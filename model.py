"""T1DMAI — encoder-only transformer for risk-space BG forecasting.

A masked patch withholds feat 0 (CGM in Kovatchev RISK space); feat 4 announces which
patches are masked. forward returns (q_tau, median) risk space; fp32-native, no bf16 autocast.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from T1DMSIM.simulator import BG_CLAMP_MIN

from config import (
    D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, FFN_DIM,
    PATCH_DIM, PATCH_SIZE,
    ROPE_BASE, BG_HEAD_HIDDEN, N_SPREADS, BG_HEAD_INIT_SCALE,
    TIME_PROBE_ENABLED, TIME_PROBE_HIDDEN, TIME_PROBE_DETACH, TIME_PROBE_INIT_SCALE,
    TIME_PROBE_N_BINS,
    CROSSING_HEAD_ENABLED, CROSSING_HEAD_HIDDEN, CROSSING_HEAD_DETACH,
    CROSSING_HEAD_INIT_SCALE, N_CROSSING,
)
from utils import assemble_quantiles, step_states


class RMSNorm(nn.Module):
    """RMS layer normalization without mean subtraction or bias, fp32-native.

    ``x / sqrt(mean(x², dim=-1) + eps) * weight``.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Init 1.0, so the block starts as the identity.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., dim)`` → same shape."""
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: int = ROPE_BASE,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cosine and sine tables, both ``(seq_len, head_dim)``.

    Built once per forward at the model level, shared across layers; ``head_dim`` must be
    even. ``base`` defaults to ``ROPE_BASE``.
    """
    half = head_dim // 2
    # Geometric series of inverse frequencies — RoPE's signature.
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)           # (T, head_dim/2)
    # Duplicated so the (T, head_dim) table multiplies Q/K element-wise.
    emb = torch.cat([freqs, freqs], dim=-1)            # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotary position embedding on Q or K; same shape in and out.

    Splits the head dim in half, rotates each pair by an angle growing geometrically with
    dim index, linearly with position. x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim).
    """
    B, H, T, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # Rotation companion [-x2, x1]: with the multiply below this is [cos -sin; sin cos].
    x_rot = torch.cat([-x2, x1], dim=-1)
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return x * c + x_rot * s


def attention_weights(
    q: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax attention weights ``(B, H, T, T)``, each row a distribution over T keys.

    SDPA forms this distribution internally, never returns it; this is the single copy,
    called by TemporalSelfAttention.forward on the tensors it hands to SDPA.
    """
    logits = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.softmax(logits.masked_fill(~attn_mask, float('-inf')), dim=-1)


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over the temporal (patch) axis.

    QK-norm bounds attention logits through the N_LAYERS-deep stack; RoPE supplies relative
    position with no learned embedding. (B, T, D_MODEL) in and out.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.w_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.w_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.w_o = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        # Diagnostic tap, off by default; None is a static branch so export/training are unaffected.
        self.attn_sink: list[torch.Tensor] | None = None

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, T, D_MODEL)`` → ``(B, T, D_MODEL)``.

        rope_cos/sin: ``(T, HEAD_DIM)`` tables, or None to build here. attn_mask: bool, True =
        attend — ``(T, T)`` shared or ``(B, 1, T, T)`` per sample, shaped once in T1DMAI.forward.
        """
        B, T, D = x.shape
        assert D == D_MODEL, f"Expected D_MODEL={D_MODEL}, got {D}"

        # (B, H, T, head_dim): SDPA wants the head axis ahead of T.
        q = self.w_q(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)  # (B, H, T, head_dim)
        k = self.w_k(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.w_v(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)

        # QK-norm BEFORE RoPE, so the rotation cannot undo the normalized norms.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # model.forward always passes precomputed tables; this is the fallback.
        if rope_cos is None or rope_sin is None:
            cos, sin = build_rope_cache(T, HEAD_DIM, device=x.device, dtype=x.dtype)
        else:
            cos, sin = rope_cos, rope_sin
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # A bare (B,T,T) mask must never arrive: broadcasting would land B on the head axis.
        assert attn_mask.dim() in (2, 4), (
            f"attn_mask must be (T, T) or (B, 1, T, T), got "
            f"{tuple(attn_mask.shape)}"
        )
        if self.attn_sink is not None:
            self.attn_sink.append(attention_weights(q, k, attn_mask))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        return self.w_o(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network, fp32-native, no biases.

    ``output = (SiLU(x @ W1) ⊙ (x @ W3)) @ W2``; ``(B, T, D_MODEL)`` in and out.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(D_MODEL, FFN_DIM, bias=False)  # gate projection
        self.w3 = nn.Linear(D_MODEL, FFN_DIM, bias=False)  # value projection
        self.w2 = nn.Linear(FFN_DIM, D_MODEL, bias=False)  # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, D_MODEL)`` → ``(B, T, D_MODEL)``."""
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


class TransformerBlock(nn.Module):
    """One pre-norm block: ``x + attn(norm1(x))`` then ``x + ffn(norm2(x))``.

    Two residual writes per block. ``(B, T, D_MODEL)`` in and out.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = RMSNorm(D_MODEL)
        self.attn = TemporalSelfAttention()
        self.norm2 = RMSNorm(D_MODEL)
        self.ffn = SwiGLUFFN()

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, T, D_MODEL)`` → ``(B, T, D_MODEL)``; mask shapes as in
        ``TemporalSelfAttention.forward``.
        """
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x

class T1DMAI(nn.Module):
    """Transformer for type-1 diabetes BG forecasting in Kovatchev RISK space.

    bg_head emits a median risk delta plus spreads per masked-patch timestep; assemble_quantiles
    anchors at f(anchor_bg). mg/dL is recovered by inference; the model never leaves risk space.
    """

    def __init__(self) -> None:
        super().__init__()

        # Masked bg z=0 is a legal reading (~142 mg/dL), not a sentinel; feat 4 announces the mask.
        self.patch_embed = nn.Linear(PATCH_DIM, D_MODEL)

        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])

        self.final_norm = RMSNorm(D_MODEL)

        # One MLP per step; FROZEN column layout, the graph cut point downstream.

        # col 0 = median delta; cols 1..N = tau>.5 spreads; cols N+1.. = tau<.5 spreads.
        self.bg_head = nn.Sequential(
            nn.Linear(D_MODEL, BG_HEAD_HIDDEN), nn.SiLU(),
            nn.Linear(BG_HEAD_HIDDEN, BG_HEAD_HIDDEN), nn.SiLU(),
            nn.Linear(BG_HEAD_HIDDEN, 1 + 2 * N_SPREADS),
        )

        # Built under saved/restored RNG state so its draws consume ZERO of the init stream.
        if TIME_PROBE_ENABLED:
            _rng_state = torch.random.get_rng_state()
            self.time_head = nn.Sequential(
                nn.Linear(D_MODEL, TIME_PROBE_HIDDEN), nn.SiLU(),
                nn.Linear(TIME_PROBE_HIDDEN, TIME_PROBE_N_BINS),
            )
            torch.random.set_rng_state(_rng_state)
        else:
            self.time_head = None

        # Same RNG discipline as the probe; two cumulative-crossing logits per step (§8.5).
        if CROSSING_HEAD_ENABLED:
            _rng_state = torch.random.get_rng_state()
            self.crossing_head = nn.Sequential(
                nn.Linear(D_MODEL, CROSSING_HEAD_HIDDEN), nn.SiLU(),
                nn.Linear(CROSSING_HEAD_HIDDEN, N_CROSSING),
            )
            torch.random.set_rng_state(_rng_state)
        else:
            self.crossing_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Width-aware init std with a residual-aware rescale.

        ``base_std`` scales as ``1/sqrt(d_model)``, anchored at GPT-2's ``0.02`` for
        ``d_model=512``, so resize_model.py doesn't land in a too-hot SwiGLU regime.
        """
        # sqrt(1.0) is exactly 1.0 in IEEE 754: bit-identical to the literal 0.02 at d_model=512.
        base_std = 0.02 * math.sqrt(512.0 / D_MODEL)
        aux_modules = set(self.time_head.modules()) if self.time_head is not None else set()
        if self.crossing_head is not None:
            aux_modules |= set(self.crossing_head.modules())
        for module in self.modules():
            if module in aux_modules:
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=base_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Residual write TWICE per block, so rescale is 1/sqrt(2*N_LAYERS); else Muon explodes it.
        residual_init_std = base_std / math.sqrt(2 * N_LAYERS)
        for block in self.blocks:
            nn.init.normal_(block.attn.w_o.weight, mean=0.0, std=residual_init_std)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=residual_init_std)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

        # Tiny final weight, zero bias -> median delta ~0 at step 0; init median is f(persistence).
        final = self.bg_head[-1]  # pyright: ignore[reportIndexIssue]
        nn.init.normal_(final.weight, mean=0.0, std=BG_HEAD_INIT_SCALE)
        nn.init.zeros_(final.bias)

        # LAST and under a saved RNG state: the probe consumes none of the init stream.
        _probe_state = torch.random.get_rng_state()
        if self.time_head is not None:
            for module in self.time_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=base_std)
                    nn.init.zeros_(module.bias)
            tfinal = self.time_head[-1]
            nn.init.normal_(tfinal.weight, mean=0.0, std=TIME_PROBE_INIT_SCALE)
            nn.init.zeros_(tfinal.bias)
        torch.random.set_rng_state(_probe_state)
        # After the probe, so a model built with either head alone is byte-identical elsewhere.
        if self.crossing_head is not None:
            for module in self.crossing_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=base_std)
                    nn.init.zeros_(module.bias)
            cfinal = self.crossing_head[-1]
            nn.init.normal_(cfinal.weight, mean=0.0, std=CROSSING_HEAD_INIT_SCALE)
            nn.init.zeros_(cfinal.bias)

    def forward(
        self,
        patches: torch.Tensor,
        attn_mask: torch.Tensor,
        anchor_bg: torch.Tensor,
        mask_idx: torch.Tensor,
        return_time: bool = False,
        return_crossing: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        anchor_bg: (B, M) mg/dL, detached before f; padded slots need a legal value (units
        tripwire reads all M) and are discarded via valid. mask_idx: padded slots gather patch 0.
        return_time=False returns the 2-tuple bit-identically.
        """
        B, T, _ = patches.shape
        assert mask_idx.dim() == 2 and mask_idx.shape[0] == B, (
            f"mask_idx must be (B, M) with B={B}, got {tuple(mask_idx.shape)}"
        )
        assert mask_idx.dtype == torch.int64, (
            f"mask_idx must be int64 (gather index), got {mask_idx.dtype}"
        )
        M = mask_idx.shape[1]
        assert anchor_bg.shape == (B, M), (
            f"anchor_bg must be (B, M)=({B}, {M}), got {tuple(anchor_bg.shape)}"
        )
        # Units tripwire: legal z < BG_CLAMP_MIN-1e-3; a normalized value here trips loudly, all M.
        assert bool((anchor_bg >= BG_CLAMP_MIN - 1e-3).all()), (
            "anchor_bg below BG_CLAMP_MIN — non-mg/dL value routed into the anchor"
        )

        x = self.patch_embed(patches)                    # (B, T, D_MODEL)

        # RoPE depends only on (T, head_dim), so build once and pass to every block.
        rope_cos, rope_sin = build_rope_cache(T, HEAD_DIM, device=x.device, dtype=x.dtype)

        # SDPA consumes the bool mask directly; nothing additive materialized for backward.

        # A per-sample (B,T,T) mask gains the head axis here, else B lands on the head axis.
        assert attn_mask.dtype == torch.bool, (
            f"attn_mask must be bool (True = attend), got {attn_mask.dtype}"
        )
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)                              # (B, 1, T, T)
            assert attn_mask.dim() == 4, (
                f"per-sample mask must be (B, 1, T, T), got {tuple(attn_mask.shape)}"
            )

        for block in self.blocks:
            x = block(x, rope_cos, rope_sin, attn_mask)

        x = self.final_norm(x)                           # (B, T, D_MODEL)

        # Masked set is arbitrary; head reads slots by index, never a slice; padded gather 0.
        h_steps = step_states(x, mask_idx, attn_mask)     # (B, M, PATCH_SIZE, D_MODEL)
        head_raw = self.bg_head(h_steps)
        assert head_raw.shape == (
            B, M, PATCH_SIZE, 1 + 2 * N_SPREADS
        ), f"head_raw shape {tuple(head_raw.shape)} unexpected"

        q_tau, median = assemble_quantiles(head_raw, anchor_bg.detach(), mask_idx)
        if not return_time and not return_crossing:
            return q_tau, median
        time_pred = None
        if return_time and self.time_head is not None:
            # Every gathered hidden state, no mean-pool, forces it to encode the absolute clock.

            # Slot j is patch mask_idx[:, j]; hour target follows mask_idx, not a fixed offset.
            pred = x.gather(1, mask_idx.unsqueeze(-1).expand(B, M, D_MODEL))
            h = pred if not TIME_PROBE_DETACH else pred.detach()  # (B, M, D_MODEL)
            time_pred = self.time_head(h)                         # (B, M, TIME_PROBE_N_BINS)
        if not return_crossing:
            return q_tau, median, time_pred
        crossing = None
        if self.crossing_head is not None:
            h_c = h_steps if not CROSSING_HEAD_DETACH else h_steps.detach()
            crossing = self.crossing_head(h_c)                    # (B, M, PATCH_SIZE, N_CROSSING)
        return q_tau, median, time_pred, crossing
