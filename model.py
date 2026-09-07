"""T1DMAI — encoder-only transformer for risk-space BG forecasting.

The input is a window of ``T`` patches, each visible or masked. A masked patch withholds
feat 0 (CGM blood glucose in Kovatchev RISK space) while carb intake, combined insulin
action and exercise keep their true or announced values; feat 4 (``bg_masked``) announces
which patches are masked. The model emits a quantile fan in RISK space for every masked
patch. A masked span ending at patch ``T−1`` is a forecast, one starting at patch 0 a
backcast, anything else infill. A single model covers windows starting at any time of
day. Inference owns the risk→mg/dL inverse (``kovatchev_f_inv``).

* Patch embedding: six consecutive 5-minute simulator timesteps make one 30-minute
  patch, so ``PATCH_DIM = PATCH_SIZE × N_INPUT_FEATURES`` values step-major, projected by
  ``patch_embed = Linear(PATCH_DIM, D_MODEL)``. No patient-conditioning embedding —
  patient identity is implicit in the context window.
* ``N_LAYERS`` pre-norm blocks, each ``TemporalSelfAttention`` (QK-norm then RoPE on Q/K,
  ``F.scaled_dot_product_attention`` under the caller's bool mask) then a SwiGLU FFN of
  width ``FFN_DIM``.
* BG head: after a final RMSNorm ``utils.step_states`` interpolates the patch states of
  each masked span and its visible neighbours into one state per within-patch timestep,
  and a 3-layer SiLU MLP maps every step state to ``1 + 2·N_SPREADS`` raw values.
  ``utils.assemble_quantiles`` anchors the median at ``f(anchor_bg)`` per slot and
  assembles the ascending fan.

``forward`` returns ``(q_tau, median)`` in risk space:
``(B, M, PATCH_SIZE, N_QUANTILES)`` and ``(B, M, PATCH_SIZE)``.

fp32-native throughout — no bf16 autocast anywhere, so ``RMSNorm`` is a plain module and
the SwiGLU gate × value product a native fp32 multiply.
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

    Built once per forward at the model level and shared across layers: the tables depend
    only on ``T`` and ``head_dim``. ``head_dim`` must be even; ``base`` defaults to
    ``ROPE_BASE``.
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

    Splits the head dim in half, treats each pair as a 2-D vector and rotates by an angle
    growing geometrically with the dim index and linearly with the position.

    x: ``(B, n_heads, T, head_dim)``. cos, sin: ``(T, head_dim)`` caches.
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

    The distribution ``F.scaled_dot_product_attention`` forms internally and never
    returns. Recomputing it outside this module would pin the QK-norm and RoPE order down
    in a second place, so this is the single copy and ``TemporalSelfAttention.forward``
    calls it on the very tensors it hands to SDPA.

    q, k: ``(B, H, T, HEAD_DIM)`` after QK-norm and RoPE.
    attn_mask: ``(T, T)`` or ``(B, 1, T, T)`` bool, True = attend.
    ``create_attention_mask_from_visible`` keeps every row's diagonal open, so no row is
    all-blocked and no row is NaN.
    """
    logits = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.softmax(logits.masked_fill(~attn_mask, float('-inf')), dim=-1)


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over the temporal (patch) axis.

    QK-norm — per-head RMSNorm on Q and K — bounds the attention logits through the
    ``N_LAYERS``-deep stack; RoPE on Q and K supplies relative position with no learned
    positional embedding. ``(B, T, D_MODEL)`` in and out.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.w_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.w_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.w_o = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        # Diagnostic tap, off by default: collects one (B, H, T, T) tensor per forward.
        # ``None`` is a static branch, so export trace and training step are unaffected.
        self.attn_sink: list[torch.Tensor] | None = None

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, T, D_MODEL)`` → ``(B, T, D_MODEL)``.

        rope_cos, rope_sin: ``(T, HEAD_DIM)`` tables, or None to build them here.
        attn_mask: bool, True = attend — ``(T, T)`` shared or ``(B, 1, T, T)`` per
            sample, shaped once in ``T1DMAI.forward``.
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

        # A bare 3-D (B, T, T) mask must never arrive here: broadcasting aligns from the
        # right, so B would land on the head axis — raising when B != N_HEADS, silently
        # masking head b with row b's mask when B == N_HEADS. T1DMAI.forward adds the
        # head axis; this asserts it did.
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

    ``bg_head`` emits, per masked-patch timestep, a median risk delta plus ``2·N_SPREADS``
    quantile spreads off that timestep's interpolated state; ``utils.assemble_quantiles``
    anchors the median at ``f(anchor_bg)`` and assembles the ascending fan. mg/dL is recovered by inference through
    ``kovatchev_f_inv`` — the model never leaves risk space.

    Forward inputs:
        patches: ``(B, T, PATCH_DIM)`` step-major
            (``PATCH_SIZE × N_INPUT_FEATURES = PATCH_DIM``); ``T <= MAX_SEQ_LEN``, the
            collate pads to the batch maximum, not to ``MAX_SEQ_LEN``.
        attn_mask: ``(T, T)`` or ``(B, T, T)`` bool, True = attend.
        anchor_bg: ``(B, M)`` mg/dL — per masked patch, the nearest visible reading; the
            risk anchor ``f(anchor_bg)`` for that slot's median.
        mask_idx: ``(B, M)`` int64 — the patch index each head slot reads.

    Forward outputs:
        q_tau: ``(B, M, PATCH_SIZE, N_QUANTILES)`` risk space.
        median: ``(B, M, PATCH_SIZE)`` risk space, the ``QUANTILE_LEVELS.index(0.5)``
            column of ``q_tau``.

    ``return_time=True`` additionally returns ``(B, M, TIME_PROBE_N_BINS)`` per-slot
    hour-of-day bin logits, or ``None`` when the probe is disabled, read off every
    gathered masked-patch hidden state. The head never feeds the forecast in the forward.
    """

    def __init__(self) -> None:
        super().__init__()

        # A masked patch carries z = 0 in its bg slots — a legal reading (~142 mg/dL),
        # not a sentinel — so feat 4 announces the mask instead; the bias lets the
        # projection use that bit as an offset.
        self.patch_embed = nn.Linear(PATCH_DIM, D_MODEL)

        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])

        self.final_norm = RMSNorm(D_MODEL)

        # One MLP shared by every within-patch step, run on that step's interpolated
        # state. FROZEN column layout, and the graph cut point downstream: col 0 = median
        # risk delta (added to that slot's f(anchor_bg)); cols 1..N = τ>.5 ascending
        # spreads (softplus → positive gaps); cols N+1.. = τ<.5 ascending spreads.
        self.bg_head = nn.Sequential(
            nn.Linear(D_MODEL, BG_HEAD_HIDDEN), nn.SiLU(),
            nn.Linear(BG_HEAD_HIDDEN, BG_HEAD_HIDDEN), nn.SiLU(),
            nn.Linear(BG_HEAD_HIDDEN, 1 + 2 * N_SPREADS),
        )

        # Built under a saved/restored RNG state, so its nn.Linear draws consume ZERO of
        # the main init stream — otherwise they land BEFORE _init_weights and shift every
        # forecast weight. _init_weights re-inits time_head LAST for the same reason.
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

        ``base_std`` scales as ``1/sqrt(d_model)``, anchored at the GPT-2 ``0.02`` for
        ``d_model = 512``, so a resize by ``resize_model.py`` does not land in a too-hot
        regime where SwiGLU saturates and Muon over-corrects in the first few hundred
        steps.
        """
        # sqrt(1.0) is exactly 1.0 in IEEE 754, so this is bit-identical to the literal
        # 0.02 at d_model = 512.
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

        # Each block writes to the residual stream TWICE (attn out, FFN out), so the
        # rescale is 1/sqrt(2*N_LAYERS) — one /sqrt(N) per independent write across the
        # depth. Without it Muon drives these layers to huge norms and the stream explodes.
        residual_init_std = base_std / math.sqrt(2 * N_LAYERS)
        for block in self.blocks:
            nn.init.normal_(block.attn.w_o.weight, mean=0.0, std=residual_init_std)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=residual_init_std)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

        # Tiny final weight and zero bias ⇒ median risk delta ≈ 0 at step 0, so the
        # initial median is f(persistence), flat from each slot's anchor_bg.
        final = self.bg_head[-1]  # pyright: ignore[reportIndexIssue]
        nn.init.normal_(final.weight, mean=0.0, std=BG_HEAD_INIT_SCALE)
        nn.init.zeros_(final.bias)

        # LAST, so every forecast-weight RNG draw above is byte-identical to a model
        # built without the probe.
        if self.time_head is not None:
            for module in self.time_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=base_std)
                    nn.init.zeros_(module.bias)
            tfinal = self.time_head[-1]
            nn.init.normal_(tfinal.weight, mean=0.0, std=TIME_PROBE_INIT_SCALE)
            nn.init.zeros_(tfinal.bias)
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
        Args:
            patches: ``(B, T, PATCH_DIM)`` step-major. ``T <= MAX_SEQ_LEN``: the collate
                left-pads to the batch maximum, so ``T`` varies batch to batch and is
                never asserted equal to ``MAX_SEQ_LEN``.
            attn_mask: ``(T, T)`` or ``(B, T, T)`` bool, True = attend.
            anchor_bg: ``(B, M)`` mg/dL — the last step of the span's left neighbour, or
                the first step of the right neighbour for a span at patch 0. Every slot
                of one span carries the same value. ``.detach()``'d before ``f``, so no
                gradient flows into it. Padded slots must still carry a legal mg/dL
                value — the units tripwire below reads all ``M`` — and their outputs are
                discarded downstream by ``valid``.
            mask_idx: ``(B, M)`` int64 — the patch index each head slot reads; padded
                slots gather position 0.
            return_time: ``False`` (default) returns the 2-tuple bit-identically.

        Returns:
            q_tau: ``(B, M, PATCH_SIZE, N_QUANTILES)`` ascending quantile fan, RISK space.
            median: ``(B, M, PATCH_SIZE)`` risk space, the ``QUANTILE_LEVELS.index(0.5)``
                column of ``q_tau``.
            time_pred: only with ``return_time=True`` — ``(B, M, TIME_PROBE_N_BINS)``
                per-slot hour-of-day bin logits, or ``None`` when ``TIME_PROBE_ENABLED``
                is False. Read off every gathered masked-patch hidden state, no
                mean-pool; with ``TIME_PROBE_DETACH=False`` its loss co-trains the trunk.
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
        # Units tripwire: every legal z satisfies z_max < BG_CLAMP_MIN - 1e-3, the floor
        # this assert reads, so a normalized-space BG routed into the anchor trips loudly.
        # Pool-independent, and it covers all M slots.
        assert bool((anchor_bg >= BG_CLAMP_MIN - 1e-3).all()), (
            "anchor_bg below BG_CLAMP_MIN — non-mg/dL value routed into the anchor"
        )

        x = self.patch_embed(patches)                    # (B, T, D_MODEL)

        # RoPE depends only on (T, head_dim), so build once and pass to every block.
        rope_cos, rope_sin = build_rope_cache(T, HEAD_DIM, device=x.device, dtype=x.dtype)

        # SDPA consumes the bool mask directly (True = attend), so nothing additive is
        # materialized and no (B, H, T, T) float is saved for backward. A shared (T, T)
        # mask broadcasts over batch and head unchanged; a per-sample (B, T, T) must gain
        # the head axis here, since broadcasting aligns from the right and would otherwise
        # put B on the head axis — raising when B != N_HEADS, and silently masking head b
        # with row b's mask when B == N_HEADS.
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

        # The masked set is arbitrary (forecast, backcast, infill), so the head reads its
        # slots by index, never by a trailing slice. Padded slots gather position 0 and
        # are discarded downstream by ``valid``.
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
            # Every gathered masked-patch hidden state, no mean-pool, so each per-patch
            # representation the BG head's nodes are drawn from is forced to encode the
            # absolute clock. With TIME_PROBE_DETACH=False the probe gradient back-props
            # into the trunk; either way the forward VALUE of q_tau/median is unchanged.
            # Slot j is patch mask_idx[:, j], so the caller's per-slot hour target must
            # follow mask_idx, not a fixed offset from the context end.
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
