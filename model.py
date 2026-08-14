"""
T1DMAI — Encoder-only transformer for risk-space BG forecasting.
=================================================================

What the model does
-------------------
T1DMAI takes a window of ``T`` patches, each visible or masked.  A masked
patch withholds feat 0 (CGM blood glucose in Kovatchev RISK space) while
carb intake, combined insulin action and exercise keep their true or
announced values; feat 4 (``bg_masked``) announces which patches are
masked.  The model emits a quantile fan in Kovatchev RISK space for every
masked patch.  A masked span ending at patch ``T−1`` is a forecast, one
starting at patch 0 a backcast, anything else infill.  The head emits,
per masked-patch timestep, a median risk delta plus a set of ascending
quantile spreads; ``utils.assemble_quantiles`` anchors the median at
``f(anchor_bg)`` and assembles the full ascending quantile fan.  A single
model is trained on windows that may start at any time of day.  Inference
owns the risk→mg/dL inverse (``kovatchev_f_inv``).

Architecture in one paragraph
-----------------------------
* **Patch embedding.**  Six consecutive 5-minute simulator timesteps are
  grouped into one 30-minute "patch", giving each patch
  ``PATCH_SIZE × N_INPUT_FEATURES`` (= ``PATCH_DIM``) values, step-major.
  ``patch_embed = Linear(PATCH_DIM, D_MODEL)`` projects each patch
  into the model's hidden dimension.  No patient-conditioning embedding is
  added — patient identity is implicit in the context window the model is
  shown.
* **N_LAYERS transformer blocks**, each with two pre-norm sub-layers:
    1. ``TemporalSelfAttention`` over patches (RoPE on Q/K, QK-norm for
       stability, ``F.scaled_dot_product_attention`` underneath the bool
       attention pattern the caller builds — visible rows see visible
       columns, masked rows see everything, padding columns blocked).
    2. ``SwiGLU FFN`` of width ``FFN_DIM`` (config-driven).
* **BG head.**  After a final RMSNorm we gather the ``M`` masked-patch
  hidden states by ``mask_idx`` and run them through a single 3-layer SiLU
  MLP that emits ``1 + 2·N_SPREADS`` raw values per timestep (one median
  delta + the ascending quantile spreads).  ``utils.assemble_quantiles``
  turns these into the ascending quantile fan anchored, per slot, at
  ``f(anchor_bg)``.

Forward pass returns the 2-tuple ``(q_tau, median)`` in risk space —
``q_tau`` of shape ``(B, M, PATCH_SIZE, N_QUANTILES)``, ``median`` of
shape ``(B, M, PATCH_SIZE)``.

Numerical care taken throughout
-------------------------------
Everything runs fp32-native — there is no bf16 autocast anywhere, so the
old precision hardening (the fp32-forcing RMSNorm autograd Function and
the fp32-cast SwiGLU product) is unnecessary.  ``RMSNorm`` is a plain
fp32 module (numerically identical to the old fused op: the same fp32
mean-square normalization, with autograd supplying the same analytic
backward) and the SwiGLU product is a native fp32 multiply.
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
    BG_HEAD_STEP_BASIS_DIM, BG_HEAD_STEP_BASIS_TYPE,
    TIME_PROBE_ENABLED, TIME_PROBE_HIDDEN, TIME_PROBE_DETACH, TIME_PROBE_INIT_SCALE,
    TIME_PROBE_N_BINS,
)
from utils import assemble_quantiles


def make_step_basis(n_steps: int, k_dim: int, kind: str = 'dct') -> torch.Tensor:
    """Fixed ``(n_steps, k_dim)`` within-patch step-expansion basis.

    The BG head emits ``k_dim`` coefficients per (patch, channel); this basis
    expands them across the ``n_steps`` within-patch timesteps as ``B @ coeff``.
    Columns are low-order / low-frequency and orthonormal over the ``n_steps``
    points, so the reconstructed within-patch curve is smooth. With
    ``k_dim < n_steps`` the highest-frequency (period-2) mode is excluded, so an
    intra-patch zigzag of the median is unrepresentable by construction;
    ``k_dim == n_steps`` spans the full space (recovers a free per-step head).

    Args:
        n_steps: ``PATCH_SIZE`` — timesteps spanned per patch.
        k_dim: ``K`` — number of coefficients / basis columns, ``1 <= K <= n_steps``.
        kind: ``'dct'`` (DCT-II cosine modes, ascending frequency) or ``'poly'``
            (orthonormal polynomials, ascending degree).

    Returns:
        ``(n_steps, k_dim)`` float32 tensor with L2-orthonormal columns.
    """
    assert 1 <= k_dim <= n_steps, (
        f"need 1 <= BG_HEAD_STEP_BASIS_DIM ({k_dim}) <= PATCH_SIZE ({n_steps})"
    )
    s = torch.arange(n_steps, dtype=torch.float64)
    if kind == 'dct':
        ks = torch.arange(k_dim, dtype=torch.float64)
        basis = torch.cos(math.pi * (s.view(-1, 1) + 0.5) * ks.view(1, -1) / n_steps)
    elif kind == 'poly':
        # Orthonormal polynomials (ascending degree) over evenly spaced t∈[-1,1],
        # via a QR of the Vandermonde — the first k_dim Legendre-like columns.
        t = 2.0 * s / (n_steps - 1) - 1.0 if n_steps > 1 else s
        vand = torch.stack([t ** p for p in range(k_dim)], dim=1)  # (S, K)
        basis, _ = torch.linalg.qr(vand)
    else:
        raise ValueError(f"unknown BG_HEAD_STEP_BASIS_TYPE {kind!r} (want 'dct' or 'poly')")
    basis = basis / basis.norm(dim=0, keepdim=True)               # orthonormal columns
    return basis.to(torch.float32)


# ============================================================================
# Building blocks
# ============================================================================

class RMSNorm(nn.Module):
    """RMS Layer Normalization without mean subtraction or bias (fp32-native).

    Equivalent to ``x / sqrt(mean(x², dim=-1) + eps) * weight``.  Cheaper
    than LayerNorm (no mean, no bias) and works well in pre-norm transformer
    blocks.

    The whole model runs fp32-native (no bf16 autocast), so this is a plain
    module rather than a custom autograd Function: the forward is the same
    fp32 mean-square normalization the old fused op used, and autograd derives
    the identical analytic backward — numerically equivalent, no hand-written
    gradient required.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Single learnable per-channel scale.  Initialized to 1.0 so the
        # block starts as the identity transform.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim)
        Returns:
            normalized x with the same shape.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: int = ROPE_BASE,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine tables for RoPE.

    Computed once per forward pass at the model level and shared across all
    layers — RoPE positions only depend on ``T`` and ``head_dim``, so re-doing
    this work inside every layer is pure waste.

    Args:
        seq_len: Length of the sequence (``T``).
        head_dim: Per-head dimension (must be even).
        base: RoPE base frequency (default ``ROPE_BASE`` = 1000).
        device: Target device.
        dtype: Target dtype.

    Returns:
        cos: (seq_len, head_dim) cosine values.
        sin: (seq_len, head_dim) sine values.
    """
    half = head_dim // 2
    # Geometric series of inverse frequencies — RoPE's signature.
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)           # (T, head_dim/2)
    # Concatenate with itself so the resulting (T, head_dim) table can be
    # multiplied element-wise with Q / K without further broadcasting.
    emb = torch.cat([freqs, freqs], dim=-1)            # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to Q or K.

    The rotation is the standard RoPE formulation: split the head dim in
    half, treat each pair as a 2D vector, and rotate by an angle that grows
    geometrically with the dim index and linearly with the position index.

    Args:
        x:   (B, n_heads, T, head_dim)
        cos: (T, head_dim) cosine cache
        sin: (T, head_dim) sine cache

    Returns:
        x with RoPE applied — same shape.
    """
    B, H, T, D = x.shape
    half = D // 2
    # Split head into [first-half, second-half].
    x1 = x[..., :half]
    x2 = x[..., half:]
    # Build the rotation companion: [-x2, x1].  This implements the 2D
    # rotation [cos -sin; sin cos] · [x1; x2] when combined as below.
    x_rot = torch.cat([-x2, x1], dim=-1)
    # Broadcast cos/sin from (T, D) to (1, 1, T, D) so they multiply across
    # batch and head axes.
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return x * c + x_rot * s


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the temporal (patch) dimension.

    Highlights:
      * QK-norm (per-head RMSNorm on Q and K) bounds attention logits and
        prevents gradient spikes through the ``N_LAYERS``-deep stack.
      * RoPE on Q and K injects relative-position information without learned
        positional embeddings.

    Input:  (B, T, D_MODEL)
    Output: (B, T, D_MODEL)
    """

    def __init__(self) -> None:
        super().__init__()
        # Standard QKV / output projections — full d_model dim.
        self.w_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.w_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.w_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.w_o = nn.Linear(D_MODEL, D_MODEL, bias=False)
        # QK-norm: a small RMSNorm per head on Q and K.  Bounds the QKᵀ
        # magnitude so attention logits stay well-scaled through the
        # ``N_LAYERS``-deep stack, preventing gradient spikes on long
        # sequences.
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, D_MODEL)
            rope_cos:  (T, HEAD_DIM) precomputed cosine table (or None)
            rope_sin:  (T, HEAD_DIM) precomputed sine table (or None)
            attn_mask: bool attention mask, True where a position may attend.
                       ``(T, T)`` for a shared mask or ``(B, 1, T, T)`` for a
                       per-sample one; shaped once in ``T1DMAI.forward``.

        Returns:
            out: (B, T, D_MODEL)
        """
        B, T, D = x.shape
        assert D == D_MODEL, f"Expected D_MODEL={D_MODEL}, got {D}"

        # Standard split into (B, H, T, head_dim).  ``transpose(1, 2)`` puts
        # the head axis ahead of T so SDPA broadcasts correctly.
        q = self.w_q(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)  # (B, H, T, head_dim)
        k = self.w_k(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.w_v(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)

        # QK-norm BEFORE RoPE — RoPE rotates Q and K but doesn't change their
        # magnitude, so we can apply the norm either before or after; doing
        # it first keeps the normalized norms from being undone by RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE on Q and K.  If the caller didn't pass tables, build them on
        # the fly — but the model.forward path always passes precomputed
        # tables to avoid redoing this work in every block.
        if rope_cos is None or rope_sin is None:
            cos, sin = build_rope_cache(T, HEAD_DIM, device=x.device, dtype=x.dtype)
        else:
            cos, sin = rope_cos, rope_sin
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # SDPA takes the bool mask as it stands (True = attend), so no per-layer
        # additive float tensor is materialized.  Both accepted shapes broadcast
        # against (B, H, T, T): (T, T) over batch and head, (B, 1, T, T) over
        # head.  A bare 3-D (B, T, T) mask must never arrive here — broadcasting
        # aligns from the right, so B would land on the head axis: it raises when
        # B != N_HEADS and silently masks head b with row b's mask when B ==
        # N_HEADS.  ``T1DMAI.forward`` adds the head axis; this asserts it did.
        assert attn_mask.dim() in (2, 4), (
            f"attn_mask must be (T, T) or (B, 1, T, T), got "
            f"{tuple(attn_mask.shape)}"
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, T, head_dim)
        # Recombine heads: (B, H, T, head_dim) → (B, T, D_MODEL)
        out = out.transpose(1, 2).contiguous().view(B, T, D_MODEL)
        return self.w_o(out)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network (fp32-native).

    ``output = (SiLU(x @ W1) ⊙ (x @ W3)) @ W2``.  No bias terms (modern
    convention).  The whole model runs fp32, so the gate × value product is a
    native fp32 multiply — the old bf16-overflow-avoiding fp32 cast is gone.

    Input:  (B, T, D_MODEL)
    Output: (B, T, D_MODEL)
    """

    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(D_MODEL, FFN_DIM, bias=False)  # gate projection
        self.w3 = nn.Linear(D_MODEL, FFN_DIM, bias=False)  # value projection
        self.w2 = nn.Linear(FFN_DIM, D_MODEL, bias=False)  # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D_MODEL)

        Returns:
            out: (B, T, D_MODEL)
        """
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


class TransformerBlock(nn.Module):
    """
    One T1DMAI transformer block.

    Pre-norm throughout — every sub-layer is ``residual + sublayer(norm(x))``.
    The two sub-layers are temporal self-attention and the SwiGLU FFN, in that
    order (two residual writes per block).

    Input:  (B, T, D_MODEL)
    Output: (B, T, D_MODEL)
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
        """
        Args:
            x:         (B, T, D_MODEL)
            rope_cos:  (T, HEAD_DIM) precomputed cosine table (or None)
            rope_sin:  (T, HEAD_DIM) precomputed sine table (or None)
            attn_mask: bool attention mask — ``(T, T)`` or ``(B, 1, T, T)``,
                       True where a position may attend.

        Returns:
            x: (B, T, D_MODEL)
        """
        # Sub-layer 1: temporal self-attention under the caller's bool mask.
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin, attn_mask)
        # Sub-layer 2: SwiGLU FFN.
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Main model
# ============================================================================

class T1DMAI(nn.Module):
    """
    T1DMAI: Transformer for Type 1 Diabetes risk-space BG forecasting.

    Predicts blood glucose over every masked patch as a fan of ascending
    quantiles in Kovatchev RISK space.  The ``bg_head`` emits, per masked-patch
    timestep, a median risk delta plus ``2·N_SPREADS`` quantile spreads;
    ``utils.assemble_quantiles`` anchors the median at ``f(anchor_bg)`` and
    assembles the full ascending fan.  Absolute BG in mg/dL is recovered
    by inference via ``kovatchev_f_inv`` — the model never leaves risk space.

    Forward pass inputs:
        patches:       (B, T, PATCH_DIM) — patched input, step-major
                       (PATCH_SIZE×N_INPUT_FEATURES = PATCH_DIM), ``T <=
                       MAX_SEQ_LEN`` — the collate pads to the batch maximum,
                       not to ``MAX_SEQ_LEN``
        attn_mask:     (T, T) or (B, T, T) bool — True = attend
        anchor_bg:     (B, M) mg/dL — per masked patch, the nearest visible
                       reading; the risk anchor ``f(anchor_bg)`` for that
                       slot's median.
        mask_idx:      (B, M) int64 — patch index each head slot reads.

    Forward pass outputs:
        q_tau:   (B, M, PATCH_SIZE, N_QUANTILES) risk space
        median:  (B, M, PATCH_SIZE) risk space
                 (== the median quantile column of ``q_tau``,
                 ``QUANTILE_LEVELS.index(0.5)``)

    ``forward`` also accepts an opt-in ``return_time`` diagnostic switch: with
    ``return_time=True`` it additionally returns the time-of-day probe output
    ``(B, M, TIME_PROBE_N_BINS)`` per-slot hour-of-day bin logits (or ``None``
    when the probe is disabled), read off every gathered masked-patch hidden
    state (the head never feeds the forecast in the forward).
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Input embeddings ---
        # patch_embed projects PATCH_DIM values per patch
        # (PATCH_SIZE × N_INPUT_FEATURES = PATCH_DIM, step-major) into D_MODEL.
        # A masked patch carries z = 0 in its bg slots — a legal reading (~142
        # mg/dL), not a sentinel — so feat 4 announces the mask instead; the
        # bias lets the projection use that bit as an offset.
        self.patch_embed = nn.Linear(PATCH_DIM, D_MODEL)

        # --- Transformer stack ---
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])

        # --- Final norm ---
        # Standard pre-norm convention: one extra RMSNorm after the last
        # block before the output head.
        self.final_norm = RMSNorm(D_MODEL)

        # --- BG quantile head (operates on the gathered masked patches only) ---
        # A single 3-layer SiLU MLP off each masked patch's hidden state that
        # emits, per patch token, ``1 + 2·N_SPREADS`` channels:
        #   col 0       = median risk delta (added to that slot's f(anchor_bg)),
        #   cols 1..N   = τ>.5 ascending spreads (softplus → positive gaps),
        #   cols N+1..  = τ<.5 ascending spreads.
        # ``utils.assemble_quantiles`` turns these into the ascending fan.
        # SMOOTH-BASIS STEP EXPANSION: rather than emit PATCH_SIZE independent
        # values per channel (which let the within-patch median zigzag freely), the
        # head emits ``K = BG_HEAD_STEP_BASIS_DIM`` coefficients per channel, expanded
        # across the PATCH_SIZE within-patch timesteps in the forward via the fixed
        # orthonormal ``step_basis`` (B, M, K, C) → (B, M, S, C). With K < PATCH_SIZE
        # the period-2 within-patch mode is unrepresentable (anti-oscillation). The
        # FROZEN head_raw COLUMN layout (1 + 2·N_SPREADS, meanings unchanged) is the
        # graph cut point downstream.
        self.bg_head = nn.Sequential(
            nn.Linear(D_MODEL, BG_HEAD_HIDDEN), nn.SiLU(),
            nn.Linear(BG_HEAD_HIDDEN, BG_HEAD_HIDDEN), nn.SiLU(),
            nn.Linear(BG_HEAD_HIDDEN, BG_HEAD_STEP_BASIS_DIM * (1 + 2 * N_SPREADS)),
        )
        # Fixed (PATCH_SIZE, K) within-patch expansion basis; a buffer so it moves
        # with .to(device) and is saved in the state dict for provenance.
        self.register_buffer(
            'step_basis',
            make_step_basis(PATCH_SIZE, BG_HEAD_STEP_BASIS_DIM, BG_HEAD_STEP_BASIS_TYPE),
        )

        # --- Time-of-day probe (co-trains the trunk when TIME_PROBE_DETACH=False) ---
        # Built under a saved/restored RNG state so its construction (nn.Linear
        # kaiming draws) consumes ZERO of the main init stream — otherwise those
        # draws land *before* _init_weights and shift every forecast weight,
        # breaking I2 (forecast weights byte-identical with/without the probe).
        # _init_weights re-inits time_head LAST, so its init draws also never
        # perturb the forecast.
        if TIME_PROBE_ENABLED:
            _rng_state = torch.random.get_rng_state()
            self.time_head = nn.Sequential(
                nn.Linear(D_MODEL, TIME_PROBE_HIDDEN), nn.SiLU(),
                nn.Linear(TIME_PROBE_HIDDEN, TIME_PROBE_N_BINS),
            )
            torch.random.set_rng_state(_rng_state)
        else:
            self.time_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with width-aware std and a residual-aware rescale.

        ``base_std`` scales as ``1/sqrt(d_model)`` so pre-activation variance
        through every ``Linear(d_model, *)`` is approximately constant
        regardless of width.  The constant ``0.02`` (the GPT-2 textbook
        value) is anchored at ``d_model = 512`` — narrower models get a
        proportionally larger std, wider models a smaller one, so resizing
        with ``resize_model.py`` doesn't push the network into a too-hot
        init regime where SwiGLU saturates and Muon over-corrects in the
        first few hundred steps.
        """
        # base_std = 0.02 at d_model=512; scales as 1/sqrt(d_model).
        # math.sqrt(1.0) is exactly 1.0 in IEEE 754 so this is bit-identical
        # to the literal 0.02 at d_model=512.
        base_std = 0.02 * math.sqrt(512.0 / D_MODEL)
        time_modules = set(self.time_head.modules()) if self.time_head is not None else set()
        for module in self.modules():
            if module in time_modules:
                continue
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=base_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale residual-branch output projections so the residual stream
        # doesn't accumulate variance with depth.  Each block writes to the
        # residual stream TWICE (temporal-attn out, FFN out), so the rescale
        # is 1/sqrt(2*N_LAYERS) — one /sqrt(N) per independent write across
        # the full depth.  Without this, Muon drives these layers to huge
        # norms and the residual stream explodes.
        residual_init_std = base_std / math.sqrt(2 * N_LAYERS)
        for block in self.blocks:
            nn.init.normal_(block.attn.w_o.weight, mean=0.0, std=residual_init_std)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=residual_init_std)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

        # BG head final layer: tiny weight (BG_HEAD_INIT_SCALE) and zero bias
        # so the median risk delta ≈ 0 at step 0 — the initial median is
        # f(persistence) (flat from each slot's anchor_bg) and the loss grows the
        # forecast from there rather than from a random kick.
        final = self.bg_head[-1]  # pyright: ignore[reportIndexIssue]
        nn.init.normal_(final.weight, mean=0.0, std=BG_HEAD_INIT_SCALE)
        nn.init.zeros_(final.bias)

        # Time-of-day probe init, done LAST so every RNG draw for the forecast
        # weights above is byte-identical to a model built without the probe.
        if self.time_head is not None:
            for module in self.time_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=base_std)
                    nn.init.zeros_(module.bias)
            tfinal = self.time_head[-1]
            nn.init.normal_(tfinal.weight, mean=0.0, std=TIME_PROBE_INIT_SCALE)
            nn.init.zeros_(tfinal.bias)

    def forward(
        self,
        patches: torch.Tensor,
        attn_mask: torch.Tensor,
        anchor_bg: torch.Tensor,
        mask_idx: torch.Tensor,
        return_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            patches:   (B, T, PATCH_DIM) — flattened feature values per patch.
                       ``T <= MAX_SEQ_LEN``: the collate left-pads to the batch
                       maximum, so ``T`` varies batch to batch and is never
                       asserted equal to ``MAX_SEQ_LEN``.
            attn_mask: (T, T) or (B, T, T) bool — True = attend
            anchor_bg: (B, M) mg/dL — per masked patch, the nearest visible
                       reading (the last step of the left neighbour, or the
                       first step of the right neighbour for a span at patch 0).
                       Every slot of one span carries the same value.  It is a
                       constant w.r.t. the model (``.detach()``'d before ``f``),
                       so no gradient flows into it.  Padded slots must still
                       carry a legal mg/dL value — the units tripwire below
                       reads all ``M`` — and their outputs are discarded
                       downstream by ``valid``.
            mask_idx:  (B, M) int64 — the patch index each head slot reads.
                       Padded slots gather position 0.
            return_time: opt-in diagnostic switch.  ``False`` (default) returns
                       the 2-tuple ``(q_tau, median)`` bit-identically; ``True``
                       additionally returns the time-of-day probe output.

        Returns:
            q_tau:  (B, M, PATCH_SIZE, N_QUANTILES) — ascending quantile fan in
                    RISK space.
            median: (B, M, PATCH_SIZE) — risk-space median
                    (== the median quantile column of ``q_tau`` at
                    ``QUANTILE_LEVELS.index(0.5)``).
            time_pred: only when ``return_time=True`` —
                    ``(B, M, TIME_PROBE_N_BINS)`` per-slot hour-of-day bin
                    logits, or ``None`` when ``TIME_PROBE_ENABLED`` is False.
                    Read off every gathered masked-patch hidden state (no
                    mean-pool); the head never feeds the forecast in the forward
                    (though with TIME_PROBE_DETACH=False its loss co-trains the
                    trunk).
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
        # Units tripwire: a z-scored value would trip this loudly, catching a
        # normalized-space BG accidentally routed into the anchor.  The
        # guarantee is pool-independent — every legal z satisfies
        # z_max < BG_CLAMP_MIN - 1e-3, the floor this assert reads.  It covers
        # all M slots: on the inference path every anchor crosses the
        # denormalize bridge (utils.last_bg_mgdl_from_context) before arriving.
        assert bool((anchor_bg >= BG_CLAMP_MIN - 1e-3).all()), (
            "anchor_bg below BG_CLAMP_MIN — non-mg/dL value routed into the anchor"
        )

        # --- Patch embedding ---
        x = self.patch_embed(patches)                    # (B, T, D_MODEL)

        # --- Precompute RoPE tables once for all layers ---
        # RoPE is a function of (T, head_dim) only, so the cos/sin tables are
        # identical across every block — we precompute and pass through.
        rope_cos, rope_sin = build_rope_cache(T, HEAD_DIM, device=x.device, dtype=x.dtype)

        # --- Shape the attention mask once for all layers ---
        # SDPA consumes the bool mask directly (True = attend), so nothing
        # additive is materialized and no (B, H, T, T) float is saved for
        # backward.  A shared (T, T) mask broadcasts over batch and head
        # unchanged.  A per-sample (B, T, T) mask must gain the head axis here:
        # broadcasting aligns from the right, so an unexpanded (B, T, T) puts B
        # on the head axis — raising when B != N_HEADS, and silently masking
        # head b with row b's mask when B == N_HEADS.
        assert attn_mask.dtype == torch.bool, (
            f"attn_mask must be bool (True = attend), got {attn_mask.dtype}"
        )
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)                              # (B, 1, T, T)
            assert attn_mask.dim() == 4, (
                f"per-sample mask must be (B, 1, T, T), got {tuple(attn_mask.shape)}"
            )

        # --- Transformer blocks ---
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin, attn_mask)

        # --- Final norm ---
        x = self.final_norm(x)                           # (B, T, D_MODEL)

        # --- BG head: gather the M masked patches by index.  The masked set is
        # arbitrary (forecast, backcast, infill), so this is a gather and not a
        # trailing slice; padded slots gather position 0 and are discarded
        # downstream by ``valid``.
        pred = x.gather(
            1, mask_idx.unsqueeze(-1).expand(B, M, D_MODEL)
        )                                                # (B, M, D_MODEL)
        # Per-patch coefficients (B, M, K, C), expanded across the PATCH_SIZE
        # within-patch timesteps by the fixed orthonormal step_basis (S, K):
        # head_raw[..., s, c] = Σ_k step_basis[s, k] · coeff[..., k, c].  K < S
        # makes the within-patch median curve smooth (no period-2 mode) by design.
        coeff = self.bg_head(pred).view(
            B, M, BG_HEAD_STEP_BASIS_DIM, 1 + 2 * N_SPREADS
        )                                                # (B, M, K, 1 + 2*N_SPREADS)
        head_raw = torch.einsum('sk,bmkc->bmsc', self.step_basis, coeff)
        assert head_raw.shape == (
            B, M, PATCH_SIZE, 1 + 2 * N_SPREADS
        ), f"head_raw shape {tuple(head_raw.shape)} unexpected after basis expansion"

        # Assemble the ascending quantile fan in risk space.  The per-slot anchor
        # f(anchor_bg.detach()) is formed inside the helper; anchor_bg is a
        # constant, so detaching keeps gradient out of it.  mask_idx goes with it:
        # it is what groups the M slots into spans, and the median mode runs per
        # span — without it the whole M axis low-passes as one.
        q_tau, median = assemble_quantiles(head_raw, anchor_bg.detach(), mask_idx)
        if not return_time:
            return q_tau, median
        time_pred = None
        if self.time_head is not None:
            # Run the probe on EVERY (final-normed) gathered masked-patch hidden
            # state (no mean-pool), so each per-patch representation the BG head
            # also reads is forced to encode the absolute clock. With
            # TIME_PROBE_DETACH=False (crucial) the probe gradient back-props into
            # the trunk, shaping that representation to carry circadian phase;
            # True detaches it (read-only diagnostic). Either way the forward
            # VALUE of q_tau/median is unchanged — the head never feeds them.
            # Slot j is patch mask_idx[:, j], so the caller's per-slot hour target
            # must follow mask_idx, not a fixed offset from the context end.
            h = pred if not TIME_PROBE_DETACH else pred.detach()  # (B, M, D_MODEL)
            time_pred = self.time_head(h)                         # (B, M, TIME_PROBE_N_BINS)
        return q_tau, median, time_pred
