"""Engine-agnostic pieces of the export: the modified head_raw forward, the
fixed-T struct-mask builder, and checkpoint loading.

The MODIFIED forward differs from ``T1DMAI.forward`` in exactly three ways, all
required by the on-device contract (PLAN §2.4):

1. Its ONLY mask input is the external **struct additive-float mask** (0.0 where a
   position may attend, ``NEG_FILL`` where blocked). The stock forward hands SDPA a
   bool mask and lets it do the blocking; here the caller supplies the additive form
   directly, and it reaches SDPA as the sole additive term on the attention logits —
   position enters through RoPE alone, so there is nothing else for it to be added
   to. ``NEG_FILL = -30000.0`` (not ``-inf``) keeps the fp16 NPU softmax finite; in
   fp32/fp64 ``exp(-30000)`` underflows to 0.0, so it is bit-identical to ``-inf``
   on the blocked positions.
2. The head reads the trailing ``PREDICTION_PATCHES`` patches as a SLICE where the
   stock forward gathers ``M`` masked patches by ``mask_idx``. The export is the
   RIGHT-EDGE SPECIALISATION of the general masked objective
   (``T1DMCOMMON/SPEC/inference.md`` §3.1), so it takes no ``mask_idx``.
3. The graph is **cut at ``head_raw``** (B, P, S, 1+2*N_SPREADS) in risk space — it
   stops before ``assemble_quantiles``, so it needs no ``anchor_bg`` and emits no
   ``q_tau``/``median``. Everything downstream of ``head_raw`` is Rust.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    HEAD_DIM, PREDICTION_PATCHES, PATCH_SIZE, N_SPREADS,
    BG_HEAD_STEP_BASIS_DIM, MAX_CONTEXT_PATCHES, MAX_SEQ_LEN,
)
from model import T1DMAI, build_rope_cache

# fp16-safe additive block fill (PLAN §2.4). exp(-30000) underflows to 0.0 in
# fp32/fp64, so the fan is identical to a -inf mask on blocked positions.
NEG_FILL: float = -30000.0


class HeadRawForward(nn.Module):
    """Wrap a loaded ``T1DMAI`` so ``forward(patches, struct) -> (head_raw, time_logits)``.

    Reuses the model's own submodules (patch_embed, blocks, final_norm, bg_head,
    step_basis buffer, and the co-trained ``time_head`` probe); it only replaces the
    forward *plumbing* — the bool mask becomes an external additive one, the
    masked-patch gather becomes the right-edge slice, and the tail
    (``assemble_quantiles``) is dropped so the graph ends at ``head_raw``.

    The graph emits TWO outputs, in this fixed order:

      0. ``head_raw``    (B, PREDICTION_PATCHES, PATCH_SIZE, 1 + 2*N_SPREADS) risk space.
      1. ``time_logits`` (B, PREDICTION_PATCHES, TIME_PROBE_N_BINS) raw per-prediction-
         patch hour-of-day bin logits from the co-trained time probe. Read off the SAME
         final-normed prediction-patch hidden states the BG head reads (no mean-pool);
         with ``TIME_PROBE_DETACH=False`` its training loss shaped the trunk, so at
         inference it is a circadian-phase belief inferred purely from the trajectory
         (no clock input). Softmax over the 12-bin hour-of-day circle downstream (Rust).
    """

    def __init__(self, model: T1DMAI) -> None:
        super().__init__()
        assert model.time_head is not None, (
            "checkpoint has no time_head (TIME_PROBE_ENABLED was False at train time); "
            "cannot export the time-probe output"
        )
        self.model = model

    def forward(
        self, patches: torch.Tensor, struct: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Args:
            patches: (B, T, PATCH_DIM) already-normalized, step-major.
            struct:  (T, T) additive float mask — 0.0 attend, NEG_FILL block.

        Returns:
            head_raw:    (B, PREDICTION_PATCHES, PATCH_SIZE, 1 + 2*N_SPREADS) risk space.
            time_logits: (B, PREDICTION_PATCHES, TIME_PROBE_N_BINS) hour-of-day bin logits.
        """
        m = self.model
        B, T, _ = patches.shape

        x = m.patch_embed(patches)                                   # (B, T, D_MODEL)

        # The RoPE tables depend only on (T, HEAD_DIM); at a fixed export shape they
        # fold to constants inside the traced graph.
        rope_cos, rope_sin = build_rope_cache(T, HEAD_DIM, device=x.device, dtype=x.dtype)

        # struct goes to SDPA as the sole additive term on the logits; it broadcasts
        # (T,T)->(B,H,T,T) there.
        for block in m.blocks:
            x = block(x, rope_cos, rope_sin, struct)

        x = m.final_norm(x)                                          # (B, T, D_MODEL)
        # RIGHT-EDGE SPECIALISATION (SPEC/inference.md §3.1): the exported graph is
        # cut to the trailing PREDICTION_PATCHES forecast and never gathers by
        # mask_idx, so the head reads a slice where the general forward gathers.
        pred = x[:, -PREDICTION_PATCHES:, :]                         # (B, P, D_MODEL)
        coeff = m.bg_head(pred).view(
            B, PREDICTION_PATCHES, BG_HEAD_STEP_BASIS_DIM, 1 + 2 * N_SPREADS
        )
        head_raw = torch.einsum('sk,bpkc->bpsc', m.step_basis, coeff)

        # Time-of-day probe on the SAME prediction-patch hidden states (mirrors the
        # eager forward's return_time=True path: h = pred, since TIME_PROBE_DETACH is
        # False; detach is a no-op under torch.no_grad export anyway).
        time_logits = m.time_head(pred)                              # (B, P, N_BINS)
        return head_raw, time_logits


def build_struct_mask(
    n_ctx: int,
    T: int = MAX_SEQ_LEN,
    neg_fill: float = NEG_FILL,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pad-aware additive struct mask for the fixed ``T`` window, left-padded.

    Layout over the ``T = MAX_CONTEXT_PATCHES + PREDICTION_PATCHES`` positions:
    ``[0, pad0)`` padding, ``[pad0, C)`` the ``n_ctx`` real context patches (right-
    aligned into the 48 context slots), ``[C, T)`` the ``P`` prediction patches.

    Allowed (0.0) blocks mirror ``utils.create_attention_mask`` on the real
    sub-window: ctx<->ctx bidirectional, pred->ctx full, pred->pred bidirectional,
    ctx->pred blocked. All padding COLUMNS are blocked for every query (so pad
    tokens never influence a prediction token). Padding ROWS are allowed to attend
    to the real context purely so their softmax is never fully-masked (NaN); those
    pad-row outputs are discarded (the head reads only the last ``P`` tokens).

    Returns:
        (T, T) additive float mask, 0.0 attend / ``neg_fill`` block.
    """
    C = MAX_CONTEXT_PATCHES
    P = PREDICTION_PATCHES
    assert 1 <= n_ctx <= C, f"n_ctx must be in [1, {C}], got {n_ctx}"
    assert T == C + P, f"expected T == {C + P}, got {T}"
    pad0 = C - n_ctx
    attend = torch.zeros(T, T, dtype=torch.bool)
    real = slice(pad0, C)
    pred = slice(C, T)
    attend[real, real] = True     # ctx <-> ctx (bidirectional)
    attend[pred, real] = True     # pred -> ctx (full)
    attend[pred, pred] = True     # pred <-> pred (bidirectional)
    if pad0 > 0:
        attend[0:pad0, real] = True   # pad rows -> ctx (anti-NaN; outputs discarded)
    struct = torch.full((T, T), float(neg_fill), dtype=dtype)
    struct[attend] = 0.0
    return struct


def load_model(ckpt_path: str) -> "tuple[T1DMAI, dict]":
    """Load a checkpoint's EMA weights into a fresh ``T1DMAI`` (eval, frozen).

    Merges the EMA shadow over the live weights (INFERENCE.md §2.2) — validation
    and every reported metric were produced under EMA. ``strict=False`` tolerates a
    ``time_head`` present on only one side (the probe is a build-time switch, and a
    checkpoint predating it carries none); the assert below rejects every other
    mismatch. ``HeadRawForward`` then refuses a checkpoint whose probe is missing,
    since the graph emits its logits.

    Returns:
        (model, checkpoint_dict).
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    ema = ck.get("model_ema_state_dict")
    merged = {k: ema.get(k, v) for k, v in sd.items()} if ema else dict(sd)

    # Fail loudly if config.py has drifted away from the checkpoint's architecture
    # (T1DMAI reads its dims from config globals at construction; a mismatch would
    # otherwise surface as an opaque load_state_dict shape error).
    tc = ck.get("training_config") or {}
    import config as _cfg
    for cfg_name, tc_key in (
        ("D_MODEL", "d_model"), ("N_LAYERS", "n_layers"), ("N_HEADS", "n_heads"),
        ("PATCH_SIZE", "patch_size"), ("PREDICTION_PATCHES", "prediction_patches"),
        ("MAX_CONTEXT_PATCHES", "max_context_patches"),
    ):
        if tc_key in tc:
            got = getattr(_cfg, cfg_name)
            assert got == tc[tc_key], (
                f"config.{cfg_name}={got} != checkpoint training_config[{tc_key!r}]="
                f"{tc[tc_key]}; align config.py with the checkpoint before exporting."
            )

    model = T1DMAI()
    missing, unexpected = model.load_state_dict(merged, strict=False)
    # Only the time_head diagnostic may legitimately differ; anything else is a bug.
    bad_missing = [k for k in missing if not k.startswith("time_head")]
    bad_unexpected = [k for k in unexpected if not k.startswith("time_head")]
    assert not bad_missing and not bad_unexpected, (
        f"unexpected state_dict mismatch: missing={bad_missing} unexpected={bad_unexpected}"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck
