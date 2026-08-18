"""Engine-agnostic pieces of the export: the modified forward, the struct-mask
builder, the slot-selection builder, and checkpoint loading.

The MODIFIED forward differs from ``T1DMAI.forward`` in exactly three ways, all
required by the on-device contract:

1. Its ONLY mask input is the external **struct additive-float mask** (0.0 where a
   position may attend, ``NEG_FILL`` where blocked). The stock forward hands SDPA a
   bool mask and lets it do the blocking; here the caller supplies the additive form
   directly, and it reaches SDPA as the sole additive term on the attention logits —
   position enters through RoPE alone, so there is nothing else for it to be added
   to. ``NEG_FILL = -30000.0`` (not ``-inf``) keeps the fp16 NPU softmax finite; in
   fp32/fp64 ``exp(-30000)`` underflows to 0.0, so it is bit-identical to ``-inf``
   on the blocked positions.
2. The head reads the ``M = MAX_MASKED_PATCHES`` masked patches through an external
   **one-hot selection matrix** where the stock forward gathers them by ``mask_idx``.
   A gather needs an int64 index tensor crossing the runtime boundary and lands on
   the portable kernel; ``slot_sel @ x`` is the same permutation as a float matmul,
   so it delegates with the rest of the graph and the app never builds an int64
   tensor. The masked set is therefore arbitrary — a trailing forecast, a leading
   backcast, or an infill span anywhere between visible patches.
3. The graph is **cut at ``head_raw``** (B, M, S, 1+2*N_SPREADS) in risk space — it
   stops before ``assemble_quantiles``, so it needs no ``anchor_bg`` and emits no
   ``q_tau``/``median``. Everything downstream of ``head_raw`` is Rust.

It emits ``slot_hidden`` (the final-normed hidden state at each selected slot)
alongside ``head_raw``. That tensor is the LoRA seam: the consumer re-runs
``bg_head`` itself, from the exported head weights, with an adapter in front of it.
With no adapter attached the two paths agree, which is what the consumer checks at
load.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    HEAD_DIM, N_SPREADS, MAX_MASKED_PATCHES,
    BG_HEAD_STEP_BASIS_DIM, PREDICTION_PATCHES, MAX_SEQ_LEN,
)
from model import T1DMAI, build_rope_cache
from utils import create_attention_mask_from_visible

# fp16-safe additive block fill. exp(-30000) underflows to 0.0 in fp32/fp64, so the
# fan is identical to a -inf mask on blocked positions.
NEG_FILL: float = -30000.0


class HeadRawForward(nn.Module):
    """Wrap a loaded ``T1DMAI`` so ``forward(patches, struct, slot_sel)`` returns
    ``(head_raw, time_logits, slot_hidden)``.

    Reuses the model's own submodules (patch_embed, blocks, final_norm, bg_head,
    step_basis buffer, and the co-trained ``time_head`` probe); it only replaces the
    forward *plumbing* — the bool mask becomes an external additive one, the
    masked-patch gather becomes a selection matmul, and the tail
    (``assemble_quantiles``) is dropped so the graph ends at ``head_raw``.

    The graph emits THREE outputs, in this fixed order:

      0. ``head_raw``    (B, M, PATCH_SIZE, 1 + 2*N_SPREADS) risk space.
      1. ``time_logits`` (B, M, TIME_PROBE_N_BINS) raw per-slot hour-of-day bin
         logits from the co-trained time probe. Read off the SAME final-normed
         hidden states the BG head reads (no mean-pool); with
         ``TIME_PROBE_DETACH=False`` its training loss shaped the trunk, so at
         inference it is a circadian-phase belief inferred purely from the
         trajectory (no clock input). Softmax over the hour-of-day circle
         downstream (Rust).
      2. ``slot_hidden`` (B, M, D_MODEL) final-normed hidden state per slot — the
         LoRA seam. The consumer needs it only when an adapter is attached; the
         plain forecast reads ``head_raw`` and ignores it.
    """

    def __init__(self, model: T1DMAI) -> None:
        super().__init__()
        assert model.time_head is not None, (
            "checkpoint has no time_head (TIME_PROBE_ENABLED was False at train time); "
            "cannot export the time-probe output"
        )
        self.model = model

    def forward(
        self, patches: torch.Tensor, struct: torch.Tensor, slot_sel: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Args:
            patches:  (B, T, PATCH_DIM) already-normalized, step-major.
            struct:   (T, T) additive float mask — 0.0 attend, NEG_FILL block.
            slot_sel: (M, T) one-hot rows; row j selects the patch head slot j reads.

        Returns:
            head_raw:    (B, M, PATCH_SIZE, 1 + 2*N_SPREADS) risk space.
            time_logits: (B, M, TIME_PROBE_N_BINS) hour-of-day bin logits.
            slot_hidden: (B, M, D_MODEL) final-normed hidden state per slot.
        """
        m = self.model
        B = patches.shape[0]
        T = patches.shape[1]

        x = m.patch_embed(patches)                                   # (B, T, D_MODEL)

        # The RoPE tables depend only on (T, HEAD_DIM); at a fixed export shape they
        # fold to constants inside the traced graph.
        rope_cos, rope_sin = build_rope_cache(T, HEAD_DIM, device=x.device, dtype=x.dtype)

        # struct goes to SDPA as the sole additive term on the logits; it broadcasts
        # (T,T)->(B,H,T,T) there.
        for block in m.blocks:
            x = block(x, rope_cos, rope_sin, struct)

        x = m.final_norm(x)                                          # (B, T, D_MODEL)
        # The general masked objective: slot j reads whichever patch slot_sel's row j
        # names. One-hot rows make this exactly the stock forward's gather.
        slot_hidden = torch.einsum('mt,btd->bmd', slot_sel, x)       # (B, M, D_MODEL)
        n_slots = slot_sel.shape[0]
        coeff = m.bg_head(slot_hidden).view(
            B, n_slots, BG_HEAD_STEP_BASIS_DIM, 1 + 2 * N_SPREADS
        )
        head_raw = torch.einsum('sk,bmkc->bmsc', m.step_basis, coeff)

        # Time-of-day probe on the SAME slot hidden states (mirrors the eager
        # forward's return_time=True path: h = the gathered slots, since
        # TIME_PROBE_DETACH is False; detach is a no-op under torch.no_grad export).
        time_logits = m.time_head(slot_hidden)                       # (B, M, N_BINS)
        return head_raw, time_logits, slot_hidden


def build_slot_selection(
    mask_idx: "list[int]",
    T: int = MAX_SEQ_LEN,
    m_slots: int = MAX_MASKED_PATCHES,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The ``(M, T)`` one-hot selection matrix naming the patch each head slot reads.

    ``mask_idx`` is the masked set in ascending patch order. Surplus slots repeat
    patch 0 — the eager forward's own padding convention: a padded slot gathers patch
    0, carries a legal anchor, and its output is discarded by ``valid``.

    Returns:
        (M, T) float, one 1.0 per row.
    """
    assert 1 <= len(mask_idx) <= m_slots, (
        f"masked set of {len(mask_idx)} patches does not fit {m_slots} head slots"
    )
    assert all(0 <= i < T for i in mask_idx), f"mask_idx out of range for T={T}"
    sel = torch.zeros(m_slots, T, dtype=dtype)
    for j in range(m_slots):
        sel[j, mask_idx[j] if j < len(mask_idx) else 0] = 1.0
    return sel


def build_struct_mask_from_visible(
    visible: torch.Tensor,
    is_pad: "torch.Tensor | None" = None,
    neg_fill: float = NEG_FILL,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The additive form of :func:`utils.create_attention_mask_from_visible`.

    One rule, one implementation: the bool mask is built by the training-time
    function and only its representation changes here (True -> 0.0, False ->
    ``neg_fill``). A second transcription of the four-line rule is exactly the
    duplicate that drifts.

    Args:
        visible: ``(T,)`` bool, True where the patch's BG is observed.
        is_pad:  ``(T,)`` bool left-padding flags, or None for no padding.

    Returns:
        (T, T) additive float mask, 0.0 attend / ``neg_fill`` block.
    """
    assert visible.ndim == 1 and visible.dtype == torch.bool, (
        f"visible must be (T,) bool, got {tuple(visible.shape)} {visible.dtype}"
    )
    pad = torch.zeros_like(visible) if is_pad is None else is_pad
    attend = create_attention_mask_from_visible(visible[None, :], pad[None, :])[0]
    struct = torch.full(attend.shape, float(neg_fill), dtype=dtype)
    struct[attend] = 0.0
    return struct


def window_labels(
    n_ctx: int,
    mask_idx: "list[int] | None" = None,
    T: int = MAX_SEQ_LEN,
    p: int = PREDICTION_PATCHES,
) -> "tuple[torch.Tensor, torch.Tensor, list[int]]":
    """Label a left-padded fixed-``T`` window: ``(visible, is_pad, mask_idx)``.

    Layout over the ``T`` positions: ``[0, pad0)`` padding, ``[pad0, T - p)`` the
    ``n_ctx`` real context patches (right-aligned into the context slots), and
    ``[T - p, T)`` the ``p`` future patches, which carry no observed BG at all and
    are therefore always masked.

    ``mask_idx`` names EXTRA masked patches (a backcast or infill span) by absolute
    position; the trailing forecast is added unconditionally and the union is
    returned in ascending order.
    """
    c = T - p
    assert 1 <= n_ctx <= c, f"n_ctx must be in [1, {c}], got {n_ctx}"
    pad0 = c - n_ctx
    is_pad = torch.zeros(T, dtype=torch.bool)
    is_pad[:pad0] = True
    visible = torch.ones(T, dtype=torch.bool)
    visible[c:] = False                       # the future patches are never observed
    for i in (mask_idx or []):
        assert pad0 <= i < c, f"extra masked patch {i} is not a real context patch"
        visible[i] = False
    idx = sorted(set(list(mask_idx or []) + list(range(c, T))))
    return visible, is_pad, idx


def build_struct_mask(
    n_ctx: int,
    T: int = MAX_SEQ_LEN,
    neg_fill: float = NEG_FILL,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Right-edge shim: the struct mask for a left-padded window whose only masked
    span is the trailing forecast. Kept for the export self-check, which compares it
    against the stock ``create_attention_mask``."""
    visible, is_pad, _ = window_labels(n_ctx, None, T)
    return build_struct_mask_from_visible(visible, is_pad, neg_fill, dtype)


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
        ("MAX_MASKED_PATCHES", "max_masked_patches"),
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
