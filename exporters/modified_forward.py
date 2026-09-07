"""Engine-agnostic export pieces: the modified forward, the struct-mask and slot-selection builders, load.

The MODIFIED forward differs from ``T1DMAI.forward`` in exactly three ways, all forced by the on-device contract:

1. Its ONLY mask input is the external additive-float struct mask — 0.0 attend, ``NEG_FILL = -30000.0`` block.
   Not ``-inf``, so an fp16 NPU softmax stays finite; in fp32/fp64 ``exp(-30000)`` underflows to 0.0, leaving
   the blocked positions bit-identical to ``-inf``. It reaches SDPA as the sole additive term on the logits —
   position enters through RoPE alone.
2. The head reads its ``M = MAX_MASKED_PATCHES`` slots through an external one-hot ``(M, T)`` matrix instead of
   gathering by ``mask_idx``: the same permutation as a float matmul, so no int64 index crosses the runtime
   boundary. The masked set is arbitrary — trailing forecast, leading backcast, or an infill span between.
3. The graph is cut at ``head_raw`` (B, M, S, 1+2*N_SPREADS), risk space: no ``anchor_bg``, no ``q_tau`` /
   ``median``. Everything downstream is Rust.

``hidden`` — the final-normed state of EVERY patch — rides alongside as the LoRA seam. The decode reads a
span's masked patches and its visible neighbours as spline nodes, so the seam carries the whole window
rather than the slot rows: the consumer gathers the nodes from ``hidden``, builds the step weights and
re-runs ``bg_head`` from the exported weights, and with no adapter attached the two paths agree.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import HEAD_DIM, MAX_MASKED_PATCHES, PREDICTION_PATCHES, MAX_SEQ_LEN
from model import T1DMAI, build_rope_cache
from utils import create_attention_mask_from_visible, step_states

# fp16-safe additive block fill; exp(-30000) underflows to 0.0 in fp32/fp64, matching a -inf mask
NEG_FILL: float = -30000.0


class HeadRawForward(nn.Module):
    """``forward(patches, struct, slot_sel)`` over a loaded ``T1DMAI``; the tail is dropped.

    FOUR outputs, fixed order: 0 ``head_raw`` (B, M, S, 1+2*N_SPREADS) risk; 1 ``time_logits``
    (B, M, N_BINS); 2 ``hidden`` (B, T, D_MODEL), the LoRA seam; 3 ``crossing_logits`` (B, M, S, 2).
    """

    def __init__(self, model: T1DMAI) -> None:
        super().__init__()
        assert model.time_head is not None, (
            "checkpoint has no time_head (TIME_PROBE_ENABLED was False at train time); "
            "cannot export the time-probe output"
        )
        assert model.crossing_head is not None, (
            "checkpoint has no crossing_head (CROSSING_HEAD_ENABLED was False at train time); "
            "cannot export the crossing output"
        )
        self.model = model

    def forward(
        self, patches: torch.Tensor, struct: torch.Tensor, slot_sel: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """``patches`` (B, T, PATCH_DIM) normalized, step-major; ``struct`` (T, T) additive, 0.0 attend /
        NEG_FILL block; ``slot_sel`` (M, T) one-hot, row j names the patch slot j reads.

        -> ``head_raw`` (B, M, PATCH_SIZE, 1 + 2*N_SPREADS) risk space, ``time_logits``
        (B, M, TIME_PROBE_N_BINS), ``hidden`` (B, T, D_MODEL).
        """
        m = self.model
        B = patches.shape[0]
        T = patches.shape[1]

        x = m.patch_embed(patches)                                   # (B, T, D_MODEL)

        # tables depend only on (T, HEAD_DIM), so at a fixed export shape they fold to constants
        rope_cos, rope_sin = build_rope_cache(T, HEAD_DIM, device=x.device, dtype=x.dtype)

        # sole additive term on the logits; broadcasts (T,T)->(B,H,T,T) inside SDPA
        for block in m.blocks:
            x = block(x, rope_cos, rope_sin, struct)

        hidden = m.final_norm(x)                                     # (B, T, D_MODEL)
        # one-hot rows make this exactly the stock forward's gather
        slot_states = torch.einsum('mt,btd->bmd', slot_sel, hidden)  # (B, M, D_MODEL)

        # slot_sel's rows are one-hot and struct is the additive form of the bool mask, so both of
        # the stock forward's arguments are recoverable — which keeps ONE node rule, step_states'.
        mask_idx = slot_sel.argmax(dim=-1).unsqueeze(0).expand(B, -1)
        h_steps = step_states(hidden, mask_idx, struct == 0.0)
        head_raw = m.bg_head(h_steps)

        # same slot hidden states as the eager return_time=True path
        time_logits = m.time_head(slot_states)                       # (B, M, N_BINS)
        crossing_logits = m.crossing_head(h_steps)                   # (B, M, S, N_CROSSING)
        return head_raw, time_logits, hidden, crossing_logits


def build_slot_selection(
    mask_idx: "list[int]",
    T: int = MAX_SEQ_LEN,
    m_slots: int = MAX_MASKED_PATCHES,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The ``(M, T)`` one-hot selection matrix — one 1.0 per row — naming the patch each head slot reads.

    ``mask_idx`` is the masked set in ascending patch order. Surplus slots repeat patch 0, the eager forward's
    padding convention: a legal anchor, and the output discarded by ``valid``.
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
    """The additive form of :func:`utils.create_attention_mask_from_visible` -> ``(T, T)``, 0.0 attend / ``neg_fill``.

    The training-time function builds the bool mask; only the representation changes here. A second
    transcription of the rule is the duplicate that drifts.
    ``visible`` ``(T,)`` bool, True where the patch's BG is observed; ``is_pad`` ``(T,)`` bool, None = no padding.
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
    """Label a left-padded fixed-``T`` window -> ``(visible, is_pad, mask_idx)``.

    Layout: ``[0, pad0)`` padding, ``[pad0, T - p)`` the ``n_ctx`` real context patches right-aligned,
    ``[T - p, T)`` the ``p`` future patches, never observed and so always masked.
    ``mask_idx`` names EXTRA masked patches by absolute position; the trailing forecast is added
    unconditionally and the union comes back ascending.
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
    """Right-edge shim: struct mask whose only masked span is the trailing forecast.

    Kept for the export self-check against the stock ``create_attention_mask``.
    """
    visible, is_pad, _ = window_labels(n_ctx, None, T)
    return build_struct_mask_from_visible(visible, is_pad, neg_fill, dtype)


def load_model(ckpt_path: str) -> "tuple[T1DMAI, dict]":
    """A checkpoint's EMA weights into a fresh ``T1DMAI``, eval and frozen -> ``(model, checkpoint_dict)``.

    The EMA shadow merges over the live weights (INFERENCE.md §2.2): every reported metric was produced
    under EMA. ``strict=False`` tolerates a ``time_head`` on one side only — the probe is a build-time
    switch — and the assert below rejects every other mismatch.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    ema = ck.get("model_ema_state_dict")
    merged = {k: ema.get(k, v) for k, v in sd.items()} if ema else dict(sd)

    # T1DMAI reads its dims from config globals at construction, so config.py drifting from the
    # checkpoint would otherwise surface as an opaque load_state_dict shape error
    tc = ck.get("training_config") or {}
    import config as _cfg
    assert ck.get("arch_version") in (None, _cfg.ARCH_VERSION), (
        f"checkpoint arch_version {ck.get('arch_version')!r} != config {_cfg.ARCH_VERSION!r}"
    )
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
