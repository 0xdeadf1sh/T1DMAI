"""Engine-agnostic export pieces: modified forward, struct-mask/slot-selection builders, load.
Differs from T1DMAI.forward in 3 ways: additive-float struct mask (NEG_FILL=-30000.0 block,
0.0 attend, no int64); one-hot (M,T) slot_sel instead of mask_idx gather; graph cut at
head_raw (B,M,S,1+2*N_SPREADS) risk space. hidden (B,T,D_MODEL) rides as the LoRA seam.
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
    """Head-raw forward over a loaded ``T1DMAI``: ``forward(patches, struct, slot_sel)``.

    Outputs: 0 ``head_raw`` (B, M, PATCH_SIZE, 1 + 2*N_SPREADS) risk space; 1 ``time_logits``
    (B, M, TIME_PROBE_N_BINS) raw, softmax in Rust; 2 ``hidden`` (B, T, D_MODEL), the LoRA seam.
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
        """patches (B,T,PATCH_DIM) normalized, step-major; struct (T,T) additive, 0.0 attend/
        NEG_FILL block; slot_sel (M,T) one-hot, row j names the patch slot j reads.

        -> head_raw (B,M,PATCH_SIZE,1+2*N_SPREADS) risk space, time_logits (B,M,N_BINS), hidden.
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

        # slot_sel's rows are one-hot and struct is additive bool mask; both stock args recoverable.
        mask_idx = slot_sel.argmax(dim=-1).unsqueeze(0).expand(B, -1)
        head_raw = m.bg_head(step_states(hidden, mask_idx, struct == 0.0))

        # same slot hidden states as the eager return_time=True path
        time_logits = m.time_head(slot_states)                       # (B, M, N_BINS)
        return head_raw, time_logits, hidden


def build_slot_selection(
    mask_idx: "list[int]",
    T: int = MAX_SEQ_LEN,
    m_slots: int = MAX_MASKED_PATCHES,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The (M,T) one-hot selection matrix, one 1.0 per row, naming the patch each slot reads.

    mask_idx is the masked set in ascending patch order. Surplus slots repeat patch 0 (eager
    forward's padding convention, a legal anchor) and are discarded by valid.
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
    """Additive form of create_attention_mask_from_visible -> (T,T), 0.0 attend / neg_fill.

    visible (T,) bool, True where BG observed; is_pad (T,) bool, None = no padding. Only the
    representation changes here — a second transcription of the rule is the duplicate that drifts.
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
    """Label a left-padded fixed-T window -> (visible, is_pad, mask_idx).
    Layout: [0,pad0) padding, [pad0,T-p) the n_ctx real context right-aligned, [T-p,T) future,
    always masked. mask_idx names EXTRA masked patches by position; trailing forecast is added
    unconditionally, union returned ascending.
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
    """A checkpoint's EMA weights into a fresh T1DMAI, eval and frozen -> (model, checkpoint_dict).

    EMA shadow merges over live weights (INFERENCE.md §2.2): every metric was under EMA.
    strict=False tolerates a time_head mismatch only (build-time switch); else asserts.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    ema = ck.get("model_ema_state_dict")
    merged = {k: ema.get(k, v) for k, v in sd.items()} if ema else dict(sd)

    # T1DMAI reads dims from config globals; a drift else surfaces as an opaque shape error.
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
