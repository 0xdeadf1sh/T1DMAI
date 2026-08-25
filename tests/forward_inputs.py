"""Forward-input fixtures for the masked-BG contract::

    forward(patches, attn_mask, anchor_bg, mask_idx) -> (q_tau, median)
      patches   (B, T, PATCH_DIM)   PATCH_DIM = PATCH_SIZE * N_INPUT_FEATURES
      attn_mask (B, T, T) bool      T <= MAX_SEQ_LEN, never == it
      anchor_bg (B, M) mg/dL        one anchor per masked patch
      mask_idx  (B, M) int64        the patch index each head slot reads

Both fixtures withhold feat 0 on masked patches and write the feat-4 bit into all
``PATCH_SIZE`` step-major columns of that patch, as ``data._build_sample`` does.
"""

from typing import Sequence

import torch

from config import (
    MAX_MASKED_PATCHES,
    MIN_CONTEXT_PATCHES,
    N_INPUT_FEATURES,
    NON_MASKABLE_FEATS,
    PATCH_DIM,
    PREDICTION_PATCHES,
)
from data import BG_MASKED_FEAT
import utils

Span = tuple[int, int]


def expand_spans(spans: Sequence[Span]) -> list[int]:
    """The masked patch indices of ``[(start, length), ...]``, left to right."""
    out: list[int] = []
    for start, length in spans:
        out.extend(range(start, start + length))
    return out


def slots(
    spans: Sequence[Span], seq_len: int, M: int | None = None,
    anchor_mgdl: float = 120.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand spans into ``M`` head slots: ``(mask_idx, valid, anchor_bg)``.

    Padded slots take ``anchor_mgdl`` too — the forward's units tripwire reads all
    ``M``, so a padded slot must still hold legal mg/dL.
    """
    idx = expand_spans(spans)
    if M is None:
        M = MAX_MASKED_PATCHES
    assert len(idx) <= M, f"{len(idx)} masked patches exceeds M={M}"
    assert all(0 <= p < seq_len for p in idx), f"span outside [0, {seq_len})"
    mask_idx = torch.zeros(M, dtype=torch.int64)
    valid = torch.zeros(M, dtype=torch.bool)
    mask_idx[: len(idx)] = torch.tensor(idx, dtype=torch.int64)
    valid[: len(idx)] = True
    anchor_bg = torch.full((M,), float(anchor_mgdl))
    return mask_idx, valid, anchor_bg


def announce(patches: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    """Withhold feat 0, set the feat-4 bit on ``masked`` (B, T) patches, in place."""
    # ``patches[..., f::N]`` is a strided VIEW, so these writes land in ``patches``
    for feat in NON_MASKABLE_FEATS:
        patches[..., feat::N_INPUT_FEATURES][masked] = 0.0
    bit = patches[..., BG_MASKED_FEAT::N_INPUT_FEATURES]
    bit[masked] = 1.0
    bit[~masked] = 0.0
    return patches


def right_edge_inputs(
    B: int = 2, n_ctx: int | None = None, M: int | None = None,
    anchor_mgdl: float = 120.0, seed: int | None = None,
    all_true_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(patches, attn_mask, anchor_bg, mask_idx)``; one masked span of
    ``PREDICTION_PATCHES`` ending at ``T - 1``, anchor in mg/dL.

    Default ``M = PREDICTION_PATCHES`` leaves no padded slot.
    """
    if n_ctx is None:
        n_ctx = MIN_CONTEXT_PATCHES
    T = n_ctx + PREDICTION_PATCHES
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    patches = torch.randn(B, T, PATCH_DIM, generator=gen)
    masked = torch.zeros(B, T, dtype=torch.bool)
    masked[:, n_ctx:] = True
    announce(patches, masked)
    if all_true_mask:
        attn_mask = torch.ones(T, T, dtype=torch.bool)
    else:
        attn_mask = utils.create_attention_mask_from_visible(~masked)
    mask_idx, _, anchor_bg = slots(
        [(n_ctx, PREDICTION_PATCHES)], T, M=M or PREDICTION_PATCHES,
        anchor_mgdl=anchor_mgdl,
    )
    return (
        patches,
        attn_mask,
        anchor_bg.unsqueeze(0).expand(B, -1).contiguous(),
        mask_idx.unsqueeze(0).expand(B, -1).contiguous(),
    )


def masked_set_inputs(
    spans_per_row: Sequence[Sequence[Span]], n_ctx: int, M: int | None = None,
    anchor_mgdl: float = 120.0, seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(patches, attn_mask, anchor_bg, mask_idx, valid)``; one ``(start_patch,
    length)`` list per batch row.

    Surplus slots are padded — the sampler leaves 41.8% padded on the average sample.
    """
    B = len(spans_per_row)
    T = n_ctx + PREDICTION_PATCHES
    if M is None:
        M = MAX_MASKED_PATCHES
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    patches = torch.randn(B, T, PATCH_DIM, generator=gen)
    masked = torch.zeros(B, T, dtype=torch.bool)
    mask_idx = torch.zeros(B, M, dtype=torch.int64)
    valid = torch.zeros(B, M, dtype=torch.bool)
    anchor_bg = torch.full((B, M), float(anchor_mgdl))
    for b, spans in enumerate(spans_per_row):
        idx, val, anc = slots(spans, T, M=M, anchor_mgdl=anchor_mgdl)
        mask_idx[b], valid[b], anchor_bg[b] = idx, val, anc
        masked[b, expand_spans(spans)] = True
    announce(patches, masked)
    attn_mask = utils.create_attention_mask_from_visible(~masked)
    return patches, attn_mask, anchor_bg, mask_idx, valid
