"""Read-only attribution for one masked span: attention rollout, and per-channel grad ⊙ input.

Attention has no channel axis — ``patch_embed`` mixes every feature into one token — so
"which channel" is the gradient's answer.  Feat 4 (``bg_masked``) is a bit, not a channel.
``rollout``: Abnar & Zuidema (2020), ``0.5·A + 0.5·I`` per layer; a heuristic, not causal.
``grad ⊙ input`` blind spot: an input at its normalized mean contributes 0 at any gradient.
``model.forward`` detaches the anchor, so a bare backward gets the BG row's SIGN wrong.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Sequence

import numpy as np
import torch

from config import N_INPUT_FEATURES
from model import T1DMAI
from normalization import CHANNEL_NAMES
from utils import kovatchev_f

# ``inference``'s chokepoint announces feat 4, checks it against the requested set, and
# crosses the anchor's denormalize bridge; rebuilding the sample here would be a second copy.
from inference import _resolve_mask_spans, _run_forward


@dataclass
class Attribution:
    """What one masked span read, over the ``T`` patches of its own window.

    ``channels`` is signed: positive raises the span's median risk, negative lowers it.
    It cannot answer on ``masked_patches`` rather than answering 0 — feat 0 is a literal
    0.0 there, so ``grad ⊙ input`` is 0 whatever the gradient.
    """
    where: np.ndarray          # (T,) rollout-composed attention mass per patch
    per_layer: np.ndarray      # (N_LAYERS, T) head-mean attention, layer by layer
    channels: np.ndarray       # (T, len(CHANNEL_NAMES)) signed grad ⊙ input
    channel_share: np.ndarray  # (len(CHANNEL_NAMES),) |saliency| share, sums to 1
    channel_names: list[str]   # the (T, C) column axis, feat order
    span: tuple[int, int]      # (start_patch, length) the maps explain
    slot_patches: np.ndarray   # (P,) patch index of every slot that span owns
    masked_patches: np.ndarray # (M,) every masked patch of the window, all spans
    n_ctx: int                 # context patches; the window is n_ctx + PREDICTION_PATCHES
    window_offset: int         # this window's patch 0, on the caller's own patch axis
    anchor_patch: int          # context patch the span's anchor was read from
    anchor_in_graph: bool      # False if the anchor's own term had to be dropped


@contextmanager
def capture_attention(model: T1DMAI) -> Generator[list[list[torch.Tensor]], None, None]:
    """Arm every block's attention tap; disarmed on the way out, exception or not.

    Yields one list per layer, layer order, each holding the ``(B, H, T, T)`` weights of
    every forward run inside the block.
    """
    attn_modules = [block.attn for block in model.blocks]
    captured: list[list[torch.Tensor]] = [[] for _ in attn_modules]
    for module, sink in zip(attn_modules, captured):
        module.attn_sink = sink
    try:
        yield captured
    finally:
        for module in attn_modules:
            module.attn_sink = None


def rollout(head_mean: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compose per-layer attention into one patch→patch flow map.

    Abnar & Zuidema (2020): ``0.5·A + 0.5·I`` per layer, renormalized, multiplied in order —
    the last layer alone credits a patch for mass that never travelled through attention.
    ``head_mean``: ``L`` × ``(T, T)``, layer order, head-averaged, row-stochastic.
    Returns ``(T, T)`` row-stochastic.
    """
    assert len(head_mean) > 0, "rollout needs at least one layer"
    T = head_mean[0].shape[-1]
    eye = torch.eye(T, dtype=head_mean[0].dtype, device=head_mean[0].device)
    composed: torch.Tensor | None = None
    for layer in head_mean:
        assert layer.shape == (T, T), (
            f"every layer must be (T, T)=({T}, {T}), got {tuple(layer.shape)}"
        )
        residual = 0.5 * layer + 0.5 * eye
        residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        composed = residual if composed is None else residual @ composed
    assert composed is not None
    return composed


def channel_saliency(
    grad: torch.Tensor, patches: torch.Tensor,
) -> torch.Tensor:
    """Fold a ``(T, PATCH_DIM)`` gradient back onto the channel axis, sign intact.

    ``PATCH_DIM = PATCH_SIZE × N_INPUT_FEATURES`` step-major, so feat ``f`` owns the stride
    ``[:, f::N_INPUT_FEATURES]``.  ``grad`` is ∂target/∂patches, ``patches`` the normalized
    inputs it was taken at.  Returns ``(T, len(CHANNEL_NAMES))`` signed, feat order — feat 4
    (``bg_masked``) is a mask announcement, not a channel, and has no column.
    """
    assert grad.shape == patches.shape, (
        f"grad {tuple(grad.shape)} and patches {tuple(patches.shape)} must match"
    )
    assert grad.ndim == 2 and grad.shape[-1] % N_INPUT_FEATURES == 0, (
        f"expected (T, PATCH_SIZE × {N_INPUT_FEATURES}), got {tuple(grad.shape)}"
    )
    contribution = grad * patches
    return torch.stack(
        [contribution[:, f::N_INPUT_FEATURES].sum(dim=-1)
         for f in range(len(CHANNEL_NAMES))],
        dim=-1,
    )


# ``z·std + mean`` vs ``f(anchor_bg)``: they differ only by a float32 round trip through
# ``kovatchev_f_inv``.  Beyond this the inverse hit its physical clamp, the anchor is no
# longer linear in the cell, and its term is dropped rather than approximated.
_ANCHOR_RECONSTRUCT_ATOL = 1e-3


def _anchor_term(
    patches: torch.Tensor, out: dict, normalization_stats: dict,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """The anchor as a live function of the ``(1, T, PATCH_DIM)`` ``patches`` leaf.

    Returns ``(anchor_patch, live, const)`` — ``(M,)`` context patch per slot, ``(M,)`` anchor
    rebuilt from the leaf, ``(M,)`` risk-space anchor the forward actually used.
    """
    bg_stats = normalization_stats[CHANNEL_NAMES[0]]
    anchor_patch = out['anchor_patch']
    # feat 0 is bg; a patch is step-major, so the cell's column is its step's block start.
    column = out['anchor_within'] * N_INPUT_FEATURES
    device = patches.device
    rows = torch.as_tensor(anchor_patch, device=device)
    cols = torch.as_tensor(column, device=device)
    live = patches[0, rows, cols] * float(bg_stats['std']) + float(bg_stats['mean'])
    return anchor_patch, live, kovatchev_f(out['anchor_bg'])


def _span_slots(
    mask_idx: torch.Tensor, valid: torch.Tensor, span: tuple[int, int],
) -> torch.Tensor:
    """``(M,)`` bool: True on a valid slot whose patch lies inside ``span`` (start_patch, length)."""
    start, length = span
    return valid & (mask_idx >= start) & (mask_idx < start + length)


def explain(
    model: T1DMAI,
    context: torch.Tensor,
    normalization_stats: dict[str, dict[str, float]],
    overrides: dict[int, torch.Tensor] | None = None,
    mask_spans: Sequence[tuple[int, int]] | None = None,
    span: tuple[int, int] | None = None,
    device: torch.device | None = None,
    window_offset: int = 0,
) -> Attribution:
    """Explain one masked span of one forward: where it read, and from which channels.

    ``context`` ``(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)`` normalized; ``overrides`` the
    announced doses as ``inference.predict`` takes them ({0: carb, 1: insulin, 2: exercise}
    → normalized).  ``mask_spans=None`` is the trailing forecast span alone; ``span=None``
    the trailing span of the set.  Every index returned is in WINDOW coordinates over the
    ``n_ctx + PREDICTION_PATCHES`` window; ``window_offset`` is the shift onto the caller's
    own axis, non-zero only once an autoregressive roll has slid the context.
    Raises ``ValueError`` when ``span`` owns no valid head slot — nothing was predicted there.
    """
    n_ctx = int(context.shape[0])
    spans = _resolve_mask_spans(mask_spans, n_ctx)
    # ``spans[-1]``, never a restated ``(n_ctx, PREDICTION_PATCHES)``: the trailing span may
    # start before ``n_ctx``, and the restated pair would miss its left half.
    target_span = (spans[-1] if span is None else (int(span[0]), int(span[1])))

    with capture_attention(model) as captured:
        out = _run_forward(
            model, context, normalization_stats, overrides=overrides,
            mask_spans=spans, device=device, grad=True,
        )
        assert all(len(calls) == 1 for calls in captured), (
            f"expected one forward per layer, captured "
            f"{[len(calls) for calls in captured]} — the maps would describe "
            f"whichever forward happened to land first"
        )
        head_mean = [calls[0][0].mean(dim=0).detach() for calls in captured]
    for calls in captured:
        calls.clear()

    sel = _span_slots(out['mask_idx'], out['valid'], target_span)
    if not bool(sel.any()):
        raise ValueError(
            f"span {target_span} owns no head slot of the masked set {spans} — "
            f"nothing was predicted there"
        )
    slot_patches = out['mask_idx'][sel].detach().cpu().numpy()

    rows = torch.as_tensor(slot_patches, device=head_mean[0].device)
    where = rollout(head_mean)[rows].mean(dim=0)
    per_layer = torch.stack([layer[rows].mean(dim=0) for layer in head_mean])

    # One scalar to differentiate: this span's median risk.  Its slots share one anchor, so
    # the pool is the span's own trajectory, not a mixture across spans.
    patches = out['patches']
    target = out['median'][sel].mean()
    anchor_patch, anchor_live, anchor_const = _anchor_term(
        patches, out, normalization_stats,
    )
    # ``anchor_live - anchor_const`` is 0 in VALUE and carries the anchor cell's gradient:
    # restores the persistence path without moving the scalar being differentiated.
    # Checked on this span's slots alone — another span's clamped anchor says nothing here.
    anchor_in_graph = bool(torch.allclose(
        anchor_live[sel].detach(), anchor_const[sel],
        rtol=0, atol=_ANCHOR_RECONSTRUCT_ATOL,
    ))
    if anchor_in_graph:
        target = target + (anchor_live[sel] - anchor_const[sel]).mean()
    grad, = torch.autograd.grad(target, patches)

    channels = channel_saliency(grad[0].detach(), patches[0].detach())
    magnitude = channels.abs().sum(dim=0)
    share = magnitude / magnitude.sum().clamp_min(1e-12)

    return Attribution(
        where=where.cpu().numpy(),
        per_layer=per_layer.cpu().numpy(),
        channels=channels.cpu().numpy(),
        channel_share=share.cpu().numpy(),
        channel_names=list(CHANNEL_NAMES),
        span=(int(target_span[0]), int(target_span[1])),
        slot_patches=slot_patches,
        masked_patches=out['mask_idx'][out['valid']].detach().cpu().numpy(),
        n_ctx=n_ctx,
        window_offset=int(window_offset),
        anchor_patch=int(anchor_patch[sel.cpu().numpy()][0]),
        anchor_in_graph=anchor_in_graph,
    )


# Ink floor for the attention ramp, as a multiple of an even share.  Three decades, not two:
# a trained model discounts a masked patch ~100-fold, and a floor at 1/100 of a share renders
# exactly that residue as nothing.
_SHARE_RAMP_FLOOR = 0.001


def share_ramp(mass: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Map an attention row onto [0, 1] by its multiple of an even share.

    The reference is ``1/T``, not zero, and the ramp is logarithmic in that multiple: the row
    spans decades, and a linear ramp renders most of the window identically black.
    ``mass`` ``(T,)`` sums to 1; ``[lo, hi)`` are the visible patches, which set the top of
    the scale — never below an even share, however dark the view.
    """
    n = int(mass.shape[0])
    if n == 0:
        return np.zeros_like(mass)
    multiple = mass * float(n)
    visible = multiple[max(0, lo):max(0, hi)]
    top = max(float(visible.max()) if visible.size else 0.0, 1.0)
    floor_log = float(np.log(_SHARE_RAMP_FLOOR))
    span = float(np.log(top)) - floor_log
    if span <= 0:
        return np.zeros_like(mass)
    ramped = (np.log(np.maximum(multiple, _SHARE_RAMP_FLOOR)) - floor_log) / span
    return np.clip(ramped, 0.0, 1.0)


def signed_ramp(values: np.ndarray, scale: float) -> np.ndarray:
    """Map signed saliency onto [-1, 1]; ``scale`` is the magnitude mapping to ±1, beyond it clips.

    Channels must share one scale for "which channel" to be readable, but BG outweighs the
    sparse dose rows enough that a linear ramp renders them blank; the square root keeps the
    ordering and the sign while lifting them into view.
    """
    if scale <= 0:
        return np.zeros_like(values)
    normalized = np.clip(values / scale, -1.0, 1.0)
    return np.sign(normalized) * np.sqrt(np.abs(normalized))
