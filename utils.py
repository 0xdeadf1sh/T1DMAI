"""
T1DMAI Utilities — seed hashing, attention mask, risk transform, weight EMA.
============================================================================

This module is the pile of small, well-defined helpers that didn't earn a
module of their own:

* ``compute_patient_seed`` — turns ``(master_seed, step, position)`` into a
  deterministic 63-bit patient seed via SHA-256.  The training loop uses this
  for every sample, so the same training run always touches the same
  patients in the same order.  The model itself doesn't see the seed —
  it is only used to drive the simulator deterministically.

* ``create_attention_mask_from_visible`` — produces the attention mask used by
  every temporal self-attention layer, from a ``(B, T)`` VISIBLE/MASKED/PAD
  labelling.  Visible positions see visible positions, masked positions see
  everything real, padding is blocked in both directions bar the diagonal.
  ``create_attention_mask(n_context, n_prediction)`` is the right-edge
  (forecast) shim over it.  Neither memoizes.

* ``kovatchev_f`` / ``kovatchev_f_target`` / ``kovatchev_f_inv`` — the sole
  bridge between mg/dL physical space (b) and Kovatchev *risk* space (c).
  ``f`` is the symmetrizing risk transform of Kovatchev et al.; it bakes the
  clinical hypo>hyper asymmetry into the loss geometry (equal risk-distance =
  larger danger at low BG).  ``f`` is NEVER differentiated — targets and the
  ``last_bg`` anchor are constants.  ``kovatchev_f`` carries the *units
  tripwire* (a hard assert that its argument is physical mg/dL, so a z-scored
  value trips it loudly); ``kovatchev_f_target`` is the target-path variant
  with a physical clamp backstop; ``kovatchev_f_inv`` is the only (c)→(b)
  inverse, clamping the risk input first and the mg/dL output second.

* ``assemble_quantiles`` — turns the BG head's raw per-slot output (a median
  delta + spreads) into an ascending 7-quantile fan in risk space, anchored
  per masked slot at ``f(anchor_bg)``.  Returns ``(q_tau, median)``.  The
  smooth-basis median projection runs PER SPAN, at the per-span dimension
  ``global_median_dim(L)``.

* ``last_bg_mgdl_from_context`` — denormalizes a context ``bg_absolute`` cell
  into mg/dL, the shared (a)→(b) bridge that produces the anchor at inference.
  Vectorised over an arbitrary set of ``(patch, step)`` cells; the rightmost
  cell is the default.

* ``ModelEMA`` — maintains a shadow copy of the model's float parameters and
  buffers, blended after every accepted optimizer step as
  ``θ_ema ← decay·θ_ema + (1-decay)·θ``.  Validation runs under EMA to
  smooth out per-batch optimizer jitter on threshold metrics like hypo
  recall.  ``apply_to`` is a context manager that swaps the shadow into
  the model for the duration of a ``with`` block, then restores the live
  weights so training keeps going on the un-smoothed parameters.
"""

import hashlib
import math
import warnings
from contextlib import contextmanager
from typing import Iterator, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn

from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX


# Kovatchev risk-transform constants, re-anchored to the [40, 400] mg/dL device
# range: f(g) = SCALE * (ln(g)^POWER - OFFSET). The clinically-validated power
# POWER = 1.084 is kept from Kovatchev et al. (it encodes the hypo > hyper danger
# asymmetry); SCALE and OFFSET are re-solved so f(40) = -sqrt(10) and
# f(400) = +sqrt(10) — i.e. the risk 10*f^2 saturates at 100 exactly at both
# anchors. The zero-risk euglycemic center consequently sits at ~128 mg/dL (the
# log^POWER center of [40, 400]). These endpoints are verified in a unit test.
# The anchors are NOT the physical clamp: [BG_CLAMP_MIN, BG_CLAMP_MAX] is
# [10, 400] mg/dL, wider below, so the realised risk range is asymmetric —
# [f(10), f(400)] = [-6.8198, +3.1623] and risk reaches ~465 at the floor. That
# clamp is applied in kovatchev_f_target and kovatchev_f_np; kovatchev_f itself
# only asserts the floor and warns above the ceiling.
_KOVATCHEV_SCALE = 2.2211457449985317
_KOVATCHEV_POWER = 1.084
_KOVATCHEV_OFFSET = 5.540076976170212

def compute_patient_seed(master_seed: int, step: int, position: int) -> int:
    """
    Deterministically hash ``(master_seed, step, position)`` to a 63-bit seed.

    SHA-256 gives us collision resistance across all (step, position) pairs in
    a training run.  We mod by 2^63 - 1 because that fits inside int64
    (the dtype torch / numpy seeds expect) while staying clear of the sign
    bit, and the resulting collision rate over a ``TOTAL_STEPS`` × batch run
    (~1e5 steps) is ~5e-7.

    Args:
        master_seed: Integer master seed for the whole training run.
        step: Training step index.
        position: Position within the batch.

    Returns:
        patient_seed: Integer in [0, 2^63 - 1].
    """
    # Stringifying the tuple before hashing means we never have to worry about
    # endian issues across machines — every Python str.encode() is UTF-8.
    key = f"{master_seed}:{step}:{position}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    # ``int(digest, 16)`` turns the hex digest into a 256-bit integer; we mod
    # down to 63 bits to keep it inside int64.
    return int(digest, 16) % (2**63 - 1)


def create_attention_mask_from_visible(
    visible: torch.Tensor, is_pad: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Per-sample attention mask for an arbitrary masked set — the general form.

    A sequence position is one of three things: VISIBLE (its ``bg_absolute`` is
    observed), MASKED (its BG is withheld and the head predicts it), or PAD (a
    left-padding slot carrying no data).  ``visible`` labels the first,
    ``is_pad`` the third, and MASKED is everything left over.  The mask is
    NOT a function of position: a masked span may sit at the right edge
    (forecast), at the left edge (backcast) or anywhere between (infill).

    Rule (True = attend, False = block)::

        visible row → visible col   allowed   (bidirectional among evidence)
        visible row → masked  col   blocked   (evidence never reads a prediction)
        masked  row → any real col  allowed   (a prediction reads everything)
        pad row / pad col           blocked, except the diagonal

    Built in exactly four lines, all four load-bearing::

        attn  = vis[:, None, :] | masked[:, :, None]
        attn &= ~is_pad[:, None, :]          # nothing reads a pad column
        attn &= ~is_pad[:, :, None]          # a pad row reads nothing but itself
        attn[:, diag, diag] = True           # no all-False row (softmax NaN)

    Dropping line 3 leaves a pad row neither masked nor row-filtered, so it
    opens onto every visible column.  The forward OUTPUT is unchanged either
    way — pad columns are blocked, so a pad row never feeds a real row — which
    is why only a full-mask comparison catches it, never a head/output gate.
    Line 4 is the SOLE reason a pad row is not all-False once line 3 is in
    place: it is not redundant with anything.

    Nothing is memoized.  The masked set varies per sample, and no cheap key
    identifies it — a memo on ``(n_context, n_prediction)`` hands one sample's
    mask to another with no shape error and no way to notice.  Each call
    returns a fresh tensor.

    Args:
        visible: ``(B, T)`` bool, True where the position's BG is observed.
            Entries at PAD positions are ignored (lines 2-3 dominate).
        is_pad: ``(B, T)`` bool left-padding flags, or None for no padding.

    Returns:
        attn: ``(B, T, T)`` bool, True = attend.  ``model.forward`` gives it the
            head axis with ``unsqueeze(1)`` before SDPA — never straight, which
            would align B onto the head axis.
    """
    assert visible.dtype == torch.bool and visible.ndim == 2, (
        f"visible must be (B, T) bool, got {tuple(visible.shape)} {visible.dtype}"
    )
    T = visible.shape[1]
    if is_pad is None:
        is_pad = torch.zeros_like(visible)
    assert is_pad.dtype == torch.bool and is_pad.shape == visible.shape, (
        f"is_pad must be (B, T) bool matching visible, got "
        f"{tuple(is_pad.shape)} {is_pad.dtype}"
    )
    vis = visible & ~is_pad
    masked = (~visible) & ~is_pad
    attn = vis[:, None, :] | masked[:, :, None]
    attn &= ~is_pad[:, None, :]
    attn &= ~is_pad[:, :, None]
    diag = torch.arange(T, device=visible.device)
    attn[:, diag, diag] = True
    return attn


def create_attention_mask(n_context: int, n_prediction: int) -> torch.Tensor:
    """Right-edge shim over :func:`create_attention_mask_from_visible`.

    The special case where the masked span is the trailing ``n_prediction``
    patches — a forecast — with no padding.  It builds the ``(1, T)`` visible
    bool internally and returns the ``(T, T)`` mask, which is what the export
    self-check and the single-sample call sites still ask for.  It does NOT
    memoize; see the general form for why no memo can be correct.

    Args:
        n_context: Number of leading VISIBLE patches (C).
        n_prediction: Number of trailing MASKED patches (P).

    Returns:
        mask: ``(C+P, C+P)`` bool tensor, freshly built on every call.
    """
    T = n_context + n_prediction
    visible = torch.zeros(1, T, dtype=torch.bool)
    visible[0, :n_context] = True
    return create_attention_mask_from_visible(visible)[0]


def kovatchev_f(g: torch.Tensor) -> torch.Tensor:
    """Kovatchev risk transform mg/dL (b) → risk (c), with the UNITS TRIPWIRE.

    ``f(g) = _KOVATCHEV_SCALE * (ln(g)^_KOVATCHEV_POWER - _KOVATCHEV_OFFSET)``
    (Kovatchev et al. 2000; the three constants live at the module top) — the
    symmetrizing transform whose risk-distance equates the clinical danger of a
    low and a high excursion, so the loss geometry inherits the hypo>hyper
    asymmetry for free.

    This variant is the **units tripwire**: it hard-asserts its argument is
    physical mg/dL (``g >= BG_CLAMP_MIN - 1e-3``), so a z-scored value trips it
    loudly: the z-score guarantee is ``z_max < BG_CLAMP_MIN - 1e-3``, the assert's
    own threshold, so no z-space tensor passes it.  It is reserved for CONTROLLED
    callers that must never carry z-space — the ``f(last_bg)`` anchor and any
    re-``f`` of an inverted value.
    The upper bound is a soft warning only (``g > BG_CLAMP_MAX + 1e-3``) since a
    legitimate physical BG never exceeds the simulator clamp.  ``f`` is NEVER
    differentiated — every call site feeds it a constant (detached) tensor.

    Args:
        g: physical BG in mg/dL, any shape; every element must be
            ``>= BG_CLAMP_MIN - 1e-3``.

    Returns:
        risk: same shape as ``g``, in Kovatchev risk space.
    """
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    assert (g >= BG_CLAMP_MIN - 1e-3).all(), (
        "kovatchev_f received a value below the physical BG floor "
        f"({BG_CLAMP_MIN} mg/dL) — likely a z-space value leaked into risk space: "
        f"min={float(g.min()):.4f}"
    )
    if (g > BG_CLAMP_MAX + 1e-3).any():
        warnings.warn(
            f"kovatchev_f received a value above the physical BG ceiling "
            f"({BG_CLAMP_MAX} mg/dL): max={float(g.max()):.4f}",
            RuntimeWarning, stacklevel=2,
        )
    return _KOVATCHEV_SCALE * (torch.log(g).pow(_KOVATCHEV_POWER) - _KOVATCHEV_OFFSET)


def kovatchev_f_target(g: torch.Tensor) -> torch.Tensor:
    """Kovatchev risk transform for the TARGET path: physical clamp, then ``f``.

    The forecast target is the RAW ``bg_observed`` (the same raw signal fed as the
    model input — no smoothing), which provably comes from cache mg/dL → never
    z-space.  As a rare physical backstop this path clamps to
    ``[BG_CLAMP_MIN, BG_CLAMP_MAX]`` BEFORE ``f`` (warning only if it actually
    clamps beyond a small tolerance), then applies the same risk transform.  The
    clamp here is a PHYSICAL backstop, not a unit guard — the unit tripwire lives
    on ``kovatchev_f`` (``last_bg`` / re-f).

    Args:
        g: smoothed target BG in mg/dL, any shape.

    Returns:
        risk: same shape as ``g``, in Kovatchev risk space.
    """
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    if (g < BG_CLAMP_MIN - 1e-3).any() or (g > BG_CLAMP_MAX + 1e-3).any():
        warnings.warn(
            "kovatchev_f_target clamped a target outside the physical BG range "
            f"[{BG_CLAMP_MIN}, {BG_CLAMP_MAX}]: "
            f"min={float(g.min()):.4f} max={float(g.max()):.4f}",
            RuntimeWarning, stacklevel=2,
        )
    g = g.clamp(BG_CLAMP_MIN, BG_CLAMP_MAX)
    return _KOVATCHEV_SCALE * (torch.log(g).pow(_KOVATCHEV_POWER) - _KOVATCHEV_OFFSET)


def kovatchev_f_inv(r: torch.Tensor) -> torch.Tensor:
    """Inverse Kovatchev risk transform risk (c) → mg/dL (b). Sole (c)→(b) helper.

    Belt-and-suspenders against both failure modes of a naive inverse:

    1. Clamp the **risk input** first to ``[f(BG_CLAMP_MIN), f(BG_CLAMP_MAX)]``,
       both bounds computed from the clamp (≈ ``[-6.8198, +3.1623]``, asymmetric:
       the floor sits well below the ``f(40) = -sqrt(10)`` anchor).  This guarantees the
       reconstructed base ``(r/_KOVATCHEV_SCALE + _KOVATCHEV_OFFSET)`` is ``>= 0``
       (no negative-base / complex / NaN for ``r`` far below range) and bounded
       above (no fp32 ``exp`` overflow for ``r`` far above range).
    2. ``g = exp((r/_KOVATCHEV_SCALE + _KOVATCHEV_OFFSET)^(1/_KOVATCHEV_POWER))`` —
       the exact inverse of ``kovatchev_f``.
    3. Clamp the **output** to ``[BG_CLAMP_MIN, BG_CLAMP_MAX]`` (a final physical
       backstop; the input clamp already keeps it inside this band).

    Used for the median AND every quantile band edge at every reporting /
    inference / rolling site.  Like ``f`` it is never differentiated.

    Args:
        r: risk-space value, any shape.

    Returns:
        g: physical BG in mg/dL, same shape, in ``[BG_CLAMP_MIN, BG_CLAMP_MAX]``.
    """
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    r_lo = _KOVATCHEV_SCALE * (math.log(BG_CLAMP_MIN) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)
    r_hi = _KOVATCHEV_SCALE * (math.log(BG_CLAMP_MAX) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)
    # ``clamp`` propagates NaN (NaN compares false against both bounds), so a
    # non-finite risk input would otherwise leak straight through to a NaN mg/dL.
    # Scrub NaN/±inf to the band edges FIRST so the inverse can never silently
    # emit a non-finite physical BG; finite values are untouched.
    r = torch.nan_to_num(r, nan=r_lo, posinf=r_hi, neginf=r_lo)
    r = r.clamp(r_lo, r_hi)
    base = r / _KOVATCHEV_SCALE + _KOVATCHEV_OFFSET  # >= 0 after the input clamp
    g = torch.exp(base.pow(1.0 / _KOVATCHEV_POWER))
    return g.clamp(BG_CLAMP_MIN, BG_CLAMP_MAX)


def kovatchev_f_np(g: "np.ndarray") -> "np.ndarray":
    """NumPy Kovatchev risk transform mg/dL (b) → risk (c) for the INPUT path.

    The ``bg_absolute`` INPUT channel is fed in risk space (``f`` applied BEFORE the
    z-score) as the sole input path, and every input-build / stat-fit site is NumPy
    (``data.py``, ``normalization.py``), whereas :func:`kovatchev_f` is torch-only and
    carries the units tripwire.  This is the NumPy sibling for those sites, with the
    same PHYSICAL clamp backstop as :func:`kovatchev_f_target` — it clamps the raw
    input bg to ``[BG_CLAMP_MIN, BG_CLAMP_MAX]`` before ``f``, keeping the stat fit
    and the input transform bit-consistent and always well-defined.

    Args:
        g: physical BG in mg/dL, any shape.

    Returns:
        risk: same shape as ``g``, in Kovatchev risk space.
    """
    g = np.clip(g, BG_CLAMP_MIN, BG_CLAMP_MAX)
    return _KOVATCHEV_SCALE * (np.log(g) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)


def kovatchev_f_inv_np(r: "np.ndarray") -> "np.ndarray":
    """NumPy inverse Kovatchev risk transform risk (c) → mg/dL (b).

    NumPy sibling of :func:`kovatchev_f_inv` for the NumPy denormalize path,
    mirroring its belt-and-suspenders guards exactly: scrub non-finite inputs to
    the band edges, clamp the risk input to ``[f(BG_CLAMP_MIN), f(BG_CLAMP_MAX)]``
    so the base is non-negative and ``exp`` cannot overflow, then clamp the mg/dL
    output to the physical range.

    Args:
        r: risk-space value, any shape.

    Returns:
        g: physical BG in mg/dL, same shape, in ``[BG_CLAMP_MIN, BG_CLAMP_MAX]``.
    """
    import numpy as np
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    r_lo = _KOVATCHEV_SCALE * (math.log(BG_CLAMP_MIN) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)
    r_hi = _KOVATCHEV_SCALE * (math.log(BG_CLAMP_MAX) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)
    r = np.nan_to_num(r, nan=r_lo, posinf=r_hi, neginf=r_lo)
    r = np.clip(r, r_lo, r_hi)
    base = r / _KOVATCHEV_SCALE + _KOVATCHEV_OFFSET  # >= 0 after the input clamp
    g = np.exp(base ** (1.0 / _KOVATCHEV_POWER))
    return np.clip(g, BG_CLAMP_MIN, BG_CLAMP_MAX)


def time_of_day_bin_centers(n_bins: int) -> torch.Tensor:
    """Center hours of the ``n_bins`` circular hour-of-day bins.

    The bins tile ``[0, 24)`` exactly with width ``24 / n_bins``; center ``k`` sits
    at the middle of bin ``k``.

    Args:
        n_bins: number of bins, ``>= 1``.

    Returns:
        ``(n_bins,)`` fp32 tensor of center hours in ``(0, 24)``.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    return (torch.arange(n_bins, dtype=torch.float32) + 0.5) * (24.0 / n_bins)


def time_of_day_bin_target(hour: torch.Tensor, n_bins: int, smooth_bins: float) -> torch.Tensor:
    """Soft circular hour-of-day target distribution over ``n_bins`` bins.

    Treats the bins as ordered and circular: the label mass placed on bin ``k`` decays
    as a wrapped Gaussian of the circular bin-distance between ``hour`` and center ``k``,
    then the row is normalized to sum to 1.  With ``smooth_bins <= 0`` the target
    collapses to a one-hot at the nearest (circular) bin.

    Args:
        hour: ``(...,)`` hours in ``[0, 24)``.
        n_bins: number of bins, ``>= 1``.
        smooth_bins: wrapped-Gaussian std in units of bins; ``<= 0`` => one-hot.

    Returns:
        ``(..., n_bins)`` fp32 distribution; rows sum to 1.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    hour = hour.to(torch.float32)
    bin_w = 24.0 / n_bins
    k = torch.arange(n_bins, dtype=torch.float32, device=hour.device)
    d = ((hour.unsqueeze(-1) / bin_w) - (k + 0.5)).abs()     # (..., n_bins) |bin offset|
    dist = torch.minimum(d, n_bins - d)                       # circular distance in bins
    if smooth_bins <= 0.0:
        idx = dist.argmin(dim=-1)
        return torch.nn.functional.one_hot(idx, n_bins).to(torch.float32)
    weight = torch.exp(-0.5 * (dist / smooth_bins) ** 2)
    return weight / weight.sum(dim=-1, keepdim=True)


def time_of_day_resultant(probs: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Mean resultant vector of a bin distribution on the hour-of-day circle.

    Maps each bin center to its angle ``th_k = 2*pi*center_k/24`` and forms the
    probability-weighted mean of ``(cos th_k, sin th_k)``.  Its length lies in
    ``[0, 1]`` and reads as concentration/confidence.

    Args:
        probs: ``(..., n_bins)`` non-negative weights (rows need not be normalized,
            but a resultant length in ``[0, 1]`` assumes rows sum to 1).
        n_bins: number of bins, ``>= 1``.

    Returns:
        ``(..., 2)`` fp32 = ``(sum_k p_k cos th_k, sum_k p_k sin th_k)``.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    centers = time_of_day_bin_centers(n_bins).to(probs.device)
    th = centers * (2.0 * math.pi / 24.0)                    # (n_bins,)
    cos = (probs * torch.cos(th)).sum(dim=-1)
    sin = (probs * torch.sin(th)).sum(dim=-1)
    return torch.stack([cos, sin], dim=-1)


def time_of_day_decode_bins(logits: torch.Tensor, n_bins: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """Decode per-bin logits to a circular-mean hour and its confidence.

    Softmaxes the logits, forms the resultant vector, and reads the hour from its
    angle and the confidence from its length.

    Args:
        logits: ``(..., n_bins)`` raw bin logits.
        n_bins: number of bins, ``>= 1``.

    Returns:
        ``(hour, R)`` where ``hour`` is ``(...,)`` in ``[0, 24)`` and ``R`` is
        ``(...,)`` resultant length in ``[0, 1]`` (concentration / confidence).
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    probs = torch.softmax(logits, dim=-1)
    res = time_of_day_resultant(probs, n_bins)               # (..., 2)
    cos, sin = res[..., 0], res[..., 1]
    two_pi = 2.0 * math.pi
    hour = (torch.atan2(sin, cos) % two_pi) * (24.0 / two_pi)
    R = torch.hypot(cos, sin)
    return hour, R


def time_of_day_bin_ce(
    logits: torch.Tensor, target_hours: torch.Tensor, n_bins: int, smooth_bins: float
) -> torch.Tensor:
    """Cross-entropy of per-bin logits against the soft circular hour target.

    Args:
        logits: ``(..., n_bins)`` raw bin logits.
        target_hours: ``(...,)`` hours in ``[0, 24)`` (broadcast-matching ``logits[..., 0]``).
        n_bins: number of bins, ``>= 1``.
        smooth_bins: wrapped-Gaussian soft-label std in bins; ``<= 0`` => one-hot.

    Returns:
        Scalar fp32 cross-entropy ``-(tgt * log_softmax(logits)).sum(-1).mean()``.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    tgt = time_of_day_bin_target(target_hours, n_bins, smooth_bins)
    logp = torch.log_softmax(logits, dim=-1)
    return -(tgt * logp).sum(dim=-1).mean()


def time_cross_window_consistency_loss(
    logits_k: torch.Tensor,
    logits_next: torch.Tensor,
    n_bins: int,
    advance_hours: float,
    valid: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Paired-window phase-advance penalty coupling two INDEPENDENT forward passes.

    Window ``k+1``'s prediction origin sits exactly ``advance_hours`` (one forecast
    horizon) after window ``k``'s, so rotating window ``k``'s origin-patch resultant
    vector by ``dtheta = 2*pi*advance_hours/24`` must land on window ``k+1``'s
    origin-patch resultant.  Matching them in the raw ``(cos, sin)`` plane (atan2-free,
    stable gradient) couples the two clocks.  Unlike the within-window advance penalty
    this is NOT redundant: the two windows are separate forwards with different
    contexts, so only an explicit cross-window term ties their difference to one horizon.
    Only the origin patch (index 0) of each window is used.

    Args:
        logits_k:    ``(B, P, n_bins)`` window k per-patch bin logits.
        logits_next: ``(B, P, n_bins)`` window k+1 per-patch bin logits.
        n_bins: number of bins, ``>= 1``.
        advance_hours: clock advance between the two origins, in hours (== the horizon).
        valid: ``(B,)`` bool or ``None``.  When given only ``True`` rows enter the mean
            (a finite ``0`` when none are valid); ``None`` => plain mean over ``B``.

    Returns:
        Scalar fp32 loss.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    assert logits_k.dim() == 3 and logits_next.dim() == 3, (
        f"expected (B, P, n_bins), got {tuple(logits_k.shape)} / {tuple(logits_next.shape)}"
    )
    assert logits_k.shape[0] == logits_next.shape[0], "batch mismatch between windows"
    pk = torch.softmax(logits_k[:, 0, :], dim=-1)            # (B, n_bins)
    pn = torch.softmax(logits_next[:, 0, :], dim=-1)         # (B, n_bins)
    rk = time_of_day_resultant(pk, n_bins)                   # (B, 2)
    rn = time_of_day_resultant(pn, n_bins)                   # (B, 2)
    c, s = rk[:, 0], rk[:, 1]
    dtheta = 2.0 * math.pi * advance_hours / 24.0
    rc = c * math.cos(dtheta) - s * math.sin(dtheta)
    rs = c * math.sin(dtheta) + s * math.cos(dtheta)
    per = (rc - rn[:, 0]) ** 2 + (rs - rn[:, 1]) ** 2        # (B,)
    if valid is None:
        return per.mean()
    vf = valid.to(per.dtype)
    return (per * vf).sum() / vf.sum().clamp(min=1.0)


def time_cross_window_jump_hours(
    logits_k: torch.Tensor,
    logits_next: torch.Tensor,
    n_bins: int,
    advance_hours: float,
) -> torch.Tensor:
    """Per-sample cross-window clock-advance deviation, in hours (the no-jump witness).

    Decodes the origin-patch (index 0) clock of both windows, measures the signed
    circular step between them, and returns ``|step - advance_hours|``.  ``~0`` means
    the rolling clock advances by exactly one horizon across the window seam.  The
    caller masks by the per-sample validity flag.

    Args:
        logits_k:    ``(B, P, n_bins)`` window k per-patch bin logits.
        logits_next: ``(B, P, n_bins)`` window k+1 per-patch bin logits.
        n_bins: number of bins, ``>= 1``.
        advance_hours: expected clock advance between the two origins, in hours.

    Returns:
        ``(B,)`` fp32 ``|advance deviation|`` in hours.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    assert logits_k.dim() == 3 and logits_next.dim() == 3, (
        f"expected (B, P, n_bins), got {tuple(logits_k.shape)} / {tuple(logits_next.shape)}"
    )
    assert logits_k.shape[0] == logits_next.shape[0], "batch mismatch between windows"
    hk, _ = time_of_day_decode_bins(logits_k[:, 0, :], n_bins)     # (B,)
    hn, _ = time_of_day_decode_bins(logits_next[:, 0, :], n_bins)  # (B,)
    r = circular_hour_residual(hn, hk)                             # (B,) signed (-12, 12]
    return (r - advance_hours).abs()


def time_inter_patch_jump_hours(
    logits: torch.Tensor, n_bins: int, advance_hours: float
) -> torch.Tensor:
    """Per-sample mean deviation of the inter-patch clock advance, in hours.

    The "no jumping" witness: decodes each patch to an hour, measures the signed
    circular step between consecutive patches, and averages ``|step - advance_hours|``.
    ~0 means the predicted clock marches forward one patch at a time.

    Args:
        logits: ``(B, P, n_bins)`` per-patch bin logits.
        n_bins: number of bins, ``>= 1``.
        advance_hours: expected clock advance between consecutive patches, in hours.

    Returns:
        ``(B,)`` fp32 mean absolute advance-deviation in hours
        (``logits.new_zeros(B)`` when ``P < 2``).
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    assert logits.dim() == 3, f"expected (B, P, n_bins), got {tuple(logits.shape)}"
    if logits.shape[1] < 2:
        return logits.new_zeros(logits.shape[0])
    hours, _ = time_of_day_decode_bins(logits, n_bins)       # (B, P)
    r = circular_hour_residual(hours[:, 1:], hours[:, :-1])  # (B, P-1)
    return (r - advance_hours).abs().mean(dim=-1)


def _resultant_np(probs: np.ndarray, n_bins: int) -> "tuple[float, float]":
    """Numpy twin of ``time_of_day_resultant`` for one bin distribution.

    Maps bin center ``center_k = (k + 0.5) * 24 / n_bins`` to its angle
    ``th_k = 2*pi*center_k/24`` and forms the probability-weighted mean of
    ``(cos th_k, sin th_k)`` — kept byte-for-byte in step with the torch version so
    the geometry core never drifts from the trained probe.

    Args:
        probs: ``(n_bins,)`` non-negative weights (assumed to sum to ~1).
        n_bins: number of bins, ``>= 1``.

    Returns:
        ``(cos, sin)`` = ``(sum_k p_k cos th_k, sum_k p_k sin th_k)``.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    assert probs.shape[-1] == n_bins, f"expected last dim {n_bins}, got {probs.shape}"
    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * (24.0 / n_bins)
    th = centers * (2.0 * math.pi / 24.0)
    cos = float((probs * np.cos(th)).sum())
    sin = float((probs * np.sin(th)).sum())
    return cos, sin


def _fractional_roll(row: np.ndarray, shift_bins: float) -> np.ndarray:
    """Wrap-correct circular shift of a ``(n_bins,)`` row by a real number of bins.

    ``np.roll(row, k)`` moves bin ``i -> i+k``; a *forward-in-time* rotation (a later
    hour, higher bin index) uses a positive ``shift_bins``. A fractional shift is the
    linear blend of the two adjacent integer rolls, so the circular adjacency of
    bin ``n_bins-1`` and bin ``0`` is preserved.

    Args:
        row: ``(n_bins,)`` values.
        shift_bins: real-valued shift in bins (may be negative / fractional).

    Returns:
        ``(n_bins,)`` shifted row.
    """
    assert row.ndim == 1, f"expected 1-D row, got {row.shape}"
    lo = int(math.floor(shift_bins))
    f = shift_bins - lo
    return (1.0 - f) * np.roll(row, lo) + f * np.roll(row, lo + 1)


def aggregate_origin_belief(probs: np.ndarray, advance_hours: float,
                            bin_hours: float) -> np.ndarray:
    """Fuse the P per-patch beliefs into a single origin-phase belief.

    Patch ``p``'s belief is the origin belief advanced by ``p*advance_hours`` (the
    probe's trained property ``tgt_hour[p] = origin + p*advance``), so DE-ROTATE patch
    ``p`` by ``-p*advance_hours`` (a shift of ``-p*advance_hours/bin_hours`` bins via
    ``_fractional_roll``), average the P de-rotated rows, and renormalize to sum 1.
    Agreement across patches sharpens the fused belief; disagreement diffuses it, so
    its resultant length self-weights the inter-patch consistency. Wrap-safe.

    Args:
        probs: ``(P, n_bins)`` per-patch softmax rows (each sums ~1).
        advance_hours: elapsed clock advance between consecutive patches, in hours.
        bin_hours: origin-hour bin width in hours (``24 / n_bins``).

    Returns:
        ``(n_bins,)`` fused origin-phase belief summing to 1; ``P == 1`` returns row 0
        unchanged.
    """
    assert probs.ndim == 2, f"expected (P, n_bins), got {probs.shape}"
    assert bin_hours > 0.0, f"need bin_hours > 0, got {bin_hours}"
    P, n_bins = probs.shape
    if P == 1:
        return probs[0].astype(np.float64, copy=True)
    acc = np.zeros(n_bins, dtype=np.float64)
    for p in range(P):
        acc += _fractional_roll(probs[p].astype(np.float64), -p * advance_hours / bin_hours)
    total = acc.sum()
    if total <= 0.0:
        return acc
    return acc / total


class ClockGeometry(NamedTuple):
    """Drawable circular-histogram geometry for one hour-of-day belief (y-up unit disk).

    Fields:
        wedges: ``(n_bins, arc_segments+2, 2)`` y-up unit coords; vertex 0 is the
            center ``(0, 0)``, the rest trace bin ``k``'s outer arc at radius ``m_k``.
        magnitudes: ``(n_bins,)`` wedge outer radii in ``[0, 1]``.
        hand: ``(2,)`` y-up unit coords of the resultant hand, length ``R``.
        R: resultant length in ``[0, 1]`` — rotation-INVARIANT.
    """
    wedges: np.ndarray
    magnitudes: np.ndarray
    hand: np.ndarray
    R: float


def _hour_to_unit(hour: np.ndarray) -> np.ndarray:
    """Map hour(s) to the FROZEN y-up clock disk: ``u(h) = (sin(2*pi*h/24), cos(2*pi*h/24))``.

    Hour 0 sits at 12-o'clock top and hours increase clockwise (h=6 -> right,
    h=12 -> bottom, h=18 -> left).

    Args:
        hour: array of hours (any shape).

    Returns:
        ``(..., 2)`` y-up unit vectors.
    """
    a = hour * (2.0 * math.pi / 24.0)
    return np.stack([np.sin(a), np.cos(a)], axis=-1)


_CLOCK_TICK_HOURS = (0.0, 6.0, 12.0, 18.0)


def clock_reference_ticks() -> np.ndarray:
    """Y-up unit vectors for the fixed 0/6/12/18-hour dial reference ticks.

    Sourced from the single frozen convention ``_hour_to_unit`` so host adapters
    (pygame, matplotlib) draw the reference ticks without re-deriving any
    trigonometry — the dial convention lives in exactly one place.

    Returns:
        ``(4, 2)`` y-up unit vectors for hours 0, 6, 12, 18.
    """
    return _hour_to_unit(np.array(_CLOCK_TICK_HOURS))


def clock_wedge_geometry(probs: np.ndarray, rotation_hours: float = 0.0,
                         arc_segments: int = 6) -> ClockGeometry:
    """Circular-histogram geometry for one belief on the hour-of-day dial.

    Bin ``k`` spans hours ``[k*bin_hours, (k+1)*bin_hours)`` with center
    ``(k+0.5)*bin_hours`` (``bin_hours = 24/n_bins`` inferred from ``probs``). A
    continuous RIGID rotation adds ``rotation_hours`` to every mapped hour BEFORE the
    ``u(h)`` map (angles only, NO re-binning) so cursor motion is smooth and exact.
    Magnitude ``m_k = p_k / p.max()`` (an all-zero belief yields zeros, ``R=0``, no
    NaN). The hand points along ``u(mean_hour + rotation_hours)`` scaled by ``R``, and
    ``R`` is invariant under ``rotation_hours`` because rotation preserves the
    resultant length.

    Args:
        probs: ``(n_bins,)`` non-negative belief (softmax row; ``n_bins`` inferred).
        rotation_hours: rigid dial rotation applied to every hour, in hours.
        arc_segments: number of arc chords per wedge (``>= 1``).

    Returns:
        ``ClockGeometry`` (wedges ``(n_bins, arc_segments+2, 2)``, magnitudes
        ``(n_bins,)``, hand ``(2,)``, R float).
    """
    assert probs.ndim == 1, f"expected (n_bins,), got {probs.shape}"
    assert arc_segments >= 1, f"need arc_segments >= 1, got {arc_segments}"
    probs = probs.astype(np.float64)
    n_bins = probs.shape[0]
    bin_hours = 24.0 / n_bins

    peak = probs.max()
    magnitudes = probs / peak if peak > 0.0 else np.zeros_like(probs)

    wedges = np.empty((n_bins, arc_segments + 2, 2), dtype=np.float64)
    wedges[:, 0, :] = 0.0
    for k in range(n_bins):
        arc_hours = np.linspace(k * bin_hours, (k + 1) * bin_hours, arc_segments + 1)
        wedges[k, 1:, :] = magnitudes[k] * _hour_to_unit(arc_hours + rotation_hours)

    cos, sin = _resultant_np(probs, n_bins)
    R = float(math.hypot(cos, sin))
    mean_hour = (math.atan2(sin, cos) % (2.0 * math.pi)) * (24.0 / (2.0 * math.pi))
    hand = R * _hour_to_unit(np.array(mean_hour + rotation_hours))

    return ClockGeometry(wedges=wedges, magnitudes=magnitudes, hand=hand, R=R)


def circular_hour_error(pred_hour: torch.Tensor, true_hour: torch.Tensor) -> torch.Tensor:
    """Absolute circular distance in hours (...,), in [0,12]."""
    d = (pred_hour - true_hour).abs() % 24.0
    return torch.minimum(d, 24.0 - d)


def circular_hour_residual(pred_hour: torch.Tensor, true_hour: torch.Tensor) -> torch.Tensor:
    """Signed circular residual (pred - true) wrapped to (-12, 12] hours (...,).

    Positive => the clock reads ahead of truth, negative => behind. Its absolute
    value equals ``circular_hour_error``; the sign is what bias/precision need.
    """
    return (pred_hour - true_hour + 12.0) % 24.0 - 12.0


def circular_bias_hours(pred_hour: torch.Tensor, true_hour: torch.Tensor) -> torch.Tensor:
    """Signed systematic clock offset: the circular mean of the residual, in (-12, 12] hours.

    ~0 means no consistent fast/slow drift; a nonzero value is a *correctable* constant
    offset. Computed as the angle of the mean resultant vector (naive averaging of angles
    is wrong across the 24 h wrap). Returns a scalar tensor.
    """
    two_pi = 2.0 * math.pi
    delta = circular_hour_residual(pred_hour, true_hour) * (two_pi / 24.0)
    mean_angle = torch.atan2(torch.sin(delta).mean(), torch.cos(delta).mean())
    return mean_angle * (24.0 / two_pi)


def circular_std_hours(pred_hour: torch.Tensor, true_hour: torch.Tensor) -> torch.Tensor:
    """Circular standard deviation of the residual, in hours (spread about the bias).

    ``sqrt(-2 ln R_bar)`` with ``R_bar`` the mean resultant length: 0 h = perfectly
    consistent phase, growing without bound as the residuals approach a uniform smear.
    ``R_bar`` is clamped to ``[1e-6, 1.0]``: the lower bound keeps a near-uniform
    distribution finite (rather than ``inf``); the upper bound guards the
    perfectly-consistent case, where ``cos.mean()**2 + sin.mean()**2`` can round just
    above ``1.0`` in fp32, making ``log(R_bar) > 0`` and ``sqrt(-2·log R_bar)`` the
    square root of a negative — a NaN. Returns a scalar tensor.
    """
    two_pi = 2.0 * math.pi
    delta = circular_hour_residual(pred_hour, true_hour) * (two_pi / 24.0)
    r_bar = torch.sqrt(
        torch.cos(delta).mean() ** 2 + torch.sin(delta).mean() ** 2
    ).clamp(1e-6, 1.0)
    # ``+ 0.0`` normalises the SIGN of zero. The perfectly-consistent case the
    # upper clamp exists for gives ``log(1.0) = 0.0``, ``-2 * 0.0 = -0.0`` and
    # ``sqrt(-0.0) = -0.0``, which formats as "-0.00 h" on the validation table —
    # a negative standard deviation. ``clamp(min=0.0)`` does NOT fix it (it
    # returns -0.0); IEEE addition of +0.0 is the operation that maps -0.0 to
    # +0.0 and leaves every other value bit-identical.
    return torch.sqrt(-2.0 * torch.log(r_bar)) * (24.0 / two_pi) + 0.0


_GLOBAL_MEDIAN_BASIS_CACHE: "dict[tuple[int, int, str, torch.device, torch.dtype], torch.Tensor]" = {}


def get_global_median_basis(
    n: int, k: int, kind: str = 'dct',
    device: "torch.device | None" = None, dtype: "torch.dtype | None" = None,
) -> torch.Tensor:
    """Fixed ``(n, k)`` orthonormal low-frequency basis for the GLOBAL median
    projection (R3), built once per ``(n, k, kind)`` and cached.

    Identical construction to :func:`model.make_step_basis` (DCT-II cosine modes or
    orthonormal polynomials, ascending frequency/degree, L2-orthonormal columns) but
    evaluated over a whole masked SPAN, ``n = L * PATCH_SIZE`` for a span of ``L``
    patches, rather than a single patch.  ``assemble_quantiles`` projects the span's
    per-patch median delta onto ``span`` of these ``k`` columns; with ``k`` small the
    median is confined to a smooth low-frequency subspace (periods below ~``2*n/k``
    steps — including the per-patch seam sawtooth — are unrepresentable), and the
    projection is an L2 contraction (``||proj(x)|| <= ||x||``) so the per-patch offset
    cannot drift.  ``k`` comes from :func:`global_median_dim`, which scales it with
    ``L``; the function itself is closed-form in ``(n, k)`` and indifferent to which
    span it serves.

    The cache is keyed by ``(n, k, kind, device, dtype)``, so the basis materializes
    once already resident on the requested ``device``/``dtype`` (no per-forward H2D copy
    or dtype cast on the hot path); each call returns a defensive clone of that
    device-resident tensor so an in-place edit by any caller can never poison the cache.

    Args:
        n: span length in timesteps, ``L * PATCH_SIZE`` (``>= 1``).
        k: number of basis columns ``G_L``, ``1 <= k <= n`` (caller clamps to ``min(G_L, n)``).
        kind: ``'dct'`` (DCT-II) or ``'poly'`` (orthonormal polynomials).
        device: target device for the returned tensor (default: CPU/cache device).
        dtype: target dtype (default: float32).

    Returns:
        ``(n, k)`` tensor with L2-orthonormal columns; col 0 is the constant DC mode.
    """
    assert 1 <= k <= n, f"need 1 <= G ({k}) <= P*S ({n})"
    resolved_device = torch.device('cpu') if device is None else torch.device(device)
    resolved_dtype = torch.float32 if dtype is None else dtype
    key = (n, k, kind, resolved_device, resolved_dtype)
    basis = _GLOBAL_MEDIAN_BASIS_CACHE.get(key)
    if basis is None:
        s = torch.arange(n, dtype=torch.float64)
        if kind == 'dct':
            ks = torch.arange(k, dtype=torch.float64)
            basis = torch.cos(math.pi * (s.view(-1, 1) + 0.5) * ks.view(1, -1) / n)
        elif kind == 'poly':
            t = 2.0 * s / (n - 1) - 1.0 if n > 1 else s
            vand = torch.stack([t ** p for p in range(k)], dim=1)  # (n, k)
            basis, _ = torch.linalg.qr(vand)
        else:
            raise ValueError(
                f"unknown BG_HEAD_STEP_BASIS_TYPE {kind!r} (want 'dct' or 'poly')")
        basis = basis / basis.norm(dim=0, keepdim=True)            # orthonormal columns
        basis = basis.to(torch.float32)                            # canonical fp32
        # Materialize once at the requested device/dtype (same op order as before:
        # fp32 canonical → .to(device, dtype)), so the hot path is a device-local
        # clone rather than a per-forward host→device copy.
        basis = basis.to(device=resolved_device, dtype=resolved_dtype)
        _GLOBAL_MEDIAN_BASIS_CACHE[key] = basis
    # Defensive clone so the returned tensor never aliases the process-wide cache:
    # an in-place edit by any caller must not poison it.
    return basis.clone()


def global_median_dim(span_patches: int) -> int:
    """``G_L`` — the smooth-basis dimension for a masked span of ``L`` patches.

    ``G_L = max(1, ceil(BG_HEAD_MEDIAN_GLOBAL_DIM * L / PREDICTION_PATCHES))``, so
    the configured ``G`` is reproduced exactly at ``L == PREDICTION_PATCHES`` and
    scales down with shorter spans::

        L        1     2     3     4( == PREDICTION_PATCHES)
        n = L*S  6    12    18    24
        G_L      3     6     9    12
        2n/G_L  4.0   4.0   4.0   4.0     (shortest representable period, steps)

    A FIXED ``G`` is a defect here rather than an approximation: at ``L = 1`` the
    projection would be ``min(G, n) = 6`` columns over ``n = 6`` points — the
    identity to 1e-5 — so the anti-drift contraction is ABSENT, not weakened, and
    every fan assert still passes.  What ``G_L`` holds roughly constant is the
    fraction of the span the basis can bend, not the cutoff period: the cutoff
    ``2n/G_L`` above is what to report, not a claim that the period is fixed.

    Args:
        span_patches: span length ``L`` in patches (``>= 1``).

    Returns:
        ``G_L >= 1``.  The caller still clamps to ``min(G_L, n)`` before building
        the basis.
    """
    from config import BG_HEAD_MEDIAN_GLOBAL_DIM, PREDICTION_PATCHES
    assert span_patches >= 1, f"span_patches must be >= 1, got {span_patches}"
    return max(1, math.ceil(BG_HEAD_MEDIAN_GLOBAL_DIM * span_patches / PREDICTION_PATCHES))


def _span_layout(
    mask_idx: "torch.Tensor | None", valid: "torch.Tensor | None",
    B: int, M: int, device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Group the ``M`` head slots into contiguous masked spans.

    Slot ``j`` continues slot ``j-1``'s span iff their patch indices are adjacent
    (``mask_idx[j] == mask_idx[j-1] + 1``) and, when ``valid`` is given, both are
    real.  The sampler never lets two masked spans abut — a visible patch always
    separates them — so adjacency in ``mask_idx`` identifies a span exactly.
    Padded slots gather patch 0, which can never continue a span (that would need
    a predecessor at patch −1), so they fall out as singletons and cannot merge
    into a real span even when a real span does start at patch 0.

    ``mask_idx is None`` means the legacy single-span layout: all ``M`` slots are
    one contiguous span, reproducing the trailing prediction zone exactly.

    Returns:
        start: ``(B, M)`` int64 — the slot index at which each slot's span begins.
        length: ``(B, M)`` int64 — the span's length ``L`` in patches, per slot.
    """
    ar = torch.arange(M, device=device)
    if mask_idx is None:
        new = ar.eq(0).unsqueeze(0).expand(B, M)
    else:
        cont = mask_idx[:, 1:] == mask_idx[:, :-1] + 1
        if valid is not None:
            cont = cont & valid[:, 1:] & valid[:, :-1]
        new = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=device), ~cont], dim=1)
    ar_b = ar.unsqueeze(0).expand(B, M)
    # Running max of "index if a span starts here else -1" = the current span's
    # start slot.  new[:, 0] is always True, so every slot resolves to >= 0.
    start = torch.cummax(
        torch.where(new, ar_b, torch.full_like(ar_b, -1)), dim=1).values
    ones = torch.ones(B, M, dtype=torch.long, device=device)
    counts = torch.zeros(B, M, dtype=torch.long, device=device).scatter_add_(1, start, ones)
    length = counts.gather(1, start)
    return start, length


def _median_global_per_span(
    delta: torch.Tensor, start: torch.Tensor, length: torch.Tensor, kind: str,
) -> torch.Tensor:
    """Project each span's median delta onto its own low-frequency subspace.

    One matmul per distinct span length present in the batch (at most four under
    ``MASK_SPAN_LENGTHS``, plus singletons from padded slots): spans of equal ``L``
    share ``n = L * PATCH_SIZE`` and ``G_L``, so they stack into a single
    ``(n_spans, n)`` block.  Slots are contiguous within a span, so the gather is
    ``start + arange(L)`` and the scatter back writes each slot exactly once.

    Args:
        delta: ``(B, M, S)`` per-slot median delta (risk space).
        start: ``(B, M)`` span-start slot index (see :func:`_span_layout`).
        length: ``(B, M)`` span length in patches.
        kind: ``BG_HEAD_STEP_BASIS_TYPE``.

    Returns:
        ``(B, M, S)`` projected delta, span by span.
    """
    B, M, S = delta.shape
    ar = torch.arange(M, device=delta.device)
    is_start = start.eq(ar.unsqueeze(0))
    out = torch.zeros_like(delta)
    # One host sync for the whole loop: the distinct span lengths present.
    for L in torch.unique(length[is_start]).tolist():
        sel = is_start & length.eq(L)
        rows, cols = sel.nonzero(as_tuple=True)               # (n_L,) each
        idx = cols.unsqueeze(1) + torch.arange(L, device=delta.device).unsqueeze(0)
        blk = delta[rows.unsqueeze(1), idx]                   # (n_L, L, S)
        n = L * S
        g = min(global_median_dim(L), n)
        Bg = get_global_median_basis(
            n, g, kind, device=delta.device, dtype=delta.dtype)
        proj = (blk.reshape(-1, n) @ Bg) @ Bg.transpose(0, 1)
        out[rows.unsqueeze(1), idx] = proj.reshape(-1, L, S)
    return out


def _carry_is_zero(carry_spread: "torch.Tensor | float") -> bool:
    """True when ``carry_spread`` widens nothing — the default, and the whole training path."""
    if isinstance(carry_spread, torch.Tensor):
        return bool((carry_spread == 0).all())
    return carry_spread == 0


def assemble_quantiles(
    head_raw: torch.Tensor, anchor_bg_mgdl: torch.Tensor,
    mask_idx: "torch.Tensor | None" = None,
    valid: "torch.Tensor | None" = None,
    carry_spread: "torch.Tensor | float" = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the BG head's raw output into an ascending risk-space quantile fan.

    The head emits ``1 + 2*N_SPREADS`` raw numbers per step (col0 = median delta;
    cols ``1..N_SPREADS`` = the ``τ>.5`` spreads nearest→far .75/.9/.95; cols
    ``N_SPREADS+1..2*N_SPREADS`` = the ``τ<.5`` spreads nearest→far .25/.1/.05).
    Spreads are passed through ``softplus`` + ``BG_QUANTILE_SPREAD_MIN`` (a strict
    positive floor against σ-collapse) and accumulated by ``cumsum`` so the fan is
    monotone by construction.  The anchor ``f(anchor_bg)`` is detached (a constant),
    held flat across the ``PATCH_SIZE`` steps of ITS OWN slot; each of the ``M``
    masked slots carries its own anchor.

    The ``M`` axis is a set of masked patches gathered by index, not a trailing
    horizon: a span may end at patch ``T−1`` (forecast), start at patch 0
    (backcast) or sit between visible patches (infill).  Slots are grouped into
    contiguous spans from ``mask_idx`` (:func:`_span_layout`) and every median
    mode is evaluated PER SPAN — nothing accumulates or low-passes across the
    visible patches that separate two spans.

    Median assembly is gated by ``config.BG_HEAD_MEDIAN_MODE`` (R3, three modes):

    * ``'global'`` (DEFAULT, R3): the per-patch median delta ``delta = head_raw[..., 0]``
      is reshaped ``(n_spans, L*S)`` C-contiguous patch-major (matching
      ``risk_loss._to_patch_major``) and PROJECTED onto a fixed low-frequency DCT-II
      subspace ``span(Bg)`` over the span's own ``n = L*S`` steps: ``z = delta_flat @ Bg``,
      ``delta_global = z @ Bgᵀ``, ``m = anchor + delta_global``.  A
      projection is an **L2 CONTRACTION** (``||proj(x)|| <= ||x||``), so the per-patch
      offset ``o[p] = m[:,p,0] − anchor`` is bounded and non-monotone in ``p`` — it
      CANNOT drift or amplify, structurally killing R1's unconstrained-integrator drift.
      The subspace dimension is ``global_median_dim(L)``, NOT a fixed ``G``: a fixed
      ``G`` degenerates to the identity at ``L = 1`` and loses the contraction with
      every fan assert still green.  The low-pass leaves the median smooth (the
      per-patch period-2 / seam sawtooth is unrepresentable), so C0 seam-continuity
      is intentionally NOT pinned.
      At init (delta≈0 ⇒ z≈0 ⇒ delta_global≈0) ``m ≈ anchor`` everywhere (persistence).
    * ``'cumulative'`` (R1, BIT-IDENTICAL legacy): each patch CONTINUES from the previous
      patch's endpoint, WITHIN ITS SPAN.  With ``d = head_raw[..., 0]``,
      ``d_rel = d − d[..., :1]``, ``rise = d[..., -1] − d[..., 0]``, and the EXCLUSIVE
      cumsum of ``rise`` over the slots of one span (``o == 0`` at each span's first
      slot) gives ``m = anchor + o[..., None] + d_rel`` — **C0
      continuity** at every seam and the FIRST step of each span pinned EXACTLY to
      that span's anchor.
    * ``'independent'`` (BIT-IDENTICAL legacy): ``m = anchor + delta`` — each patch's
      within-patch median curve is INDEPENDENT, so the median can jump at the seams.

    Under every mode ``median == q_tau[..., 3]`` and the ascending fan hold: only the
    value of ``m`` changes; the ± band structure (``head_raw[..., 1:]`` → softplus →
    cumsum, plus ``carry_spread``) is untouched.

    ``carry_spread`` (risk space, default ``0.0`` → bit-identical to a bare fan)
    seeds the cumulative spread base on BOTH sides of the median:
    ``q(τ>.5) = m + hypot(c_up, cumsum(d+))`` and
    ``q(τ<.5) = m − hypot(c_dn, cumsum(d−))`` — QUADRATURE, because the carry is
    another roll's increment and independent increments add variances; adding the
    two is the perfectly-correlated bound.
    It is PER LEVEL — a trailing axis of ``2*N_SPREADS`` in the spread columns' own
    layout, ``[.75 .9 .95 | .25 .1 .05]`` — and a scalar widens all six alike.  One
    value across the levels re-seeds every level from the outermost one's carry, so
    the fan flattens into a slab (`SPEC/inference.md` §8.1).  NO runtime caller passes it — the
    sole non-test call site is ``model.py``'s forward, which takes the default.
    ``inference.predict_rolling`` needs the same widening but cannot reach this
    argument (the assembly runs inside ``model.forward``), so it accumulates its own
    six per-level risk-space offsets and applies the identical quadrature composition
    post-forward on the returned ``q_tau``.  Keep the two in step: a change to the algebra here must be
    mirrored there, or the rolling band silently stops matching the fan it widens.

    Args:
        head_raw: ``(B, M, S, 1 + 2*N_SPREADS)`` raw head output (risk space), one
            slot per masked patch.
        anchor_bg_mgdl: ``(B, M)`` per-slot anchor BG (mg/dL), ONE-SIDED and
            left-preferring: the last step of the span's LEFT neighbour, or the
            first step of the right neighbour when the span starts at patch 0.
            Every slot of a span carries the same value, so the anchor is NOT the
            nearest visible evidence for a slot near the span's right edge — it
            ignores the near side, which costs no information (masked rows attend
            both ways) and only makes this offset parameterisation work harder.
            Evaluation bins on the two-sided distance ``d``, never on this.
            ``(B,)`` is accepted as the legacy single-span form (one broadcast
            ``last_bg``) and then ``mask_idx`` must be None.
        mask_idx: ``(B, M)`` int64 patch index of each slot.  Required whenever
            ``anchor_bg_mgdl`` is ``(B, M)``: it is what identifies the spans, and
            without it the whole ``M`` axis would silently low-pass as one span.
            None selects the legacy layout (all ``M`` slots one contiguous span),
            which reproduces the trailing prediction zone exactly.
        valid: ``(B, M)`` bool, False on padded slots.  Optional — padded slots
            gather patch 0 and fall out as their own singleton spans either way;
            passing ``valid`` additionally pins their median to their anchor, so
            no gradient reaches ``head_raw[..., 0]`` there.
        carry_spread: risk-space scalar, tensor broadcastable to ``(B, M, S, 1)``
            (every level alike), or one broadcastable to ``(B, M, S, 2*N_SPREADS)``
            in the spread columns' layout ``[.75 .9 .95 | .25 .1 .05]`` (per level).
            Seeds the cumulative spread base (default ``0.0``).

    Returns:
        q_tau: ``(B, M, S, N_QUANTILES)`` quantiles in risk space, ascending in τ
            (index-for-index with ``QUANTILE_LEVELS``).
        median: ``(B, M, S)`` median in risk space (== ``q_tau[..., 3]``).
    """
    from config import (
        N_SPREADS, N_QUANTILES, BG_QUANTILE_SPREAD_MIN,
        BG_HEAD_MEDIAN_MODE, BG_HEAD_STEP_BASIS_TYPE,
    )
    import torch.nn.functional as F
    assert head_raw.ndim == 4 and head_raw.shape[-1] == 1 + 2 * N_SPREADS, (
        f"head_raw must be (B, M, S, {1 + 2 * N_SPREADS}), got {tuple(head_raw.shape)}"
    )
    B_, M_, S_ = head_raw.shape[:3]
    if anchor_bg_mgdl.ndim == 1:
        assert mask_idx is None, (
            "a (B,) anchor is the legacy single-span form and cannot describe a "
            "general masked set — pass a (B, M) anchor with mask_idx"
        )
        assert anchor_bg_mgdl.shape[0] == B_, (
            f"anchor_bg_mgdl must be (B,)=({B_},), got {tuple(anchor_bg_mgdl.shape)}"
        )
        anchor_bm = anchor_bg_mgdl.unsqueeze(1).expand(B_, M_)
    else:
        assert anchor_bg_mgdl.shape == (B_, M_), (
            f"anchor_bg_mgdl must be (B, M)=({B_}, {M_}), got "
            f"{tuple(anchor_bg_mgdl.shape)}"
        )
        assert mask_idx is not None, (
            "a (B, M) anchor needs mask_idx (B, M) to identify the spans"
        )
        anchor_bm = anchor_bg_mgdl
    if mask_idx is not None:
        assert mask_idx.shape == (B_, M_) and mask_idx.dtype == torch.int64, (
            f"mask_idx must be (B, M)=({B_}, {M_}) int64, got "
            f"{tuple(mask_idx.shape)} {mask_idx.dtype}"
        )
    if valid is not None:
        assert valid.shape == (B_, M_) and valid.dtype == torch.bool, (
            f"valid must be (B, M)=({B_}, {M_}) bool, got "
            f"{tuple(valid.shape)} {valid.dtype}"
        )

    # Anchor: f(anchor_bg), detached (a constant), flat across the slot's S steps.
    # Clamp the persistence anchor into the physical BG range first — a CGM-noisy
    # last reading can sit just above the simulator ceiling (e.g. 402 mg/dL), and
    # the anchor is a constant, so clamping silences kovatchev_f's ceiling warning
    # without touching any gradient.
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    anchor = kovatchev_f(anchor_bm.detach().clamp(BG_CLAMP_MIN, BG_CLAMP_MAX))  # (B,M)
    anchor = anchor.unsqueeze(-1)                        # (B, M, 1)
    delta = head_raw[..., 0]                             # (B, M, S)
    start, length = _span_layout(mask_idx, valid, B_, M_, delta.device)
    assert BG_HEAD_MEDIAN_MODE in ('global', 'cumulative', 'independent'), (
        f"BG_HEAD_MEDIAN_MODE must be 'global'/'cumulative'/'independent', "
        f"got {BG_HEAD_MEDIAN_MODE!r}")
    if BG_HEAD_MEDIAN_MODE == 'global':
        # R3 GLOBAL SMOOTH-BASIS median: project each span's median delta onto a
        # fixed low-frequency DCT-II subspace over that span's own L*S steps.  A
        # projection is an L2 contraction, so the per-patch offset cannot drift
        # (kills R1's integrator).  The flattening is C-contiguous patch-major
        # (flat = p*S + s), matching risk_loss._to_patch_major so the basis
        # low-passes the TIME axis.
        delta_med = _median_global_per_span(
            delta, start, length, BG_HEAD_STEP_BASIS_TYPE)
    elif BG_HEAD_MEDIAN_MODE == 'cumulative':
        # Cumulative cross-patch-continuity median (R1): each patch continues from
        # the previous patch's endpoint.  d_rel zeroes each patch's curve at step 0;
        # the exclusive cumsum of the per-patch rise carries the offset forward —
        # restarted at every span, by subtracting the cumsum at the span's start.
        d_rel = delta - delta[..., :1]                   # (B,M,S) zero-based; d_rel[...,0]==0
        rise = delta[..., -1] - delta[..., 0]            # (B,M) net within-patch rise
        excl = torch.cumsum(rise, dim=1) - rise          # (B,M) EXCLUSIVE cumsum
        o = excl - excl.gather(1, start)                 # zero at each span's first slot
        delta_med = o.unsqueeze(-1) + d_rel              # (B,M,S)
    else:  # 'independent'
        delta_med = delta                                # legacy flat — BIT-IDENTICAL
    if valid is not None:
        # Padded slots are anchor-flat: no median gradient reaches their head_raw.
        delta_med = delta_med * valid.to(delta_med.dtype).unsqueeze(-1)
    m = anchor + delta_med                               # (B,M,S) median (risk)

    spread = F.softplus(head_raw[..., 1:]) + BG_QUANTILE_SPREAD_MIN  # (B,M,S,2*N_SPREADS)
    d_up = spread[..., :N_SPREADS]                       # τ>.5: .75/.9/.95
    d_dn = spread[..., N_SPREADS:]                       # τ<.5: .25/.1/.05
    # The carry is PER LEVEL when its last axis is 2*N_SPREADS, in the spread
    # columns' own layout ([.75 .9 .95 | .25 .1 .05]); a scalar or a trailing 1
    # widens every level alike, which is what the legacy default 0.0 does.
    c_up = c_dn = carry_spread
    if isinstance(carry_spread, torch.Tensor) and carry_spread.ndim and carry_spread.shape[-1] != 1:
        assert carry_spread.shape[-1] == 2 * N_SPREADS, (
            f"carry_spread's last axis must be 1 or {2 * N_SPREADS} (per level), "
            f"got {tuple(carry_spread.shape)}"
        )
        c_up, c_dn = carry_spread[..., :N_SPREADS], carry_spread[..., N_SPREADS:]
    o_up, o_dn = torch.cumsum(d_up, dim=-1), torch.cumsum(d_dn, dim=-1)
    # The carry is a DIFFERENT roll's increment, so it composes with this span's own spread in
    # QUADRATURE — independent increments add variances, and adding the two is the
    # perfectly-correlated bound (twice too wide by the fourth roll).  Skipped outright when there
    # is no carry, which keeps every training and single-window caller bit-identical.
    if not _carry_is_zero(carry_spread):
        c_up = torch.as_tensor(c_up, dtype=o_up.dtype, device=o_up.device)
        c_dn = torch.as_tensor(c_dn, dtype=o_dn.dtype, device=o_dn.device)
        o_up, o_dn = torch.hypot(c_up, o_up), torch.hypot(c_dn, o_dn)
    up = m.unsqueeze(-1) + o_up                          # (B,M,S,N_SPREADS) ascending
    dn = m.unsqueeze(-1) - o_dn                          # (B,M,S,N_SPREADS) descending

    # Assemble ascending τ: [.05 .1 .25 | .5 | .75 .9 .95].
    # dn is [.25 .1 .05] (descending in value) → flip to [.05 .1 .25] (ascending).
    q_tau = torch.cat([dn.flip(-1), m.unsqueeze(-1), up], dim=-1)  # (B,M,S,N_QUANTILES)
    assert q_tau.shape[-1] == N_QUANTILES, (
        f"assembled {q_tau.shape[-1]} quantiles, expected {N_QUANTILES}"
    )
    return q_tau, m


def last_bg_mgdl_from_context(
    context: torch.Tensor, stats: dict[str, dict[str, float]],
    patch_idx: "torch.Tensor | Sequence[int] | None" = None,
    step_idx: "torch.Tensor | Sequence[int] | None" = None,
) -> torch.Tensor:
    """Anchor BG (mg/dL) read out of the normalized context — the (a)→(b) bridge.

    The pipeline runs on RAW post-noise signals (no input/target smoothing): the
    context ``bg_absolute`` channel is the raw observed BG that produced the
    training input and target.  So the anchor is simply that context cell
    denormalized to mg/dL — it matches the training anchor as the SAME physical
    mg/dL, to within a sub-ulp round-trip difference (``data._build_sample`` reads
    the anchor off the raw mg/dL array directly, while inference reconstructs it
    via a z-unscale→f_inv round-trip training never does).  ``bg_absolute`` is the
    RISK-space input channel — feat 0 is ``z( f(bg) )`` (Kovatchev ``f`` applied
    before the z-score) as the sole input path — so the inverse is the plain
    z-score un-scale followed unconditionally by ``kovatchev_f_inv_np`` back to
    mg/dL; ``model.forward`` re-applies ``f`` internally.

    The general masked objective needs ``M`` anchors per sample, one per masked
    slot, each at a different ``(patch, step)`` cell — the last step of the span's
    left neighbour, or the first step of the right neighbour for a span starting
    at patch 0.  Pass them as index tensors and all ``M`` cross in ONE host
    transfer and one float64 NumPy inverse; the default reads the rightmost cell
    and returns ``(1,)``, bit-identical to the single-anchor form it replaces.
    Only visible cells may be indexed: feat 0 of a MASKED patch is a legal-looking
    ``z`` that decodes to an ordinary mg/dL, so a wrong index yields a plausible
    anchor rather than an error.

    Args:
        context: ``(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)`` normalized context.
        stats: normalization statistics dict (must carry ``bg_absolute``).
        patch_idx: ``(M,)`` patch indices (negatives count from the right).
            Default: the last patch.
        step_idx: ``(M,)`` within-patch step indices, same length as
            ``patch_idx``.  Default: the last step.

    Returns:
        anchor: ``(M,)`` BG in mg/dL, clamped to the physical range — ``(1,)`` in
        the default single-cell form.
    """
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    import numpy as np
    assert context.ndim == 3, (
        f"context must be (n_ctx, PATCH_SIZE, N_INPUT_FEATURES), got {tuple(context.shape)}"
    )
    n_ctx, n_steps = context.shape[0], context.shape[1]
    assert (patch_idx is None) == (step_idx is None), (
        "pass patch_idx and step_idx together, or neither"
    )
    if patch_idx is None:
        # Last context BG cell (newest patch, newest step).  Left-padding sits at
        # the FAR left, so the rightmost cell is always real data.
        p = torch.tensor([-1], dtype=torch.long)
        s = torch.tensor([-1], dtype=torch.long)
    else:
        p = torch.as_tensor(patch_idx, dtype=torch.long).reshape(-1)
        s = torch.as_tensor(step_idx, dtype=torch.long).reshape(-1)
        assert p.shape == s.shape, (
            f"patch_idx and step_idx must have the same length, got "
            f"{tuple(p.shape)} and {tuple(s.shape)}"
        )
    p = torch.where(p < 0, p + n_ctx, p)
    s = torch.where(s < 0, s + n_steps, s)
    assert bool(((p >= 0) & (p < n_ctx) & (s >= 0) & (s < n_steps)).all()), (
        f"anchor cell out of range for a ({n_ctx}, {n_steps}) context"
    )
    bg_mean = stats['bg_absolute']['mean']
    bg_std = stats['bg_absolute']['std']
    # feat 0 is z( f(bg) ) — the sole input path — so un-z-scoring yields a RISK
    # value; invert it back to mg/dL so the anchor handed to model.forward stays
    # physical (forward re-applies f internally — this keeps model.py / the loss
    # untouched).  One transfer for all M cells; the arithmetic is float64, as it
    # was when this read a single Python float.
    p = p.to(context.device)
    s = s.to(context.device)
    z = context[p, s, 0].detach().float().cpu().numpy().astype(np.float64)
    risk = z * (bg_std + 1e-8) + bg_mean
    mgdl = np.clip(kovatchev_f_inv_np(risk), BG_CLAMP_MIN, BG_CLAMP_MAX)
    return torch.tensor(mgdl, dtype=torch.float32, device=context.device)


class ModelEMA:
    """
    Exponential moving average of a model's floating-point state.

    Maintains a shadow ``state_dict`` that's blended after every accepted
    optimizer step.  Validation can be run under the EMA via the
    ``apply_to(model)`` context manager — the shadow is swapped into the
    model for the duration of the ``with`` block, then the live weights
    are restored on exit so training continues on the un-smoothed
    parameters.

    Why bother?  Threshold-crossing metrics (hypo recall, TIR error)
    are extremely sensitive to small μ shifts.  With Muon at momentum
    ``MUON_MOMENTUM`` the per-step weight jitter in μ-space is large relative to the
    clinical cutoffs, so consecutive validations 100 steps apart bounce
    around even when the loss is monotonically decreasing.  Averaging
    weights over a ~1k-step window smooths that out without altering
    the training trajectory itself.

    Args:
        model: The live model whose parameters will be tracked.
        decay: EMA decay factor in [0, 1).  Typical 0.999 (≈1k-step
            smoothing) or 0.9999 (≈10k-step smoothing).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        # Two parallel data structures so update() can iterate without doing
        # a fresh ``model.named_*()`` walk on every step:
        #   shadow      — the smoothed copy (the thing we read at validation)
        #   _param_refs — direct references into the live model so update()
        #                 can blend without needing the model again
        self.shadow: dict[str, torch.Tensor] = {}
        self._param_refs: list[tuple[str, torch.Tensor]] = []
        # Skip non-persistent buffers — they're typically things like RoPE
        # caches or attention masks that don't represent learned state, so
        # there is no value in averaging them and including them would just
        # bloat the checkpoint.
        non_persistent = getattr(model, '_non_persistent_buffers_set', set())
        for name, param in model.named_parameters():
            # Only float tensors are EMA-tracked (int64 buffers, bool masks,
            # etc. are passed through unchanged by apply_to below).
            if torch.is_floating_point(param):
                self.shadow[name] = param.detach().clone()
                self._param_refs.append((name, param))
        for name, buf in model.named_buffers():
            if name in non_persistent:
                continue
            if torch.is_floating_point(buf) and name not in self.shadow:
                self.shadow[name] = buf.detach().clone()
                self._param_refs.append((name, buf))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the model's current float tensors into the shadow copy."""
        # Gather the (shadow, live) pairs whose incoming live weights are FINITE.
        # A single NaN/inf folded into the shadow would persist forever
        # (decay*NaN + ... == NaN on every subsequent step), permanently poisoning
        # the EMA validation weights, so a non-finite tensor is skipped per-tensor
        # (its shadow keeps its last good value).  Selecting the finite subset
        # BEFORE the foreach preserves that exact per-tensor skip semantics.
        sel_shadow: list[torch.Tensor] = []
        sel_live: list[torch.Tensor] = []
        for name, ref in self._param_refs:
            if name in self.shadow:
                ref = ref.detach()
                if not torch.isfinite(ref).all():
                    continue
                sel_shadow.append(self.shadow[name])
                sel_live.append(ref)
        if sel_shadow:
            # Batched in-place blend: mathematically identical, per tensor, to
            # ``shadow.mul_(decay).add_(live, alpha=1-decay)`` == decay*shadow +
            # (1-decay)*live, but fused across all tensors to cut launch overhead.
            torch._foreach_mul_(sel_shadow, self.decay)
            torch._foreach_add_(sel_shadow, sel_live, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Defensive clone so the caller can't mutate our shadow accidentally
        # (checkpoints are saved from this dict).
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        # Same defensive clone on the way in — the caller might keep a
        # reference to the dict and mutate it after handing it to us.
        self.shadow = {k: v.detach().clone() for k, v in sd.items()}

    def to(self, device: torch.device) -> "ModelEMA":
        # Move every shadow tensor to ``device``.  Important when the
        # checkpoint was loaded onto CPU and we then ``.to('cuda')`` later.
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """
        Context manager that swaps shadow weights into ``model`` for the
        duration of the ``with`` block, then restores the live weights on
        exit.  The backup snapshot is taken from ``model.state_dict()``
        directly (not from ``self._param_refs``) so we restore *exactly*
        the live state even if other code mutated the model during the block.
        """
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # Start from a copy of the backup so non-EMA-tracked buffers (int64
        # counters, etc.) keep their live values even while the float weights
        # are temporarily replaced by the shadow copy.
        merged = dict(backup)
        for k, v in self.shadow.items():
            merged[k] = v
        model.load_state_dict(merged, strict=True)
        try:
            yield
        finally:
            # Always restore — even on exceptions raised inside the with block.
            model.load_state_dict(backup, strict=True)
