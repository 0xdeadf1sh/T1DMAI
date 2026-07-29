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

* ``create_attention_mask`` — produces the hybrid context-vs-prediction mask
  used by every temporal self-attention layer.  Bidirectional inside the
  context, bidirectional inside the prediction horizon, full attention from
  prediction → context, blocked from context → prediction.

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

* ``assemble_quantiles`` — turns the BG head's raw per-step output (a median
  delta + spreads) into an ascending 7-quantile fan in risk space, anchored at
  ``f(last_bg)``.  Returns ``(q_tau, median)``.

* ``last_bg_mgdl_from_context`` — denormalizes the rightmost context
  ``bg_absolute`` cell into mg/dL, the shared (a)→(b) bridge that produces the
  ``last_bg`` anchor at inference.

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
from typing import Iterator, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX


# Kovatchev risk-transform constants, re-anchored to the [40, 400] mg/dL device
# range (BG_CLAMP_MIN / BG_CLAMP_MAX): f(g) = SCALE * (ln(g)^POWER - OFFSET).
# The clinically-validated power POWER = 1.084 is kept from Kovatchev et al. (it
# encodes the hypo > hyper danger asymmetry); SCALE and OFFSET are re-solved so
# f(40) = -sqrt(10) and f(400) = +sqrt(10) — i.e. the risk 10*f^2 saturates at
# 100 exactly at both CGM rails. The zero-risk euglycemic center consequently
# sits at ~128 mg/dL (the log^POWER center of [40, 400]). These endpoints are
# verified in a unit test.
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



_ATTENTION_MASK_CACHE: "dict[tuple[int, int], torch.Tensor]" = {}


def create_attention_mask(n_context: int, n_prediction: int) -> torch.Tensor:
    """
    Create the hybrid attention mask for T1DMAI sequences.

    The mask shape is intentionally non-causal: context patches are fully
    observed and benefit from bidirectional attention, and prediction patches
    attend bidirectionally among themselves as well (the whole prediction
    horizon is decoded jointly in one forward pass).

    Pattern (True = attend, False = block)::

        Context  → Context     bidirectional  (every ctx sees every ctx)
        Context  → Prediction  blocked        (ctx can't peek at the future)
        Pred     → Context     full           (every pred sees all of ctx)
        Pred     → Prediction  bidirectional  (every pred sees every pred)

    Args:
        n_context: Number of context patches (C).
        n_prediction: Number of prediction patches (P).

    Returns:
        mask: (C+P, C+P) bool tensor.  Memoized by ``(n_context, n_prediction)``
            and returned as a SHARED read-only tensor — every caller treats it
            read-only (``collate_fn`` copies it into the batch mask; inference
            passes it straight through), so aliasing the cache is safe.
    """
    key = (n_context, n_prediction)
    cached = _ATTENTION_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    T = n_context + n_prediction
    # Start with all-False (block everything) and switch on the three regions
    # we want to allow.  The fourth region (context → prediction) stays False.
    mask = torch.zeros(T, T, dtype=torch.bool)

    # Context-to-context: bidirectional — every observed timestep can see
    # every other observed timestep, regardless of order.
    mask[:n_context, :n_context] = True

    # Prediction-to-context: full attention — predicted patches always see
    # every context patch.
    mask[n_context:, :n_context] = True

    # Prediction-to-prediction: bidirectional — every prediction patch sees
    # every other prediction patch.  The full horizon is decoded jointly in a
    # single forward pass, so there is no within-horizon causal constraint;
    # future leak is prevented by the blocked context → prediction region, not
    # by triangulating this block.
    mask[n_context:, n_context:] = True

    _ATTENTION_MASK_CACHE[key] = mask
    return mask


def kovatchev_f(g: torch.Tensor) -> torch.Tensor:
    """Kovatchev risk transform mg/dL (b) → risk (c), with the UNITS TRIPWIRE.

    ``f(g) = _KOVATCHEV_SCALE * (ln(g)^_KOVATCHEV_POWER - _KOVATCHEV_OFFSET)``
    (Kovatchev et al. 2000; the three constants live at the module top) — the
    symmetrizing transform whose risk-distance equates the clinical danger of a
    low and a high excursion, so the loss geometry inherits the hypo>hyper
    asymmetry for free.

    This variant is the **units tripwire**: it hard-asserts its argument is
    physical mg/dL (``g >= BG_CLAMP_MIN - 1e-3``), so a z-scored value (~[-3, 3])
    trips it loudly.  It is reserved for CONTROLLED callers that must never carry
    z-space — the ``f(last_bg)`` anchor and any re-``f`` of an inverted value.
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

    1. Clamp the **risk input** first to ``[f(BG_CLAMP_MIN), f(BG_CLAMP_MAX)]``
       (≈ ``[-3.1623, +3.1623]`` = ``[f(40), f(400)]`` = ±sqrt(10)).  This guarantees the
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
    return torch.sqrt(-2.0 * torch.log(r_bar)) * (24.0 / two_pi)


_GLOBAL_MEDIAN_BASIS_CACHE: "dict[tuple[int, int, str, torch.device, torch.dtype], torch.Tensor]" = {}


def get_global_median_basis(
    n: int, k: int, kind: str = 'dct',
    device: "torch.device | None" = None, dtype: "torch.dtype | None" = None,
) -> torch.Tensor:
    """Fixed ``(n, k)`` orthonormal low-frequency basis for the GLOBAL median
    projection (R3), built once per ``(n, k, kind)`` and cached.

    Identical construction to :func:`model.make_step_basis` (DCT-II cosine modes or
    orthonormal polynomials, ascending frequency/degree, L2-orthonormal columns) but
    evaluated over the FULL ``n = PREDICTION_PATCHES * PATCH_SIZE`` prediction horizon
    rather than a single patch.  ``assemble_quantiles`` projects the per-patch median
    delta onto ``span`` of these ``k`` columns; with ``k`` small the median is confined
    to a smooth low-frequency subspace (periods below ~``2*n/k`` steps — including the
    per-patch seam sawtooth — are unrepresentable), and the projection is an L2
    contraction (``||proj(x)|| <= ||x||``) so the per-patch offset cannot drift.

    The cache is keyed by ``(n, k, kind, device, dtype)``, so the basis materializes
    once already resident on the requested ``device``/``dtype`` (no per-forward H2D copy
    or dtype cast on the hot path); each call returns a defensive clone of that
    device-resident tensor so an in-place edit by any caller can never poison the cache.

    Args:
        n: horizon length ``PREDICTION_PATCHES * PATCH_SIZE`` (``>= 1``).
        k: number of basis columns ``G``, ``1 <= k <= n`` (caller clamps to ``min(G, n)``).
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


def assemble_quantiles(
    head_raw: torch.Tensor, last_bg_mgdl: torch.Tensor,
    carry_spread: "torch.Tensor | float" = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the BG head's raw output into an ascending risk-space quantile fan.

    The head emits ``1 + 2*N_SPREADS`` raw numbers per step (col0 = median delta;
    cols ``1..N_SPREADS`` = the ``τ>.5`` spreads nearest→far .75/.9/.95; cols
    ``N_SPREADS+1..2*N_SPREADS`` = the ``τ<.5`` spreads nearest→far .25/.1/.05).
    Spreads are passed through ``softplus`` + ``BG_QUANTILE_SPREAD_MIN`` (a strict
    positive floor against σ-collapse) and accumulated by ``cumsum`` so the fan is
    monotone by construction.  The anchor ``f(last_bg)`` is detached (a constant)
    and held flat across the whole horizon.

    Median assembly is gated by ``config.BG_HEAD_MEDIAN_MODE`` (R3, three modes):

    * ``'global'`` (DEFAULT, R3): the per-patch median delta ``delta = head_raw[..., 0]``
      is reshaped ``(B, P*S)`` C-contiguous patch-major (matching
      ``risk_loss._to_patch_major``) and PROJECTED onto a fixed low-frequency DCT-II
      subspace ``span(Bg)`` over the FULL ``P*S`` horizon: ``z = delta_flat @ Bg``,
      ``delta_global = z @ Bgᵀ``, ``m = anchor + delta_global.reshape(B,P,S)``.  A
      projection is an **L2 CONTRACTION** (``||proj(x)|| <= ||x||``), so the per-patch
      offset ``o[p] = m[:,p,0] − anchor`` is bounded and non-monotone in ``p`` — it
      CANNOT drift or amplify, structurally killing R1's unconstrained-integrator drift.
      The low-pass (small ``G``) leaves the median smooth (the per-patch period-2 / seam
      sawtooth is unrepresentable), so C0 seam-continuity is intentionally NOT pinned.
      At init (delta≈0 ⇒ z≈0 ⇒ delta_global≈0) ``m ≈ anchor`` everywhere (persistence).
    * ``'cumulative'`` (R1, BIT-IDENTICAL legacy): each patch CONTINUES from the previous
      patch's endpoint.  With ``d = head_raw[..., 0]``, ``d_rel = d − d[..., :1]``,
      ``rise = d[..., -1] − d[..., 0]``, and the EXCLUSIVE cumsum of ``rise`` over the
      patch dim (``o[..., 0]==0``) gives ``m = anchor + o[..., None] + d_rel`` — **C0
      continuity** at every seam and the FIRST step pinned EXACTLY to the anchor.
    * ``'independent'`` (BIT-IDENTICAL legacy): ``m = anchor + delta`` — each patch's
      within-patch median curve is INDEPENDENT, so the median can jump at the seams.

    Under every mode ``median == q_tau[..., 3]`` and the ascending fan hold: only the
    value of ``m`` changes; the ± band structure (``head_raw[..., 1:]`` → softplus →
    cumsum, plus ``carry_spread``) is untouched.

    ``carry_spread`` (risk space, default ``0.0`` → bit-identical to a bare fan)
    seeds the cumulative spread base on BOTH sides of the median, widening the band
    symmetrically: ``q(τ>.5) = m + carry_spread + cumsum(d+)`` and
    ``q(τ<.5) = m − carry_spread − cumsum(d−)``.  NO runtime caller passes it — the
    sole non-test call site is ``model.py``'s forward, which takes the default.
    ``inference.predict_rolling`` needs the same widening but cannot reach this
    argument (the assembly runs inside ``model.forward``), so it accumulates its own
    risk-space half-width and applies the identical shift post-forward on the
    returned ``q_tau``.  Keep the two in step: a change to the algebra here must be
    mirrored there, or the rolling band silently stops matching the fan it widens.

    Args:
        head_raw: ``(B, P, S, 1 + 2*N_SPREADS)`` raw head output (risk space).
        last_bg_mgdl: ``(B,)`` last context BG (mg/dL), the persistence anchor.
        carry_spread: risk-space scalar or tensor broadcastable to ``(B, P, S, 1)``
            that seeds the cumulative spread base (default ``0.0``).

    Returns:
        q_tau: ``(B, P, S, N_QUANTILES)`` quantiles in risk space, ascending in τ
            (index-for-index with ``QUANTILE_LEVELS``).
        median: ``(B, P, S)`` median in risk space (== ``q_tau[..., 3]``).
    """
    from config import (
        N_SPREADS, N_QUANTILES, BG_QUANTILE_SPREAD_MIN,
        BG_HEAD_MEDIAN_MODE, BG_HEAD_MEDIAN_GLOBAL_DIM, BG_HEAD_STEP_BASIS_TYPE,
    )
    import torch.nn.functional as F
    assert head_raw.ndim == 4 and head_raw.shape[-1] == 1 + 2 * N_SPREADS, (
        f"head_raw must be (B, P, S, {1 + 2 * N_SPREADS}), got {tuple(head_raw.shape)}"
    )
    assert last_bg_mgdl.ndim == 1 and last_bg_mgdl.shape[0] == head_raw.shape[0], (
        f"last_bg_mgdl must be (B,) matching head_raw batch, got {tuple(last_bg_mgdl.shape)}"
    )

    # Anchor: f(last_bg), detached (a constant), broadcast flat across (P, S).
    # Clamp the persistence anchor into the physical BG range first — a CGM-noisy
    # last reading can sit just above the simulator ceiling (e.g. 402 mg/dL), and
    # the anchor is a constant, so clamping silences kovatchev_f's ceiling warning
    # without touching any gradient.
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    anchor = kovatchev_f(last_bg_mgdl.detach().clamp(BG_CLAMP_MIN, BG_CLAMP_MAX))  # (B,)
    anchor = anchor.view(-1, 1, 1)                       # (B, 1, 1)
    delta = head_raw[..., 0]                             # (B, P, S)
    assert BG_HEAD_MEDIAN_MODE in ('global', 'cumulative', 'independent'), (
        f"BG_HEAD_MEDIAN_MODE must be 'global'/'cumulative'/'independent', "
        f"got {BG_HEAD_MEDIAN_MODE!r}")
    if BG_HEAD_MEDIAN_MODE == 'global':
        # R3 GLOBAL SMOOTH-BASIS median: project the per-patch median delta onto a
        # fixed low-frequency DCT-II subspace over the full P*S horizon.  A projection
        # is an L2 contraction, so the per-patch offset cannot drift (kills R1's
        # integrator).  delta_flat is C-contiguous patch-major (flat = p*S + s),
        # matching risk_loss._to_patch_major so the basis low-passes the TIME axis.
        B_, P_, S_ = delta.shape
        n = P_ * S_
        g = min(BG_HEAD_MEDIAN_GLOBAL_DIM, n)            # clamp G (P==1 edge: g<=S)
        Bg = get_global_median_basis(
            n, g, BG_HEAD_STEP_BASIS_TYPE,
            device=delta.device, dtype=delta.dtype,
        )                                                # (n, g) orthonormal columns
        delta_flat = delta.reshape(B_, n)                # (B, P*S) patch-major
        delta_global = (delta_flat @ Bg) @ Bg.transpose(0, 1)  # proj onto span(Bg)
        m = anchor + delta_global.reshape(B_, P_, S_)    # (B,P,S) median (risk)
    elif BG_HEAD_MEDIAN_MODE == 'cumulative':
        # Cumulative cross-patch-continuity median (R1): each patch continues from
        # the previous patch's endpoint.  d_rel zeroes each patch's curve at step 0;
        # the exclusive cumsum of the per-patch rise carries the offset forward.
        d_rel = delta - delta[..., :1]                   # (B,P,S) zero-based; d_rel[...,0]==0
        rise = delta[..., -1] - delta[..., 0]            # (B,P) net within-patch rise
        o = torch.cumsum(rise, dim=1) - rise             # (B,P) EXCLUSIVE cumsum: o[...,0]==0
        delta_cum = o.unsqueeze(-1) + d_rel              # (B,P,S)
        m = anchor + delta_cum                           # (B,P,S) median (risk), C0 at seams
    else:  # 'independent'
        m = anchor + delta                               # legacy flat — BIT-IDENTICAL

    spread = F.softplus(head_raw[..., 1:]) + BG_QUANTILE_SPREAD_MIN  # (B,P,S,2*N_SPREADS)
    d_up = spread[..., :N_SPREADS]                       # τ>.5: .75/.9/.95
    d_dn = spread[..., N_SPREADS:]                       # τ<.5: .25/.1/.05
    up = m.unsqueeze(-1) + carry_spread + torch.cumsum(d_up, dim=-1)  # (B,P,S,N_SPREADS) ascending
    dn = m.unsqueeze(-1) - carry_spread - torch.cumsum(d_dn, dim=-1)  # (B,P,S,N_SPREADS) descending

    # Assemble ascending τ: [.05 .1 .25 | .5 | .75 .9 .95].
    # dn is [.25 .1 .05] (descending in value) → flip to [.05 .1 .25] (ascending).
    q_tau = torch.cat([dn.flip(-1), m.unsqueeze(-1), up], dim=-1)  # (B,P,S,N_QUANTILES)
    assert q_tau.shape[-1] == N_QUANTILES, (
        f"assembled {q_tau.shape[-1]} quantiles, expected {N_QUANTILES}"
    )
    return q_tau, m


def last_bg_mgdl_from_context(
    context: torch.Tensor, stats: dict[str, dict[str, float]],
) -> torch.Tensor:
    """The ``last_bg`` anchor (mg/dL) for inference — the LAST context BG input.

    The pipeline runs on RAW post-noise signals (no input/target smoothing): the
    context ``bg_absolute`` channel is the raw observed BG that produced the
    training input and target.  So the anchor is simply that last context value
    denormalized to mg/dL — it matches the training anchor as the SAME physical
    mg/dL, to within a sub-ulp round-trip difference (``data._build_sample`` uses
    ``last_bg = bg[pred_start-1]`` directly, while inference reconstructs it via a
    z-unscale→f_inv round-trip training never does).  ``bg_absolute`` is the
    RISK-space input channel — feat 0 is ``z( f(bg) )`` (Kovatchev ``f`` applied
    before the z-score) as the sole input path — so the inverse is the plain
    z-score un-scale followed unconditionally by ``kovatchev_f_inv_np`` back to
    mg/dL; ``model.forward`` re-applies ``f`` internally.

    Args:
        context: ``(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)`` normalized context.
        stats: normalization statistics dict (must carry ``bg_absolute``).

    Returns:
        last_bg: ``(1,)`` anchor BG in mg/dL, clamped to the physical range.
    """
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    assert context.ndim == 3, (
        f"context must be (n_ctx, PATCH_SIZE, N_INPUT_FEATURES), got {tuple(context.shape)}"
    )
    bg_mean = stats['bg_absolute']['mean']
    bg_std = stats['bg_absolute']['std']
    # Last context BG cell (newest patch, newest step), un-z-scored.  Left-padding
    # sits at the FAR left, so the rightmost cell is always real data.
    last_z = float(context[-1, -1, 0].detach().float().cpu())
    # feat 0 is z( f(bg) ) — the sole input path — so un-z-scoring yields a RISK
    # value; invert it back to mg/dL so the anchor handed to model.forward stays
    # physical (forward re-applies f internally — this keeps model.py / the loss
    # untouched).
    import numpy as np
    last_risk = last_z * (bg_std + 1e-8) + bg_mean
    last_val = float(kovatchev_f_inv_np(np.asarray(last_risk, dtype=np.float64)))
    last_mgdl = float(min(max(last_val, BG_CLAMP_MIN), BG_CLAMP_MAX))
    return torch.tensor([last_mgdl], dtype=torch.float32, device=context.device)


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
