"""Seed hashing, attention masks, Kovatchev risk transform, quantile assembly, weight EMA."""

import hashlib
import math
import warnings
from contextlib import contextmanager
from typing import Iterator, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn

from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX


# f(g) = SCALE*(ln(g)^POWER - OFFSET), Kovatchev et al. POWER kept; SCALE/OFFSET
# re-solved so f(40) = -sqrt(10), f(400) = +sqrt(10) — risk 10*f^2 hits 100 at both.
# Anchors are NOT the clamp [BG_CLAMP_MIN, BG_CLAMP_MAX] = [10, 400] mg/dL, so the
# realised risk range [f(10), f(400)] = [-6.8198, +3.1623] is asymmetric by design.
_KOVATCHEV_SCALE = 2.2211457449985317
_KOVATCHEV_POWER = 1.084
_KOVATCHEV_OFFSET = 5.540076976170212

def compute_patient_seed(master_seed: int, step: int, position: int) -> int:
    """SHA-256 of (master_seed, step, position) to a 63-bit seed.

    Mod 2^63-1 keeps it inside int64 clear of the sign bit; collision rate over a
    ~1e5-step run is ~5e-7.
    """
    key = f"{master_seed}:{step}:{position}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest, 16) % (2**63 - 1)


def create_attention_mask_from_visible(
    visible: torch.Tensor, is_pad: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Per-sample attention mask for an arbitrary masked set. True = attend.

    visible row → visible col allowed; visible row → masked col BLOCKED (evidence
    never reads a prediction); masked row → any real col allowed; pad row and pad col
    blocked except the diagonal.

    Trap: dropping ``attn &= ~is_pad[:, :, None]`` opens a pad row onto every visible
    column and leaves the forward OUTPUT unchanged — only a full-mask comparison
    catches it. The diagonal write is the sole guard against an all-False row
    (softmax NaN). Never memoized: no cheap key identifies a masked set.

    Args:
        visible: ``(B, T)`` bool, True where the position's BG is observed; entries
            at PAD positions are ignored.
        is_pad: ``(B, T)`` bool left-padding flags, or None for no padding.

    Returns:
        attn: ``(B, T, T)`` bool. ``model.forward`` gives it the head axis with
            ``unsqueeze(1)`` — passing it straight aligns B onto the head axis.
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
    """``(C+P, C+P)`` bool right-edge mask: C visible patches then P masked, no padding.

    Built fresh on every call; see the general form for why no memo can be correct.
    """
    T = n_context + n_prediction
    visible = torch.zeros(1, T, dtype=torch.bool)
    visible[0, :n_context] = True
    return create_attention_mask_from_visible(visible)[0]


def kovatchev_f(g: torch.Tensor) -> torch.Tensor:
    """Kovatchev risk transform mg/dL (b) → risk (c). The UNITS TRIPWIRE.

    Hard-asserts ``g >= BG_CLAMP_MIN - 1e-3``, which no z-scored value satisfies, so
    a z-space leak trips loudly. Reserved for controlled callers that must never
    carry z-space — ``f(anchor_bg)`` and any re-``f`` of an inverted value. Above
    ``BG_CLAMP_MAX + 1e-3`` warns only, and does not clamp. Never differentiated.
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
    """Kovatchev risk transform, TARGET path: physical clamp then ``f``.

    Clamps to ``[BG_CLAMP_MIN, BG_CLAMP_MAX]``, warning when it bites beyond a small
    tolerance — a physical backstop, not a unit guard. The tripwire is on
    :func:`kovatchev_f`.
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

    Scrubs non-finite input to the band edges (``clamp`` propagates NaN), clamps the
    risk input to ``[f(BG_CLAMP_MIN), f(BG_CLAMP_MAX)]`` ≈ ``[-6.8198, +3.1623]`` so
    the base stays >= 0 and ``exp`` cannot overflow fp32, then clamps the mg/dL output
    to ``[BG_CLAMP_MIN, BG_CLAMP_MAX]``. Never differentiated.
    """
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    r_lo = _KOVATCHEV_SCALE * (math.log(BG_CLAMP_MIN) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)
    r_hi = _KOVATCHEV_SCALE * (math.log(BG_CLAMP_MAX) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)
    r = torch.nan_to_num(r, nan=r_lo, posinf=r_hi, neginf=r_lo)
    r = r.clamp(r_lo, r_hi)
    base = r / _KOVATCHEV_SCALE + _KOVATCHEV_OFFSET  # >= 0 after the input clamp
    g = torch.exp(base.pow(1.0 / _KOVATCHEV_POWER))
    return g.clamp(BG_CLAMP_MIN, BG_CLAMP_MAX)


def kovatchev_f_np(g: "np.ndarray") -> "np.ndarray":
    """NumPy Kovatchev risk transform mg/dL (b) → risk (c), INPUT path.

    Sibling of :func:`kovatchev_f` for the NumPy input-build and stat-fit sites
    (``data.py``, ``normalization.py``), carrying the same physical clamp to
    ``[BG_CLAMP_MIN, BG_CLAMP_MAX]`` as :func:`kovatchev_f_target` — not the tripwire —
    so the stat fit and the input transform stay bit-consistent.
    """
    g = np.clip(g, BG_CLAMP_MIN, BG_CLAMP_MAX)
    return _KOVATCHEV_SCALE * (np.log(g) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)


def kovatchev_f_inv_np(r: "np.ndarray") -> "np.ndarray":
    """NumPy inverse Kovatchev risk transform risk (c) → mg/dL (b).

    Mirrors :func:`kovatchev_f_inv` guard for guard.
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
    """``(n_bins,)`` center hours of the circular hour-of-day bins, width ``24/n_bins``."""
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    return (torch.arange(n_bins, dtype=torch.float32) + 0.5) * (24.0 / n_bins)


def time_of_day_bin_target(hour: torch.Tensor, n_bins: int, smooth_bins: float) -> torch.Tensor:
    """``(..., n_bins)`` soft circular hour-of-day target; rows sum to 1.

    Label mass decays as a wrapped Gaussian of circular bin-distance, std
    ``smooth_bins`` in bins; ``smooth_bins <= 0`` gives a one-hot at the nearest bin.
    ``hour`` is ``(...,)`` in ``[0, 24)``.
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
    """``(..., 2)`` probability-weighted mean of ``(cos, sin)`` over the bin-center angles.

    ``probs`` is ``(..., n_bins)`` non-negative; a resultant length in ``[0, 1]``
    assumes rows sum to 1.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    centers = time_of_day_bin_centers(n_bins).to(probs.device)
    th = centers * (2.0 * math.pi / 24.0)                    # (n_bins,)
    cos = (probs * torch.cos(th)).sum(dim=-1)
    sin = (probs * torch.sin(th)).sum(dim=-1)
    return torch.stack([cos, sin], dim=-1)


def time_of_day_decode_bins(logits: torch.Tensor, n_bins: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(..., n_bins)`` logits → ``(hour, R)``.

    ``hour`` is ``(...,)`` in ``[0, 24)`` from the resultant angle; ``R`` is ``(...,)``
    its length in ``[0, 1]``, read as confidence.
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
    """Scalar fp32 CE of ``(..., n_bins)`` logits against the soft circular hour target.

    ``target_hours`` is ``(...,)`` in ``[0, 24)``; ``smooth_bins <= 0`` => one-hot.
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
    """Scalar phase-advance penalty coupling two INDEPENDENT forward passes.

    Window k+1's origin sits ``advance_hours`` (one horizon) after window k's, so k's
    origin-patch resultant rotated by ``2*pi*advance_hours/24`` must land on k+1's.
    Matched in the raw ``(cos, sin)`` plane — atan2-free, stable gradient. Only patch 0
    of each window is used. Not redundant with a within-window advance penalty: the
    two windows are separate forwards over different contexts.

    Args:
        logits_k, logits_next: ``(B, P, n_bins)`` per-patch bin logits.
        valid: ``(B,)`` bool — only True rows enter the mean (a finite 0 when none);
            None => plain mean over B.
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
    """``(B,)`` fp32 ``|cross-window clock step − advance_hours|`` in hours, off patch 0.

    ~0 means the rolling clock advances by exactly one horizon across the window seam.
    The caller masks by the per-sample validity flag.
    ``logits_k``/``logits_next`` are ``(B, P, n_bins)``.
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
    """``(B,)`` fp32 mean ``|inter-patch clock step − advance_hours|`` in hours.

    ~0 means the predicted clock marches forward one patch at a time.
    ``logits`` is ``(B, P, n_bins)``; ``P < 2`` returns zeros.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    assert logits.dim() == 3, f"expected (B, P, n_bins), got {tuple(logits.shape)}"
    if logits.shape[1] < 2:
        return logits.new_zeros(logits.shape[0])
    hours, _ = time_of_day_decode_bins(logits, n_bins)       # (B, P)
    r = circular_hour_residual(hours[:, 1:], hours[:, :-1])  # (B, P-1)
    return (r - advance_hours).abs().mean(dim=-1)


def _resultant_np(probs: np.ndarray, n_bins: int) -> "tuple[float, float]":
    """``(cos, sin)`` resultant of one ``(n_bins,)`` distribution.

    Kept in step with :func:`time_of_day_resultant` so the geometry core never drifts
    from the trained probe.
    """
    assert n_bins >= 1, f"need n_bins >= 1, got {n_bins}"
    assert probs.shape[-1] == n_bins, f"expected last dim {n_bins}, got {probs.shape}"
    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * (24.0 / n_bins)
    th = centers * (2.0 * math.pi / 24.0)
    cos = float((probs * np.cos(th)).sum())
    sin = float((probs * np.sin(th)).sum())
    return cos, sin


def _fractional_roll(row: np.ndarray, shift_bins: float) -> np.ndarray:
    """Circular shift of a ``(n_bins,)`` row by a real number of bins.

    Positive ``shift_bins`` moves forward in time (higher bin index). A fractional
    shift blends the two adjacent integer rolls, preserving the ``n_bins-1``↔``0``
    adjacency.
    """
    assert row.ndim == 1, f"expected 1-D row, got {row.shape}"
    lo = int(math.floor(shift_bins))
    f = shift_bins - lo
    return (1.0 - f) * np.roll(row, lo) + f * np.roll(row, lo + 1)


def aggregate_origin_belief(probs: np.ndarray, advance_hours: float,
                            bin_hours: float) -> np.ndarray:
    """Fuse ``(P, n_bins)`` per-patch beliefs into one ``(n_bins,)`` origin belief, sum 1.

    Patch ``p``'s belief is the origin advanced by ``p*advance_hours``, so de-rotate it
    by ``-p*advance_hours/bin_hours`` bins, average, renormalize. Agreement across
    patches sharpens the fused belief, disagreement diffuses it, so its resultant
    length self-weights inter-patch consistency. Wrap-safe. ``bin_hours`` is
    ``24/n_bins``; ``P == 1`` returns row 0 unchanged.
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

    wedges: ``(n_bins, arc_segments+2, 2)``; vertex 0 is the center ``(0, 0)``, the
        rest trace bin ``k``'s outer arc at radius ``m_k``.
    magnitudes: ``(n_bins,)`` wedge outer radii in ``[0, 1]``.
    hand: ``(2,)`` resultant hand, length ``R``.
    R: resultant length in ``[0, 1]`` — rotation-INVARIANT.
    """
    wedges: np.ndarray
    magnitudes: np.ndarray
    hand: np.ndarray
    R: float


def _hour_to_unit(hour: np.ndarray) -> np.ndarray:
    """FROZEN y-up clock map ``u(h) = (sin(2*pi*h/24), cos(2*pi*h/24))`` → ``(..., 2)``.

    Hour 0 at 12-o'clock top, hours increasing clockwise (6 right, 12 bottom, 18 left).
    """
    a = hour * (2.0 * math.pi / 24.0)
    return np.stack([np.sin(a), np.cos(a)], axis=-1)


_CLOCK_TICK_HOURS = (0.0, 6.0, 12.0, 18.0)


def clock_reference_ticks() -> np.ndarray:
    """``(4, 2)`` y-up unit vectors for the fixed 0/6/12/18 h dial ticks.

    From :func:`_hour_to_unit`, so host adapters re-derive no trigonometry.
    """
    return _hour_to_unit(np.array(_CLOCK_TICK_HOURS))


def clock_wedge_geometry(probs: np.ndarray, rotation_hours: float = 0.0,
                         arc_segments: int = 6) -> ClockGeometry:
    """``ClockGeometry`` for one ``(n_bins,)`` belief on the hour-of-day dial.

    Bin ``k`` spans hours ``[k*bin_hours, (k+1)*bin_hours)``, ``bin_hours = 24/n_bins``
    inferred from ``probs``. ``rotation_hours`` is added to every mapped hour BEFORE the
    ``u(h)`` map — angles only, no re-binning — so cursor motion is smooth and ``R`` is
    invariant under it. ``m_k = p_k / p.max()``; an all-zero belief gives zeros and
    ``R = 0``, no NaN. ``arc_segments >= 1`` chords per wedge.
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
    """Signed circular residual (pred − true) wrapped to (-12, 12] hours (...,).

    Positive => the clock reads ahead of truth. Its absolute value equals
    ``circular_hour_error``; the sign is what bias/precision need.
    """
    return (pred_hour - true_hour + 12.0) % 24.0 - 12.0


def circular_bias_hours(pred_hour: torch.Tensor, true_hour: torch.Tensor) -> torch.Tensor:
    """Scalar signed systematic clock offset in (-12, 12] hours — a correctable constant.

    The angle of the mean resultant vector; naive angle averaging breaks at the 24 h wrap.
    """
    two_pi = 2.0 * math.pi
    delta = circular_hour_residual(pred_hour, true_hour) * (two_pi / 24.0)
    mean_angle = torch.atan2(torch.sin(delta).mean(), torch.cos(delta).mean())
    return mean_angle * (24.0 / two_pi)


def circular_std_hours(pred_hour: torch.Tensor, true_hour: torch.Tensor) -> torch.Tensor:
    """Scalar circular std of the residual in hours: ``sqrt(-2 ln R_bar)``.

    ``R_bar`` is clamped to ``[1e-6, 1.0]``: the floor keeps a near-uniform smear finite
    rather than ``inf``; the ceiling stops fp32 rounding just above 1.0 turning
    ``sqrt(-2 log R_bar)`` into the square root of a negative — a NaN.
    """
    two_pi = 2.0 * math.pi
    delta = circular_hour_residual(pred_hour, true_hour) * (two_pi / 24.0)
    r_bar = torch.sqrt(
        torch.cos(delta).mean() ** 2 + torch.sin(delta).mean() ** 2
    ).clamp(1e-6, 1.0)
    # ``+ 0.0`` maps -0.0 to +0.0 (``sqrt(-0.0) == -0.0`` prints as "-0.00 h" on the
    # validation table); ``clamp(min=0.0)`` returns -0.0 and does not fix it.
    return torch.sqrt(-2.0 * torch.log(r_bar)) * (24.0 / two_pi) + 0.0


_BSPLINE_STEP_WEIGHT_CACHE: "dict[tuple[int, bool, bool], torch.Tensor]" = {}


def bspline_step_weights(L: int, has_left: bool, has_right: bool) -> torch.Tensor:
    """``(L*PATCH_SIZE, L + has_left + has_right)`` fp32 step-state weight matrix.

    Rows are the span's steps, patch-major: row ``(i-1)*S + j`` is step ``j`` of masked
    patch ``i`` (``1..L``). Columns are the span's nodes in order, starting at node
    ``lo``: the left visible neighbour when ``has_left``, then the ``L`` masked patches,
    then the right visible neighbour when ``has_right``.

    Each node sits at its patch centre, so step ``j`` of patch ``i`` sits at
    ``c = i + (j - (S-1)/2) / S`` in node units, and the row is the uniform cubic
    B-spline evaluated there over nodes ``k-1..k+2``, ``k = floor(c)``, each index
    clamped into ``[lo, hi]`` — the end node repeats. Every row sums to 1.

    Cached per ``(L, has_left, has_right)``; each call returns a clone, so an in-place
    edit cannot poison the cache. :func:`step_states` computes the same states by
    gathering nodes per slot, which needs no per-span grouping.
    """
    from config import PATCH_SIZE
    assert L >= 1, f"span length must be >= 1, got {L}"
    key = (int(L), bool(has_left), bool(has_right))
    W = _BSPLINE_STEP_WEIGHT_CACHE.get(key)
    if W is None:
        S = PATCH_SIZE
        lo = 0 if has_left else 1
        hi = L + 1 if has_right else L
        W = torch.zeros(L * S, hi - lo + 1, dtype=torch.float32)
        for i in range(1, L + 1):
            for j in range(S):
                dc = (j - (S - 1) / 2) / S       # offset from the patch centre
                k, u = (i - 1, dc + 1.0) if dc < 0 else (i, dc)
                w = ((1 - u) ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
                     (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6, u ** 3 / 6)
                for w_o, o in zip(w, (-1, 0, 1, 2)):
                    W[(i - 1) * S + j, min(max(k + o, lo), hi) - lo] += w_o
        _BSPLINE_STEP_WEIGHT_CACHE[key] = W
    return W.clone()


def _span_layout(
    mask_idx: "torch.Tensor | None", valid: "torch.Tensor | None",
    B: int, M: int, device: torch.device,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Group the ``M`` head slots into contiguous masked spans.

    Slot ``j`` continues slot ``j-1`` iff ``mask_idx[j] == mask_idx[j-1] + 1`` and, when
    ``valid`` is given, both are real. The sampler never lets two masked spans abut, so
    adjacency in ``mask_idx`` identifies a span exactly. Padded slots gather patch 0 and
    cannot continue a span (that needs a predecessor at patch −1), so they fall out as
    singletons even when a real span does start at patch 0.
    ``mask_idx is None`` => all ``M`` slots are one contiguous span.

    Returns:
        start: ``(B, M)`` int64 slot index at which each slot's span begins.
        length: ``(B, M)`` int64 span length ``L`` in patches, per slot.
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
    # Running max of "index if a span starts here else -1" = the current span's start.
    # Written as a prefix max under an (M, M) causal mask, not torch.cummax: cummax is outside
    # the Core ATen opset and refuses to lower to ExecuTorch. M is MAX_MASKED_PATCHES.
    src = torch.where(new, ar_b, torch.full_like(ar_b, -1)).unsqueeze(1)
    causal = torch.ones(M, M, dtype=torch.bool, device=device).tril().unsqueeze(0)
    start = torch.where(causal, src, torch.full_like(src, -1)).amax(dim=2)
    ones = torch.ones(B, M, dtype=torch.long, device=device)
    counts = torch.zeros(B, M, dtype=torch.long, device=device).scatter_add_(1, start, ones)
    length = counts.gather(1, start)
    return start, length


def step_states(
    x: torch.Tensor, mask_idx: torch.Tensor, attn_mask: torch.Tensor,
    valid: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """``(B, M, PATCH_SIZE, D)`` per-step hidden states for the BG head.

    A masked span's nodes are its ``L`` patches plus the visible patch on each side where
    one exists — inside the window and readable by the span under ``attn_mask``, so a pad
    row is never a node. Node states are ``x`` at those patches; the head's step states
    are the uniform cubic B-spline over them at the step coordinates of
    :func:`bspline_step_weights`. One code path for forecast, backcast and infill, and
    the interpolation is C2 across every patch edge inside a span.

    Nodes are gathered per slot rather than through the matrix, so spans of different
    length in one batch need no grouping. A padded or invalid slot is a singleton span
    and its states are arbitrary but finite; downstream ``valid`` discards them.

    Args:
        x: ``(B, T, D)`` final-normed patch states.
        mask_idx: ``(B, M)`` int64 patch index per head slot.
        attn_mask: ``(T, T)``, ``(B, T, T)`` or ``(B, 1, T, T)`` bool, True = attend.
        valid: ``(B, M)`` bool, False on padded slots. Optional — see
            :func:`_span_layout`.
    """
    from config import PATCH_SIZE
    assert x.ndim == 3, f"x must be (B, T, D), got {tuple(x.shape)}"
    B, T, D = x.shape
    assert mask_idx.shape[0] == B and mask_idx.ndim == 2, (
        f"mask_idx must be (B, M) with B={B}, got {tuple(mask_idx.shape)}"
    )
    assert attn_mask.dtype == torch.bool, (
        f"attn_mask must be bool (True = attend), got {attn_mask.dtype}"
    )
    M = mask_idx.shape[1]
    S = PATCH_SIZE
    dev = x.device
    attn = attn_mask
    if attn.ndim == 2:
        attn = attn.unsqueeze(0).expand(B, T, T)
    elif attn.ndim == 4:
        attn = attn[:, 0]

    node_state = x.gather(1, mask_idx.unsqueeze(-1).expand(B, M, D))   # (B, M, D)
    start, length = _span_layout(mask_idx, valid, B, M, dev)
    end = start + length - 1
    p_first, p_last = mask_idx.gather(1, start), mask_idx.gather(1, end)
    la_pos, ra_pos = (p_first - 1).clamp(min=0), (p_last + 1).clamp(max=T - 1)
    rows = torch.arange(B, device=dev).unsqueeze(1)
    has_l = (p_first > 0) & attn[rows, p_first, la_pos]
    has_r = (p_last + 1 < T) & attn[rows, p_last, ra_pos]
    xl = x.gather(1, la_pos.unsqueeze(-1).expand(B, M, D))
    xr = x.gather(1, ra_pos.unsqueeze(-1).expand(B, M, D))
    slot = torch.arange(M, device=dev).unsqueeze(0).expand(B, M)
    i = slot - start + 1                                 # this slot's node index, 1..L
    L = length
    lo = torch.where(has_l, torch.zeros_like(L), torch.ones_like(L))
    hi = torch.where(has_r, L + 1, L)

    def node(n: torch.Tensor) -> torch.Tensor:
        n = torch.maximum(torch.minimum(n, hi), lo)      # the end node repeats
        st = node_state.gather(
            1, (start + n - 1).clamp(0, M - 1).unsqueeze(-1).expand(B, M, D))
        st = torch.where((n == 0).unsqueeze(-1), xl, st)
        return torch.where((n == L + 1).unsqueeze(-1), xr, st)

    out = []
    for j in range(S):
        dc = (j - (S - 1) / 2) / S                       # offset from the patch centre
        k, u = (i - 1, dc + 1.0) if dc < 0 else (i, dc)
        w = ((1 - u) ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
             (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6, u ** 3 / 6)
        out.append(sum(w_o * node(k + o) for w_o, o in zip(w, (-1, 0, 1, 2))))
    return torch.stack(out, 2)                           # (B, M, S, D)


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

    ``head_raw`` columns: col 0 median delta; cols ``1..N_SPREADS`` the ``τ>.5`` spreads
    nearest→far ``.75/.9/.95``; cols ``N_SPREADS+1..2*N_SPREADS`` the ``τ<.5`` spreads
    nearest→far ``.25/.1/.05``. Spreads pass through ``softplus`` +
    ``BG_QUANTILE_SPREAD_MIN`` (a strict positive floor against σ-collapse) and are
    accumulated by ``cumsum``, so the fan is monotone by construction. The anchor
    ``f(anchor_bg)`` is detached and held flat across the ``PATCH_SIZE`` steps of its own
    slot.

    The ``M`` axis is a gathered set of masked patches, not a trailing horizon: a span may
    end at patch ``T−1`` (forecast), start at patch 0 (backcast) or sit between visible
    patches (infill). The median is ``m = anchor + head_raw[..., 0]``, per slot and per
    step, with ``median == q_tau[..., 3]``. Nothing here couples the slots: the median is
    continuous across a span's patch edges because :func:`step_states` interpolates the
    head's INPUT, not because this assembly smooths its output.

    ``carry_spread`` (risk space, default ``0.0`` → bit-identical to a bare fan) seeds the
    cumulative spread base on BOTH sides: ``q(τ>.5) = m + hypot(c_up, cumsum(d+))``,
    ``q(τ<.5) = m − hypot(c_dn, cumsum(d−))`` — QUADRATURE, because the carry is another
    roll's increment and independent increments add variances; adding the two is the
    perfectly-correlated bound. It is PER LEVEL — a trailing axis of ``2*N_SPREADS`` in
    the spread columns' own layout ``[.75 .9 .95 | .25 .1 .05]`` — and a scalar widens all
    six alike. One value shared across the levels re-seeds every level from the outermost
    one's carry and flattens the fan into a slab
    (``../T1DMCOMMON/SPEC/inference.md`` §8.1). No runtime caller passes it: the sole
    non-test call site is ``model.forward``, which takes the default.
    ``inference.predict_rolling`` needs the same widening but cannot reach this argument
    (the assembly runs inside ``model.forward``), so it repeats the identical quadrature
    post-forward on the returned ``q_tau``. Change the algebra here and it must be mirrored
    there, or the rolling band silently stops matching the fan it widens.

    Args:
        head_raw: ``(B, M, S, 1 + 2*N_SPREADS)`` risk space, one slot per masked patch.
        anchor_bg_mgdl: ``(B, M)`` per-slot anchor BG in mg/dL, ONE-SIDED and
            left-preferring — the last step of the span's LEFT neighbour, or the first step
            of the right neighbour when the span starts at patch 0. Every slot of a span
            carries the same value, so it is NOT the nearest visible evidence for a slot at
            the span's right edge; evaluation bins on the two-sided distance ``d``, never
            on this. ``(B,)`` is the legacy single-span form, and then ``mask_idx`` must be
            None.
        mask_idx: ``(B, M)`` int64 patch index of each slot, shape-checked and required
            whenever ``anchor_bg_mgdl`` is ``(B, M)``. None selects the legacy
            single-span form.
        valid: ``(B, M)`` bool, False on padded slots. Optional — passing it pins a padded
            slot's median to its anchor, so no gradient reaches ``head_raw[..., 0]`` there.
        carry_spread: risk-space scalar, a tensor broadcastable to ``(B, M, S, 1)`` (every
            level alike), or one broadcastable to ``(B, M, S, 2*N_SPREADS)`` in the spread
            columns' layout ``[.75 .9 .95 | .25 .1 .05]`` (per level).

    Returns:
        q_tau: ``(B, M, S, N_QUANTILES)`` risk space, ascending in τ, index-for-index with
            ``QUANTILE_LEVELS``.
        median: ``(B, M, S)`` risk space (== ``q_tau[..., 3]``).
    """
    from config import N_SPREADS, N_QUANTILES, BG_QUANTILE_SPREAD_MIN
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
            "a (B, M) anchor is a general masked set and must come with its mask_idx"
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

    # Clamp first: a CGM-noisy last reading can sit just above the simulator ceiling
    # (e.g. 402 mg/dL). The anchor is a constant, so this silences kovatchev_f's
    # ceiling warning and touches no gradient.
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    anchor = kovatchev_f(anchor_bm.detach().clamp(BG_CLAMP_MIN, BG_CLAMP_MAX))  # (B,M)
    anchor = anchor.unsqueeze(-1)                        # (B, M, 1)
    delta_med = head_raw[..., 0]                         # (B, M, S)
    if valid is not None:
        # Padded slots are anchor-flat: no median gradient reaches their head_raw.
        delta_med = delta_med * valid.to(delta_med.dtype).unsqueeze(-1)
    m = anchor + delta_med                               # (B,M,S) median (risk)

    spread = F.softplus(head_raw[..., 1:]) + BG_QUANTILE_SPREAD_MIN  # (B,M,S,2*N_SPREADS)
    d_up = spread[..., :N_SPREADS]                       # τ>.5: .75/.9/.95
    d_dn = spread[..., N_SPREADS:]                       # τ<.5: .25/.1/.05
    c_up = c_dn = carry_spread
    if isinstance(carry_spread, torch.Tensor) and carry_spread.ndim and carry_spread.shape[-1] != 1:
        assert carry_spread.shape[-1] == 2 * N_SPREADS, (
            f"carry_spread's last axis must be 1 or {2 * N_SPREADS} (per level), "
            f"got {tuple(carry_spread.shape)}"
        )
        c_up, c_dn = carry_spread[..., :N_SPREADS], carry_spread[..., N_SPREADS:]
    o_up, o_dn = torch.cumsum(d_up, dim=-1), torch.cumsum(d_dn, dim=-1)
    # Skipped outright when there is no carry, which keeps every training and
    # single-window caller bit-identical.
    if not _carry_is_zero(carry_spread):
        c_up = torch.as_tensor(c_up, dtype=o_up.dtype, device=o_up.device)
        c_dn = torch.as_tensor(c_dn, dtype=o_dn.dtype, device=o_dn.device)
        o_up, o_dn = torch.hypot(c_up, o_up), torch.hypot(c_dn, o_dn)
    up = m.unsqueeze(-1) + o_up                          # (B,M,S,N_SPREADS) ascending
    dn = m.unsqueeze(-1) - o_dn                          # (B,M,S,N_SPREADS) descending

    # dn is [.25 .1 .05], descending in value → flip to ascending [.05 .1 .25].
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
    """Anchor BG in mg/dL read out of the normalized context — the (a)→(b) bridge.

    feat 0 is ``z(f(bg))``, the sole input path, so the inverse is a plain z-unscale
    followed unconditionally by ``kovatchev_f_inv_np``; ``model.forward`` re-applies ``f``
    internally. The pipeline runs on RAW post-noise signals, so this matches the training
    anchor (``data._build_sample`` reads it off the raw mg/dL array) as the same physical
    value, to within a sub-ulp round-trip difference.

    All ``M`` cells cross in ONE host transfer and one float64 NumPy inverse. Only VISIBLE
    cells may be indexed: feat 0 of a MASKED patch is a legal-looking ``z`` that decodes to
    an ordinary mg/dL, so a wrong index yields a plausible anchor rather than an error.

    Args:
        context: ``(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)`` normalized context.
        stats: normalization statistics; must carry ``bg_absolute``.
        patch_idx: ``(M,)`` patch indices, negatives counting from the right.
            Default: the last patch.
        step_idx: ``(M,)`` within-patch step indices, same length as ``patch_idx``.
            Default: the last step.

    Returns:
        anchor: ``(M,)`` mg/dL clamped to the physical range — ``(1,)`` in the default
        single-cell form.
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
        # Left-padding sits at the FAR left, so the rightmost cell is always real data.
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
    # float64 arithmetic, as when this read a single Python float.
    p = p.to(context.device)
    s = s.to(context.device)
    z = context[p, s, 0].detach().float().cpu().numpy().astype(np.float64)
    risk = z * (bg_std + 1e-8) + bg_mean
    mgdl = np.clip(kovatchev_f_inv_np(risk), BG_CLAMP_MIN, BG_CLAMP_MAX)
    return torch.tensor(mgdl, dtype=torch.float32, device=context.device)


class ModelEMA:
    """Exponential moving average of a model's float parameters and buffers.

    ``apply_to(model)`` swaps the shadow in for the duration of a ``with`` block, then
    restores the live weights so training continues on the un-smoothed parameters.

    Validation runs under the shadow because threshold-crossing metrics (hypo recall, TIR
    error) are very sensitive to small μ shifts, and Muon's per-step weight jitter is large
    next to the clinical cutoffs.

    decay: in [0, 1). 0.999 ≈ a 1k-step window, 0.9999 ≈ 10k.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        # shadow — the smoothed copy read at validation.
        # _param_refs — live-model references, so update() need not re-walk named_*().
        self.shadow: dict[str, torch.Tensor] = {}
        self._param_refs: list[tuple[str, torch.Tensor]] = []
        # Non-persistent buffers (RoPE caches, masks) are not learned state.
        non_persistent = getattr(model, '_non_persistent_buffers_set', set())
        for name, param in model.named_parameters():
            # Non-float tensors pass through apply_to unchanged.
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
        # One NaN/inf folded into the shadow would persist forever (decay*NaN == NaN),
        # so a non-finite tensor is skipped per tensor and its shadow keeps its last
        # good value. Selecting the finite subset BEFORE the foreach preserves that.
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
            # Fused equivalent of shadow.mul_(decay).add_(live, alpha=1-decay).
            torch._foreach_mul_(sel_shadow, self.decay)
            torch._foreach_add_(sel_shadow, sel_live, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Defensive clone: checkpoints are saved from this dict.
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        # Defensive clone: the caller may keep the dict and mutate it afterwards.
        self.shadow = {k: v.detach().clone() for k, v in sd.items()}

    def to(self, device: torch.device) -> "ModelEMA":
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """Swap the shadow into ``model`` for the ``with`` block; restore live weights on exit.

        The backup is taken from ``model.state_dict()``, not ``_param_refs``, so the
        restore is exact even if other code mutated the model during the block.
        """
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # Start from the backup so non-EMA-tracked buffers (int64 counters) keep their
        # live values while the float weights are replaced.
        merged = dict(backup)
        for k, v in self.shadow.items():
            merged[k] = v
        model.load_state_dict(merged, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)
