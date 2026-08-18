"""
Proper scoring rules for the masked-BG quantile fan, binned on ``d``.

Five rules live here, none of which is measured anywhere else in this tree:

  1. ``crps_by_d``                — continuous ranked probability score
  2. ``winkler_by_d``            — interval / Winkler score per nominal central level
  3. ``coverage_sharpness_by_d`` — coverage AND the width that bought it, never apart
  4. ``joint_coverage_by_d``     — simultaneous horizon coverage, distinct from marginal
  5. ``alarm_operating_curve``   — hypo detection rate vs false alarms/day + median lead

SPACE — every array crossing this module's boundary is (b) mg/dL PHYSICAL.
Not (c) Kovatchev risk space, not (a) normalized z-space.  The model emits its
fan in risk space; the caller crosses once with ``utils.kovatchev_f_inv`` and
passes the decoded result here.  Nothing in this module transforms a space, and
every public entry point runs the same units tripwire (``_assert_mgdl``): a
risk-space fan lies in ``[f(10), f(400)] = [-6.82, +3.16]`` and a z-space one is
smaller still, so both trip the ``>= BG_CLAMP_MIN`` floor loudly rather than
producing a plausible score.  See ``CLAUDE.md`` "Three spaces and the only two
bridges" — the single copy of that contract.

THE ``d`` AXIS — ``d`` is the distance in patches to the nearest visible
evidence on EITHER side (``data.py`` ``_mask_slots``).  Every rule here bins on
it, and on nothing else: not on span length, which confounds one-sided and
two-sided cases at equal difficulty, and not on arm.  The fixed forecast
protocol's @30/@60/@90/@120 min columns ARE ``d = 1..4`` one-sided.

POOLED FIGURES ARE NOT A SELECTION METRIC.  Every rule also reports a figure
pooled over ``d``, and every one of those carries ``POOLED_NOT_COMPARABLE``.
The training sampler concentrates its supervision at small ``d`` and on the
two-sided case (``metrics.protocols.SAMPLER_REFERENCE`` carries the enumerated
shares), so a pooled masked-BG scalar is an average over a mask distribution
rather than over a difficulty: it improves when the mixture softens and moves
between protocols that share no mixture.  Compare pooled
against pooled only within one fixed protocol, and select on nothing pooled.

SCORING UNIT — a masked patch.  Arrays arrive as

    q         (N, S, K)  quantile fan, mg/dL, ascending in K
    true      (N, S)     realized BG, mg/dL
    d         (N,)       int >= 1, patches to nearest visible evidence
    group     (N,)       int, the window (forecast origin) each unit belongs to
    lead_min  (N, S)     minutes from that window's alarm-decision time to the step

with ``S = PATCH_SIZE`` steps per patch and ``K = len(QUANTILE_LEVELS)``.  A
forward emits ``q_tau (B, M, S, K)`` over ``M = MAX_MASKED_PATCHES`` head slots
of which only ``valid`` ones carry supervision; the caller flattens the valid
slots into ``N``.  Padded slots carry ``d = 0``, so the ``d >= 1`` assert here
is also the tripwire for an unfiltered ``(B, M)`` flatten.

NO GATES.  This module reports numbers and defines no pass/fail threshold on any
of them: no reference pretrain exists to read one off, and a guessed gate is
worse than an absent one.  Selection and admission thresholds belong wherever
they are eventually measured, not here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
from config import BG_HYPO_THRESHOLD, HYPO_ALARM_QUANTILE_TAU, PATCH_SIZE, QUANTILE_LEVELS
from metrics.core.suite import _FAN_ORDER_TOL_MGDL as FAN_ORDER_TOL_MGDL
from metrics.core.schema import GRID_MIN

__all__ = [
    'POOLED_NOT_COMPARABLE', 'CRPS_PWL', 'CRPS_TRAPEZOID',
    'ByD', 'CoverageSharpness', 'CoverageByD', 'JointCoverage',
    'AlarmPoint', 'AlarmCurve', 'AlarmCurves',
    'central_levels', 'predictive_cdf', 'forecast_lead_minutes',
    'crps_steps', 'crps_by_d', 'winkler_steps', 'winkler_by_d',
    'coverage_sharpness_by_d', 'joint_coverage_by_d', 'alarm_operating_curve',
    'FanScores', 'score_fan',
]

# Marker carried by every pooled-over-d figure this module emits.
POOLED_NOT_COMPARABLE = (
    "pooled over d: an average over the protocol's mask distribution, not over a "
    "difficulty — comparable only within one fixed protocol, never a selection metric"
)

# CRPS quadrature rules; see `crps_steps` for the error of each.
CRPS_PWL = 'pwl'
CRPS_TRAPEZOID = 'trapezoid'

_HYPO_EDGE_IDX = QUANTILE_LEVELS.index(HYPO_ALARM_QUANTILE_TAU)


# --------------------------------------------------------------------------- #
# Input contract
# --------------------------------------------------------------------------- #
def _assert_mgdl(a: np.ndarray, name: str) -> None:
    """Units tripwire: ``a`` must be (b) mg/dL physical, not risk and not z-space.

    Risk space spans ``[f(10), f(400)] = [-6.82, +3.16]`` and z-space is narrower
    still, so both fail the ``BG_CLAMP_MIN`` floor by a wide margin.  Non-finite
    entries are ignored here and dropped by each rule's own masking.
    """
    v = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(v)
    if not finite.any():
        raise AssertionError(f"{name}: no finite values")
    lo, hi = float(v[finite].min()), float(v[finite].max())
    assert lo >= BG_CLAMP_MIN - 1e-3, (
        f"{name} min {lo:.4f} < BG_CLAMP_MIN ({BG_CLAMP_MIN}) mg/dL — this module "
        f"scores in mg/dL physical space only; a risk-space fan ([-6.82, +3.16]) or "
        f"a z-space one reaches here through a missing kovatchev_f_inv")
    assert hi <= BG_CLAMP_MAX + 1e-3, (
        f"{name} max {hi:.4f} > BG_CLAMP_MAX ({BG_CLAMP_MAX}) mg/dL")


def _check_fan(q: np.ndarray, true: np.ndarray, d: np.ndarray,
               levels: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize ``(q, true, d, levels)``; return them as arrays.

    All of ``q`` / ``true`` are mg/dL physical.  ``d`` must be >= 1: a padded head
    slot carries ``d = 0``, so a zero here means an unfiltered ``(B, M)`` flatten
    reached the scorer and the padded slots would be scored against patch 0.
    """
    q = np.asarray(q, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    d = np.asarray(d)
    lv = np.asarray(levels, dtype=np.float64)
    assert q.ndim == 3, f"q must be (N, S, K), got {q.shape}"
    N, S, K = q.shape
    assert true.shape == (N, S), f"true must be {(N, S)}, got {true.shape}"
    assert d.shape == (N,), f"d must be {(N,)}, got {d.shape}"
    assert np.issubdtype(d.dtype, np.integer), f"d must be integer, got {d.dtype}"
    assert (d >= 1).all(), (
        "d must be >= 1 (distance in patches to the nearest visible evidence); "
        "d == 0 is a PADDED head slot — filter on `valid` before scoring")
    assert K == lv.size, f"fan has K={K} levels, `levels` has {lv.size}"
    assert lv.ndim == 1 and (np.diff(lv) > 0).all(), "levels must be ascending"
    assert 0.0 < lv[0] and lv[-1] < 1.0, "levels must lie strictly inside (0, 1)"
    assert (np.diff(q, axis=-1) >= -FAN_ORDER_TOL_MGDL).all(), (
        "quantile fan must be ascending in K (mg/dL)")
    _assert_mgdl(q, 'q')
    _assert_mgdl(true, 'true')
    return q, true, d, lv


def _check_group(group, n: int) -> np.ndarray:
    group = np.asarray(group)
    assert group.shape == (n,), f"group must be {(n,)}, got {group.shape}"
    assert np.issubdtype(group.dtype, np.integer), f"group must be integer, got {group.dtype}"
    return group


def central_levels(levels: Sequence[float] = QUANTILE_LEVELS) -> tuple[tuple[float, int, int], ...]:
    """The nominal central levels the fan's τ pairs imply, widest first.

    A pair ``(τ, 1 − τ)`` present in ``levels`` brackets a central interval of
    nominal level ``1 − 2τ``.  With ``QUANTILE_LEVELS`` this is 0.90 (0.05/0.95),
    0.80 (0.1/0.9) and 0.50 (0.25/0.75).  Returns ``(nominal, lo_idx, hi_idx)``
    triples; a level with no partner is skipped rather than approximated.
    """
    lv = [float(x) for x in levels]
    out: list[tuple[float, int, int]] = []
    for i, tau in enumerate(lv):
        if tau >= 0.5:
            continue
        for j, other in enumerate(lv):
            if abs(other - (1.0 - tau)) <= 1e-12:
                out.append((round(1.0 - 2.0 * tau, 12), i, j))
                break
    return tuple(out)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ByD:
    """One scalar reported per ``d``, plus the pooled figure and its warning."""
    name: str
    unit: str
    by_d: Mapping[int, float]
    n_by_d: Mapping[int, int]
    pooled: float
    n_pooled: int
    pooled_note: str = POOLED_NOT_COMPARABLE

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for dd in sorted(self.by_d):
            out[f'{self.name}@d{dd}'] = self.by_d[dd]
            out[f'{self.name}_n@d{dd}'] = float(self.n_by_d[dd])
        out[f'{self.name}_pooled_NOT_COMPARABLE'] = self.pooled
        out[f'{self.name}_n_pooled'] = float(self.n_pooled)
        return out


@dataclass(frozen=True)
class CoverageSharpness:
    """A coverage figure and the interval width that bought it — always together.

    Both fields are required at construction, so no caller can build a coverage
    number that has lost its sharpness.  ``render`` emits the pair on one line;
    a coverage reported without its width is not interpretable, since coverage
    alone is bought by widening the band.

    Widths are mg/dL.  ``kind`` names WHICH coverage this is — marginal per step,
    or one of the simultaneous variants — so the two can never be confused in a
    table.  ``n`` counts that kind's own scoring units: (unit, step) pairs for a
    marginal figure, groups for a joint one.
    """
    kind: str
    nominal: float
    coverage: float
    mean_width: float
    median_width: float
    n: int

    def render(self) -> str:
        return (f"{self.kind} cov {self.coverage:.4f} (nominal {self.nominal:.2f}) "
                f"@ width {self.mean_width:.2f} mg/dL (median {self.median_width:.2f}), "
                f"n={self.n}")

    def to_dict(self, prefix: str) -> dict[str, float]:
        return {f'{prefix}_cov': self.coverage,
                f'{prefix}_width_mean': self.mean_width,
                f'{prefix}_width_median': self.median_width,
                f'{prefix}_n': float(self.n)}


@dataclass(frozen=True)
class CoverageByD:
    """Marginal (per-step) coverage + sharpness, per ``d`` and pooled."""
    nominal: float
    by_d: Mapping[int, CoverageSharpness]
    pooled: CoverageSharpness
    pooled_note: str = POOLED_NOT_COMPARABLE

    def to_dict(self) -> dict[str, float]:
        pct = int(round(self.nominal * 100))
        out: dict[str, float] = {}
        for dd in sorted(self.by_d):
            out.update(self.by_d[dd].to_dict(f'marginal{pct}@d{dd}'))
        out.update(self.pooled.to_dict(f'marginal{pct}_pooled_NOT_COMPARABLE'))
        return out

    def render(self) -> list[str]:
        return ([self.by_d[dd].render() for dd in sorted(self.by_d)]
                + [self.pooled.render() + f"  [{self.pooled_note}]"])


@dataclass(frozen=True)
class JointCoverage:
    """Simultaneous coverage, kept apart from the marginal figure by construction.

    ``marginal_*`` is per (unit, step): the fraction of individual steps inside
    the band.  The three ``joint_*`` fields are per GROUP (one forecast origin)
    and require EVERY step in their scope to be inside at once:

      * ``joint_within_d``  — every step of the group's patches at that ``d``
      * ``joint_path_to_d`` — every step at ``d' <= d``: the simultaneous
        coverage of the whole forecast path out to that distance
      * ``joint_pooled``    — every scored step of the group

    Joint coverage is at most the smallest marginal in its scope and falls fast
    with scope; reporting one as the other overstates calibration.  Every field
    carries its own width, so no coverage travels without its sharpness.
    """
    nominal: float
    marginal_by_d: Mapping[int, CoverageSharpness]
    marginal_pooled: CoverageSharpness
    joint_within_d: Mapping[int, CoverageSharpness]
    joint_path_to_d: Mapping[int, CoverageSharpness]
    joint_pooled: CoverageSharpness
    pooled_note: str = POOLED_NOT_COMPARABLE

    def to_dict(self) -> dict[str, float]:
        pct = int(round(self.nominal * 100))
        out: dict[str, float] = {}
        for dd in sorted(self.marginal_by_d):
            out.update(self.marginal_by_d[dd].to_dict(f'marginal{pct}@d{dd}'))
        out.update(self.marginal_pooled.to_dict(f'marginal{pct}_pooled_NOT_COMPARABLE'))
        for dd in sorted(self.joint_within_d):
            out.update(self.joint_within_d[dd].to_dict(f'joint{pct}_within@d{dd}'))
        for dd in sorted(self.joint_path_to_d):
            out.update(self.joint_path_to_d[dd].to_dict(f'joint{pct}_path_to_d{dd}'))
        out.update(self.joint_pooled.to_dict(f'joint{pct}_pooled_NOT_COMPARABLE'))
        return out

    def render(self) -> list[str]:
        rows = [self.marginal_by_d[dd].render() for dd in sorted(self.marginal_by_d)]
        rows.append(self.marginal_pooled.render() + f"  [{self.pooled_note}]")
        rows += [self.joint_within_d[dd].render() for dd in sorted(self.joint_within_d)]
        rows += [self.joint_path_to_d[dd].render() for dd in sorted(self.joint_path_to_d)]
        rows.append(self.joint_pooled.render() + f"  [{self.pooled_note}]")
        return rows


@dataclass(frozen=True)
class AlarmPoint:
    """One operating point of the hypo alarm.

    ``median_lead_min`` is the median, over the detected events, of the minutes
    between the alarm's decision time and the first true sub-threshold step it
    caught.  It is what makes the rest of the point actionable: a detection rate
    bought at a two-minute lead is not a usable alarm, and the rate alone cannot
    show that.
    """
    rule: str
    score_threshold: float | None
    detection_rate: float
    false_alarms_per_day: float | None
    median_lead_min: float
    lead_min_p25: float
    lead_min_p75: float
    n_groups: int
    n_events: int
    n_detected: int
    n_alarms: int
    n_false_alarms: int


@dataclass(frozen=True)
class AlarmCurve:
    """The swept curve for one ``d`` (or pooled), plus the deployed rule's point."""
    d: int | None
    points: tuple[AlarmPoint, ...]
    deployed: AlarmPoint
    threshold_mgdl: float
    observed_days: float | None
    pooled_note: str | None = None


@dataclass(frozen=True)
class AlarmCurves:
    by_d: Mapping[int, AlarmCurve]
    pooled: AlarmCurve
    pooled_note: str = POOLED_NOT_COMPARABLE


# --------------------------------------------------------------------------- #
# 1. CRPS
# --------------------------------------------------------------------------- #
def crps_steps(q: np.ndarray, true: np.ndarray,
               levels: Sequence[float] = QUANTILE_LEVELS,
               rule: str = CRPS_PWL) -> np.ndarray:
    """Per-(unit, step) CRPS in mg/dL from the discrete quantile fan.

    SPACE: ``q`` ``(N, S, K)`` and ``true`` ``(N, S)`` are (b) mg/dL physical,
    and the returned score is in mg/dL (CRPS carries the unit of the variable).

    The quantile decomposition is used throughout:

        CRPS(F, y) = 2 ∫₀¹ ρ_τ(y − F⁻¹(τ)) dτ,   ρ_τ(u) = u·(τ − 1{u < 0})

    i.e. twice the integral of the pinball loss over τ.  The fan supplies F⁻¹ at
    ``K = 7`` nodes only, so the integral needs a quadrature rule and the choice
    is not neutral at that spacing.  Both rules here share the SAME closed-form
    tails, so they differ only in the interior and the difference is attributable.

    TAILS (τ < τ₁ and τ > τ_K), identical under both rules.  F⁻¹ is extended
    flat, which is the quantile function of a law holding mass τ₁ as an atom at
    q₁ and 1 − τ_K as an atom at q_K.  The contribution is then exact for that
    law: ``τ₁²·(y − q₁)`` when ``y >= q₁`` and ``(2τ₁ − τ₁²)·(q₁ − y)`` when it
    is below, and symmetrically at the top.  The weight on an escape below the
    fan is therefore ``2τ₁ − τ₁² = 0.0975`` mg/dL per mg/dL at ``τ₁ = 0.05``.
    Against a true law with a genuine tail the sign of this error is not fixed:
    concentrating the tail mass at the edge overstates the score for x between y
    and the edge and understates it beyond, so it is a discretization error of
    the fan and not a bias with a known direction.

    INTERIOR, ``rule = CRPS_PWL`` (default).  F⁻¹ is the linear interpolation of
    the fan between adjacent nodes, and the integral is evaluated in closed form
    on every subinterval, splitting analytically at the crossing τ* where
    F⁻¹(τ*) = y.  Quadrature error is ZERO for that piecewise-linear-quantile
    law; the residual against the model's own predictive law is the fan's
    discretization, which no rule can recover from 7 nodes.

    INTERIOR, ``rule = CRPS_TRAPEZOID``.  Trapezoidal quadrature on the τ nodes.
    ρ_τ is V-shaped around τ*, so a chord across the subinterval containing the
    crossing lies above the true integrand and the rule OVERSTATES.  On one
    subinterval of width h whose quantile gap is D, with the crossing at
    fraction u of the gap, the excess is exactly

        2·[ D·h·u(1−u)(1−h)/2 − D·h²·(u³ + (1−u)³)/6 ]

    maximised at u = 1/2.  ``QUANTILE_LEVELS`` has τ gaps
    (0.05, 0.15, 0.25, 0.25, 0.15, 0.05), so the worst case is h = 0.25 and the
    excess is at most exactly ``D/24 ≈ 0.0417·D`` mg/dL, D being that gap's
    spread — the fan's ``q(0.5)..q(0.75)`` (or ``q(0.25)..q(0.5)``) width.  A
    20 mg/dL half-IQR therefore inflates CRPS by up to 0.83 mg/dL.  The two rules
    agree exactly on a degenerate fan, where every gap has D = 0.

    Args:
        q:      ``(N, S, K)`` ascending fan, mg/dL.
        true:   ``(N, S)`` realized BG, mg/dL.
        levels: the ``K`` τ of the fan, ascending, strictly inside (0, 1).
        rule:   ``CRPS_PWL`` or ``CRPS_TRAPEZOID``.

    Returns:
        ``(N, S)`` CRPS in mg/dL.  Non-finite ``true`` propagates as NaN.
    """
    assert rule in (CRPS_PWL, CRPS_TRAPEZOID), f"unknown CRPS rule {rule!r}"
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(true, dtype=np.float64)
    lv = np.asarray(levels, dtype=np.float64)
    assert q.ndim == 3 and y.shape == q.shape[:2] and lv.size == q.shape[2]

    t1, tK = float(lv[0]), float(lv[-1])
    q1, qK = q[..., 0], q[..., -1]

    below_lo = y < q1
    tail_lo = np.where(below_lo,
                       (2.0 * t1 - t1 * t1) * (q1 - y),
                       (t1 * t1) * (y - q1))
    above_hi = y > qK
    tail_hi = np.where(above_hi,
                       (1.0 - tK * tK) * (y - qK),
                       ((1.0 - tK) ** 2) * (qK - y))

    qa, qb = q[..., :-1], q[..., 1:]                     # (N, S, K-1)
    ta, tb = lv[:-1], lv[1:]
    h = (tb - ta)[None, None, :]
    ta_b, tb_b = ta[None, None, :], tb[None, None, :]
    yy = y[..., None]
    D = qb - qa

    if rule == CRPS_PWL:
        # Case A: y >= qb (fan entirely below the truth on this subinterval).
        a = 2.0 * (ta_b * h * (yy - qa) - ta_b * D * h / 2.0
                   + (yy - qa) * h * h / 2.0 - D * h * h / 3.0)
        # Case B: y <= qa (fan entirely above).
        b = 2.0 * ((1.0 - ta_b) * h * (qa - yy) + (1.0 - ta_b) * D * h / 2.0
                   - (qa - yy) * h * h / 2.0 - D * h * h / 3.0)
        # Case C: the crossing τ* lies inside; u is its position in the gap.
        safe_D = np.where(D > 0.0, D, 1.0)
        u = np.clip((yy - qa) / safe_D, 0.0, 1.0)
        c = D * (ta_b * u * u * h + (u ** 3) * h * h / 3.0
                 + (1.0 - tb_b) * (1.0 - u) ** 2 * h + ((1.0 - u) ** 3) * h * h / 3.0)
        inside = (D > 0.0) & (yy > qa) & (yy < qb)
        interior = np.where(inside, c, np.where(yy >= qb, a, b))
    else:
        pin_a = (ta_b - (yy < qa)) * (yy - qa)
        pin_b = (tb_b - (yy < qb)) * (yy - qb)
        interior = h * (pin_a + pin_b)                    # 2 · (h/2)·(f_a + f_b)

    return tail_lo + interior.sum(axis=-1) + tail_hi


def crps_by_d(q: np.ndarray, true: np.ndarray, d: np.ndarray,
              levels: Sequence[float] = QUANTILE_LEVELS,
              rule: str = CRPS_PWL) -> ByD:
    """CRPS averaged per ``d``, and pooled.

    SPACE: (b) mg/dL physical on both inputs and the score.  See ``crps_steps``
    for the quadrature rule and its error.

    Every step of one masked patch shares that patch's ``d``, so the per-``d``
    figure is the mean over all (unit at that ``d``, step) pairs.  The pooled
    figure averages over the protocol's own mask distribution and carries
    ``POOLED_NOT_COMPARABLE``.
    """
    q, true, d, lv = _check_fan(q, true, d, levels)
    s = crps_steps(q, true, lv, rule)
    ok = np.isfinite(s)
    by_d, n_by_d = {}, {}
    for dd in np.unique(d):
        sel = ok & (d[:, None] == dd)
        by_d[int(dd)] = float(s[sel].mean()) if sel.any() else math.nan
        n_by_d[int(dd)] = int(sel.sum())
    return ByD(name=f'crps_{rule}', unit='mg/dL', by_d=by_d, n_by_d=n_by_d,
               pooled=float(s[ok].mean()) if ok.any() else math.nan,
               n_pooled=int(ok.sum()))


# --------------------------------------------------------------------------- #
# 2. Interval / Winkler score
# --------------------------------------------------------------------------- #
def winkler_steps(q: np.ndarray, true: np.ndarray, lo_idx: int, hi_idx: int,
                  alpha: float) -> np.ndarray:
    """Per-(unit, step) Winkler interval score for one central interval, mg/dL.

    SPACE: (b) mg/dL physical throughout.

        W = (u − l) + (2/α)·(l − y)·1{y < l} + (2/α)·(y − u)·1{y > u}

    for a central interval ``[l, u]`` of nominal level ``1 − α``.  The score is
    proper: the width term alone is minimized by a collapsed interval and the
    escape terms alone by an infinite one, so it prices coverage and sharpness
    against each other on one axis.  Smaller is better; the unit is mg/dL.
    """
    lo, hi = q[..., lo_idx], q[..., hi_idx]
    w = hi - lo
    w = w + (2.0 / alpha) * np.where(true < lo, lo - true, 0.0)
    w = w + (2.0 / alpha) * np.where(true > hi, true - hi, 0.0)
    return w


def winkler_by_d(q: np.ndarray, true: np.ndarray, d: np.ndarray,
                 levels: Sequence[float] = QUANTILE_LEVELS) -> dict[float, ByD]:
    """Winkler score per ``d`` and pooled, at every nominal level the fan implies.

    SPACE: (b) mg/dL physical on inputs and score.

    Returns one ``ByD`` per nominal central level from ``central_levels`` — 0.90,
    0.80 and 0.50 under ``QUANTILE_LEVELS`` — keyed by that nominal level.
    """
    q, true, d, lv = _check_fan(q, true, d, levels)
    out: dict[float, ByD] = {}
    for nominal, lo_idx, hi_idx in central_levels(lv):
        alpha = 1.0 - nominal
        w = winkler_steps(q, true, lo_idx, hi_idx, alpha)
        ok = np.isfinite(w)
        by_d, n_by_d = {}, {}
        for dd in np.unique(d):
            sel = ok & (d[:, None] == dd)
            by_d[int(dd)] = float(w[sel].mean()) if sel.any() else math.nan
            n_by_d[int(dd)] = int(sel.sum())
        out[nominal] = ByD(name=f'winkler{int(round(nominal * 100))}', unit='mg/dL',
                           by_d=by_d, n_by_d=n_by_d,
                           pooled=float(w[ok].mean()) if ok.any() else math.nan,
                           n_pooled=int(ok.sum()))
    return out


# --------------------------------------------------------------------------- #
# 3. Coverage AND sharpness
# --------------------------------------------------------------------------- #
def _cov_sharp(kind: str, nominal: float, inside: np.ndarray, width: np.ndarray,
               sel: np.ndarray) -> CoverageSharpness:
    """Build a coverage/width pair over the selected mask; empty ⇒ NaN with n = 0."""
    if not sel.any():
        return CoverageSharpness(kind=kind, nominal=nominal, coverage=math.nan,
                                 mean_width=math.nan, median_width=math.nan, n=0)
    return CoverageSharpness(kind=kind, nominal=nominal,
                             coverage=float(inside[sel].mean()),
                             mean_width=float(width[sel].mean()),
                             median_width=float(np.median(width[sel])),
                             n=int(sel.sum()))


def coverage_sharpness_by_d(q: np.ndarray, true: np.ndarray, d: np.ndarray,
                            levels: Sequence[float] = QUANTILE_LEVELS
                            ) -> dict[float, CoverageByD]:
    """Marginal per-step coverage, reported with the width that bought it.

    SPACE: (b) mg/dL physical; widths are mg/dL.

    Coverage and sharpness are returned in one ``CoverageSharpness`` object with
    both fields required, so a coverage figure cannot be rendered, tabulated or
    logged without the width beside it.  A band widens to any coverage, so the
    number alone says nothing about the forecast.

    This is the MARGINAL figure — the fraction of individual (unit, step) pairs
    inside the band, one step at a time.  ``joint_coverage_by_d`` carries the
    simultaneous variant; the two are labelled by ``CoverageSharpness.kind`` and
    are not interchangeable.
    """
    q, true, d, lv = _check_fan(q, true, d, levels)
    out: dict[float, CoverageByD] = {}
    for nominal, lo_idx, hi_idx in central_levels(lv):
        lo, hi = q[..., lo_idx], q[..., hi_idx]
        inside = (true >= lo) & (true <= hi)
        width = hi - lo
        ok = np.isfinite(true) & np.isfinite(width)
        by_d = {int(dd): _cov_sharp('marginal', nominal, inside, width,
                                    ok & (d[:, None] == dd))
                for dd in np.unique(d)}
        out[nominal] = CoverageByD(nominal=nominal, by_d=by_d,
                                   pooled=_cov_sharp('marginal(pooled)', nominal,
                                                     inside, width, ok))
    return out


# --------------------------------------------------------------------------- #
# 4. Joint (simultaneous) horizon coverage
# --------------------------------------------------------------------------- #
def _joint(kind: str, nominal: float, inside: np.ndarray, width: np.ndarray,
           ok: np.ndarray, gidx: np.ndarray, n_groups: int,
           unit_sel: np.ndarray) -> CoverageSharpness:
    """Fraction of groups with EVERY selected step inside, plus that scope's width.

    A group counts only when it carries at least one selected finite step.  A
    step that is not finite is excluded rather than treated as an escape.
    """
    step_sel = ok & unit_sel[:, None]
    if not step_sel.any():
        return CoverageSharpness(kind=kind, nominal=nominal, coverage=math.nan,
                                 mean_width=math.nan, median_width=math.nan, n=0)
    rows = np.repeat(gidx[:, None], inside.shape[1], axis=1)[step_sel]
    esc = (~inside)[step_sel].astype(np.float64)
    present = np.bincount(rows, minlength=n_groups)
    escaped = np.bincount(rows, weights=esc, minlength=n_groups)
    have = present > 0
    return CoverageSharpness(kind=kind, nominal=nominal,
                             coverage=float((escaped[have] == 0).mean()),
                             mean_width=float(width[step_sel].mean()),
                             median_width=float(np.median(width[step_sel])),
                             n=int(have.sum()))


def joint_coverage_by_d(q: np.ndarray, true: np.ndarray, d: np.ndarray,
                        group: np.ndarray,
                        levels: Sequence[float] = QUANTILE_LEVELS
                        ) -> dict[float, JointCoverage]:
    """Simultaneous horizon coverage, reported beside — and labelled apart from —
    the per-step marginal figure.

    SPACE: (b) mg/dL physical; widths mg/dL.

    ``group`` labels the window (one forecast origin) each masked patch belongs
    to.  Marginal coverage asks whether a single step landed inside its band;
    joint coverage asks whether an entire path did at once, which is the question
    a rolled-out trajectory or an alarm actually poses.  Joint coverage is
    bounded above by the smallest marginal in its scope and decays with the
    number of steps in it, so the two figures cannot be read as the same
    quantity: each ``CoverageSharpness`` here names its own ``kind``.

    Three joint scopes are reported (see ``JointCoverage``): within one ``d``,
    cumulatively out to a ``d`` (the forecast path), and over every scored step
    of the group.
    """
    q, true, d, lv = _check_fan(q, true, d, levels)
    group = _check_group(group, d.size)
    _, gidx = np.unique(group, return_inverse=True)
    n_groups = int(gidx.max()) + 1 if gidx.size else 0
    ds = [int(x) for x in np.unique(d)]

    out: dict[float, JointCoverage] = {}
    for nominal, lo_idx, hi_idx in central_levels(lv):
        lo, hi = q[..., lo_idx], q[..., hi_idx]
        inside = (true >= lo) & (true <= hi)
        width = hi - lo
        ok = np.isfinite(true) & np.isfinite(width)
        all_units = np.ones(d.size, dtype=bool)
        out[nominal] = JointCoverage(
            nominal=nominal,
            marginal_by_d={dd: _cov_sharp('marginal', nominal, inside, width,
                                          ok & (d[:, None] == dd)) for dd in ds},
            marginal_pooled=_cov_sharp('marginal(pooled)', nominal, inside, width, ok),
            joint_within_d={dd: _joint(f'joint-within-d{dd}', nominal, inside, width,
                                       ok, gidx, n_groups, d == dd) for dd in ds},
            joint_path_to_d={dd: _joint(f'joint-path-to-d{dd}', nominal, inside, width,
                                        ok, gidx, n_groups, d <= dd) for dd in ds},
            joint_pooled=_joint('joint(pooled)', nominal, inside, width, ok, gidx,
                                n_groups, all_units))
    return out


# --------------------------------------------------------------------------- #
# 5. Alarm operating curve
# --------------------------------------------------------------------------- #
def predictive_cdf(q: np.ndarray, x: float,
                   levels: Sequence[float] = QUANTILE_LEVELS) -> np.ndarray:
    """``P(Y <= x)`` under the same piecewise-linear-quantile law ``CRPS_PWL`` scores.

    SPACE: ``q`` and ``x`` are (b) mg/dL physical; the result is a probability.

    F⁻¹ is linear between fan nodes and flat outside them, so the mass below the
    lowest node sits as an atom at that node: the probability is exactly 0 below
    ``q₁`` and exactly 1 at or above ``q_K``.  Using the whole fan rather than a
    single band edge is what gives the alarm a continuous score to sweep — the
    fan carries only three τ below the median, far too coarse an axis on its own.
    """
    q = np.asarray(q, dtype=np.float64)
    lv = np.asarray(levels, dtype=np.float64)
    assert q.shape[-1] == lv.size
    p = np.where(x >= q[..., -1], 1.0, 0.0)
    for k in range(lv.size - 1):
        qa, qb = q[..., k], q[..., k + 1]
        gap = qb - qa
        frac = np.where(gap > 0.0, (x - qa) / np.where(gap > 0.0, gap, 1.0), 1.0)
        val = lv[k] + (lv[k + 1] - lv[k]) * np.clip(frac, 0.0, 1.0)
        p = np.where((x >= qa) & (x < qb), val, p)
    return p


def forecast_lead_minutes(d: np.ndarray, n_steps: int = PATCH_SIZE) -> np.ndarray:
    """``(N, S)`` minutes from the forecast origin to each step, for the RIGHT-EDGE
    forecast protocol only.

    In that protocol the masked span is the trailing ``PREDICTION_PATCHES``
    patches, so a patch at distance ``d`` is the ``d``-th patch after the origin
    and step ``s`` of it lands at ``((d − 1)·n_steps + s + 1)·GRID_MIN`` minutes
    — the same 0-based patch-end convention as ``metrics.core.horizons``.  The
    identity ``d = 1..4 ⇔ @30/@60/@90/@120`` min holds only here: for an infill
    span ``d`` is a distance to evidence on either side and says nothing about
    the time to the origin, so that protocol must pass its own ``lead_min``.
    """
    d = np.asarray(d)
    assert np.issubdtype(d.dtype, np.integer) and (d >= 1).all()
    s = np.arange(n_steps)[None, :]
    return (((d[:, None] - 1) * n_steps) + s + 1) * float(GRID_MIN)


def _alarm_point(rule: str, score_threshold: float | None, fires: np.ndarray,
                 has_event: np.ndarray, onset_lead: np.ndarray,
                 observed_days: float | None) -> AlarmPoint:
    detected = fires & has_event
    false = fires & ~has_event
    n_events = int(has_event.sum())
    n_false = int(false.sum())
    leads = onset_lead[detected]
    leads = leads[np.isfinite(leads)]
    return AlarmPoint(
        rule=rule,
        score_threshold=score_threshold,
        detection_rate=float(detected.sum()) / n_events if n_events else math.nan,
        false_alarms_per_day=(n_false / observed_days) if observed_days else None,
        median_lead_min=float(np.median(leads)) if leads.size else math.nan,
        lead_min_p25=float(np.percentile(leads, 25)) if leads.size else math.nan,
        lead_min_p75=float(np.percentile(leads, 75)) if leads.size else math.nan,
        n_groups=int(fires.size), n_events=n_events, n_detected=int(detected.sum()),
        n_alarms=int(fires.sum()), n_false_alarms=n_false)


def _alarm_curve_for(d_label: int | None, sel_units: np.ndarray, score_unit: np.ndarray,
                     edge_unit: np.ndarray, true: np.ndarray, lead_min: np.ndarray,
                     gidx: np.ndarray, n_groups: int, threshold: float,
                     observed_days: float | None, max_points: int) -> AlarmCurve:
    """One operating curve over the groups touched by ``sel_units``.

    Per group: the alarm score is the maximum over the scanned steps, the
    deployed band edge the minimum, and the event onset the earliest scanned step
    with ``true < threshold``.  Groups no selected unit touches are dropped.
    """
    big = np.finfo(np.float64).max
    rows = np.nonzero(sel_units)[0]
    gr = gidx[rows]
    score_g = np.full(n_groups, -np.inf)
    onset_g = np.full(n_groups, np.inf)
    edge_g = np.full(n_groups, big)
    if rows.size:
        s_fin = np.isfinite(score_unit[rows])
        np.maximum.at(score_g, gr,
                      np.max(np.where(s_fin, score_unit[rows], -np.inf), axis=1))
        e_fin = np.isfinite(edge_unit[rows])
        np.minimum.at(edge_g, gr, np.min(np.where(e_fin, edge_unit[rows], big), axis=1))
        hypo = np.isfinite(true[rows]) & (true[rows] < threshold)
        np.minimum.at(onset_g, gr, np.min(np.where(hypo, lead_min[rows], np.inf), axis=1))
    present = np.bincount(gr, minlength=n_groups) > 0

    score_g, onset_g, edge_g = score_g[present], onset_g[present], edge_g[present]
    has_event = np.isfinite(onset_g)

    # The swept cuts are the realized scores themselves (an exact ROC sweep), plus
    # the always-fire endpoint at 0.
    cuts = np.unique(score_g[score_g > 0.0])
    if cuts.size > max_points - 1:
        cuts = np.unique(np.quantile(cuts, np.linspace(0.0, 1.0, max_points - 1)))
    cuts = np.concatenate([[0.0], cuts])
    points = [_alarm_point(f'P(BG<={threshold:g}) >= {c:.6f}', float(c), score_g >= c,
                           has_event, onset_g, observed_days)
              for c in cuts[::-1]]
    deployed = _alarm_point(
        f'band edge tau={HYPO_ALARM_QUANTILE_TAU:g} < {threshold:g} mg/dL', None,
        edge_g < threshold, has_event, onset_g, observed_days)
    return AlarmCurve(d=d_label, points=tuple(points), deployed=deployed,
                      threshold_mgdl=threshold, observed_days=observed_days,
                      pooled_note=None if d_label is not None else POOLED_NOT_COMPARABLE)


def alarm_operating_curve(q: np.ndarray, true: np.ndarray, d: np.ndarray,
                          group: np.ndarray, lead_min: np.ndarray,
                          observed_days: float | None,
                          levels: Sequence[float] = QUANTILE_LEVELS,
                          threshold: float = BG_HYPO_THRESHOLD,
                          max_points: int = 101) -> AlarmCurves:
    """Hypo detection rate against false alarms per day, with the median lead time.

    SPACE: (b) mg/dL physical — ``q``, ``true`` and ``threshold`` are mg/dL; the
    swept score is a probability and the lead time is minutes.

    One alarm decision is made per ``group`` (one forecast origin).  Within a
    group, the scanned steps are those of the selected masked patches — every
    patch for the pooled curve, the patches at one ``d`` for a per-``d`` curve.
    Then:

      * SCORE — ``max`` over the scanned steps of ``P(BG <= threshold)`` from
        ``predictive_cdf``.  Sweeping a cut on this score generates the curve.
      * EVENT — the group carries a true hypo if any scanned step has
        ``true < threshold`` (the strict convention ``metrics.core.suite`` and the
        validation table already use).  ONSET is the earliest such step.
      * DETECTION RATE — detected events / events.  Groups with no event never
        enter this denominator.
      * FALSE ALARMS PER DAY — firing groups with no event, over
        ``observed_days``.  This is an ISSUANCE rate: one alarm per group, so
        overlapping windows raise it and the caller's window cadence sets it.
        ``observed_days=None`` leaves the rate ``None`` and reports the count —
        the denominator is a property of the evaluation set, never a default.
      * MEDIAN LEAD TIME — median over detected events of the onset's
        ``lead_min``, i.e. the minutes between the alarm and the first true
        sub-threshold reading it caught.  Measured per issuance: a group that
        fires 30 min before another for the same physiological event is a
        separate detection with its own lead, so this is the lead a single
        forecast delivers rather than the earliest warning a stream could give.

    The deployed rule is scored alongside the swept curve at every ``d``: the
    band-edge detector the app ships (``HYPO_ALARM_QUANTILE_TAU``'s lower edge
    dipping below ``BG_HYPO_THRESHOLD``), so the curve carries the operating
    point actually in use rather than only the envelope around it.

    Args:
        lead_min: ``(N, S)`` minutes from the group's decision time to each step.
            ``forecast_lead_minutes`` builds it for the right-edge forecast
            protocol; an infill protocol must supply its own.
    """
    q, true, d, lv = _check_fan(q, true, d, levels)
    group = _check_group(group, d.size)
    lead_min = np.asarray(lead_min, dtype=np.float64)
    assert lead_min.shape == true.shape, f"lead_min must be {true.shape}, got {lead_min.shape}"
    assert observed_days is None or observed_days > 0.0, "observed_days must be positive"
    _, gidx = np.unique(group, return_inverse=True)
    n_groups = int(gidx.max()) + 1 if gidx.size else 0

    score_unit = predictive_cdf(q, threshold, lv)          # (N, S)
    edge_unit = q[..., _HYPO_EDGE_IDX]                     # (N, S)
    by_d = {int(dd): _alarm_curve_for(int(dd), d == dd, score_unit, edge_unit, true,
                                      lead_min, gidx, n_groups, threshold,
                                      observed_days, max_points)
            for dd in np.unique(d)}
    pooled = _alarm_curve_for(None, np.ones(d.size, dtype=bool), score_unit, edge_unit,
                              true, lead_min, gidx, n_groups, threshold, observed_days,
                              max_points)
    return AlarmCurves(by_d=by_d, pooled=pooled)


# --------------------------------------------------------------------------- #
# Convenience aggregate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FanScores:
    """The five rules over one protocol's scored masked patches."""
    crps: ByD
    winkler: Mapping[float, ByD]
    coverage: Mapping[float, CoverageByD]
    joint: Mapping[float, JointCoverage]
    alarm: AlarmCurves | None
    pooled_note: str = POOLED_NOT_COMPARABLE


def score_fan(q: np.ndarray, true: np.ndarray, d: np.ndarray, group: np.ndarray,
              lead_min: np.ndarray | None = None,
              observed_days: float | None = None,
              levels: Sequence[float] = QUANTILE_LEVELS,
              crps_rule: str = CRPS_PWL) -> FanScores:
    """Run all five rules over one protocol's scored masked patches.

    SPACE: (b) mg/dL physical throughout — the caller has already crossed out of
    risk space with ``kovatchev_f_inv``.

    ``lead_min=None`` skips the alarm curve (no lead time can be defined without
    it, and a detection rate without a lead time is not an operating point).
    Nothing here is a gate: no threshold is applied to any figure it returns.
    """
    return FanScores(
        crps=crps_by_d(q, true, d, levels, crps_rule),
        winkler=winkler_by_d(q, true, d, levels),
        coverage=coverage_sharpness_by_d(q, true, d, levels),
        joint=joint_coverage_by_d(q, true, d, group, levels),
        alarm=(None if lead_min is None else
               alarm_operating_curve(q, true, d, group, lead_min, observed_days, levels)))
