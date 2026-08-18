"""Proper scoring rules for the masked-BG fan (``metrics.scoring``).

Deterministic synthetic checks against hand-computable cases — no model, no
simulator, no checkpoint.  Three fans recur:

  * a DEGENERATE fan (every τ on one value), where CRPS collapses to the absolute
    error, the Winkler score to the pure escape penalty and coverage to 0/1;
  * a PERFECTLY CALIBRATED fan (the exact quantiles of the law the truth is drawn
    from), where marginal coverage must land on nominal;
  * a KNOWN-MISCALIBRATED fan (the calibrated one shrunk by a fixed factor),
    which must score worse on every rule while reading SHARPER — the pair of
    numbers is the reason sharpness never travels without coverage.

CRPS additionally gets an independent reference: dense trapezoidal quadrature of
the pinball loss over τ on the same piecewise-linear-quantile law, which the
closed form must reproduce.

Every array here is mg/dL physical space; the space guard is tested directly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from config import (BG_HYPO_THRESHOLD, HYPO_ALARM_QUANTILE_TAU, N_QUANTILES,
                    PATCH_SIZE, QUANTILE_LEVELS)
from metrics.scoring import (CRPS_PWL, CRPS_TRAPEZOID, POOLED_NOT_COMPARABLE,
                             CoverageSharpness, alarm_operating_curve, central_levels,
                             coverage_sharpness_by_d, crps_by_d, crps_steps,
                             forecast_lead_minutes, joint_coverage_by_d,
                             predictive_cdf, score_fan, winkler_by_d, winkler_steps)
from metrics.core.horizons import GRID_MIN, HORIZON_IDX
from utils import kovatchev_f_np

LV = np.asarray(QUANTILE_LEVELS, dtype=np.float64)
MED = QUANTILE_LEVELS.index(0.5)
HYPO_EDGE = QUANTILE_LEVELS.index(HYPO_ALARM_QUANTILE_TAU)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _fan(center: np.ndarray, width: float) -> np.ndarray:
    """``(N, S, K)`` ascending fan: level τ at ``center + 2·width·(τ − 0.5)``.

    ``width`` scales the whole fan, so the central-90% band spans
    ``center ± 0.9·width`` and the central-50% band ``center ± 0.5·width``.
    """
    off = 2.0 * (LV - 0.5)
    return np.asarray(center, dtype=np.float64)[:, :, None] + width * off[None, None, :]


def _degenerate(center: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(center, dtype=np.float64)[:, :, None], N_QUANTILES, axis=2)


def _uniform_fan(lo: float, hi: float, n: int, s: int) -> np.ndarray:
    """The EXACT quantiles of ``U(lo, hi)`` — a perfectly calibrated fan."""
    row = lo + (hi - lo) * LV
    return np.broadcast_to(row, (n, s, N_QUANTILES)).copy()


def _crps_dense(qrow: np.ndarray, y: float, n: int = 400001) -> float:
    """Independent CRPS reference: dense quadrature of ``2·∫ pinball dτ``.

    ``np.interp`` clamps outside the node range, which is exactly the flat-tail
    extension the closed form integrates analytically.
    """
    tau = np.linspace(0.0, 1.0, n)
    qq = np.interp(tau, LV, qrow)
    pin = (tau - (y < qq)) * (y - qq)
    return float(2.0 * np.trapezoid(pin, tau))


# --------------------------------------------------------------------------- #
# The input contract: space, and the padded-slot tripwire
# --------------------------------------------------------------------------- #
def test_risk_space_fan_is_rejected():
    """A fan left in Kovatchev risk space must not score as if it were mg/dL."""
    true_mgdl = np.full((4, 2), 120.0)
    risk = kovatchev_f_np(np.asarray(_fan(true_mgdl, 20.0)))
    print(f"[DUMP] risk-space fan range: [{risk.min():.4f}, {risk.max():.4f}]")
    with pytest.raises(AssertionError, match='mg/dL'):
        crps_by_d(risk, true_mgdl, np.ones(4, dtype=np.int64))


def test_z_space_truth_is_rejected():
    q = _fan(np.full((4, 2), 120.0), 20.0)
    z_true = np.full((4, 2), -0.35)          # a plausible z, an impossible BG
    with pytest.raises(AssertionError, match='mg/dL'):
        crps_by_d(q, z_true, np.ones(4, dtype=np.int64))


def test_padded_head_slot_is_rejected():
    """``d = 0`` is a padded slot; scoring it would score patch 0's BG."""
    q = _fan(np.full((3, 2), 120.0), 20.0)
    true = np.full((3, 2), 120.0)
    d = np.array([1, 0, 2], dtype=np.int64)
    with pytest.raises(AssertionError, match='PADDED'):
        crps_by_d(q, true, d)


def test_descending_fan_is_rejected():
    q = _fan(np.full((2, 2), 120.0), 20.0)[:, :, ::-1].copy()
    with pytest.raises(AssertionError, match='ascending'):
        crps_by_d(q, np.full((2, 2), 120.0), np.ones(2, dtype=np.int64))


def test_central_levels_are_the_fans_own_pairs():
    assert central_levels() == ((0.9, 0, 6), (0.8, 1, 5), (0.5, 2, 4))


# --------------------------------------------------------------------------- #
# 1. CRPS
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('rule', [CRPS_PWL, CRPS_TRAPEZOID])
@pytest.mark.parametrize('y', [80.0, 120.0, 200.0])
def test_crps_degenerate_fan_is_the_absolute_error(rule, y):
    """A fan collapsed onto one value is a point mass: CRPS == |y − m|, exactly.

    Both quadrature rules must agree here — every τ gap has zero spread, so the
    interior integrand is linear in τ and the trapezoid excess vanishes.
    """
    q = _degenerate(np.full((1, 1), 120.0))
    got = float(crps_steps(q, np.array([[y]]), LV, rule)[0, 0])
    print(f"[DUMP] rule={rule} y={y} crps={got:.10f} |y-m|={abs(y - 120.0)}")
    assert got == pytest.approx(abs(y - 120.0), abs=1e-9)


def test_crps_pwl_matches_dense_quadrature():
    """The closed form reproduces dense numerical integration of 2·∫ pinball dτ."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        center = rng.uniform(60.0, 300.0)
        spread = rng.uniform(0.5, 60.0)
        qrow = center + spread * np.array([-1.64, -1.28, -0.67, 0.0, 0.67, 1.28, 1.64])
        y = float(np.clip(rng.uniform(qrow[0] - 40.0, qrow[-1] + 40.0), 10.0, 400.0))
        got = float(crps_steps(qrow[None, None, :], np.array([[y]]), LV, CRPS_PWL)[0, 0])
        worst = max(worst, abs(got - _crps_dense(qrow, y)))
    print(f"[DUMP] max |closed form − dense quadrature| = {worst:.3e} mg/dL")
    assert worst < 1e-6


def test_crps_trapezoid_excess_matches_its_closed_form():
    """The documented quadrature error, verified on the widest τ gap.

    Only ``q(0.5)..q(0.75)`` is non-degenerate here, so the whole difference
    between the rules comes from that one subinterval (h = 0.25, spread D) and
    must equal ``2·[D·h·u(1−u)(1−h)/2 − D·h²·(u³+(1−u)³)/6]``, peaking at
    ``D/24`` when the truth sits mid-gap.
    """
    base, D, h, ta = 150.0, 20.0, 0.25, 0.5
    qrow = np.array([base] * 4 + [base + D] * 3, dtype=np.float64)
    for u in (0.1, 0.25, 0.5, 0.75, 0.9):
        y = base + u * D
        pwl = float(crps_steps(qrow[None, None, :], np.array([[y]]), LV, CRPS_PWL)[0, 0])
        trap = float(crps_steps(qrow[None, None, :], np.array([[y]]), LV, CRPS_TRAPEZOID)[0, 0])
        closed = 2.0 * (D * h * u * (1 - u) * (1 - h) / 2.0
                        - D * h * h * (u ** 3 + (1 - u) ** 3) / 6.0)
        print(f"[DUMP] u={u} pwl={pwl:.6f} trap={trap:.6f} excess={trap - pwl:.6f} "
              f"closed={closed:.6f}")
        assert trap - pwl == pytest.approx(closed, abs=1e-9)
        assert trap >= pwl - 1e-12                      # the chord never undercuts
    y_mid = base + 0.5 * D
    pwl = float(crps_steps(qrow[None, None, :], np.array([[y_mid]]), LV, CRPS_PWL)[0, 0])
    trap = float(crps_steps(qrow[None, None, :], np.array([[y_mid]]), LV, CRPS_TRAPEZOID)[0, 0])
    assert trap - pwl == pytest.approx(D / 24.0, abs=1e-9)


def test_crps_prefers_the_calibrated_fan():
    """CRPS is proper: the fan of the law that generated the truth wins.

    Both a too-narrow and a too-wide fan must score worse than the calibrated one
    over the same draws.
    """
    rng = np.random.default_rng(11)
    n = 4000
    y = rng.uniform(80.0, 220.0, size=(n, 1))
    cal = _uniform_fan(80.0, 220.0, n, 1)
    mid = 0.5 * (cal[..., :1] + cal[..., -1:])
    narrow = mid + 0.4 * (cal - mid)
    wide = mid + 2.0 * (cal - mid)
    d = np.ones(n, dtype=np.int64)
    s_cal = crps_by_d(cal, y, d).pooled
    s_nar = crps_by_d(narrow, y, d).pooled
    s_wid = crps_by_d(wide, y, d).pooled
    print(f"[DUMP] CRPS calibrated={s_cal:.4f} narrow={s_nar:.4f} wide={s_wid:.4f} mg/dL "
          f"(U(80,220) closed form for the untrimmed law = {140 / 6:.4f})")
    assert s_cal < s_nar and s_cal < s_wid
    # The flat-tail atoms beyond τ=0.05/0.95 are the only gap to the true law.
    assert s_cal == pytest.approx(140.0 / 6.0, rel=0.05)


def test_crps_by_d_bins_and_marks_the_pooled_figure():
    """Per-``d`` means are exact, and pooled is their supervision-weighted mean."""
    true = np.array([[100.0], [100.0], [100.0], [100.0]])
    q = _degenerate(np.array([[110.0], [130.0], [100.0], [140.0]]))   # errors 10/30/0/40
    d = np.array([1, 1, 2, 2], dtype=np.int64)
    out = crps_by_d(q, true, d)
    print(f"[DUMP] {out.by_d} pooled={out.pooled}")
    assert out.by_d == {1: pytest.approx(20.0), 2: pytest.approx(20.0)}
    assert out.n_by_d == {1: 2, 2: 2}
    assert out.pooled == pytest.approx(20.0)
    assert out.pooled_note == POOLED_NOT_COMPARABLE
    assert 'crps_pwl_pooled_NOT_COMPARABLE' in out.to_dict()


def test_pooled_moves_with_the_d_mixture_alone():
    """The pooling hazard, pinned: identical per-``d`` skill, different pooled score.

    Two protocols with the same error at every ``d`` but different shares of
    supervision at each ``d`` produce different pooled CRPS.  That is why the
    pooled figure carries ``POOLED_NOT_COMPARABLE`` and cannot select a
    checkpoint: it improves when the mask mixture softens.
    """
    def pooled_for(n_easy: int, n_hard: int) -> float:
        true = np.full((n_easy + n_hard, 1), 100.0)
        center = np.concatenate([np.full(n_easy, 105.0), np.full(n_hard, 145.0)])[:, None]
        d = np.concatenate([np.ones(n_easy, dtype=np.int64),
                            np.full(n_hard, 4, dtype=np.int64)])
        out = crps_by_d(_degenerate(center), true, d)
        assert out.by_d[1] == pytest.approx(5.0) and out.by_d[4] == pytest.approx(45.0)
        return out.pooled

    heavy_easy, balanced = pooled_for(90, 10), pooled_for(50, 50)
    print(f"[DUMP] pooled CRPS: 90/10 mixture={heavy_easy:.2f}, 50/50={balanced:.2f} mg/dL")
    assert heavy_easy == pytest.approx(9.0)
    assert balanced == pytest.approx(25.0)


# --------------------------------------------------------------------------- #
# 2. Interval / Winkler score
# --------------------------------------------------------------------------- #
def test_winkler_hand_computed():
    """W = width + (2/α)·escape, on one interval with the truth in each position."""
    lo_idx, hi_idx = 0, N_QUANTILES - 1                # the central-90% pair, α = 0.1
    q = _fan(np.full((3, 1), 150.0), 100.0)            # 90% band = 150 ± 90 = [60, 240]
    assert q[0, 0, lo_idx] == pytest.approx(60.0) and q[0, 0, hi_idx] == pytest.approx(240.0)
    true = np.array([[150.0], [50.0], [250.0]])        # inside, 10 below, 10 above
    w = winkler_steps(q, true, lo_idx, hi_idx, alpha=0.1)
    print(f"[DUMP] winkler90 = {w.ravel()}")
    assert w[0, 0] == pytest.approx(180.0)                    # width only
    assert w[1, 0] == pytest.approx(180.0 + 20.0 * 10.0)      # + (2/0.1)·10
    assert w[2, 0] == pytest.approx(180.0 + 20.0 * 10.0)


def test_winkler_by_d_reports_every_nominal_level():
    q = _fan(np.full((2, 1), 150.0), 100.0)
    true = np.array([[150.0], [150.0]])
    d = np.array([1, 2], dtype=np.int64)
    out = winkler_by_d(q, true, d)
    assert set(out) == {0.9, 0.8, 0.5}
    # All truths inside ⇒ the score is the band width: 180 / 160 / 100.
    print(f"[DUMP] " + ", ".join(f"{k}:{v.pooled:.1f}" for k, v in out.items()))
    assert out[0.9].pooled == pytest.approx(180.0)
    assert out[0.8].pooled == pytest.approx(160.0)
    assert out[0.5].pooled == pytest.approx(100.0)
    assert out[0.9].by_d == {1: pytest.approx(180.0), 2: pytest.approx(180.0)}
    assert out[0.5].pooled_note == POOLED_NOT_COMPARABLE


def test_winkler_prefers_the_calibrated_fan():
    rng = np.random.default_rng(5)
    n = 4000
    y = rng.uniform(80.0, 220.0, size=(n, 1))
    cal = _uniform_fan(80.0, 220.0, n, 1)
    mid = 0.5 * (cal[..., :1] + cal[..., -1:])
    d = np.ones(n, dtype=np.int64)
    s_cal = winkler_by_d(cal, y, d)[0.9].pooled
    s_nar = winkler_by_d(mid + 0.4 * (cal - mid), y, d)[0.9].pooled
    s_wid = winkler_by_d(mid + 2.0 * (cal - mid), y, d)[0.9].pooled
    print(f"[DUMP] winkler90 calibrated={s_cal:.2f} narrow={s_nar:.2f} wide={s_wid:.2f}")
    assert s_cal < s_nar and s_cal < s_wid


# --------------------------------------------------------------------------- #
# 3. Coverage, always with the width that bought it
# --------------------------------------------------------------------------- #
def test_coverage_cannot_be_built_without_its_width():
    """The container makes a widthless coverage figure unconstructible."""
    with pytest.raises(TypeError):
        CoverageSharpness(kind='marginal', nominal=0.9, coverage=0.9)   # type: ignore[call-arg]


def test_coverage_and_width_hand_counted():
    """Three of four steps inside a 180 mg/dL-wide 90% band."""
    q = _fan(np.full((2, 2), 150.0), 100.0)            # 90% band [60, 240], width 180
    true = np.array([[150.0, 200.0], [59.0, 100.0]])   # one escape, below
    d = np.array([1, 2], dtype=np.int64)
    out = coverage_sharpness_by_d(q, true, d)[0.9]
    print(f"[DUMP] " + " | ".join(out.render()))
    assert out.pooled.coverage == pytest.approx(0.75)
    assert out.pooled.mean_width == pytest.approx(180.0)
    assert out.pooled.n == 4
    assert out.by_d[1].coverage == pytest.approx(1.0)
    assert out.by_d[2].coverage == pytest.approx(0.5)
    assert out.pooled.kind.startswith('marginal')
    assert out.pooled_note == POOLED_NOT_COMPARABLE


def test_calibrated_fan_covers_at_nominal_and_the_narrow_one_does_not():
    """The pair is the point: the narrow fan reads SHARPER while covering less."""
    rng = np.random.default_rng(7)
    n = 20000
    y = rng.uniform(80.0, 220.0, size=(n, 1))
    cal = _uniform_fan(80.0, 220.0, n, 1)
    mid = 0.5 * (cal[..., :1] + cal[..., -1:])
    narrow = mid + 0.5 * (cal - mid)
    d = np.ones(n, dtype=np.int64)
    c = coverage_sharpness_by_d(cal, y, d)
    m = coverage_sharpness_by_d(narrow, y, d)
    for nominal in (0.9, 0.8, 0.5):
        print(f"[DUMP] {c[nominal].pooled.render()}   vs narrow: {m[nominal].pooled.render()}")
        assert c[nominal].pooled.coverage == pytest.approx(nominal, abs=0.015)
        assert m[nominal].pooled.coverage < nominal - 0.05
        assert m[nominal].pooled.mean_width == pytest.approx(
            0.5 * c[nominal].pooled.mean_width, rel=1e-9)


# --------------------------------------------------------------------------- #
# 4. Joint (simultaneous) coverage vs per-step marginal
# --------------------------------------------------------------------------- #
def _two_group_case():
    """Two groups × four patches (``d`` = 1..4) × two steps; one step escapes."""
    n, s = 8, 2
    q = _fan(np.full((n, s), 150.0), 100.0)            # 90% band [60, 240]
    true = np.full((n, s), 150.0)
    d = np.tile(np.arange(1, 5, dtype=np.int64), 2)
    group = np.repeat(np.array([0, 1], dtype=np.int64), 4)
    true[1, 1] = 400.0                                 # group 0, d = 2, step 1
    return q, true, d, group


def test_joint_coverage_is_not_the_marginal_figure():
    q, true, d, group = _two_group_case()
    out = joint_coverage_by_d(q, true, d, group)[0.9]
    print(f"[DUMP] " + " | ".join(out.render()))
    # Marginal: 15 of 16 steps inside; at d = 2, 3 of 4.
    assert out.marginal_pooled.coverage == pytest.approx(15.0 / 16.0)
    assert out.marginal_by_d[2].coverage == pytest.approx(0.75)
    # Joint: group 0 fails wherever its escape is in scope.
    assert out.joint_within_d[1].coverage == pytest.approx(1.0)
    assert out.joint_within_d[2].coverage == pytest.approx(0.5)
    assert out.joint_within_d[3].coverage == pytest.approx(1.0)
    assert out.joint_pooled.coverage == pytest.approx(0.5)
    # n counts groups for a joint figure, steps for a marginal one.
    assert out.joint_pooled.n == 2 and out.marginal_pooled.n == 16
    assert 'joint' in out.joint_pooled.kind and 'marginal' in out.marginal_pooled.kind


def test_joint_path_coverage_is_monotone_in_d():
    """Widening the scope can only lose groups: the path figure never rises."""
    q, true, d, group = _two_group_case()
    out = joint_coverage_by_d(q, true, d, group)[0.9]
    path = [out.joint_path_to_d[dd].coverage for dd in sorted(out.joint_path_to_d)]
    print(f"[DUMP] joint path coverage by d: {path}")
    assert path == [pytest.approx(1.0), pytest.approx(0.5), pytest.approx(0.5),
                    pytest.approx(0.5)]
    assert all(b <= a + 1e-12 for a, b in zip(path, path[1:]))
    for dd in sorted(out.joint_path_to_d):
        assert out.joint_path_to_d[dd].coverage <= out.marginal_by_d[dd].coverage + 1e-12


def test_joint_equals_marginal_when_the_scope_is_one_step():
    """With one unit of one step per group the two figures must coincide."""
    q = _fan(np.array([[150.0], [150.0], [150.0], [150.0]]), 100.0)
    true = np.array([[150.0], [400.0], [150.0], [150.0]])
    d = np.ones(4, dtype=np.int64)
    group = np.arange(4, dtype=np.int64)
    out = joint_coverage_by_d(q, true, d, group)[0.9]
    assert out.joint_pooled.coverage == pytest.approx(out.marginal_pooled.coverage)
    assert out.joint_pooled.coverage == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# 5. Alarm operating curve
# --------------------------------------------------------------------------- #
def test_forecast_lead_minutes_reproduces_the_horizon_grid():
    """``d`` = 1..4 one-sided IS @30/@60/@90/@120 min on the right-edge protocol."""
    lead = forecast_lead_minutes(np.array([1, 2, 3, 4], dtype=np.int64))
    print(f"[DUMP] lead minutes by d:\n{lead}")
    assert lead.shape == (4, PATCH_SIZE)
    assert list(lead[:, -1]) == [30.0, 60.0, 90.0, 120.0]
    assert lead[0, 0] == GRID_MIN
    # The step index the real-data suite reads @30 / @60 / @120 at, flattened.
    flat = lead.ravel()
    for h_min, idx in HORIZON_IDX.items():
        assert flat[idx] == float(h_min)


def test_predictive_cdf_is_the_fan_interpolated():
    q = _fan(np.array([[150.0]]), 100.0)               # τ=0.25 → 100, τ=0.5 → 150
    assert float(predictive_cdf(q, 100.0)[0, 0]) == pytest.approx(0.25)
    assert float(predictive_cdf(q, 125.0)[0, 0]) == pytest.approx(0.375)
    assert float(predictive_cdf(q, 59.0)[0, 0]) == 0.0        # below the whole fan
    assert float(predictive_cdf(q, 241.0)[0, 0]) == 1.0


def _alarm_case():
    """Four groups × four patches (``d`` = 1..4) × ``PATCH_SIZE`` steps.

    A patch centred on 80 mg/dL dips its τ=0.25 edge to 60 and alarms; one
    centred on 150 never does.  Truth is flat 150 apart from the two planted
    hypos.

      g0: alarms at d = 3, true hypo at d = 3 step 0  → detected, lead 65 min
      g1: alarms at d = 1, true hypo at d = 1 step 5  → detected, lead 30 min
      g2: alarms at d = 2, no hypo                    → false alarm
      g3: never alarms, no hypo
    """
    n, s = 16, PATCH_SIZE
    center = np.full((n, s), 150.0)
    center[0 * 4 + 2] = 80.0            # g0, d = 3
    center[1 * 4 + 0] = 80.0            # g1, d = 1
    center[2 * 4 + 1] = 80.0            # g2, d = 2
    q = _fan(center, 40.0)              # τ=0.25 edge at centre − 20
    true = np.full((n, s), 150.0)
    true[0 * 4 + 2, 0] = 60.0
    true[1 * 4 + 0, 5] = 65.0
    d = np.tile(np.arange(1, 5, dtype=np.int64), 4)
    group = np.repeat(np.arange(4, dtype=np.int64), 4)
    lead = forecast_lead_minutes(d)
    return q, true, d, group, lead


def test_alarm_curve_hand_computed():
    q, true, d, group, lead = _alarm_case()
    assert q[2, 0, HYPO_EDGE] == pytest.approx(60.0)          # the dipping edge
    curves = alarm_operating_curve(q, true, d, group, lead, observed_days=2.0)
    dep = curves.pooled.deployed
    print(f"[DUMP] deployed: rule={dep.rule} det={dep.detection_rate} "
          f"fa/day={dep.false_alarms_per_day} median lead={dep.median_lead_min} min "
          f"(n_events={dep.n_events}, n_false={dep.n_false_alarms})")
    assert dep.n_events == 2 and dep.n_detected == 2
    assert dep.detection_rate == pytest.approx(1.0)
    assert dep.n_false_alarms == 1
    assert dep.false_alarms_per_day == pytest.approx(0.5)     # 1 false alarm / 2 days
    assert dep.median_lead_min == pytest.approx(47.5)         # median(65, 30)
    assert dep.lead_min_p25 == pytest.approx(38.75)
    assert curves.pooled.pooled_note == POOLED_NOT_COMPARABLE


def test_alarm_lead_time_is_per_d_and_not_optional():
    """Restricting the scan to one ``d`` restricts the lead to that patch's window."""
    q, true, d, group, lead = _alarm_case()
    curves = alarm_operating_curve(q, true, d, group, lead, observed_days=2.0)
    by_d = curves.by_d
    print(f"[DUMP] " + " | ".join(
        f"d={k}: det={v.deployed.detection_rate} lead={v.deployed.median_lead_min} "
        f"fa/day={v.deployed.false_alarms_per_day}" for k, v in sorted(by_d.items())))
    assert by_d[1].deployed.detection_rate == pytest.approx(1.0)
    assert by_d[1].deployed.median_lead_min == pytest.approx(30.0)
    assert by_d[3].deployed.detection_rate == pytest.approx(1.0)
    assert by_d[3].deployed.median_lead_min == pytest.approx(65.0)
    # d = 2 sees the false alarm and no event at all: a rate without a denominator.
    assert by_d[2].deployed.n_events == 0
    assert math.isnan(by_d[2].deployed.detection_rate)
    assert by_d[2].deployed.false_alarms_per_day == pytest.approx(0.5)
    assert by_d[4].deployed.n_alarms == 0
    # Every per-d lead sits inside that patch's own 30-minute slice.
    for dd, curve in by_d.items():
        if curve.deployed.n_detected:
            assert (dd - 1) * 30.0 < curve.deployed.median_lead_min <= dd * 30.0


def test_alarm_curve_sweeps_monotonically():
    """Raising the score cut can only lose detections and false alarms."""
    q, true, d, group, lead = _alarm_case()
    pts = alarm_operating_curve(q, true, d, group, lead, observed_days=2.0).pooled.points
    cuts = [p.score_threshold for p in pts]
    det = [p.detection_rate for p in pts]
    fa = [p.false_alarms_per_day for p in pts]
    print(f"[DUMP] cuts={cuts} det={det} fa/day={fa}")
    assert cuts == sorted(cuts, reverse=True)
    # The list runs from the strictest cut down, so both rates only rise along it.
    assert all(b >= a - 1e-12 for a, b in zip(det, det[1:]))
    assert all(b >= a - 1e-12 for a, b in zip(fa, fa[1:]))
    assert pts[-1].score_threshold == 0.0                     # the always-fire endpoint
    assert pts[-1].detection_rate == pytest.approx(1.0)


def test_alarm_without_an_observation_span_reports_counts_only():
    """No wall-clock denominator is invented: the rate is ``None``, the count stands."""
    q, true, d, group, lead = _alarm_case()
    dep = alarm_operating_curve(q, true, d, group, lead,
                                observed_days=None).pooled.deployed
    assert dep.false_alarms_per_day is None
    assert dep.n_false_alarms == 1


# --------------------------------------------------------------------------- #
# The aggregate
# --------------------------------------------------------------------------- #
def test_score_fan_runs_all_five_and_marks_every_pooled_figure():
    q, true, d, group, lead = _alarm_case()
    out = score_fan(q, true, d, group, lead, observed_days=2.0)
    assert set(out.winkler) == {0.9, 0.8, 0.5}
    assert set(out.coverage) == {0.9, 0.8, 0.5}
    assert out.alarm is not None
    keys = set(out.crps.to_dict()) | set(out.joint[0.9].to_dict())
    marked = {k for k in keys if 'pooled' in k}
    print(f"[DUMP] pooled keys: {sorted(marked)}")
    assert marked and all('NOT_COMPARABLE' in k or k.endswith('_n_pooled') for k in marked)
    assert score_fan(q, true, d, group).alarm is None         # no lead ⇒ no curve
