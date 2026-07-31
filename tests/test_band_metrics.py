"""Band-scored metric basis (``realdata.metrics``).

The headline real/sim level metrics score the BAND-PROJECTED forecast
``pred_eff = clip(true, q[METRIC_BAND_TAU_LO], q[METRIC_BAND_TAU_HI])`` instead of the
median line. These are deterministic synthetic checks of the four properties the
change rests on:

  * ``band_project`` is zero-error inside the band, the distance to the nearer edge
    outside it, and the identity on a collapsed band;
  * every band-scored level metric is ≤ its ``median_line`` twin on the same windows,
    with EXACT equality when the band is degenerate;
  * ``band_cov50`` / ``band_width`` are read at the horizon step only, and match a
    hand-counted construction;
  * ``bands=None`` still emits the pre-band block verbatim — same key set, same values
    as the banded run's ``median_line``.

No model and no simulator: pure numpy arrays, fast.
"""
from __future__ import annotations

import numpy as np

from config import (METRIC_BAND_TAU_LO, METRIC_BAND_TAU_HI, QUANTILE_LEVELS,
                    N_QUANTILES, PREDICTION_PATCHES, PATCH_SIZE,
                    BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD)
from realdata.horizons import HORIZONS, HORIZON_IDX
from realdata.metrics import band_project, compute_suite, conformal_intervals

PRED_STEPS = PREDICTION_PATCHES * PATCH_SIZE
LO = QUANTILE_LEVELS.index(METRIC_BAND_TAU_LO)
HI = QUANTILE_LEVELS.index(METRIC_BAND_TAU_HI)
MED = QUANTILE_LEVELS.index(0.5)

# The exact per-horizon key set ``compute_suite`` emitted before the band basis landed;
# ``bands=None`` must still emit it verbatim (no median_line, no band_* keys).
PRE_BAND_KEYS = {
    'rmse_point', 'mae_point', 'rmse_winmean', 'mae_winmean', 'rmse_macro', 'mard',
    'clarke_A', 'clarke_AB', 'clarke_D', 'clarke_E', 'skill_point',
    'rmse_persist_point', 'rmse_persist_winmean', 'hypo', 'hyper', 'n_windows',
}
# The block ``_point_block`` emits for one basis — the keys the band-scored row and its
# ``median_line`` twin share.
POINT_KEYS = ('rmse_point', 'mae_point', 'rmse_winmean', 'mae_winmean', 'rmse_macro',
              'mard', 'clarke_A', 'clarke_AB', 'clarke_D', 'clarke_E', 'skill_point')
ERROR_KEYS = ('rmse_point', 'mae_point', 'rmse_winmean', 'mae_winmean', 'rmse_macro', 'mard')


def _fan(center: np.ndarray, half: float) -> np.ndarray:
    """``(N, S, N_QUANTILES)`` ascending fan centred on ``center`` ``(N, S)``.

    Level τ sits at ``center + 2·half·(τ − 0.5)``, so the scored band
    ``[τ_LO, τ_HI] = [0.25, 0.75]`` spans ``center ± half/2``.
    """
    off = np.array([(t - 0.5) * 2.0 for t in QUANTILE_LEVELS])
    return center[:, :, None] + half * off[None, None, :]


def _degenerate_fan(center: np.ndarray, spread: float = 20.0) -> np.ndarray:
    """``(N, S, N_QUANTILES)`` fan whose scored band has COLLAPSED onto the median.

    Every level in ``[METRIC_BAND_TAU_LO, METRIC_BAND_TAU_HI]`` (the alarm taus too,
    which are numerically the same pair) sits exactly on ``center``; the outer levels
    still spread, so the fan stays ascending and the band-projected forecast must
    reduce to the median line exactly.
    """
    off = np.array([0.0 if METRIC_BAND_TAU_LO <= t <= METRIC_BAND_TAU_HI
                    else (t - 0.5) * 2.0 * spread for t in QUANTILE_LEVELS])
    return center[:, :, None] + off[None, None, :]


def _synthetic(n_windows: int = 60, seed: int = 3, half: float = 20.0
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """A band-vs-median test case: ``(pred, true, last_bg, patients, bands)``.

    ``pred`` ``(n_windows, PRED_STEPS)`` is the median line — a random walk off a
    per-window level spread across the whole clinical range, so both excursion regions
    are populated — and ``true`` a noisy copy whose spread (σ ≈ 18 mg/dL) leaves a good
    fraction of the truth INSIDE the ±``half``/2 band and the rest outside, so both
    regimes of the projection are exercised.
    """
    rng = np.random.default_rng(seed)
    drift = np.cumsum(rng.standard_normal((n_windows, PRED_STEPS)) * 6.0, axis=1)
    pred = rng.uniform(70.0, 210.0, n_windows)[:, None] + drift
    true = np.clip(pred + rng.standard_normal((n_windows, PRED_STEPS)) * 18.0, 40.0, 400.0)
    last_bg = pred[:, 0] - rng.standard_normal(n_windows) * 5.0
    patients = [f"p{i % 4}" for i in range(n_windows)]
    return pred, true, last_bg, patients, _fan(pred, half)


# ------------------------------- band_project ------------------------------- #
def test_band_project_zero_inside_distance_outside():
    """Zero residual inside the band; the distance to the NEARER edge outside it."""
    true = np.array([[110.0, 50.0, 200.0, 100.0, 120.0]])
    lo = np.full_like(true, 100.0)
    hi = np.full_like(true, 120.0)
    eff = band_project(true, lo, hi)
    resid = eff - true
    assert eff.shape == true.shape
    assert resid[0, 0] == 0.0, "truth strictly inside the band must score zero error"
    assert resid[0, 1] == 50.0, "below the band: distance to the LOWER edge"
    assert resid[0, 2] == -80.0, "above the band: distance to the UPPER edge"
    assert resid[0, 3] == 0.0 and resid[0, 4] == 0.0, "the edges themselves are inside"
    # closed form: |residual| == max(0, lo − true, true − hi)
    expect = np.maximum(0.0, np.maximum(lo - true, true - hi))
    assert np.array_equal(np.abs(resid), expect)
    print(f"\n[DUMP] band_project true {true[0].tolist()} band [100,120] -> "
          f"{eff[0].tolist()}, |resid| {np.abs(resid)[0].tolist()} ✓")


def test_band_project_degenerate_band_is_the_median():
    """A collapsed band (``lo == hi``) returns that common value for every truth."""
    rng = np.random.default_rng(11)
    med = 120.0 + rng.standard_normal((7, PRED_STEPS)) * 25.0
    true = med + rng.standard_normal((7, PRED_STEPS)) * 40.0
    eff = band_project(true, med, med.copy())
    assert np.array_equal(eff, med), "lo == hi must project every truth onto the median"
    print(f"[DUMP] degenerate band: projection == median exactly "
          f"(max|Δ| {np.abs(eff - med).max():.1e}) ✓")


def test_band_project_rejects_mismatched_shapes():
    true = np.zeros((3, PRED_STEPS))
    try:
        band_project(true, np.zeros((3, PRED_STEPS - 1)), np.zeros((3, PRED_STEPS)))
    except AssertionError:
        print("[DUMP] band_project shape assert fires ✓")
    else:
        raise AssertionError("band_project accepted a shape mismatch")


# ----------------------------- compute_suite -------------------------------- #
def test_band_scored_never_worse_than_median_line():
    """Every band-scored level metric ≤ its ``median_line`` twin, strictly so here.

    The projection is the closest point of ``[q₂₅, q₇₅]`` to the truth and the median
    lies inside that interval, so the per-step error can only shrink — element-wise,
    hence for RMSE / MAE / MARD / macro-RMSE alike.
    """
    pred, true, last_bg, pats, bands = _synthetic()
    res = compute_suite(pred, true, last_bg, pats, bands=bands)
    for h in HORIZONS:
        row, med = res[h], res[h]['median_line']
        for k in ERROR_KEYS:
            assert row[k] <= med[k] + 1e-9, (h, k, row[k], med[k])
        assert row['rmse_point'] < med['rmse_point'], \
            f"a non-degenerate band must strictly reduce RMSE @{h}"
        assert row['skill_point'] >= med['skill_point'] - 1e-9, \
            "skill is measured against the SAME point-persistence baseline on both bases"
        assert row['clarke_A'] >= med['clarke_A'] - 1e-9
        print(f"[DUMP] @{h:>3}m  rmse band {row['rmse_point']:6.2f} vs median "
              f"{med['rmse_point']:6.2f} | mard {row['mard']:5.2f} vs {med['mard']:5.2f} "
              f"| clarkeA {row['clarke_A']:5.1f} vs {med['clarke_A']:5.1f}")


def test_band_cov50_and_width_hand_counted():
    """``band_cov50`` / ``band_width`` are read at the horizon step, hand-countable."""
    n = 4
    pred = np.full((n, PRED_STEPS), 120.0)
    # Band [90, 150] at every step: width 60. Truth constant per window so the count is
    # the same at every horizon: in, below, above, exactly on the upper edge.
    base = np.array([60.0, 75.0, 90.0, 120.0, 150.0, 165.0, 180.0])
    assert base.shape == (N_QUANTILES,) and base[LO] == 90.0 and base[HI] == 150.0
    bands = np.broadcast_to(base, (n, PRED_STEPS, N_QUANTILES)).copy()
    true = np.repeat(np.array([[120.0], [80.0], [200.0], [150.0]]), PRED_STEPS, axis=1)
    last_bg = true[:, 0].copy()
    res = compute_suite(pred, true, last_bg, ['a', 'a', 'b', 'b'], bands=bands)
    for h in HORIZONS:
        assert res[h]['band_cov50'] == 0.5, (h, res[h]['band_cov50'])
        assert res[h]['band_width'] == 60.0, (h, res[h]['band_width'])
        assert res[h]['n_windows'] == n
    print(f"[DUMP] hand-counted band: cov50 {res[120]['band_cov50']:.2f} (2 of 4 inside "
          f"[90,150]; the 150 edge counts as inside), width "
          f"{res[120]['band_width']:.1f} mg/dL ✓")


def test_band_cov50_tracks_the_horizon_step_only():
    """Coverage is the realized fraction at step ``HORIZON_IDX[h]``, NOT pooled 0..k."""
    n = 5
    pred = np.full((n, PRED_STEPS), 120.0)
    bands = _fan(pred, half=20.0)                       # band = 120 ± 10
    true = np.full((n, PRED_STEPS), 300.0)              # everywhere outside …
    k30 = HORIZON_IDX[30]
    true[:3, k30] = 120.0                               # … except 3 of 5 at the 30-min step
    last_bg = np.full(n, 120.0)
    res = compute_suite(pred, true, last_bg, [f"p{i}" for i in range(n)], bands=bands)
    assert res[30]['band_cov50'] == 3 / 5, res[30]['band_cov50']
    assert res[60]['band_cov50'] == 0.0 and res[120]['band_cov50'] == 0.0
    print(f"[DUMP] cov50 @30 {res[30]['band_cov50']:.2f} (step {k30} only), "
          f"@60 {res[60]['band_cov50']:.2f}, @120 {res[120]['band_cov50']:.2f} ✓")


def test_bands_none_emits_the_pre_band_block():
    """``bands=None`` keeps the pre-band key set AND the banded run's median values."""
    pred, true, last_bg, pats, bands = _synthetic()
    res_b = compute_suite(pred, true, last_bg, pats, bands=bands)
    res_n = compute_suite(pred, true, last_bg, pats, bands=None)
    assert set(res_n.keys()) == set(res_b.keys()) == set(HORIZONS) | {'cgega'}
    for h in HORIZONS:
        assert set(res_n[h].keys()) == PRE_BAND_KEYS, sorted(res_n[h].keys())
        assert 'median_line' not in res_n[h]
        assert 'band_cov50' not in res_n[h] and 'band_width' not in res_n[h]
        med = res_b[h]['median_line']
        for k in POINT_KEYS:
            assert med[k] == res_n[h][k], (h, k, med[k], res_n[h][k])
        # persistence carries no band, so it is shared verbatim by both bases
        assert res_b[h]['rmse_persist_point'] == res_n[h]['rmse_persist_point']
        assert res_b[h]['rmse_persist_winmean'] == res_n[h]['rmse_persist_winmean']
        assert res_b[h]['n_windows'] == res_n[h]['n_windows']
    print(f"[DUMP] bands=None keys {sorted(res_n[120].keys())}")
    print(f"[DUMP] median_line == bands=None block, exact: rmse@120 "
          f"{res_b[120]['median_line']['rmse_point']!r} == {res_n[120]['rmse_point']!r} ✓")


def test_degenerate_band_reproduces_the_median_line_exactly():
    """A collapsed scored band makes the banded run identical to the band-less one."""
    pred, true, last_bg, pats, _ = _synthetic()
    bands = _degenerate_fan(pred)
    res_b = compute_suite(pred, true, last_bg, pats, bands=bands)
    res_n = compute_suite(pred, true, last_bg, pats, bands=None)
    for h in HORIZONS:
        for k in POINT_KEYS:
            assert res_b[h][k] == res_n[h][k] == res_b[h]['median_line'][k], (h, k)
        assert res_b[h]['band_width'] == 0.0
        # the alarm edges collapse onto the median too, so the detectors coincide
        assert res_b[h]['hypo'] == res_n[h]['hypo']
        assert res_b[h]['hyper'] == res_n[h]['hyper']
    assert res_b['cgega'] == res_n['cgega'], "CG-EGA must be identical on a collapsed band"
    print(f"[DUMP] degenerate band == median line == bands=None: rmse@120 "
          f"{res_b[120]['rmse_point']!r}; band_width 0.0; cgega identical ✓")


def test_band_scoring_does_not_disturb_the_alarm_edges():
    """hypo/hyper stay on the ALARM taus, independent of the metric-band projection.

    The truth side is untouched by the projection, so ``n_true`` must still be the raw
    threshold crossing count at the horizon step; the alarm side reads the alarm band
    edges, which are at least as sensitive as the median line.
    """
    pred, true, last_bg, pats, bands = _synthetic()
    res = compute_suite(pred, true, last_bg, pats, bands=bands)
    res_n = compute_suite(pred, true, last_bg, pats, bands=None)
    for h in HORIZONS:
        tk = true[:, HORIZON_IDX[h]]
        assert set(res[h]['hypo'].keys()) == {'recall', 'precision', 'n_true', 'n_pred'}
        assert res[h]['hypo']['n_true'] == int((tk < BG_HYPO_THRESHOLD).sum())
        assert res[h]['hyper']['n_true'] == int((tk > BG_HYPER_THRESHOLD).sum())
        assert res[h]['hypo']['n_pred'] >= res_n[h]['hypo']['n_pred'], \
            "the τ-lower alarm edge cannot fire less often than the median"
        assert res[h]['hyper']['n_pred'] >= res_n[h]['hyper']['n_pred'], \
            "the τ-upper alarm edge cannot fire less often than the median"
    print(f"[DUMP] alarm blocks intact under band scoring: @120 hypo n_true "
          f"{res[120]['hypo']['n_true']}, n_pred edge/median "
          f"{res[120]['hypo']['n_pred']}/{res_n[120]['hypo']['n_pred']}; hyper n_true "
          f"{res[120]['hyper']['n_true']}, n_pred edge/median "
          f"{res[120]['hyper']['n_pred']}/{res_n[120]['hyper']['n_pred']} ✓")


def test_cgega_region_totals_depend_only_on_the_truth():
    """Hold the truth and the window set fixed, vary only the forecast: the three
    CG-EGA per-region totals must not move.

    A point's glycemic region is that of its TRUE glucose, so ``ap+be+ep`` per region is
    a function of the truth and the window set alone — the forecast may only redistribute
    each total across AP/BE/EP. Score the forecast as the reference instead and the
    region follows the forecast, so the denominators shift between two runs over
    identical truth and the per-region percentages are no longer comparable.

    Both forecasts here sit far outside their own band, so the projected series is
    displaced ~51 mg/dL below the truth in one run and ~51 above it in the other — well
    across both the 70 and the 180 mg/dL region boundaries for a large share of points.
    """
    rng = np.random.default_rng(17)
    n = 40
    true = np.clip(
        rng.uniform(60.0, 280.0, n)[:, None]
        + np.cumsum(rng.standard_normal((n, PRED_STEPS)) * 5.0, axis=1),
        40.0, 400.0)
    last_bg = true[:, 0] - rng.standard_normal(n) * 5.0
    pats = [f"p{i % 4}" for i in range(n)]

    low = np.clip(true - 55.0, 40.0, 400.0)
    high = np.clip(true + 55.0, 40.0, 400.0)
    res_lo = compute_suite(low, true, last_bg, pats, bands=_fan(low, 8.0))
    res_hi = compute_suite(high, true, last_bg, pats, bands=_fan(high, 8.0))

    def _tot(res, reg):
        c = res['cgega']['counts']
        return c[f'ap_{reg}'] + c[f'be_{reg}'] + c[f'ep_{reg}']

    for reg in ('hypo', 'eu', 'hyper'):
        assert _tot(res_lo, reg) == _tot(res_hi, reg), (
            f"CG-EGA {reg} denominator moved with the forecast: "
            f"{_tot(res_lo, reg)} vs {_tot(res_hi, reg)} — the region is being binned "
            f"on the forecast, not the truth"
        )
    # not vacuous: the two forecasts genuinely score differently inside those totals
    assert res_lo['cgega']['counts'] != res_hi['cgega']['counts']
    assert sum(res_lo['cgega']['counts'].values()) == true.size
    print(f"[DUMP] region totals invariant to the forecast: "
          f"{[(r, _tot(res_lo, r)) for r in ('hypo', 'eu', 'hyper')]} "
          f"== {[(r, _tot(res_hi, r)) for r in ('hypo', 'eu', 'hyper')]} ✓")


# ----------------------------- conformal basis ------------------------------ #
def test_conformal_intervals_on_the_projected_basis_is_tighter():
    """Fed the band-projected arrays, the conformal half-width shrinks.

    ``conformal_intervals`` takes no basis flag — the caller decides. The residual
    ``|band_project(true) − true|`` is ≤ ``|median − true|`` element-wise, so its
    empirical quantile (the half-width) can only shrink; it then reads as the extra
    width needed ON TOP OF the band to reach 90%.
    """
    n = 400        # enough windows that the 90% split-conformal coverage is tight
    cal_pred, cal_true, _, _, cal_bands = _synthetic(n_windows=n, seed=5)
    test_pred, test_true, _, _, test_bands = _synthetic(n_windows=n, seed=6)
    cal_eff = band_project(cal_true, cal_bands[..., LO], cal_bands[..., HI])
    test_eff = band_project(test_true, test_bands[..., LO], test_bands[..., HI])
    band = conformal_intervals(cal_eff, cal_true, test_eff, test_true)
    point = conformal_intervals(cal_pred, cal_true, test_pred, test_true)
    for h in HORIZONS:
        assert band[h]['half_width'] < point[h]['half_width'], (h, band[h], point[h])
        for basis in (band, point):
            assert 0.85 <= basis[h]['coverage'] <= 0.96, (h, basis[h])
        print(f"[DUMP] conformal @{h:>3}m  half_width band {band[h]['half_width']:6.2f} "
              f"vs median {point[h]['half_width']:6.2f} | cov "
              f"{band[h]['coverage']:.3f} / {point[h]['coverage']:.3f}")


# --------------------------------- config ----------------------------------- #
def test_metric_band_taus_are_quantile_levels_straddling_the_median():
    assert METRIC_BAND_TAU_LO in QUANTILE_LEVELS and METRIC_BAND_TAU_HI in QUANTILE_LEVELS
    assert METRIC_BAND_TAU_LO < 0.5 < METRIC_BAND_TAU_HI
    assert LO < MED < HI, (LO, MED, HI)
    assert len(QUANTILE_LEVELS) == N_QUANTILES
    print(f"[DUMP] metric band τ ({METRIC_BAND_TAU_LO}, {METRIC_BAND_TAU_HI}) "
          f"-> fan columns ({LO}, {HI}) straddling the median column {MED} ✓")
