"""Quantile-CQR re-fit wiring tests (additive band path):

* ``Window`` carries an optional RAW band fan; ``forecast_bands`` stacks it and
  returns None whenever a window lacks bands (old-cache compat).
* ``run_eval.evaluate_from_windows`` grows a ``conformal_cqr`` block when bands are
  present, calibrating the test bands toward nominal coverage — and leaves it None on
  the band-less path (additivity / None ⇒ identity).
* the METRICS are no longer basis-invariant: with bands the suite is BAND-SCORED
  (``pred_eff = clip(true, q[METRIC_BAND_TAU_LO], q[METRIC_BAND_TAU_HI])``), so the
  headline block is strictly better than the band-less run, and the median-line basis
  survives bit-identically under ``metrics[h]['median_line']``.
* the point ``conformal`` block carries whatever basis ``evaluate_from_windows`` hands
  ``conformal_intervals`` — the band-projected arrays when both splits have bands, the
  median line otherwise. Both are legitimate; the invariant asserted here is the one
  that holds either way (the projected residual is ≤ the median residual element-wise,
  so the band half-width can only shrink).
* the sim ``collect_sim_rows`` band capture is single-pass-only and leaves
  median/pred bit-identical to the no-delta run.
"""
import numpy as np
import pytest

from config import (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES, QUANTILE_LEVELS,
                    METRIC_BAND_TAU_LO, METRIC_BAND_TAU_HI)
from realdata.calibrate import Window, forecast_bands, forecast_windows
from realdata.metrics import band_project, conformal_intervals
from realdata.run_eval import evaluate_from_windows

PRED_STEPS = PREDICTION_PATCHES * PATCH_SIZE
LEVELS = QUANTILE_LEVELS
MED = LEVELS.index(0.5)
LO, HI = LEVELS.index(0.05), LEVELS.index(0.95)
BAND_LO, BAND_HI = LEVELS.index(METRIC_BAND_TAU_LO), LEVELS.index(METRIC_BAND_TAU_HI)

# The block ``_point_block`` emits per basis — the keys the band-scored row and its
# ``median_line`` twin share, and on which the median line must reproduce the
# band-less run exactly.
POINT_KEYS = ('rmse_point', 'mae_point', 'rmse_winmean', 'mae_winmean', 'rmse_macro',
              'mard', 'clarke_A', 'clarke_AB', 'clarke_D', 'clarke_E', 'skill_point')
ERROR_KEYS = ('rmse_point', 'mae_point', 'rmse_winmean', 'mae_winmean', 'rmse_macro', 'mard')


def _ascending_fan(center: np.ndarray, half: float) -> np.ndarray:
    """(PRED_STEPS, K) ascending fan centred on ``center`` with (τ-0.5)·2·half offsets."""
    off = np.array([(t - 0.5) * 2.0 for t in LEVELS])     # -0.9..+0.9
    return center[:, None] + half * off[None, :]


def _make_window(rng, patient: str, with_bands: bool, half: float = 8.0) -> Window:
    center = 120.0 + rng.standard_normal(PRED_STEPS) * 5.0
    cgm = center + rng.standard_normal(PRED_STEPS) * 30.0     # truth far wider than band
    bands = _ascending_fan(center, half) if with_bands else None
    return Window(patient=patient, pred_bg=center.copy(), last_bg=float(center[0]),
                  cgm=cgm.astype(np.float64), bands=bands)


def test_forecast_bands_shape_and_none():
    rng = np.random.default_rng(0)
    ws = [_make_window(rng, f"p{i}", with_bands=True) for i in range(5)]
    b = forecast_bands(ws)
    assert b is not None and b.shape == (5, PRED_STEPS, N_QUANTILES), b.shape
    # any window missing bands ⇒ None (old-cache compat)
    ws2 = ws + [_make_window(rng, "px", with_bands=False)]
    assert forecast_bands(ws2) is None
    assert forecast_bands([]) is None
    print(f"\n[DUMP] forecast_bands -> {b.shape}; None when a window lacks bands ✓")


def test_evaluate_from_windows_cqr_and_additivity():
    """CQR additivity + the BAND-SCORED suite contract.

    ``conformal_cqr`` is still purely additive (present with bands, None without). The
    metric suite is NOT basis-invariant any more: with bands it scores the
    band-projected forecast, so its level metrics are strictly better than the
    band-less run, and the old median-line numbers move to
    ``metrics[h]['median_line']``, reproduced bit-identically. The point ``conformal``
    block moves to the same band-projected basis on both splits (cal and test must
    share one basis for the residuals to stay exchangeable), which can only shrink its
    half-width.
    """
    rng = np.random.default_rng(7)
    # Two synthetic cohorts' worth of windows; bands are DELIBERATELY too narrow
    # (half=6) so raw 90% coverage is well below 0.90 and CQR must widen it.
    cal_b = [_make_window(rng, f"c{i%3}", with_bands=True, half=6.0) for i in range(120)]
    test_b = [_make_window(rng, f"t{i%3}", with_bands=True, half=6.0) for i in range(120)]
    # Identical windows but with bands stripped (old-cache path).
    cal_n = [Window(w.patient, w.pred_bg, w.last_bg, w.cgm, None) for w in cal_b]
    test_n = [Window(w.patient, w.pred_bg, w.last_bg, w.cgm, None) for w in test_b]

    res_b = evaluate_from_windows(cal_b, test_b)
    res_n = evaluate_from_windows(cal_n, test_n)

    # --- CQR present + calibrated coverage closer to 0.90 than raw -----------
    cq = res_b['conformal_cqr']
    assert cq is not None
    for h in ('30', '60', '120'):
        d = cq[h]
        raw_gap = abs(d['raw_cov90'] - 0.90)
        cal_gap = abs(d['cal_cov90'] - 0.90)
        assert cal_gap <= raw_gap + 1e-9, (h, d)
        assert d['n_cal'] == 120 and d['n_test'] == 120
    print(f"\n[DUMP] CQR cov90 @120 raw={cq['120']['raw_cov90']:.3f} "
          f"-> cal={cq['120']['cal_cov90']:.3f} (target 0.90) ✓")

    # --- no-bands path: conformal_cqr is None --------------------------------
    assert res_n['conformal_cqr'] is None

    # --- the suite is BAND-SCORED with bands, median-line without ------------
    for h in ('30', '60', '120'):
        rb, rn = res_b['metrics'][h], res_n['metrics'][h]
        assert 'median_line' not in rn and 'band_cov50' not in rn and 'band_width' not in rn
        for k in ERROR_KEYS:
            assert rb[k] <= rn[k] + 1e-9, (h, k, rb[k], rn[k])
        assert rb['rmse_point'] < rn['rmse_point'], \
            f"a non-degenerate band must strictly reduce the scored RMSE @{h}"
        # the median-line basis survives verbatim
        for k in POINT_KEYS:
            assert rb['median_line'][k] == rn[k], (h, k, rb['median_line'][k], rn[k])
        assert rb['rmse_persist_point'] == rn['rmse_persist_point']
        assert rb['rmse_persist_winmean'] == rn['rmse_persist_winmean']
        assert rb['n_windows'] == rn['n_windows']
        assert 0.0 <= rb['band_cov50'] <= 1.0 and rb['band_width'] > 0.0
    print(f"[DUMP] suite @120 rmse band {res_b['metrics']['120']['rmse_point']:.2f} < "
          f"median {res_n['metrics']['120']['rmse_point']:.2f}; median_line reproduces "
          f"the band-less block exactly; band cov50 "
          f"{res_b['metrics']['120']['band_cov50']:.3f}, width "
          f"{res_b['metrics']['120']['band_width']:.1f} mg/dL ✓")

    # --- point conformal: same projected basis on BOTH splits ----------------
    cal_bands, test_bands = forecast_bands(cal_b), forecast_bands(test_b)
    _, cal_true, _, _ = forecast_windows(cal_b)
    test_pred, test_true, _, _ = forecast_windows(test_b)
    cal_eff = band_project(cal_true, cal_bands[..., BAND_LO], cal_bands[..., BAND_HI])
    test_eff = band_project(test_true, test_bands[..., BAND_LO], test_bands[..., BAND_HI])
    expect = {str(h): v for h, v in
              conformal_intervals(cal_eff, cal_true, test_eff, test_true).items()}
    assert res_b['conformal'] == expect, "banded run must score conformal on the band basis"
    for h in ('30', '60', '120'):
        assert res_b['conformal'][h]['half_width'] <= res_n['conformal'][h]['half_width'] + 1e-9
    print(f"[DUMP] point conformal basis: band-projected (both splits) — half_width @120 "
          f"{res_b['conformal']['120']['half_width']:.2f} vs median-line "
          f"{res_n['conformal']['120']['half_width']:.2f}; median-line run unchanged ✓")

    # --- decision-offset sweep: the band edge is the more sensitive detector --
    for side in ('hypo', 'hyper'):
        for h in ('30', '60', '120'):
            cb, cn = res_b['threshold_curves'][side][h], res_n['threshold_curves'][side][h]
            assert len(cb) == len(cn)
            for pb, pn in zip(cb, cn):
                assert pb['offset'] == pn['offset']
                assert pb['n_true'] == pn['n_true'], "the truth side is basis-independent"
                assert pb['n_pred'] >= pn['n_pred'], (side, h, pb, pn)
                if pn['recall'] is not None:
                    assert pb['recall'] >= pn['recall'] - 1e-12, (side, h, pb, pn)
    floor = res_b['selected_offsets']['min_precision']
    assert res_n['selected_offsets']['min_precision'] == floor
    for h in ('30', '60', '120'):
        d = res_b['selected_offsets']['hypo'][h]
        assert (d['cal_precision'] is None or d['cal_precision'] >= floor
                or d['offset'] == 0.0), d
    swept_b = sum(p['n_pred'] for p in res_b['threshold_curves']['hypo']['120'])
    swept_n = sum(p['n_pred'] for p in res_n['threshold_curves']['hypo']['120'])
    print(f"[DUMP] hypo sweep @120 alarms over the whole δ grid, band edge/median "
          f"{swept_b}/{swept_n}; selected offset "
          f"{res_b['selected_offsets']['hypo']['120']['offset']:.1f} vs "
          f"{res_n['selected_offsets']['hypo']['120']['offset']:.1f} (floor {floor:.2f}) ✓")


@pytest.mark.parametrize("delta_none", [True, False])
def test_sim_collect_rows_band_capture(delta_none):
    """Single-pass branch captures a (PRED_STEPS, K) band fan; median/pred are
    bit-identical to the no-delta run. Requires the simulator + a checkpoint, so
    skip cleanly when neither is available."""
    import os
    import torch
    sys_path = os.path.join(os.path.dirname(__file__), '..', 'metrics', 'sim')
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    try:
        from sim_data import collect_sim_rows, make_sim_runs, FIG_ROLLS, FIG_STEPS
        from realdata.report import load_model
    except Exception as e:
        pytest.skip(f"sim/report imports unavailable: {e}")
    import glob
    ckpts = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '..', 'checkpoints',
                                          't1dmai_step_*.pt')))
    if not ckpts:
        pytest.skip("no checkpoint available")
    device = torch.device('cpu')
    model, stats, _ = load_model(device, ckpts[-1])
    runs = make_sim_runs((8000,), 90.0)
    delta = None if delta_none else getattr(model, 'conformal_delta', None)
    rows = collect_sim_rows(model, stats, runs, device, cap=2, conformal_delta=delta)
    assert rows, "expected at least one sim row"
    for r in rows:
        if FIG_ROLLS == 1:
            assert r['bands'] is not None
            assert r['bands'].shape == (min(FIG_STEPS, PRED_STEPS), N_QUANTILES), r['bands'].shape
        else:
            assert r['bands'] is None     # rolling: sim delta cannot span FIG_STEPS

    # median/pred identical regardless of conformal_delta
    rows0 = collect_sim_rows(model, stats, runs, device, cap=2, conformal_delta=None)
    for ra, rb in zip(rows, rows0):
        assert np.array_equal(ra['pred'], rb['pred']), "median/pred must be delta-invariant"
    print(f"\n[DUMP] sim band capture: FIG_ROLLS={FIG_ROLLS}, bands shape "
          f"{rows[0]['bands'].shape if rows[0]['bands'] is not None else None}; "
          f"pred delta-invariant ✓")
