"""
Comparison metric suite for the evaluation harness.

Replicates train.py's exact definitions (Clarke Error Grid zones, MARD, band-edge
hypo/hyper recall/precision — hypo off the τ-lower band edge, hyper off the τ-upper,
with strict recall and a precision-only ±EXCURSION_PRECISION_TOLERANCE_MGDL forgiveness
band) but operates on plain ``(pred_bg, true_cgm)`` arrays of shape
(n_windows, PRED_STEPS), where ``pred_bg`` is the median line ``median_bg =
f_inv(median)`` and the band fan is passed alongside.

The headline level metrics are BAND-SCORED. The forecast is a quantile fan, so the
effective point forecast is the band point nearest the truth,
``pred_eff = clip(true, q[METRIC_BAND_TAU_LO], q[METRIC_BAND_TAU_HI])`` — zero error
where the truth lies inside the band, the distance to the nearer edge where it lies
outside. Every downstream formula (RMSE / MAE / MARD / Clarke / CG-EGA / skill) is
unchanged; only the array it consumes changes, and a degenerate band reproduces the
median-line numbers exactly. The same block computed on the median line is kept
alongside under ``out[h]['median_line']`` — the basis for peer / literature
comparisons, whose published numbers are point forecasts. Persistence carries no band,
so the skill column compares a band-scored model against a point baseline.

``bands=None`` keeps the pre-band behaviour exactly: every metric off the median
``pred``, and no ``median_line`` / ``band_cov50`` / ``band_width`` keys.

Also carries the comparison-audit reporting fixes:

  * STRICT point-at-horizon RMSE/MAE  (our original convention)
  * WINDOW-MEAN RMSE/MAE pooled over steps 0..k  (the Crossformer/PatchTST basis)
  * naive-PERSISTENCE baseline + skill, per horizon  (last-CGM-value forecast)
  * per-patient MACRO RMSE  (secondary aggregation)
  * realized band coverage ``band_cov50`` + mean ``band_width``, per horizon
  * split-conformal calibrated 90% interval  (needs only point residuals)
"""
from __future__ import annotations

import numpy as np

import cg_ega
from config import (BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD, QUANTILE_LEVELS,
                    HYPO_ALARM_QUANTILE_TAU, HYPER_ALARM_QUANTILE_TAU,
                    METRIC_BAND_TAU_LO, METRIC_BAND_TAU_HI,
                    EXCURSION_PRECISION_TOLERANCE_MGDL)

from .horizons import HORIZONS, HORIZON_IDX as _IDX  # step index (0-based) per minute-horizon

# Clinical hypo/hyper recall/precision key off the QUANTILE BAND EDGES (matching
# train.py's validation table): hypo off the τ=HYPO_ALARM_QUANTILE_TAU lower edge,
# hyper off the τ=HYPER_ALARM_QUANTILE_TAU upper edge, from the per-window band fan
# (``Window.bands``) when available, else the median. RECALL is strict; PRECISION
# forgives a near-boundary false alarm within EXCURSION_PRECISION_TOLERANCE_MGDL of the
# true value (CGM sensor noise near a threshold does not deflate it) — same as train.py.
_HYPO_BAND_IDX = QUANTILE_LEVELS.index(HYPO_ALARM_QUANTILE_TAU)
_HYPER_BAND_IDX = QUANTILE_LEVELS.index(HYPER_ALARM_QUANTILE_TAU)

# Band the LEVEL metrics score against (a knob distinct from the alarm taus above).
_BAND_LO_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_LO)
_BAND_HI_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_HI)
_N_QUANTILES = len(QUANTILE_LEVELS)

# Slack on the ascending-fan assertion: the fan reaches mg/dL through f_inv, so two
# adjacent quantiles that coincide in risk space can differ by a float rounding step.
_FAN_ORDER_TOL_MGDL = 1e-6


def _clarke(pred: np.ndarray, true: np.ndarray):
    """Clarke Error Grid zone masks (A,B,C,D,E), matching train.py:856-872."""
    pb = np.clip(pred, 1.0, None); tb = np.clip(true, 1.0, None)
    rel = np.abs(pb - tb) / tb
    A = (rel <= 0.20) | ((pb <= 70) & (tb <= 70))
    E = ((pb <= 70) & (tb >= 180)) | ((pb >= 180) & (tb <= 70))
    c_up = (tb >= 70) & (tb <= 290) & (pb >= tb + 110)
    c_lo = (tb >= 130) & (tb <= 180) & (pb <= (7.0 / 5.0) * tb - 182.0)
    C = (~A) & (~E) & (c_up | c_lo)
    D = (~A) & (~E) & (~C) & ((tb <= 70) | (tb >= 240)) & (pb >= 70) & (pb <= 180)
    B = (~A) & (~E) & (~C) & (~D)
    return A, B, C, D, E


def _excursion(pred_edge, true, thr, side):
    """Band-edge recall/precision for a threshold crossing (matches train.py's band-edge
    detectors). ``pred_edge`` is the τ-lower edge for hypo / τ-upper edge for hyper.
    RECALL is strict; PRECISION forgives a false alarm whose edge is within
    EXCURSION_PRECISION_TOLERANCE_MGDL of the true value."""
    if side == 'hypo':
        te, pe = true < thr, pred_edge < thr
    else:
        te, pe = true > thr, pred_edge > thr
    close_p = np.abs(pred_edge - true) <= EXCURSION_PRECISION_TOLERANCE_MGDL
    nt, npd = int(te.sum()), int(pe.sum())
    tp = int((te & pe).sum())
    recall = tp / nt if nt else None
    prec = float((pe & (te | close_p)).sum()) / npd if npd else None
    return {'recall': recall, 'precision': prec, 'n_true': nt, 'n_pred': npd}


def _rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e * e)))


def band_project(true: np.ndarray, band_lo: np.ndarray, band_hi: np.ndarray) -> np.ndarray:
    """Project the truth onto the forecast band — the effective point forecast.

    ``true`` / ``band_lo`` / ``band_hi`` are matching mg/dL arrays, canonically
    ``(n_windows, PRED_STEPS)``; returns the same shape. ``clip(true, lo, hi)`` is the
    band point nearest the truth: zero error where the truth lies inside the band, the
    distance to the nearer edge where it lies outside. A degenerate band (``lo == hi``)
    returns that common value, so the projection reduces to a point forecast exactly.
    """
    assert true.shape == band_lo.shape == band_hi.shape, (
        f"band_project shape mismatch: true {true.shape}, lo {band_lo.shape}, "
        f"hi {band_hi.shape}")
    assert bool(np.all(band_hi >= band_lo - _FAN_ORDER_TOL_MGDL)), \
        "band_project: band_hi must not fall below band_lo (ascending quantile fan)"
    return np.clip(true, band_lo, band_hi)


def _point_block(pred_k: np.ndarray, true_k: np.ndarray, err_winmean: np.ndarray,
                 pats: np.ndarray, rmse_persist_point: float) -> dict:
    """Per-horizon point-error block for ONE forecast basis (band-projected or median).

    ``pred_k`` / ``true_k`` are ``(n_windows,)`` mg/dL at the horizon step;
    ``err_winmean`` the ``(n_windows, k+1)`` error over steps 0..k on the same basis;
    ``pats`` the ``(n_windows,)`` patient ids. ``rmse_persist_point`` is the shared
    persistence baseline — persistence has no band, so it is computed once by the
    caller and fed to both bases; ``skill_point`` then differs only through the model
    side. Returns rmse_point, mae_point, rmse_winmean, mae_winmean, rmse_macro, mard,
    clarke_A, clarke_AB, clarke_D, clarke_E, skill_point.
    """
    assert pred_k.ndim == 1 and pred_k.shape == true_k.shape == pats.shape, (
        f"_point_block shape mismatch: pred {pred_k.shape}, true {true_k.shape}, "
        f"pats {pats.shape}")
    assert err_winmean.ndim == 2 and err_winmean.shape[0] == pred_k.shape[0], \
        f"_point_block window-mean error shape {err_winmean.shape}"
    e = pred_k - true_k
    A, B, C, D, E = _clarke(pred_k, true_k)
    macro = np.mean([_rmse(e[pats == p]) for p in np.unique(pats)])
    rmse_point = _rmse(e)
    return {
        'rmse_point': rmse_point,
        'mae_point': float(np.mean(np.abs(e))),
        'rmse_winmean': _rmse(err_winmean),
        'mae_winmean': float(np.mean(np.abs(err_winmean))),
        'rmse_macro': float(macro),
        'mard': float(100 * np.mean(np.abs(e) / np.clip(true_k, 1, None))),
        'clarke_A': float(100 * A.mean()),
        'clarke_AB': float(100 * (A | B).mean()),
        'clarke_D': float(100 * D.mean()),
        'clarke_E': float(100 * E.mean()),
        'skill_point': (float((rmse_persist_point - rmse_point) / rmse_persist_point)
                        if rmse_persist_point > 1e-9 else float('nan')),
    }


def compute_suite(pred: np.ndarray, true: np.ndarray, last_bg: np.ndarray,
                  patients: list[str], bands: np.ndarray | None = None) -> dict:
    """Full per-horizon metric suite from (n_windows, PRED_STEPS) BG arrays.

    ``pred`` is the median line ``median_bg``, ``true`` the CGM truth, ``last_bg``
    ``(n_windows,)`` the persistence anchor, ``bands`` the optional per-window mg/dL
    quantile fan ``(n_windows, PRED_STEPS, N_QUANTILES)`` (``forecast_bands``).

    With ``bands``: the headline per-horizon block and CG-EGA score the BAND-PROJECTED
    forecast ``band_project(true, q[METRIC_BAND_TAU_LO], q[METRIC_BAND_TAU_HI])``; the
    same block on the median line is nested under ``out[h]['median_line']``; each
    horizon also carries ``band_cov50`` (realized fraction of truth inside the band,
    target 0.50) and ``band_width`` (mean edge-to-edge width, mg/dL). hypo/hyper
    recall/precision key off the τ-lower / τ-upper ALARM band edges (matching train.py).

    With ``bands=None``: the pre-band behaviour exactly — every metric off the median
    ``pred``, no ``median_line`` / ``band_cov50`` / ``band_width`` keys, hypo/hyper off
    the median.

    Returns ``{h: {...} for h in HORIZONS} | {'cgega': {...}}``.
    """
    assert pred.ndim == 2 and pred.shape == true.shape, \
        f"compute_suite shape mismatch: pred {pred.shape}, true {true.shape}"
    assert last_bg.shape == (pred.shape[0],), \
        f"compute_suite last_bg shape {last_bg.shape}, expected {(pred.shape[0],)}"
    assert len(patients) == pred.shape[0], \
        f"compute_suite patients {len(patients)} vs {pred.shape[0]} windows"
    pats = np.asarray(patients)
    if bands is not None:
        assert bands.shape == pred.shape + (_N_QUANTILES,), \
            f"compute_suite bands shape {bands.shape}, expected {pred.shape + (_N_QUANTILES,)}"
        band_lo, band_hi = bands[..., _BAND_LO_IDX], bands[..., _BAND_HI_IDX]
        pred_eff = band_project(true, band_lo, band_hi)
    else:
        pred_eff = pred
    out: dict = {}
    for h in HORIZONS:
        k = _IDX[h]
        tk = true[:, k]
        # persistence (predict last_bg held flat) — no band, computed once per horizon
        rmse_persist_point = _rmse(last_bg - tk)
        row = _point_block(pred_eff[:, k], tk,
                           pred_eff[:, :k + 1] - true[:, :k + 1],   # window-mean basis
                           pats, rmse_persist_point)
        row['rmse_persist_point'] = rmse_persist_point
        row['rmse_persist_winmean'] = _rmse(last_bg[:, None] - true[:, :k + 1])
        if bands is not None:
            row['band_cov50'] = float(np.mean((tk >= band_lo[:, k]) & (tk <= band_hi[:, k])))
            row['band_width'] = float(np.mean(band_hi[:, k] - band_lo[:, k]))
            row['median_line'] = _point_block(pred[:, k], tk,
                                              pred[:, :k + 1] - true[:, :k + 1],
                                              pats, rmse_persist_point)
            pred_lo_k, pred_hi_k = bands[:, k, _HYPO_BAND_IDX], bands[:, k, _HYPER_BAND_IDX]
        else:
            pred_lo_k = pred_hi_k = pred[:, k]    # median fallback (band-less cohort)
        row['hypo'] = _excursion(pred_lo_k, tk, BG_HYPO_THRESHOLD, 'hypo')
        row['hyper'] = _excursion(pred_hi_k, tk, BG_HYPER_THRESHOLD, 'hyper')
        row['n_windows'] = len(tk)
        out[h] = row
    # CG-EGA clinical accuracy over the FULL forecast window (all PRED_STEPS, not a
    # single horizon) — per-region %AP/%BE/%EP, anchored at last_bg for the t=0 rate.
    # GRID_MIN = 5 min per step. The TRUTH is the reference on every axis (region,
    # acceptance band, mod widening, R-EGA abscissa); the band-projected forecast is
    # the scored series, and where the truth lies inside the band the two coincide.
    cg_counts = cg_ega.cg_ega_counts(true, pred_eff, last_bg, freq_min=5.0)
    cg_fr = cg_ega.cg_ega_fractions(cg_counts)
    out['cgega'] = {
        f'{m}_{reg}': (None if cg_fr[f'{m}_{reg}'] is None
                       else 100.0 * cg_fr[f'{m}_{reg}'])
        for reg in ('hypo', 'eu', 'hyper') for m in ('ap', 'be', 'ep')
    }
    out['cgega']['counts'] = cg_counts
    return out


def conformal_intervals(cal_pred: np.ndarray, cal_true: np.ndarray,
                        test_pred: np.ndarray, test_true: np.ndarray,
                        alpha: float = 0.1) -> dict:
    """Split-conformal constant-width 90% intervals: fit ±q on calibration
    residuals, evaluate coverage + width on test (per horizon).  Point-only —
    no MC sampling needed.

    All four arrays are ``(n_windows, PRED_STEPS)`` mg/dL. The BASIS is whatever the
    caller passes: hand it ``band_project(true, lo, hi)`` on both splits and the
    half-width reads as the extra width needed on top of the band to reach 90%; hand it
    the median line on both and it is the classic point interval. Both splits must use
    the SAME basis for the residuals to be exchangeable.
    """
    out = {}
    for h in HORIZONS:
        k = _IDX[h]
        scores = np.abs(cal_pred[:, k] - cal_true[:, k])
        nq = len(scores)
        if nq == 0 or len(test_pred) == 0:
            # A window-starved source (a single short segment) yields an
            # empty calibration or test split: np.quantile on an empty array (and the
            # nq+1 / nq term) would crash. Degrade gracefully to NaN rather than abort
            # the whole eval.
            out[h] = {'half_width': float('nan'), 'coverage': float('nan')}
            continue
        q = float(np.quantile(scores, min(1.0, np.ceil((nq + 1) * (1 - alpha)) / nq), method='higher'))
        cov = float(np.mean(np.abs(test_pred[:, k] - test_true[:, k]) <= q))
        out[h] = {'half_width': q, 'coverage': cov}
    return out
