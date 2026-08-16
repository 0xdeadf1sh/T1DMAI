"""
Per-dataset evaluation driver — produces the full comparison metric suite for a
real CGM dataset, with the comparison-audit reporting protocol baked in:

  * headline forecast = the model's quantile BAND projected onto the truth,
    ``clip(true, q[METRIC_BAND_TAU_LO], q[METRIC_BAND_TAU_HI])``; the median line
    ``median_bg`` is kept alongside under ``metrics[h]['median_line']`` as the
    point basis for peer/literature comparison. Without a band fan (legacy caches,
    or a split that captured none) every metric falls back to the median line
  * canonical split: OhioT1DM train/test XML; AZT1D/Shanghai temporal per-patient
  * metrics: strict-point + window-mean RMSE/MAE, MARD, Clarke A/A+B, hypo/hyper
    recall+precision (STRICT crossings), persistence skill (persistence carries no
    band), macro RMSE, realized band coverage/width, split-conformal 90% bands

``evaluate_from_windows`` is model-free (operates on collected Window forecasts),
so it is reused for the cached OhioT1DM windows in validation.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import timedelta

import numpy as np
import torch

from .schema import Segment, GRID_MIN, MIN_SEGMENT_STEPS, STEPS_PER_HOUR
from .calibrate import (
    collect_windows, calibrate_threshold, select_offset,
    threshold_curve, forecast_windows, forecast_bands, Window, CTX_STEPS, PRED_STEPS,
    _future_overrides,
)
from .features import build_feature_stack, context_window, smoothed_cgm
from .metrics import compute_suite, conformal_intervals, band_project
from .horizons import HORIZONS, HORIZON_IDX as _HORIZON_IDX, FIGURE_HORIZONS
import conformal
import mondrian
from config import (
    BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD,
    PATCH_SIZE, PREDICTION_PATCHES, PREDICTION_HORIZON_HOURS,
    NIGHT_LONG_HORIZON_HOURS, NOCTURNAL_START_HOUR, NOCTURNAL_END_HOUR,
    MAX_CONTEXT_PATCHES, CHANNEL_TO_FEAT, QUANTILE_LEVELS, N_QUANTILES,
    HYPO_ALARM_QUANTILE_TAU, HYPER_ALARM_QUANTILE_TAU,
    METRIC_BAND_TAU_LO, METRIC_BAND_TAU_HI,
)

_LO_IDX = QUANTILE_LEVELS.index(0.05)
_HI_IDX = QUANTILE_LEVELS.index(0.95)
_MED_IDX = QUANTILE_LEVELS.index(0.5)
_HYPO_TAU_IDX = QUANTILE_LEVELS.index(0.10)
_HYPO_BAND_IDX = QUANTILE_LEVELS.index(HYPO_ALARM_QUANTILE_TAU)   # clinical alarm τ-lower (config)
_HYPER_BAND_IDX = QUANTILE_LEVELS.index(HYPER_ALARM_QUANTILE_TAU)  # clinical alarm τ-upper (config)
_BAND_LO_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_LO)    # metric band τ-lower (config)
_BAND_HI_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_HI)    # metric band τ-upper (config)
from normalization import CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS
from utils import kovatchev_f_inv_np

# Medically-useful precision floor for the per-horizon hypo decision-offset
# selection. The risk-space redesign removed ``config.EXCURSION_DECISION_MIN_PRECISION``
# (realdata was out of scope for that cycle's config), so the deployment knob lives
# here as a module-level default — the highest-recall offset whose CAL precision
# clears this floor is selected (else the strict δ=0 crossing).
EXCURSION_DECISION_MIN_PRECISION = 0.7

_PRED_STEPS = PREDICTION_PATCHES * PATCH_SIZE

# --------------------------------------------------------------------------- #
# The forecast protocol, as a masked set
# --------------------------------------------------------------------------- #
# Every window this module scores runs one fixed protocol: a single masked span of
# ``PREDICTION_PATCHES`` patches ending at the window's LAST patch, with the whole
# context visible.  That is the FORECAST case of the masked-BG objective, not a
# mode of its own — a span starting at patch 0 is a backcast and a span between
# visible patches is infill, and neither is scored here.
#
# The masked set is expanded by ``data._mask_slots``, the single definition of the
# slot layout and of ``d``; nothing below restates the rule.  ``d`` is the distance
# in patches from a masked patch to the nearest visible evidence ON EITHER SIDE,
# and it is the only quantity a masked-BG metric may be binned on — never span
# length, which confounds one-sided and two-sided cases at equal difficulty, and
# never arm.  A right-edge span has no right neighbour, so d is ONE-SIDED here and
# slot j sits at d = j + 1: the reported 30 / 60 / 90 / 120 min horizons ARE
# d = 1..4 one-sided.
#
# The ANCHOR is a different quantity and is deliberately not d.  It is ONE-SIDED
# and LEFT-PREFERRING: every slot of a span takes the last step of the left
# neighbour, or the first step of the right neighbour only when the span starts at
# patch 0, and every slot of one span gets the same value.  It therefore IGNORES
# the near side for the right-half slots of a two-sided span.  For the right-edge
# span scored here there is no near side to ignore, but the anchor still sits a
# fixed 1 patch away while d runs 1..PREDICTION_PATCHES.
from data import _mask_slots as _expand_mask_slots

_FORECAST_SEQ_LEN = MAX_CONTEXT_PATCHES + PREDICTION_PATCHES
_fc_idx, _fc_valid, _fc_d, _fc_anchor_step = _expand_mask_slots(
    [(MAX_CONTEXT_PATCHES, PREDICTION_PATCHES)], _FORECAST_SEQ_LEN)
FORECAST_MASK_IDX = tuple(int(i) for i in _fc_idx[_fc_valid])
FORECAST_D_PATCHES = tuple(int(x) for x in _fc_d[_fc_valid])
assert FORECAST_D_PATCHES == tuple(range(1, PREDICTION_PATCHES + 1)), (
    f"right-edge forecast d {FORECAST_D_PATCHES} is not one-sided "
    f"1..{PREDICTION_PATCHES}")
assert int(_fc_anchor_step[_fc_valid][0]) == CTX_STEPS - 1, (
    "forecast anchor is not the last visible step")


def horizon_d_patches(h_min: int) -> int:
    """Distance-to-evidence bin ``d``, in patches, of a horizon of ``h_min`` minutes.

    A single forward pass masks ``PREDICTION_PATCHES`` patches at the right edge,
    so the step at ``h_min`` minutes falls in patch ``ceil(h_min / patch_minutes)``
    of that span and carries ``d`` = that patch number, one-sided.

    Past a single pass the forecast is ROLLED, and ``d`` does not keep growing: roll
    ``r`` re-runs the same right-edge span against a context whose tail is the
    previous roll's own output, so its slots are again ``d`` = 1..PREDICTION_PATCHES
    — but measured from FABRICATED evidence, not from an observed reading.  This
    returns the within-roll ``d`` and the roll index, so a table can say which.
    """
    patch_min = PATCH_SIZE * GRID_MIN
    patch_no = -(-h_min // patch_min)                     # ceil
    return FORECAST_D_PATCHES[(patch_no - 1) % PREDICTION_PATCHES]


def horizon_roll_index(h_min: int) -> int:
    """0-based roll a horizon of ``h_min`` minutes lands in (0 = the single pass)."""
    patch_min = PATCH_SIZE * GRID_MIN
    patch_no = -(-h_min // patch_min)
    return (patch_no - 1) // PREDICTION_PATCHES


def _slice(seg: Segment, a: int, b: int) -> Segment:
    """A sub-Segment over steps [a, b), with t0 advanced accordingly.

    Every length-N array on the Segment must be named here, the optional
    pre-resolved curves included: ``replace`` passes un-named fields through
    verbatim, so an omitted channel would leave a full-length array on a
    sub-Segment and silently misalign it against the sliced CGM.
    """
    return replace(
        seg, t0=seg.t0 + timedelta(minutes=GRID_MIN * a),
        cgm=seg.cgm[a:b], carb_grams=seg.carb_grams[a:b], bolus_units=seg.bolus_units[a:b],
        basal_rate=seg.basal_rate[a:b], exercise=seg.exercise[a:b],
        carb_curve=(None if seg.carb_curve is None else seg.carb_curve[a:b]),
        insulin_curve=(None if seg.insulin_curve is None else seg.insulin_curve[a:b]))


def split_segments(segs: list[Segment], dataset: str, cal_frac: float = 0.6):
    """Canonical OhioT1DM train/test split; INTRA-segment temporal split otherwise.

    For AZT1D/Shanghai (no canonical split, many single-segment patients) each
    segment is cut into an early calibration sub-segment and a late test
    sub-segment separated by a ``CTX_STEPS`` gap so the test window's context
    never overlaps the calibration span — every patient thus contributes both.
    """
    if dataset in ('ohiot1dm', 'ohio'):
        return ([s for s in segs if s.split == 'training'],
                [s for s in segs if s.split == 'testing'])
    cal, test = [], []
    need = CTX_STEPS + PRED_STEPS
    for s in segs:
        n = len(s)
        cut = int(cal_frac * n)
        if cut >= max(need, MIN_SEGMENT_STEPS):
            cal.append(_slice(s, 0, cut))
        ts = cut + CTX_STEPS
        if n - ts >= need:
            test.append(_slice(s, ts, n))
    return cal, test


def _quantile_cqr(cal_w: list[Window], test_w: list[Window]) -> dict | None:
    """Per-cohort quantile-CQR band recalibration, REGION-BINNED (Mondrian).

    Captures the model's RAW per-window mg/dL quantile fans (``Window.bands``) and
    fits the split-conformal correction on the CALIBRATION split ONCE PER REGION
    BIN (``mondrian.fit_mondrian``), the region being where that window's forecast
    is HEADING. The fit is automatically PER-COHORT (this runs inside
    ``evaluate_from_windows`` on one cohort's own cal/test windows). Returns
    ``None`` when either split lacks bands (old caches / a path that did not
    capture them), so the renderer can skip.

    THREE ARMS, ALL MEASURED IN THIS RUN so no figure is ever compared across
    runs (or across a clamp change): ``raw`` uncalibrated, ``marg`` the marginal
    pre-Mondrian fit that ``conformal.py`` alone gives — which is also the stated
    fallback for a bin under ``mondrian.MIN_N_OWN_FIT`` — and the region-binned
    arm, carried under the historical ``cal_*`` names since it is the correction
    in force.

    Every coverage figure carries its ``n``, its DISTINCT PATIENT count and its
    MEAN BAND WIDTH: coverage is bought with width, and n windows drawn from a
    handful of patients are not n independent observations.

    Per-horizon rows are per-``d`` rows — the right-edge forecast span puts the
    30/60/90/120 min horizons at one-sided ``d`` = 1..4 (``horizon_d_patches``).
    """
    cal_bands = forecast_bands(cal_w)
    test_bands = forecast_bands(test_w)
    if cal_bands is None or test_bands is None:
        return None
    _, cal_true, _, cal_pats = forecast_windows(cal_w)
    _, test_true, _, test_pats = forecast_windows(test_w)

    # The region reads the fan's own median line, which conformal holds FIXED — a
    # window's bin is therefore identical before and after correction and the
    # assignment is not circular.
    cal_bin = mondrian.region_bin(mondrian.forecast_destination(cal_bands, _MED_IDX))
    test_bin = mondrian.region_bin(mondrian.forecast_destination(test_bands, _MED_IDX))

    delta, marginal, meta = mondrian.fit_mondrian(
        cal_bands, cal_true, cal_bin, QUANTILE_LEVELS, _MED_IDX, patients=cal_pats)

    test_marg = conformal.apply_quantile_conformal(test_bands, marginal, _MED_IDX)
    test_mond = mondrian.apply_mondrian(test_bands, delta, test_bin, _MED_IDX)
    arms = {'raw': test_bands, 'marginal': test_marg, 'mondrian': test_mond}

    cov = {a: conformal.band_coverage(q, test_true, _LO_IDX, _HI_IDX)      # (S,)
           for a, q in arms.items()}
    # hypo escape = fraction of truth BELOW the τ=0.10 lower edge (target 0.10).
    esc = {a: (test_true < q[:, :, _HYPO_TAU_IDX]).mean(axis=0) for a, q in arms.items()}
    wid = {a: (q[:, :, _HI_IDX] - q[:, :, _LO_IDX]).mean(axis=0) for a, q in arms.items()}

    out: dict = {
        'delta': delta.tolist(),                 # (n_bins, S, K) — region-binned
        'delta_marginal': marginal.tolist(),     # (S, K) — the pre-Mondrian baseline
        'fit': meta,
        # Per bin AND per d: d is the only axis a masked-BG metric may be binned
        # on, and the right-edge forecast span puts patch p at one-sided d = p+1.
        'bins': mondrian.bin_report(
            arms, test_true, test_bin, _LO_IDX, _HI_IDX, patients=test_pats,
            step_groups=mondrian.forecast_d_step_groups(PREDICTION_PATCHES, PATCH_SIZE)),
    }
    n_cal_pat = len(set(cal_pats))
    n_test_pat = len(set(test_pats))
    for h in HORIZONS:
        k = _HORIZON_IDX[h]
        out[str(h)] = {
            'raw_cov90': float(cov['raw'][k]), 'cal_cov90': float(cov['mondrian'][k]),
            'marg_cov90': float(cov['marginal'][k]),
            'raw_hypo_escape': float(esc['raw'][k]),
            'cal_hypo_escape': float(esc['mondrian'][k]),
            'marg_hypo_escape': float(esc['marginal'][k]),
            'raw_width': float(wid['raw'][k]), 'cal_width': float(wid['mondrian'][k]),
            'marg_width': float(wid['marginal'][k]),
            'n_cal': int(cal_bands.shape[0]), 'n_test': int(test_bands.shape[0]),
            'n_cal_patients': n_cal_pat, 'n_test_patients': n_test_pat,
            'd_patches': horizon_d_patches(h), 'roll': horizon_roll_index(h),
        }
    return out


def evaluate_from_windows(cal_w: list[Window], test_w: list[Window]) -> dict:
    """Score the headline metric suite from pre-collected windows (no model).

    The headline forecast is the model's quantile BAND projected onto the truth
    (``realdata.metrics.band_project`` over the τ=``METRIC_BAND_TAU_LO`` /
    τ=``METRIC_BAND_TAU_HI`` edges of the fan captured at collection); the median line
    ``median_bg`` is scored alongside under ``metrics[h]['median_line']``. The suite,
    the split-conformal intervals and the excursion decision-offset sweep all read the
    SAME basis: a split whose windows carry no fan demotes every one of them to the
    median line, so the CAL-fit operating point and its TEST readout stay comparable.
    """
    test_pred, test_true, test_last, test_pats = forecast_windows(test_w)
    cal_pred, cal_true, _, _ = forecast_windows(cal_w)
    cal_bands = forecast_bands(cal_w)
    test_bands = forecast_bands(test_w)
    banded = cal_bands is not None and test_bands is not None

    suite = compute_suite(test_pred, test_true, test_last, test_pats,
                          bands=test_bands if banded else None)
    if banded:
        cal_eff = band_project(cal_true, cal_bands[..., _BAND_LO_IDX], cal_bands[..., _BAND_HI_IDX])
        test_eff = band_project(test_true, test_bands[..., _BAND_LO_IDX], test_bands[..., _BAND_HI_IDX])
    else:
        cal_eff, test_eff = cal_pred, test_pred
    conf = conformal_intervals(cal_eff, cal_true, test_eff, test_true)
    # Quantile-CQR band recalibration reads the RAW fan (not the projected basis):
    # additive, None when windows lack bands. Per-cohort by construction.
    conf_cqr = _quantile_cqr(cal_w, test_w)

    # --- Excursion decision-offset sweep ----------------------------------- #
    # The excursion decision runs on the SAME basis as the suite — the τ-lower band
    # edge (hypo) / τ-upper edge (hyper), else the median line — with a per-horizon
    # offset δ (``edge < thr + δ``) as the operating point, chosen from the
    # recall–precision curve. The curves are fit on the CAL split (STRICT crossings;
    # the offset δ is itself the band lever) so the operating point is picked from data.
    curves = calibrate_threshold(cal_pred, cal_true, bands=cal_bands if banded else None)
    threshold_curves = {side: {str(h): curves[side][h] for h in HORIZONS}
                        for side in ('hypo', 'hyper')}

    # Per-horizon hypo decision offset, selected on the CAL split under the
    # medically-useful precision floor (EXCURSION_DECISION_MIN_PRECISION): the
    # highest-recall δ whose precision clears the floor, else the strict δ=0
    # crossing (no low-precision alarm). This is the deployable operating point
    # the excursion decision should run at; it is fit on cal and reported so the
    # test-split recall/precision can be read at it.
    selected_offsets = {'min_precision': EXCURSION_DECISION_MIN_PRECISION, 'hypo': {}}
    for h in HORIZONS:
        kk = _HORIZON_IDX[h]   # step index for horizon h (30->5, 60->11, 120->23)
        off, cal_rec, cal_prec = select_offset(
            curves['hypo'][h], min_precision=EXCURSION_DECISION_MIN_PRECISION)
        test_edge = test_bands[:, kk, _BAND_LO_IDX] if banded else test_pred[:, kk]
        test_pt = threshold_curve(
            test_edge, test_true[:, kk], BG_HYPO_THRESHOLD, 'hypo',
            offsets=[off])[0]
        selected_offsets['hypo'][str(h)] = {
            'offset': off, 'cal_recall': cal_rec, 'cal_precision': cal_prec,
            'test_recall': test_pt['recall'], 'test_precision': test_pt['precision']}

    return {
        'n_cal_windows': len(cal_w), 'n_test_windows': len(test_w),
        'n_patients': len({w.patient for w in test_w}),
        # What every per-horizon row above is binned on: d, the distance in patches
        # to the nearest visible evidence on either side, one-sided for the
        # right-edge forecast span these windows run.  Recorded so a reader never
        # has to infer the bin from the horizon label.
        'horizon_d': {str(h): {'d_patches': horizon_d_patches(h),
                               'one_sided': True,
                               'roll': horizon_roll_index(h)}
                      for h in HORIZONS},
        'metrics': {str(h): suite[h] for h in HORIZONS},
        'cgega': suite['cgega'],
        'conformal': {str(h): conf[h] for h in HORIZONS},
        'conformal_cqr': conf_cqr,
        'threshold_curves': threshold_curves,
        'selected_offsets': selected_offsets,
    }


# --------------------------------------------------------------------------- #
# Night-onset nocturnal excursion prediction. Offline evaluation only — the
# training loop no longer carries a mirror of it, so this is the definition.
# --------------------------------------------------------------------------- #
def _night_len_hours() -> float:
    """Length of the nocturnal window (NOCTURNAL_START_HOUR → NOCTURNAL_END_HOUR),
    wrapping past midnight; a same-hour pair is read as a full 24 h night."""
    h = (NOCTURNAL_END_HOUR - NOCTURNAL_START_HOUR) % 24.0
    return 24.0 if h == 0.0 else h


def _denorm_channel(col_norm: np.ndarray, name: str, stats: dict) -> np.ndarray:
    """Inverse-normalize one named channel column (mirrors ``normalization.denormalize``
    for a single channel): z-score un-scale, then the per-channel inverse — Kovatchev
    ``f_inv`` for risk-space channels (bg), ``expm1``+clamp for log1p channels."""
    x = col_norm.astype(np.float64) * (stats[name]['std'] + 1e-8) + stats[name]['mean']
    if name in RISK_SPACE_CHANNELS:
        x = kovatchev_f_inv_np(x)
    elif name in SPARSE_LOG1P_CHANNELS:
        x = np.maximum(np.expm1(x), 0.0)
    return x


def _make_night_overrides_fn(feats: np.ndarray, pred_start: int,
                             announce: tuple[int, ...], stats: dict):
    """Per-roll announced carb(0)/insulin(1)/exercise(2) overrides for ``predict_rolling``
    across a whole night, sliced from the normalized feature stack of the segment.

    Roll ``r`` masks the same right-edge span of ``PREDICTION_PATCHES`` patches,
    advanced by one horizon: the announced window is the feature span
    ``[pred_start + r·PRED_STEPS, pred_start + (r+1)·PRED_STEPS)``; the announced
    channels are returned both normalized (sliced straight from ``feats``) and raw
    (denormalized) as ``{ch: (PREDICTION_PATCHES, PATCH_SIZE)}`` dicts, matching the
    ``overrides_fn`` contract. Returns ``None`` for a roll whose window runs past the
    segment, so the rollout falls back to the BG-autoregressive prediction there.
    """
    n = feats.shape[0]

    def fn(roll_idx: int, mu_np, abs_n_ctx: int):
        a = pred_start + roll_idx * _PRED_STEPS
        b = a + _PRED_STEPS
        if b > n:
            return None
        ov_norm: dict[int, np.ndarray] = {}
        ov_raw: dict[int, np.ndarray] = {}
        for ch in announce:
            fidx = CHANNEL_TO_FEAT[ch]
            col = feats[a:b, fidx].astype(np.float32)
            ov_norm[ch] = col.reshape(PREDICTION_PATCHES, PATCH_SIZE).copy()
            raw = _denorm_channel(col, CHANNEL_NAMES[fidx], stats).astype(np.float32)
            ov_raw[ch] = raw.reshape(PREDICTION_PATCHES, PATCH_SIZE).copy()
        return ov_norm, ov_raw
    return fn


def _night_onset_origins(hod: np.ndarray, n_steps: int, night_steps: int,
                         tol_hours: float = 0.75) -> list[int]:
    """Patch-aligned prediction-start indices whose hour-of-day (``hod``) is within
    ``tol_hours`` of ``NOCTURNAL_START_HOUR`` and that leave the full night
    (``night_steps``) ahead, with ``CTX_STEPS`` of context behind.

    ``n_steps`` is the patch-trimmed grid length. At most one origin per calendar
    night is kept (≥12 h apart), so successive 5-min grid points inside the tolerance
    band do not each spawn a near-duplicate night.
    """
    if n_steps < CTX_STEPS + night_steps:
        return []
    origins: list[int] = []
    last_kept = -10 ** 9
    min_gap = int(round(12.0 * STEPS_PER_HOUR))   # ≥12 h between kept night origins
    for ps in range(CTX_STEPS, n_steps - night_steps + 1, PATCH_SIZE):
        dist = abs(((hod[ps] - NOCTURNAL_START_HOUR + 12.0) % 24.0) - 12.0)
        if dist <= tol_hours and ps - last_kept >= min_gap:
            origins.append(ps)
            last_kept = ps
    return origins


def _score_night(model, feats: np.ndarray, cgm: np.ndarray, pred_start: int,
                 night_steps: int, n_rolls: int, stats: dict, device,
                 announce: tuple[int, ...]) -> tuple[bool, bool, bool, bool]:
    """Roll one night from ``pred_start`` to night-end and return the per-night binary
    excursion calls ``(hypo_true, hypo_pred, hyper_true, hyper_pred)``.

    ``hypo_true``/``hyper_true``: the TRUE CGM (``cgm``) crosses the threshold anywhere
    in the clipped night window. ``*_pred``: the rolled forecast crosses it likewise.
    The night's announced overnight carbs+insulin+exercise are fed to every roll via
    ``predict_rolling``'s ``overrides_fn`` (the model is always conditioned).
    """
    from inference import predict_rolling

    ctx = context_window(feats, pred_start, MAX_CONTEXT_PATCHES)
    overrides_fn = _make_night_overrides_fn(feats, pred_start, announce, stats)
    result = predict_rolling(
        model, ctx, patient_seed=None, n_rolls=n_rolls,
        normalization_stats=stats, device=device,
        overrides_fn=overrides_fn,
    )
    pred_bg = result['pred_bg'].detach().cpu().numpy()
    # Band-edge detectors, as every other clinical hypo/hyper metric: hypo off the τ-lower edge,
    # hyper off the τ-upper edge; truth off the TRUE CGM. bands: (rolls*P, S, K) -> (T, K).
    bands = result['bands'].detach().cpu().numpy().reshape(-1, N_QUANTILES)
    true_bg = cgm[pred_start:pred_start + night_steps].astype(np.float64)
    usable = min(pred_bg.shape[0], true_bg.shape[0], night_steps)
    tb = true_bg[:usable]
    pred_lo, pred_hi = bands[:usable, _HYPO_BAND_IDX], bands[:usable, _HYPER_BAND_IDX]
    return (bool((tb < BG_HYPO_THRESHOLD).any()), bool((pred_lo < BG_HYPO_THRESHOLD).any()),
            bool((tb > BG_HYPER_THRESHOLD).any()), bool((pred_hi > BG_HYPER_THRESHOLD).any()))


def _finalize_night_side(tr: int, pr: int, tp: int) -> dict:
    return {'recall': (tp / tr) if tr > 0 else None,
            'precision': (tp / pr) if pr > 0 else None,
            'n_true': tr, 'n_pred': pr}


def night_onset_from_records(model, stats, records, device,
                             announce: tuple[int, ...] = (0, 1, 2),
                             max_nights: int | None = None) -> dict:
    """Core per-night nocturnal-excursion scorer over generic records.

    Each record is ``(feats, cgm, hod)``: the normalized (N, F) feature stack, the
    raw (bg-clamped) comparison-truth CGM (N,) in mg/dL, and the fractional
    hour-of-day (N,). The real-data path (``evaluate_night_onset``) and the sim path
    both adapt their data to this shape (each clamping its truth before passing it).

    At each night-start origin (``hod`` ≈ ``NOCTURNAL_START_HOUR``) the forecast is
    autoregressively rolled (``inference.predict_rolling``) to night-end
    (``NIGHT_LONG_HORIZON_HOURS`` ≈ the nocturnal span) and a PER-NIGHT binary
    hypo/hyper call is emitted from the true CGM and from the rolled forecast (crossing
    the clinical threshold anywhere in the clipped night window). Recall = fraction of
    true-excursion nights flagged; precision = fraction of flagged nights that truly had
    one. The roll is always CONDITIONED (announced overnight carbs+insulin fed per roll
    via ``predict_rolling``'s ``overrides_fn``); there is no unconditioned regime.

    Returns::
        {'hypo': {recall, precision, n_true, n_pred},  'hyper': {...},
         'n_nights': int}

    Empty (``{}``) when the night fits in a single forward pass (n_rolls ≤ 1) or no
    night-onset window is available.
    """
    n_rolls = math.ceil(NIGHT_LONG_HORIZON_HOURS / PREDICTION_HORIZON_HOURS)
    if n_rolls <= 1:
        return {}

    night_steps = int(round(_night_len_hours() * STEPS_PER_HOUR))
    counts = {s: {'true': 0, 'pred': 0, 'tp': 0} for s in ('hypo', 'hyper')}
    n_nights = 0

    model.eval()
    with torch.no_grad():
        for feats, cgm, hod in records:
            n_steps = (len(cgm) // PATCH_SIZE) * PATCH_SIZE
            for ps in _night_onset_origins(hod, n_steps, night_steps):
                if max_nights is not None and n_nights >= max_nights:
                    break
                n_nights += 1
                ht, hp, yt, yp = _score_night(
                    model, feats, cgm, ps, night_steps, n_rolls, stats, device,
                    announce=announce)
                counts['hypo']['true'] += int(ht)
                counts['hypo']['pred'] += int(hp)
                counts['hypo']['tp'] += int(ht and hp)
                counts['hyper']['true'] += int(yt)
                counts['hyper']['pred'] += int(yp)
                counts['hyper']['tp'] += int(yt and yp)
            if max_nights is not None and n_nights >= max_nights:
                break

    if n_nights == 0:
        return {}
    out: dict = {'n_nights': n_nights}
    for s in ('hypo', 'hyper'):
        out[s] = _finalize_night_side(
            counts[s]['true'], counts[s]['pred'], counts[s]['tp'])
    return out


def evaluate_night_onset(model, stats, test_segs: list[Segment], device,
                         announce: tuple[int, ...] = (0, 1, 2),
                         max_nights: int | None = None) -> dict:
    """Per-night nocturnal excursion prediction on a dataset's test segments.

    Adapts each :class:`Segment` to a ``(feats, cgm, hod)`` record and delegates to
    :func:`night_onset_from_records`; see it for the per-night metric definition and
    the returned dict shape.
    """
    # Score against the raw (bg-clamped) CGM (the model lives in raw post-noise space).
    records = ((build_feature_stack(seg, stats), smoothed_cgm(seg.cgm), seg.hour_of_day())
               for seg in test_segs)
    return night_onset_from_records(model, stats, records, device,
                                    announce=announce, max_nights=max_nights)


# --------------------------------------------------------------------------- #
# Hour-by-hour RMSE-vs-horizon for the figures (rolled past the single forward
# pass). Figure-only: the reported metric suite keeps the canonical HORIZONS.
# --------------------------------------------------------------------------- #
def rmse_by_horizon_from_records(model, stats, records, device,
                                 horizons_min: tuple[int, ...] = FIGURE_HORIZONS,
                                 conditional: bool = True,
                                 announce: tuple[int, ...] = (0, 1, 2),
                                 stride_patches: int = 8,
                                 max_windows: int = 200) -> dict:
    """Per-horizon point and window-mean RMSE — model and naive persistence — from
    a forecast ROLLED out to the largest requested horizon, for the hour-by-hour
    ``rmse_vs_horizon`` figure.

    Each record is ``(feats, cgm)``: the normalized (N, F) feature stack and the
    raw (bg-clamped) comparison-truth CGM (N,) in mg/dL (the real path wraps
    :class:`Segment`s, the sim path its run dicts; each clamps its truth before
    passing it). Squared errors are accumulated per 5-min step across windows (a window
    near a segment end contributes only to the steps it reaches); point RMSE reads
    the step at the horizon, window-mean RMSE pools steps 0..horizon. Persistence is
    the last context BG held flat. Each window's future carbs/insulin/exercise are
    announced per roll (the model is always conditioned; ``conditional`` is a deprecated
    no-op), matching the report regime.

    Two forecast bases are accumulated over the same windows, matching the suite: the
    BAND-projected forecast ``band_project(true, q[METRIC_BAND_TAU_LO],
    q[METRIC_BAND_TAU_HI])`` (``rmse_point`` / ``rmse_winmean``) and the median line
    ``f_inv(median)`` (``rmse_point_median`` / ``rmse_winmean_median``). Persistence
    carries no band and is accumulated once.

    Returns ``{str(h_min): {rmse_point, rmse_winmean, rmse_point_median,
    rmse_winmean_median, rmse_persist_point, rmse_persist_winmean, n}}`` for every
    horizon the data reach.
    """
    from inference import predict, predict_rolling

    hmax = max(horizons_min)
    n_rolls = max(1, math.ceil(hmax / 60.0 / PREDICTION_HORIZON_HOURS))
    H = n_rolls * _PRED_STEPS
    se = np.zeros(H); se_m = np.zeros(H); se_p = np.zeros(H); cnt = np.zeros(H)
    stride = stride_patches * PATCH_SIZE
    nwin = 0

    model.eval()
    with torch.no_grad():
        for feats, cgm in records:
            if nwin >= max_windows:
                break
            cgm = np.asarray(cgm, dtype=np.float64)
            ntot = len(cgm)
            n = (ntot // PATCH_SIZE) * PATCH_SIZE
            if n < CTX_STEPS + _PRED_STEPS:
                continue
            for ps in range(CTX_STEPS, n - _PRED_STEPS + 1, stride):
                if nwin >= max_windows:
                    break
                ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)
                if n_rolls == 1:
                    ov = _future_overrides(feats, ps, announce)
                    out = predict(model, ctx, normalization_stats=stats,
                                  overrides=ov, device=device)
                    pred_t = out['median_bg']
                else:
                    fn = _make_night_overrides_fn(feats, ps, announce, stats)
                    out = predict_rolling(model, ctx, n_rolls=n_rolls,
                                          normalization_stats=stats,
                                          overrides_fn=fn, device=device)
                    pred_t = out['pred_bg']
                pred = pred_t.detach().cpu().numpy()
                # (P, S, K) single-pass / (rolls*P, S, K) rolled -> per-step fan (T, K).
                fan = out['bands'].detach().cpu().numpy().reshape(-1, N_QUANTILES)
                assert fan.shape[0] == pred.shape[0], \
                    f"band fan {fan.shape} vs forecast {pred.shape}"
                m = min(H, len(pred), ntot - ps)
                if m <= 0:
                    continue
                true = cgm[ps:ps + m]
                pred_eff = band_project(true, fan[:m, _BAND_LO_IDX], fan[:m, _BAND_HI_IDX])
                d = pred_eff - true
                dm = pred[:m] - true               # median line, the peer-comparable basis
                dp = cgm[ps - 1] - true            # persistence: last context BG, flat
                se[:m] += d * d; se_m[:m] += dm * dm; se_p[:m] += dp * dp; cnt[:m] += 1
                nwin += 1

    out: dict = {}
    for h in horizons_min:
        k = h // GRID_MIN - 1
        if k >= H or cnt[k] == 0:
            continue
        msk = cnt[:k + 1] > 0
        pooled = cnt[:k + 1][msk].sum()
        out[str(h)] = {
            'rmse_point': math.sqrt(se[k] / cnt[k]),
            'rmse_winmean': math.sqrt(se[:k + 1][msk].sum() / pooled),
            'rmse_point_median': math.sqrt(se_m[k] / cnt[k]),
            'rmse_winmean_median': math.sqrt(se_m[:k + 1][msk].sum() / pooled),
            'rmse_persist_point': math.sqrt(se_p[k] / cnt[k]),
            'rmse_persist_winmean': math.sqrt(se_p[:k + 1][msk].sum() / pooled),
            'n': int(cnt[k]),
            # The bin, carried beside the number: d is the distance in patches to
            # the nearest visible evidence, one-sided for a right-edge span.  Past
            # roll 0 that evidence is the previous roll's own output, so d restarts
            # at 1 and is measured from a fabricated reading — ``roll`` says which.
            'd_patches': horizon_d_patches(h),
            'one_sided': True,
            'roll': horizon_roll_index(h),
        }
    return out


def rmse_by_horizon_rolling(model, stats, test_segs: list[Segment], device,
                            horizons_min: tuple[int, ...] = FIGURE_HORIZONS,
                            conditional: bool = True,
                            announce: tuple[int, ...] = (0, 1, 2),
                            stride_patches: int = 8,
                            max_windows: int = 200) -> dict:
    """Hour-by-hour RMSE-vs-horizon over a dataset's test segments; see
    :func:`rmse_by_horizon_from_records` for the metric definition and dict shape."""
    # Score against the raw (bg-clamped) CGM (the model lives in raw post-noise space).
    records = ((build_feature_stack(seg, stats), smoothed_cgm(seg.cgm)) for seg in test_segs)
    return rmse_by_horizon_from_records(
        model, stats, records, device, horizons_min, conditional=conditional,
        announce=announce, stride_patches=stride_patches, max_windows=max_windows)


def evaluate_dataset(name: str, model, stats, device, cal_stride: int = 8,
                     cal_cap: int = 24, test_stride: int = 6, test_cap: int = 60,
                     conditional: bool = True, announce: tuple[int, ...] = (0, 1, 2),
                     augment_fn=None) -> dict:
    """Full pipeline for one dataset: load -> (augment) -> split -> collect -> evaluate.

    The model is ALWAYS conditioned: each window's true future carbs/insulin/exercise
    are announced (the deployment what-if regime); there is no unconditioned companion
    suite. ``conditional`` is retained only for call-compatibility and no longer
    toggles anything. The exercise column is identically zero on every real cohort,
    so its announcement declares "no session" rather than adding information.

    Window counts are capped (``cal_cap``/``test_cap`` per patient) so a large
    cohort like AZT1D stays tractable; a few hundred windows give stable metrics.

    Args:
        conditional: deprecated no-op (the forecast is always conditioned).
        announce: output-channel indices announced; see ``collect_windows``.
        augment_fn: optional ``Segment -> Segment`` map applied right after load
            (e.g. injecting reconstructed meal/bolus events into the raw record);
            ``None`` evaluates the unmodified record.
    """
    from . import load_dataset
    segs = load_dataset(name)
    if augment_fn is not None:
        segs = [augment_fn(s) for s in segs]
    cal_segs, test_segs = split_segments(segs, name)

    cal_w = collect_windows(model, stats, cal_segs, device, stride_patches=cal_stride,
                            max_per_patient=cal_cap, announce=announce)
    test_w = collect_windows(model, stats, test_segs, device, stride_patches=test_stride,
                             max_per_patient=test_cap, announce=announce)
    res = evaluate_from_windows(cal_w, test_w)
    res['dataset'] = name
    # Night-onset nocturnal excursion prediction (conditioned roll) on the same test
    # segments — additive, never alters the suite.
    res['night_onset'] = evaluate_night_onset(model, stats, test_segs, device, announce=announce)
    # Hour-by-hour RMSE-vs-horizon (rolled) for the rmse_vs_horizon figure —
    # figure-only, never alters the suite; matches the report's announce regime.
    res['rmse_by_hour'] = rmse_by_horizon_rolling(
        model, stats, test_segs, device, announce=announce)
    return res


def _print(res: dict):
    m = res['metrics']
    banded = 'median_line' in m[str(HORIZONS[0])]
    print(f"\n{res.get('dataset','?')}: {res['n_test_windows']} test windows, "
          f"{res['n_patients']} patients")
    print("level-metric basis: "
          + ("band-projected forecast (median line under metrics[h]['median_line'])"
             if banded else "median line (no band fan on one of the splits)"))
    print("binned on d, the distance in patches to the nearest visible evidence "
          "(one-sided; the anchor is left-preferring and reads a different "
          "distance): "
          + "  ".join(f"{h}m=d{horizon_d_patches(h)}" for h in HORIZONS))
    print(f"{'horizon':>7} | {'RMSE pt':>7} {'RMSE wm':>7} | {'persist':>7} {'skill%':>6} | "
          f"{'MARD':>5} {'ClkA':>5} {'ClkA+B':>6} | {'hypoRec':>7} {'hyperRec':>8} | {'conf±':>6} {'cov%':>5}")
    for h in HORIZONS:
        d = m[str(h)]; c = res['conformal'][str(h)]
        hr = d['hypo']['recall']; yr = d['hyper']['recall']
        print(f"{h:>5}m | {d['rmse_point']:7.1f} {d['rmse_winmean']:7.1f} | "
              f"{d['rmse_persist_point']:7.1f} {100*d['skill_point']:6.1f} | "
              f"{d['mard']:5.1f} {d['clarke_A']:5.1f} {d['clarke_AB']:6.1f} | "
              f"{('%.2f'%hr if hr is not None else '  n/a'):>7} "
              f"{('%.2f'%yr if yr is not None else '  n/a'):>8} | "
              f"{c['half_width']:6.1f} {100*c['coverage']:5.0f}")
    cg = res.get('cgega')
    if cg is not None:
        def _f(v):
            return '  n/a' if v is None else f'{v:5.1f}'
        print("CG-EGA (Kovatchev 2004) %AP/%EP: "
              f"hypo {_f(cg['ap_hypo'])}/{_f(cg['ep_hypo'])}  "
              f"eu {_f(cg['ap_eu'])}/{_f(cg['ep_eu'])}  "
              f"hyper {_f(cg['ap_hyper'])}/{_f(cg['ep_hyper'])}")
    so = res.get('selected_offsets')
    if so and so.get('hypo'):
        print(f"hypo decision offset (cal precision floor {so['min_precision']:.2f}):")
        for h in HORIZONS:
            d = so['hypo'].get(str(h))
            if d is None:
                continue
            def _p(v):
                return ' n/a' if v is None else f'{v:.2f}'
            print(f"  {h:>3}m  offset {d['offset']:5.1f}  test rec/prec "
                  f"{_p(d['test_recall'])}/{_p(d['test_precision'])}")
    cq = res.get('conformal_cqr')
    if cq:
        fit = cq.get('fit') or {}
        print("quantile-CQR band coverage (cohort re-fit), region-binned on where the "
              f"forecast is heading; edges {fit.get('region_edges')} mg/dL, "
              f"marginal fallback below n={fit.get('min_n_own_fit')}:")
        print(f"  {'d':>2} {'horizon':>7} | {'cov90 raw':>9} {'marg':>6} {'binned':>7} | "
              f"{'width raw':>9} {'marg':>6} {'binned':>7} | "
              f"{'hypo-esc raw':>12} {'marg':>6} {'binned':>7}")
        for h in HORIZONS:
            d = cq.get(str(h))
            if d is None:
                continue
            print(f"  {d['d_patches']:>2} {h:>6}m | {100*d['raw_cov90']:8.0f}% "
                  f"{100*d['marg_cov90']:5.0f}% {100*d['cal_cov90']:6.0f}% | "
                  f"{d['raw_width']:9.1f} {d['marg_width']:6.1f} {d['cal_width']:7.1f} | "
                  f"{100*d['raw_hypo_escape']:11.0f}% "
                  f"{100*d['marg_hypo_escape']:5.0f}% {100*d['cal_hypo_escape']:6.0f}%")
        h0 = cq.get(str(HORIZONS[0])) or {}
        print(f"  n_cal {h0.get('n_cal')} ({h0.get('n_cal_patients')} patients), "
              f"n_test {h0.get('n_test')} ({h0.get('n_test_patients')} patients)")
        for rec in fit.get('bins', []):
            print(f"  calibration region {rec['label']:>12} n={rec['n']:<5} "
                  f"patients={rec['n_patients']}  "
                  + ('own fit' if rec['own_fit'] else f"MARGINAL: {rec['fallback_reason']}"))
        if cq.get('bins'):
            mondrian.print_bin_report(cq['bins'], 0.90,
                                      "  test-split coverage per region bin")


if __name__ == '__main__':
    import sys
    import torch
    cache = torch.load('scratch/_ohio_windows.pt', weights_only=False)
    # Risk-space forecast: Windows must carry the headline ``pred_bg`` (median_bg)
    # field. An old cache predates the redesign (carried dynamics/trend fields) —
    # skip rather than crash, the cached path is only a convenience.
    if not all(hasattr(w, 'pred_bg') for w in cache['cal_w'] + cache['test_w']):
        print("cached windows predate the risk-space redesign (no 'pred_bg' field) — "
              "re-collect with evaluate_dataset; skipping.", file=sys.stderr)
        sys.exit(0)
    res = evaluate_from_windows(cache['cal_w'], cache['test_w'])
    res['dataset'] = 'ohiot1dm (cached)'
    _print(res)
