"""Evaluation driver: the comparison metric suite over collected windows, reporting protocol included.

  * headline forecast = the band projected onto the truth,
    ``clip(true, q[METRIC_BAND_TAU_LO], q[METRIC_BAND_TAU_HI])``; the median line stays under
    ``metrics[h]['median_line']`` as the point basis peers publish. No fan ⇒ every metric is median-line
  * strict-point and window-mean RMSE/MAE, MARD, Clarke A/A+B, hypo/hyper recall and precision on STRICT
    crossings, persistence skill (persistence has no band, so its rows are median-line either way)
  * split-conformal band recalibration, region-binned, fit on cal and reported on test

``evaluate_from_windows`` is the entry point; ``metrics/sim/`` is the collector.
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
from .suite import compute_suite, conformal_intervals, band_project
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

# Precision floor for the per-horizon hypo decision offset: the highest-recall offset whose CAL precision
# clears it wins, else the strict δ=0 crossing.
EXCURSION_DECISION_MIN_PRECISION = 0.7

_PRED_STEPS = PREDICTION_PATCHES * PATCH_SIZE

# The forecast protocol as a masked set: one span of ``PREDICTION_PATCHES`` at the window's LAST patch,
# whole context visible. Backcast and infill are the same objective and are not scored here.
# ``data._mask_slots`` is the single definition of the slot layout and of ``d``, the distance in patches to
# the nearest visible evidence ON EITHER SIDE — the only axis a masked-BG metric may bin on, never span
# length. A right-edge span has no right neighbour, so slot j sits at d = j + 1: 30/60/90/120 min ARE d = 1..4.
# The ANCHOR is NOT d: one-sided and LEFT-PREFERRING, the left neighbour's last step (the right neighbour's
# first only at patch 0), one value per span. Here it stays a fixed 1 patch away while d runs 1..PREDICTION_PATCHES.
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
    """The distance-to-evidence bin ``d``, in patches, for a horizon of ``h_min`` minutes.

    One pass masks ``PREDICTION_PATCHES`` at the right edge, so the step lands in patch
    ``ceil(h_min / patch_minutes)`` and carries that ``d``, one-sided.
    Past one pass ``d`` does NOT keep growing: each roll re-runs the same span against the previous roll's
    own output, so d = 1..PREDICTION_PATCHES again, measured from FABRICATED evidence.
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
    """A sub-Segment over steps ``[a, b)``, ``t0`` advanced to match.

    Every length-N array must be named, pre-resolved curves included: ``replace`` passes an un-named field
    through at FULL length, misaligned against the sliced CGM.
    """
    return replace(
        seg, t0=seg.t0 + timedelta(minutes=GRID_MIN * a),
        cgm=seg.cgm[a:b], carb_grams=seg.carb_grams[a:b], bolus_units=seg.bolus_units[a:b],
        basal_rate=seg.basal_rate[a:b], exercise=seg.exercise[a:b],
        carb_curve=(None if seg.carb_curve is None else seg.carb_curve[a:b]),
        insulin_curve=(None if seg.insulin_curve is None else seg.insulin_curve[a:b]))




def _quantile_cqr(cal_w: list[Window], test_w: list[Window]) -> dict | None:
    """Quantile-CQR band recalibration, REGION-BINNED (Mondrian), fit on the CAL split of this run.

    The RAW per-window fans (``Window.bands``) are corrected once per region bin, the region being where
    that window's forecast is HEADING. ``None`` when either split lacks bands, so the renderer can skip.
    THREE ARMS, all measured in this run so nothing is compared across runs: ``raw``; ``marg``, the marginal
    ``conformal.py`` fit, which is also the fallback for a bin under ``mondrian.MIN_N_OWN_FIT``; and the
    region-binned arm, under the ``cal_*`` names, being the correction in force.
    Every coverage figure carries its ``n``, its DISTINCT PATIENT count and its MEAN BAND WIDTH: coverage is
    bought with width, and n windows from a handful of patients are not n independent observations.
    Per-horizon rows are per-``d`` rows.
    """
    cal_bands = forecast_bands(cal_w)
    test_bands = forecast_bands(test_w)
    if cal_bands is None or test_bands is None:
        return None
    _, cal_true, _, cal_pats = forecast_windows(cal_w)
    _, test_true, _, test_pats = forecast_windows(test_w)

    # the region reads the median line, which conformal holds FIXED, so a window's bin is the same before
    # and after correction and the assignment is not circular
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
        # per bin AND per d, the only axis a masked-BG metric may bin on; a right-edge span puts patch p at d = p+1
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
    """Score the headline suite from pre-collected windows; no model needed.

    The headline forecast is the band projected onto the truth over the τ=``METRIC_BAND_TAU_LO`` /
    ``METRIC_BAND_TAU_HI`` edges; the median line is scored alongside under ``metrics[h]['median_line']``.
    Suite, conformal intervals and the decision-offset sweep read ONE basis: a split with no fan demotes all
    three to the median line, so the CAL-fit operating point and its TEST readout stay comparable.
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
    # reads the RAW fan, not the projected basis; None when windows lack bands
    conf_cqr = _quantile_cqr(cal_w, test_w)

    # Same basis as the suite — τ-lower edge for hypo, τ-upper for hyper, else the median line — with a
    # per-horizon offset δ (``edge < thr + δ``) as the operating point, fit on the CAL split.
    curves = calibrate_threshold(cal_pred, cal_true, bands=cal_bands if banded else None)
    threshold_curves = {side: {str(h): curves[side][h] for h in HORIZONS}
                        for side in ('hypo', 'hyper')}

    # highest-recall δ whose CAL precision clears EXCURSION_DECISION_MIN_PRECISION, else the strict δ=0
    # crossing — no low-precision alarm. Reported so the test-split rates can be read at it.
    selected_offsets = {'min_precision': EXCURSION_DECISION_MIN_PRECISION, 'hypo': {}}
    for h in HORIZONS:
        kk = _HORIZON_IDX[h]   # step index for horizon h
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
        # what every per-horizon row is binned on, recorded so no reader infers it from the horizon label
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


# Night-onset nocturnal excursion prediction. Offline only, and the definition: the training loop has no mirror.
def _night_len_hours() -> float:
    """Nocturnal window length, wrapping past midnight; a same-hour pair reads as a full 24 h night."""
    h = (NOCTURNAL_END_HOUR - NOCTURNAL_START_HOUR) % 24.0
    return 24.0 if h == 0.0 else h


def _denorm_channel(col_norm: np.ndarray, name: str, stats: dict) -> np.ndarray:
    """One named channel column inverse-normalized: z un-scale, then ``f_inv`` (risk space) or ``expm1``+clamp (log1p).

    Mirrors ``normalization.denormalize`` for a single channel.
    """
    x = col_norm.astype(np.float64) * (stats[name]['std'] + 1e-8) + stats[name]['mean']
    if name in RISK_SPACE_CHANNELS:
        x = kovatchev_f_inv_np(x)
    elif name in SPARSE_LOG1P_CHANNELS:
        x = np.maximum(np.expm1(x), 0.0)
    return x


def _make_night_overrides_fn(feats: np.ndarray, pred_start: int,
                             announce: tuple[int, ...], stats: dict):
    """Per-roll announced carb(0)/insulin(1)/exercise(2) overrides for ``predict_rolling``, across one night.

    Roll ``r`` masks the same right-edge span advanced by one horizon: the feature span
    ``[pred_start + r·PRED_STEPS, pred_start + (r+1)·PRED_STEPS)``, returned both normalized and raw as
    ``{ch: (PREDICTION_PATCHES, PATCH_SIZE)}``. ``None`` for a roll past the segment end, which leaves that
    roll BG-autoregressive.
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
    """Patch-aligned origins within ``tol_hours`` of ``NOCTURNAL_START_HOUR``, with the full night ahead.

    ``n_steps`` is the patch-trimmed grid length; ``CTX_STEPS`` of context must sit behind.
    One origin per night at most, ≥12 h apart, so neighbouring grid points inside the tolerance band do not
    each spawn a near-duplicate night.
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
    """One night rolled to night-end -> ``(hypo_true, hypo_pred, hyper_true, hyper_pred)``.

    A call is True when the series crosses the threshold ANYWHERE in the clipped night: truth off the TRUE
    CGM, prediction off the rolled forecast. Every roll is fed the night's announced carb, insulin and exercise.
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
    # band-edge detectors, as everywhere else: hypo off the τ-lower edge, hyper off the τ-upper.
    # bands: (rolls*P, S, K) -> (T, K)
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
    """Per-night nocturnal-excursion scorer over generic records -> ``{'hypo': {...}, 'hyper': {...}, 'n_nights'}``.

    A record is ``(feats, cgm, hod)``: the normalized ``(N, F)`` stack, the raw bg-clamped truth CGM ``(N,)``
    in mg/dL, and the fractional hour-of-day ``(N,)``. Every path adapts to this shape.
    From each night-start origin the forecast is rolled to night-end and one binary hypo/hyper call comes off
    the truth and off the forecast — crossing the threshold anywhere in the clipped night. Recall is the
    fraction of true-excursion nights flagged, precision the fraction of flagged nights that had one.
    Always conditioned: the overnight doses are announced per roll.
    ``{}`` when the night fits one forward pass (n_rolls ≤ 1) or no night-onset window exists.
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

    Each :class:`Segment` becomes a ``(feats, cgm, hod)`` record for :func:`night_onset_from_records`.
    """
    # scored against the raw bg-clamped CGM
    records = ((build_feature_stack(seg, stats), smoothed_cgm(seg.cgm), seg.hour_of_day())
               for seg in test_segs)
    return night_onset_from_records(model, stats, records, device,
                                    announce=announce, max_nights=max_nights)


# Hour-by-hour RMSE-vs-horizon, rolled past the single forward pass. Figure-only: the suite keeps HORIZONS.
def rmse_by_horizon_from_records(model, stats, records, device,
                                 horizons_min: tuple[int, ...] = FIGURE_HORIZONS,
                                 conditional: bool = True,
                                 announce: tuple[int, ...] = (0, 1, 2),
                                 stride_patches: int = 8,
                                 max_windows: int = 200) -> dict:
    """Per-horizon point and window-mean RMSE, model and naive persistence, off a forecast ROLLED to ``hmax``.

    A record is ``(feats, cgm)``: the normalized ``(N, F)`` stack and the raw bg-clamped truth ``(N,)`` mg/dL.
    Squared errors accumulate per 5-min step across windows — a window near a segment end contributes only
    the steps it reaches. Point RMSE reads the horizon step, window-mean pools steps 0..horizon; persistence
    is the last context BG held flat. Always conditioned; ``conditional`` is a no-op.
    Two bases over the same windows, matching the suite: BAND-projected (``rmse_point`` / ``rmse_winmean``)
    and the median line (``rmse_point_median`` / ``rmse_winmean_median``). Persistence has no band and is
    accumulated once.
    -> ``{str(h_min): {rmse_point, rmse_winmean, rmse_point_median, rmse_winmean_median,
    rmse_persist_point, rmse_persist_winmean, n}}`` for every horizon the data reach.
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
            # the bin beside the number: past roll 0 the evidence is the previous roll's own output,
            # so d restarts at 1 from a fabricated reading and ``roll`` says which
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
    """Hour-by-hour RMSE-vs-horizon over a dataset's test segments; defined in :func:`rmse_by_horizon_from_records`."""
    # scored against the raw bg-clamped CGM
    records = ((build_feature_stack(seg, stats), smoothed_cgm(seg.cgm)) for seg in test_segs)
    return rmse_by_horizon_from_records(
        model, stats, records, device, horizons_min, conditional=conditional,
        announce=announce, stride_patches=stride_patches, max_windows=max_windows)




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
        print("quantile-CQR band coverage (re-fit), region-binned on where the "
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


