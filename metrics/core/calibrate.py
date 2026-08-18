"""
Window collection + excursion-decision calibration on the model-input bridge.

The risk-space redesign makes the HEADLINE BG forecast the model's quantile
median, ``median_bg = f_inv(median)`` from ``inference.predict`` — there is no
trend head, no physics reconstruction, and no per-patient ``icr`` / ``bg_scale``
to fit (those depended on the deleted ``utils.reconstruct_bg``).  The deployment
trend-gain machinery (``calibrate_gain`` / ``g_down`` / ``g_up``) is likewise
gone: the model emits a calibrated quantile forecast directly.

What survives is the per-patient EXCURSION-DECISION calibration, which only ever
needed plain ``(pred_bg, true_bg)`` arrays:

    collect_windows(model, stats, segments)  -> list[Window]  (median_bg + true CGM)
    forecast_windows(windows)                -> (pred (N,T), true (N,T), last_bg (N,), patients)
    calibrate_threshold(pred, true, bands=…) -> per-horizon recall–precision curves
    select_offset(curve, …)                  -> one operating point under a precision floor

The decision sweep runs off the metric BAND EDGES (τ=``METRIC_BAND_TAU_LO`` for hypo,
τ=``METRIC_BAND_TAU_HI`` for hyper) when the quantile fan is supplied, matching the
band basis of the level metrics; without a fan it runs off the median line as before.

``forecast_windows`` returns the raw per-window predicted/true BG so the metrics
harness can compute MARD / Clarke / hypo-hyper without re-running the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import (
    MAX_CONTEXT_PATCHES, PATCH_SIZE, PREDICTION_PATCHES,
    BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD, CHANNEL_TO_FEAT, N_QUANTILES,
    QUANTILE_LEVELS, METRIC_BAND_TAU_LO, METRIC_BAND_TAU_HI,
)
from inference import predict
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

from .features import build_feature_stack, context_window, smoothed_cgm
from .schema import Segment
from .horizons import HORIZONS, HORIZON_IDX as _HORIZON_IDX

CTX_STEPS = MAX_CONTEXT_PATCHES * PATCH_SIZE
PRED_STEPS = PREDICTION_PATCHES * PATCH_SIZE
# Decision-offset sweep for ``calibrate_threshold``: δ ∈ [0, 60] mg/dL, 2.5 steps.
# A wide, fine grid so the user reads the whole recall–precision curve; no floor
# is baked in (PLAN §6/§7).
_OFFSET_GRID = tuple(float(x) for x in np.arange(0.0, 60.0 + 1e-9, 2.5))

# Band the decision sweep reads when a quantile fan is supplied — the same knob the
# level metrics score against (``metrics.core.suite``), distinct from the clinical
# alarm taus.
_BAND_LO_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_LO)
_BAND_HI_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_HI)


def threshold_curve(pred: np.ndarray, true: np.ndarray, thr: float, side: str,
                    offsets=_OFFSET_GRID) -> list[dict]:
    """Recall–precision curve over a decision-offset sweep at ONE horizon point.

    For each δ in ``offsets`` the alarm fires when ``pred < thr + δ`` (hypo) or
    ``pred > thr − δ`` (hyper) — a positive δ widens the band toward higher recall.
    Truth and decision are STRICT clinical crossings (no tolerance band); the band
    is exactly the lever the user sets the operating point with.

    Args:
        pred: ``(N,)`` forecast BG at the horizon point — the median line or the
            band edge the caller decides on.
        true: ``(N,)`` true BG at the horizon point.
        thr:  clinical threshold (``BG_HYPO_THRESHOLD`` / ``BG_HYPER_THRESHOLD``).
        side: ``'hypo'`` | ``'hyper'``.
        offsets: iterable of δ in mg/dL.

    Returns:
        list of ``{offset, recall, precision, n_true, n_pred}`` (``n_true`` is
        constant across the sweep; ``n_pred`` grows with δ; recall/precision are
        ``None`` when their denominator is 0).
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    te = true < thr if side == 'hypo' else true > thr
    nt = int(te.sum())
    curve = []
    for d in offsets:
        pe = pred < thr + d if side == 'hypo' else pred > thr - d
        npd = int(pe.sum())
        recall = float((te & pe).sum()) / nt if nt else None
        prec = float((pe & te).sum()) / npd if npd else None
        curve.append({'offset': float(d), 'recall': recall, 'precision': prec,
                      'n_true': nt, 'n_pred': npd})
    return curve


def calibrate_threshold(pred: np.ndarray, true: np.ndarray,
                        offsets=_OFFSET_GRID,
                        bands: np.ndarray | None = None) -> dict:
    """Per-horizon hypo & hyper recall–precision curves over a decision-offset sweep.

    Fit on the CALIBRATION split. Emits the full curve per horizon so the operating
    point is chosen from data; bakes in NO precision floor or target recall
    (PLAN §6/§7).

    With ``bands`` the hypo sweep reads the τ=``METRIC_BAND_TAU_LO`` lower edge and the
    hyper sweep the τ=``METRIC_BAND_TAU_HI`` upper edge, the same band the level
    metrics score against. With ``bands=None`` both sweeps read the median line
    ``pred`` exactly as before.

    Args:
        pred: ``(N, PRED_STEPS)`` median-line forecast BG, mg/dL.
        true: ``(N, PRED_STEPS)`` matched true BG, mg/dL.
        offsets: δ grid in mg/dL.
        bands: optional ``(N, PRED_STEPS, N_QUANTILES)`` mg/dL quantile fan
            (ascending τ), as stacked by :func:`forecast_bands`.

    Returns:
        ``{'hypo': {h: curve}, 'hyper': {h: curve}}`` for ``h`` in ``HORIZONS``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    assert pred.ndim == 2 and pred.shape == true.shape, \
        f"calibrate_threshold shape mismatch: pred {pred.shape}, true {true.shape}"
    if bands is not None:
        bands = np.asarray(bands, dtype=np.float64)
        assert bands.shape == pred.shape + (N_QUANTILES,), \
            f"calibrate_threshold bands shape {bands.shape}, expected {pred.shape + (N_QUANTILES,)}"
    out: dict[str, dict] = {'hypo': {}, 'hyper': {}}
    for h in HORIZONS:
        k = _HORIZON_IDX[h]
        lo_k = pred[:, k] if bands is None else bands[:, k, _BAND_LO_IDX]
        hi_k = pred[:, k] if bands is None else bands[:, k, _BAND_HI_IDX]
        out['hypo'][h] = threshold_curve(lo_k, true[:, k], BG_HYPO_THRESHOLD, 'hypo', offsets)
        out['hyper'][h] = threshold_curve(hi_k, true[:, k], BG_HYPER_THRESHOLD, 'hyper', offsets)
    return out


def select_offset(curve: list[dict], min_precision: float | None = None,
                  target_recall: float | None = None) -> tuple[float, float | None, float | None]:
    """Pick one operating point from a ``threshold_curve``.

    ``min_precision``: the δ of highest recall whose precision ≥ the floor
    (recall-first under a precision constraint). ``target_recall``: the smallest δ
    that reaches that recall (least precision sacrificed). Neither: the strict
    (δ≈0) point — no operating point is baked in by default. Returns
    ``(offset, recall, precision)``.
    """
    pts = [p for p in curve if p['recall'] is not None]
    if not pts:
        return 0.0, None, None
    if min_precision is not None:
        ok = [p for p in pts if p['precision'] is not None and p['precision'] >= min_precision]
        best = max(ok, key=lambda p: p['recall']) if ok else min(pts, key=lambda p: p['offset'])
        return best['offset'], best['recall'], best['precision']
    if target_recall is not None:
        ok = [p for p in pts if p['recall'] >= target_recall]
        best = min(ok, key=lambda p: p['offset']) if ok else max(pts, key=lambda p: p['recall'])
        return best['offset'], best['recall'], best['precision']
    strict = min(pts, key=lambda p: abs(p['offset']))
    return strict['offset'], strict['recall'], strict['precision']


@dataclass
class Window:
    """One prediction window: the headline median BG forecast + the true CGM.

    ``bands`` (optional, default None) carries the model's full per-step mg/dL
    quantile fan ``(PRED_STEPS, N_QUANTILES)`` (ascending τ, RAW ``f_inv(q_tau)`` —
    uncalibrated) for the quantile-CQR re-fit in ``run_eval.evaluate_from_windows``.
    Defaulting to None keeps every pre-redesign cached/sim Window construction
    (which never set it) valid, and ``forecast_bands`` returns None unless every
    window carries it.
    """
    patient: str
    pred_bg: np.ndarray          # (PRED_STEPS,) median_bg = f_inv(median), mg/dL
    last_bg: float               # raw (bg-clamped) last-context CGM, mg/dL
    cgm: np.ndarray              # (PRED_STEPS,) raw (bg-clamped) true CGM, mg/dL
    bands: np.ndarray | None = None   # (PRED_STEPS, N_QUANTILES) raw mg/dL fan, or None


def _future_overrides(feats: np.ndarray, pred_start: int,
                      announce: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """Slice the prediction-zone future of the announced output channels into
    ``predict`` overrides.

    ``feats`` is the normalized (N, F) input stack; output channel ``ch`` lives at
    input feature ``CHANNEL_TO_FEAT[ch]`` (carb 0→feat 1, insulin 1→feat 2,
    exercise 2→feat 3).
    Returns a ``{ch: (PREDICTION_PATCHES, PATCH_SIZE)}`` dict of the
    already-normalized future values, ready to hand to
    ``inference.predict(overrides=…)``.
    """
    fut = feats[pred_start:pred_start + PRED_STEPS]      # (PRED_STEPS, F), normalized
    ov: dict[int, torch.Tensor] = {}
    for ch in announce:
        feat_idx = CHANNEL_TO_FEAT[ch]
        col = fut[:, feat_idx].reshape(PREDICTION_PATCHES, PATCH_SIZE)
        ov[ch] = torch.from_numpy(col.copy())
    return ov


def collect_windows(model, stats, segments: list[Segment], device,
                    stride_patches: int = 8, max_per_patient: int | None = None,
                    seed: int = 0, conditional: bool = True,
                    announce: tuple[int, ...] = (0, 1, 2)) -> list[Window]:
    """Slide prediction windows across each segment and capture the BG forecast.

    The model is ALWAYS conditioned: each window's true future ``announce``
    channels (the logged carbs/insulin/exercise over the prediction zone) are
    announced to the model, so the captured forecast is conditioned on them — the
    deployment what-if regime in which the patient declares the meal/dose/session.
    There is no unconditioned regime; ``conditional`` is retained only for
    call-compatibility and no longer toggles anything.

    Args:
        stride_patches: gap (in patches) between successive window starts.
        max_per_patient: cap on windows per patient (random subsample if exceeded).
        conditional: deprecated no-op (the forecast is always conditioned).
        announce: output-channel indices announced — carbs (0), insulin (1) and
            exercise (2); BG is never conditionable. Exercise is identically zero
            wherever a source has no activity record, so announcing it declares
            "no session".
    """
    by_patient: dict[str, list[Window]] = {}
    for seg in segments:
        n = (len(seg) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX_STEPS + PRED_STEPS:
            continue
        feats = build_feature_stack(seg, stats)
        # The forecast is scored against the raw (bg-clamped) CGM (the model
        # consumes raw post-noise signals); both the per-window truth and the
        # last-context anchor are sliced from it.
        cgm_smooth = smoothed_cgm(seg.cgm)
        stride = stride_patches * PATCH_SIZE
        for pred_start in range(CTX_STEPS, n - PRED_STEPS + 1, stride):
            ctx = context_window(feats, pred_start, MAX_CONTEXT_PATCHES)
            overrides = _future_overrides(feats, pred_start, announce)
            out = predict(model, ctx, normalization_stats=stats, device=device,
                          overrides=overrides)
            pred_bg = out['median_bg'].detach().cpu().numpy().astype(np.float64)
            # RAW mg/dL band fan (no conformal_delta passed) so the downstream CQR
            # fit in run_eval is on UNcalibrated bands. (P, S, K) -> (PRED_STEPS, K).
            bands = out['bands'].detach().cpu().numpy().reshape(-1, N_QUANTILES).astype(np.float64)
            w = Window(
                patient=seg.patient,
                pred_bg=pred_bg,
                last_bg=float(min(max(cgm_smooth[pred_start - 1], BG_CLAMP_MIN), BG_CLAMP_MAX)),
                cgm=cgm_smooth[pred_start:pred_start + PRED_STEPS].copy(),
                bands=bands,
            )
            by_patient.setdefault(seg.patient, []).append(w)

    rng = np.random.default_rng(seed)
    out_windows: list[Window] = []
    for p, ws in by_patient.items():
        if max_per_patient and len(ws) > max_per_patient:
            ws = [ws[i] for i in rng.choice(len(ws), max_per_patient, replace=False)]
        out_windows.extend(ws)
    return out_windows


def forecast_windows(windows: list[Window]):
    """Stack a window list into ``(pred (N,T), true (N,T), last_bg (N,), patients)``.

    The headline ``pred`` is the model's median BG forecast (``median_bg``)
    captured at collection; no per-patient calibration is applied.
    """
    if not windows:
        empty = np.zeros((0, PRED_STEPS), dtype=np.float64)
        return empty, empty.copy(), np.zeros(0, dtype=np.float64), []
    pred = np.stack([w.pred_bg for w in windows]).astype(np.float64)
    true = np.stack([w.cgm for w in windows]).astype(np.float64)
    last_bg = np.array([w.last_bg for w in windows], dtype=np.float64)
    return pred, true, last_bg, [w.patient for w in windows]


def forecast_bands(windows: list[Window]) -> np.ndarray | None:
    """Stack the per-window RAW mg/dL quantile fans into ``(N, PRED_STEPS, N_QUANTILES)``.

    Returns ``None`` if the list is empty or ANY window lacks ``bands`` (legacy caches
    or any window that did not capture a band fan), so callers can cleanly skip the
    quantile-CQR path. Ascending τ, mg/dL, uncalibrated — the input to
    ``conformal.fit_quantile_conformal`` / ``apply_quantile_conformal``.
    """
    if not windows or any(w.bands is None for w in windows):
        return None
    return np.stack([w.bands for w in windows]).astype(np.float64)
