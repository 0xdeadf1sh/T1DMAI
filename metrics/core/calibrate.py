"""Window collection and excursion-decision calibration on the model-input bridge.

The headline forecast is the quantile median ``median_bg = f_inv(median)``; the decision calibration needs
only plain ``(pred_bg, true_bg)`` arrays.

    collect_windows(model, stats, segments)  -> list[Window]  (median_bg + true CGM)
    forecast_windows(windows)                -> (pred (N,T), true (N,T), last_bg (N,), patients)
    calibrate_threshold(pred, true, bands=…) -> per-horizon recall–precision curves
    select_offset(curve, …)                  -> one operating point under a precision floor

The decision sweep reads the metric BAND EDGES (τ=``METRIC_BAND_TAU_LO`` hypo, ``METRIC_BAND_TAU_HI`` hyper)
when a fan is supplied, matching the level metrics' basis; without one it reads the median line.
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
# δ ∈ [0, 60] mg/dL in 2.5 steps — the whole recall–precision curve, no floor baked in (PLAN §6/§7)
_OFFSET_GRID = tuple(float(x) for x in np.arange(0.0, 60.0 + 1e-9, 2.5))

# the band the sweep reads with a fan supplied: the level metrics' knob, not the clinical alarm taus
_BAND_LO_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_LO)
_BAND_HI_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_HI)


def threshold_curve(pred: np.ndarray, true: np.ndarray, thr: float, side: str,
                    offsets=_OFFSET_GRID) -> list[dict]:
    """Recall–precision curve over a decision-offset sweep at ONE horizon point.

    ``pred`` / ``true`` ``(N,)`` mg/dL at that point; the alarm fires at ``pred < thr + δ`` (hypo) or
    ``pred > thr − δ`` (hyper), so a positive δ buys recall. Truth and decision are STRICT crossings.
    -> ``{offset, recall, precision, n_true, n_pred}`` per δ: ``n_true`` constant, ``n_pred`` grows with δ,
    recall and precision ``None`` on a zero denominator.
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
    """Per-horizon hypo and hyper recall–precision curves -> ``{'hypo': {h: curve}, 'hyper': {h: curve}}``.

    Fit on the CALIBRATION split; no precision floor or target recall baked in (PLAN §6/§7).
    ``pred`` / ``true`` ``(N, PRED_STEPS)`` mg/dL; ``bands`` optional ``(N, PRED_STEPS, N_QUANTILES)`` mg/dL,
    ascending τ. With ``bands`` the sweeps read the τ=``METRIC_BAND_TAU_LO`` / ``METRIC_BAND_TAU_HI`` edges,
    without it the median line.
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
    """One operating point from a ``threshold_curve`` -> ``(offset, recall, precision)``.

    ``min_precision``: highest-recall δ whose precision clears the floor. ``target_recall``: smallest δ that
    reaches it. Neither: the strict δ≈0 point, so nothing is baked in by default.
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
    """One prediction window: the median BG forecast and the true CGM.

    ``bands`` is the per-step mg/dL fan ``(PRED_STEPS, N_QUANTILES)``, ascending τ, RAW ``f_inv(q_tau)`` and
    uncalibrated, for the CQR re-fit in ``run_eval.evaluate_from_windows``. None is legal — a window built
    without one — and ``forecast_bands`` then returns None for the whole list.
    """
    patient: str
    pred_bg: np.ndarray          # (PRED_STEPS,) median_bg = f_inv(median), mg/dL
    last_bg: float               # raw (bg-clamped) last-context CGM, mg/dL
    cgm: np.ndarray              # (PRED_STEPS,) raw (bg-clamped) true CGM, mg/dL
    bands: np.ndarray | None = None   # (PRED_STEPS, N_QUANTILES) raw mg/dL fan, or None


def _future_overrides(feats: np.ndarray, pred_start: int,
                      announce: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """The announced channels' prediction-zone future -> ``{ch: (PREDICTION_PATCHES, PATCH_SIZE)}``, normalized.

    ``feats`` is the normalized ``(N, F)`` stack; output channel ``ch`` sits at input feature
    ``CHANNEL_TO_FEAT[ch]`` — carb 0→feat 1, insulin 1→feat 2, exercise 2→feat 3.
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

    ALWAYS conditioned: each window's true future ``announce`` channels reach the model, the deployment
    regime where the patient declares the meal, dose or session. ``conditional`` is a no-op kept for callers.
    ``stride_patches`` in patches between window starts; ``max_per_patient`` subsamples at random.
    ``announce``: carb 0, insulin 1, exercise 2 — BG is never conditionable, and announcing an exercise
    column of zeros declares "no session".
    """
    by_patient: dict[str, list[Window]] = {}
    for seg in segments:
        n = (len(seg) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX_STEPS + PRED_STEPS:
            continue
        feats = build_feature_stack(seg, stats)
        # truth and last-context anchor both slice the raw bg-clamped CGM
        cgm_smooth = smoothed_cgm(seg.cgm)
        stride = stride_patches * PATCH_SIZE
        for pred_start in range(CTX_STEPS, n - PRED_STEPS + 1, stride):
            ctx = context_window(feats, pred_start, MAX_CONTEXT_PATCHES)
            overrides = _future_overrides(feats, pred_start, announce)
            out = predict(model, ctx, normalization_stats=stats, device=device,
                          overrides=overrides)
            pred_bg = out['median_bg'].detach().cpu().numpy().astype(np.float64)
            # RAW fan, no conformal_delta: run_eval's CQR fit needs uncalibrated bands. (P,S,K) -> (PRED_STEPS,K)
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

    ``pred`` is the median forecast as captured; no per-patient calibration is applied.
    """
    if not windows:
        empty = np.zeros((0, PRED_STEPS), dtype=np.float64)
        return empty, empty.copy(), np.zeros(0, dtype=np.float64), []
    pred = np.stack([w.pred_bg for w in windows]).astype(np.float64)
    true = np.stack([w.cgm for w in windows]).astype(np.float64)
    last_bg = np.array([w.last_bg for w in windows], dtype=np.float64)
    return pred, true, last_bg, [w.patient for w in windows]


def forecast_bands(windows: list[Window]) -> np.ndarray | None:
    """Per-window RAW fans stacked into ``(N, PRED_STEPS, N_QUANTILES)``: ascending τ, mg/dL, uncalibrated.

    ``None`` when the list is empty or ANY window lacks ``bands``, so a caller skips the CQR path whole.
    """
    if not windows or any(w.bands is None for w in windows):
        return None
    return np.stack([w.bands for w in windows]).astype(np.float64)
