"""
Simulator-cohort bridge for the in-domain `metrics/sim/` report.

Fresh patients are drawn from the T1DMSIM behavioral simulator at chosen seeds
(each seed → a distinct random patient), run through the same warmup-discard the
training pipeline uses, and turned into the model's normalized ``N_INPUT_FEATURES``-feature
input stack ``[bg_absolute, carbs, insulin]``. Unlike
the real-data bridge, the simulator emits the model's signal channels DIRECTLY —
``total_carb`` / ``total_insulin`` are the post-noise per-step carb / insulin the
normalization stats were fit on (``data._build_sample`` consumes them verbatim) —
so no absorption/action kernel reconstruction is needed.

This is the model's TRAINING distribution, so the report is an in-domain
reference, not a peer comparison.

Risk-space redesign note: the model now outputs ONLY a BG quantile forecast (no
carb/insulin/IS/HGO dynamics channels and no trend head), so the former
``bg_delta`` input channel, the IS/HGO mean-imputation, and the four
dynamics-channel ground-truth overlays in the day figures are gone. The day
figure keeps the BG true-vs-predicted comparison only.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from config import (N_INPUT_FEATURES, PATCH_SIZE, MAX_CONTEXT_PATCHES, PREDICTION_PATCHES,
                    PREDICTION_HORIZON_HOURS, NIGHT_LONG_HORIZON_HOURS,
                    N_QUANTILES, TIME_PROBE_N_BINS)
from data import _make_simulator, simulate_discard_warmup
from normalization import CHANNEL_NAMES, normalize
from inference import predict, predict_rolling
from utils import time_of_day_decode_bins
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
from realdata.features import context_window
from realdata.calibrate import Window, _future_overrides
from realdata.run_eval import _make_night_overrides_fn


def _smooth_sim_bg(bg_raw: np.ndarray) -> np.ndarray:
    """RAW simulator BG, bg-clamped (the ground truth every metric scores against).

    The pipeline consumes raw post-noise signals (``data._build_sample``): every
    scored ground-truth BG is the raw ``bg_observed`` that produced the model input,
    physical-range clamped. Name kept for its call-sites; no FIR smoothing applied.
    """
    return np.clip(np.asarray(bg_raw, dtype=np.float64),
                   BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)

CTX_STEPS = MAX_CONTEXT_PATCHES * PATCH_SIZE
PRED_STEPS = PREDICTION_PATCHES * PATCH_SIZE
# Parity/clarke forecasts roll out to the night long horizon for hour-by-hour
# figure granularity (mirrors realdata.report.FIG_ROLLS); single pass when unset.
FIG_ROLLS = max(1, math.ceil(NIGHT_LONG_HORIZON_HOURS / PREDICTION_HORIZON_HOURS))
FIG_STEPS = FIG_ROLLS * PRED_STEPS

# Fixed (arbitrary, reproducible) seed pools — each seed draws a distinct random
# patient; calibration and test pools are disjoint.
DEFAULT_HOURS = 120.0
CAL_SEEDS = tuple(range(7000, 7012))     # 12 calibration patients
TEST_SEEDS = tuple(range(8000, 8030))    # 30 test patients


def make_sim_runs(seeds, hours: float) -> list[tuple[str, dict]]:
    """Generate one fresh simulator patient per seed.

    Args:
        seeds: iterable of integer patient seeds (each → a distinct random patient).
        hours: post-warmup trajectory length per patient.

    Returns:
        list of ``(patient_id, data_dict)`` where ``data_dict`` is the simulator's
        per-channel array dict (post-warmup), keys ``bg_observed`` / ``total_carb``
        / ``total_insulin`` / ``hour_of_day`` / ``day`` (the dynamics channels
        ``insulin_resistance`` / ``hgo`` / ``total_exercise`` remain in the dict
        but are no longer model inputs or outputs).
    """
    runs = []
    for s in seeds:
        sim = _make_simulator(int(s), uniform_skills=False)
        runs.append((f"sim{int(s)}", simulate_discard_warmup(sim, float(hours))))
    return runs


def build_sim_feature_stack(d: dict, stats: dict) -> np.ndarray:
    """Normalized (N, N_INPUT_FEATURES) input stack from a simulator run.

    The three channels (bg_observed, carb, insulin — the trimmed
    ``CHANNEL_NAMES``) are normalized per ``stats`` via ``normalization.normalize``,
    so bg (feat 0) takes the Kovatchev risk transform BEFORE the z-score and the
    sparse carb/insulin take log1p — the SOLE input path, mirroring
    ``realdata.features.build_feature_stack`` / ``data._build_sample`` (the redesign
    dropped ``bg_delta``, the IS/HGO latents, and the four sin/cos temporal
    features from the stack).

    The model consumes RAW post-noise signals: every signal channel (bg feat0,
    carb feat1, insulin feat2) is used raw BEFORE normalization, with the SAME
    clamps as ``data._build_sample`` (bg → the physical BG range; the sparse
    carb/insulin floored at 0; no FIR smoothing).
    """
    bg = d['bg_observed'].astype(np.float64)
    n = len(bg)
    assert list(CHANNEL_NAMES) == ['bg_absolute', 'carb_intake', 'insulin_combined'], (
        f"unexpected CHANNEL_NAMES {list(CHANNEL_NAMES)}"
    )
    raw = np.empty((n, N_INPUT_FEATURES), dtype=np.float64)
    raw[:, 0] = np.clip(bg, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)
    raw[:, 1] = np.clip(d['total_carb'].astype(np.float64), 0.0, None).astype(np.float64)
    raw[:, 2] = np.clip(d['total_insulin'].astype(np.float64), 0.0, None).astype(np.float64)
    # normalize() owns the per-channel forward transform (risk-f on bg, log1p on
    # the sparse channels) + z-score — the single source of truth for the input path.
    return normalize(raw, stats).astype(np.float32)


def collect_sim_windows(model, stats, runs, device, stride_patches: int = 8,
                        max_per_patient: int | None = None,
                        announce: tuple[int, ...] = (0, 1), seed: int = 0,
                        conformal_delta=None) -> list[Window]:
    """Slide prediction windows across each simulated patient and capture the
    headline median BG forecast — the simulator analogue of
    ``realdata.calibrate.collect_windows`` (same trimmed ``Window`` of
    ``pred_bg`` = ``median_bg`` + true CGM, now also the RAW mg/dL band fan). The
    model is always conditioned: each window announces its true future carbs/insulin.

    NOTE on ``conformal_delta``: it is accepted only for call-symmetry and is
    DELIBERATELY NOT used to pre-calibrate the captured bands. The metric CQR re-fit
    in ``run_eval.evaluate_from_windows`` owns the fit/apply on a held-out split, so
    the bands stored here MUST stay RAW — pre-applying the stored sim delta here
    would double-calibrate. The FIGURE path (``collect_sim_rows``) is where the
    stored sim delta is legitimately applied for display."""
    del conformal_delta  # never applied here (see docstring): bands captured RAW
    by_patient: dict[str, list[Window]] = {}
    for pid, d in runs:
        feats = build_sim_feature_stack(d, stats)
        # Scored ground-truth BG is the raw (bg-clamped) CGM (one space; never the
        # future) — both the ``last_bg`` anchor and the truth the forecast is scored on.
        cgm = _smooth_sim_bg(d['bg_observed'])
        n = (len(cgm) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX_STEPS + PRED_STEPS:
            continue
        stride = stride_patches * PATCH_SIZE
        for ps in range(CTX_STEPS, n - PRED_STEPS + 1, stride):
            ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)
            overrides = _future_overrides(feats, ps, announce)
            out = predict(model, ctx, normalization_stats=stats, device=device,
                          overrides=overrides)
            pred_bg = out['median_bg'].detach().cpu().numpy().astype(np.float64)
            bands = out['bands'].detach().cpu().numpy().reshape(-1, N_QUANTILES).astype(np.float64)
            by_patient.setdefault(pid, []).append(Window(
                patient=pid, pred_bg=pred_bg, last_bg=float(cgm[ps - 1]),
                cgm=cgm[ps:ps + PRED_STEPS].copy(), bands=bands))

    rng = np.random.default_rng(seed)
    out_windows: list[Window] = []
    for _, ws in by_patient.items():
        if max_per_patient and len(ws) > max_per_patient:
            ws = [ws[i] for i in rng.choice(len(ws), max_per_patient, replace=False)]
        out_windows.extend(ws)
    return out_windows


def collect_sim_rows(model, stats, runs, device,
                     announce: tuple[int, ...] = (0, 1), cap: int = 24,
                     conformal_delta=None) -> list[dict]:
    """Per-window conditional BG forecasts for the trajectory/parity/clarke figures.

    The model is always conditioned: each window announces its true future
    carbs/insulin. Forecasts roll out to ``FIG_STEPS`` (the night long horizon) so
    the parity/clarke panels read hour-by-hour; ``true`` is NaN-padded to that
    length past a segment end and the per-horizon plotters drop the unfilled tail.

    ``conformal_delta`` (the STORED sim delta from the checkpoint) is the FIGURE
    path's legitimate consumer: in the single-pass branch (``FIG_ROLLS == 1``) it is
    threaded into ``predict`` so the captured ``row['bands']`` are CALIBRATED
    (median untouched). It is shaped ``(PRED_STEPS, N_QUANTILES)`` and covers only
    the single forward pass, so in the ROLLING branch it is NOT applied (it cannot
    span ``FIG_STEPS`` > ``PRED_STEPS``) and ``row['bands']`` is left None.
    ``None`` ⇒ raw bands, bit-identical median/pred.

    Each row also carries the auxiliary time-of-day probe read at the forecast
    origin: ``time_probs`` (the per-patch ``(P, TIME_PROBE_N_BINS)`` softmax belief,
    or None) rides on the SAME forward as the forecast (``return_time=True``), and
    the scalar ``pred_hour``/``tod_R`` decode from its patch 0 (plus the simulator's
    true ``true_hour``); all three are NaN / None when the probe is disabled. The
    probe is read-only — the BG forecast is bit-identical to the probe-off path."""
    rows = []
    for pid, d in runs:
        feats = build_sim_feature_stack(d, stats)
        # Scored ground-truth BG (figure ``true`` + context tail) is the
        # raw (bg-clamped) CGM — one space, never the future.
        cgm = _smooth_sim_bg(d['bg_observed'])
        n = (len(cgm) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX_STEPS + PRED_STEPS:
            continue
        cnt = 0
        for ps in range(CTX_STEPS, n - PRED_STEPS + 1, 8 * PATCH_SIZE):
            if cnt >= cap:
                break
            ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)
            bands = None
            if FIG_ROLLS == 1:
                overrides = _future_overrides(feats, ps, announce)
                out = predict(model, ctx, normalization_stats=stats, device=device,
                              overrides=overrides, conformal_delta=conformal_delta,
                              return_time=True)
                pred = out['median_bg'].detach().cpu().numpy()[:FIG_STEPS]
                # (P, S, K) -> (PRED_STEPS, K), sliced to the plotted length (== PRED_STEPS
                # in the single-pass branch). Calibrated when conformal_delta is set.
                bands = out['bands'].detach().cpu().numpy().reshape(
                    -1, N_QUANTILES)[:FIG_STEPS].astype(np.float64)
            else:
                # Rolling: the (PRED_STEPS, K) sim delta cannot span FIG_STEPS; leave
                # bands None so the figure draws a band ribbon only when present.
                fn = _make_night_overrides_fn(feats, ps, announce, stats)
                out = predict_rolling(model, ctx, n_rolls=FIG_ROLLS,
                                      normalization_stats=stats, device=device,
                                      overrides_fn=fn, return_time=True)
                pred = out['pred_bg'].detach().cpu().numpy()[:FIG_STEPS]
            tr = cgm[ps:ps + FIG_STEPS]
            if len(tr) < FIG_STEPS:
                tr = np.concatenate([tr, np.full(FIG_STEPS - len(tr), np.nan)])
            # Auxiliary time-of-day probe at the forecast origin (read-only; the BG
            # forecast above is untouched). The per-patch bin logits ride on the SAME
            # forward as the forecast (``return_time=True``), so the scalar origin
            # clock decodes from patch 0 — no extra forward. ``time_probs`` are NaN /
            # None when the probe is disabled; ``true_hour`` is the simulator's
            # wall-clock hour at the origin step.
            tp = out.get('time_pred')
            time_probs = None if tp is None else torch.softmax(tp, -1).cpu().numpy()
            if tp is None:
                pred_hour, tod_r = float('nan'), float('nan')
            else:
                h_t, r_t = time_of_day_decode_bins(tp[0:1, :], TIME_PROBE_N_BINS)
                pred_hour, tod_r = float(h_t.item()), float(r_t.item())
            true_hour = float(d['hour_of_day'][ps])
            rows.append({'patient': pid, 'pred': pred, 'true': tr,
                         'ctx_tail': cgm[max(0, ps - 12):ps], 'bands': bands,
                         'pred_hour': pred_hour, 'true_hour': true_hour,
                         'tod_R': tod_r, 'time_probs': time_probs})
            cnt += 1
    return rows
