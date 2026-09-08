"""Simulator-cohort bridge for the in-domain `metrics/sim/` report.

One fresh T1DMSIM patient per seed, warmup discarded, into the 5-column stack; mask bit is
0.0 throughout (all OBSERVED). Signal channels are post-noise curves — no reconstruction.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import torch

from config import (N_INPUT_FEATURES, PATCH_SIZE, MAX_CONTEXT_PATCHES, PREDICTION_PATCHES,
                    PREDICTION_HORIZON_HOURS, NIGHT_LONG_HORIZON_HOURS,
                    N_QUANTILES, TIME_PROBE_N_BINS, CHANNEL_TO_FEAT)
from data import (_make_simulator, simulate_discard_warmup, _mask_slots,
                  BG_MASKED_FEAT)
from normalization import CHANNEL_NAMES, normalize
from inference import predict, predict_rolling
from utils import time_of_day_decode_bins
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
from metrics.core.schema import Segment
from metrics.core.features import context_window
from metrics.core.calibrate import Window, _future_overrides
from metrics.core.run_eval import _make_night_overrides_fn


def _assert_announces_all(announce: tuple[int, ...]) -> None:
    """Every announceable channel must be announced, on every evaluation path.

    A dropped slot sits at ``normalize(0)`` — for exercise_equiv a legal "no session" (-0.139 z on
    the balanced pool) — so the window scores as if nothing happened and the table looks right.
    """
    assert tuple(announce) == tuple(CHANNEL_TO_FEAT), (
        f"announced set {tuple(announce)} != announceable set {tuple(CHANNEL_TO_FEAT)}")


def _smooth_sim_bg(bg_raw: np.ndarray) -> np.ndarray:
    """RAW simulator BG, bg-clamped — the ground truth every metric scores against.

    No smoothing despite the name: the scored truth is the raw ``bg_observed`` that produced the
    model input.
    """
    return np.clip(np.asarray(bg_raw, dtype=np.float64),
                   BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)

# Every window runs FORECAST: one PREDICTION_PATCHES span at the right edge, whole context visible.
CTX_STEPS = MAX_CONTEXT_PATCHES * PATCH_SIZE
PRED_STEPS = PREDICTION_PATCHES * PATCH_SIZE
# parity/clarke roll to the night long horizon for hour-by-hour figures; single pass when unset
FIG_ROLLS = max(1, math.ceil(NIGHT_LONG_HORIZON_HOURS / PREDICTION_HORIZON_HOURS))
FIG_STEPS = FIG_ROLLS * PRED_STEPS

# Masked set from data._mask_slots; FORECAST_D bins on d, distance to visible evidence.

# Right-edge span has no right neighbour: slot j sits at d=j+1, so 30/60/90/120 min ARE d=1..4.
_fc_idx, _fc_valid, _fc_d, _fc_anchor_step = _mask_slots(
    [(MAX_CONTEXT_PATCHES, PREDICTION_PATCHES)],
    MAX_CONTEXT_PATCHES + PREDICTION_PATCHES,
)
FORECAST_D = tuple(int(x) for x in _fc_d[_fc_valid])
assert FORECAST_D == tuple(range(1, PREDICTION_PATCHES + 1)), (
    f"right-edge forecast d {FORECAST_D} is not one-sided 1..{PREDICTION_PATCHES}")
# Anchor is ONE-SIDED, LEFT-PREFERRING (right neighbour only at patch 0); Window.last_bg is it.
assert int(_fc_anchor_step[_fc_valid][0]) == CTX_STEPS - 1, (
    "forecast anchor is not the last visible step")

# Fixed seed pools, one patient/seed, cal/test disjoint; a window costs 340 patches (170h).

# At 288h the caps bind exactly: 60 test windows at stride 4, 24 calibration at stride 8.
DEFAULT_HOURS = 288.0
CAL_SEEDS = tuple(range(7000, 7012))     # 12 calibration patients
TEST_SEEDS = tuple(range(8000, 8030))    # 30 test patients


def make_sim_runs(seeds, hours: float) -> list[tuple[str, dict]]:
    """One fresh simulator patient per seed -> ``[(patient_id, data_dict)]``, ``hours`` each.

    ``data_dict`` keys: bg_observed/total_carb/total_insulin/total_exercise/hour_of_day/day, plus
    insulin_resistance/hgo (neither input nor output).
    """
    runs = []
    for s in seeds:
        sim = _make_simulator(int(s), uniform_skills=False)
        runs.append((f"sim{int(s)}", simulate_discard_warmup(sim, float(hours))))
    return runs


SIM_DATASET = 'sim'
# Fixed epoch: simulator has no wall clock; t0 offset by opening hour so hour_of_day() matches.
SIM_EPOCH = datetime(2024, 1, 1, 0, 0, 0)


def run_to_segment(pid: str, d: dict) -> Segment:
    """One simulator run as a ``Segment``, for the consumers that take Segments.

    carb_curve/insulin_curve carry total_carb/total_insulin verbatim (already curves); exercise is
    UNRESCALED total_exercise. EVENT CHANNELS ARE EMPTY: convolved into the curves, unrecoverable.
    """
    bg = np.clip(np.asarray(d['bg_observed'], dtype=np.float64),
                 BG_CLAMP_MIN, BG_CLAMP_MAX)
    n = len(bg)
    assert np.isfinite(bg).all(), f"{pid}: simulator emitted a non-finite BG"
    zeros = lambda: np.zeros(n, dtype=np.float64)          # noqa: E731 — event channels
    return Segment(
        dataset=SIM_DATASET, patient=pid,
        t0=SIM_EPOCH + timedelta(hours=float(d['hour_of_day'][0])),
        cgm=bg,
        carb_grams=zeros(), bolus_units=zeros(), basal_rate=zeros(),
        exercise=np.clip(np.asarray(d['total_exercise'], dtype=np.float64), 0.0, None),
        carb_curve=np.clip(np.asarray(d['total_carb'], dtype=np.float64), 0.0, None),
        insulin_curve=np.clip(np.asarray(d['total_insulin'], dtype=np.float64), 0.0, None),
    )


def make_sim_segments(seeds=None, hours: float = DEFAULT_HOURS) -> list[Segment]:
    """Fresh simulator patients as Segments, one per seed; ``seeds`` defaults to ``TEST_SEEDS``."""
    return [run_to_segment(pid, d)
            for pid, d in make_sim_runs(TEST_SEEDS if seeds is None else seeds, hours)]


def build_sim_feature_stack(d: dict, stats: dict) -> np.ndarray:
    """Normalized ``(N, N_INPUT_FEATURES)`` input stack from a simulator run.

    normalize() per stats: bg Kovatchev before the z-score, sparse three log1p. Feat 3 is g/step
    carb-equivalent, never risk-transformed. Feat BG_MASKED_FEAT is 0.0 throughout (all OBSERVED).
    """
    bg = d['bg_observed'].astype(np.float64)
    n = len(bg)
    # Width, not names (normalization.py's copy); channels 0..BG_MASKED_FEAT-1, mask bit one wider.
    assert len(CHANNEL_NAMES) == BG_MASKED_FEAT < N_INPUT_FEATURES, (
        f"unexpected CHANNEL_NAMES {list(CHANNEL_NAMES)} against "
        f"BG_MASKED_FEAT={BG_MASKED_FEAT}, N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )
    raw = np.empty((n, len(CHANNEL_NAMES)), dtype=np.float64)
    raw[:, 0] = np.clip(bg, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)
    raw[:, 1] = np.clip(d['total_carb'].astype(np.float64), 0.0, None).astype(np.float64)
    raw[:, 2] = np.clip(d['total_insulin'].astype(np.float64), 0.0, None).astype(np.float64)
    raw[:, 3] = np.clip(d['total_exercise'].astype(np.float64), 0.0, None).astype(np.float64)
    # normalize() owns the per-channel transform and z-score: single source of truth for the input.
    feats = np.zeros((n, N_INPUT_FEATURES), dtype=np.float32)
    feats[:, :BG_MASKED_FEAT] = normalize(raw, stats)
    return feats


def collect_sim_windows(model, stats, runs, device, stride_patches: int = 8,
                        max_per_patient: int | None = None,
                        announce: tuple[int, ...] = (0, 1, 2), seed: int = 0,
                        conformal_delta=None) -> list[Window]:
    """Slide windows per simulated patient, capturing median BG and RAW mg/dL fan per Window.

    Always conditioned (true future doses announced). conformal_delta is taken for call-symmetry
    and DELIBERATELY not applied -- run_eval owns CQR fit/apply; pre-calibrating would double it.
    """
    del conformal_delta  # never applied here (see docstring): bands captured RAW
    _assert_announces_all(announce)
    by_patient: dict[str, list[Window]] = {}
    for pid, d in runs:
        feats = build_sim_feature_stack(d, stats)
        # truth and the ``last_bg`` anchor both slice the raw bg-clamped CGM, never the future
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
                     announce: tuple[int, ...] = (0, 1, 2), cap: int = 24,
                     conformal_delta=None) -> list[dict]:
    """Per-window conditional BG forecasts for the trajectory/parity/clarke figures.

    Forecasts roll to FIG_STEPS; true is NaN-padded past a segment end. conformal_delta applies
    only in the single-pass branch; each row also carries the time-of-day probe at the origin.
    """
    _assert_announces_all(announce)
    rows = []
    for pid, d in runs:
        feats = build_sim_feature_stack(d, stats)
        # figure ``true`` and the context tail both slice the raw bg-clamped CGM
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
                # (P, S, K) -> (PRED_STEPS, K); calibrated when conformal_delta is set
                bands = out['bands'].detach().cpu().numpy().reshape(
                    -1, N_QUANTILES)[:FIG_STEPS].astype(np.float64)
            else:
                # (PRED_STEPS, K) delta cannot span FIG_STEPS, so bands stay None, no ribbon drawn.
                fn = _make_night_overrides_fn(feats, ps, announce, stats)
                out = predict_rolling(model, ctx, n_rolls=FIG_ROLLS,
                                      normalization_stats=stats, device=device,
                                      overrides_fn=fn, return_time=True)
                pred = out['pred_bg'].detach().cpu().numpy()[:FIG_STEPS]
            tr = cgm[ps:ps + FIG_STEPS]
            if len(tr) < FIG_STEPS:
                tr = np.concatenate([tr, np.full(FIG_STEPS - len(tr), np.nan)])
            # Probe rides the SAME forward (origin decodes from patch 0); NaN/None if off.
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
