"""Simulator-cohort bridge for the in-domain `metrics/sim/` report.

One fresh T1DMSIM patient per seed, warmup discarded as in training, into the stack
``[bg_absolute, carbs, insulin, exercise, bg_masked]`` — four normalized channels plus the announcement bit,
0.0 throughout a stack of observed readings.
The simulator emits the SIGNAL CHANNELS directly: ``total_carb`` / ``total_insulin`` / ``total_exercise`` are
the post-noise per-step curves the stats were fit on, so no kernel reconstruction happens.
This is the TRAINING distribution — an in-domain reference, not a peer comparison.
``run_to_segment`` / ``make_sim_segments`` expose the same runs as ``Segment``s. The exercise column is why:
no source of bare logged events carries a session in the channel's units, so a probe of it needs these.
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

    A set short of ``CHANNEL_TO_FEAT`` leaves the dropped slot at ``normalize(0)`` — for exercise_equiv a
    legal "no session" (−0.139 z on the balanced pool), so the window scores as if none happened and the
    table looks right.
    """
    assert tuple(announce) == tuple(CHANNEL_TO_FEAT), (
        f"announced set {tuple(announce)} != announceable set {tuple(CHANNEL_TO_FEAT)}")


def _smooth_sim_bg(bg_raw: np.ndarray) -> np.ndarray:
    """RAW simulator BG, bg-clamped — the ground truth every metric scores against.

    No smoothing despite the name, kept for its call sites: the scored truth is the raw ``bg_observed`` that
    produced the model input.
    """
    return np.clip(np.asarray(bg_raw, dtype=np.float64),
                   BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)

# Every window here runs the FORECAST protocol: one masked span of ``PREDICTION_PATCHES`` at the right edge,
# whole context visible — one case of the masked-BG objective, not a mode.
# ``CTX_STEPS`` the visible prefix, ``PRED_STEPS`` the masked span, both in steps.
CTX_STEPS = MAX_CONTEXT_PATCHES * PATCH_SIZE
PRED_STEPS = PREDICTION_PATCHES * PATCH_SIZE
# parity/clarke roll to the night long horizon for hour-by-hour figures; single pass when unset
FIG_ROLLS = max(1, math.ceil(NIGHT_LONG_HORIZON_HOURS / PREDICTION_HORIZON_HOURS))
FIG_STEPS = FIG_ROLLS * PRED_STEPS

# Masked set expanded by ``data._mask_slots``, never restated. ``FORECAST_D`` is the distance in patches to
# the nearest visible evidence on EITHER side — the only axis a masked-BG metric bins on, never span length.
# A right-edge span has no right neighbour, so slot j sits at d = j + 1: 30/60/90/120 min ARE d = 1..4.
_fc_idx, _fc_valid, _fc_d, _fc_anchor_step = _mask_slots(
    [(MAX_CONTEXT_PATCHES, PREDICTION_PATCHES)],
    MAX_CONTEXT_PATCHES + PREDICTION_PATCHES,
)
FORECAST_D = tuple(int(x) for x in _fc_d[_fc_valid])
assert FORECAST_D == tuple(range(1, PREDICTION_PATCHES + 1)), (
    f"right-edge forecast d {FORECAST_D} is not one-sided 1..{PREDICTION_PATCHES}")
# The anchor is ONE-SIDED and LEFT-PREFERRING: the left neighbour's last step (the right neighbour's first
# only at patch 0), so it ignores the near side. ``Window.last_bg`` is that step here.
assert int(_fc_anchor_step[_fc_valid][0]) == CTX_STEPS - 1, (
    "forecast anchor is not the last visible step")

# Fixed seed pools, one distinct patient per seed; calibration and test are disjoint.
# Post-warmup hours per patient: one forecast window costs MAX_CONTEXT_PATCHES + PREDICTION_PATCHES = 340
# patches (170 h), and a shorter run yields ZERO windows and an empty report, not an error. At 288 h the
# collectors' caps bind exactly — 60 test windows at stride 4, 24 calibration at stride 8 — and curves_sim's
# MAX_CONTEXT_PATCHES + DAY_PATCHES figure fits.
DEFAULT_HOURS = 288.0
CAL_SEEDS = tuple(range(7000, 7012))     # 12 calibration patients
TEST_SEEDS = tuple(range(8000, 8030))    # 30 test patients


def make_sim_runs(seeds, hours: float) -> list[tuple[str, dict]]:
    """One fresh simulator patient per seed -> ``[(patient_id, data_dict)]``, ``hours`` post-warmup each.

    ``data_dict`` keys: ``bg_observed`` / ``total_carb`` / ``total_insulin`` / ``total_exercise`` /
    ``hour_of_day`` / ``day``, plus ``insulin_resistance`` / ``hgo``, which are neither input nor output.
    """
    runs = []
    for s in seeds:
        sim = _make_simulator(int(s), uniform_skills=False)
        runs.append((f"sim{int(s)}", simulate_discard_warmup(sim, float(hours))))
    return runs


SIM_DATASET = 'sim'
# Fixed epoch: the simulator has no wall clock, only ``hour_of_day``. A Segment clocks off ``t0``, so the
# epoch is offset by the run's opening hour and ``Segment.hour_of_day()`` reproduces it step for step.
SIM_EPOCH = datetime(2024, 1, 1, 0, 0, 0)


def run_to_segment(pid: str, d: dict) -> Segment:
    """One simulator run as a ``Segment``, for the consumers that take Segments.

    ``total_carb`` / ``total_insulin`` are already appearance and action curves and ride in the pre-resolved
    ``carb_curve`` / ``insulin_curve``, so ``features.segment_to_channels`` returns them verbatim and this
    Segment feeds ``build_feature_stack`` what ``build_sim_feature_stack`` does.
    ``exercise`` takes ``total_exercise`` UNRESCALED — the trained scale is g/step carbohydrate-equivalent —
    and this is the only source in the suite that fills the column.
    THE EVENT CHANNELS ARE EMPTY, a property of the source: ``carb_grams`` / ``bolus_units`` / ``basal_rate``
    are raw events, and every meal and bolus is already convolved into the curves above, unrecoverable.
    A consumer that strips or counts events must detect that and refuse, not read the zeros as "nothing".
    """
    bg = np.clip(np.asarray(d['bg_observed'], dtype=np.float64),
                 BG_CLAMP_MIN, BG_CLAMP_MAX)
    n = len(bg)
    assert np.isfinite(bg).all(), f"{pid}: simulator emitted a non-finite BG"
    zeros = lambda: np.zeros(n, dtype=np.float64)          # noqa: E731 — event channels, see docstring
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
    """Fresh simulator patients as ``Segment``s, one per seed; ``seeds`` defaults to the disjoint ``TEST_SEEDS``."""
    return [run_to_segment(pid, d)
            for pid, d in make_sim_runs(TEST_SEEDS if seeds is None else seeds, hours)]


def build_sim_feature_stack(d: dict, stats: dict) -> np.ndarray:
    """Normalized ``(N, N_INPUT_FEATURES)`` input stack from a simulator run.

    ``normalization.normalize`` per ``stats``: bg (feat 0) Kovatchev BEFORE the z-score, the sparse three
    log1p — the SOLE input path, mirroring ``metrics.core.features`` and ``data._build_sample``.
    Feat 3 is the simulator's carbohydrate-EQUIVALENT disposal in g/step, the scale the stats were fit on:
    never risk-transformed, never rescaled to an intensity.
    Signals go in RAW post-noise, with ``data._build_sample``'s clamps: bg to the physical range, the sparse
    three floored at 0, no smoothing.
    Feat ``BG_MASKED_FEAT`` is the announcement bit — no statistics, never through ``normalize``, 0.0
    throughout, since every step here is OBSERVED.
    """
    bg = d['bg_observed'].astype(np.float64)
    n = len(bg)
    # width, not names: normalization.py holds the single copy of those.
    # Normalized channels are 0..BG_MASKED_FEAT-1, mask bit above: the stack is one column wider.
    assert len(CHANNEL_NAMES) == BG_MASKED_FEAT < N_INPUT_FEATURES, (
        f"unexpected CHANNEL_NAMES {list(CHANNEL_NAMES)} against "
        f"BG_MASKED_FEAT={BG_MASKED_FEAT}, N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )
    raw = np.empty((n, len(CHANNEL_NAMES)), dtype=np.float64)
    raw[:, 0] = np.clip(bg, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)
    raw[:, 1] = np.clip(d['total_carb'].astype(np.float64), 0.0, None).astype(np.float64)
    raw[:, 2] = np.clip(d['total_insulin'].astype(np.float64), 0.0, None).astype(np.float64)
    raw[:, 3] = np.clip(d['total_exercise'].astype(np.float64), 0.0, None).astype(np.float64)
    # normalize() owns the per-channel transform and z-score: the single source of truth for the input path
    feats = np.zeros((n, N_INPUT_FEATURES), dtype=np.float32)
    feats[:, :BG_MASKED_FEAT] = normalize(raw, stats)
    return feats


def collect_sim_windows(model, stats, runs, device, stride_patches: int = 8,
                        max_per_patient: int | None = None,
                        announce: tuple[int, ...] = (0, 1, 2), seed: int = 0,
                        conformal_delta=None) -> list[Window]:
    """Slide windows across each simulated patient, capturing median BG and the RAW mg/dL fan per ``Window``.

    Always conditioned: each window announces its true future carb/insulin/exercise.
    Forecast protocol, right-edge span, so downstream numbers bin on the one-sided ``FORECAST_D``.
    ``Window.last_bg`` is the span's anchor, the last visible step — left-preferring, so NOT the ``d`` these
    metrics bin on; for a right-edge span the two coincide only at d = 1.
    ``conformal_delta`` is taken for call-symmetry and DELIBERATELY not applied: ``run_eval`` owns the CQR
    fit/apply on a held-out split, so pre-calibrating here would double-calibrate. The figure path
    (``collect_sim_rows``) is where the stored sim delta legitimately applies.
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

    Always conditioned. Forecasts roll to ``FIG_STEPS``, the night long horizon; ``true`` is NaN-padded to
    that length past a segment end and the plotters drop the unfilled tail.
    ``conformal_delta``, the stored sim delta, is ``(PRED_STEPS, N_QUANTILES)`` and applies only in the
    single-pass branch (``FIG_ROLLS == 1``), leaving ``row['bands']`` calibrated with the median untouched;
    rolling cannot span ``FIG_STEPS``, so bands stay None there. ``None`` ⇒ raw bands.
    Each row also carries the time-of-day probe at the origin — ``time_probs`` ``(P, TIME_PROBE_N_BINS)``,
    ``pred_hour`` / ``tod_R`` decoded from patch 0, ``true_hour`` from the simulator — NaN or None with the
    probe off. Read-only: the BG forecast is bit-identical to the probe-off path.
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
                # the (PRED_STEPS, K) delta cannot span FIG_STEPS, so bands stay None and no ribbon is drawn
                fn = _make_night_overrides_fn(feats, ps, announce, stats)
                out = predict_rolling(model, ctx, n_rolls=FIG_ROLLS,
                                      normalization_stats=stats, device=device,
                                      overrides_fn=fn, return_time=True)
                pred = out['pred_bg'].detach().cpu().numpy()[:FIG_STEPS]
            tr = cgm[ps:ps + FIG_STEPS]
            if len(tr) < FIG_STEPS:
                tr = np.concatenate([tr, np.full(FIG_STEPS - len(tr), np.nan)])
            # probe rides the SAME forward, so the origin clock decodes from patch 0 with no extra pass;
            # NaN / None with the probe off
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
