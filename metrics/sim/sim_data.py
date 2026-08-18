"""
Simulator-cohort bridge for the in-domain `metrics/sim/` report.

Fresh patients are drawn from the T1DMSIM behavioral simulator at chosen seeds
(each seed → a distinct random patient), run through the same warmup-discard the
training pipeline uses, and turned into the model's ``N_INPUT_FEATURES``-feature
input stack ``[bg_absolute, carbs, insulin, exercise, bg_masked]`` — four
normalized signal channels plus the per-patch ``bg_masked`` announcement bit,
which is 0.0 throughout a stack of observed readings. Unlike
a source of logged events, the simulator emits the model's signal channels DIRECTLY —
``total_carb`` / ``total_insulin`` / ``total_exercise`` are the post-noise per-step
carb / insulin / carbohydrate-equivalent exercise disposal the normalization stats
were fit on (``data._build_sample`` consumes them verbatim) — so no
absorption/action kernel reconstruction is needed.

This is the model's TRAINING distribution, so the report is an in-domain
reference, not a peer comparison.

``run_to_segment`` / ``make_sim_segments`` expose the same runs as ``Segment``s,
for the consumers built on ``metrics.core.schema`` rather than on this module's
``Window``. The exercise column is the reason one exists: a source of bare logged
events carries no session in the channel's units, so the simulator is the only
source that announces one at all, and any probe of the exercise channel needs it.

Risk-space redesign note: the model now outputs ONLY a BG quantile forecast (no
carb/insulin/IS/HGO dynamics channels and no trend head), so the former
``bg_delta`` input channel, the IS/HGO mean-imputation, and the four
dynamics-channel ground-truth overlays in the day figures are gone. The day
figure keeps the BG true-vs-predicted comparison only.
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

    Checked rather than left to read correctly: an announced set short of
    ``CHANNEL_TO_FEAT`` leaves the dropped slot at ``normalize(0)``, which for
    exercise_equiv is a legal "no session" value (−0.139 z on the balanced pool),
    so a window trained on an announced session is scored as if none happened and
    nothing about the resulting table looks wrong.
    """
    assert tuple(announce) == tuple(CHANNEL_TO_FEAT), (
        f"announced set {tuple(announce)} != announceable set {tuple(CHANNEL_TO_FEAT)}")


def _smooth_sim_bg(bg_raw: np.ndarray) -> np.ndarray:
    """RAW simulator BG, bg-clamped (the ground truth every metric scores against).

    The pipeline consumes raw post-noise signals (``data._build_sample``): every
    scored ground-truth BG is the raw ``bg_observed`` that produced the model input,
    physical-range clamped. Name kept for its call-sites; no FIR smoothing applied.
    """
    return np.clip(np.asarray(bg_raw, dtype=np.float64),
                   BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)

# Every window this module builds runs the FORECAST protocol: one masked span of
# ``PREDICTION_PATCHES`` patches ending at the last patch of the window, with the
# whole context visible.  That is one case of the masked-BG objective, not a mode
# of its own — the same objective covers a span at patch 0 (backcast) and a span
# between visible patches (infill), neither of which this module scores.
# ``CTX_STEPS`` is the visible prefix and ``PRED_STEPS`` the masked span, in steps.
CTX_STEPS = MAX_CONTEXT_PATCHES * PATCH_SIZE
PRED_STEPS = PREDICTION_PATCHES * PATCH_SIZE
# Parity/clarke forecasts roll out to the night long horizon for hour-by-hour
# figure granularity (mirrors metrics.core.report.FIG_ROLLS); single pass when unset.
FIG_ROLLS = max(1, math.ceil(NIGHT_LONG_HORIZON_HOURS / PREDICTION_HORIZON_HOURS))
FIG_STEPS = FIG_ROLLS * PRED_STEPS

# The forecast protocol's masked set, expanded by the single definition in
# ``data._mask_slots`` rather than restated here.  ``FORECAST_D`` is the distance
# in patches from each masked patch to the nearest visible evidence ON EITHER
# SIDE — the only quantity a masked-BG metric is binned on (never span length,
# which confounds one-sided and two-sided cases at equal difficulty, and never
# arm).  A right-edge span has no right neighbour, so the distance is one-sided
# and slot j sits at d = j + 1: the 30 / 60 / 90 / 120 min horizons ARE d = 1..4.
_fc_idx, _fc_valid, _fc_d, _fc_anchor_step = _mask_slots(
    [(MAX_CONTEXT_PATCHES, PREDICTION_PATCHES)],
    MAX_CONTEXT_PATCHES + PREDICTION_PATCHES,
)
FORECAST_D = tuple(int(x) for x in _fc_d[_fc_valid])
assert FORECAST_D == tuple(range(1, PREDICTION_PATCHES + 1)), (
    f"right-edge forecast d {FORECAST_D} is not one-sided 1..{PREDICTION_PATCHES}")
# The anchor is ONE-SIDED and LEFT-PREFERRING: every slot of a span takes the last
# step of the left neighbour (the first step of the right neighbour only when the
# span starts at patch 0), so it IGNORES the near side for a span's right-half
# slots.  ``Window.last_bg`` below is that step for the forecast protocol.
assert int(_fc_anchor_step[_fc_valid][0]) == CTX_STEPS - 1, (
    "forecast anchor is not the last visible step")

# Fixed (arbitrary, reproducible) seed pools — each seed draws a distinct random
# patient; calibration and test pools are disjoint.
# Post-warmup trajectory length per patient. It has to clear the context window
# before a single window exists at all: one forecast window costs
# MAX_CONTEXT_PATCHES + PREDICTION_PATCHES = 340 patches (170 h), so a shorter
# run yields ZERO windows and an empty report rather than an error. 288 h is the
# length at which the collectors' own caps bind exactly — 60 test windows per
# patient at stride 4, and the 24-window calibration cap at stride 8 — and it
# also clears curves_sim's MAX_CONTEXT_PATCHES + DAY_PATCHES day figure.
DEFAULT_HOURS = 288.0
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
        / ``total_insulin`` / ``total_exercise`` / ``hour_of_day`` / ``day`` (the
        dynamics channels ``insulin_resistance`` / ``hgo`` remain in the dict but
        are no longer model inputs or outputs).
    """
    runs = []
    for s in seeds:
        sim = _make_simulator(int(s), uniform_skills=False)
        runs.append((f"sim{int(s)}", simulate_discard_warmup(sim, float(hours))))
    return runs


SIM_DATASET = 'sim'
# Arbitrary but fixed grid epoch: the simulator has no wall clock, only an
# ``hour_of_day`` array. A Segment derives its clock from ``t0``, so the epoch is
# offset by the run's own opening hour and ``Segment.hour_of_day()`` then
# reproduces ``d['hour_of_day']`` step for step.
SIM_EPOCH = datetime(2024, 1, 1, 0, 0, 0)


def run_to_segment(pid: str, d: dict) -> Segment:
    """One simulator run as a ``Segment``, for the consumers that take Segments.

    The simulator emits the model's SIGNAL CHANNELS directly — ``total_carb`` /
    ``total_insulin`` are already the appearance / action curves, and
    ``total_exercise`` the carbohydrate-equivalent glucose-disposal curve in
    g/step — so they are carried in the Segment's pre-resolved ``carb_curve`` /
    ``insulin_curve`` fields and its ``exercise`` field, and
    ``features.segment_to_channels`` returns them verbatim rather than
    reconstructing them through its kernels.  A Segment built this way therefore
    feeds ``build_feature_stack`` the same numbers ``build_sim_feature_stack``
    does.

    ``exercise`` is populated from ``total_exercise`` UNRESCALED: the trained
    scale is g/step carbohydrate-equivalent, so any conversion here would train
    one quantity and probe another.  This is the only source in the suite that
    fills the column — all five real adapters write zeros.

    THE EVENT CHANNELS ARE EMPTY, and that is a property of the source rather than
    of the run.  ``carb_grams`` / ``bolus_units`` / ``basal_rate`` are the RAW
    events a real adapter parses, and the simulator keeps no such record: by the
    time a run is returned, every meal and bolus has already been convolved into
    the curves above and cannot be recovered from them.  A consumer that strips or
    counts events (``metrics.whatif``'s empty-future arm, its quiet-window test)
    must detect that and refuse, not read the zeros as "nothing happened".
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
    """Fresh simulator patients as ``Segment``s — one per seed.

    ``seeds`` defaults to ``TEST_SEEDS``, disjoint from the calibration pool, so a
    caller that wants unseen patients and nothing else can pass ``hours`` alone.
    """
    return [run_to_segment(pid, d)
            for pid, d in make_sim_runs(TEST_SEEDS if seeds is None else seeds, hours)]


def build_sim_feature_stack(d: dict, stats: dict) -> np.ndarray:
    """Normalized (N, N_INPUT_FEATURES) input stack from a simulator run.

    The channels (bg_observed, carb, insulin, exercise — the trimmed
    ``CHANNEL_NAMES``) are normalized per ``stats`` via ``normalization.normalize``,
    so bg (feat 0) takes the Kovatchev risk transform BEFORE the z-score and the
    sparse carb/insulin/exercise take log1p — the SOLE input path, mirroring
    ``metrics.core.features.build_feature_stack`` / ``data._build_sample`` (the redesign
    dropped ``bg_delta``, the IS/HGO latents, and the four sin/cos temporal
    features from the stack).

    ``total_exercise`` (feat 3) is the simulator's carbohydrate-EQUIVALENT glucose
    disposal in g/step — the quantity the normalization stats were fit on. It is
    never risk-transformed (it is not a glucose) and never rescaled to an
    intensity; the trained scale is g/step.

    The model consumes RAW post-noise signals: every signal channel (bg feat0,
    carb feat1, insulin feat2, exercise feat3) is used raw BEFORE normalization,
    with the SAME clamps as ``data._build_sample`` (bg → the physical BG range; the
    sparse carb/insulin/exercise floored at 0; no FIR smoothing).

    Feat ``BG_MASKED_FEAT`` is the ``bg_masked`` announcement bit, not a signal: it
    carries no statistics and never crosses ``normalize``.  Every step of this
    stack is an OBSERVED reading, so the column is 0.0 throughout; the masked set
    is written into the patches downstream, by the builder that knows it.
    """
    bg = d['bg_observed'].astype(np.float64)
    n = len(bg)
    # Length check, not a name list: the names themselves live in normalization.py
    # (the single copy), and what this stack must agree with is the width it fills.
    # The normalized channels occupy columns 0..BG_MASKED_FEAT-1 and the mask bit
    # sits above them, so the stack is one column WIDER than CHANNEL_NAMES.
    assert len(CHANNEL_NAMES) == BG_MASKED_FEAT < N_INPUT_FEATURES, (
        f"unexpected CHANNEL_NAMES {list(CHANNEL_NAMES)} against "
        f"BG_MASKED_FEAT={BG_MASKED_FEAT}, N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )
    raw = np.empty((n, len(CHANNEL_NAMES)), dtype=np.float64)
    raw[:, 0] = np.clip(bg, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)
    raw[:, 1] = np.clip(d['total_carb'].astype(np.float64), 0.0, None).astype(np.float64)
    raw[:, 2] = np.clip(d['total_insulin'].astype(np.float64), 0.0, None).astype(np.float64)
    raw[:, 3] = np.clip(d['total_exercise'].astype(np.float64), 0.0, None).astype(np.float64)
    # normalize() owns the per-channel forward transform (risk-f on bg, log1p on
    # the sparse channels) + z-score — the single source of truth for the input path.
    feats = np.zeros((n, N_INPUT_FEATURES), dtype=np.float32)
    feats[:, :BG_MASKED_FEAT] = normalize(raw, stats)
    return feats


def collect_sim_windows(model, stats, runs, device, stride_patches: int = 8,
                        max_per_patient: int | None = None,
                        announce: tuple[int, ...] = (0, 1, 2), seed: int = 0,
                        conformal_delta=None) -> list[Window]:
    """Slide prediction windows across each simulated patient and capture the
    headline median BG forecast — the simulator analogue of
    ``metrics.core.calibrate.collect_windows`` (same trimmed ``Window`` of
    ``pred_bg`` = ``median_bg`` + true CGM, now also the RAW mg/dL band fan). The
    model is always conditioned: each window announces its true future
    carbs/insulin/exercise.

    Every window runs the forecast protocol — a masked span at the right edge —
    so the per-horizon numbers downstream are binned on ``FORECAST_D``, one-sided.
    ``Window.last_bg`` is the span's anchor, the last visible step; the anchor is
    left-preferring and ignores the near side, so it is NOT the distance ``d``
    these metrics bin on (for a right-edge span the two coincide only at d = 1).

    NOTE on ``conformal_delta``: it is accepted only for call-symmetry and is
    DELIBERATELY NOT used to pre-calibrate the captured bands. The metric CQR re-fit
    in ``run_eval.evaluate_from_windows`` owns the fit/apply on a held-out split, so
    the bands stored here MUST stay RAW — pre-applying the stored sim delta here
    would double-calibrate. The FIGURE path (``collect_sim_rows``) is where the
    stored sim delta is legitimately applied for display."""
    del conformal_delta  # never applied here (see docstring): bands captured RAW
    _assert_announces_all(announce)
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
                     announce: tuple[int, ...] = (0, 1, 2), cap: int = 24,
                     conformal_delta=None) -> list[dict]:
    """Per-window conditional BG forecasts for the trajectory/parity/clarke figures.

    The model is always conditioned: each window announces its true future
    carbs/insulin/exercise. Forecasts roll out to ``FIG_STEPS`` (the night long
    horizon) so the parity/clarke panels read hour-by-hour; ``true`` is NaN-padded to that
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
    _assert_announces_all(announce)
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
