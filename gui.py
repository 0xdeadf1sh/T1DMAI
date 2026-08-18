"""
T1DMAI GUI — Interactive visualization and what-if analysis.
=============================================================

What this app is for
--------------------
A pygame-based front end for inspecting model predictions on a per-patient
basis.  It builds a context window from the simulator, runs a forward pass
through a loaded checkpoint, then renders the model's median BG forecast
with a per-τ quantile uncertainty envelope and a draggable cursor.  The
model is risk-space: its head emits ONLY a BG quantile forecast (no carb /
insulin / IS / HGO dynamics outputs, no trend head, no physics
reconstruction).  The user can place 3-point curve events (raised-cosine
bells) for carbs / insulin / exercise into the prediction zone and re-run a
what-if pass: the announced carb / insulin / exercise INPUT is perturbed and
``predict`` is re-run, so the median BG forecast moves in response to the plan
— useful for sanity-checking the model's response to specific meal / bolus /
session plans.

Architecture
------------
* ``gui.py``           — main app, window / event loop, sidebar, all
                           high-level interactions.  Keeps a ``GUIState``
                           and a chart transform.
* ``gui_state.py``     — central mutable state (visibility toggles, active
                           tool, overrides, prediction results, last BG …).
* ``gui_controls.py``  — input widgets for the meal / bolus
                            builders.
* ``gui_renderer.py``  — chart drawing primitives (gridlines, curves,
                           confidence bands, hover overlay).
* ``inference.py``     — model entry points (``predict``,
                           ``predict_what_if``, ``predict_rolling``).

Threading
---------
Predictions run on a background thread so the UI stays responsive during a
forward pass.  ``state.is_computing`` flips to True while inference is in
flight; the main loop reads it to skip redrawing the prediction layer
until the worker writes ``median_bg`` / ``bands`` back into
``state.prediction``.

Channel layout convention
-------------------------
* Display channels (what the user sees on screen): four entries —
  ``[BG, Carbs, Insulin, Exercise]``.  BG is the model's median forecast;
  carbs, insulin and exercise are the announced what-if inputs the user can
  paint.  Exercise is painted in g/step carbohydrate-equivalent glucose
  disposal, the scale the model is trained on.
* The model emits no dynamics channels.  The green BG curve drawn on the
  chart is ``predict``'s ``median_bg``; the shaded envelope is its ``bands``
  (per-τ quantile edges in mg/dL).

The mapping table ``DISPLAY_TO_OUTPUT_CH`` translates display → announced
output-channel index (carbs → 0, insulin → 1, exercise → 2).  BG has no
announced channel.

Usage::

    python gui.py --checkpoint checkpoints/t1dmai_best.pt --seed 42
    python gui.py --no-model    # UI testing with random weights

Controls::

    SPACE/Enter   Run prediction (respecting curve overrides)
    W / E         Toggle curve editor (a.k.a. what-if mode) — both keys toggle
    Tab           Cycle selected edit channel
    F             Roll prediction forward (one model horizon per press)
    G             Step the simulator forward (extends real ground-truth)
    V             Eval vs Sim — score the current prediction against the simulator
    R             Reset overrides
    C             Clear all curves (blank canvas)
    N             New random patient
    M             Meal builder
    B             Bolus builder
    S             Screenshot
    1-4           Toggle curve visibility
    A             Toggle all curves
    +/-           Zoom in/out
    Scroll        Zoom around cursor
    Middle/Right-drag  Pan
    Delete        Remove selected curve event
    Ctrl+Z        Undo
    Q/ESC         Quit
"""

import os
import sys
import copy
import argparse
import math
import time
import threading
import traceback
import datetime

if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
    os.environ['SDL_VIDEODRIVER'] = 'x11'

from typing import Any
import numpy as np
import torch

from config import BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD, PATCH_SIZE

from gui_renderer import ui_px  # used by the constants below

WINDOW_WIDTH = ui_px(1600)
WINDOW_HEIGHT = ui_px(900)
SIDEBAR_WIDTH = ui_px(320)
RIGHT_PANEL_WIDTH = ui_px(280)
CHART_PADDING = ui_px(40)
CONTROL_PANEL_HEIGHT = ui_px(200)
STATUS_BAR_HEIGHT = ui_px(28)
FPS = 30
FONT_SIZE = ui_px(14)
FONT_SIZE_SMALL = ui_px(11)
FONT_SIZE_LARGE = ui_px(18)

# Layout primitives — design-time values scaled at import. Reused by the
# sidebar layout (`_draw_sidebar`) and by control-panel widgets so the
# whole interface scales with ``UI_SCALE``.
TOGGLE_HEIGHT = ui_px(32)
TOGGLE_ROW_PITCH = ui_px(28)
BUTTON_HEIGHT = ui_px(34)
BUTTON_ROW_PITCH = ui_px(44)
LINE_GAP = ui_px(18)
LINE_GAP_SM = ui_px(15)
HEADER_GAP = ui_px(22)
HEADER_GAP_LG = ui_px(24)
SECTION_GAP = ui_px(10)
SECTION_GAP_SM = ui_px(6)
BG_COLOR = (18, 18, 24)
CHART_BG_COLOR = (28, 28, 36)
SIDEBAR_BG_COLOR = (22, 22, 30)
GRID_COLOR = (48, 48, 56)
TEXT_COLOR = (225, 228, 240)
TEXT_DIM_COLOR = (140, 145, 165)
TEXT_FAINT_COLOR = (105, 110, 130)
CURSOR_COLOR = (255, 255, 255)
# Clock-face overlay cosmetics (the time-of-day probe's circular histogram,
# rotated to the cursor's elapsed offset). Model/training knobs live in
# config.py; these are purely GUI presentation, so they live here.
CLOCK_FACE_RADIUS_PX = ui_px(52)
CLOCK_FACE_MARGIN_PX = ui_px(12)
CLOCK_FACE_BG_COLOR = (32, 34, 48)
CLOCK_WEDGE_COLOR = (90, 150, 230)
CLOCK_HAND_COLOR = (245, 245, 250)
CLOCK_TICK_COLOR = (120, 125, 145)
CONFIDENCE_ALPHA = 40
# Subtle blue tint over the model's input patches. Mirrors the darker
# overlay on the prediction side of NOW so the two halves are visually
# distinct without overpowering the curves.
CONTEXT_SHADE_COLOR = (80, 130, 200)
CONTEXT_SHADE_ALPHA = 20

# Sidebar palette — soft dark panels with colored section accents.
SIDEBAR_PANEL_BG = (32, 34, 48)
SIDEBAR_PANEL_BORDER = (56, 60, 80)
ACCENT_PATIENT = (110, 170, 245)
ACCENT_CHANNELS = (120, 200, 140)
ACCENT_PREDICTION = (245, 175, 95)
ACCENT_ACTIONS = (190, 140, 230)
ACCENT_OVERRIDES = (240, 120, 130)
ACCENT_EVAL = (90, 210, 220)
MODE_PILL_STANDARD = ((28, 42, 62), (110, 170, 245))      # (bg, fg)
MODE_PILL_WHATIF = ((58, 44, 26), (245, 175, 95))
MODE_PILL_OTHER = ((42, 42, 56), (180, 185, 210))
TIR_IN_RANGE_COLOR = (90, 200, 130)
TIR_HIGH_COLOR = (240, 180, 90)
TIR_LOW_COLOR = (230, 100, 100)

# Chart background shading for glucose ranges. Thresholds are the clinical
# hypo/hyper cutoffs ``BG_HYPO_THRESHOLD`` / ``BG_HYPER_THRESHOLD`` (config is
# the single source of truth). Alpha is kept low so the shading is a subtle
# visual cue and doesn't fight the curves on top.
HYPO_BAND_COLOR = TIR_LOW_COLOR
IN_RANGE_BAND_COLOR = TIR_IN_RANGE_COLOR
HYPER_BAND_COLOR = TIR_HIGH_COLOR
GLUCOSE_BAND_ALPHA = 28
HYPO_THRESHOLD_MGDL = BG_HYPO_THRESHOLD
HYPER_THRESHOLD_MGDL = BG_HYPER_THRESHOLD

COLOR_BG_CURVE = (70, 180, 70)
COLOR_CARBS = (230, 160, 50)
COLOR_INSULIN = (80, 140, 240)
COLOR_EXERCISE = (190, 125, 235)

PREDICTION_LINE_ALPHA = 200
OVERRIDE_LINE_COLOR = (255, 80, 80)
OVERRIDE_POINT_RADIUS = ui_px(6)
CURVE_EVENT_FILL_ALPHA = 60
CONTROL_POINT_HIT_RADIUS = ui_px(12)

# ---- Free-form masking ----
# A masked span hides the true BG under it: the model is being asked to fill
# that stretch, so leaving the curve drawn there invites reading the answer off
# the chart. The overlay is what marks the span; the context curve is clipped
# out of it in _draw_chart.
MASK_SPAN_COLOR = (150, 120, 210)
MASK_SPAN_ALPHA = 46
MASK_SPAN_SELECTED_ALPHA = 78
MASK_DRAG_ALPHA = 30
MASK_EDGE_COLOR = (185, 160, 240)
MASK_OOD_COLOR = (240, 175, 70)
# Buttons that write an announced dose; disabled under a blind checkpoint.
_DOSE_PAINTING_BUTTONS = frozenset({
    "What-If", "Pencil", "Basal +1 U/h", "Basal -1 U/h",
})

# Visual smoothing windows applied to the predicted curves, in timesteps
# (5 min each). Both are purely cosmetic — the underlying prediction is
# unchanged. The σ envelope gets a wider window because edge wobble is
# what makes the band look "spiky"; the mean line uses a tighter window
# so real trend changes still come through. Both can be toggled off via
# the sidebar checkboxes (state.smooth_band, state.smooth_mu).
CONFIDENCE_BAND_SMOOTH_STEPS = 25
MU_SMOOTH_STEPS = 13

# The four display channels: BG (the model's median forecast) and the three
# announced what-if inputs (carbs, insulin, exercise). Their stats-names match
# the input feature stack
# [bg_absolute, carb_intake, insulin_combined, exercise_equiv].
OUTPUT_CHANNEL_ORDER = ['carb_intake', 'insulin_combined', 'exercise_equiv']

# Number of display channels (BG + carbs + insulin + exercise).
N_DISPLAY_CHANNELS = 4

# Per-channel y-axis span for the painted / drawn curves, in the channel's own
# raw unit. Exercise shares carb's span: it is a carbohydrate-EQUIVALENT
# glucose-disposal rate in the same grams-per-step unit, so the two read on one
# scale.
DISPLAY_CHANNEL_RAW_RANGES: list[tuple[float, float]] = [
    (0.0,     400.0),
    (0.0,     10.0),
    (0.0,     1.0),
    (0.0,     10.0),
]

DISPLAY_TO_STATS_NAME: list[str] = [
    'bg_absolute', 'carb_intake', 'insulin_combined', 'exercise_equiv',
]

# Display channel → input feature slot. The 4-feature input stack is
# [bg_absolute, carb_intake, insulin_combined, exercise_equiv], so
# BG/carb/insulin/exercise sit at feats 0/1/2/3 (all but BG are announceable).
DISPLAY_TO_FEATURE_IDX: list[int] = [0, 1, 2, 3]

# Display channel → announced output-channel index (carbs → 0, insulin → 1,
# exercise → 2). BG (display 0) is the model's forecast, never an announced
# input.
DISPLAY_TO_OUTPUT_CH: dict[int, int] = {1: 0, 2: 1, 3: 2}

# Announced output-channel index → display channel (the inverse of
# DISPLAY_TO_OUTPUT_CH); used to route a pencil stroke's display channel back
# through the output-channel override compiler.
OUTPUT_TO_DISPLAY_CH: dict[int, int] = {v: k for k, v in DISPLAY_TO_OUTPUT_CH.items()}

DISPLAY_CHANNEL_CLEAR_DEFAULTS: list[float] = [100.0, 0.0, 0.0, 0.0]

# Short display names per announced output channel for status messages.
OUTPUT_CHANNEL_SHORT_NAMES: list[str] = ['carbs', 'insulin', 'exercise']

# The one announced channel the basal ramp writes into.
INSULIN_OUTPUT_CH: int = OUTPUT_CHANNEL_ORDER.index('insulin_combined')

# Exercise is announced at the trained g/step scale — a carbohydrate-equivalent
# glucose disposal rate, not a 0-1 intensity. Any conversion belongs to whoever
# supplies the session, never to the display.
DISPLAY_CHANNEL_UNITS: list[str] = [
    'mg/dL',
    'g/5min',
    'U/5min',
    'g/step',
]

SCROLL_SPEED = 48
ZOOM_FACTOR = 1.3
# Hours of the context window the chart opens on. The window itself is days
# wide; drawn whole, neither the CGM trace nor the forecast at its right edge is
# legible, and the curve editor's control points land within a pixel of each
# other. Zoom and pan reach the rest.
CHART_VIEW_HOURS = 24.0
HOVER_TOOLTIP_DELAY_MS = 200

# One colour per display channel — indexed by ``disp_ch`` over
# ``range(N_DISPLAY_CHANNELS)`` on every draw path, so it stays the same length
# as the table above.
CHANNEL_COLORS = [
    COLOR_BG_CURVE, COLOR_CARBS, COLOR_INSULIN, COLOR_EXERCISE,
]
assert len(CHANNEL_COLORS) == N_DISPLAY_CHANNELS


def _safe_import_pygame():
    try:
        import pygame
        return pygame
    except ImportError:
        print("ERROR: pygame is not installed. Run: pip install pygame")
        sys.exit(1)


def _features_from_raw(
    raw: dict,
    norm_stats: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a simulator raw-dict chunk into the normalized feature
    stack and per-step bookkeeping arrays the GUI keeps around.

    The risk-space model takes an ``N_INPUT_FEATURES``-feature input stack
    ``[bg_absolute, carb_intake, insulin_combined, exercise_equiv, bg_masked]``
    — four normalized signal channels plus the per-patch ``bg_masked``
    announcement bit — with no bg_delta / IS / HGO input and no temporal sin/cos
    columns.  Every step of this stack is an OBSERVED reading, so the bit column
    is 0.0 throughout; the masked set is written into the patches downstream, by
    the builder that knows it.  bg (feat 0)
    is fed through the Kovatchev risk transform ``f`` BEFORE the z-score
    (``normalize`` applies it via ``RISK_SPACE_CHANNELS``); carb / insulin /
    exercise keep log1p + z.  ``total_exercise`` is the carbohydrate-EQUIVALENT
    glucose-disposal curve in g/step (the simulator subtracts it from the
    appearance term), so it is fed at its trained g/step scale and never
    rescaled to an intensity, and never through the risk transform — it is not
    a glucose.

    Returns (features_norm[N, N_INPUT_FEATURES], bg_raw[N], context_raw[N,4])
    where N is trimmed to a multiple of PATCH_SIZE.  ``context_raw`` stays at the
    four signal channels — it is the chart's data, not the model's input.
    """
    from config import PATCH_SIZE, N_INPUT_FEATURES
    from data import BG_MASKED_FEAT
    from normalization import CHANNEL_NAMES, normalize
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

    # Use bg_observed (post-CGM-noise) so the GUI matches what the model was
    # trained on.  The model consumes RAW post-noise signals: bg is only clamped
    # to the physical BG range, carb/insulin/exercise floored at 0 (mirroring
    # ``data._build_sample`` — no smoothing).  ``bg_raw`` / ``context_raw`` below
    # stay unclamped for the chart; only the model-input ``features`` are clamped.
    bg_obs_raw = raw['bg_observed'].astype(np.float32)
    carb_raw = raw['total_carb'].astype(np.float32)
    insulin_raw = raw['total_insulin'].astype(np.float32)
    exercise_raw = raw['total_exercise'].astype(np.float32)
    bg_obs = np.clip(bg_obs_raw, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.clip(carb_raw, 0.0, None).astype(np.float32)
    insulin = np.clip(insulin_raw, 0.0, None).astype(np.float32)
    exercise = np.clip(exercise_raw, 0.0, None).astype(np.float32)
    N = (len(bg_obs) // PATCH_SIZE) * PATCH_SIZE
    bg_obs = bg_obs[:N]; carb = carb[:N]; insulin = insulin[:N]
    exercise = exercise[:N]
    bg_obs_raw = bg_obs_raw[:N]; carb_raw = carb_raw[:N]; insulin_raw = insulin_raw[:N]
    exercise_raw = exercise_raw[:N]

    # The displayed/plotted CGM + carb + insulin + exercise stay RAW (chart
    # realism); only the model-input ``features`` carry the clamp + normalization.
    bg_raw = bg_obs_raw.copy()
    context_raw = np.stack(
        [bg_obs_raw, carb_raw, insulin_raw, exercise_raw], axis=-1,
    ).astype(np.float32)

    # Normalize the four clamped signal channels through the shared transform:
    # bg (feat 0) via the Kovatchev risk transform BEFORE z (RISK_SPACE_CHANNELS),
    # the sparse carb/insulin/exercise via log1p + z.  One CHANNEL_NAMES entry per
    # SIGNAL channel, in stack order; the bg_masked bit above them is not a signal
    # and carries no statistics, so the stack is one column wider than CHANNEL_NAMES.
    assert len(CHANNEL_NAMES) == BG_MASKED_FEAT < N_INPUT_FEATURES, (
        f"CHANNEL_NAMES has {len(CHANNEL_NAMES)} entries against "
        f"BG_MASKED_FEAT={BG_MASKED_FEAT}, N_INPUT_FEATURES={N_INPUT_FEATURES}: "
        f"{list(CHANNEL_NAMES)}"
    )
    signal = np.stack([bg_obs, carb, insulin, exercise], axis=-1).astype(np.float32)
    features = np.zeros((len(bg_obs), N_INPUT_FEATURES), dtype=np.float32)
    features[:, :BG_MASKED_FEAT] = normalize(signal, norm_stats)

    return features, bg_raw, context_raw


def default_context_hours() -> float:
    """Post-warmup simulator hours that fill the model's context window.

    ``MAX_CONTEXT_PATCHES`` is the ceiling the model was trained to, and the
    forward accepts anything from ``MIN_CONTEXT_PATCHES`` up to it. Bootstrapping
    short of the floor is the one failure the GUI cannot show you: nothing
    raises, the fan is simply drawn from a window the weights never saw.
    """
    from config import MAX_CONTEXT_PATCHES, PATCH_SIZE
    return MAX_CONTEXT_PATCHES * PATCH_SIZE * 5.0 / 60.0


def _build_context_from_sim(
    seed: int,
    norm_stats: dict,
    context_hours: float | None = None,
) -> tuple[torch.Tensor, dict, np.ndarray, np.ndarray, float, int, Any]:
    from T1DMSIM.simulator import T1DMSimulator
    from data import simulate_discard_warmup
    from config import MIN_CONTEXT_PATCHES, PATCH_SIZE, N_INPUT_FEATURES

    hours = default_context_hours() if context_hours is None else float(context_hours)
    sim = T1DMSimulator(seed=seed)
    raw = simulate_discard_warmup(sim, hours)
    features, bg_raw, context_raw = _features_from_raw(raw, norm_stats)
    n_patches = features.shape[0] // PATCH_SIZE
    if n_patches < MIN_CONTEXT_PATCHES:
        raise ValueError(
            f"{hours:g} h of simulator gives {n_patches} context patches, under the "
            f"model's MIN_CONTEXT_PATCHES={MIN_CONTEXT_PATCHES} "
            f"({MIN_CONTEXT_PATCHES * PATCH_SIZE * 5.0 / 60.0:g} h). Raise "
            f"--context-hours."
        )

    context = torch.from_numpy(features).reshape(-1, PATCH_SIZE, N_INPUT_FEATURES)
    start_hour = float(raw['hour_of_day'][0]) % 24.0
    start_day = int(raw['day'][0]) % 7
    return (context, sim.get_patient_summary(), bg_raw, context_raw,
            start_hour, start_day, sim)


def _advance_sim_hours(
    state,
    hours: float,
    norm_stats: dict,
) -> int:
    """Step ``state.sim`` forward by ``hours`` real hours and append the
    new ground-truth steps to the state buffers (context tensor, bg_raw,
    context_raw).

    Returns the NET patch shift: appended minus dropped off the left. Once the
    buffer is saturated at ``MAX_CONTEXT_PATCHES`` the two cancel and the return
    is 0 — the window slides rather than grows, and every buffer index, the
    chart viewport included, stays where it was.
    """
    from config import PATCH_SIZE, N_INPUT_FEATURES

    sim = state.sim
    if sim is None or hours <= 0.0:
        return 0

    raw_new = sim.generate_hours(hours)
    features_new, bg_raw_new, context_raw_new = _features_from_raw(
        raw_new, norm_stats,
    )

    n_new_steps = features_new.shape[0]
    n_new_patches = n_new_steps // PATCH_SIZE
    if n_new_patches <= 0:
        return 0

    new_patches = torch.from_numpy(features_new).reshape(
        n_new_patches, PATCH_SIZE, N_INPUT_FEATURES,
    )
    state.context = (
        torch.cat([state.context, new_patches], dim=0)
        if state.context is not None else new_patches
    )
    state.bg_raw = (
        np.concatenate([state.bg_raw, bg_raw_new])
        if state.bg_raw is not None else bg_raw_new
    )
    state.context_raw = (
        np.concatenate([state.context_raw, context_raw_new], axis=0)
        if state.context_raw is not None else context_raw_new
    )

    # Drop from the left so the buffer never outgrows MAX_CONTEXT_PATCHES. The
    # forward reads whatever it is handed and RoPE extrapolates, so an overrun
    # does not raise — it silently forecasts from a window longer than any the
    # model was trained on, and grows by ``hours`` every press. The bootstrap now
    # starts AT the ceiling, so the very first step would overrun without this.
    from config import MAX_CONTEXT_PATCHES
    n_drop = max(0, int(state.context.shape[0]) - MAX_CONTEXT_PATCHES)
    if n_drop:
        state.context = state.context[n_drop:]
        state.bg_raw = state.bg_raw[n_drop * PATCH_SIZE:]
        state.context_raw = state.context_raw[n_drop * PATCH_SIZE:]
        state.sim_start_hour = (
            float(getattr(state, 'sim_start_hour', 0.0)) + n_drop * 0.5) % 24.0
    return n_new_patches - n_drop


def _hour_at_pred_start(state) -> float:
    """Cosmetic hour-of-day (0..24) at the prediction-zone start.

    The temporal sin/cos inputs are gone, so the clock label is derived purely
    from the run's start hour plus the elapsed context: each context patch spans
    ``PATCH_SIZE`` 5-min steps (30 min = 0.5 h), and the prediction zone begins
    immediately after the last context patch.
    """
    start_hour = float(getattr(state, 'sim_start_hour', 0.0))
    ctx = getattr(state, 'context', None)
    n_ctx = 0 if ctx is None else int(ctx.shape[0])
    return (start_hour + n_ctx * 0.5) % 24.0


# Pharmacokinetic shape templates for high-level events. Times in
# minutes from the placement (event time_offset_min). Each entry yields
# one or more raised-cosine bells; amplitudes are back-solved from the
# event's magnitude so the AUC matches grams/units.
#
# Carbs unit is g/5min, insulin is U/5min; a 5-min step's value already
# equals its contribution in grams (resp. units), so total grams =
# sum(vals). For a half raised-cosine of half-span L patches, the
# integral is 0.5 L; total bell integral = 0.5 (L_left + L_right). In
# steps (× patch_size), sum_of_vals = amplitude × 0.5 × (L_l + L_r) × ps.
# Hence amplitude = magnitude / (0.5 × (L_l + L_r) × ps).
EVENT_SHAPES = {
    'juice':         {'rise_min': 15.0, 'fall_min': 30.0,   'channel': 1},
    'fast_insulin':  {'rise_min': 75.0, 'fall_min': 165.0,  'channel': 2},
    'basal_insulin': {'rise_min': 300.0, 'fall_min': 660.0, 'channel': 2},
    'meal_fast':     {'rise_min': 30.0, 'fall_min': 60.0,   'channel': 1},
    'meal_slow':     {'rise_min': 120.0, 'fall_min': 180.0, 'channel': 1},
}


def _make_curve_event(
    channel: int,
    placement_min: float,
    rise_min: float,
    fall_min: float,
    magnitude: float,
    n_ctx: int,
    patch_size: int,
):
    """Build a raised-cosine CurveEvent whose AUC equals ``magnitude``.

    ``placement_min`` is the event's nominal start time in minutes
    relative to the start of the prediction zone (n_ctx)."""
    from gui_state import CurveEvent

    min_per_patch = 30.0
    left_patch  = n_ctx + placement_min / min_per_patch
    peak_patch  = left_patch + rise_min / min_per_patch
    right_patch = peak_patch + fall_min / min_per_patch

    L_left  = max(peak_patch - left_patch,  1e-3)
    L_right = max(right_patch - peak_patch, 1e-3)
    bell_integral_steps = 0.5 * (L_left + L_right) * patch_size
    amplitude = magnitude / bell_integral_steps if bell_integral_steps > 0 else 0.0

    return CurveEvent(
        channel=channel,
        left_patch=left_patch,
        peak_patch=peak_patch,
        right_patch=right_patch,
        amplitude=amplitude,
    )


def _events_to_curve_events(
    events: list,
    n_ctx: int,
    patch_size: int = 6,
) -> list:
    """Expand high-level Events (juice / insulin / meal) into one or
    more CurveEvents suitable for ``_compile_overrides_from_edits``."""
    from gui_state import (
        Event, EVENT_KIND_JUICE, EVENT_KIND_FAST_INSULIN,
        EVENT_KIND_BASAL_INSULIN, EVENT_KIND_MEAL,
    )
    out: list = []
    for ev in events:
        if not isinstance(ev, Event):
            continue
        t = float(ev.time_offset_min)
        mag = float(ev.magnitude)
        if mag <= 0.0:
            continue
        if ev.kind == EVENT_KIND_JUICE:
            shape = EVENT_SHAPES['juice']
            out.append(_make_curve_event(
                shape['channel'], t, shape['rise_min'], shape['fall_min'],
                mag, n_ctx, patch_size,
            ))
        elif ev.kind == EVENT_KIND_FAST_INSULIN:
            shape = EVENT_SHAPES['fast_insulin']
            out.append(_make_curve_event(
                shape['channel'], t, shape['rise_min'], shape['fall_min'],
                mag, n_ctx, patch_size,
            ))
        elif ev.kind == EVENT_KIND_BASAL_INSULIN:
            shape = EVENT_SHAPES['basal_insulin']
            out.append(_make_curve_event(
                shape['channel'], t, shape['rise_min'], shape['fall_min'],
                mag, n_ctx, patch_size,
            ))
        elif ev.kind == EVENT_KIND_MEAL:
            fast_frac = max(0.0, min(1.0, float(ev.fast_frac)))
            slow_frac = 1.0 - fast_frac
            if fast_frac > 0.0:
                shape = EVENT_SHAPES['meal_fast']
                out.append(_make_curve_event(
                    shape['channel'], t, shape['rise_min'], shape['fall_min'],
                    mag * fast_frac, n_ctx, patch_size,
                ))
            if slow_frac > 0.0:
                shape = EVENT_SHAPES['meal_slow']
                out.append(_make_curve_event(
                    shape['channel'], t, shape['rise_min'], shape['fall_min'],
                    mag * slow_frac, n_ctx, patch_size,
                ))
    return out


def _curve_event_to_raw_values(
    event,
    n_ctx: int,
    n_pred_patches: int,
    patch_size: int,
) -> np.ndarray:
    vals = np.zeros((n_pred_patches, patch_size), dtype=np.float32)
    for p in range(n_pred_patches):
        for s in range(patch_size):
            abs_pos = float(n_ctx + p + s / patch_size)
            if abs_pos < event.left_patch or abs_pos > event.right_patch:
                continue
            if event.left_patch >= event.right_patch:
                bell = 1.0
            elif abs_pos <= event.peak_patch:
                half_span = max(event.peak_patch - event.left_patch, 0.01)
                t = (abs_pos - event.left_patch) / half_span
                t = max(0.0, min(1.0, t))
                bell = 0.5 * (1.0 - math.cos(math.pi * t))
            else:
                half_span = max(event.right_patch - event.peak_patch, 0.01)
                t = (event.right_patch - abs_pos) / half_span
                t = max(0.0, min(1.0, t))
                bell = 0.5 * (1.0 - math.cos(math.pi * t))
            bell = max(0.0, min(1.0, bell))
            vals[p, s] = event.amplitude * bell
    return vals


def _pencil_strokes_to_raw_values(
    strokes: list,
    disp_channel: int,
    n_ctx: int,
    n_pred_patches: int,
    patch_size: int,
) -> np.ndarray:
    """Resample freehand pencil strokes for ``disp_channel`` onto the model's
    ``(n_pred_patches, patch_size)`` prediction grid, smoothed.

    Each stroke stores ``(absolute-patch, raw-value)`` samples.  It contributes a
    dose of 0 outside its drawn abs-patch span (``np.interp`` ``left=right=0``);
    overlapping strokes on the same channel are max-combined (redraw-friendly).
    The resampled dense curve is Gaussian-smoothed (``GUI_PENCIL_SMOOTH_STEPS``)
    so the hand-drawn shape reads cleanly, and floored at 0 (this is a cosmetic
    drawing aid, not a model-input denoiser).

    Args:
        strokes: list of ``gui_state.PencilStroke``.
        disp_channel: display channel to gather (1=carbs, 2=insulin).
        n_ctx: number of context patches (prediction zone starts here).
        n_pred_patches: number of prediction patches to fill.
        patch_size: timesteps per patch.

    Returns:
        ``(n_pred_patches, patch_size)`` raw-unit dose array (g/5min or U/5min).
    """
    from config import GUI_PENCIL_SMOOTH_STEPS

    grid = n_ctx + np.arange(n_pred_patches * patch_size, dtype=np.float32) / patch_size
    out = np.zeros(n_pred_patches * patch_size, dtype=np.float32)
    for stroke in strokes:
        if getattr(stroke, 'channel', None) != disp_channel:
            continue
        xs = np.asarray(stroke.xs, dtype=np.float32)
        ys = np.maximum(np.asarray(stroke.ys, dtype=np.float32), 0.0)
        if xs.size == 0:
            continue
        # ``np.interp`` needs strictly-increasing sample x, but a stroke drawn with
        # back-and-forth motion is not monotone — collapse duplicate x to their max.
        uniq_x, inv = np.unique(xs, return_inverse=True)
        uniq_y = np.zeros(uniq_x.shape, dtype=np.float32)
        np.maximum.at(uniq_y, inv, ys)
        vals = np.interp(grid, uniq_x, uniq_y, left=0.0, right=0.0).astype(np.float32)
        out = np.maximum(out, vals)

    out = np.maximum(out, 0.0)
    out = _smooth_1d(out, GUI_PENCIL_SMOOTH_STEPS)
    return out.reshape(n_pred_patches, patch_size)


def _basal_curve_raw_values(
    basal_rate_delta: float,
    n_pred_patches: int,
    patch_size: int,
    ramp_up_h: float,
    ramp_down_h: float,
    duration_h: float,
) -> np.ndarray:
    vals = np.zeros((n_pred_patches, patch_size), dtype=np.float32)
    if basal_rate_delta == 0.0:
        return vals

    steps_per_hour = 60.0 / 5.0
    patches_per_hour = steps_per_hour / patch_size
    ramp_up_patches = ramp_up_h * patches_per_hour
    ramp_down_patches = ramp_down_h * patches_per_hour
    plateau_patches = max(duration_h * patches_per_hour - ramp_up_patches - ramp_down_patches, 0.0)
    total_patches = ramp_up_patches + plateau_patches + ramp_down_patches

    for p in range(n_pred_patches):
        for s in range(patch_size):
            abs_pos = float(p + s / patch_size)
            if abs_pos >= total_patches:
                continue
            if abs_pos < ramp_up_patches:
                frac = abs_pos / max(ramp_up_patches, 0.01)
                rate = basal_rate_delta * frac
            elif abs_pos < ramp_up_patches + plateau_patches:
                rate = basal_rate_delta
            else:
                frac = (abs_pos - ramp_up_patches - plateau_patches) / max(ramp_down_patches, 0.01)
                rate = basal_rate_delta * (1.0 - frac)
            vals[p, s] = rate / steps_per_hour

    return vals


def _normalize_channel_array(
    raw_vals: np.ndarray,
    channel_name: str,
    norm_stats: dict,
) -> np.ndarray:
    from normalization import SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS
    mean = norm_stats[channel_name]['mean']
    std = norm_stats[channel_name]['std']
    x = raw_vals.copy().astype(np.float32)
    if channel_name in RISK_SPACE_CHANNELS:
        from utils import kovatchev_f_np
        x = kovatchev_f_np(x)
    elif channel_name in SPARSE_LOG1P_CHANNELS:
        x = np.log1p(np.maximum(x, 0.0))
    return (x - mean) / (std + 1e-8)


def _compile_overrides_from_edits(
    n_pred: int,
    curve_events: list,
    basal_rate_delta: float,
    n_ctx: int,
    norm_stats: dict,
    basal_ramp_up_h: float,
    basal_ramp_down_h: float,
    basal_duration_h: float,
    pencil_strokes: list | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compile the user's painted curve / pencil / basal edits into announced
    carb / insulin / exercise INPUT overrides for the prediction zone.

    The model has no dynamics outputs, so the announced inputs are built from
    a ZERO baseline (no carb, no insulin, no exercise) plus the user's
    raised-cosine curves, freehand pencil strokes, and basal ramp — there is
    nothing to seed from a model prediction.  Curve bells and pencil strokes on
    the same channel sum together.  Returns ``(overrides_norm, overrides_raw)``
    over output channels {0: carb, 1: insulin, 2: exercise}; ``predict_what_if``
    consumes ``overrides_norm``.

    Every announced channel carries only what the user painted.  Exercise is a
    PLAN channel like the other two: nothing the patient did not announce is
    ever written into the prediction zone.
    """
    from config import PATCH_SIZE, CHANNEL_TO_FEAT

    pencil_strokes = pencil_strokes or []

    # The announceable set is the display table's, not a literal — a display
    # channel added without its entry here is silently dropped, taking the
    # user's stroke with it.
    has_edit = {out_ch: False for out_ch in DISPLAY_TO_OUTPUT_CH.values()}
    for event in curve_events:
        out_ch = DISPLAY_TO_OUTPUT_CH.get(event.channel)
        if out_ch in has_edit:
            has_edit[out_ch] = True
    for stroke in pencil_strokes:
        out_ch = DISPLAY_TO_OUTPUT_CH.get(stroke.channel)
        if out_ch in has_edit and len(stroke.xs) > 0:
            has_edit[out_ch] = True
    if basal_rate_delta != 0.0:
        has_edit[INSULIN_OUTPUT_CH] = True

    overrides_norm: dict[int, np.ndarray] = {}
    overrides_raw: dict[int, np.ndarray] = {}

    for out_ch in sorted(has_edit):
        # Announceable channels are the ones the model conditions on — the
        # input feats CHANNEL_TO_FEAT names.
        if out_ch not in CHANNEL_TO_FEAT:
            continue
        if not has_edit[out_ch]:
            continue
        ch_name = OUTPUT_CHANNEL_ORDER[out_ch]

        # Zero baseline: nothing announced on this channel until the user
        # paints it.
        modified_raw = np.zeros((n_pred, PATCH_SIZE), dtype=np.float32)

        for event in curve_events:
            evt_out_ch = DISPLAY_TO_OUTPUT_CH.get(event.channel)
            if evt_out_ch != out_ch:
                continue
            event_vals = _curve_event_to_raw_values(event, n_ctx, n_pred, PATCH_SIZE)
            modified_raw = modified_raw + event_vals

        # Freehand pencil strokes for this channel (max-combined internally,
        # summed on top of the raised-cosine bells above).
        if pencil_strokes:
            disp_ch = OUTPUT_TO_DISPLAY_CH[out_ch]
            pencil_vals = _pencil_strokes_to_raw_values(
                pencil_strokes, disp_ch, n_ctx, n_pred, PATCH_SIZE,
            )
            modified_raw = modified_raw + pencil_vals

        if out_ch == INSULIN_OUTPUT_CH and basal_rate_delta != 0.0:
            basal_vals = _basal_curve_raw_values(
                basal_rate_delta, n_pred, PATCH_SIZE,
                basal_ramp_up_h, basal_ramp_down_h, basal_duration_h,
            )
            modified_raw = modified_raw + basal_vals

        modified_raw = np.maximum(modified_raw, 0.0)

        # The model consumes RAW post-noise doses (carb/insulin/exercise floored
        # at 0); the announced what-if dose is already floored above, so
        # normalize it directly — no smoothing.  ``overrides_raw`` (chart
        # overlay) keeps the RAW announced value, at its own trained scale:
        # exercise in g/step carbohydrate-equivalent, never an intensity.
        norm_vals = _normalize_channel_array(
            modified_raw.flatten(), ch_name, norm_stats
        ).reshape(n_pred, PATCH_SIZE)

        overrides_norm[out_ch] = norm_vals.astype(np.float32)
        overrides_raw[out_ch] = modified_raw

    return overrides_norm, overrides_raw


def _decode_tod(
    model,
    context: torch.Tensor,
    norm_stats: dict,
    device: torch.device,
    overrides: dict[int, torch.Tensor] | None = None,
) -> tuple[float | None, float | None, np.ndarray | None]:
    """Decode the diagnostic time-of-day probe for ``context``.

    Runs one cheap ``B=1`` forecast through ``inference.predict`` with
    ``return_time=True`` (the forecast ``q_tau`` / ``median`` are computed
    identically and untouched — the probe reads a detached hidden state) and
    decodes the resultant length R of the per-bin softmax belief (via
    ``utils.time_of_day_decode_bins``).  ``overrides`` should mirror the
    announced-dose overrides used for the displayed forecast so the probe's
    context matches the shown bands on the what-if path.

    The probe is read off the MASKED patches, one row per head slot, and the
    masked set is no longer implied by position — it is built and passed to the
    model explicitly.  This goes through ``predict`` rather than assembling the
    forward's arguments here so that the masked set, the per-slot anchors and the
    patches' ``bg_masked`` bits are all built in one place: a second builder here
    would be a second chance for them to disagree.  Slot 0 is the first masked
    patch, which for the forecast protocol is the forecast origin.

    Args:
        model: loaded T1DMAI model.
        context: (n_ctx, PATCH_SIZE, N_INPUT_FEATURES) normalized context.
        norm_stats: normalization statistics (the ``normalize(0)`` no-dose
            baseline + the mg/dL anchor).
        device: torch device to run the forward on.
        overrides: optional {output-channel: (PREDICTION_PATCHES, PATCH_SIZE)}
            normalized announced-dose overrides.

    Returns:
        (pred_hour, confidence, bin_probs): ``pred_hour`` the decoded
        prediction-origin hour-of-day in [0, 24), ``confidence`` the resultant
        length R in [0, 1] of the FIRST masked patch's bin distribution (higher =
        more confident), ``bin_probs`` the ``(masked patches, TIME_PROBE_N_BINS)``
        per-patch softmax belief. All ``None`` when the probe is disabled
        (``time_pred is None``).
    """
    from inference import predict
    from utils import time_of_day_decode_bins
    from config import TIME_PROBE_N_BINS

    with torch.no_grad():
        out = predict(model, context, normalization_stats=norm_stats,
                      device=device, overrides=overrides, return_time=True)
    time_pred = out.get('time_pred')
    if time_pred is None:
        return None, None, None
    hours, Rs = time_of_day_decode_bins(time_pred[0:1, :], TIME_PROBE_N_BINS)
    hour = float(hours.reshape(-1)[0].item())
    R = float(Rs.reshape(-1)[0].item())
    bin_probs = torch.softmax(time_pred, dim=-1).cpu().numpy()
    return hour, R, bin_probs


def _painted_rolls_for_state(state, min_rolls: int, max_rolls: int) -> int:
    """Number of rolls needed to cover the furthest painted dose (curve events +
    high-level events + pencil strokes), clamped to ``[min_rolls, max_rolls]``.
    Positions are absolute patch units, so the furthest painted point minus
    ``n_ctx`` is the painted span in patches.

    One roll masks a span of ``PREDICTION_PATCHES`` patches at the right edge of
    its window — the forecast case of the masked-BG objective — so that is the
    span each roll covers.

    This is the *fit-to-drawing* horizon shared by the long-prediction dispatch
    and the announced-dose preview overlay."""
    from config import PREDICTION_PATCHES
    n_ctx = state.context.shape[0] if state.context is not None else 0
    furthest = float(n_ctx)
    for ev in state.curve_events:
        furthest = max(furthest, float(ev.right_patch))
    for st in state.pencil_strokes:
        if st.xs:
            furthest = max(furthest, float(max(st.xs)))
    if state.events:
        for cev in _events_to_curve_events(state.events, n_ctx):
            furthest = max(furthest, float(cev.right_patch))
    painted_patches = max(0.0, furthest - n_ctx)
    rolls = math.ceil(painted_patches / PREDICTION_PATCHES) if painted_patches > 0 else 0
    return int(max(min_rolls, min(max_rolls, rolls)))


def _preview_horizon_patches_for_state(state) -> int:
    """Length in patches, past ``n_ctx``, to compile/draw the announced-dose
    preview over.  At least one forward window, grown to cover the drawing
    (fit-to-drawing, capped at ``GUI_MAX_PREDICTION_HOURS``) and to match an
    existing forecast's rolled horizon — so a dose painted past 2 h stays
    visible rather than clipping to the single-pass window.

    This is a DRAWING extent, not the model's masked set: the announced plan is
    painted over patches the model may never have masked in one pass.  The masked
    set the model actually reads is built per forward, one roll at a time."""
    from config import PREDICTION_PATCHES, PREDICTION_HORIZON_HOURS, GUI_MAX_PREDICTION_HOURS
    max_rolls = max(1, round(GUI_MAX_PREDICTION_HOURS / PREDICTION_HORIZON_HOURS))
    n = _painted_rolls_for_state(state, 1, max_rolls) * PREDICTION_PATCHES
    if state.prediction.bands is not None:
        n = max(n, int(state.prediction.bands.shape[0]))
    return int(n)


def _masked_context(state, spans: list[tuple[int, int]]):
    """``state.context`` with the masked CONTEXT spans' dose channels withheld.

    Only under the blind policy, and only for spans inside the context: the
    trailing forecast zone is seeded to ``normalize(0)`` by
    ``inference._build_patches_tensor`` already, which IS the blind fill.
    ``inference`` withholds bg alone on a masked context patch — the announced
    convention — so under ``blind`` this is where the dose channels go with it,
    the same place ``calibrate_conformal._blind_context`` puts them.

    Args:
        state: the GUI state.
        spans: the emitted masked set, trailing span included.

    Returns:
        A new tensor under ``blind``; ``state.context`` itself otherwise.
    """
    from config import N_INPUT_FEATURES, PATCH_SIZE
    from data import blind_masked_doses
    from gui_state import mask_dose_fill

    fill = mask_dose_fill(state.masked_channel_policy, state.norm_stats)
    if fill is None:
        return state.context
    n_ctx = int(state.context.shape[0])
    flat = state.context.reshape(n_ctx, PATCH_SIZE * N_INPUT_FEATURES).clone()
    masked = torch.zeros(n_ctx, dtype=torch.bool)
    for start, length in spans:
        lo, hi = max(0, start), min(n_ctx, start + length)
        if hi > lo:
            masked[lo:hi] = True
    blind_masked_doses(flat, masked, fill)
    return flat.reshape(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)


def _run_prediction(
    state,
    model,
    device: torch.device,
    basal_ramp_up_h: float,
    basal_ramp_down_h: float,
    basal_duration_h: float,
) -> None:
    import inference
    from inference import predict
    from config import PATCH_SIZE, PREDICTION_PATCHES

    state.is_computing = True
    state.status_message = "Computing..."

    try:
        hour = _hour_at_pred_start(state)
        state.active_band_label = f"{hour:0.1f}h"
        n_ctx = state.context.shape[0]
        # The masked set: the user's context spans plus the mandatory trailing
        # forecast span. ``inference.PREDICTION_PATCHES`` (not config's) is the
        # window ``predict`` builds — ``main`` rewrites it to the checkpoint's
        # horizon — so the trailing span has to be measured against that one.
        n_pred = int(inference.PREDICTION_PATCHES)
        mask_spans = state.emitted_mask_spans(n_pred)
        ctx = _masked_context(state, mask_spans)
        # The context edge, which is the trailing forecast span's own anchor. The
        # anchor rule is one-sided and LEFT-PREFERRING: every slot of a span takes
        # the last step of the left neighbour patch (the first step of the right
        # neighbour only when the span starts at patch 0), and every slot of one
        # span gets the same value. Per-span anchors are read off
        # ``gui_state.span_anchor_cell`` for the chart's readout; this scalar
        # stays what it always was. The anchor is not the distance a metric bins
        # on: it ignores the near side.
        last_bg_idx = n_ctx * PATCH_SIZE - 1
        if state.bg_raw is not None and last_bg_idx >= 0:
            state.last_bg = float(state.bg_raw[last_bg_idx])

        if state.has_edits():
            all_curve_events = (
                list(state.curve_events)
                + _events_to_curve_events(state.events, n_ctx)
            )
            overrides_norm, overrides_raw = _compile_overrides_from_edits(
                PREDICTION_PATCHES,
                all_curve_events,
                state.basal_rate_delta,
                n_ctx,
                state.norm_stats,
                basal_ramp_up_h, basal_ramp_down_h, basal_duration_h,
                pencil_strokes=state.pencil_strokes,
            )
            torch_overrides = {
                ch: torch.from_numpy(vals.astype(np.float32))
                for ch, vals in overrides_norm.items()
            }
            # The model only conditions on the single-pass 2 h window, but the
            # announced dose the user painted may extend past it (fit-to-drawing).
            # Recompile the RAW overlay over the full painted horizon so a dose
            # drawn past 2 h stays visible on the chart (the model override above
            # stays at PREDICTION_PATCHES).
            disp_n_pred = _preview_horizon_patches_for_state(state)
            if disp_n_pred > PREDICTION_PATCHES:
                _, overrides_raw = _compile_overrides_from_edits(
                    disp_n_pred,
                    all_curve_events,
                    state.basal_rate_delta,
                    n_ctx,
                    state.norm_stats,
                    basal_ramp_up_h, basal_ramp_down_h, basal_duration_h,
                    pencil_strokes=state.pencil_strokes,
                )
            # Re-run predict with the announced carb/insulin/exercise INPUT perturbed;
            # the model's median BG forecast shifts in response.
            # ``predict_what_if`` is right-edge by construction — it takes no
            # ``mask_spans`` — and it is otherwise a conditioned ``predict``, so
            # the masked-set call goes straight to ``predict`` with the same
            # overrides rather than widening the wrapper. Painting still writes
            # the trailing zone only: the overrides are (n_pred, PATCH_SIZE).
            result = predict(
                model, ctx, state.patient_seed,
                overrides=torch_overrides,
                normalization_stats=state.norm_stats,
                device=device,
                mask_spans=mask_spans,
            )
            state.prediction.is_what_if = True
            state.mode_label = f"What-If · {state.active_band_label}"
            state.overrides = overrides_norm
            state.prediction.overrides_raw = overrides_raw
            probe_overrides = torch_overrides
        else:
            result = predict(
                model, ctx, state.patient_seed,
                normalization_stats=state.norm_stats,
                device=device,
                mask_spans=mask_spans,
            )
            state.prediction.is_what_if = False
            state.mode_label = f"Standard · {state.active_band_label}"
            state.overrides.clear()
            state.prediction.overrides_raw = None
            probe_overrides = None

        # Headline forecast: median_bg (mg/dL) + per-τ quantile bands.
        state.prediction.median_bg = result['median_bg'].cpu().numpy()
        state.prediction.bands = result['bands'].cpu().numpy()
        # WHICH patch each band row predicts. The head reads its slots by gather
        # over an arbitrary masked set, so slot j is patch mask_idx[j] and the
        # chart must place the row there — a fixed offset from the context end is
        # right only for the forecast.
        state.prediction.span_patches = result['mask_idx'].cpu().numpy()
        state.prediction.n_rolls = 1

        # Diagnostic time-of-day probe: decode the prediction-origin hour +
        # confidence R off the SAME (optionally what-if-overridden) context so
        # the readout stays consistent with the shown bands. Read-only — the
        # forecast above is untouched. None,None when the probe is disabled.
        (state.prediction.tod_pred_hour,
         state.prediction.tod_confidence,
         state.prediction.tod_bin_probs) = _decode_tod(
            model, state.context, state.norm_stats, device,
            overrides=probe_overrides,
        )

        state.status_message = f"Prediction complete ({datetime.datetime.now().strftime('%H:%M:%S')})"

    except Exception as e:
        traceback.print_exc()
        state.status_message = f"Error: {e}"
    finally:
        state.is_computing = False


def _normalize_channel(
    raw_val: float,
    channel_name: str,
    norm_stats: dict,
) -> float:
    from normalization import SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS
    mean = norm_stats[channel_name]['mean']
    std = norm_stats[channel_name]['std']
    x = raw_val
    if channel_name in RISK_SPACE_CHANNELS:
        from utils import kovatchev_f_np
        x = float(kovatchev_f_np(np.float32(x)))
    elif channel_name in SPARSE_LOG1P_CHANNELS:
        x = math.log1p(max(x, 0.0))
    return (x - mean) / (std + 1e-8)


def _scale_to_chart_y(
    raw_vals: np.ndarray,
    raw_min: float,
    raw_max: float,
    cy_min: float,
    cy_max: float,
) -> np.ndarray:
    frac = (raw_vals - raw_min) / max(raw_max - raw_min, 1e-9)
    return cy_min + frac * (cy_max - cy_min)


def _adaptive_time_intervals(
    span_patches: float,
) -> tuple[float, float, str]:
    """Pick a (major, minor, label_format) triple appropriate for the
    visible time span. One patch = 30 minutes.

    ``label_format`` is ``'h'`` for whole-hour-only labels (used when
    zoomed way out so we don't waste pixels on ":00"), ``'hm'`` for
    ``HH:MM`` (the normal case), and ``'hms'`` for ``HH:MM:SS`` when
    zoomed in tightly enough that minutes alone start looking coarse.
    Targets ~6–10 major labels visible at any zoom.
    """
    s = float(span_patches)
    if s >= 192:    return 48.0,  12.0,  'h'    # 24h major, 6h minor
    if s >= 96:     return 24.0,  4.0,   'h'    # 12h, 2h
    if s >= 48:     return 12.0,  2.0,   'h'    # 6h, 1h
    if s >= 24:     return 6.0,   1.0,   'hm'   # 3h, 30m
    if s >= 12:     return 4.0,   1.0,   'hm'   # 2h, 30m
    if s >= 6:      return 2.0,   0.5,   'hm'   # 1h, 15m
    if s >= 3:      return 1.0,   1/6,   'hm'   # 30m, 5m
    if s >= 1.5:    return 0.5,   1/6,   'hm'   # 15m, 5m
    if s >= 0.6:    return 1/6,   1/30,  'hm'   # 5m, 1m
    return                1/30,  1/300, 'hms'  # 1m, 10s — beyond data resolution


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Gaussian smoothing with edge-replicated padding so output length
    matches input. ``window=1`` is a no-op; even windows are bumped to the
    next odd value so the kernel stays centered. The kernel is a discrete
    Gaussian with σ = window/3, which puts ±3σ at the kernel edges and
    gives a noticeably softer envelope than a flat boxcar of the same
    width."""
    if window <= 1 or x.size < 2:
        return x
    w = min(int(window), x.size)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    sigma = max(w / 3.0, 0.5)
    k = np.arange(w, dtype=np.float32) - pad
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, pad, mode='edge')
    return np.convolve(padded, kernel, mode='valid').astype(x.dtype, copy=False)


def _contiguous_runs(patches: np.ndarray) -> list[tuple[int, int]]:
    """Group ascending patch indices into ``[lo, hi)`` runs of consecutive ones.

    The head's ``mask_idx`` is one slot per masked patch in span order, and the
    sampler guarantees a visible separator between spans, so a break in the
    sequence IS a span boundary — the same adjacency rule ``utils._span_layout``
    uses to group slots for the per-span median basis.

    Args:
        patches: ``(P,)`` ascending patch indices.

    Returns:
        ``[(lo, hi), ...]`` half-open index ranges into ``patches``.
    """
    if len(patches) == 0:
        return []
    runs: list[tuple[int, int]] = []
    lo = 0
    for i in range(1, len(patches)):
        if int(patches[i]) != int(patches[i - 1]) + 1:
            runs.append((lo, i))
            lo = i
    runs.append((lo, len(patches)))
    return runs


def _split_at_masked(
    times: np.ndarray, values: np.ndarray, spans,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Break a context curve into the stretches that are NOT masked.

    ``times`` are in patch units, so a sample belongs to patch ``floor(t)``.

    Args:
        times: ``(N,)`` x values in absolute patch units.
        values: ``(N,)`` chart-y values.
        spans: the user's ``MaskSpan`` list.

    Returns:
        ``[(times, values), ...]`` for each visible run of 2+ samples.
    """
    if len(times) == 0:
        return []
    hidden = np.zeros(len(times), dtype=bool)
    patch_of = np.floor(times).astype(np.int64)
    for s in spans:
        hidden |= (patch_of >= s.start) & (patch_of <= s.last)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    start = None
    for i, h in enumerate(hidden):
        if not h and start is None:
            start = i
        elif h and start is not None:
            if i - start >= 2:
                out.append((times[start:i], values[start:i]))
            start = None
    if start is not None and len(times) - start >= 2:
        out.append((times[start:], values[start:]))
    return out


def _draw_band_polygon(
    surf,
    transform,
    times: np.ndarray,
    y_upper: np.ndarray,
    y_lower: np.ndarray,
    color: tuple[int, int, int],
    alpha: int = 40,
) -> None:
    try:
        import pygame as _pg
    except ImportError:
        return
    if len(times) < 2:
        return

    def to_local(t: float, v: float) -> tuple[int, int]:
        sx, sy = transform.chart_to_screen(t, v)
        sx_l = int(max(0, min(transform.sw, sx - transform.sx)))
        sy_l = int(max(0, min(transform.sh, sy - transform.sy)))
        return sx_l, sy_l

    band_surf = _pg.Surface((int(transform.sw), int(transform.sh)), _pg.SRCALPHA)
    upper_pts = [to_local(float(t), float(u)) for t, u in zip(times, y_upper)]
    lower_pts = [to_local(float(t), float(l))
                 for t, l in reversed(list(zip(times, y_lower)))]
    polygon = upper_pts + lower_pts
    if len(polygon) >= 3:
        _pg.draw.polygon(band_surf, (*color, alpha), polygon)
    surf.blit(band_surf, (int(transform.sx), int(transform.sy)))


class T1DMAIGui:

    def __init__(
        self,
        model=None,
        norm_stats: dict | None = None,
        device: torch.device | None = None,
        patient_seed: int = 42,
        width: int = WINDOW_WIDTH,
        height: int = WINDOW_HEIGHT,
        basal_ramp_up_h: float = 0.5,
        basal_ramp_down_h: float = 0.5,
        basal_duration_h: float = 4.0,
        masked_channel_policy: str | None = None,
        context_hours: float | None = None,
    ) -> None:
        self.pygame = _safe_import_pygame()
        self._context_hours = context_hours
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        self.width = width
        self.height = height
        self._basal_ramp_up_h = basal_ramp_up_h
        self._basal_ramp_down_h = basal_ramp_down_h
        self._basal_duration_h = basal_duration_h
        if norm_stats is None:
            norm_stats = {}

        from gui_state import GUIState
        self.state = GUIState()
        self.state.norm_stats = norm_stats
        self.state.patient_seed = patient_seed
        if masked_channel_policy is not None:
            self.state.masked_channel_policy = masked_channel_policy

        right_panel_w = self._right_panel_w()
        chart_x = SIDEBAR_WIDTH + CHART_PADDING
        chart_y = CHART_PADDING
        chart_w = width - SIDEBAR_WIDTH - right_panel_w - 2 * CHART_PADDING
        chart_h = (height - CONTROL_PANEL_HEIGHT - STATUS_BAR_HEIGHT
                   - 2 * CHART_PADDING - ui_px(40))

        from gui_renderer import ChartTransform
        self.chart_transform = ChartTransform(
            screen_x=chart_x, screen_y=chart_y,
            screen_w=chart_w, screen_h=chart_h,
            chart_x_min=0, chart_x_max=100,
            chart_y_min=40, chart_y_max=400,
        )

        self.control_panel_rect = (
            SIDEBAR_WIDTH, height - CONTROL_PANEL_HEIGHT - STATUS_BAR_HEIGHT,
            width - SIDEBAR_WIDTH - right_panel_w, CONTROL_PANEL_HEIGHT,
        )

        from gui_controls import (
            Button, Toggle,
            HelpWindow, EventEditorModal,
        )

        self._toggles = []
        for i, (name, color) in enumerate(zip(self.state.channel_names, CHANNEL_COLORS)):
            t = Toggle(
                10, 0, SIDEBAR_WIDTH - 20, TOGGLE_HEIGHT,
                name, color,
                initial_state=self.state.channel_visible[i],
                callback=lambda state, idx=i: self._toggle_channel(idx),
            )
            self._toggles.append(t)

        # Display-options toggles: cosmetic smoothing on the mean line and
        # on the σ band edges. Toggling them just queues a redraw — no
        # re-prediction is needed since smoothing is post-hoc.
        self._display_toggles = [
            Toggle(
                10, 0, SIDEBAR_WIDTH - 20, TOGGLE_HEIGHT,
                "Smooth μ (mean line)", color=(200, 205, 220),
                initial_state=self.state.smooth_mu,
                callback=lambda s: self._set_smooth_mu(s),
            ),
            Toggle(
                10, 0, SIDEBAR_WIDTH - 20, TOGGLE_HEIGHT,
                "Smooth σ (band edges)", color=(200, 205, 220),
                initial_state=self.state.smooth_band,
                callback=lambda s: self._set_smooth_band(s),
            ),
        ]

        btn_w = (SIDEBAR_WIDTH - 30) // 2
        bh = BUTTON_HEIGHT
        self._buttons = [
            Button(10, 0, btn_w, bh, "Predict [SPC]", callback=self._do_predict),
            Button(10 + btn_w + 10, 0, btn_w, bh, "Long Pred [L]",
                   callback=self._do_long_predict),
            Button(10, 0, btn_w, bh, "What-If [W]",
                   callback=self._do_what_if),
            Button(10 + btn_w + 10, 0, btn_w, bh, "Pencil [P]",
                   callback=self._do_pencil),
            Button(10, 0, btn_w, bh, "Roll Pred [F]",
                   callback=self._do_roll_forward),
            Button(10 + btn_w + 10, 0, btn_w, bh, "Sim Fwd [G]",
                   callback=self._do_sim_forward),
            Button(10, 0, btn_w, bh, "Eval vs Sim [V]",
                   callback=self._do_eval_against_sim),
            Button(10 + btn_w + 10, 0, btn_w, bh, "Reset [R]",
                   callback=self._do_reset),
            Button(10, 0, btn_w, bh, "New Patient [N]",
                   callback=self._do_new_patient),
            Button(10, 0, btn_w, bh, "Screenshot [S]",
                   callback=self._do_screenshot),
            Button(10 + btn_w + 10, 0, btn_w, bh, "Clear All [C]",
                   callback=self._do_clear_all),
            Button(10, 0, btn_w, bh, "Basal +1 U/h",
                   callback=self._do_basal_plus),
            Button(10 + btn_w + 10, 0, btn_w, bh, "Basal -1 U/h",
                   callback=self._do_basal_minus),
            Button(10, 0, btn_w, bh, "Mask [M]", callback=self._do_mask_tool),
            Button(10 + btn_w + 10, 0, btn_w, bh, "Clear Masks",
                   callback=self._do_clear_masks),
        ]
        # Dose painting is inert under a blind checkpoint (the masked-patch dose
        # channels were pinned at the no-dose fill throughout training), so the
        # controls that write one are disabled rather than left to do nothing
        # visible. Matched by label, since the list above is positional.
        from gui_state import dose_painting_enabled
        if not dose_painting_enabled(self.state.masked_channel_policy):
            for b in self._buttons:
                if b.label.split(' [')[0] in _DOSE_PAINTING_BUTTONS:
                    b.enabled = False

        # Modal sub-windows. They auto-center on the screen each frame
        # via their own _layout, so no rect plumbing is needed here.
        self._help_window = HelpWindow()
        self._event_editor = EventEditorModal()

        # ---- Events right-panel state ----
        self._events_panel_scroll: int = 0
        self._events_panel_content_h: int = 0
        # (kind, meal_name, rect) for each create-event button.
        self._event_create_rects: list[tuple[str, str, tuple]] = []
        # (event_idx, row_rect, delete_rect) for each list row.
        self._event_row_rects: list[tuple[int, tuple, tuple]] = []

        self._needs_redraw = True
        self._chart_cache: self.pygame.Surface | None = None
        self._prediction_thread: threading.Thread | None = None
        self._prev_is_computing = False

        self._undo_stack: list[tuple[list, list, float]] = []

        # ---- Pan-drag state (middle / right mouse button) ----
        self._panning: bool = False
        self._pan_start_screen_x: int = 0
        self._pan_start_cx_min: float = 0.0
        self._pan_start_cx_max: float = 0.0

        # ---- Pencil-draw state (left-drag while the pencil tool is active) ----
        # The in-progress freehand stroke is accumulated here and appended to
        # ``state.pencil_strokes`` on release.
        self._pencil_drawing: bool = False
        self._active_stroke: Any = None  # gui_state.PencilStroke | None

        # ---- Sidebar scroll state ----
        # Content height is measured at the end of each _draw_sidebar pass;
        # the wheel handler clamps against the *previous* frame's value,
        # which is fine because layout only changes in response to user
        # actions that also trigger a redraw before the next scroll input.
        self._sidebar_scroll: int = 0
        self._sidebar_content_h: int = 0

        if patient_seed is not None and norm_stats:
            try:
                (self.state.context, self.state.patient_summary,
                 self.state.bg_raw, self.state.context_raw,
                 self.state.sim_start_hour, self.state.sim_start_day,
                 self.state.sim) = \
                    _build_context_from_sim(patient_seed, norm_stats,
                                            self._context_hours)
                self._reset_chart_view()
                self.state.status_message = "Ready — press SPACE/Enter to predict"
            except Exception as e:
                traceback.print_exc()
                self.state.status_message = f"Sim error: {e}"

    def _update_layout(self, width: int, height: int) -> None:
        from gui_renderer import ChartTransform

        self.width = width
        self.height = height

        right_panel_w = self._right_panel_w()
        chart_x = SIDEBAR_WIDTH + CHART_PADDING
        chart_y = CHART_PADDING
        chart_w = max(width - SIDEBAR_WIDTH - right_panel_w - 2 * CHART_PADDING, 100)
        chart_h = max(height - CONTROL_PANEL_HEIGHT - STATUS_BAR_HEIGHT
                      - 2 * CHART_PADDING - ui_px(40), 100)

        self.chart_transform = ChartTransform(
            screen_x=chart_x, screen_y=chart_y,
            screen_w=chart_w, screen_h=chart_h,
            chart_x_min=self.chart_transform.cx_min,
            chart_x_max=self.chart_transform.cx_max,
            chart_y_min=self.chart_transform.cy_min,
            chart_y_max=self.chart_transform.cy_max,
        )

        self.control_panel_rect = (
            SIDEBAR_WIDTH, height - CONTROL_PANEL_HEIGHT - STATUS_BAR_HEIGHT,
            width - SIDEBAR_WIDTH - right_panel_w, CONTROL_PANEL_HEIGHT,
        )

        # Modal sub-windows reposition themselves on every draw; no
        # layout fixup needed when the host resizes.

    def _toggle_channel(self, idx: int) -> None:
        self.state.channel_visible[idx] = not self.state.channel_visible[idx]
        self._needs_redraw = True

    def _set_smooth_mu(self, on: bool) -> None:
        self.state.smooth_mu = on
        self._needs_redraw = True

    def _set_smooth_band(self, on: bool) -> None:
        self.state.smooth_band = on
        self._needs_redraw = True

    def _do_predict(self) -> None:
        if not self.state.is_computing:
            self._run_prediction_async()

    def _n_pred(self) -> int:
        """The trailing forecast span's length, in patches.

        Read off ``inference``, not ``config``: ``main`` rewrites
        ``inference.PREDICTION_PATCHES`` to the loaded checkpoint's horizon and
        leaves ``config`` alone, and this is the window ``predict`` actually
        builds — the one the masked set has to fit.
        """
        import inference
        return int(inference.PREDICTION_PATCHES)

    def _dose_painting_blocked(self) -> str:
        """Why a painted dose cannot reach this model, or ``''``.

        The blind policy pinned the masked-patch dose channels at
        ``data.zero_dose_fill`` for the whole of training, so an override written
        there is a channel the weights learned carries no information: the
        forecast would not move and the user would read that as a model result.
        """
        from gui_state import dose_painting_enabled
        if dose_painting_enabled(self.state.masked_channel_policy):
            return ''
        return ("Dose painting is off: this checkpoint was trained blind, with "
                "masked-patch carb / insulin / exercise pinned at the no-dose "
                "fill — a painted override is invisible to it.")

    def _do_mask_tool(self) -> None:
        from gui_state import TOOL_MASK, TOOL_NONE
        if self.state.active_tool == TOOL_MASK:
            self.state.set_tool(TOOL_NONE)
            self.state.status_message = "Mask tool off"
        else:
            self.state.set_tool(TOOL_MASK)
            left = self.state.mask_budget_left(self._n_pred())
            self.state.status_message = (
                f"Mask ON — drag over the context to mask patches ({left} free). "
                "1/2/3 on the numpad row = forecast / begin-fill / infill preset."
            )
        self._needs_redraw = True

    def _do_clear_masks(self) -> None:
        self.state.clear_mask_spans()
        self._clear_prediction()
        self.state.status_message = "Masks cleared — forecast only"
        self._needs_redraw = True

    def _do_mask_preset(self, preset: str) -> None:
        from gui_state import MASK_PRESET_LABELS
        if self.state.context is None:
            return
        self.state.apply_mask_preset(preset, self._n_pred())
        self._clear_prediction()
        spans = self.state.emitted_mask_spans(self._n_pred())
        # Frame the span the user asked for, not the whole emitted set: the
        # trailing forecast span is always in that set and sits a window away
        # from a backcast one, so framing both frames neither.
        chosen = [(sp.start, sp.length) for sp in self.state.mask_spans]
        self._scroll_to_patches(chosen or spans)
        self.state.status_message = (
            f"{MASK_PRESET_LABELS[preset]} preset — masked set {spans}"
        )
        self._needs_redraw = True

    def _scroll_to_patches(self, spans) -> None:
        """Pan (never zoom) so every span in ``spans`` is on the chart.

        The view opens on the trailing hours of a multi-day window, so the
        backcast preset's span at patch 0 and an interior infill both land far
        off the left edge. Their fan would be drawn correctly and seen by nobody.
        """
        if not spans:
            return
        ct = self.chart_transform
        lo = min(float(s) for s, _L in spans)
        hi = max(float(s + L) for s, L in spans)
        width = ct.cx_max - ct.cx_min
        if lo >= ct.cx_min and hi <= ct.cx_max:
            return
        if hi - lo >= width:                      # too wide to frame: show its head
            new_min = max(0.0, lo - 1.0)
        else:
            new_min = max(0.0, (lo + hi) / 2.0 - width / 2.0)
        ct.update(chart_x_min=new_min, chart_x_max=new_min + width)

    def _clear_prediction(self) -> None:
        """Drop the shown forecast.

        Any change to the masked set or the context invalidates it: the rows are
        keyed to the patches the head was asked about, so keeping them would draw
        one masked set's fan over another's spans.
        """
        self.state.prediction.median_bg = None
        self.state.prediction.bands = None
        self.state.prediction.span_patches = None
        self.state.prediction.overrides_raw = None
        self.state.prediction_rolls = 1

    def _do_what_if(self) -> None:
        if self._refuse_dose_painting():
            return
        from gui_state import TOOL_CURVE_EDITOR, TOOL_NONE
        if self.state.active_tool == TOOL_CURVE_EDITOR:
            self.state.set_tool(TOOL_NONE)
            self.state.status_message = "Curve editor off"
        else:
            self.state.set_tool(TOOL_CURVE_EDITOR)
            ch_name = self.state.channel_names[self.state.selected_edit_channel]
            self.state.status_message = (
                f"Curve editor ON — editing {ch_name} (Tab to cycle)"
            )
        self._needs_redraw = True

    def _do_pencil(self) -> None:
        if self._refuse_dose_painting():
            return
        from gui_state import TOOL_PENCIL, TOOL_NONE
        if self.state.active_tool == TOOL_PENCIL:
            self.state.set_tool(TOOL_NONE)
            self._pencil_drawing = False
            self._active_stroke = None
            self.state.status_message = "Pencil off"
        else:
            self.state.set_tool(TOOL_PENCIL)
            ch_name = self.state.channel_names[self.state.selected_edit_channel]
            self.state.status_message = (
                f"Pencil ON — drag to draw {ch_name} "
                "(Tab = carb/insulin/exercise, L = long predict)"
            )
        self._needs_redraw = True

    def _do_roll_forward(self) -> None:
        if not self.state.is_computing and self.state.context is not None:
            self.state.prediction_rolls += 1
            self._run_rolling_async()

    def _painted_rolls(self, min_rolls: int, max_rolls: int) -> int:
        """Fit-to-drawing roll count over ``self.state`` (see
        ``_painted_rolls_for_state``)."""
        return _painted_rolls_for_state(self.state, min_rolls, max_rolls)

    def _preview_horizon_patches(self) -> int:
        """Announced-dose preview length in patches over ``self.state`` (see
        ``_preview_horizon_patches_for_state``)."""
        return _preview_horizon_patches_for_state(self.state)

    def _do_long_predict(self) -> None:
        """Fit-to-drawing long prediction: roll ``predict_rolling`` out far
        enough to cover the painted doses (floor ``GUI_LONG_PREDICTION_HOURS``,
        cap ``GUI_MAX_PREDICTION_HOURS``), conditioning every roll on the
        announced carb / insulin / exercise.  SPACE stays the single-pass 2 h
        forecast."""
        if self.state.is_computing or self.state.context is None or self.model is None:
            return
        from config import (
            PREDICTION_HORIZON_HOURS, GUI_LONG_PREDICTION_HOURS, GUI_MAX_PREDICTION_HOURS,
        )
        per_roll_h = PREDICTION_HORIZON_HOURS
        min_rolls = max(1, round(GUI_LONG_PREDICTION_HOURS / per_roll_h))
        max_rolls = max(min_rolls, round(GUI_MAX_PREDICTION_HOURS / per_roll_h))
        rolls = self._painted_rolls(min_rolls, max_rolls)
        self.state.prediction_rolls = rolls
        self.state.status_message = (
            f"Long prediction — {rolls * per_roll_h:.0f} h ({rolls} rolls)…"
        )
        self._run_rolling_async()

    def _do_sim_forward(self, hours: float = 2.0) -> None:
        """Step the live simulator forward by ``hours`` real hours, append
        the new ground-truth steps to the context buffers, and re-run a
        single-horizon prediction from the new context end. Distinct from
        ``_do_roll_forward``, which extends only the model's predicted
        horizon without advancing the simulator."""
        if self.state.is_computing or self.state.sim is None:
            return
        if self.state.norm_stats is None:
            return
        try:
            n_added = _advance_sim_hours(
                self.state, hours, self.state.norm_stats,
            )
        except Exception as e:
            traceback.print_exc()
            self.state.status_message = f"Sim advance error: {e}"
            return
        if n_added <= 0:
            self.state.status_message = "Sim advance produced no new patches"
            return
        self.state.clear_overrides()
        # The context just grew, so every masked span now sits on different data
        # than the user drew it over and the trailing forecast span has moved.
        # Reset to the forecast preset rather than re-anchor silently.
        self.state.clear_mask_spans()
        self._clear_prediction()
        # Keep the user's zoom, but follow NOW: ``n_added`` is the NET shift, so
        # it is 0 once the buffer is saturated (the window slides, indices hold)
        # and positive only while it is still filling. Holding the view fixed
        # through that phase walks the forecast zone off the right edge.
        if n_added:
            ct = self.chart_transform
            ct.update(chart_x_min=ct.cx_min + n_added, chart_x_max=ct.cx_max + n_added)
        self.state.status_message = (
            f"Sim advanced by {hours:.1f}h ({n_added} patches) — "
            "press SPACE to predict"
        )

    def _do_eval_against_sim(self) -> None:
        """Score the current model prediction against the simulator's
        ground truth. Snapshots ``state.prediction.median_bg``, advances the
        simulator by the prediction horizon so the same window now has
        real BG, compares the two, and stores the error stats in
        ``state.last_eval`` for the sidebar Score card to render. The
        prediction zone gets consumed (the new ground truth becomes
        context), so the stale prediction is cleared — press SPACE to
        predict again from the new tail."""
        from config import PATCH_SIZE
        from gui_state import EvalResult
        if self.state.is_computing or self.state.sim is None:
            return
        if self.state.norm_stats is None:
            return
        if self.state.prediction.median_bg is None or self.state.context is None:
            self.state.status_message = "Eval: predict first, then click Eval"
            return

        pred_bg_snap = np.asarray(
            self.state.prediction.median_bg, dtype=np.float32,
        ).copy()
        horizon_steps = int(len(pred_bg_snap))
        horizon_h = horizon_steps * 5.0 / 60.0
        eval_start_step = int(self.state.context.shape[0]) * PATCH_SIZE

        try:
            n_added = _advance_sim_hours(
                self.state, horizon_h, self.state.norm_stats,
            )
        except Exception as e:
            traceback.print_exc()
            self.state.status_message = f"Eval error: {e}"
            return
        if n_added <= 0 or self.state.bg_raw is None:
            self.state.status_message = "Eval: simulator advance failed"
            return

        # Score against the RAW post-noise truth: the model forecast targets raw
        # clamped BG, so the ground truth uses the same physical-range clamp.
        from T1DMSIM.simulator import BG_CLAMP_MIN as _BMIN, BG_CLAMP_MAX as _BMAX
        truth_bg = np.asarray(
            np.clip(self.state.bg_raw, _BMIN, _BMAX)[
                eval_start_step:eval_start_step + horizon_steps],
            dtype=np.float32,
        )
        n_compare = int(min(len(truth_bg), len(pred_bg_snap)))
        if n_compare <= 0:
            self.state.status_message = "Eval: no truth data available"
            return

        pred = pred_bg_snap[:n_compare]
        truth = truth_bg[:n_compare]
        err = pred - truth
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        bias = float(np.mean(err))
        max_abs = float(np.max(np.abs(err)))

        self.state.last_eval = EvalResult(
            mae=mae, rmse=rmse, bias=bias, max_abs=max_abs,
            horizon_h=horizon_h, n_steps=n_compare,
            pred_bg=pred.copy(), truth_bg=truth.copy(),
            eval_at_patch=eval_start_step // PATCH_SIZE,
        )

        # The just-evaluated window is now context with real data; clear
        # the stale prediction so the chart doesn't keep drawing a band
        # built for a different prediction zone, and the masked spans with it —
        # the simulator advanced, so they no longer cover what they were drawn on.
        self._clear_prediction()
        self.state.clear_mask_spans()

        self.state.status_message = (
            f"Eval: MAE {mae:.1f} · RMSE {rmse:.1f} · "
            f"Bias {bias:+.1f} mg/dL over {horizon_h:.1f}h"
        )
        self._needs_redraw = True

    def _reset_chart_view(self) -> None:
        """Open on the trailing ``CHART_VIEW_HOURS`` of the window.

        The whole context is several days wide, and drawn end to end nothing in
        it is legible — least of all the forecast, which is the last 2 h of it.
        The view is a viewport, not the model's input: every patch is still fed
        to the forward. Scroll-wheel zooms, drag pans, and ``R`` returns here.
        """
        if self.state.context is None:
            return
        from config import GUI_LONG_PREDICTION_HOURS, PREDICTION_HORIZON_HOURS
        n_ctx = self.state.context.shape[0]
        # Reserve the LONG fan, not the single-pass one: L rolls the forecast out
        # to GUI_LONG_PREDICTION_HOURS, and a right edge cut to the 2 h horizon
        # draws most of that off the chart.
        rolls = max(1, round(GUI_LONG_PREDICTION_HOURS / PREDICTION_HORIZON_HOURS))
        end = float(n_ctx + rolls * self._n_pred() + 2)
        history = CHART_VIEW_HOURS * 60.0 / (PATCH_SIZE * 5.0)
        self.chart_transform.update(
            chart_x_min=max(0.0, float(n_ctx) - history),
            chart_x_max=end,
        )

    def _do_reset(self) -> None:
        self.state.clear_overrides()
        self.state.prediction_rolls = 1
        self._reset_chart_view()
        self.state.status_message = "Reset — press SPACE/Enter to predict"
        self._needs_redraw = True

    def _do_clear_all(self) -> None:
        if self.state.context is None or self.state.norm_stats is None:
            return
        for disp_ch in range(N_DISPLAY_CHANNELS):
            raw_default = DISPLAY_CHANNEL_CLEAR_DEFAULTS[disp_ch]
            ch_name = DISPLAY_TO_STATS_NAME[disp_ch]
            feat_idx = DISPLAY_TO_FEATURE_IDX[disp_ch]
            if self.state.context_raw is not None:
                self.state.context_raw[:, disp_ch] = raw_default
            norm_val = float(_normalize_channel(raw_default, ch_name, self.state.norm_stats))
            self.state.context[:, :, feat_idx] = norm_val
        if self.state.bg_raw is not None:
            self.state.bg_raw[:] = DISPLAY_CHANNEL_CLEAR_DEFAULTS[0]
        self.state.clear_overrides()
        self.state.clear_mask_spans()
        self._clear_prediction()
        self.state.status_message = "Cleared all curves — paint and press Enter"
        self._needs_redraw = True

    def _do_new_patient(self) -> None:
        import random
        self.state.patient_seed = random.randint(0, 100000)
        self.state.clear_overrides()
        self.state.clear_mask_spans()
        self.state.last_eval = None
        # Drop the previous patient's prediction so the chart doesn't render a
        # stale forecast overlay (built for a different context / last_bg)
        # against the new patient until the async worker finishes.
        self._clear_prediction()
        if self.state.norm_stats:
            try:
                (ctx, summ, bg_raw, ctx_raw,
                 start_hour, start_day, sim) = _build_context_from_sim(
                    self.state.patient_seed, self.state.norm_stats,
                    self._context_hours,
                )
                self.state.context = ctx
                self.state.patient_summary = summ
                self.state.bg_raw = bg_raw
                self.state.context_raw = ctx_raw
                self.state.sim_start_hour = start_hour
                self.state.sim_start_day = start_day
                self.state.sim = sim
                self._reset_chart_view()
                self.state.status_message = (
                    f"New patient (seed {self.state.patient_seed}) — "
                    "press SPACE/Enter to predict"
                )
                self._needs_redraw = True
            except Exception as e:
                traceback.print_exc()
                self.state.status_message = f"Error: {e}"

    def _do_screenshot(self) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"t1dmai_screenshot_{ts}.png"
        self.pygame.image.save(self._screen, path)
        self.state.status_message = f"Saved: {path}"

    def _is_in_chart(self, mx: int, my: int) -> bool:
        ct = self.chart_transform
        return (ct.sx <= mx <= ct.sx + ct.sw and
                ct.sy <= my <= ct.sy + ct.sh)

    def _zoom_at_cursor(self, mx: int, wheel_y: int) -> None:
        """Zoom the chart x-axis around the cursor's chart-x position so
        the value under the cursor stays fixed. ``wheel_y`` is the pygame
        MOUSEWHEEL y delta (+1 = wheel up = zoom in)."""
        if wheel_y == 0:
            return
        ct = self.chart_transform
        cx_cursor, _ = ct.screen_to_chart(float(mx), float(ct.sy))
        factor = 1.0 / ZOOM_FACTOR if wheel_y > 0 else ZOOM_FACTOR
        new_min = cx_cursor + (ct.cx_min - cx_cursor) * factor
        new_max = cx_cursor + (ct.cx_max - cx_cursor) * factor
        ct.update(chart_x_min=new_min, chart_x_max=new_max)
        self._needs_redraw = True

    def _begin_pan(self, mx: int) -> None:
        self._panning = True
        self._pan_start_screen_x = mx
        self._pan_start_cx_min = self.chart_transform.cx_min
        self._pan_start_cx_max = self.chart_transform.cx_max

    def _handle_pan_motion(self, mx: int) -> None:
        ct = self.chart_transform
        span = self._pan_start_cx_max - self._pan_start_cx_min
        dx_pixels = mx - self._pan_start_screen_x
        dx_chart = -dx_pixels / max(ct.sw, 1.0) * span
        ct.update(
            chart_x_min=self._pan_start_cx_min + dx_chart,
            chart_x_max=self._pan_start_cx_max + dx_chart,
        )
        self._needs_redraw = True

    def _control_point_screen_pos(
        self, event, point_name: str,
    ) -> tuple[int, int]:
        disp_ch = event.channel
        raw_min, raw_max = DISPLAY_CHANNEL_RAW_RANGES[disp_ch]
        cy_min = self.chart_transform.cy_min
        cy_max = self.chart_transform.cy_max

        if point_name == 'left':
            cx = event.left_patch
            raw_val = raw_min
        elif point_name == 'peak':
            cx = event.peak_patch
            raw_val = event.amplitude
        else:
            cx = event.right_patch
            raw_val = raw_min

        frac = (raw_val - raw_min) / max(raw_max - raw_min, 1e-9)
        cy = cy_min + frac * (cy_max - cy_min)
        sx, sy = self.chart_transform.chart_to_screen(cx, cy)
        return int(sx), int(sy)

    def _hit_test_control_points(
        self, mx: int, my: int,
    ) -> tuple[int, str] | None:
        for evt_idx, event in enumerate(self.state.curve_events):
            if event.channel != self.state.selected_edit_channel:
                continue
            for point_name in ('left', 'peak', 'right'):
                sx, sy = self._control_point_screen_pos(event, point_name)
                if (mx - sx) ** 2 + (my - sy) ** 2 <= CONTROL_POINT_HIT_RADIUS ** 2:
                    return evt_idx, point_name
        return None

    def _handle_curve_click(self, mx: int, my: int) -> None:
        if not self._is_in_chart(mx, my):
            return
        if self.state.context is None:
            return

        cx, cy = self.chart_transform.screen_to_chart(float(mx), float(my))
        n_ctx = self.state.context.shape[0]

        if cx < n_ctx:
            return

        hit = self._hit_test_control_points(mx, my)
        if hit is not None:
            evt_idx, point_name = hit
            self.state.selected_event_idx = evt_idx
            self.state.dragging_point = point_name
            self._snapshot_for_undo()
            return

        disp_ch = self.state.selected_edit_channel
        if disp_ch not in (1, 2):
            return

        raw_min, raw_max = DISPLAY_CHANNEL_RAW_RANGES[disp_ch]
        cy_min = self.chart_transform.cy_min
        cy_max = self.chart_transform.cy_max
        frac = (cy - cy_min) / max(cy_max - cy_min, 1e-9)
        amplitude = raw_min + frac * (raw_max - raw_min)
        amplitude = float(max(raw_min, min(raw_max, amplitude)))

        from gui_state import CurveEvent
        event = CurveEvent(
            channel=disp_ch,
            left_patch=cx - 2.0,
            peak_patch=cx,
            right_patch=cx + 2.0,
            amplitude=amplitude,
        )

        self._snapshot_for_undo()
        self.state.curve_events.append(event)
        self.state.selected_event_idx = len(self.state.curve_events) - 1
        self.state.dragging_point = 'peak'
        self._compile_curve_overrides()

    def _handle_curve_drag(self, mx: int, my: int) -> None:
        if self.state.selected_event_idx < 0 or self.state.dragging_point is None:
            return
        if not self._is_in_chart(mx, my):
            return
        if self.state.context is None:
            return

        cx, cy = self.chart_transform.screen_to_chart(float(mx), float(my))
        n_ctx = self.state.context.shape[0]

        event = self.state.curve_events[self.state.selected_event_idx]
        disp_ch = event.channel
        raw_min, raw_max = DISPLAY_CHANNEL_RAW_RANGES[disp_ch]
        cy_min = self.chart_transform.cy_min
        cy_max = self.chart_transform.cy_max

        if self.state.dragging_point == 'peak':
            frac = (cy - cy_min) / max(cy_max - cy_min, 1e-9)
            amplitude = raw_min + frac * (raw_max - raw_min)
            event.amplitude = float(max(raw_min, min(raw_max, amplitude)))
            event.peak_patch = max(float(n_ctx), cx)
        elif self.state.dragging_point == 'left':
            event.left_patch = min(cx, event.peak_patch - 0.5)
        elif self.state.dragging_point == 'right':
            event.right_patch = max(cx, event.peak_patch + 0.5)

        self._compile_curve_overrides()

    def _handle_curve_release(self, mx: int, my: int) -> None:
        if self.state.dragging_point is not None:
            self.state.dragging_point = None
            self._compile_curve_overrides()

    # ------------------------------------------------------------------
    # Pencil tool (freehand announced-dose drawing)
    # ------------------------------------------------------------------

    def _pencil_sample(self, mx: int, my: int) -> tuple[float, float] | None:
        """Map a screen click to a ``(abs_patch, raw_value)`` pencil sample for
        the selected edit channel, or None if outside the drawable region.

        x is clamped to the prediction zone ``[n_ctx, n_ctx + max horizon]`` so a
        stroke never lands in the context or absurdly far out; y is mapped from
        the selected channel's raw range and floored at 0."""
        if self.state.context is None:
            return None
        disp_ch = self.state.selected_edit_channel
        if disp_ch not in (1, 2):
            return None
        from config import PREDICTION_HORIZON_HOURS, GUI_MAX_PREDICTION_HOURS, PREDICTION_PATCHES
        cx, cy = self.chart_transform.screen_to_chart(float(mx), float(my))
        n_ctx = self.state.context.shape[0]
        max_rolls = max(1, round(GUI_MAX_PREDICTION_HOURS / PREDICTION_HORIZON_HOURS))
        cx = float(min(max(cx, float(n_ctx)), float(n_ctx + max_rolls * PREDICTION_PATCHES)))
        raw_min, raw_max = DISPLAY_CHANNEL_RAW_RANGES[disp_ch]
        cy_min = self.chart_transform.cy_min
        cy_max = self.chart_transform.cy_max
        frac = (cy - cy_min) / max(cy_max - cy_min, 1e-9)
        raw_val = raw_min + frac * (raw_max - raw_min)
        raw_val = float(max(raw_min, min(raw_max, raw_val)))
        return cx, raw_val

    def _handle_pencil_down(self, mx: int, my: int) -> None:
        if not self._is_in_chart(mx, my):
            return
        sample = self._pencil_sample(mx, my)
        if sample is None:
            return
        from gui_state import PencilStroke
        self._snapshot_for_undo()
        self._active_stroke = PencilStroke(
            channel=self.state.selected_edit_channel,
            xs=[sample[0]], ys=[sample[1]],
        )
        self.state.pencil_strokes.append(self._active_stroke)
        self._pencil_drawing = True
        self._compile_curve_overrides()

    def _handle_pencil_motion(self, mx: int, my: int) -> None:
        if not self._pencil_drawing or self._active_stroke is None:
            return
        sample = self._pencil_sample(mx, my)
        if sample is None:
            return
        self._active_stroke.xs.append(sample[0])
        self._active_stroke.ys.append(sample[1])
        self._compile_curve_overrides()

    def _handle_pencil_release(self, mx: int, my: int) -> None:
        if not self._pencil_drawing:
            return
        self._pencil_drawing = False
        # Drop a degenerate (single-point / zero-length) stroke so a stray click
        # doesn't leave an invisible edit that flips the mode to What-If.
        if self._active_stroke is not None and len(self._active_stroke.xs) < 2:
            try:
                self.state.pencil_strokes.remove(self._active_stroke)
            except ValueError:
                pass
        self._active_stroke = None
        self._compile_curve_overrides()

    # ------------------------------------------------------------------
    # Free-form masking — drag, select, remove
    # ------------------------------------------------------------------

    def _patch_at(self, mx: int) -> int:
        """The patch under screen x, floored.

        Masking is PATCH-aligned because a patch is the head's unit: it emits one
        slot per masked patch, so a half-patch mask has no representation.
        """
        cx, _cy = self.chart_transform.screen_to_chart(float(mx), float(self.chart_transform.sy))
        return int(math.floor(cx))

    def _handle_mask_down(self, mx: int, my: int) -> None:
        """Begin a drag, or select / remove an existing span.

        A click inside a span selects it (the anchor readout follows the
        selection); ctrl-click removes it. A click on bare context starts a new
        span.
        """
        if self.state.context is None or not self._is_in_chart(mx, my):
            return
        patch = self._patch_at(mx)
        for idx, span in enumerate(self.state.mask_spans):
            if span.start <= patch <= span.last:
                if self.pygame.key.get_mods() & self.pygame.KMOD_CTRL:
                    self.state.remove_mask_span(idx)
                    self._clear_prediction()
                    self.state.status_message = "Span removed"
                else:
                    self.state.selected_mask_idx = idx
                    self.state.status_message = self._anchor_readout()
                self._needs_redraw = True
                return
        self.state.mask_drag_start = patch
        self.state.mask_drag_end = patch
        self._needs_redraw = True

    def _handle_mask_motion(self, mx: int, my: int) -> None:
        if self.state.mask_drag_start < 0:
            return
        self.state.mask_drag_end = self._patch_at(mx)
        self._needs_redraw = True

    def _handle_mask_release(self, mx: int, my: int) -> None:
        """Commit the drag as one span, or report why it was refused."""
        if self.state.mask_drag_start < 0:
            return
        lo = min(self.state.mask_drag_start, self.state.mask_drag_end)
        hi = max(self.state.mask_drag_start, self.state.mask_drag_end)
        self.state.mask_drag_start = -1
        self.state.mask_drag_end = -1
        n_ctx = self.state.n_ctx()
        # Clamp to the maskable stretch before asking: the trailing forecast span
        # is already masked and patch n_ctx-1 is its mandatory separator, so a
        # drag that runs off the right edge should mask what it legally can
        # rather than be refused whole.
        lo = max(0, lo)
        hi = min(hi, n_ctx - 2)
        if hi < lo:
            self.state.status_message = (
                f"Nothing to mask there — the forecast already covers patch "
                f"{n_ctx} on, and patch {n_ctx - 1} is its separator"
            )
            self._needs_redraw = True
            return
        reason = self.state.add_mask_span(lo, hi - lo + 1, self._n_pred())
        if reason:
            self.state.status_message = f"Refused: {reason}"
        else:
            from config import MAX_MASKED_PATCHES
            self._clear_prediction()
            left = self.state.mask_budget_left(self._n_pred())
            self.state.status_message = (
                f"Masked patches {lo}–{hi} · "
                f"{self.state.masked_patch_count(self._n_pred())}/"
                f"{MAX_MASKED_PATCHES} slots, {left} free"
            )
        self._needs_redraw = True

    def _forecast_rows(self) -> tuple[np.ndarray | None, int]:
        """The prediction's FORECAST steps, and how many other patches it filled.

        The masked set may hold backcast and infill spans as well, and those rows
        are not on the forecast's timeline: pooling them would make the sidebar's
        horizon the total masked patch count and its time-in-range a mixture of
        two different questions. ``span_patches`` is what separates them — rows
        at patch ``>= n_ctx`` are the trailing span.

        Returns:
            ``(median_bg over the forecast rows, count of other masked patches)``,
            or ``(None, 0)`` when there is no prediction.
        """
        pred = self.state.prediction
        if pred.median_bg is None or len(pred.median_bg) == 0:
            return None, 0
        patches = pred.span_patches
        if patches is None or pred.bands is None:
            return pred.median_bg, 0
        P = pred.bands.shape[0]
        if len(patches) != P or len(pred.median_bg) % P:
            return pred.median_bg, 0
        rows = np.asarray(pred.median_bg).reshape(P, -1)
        keep = np.asarray(patches) >= self.state.n_ctx()
        if not keep.any():
            return pred.median_bg, 0
        return rows[keep].reshape(-1), int((~keep).sum())

    def _anchor_readout(self) -> str:
        """The SELECTED span's anchor, in mg/dL, or the forecast's when none is.

        The anchor rule is one-sided and left-preferring, so this is the last
        step of the span's left neighbour — or the first step of its right
        neighbour for a span at patch 0. Showing the context edge for every span,
        as the readout did when the forecast was the only masked set, would name
        a cell the forward never read.
        """
        from config import PATCH_SIZE
        if self.state.bg_raw is None:
            return "Anchor: —"
        patch, step = self.state.selected_anchor_cell(self._n_pred())
        idx = patch * PATCH_SIZE + step
        if not 0 <= idx < len(self.state.bg_raw):
            return "Anchor: —"
        idx_sel = self.state.selected_mask_idx
        if 0 <= idx_sel < len(self.state.mask_spans):
            span = self.state.mask_spans[idx_sel]
            which = f"span {span.start}–{span.last}"
            side = "right" if span.start == 0 else "left"
        else:
            which, side = "forecast", "left"
        return (f"Anchor: {float(self.state.bg_raw[idx]):.0f} mg/dL "
                f"({which}, {side} · patch {patch})")

    def _compile_curve_overrides(self) -> None:
        """Recompile the painted carb / insulin / exercise announcement for
        the canvas preview.

        The risk-space model has no dynamics outputs and cannot reconstruct
        BG without a forward pass, so this only refreshes ``state.overrides`` /
        ``prediction.overrides_raw`` (the painted curves shown on the chart)
        and the mode label.  The BG forecast itself updates only when the user
        runs Predict / What-If (which re-runs the model with the announced
        input perturbed)."""
        if self.state.context is None or self.state.norm_stats is None:
            return
        from config import PREDICTION_PATCHES

        n_ctx = self.state.context.shape[0]
        all_curve_events = (
            list(self.state.curve_events)
            + _events_to_curve_events(self.state.events, n_ctx)
        )
        # Preview over the fit-to-drawing horizon so a dose painted past 2 h shows
        # immediately, not clipped to the single-pass window.
        n_pred = self._preview_horizon_patches()
        overrides_norm, overrides_raw = _compile_overrides_from_edits(
            n_pred,
            all_curve_events,
            self.state.basal_rate_delta,
            n_ctx,
            self.state.norm_stats,
            self._basal_ramp_up_h, self._basal_ramp_down_h, self._basal_duration_h,
            pencil_strokes=self.state.pencil_strokes,
        )

        self.state.overrides = overrides_norm
        self.state.prediction.overrides_raw = overrides_raw

        if self.state.has_edits():
            self.state.prediction.is_what_if = True
            self.state.mode_label = f"What-If (curves) · {self.state.active_band_label}"
        else:
            self.state.prediction.is_what_if = False
            self.state.mode_label = f"Standard · {self.state.active_band_label}"

        self._needs_redraw = True

    def _snapshot_for_undo(self) -> None:
        events_copy = copy.deepcopy(self.state.curve_events)
        pencil_copy = copy.deepcopy(self.state.pencil_strokes)
        self._undo_stack.append((events_copy, pencil_copy, self.state.basal_rate_delta))
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)

    def _do_undo(self) -> None:
        if not self._undo_stack:
            self.state.status_message = "Nothing to undo"
            return
        events_copy, pencil_copy, basal_copy = self._undo_stack.pop()
        self.state.curve_events = events_copy
        self.state.pencil_strokes = pencil_copy
        self.state.basal_rate_delta = basal_copy
        self.state.selected_event_idx = -1
        self.state.dragging_point = None
        self._pencil_drawing = False
        self._active_stroke = None
        self._compile_curve_overrides()
        self.state.status_message = "Undo"
        self._needs_redraw = True

    def _do_basal_plus(self) -> None:
        if self._refuse_dose_painting():
            return
        self._snapshot_for_undo()
        self.state.basal_rate_delta += 1.0
        self._compile_curve_overrides()

    def _do_basal_minus(self) -> None:
        if self._refuse_dose_painting():
            return
        self._snapshot_for_undo()
        self.state.basal_rate_delta -= 1.0
        self._compile_curve_overrides()

    def _refuse_dose_painting(self) -> bool:
        """Surface the blind-policy reason and return True when painting is off."""
        blocked = self._dose_painting_blocked()
        if not blocked:
            return False
        self.state.status_message = blocked
        self._needs_redraw = True
        return True

    def _run_prediction_async(self) -> None:
        if self.state.context is None or self.model is None:
            return
        # Claim the compute slot synchronously (the event loop is single-threaded)
        # so a second Predict/SPACE in the same frame batch cannot slip past the
        # _do_predict guard before the worker thread sets the flag and launch a
        # concurrent prediction writing the same state slots.
        self.state.is_computing = True
        self._undo_stack.clear()
        t = threading.Thread(
            target=_run_prediction,
            args=(self.state, self.model, self.device,
                  self._basal_ramp_up_h, self._basal_ramp_down_h,
                  self._basal_duration_h),
            daemon=True,
        )
        t.start()
        self._prediction_thread = t
        self._needs_redraw = True

    def _run_rolling_async(self) -> None:
        if self.state.context is None or self.model is None:
            return
        # ``predict_rolling`` is right-edge by construction and takes no masked
        # set, so a user span would be silently ignored while the chart still
        # shaded it. Drop the spans rather than show a fan that does not answer
        # what the overlay says was asked.
        if self.state.mask_spans:
            self.state.clear_mask_spans()
            self.state.status_message = (
                "Rolling is right-edge only — masked spans cleared"
            )
        # Claim the compute slot synchronously (see _run_prediction_async): closes
        # the double-thread window AND prevents a second _do_roll_forward in the
        # same batch from over-incrementing prediction_rolls before the flag is set.
        self.state.is_computing = True

        def _roll():
            # PREDICTION_PATCHES comes from ``inference`` (not ``config``):
            # main() patches ``inference.PREDICTION_PATCHES`` to the loaded
            # checkpoint's horizon, and ``predict_rolling`` consumes that same
            # value internally. Importing from ``config`` here would use the
            # stale module default and mis-size the trajectory slices below
            # whenever the checkpoint's horizon differs from the config default.
            from inference import predict_rolling, PREDICTION_PATCHES
            from config import PATCH_SIZE
            self.state.is_computing = True
            self.state.status_message = "Rolling forward..."
            try:
                assert self.state.context is not None
                assert self.model is not None
                hour = _hour_at_pred_start(self.state)
                self.state.active_band_label = f"{hour:0.1f}h"

                # Per-roll override callback so curve events condition the
                # model on every roll — not just get pasted on after the fact.
                # Each call receives the absolute patch index at this roll's
                # prediction-zone start; we hand it to
                # ``_compile_overrides_from_edits`` (the same compiler used by
                # the single-shot what-if path) so a curve event spanning
                # multiple rolls produces the right per-roll slice via
                # ``_curve_event_to_raw_values``'s absolute-patch math.
                # ``basal_rate_delta`` is only forwarded on roll 0 because
                # ``_basal_curve_raw_values`` measures position from the start
                # of *this* prediction window — applying it again on roll 1+
                # would re-trigger the basal ramp at the wrong absolute time.
                state = self.state
                ramp_up = self._basal_ramp_up_h
                ramp_down = self._basal_ramp_down_h
                duration = self._basal_duration_h
                def overrides_fn(roll_idx, base_mu_np, abs_n_ctx):
                    if not state.has_edits() or state.norm_stats is None:
                        return None
                    all_curve_events = (
                        list(state.curve_events)
                        + _events_to_curve_events(state.events, abs_n_ctx)
                    )
                    norm_dict, raw_dict = _compile_overrides_from_edits(
                        PREDICTION_PATCHES,
                        all_curve_events,
                        state.basal_rate_delta if roll_idx == 0 else 0.0,
                        abs_n_ctx,
                        state.norm_stats,
                        ramp_up, ramp_down, duration,
                        pencil_strokes=state.pencil_strokes,
                    )
                    if not norm_dict:
                        return None
                    return norm_dict, raw_dict

                result = predict_rolling(
                    self.model, self.state.context, self.state.patient_seed,
                    n_rolls=self.state.prediction_rolls,
                    normalization_stats=self.state.norm_stats,
                    device=self.device,
                    overrides_fn=overrides_fn,
                )
                # Headline forecast over the full rolled horizon: pred_bg is the
                # per-roll median forecast concatenated; bands is its per-τ
                # quantile envelope.
                self.state.prediction.median_bg = result['pred_bg'].cpu().numpy()
                self.state.prediction.bands = result['bands'].cpu().numpy()
                # Rolling is right-edge by construction: its rows ARE the
                # trailing zone, roll after roll. Clear any per-span mapping a
                # previous masked prediction left, or the chart would place these
                # rows over that set's spans.
                self.state.prediction.span_patches = None
                self.state.prediction.n_rolls = self.state.prediction_rolls

                # Time-of-day probe: the prediction origin is fixed across
                # rolls, so decode once from the initial context (read-only,
                # None,None when the probe is disabled).
                (self.state.prediction.tod_pred_hour,
                 self.state.prediction.tod_confidence,
                 self.state.prediction.tod_bin_probs) = _decode_tod(
                    self.model, self.state.context, self.state.norm_stats,
                    self.device,
                )

                if self.state.has_edits():
                    self.state.prediction.is_what_if = True
                    self.state.mode_label = f"What-If (curves) · {self.state.active_band_label}"
                    # Capture the painted announcement over the rolled horizon
                    # for the chart's curve overlay.
                    n_ctx = self.state.context.shape[0]
                    all_curve_events = (
                        list(self.state.curve_events)
                        + _events_to_curve_events(self.state.events, n_ctx)
                    )
                    n_pred = self.state.prediction.bands.shape[0]
                    overrides_norm, overrides_raw = _compile_overrides_from_edits(
                        n_pred,
                        all_curve_events,
                        self.state.basal_rate_delta,
                        n_ctx,
                        self.state.norm_stats,
                        self._basal_ramp_up_h, self._basal_ramp_down_h,
                        self._basal_duration_h,
                        pencil_strokes=self.state.pencil_strokes,
                    )
                    self.state.overrides = overrides_norm
                    self.state.prediction.overrides_raw = overrides_raw
                else:
                    self.state.overrides.clear()
                    self.state.prediction.overrides_raw = None
                    self.state.prediction.is_what_if = False
                    self.state.mode_label = f"Standard · {self.state.active_band_label}"

                # Leave the chart view alone — preserve the user's current
                # zoom/pan after a roll. Press R to reset, or scroll-wheel /
                # middle-drag to navigate to the new prediction zone.
                self.state.status_message = f"Roll {self.state.prediction_rolls} complete"
            except Exception as e:
                traceback.print_exc()
                self.state.status_message = f"Roll error: {e}"
            finally:
                self.state.is_computing = False

        t = threading.Thread(target=_roll, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Sidebar drawing helpers
    # ------------------------------------------------------------------

    def _draw_panel_card(self, surface, x, y, w, h):
        """Soft rounded card used as the background for a sidebar
        section. Subtle alpha fill + thin border line."""
        pygame = self.pygame
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        card.fill((*SIDEBAR_PANEL_BG, 235))
        surface.blit(card, (x, y))
        pygame.draw.rect(
            surface, SIDEBAR_PANEL_BORDER, (x, y, w, h),
            1, border_radius=ui_px(6),
        )

    def _draw_section_header(self, surface, x, y, title, accent):
        """Colored vertical accent + bold-ish title. Returns y after."""
        pygame = self.pygame
        bar_w = ui_px(3)
        bar_h = self._font_large.get_height() - ui_px(2)
        pygame.draw.rect(
            surface, accent,
            (x, y + ui_px(2), bar_w, bar_h), border_radius=ui_px(2),
        )
        img = self._font_large.render(title, True, TEXT_COLOR)
        surface.blit(img, (x + bar_w + ui_px(8), y))
        return y + img.get_height() + ui_px(4)

    def _draw_kv(self, surface, x, y, w, key, value,
                 key_color=None, val_color=None):
        """Compact key/value row: left-aligned key, right-aligned value."""
        kc = key_color if key_color is not None else TEXT_DIM_COLOR
        vc = val_color if val_color is not None else TEXT_COLOR
        key_img = self._font_small.render(str(key), True, kc)
        val_img = self._font_small.render(str(value), True, vc)
        surface.blit(key_img, (x, y))
        surface.blit(val_img, (x + w - val_img.get_width(), y))

    def _draw_mode_pill(self, surface, x, y, w, h):
        pygame = self.pygame
        mode = self.state.mode_label
        if "What-If" in mode:
            bg, fg = MODE_PILL_WHATIF
        elif "Standard" in mode:
            bg, fg = MODE_PILL_STANDARD
        else:
            bg, fg = MODE_PILL_OTHER
        pill = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(pill, bg, (0, 0, w, h), border_radius=ui_px(6))
        pygame.draw.rect(pill, (*fg, 110), (0, 0, w, h), 1,
                         border_radius=ui_px(6))
        surface.blit(pill, (x, y))
        dot_r = ui_px(4)
        dot_x = x + ui_px(12) + dot_r
        dot_y = y + h // 2
        pygame.draw.circle(surface, fg, (dot_x, dot_y), dot_r)
        label = self._font.render(mode, True, fg)
        surface.blit(label, (dot_x + dot_r + ui_px(10),
                             dot_y - label.get_height() // 2))

    def _draw_tir_bar(self, surface, x, y, w, h, bg_arr):
        """Stacked horizontal bar: low(<BG_HYPO_THRESHOLD) / in-range /
        high(>BG_HYPER_THRESHOLD).
        Returns (in_frac, low_frac, high_frac) so the caller can label."""
        pygame = self.pygame
        n = len(bg_arr)
        low_n = int(np.sum(bg_arr < HYPO_THRESHOLD_MGDL))
        high_n = int(np.sum(bg_arr > HYPER_THRESHOLD_MGDL))
        in_n = n - low_n - high_n
        low_f = low_n / n
        in_f = in_n / n
        high_f = high_n / n
        bar = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(bar, (45, 48, 60), (0, 0, w, h),
                         border_radius=ui_px(3))
        cur = 0
        for frac, color in (
            (low_f, TIR_LOW_COLOR),
            (in_f, TIR_IN_RANGE_COLOR),
            (high_f, TIR_HIGH_COLOR),
        ):
            seg_w = int(round(frac * w))
            if seg_w > 0:
                pygame.draw.rect(bar, color, (cur, 0, seg_w, h))
            cur += seg_w
        pygame.draw.rect(bar, (80, 84, 100), (0, 0, w, h), 1,
                         border_radius=ui_px(3))
        surface.blit(bar, (x, y))
        return in_f, low_f, high_f

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _draw_sidebar(self, surface: 'pygame.Surface') -> None:
        pygame = self.pygame
        pygame.draw.rect(surface, SIDEBAR_BG_COLOR, (0, 0, SIDEBAR_WIDTH, self.height))
        pygame.draw.line(surface, (44, 48, 64),
                         (SIDEBAR_WIDTH - 1, 0),
                         (SIDEBAR_WIDTH - 1, self.height))

        # Sidebar is clipped to the area above the status bar; scrolling
        # shifts the content origin by -_sidebar_scroll, so interactive
        # widget rects assigned during this pass already land in screen
        # coords (no extra adjustment in event handlers).
        visible_h = max(self.height - STATUS_BAR_HEIGHT, 1)
        max_scroll = max(0, self._sidebar_content_h - visible_h)
        self._sidebar_scroll = max(0, min(self._sidebar_scroll, max_scroll))
        prev_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(0, 0, SIDEBAR_WIDTH, visible_h))

        pad = ui_px(14)
        inner_x = pad
        inner_w = SIDEBAR_WIDTH - 2 * pad
        y = pad - self._sidebar_scroll

        # ---- App header: title + subtitle ----
        title_img = self._font_large.render("T1DMAI", True, TEXT_COLOR)
        surface.blit(title_img, (inner_x, y))
        y += title_img.get_height()
        sub_img = self._font_small.render(
            "Behavior prediction · interactive what-if",
            True, TEXT_FAINT_COLOR,
        )
        surface.blit(sub_img, (inner_x, y))
        y += sub_img.get_height() + ui_px(10)

        # ---- Mode pill ----
        pill_h = ui_px(30)
        self._draw_mode_pill(surface, inner_x, y, inner_w, pill_h)
        y += pill_h + ui_px(6)

        # ---- Compact system info (one line, key/value pairs) ----
        n_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        sys_rows = [
            ("Model", "loaded" if self.model is not None else "none"),
            ("Params", f"{n_params/1e6:.2f}M"),
            ("Device", str(self.device).upper()),
            ("Pred-start", self.state.active_band_label or "—"),
        ]
        for k, v in sys_rows:
            self._draw_kv(surface, inner_x, y, inner_w, k, v)
            y += LINE_GAP_SM
        y += ui_px(10)

        # ---- Patient card ----
        n_summary_rows = (
            min(5, len(self.state.patient_summary))
            if self.state.patient_summary else 0
        )
        card_h = (
            ui_px(14)  # padding
            + self._font_large.get_height() + ui_px(4)  # header
            + self._font.get_height() + ui_px(4)        # seed row
            + n_summary_rows * LINE_GAP_SM
            + ui_px(10)  # padding
        )
        self._draw_panel_card(surface, inner_x, y, inner_w, card_h)
        cy = y + ui_px(10)
        cx = inner_x + ui_px(12)
        cw = inner_w - ui_px(24)
        cy = self._draw_section_header(surface, cx, cy, "Patient", ACCENT_PATIENT)
        seed_img = self._font.render(
            f"Seed #{self.state.patient_seed}", True, TEXT_COLOR,
        )
        surface.blit(seed_img, (cx, cy))
        cy += seed_img.get_height() + ui_px(4)
        if self.state.patient_summary:
            for k, v in list(self.state.patient_summary.items())[:5]:
                self._draw_kv(
                    surface, cx, cy, cw,
                    k.replace('_', ' '), str(v),
                    val_color=ACCENT_PATIENT,
                )
                cy += LINE_GAP_SM
        y += card_h + ui_px(10)

        # ---- Channels card ----
        ch_card_h = (
            ui_px(14)
            + self._font_large.get_height() + ui_px(4)
            + len(self._toggles) * TOGGLE_ROW_PITCH
            + ui_px(8)
        )
        self._draw_panel_card(surface, inner_x, y, inner_w, ch_card_h)
        cy = y + ui_px(10)
        cx = inner_x + ui_px(12)
        cw = inner_w - ui_px(24)
        cy = self._draw_section_header(surface, cx, cy, "Channels", ACCENT_CHANNELS)
        for i, toggle in enumerate(self._toggles):
            toggle.rect = (cx, cy, cw, toggle.rect[3])
            toggle.draw(surface, self._font_small)
            from gui_state import TOOL_CURVE_EDITOR
            if (self.state.active_tool == TOOL_CURVE_EDITOR
                    and self.state.selected_edit_channel == i):
                pygame.draw.rect(
                    surface, (255, 180, 100),
                    (cx - 2, cy - 2, cw + 4, toggle.rect[3] + 4), 2,
                    border_radius=ui_px(3),
                )
            cy += TOGGLE_ROW_PITCH
        y += ch_card_h + ui_px(10)

        # ---- Mask card ----
        # The masked set is what the model was asked about, so it belongs beside
        # the patient rather than buried in the tool's help text.
        from config import MAX_MASKED_PATCHES
        from gui_state import MASK_PRESET_LABELS
        n_pred_ui = self._n_pred()
        span_rows = self.state.mask_spans[:4]
        mask_card_h = (
            ui_px(14)
            + self._font_large.get_height() + ui_px(4)
            + (3 + len(span_rows)) * LINE_GAP_SM
            + ui_px(8)
        )
        self._draw_panel_card(surface, inner_x, y, inner_w, mask_card_h)
        cy = y + ui_px(10)
        cx = inner_x + ui_px(12)
        cw = inner_w - ui_px(24)
        cy = self._draw_section_header(surface, cx, cy, "Mask", ACCENT_ACTIONS)
        self._draw_kv(surface, cx, cy, cw, "Preset",
                      MASK_PRESET_LABELS.get(self.state.mask_preset, "custom"))
        cy += LINE_GAP_SM
        self._draw_kv(
            surface, cx, cy, cw, "Slots",
            f"{self.state.masked_patch_count(n_pred_ui)} / {MAX_MASKED_PATCHES}"
            f"  ({self.state.mask_budget_left(n_pred_ui)} free)",
        )
        cy += LINE_GAP_SM
        self._draw_kv(surface, cx, cy, cw, "Doses",
                      self.state.masked_channel_policy,
                      val_color=(MASK_OOD_COLOR
                                 if self.state.masked_channel_policy != 'announced'
                                 else TEXT_COLOR))
        cy += LINE_GAP_SM
        for i, span in enumerate(span_rows):
            sel = (i == self.state.selected_mask_idx)
            self._draw_kv(
                surface, cx, cy, cw,
                f"{'▸' if sel else ' '} patches", f"{span.start}–{span.last}",
                val_color=MASK_EDGE_COLOR if sel else TEXT_DIM_COLOR,
            )
            cy += LINE_GAP_SM
        y += mask_card_h + ui_px(10)

        # ---- Display options (smoothing toggles, no card bg) ----
        display_label = self._font_small.render(
            "DISPLAY", True, ACCENT_ACTIONS,
        )
        surface.blit(display_label, (inner_x, y))
        y += display_label.get_height() + ui_px(4)
        for toggle in self._display_toggles:
            toggle.rect = (inner_x, y, inner_w, toggle.rect[3])
            toggle.draw(surface, self._font_small)
            y += TOGGLE_ROW_PITCH
        y += ui_px(4)
        y += ui_px(8)

        # ---- Prediction summary card (only when a prediction exists) ----
        # Scoped to the FORECAST rows. This card is about what comes next —
        # horizon, TIR, the value at the end — and a backcast or infill row is
        # not on that timeline: pooling them would make the horizon the total
        # masked patch count and the TIR a mixture of two different questions.
        pred_bg, n_filled = self._forecast_rows()
        if pred_bg is not None and len(pred_bg) > 0:
            from config import PATCH_SIZE
            n_steps = len(pred_bg)
            horizon_h = n_steps * 5.0 / 60.0  # 5-min steps
            pred_min = float(np.min(pred_bg))
            pred_max = float(np.max(pred_bg))
            pred_mean = float(np.mean(pred_bg))
            pred_end = float(pred_bg[-1])
            bar_h = ui_px(12)
            pred_card_h = (
                ui_px(14)
                + self._font_large.get_height() + ui_px(4)
                + bar_h + ui_px(6)               # TIR bar
                + self._font_small.get_height() + ui_px(8)  # TIR label
                + (5 if n_filled else 4) * LINE_GAP_SM
                + SECTION_GAP_SM + 3 * LINE_GAP_SM  # Clock sub-block (up to 3 rows)
                + ui_px(8)
            )
            self._draw_panel_card(surface, inner_x, y, inner_w, pred_card_h)
            cy = y + ui_px(10)
            cx = inner_x + ui_px(12)
            cw = inner_w - ui_px(24)
            cy = self._draw_section_header(
                surface, cx, cy, "Prediction", ACCENT_PREDICTION,
            )
            in_f, low_f, high_f = self._draw_tir_bar(
                surface, cx, cy, cw, bar_h, pred_bg,
            )
            cy += bar_h + ui_px(6)
            parts = [f"TIR {in_f:.0%}"]
            if low_f > 0.005:
                parts.append(f"Low {low_f:.0%}")
            if high_f > 0.005:
                parts.append(f"High {high_f:.0%}")
            tir_label = self._font_small.render(
                "  ·  ".join(parts), True, TEXT_DIM_COLOR,
            )
            surface.blit(tir_label, (cx, cy))
            cy += tir_label.get_height() + ui_px(8)

            def _bg_color(v: float) -> tuple[int, int, int]:
                if v < HYPO_THRESHOLD_MGDL:
                    return TIR_LOW_COLOR
                if v > HYPER_THRESHOLD_MGDL:
                    return TIR_HIGH_COLOR
                return TIR_IN_RANGE_COLOR

            self._draw_kv(surface, cx, cy, cw, "Horizon", f"{horizon_h:.1f} h")
            cy += LINE_GAP_SM
            if n_filled:
                self._draw_kv(surface, cx, cy, cw, "Also filled",
                              f"{n_filled} masked patches",
                              val_color=MASK_EDGE_COLOR)
                cy += LINE_GAP_SM
            self._draw_kv(surface, cx, cy, cw, "Range",
                          f"{pred_min:.0f} – {pred_max:.0f} mg/dL",
                          val_color=TEXT_COLOR)
            cy += LINE_GAP_SM
            self._draw_kv(surface, cx, cy, cw, "Mean BG",
                          f"{pred_mean:.0f} mg/dL",
                          val_color=_bg_color(pred_mean))
            cy += LINE_GAP_SM
            self._draw_kv(surface, cx, cy, cw, f"At +{horizon_h:.1f}h",
                          f"{pred_end:.0f} mg/dL",
                          val_color=_bg_color(pred_end))
            cy += LINE_GAP_SM

            # ---- Clock sub-block: model-decoded prediction-origin hour-of-day
            # (the diagnostic time-of-day probe) beside the true origin clock,
            # with the resultant length R of the per-bin softmax belief as a
            # confidence read-out. ----
            cy += SECTION_GAP_SM
            true_h = _hour_at_pred_start(self.state)
            self._draw_kv(surface, cx, cy, cw, "Origin (true)",
                          f"{true_h:04.1f}h")
            cy += LINE_GAP_SM
            tod_h = getattr(self.state.prediction, 'tod_pred_hour', None)
            tod_R = getattr(self.state.prediction, 'tod_confidence', None)
            if tod_h is not None:
                d = abs(tod_h - true_h) % 24.0
                err = min(d, 24.0 - d)  # circular hour error, in [0, 12]
                if err < 1.0:
                    clk_color = TIR_IN_RANGE_COLOR
                elif err < 3.0:
                    clk_color = TIR_HIGH_COLOR
                else:
                    clk_color = TIR_LOW_COLOR
                r_txt = f"  R{tod_R:.2f}" if tod_R is not None else ""
                self._draw_kv(surface, cx, cy, cw, "Origin (pred)",
                              f"{tod_h:04.1f}h{r_txt}", val_color=clk_color)
                cy += LINE_GAP_SM
                self._draw_kv(surface, cx, cy, cw, "Clock err",
                              f"{err:.1f} h", val_color=clk_color)
                cy += LINE_GAP_SM
            else:
                self._draw_kv(surface, cx, cy, cw, "Origin (pred)",
                              "— (probe off)", val_color=TEXT_DIM_COLOR)
                cy += LINE_GAP_SM
            y += pred_card_h + ui_px(10)

        # ---- Model Score card (last Eval-vs-Sim result) ----
        ev = self.state.last_eval
        if ev is not None:
            score_card_h = (
                ui_px(14)
                + self._font_large.get_height() + ui_px(4)
                + 5 * LINE_GAP_SM
                + ui_px(8)
            )
            self._draw_panel_card(surface, inner_x, y, inner_w, score_card_h)
            cy = y + ui_px(10)
            cx = inner_x + ui_px(12)
            cw = inner_w - ui_px(24)
            cy = self._draw_section_header(
                surface, cx, cy, "Model Score", ACCENT_EVAL,
            )
            # Color the MAE by how good it is: <15 mg/dL green,
            # <30 yellow, otherwise red. Same scale roughly tracks
            # clinically-meaningful CGM error tiers.
            if ev.mae < 15.0:
                mae_color = TIR_IN_RANGE_COLOR
            elif ev.mae < 30.0:
                mae_color = TIR_HIGH_COLOR
            else:
                mae_color = TIR_LOW_COLOR
            self._draw_kv(surface, cx, cy, cw, "Window",
                          f"{ev.horizon_h:.1f} h · {ev.n_steps} steps")
            cy += LINE_GAP_SM
            self._draw_kv(surface, cx, cy, cw, "MAE",
                          f"{ev.mae:.1f} mg/dL", val_color=mae_color)
            cy += LINE_GAP_SM
            self._draw_kv(surface, cx, cy, cw, "RMSE",
                          f"{ev.rmse:.1f} mg/dL")
            cy += LINE_GAP_SM
            bias_color = (
                TIR_HIGH_COLOR if ev.bias > 0 else TIR_LOW_COLOR
                if ev.bias < 0 else TEXT_COLOR
            )
            self._draw_kv(surface, cx, cy, cw, "Bias",
                          f"{ev.bias:+.1f} mg/dL", val_color=bias_color)
            cy += LINE_GAP_SM
            self._draw_kv(surface, cx, cy, cw, "Max |err|",
                          f"{ev.max_abs:.1f} mg/dL")
            y += score_card_h + ui_px(10)

        # ---- Action buttons (no panel; sit directly on sidebar bg) ----
        actions_label = self._font_small.render(
            "ACTIONS", True, ACCENT_ACTIONS,
        )
        surface.blit(actions_label, (inner_x, y))
        y += actions_label.get_height() + ui_px(4)
        btn_w = (inner_w - ui_px(10)) // 2
        for i, btn in enumerate(self._buttons):
            col = i % 2
            row = i // 2
            btn.rect = (inner_x + col * (btn_w + ui_px(10)),
                        y + row * BUTTON_ROW_PITCH, btn_w, BUTTON_HEIGHT)
            btn.draw(surface, self._font_small)
        y += ((len(self._buttons) + 1) // 2) * BUTTON_ROW_PITCH

        # ---- Active edits footer (basal / curves badges) ----
        y += ui_px(8)
        active_bits = []
        if self.state.basal_rate_delta != 0.0:
            active_bits.append(
                ("Basal", f"{self.state.basal_rate_delta:+.0f} U/h",
                 ACCENT_OVERRIDES),
            )
        if self.state.curve_events:
            active_bits.append(
                ("Curves", str(len(self.state.curve_events)),
                 ACCENT_OVERRIDES),
            )
        if active_bits:
            for k, v, color in active_bits:
                self._draw_kv(surface, inner_x, y, inner_w, k, v,
                              val_color=color)
                y += LINE_GAP_SM

        # Record content height (in unscrolled coords) for the wheel
        # handler / scrollbar to use on the next frame, then drop clip.
        self._sidebar_content_h = y + self._sidebar_scroll + pad
        surface.set_clip(prev_clip)

        # Scrollbar overlay, drawn on top of the sidebar (outside the
        # content clip so it isn't itself scrolled).
        if self._sidebar_content_h > visible_h:
            track_w = ui_px(6)
            track_x = SIDEBAR_WIDTH - track_w - ui_px(3)
            track_y = ui_px(4)
            track_h = visible_h - ui_px(8)
            pygame.draw.rect(
                surface, (40, 42, 56),
                (track_x, track_y, track_w, track_h),
                border_radius=track_w // 2,
            )
            thumb_h = max(
                ui_px(24),
                int(track_h * visible_h / self._sidebar_content_h),
            )
            thumb_range = max(track_h - thumb_h, 1)
            scroll_frac = (
                self._sidebar_scroll / max_scroll if max_scroll > 0 else 0.0
            )
            thumb_y = track_y + int(scroll_frac * thumb_range)
            pygame.draw.rect(
                surface, (110, 115, 140),
                (track_x, thumb_y, track_w, thumb_h),
                border_radius=track_w // 2,
            )

    def _is_in_sidebar(self, mx: int, my: int) -> bool:
        return (0 <= mx < SIDEBAR_WIDTH
                and 0 <= my < self.height - STATUS_BAR_HEIGHT)

    def _right_panel_w(self) -> int:
        return RIGHT_PANEL_WIDTH if self.state.events_panel_visible else 0

    def _right_panel_x(self) -> int:
        return self.width - self._right_panel_w()

    def _is_in_events_panel(self, mx: int, my: int) -> bool:
        w = self._right_panel_w()
        if w == 0:
            return False
        x0 = self._right_panel_x()
        return (x0 <= mx < x0 + w
                and 0 <= my < self.height - STATUS_BAR_HEIGHT)

    # ------------------------------------------------------------------
    # Events right panel
    # ------------------------------------------------------------------

    _EVENT_CREATE_BUTTONS: list[tuple[str, str, str]] = [
        # (kind, meal_name, label)
        ('juice',         '',          '+ Juice'),
        ('fast_insulin',  '',          '+ Fast Ins'),
        ('basal_insulin', '',          '+ Basal'),
        ('meal',          'breakfast', '+ Breakfast'),
        ('meal',          'lunch',     '+ Lunch'),
        ('meal',          'dinner',    '+ Dinner'),
    ]

    @staticmethod
    def _event_summary(ev) -> tuple[str, tuple]:
        """Return (label, color) for an event-list row."""
        kind = ev.kind
        t = ev.time_offset_min
        sign = '+' if t >= 0 else '−'
        t_abs = abs(t)
        if kind == 'juice':
            return (f"Juice {ev.magnitude:.0f}g  {sign}{t_abs:.0f}m", COLOR_CARBS)
        if kind == 'fast_insulin':
            return (f"Fast {ev.magnitude:.1f}U  {sign}{t_abs:.0f}m", COLOR_INSULIN)
        if kind == 'basal_insulin':
            return (f"Basal {ev.magnitude:.0f}U  {sign}{t_abs:.0f}m", COLOR_INSULIN)
        if kind == 'meal':
            name = (ev.name or 'meal').capitalize()
            return (f"{name} {ev.magnitude:.0f}g  {sign}{t_abs:.0f}m", COLOR_CARBS)
        return (str(kind), TEXT_DIM_COLOR)

    def _draw_events_panel(self, surface: 'pygame.Surface') -> None:
        if not self.state.events_panel_visible:
            self._event_create_rects = []
            self._event_row_rects = []
            return
        pygame = self.pygame
        w = self._right_panel_w()
        x0 = self._right_panel_x()
        panel_h = self.height - STATUS_BAR_HEIGHT

        pygame.draw.rect(surface, SIDEBAR_BG_COLOR, (x0, 0, w, panel_h))
        pygame.draw.line(surface, (44, 48, 64),
                         (x0, 0), (x0, panel_h))

        pad = ui_px(12)
        inner_x = x0 + pad
        inner_w = w - 2 * pad

        prev_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(x0, 0, w, panel_h))

        y = pad - self._events_panel_scroll

        title_img = self._font_large.render("Events", True, TEXT_COLOR)
        surface.blit(title_img, (inner_x, y))
        y += title_img.get_height() + ui_px(4)
        sub = self._font_small.render(
            "Click a button to add an event.",
            True, TEXT_FAINT_COLOR,
        )
        surface.blit(sub, (inner_x, y))
        y += sub.get_height() + ui_px(10)

        # 2-column grid of create-event buttons.
        btn_h = ui_px(30)
        btn_gap = ui_px(8)
        col_w = (inner_w - btn_gap) // 2
        self._event_create_rects = []
        for i, (kind, meal_name, label) in enumerate(self._EVENT_CREATE_BUTTONS):
            row = i // 2
            col = i % 2
            bx = inner_x + col * (col_w + btn_gap)
            by = y + row * (btn_h + btn_gap)
            color = COLOR_CARBS if kind in ('juice', 'meal') else COLOR_INSULIN
            # Dim background, colored left border, white-ish label.
            pygame.draw.rect(surface, (44, 48, 64),
                             (bx, by, col_w, btn_h),
                             border_radius=ui_px(4))
            pygame.draw.rect(surface, color,
                             (bx, by, ui_px(3), btn_h),
                             border_radius=ui_px(2))
            img = self._font_small.render(label, True, TEXT_COLOR)
            surface.blit(
                img,
                (bx + ui_px(10),
                 by + (btn_h - img.get_height()) // 2),
            )
            self._event_create_rects.append((kind, meal_name, (bx, by, col_w, btn_h)))

        n_rows = (len(self._EVENT_CREATE_BUTTONS) + 1) // 2
        y += n_rows * (btn_h + btn_gap) + ui_px(8)

        pygame.draw.line(surface, (56, 60, 80),
                         (inner_x, y), (inner_x + inner_w, y))
        y += ui_px(10)

        if not self.state.events:
            empty = self._font_small.render(
                "No events yet.", True, TEXT_FAINT_COLOR,
            )
            surface.blit(empty, (inner_x, y))
            y += empty.get_height() + ui_px(4)
        else:
            count_img = self._font_small.render(
                f"{len(self.state.events)} event(s):",
                True, TEXT_DIM_COLOR,
            )
            surface.blit(count_img, (inner_x, y))
            y += count_img.get_height() + ui_px(6)

        row_h = ui_px(28)
        row_gap = ui_px(4)
        del_w = ui_px(22)
        self._event_row_rects = []
        for idx, ev in enumerate(self.state.events):
            label, color = self._event_summary(ev)
            row_rect = (inner_x, y, inner_w - del_w - ui_px(4), row_h)
            del_rect = (inner_x + inner_w - del_w, y, del_w, row_h)

            rx, ry, rw, rh = row_rect
            pygame.draw.rect(surface, (38, 42, 58),
                             (rx, ry, rw, rh),
                             border_radius=ui_px(3))
            pygame.draw.rect(surface, color,
                             (rx, ry, ui_px(3), rh),
                             border_radius=ui_px(2))
            img = self._font_small.render(label, True, TEXT_COLOR)
            surface.blit(
                img,
                (rx + ui_px(10),
                 ry + (rh - img.get_height()) // 2),
            )

            dx, dy, dw, dh = del_rect
            pygame.draw.rect(surface, (70, 36, 44),
                             (dx, dy, dw, dh),
                             border_radius=ui_px(3))
            x_img = self._font_small.render("×", True, (240, 200, 200))
            surface.blit(
                x_img,
                (dx + (dw - x_img.get_width()) // 2,
                 dy + (dh - x_img.get_height()) // 2),
            )

            self._event_row_rects.append((idx, row_rect, del_rect))
            y += row_h + row_gap

        content_bottom = y + self._events_panel_scroll
        self._events_panel_content_h = content_bottom + pad

        surface.set_clip(prev_clip)

        max_scroll = max(0, self._events_panel_content_h - panel_h)
        if max_scroll > 0:
            track_w = ui_px(4)
            track_x = x0 + w - track_w - ui_px(3)
            thumb_h = max(ui_px(24),
                          int(panel_h * panel_h / self._events_panel_content_h))
            thumb_y = int(self._events_panel_scroll
                          * (panel_h - thumb_h) / max_scroll)
            pygame.draw.rect(surface, (110, 115, 140),
                             (track_x, thumb_y, track_w, thumb_h),
                             border_radius=track_w // 2)

    def _open_event_editor(self, kind: str, meal_name: str = '',
                           editing_idx: int = -1) -> None:
        from gui_state import Event
        initial: dict | None = None
        if editing_idx >= 0 and editing_idx < len(self.state.events):
            ev = self.state.events[editing_idx]
            initial = {
                'time_offset_min': ev.time_offset_min,
                'magnitude': ev.magnitude,
                'fast_frac': ev.fast_frac,
            }
            kind = ev.kind
            meal_name = ev.name

        def on_save(values: dict) -> None:
            new_ev = Event(
                kind=values['kind'],
                time_offset_min=float(values.get('time_offset_min', 0.0)),
                magnitude=float(values.get('magnitude', 0.0)),
                fast_frac=float(values.get('fast_frac', 0.5)),
                name=values.get('name', meal_name),
            )
            if editing_idx >= 0 and editing_idx < len(self.state.events):
                self.state.events[editing_idx] = new_ev
            else:
                self.state.events.append(new_ev)
            self._compile_curve_overrides()
            self._needs_redraw = True

        def on_delete() -> None:
            if 0 <= editing_idx < len(self.state.events):
                del self.state.events[editing_idx]
                self._compile_curve_overrides()
                self._needs_redraw = True

        self._event_editor.open_for(
            kind=kind,
            save_callback=on_save,
            initial=initial,
            editing_idx=editing_idx,
            delete_callback=on_delete if editing_idx >= 0 else None,
            meal_name=meal_name,
        )
        self._needs_redraw = True

    def _handle_events_panel_click(self, mx: int, my: int) -> bool:
        for kind, meal_name, rect in self._event_create_rects:
            x, y, w, h = rect
            if x <= mx < x + w and y <= my < y + h:
                if self._refuse_dose_painting():
                    return True
                self._open_event_editor(kind, meal_name=meal_name)
                return True
        for idx, row_rect, del_rect in self._event_row_rects:
            dx, dy, dw, dh = del_rect
            if dx <= mx < dx + dw and dy <= my < dy + dh:
                if 0 <= idx < len(self.state.events):
                    del self.state.events[idx]
                    self._compile_curve_overrides()
                    self._needs_redraw = True
                return True
            rx, ry, rw, rh = row_rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                self._open_event_editor('', editing_idx=idx)
                return True
        return False

    def _draw_chart(self) -> 'pygame.Surface':
        pygame = self.pygame
        from gui_renderer import (
            ChartTransform, draw_curve, draw_grid, draw_now_line, draw_y_band,
        )
        from config import PATCH_SIZE

        w = int(self.chart_transform.sw)
        h = int(self.chart_transform.sh)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill(CHART_BG_COLOR)

        n_ctx = self.state.context.shape[0] if self.state.context is not None else 0

        local_transform = ChartTransform(
            screen_x=0, screen_y=0,
            screen_w=w, screen_h=h,
            chart_x_min=self.chart_transform.cx_min,
            chart_x_max=self.chart_transform.cx_max,
            chart_y_min=self.chart_transform.cy_min,
            chart_y_max=self.chart_transform.cy_max,
        )
        cy_min = local_transform.cy_min
        cy_max = local_transform.cy_max

        # Glucose-range shading. The BG channel's raw range maps mg/dL into
        # chart-y space, so the band edges track any vertical pan/zoom.
        bg_raw_min, bg_raw_max = DISPLAY_CHANNEL_RAW_RANGES[0]
        cy_hypo = float(_scale_to_chart_y(
            np.array([HYPO_THRESHOLD_MGDL], dtype=np.float32),
            bg_raw_min, bg_raw_max, cy_min, cy_max,
        )[0])
        cy_hyper = float(_scale_to_chart_y(
            np.array([HYPER_THRESHOLD_MGDL], dtype=np.float32),
            bg_raw_min, bg_raw_max, cy_min, cy_max,
        )[0])
        draw_y_band(surf, local_transform, cy_min, cy_hypo,
                    HYPO_BAND_COLOR, GLUCOSE_BAND_ALPHA)
        draw_y_band(surf, local_transform, cy_hypo, cy_hyper,
                    IN_RANGE_BAND_COLOR, GLUCOSE_BAND_ALPHA)
        draw_y_band(surf, local_transform, cy_hyper, cy_max,
                    HYPER_BAND_COLOR, GLUCOSE_BAND_ALPHA)

        if n_ctx > 0:
            # The model only ever sees the trailing MAX_CONTEXT_PATCHES
            # patches before NOW, even if the chart shows more accumulated
            # history. As predictions roll forward and n_ctx grows past
            # the cap, the visible window slides with it.
            from config import MAX_CONTEXT_PATCHES
            ctx_start = max(0, n_ctx - MAX_CONTEXT_PATCHES)
            ctx_sx_l = int(max(0.0, local_transform.x_to_screen(ctx_start)))
            ctx_sx_r = int(min(float(w), local_transform.x_to_screen(n_ctx)))
            if ctx_sx_r > ctx_sx_l:
                ctx_surf = pygame.Surface(
                    (ctx_sx_r - ctx_sx_l, h), pygame.SRCALPHA,
                )
                ctx_surf.fill((*CONTEXT_SHADE_COLOR, CONTEXT_SHADE_ALPHA))
                surf.blit(ctx_surf, (ctx_sx_l, 0))

        pred_sx = int(self.chart_transform.x_to_screen(n_ctx) - self.chart_transform.sx)
        if 0 <= pred_sx < w:
            pred_surf = pygame.Surface((w - pred_sx, h), pygame.SRCALPHA)
            pred_surf.fill((40, 40, 55, 30))
            surf.blit(pred_surf, (pred_sx, 0))

        grid_major, grid_minor, _ = _adaptive_time_intervals(
            float(local_transform.cx_max - local_transform.cx_min),
        )
        draw_grid(
            surf, local_transform, n_ctx,
            major_interval_patches=grid_major,
            minor_interval_patches=grid_minor,
            font=None, text_color=TEXT_DIM_COLOR,
        )

        median_bg = self.state.prediction.median_bg
        bands = self.state.prediction.bands
        overrides_raw = self.state.prediction.overrides_raw
        has_pred = (
            median_bg is not None and bands is not None
            and self.state.norm_stats is not None
        )
        S = bands.shape[1] if has_pred else PATCH_SIZE

        for disp_ch in range(N_DISPLAY_CHANNELS):
            if not self.state.channel_visible[disp_ch]:
                continue
            color = CHANNEL_COLORS[disp_ch]
            raw_min, raw_max = DISPLAY_CHANNEL_RAW_RANGES[disp_ch]

            # --- Context history curve ---
            ctx_vals: np.ndarray | None = None
            if disp_ch == 0 and self.state.bg_raw is not None:
                ctx_vals = self.state.bg_raw[:n_ctx * PATCH_SIZE]
            elif self.state.context_raw is not None and disp_ch < self.state.context_raw.shape[1]:
                ctx_vals = self.state.context_raw[:n_ctx * PATCH_SIZE, disp_ch]

            if ctx_vals is not None and len(ctx_vals) >= 2:
                visible_start = max(
                    0, int(local_transform.cx_min * PATCH_SIZE) - 1
                )
                visible_end = min(
                    len(ctx_vals),
                    int(math.ceil(local_transform.cx_max * PATCH_SIZE)) + 1,
                )
                if visible_end - visible_start >= 2:
                    ctx_times = (
                        np.arange(visible_start, visible_end, dtype=np.float32)
                        / PATCH_SIZE
                    )
                    y_ctx = _scale_to_chart_y(
                        ctx_vals[visible_start:visible_end],
                        raw_min, raw_max, cy_min, cy_max,
                    )
                    if disp_ch == 0 and self.state.mask_spans:
                        # BG under a masked span is what the model is being asked
                        # to produce; drawing the truth there turns the fan into
                        # a comparison the user did not ask for. Break the line
                        # over each span instead. The other channels ride through
                        # a masked patch under the announced policy, so they stay
                        # whole; under 'blind' the model does not see them, but
                        # what the chart shows is the patient's record either way.
                        for seg_t, seg_y in _split_at_masked(
                            ctx_times, y_ctx, self.state.mask_spans,
                        ):
                            draw_curve(surf, local_transform, seg_t, seg_y,
                                       color, width=2)
                    else:
                        draw_curve(surf, local_transform, ctx_times, y_ctx,
                                   color, width=2)

            # --- BG forecast: median_bg line + per-τ quantile envelope ---
            # One fan PER SPAN, placed where the head read it. ``span_patches`` is
            # ``predict``'s own ``mask_idx``: slot j is patch mask_idx[j], so a
            # fixed offset from the context end is right only for the forecast.
            if has_pred and disp_ch == 0:
                pred_bg = np.asarray(median_bg, dtype=np.float32)
                P = bands.shape[0]
                rows = pred_bg.reshape(P, -1)
                patches = self.state.prediction.span_patches
                if patches is None or len(patches) != P:
                    patches = n_ctx + np.arange(P, dtype=np.int64)
                for lo, hi in _contiguous_runs(np.asarray(patches, dtype=np.int64)):
                    seg = rows[lo:hi].reshape(-1)
                    seg_drawn = (
                        _smooth_1d(seg, MU_SMOOTH_STEPS)
                        if self.state.smooth_mu else seg
                    )
                    # Outermost quantile pair (ascending τ) is the widest band;
                    # use it as the ~95% uncertainty envelope.
                    bflat = bands[lo:hi].reshape(-1, bands.shape[-1])
                    upper_raw = bflat[:, -1].astype(np.float32)
                    lower_raw = bflat[:, 0].astype(np.float32)
                    if self.state.smooth_band:
                        upper_raw = _smooth_1d(upper_raw, CONFIDENCE_BAND_SMOOTH_STEPS)
                        lower_raw = _smooth_1d(lower_raw, CONFIDENCE_BAND_SMOOTH_STEPS)
                    start_patch = float(patches[lo])
                    pred_times = start_patch + np.arange(
                        len(seg_drawn), dtype=np.float32) / S
                    y_up = _scale_to_chart_y(upper_raw, raw_min, raw_max, cy_min, cy_max)
                    y_lo = _scale_to_chart_y(lower_raw, raw_min, raw_max, cy_min, cy_max)
                    _draw_band_polygon(surf, local_transform, pred_times, y_up, y_lo,
                                       color, alpha=CONFIDENCE_ALPHA)
                    y_pred = _scale_to_chart_y(seg_drawn, raw_min, raw_max, cy_min, cy_max)
                    draw_curve(surf, local_transform, pred_times, y_pred, color,
                               width=2, alpha=PREDICTION_LINE_ALPHA)

            # --- Announced carb / insulin / exercise input in the pred zone ---
            out_ch = DISPLAY_TO_OUTPUT_CH.get(disp_ch)
            if (out_ch is not None and overrides_raw is not None
                    and out_ch in overrides_raw):
                ov = np.asarray(overrides_raw[out_ch], dtype=np.float32)
                ov_flat = ov.flatten()
                ov_times = n_ctx + np.arange(len(ov_flat), dtype=np.float32) / ov.shape[1]
                y_ov = _scale_to_chart_y(ov_flat, raw_min, raw_max, cy_min, cy_max)
                draw_curve(surf, local_transform, ov_times, y_ov, color,
                           width=2, alpha=PREDICTION_LINE_ALPHA)

        self._draw_curve_events(surf, local_transform, n_ctx)
        self._draw_mask_spans(surf, local_transform, n_ctx)

        draw_now_line(surf, local_transform, n_ctx,
                      font=self._font_small, text_color=TEXT_DIM_COLOR)

        return surf

    def _draw_mask_spans(
        self,
        surf: 'pygame.Surface',
        local_transform,
        n_ctx: int,
    ) -> None:
        """Shade the user's masked spans, and the drag in flight.

        The trailing forecast span is not drawn here — the NOW line and the
        prediction-zone shading already mark it, and it is not something the user
        can remove.
        """
        pygame = self.pygame
        h = int(local_transform.sh)
        w = int(local_transform.sw)
        if h <= 0 or w <= 0:
            return

        ood = {}
        if self.state.mask_spans and self.state.context is not None:
            from gui_state import mask_span_ood
            n_pred = self._n_pred()
            try:
                emitted = self.state.emitted_mask_spans(n_pred)
                ood = mask_span_ood(emitted, n_ctx, n_pred)
            except AssertionError:
                ood = {}

        def _shade(lo: float, hi: float, alpha: int, edge: bool) -> None:
            sx_l = int(max(0.0, local_transform.x_to_screen(lo)))
            sx_r = int(min(float(w), local_transform.x_to_screen(hi)))
            if sx_r <= sx_l:
                return
            band = pygame.Surface((sx_r - sx_l, h), pygame.SRCALPHA)
            band.fill((*MASK_SPAN_COLOR, alpha))
            surf.blit(band, (sx_l, 0))
            if edge:
                pygame.draw.line(surf, MASK_EDGE_COLOR, (sx_l, 0), (sx_l, h), 1)
                pygame.draw.line(surf, MASK_EDGE_COLOR, (sx_r - 1, 0), (sx_r - 1, h), 1)

        for idx, span in enumerate(self.state.mask_spans):
            selected = (idx == self.state.selected_mask_idx)
            _shade(float(span.start), float(span.end),
                   MASK_SPAN_SELECTED_ALPHA if selected else MASK_SPAN_ALPHA,
                   edge=True)
            label = f"{span.start}–{span.last}"
            sx = int(local_transform.x_to_screen(float(span.start))) + 4
            img = self._font_small.render(label, True, MASK_EDGE_COLOR)
            surf.blit(img, (sx, 4))
            # OOD is a HINT, never a block: the fan is still drawn, it is simply
            # not calibrated by anything the sampler supervised.
            if idx in ood:
                warn = self._font_small.render("OOD", True, MASK_OOD_COLOR)
                surf.blit(warn, (sx, 4 + img.get_height() + 2))

        if self.state.mask_drag_start >= 0:
            lo = min(self.state.mask_drag_start, self.state.mask_drag_end)
            hi = max(self.state.mask_drag_start, self.state.mask_drag_end)
            _shade(float(lo), float(hi + 1), MASK_DRAG_ALPHA, edge=False)

    def _draw_curve_events(
        self,
        surf: 'pygame.Surface',
        local_transform,
        n_ctx: int,
    ) -> None:
        pygame = self.pygame
        from config import PATCH_SIZE

        n_pred = self._preview_horizon_patches()

        for evt_idx, event in enumerate(self.state.curve_events):
            disp_ch = event.channel
            if not self.state.channel_visible[disp_ch]:
                continue

            raw_min, raw_max = DISPLAY_CHANNEL_RAW_RANGES[disp_ch]
            color = CHANNEL_COLORS[disp_ch]
            cy_min = local_transform.cy_min
            cy_max = local_transform.cy_max

            if n_pred == 0:
                continue

            raw_vals = _curve_event_to_raw_values(event, n_ctx, n_pred, PATCH_SIZE)

            times_list: list[float] = []
            y_list: list[float] = []
            for p in range(n_pred):
                for s in range(PATCH_SIZE):
                    abs_pos = float(n_ctx + p + s / PATCH_SIZE)
                    times_list.append(abs_pos)
                    y_list.append(float(raw_vals[p, s]))

            times_arr = np.array(times_list, dtype=np.float32)
            y_vals = np.array(y_list, dtype=np.float32)
            y_curve = _scale_to_chart_y(y_vals, raw_min, raw_max, cy_min, cy_max)
            y_baseline_val = _scale_to_chart_y(
                np.array([raw_min]), raw_min, raw_max, cy_min, cy_max,
            )[0]

            upper_pts: list[tuple[int, int]] = []
            lower_pts: list[tuple[int, int]] = []
            for i in range(len(times_arr)):
                sx, sy = local_transform.chart_to_screen(times_arr[i], y_curve[i])
                upper_pts.append((int(sx), int(sy)))
                sx2, sy2 = local_transform.chart_to_screen(times_arr[i], y_baseline_val)
                lower_pts.append((int(sx2), int(sy2)))

            polygon = upper_pts + lower_pts[::-1]
            if len(polygon) >= 3:
                poly_w = int(local_transform.sw)
                poly_h = int(local_transform.sh)
                if poly_w > 0 and poly_h > 0:
                    poly_surf = pygame.Surface((poly_w, poly_h), pygame.SRCALPHA)
                    pygame.draw.polygon(poly_surf, (*color, CURVE_EVENT_FILL_ALPHA), polygon)
                    surf.blit(poly_surf, (0, 0))

            is_selected = (evt_idx == self.state.selected_event_idx)

            for point_name, abs_patch, raw_val in [
                ('left', event.left_patch, raw_min),
                ('peak', event.peak_patch, event.amplitude),
                ('right', event.right_patch, raw_min),
            ]:
                cy_pt = float(_scale_to_chart_y(
                    np.array([raw_val]), raw_min, raw_max, cy_min, cy_max,
                )[0])
                sx, sy = local_transform.chart_to_screen(abs_patch, cy_pt)
                sx, sy = int(sx), int(sy)
                radius = OVERRIDE_POINT_RADIUS
                if is_selected:
                    if point_name == self.state.dragging_point:
                        pygame.draw.circle(surf, (255, 255, 255), (sx, sy), radius + 2)
                    pygame.draw.circle(surf, color, (sx, sy), radius)
                    pygame.draw.circle(surf, (255, 255, 255), (sx, sy), radius, 1)
                else:
                    pygame.draw.circle(surf, color, (sx, sy), radius - 1)

            auc = event.auc_raw(n_ctx, n_pred, PATCH_SIZE)
            if disp_ch == 1:
                auc_label = f"AUC: {auc:.1f} g"
            elif disp_ch == 2:
                auc_label = f"AUC: {auc:.2f} U"
            else:
                auc_label = f"AUC: {auc:.1f}"
            cy_peak = float(_scale_to_chart_y(
                np.array([event.amplitude]), raw_min, raw_max, cy_min, cy_max,
            )[0])
            sx, sy = local_transform.chart_to_screen(event.peak_patch, cy_peak)
            img = self._font_small.render(auc_label, True, color)
            surf.blit(img, (int(sx) + 8, int(sy) - 16))

    def _draw_control_panel(self, surface: 'pygame.Surface') -> None:
        pygame = self.pygame
        cp_x, cp_y, cp_w, cp_h = self.control_panel_rect
        pygame.draw.rect(surface, (22, 22, 30), (cp_x, cp_y, cp_w, cp_h))
        pygame.draw.line(surface, GRID_COLOR, (cp_x, cp_y), (cp_x + cp_w, cp_y))

        from gui_state import TOOL_CURVE_EDITOR, TOOL_MASK, TOOL_PENCIL
        tool = self.state.active_tool

        if tool == TOOL_MASK:
            from config import MAX_MASKED_PATCHES
            n_pred = self._n_pred()
            n_ctx = self.state.n_ctx()
            lines = [
                f"Mask — drag over the context to mask patches. "
                f"{self.state.masked_patch_count(n_pred)}/{MAX_MASKED_PATCHES} slots, "
                f"{self.state.mask_budget_left(n_pred)} free.",
                f"1 = forecast  2 = begin-fill  3 = infill. Click a span to select "
                f"(its anchor is shown), Ctrl+click or Del to remove.",
                f"The trailing {n_pred} patches from {n_ctx} on are always masked — "
                f"the future carries no reading. M = exit tool.",
            ]
            line_h = self._font_small.get_height() + ui_px(4)
            for j, line in enumerate(lines):
                img = self._font_small.render(line, True, TEXT_COLOR)
                surface.blit(img, (cp_x + 10, cp_y + 10 + j * line_h))
            img = self._font_small.render(
                self._anchor_readout(), True, MASK_EDGE_COLOR)
            surface.blit(img, (cp_x + 10, cp_y + 10 + len(lines) * line_h))
        elif tool == TOOL_CURVE_EDITOR:
            ch_name = self.state.channel_names[self.state.selected_edit_channel]
            lines = [
                f"Curve Editor — channel: {ch_name}  (Tab to cycle)",
                "Click in PRED zone to add curve. Drag points to edit. Del = remove. Ctrl+Z = undo.",
                "Basal +/-1 U/h buttons in sidebar. Enter = predict with overrides. W/E = exit tool",
            ]
            line_h = self._font_small.get_height() + ui_px(4)
            for j, line in enumerate(lines):
                img = self._font_small.render(line, True, TEXT_COLOR)
                surface.blit(img, (cp_x + 10, cp_y + 10 + j * line_h))
        elif tool == TOOL_PENCIL:
            ch_name = self.state.channel_names[self.state.selected_edit_channel]
            lines = [
                f"Pencil — channel: {ch_name}  (Tab to cycle carb/insulin/exercise)",
                "Drag in the PRED zone to draw a dose curve (smoothed). Draw past 2 h to plan ahead.",
                "L / Shift+Space = long predict (covers your drawing). Ctrl+Z = undo. P = exit tool",
            ]
            line_h = self._font_small.get_height() + ui_px(4)
            for j, line in enumerate(lines):
                img = self._font_small.render(line, True, TEXT_COLOR)
                surface.blit(img, (cp_x + 10, cp_y + 10 + j * line_h))
        else:
            lines = [
                "SPC=Predict 2h  L/Shift+SPC=Long Predict  W=What-If  P=Pencil  M=Mask  F=Roll  G=Sim Fwd  V=Eval  R=Reset",
                "E=Curve Editor  Tab=Cycle edit channel  C=Clear All curves  N=New Patient  S=Screenshot",
                "1-4=Toggle channels (BG/Carbs/Insulin/Exercise)  A=Toggle all  Q=Quit",
                "Scroll=Zoom at cursor  +/-=Zoom  Middle/Right-drag=Pan",
            ]
            line_h = self._font_small.get_height() + ui_px(4)
            for j, line in enumerate(lines):
                img = self._font_small.render(line, True, TEXT_DIM_COLOR)
                surface.blit(img, (cp_x + 10, cp_y + 10 + j * line_h))

    def _draw_axes(self, surface: 'pygame.Surface') -> None:
        pygame = self.pygame
        ct = self.chart_transform

        disp_ch = self.state.selected_edit_channel
        if 0 <= disp_ch < N_DISPLAY_CHANNELS:
            raw_min, raw_max = DISPLAY_CHANNEL_RAW_RANGES[disp_ch]
            unit = DISPLAY_CHANNEL_UNITS[disp_ch]
            ch_name = self.state.channel_names[disp_ch]
            color = CHANNEL_COLORS[disp_ch]

            n_ticks = 5
            cy_min, cy_max = ct.cy_min, ct.cy_max
            for i in range(n_ticks + 1):
                frac = i / n_ticks
                raw_val = raw_min + frac * (raw_max - raw_min)
                cy = cy_min + frac * (cy_max - cy_min)
                sy = int(ct.y_to_screen(cy))
                sx = int(ct.sx)
                pygame.draw.line(surface, (120, 120, 130),
                                 (sx - ui_px(4), sy), (sx + ui_px(2), sy), 1)
                span = abs(raw_max - raw_min)
                if span < 5:
                    label = f'{raw_val:.2f}'
                elif span < 50:
                    label = f'{raw_val:.1f}'
                else:
                    label = f'{raw_val:.0f}'
                img = self._font_small.render(label, True, color)
                surface.blit(
                    img, (sx - img.get_width() - ui_px(6), sy - img.get_height() // 2)
                )

            title = f'{ch_name} [{unit}]'
            img = self._font_small.render(title, True, color)
            surface.blit(img, (int(ct.sx) - img.get_width() - ui_px(8),
                               int(ct.sy) - img.get_height() - ui_px(2)))

        sy_bottom = int(ct.sy + ct.sh)
        span = float(ct.cx_max - ct.cx_min)
        major, _minor, fmt = _adaptive_time_intervals(span)
        # Snap the first label to a multiple of the major interval that's
        # at or just before cx_min, so labels land on round times.
        start = math.floor(ct.cx_min / major) * major
        end = ct.cx_max + major
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        prev_day_idx = -1
        for patch in np.arange(start, end, major):
            sx = int(ct.x_to_screen(float(patch)))
            if sx < ct.sx or sx > ct.sx + ct.sw:
                continue
            hours_elapsed = float(patch) * 0.5
            total_min_f = self.state.sim_start_hour * 60.0 + hours_elapsed * 60.0
            day_offset = int(total_min_f // (24 * 60))
            mod_min_f = total_min_f - day_offset * 24 * 60
            abs_day = (self.state.sim_start_day + day_offset) % 7
            hh = int(mod_min_f // 60)
            mm = int(mod_min_f - hh * 60)
            ss = int(round((mod_min_f - hh * 60 - mm) * 60))
            if ss == 60:
                ss = 0; mm += 1
                if mm == 60:
                    mm = 0; hh = (hh + 1) % 24
            if fmt == 'hms':
                time_label = f'{hh:02d}:{mm:02d}:{ss:02d}'
            elif fmt == 'h' and mm == 0:
                time_label = f'{hh:02d}:00'
            else:
                time_label = f'{hh:02d}:{mm:02d}'
            img = self._font_small.render(time_label, True, TEXT_DIM_COLOR)
            surface.blit(img, (sx - img.get_width() // 2, sy_bottom + ui_px(4)))
            # Show the day label only when the day changes — otherwise it
            # repeats under every hour tick and adds clutter.
            if abs_day != prev_day_idx:
                dimg = self._font_small.render(
                    day_names[abs_day], True, (100, 130, 170),
                )
                surface.blit(dimg, (sx - dimg.get_width() // 2,
                                    sy_bottom + ui_px(4) + img.get_height() + ui_px(2)))
                prev_day_idx = abs_day

        title = 'Time of day'
        img = self._font_small.render(title, True, TEXT_DIM_COLOR)
        surface.blit(
            img,
            (int(ct.sx + ct.sw // 2 - img.get_width() // 2),
             sy_bottom + ui_px(4) + 2 * (self._font_small.get_height() + ui_px(2))),
        )

    def _draw_status_bar(self, surface: 'pygame.Surface') -> None:
        pygame = self.pygame
        bar_y = self.height - STATUS_BAR_HEIGHT
        pygame.draw.rect(surface, (20, 20, 28), (0, bar_y, self.width, STATUS_BAR_HEIGHT))
        pygame.draw.line(surface, GRID_COLOR, (0, bar_y), (self.width, bar_y))

        n_ctx = self.state.context.shape[0] if self.state.context is not None else 0
        n_pred = self._n_pred()
        status_text = (
            f"Mode: {self.state.mode_label} | "
            f"Context: {n_ctx} patches | "
            f"Masked: {self.state.masked_patch_count(n_pred)} | "
            f"Policy: {self.state.masked_channel_policy} | "
            f"Seed: {self.state.patient_seed} | "
            f"{self.state.status_message}"
        )
        if self.state.is_computing:
            status_text = "⟳ Computing... | " + status_text

        img = self._font_small.render(status_text, True, TEXT_DIM_COLOR)
        surface.blit(img, (10, bar_y + (STATUS_BAR_HEIGHT - img.get_height()) // 2))

    def _draw_cursor(self, surface: 'pygame.Surface', mx: int, my: int) -> None:
        pygame = self.pygame
        ct = self.chart_transform
        from config import PATCH_SIZE

        if not (ct.sx <= mx <= ct.sx + ct.sw):
            return
        pygame.draw.line(surface, CURSOR_COLOR,
                         (mx, int(ct.sy)), (mx, int(ct.sy + ct.sh)), 1)

        cx, _ = ct.screen_to_chart(float(mx), float(my))
        ts_idx = int(round(cx * PATCH_SIZE))
        n_ctx = self.state.context.shape[0] if self.state.context is not None else 0
        ctx_ts = n_ctx * PATCH_SIZE

        tooltip_lines: list[tuple[str, tuple[int, int, int]]] = []
        for disp_ch in range(N_DISPLAY_CHANNELS):
            if not self.state.channel_visible[disp_ch]:
                continue
            color = CHANNEL_COLORS[disp_ch]
            unit = DISPLAY_CHANNEL_UNITS[disp_ch]
            ch_name = self.state.channel_names[disp_ch]
            raw_val: float | None = None

            if ts_idx < ctx_ts:
                if disp_ch == 0 and self.state.bg_raw is not None:
                    if 0 <= ts_idx < len(self.state.bg_raw):
                        raw_val = float(self.state.bg_raw[ts_idx])
                elif (self.state.context_raw is not None
                      and disp_ch < self.state.context_raw.shape[1]):
                    if 0 <= ts_idx < self.state.context_raw.shape[0]:
                        raw_val = float(self.state.context_raw[ts_idx, disp_ch])
            else:
                pred_ts = ts_idx - ctx_ts
                if disp_ch == 0 and self.state.prediction.median_bg is not None:
                    pred_bg = self.state.prediction.median_bg
                    if 0 <= pred_ts < len(pred_bg):
                        raw_val = float(pred_bg[pred_ts])
                else:
                    out_ch = DISPLAY_TO_OUTPUT_CH.get(disp_ch)
                    ovr = self.state.prediction.overrides_raw
                    if out_ch is not None and ovr is not None and out_ch in ovr:
                        arr = ovr[out_ch]
                        patch_idx = pred_ts // PATCH_SIZE
                        step_idx = pred_ts % PATCH_SIZE
                        if 0 <= patch_idx < arr.shape[0]:
                            raw_val = float(arr[patch_idx, step_idx])

            if raw_val is not None:
                fmt = f'{raw_val:.2f}' if abs(raw_val) < 5 else (f'{raw_val:.1f}' if abs(raw_val) < 50 else f'{raw_val:.0f}')
                tooltip_lines.append((f'{ch_name}: {fmt} {unit}', color))

        if not tooltip_lines:
            return

        line_h = self._font_small.get_height() + ui_px(2)
        pad = ui_px(4)
        max_w = max(self._font_small.size(text)[0] for text, _ in tooltip_lines)
        box_w = max_w + 2 * pad
        box_h = len(tooltip_lines) * line_h + 2 * pad

        tip_x = mx + 10
        tip_y = my - box_h // 2
        if tip_x + box_w > self.width:
            tip_x = mx - box_w - 10
        tip_y = max(int(ct.sy), min(tip_y, int(ct.sy + ct.sh - box_h)))

        tip_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        tip_surf.fill((20, 20, 30, 220))
        pygame.draw.rect(tip_surf, (80, 80, 100), (0, 0, box_w, box_h), 1)

        for i, (text, color) in enumerate(tooltip_lines):
            img = self._font_small.render(text, True, color)
            tip_surf.blit(img, (pad, pad + i * line_h))

        surface.blit(tip_surf, (tip_x, tip_y))

    def _draw_clock_face_overlay(self, surface: 'pygame.Surface', mx: int, my: int) -> None:
        """Draw the time-of-day probe's clock-face histogram, rotated to the cursor.

        The probe emits one row per MASKED patch, and the rows are ticked at the
        patches they were read off rather than at a fixed trailing zone: the row
        count is ``probs.shape[0]``, not ``PREDICTION_PATCHES``. For the GUI's
        forecast the masked set is the contiguous span that starts at ``n_ctx``, so
        the ticks land there — but the count comes from the probe output, so a
        shorter or longer masked set ticks correctly instead of silently drawing a
        2 h zone.

        Those per-patch beliefs are ONE origin-phase belief sampled at deterministic
        elapsed offsets (row p == origin advanced by ``p * ADV_HOURS``); they are
        fused once (``utils.aggregate_origin_belief``) into a single origin belief,
        then rendered at the cursor's continuous elapsed offset ``t`` by rigidly
        rotating that belief on the dial by ``2*pi*t/24`` (angles only, no re-binning,
        so cursor motion is smooth and exact). Diagnostic only.
        """
        import utils
        import gui_renderer
        from config import PREDICTION_HORIZON_HOURS, PREDICTION_PATCHES, TIME_PROBE_BIN_HOURS

        if self.state.is_computing or self.state.prediction is None:
            return
        probs = self.state.prediction.tod_bin_probs
        if probs is None or self.state.context is None:
            return

        pygame = self.pygame
        ct = self.chart_transform
        n_ctx = self.state.context.shape[0]

        n_masked = int(probs.shape[0])
        adv_hours = PREDICTION_HORIZON_HOURS / PREDICTION_PATCHES

        tick_top = int(ct.sy + ct.sh - ui_px(8))
        tick_bot = int(ct.sy + ct.sh)
        for p in range(n_masked + 1):
            tx = int(round(ct.x_to_screen(n_ctx + p)))
            if ct.sx <= tx <= ct.sx + ct.sw:
                pygame.draw.line(surface, CLOCK_TICK_COLOR, (tx, tick_top), (tx, tick_bot), 1)

        if not (ct.sx <= mx <= ct.sx + ct.sw):
            return
        cx_chart, _ = ct.screen_to_chart(float(mx), float(my))
        if cx_chart < n_ctx:
            return
        t_hours = (cx_chart - n_ctx) * adv_hours
        t_hours = max(0.0, min(t_hours, self.state.prediction.n_rolls * PREDICTION_HORIZON_HOURS))

        origin = utils.aggregate_origin_belief(probs, adv_hours, TIME_PROBE_BIN_HOURS)
        geom = utils.clock_wedge_geometry(origin, rotation_hours=t_hours)

        face_cx = int(ct.sx + ct.sw - CLOCK_FACE_RADIUS_PX - CLOCK_FACE_MARGIN_PX)
        face_cy = int(ct.sy + CLOCK_FACE_RADIUS_PX + CLOCK_FACE_MARGIN_PX)
        gui_renderer.draw_clock_face(
            surface, cx=face_cx, cy=face_cy, radius=CLOCK_FACE_RADIUS_PX, geom=geom,
            face_color=CLOCK_FACE_BG_COLOR, wedge_color=CLOCK_WEDGE_COLOR,
            hand_color=CLOCK_HAND_COLOR, tick_color=CLOCK_TICK_COLOR, R=geom.R,
        )

        hx, hy = float(geom.hand[0]), float(geom.hand[1])  # u(h) = (sin a, cos a)
        disp_hour = (math.atan2(hx, hy) % (2.0 * math.pi)) * (24.0 / (2.0 * math.pi))
        cap = f"{disp_hour:04.1f}h  R{geom.R:.2f}"
        img = self._font_small.render(cap, True, TEXT_DIM_COLOR)
        surface.blit(
            img,
            (face_cx - img.get_width() // 2, face_cy + CLOCK_FACE_RADIUS_PX + ui_px(2)),
        )

    def _handle_keyboard(self, event) -> bool:
        pygame = self.pygame
        key = event.key

        if key in (pygame.K_q, pygame.K_ESCAPE):
            return True

        elif key == pygame.K_SPACE:
            # Shift+Space runs the fit-to-drawing long prediction; plain Space
            # stays the single-pass 2 h forecast.
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self._do_long_predict()
            else:
                self._do_predict()

        elif key == pygame.K_l:
            self._do_long_predict()

        elif key == pygame.K_w:
            self._do_what_if()

        elif key == pygame.K_p:
            self._do_pencil()

        elif key == pygame.K_m:
            self._do_mask_tool()

        elif key == pygame.K_f:
            self._do_roll_forward()

        elif key == pygame.K_g:
            self._do_sim_forward()

        elif key == pygame.K_v:
            self._do_eval_against_sim()

        elif key == pygame.K_r:
            self._do_reset()

        elif key == pygame.K_n:
            self._do_new_patient()

        elif key == pygame.K_s:
            self._do_screenshot()

        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            from gui_state import (
                MASK_PRESET_BEGIN_FILL, MASK_PRESET_FORECAST, MASK_PRESET_INFILL,
                TOOL_MASK,
            )
            idx = key - pygame.K_1
            # While the mask tool is up, 1/2/3 place the presets; the channel
            # toggles keep the keys otherwise.
            presets = (MASK_PRESET_FORECAST, MASK_PRESET_BEGIN_FILL,
                       MASK_PRESET_INFILL)
            if self.state.active_tool == TOOL_MASK and idx < len(presets):
                self._do_mask_preset(presets[idx])
            elif idx < len(self.state.channel_visible):
                self.state.toggle_channel(idx)
                self._toggles[idx].state = self.state.channel_visible[idx]
                self._needs_redraw = True

        elif key == pygame.K_a:
            self.state.toggle_all_channels()
            for i, t in enumerate(self._toggles):
                t.state = self.state.channel_visible[i]
            self._needs_redraw = True

        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            cx = (self.chart_transform.cx_min + self.chart_transform.cx_max) / 2
            span = (self.chart_transform.cx_max - self.chart_transform.cx_min) / ZOOM_FACTOR
            self.chart_transform.update(
                chart_x_min=cx - span / 2,
                chart_x_max=cx + span / 2,
            )
            self._needs_redraw = True

        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            cx = (self.chart_transform.cx_min + self.chart_transform.cx_max) / 2
            span = (self.chart_transform.cx_max - self.chart_transform.cx_min) * ZOOM_FACTOR
            self.chart_transform.update(
                chart_x_min=cx - span / 2,
                chart_x_max=cx + span / 2,
            )
            self._needs_redraw = True

        elif key == pygame.K_c:
            self._do_clear_all()

        elif key == pygame.K_h:
            self._help_window.toggle()
            if self._help_window.visible:
                self._event_editor.visible = False
            self._needs_redraw = True

        elif key == pygame.K_e:
            from gui_state import TOOL_CURVE_EDITOR, TOOL_NONE
            if self.state.active_tool == TOOL_CURVE_EDITOR:
                self.state.set_tool(TOOL_NONE)
            else:
                self.state.set_tool(TOOL_CURVE_EDITOR)
            self._needs_redraw = True

        elif key == pygame.K_TAB:
            mods = pygame.key.get_mods()
            delta = -1 if (mods & pygame.KMOD_SHIFT) else 1
            self.state.cycle_edit_channel(delta)
            ch = self.state.channel_names[self.state.selected_edit_channel]
            self.state.status_message = f"Editing: {ch}"
            self._needs_redraw = True

        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._do_predict()

        elif key == pygame.K_z:
            mods = pygame.key.get_mods()
            if mods & pygame.KMOD_CTRL:
                self._do_undo()

        elif key == pygame.K_DELETE:
            from gui_state import TOOL_MASK
            if (self.state.active_tool == TOOL_MASK
                    and 0 <= self.state.selected_mask_idx < len(self.state.mask_spans)):
                self.state.remove_mask_span(self.state.selected_mask_idx)
                self._clear_prediction()
                self.state.status_message = "Span removed"
                self._needs_redraw = True
            elif (0 <= self.state.selected_event_idx < len(self.state.curve_events)):
                self._snapshot_for_undo()
                del self.state.curve_events[self.state.selected_event_idx]
                self.state.selected_event_idx = -1
                self.state.dragging_point = None
                self._compile_curve_overrides()

        return False

    def run(self) -> None:
        pygame = self.pygame
        pygame.init()
        pygame.display.set_caption(
            f"T1DMAI — Behavior Prediction  ·  masked doses: "
            f"{self.state.masked_channel_policy}"
        )

        self._screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF | pygame.RESIZABLE,
        )
        try:
            from pygame._sdl2 import video as _sdl2_video
            win = _sdl2_video.Window.from_display_module()
            win.maximize()
        except Exception:
            pass
        clock = pygame.time.Clock()

        pygame.font.init()
        self._font = pygame.font.SysFont('DejaVuSans', FONT_SIZE)
        self._font_small = pygame.font.SysFont('DejaVuSans', FONT_SIZE_SMALL)
        self._font_large = pygame.font.SysFont('DejaVuSans', FONT_SIZE_LARGE, bold=True)

        running = True
        last_mx = 0
        last_my = 0

        while running:
            for event in pygame.event.get():
                # Modals get first crack so chart zoom / pan and sidebar
                # scroll don't fire on events landing inside an open
                # sub-window. The modal swallows mouse buttons + wheel
                # and Esc; mouse motion and other keys pass through.
                modal_consumed = False
                for modal in (self._help_window, self._event_editor):
                    if modal.visible and modal.handle_event(event):
                        modal_consumed = True
                        self._needs_redraw = True
                        break
                if modal_consumed:
                    continue

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(
                        event.size, pygame.DOUBLEBUF | pygame.RESIZABLE,
                    )
                    self._update_layout(event.w, event.h)
                    self._needs_redraw = True

                elif event.type == pygame.KEYDOWN:
                    if self._handle_keyboard(event):
                        running = False

                elif event.type == pygame.MOUSEMOTION:
                    last_mx = event.pos[0]
                    last_my = event.pos[1]
                    self._needs_redraw = True
                    if self._panning:
                        self._handle_pan_motion(event.pos[0])
                    else:
                        from gui_state import TOOL_CURVE_EDITOR, TOOL_MASK, TOOL_PENCIL
                        if (self.state.active_tool == TOOL_CURVE_EDITOR
                                and self.state.dragging_point is not None):
                            self._handle_curve_drag(event.pos[0], event.pos[1])
                        elif (self.state.active_tool == TOOL_PENCIL
                                and self._pencil_drawing):
                            self._handle_pencil_motion(event.pos[0], event.pos[1])
                        elif self.state.active_tool == TOOL_MASK:
                            self._handle_mask_motion(event.pos[0], event.pos[1])

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    from gui_state import TOOL_CURVE_EDITOR, TOOL_MASK, TOOL_PENCIL
                    if (event.button == 1
                            and self._is_in_events_panel(event.pos[0], event.pos[1])):
                        if self._handle_events_panel_click(event.pos[0], event.pos[1]):
                            self._needs_redraw = True
                            continue
                    if (event.button in (2, 3)
                            and self._is_in_chart(event.pos[0], event.pos[1])):
                        self._begin_pan(event.pos[0])
                    elif (event.button == 1
                            and self.state.active_tool == TOOL_CURVE_EDITOR
                            and self._is_in_chart(event.pos[0], event.pos[1])):
                        self._handle_curve_click(event.pos[0], event.pos[1])
                    elif (event.button == 1
                            and self.state.active_tool == TOOL_PENCIL
                            and self._is_in_chart(event.pos[0], event.pos[1])):
                        self._handle_pencil_down(event.pos[0], event.pos[1])
                    elif (event.button == 1
                            and self.state.active_tool == TOOL_MASK
                            and self._is_in_chart(event.pos[0], event.pos[1])):
                        self._handle_mask_down(event.pos[0], event.pos[1])

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in (2, 3):
                        self._panning = False
                    if event.button == 1:
                        self._handle_curve_release(event.pos[0], event.pos[1])
                        self._handle_pencil_release(event.pos[0], event.pos[1])
                        self._handle_mask_release(event.pos[0], event.pos[1])

                elif event.type == pygame.MOUSEWHEEL:
                    if self._is_in_sidebar(last_mx, last_my):
                        step = SCROLL_SPEED * (-event.y)
                        visible_h = max(self.height - STATUS_BAR_HEIGHT, 1)
                        max_scroll = max(0, self._sidebar_content_h - visible_h)
                        self._sidebar_scroll = max(
                            0, min(self._sidebar_scroll + step, max_scroll)
                        )
                        self._needs_redraw = True
                    elif self._is_in_events_panel(last_mx, last_my):
                        step = SCROLL_SPEED * (-event.y)
                        visible_h = max(self.height - STATUS_BAR_HEIGHT, 1)
                        max_scroll = max(0, self._events_panel_content_h - visible_h)
                        self._events_panel_scroll = max(
                            0, min(self._events_panel_scroll + step, max_scroll)
                        )
                        self._needs_redraw = True
                    elif self._is_in_chart(last_mx, last_my):
                        self._zoom_at_cursor(last_mx, event.y)

                for toggle in self._toggles:
                    toggle.handle_event(event)
                for toggle in self._display_toggles:
                    toggle.handle_event(event)
                for btn in self._buttons:
                    btn.handle_event(event)

            if self._prev_is_computing and not self.state.is_computing:
                self._needs_redraw = True
            self._prev_is_computing = self.state.is_computing

            if self._needs_redraw or self.state.is_computing:
                self._screen.fill(BG_COLOR)

                try:
                    self._chart_cache = self._draw_chart()
                except Exception:
                    traceback.print_exc()
                    self._chart_cache = None

                if self._chart_cache is not None:
                    self._screen.blit(
                        self._chart_cache,
                        (int(self.chart_transform.sx), int(self.chart_transform.sy))
                    )

                if self.state.is_computing:
                    msg = self._font_large.render("Computing...", True, (255, 200, 80))
                    cx = int(self.chart_transform.sx + self.chart_transform.sw // 2)
                    self._screen.blit(msg, (cx - msg.get_width() // 2, int(self.chart_transform.sy + 10)))

                self._draw_axes(self._screen)
                self._draw_sidebar(self._screen)
                self._draw_events_panel(self._screen)
                self._draw_control_panel(self._screen)
                self._draw_status_bar(self._screen)
                self._draw_cursor(self._screen, last_mx, last_my)
                self._draw_clock_face_overlay(self._screen, last_mx, last_my)

                # Modals last so they paint on top of everything else,
                # including the cursor overlay.
                for modal in (self._help_window, self._event_editor):
                    modal.draw(self._screen, self._font, self._font_large)

                pygame.display.flip()
                self._needs_redraw = False

            clock.tick(FPS)

        pygame.quit()


# The trained-checkpoint tree, one capacity per subdirectory. ``compare.py``
# reads the same root; a second one is how the two start disagreeing about which
# checkpoint 'medium' names.
MODEL_DIR = 'new_models'


def _discover_checkpoint() -> str | None:
    """The best checkpoint of the capacity the live ``config.py`` describes.

    Every builder refuses a checkpoint whose ``training_config`` disagrees with
    the live config, so there is exactly one capacity under ``MODEL_DIR`` that
    can be loaded at all — matching on ``D_MODEL`` / ``N_LAYERS`` / ``N_HEADS``
    picks it without reading a state dict.
    """
    from config import D_MODEL, N_LAYERS, N_HEADS
    for capacity in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []:
        path = os.path.join(MODEL_DIR, capacity, 'checkpoints', 't1dmai_best.pt')
        if not os.path.isfile(path):
            continue
        try:
            tc = torch.load(path, map_location='cpu',
                            weights_only=True).get('training_config') or {}
        except Exception:
            continue
        if (tc.get('d_model'), tc.get('n_layers'), tc.get('n_heads')) == \
                (D_MODEL, N_LAYERS, N_HEADS):
            print(f"Using {path} (capacity '{capacity}' matches config.py)")
            return path
    return None


def _load_checkpoint_into_model(
    path: str, device: torch.device, use_ema: bool
) -> tuple[object, int, dict, str]:
    """Load a checkpoint's weights, horizon, stats and masked-channel policy.

    The policy is part of the return because no parameter shape records it: the
    strict state-dict load accepts weights trained under either convention, so a
    blind checkpoint would otherwise be driven as a conditioned one — painted
    doses landing on channels its weights were trained to read as constant, and
    a forecast that plausibly does not move.

    Returns:
        ``(model, prediction_patches, normalization_stats, masked_channel_policy)``.
    """
    from model import T1DMAI
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = T1DMAI().to(device)
    live_sd = ckpt['model_state_dict']
    ema_sd = ckpt.get('model_ema_state_dict')
    if use_ema and ema_sd:
        merged = {k: ema_sd.get(k, v) for k, v in live_sd.items()}
        model.load_state_dict(merged, strict=True)
        print(f"Loaded EMA shadow from checkpoint: {path}")
    else:
        if use_ema:
            print(f"Checkpoint {path} has no EMA state; falling back to live weights.")
        model.load_state_dict(live_sd)
        print(f"Loaded live weights from checkpoint: {path}")
    model.eval()
    tc = ckpt.get('training_config') or {}
    from config import PREDICTION_PATCHES as _PP
    from data import stored_masked_channel_policy
    pp = int(tc.get('prediction_patches', _PP))
    norm_stats = ckpt.get('normalization_stats', {}) or {}
    policy = stored_masked_channel_policy(tc)
    print(f"masked_channel_policy: {policy}"
          + ('' if 'masked_channel_policy' in tc else ' (absent stamp — announced)'))
    return model, pp, norm_stats, policy


def main() -> None:
    parser = argparse.ArgumentParser(description='T1DMAI GUI')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the trained model checkpoint. With no path '
                             'and no --no-model, the capacity matching the live '
                             'config.py is looked up under new_models/.')
    parser.add_argument('--no-model', action='store_true',
                        help='Use random weights for UI testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Patient seed for simulator context')
    parser.add_argument('--width', type=int, default=WINDOW_WIDTH)
    parser.add_argument('--height', type=int, default=WINDOW_HEIGHT)
    parser.add_argument('--live-weights', action='store_true',
                        help='Load the live (training) weights instead of the '
                             'EMA shadow. Validation runs under EMA, so the '
                             'GUI mirrors that by default; use this flag to '
                             'inspect the raw training-step weights.')
    parser.add_argument('--basal-ramp-up', type=float, default=0.5,
                        help='Basal ramp-up duration in hours (default: 0.5)')
    parser.add_argument('--basal-ramp-down', type=float, default=0.5,
                        help='Basal ramp-down duration in hours (default: 0.5)')
    parser.add_argument('--basal-duration', type=float, default=4.0,
                        help='Basal adjustment total duration in hours (default: 4.0)')
    parser.add_argument('--context-hours', type=float, default=None,
                        help='Post-warmup simulator hours behind the forecast '
                             f'origin (default: {default_context_hours():g}, which '
                             'fills MAX_CONTEXT_PATCHES). Anything under '
                             'MIN_CONTEXT_PATCHES is refused.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    norm_stats: dict = {}
    model = None
    from data import masked_channel_policy as _policy_name
    policy = _policy_name(blind=False)

    if args.no_model:
        from model import T1DMAI
        model = T1DMAI().to(device)
        model.eval()
    else:
        checkpoint = args.checkpoint or _discover_checkpoint()
        if not checkpoint:
            print("No checkpoint provided and none found under "
                  f"{MODEL_DIR}/<capacity>/checkpoints/. Pass --checkpoint, or "
                  "--no-model for a UI-only run.")
            return
        use_ema = not args.live_weights
        model, ckpt_pred_patches, ns, policy = _load_checkpoint_into_model(
            checkpoint, device, use_ema=use_ema,
        )
        norm_stats = norm_stats or ns
        import model as _model_mod
        import inference as _infer_mod
        _model_mod.PREDICTION_PATCHES = ckpt_pred_patches
        _infer_mod.PREDICTION_PATCHES = ckpt_pred_patches

    if not norm_stats:
        from normalization import load_normalization_stats, NORM_STATS_FILE
        if os.path.exists(NORM_STATS_FILE):
            norm_stats = load_normalization_stats()
        else:
            from normalization import compute_normalization_stats, save_normalization_stats
            print("Computing normalization stats...")
            norm_stats = compute_normalization_stats()
            save_normalization_stats(norm_stats)

    gui = T1DMAIGui(
        model=model,
        norm_stats=norm_stats,
        device=device,
        patient_seed=args.seed,
        width=args.width,
        height=args.height,
        basal_ramp_up_h=args.basal_ramp_up,
        basal_ramp_down_h=args.basal_ramp_down,
        basal_duration_h=args.basal_duration,
        masked_channel_policy=policy,
        context_hours=args.context_hours,
    )
    gui.run()


if __name__ == '__main__':
    main()
