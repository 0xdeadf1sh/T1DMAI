"""Pygame front end: context window, checkpoint forward, per-tau BG band, doses/masked spans.

    python gui.py --checkpoint checkpoints/t1dmai_best.pt --seed 42
    python gui.py --no-model    # UI testing with random weights
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
from normalization import CHANNEL_NAMES as _CHANNEL_NAMES  # the strips' row order

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

# design-time values scaled at import, so the whole interface moves with ``UI_SCALE``
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
# Presentation only, so they live here rather than in config.py.
CLOCK_FACE_RADIUS_PX = ui_px(52)
CLOCK_FACE_MARGIN_PX = ui_px(12)
CLOCK_FACE_BG_COLOR = (32, 34, 48)
CLOCK_WEDGE_COLOR = (90, 150, 230)
CLOCK_HAND_COLOR = (245, 245, 250)
CLOCK_TICK_COLOR = (120, 125, 145)
CONFIDENCE_ALPHA = 40
# tint over the model's input patches, against the darker one past NOW
CONTEXT_SHADE_COLOR = (80, 130, 200)
CONTEXT_SHADE_ALPHA = 20

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

# glucose-range shading; the cutoffs come from config, the single source of truth
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

# Masked span hides true BG; overlay marks it, _draw_chart clips the context curve there.
MASK_SPAN_COLOR = (150, 120, 210)
MASK_SPAN_ALPHA = 46
MASK_SPAN_SELECTED_ALPHA = 78
MASK_DRAG_ALPHA = 30
MASK_EDGE_COLOR = (185, 160, 240)
MASK_OOD_COLOR = (240, 175, 70)
# Attention is positive-only (one hue ramp); saliency is signed (diverging, up raises forecast).
ATTN_ROW_H = ui_px(11)
ATTN_ROW_GAP = ui_px(2)
ATTN_BLOCK_PAD = ui_px(4)
ATTN_STRIP_BG = (24, 24, 32)
ATTN_ROW_BG = (34, 34, 44)
ATTN_MASS_COLOR = (120, 190, 250)
SALIENCY_UP_COLOR = (240, 110, 110)
SALIENCY_DOWN_COLOR = (90, 150, 240)
ATTN_SPAN_EDGE_COLOR = MASK_EDGE_COLOR
# Row order follows CHANNEL_NAMES (Attribution.channels column order); only abbreviations local.
ATTN_CHANNEL_ABBREV = {
    'bg_absolute': 'BG',
    'carb_intake': 'carb',
    'insulin_combined': 'ins',
    'exercise_equiv': 'exer',
}
ATTN_ROW_LABELS = ['attn'] + [
    ATTN_CHANNEL_ABBREV[name] for name in _CHANNEL_NAMES
]
ATTN_ROW_MAX_ALPHA = 235
# A masked patch's true attribution is 0.0; mark withheld, not zero, or it reads as "ignored".
ATTN_WITHHELD_COLOR = (96, 100, 118)
ATTN_WITHHELD_ALPHA = 120
# Floor on the visible-patch ink scale (fraction of window peak) so an empty stretch stays dark.
ATTN_VIEW_SCALE_FLOOR = 0.15
ATTN_BLOCK_HEIGHT = (2 * ATTN_BLOCK_PAD
                     + len(ATTN_ROW_LABELS) * (ATTN_ROW_H + ATTN_ROW_GAP))

# Buttons that write an announced dose; disabled under a blind checkpoint.
_DOSE_PAINTING_BUTTONS = frozenset({
    "What-If", "Pencil", "Basal +1 U/h", "Basal -1 U/h",
})

# Timesteps, cosmetic only. Band gets a wider smoothing window than median (median stays reactive).
CONFIDENCE_BAND_SMOOTH_STEPS = 25
MU_SMOOTH_STEPS = 13

# the three announceable channels, in output-channel order
OUTPUT_CHANNEL_ORDER = ['carb_intake', 'insulin_combined', 'exercise_equiv']

N_DISPLAY_CHANNELS = 4

# y-axis span per channel, own raw unit; exercise shares carb's range (carb-equivalent, g/step).
DISPLAY_CHANNEL_RAW_RANGES: list[tuple[float, float]] = [
    (0.0,     400.0),
    (0.0,     10.0),
    (0.0,     1.0),
    (0.0,     10.0),
]

DISPLAY_TO_STATS_NAME: list[str] = [
    'bg_absolute', 'carb_intake', 'insulin_combined', 'exercise_equiv',
]

# display channel → input feat; BG/carb/insulin/exercise at 0/1/2/3, all but BG announceable
DISPLAY_TO_FEATURE_IDX: list[int] = [0, 1, 2, 3]

# display channel → announced output channel; BG (display 0) is a forecast, never an input
DISPLAY_TO_OUTPUT_CH: dict[int, int] = {1: 0, 2: 1, 3: 2}

# inverse of DISPLAY_TO_OUTPUT_CH, routing a pencil stroke back through the override compiler
OUTPUT_TO_DISPLAY_CH: dict[int, int] = {v: k for k, v in DISPLAY_TO_OUTPUT_CH.items()}

DISPLAY_CHANNEL_CLEAR_DEFAULTS: list[float] = [100.0, 0.0, 0.0, 0.0]

OUTPUT_CHANNEL_SHORT_NAMES: list[str] = ['carbs', 'insulin', 'exercise']

INSULIN_OUTPUT_CH: int = OUTPUT_CHANNEL_ORDER.index('insulin_combined')

# Exercise stays at the trained g/step carb-equivalent scale, never a 0-1 intensity.
DISPLAY_CHANNEL_UNITS: list[str] = [
    'mg/dL',
    'g/5min',
    'U/5min',
    'g/step',
]

SCROLL_SPEED = 48
ZOOM_FACTOR = 1.3
# Fraction of the visible span, not a patch count, so one press covers the same screen distance.
PAN_STEP_FRACTION = 0.25
PAN_STEP_FAST_FRACTION = 1.0
# Hours the chart opens on; drawn full-width the window is illegible. Zoom/pan reach the rest.
CHART_VIEW_HOURS = 24.0
HOVER_TOOLTIP_DELAY_MS = 200

# indexed by ``disp_ch`` on every draw path, so it stays as long as the tables above
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
    """A simulator raw-dict chunk as the normalized feature stack, plus the GUI's raw arrays.

    Returns ``(features_norm[N, N_INPUT_FEATURES], bg_raw[N], context_raw[N, 4])``, N trimmed to
    a multiple of PATCH_SIZE; ``context_raw`` is the chart's data, not the model's input.
    """
    from config import PATCH_SIZE, N_INPUT_FEATURES
    from data import BG_MASKED_FEAT
    from normalization import CHANNEL_NAMES, normalize
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

    # bg_observed post-CGM-noise, clamped; sparse three floored at 0 (mirrors _build_sample).
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

    # the plotted channels stay RAW; only ``features`` carries the clamp and normalization
    bg_raw = bg_obs_raw.copy()
    context_raw = np.stack(
        [bg_obs_raw, carb_raw, insulin_raw, exercise_raw], axis=-1,
    ).astype(np.float32)

    # bg_masked bit carries no statistics, so the stack is one column wider than CHANNEL_NAMES.
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
    """Post-warmup simulator hours that fill the context window to ``MAX_CONTEXT_PATCHES``.

    Bootstrapping below ``MIN_CONTEXT_PATCHES`` raises nothing — the fan is simply drawn from a
    window the weights never saw, which is the one failure the GUI cannot show.
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
    """Step ``state.sim`` by ``hours`` and append the new truth to the state buffers.

    Returns the NET patch shift, appended minus dropped off the left; 0 once the window is
    sliding at ``MAX_CONTEXT_PATCHES`` rather than growing.
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

    # Drop from the left: no raise on overrun, RoPE just extrapolates past the trained window.
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
    """Cosmetic hour-of-day [0, 24) at the prediction-zone start: run start plus 0.5 h a patch."""
    start_hour = float(getattr(state, 'sim_start_hour', 0.0))
    ctx = getattr(state, 'context', None)
    n_ctx = 0 if ctx is None else int(ctx.shape[0])
    return (start_hour + n_ctx * 0.5) % 24.0


# Raised-cosine bells, amplitude back-solved so AUC matches magnitude; carbs g/5min, insulin U/5min.
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
    """A raised-cosine CurveEvent whose AUC equals ``magnitude``; ``placement_min`` is minutes
    from the start of the prediction zone (n_ctx)."""
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
    """Expand juice / insulin / meal Events into CurveEvents for the override compiler."""
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
    """``disp_channel``'s pencil strokes onto the ``(n_pred_patches, patch_size)`` grid, smoothed.

    Overlapping strokes on one channel are max-combined, so a redraw replaces rather than adds.
    Returns raw units, g/5min or U/5min.
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
        # np.interp needs strictly-increasing x; collapse a back-and-forth stroke's dupes to max.
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
    """The painted curve / pencil / basal edits as announced dose overrides for the pred zone.

    Built from a ZERO baseline plus what the user painted; bells and pencil strokes SUM.
    Returns ``(overrides_norm, overrides_raw)`` over {0: carb, 1: insulin, 2: exercise}.
    """
    from config import PATCH_SIZE, CHANNEL_TO_FEAT

    pencil_strokes = pencil_strokes or []

    # Announceable set is the display table's; a channel missing here silently drops its stroke.
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
        if out_ch not in CHANNEL_TO_FEAT:
            continue
        if not has_edit[out_ch]:
            continue
        ch_name = OUTPUT_CHANNEL_ORDER[out_ch]

        # zero baseline: nothing announced on this channel until the user paints it
        modified_raw = np.zeros((n_pred, PATCH_SIZE), dtype=np.float32)

        for event in curve_events:
            evt_out_ch = DISPLAY_TO_OUTPUT_CH.get(event.channel)
            if evt_out_ch != out_ch:
                continue
            event_vals = _curve_event_to_raw_values(event, n_ctx, n_pred, PATCH_SIZE)
            modified_raw = modified_raw + event_vals

        # max-combined internally, summed on top of the bells above
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

        # overrides_raw keeps the RAW announced value at its trained scale: exercise g/step.
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
    """Decode the diagnostic time-of-day probe for ``context``; all None when the probe is off.

    ``overrides`` should mirror the displayed forecast's. Returns ``(pred_hour [0, 24), confidence
    R [0, 1] of the FIRST masked patch, bin_probs (masked patches, TIME_PROBE_N_BINS))``.
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
    """Rolls needed to cover the furthest painted dose, clamped to ``[min_rolls, max_rolls]``.

    Positions are absolute patches; one roll covers ``PREDICTION_PATCHES``.
    """
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
    """Patches past ``n_ctx`` to draw the announced-dose preview over.

    A DRAWING extent, not a masked set — the model's is built per forward, one roll at a time.
    """
    from config import PREDICTION_PATCHES, PREDICTION_HORIZON_HOURS, GUI_MAX_PREDICTION_HOURS
    max_rolls = max(1, round(GUI_MAX_PREDICTION_HOURS / PREDICTION_HORIZON_HOURS))
    n = _painted_rolls_for_state(state, 1, max_rolls) * PREDICTION_PATCHES
    if state.prediction.bands is not None:
        n = max(n, int(state.prediction.bands.shape[0]))
    return int(n)


def _masked_context(state, spans: list[tuple[int, int]]):
    """``state.context`` with the masked CONTEXT spans' dose channels withheld.

    Blind policy only, context spans only; mirrors ``calibrate_conformal._blind_context``.
    Returns a new tensor under ``blind``, ``state.context`` itself otherwise.
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
        # inference.PREDICTION_PATCHES, not config's: main rewrites it to the checkpoint's horizon.
        n_pred = int(inference.PREDICTION_PATCHES)
        mask_spans = state.emitted_mask_spans(n_pred)
        # Snapshotted HERE so a mid-compute mask edit can't hand explain() a span not forwarded.
        target_span = state.selected_span(n_pred)
        ctx = _masked_context(state, mask_spans)
        # Anchor rule is one-sided, LEFT-PREFERRING; see gui_state.span_anchor_cell.
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
            # RAW overlay recompiled over the full painted horizon; model override stays fixed.
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
            # predict_what_if is right-edge only; a masked-set call goes to predict directly.
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

        state.prediction.median_bg = result['median_bg'].cpu().numpy()
        state.prediction.bands = result['bands'].cpu().numpy()
        # Slot j is patch mask_idx[j]; a fixed offset from context end is right only for forecast.
        state.prediction.span_patches = result['mask_idx'].cpu().numpy()
        state.prediction.n_rolls = 1

        # Decoded off the SAME context as the shown bands; None when the probe is off.
        (state.prediction.tod_pred_hour,
         state.prediction.tod_confidence,
         state.prediction.tod_bin_probs) = _decode_tod(
            model, state.context, state.norm_stats, device,
            overrides=probe_overrides,
        )

        # What this forward consumed, so strips can be filled later without reproducing it.
        state.last_forward = {
            'rolls': [{
                'context': ctx,
                'mask_spans': mask_spans,
                'overrides': probe_overrides,
                'span': target_span,
                'offset': 0,
                'label': '',
            }],
            'index': 0,
        }

        # Off by default: one extra grad-enabled forward per prediction.
        state.prediction.attribution = None
        attribution_note = ""
        if state.attn_overlay_visible:
            attribution_note = _fill_attribution(state, model, device)

        state.status_message = (
            f"Prediction complete "
            f"({datetime.datetime.now().strftime('%H:%M:%S')}){attribution_note}"
        )

    except Exception as e:
        traceback.print_exc()
        state.status_message = f"Error: {e}"
    finally:
        state.is_computing = False


def _selected_roll(state) -> dict | None:
    """The recorded forward the strips explain, None when there is none.

    A single pass records one roll, a rolling forecast one per roll, and ``index`` picks which.
    """
    forward = state.last_forward
    if not forward or not forward.get('rolls'):
        return None
    rolls = forward['rolls']
    return rolls[max(0, min(int(forward.get('index', 0)), len(rolls) - 1))]


def _fill_attribution(state, model, device) -> str:
    """Maps for the recorded forward the roll index selects; returns '' or the failure text."""
    from attribution import explain

    roll = _selected_roll(state)
    state.prediction.attribution = None
    if roll is None or state.norm_stats is None:
        return ""
    try:
        state.prediction.attribution = explain(
            model, roll['context'], state.norm_stats,
            overrides=roll['overrides'],
            mask_spans=roll['mask_spans'],
            span=roll['span'],
            device=device,
            window_offset=roll['offset'],
        )
    except Exception as e:
        traceback.print_exc()
        return f" | attribution failed: {e}"
    return ""


def _run_attribution(state, model, device) -> None:
    """Fill the strips off the recorded forward, on a worker thread."""
    state.is_computing = True
    state.status_message = "Reading attention..."
    try:
        note = _fill_attribution(state, model, device)
        attrib = state.prediction.attribution
        if note or attrib is None:
            state.status_message = note.lstrip(" |") or "Attention unavailable"
        else:
            roll = _selected_roll(state) or {}
            n_rolls = len(state.last_forward['rolls']) if state.last_forward else 1
            start = attrib.span[0] + attrib.window_offset
            where = f"patches {start}-{start + attrib.span[1] - 1}"
            state.status_message = (
                f"Attention: {roll['label']} of {n_rolls}, {where}"
                if roll.get('label') else f"Attention: {where}"
            )
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
    """(major, minor, label_format) for the visible span; one patch = 30 min.

    ``label_format``: 'h' whole hours, 'hm' HH:MM, 'hms' HH:MM:SS. Targets 6–10 major labels.
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
    """Gaussian smoothing, edge-replicated so the length is unchanged; ``window=1`` is a no-op.

    Even windows bump to the next odd so the kernel stays centred; σ = window/3 puts ±3σ at the
    kernel edges, softer than a boxcar of the same width.
    """
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
    """``(P,)`` ascending patch indices as ``[(lo, hi), ...]`` half-open ranges INTO ``patches``.

    A visible separator is guaranteed between spans, so a break in the sequence IS a span
    boundary — the adjacency rule ``utils._span_layout`` groups slots by.
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
    """The stretches of a context curve that are NOT masked, runs of 2+ samples only.

    ``times`` are absolute patch units, so a sample belongs to patch ``floor(t)``.
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
                   - 2 * CHART_PADDING - ui_px(40)
                   - self._attn_block_h())

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

        # smoothing is post-hoc, so toggling only queues a redraw
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
        # Disabled under a blind checkpoint, where dose painting is inert; matched by LABEL.
        from gui_state import dose_painting_enabled
        if not dose_painting_enabled(self.state.masked_channel_policy):
            for b in self._buttons:
                if b.label.split(' [')[0] in _DOSE_PAINTING_BUTTONS:
                    b.enabled = False

        # they re-centre themselves each frame, so no rect plumbing here
        self._help_window = HelpWindow()
        self._event_editor = EventEditorModal()

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

        self._panning: bool = False
        self._pan_start_screen_x: int = 0
        self._pan_start_cx_min: float = 0.0
        self._pan_start_cx_max: float = 0.0

        # the in-flight stroke, appended to ``state.pencil_strokes`` on release
        self._pencil_drawing: bool = False
        self._active_stroke: Any = None  # gui_state.PencilStroke | None

        # Wheel handler clamps against the PREVIOUS frame's content height, measured at draw end.
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
                      - 2 * CHART_PADDING - ui_px(40)
                      - self._attn_block_h(), 100)

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

        # modals re-centre on every draw, so a resize needs no layout fixup here

    def _attn_block_h(self) -> int:
        """Vertical space the strip block takes out of the chart, 0 when hidden."""
        return ATTN_BLOCK_HEIGHT if self.state.attn_overlay_visible else 0

    def _toggle_attn_overlay(self) -> None:
        self.state.attn_overlay_visible = not self.state.attn_overlay_visible
        self._update_layout(self.width, self.height)
        if not self.state.attn_overlay_visible:
            self.state.status_message = "Attention strips off"
        elif self.state.prediction.attribution is not None:
            self.state.status_message = "Attention strips on"
        elif not self._run_attribution_async():
            self.state.status_message = "Attention strips on — predict to fill them"
        self._needs_redraw = True

    def _run_attribution_async(self) -> bool:
        """Fill the strips off the forward already on screen; True when a worker started.

        No re-prediction: the maps have to describe the forward that produced the bands on
        screen, and only the recorded one is guaranteed to be it.
        """
        if (self.model is None or self.state.is_computing
                or self.state.last_forward is None
                or self.state.norm_stats is None):
            return False
        # Claimed synchronously, blocking a second T/SPACE in the same frame from racing it.
        self.state.is_computing = True
        threading.Thread(
            target=_run_attribution,
            args=(self.state, self.model, self.device),
            daemon=True,
        ).start()
        self._needs_redraw = True
        return True

    def _cycle_attn_layer(self, delta: int) -> None:
        """Step through the layers, with the rollout (-1) as one more position."""
        from config import N_LAYERS
        attrib = self.state.prediction.attribution
        n_layers = (int(attrib.per_layer.shape[0]) if attrib is not None
                    else int(N_LAYERS))
        self.state.attn_layer = (
            (self.state.attn_layer + 1 + delta) % (n_layers + 1)
        ) - 1
        self.state.status_message = (
            f"Attention: {self._attn_layer_label()}"
        )
        self._needs_redraw = True

    def _attn_layer_label(self) -> str:
        # 'all', never 'roll': the strips step through rolls too, and one word for both is a trap
        layer = self.state.attn_layer
        return f'L{layer}' if layer >= 0 else 'all'

    def _cycle_attn_roll(self, delta: int) -> None:
        """Step which recorded forward the strips explain.

        Only roll 0 reads evidence that is entirely observed; every later one attends to a
        context partly built from the model's own output.
        """
        forward = self.state.last_forward
        rolls = forward.get('rolls') if forward else None
        if not rolls:
            self.state.status_message = "No forward recorded — predict first"
        elif len(rolls) == 1:
            self.state.status_message = "One forward — no rolls to step through"
        else:
            forward['index'] = (int(forward.get('index', 0)) + delta) % len(rolls)
            if self.state.attn_overlay_visible:
                self.state.prediction.attribution = None
                if not self._run_attribution_async():
                    self.state.status_message = "Attention busy — try again"
            else:
                self.state.status_message = (
                    f"Attention set to {rolls[forward['index']]['label']} "
                    f"of {len(rolls)} — T to show"
                )
        self._needs_redraw = True

    def _pan_view(self, fraction: float) -> None:
        """Slide the time axis by a fraction of the visible span, so one press covers the same
        screen distance at two hours or four days."""
        ct = self.chart_transform
        step = (ct.cx_max - ct.cx_min) * fraction
        ct.update(chart_x_min=ct.cx_min + step, chart_x_max=ct.cx_max + step)
        self._needs_redraw = True

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
        """The trailing forecast span's length in patches, off ``inference``, never ``config``:
        ``main`` rewrites ``inference.PREDICTION_PATCHES`` to the checkpoint's horizon."""
        import inference
        return int(inference.PREDICTION_PATCHES)

    def _dose_painting_blocked(self) -> str:
        """Why a painted dose cannot reach this model, or ``''``.

        Blind training pinned masked-patch doses at ``data.zero_dose_fill``, so an override there
        moves nothing.
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
        # Frame the requested span, not the emitted set — trailing sits far from a backcast one.
        chosen = [(sp.start, sp.length) for sp in self.state.mask_spans]
        self._scroll_to_patches(chosen or spans)
        self.state.status_message = (
            f"{MASK_PRESET_LABELS[preset]} preset — masked set {spans}"
        )
        self._needs_redraw = True

    def _scroll_to_patches(self, spans) -> None:
        """Pan, never zoom, so every span in ``spans`` is on the chart.

        The view opens on the trailing hours of a multi-day window, so a backcast span at patch
        0 and an interior infill both land off the left edge — drawn correctly, seen by nobody.
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
        """Drop the shown forecast: its rows are keyed to the patches the head was asked about,
        so any change to the masked set or the context would draw one set's fan over another's."""
        self.state.prediction.median_bg = None
        self.state.prediction.bands = None
        self.state.prediction.span_patches = None
        self.state.prediction.overrides_raw = None
        self.state.prediction.attribution = None
        self.state.last_forward = None
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
        """Fit-to-drawing roll count over ``self.state``."""
        return _painted_rolls_for_state(self.state, min_rolls, max_rolls)

    def _preview_horizon_patches(self) -> int:
        """Announced-dose preview length in patches over ``self.state``."""
        return _preview_horizon_patches_for_state(self.state)

    def _do_long_predict(self) -> None:
        """Roll out far enough to cover the painted doses, floor ``GUI_LONG_PREDICTION_HOURS``
        and cap ``GUI_MAX_PREDICTION_HOURS``, every roll conditioned on the announced doses."""
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
        """Advance the simulator ``hours`` and append the new truth to the context buffers.

        Distinct from ``_do_roll_forward``, which extends the predicted horizon and leaves the
        simulator where it stands.
        """
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
        # Context grew, so spans sit on different data now; reset rather than re-anchor silently.
        self.state.clear_mask_spans()
        self._clear_prediction()
        # Keep zoom but follow NOW; n_added is the NET shift, 0 once the buffer is saturated.
        if n_added:
            ct = self.chart_transform
            ct.update(chart_x_min=ct.cx_min + n_added, chart_x_max=ct.cx_max + n_added)
        self.state.status_message = (
            f"Sim advanced by {hours:.1f}h ({n_added} patches) — "
            "press SPACE to predict"
        )

    def _do_eval_against_sim(self) -> None:
        """Score the shown forecast against the simulator, into ``state.last_eval``.

        Advances the simulator by the horizon so the window carries real BG; consumes the
        prediction zone (it becomes context), clearing the forecast with it.
        """
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

        # RAW post-noise truth under the same physical clamp the forecast targets
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

        # Evaluated window is context now; the band's prediction zone no longer exists.
        self._clear_prediction()
        self.state.clear_mask_spans()

        self.state.status_message = (
            f"Eval: MAE {mae:.1f} · RMSE {rmse:.1f} · "
            f"Bias {bias:+.1f} mg/dL over {horizon_h:.1f}h"
        )
        self._needs_redraw = True

    def _reset_chart_view(self) -> None:
        """Open on the trailing ``CHART_VIEW_HOURS``.

        The context is days wide and illegible drawn end to end, the forecast most of all. A
        viewport only — every patch is still fed to the forward.
        """
        if self.state.context is None:
            return
        from config import GUI_LONG_PREDICTION_HOURS, PREDICTION_HORIZON_HOURS
        n_ctx = self.state.context.shape[0]
        # Reserve the LONG fan (rolls to GUI_LONG_PREDICTION_HOURS), not the single-pass horizon.
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
        # the previous patient's band was built for a different context and last_bg
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
        """Zoom x around the cursor so the value under it stays fixed; ``wheel_y`` is the raw
        MOUSEWHEEL delta, +1 up = in."""
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


    def _pencil_sample(self, mx: int, my: int) -> tuple[float, float] | None:
        """A screen click as an ``(abs_patch, raw_value)`` sample, or None outside the region.

        x is clamped to ``[n_ctx, n_ctx + max horizon]`` so a stroke never lands in the context;
        y maps from the selected channel's raw range, floored at 0.
        """
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
        # a degenerate stroke would be an invisible edit that still flips the mode to What-If
        if self._active_stroke is not None and len(self._active_stroke.xs) < 2:
            try:
                self.state.pencil_strokes.remove(self._active_stroke)
            except ValueError:
                pass
        self._active_stroke = None
        self._compile_curve_overrides()


    def _patch_at(self, mx: int) -> int:
        """The patch under screen x, floored: the head emits one slot per masked patch, so a
        half-patch mask has no representation."""
        cx, _cy = self.chart_transform.screen_to_chart(float(mx), float(self.chart_transform.sy))
        return int(math.floor(cx))

    def _handle_mask_down(self, mx: int, my: int) -> None:
        """Begin a drag, or select an existing span; ctrl-click removes one."""
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
                    # Strips explain ONE span; selection moved, so re-aim rather than drop.
                    if self.state.attn_overlay_visible:
                        self.state.prediction.attribution = None
                        roll = _selected_roll(self.state)
                        if roll is not None:
                            roll['span'] = self.state.selected_span(self._n_pred())
                        if not self._run_attribution_async():
                            self.state.status_message += ' — predict to remap'
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
        # Clamp before asking: patch n_ctx-1 is the trailing span's separator, already masked.
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
        """``(median_bg over the forecast rows, count of other masked patches)``, ``(None, 0)``
        with no prediction.

        Rows at patch ``>= n_ctx`` are the trailing span; backcast/infill rows pool separately.
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
        """The SELECTED span's anchor in mg/dL, the forecast's when none is.

        One-sided, left-preferring: left neighbour's last step, or right's first at patch 0.
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
        """Recompile the painted announcement for the canvas preview.

        Refreshes ``state.overrides`` / ``prediction.overrides_raw`` and the mode label only;
        the BG forecast moves only on a Predict / What-If, which re-runs the model.
        """
        if self.state.context is None or self.state.norm_stats is None:
            return
        from config import PREDICTION_PATCHES

        n_ctx = self.state.context.shape[0]
        all_curve_events = (
            list(self.state.curve_events)
            + _events_to_curve_events(self.state.events, n_ctx)
        )
        # fit-to-drawing horizon, so a dose painted past the single-pass window still shows
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
        """Surface the blind-policy reason; True when painting is off."""
        blocked = self._dose_painting_blocked()
        if not blocked:
            return False
        self.state.status_message = blocked
        self._needs_redraw = True
        return True

    def _run_prediction_async(self) -> None:
        if self.state.context is None or self.model is None:
            return
        # Claimed synchronously, blocking a second Predict in the same frame from racing it.
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
        # predict_rolling is right-edge only; drop spans rather than shade a span it ignores.
        if self.state.mask_spans:
            self.state.clear_mask_spans()
            self.state.status_message = (
                "Rolling is right-edge only — masked spans cleared"
            )
        # Claimed synchronously; stops a second _do_roll_forward over-incrementing prediction_rolls.
        self.state.is_computing = True

        def _roll():
            # From inference, never config: main() rewrites it to the checkpoint's horizon.
            from inference import predict_rolling, PREDICTION_PATCHES
            from config import PATCH_SIZE
            self.state.is_computing = True
            self.state.status_message = "Rolling forward..."
            try:
                assert self.state.context is not None
                assert self.model is not None
                hour = _hour_at_pred_start(self.state)
                self.state.active_band_label = f"{hour:0.1f}h"

                # Per-roll: same compiler as the single-shot path, keyed to this roll's start patch.
                state = self.state
                # basal_rate_delta rides roll 0 alone, or a later roll re-triggers the ramp.
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
                    return_rolls=True,
                )
                # per-roll medians concatenated, with their per-τ envelope
                self.state.prediction.median_bg = result['pred_bg'].cpu().numpy()
                self.state.prediction.bands = result['bands'].cpu().numpy()
                # Rolling is right-edge; clear any per-span mapping the last masked prediction left.
                self.state.prediction.span_patches = None
                # Record every roll's inputs; the strips read whichever roll the user steps to.
                self.state.prediction.attribution = None
                self.state.last_forward = {
                    'rolls': [{
                        'context': ri['context'],
                        'mask_spans': None,
                        'span': None,
                        'overrides': ri['overrides'],
                        'offset': int(ri['offset']),
                        'label': f"roll {k}",
                    } for k, ri in enumerate(result['roll_inputs'])],
                    'index': 0,
                }
                self.state.prediction.n_rolls = self.state.prediction_rolls

                # Prediction origin is fixed across rolls; decode once from the initial context.
                (self.state.prediction.tod_pred_hour,
                 self.state.prediction.tod_confidence,
                 self.state.prediction.tod_bin_probs) = _decode_tod(
                    self.model, self.state.context, self.state.norm_stats,
                    self.device,
                )

                if self.state.has_edits():
                    self.state.prediction.is_what_if = True
                    self.state.mode_label = f"What-If (curves) · {self.state.active_band_label}"
                    # the painted announcement over the rolled horizon, for the chart overlay
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

                # the chart view is the user's; a roll does not move it
                self.state.status_message = f"Roll {self.state.prediction_rolls} complete"
            except Exception as e:
                traceback.print_exc()
                self.state.status_message = f"Roll error: {e}"
            finally:
                self.state.is_computing = False

        t = threading.Thread(target=_roll, daemon=True)
        t.start()


    def _draw_panel_card(self, surface, x, y, w, h):
        """Rounded card behind a sidebar section: alpha fill, thin border."""
        pygame = self.pygame
        card = pygame.Surface((w, h), pygame.SRCALPHA)
        card.fill((*SIDEBAR_PANEL_BG, 235))
        surface.blit(card, (x, y))
        pygame.draw.rect(
            surface, SIDEBAR_PANEL_BORDER, (x, y, w, h),
            1, border_radius=ui_px(6),
        )

    def _draw_section_header(self, surface, x, y, title, accent):
        """Coloured accent bar and title; returns the y below."""
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
        """Key/value row: key left, value right."""
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
        """Stacked bar of the forecast's low / in-range / high fractions, off the config
        thresholds; returns ``(in_frac, low_frac, high_frac)``."""
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


    def _draw_sidebar(self, surface: 'pygame.Surface') -> None:
        pygame = self.pygame
        pygame.draw.rect(surface, SIDEBAR_BG_COLOR, (0, 0, SIDEBAR_WIDTH, self.height))
        pygame.draw.line(surface, (44, 48, 64),
                         (SIDEBAR_WIDTH - 1, 0),
                         (SIDEBAR_WIDTH - 1, self.height))

        # Clipped above the status bar; widget rects assigned this pass are already screen coords.
        visible_h = max(self.height - STATUS_BAR_HEIGHT, 1)
        max_scroll = max(0, self._sidebar_content_h - visible_h)
        self._sidebar_scroll = max(0, min(self._sidebar_scroll, max_scroll))
        prev_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(0, 0, SIDEBAR_WIDTH, visible_h))

        pad = ui_px(14)
        inner_x = pad
        inner_w = SIDEBAR_WIDTH - 2 * pad
        y = pad - self._sidebar_scroll

        title_img = self._font_large.render("T1DMAI", True, TEXT_COLOR)
        surface.blit(title_img, (inner_x, y))
        y += title_img.get_height()
        sub_img = self._font_small.render(
            "Behavior prediction · interactive what-if",
            True, TEXT_FAINT_COLOR,
        )
        surface.blit(sub_img, (inner_x, y))
        y += sub_img.get_height() + ui_px(10)

        pill_h = ui_px(30)
        self._draw_mode_pill(surface, inner_x, y, inner_w, pill_h)
        y += pill_h + ui_px(6)

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

        # the masked set is what the model was asked about, so it sits beside the patient
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

        # FORECAST rows only; pooling backcast/infill in would mix two different questions.
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

            # the probe's decoded origin hour beside the true one, with R as its confidence
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
            # <15 mg/dL green, <30 yellow, else red — roughly the CGM error tiers
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

        # content height in UNSCROLLED coords, for next frame's wheel handler and scrollbar
        self._sidebar_content_h = y + self._sidebar_scroll + pad
        surface.set_clip(prev_clip)

        # outside the content clip, so the scrollbar is not itself scrolled
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


    _EVENT_CREATE_BUTTONS: list[tuple[str, str, str]] = [
        ('juice',         '',          '+ Juice'),
        ('fast_insulin',  '',          '+ Fast Ins'),
        ('basal_insulin', '',          '+ Basal'),
        ('meal',          'breakfast', '+ Breakfast'),
        ('meal',          'lunch',     '+ Lunch'),
        ('meal',          'dinner',    '+ Dinner'),
    ]

    @staticmethod
    def _event_summary(ev) -> tuple[str, tuple]:
        """``(label, color)`` for one event-list row."""
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

        # the BG channel's raw range maps mg/dL into chart-y, so the band edges track pan/zoom
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
            # Model sees only the trailing MAX_CONTEXT_PATCHES; that window slides past the cap.
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
                        # Break the truth line under a masked span rather than reveal the answer.
                        for seg_t, seg_y in _split_at_masked(
                            ctx_times, y_ctx, self.state.mask_spans,
                        ):
                            draw_curve(surf, local_transform, seg_t, seg_y,
                                       color, width=2)
                    else:
                        draw_curve(surf, local_transform, ctx_times, y_ctx,
                                   color, width=2)

            # One fan PER SPAN: span_patches is predict's mask_idx, slot j is patch mask_idx[j].
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
                    # outermost τ pair, the widest band
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
        """Shade the user's masked spans and the drag in flight.

        Not the trailing forecast span: the NOW line and the prediction-zone shading mark it
        already, and the user cannot remove it.
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
            # A HINT, never a block: the fan is drawn, just uncalibrated by the sampler.
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
                "1-4=Toggle channels (BG/Carbs/Insulin/Exercise)  A=Toggle all  "
                "T=Attention strips  [ / ]=Attention layer  , / .=Attention roll  Q=Quit",
                "Left/Right=Pan (Shift=a full screen)  Scroll=Zoom at cursor  "
                "+/-=Zoom  Middle/Right-drag=Pan",
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

        sy_bottom = int(ct.sy + ct.sh) + self._attn_block_h()
        span = float(ct.cx_max - ct.cx_min)
        major, _minor, fmt = _adaptive_time_intervals(span)
        # Snap to a multiple of the major interval at/before cx_min, so labels land on round times.
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
            # only on a day change; otherwise it repeats under every hour tick
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

    def _attn_mass(self) -> 'np.ndarray | None':
        """The attention row currently on display — one layer's, or the rollout."""
        attrib = self.state.prediction.attribution
        if attrib is None:
            return None
        layer = self.state.attn_layer
        if 0 <= layer < attrib.per_layer.shape[0]:
            return attrib.per_layer[layer]
        return attrib.where

    @staticmethod
    def _display_scale(values: np.ndarray, lo: int, hi: int) -> float:
        """The magnitude that maps to full ink, over the VISIBLE patches.

        Floored at a fraction of the window's peak, so an empty region does not brighten to
        look busy while the true peak sits off-screen.
        """
        window_peak = float(np.abs(values).max()) if values.size else 0.0
        visible = values[max(0, lo):max(0, hi)]
        view_peak = float(np.abs(visible).max()) if visible.size else 0.0
        return max(view_peak, ATTN_VIEW_SCALE_FLOOR * window_peak)

    def _attn_row_data(self, lo: int, hi: int) -> list[tuple] | None:
        """``[(values, up_color, down_color, note), ...]``, one per ``ATTN_ROW_LABELS`` row.

        ``[lo, hi)`` are visible patches in the EXPLAINED window's own coordinates; ``values``
        are in [-1, 1] over that window's patch axis. None when no map has been computed.
        """
        from attribution import share_ramp, signed_ramp
        attrib = self.state.prediction.attribution
        mass = self._attn_mass()
        if attrib is None or mass is None:
            return None
        # Log ramp: mass spans decades, a linear ramp would render most of the row black.
        rows: list[tuple] = [(
            share_ramp(mass, lo, hi),
            ATTN_MASS_COLOR, ATTN_MASS_COLOR, self._attn_layer_label(), None,
        )]
        # Cells with no input to attribute: masked BG, plus dose channels under a blind policy.
        from gui_state import dose_painting_enabled
        withheld = np.zeros(attrib.channels.shape[0], dtype=bool)
        withheld[attrib.masked_patches] = True
        blind = not dose_painting_enabled(self.state.masked_channel_policy)
        # ONE scale across all four channels, else a per-row scale flattens their relative import.
        scale = self._display_scale(attrib.channels, lo, hi)
        for ch in range(attrib.channels.shape[1]):
            rows.append((
                signed_ramp(attrib.channels[:, ch], scale),
                SALIENCY_UP_COLOR, SALIENCY_DOWN_COLOR,
                f'{attrib.channel_share[ch] * 100:.0f}%',
                withheld if (ch == 0 or blind) else None,
            ))
        assert len(rows) == len(ATTN_ROW_LABELS), (
            f"{len(rows)} rows for {len(ATTN_ROW_LABELS)} labels — the channel "
            f"axis and the row labels have diverged"
        )
        return rows

    def _draw_attention_strips(self, surface: 'pygame.Surface') -> None:
        """The attention and saliency rows, on the chart's own x-transform so a bright column
        stays under the stretch of trace it refers to through a pan or zoom."""
        if not self.state.attn_overlay_visible:
            return
        pygame = self.pygame
        ct = self.chart_transform
        block_x, block_w = int(ct.sx), int(ct.sw)
        block_y = int(ct.sy + ct.sh)
        pygame.draw.rect(surface, ATTN_STRIP_BG,
                         (block_x, block_y, block_w, ATTN_BLOCK_HEIGHT))

        attrib = self.state.prediction.attribution
        # Map indices are in the explained forward's OWN window, offset once context slides.
        offset = int(attrib.window_offset) if attrib is not None else 0
        n_patches = int(attrib.where.shape[0]) if attrib is not None else 0
        p_lo = max(offset, int(math.floor(ct.cx_min)))
        p_hi = min(offset + n_patches, int(math.ceil(ct.cx_max)) + 1)
        rows = self._attn_row_data(p_lo - offset, p_hi - offset)

        for i, label in enumerate(ATTN_ROW_LABELS):
            row_y = block_y + ATTN_BLOCK_PAD + i * (ATTN_ROW_H + ATTN_ROW_GAP)
            pygame.draw.rect(surface, ATTN_ROW_BG,
                             (block_x, row_y, block_w, ATTN_ROW_H))
            img = self._font_small.render(label, True, TEXT_DIM_COLOR)
            surface.blit(img, (block_x - img.get_width() - ui_px(6),
                               row_y + (ATTN_ROW_H - img.get_height()) // 2))
            if rows is None:
                continue
            values, up_color, down_color, note, withheld = rows[i]
            row_surf = pygame.Surface((block_w, ATTN_ROW_H), pygame.SRCALPHA)
            for patch in range(p_lo, p_hi):
                idx = patch - offset
                blank = withheld is not None and bool(withheld[idx])
                v = float(values[idx])
                if v == 0.0 and not blank:
                    continue
                sx_l = max(0, int(round(ct.x_to_screen(float(patch)))) - block_x)
                sx_r = min(block_w, int(round(ct.x_to_screen(float(patch + 1)))) - block_x)
                if sx_r <= sx_l:
                    sx_r = min(block_w, sx_l + 1)
                if sx_r <= sx_l:
                    continue
                if blank:
                    color, alpha = ATTN_WITHHELD_COLOR, ATTN_WITHHELD_ALPHA
                else:
                    color = up_color if v >= 0 else down_color
                    alpha = int(min(1.0, abs(v)) * ATTN_ROW_MAX_ALPHA)
                row_surf.fill((*color, alpha), (sx_l, 0, sx_r - sx_l, ATTN_ROW_H))
            surface.blit(row_surf, (block_x, row_y))
            nimg = self._font_small.render(note, True, TEXT_FAINT_COLOR)
            surface.blit(nimg, (block_x + block_w + ui_px(5),
                                row_y + (ATTN_ROW_H - nimg.get_height()) // 2))

        # NOW and the explained span: without both, a strip is a heat bar with no referent
        n_ctx = self.state.context.shape[0] if self.state.context is not None else 0
        now_x = int(round(ct.x_to_screen(float(n_ctx))))
        if block_x <= now_x <= block_x + block_w:
            pygame.draw.line(surface, GRID_COLOR, (now_x, block_y),
                             (now_x, block_y + ATTN_BLOCK_HEIGHT))
        if attrib is not None:
            start, length = attrib.span
            start += offset
            sx_l = max(block_x, min(int(round(ct.x_to_screen(float(start)))),
                                    block_x + block_w))
            sx_r = max(block_x, min(int(round(ct.x_to_screen(float(start + length)))),
                                    block_x + block_w))
            if sx_r > sx_l:
                pygame.draw.rect(
                    surface, ATTN_SPAN_EDGE_COLOR,
                    (sx_l, block_y, sx_r - sx_l, ATTN_BLOCK_HEIGHT), 1,
                )

    def _draw_cursor(self, surface: 'pygame.Surface', mx: int, my: int) -> None:
        pygame = self.pygame
        ct = self.chart_transform
        from config import PATCH_SIZE

        if not (ct.sx <= mx <= ct.sx + ct.sw):
            return
        pygame.draw.line(
            surface, CURSOR_COLOR, (mx, int(ct.sy)),
            (mx, int(ct.sy + ct.sh) + self._attn_block_h()), 1,
        )

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

        attrib = self.state.prediction.attribution
        mass = self._attn_mass()
        if self.state.attn_overlay_visible and attrib is not None and mass is not None:
            patch_idx = int(math.floor(cx)) - int(attrib.window_offset)
            if 0 <= patch_idx < mass.shape[0]:
                share = float(mass[patch_idx]) * float(mass.shape[0])
                tooltip_lines.append((
                    f'attn {self._attn_layer_label()}: '
                    f'{float(mass[patch_idx]):.4f} ({share:.2f}x even)',
                    ATTN_MASS_COLOR,
                ))
                masked_here = patch_idx in set(attrib.masked_patches.tolist())
                from gui_state import dose_painting_enabled
                blind = not dose_painting_enabled(self.state.masked_channel_policy)
                for ch, ch_label in enumerate(ATTN_ROW_LABELS[1:]):
                    if masked_here and (ch == 0 or blind):
                        tooltip_lines.append((
                            f'{ch_label}: withheld here', ATTN_WITHHELD_COLOR,
                        ))
                        continue
                    pull = float(attrib.channels[patch_idx, ch])
                    tooltip_lines.append((
                        f'{ch_label} pull: {pull:+.3f}',
                        SALIENCY_UP_COLOR if pull >= 0 else SALIENCY_DOWN_COLOR,
                    ))

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
        """The time-of-day probe's clock face, rotated to the cursor. Diagnostic only.

        One row per MASKED patch, count is ``probs.shape[0]``, never ``PREDICTION_PATCHES``.
        Rotated on the dial by ``2*pi*t/24`` — angles only, no re-binning.
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
            # Shift+Space is the fit-to-drawing long prediction; plain Space the single pass
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
            # 1/2/3 place the presets while the mask tool is up, else toggle channels
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

        elif key == pygame.K_t:
            self._toggle_attn_overlay()

        elif key == pygame.K_LEFTBRACKET:
            self._cycle_attn_layer(-1)

        elif key == pygame.K_RIGHTBRACKET:
            self._cycle_attn_layer(1)

        elif key == pygame.K_COMMA:
            self._cycle_attn_roll(-1)

        elif key == pygame.K_PERIOD:
            self._cycle_attn_roll(1)

        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            fast = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
            step = PAN_STEP_FAST_FRACTION if fast else PAN_STEP_FRACTION
            self._pan_view(step if key == pygame.K_RIGHT else -step)

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
                # Modals first, so chart zoom/pan and sidebar scroll don't fire inside a sub-window.
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
                self._draw_attention_strips(self._screen)
                self._draw_sidebar(self._screen)
                self._draw_events_panel(self._screen)
                self._draw_control_panel(self._screen)
                self._draw_status_bar(self._screen)
                self._draw_cursor(self._screen, last_mx, last_my)
                self._draw_clock_face_overlay(self._screen, last_mx, last_my)

                # last, so they paint over everything including the cursor overlay
                for modal in (self._help_window, self._event_editor):
                    modal.draw(self._screen, self._font, self._font_large)

                pygame.display.flip()
                self._needs_redraw = False

            clock.tick(FPS)

        pygame.quit()


# One capacity per subdirectory; compare.py reads the same root, so a second would disagree.
MODEL_DIR = 'models'


def _discover_checkpoint() -> str | None:
    """The best checkpoint of the capacity the live ``config.py`` describes.

    Matches ``D_MODEL`` / ``N_LAYERS`` / ``N_HEADS`` in ``training_config`` without loading it.
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
    """``(model, prediction_patches, normalization_stats, masked_channel_policy)``.

    Policy is returned because no parameter shape records it, or a blind checkpoint would be
    driven as a conditioned one.
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
                             'config.py is looked up under models/.')
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
