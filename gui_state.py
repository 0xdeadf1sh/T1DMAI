"""Everything the GUI remembers between frames, in one mutable ``GUIState`` passed by reference.

Threading is confined to ``gui._run_prediction_async``: the worker writes ``is_computing``,
``status_message`` and the ``prediction.*`` slots, the main loop reads them every frame and
defers the prediction layer until ``is_computing`` goes False.  Attribute assignment is
atomic, so no lock.
"""

import copy
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any

import config


TOOL_NONE = 'none'
TOOL_CURVE_EDITOR = 'curve_editor'
TOOL_PENCIL = 'pencil'
TOOL_MASK = 'mask'
TOOL_MEAL_BUILDER = 'meal_builder'
TOOL_BOLUS_BUILDER = 'bolus_builder'
TOOL_SCENARIO = 'scenario'


EVENT_KIND_JUICE = 'juice'
EVENT_KIND_FAST_INSULIN = 'fast_insulin'
EVENT_KIND_BASAL_INSULIN = 'basal_insulin'
EVENT_KIND_MEAL = 'meal'

MEAL_NAMES = ('breakfast', 'lunch', 'dinner', 'snack')


@dataclass
class Event:
    """Intervention template compiled to CurveEvents at prediction time.

    ``time_offset_min`` is minutes from the start of the prediction zone; ``magnitude`` is the
    kind's natural unit (g carbs, U insulin); ``fast_frac`` splits a meal's carbs between the
    fast and slow absorption bells.
    """
    kind: str
    time_offset_min: float
    magnitude: float
    fast_frac: float = 0.5
    name: str = ''


@dataclass
class CurveEvent:
    """One user-placed raised-cosine bell in the prediction zone, zero outside its tails.

    Positions are ABSOLUTE patch units from the start of the context window, so they stay put
    when a roll extends the prediction zone.
    """
    channel: int       # display channel (1=carbs, 2=insulin, 3=exercise)
    left_patch: float   # absolute patch position of left tail
    peak_patch: float   # absolute patch position of peak
    right_patch: float  # absolute patch position of right tail
    amplitude: float    # peak raw value in channel native units

    def auc_raw(self, n_ctx: int, n_pred_patches: int, patch_size: int) -> float:
        """Area under the curve in raw units: total grams, or units of insulin."""
        from gui import _curve_event_to_raw_values
        vals = _curve_event_to_raw_values(self, n_ctx, n_pred_patches, patch_size)
        return float(vals.sum())


@dataclass
class PencilStroke:
    """One freehand announced-dose stroke, as parallel ``xs`` / ``ys`` samples.

    ``xs`` are ABSOLUTE patch positions, like ``CurveEvent``; ``ys`` raw doses — g/5min carbs,
    U/5min insulin, g/step carb-equivalent exercise.  Resampled onto the (patch, timestep) grid
    and smoothed at compile time (``gui._pencil_strokes_to_raw_values``); 0 outside its ``xs``.
    """
    channel: int             # display channel (1=carbs, 2=insulin, 3=exercise)
    xs: list[float] = field(default_factory=list)   # absolute patch positions
    ys: list[float] = field(default_factory=list)   # raw values (g/5min, U/5min or g/step)


# Free-form masking: forecast, begin-fill and infill are POSITIONS of one masked-BG objective,
# not modes.  Everything here is pure — no pygame, no model, no torch — so it is testable with
# no display.  ``n_pred`` is a parameter, never ``config``: ``gui.main`` rewrites
# ``inference.PREDICTION_PATCHES`` to the loaded checkpoint's horizon and leaves ``config``
# alone, so a module-level binding would go stale against the window inference builds.
#
# THE TRAILING SPAN IS NOT OPTIONAL.  ``inference._resolve_mask_spans`` requires every patch of
# ``[n_ctx, n_ctx + n_pred)`` masked: that zone carries no observed BG, so a visible patch there
# announces a fabricated ``z = 0`` (~142 mg/dL) as a reading.  Every emitted set therefore ends
# with ``(n_ctx, n_pred)``, the user's budget is what is left of ``MAX_MASKED_PATCHES``, and a
# pure interior infill is unreachable.

MASK_PRESET_FORECAST = 'forecast'
MASK_PRESET_BEGIN_FILL = 'begin_fill'
MASK_PRESET_INFILL = 'infill'
MASK_PRESETS = (MASK_PRESET_FORECAST, MASK_PRESET_BEGIN_FILL, MASK_PRESET_INFILL)
# any masked set the three presets do not name
MASK_PRESET_CUSTOM = 'custom'

MASK_PRESET_LABELS = {
    MASK_PRESET_FORECAST: 'Forecast',
    MASK_PRESET_BEGIN_FILL: 'Begin-fill',
    MASK_PRESET_INFILL: 'Infill',
    MASK_PRESET_CUSTOM: 'Custom',
}


@dataclass
class MaskSpan:
    """One user-masked span of the CONTEXT: the ``(start_patch, length)`` pair in absolute
    patch units that ``data.sample_mask_spans`` draws.  The trailing forecast span is never
    one of these — it is appended at emit time."""
    start: int
    length: int

    @property
    def end(self) -> int:
        """One past the last masked patch."""
        return self.start + self.length

    @property
    def last(self) -> int:
        return self.start + self.length - 1

    def as_tuple(self) -> tuple[int, int]:
        return int(self.start), int(self.length)


def user_mask_capacity(n_pred: int) -> int:
    """CONTEXT patches the user may still mask: the head's ``MAX_MASKED_PATCHES`` slots less
    the ``n_pred`` the mandatory trailing forecast span already spends."""
    return max(0, int(config.MAX_MASKED_PATCHES) - int(n_pred))


def merge_mask_spans(spans: list[MaskSpan]) -> list[MaskSpan]:
    """Sort by start and fuse every overlapping OR ABUTTING pair into one span.

    Two masked spans never abut: the visible separator is what makes the anchor, the per-span
    spline's node sequence and the DILATE length bucket well defined, and ``utils._span_layout``
    identifies spans by adjacency in ``mask_idx`` because of it.  An abutting pair IS one
    longer span, so this fuses rather than rejects.
    """
    out: list[MaskSpan] = []
    for s in sorted(spans, key=lambda z: (z.start, z.length)):
        if s.length < 1:
            continue
        if out and s.start <= out[-1].end:
            prev = out[-1]
            out[-1] = MaskSpan(prev.start, max(prev.end, s.end) - prev.start)
        else:
            out.append(MaskSpan(int(s.start), int(s.length)))
    return out


def validate_user_spans(spans: list[MaskSpan], n_ctx: int, n_pred: int) -> str:
    """One-line reason ``spans`` cannot be emitted, or ``''`` when they can.

    Assumes ``merge_mask_spans`` has run, so overlap and abutment are resolved rather than
    reported; what is left is the head's slot budget and the window's edges.
    """
    cap = user_mask_capacity(n_pred)
    total = sum(int(s.length) for s in spans)
    if total > cap:
        return (f"masked set is {total + n_pred} patches, over the head's "
                f"{config.MAX_MASKED_PATCHES} slots ({cap} left after the forecast)")
    for s in spans:
        if s.length < 1:
            return f"span at patch {s.start} has no patches"
        if s.start < 0:
            return f"span starts at patch {s.start}, before the window"
        # the trailing span starts at n_ctx, so patch n_ctx - 1 stays visible as the separator
        if s.last > n_ctx - 2:
            return (f"span {s.start}-{s.last} reaches the forecast — leave patch "
                    f"{n_ctx - 1} visible as the separator")
    return ''


def add_user_span(
    spans: list[MaskSpan], new: MaskSpan, n_ctx: int, n_pred: int,
) -> tuple[list[MaskSpan], str]:
    """Add one span, merging and validating; returns ``(spans, reason)``, ``reason`` ``''`` on success.

    Refuses as a whole — on failure the set returned is the ORIGINAL one, never half-applied.
    """
    merged = merge_mask_spans(list(spans) + [new])
    reason = validate_user_spans(merged, n_ctx, n_pred)
    if reason:
        return list(spans), reason
    return merged, ''


def emit_mask_spans(
    spans: list[MaskSpan], n_ctx: int, n_pred: int,
) -> list[tuple[int, int]]:
    """``[(start_patch, length), ...]`` over the ``n_ctx + n_pred`` window, for ``inference.predict``.

    Sorted, non-abutting, inside the window, always ending with the trailing forecast span —
    the four rules ``inference._resolve_mask_spans`` asserts.  ``spans`` need not be merged.
    """
    merged = merge_mask_spans(spans)
    reason = validate_user_spans(merged, n_ctx, n_pred)
    assert not reason, f"illegal masked set: {reason}"
    return [s.as_tuple() for s in merged] + [(int(n_ctx), int(n_pred))]


def preset_user_spans(preset: str, n_ctx: int, n_pred: int) -> list[MaskSpan]:
    """The user spans one preset places, empty when it does not fit.

    ``forecast`` places none — the trailing span is emitted either way.  The other two place
    one span of the forecast's own length, so the presets differ only in WHERE it sits.
    """
    assert preset in MASK_PRESETS, f"unknown mask preset {preset!r}"
    if preset == MASK_PRESET_FORECAST:
        return []
    length = min(int(n_pred), user_mask_capacity(n_pred), max(0, n_ctx - 2))
    if length < 1:
        return []
    if preset == MASK_PRESET_BEGIN_FILL:
        return [MaskSpan(0, length)]
    # centred, never at patch 0 — that is begin-fill, anchored on its RIGHT neighbour
    start = max(1, (n_ctx - length) // 2)
    start = min(start, n_ctx - 1 - length)
    if start < 1:
        return []
    return [MaskSpan(start, length)]


def mask_span_ood(
    spans: list[tuple[int, int]], n_ctx: int, n_pred: int,
) -> dict[int, str]:
    """``{index into the emitted spans: reason}`` for spans the sampler never supervised.

    A hint, never a block: the fan is still a fan, but it was never trained on a mask of this
    shape, so its calibration is evidence of nothing.  Absent keys are in-distribution.
    Three conditions, every threshold read off the sampler's constants: a span longer than
    ``max(MASK_SPAN_LENGTHS)``; a masked patch farther than that from visible evidence on
    either side (``d``, from ``data._mask_slots``, the distance every masked-BG metric bins on);
    more spans than ``MASK_MAX_SPANS``.
    Only the third fires at today's constants — the user's budget is
    ``MAX_MASKED_PATCHES - n_pred`` = 8 = ``max(MASK_SPAN_LENGTHS)``, so no admissible span is
    longer than the length law, and ``d <= length`` makes the second subsume into the first at
    ANY constants.  ``tests/test_gui_masking.py`` pins each relation.
    """
    from data import _mask_slots
    max_len = max(config.MASK_SPAN_LENGTHS)
    seq_len = int(n_ctx) + int(n_pred)
    out: dict[int, str] = {}
    total = sum(int(L) for _s, L in spans)
    if total > config.MAX_MASKED_PATCHES:
        # unemittable, not merely unsupervised: ``_mask_slots`` has nowhere to put the
        # surplus, so say so instead of indexing past the head's slots
        return {i: (f"{total} masked patches exceeds the head's "
                    f"{config.MAX_MASKED_PATCHES} slots")
                for i in range(len(spans))}
    _mask_idx, _valid, d, _anchor = _mask_slots(list(spans), seq_len)
    if len(spans) > config.MASK_MAX_SPANS:
        for i in range(len(spans)):
            out[i] = (f"{len(spans)} spans — the sampler draws at most "
                      f"{config.MASK_MAX_SPANS}")
    slot = 0
    for i, (_start, length) in enumerate(spans):
        span_d = int(d[slot:slot + length].max()) if length else 0
        slot += length
        if length > max_len:
            out[i] = f"span of {length} patches — the sampler draws at most {max_len}"
        elif span_d > max_len:
            out[i] = (f"{span_d} patches from the nearest visible reading — the "
                      f"sampler never leaves more than {max_len}")
    return out


def mask_dose_fill(
    policy: str, stats: dict[str, dict[str, float]] | None,
) -> dict[int, float] | None:
    """``{feat_idx: z}`` under the checkpoint's ``blind`` policy with stats to hand, else ``None``.

    ``announced`` rides the recorded doses through a masked patch; ``blind`` pins them at
    ``normalize(0)`` as training did, and handing such a checkpoint the recorded doses feeds it
    a channel it learned to read as constant.
    """
    from data import MASKED_CHANNEL_POLICY_BLIND, zero_dose_fill
    if policy != MASKED_CHANNEL_POLICY_BLIND or not stats:
        return None
    return zero_dose_fill(stats)


def dose_painting_enabled(policy: str) -> bool:
    """False under ``blind``: those channels were pinned at the no-dose fill throughout
    training, so a painted override is silently inert rather than merely weak."""
    from data import MASKED_CHANNEL_POLICY_BLIND
    return policy != MASKED_CHANNEL_POLICY_BLIND


def span_anchor_cell(start: int, length: int) -> tuple[int, int]:
    """``(patch, step)`` of one span's anchor, in window coordinates.

    Splits ``data._anchor_step_for_span``'s step index — the SAME one-sided, left-preferring
    rule the model is handed, so the readout cannot show an anchor the forward did not use.
    A span at patch 0 takes its RIGHT neighbour's first step, every other its left's last.
    """
    from data import _anchor_step_for_span
    step = _anchor_step_for_span(int(start), int(length))
    return step // config.PATCH_SIZE, step % config.PATCH_SIZE


@dataclass
class EvalResult:
    """One Eval-vs-Sim run: the model's BG forecast against the simulator's later truth.

    ``eval_at_patch`` is the ABSOLUTE patch where the evaluated window starts — where the
    prediction zone sat at eval time.
    """
    mae: float = 0.0          # mean absolute error  (mg/dL)
    rmse: float = 0.0         # root mean squared error  (mg/dL)
    bias: float = 0.0         # mean signed error (pred − truth)  (mg/dL)
    max_abs: float = 0.0      # worst |pred − truth| over the window (mg/dL)
    horizon_h: float = 0.0
    n_steps: int = 0
    pred_bg: np.ndarray | None = None
    truth_bg: np.ndarray | None = None
    eval_at_patch: int = 0


@dataclass
class PredictionResult:
    """The latest BG quantile forecast, risk-space quantiles already inverted to mg/dL.

    ``P`` is ``PREDICTION_PATCHES``, or ``n_rolls ×`` that after a rolling forecast.
    """
    median_bg: np.ndarray | None = None     # (P*S,) headline BG forecast (mg/dL)
    bands: np.ndarray | None = None         # (P, S, N_QUANTILES) per-τ band edges (mg/dL)
    # (P,) ABSOLUTE patch of each ``bands`` row — ``predict``'s own ``mask_idx``. The masked set
    # is arbitrary, so the chart places row j at patch mask_idx[j], never at a fixed offset from
    # the context end. None ⇒ the rows ARE the trailing forecast, in order (and the rolling path).
    span_patches: np.ndarray | None = None
    n_rolls: int = 1                        # number of rolls for extended prediction
    is_what_if: bool = False                # whether this is an input-perturbation what-if
    overrides_raw: dict[int, np.ndarray] | None = None  # raw carb/insulin/exercise announced in the pred zone
    # Time-of-day probe, diagnostic only (``gui._decode_tod``). All None when the probe is off
    # (``TIME_PROBE_ENABLED`` False) or the decode failed.
    tod_pred_hour: float | None = None      # model-decoded prediction-origin hour-of-day, [0, 24)
    tod_confidence: float | None = None     # resultant length R of the per-bin belief, [0, 1], higher = more confident
    tod_bin_probs: np.ndarray | None = None  # (P, TIME_PROBE_N_BINS) per-patch softmax belief; None when probe off
    # ``attribution.Attribution`` for the selected span; None when the overlay is off or it
    # failed. Keyed to the same masked set and doses as the bands above — a map from another
    # forward would explain a forecast nobody is looking at.
    attribution: Any = None


class GUIState:
    def __init__(self) -> None:
        # One entry per display channel, zipped against gui.CHANNEL_COLORS to build the sidebar
        # toggles — a short list truncates that zip and the missing toggle just disappears.
        self.channel_visible: list[bool] = [True, True, True, True]
        self.channel_names: list[str] = [
            'Blood Glucose', 'Carbs', 'Insulin', 'Exercise',
        ]

        self.active_tool: str = TOOL_NONE
        self.selected_channel: int = -1       # which channel is selected for editing

        # {output channel 0=carbs, 1=insulin, 2=exercise: (P, S) NORMALIZED}, passed straight
        # to inference.predict_what_if. BG is never overridable — the model always predicts it.
        self.overrides: dict[int, np.ndarray] = {}

        self.prediction: PredictionResult = PredictionResult()
        self.prediction_rolls: int = 1        # number of prediction-horizon-long rolls

        self.context: torch.Tensor | None = None  # (n_ctx, PATCH_SIZE, N_INPUT_FEATURES)
        self.patient_seed: int = 42
        self.patient_summary: dict[str, str] | None = None
        self.bg_raw: np.ndarray | None = None  # raw BG for context region (mg/dL)
        self.context_raw: np.ndarray | None = None  # (n_ctx*PATCH_SIZE, N_INPUT_FEATURES) raw display values
        self.norm_stats: dict | None = None
        self.last_bg: float = 100.0           # last observed BG before prediction zone
        self.sim_start_hour: float = 0.0      # hour-of-day at first context timestep
        self.sim_start_day: int = 0            # day index at first context timestep
        # Live simulator, so "Sim Fwd" advances the truth in place and appends patches to the
        # context. None until a patient is loaded.
        self.sim: Any = None

        # display channel being edited: 1=Carbs, 2=Insulin, 3=Exercise (BG is not editable)
        self.selected_edit_channel: int = 1

        self.view_start_patch: float = 0.0    # leftmost visible patch index
        self.view_end_patch: float = 100.0    # rightmost visible patch index
        self.y_scale_left: tuple[float, float] = (0.0, 400.0)   # (min, max) for left axis
        self.y_scale_right: tuple[float, float] = (0.5, 2.5)    # secondary axis

        self.cursor_patch: float = -1.0       # cursor position in patch units
        self.is_computing: bool = False       # model inference in progress
        self.status_message: str = "Ready"
        self.mode_label: str = "Standard"
        self.active_band_label: str = ""

        self.curve_events: list[CurveEvent] = []
        self.selected_event_idx: int = -1
        self.dragging_point: str | None = None  # 'peak', 'left', 'right', or None

        # Pencil strokes and CurveEvents coexist: both compile to announced doses and SUM
        # (``gui._compile_overrides_from_edits``).
        self.pencil_strokes: list[PencilStroke] = []

        # juice / meal / bolus templates, compiled to CurveEvents at prediction time by
        # ``gui._events_to_curve_events``
        self.events: list[Event] = []
        self.events_panel_visible: bool = True

        self.basal_rate_delta: float = 0.0    # U/h (can be negative)

        self.builder_time_patch: float = -1.0  # where user clicked for event time
        self.builder_params: dict[str, Any] = {}

        # Independent of ``prediction``, so it survives Predict / Sim Fwd / Reset; cleared on
        # New Patient.
        self.last_eval: EvalResult | None = None

        # Cosmetic only — the prediction tensors are never modified.
        self.smooth_mu: bool = True
        self.smooth_band: bool = True

        self.screenshot_count: int = 0

        # User spans over the CONTEXT only; the mandatory trailing span is appended at emit
        # time. Cleared on every context change — Sim Fwd appends patches and New Patient
        # replaces them, so a kept span would mask different data than the user drew.
        self.mask_spans: list[MaskSpan] = []
        self.selected_mask_idx: int = -1       # index into mask_spans, -1 = the forecast
        self.mask_preset: str = MASK_PRESET_FORECAST
        # the checkpoint's own ``data.stored_masked_channel_policy``; under 'blind' the masked
        # spans carry ``data.zero_dose_fill`` and dose painting is disabled
        self.masked_channel_policy: str = 'announced'

        # The exact inputs ``inference.predict`` consumed for the forecast on screen, so the
        # maps can be filled or re-aimed without a second prediction — a re-run would have to
        # reproduce this forward exactly and nothing would check that it had. None until a
        # prediction runs; cleared with the prediction it belongs to.
        self.last_forward: dict[str, Any] | None = None

        # Off by default: the maps cost an extra grad-enabled forward per prediction.
        # ``attn_layer`` -1 is the rollout across all layers, else that layer's own attention.
        self.attn_overlay_visible: bool = False
        self.attn_layer: int = -1

        self.mask_drag_start: int = -1         # first patch of the in-flight drag
        self.mask_drag_end: int = -1           # last patch of the in-flight drag

    def set_override(self, channel: int, values: np.ndarray) -> None:
        """``channel`` 0=carbs, 1=insulin, 2=exercise; ``values`` ``(P, PATCH_SIZE)`` normalized."""
        assert values.ndim == 2, f"Override must be 2D (patches, timesteps), got {values.shape}"
        self.overrides[channel] = values.copy()

    def clear_overrides(self) -> None:
        self.overrides.clear()
        self.curve_events.clear()
        self.pencil_strokes.clear()
        self.events.clear()
        self.basal_rate_delta = 0.0
        self.selected_event_idx = -1
        self.dragging_point = None
        self.prediction.is_what_if = False
        self.mode_label = "Standard"

    def has_overrides(self) -> bool:
        return len(self.overrides) > 0

    def has_edits(self) -> bool:
        return (len(self.curve_events) > 0
                or len(self.pencil_strokes) > 0
                or len(self.events) > 0
                or self.basal_rate_delta != 0.0)

    def n_ctx(self) -> int:
        """Context length in patches, 0 when no patient is loaded."""
        return int(self.context.shape[0]) if self.context is not None else 0

    def add_mask_span(self, start: int, length: int, n_pred: int) -> str:
        """Mask ``length`` patches from absolute patch ``start``; ``''``, or why nothing changed."""
        spans, reason = add_user_span(
            self.mask_spans, MaskSpan(int(start), int(length)),
            self.n_ctx(), n_pred,
        )
        if reason:
            return reason
        self.mask_spans = spans
        self.selected_mask_idx = -1
        self.mask_preset = MASK_PRESET_CUSTOM
        return ''

    def remove_mask_span(self, idx: int) -> None:
        """Drop one user span by index; no-op out of range."""
        if 0 <= idx < len(self.mask_spans):
            del self.mask_spans[idx]
            self.selected_mask_idx = -1
            self.mask_preset = (
                MASK_PRESET_FORECAST if not self.mask_spans else MASK_PRESET_CUSTOM
            )

    def clear_mask_spans(self) -> None:
        self.mask_spans = []
        self.selected_mask_idx = -1
        self.mask_preset = MASK_PRESET_FORECAST
        self.mask_drag_start = -1
        self.mask_drag_end = -1

    def apply_mask_preset(self, preset: str, n_pred: int) -> None:
        self.mask_spans = preset_user_spans(preset, self.n_ctx(), n_pred)
        self.mask_preset = preset
        self.selected_mask_idx = -1

    def emitted_mask_spans(self, n_pred: int) -> list[tuple[int, int]]:
        """The masked set for ``inference.predict``, trailing span included."""
        return emit_mask_spans(self.mask_spans, self.n_ctx(), n_pred)

    def masked_patch_count(self, n_pred: int) -> int:
        """Total masked patches, the mandatory trailing span included."""
        return sum(s.length for s in merge_mask_spans(self.mask_spans)) + int(n_pred)

    def mask_budget_left(self, n_pred: int) -> int:
        used = sum(s.length for s in merge_mask_spans(self.mask_spans))
        return max(0, user_mask_capacity(n_pred) - used)

    def selected_span(self, n_pred: int) -> tuple[int, int]:
        """``(start_patch, length)`` of the SELECTED span; the trailing forecast span when
        nothing is selected, that being the one span always in the emitted set."""
        if 0 <= self.selected_mask_idx < len(self.mask_spans):
            return self.mask_spans[self.selected_mask_idx].as_tuple()
        return (self.n_ctx(), int(n_pred))

    def selected_anchor_cell(self, n_pred: int) -> tuple[int, int]:
        """``(patch, step)`` of the SELECTED span's anchor; the context edge when nothing is
        selected."""
        start, length = self.selected_span(n_pred)
        return span_anchor_cell(start, length)

    def set_tool(self, tool: str) -> None:
        self.active_tool = tool
        self.mask_drag_start = -1
        self.mask_drag_end = -1
        if tool == TOOL_NONE:
            self.builder_time_patch = -1.0
            self.builder_params = {}

    def toggle_channel(self, channel: int) -> None:
        if 0 <= channel < len(self.channel_visible):
            self.channel_visible[channel] = not self.channel_visible[channel]

    def toggle_all_channels(self) -> None:
        """Any visible → hide all; all hidden → show all."""
        if any(self.channel_visible):
            self.channel_visible = [False] * len(self.channel_visible)
        else:
            self.channel_visible = [True] * len(self.channel_visible)

    def cycle_edit_channel(self, delta: int = 1) -> None:
        """Cycle ``selected_edit_channel`` over the paintable channels.

        Display channel 0 is BG, the model's forecast and the only non-editable one; the rest
        come off ``channel_names``, so a channel added to that table is reachable here.
        """
        editable = list(range(1, len(self.channel_names)))
        try:
            idx = editable.index(self.selected_edit_channel)
        except ValueError:
            idx = 0
        idx = (idx + delta) % len(editable)
        self.selected_edit_channel = editable[idx]

    def get_predicted_tir(self) -> float | None:
        """Fraction of the median BG forecast inside 70–180 mg/dL, [0, 1]; None with no forecast."""
        if self.prediction.median_bg is None:
            return None
        bg = self.prediction.median_bg
        # 70.0 / 180.0 == train.py's BG_TARGET_LO / BG_TARGET_HI, literal to avoid the import
        in_range = np.sum((bg >= 70.0) & (bg <= 180.0))
        return float(in_range / len(bg))

    def get_mean_uncertainty(self) -> float | None:
        """Mean half-width ``(q_hi − q_lo)/2`` of the OUTERMOST quantile pair, mg/dL; None with
        no bands."""
        bands = self.prediction.bands
        if bands is None or bands.shape[-1] < 2:
            return None
        half = (bands[..., -1] - bands[..., 0]) * 0.5
        return float(np.mean(half))

    def get_max_uncertainty(self) -> float | None:
        """Max half-width of the outermost quantile pair, mg/dL."""
        bands = self.prediction.bands
        if bands is None or bands.shape[-1] < 2:
            return None
        half = (bands[..., -1] - bands[..., 0]) * 0.5
        return float(np.max(half))
