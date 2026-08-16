"""
T1DMAI GUI State Management — central mutable state container.
================================================================

This module centralizes everything the GUI needs to remember between frames:

* What's visible (per-channel toggles, all-channels toggle).
* Which tool is active (curve editor, meal builder, bolus builder, …).
* What overrides the user has painted into the prediction zone.
* Which patches the user has MASKED, and the constraints that set must satisfy.
* The latest prediction result (median BG forecast + per-τ quantile bands).
* The chart view state (visible patch range, y-axis scales).
* The current patient context (raw + normalized buffers, last observed BG).
* Builder ephemeral state (last clicked patch, in-flight builder params).

Risk-space model note
---------------------
The model emits ONLY a BG quantile forecast:
``predict`` returns ``{q_tau, median_bg, bands, last_bg}``.  There are no
dynamics outputs (carb / insulin / IS / HGO), no trend head, and no physics
reconstruction.  The headline forecast is ``median_bg`` (mg/dL) and the
uncertainty envelope is ``bands`` (per-τ quantile edges in mg/dL).  The
what-if path perturbs the announced carb / insulin / exercise INPUT and
re-runs ``predict`` — the model's BG response moves accordingly.

Design choices
--------------
The state lives in a single mutable ``GUIState`` object that is passed by
reference everywhere.  This is intentional: the GUI is one process, one
event loop, no concurrency concerns, and the alternative (events / pubsub
/ Redux-style reducers) would add code without buying anything.

The one place threading matters is during inference — see
``gui._run_prediction_async``.  The prediction worker writes to
``state.is_computing``, ``state.status_message`` and the
``state.prediction.*`` slots; the main loop reads them on every frame and
defers redrawing the prediction layer until ``is_computing`` flips to
False.  Writes to those slots are atomic (Python attribute assignment) so
no extra locking is needed.
"""

import copy
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any

import config


# === Tool identifiers ===
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
    """User-friendly intervention template that compiles to one or more
    CurveEvents at prediction time.

    Time is relative to the start of the prediction zone ('now'), in
    minutes. ``magnitude`` is in the natural unit for the kind (grams for
    carb events, units for insulin). For meals, ``fast_frac`` splits the
    carbs between a fast and slow absorption bell."""
    kind: str
    time_offset_min: float
    magnitude: float
    fast_frac: float = 0.5
    name: str = ''


@dataclass
class CurveEvent:
    """A single user-placed curve in the prediction zone.

    Defined by three control points:
      * Left tail  — draggable left/right, sets curve start.
      * Peak       — draggable up/down, sets amplitude.
      * Right tail — draggable left/right, sets curve end.

    The curve shape is a raised-cosine bell: smooth rise from left to
    peak, smooth fall from peak to right, zero elsewhere.
    Positions are in *absolute* patch units (from the start of the
    context window), so they stay anchored when the prediction zone
    extends via rolling.
    """
    channel: int       # display channel (1=carbs, 2=insulin, 3=exercise)
    left_patch: float   # absolute patch position of left tail
    peak_patch: float   # absolute patch position of peak
    right_patch: float  # absolute patch position of right tail
    amplitude: float    # peak raw value in channel native units

    def auc_raw(self, n_ctx: int, n_pred_patches: int, patch_size: int) -> float:
        """Return the area-under-curve in raw units (total grams / units)."""
        from gui import _curve_event_to_raw_values
        vals = _curve_event_to_raw_values(self, n_ctx, n_pred_patches, patch_size)
        return float(vals.sum())


@dataclass
class PencilStroke:
    """A single freehand pencil stroke over the prediction zone.

    The user drags the pencil to draw an announced carb / insulin / exercise
    curve directly on the chart. Samples are stored as parallel arrays of
    *absolute* patch positions (``xs``, from the start of the context
    window, so they stay anchored when the prediction zone extends via
    rolling — exactly like ``CurveEvent``) and raw-unit dose values
    (``ys``, g/5min for carbs, U/5min for insulin, g/step carbohydrate-
    equivalent for exercise).

    The stroke is resampled onto the model's (patch, timestep) grid and
    smoothed at compile time (``gui._pencil_strokes_to_raw_values``); it
    contributes a dose of 0 outside its drawn ``xs`` span.
    """
    channel: int             # display channel (1=carbs, 2=insulin, 3=exercise)
    xs: list[float] = field(default_factory=list)   # absolute patch positions
    ys: list[float] = field(default_factory=list)   # raw values (g/5min, U/5min or g/step)


# ============================================================================
# Free-form masking
# ============================================================================
#
# The user masks patch-aligned spans of the context and the model fills them.
# Forecast, begin-fill and infill are POSITIONS of the one masked-BG objective,
# not modes: the only thing that changes between them is where the spans sit.
#
# Everything here is pure — no pygame, no model, no torch — so the constraints
# are testable without a display.  ``n_pred`` is passed in rather than read from
# ``config``: ``gui.main`` rewrites ``inference.PREDICTION_PATCHES`` to the
# loaded checkpoint's horizon and leaves ``config`` alone, so a module-level
# binding here would go stale against the window inference actually builds.
#
# THE TRAILING SPAN IS NOT OPTIONAL.  ``inference._resolve_mask_spans`` requires
# every patch of the future zone ``[n_ctx, n_ctx + n_pred)`` to be masked — that
# zone carries no observed BG, so a visible patch there announces a fabricated
# ``z = 0`` (~142 mg/dL) as a reading.  Every emitted masked set therefore ends
# with ``(n_ctx, n_pred)``, and the user's own spans draw on what is left of the
# head's ``MAX_MASKED_PATCHES`` slots.  A pure interior infill is unreachable.

MASK_PRESET_FORECAST = 'forecast'
MASK_PRESET_BEGIN_FILL = 'begin_fill'
MASK_PRESET_INFILL = 'infill'
MASK_PRESETS = (MASK_PRESET_FORECAST, MASK_PRESET_BEGIN_FILL, MASK_PRESET_INFILL)
# Free-form dragging is the point; the presets are three named positions of the
# one objective, and anything else is this.
MASK_PRESET_CUSTOM = 'custom'

MASK_PRESET_LABELS = {
    MASK_PRESET_FORECAST: 'Forecast',
    MASK_PRESET_BEGIN_FILL: 'Begin-fill',
    MASK_PRESET_INFILL: 'Infill',
    MASK_PRESET_CUSTOM: 'Custom',
}


@dataclass
class MaskSpan:
    """One user-masked span of the CONTEXT, in absolute patch units.

    ``start`` is the first masked patch and ``length`` the patch count, the same
    ``(start_patch, length)`` pair ``data.sample_mask_spans`` draws and
    ``inference.predict`` takes.  The trailing forecast span is never one of
    these — it is appended at emit time.
    """
    start: int
    length: int

    @property
    def end(self) -> int:
        """One past the last masked patch."""
        return self.start + self.length

    @property
    def last(self) -> int:
        """Index of the last masked patch."""
        return self.start + self.length - 1

    def as_tuple(self) -> tuple[int, int]:
        return int(self.start), int(self.length)


def user_mask_capacity(n_pred: int) -> int:
    """How many CONTEXT patches the user may still mask.

    The head has exactly ``MAX_MASKED_PATCHES`` slots and the mandatory trailing
    forecast span already spends ``n_pred`` of them.

    Args:
        n_pred: length of the trailing forecast span, in patches.

    Returns:
        The user's patch budget (>= 0).
    """
    return max(0, int(config.MAX_MASKED_PATCHES) - int(n_pred))


def merge_mask_spans(spans: list[MaskSpan]) -> list[MaskSpan]:
    """Sort by start and fuse every overlapping OR ABUTTING pair.

    Two masked spans never abut: one visible separator patch is what makes the
    anchor, the per-span median basis and the DILATE length bucket well defined
    per span, and ``utils._span_layout`` identifies spans by adjacency in
    ``mask_idx`` precisely because of it.  An abutting pair IS one longer span,
    so this fuses rather than rejects.

    Args:
        spans: any collection of spans, in any order.

    Returns:
        A new sorted list with no overlaps and no abutments.
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
    """Why ``spans`` cannot be emitted, or ``''`` when they can.

    Assumes ``merge_mask_spans`` has already run, so overlap and abutment
    between user spans are resolved rather than reported.  What is left is the
    head's slot budget and the window's edges — including the separator patch
    the trailing forecast span needs on its left.

    Args:
        spans: merged user spans.
        n_ctx: context length in patches.
        n_pred: trailing forecast span length in patches.

    Returns:
        A one-line reason, or ``''`` if the set is legal.
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
        # ``<= n_ctx - 2`` is the separator: the trailing span starts at n_ctx,
        # and patch n_ctx - 1 must stay visible between them.
        if s.last > n_ctx - 2:
            return (f"span {s.start}-{s.last} reaches the forecast — leave patch "
                    f"{n_ctx - 1} visible as the separator")
    return ''


def add_user_span(
    spans: list[MaskSpan], new: MaskSpan, n_ctx: int, n_pred: int,
) -> tuple[list[MaskSpan], str]:
    """Add one span to the set, merging and validating.

    Refuses as a whole: on failure the returned set is the ORIGINAL one, so a
    drag that would breach the cap leaves the existing spans alone rather than
    half-applying.

    Args:
        spans: the current user spans.
        new: the span the user just dragged.
        n_ctx: context length in patches.
        n_pred: trailing forecast span length in patches.

    Returns:
        ``(spans, reason)`` — ``reason`` is ``''`` on success.
    """
    merged = merge_mask_spans(list(spans) + [new])
    reason = validate_user_spans(merged, n_ctx, n_pred)
    if reason:
        return list(spans), reason
    return merged, ''


def emit_mask_spans(
    spans: list[MaskSpan], n_ctx: int, n_pred: int,
) -> list[tuple[int, int]]:
    """The masked set to hand ``inference.predict``.

    Sorted, non-abutting, inside the window, and always ending with the trailing
    forecast span — the four rules ``inference._resolve_mask_spans`` asserts.

    Args:
        spans: the user's context spans (need not be merged).
        n_ctx: context length in patches.
        n_pred: trailing forecast span length in patches.

    Returns:
        ``[(start_patch, length), ...]`` over the ``n_ctx + n_pred`` window.
    """
    merged = merge_mask_spans(spans)
    reason = validate_user_spans(merged, n_ctx, n_pred)
    assert not reason, f"illegal masked set: {reason}"
    return [s.as_tuple() for s in merged] + [(int(n_ctx), int(n_pred))]


def preset_user_spans(preset: str, n_ctx: int, n_pred: int) -> list[MaskSpan]:
    """The user spans one preset places.

    ``forecast`` places none — the trailing span alone is the forecast, and it
    is emitted either way.  The other two place a single span of the forecast's
    own length, so the three presets differ only in WHERE the span sits.

    Args:
        preset: one of ``MASK_PRESETS``.
        n_ctx: context length in patches.
        n_pred: trailing forecast span length in patches.

    Returns:
        The preset's spans, empty when it does not fit.
    """
    assert preset in MASK_PRESETS, f"unknown mask preset {preset!r}"
    if preset == MASK_PRESET_FORECAST:
        return []
    length = min(int(n_pred), user_mask_capacity(n_pred), max(0, n_ctx - 2))
    if length < 1:
        return []
    if preset == MASK_PRESET_BEGIN_FILL:
        return [MaskSpan(0, length)]
    # Interior: centred in the context, and never at patch 0 (that is begin-fill,
    # whose anchor comes from the right neighbour instead of the left).
    start = max(1, (n_ctx - length) // 2)
    start = min(start, n_ctx - 1 - length)
    if start < 1:
        return []
    return [MaskSpan(start, length)]


def mask_span_ood(
    spans: list[tuple[int, int]], n_ctx: int, n_pred: int,
) -> dict[int, str]:
    """Which EMITTED spans sit outside what the sampler ever supervised.

    A hint, never a block: the model still runs, and the fan is still a fan —
    it was simply never trained on a mask of this shape, so its calibration is
    not evidence of anything.  Every threshold is read off the sampler's own
    constants rather than written down.

    Three conditions:

    * a span longer than ``max(MASK_SPAN_LENGTHS)`` — no training span was;
    * a masked patch farther than ``max(MASK_SPAN_LENGTHS)`` from visible
      evidence on either side (``d``, reused from ``data._mask_slots`` so the
      distance is the one every masked-BG metric bins on);
    * more spans than ``MASK_MAX_SPANS``, which the sampler never draws.

    Only the third fires today.  The first is unreachable at the current
    constants — the user's budget is ``MAX_MASKED_PATCHES - n_pred`` = 8, exactly
    ``max(MASK_SPAN_LENGTHS)``, so no span the cap admits is longer than the
    length law — and comes alive the moment the head's spare slots outgrow it.
    The second is subsumed by the first at ANY constants: ``d <= length`` for
    every span, so a patch farther than the law from evidence sits in a span
    longer than the law.  Both are still read off the constants rather than
    dropped; ``tests/test_gui_masking.py`` pins each relation.

    Args:
        spans: the emitted masked set, trailing span included.
        n_ctx: context length in patches.
        n_pred: trailing forecast span length in patches.

    Returns:
        ``{index into spans: reason}`` — absent keys are in-distribution.
    """
    from data import _mask_slots
    max_len = max(config.MASK_SPAN_LENGTHS)
    seq_len = int(n_ctx) + int(n_pred)
    out: dict[int, str] = {}
    total = sum(int(L) for _s, L in spans)
    if total > config.MAX_MASKED_PATCHES:
        # Unemittable, not merely unsupervised — ``validate_user_spans`` refuses
        # it upstream and ``_mask_slots`` has nowhere to put the surplus. Say so
        # instead of indexing past the head's slots.
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
    """What the masked spans' dose channels carry, per the checkpoint's policy.

    ``announced``: ``None`` — the recorded carb / insulin / exercise ride through
    a masked patch, which is that convention.  ``blind``: ``data.zero_dose_fill``,
    matching training, where those channels were pinned at ``normalize(0)`` on
    every masked patch.  Handing a blind checkpoint the recorded doses feeds it a
    channel it was trained to read as constant.

    Args:
        policy: the checkpoint's ``masked_channel_policy``.
        stats: the run's normalization statistics.

    Returns:
        ``{feat_idx: z}`` under ``blind`` with stats available, else ``None``.
    """
    from data import MASKED_CHANNEL_POLICY_BLIND, zero_dose_fill
    if policy != MASKED_CHANNEL_POLICY_BLIND or not stats:
        return None
    return zero_dose_fill(stats)


def dose_painting_enabled(policy: str) -> bool:
    """Whether painted doses can reach the model under ``policy``.

    False under ``blind``: the masked-patch dose channels were pinned at the
    no-dose fill throughout training, so a painted override lands on a channel
    the weights learned to ignore and the forecast does not move.  A silently
    inert control is worse than a disabled one.
    """
    from data import MASKED_CHANNEL_POLICY_BLIND
    return policy != MASKED_CHANNEL_POLICY_BLIND


def span_anchor_cell(start: int, length: int) -> tuple[int, int]:
    """``(patch, step)`` of one span's anchor, in window coordinates.

    Thin split of ``data._anchor_step_for_span``'s window-relative step index —
    the SAME one-sided, left-preferring rule the model is handed, so the chart's
    readout cannot show an anchor the forward did not use.  A span at patch 0
    takes the first step of its RIGHT neighbour; every other span takes the last
    step of its left neighbour.

    Args:
        start: first masked patch of the span.
        length: span length in patches.

    Returns:
        ``(patch_idx, step_idx)`` of the anchor cell.
    """
    from data import _anchor_step_for_span
    step = _anchor_step_for_span(int(start), int(length))
    return step // config.PATCH_SIZE, step % config.PATCH_SIZE


@dataclass
class EvalResult:
    """Outcome of a single Eval-vs-Sim run: snapshot of the model's BG
    prediction and the simulator's later ground-truth BG over the same
    window, plus the standard error stats.

    ``pred_bg`` / ``truth_bg`` are kept around so the chart can overlay
    the prediction on the now-revealed ground truth if desired.
    ``eval_at_patch`` is the absolute patch index where the evaluated
    window starts (i.e. where the prediction zone was at eval time)."""
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
    """Holds the latest BG quantile forecast from the model.

    The model emits ONLY a BG forecast (risk-space quantiles inverted to
    mg/dL).  ``median_bg`` is the headline point forecast and ``bands`` is the
    per-τ quantile envelope.  Shapes: ``median_bg`` is ``(P*S,)`` mg/dL and
    ``bands`` is ``(P, S, N_QUANTILES)`` mg/dL, where ``P`` is
    ``PREDICTION_PATCHES`` for a single prediction or ``n_rolls ×
    PREDICTION_PATCHES`` after a rolling forecast.  There are no dynamics
    channels, no σ, and no physics reconstruction.
    """
    median_bg: np.ndarray | None = None     # (P*S,) headline BG forecast (mg/dL)
    bands: np.ndarray | None = None         # (P, S, N_QUANTILES) per-τ band edges (mg/dL)
    # (P,) absolute patch index of each row of ``bands`` — ``predict``'s own
    # ``mask_idx``, kept because the masked set is arbitrary. Slot j is patch
    # mask_idx[j], so the chart must place each row where the head read it and
    # never at a fixed offset from the context end. None ⇒ the rows are the
    # trailing forecast, in order (the pre-masking case, and the rolling path).
    span_patches: np.ndarray | None = None
    n_rolls: int = 1                        # number of rolls for extended prediction
    is_what_if: bool = False                # whether this is an input-perturbation what-if
    overrides_raw: dict[int, np.ndarray] | None = None  # raw carb/insulin/exercise announced in the pred zone
    # Time-of-day probe read-out (diagnostic; decoded from the model's
    # ``return_time=True`` head — see ``gui._decode_tod``). Both are None when
    # the probe is disabled (``TIME_PROBE_ENABLED`` False) or on decode failure.
    tod_pred_hour: float | None = None      # model-decoded prediction-origin hour-of-day, [0, 24)
    tod_confidence: float | None = None     # resultant length R of the per-bin softmax belief (via utils.time_of_day_decode_bins; higher = more confident)
    tod_bin_probs: np.ndarray | None = None  # (P, TIME_PROBE_N_BINS) per-patch softmax belief; None when probe off


class GUIState:
    """
    Central state container for the T1DMAI GUI.

    Tracks visibility toggles, active tool, overrides, prediction results,
    chart transforms, and interaction state.
    """

    def __init__(self) -> None:
        # ---- Channel visibility toggles (N_INPUT_FEATURES channels) ----
        # Channels: BG (the model's forecast), Carbs, Insulin, Exercise (the
        # announced what-if inputs). IS / HGO / BG-delta are no longer model
        # signals. Both lists carry one entry per display channel and are zipped
        # against gui.CHANNEL_COLORS to build the sidebar toggles — a short list
        # truncates that zip and the missing toggle just disappears.
        self.channel_visible: list[bool] = [True, True, True, True]
        self.channel_names: list[str] = [
            'Blood Glucose', 'Carbs', 'Insulin', 'Exercise',
        ]

        # ---- Active tool ----
        self.active_tool: str = TOOL_NONE
        self.selected_channel: int = -1       # which channel is selected for editing

        # ---- Overrides: output-channel index → (P, S) array ----
        # Keys are output-channel indices [0=carbs, 1=insulin, 2=exercise]
        # passed straight to inference.predict_what_if's overrides dict
        # (NORMALIZED values). BG can't be overridden — the model always
        # predicts it.
        self.overrides: dict[int, np.ndarray] = {}

        # ---- Prediction results ----
        self.prediction: PredictionResult = PredictionResult()
        self.prediction_rolls: int = 1        # number of prediction-horizon-long rolls

        # ---- Context data (from simulator or loaded) ----
        self.context: torch.Tensor | None = None  # (n_ctx, PATCH_SIZE, N_INPUT_FEATURES)
        self.patient_seed: int = 42
        self.patient_summary: dict[str, str] | None = None
        self.bg_raw: np.ndarray | None = None  # raw BG for context region (mg/dL)
        self.context_raw: np.ndarray | None = None  # (n_ctx*PATCH_SIZE, N_INPUT_FEATURES) raw display values
        self.norm_stats: dict | None = None
        self.last_bg: float = 100.0           # last observed BG before prediction zone
        self.sim_start_hour: float = 0.0      # hour-of-day at first context timestep
        self.sim_start_day: int = 0            # day index at first context timestep
        # Live simulator instance — kept so the "Sim Fwd" action can advance
        # the ground-truth trajectory in place and append new patches to the
        # context. ``None`` until a patient is loaded.
        self.sim: Any = None

        # ---- Curve-editor channel selection ----
        # Display channel index being edited in curve editor mode.
        # 1=Carbs, 2=Insulin, 3=Exercise (BG is the model's forecast,
        # non-editable).
        self.selected_edit_channel: int = 1

        # ---- Chart view state ----
        self.view_start_patch: float = 0.0    # leftmost visible patch index
        self.view_end_patch: float = 100.0    # rightmost visible patch index
        self.y_scale_left: tuple[float, float] = (0.0, 400.0)   # (min, max) for left axis
        self.y_scale_right: tuple[float, float] = (0.5, 2.5)    # secondary axis

        # ---- Interaction state ----
        self.cursor_patch: float = -1.0       # cursor position in patch units
        self.is_computing: bool = False       # model inference in progress
        self.status_message: str = "Ready"
        self.mode_label: str = "Standard"
        self.active_band_label: str = ""

        # ---- Curve editor: 3-point curve events ----
        self.curve_events: list[CurveEvent] = []
        self.selected_event_idx: int = -1
        self.dragging_point: str | None = None  # 'peak', 'left', 'right', or None

        # ---- Pencil tool: freehand announced-dose strokes ----
        # Coexists with the raised-cosine curve editor: pencil strokes and
        # CurveEvents both compile to announced carb / insulin / exercise
        # and sum together (``gui._compile_overrides_from_edits``).
        self.pencil_strokes: list[PencilStroke] = []

        # ---- High-level intervention events (right panel) ----
        # User-friendly templates (juice, meal, bolus) that compile to
        # CurveEvents at prediction time via ``_events_to_curve_events``.
        self.events: list[Event] = []
        self.events_panel_visible: bool = True

        # ---- Basal insulin adjustment ----
        self.basal_rate_delta: float = 0.0    # U/h (can be negative)

        # ---- Meal/bolus builder state ----
        self.builder_time_patch: float = -1.0  # where user clicked for event time
        self.builder_params: dict[str, Any] = {}

        # ---- Last Eval-vs-Sim result ----
        # Populated by the Eval action. Lives independently of
        # ``prediction`` so it survives subsequent Predict / Sim Fwd /
        # Reset clicks; cleared on New Patient.
        self.last_eval: EvalResult | None = None

        # ---- Display smoothing toggles ----
        # Both are purely cosmetic — the underlying prediction tensors are
        # never modified. ``smooth_mu`` smooths the median BG forecast line;
        # ``smooth_band`` smooths the upper/lower edges of the quantile band.
        self.smooth_mu: bool = True
        self.smooth_band: bool = True

        # ---- Screenshot counter ----
        self.screenshot_count: int = 0

        # ---- Free-form masking ----
        # User-drawn masked spans over the CONTEXT, in absolute patch units. The
        # trailing forecast span is not in here — it is mandatory and appended at
        # emit time (see the module comment). Cleared on every context change:
        # Sim Fwd appends patches and New Patient replaces them, and a span kept
        # across either would mask different data than the user drew.
        self.mask_spans: list[MaskSpan] = []
        self.selected_mask_idx: int = -1       # index into mask_spans, -1 = the forecast
        self.mask_preset: str = MASK_PRESET_FORECAST
        # The loaded checkpoint's masked_channel_policy (data.stored_masked_channel_policy).
        # Under 'blind' the masked spans carry data.zero_dose_fill instead of the
        # recorded doses, and dose painting is disabled — the model was trained
        # with those channels pinned, so a painted override is invisible to it.
        self.masked_channel_policy: str = 'announced'

        # ---- Mask-drag state ----
        self.mask_drag_start: int = -1         # first patch of the in-flight drag
        self.mask_drag_end: int = -1           # last patch of the in-flight drag

    # ------------------------------------------------------------------
    # Override management
    # ------------------------------------------------------------------

    def set_override(self, channel: int, values: np.ndarray) -> None:
        """
        Set an announced-input override for a what-if output channel.

        Args:
            channel: Output channel index (0=carbs, 1=insulin, 2=exercise).
                     BG is the model's forecast and cannot be overridden.
            values: (P, PATCH_SIZE) array of normalized override values, where
                     P is PREDICTION_PATCHES (or n_rolls × PREDICTION_PATCHES
                     for an extended rolling forecast).
        """
        assert values.ndim == 2, f"Override must be 2D (patches, timesteps), got {values.shape}"
        self.overrides[channel] = values.copy()

    def clear_overrides(self) -> None:
        """Remove all channel overrides, curve events, pencil strokes, and
        basal; reset to standard."""
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
        """Return True if any channel has been overridden."""
        return len(self.overrides) > 0

    def has_edits(self) -> bool:
        """Return True if any curve events, pencil strokes, high-level events,
        or basal adjustment is active."""
        return (len(self.curve_events) > 0
                or len(self.pencil_strokes) > 0
                or len(self.events) > 0
                or self.basal_rate_delta != 0.0)

    # ------------------------------------------------------------------
    # Free-form masking
    # ------------------------------------------------------------------

    def n_ctx(self) -> int:
        """Context length in patches, 0 when no patient is loaded."""
        return int(self.context.shape[0]) if self.context is not None else 0

    def add_mask_span(self, start: int, length: int, n_pred: int) -> str:
        """Mask ``length`` patches from ``start``; return ``''`` or the refusal.

        Args:
            start: first patch to mask, absolute.
            length: patch count.
            n_pred: trailing forecast span length in patches.

        Returns:
            ``''`` on success, else why the set was left unchanged.
        """
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
        """Drop one user span by index; a no-op when the index is out of range."""
        if 0 <= idx < len(self.mask_spans):
            del self.mask_spans[idx]
            self.selected_mask_idx = -1
            self.mask_preset = (
                MASK_PRESET_FORECAST if not self.mask_spans else MASK_PRESET_CUSTOM
            )

    def clear_mask_spans(self) -> None:
        """Reset to the forecast preset — no user spans, nothing selected."""
        self.mask_spans = []
        self.selected_mask_idx = -1
        self.mask_preset = MASK_PRESET_FORECAST
        self.mask_drag_start = -1
        self.mask_drag_end = -1

    def apply_mask_preset(self, preset: str, n_pred: int) -> None:
        """Replace the user spans with a preset's."""
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
        """Context patches the user may still mask."""
        used = sum(s.length for s in merge_mask_spans(self.mask_spans))
        return max(0, user_mask_capacity(n_pred) - used)

    def selected_anchor_cell(self, n_pred: int) -> tuple[int, int]:
        """``(patch, step)`` of the SELECTED span's anchor.

        With nothing selected this is the trailing forecast span's anchor — the
        context edge, which is what the readout has always shown.
        """
        if 0 <= self.selected_mask_idx < len(self.mask_spans):
            s = self.mask_spans[self.selected_mask_idx]
            return span_anchor_cell(s.start, s.length)
        return span_anchor_cell(self.n_ctx(), int(n_pred))

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        """Activate a tool by name."""
        self.active_tool = tool
        self.mask_drag_start = -1
        self.mask_drag_end = -1
        if tool == TOOL_NONE:
            self.builder_time_patch = -1.0
            self.builder_params = {}

    def toggle_channel(self, channel: int) -> None:
        """Toggle visibility of a channel."""
        if 0 <= channel < len(self.channel_visible):
            self.channel_visible[channel] = not self.channel_visible[channel]

    def toggle_all_channels(self) -> None:
        """Toggle all channels: if any visible → hide all; if all hidden → show all."""
        if any(self.channel_visible):
            self.channel_visible = [False] * len(self.channel_visible)
        else:
            self.channel_visible = [True] * len(self.channel_visible)

    def cycle_edit_channel(self, delta: int = 1) -> None:
        """
        Cycle `selected_edit_channel` through paintable display channels
        (Carbs, Insulin, Exercise). BG is display channel 0, the model's
        forecast, and the only non-editable one — the rest are read off
        `channel_names` so a channel added to the table is reachable here.
        """
        editable = list(range(1, len(self.channel_names)))
        try:
            idx = editable.index(self.selected_edit_channel)
        except ValueError:
            idx = 0
        idx = (idx + delta) % len(editable)
        self.selected_edit_channel = editable[idx]

    # ------------------------------------------------------------------
    # Statistics computed from prediction
    # ------------------------------------------------------------------

    def get_predicted_tir(self) -> float | None:
        """
        Compute predicted time-in-range from the median BG forecast, over the
        fixed clinical band ``[BG_TARGET_LO, BG_TARGET_HI]`` (70–180 mg/dL).
        Returns a fraction in [0, 1], or None if no prediction is available.
        """
        if self.prediction.median_bg is None:
            return None
        bg = self.prediction.median_bg
        # 70.0 / 180.0 == BG_TARGET_LO / BG_TARGET_HI (train.py); the clinical
        # TIR band, kept literal here to avoid importing the training module.
        in_range = np.sum((bg >= 70.0) & (bg <= 180.0))
        return float(in_range / len(bg))

    def get_mean_uncertainty(self) -> float | None:
        """Return mean band half-width (mg/dL) across the forecast horizon, or
        None if no bands are available. The half-width is (q_high − q_low)/2
        using the outermost quantile pair."""
        bands = self.prediction.bands
        if bands is None or bands.shape[-1] < 2:
            return None
        half = (bands[..., -1] - bands[..., 0]) * 0.5
        return float(np.mean(half))

    def get_max_uncertainty(self) -> float | None:
        """Return max band half-width (mg/dL) over the forecast horizon."""
        bands = self.prediction.bands
        if bands is None or bands.shape[-1] < 2:
            return None
        half = (bands[..., -1] - bands[..., 0]) * 0.5
        return float(np.max(half))
