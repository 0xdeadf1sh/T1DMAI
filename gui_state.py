"""
T1DMAI GUI State Management — central mutable state container.
================================================================

This module centralizes everything the GUI needs to remember between frames:

* What's visible (per-channel toggles, all-channels toggle).
* Which tool is active (curve editor, meal builder, bolus builder, …).
* What overrides the user has painted into the prediction zone.
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


# === Tool identifiers ===
TOOL_NONE = 'none'
TOOL_CURVE_EDITOR = 'curve_editor'
TOOL_PENCIL = 'pencil'
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
    # Tool management
    # ------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        """Activate a tool by name."""
        self.active_tool = tool
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
