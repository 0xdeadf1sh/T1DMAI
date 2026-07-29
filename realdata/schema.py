"""
Canonical real-data representation shared by every dataset adapter.

Each adapter parses its source (OhioT1DM XML, AZT1D CSV, ShanghaiT1DM Excel) into
a list of :class:`Segment` — a contiguous, gap-free, 5-minute-grid stretch of one
patient's record carrying the *raw events* the model's input bridge needs:

    cgm          mg/dL          the CGM backbone (already gap-interpolated/split)
    carb_grams   grams/step     meal carbohydrate ingested in that 5-min step
    bolus_units  IU/step        discrete bolus insulin delivered in that step
    basal_rate   IU/hour        piecewise-constant basal rate (forward-filled)
    exercise     intensity      activity proxy (0 where the source has none)

Why raw events and not the model's channels?  The model consumes *absorption /
action curves* (the simulator's gamma carb-absorption and first-order basal +
gamma bolus-action, EMA-smoothed), NOT ingestion/injection spikes.  Converting
these raw events into those curves is the model-input bridge's job; it
needs basal and bolus kept *separate* (different kernels), which is why this
schema preserves them rather than pre-summing an ``insulin_combined`` channel.

All BG is in mg/dL.  Adapters that read a mmol/L source must convert with
``MGDL_PER_MMOL`` before constructing a Segment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

GRID_MIN: int = 5                       # resample grid resolution (minutes)
STEPS_PER_HOUR: int = 60 // GRID_MIN
MGDL_PER_MMOL: float = 18.0156          # glucose mg/dL per mmol/L
MAX_INTERP_GAP_MIN: int = 30            # CGM gaps ≤ this are linearly interpolated; beyond → split
MIN_SEGMENT_STEPS: int = 60             # drop runt segments shorter than 5 h (60 × 5 min)


@dataclass
class Segment:
    """A contiguous 5-minute-grid stretch of one patient's record.

    All arrays share length ``N`` and are aligned to a uniform grid starting at
    ``t0`` with ``GRID_MIN``-minute spacing.  ``cgm`` is finite everywhere
    (small gaps interpolated, large gaps split into separate Segments).
    """
    dataset: str
    patient: str
    t0: datetime
    cgm: np.ndarray            # (N,) mg/dL
    carb_grams: np.ndarray     # (N,) grams ingested in this step
    bolus_units: np.ndarray    # (N,) bolus IU delivered in this step
    basal_rate: np.ndarray     # (N,) basal IU/hour, piecewise-constant
    exercise: np.ndarray       # (N,) activity proxy (0 if unavailable)
    split: str = ''            # 'training' | 'testing' | '' — canonical-protocol origin

    def __post_init__(self) -> None:
        n = len(self.cgm)
        for name in ('carb_grams', 'bolus_units', 'basal_rate', 'exercise'):
            arr = getattr(self, name)
            assert arr.shape == (n,), f"{name} length {arr.shape} != cgm length {n}"
        assert np.isfinite(self.cgm).all(), "cgm must be gap-free (interpolate/split first)"

    def __len__(self) -> int:
        return len(self.cgm)

    def timestamps(self) -> list[datetime]:
        """Per-step wall-clock timestamps (length N)."""
        return [self.t0 + timedelta(minutes=GRID_MIN * i) for i in range(len(self))]

    def hour_of_day(self) -> np.ndarray:
        """Fractional hour-of-day in [0, 24) for each step (the time-of-day probe target)."""
        base = self.t0.hour + self.t0.minute / 60.0 + self.t0.second / 3600.0
        return (base + np.arange(len(self)) * (GRID_MIN / 60.0)) % 24.0


def segment_grid(
    dataset: str,
    patient: str,
    grid_t0: datetime,
    cgm: np.ndarray,
    carb_grams: np.ndarray,
    bolus_units: np.ndarray,
    basal_rate: np.ndarray,
    exercise: np.ndarray,
) -> list[Segment]:
    """Split a full uniform grid into gap-free Segments.

    ``cgm`` may contain NaN where the source had no reading.  Runs of NaN whose
    span is ≤ ``MAX_INTERP_GAP_MIN`` are linearly interpolated and kept inline;
    longer runs break the record into separate Segments.  Event channels
    (carbs/bolus/basal/exercise) are assumed already laid onto the same grid and
    are simply sliced per Segment.

    Args:
        dataset, patient: provenance labels.
        grid_t0: timestamp of grid index 0.
        cgm: (M,) mg/dL with NaN at missing steps.
        carb_grams, bolus_units, basal_rate, exercise: (M,) event channels.

    Returns:
        List of Segments (possibly empty), each ≥ ``MIN_SEGMENT_STEPS`` long.
    """
    m = len(cgm)
    assert all(len(a) == m for a in (carb_grams, bolus_units, basal_rate, exercise))
    max_gap = MAX_INTERP_GAP_MIN // GRID_MIN          # in steps

    finite = np.isfinite(cgm)
    if not finite.any():
        return []

    # Boundaries of valid coverage: trim leading/trailing NaN.
    first, last = int(np.argmax(finite)), m - int(np.argmax(finite[::-1]))
    cgm = cgm[first:last].copy()
    chans = {k: v[first:last].copy() for k, v in (
        ('carb_grams', carb_grams), ('bolus_units', bolus_units),
        ('basal_rate', basal_rate), ('exercise', exercise))}
    seg_t0 = grid_t0 + timedelta(minutes=GRID_MIN * first)
    finite = np.isfinite(cgm)

    # Find NaN runs; mark long ones as split points, interpolate short ones.
    split_after: list[int] = []        # last good index before an un-bridgeable gap
    i = 0
    n = len(cgm)
    while i < n:
        if finite[i]:
            i += 1
            continue
        j = i
        while j < n and not finite[j]:
            j += 1
        gap_len = j - i                # number of consecutive missing steps
        if i > 0 and j < n and gap_len <= max_gap:
            lo, hi = cgm[i - 1], cgm[j]
            for k in range(i, j):
                cgm[k] = lo + (hi - lo) * (k - i + 1) / (gap_len + 1)
        elif i > 0:
            split_after.append(i - 1)  # break before this gap
        i = j

    # Cut points -> contiguous index ranges of finite, bridged CGM.
    bounds = [0] + [s + 1 for s in split_after] + [n]
    segments: list[Segment] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        # Trim any trailing/leading non-finite left by an unbridged boundary gap.
        sl = slice(a, b)
        sub = cgm[sl]
        good = np.isfinite(sub)
        if not good.any():
            continue
        lo = a + int(np.argmax(good))
        hi = b - int(np.argmax(good[::-1]))
        if hi - lo < MIN_SEGMENT_STEPS or not np.isfinite(cgm[lo:hi]).all():
            continue
        segments.append(Segment(
            dataset=dataset, patient=patient,
            t0=seg_t0 + timedelta(minutes=GRID_MIN * lo),
            cgm=cgm[lo:hi],
            carb_grams=chans['carb_grams'][lo:hi],
            bolus_units=chans['bolus_units'][lo:hi],
            basal_rate=chans['basal_rate'][lo:hi],
            exercise=chans['exercise'][lo:hi],
        ))
    return segments


def lay_on_grid(grid_t0: datetime, n_steps: int, events: list[tuple[datetime, float]]) -> np.ndarray:
    """Accumulate point events (timestamp, amount) into a (n_steps,) grid by binning.

    Each event's amount is added to the step whose grid time is *nearest* the
    event timestamp (nearest-index ``round``), matching the convention every
    adapter uses for CGM placement so an event and its aligned CGM sample land in
    the same cell.  Events outside [grid_t0, grid_t0 + n_steps·GRID_MIN) are dropped.
    """
    out = np.zeros(n_steps, dtype=np.float64)
    for ts, amt in events:
        idx = int(round((ts - grid_t0).total_seconds() / (GRID_MIN * 60)))
        if 0 <= idx < n_steps:
            out[idx] += amt
    return out
