"""Canonical record every evaluation source parses into.

A Segment is a contiguous, gap-free, 5-min-grid stretch of RAW events (see field comments
for units). exercise is g/step already; convert other sources before this field.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

GRID_MIN: int = 5                       # minutes
STEPS_PER_HOUR: int = 60 // GRID_MIN
MGDL_PER_MMOL: float = 18.0156          # glucose mg/dL per mmol/L
MAX_INTERP_GAP_MIN: int = 30            # CGM gaps ≤ this are linearly interpolated; beyond → split
MIN_SEGMENT_STEPS: int = 60             # drop runt segments shorter than 5 h (60 × 5 min)


@dataclass
class Segment:
    """A contiguous 5-minute-grid stretch of one patient's record.

    All arrays share length N, aligned to a uniform grid from t0; cgm is finite everywhere.
    carb_curve/insulin_curve: supplied together or both None; preferred over the kernel default.
    """
    dataset: str
    patient: str
    t0: datetime
    cgm: np.ndarray            # (N,) mg/dL
    carb_grams: np.ndarray     # (N,) grams ingested in this step
    bolus_units: np.ndarray    # (N,) bolus IU delivered in this step
    basal_rate: np.ndarray     # (N,) basal IU/hour, piecewise-constant
    exercise: np.ndarray       # (N,) g/step carb-equivalent disposal (0 if unavailable)
    split: str = ''            # 'training' | 'testing' | '' — canonical-protocol origin
    carb_curve: np.ndarray | None = None      # (N,) g/step appearance, pre-resolved
    insulin_curve: np.ndarray | None = None   # (N,) IU/step action, pre-resolved (basal+bolus)

    def __post_init__(self) -> None:
        n = len(self.cgm)
        for name in ('carb_grams', 'bolus_units', 'basal_rate', 'exercise'):
            arr = getattr(self, name)
            assert arr.shape == (n,), f"{name} length {arr.shape} != cgm length {n}"
        assert (self.carb_curve is None) == (self.insulin_curve is None), (
            "carb_curve and insulin_curve must be supplied together or both left None"
        )
        for name in ('carb_curve', 'insulin_curve'):
            arr = getattr(self, name)
            assert arr is None or arr.shape == (n,), (
                f"{name} length {None if arr is None else arr.shape} != cgm length {n}"
            )
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
    carb_curve: np.ndarray | None = None,
    insulin_curve: np.ndarray | None = None,
) -> list[Segment]:
    """Split a full uniform grid into gap-free Segments, each ≥ ``MIN_SEGMENT_STEPS`` long.

    ``cgm`` (M,) mg/dL, NaN at missing steps: runs ≤ ``MAX_INTERP_GAP_MIN`` interpolate, longer
    ones break the record. ``grid_t0`` is grid index 0's time; event channels are already gridded.
    """
    m = len(cgm)
    assert all(len(a) == m for a in (carb_grams, bolus_units, basal_rate, exercise))
    assert (carb_curve is None) == (insulin_curve is None), (
        "carb_curve and insulin_curve must be supplied together or both left None"
    )
    if carb_curve is not None and insulin_curve is not None:
        assert len(carb_curve) == m and len(insulin_curve) == m, (
            "pre-resolved curve length does not match the grid"
        )
    max_gap = MAX_INTERP_GAP_MIN // GRID_MIN          # steps

    finite = np.isfinite(cgm)
    if not finite.any():
        return []

    # trim leading/trailing NaN
    first, last = int(np.argmax(finite)), m - int(np.argmax(finite[::-1]))
    cgm = cgm[first:last].copy()
    named = [('carb_grams', carb_grams), ('bolus_units', bolus_units),
             ('basal_rate', basal_rate), ('exercise', exercise)]
    if carb_curve is not None:
        named += [('carb_curve', carb_curve), ('insulin_curve', insulin_curve)]
    chans = {k: v[first:last].copy() for k, v in named}
    seg_t0 = grid_t0 + timedelta(minutes=GRID_MIN * first)
    finite = np.isfinite(cgm)

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
        gap_len = j - i                # steps
        if i > 0 and j < n and gap_len <= max_gap:
            lo, hi = cgm[i - 1], cgm[j]
            for k in range(i, j):
                cgm[k] = lo + (hi - lo) * (k - i + 1) / (gap_len + 1)
        elif i > 0:
            split_after.append(i - 1)
        i = j

    bounds = [0] + [s + 1 for s in split_after] + [n]
    segments: list[Segment] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        # an unbridged boundary gap can leave non-finite ends
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
            carb_curve=(None if carb_curve is None else chans['carb_curve'][lo:hi]),
            insulin_curve=(None if carb_curve is None else chans['insulin_curve'][lo:hi]),
        ))
    return segments


def lay_on_grid(grid_t0: datetime, n_steps: int, events: list[tuple[datetime, float]]) -> np.ndarray:
    """Bin point events ``(timestamp, amount)`` into a ``(n_steps,)`` grid.

    Nearest-index ``round``, the same placement every adapter uses for CGM, so an event and its
    CGM sample share a cell. Events outside ``[grid_t0, grid_t0 + n_steps·GRID_MIN)`` are dropped.
    """
    out = np.zeros(n_steps, dtype=np.float64)
    for ts, amt in events:
        idx = int(round((ts - grid_t0).total_seconds() / (GRID_MIN * 60)))
        if 0 <= idx < n_steps:
            out[idx] += amt
    return out
