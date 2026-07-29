"""
AZT1D adapter — parses the 25-subject AZT1D CSV records into canonical Segments.

One CSV per subject (``Subject N/Subject N.csv``), 5-minute CGM with event columns
populated only when an event occurs:
    CGM                          mg/dL                CGM backbone
    CarbSize                     grams                carbohydrate ingestion
    TotalBolusInsulinDelivered   IU                   bolus (already includes correction)
    Basal                        IU/hour              AID basal rate (sparse; forward-filled)
    DeviceMode                   text                 normal/sleep/exercise (often empty)

These are AID (automated insulin delivery) records, so basal is algorithm-modulated
step to step.  ``TotalBolusInsulinDelivered`` subsumes ``CorrectionDelivered`` — do
not sum them.  A few corrupt ``Basal`` rows carry absurd values (≫ physiological),
clipped at ``BASAL_CLIP_IU_PER_H``.  Exercise is set to zero (see package note).
"""
from __future__ import annotations

import csv
import glob
import os
from datetime import datetime

import numpy as np

from .schema import GRID_MIN, Segment, segment_grid, lay_on_grid

_TS = "%Y-%m-%d %H:%M:%S"
BASAL_CLIP_IU_PER_H: float = 10.0       # physiological basal ceiling; guards corrupt rows


def _ts(s: str) -> datetime:
    return datetime.strptime(s.strip(), _TS)


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_csv(path: str, patient: str) -> list[Segment]:
    """Parse one AZT1D subject CSV into gap-split Segments."""
    rows = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            t = _num(row.get('CGM'))
            try:
                ts = _ts(row['EventDateTime'])
            except (KeyError, ValueError):
                continue
            rows.append((ts, row))
    rows = [r for r in rows if r[0] is not None]
    if len(rows) < 2:
        return []
    rows.sort(key=lambda r: r[0])
    base = rows[0][0]
    n = int(round((rows[-1][0] - base).total_seconds() / (GRID_MIN * 60))) + 1

    cgm = np.full(n, np.nan)
    carb_events, bolus_events, basal_events = [], [], []
    for ts, row in rows:
        idx = int(round((ts - base).total_seconds() / (GRID_MIN * 60)))
        cg = _num(row.get('CGM'))
        if cg is not None and 0 <= idx < n:
            cgm[idx] = cg
        carb = _num(row.get('CarbSize'))
        if carb:
            carb_events.append((ts, carb))
        bol = _num(row.get('TotalBolusInsulinDelivered'))
        if bol:
            bolus_events.append((ts, bol))
        bas = _num(row.get('Basal'))
        if bas is not None:
            basal_events.append((ts, min(bas, BASAL_CLIP_IU_PER_H)))

    carb_grams = lay_on_grid(base, n, carb_events)
    bolus_units = lay_on_grid(base, n, bolus_events)
    basal_rate = _piecewise(base, n, basal_events)
    exercise = np.zeros(n, dtype=np.float64)

    return segment_grid('azt1d', patient, base, cgm,
                        carb_grams, bolus_units, basal_rate, exercise)


def _piecewise(base: datetime, n: int, events: list[tuple[datetime, float]]) -> np.ndarray:
    """Forward-filled piecewise-constant rate over the grid from (ts, rate) events."""
    out = np.zeros(n, dtype=np.float64)
    if not events:
        return out
    events.sort(key=lambda x: x[0])
    times = np.array([(t - base).total_seconds() for t, _ in events])
    vals = np.array([v for _, v in events])
    step_secs = np.arange(n) * (GRID_MIN * 60)
    pos = np.clip(np.searchsorted(times, step_secs, side='right') - 1, 0, len(vals) - 1)
    return vals[pos]


def load(root_dir: str = '../T1DMSIM/datasets/AZT1D/CGM Records') -> list[Segment]:
    """Load every AZT1D subject CSV under ``root_dir`` into Segments."""
    segs: list[Segment] = []
    for path in sorted(glob.glob(os.path.join(root_dir, 'Subject *', '*.csv'))):
        patient = os.path.basename(os.path.dirname(path)).replace('Subject ', 'AZ')
        segs.extend(parse_csv(path, patient))
    return segs
