"""
OhioT1DM adapter — parses the 2018/2020 OhioT1DM XML into canonical Segments.

The XML nests one section per channel, each a list of ``<event>`` elements:
    glucose_level  ts, value (mg/dL)              the 5-min CGM backbone
    meal           ts, carbs (grams)              carbohydrate ingestion
    bolus          ts_begin, dose (IU)            discrete bolus
    basal          ts, value (IU/hour)            scheduled basal rate (piecewise)
    temp_basal     ts_begin, ts_end, value        temporary basal override
Timestamps are ``dd-mm-yyyy HH:MM:SS`` local time.  CGM is already mg/dL.

The canonical set ships train+test XML per patient; both are parsed (the split is
applied later, at evaluation time, not here).  Exercise is set to zero (see the
package note — no faithful magnitude in simulator units).
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import numpy as np

from .schema import GRID_MIN, Segment, segment_grid, lay_on_grid

_TS = "%d-%m-%Y %H:%M:%S"


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s.strip(), _TS)


def _events(root: ET.Element, section: str) -> list[ET.Element]:
    node = root.find(section)
    return list(node.findall('event')) if node is not None else []


def _step_index(ts: datetime, base: datetime) -> int:
    """Nearest 5-min grid index of ``ts`` relative to ``base``."""
    return int(round((ts - base).total_seconds() / (GRID_MIN * 60)))


def parse_xml(path: str) -> list[Segment]:
    """Parse one OhioT1DM XML file into gap-split Segments."""
    root = ET.parse(path).getroot()
    pid = root.get('id', os.path.basename(path).split('-')[0])

    glucose = [(_parse_ts(e.get('ts')), float(e.get('value')))
               for e in _events(root, 'glucose_level') if e.get('value')]
    if len(glucose) < 2:
        return []
    glucose.sort(key=lambda x: x[0])
    base = glucose[0][0]
    n = _step_index(glucose[-1][0], base) + 1

    cgm = np.full(n, np.nan)
    for ts, val in glucose:
        idx = _step_index(ts, base)
        if 0 <= idx < n:
            cgm[idx] = val

    carb_events = [(_parse_ts(e.get('ts')), float(e.get('carbs')))
                   for e in _events(root, 'meal') if e.get('carbs')]
    bolus_events = [(_parse_ts(e.get('ts_begin')), float(e.get('dose')))
                    for e in _events(root, 'bolus') if e.get('dose')]
    carb_grams = lay_on_grid(base, n, carb_events)
    bolus_units = lay_on_grid(base, n, bolus_events)

    basal_rate = _build_basal(root, base, n)
    exercise = np.zeros(n, dtype=np.float64)

    split = ('testing' if 'test' in os.path.basename(path).lower()
             else 'training' if 'train' in os.path.basename(path).lower() else '')
    segs = segment_grid('ohiot1dm', pid, base, cgm,
                        carb_grams, bolus_units, basal_rate, exercise)
    for s in segs:
        s.split = split
    return segs


def _build_basal(root: ET.Element, base: datetime, n: int) -> np.ndarray:
    """Piecewise-constant basal rate (IU/h) over the grid, with temp-basal overrides."""
    basal = [(_parse_ts(e.get('ts')), float(e.get('value')))
             for e in _events(root, 'basal') if e.get('value')]
    out = np.zeros(n, dtype=np.float64)
    if basal:
        basal.sort(key=lambda x: x[0])
        times = np.array([(t - base).total_seconds() for t, _ in basal])
        vals = np.array([v for _, v in basal])
        step_secs = np.arange(n) * (GRID_MIN * 60)
        pos = np.searchsorted(times, step_secs, side='right') - 1
        pos = np.clip(pos, 0, len(vals) - 1)          # before first event → first rate
        out = vals[pos]

    for e in _events(root, 'temp_basal'):
        if not (e.get('ts_begin') and e.get('ts_end') and e.get('value')):
            continue
        lo = _step_index(_parse_ts(e.get('ts_begin')), base)
        hi = _step_index(_parse_ts(e.get('ts_end')), base)
        lo, hi = max(0, lo), min(n, hi + 1)
        if lo < hi:
            out[lo:hi] = float(e.get('value'))
    return out


def load(root_dir: str = '../T1DMSIM/datasets/ohiot1dm') -> list[Segment]:
    """Load every OhioT1DM XML (train + test) under ``root_dir`` into Segments."""
    segs: list[Segment] = []
    for path in sorted(glob.glob(os.path.join(root_dir, '*.xml'))):
        segs.extend(parse_xml(path))
    return segs
