"""
ShanghaiT1DM adapter — parses the Shanghai_T1DM Excel records into Segments.

One workbook per patient-admission.  CGM is **already mg/dL** (header
``CGM (mg / dl)`` — the mmol/L trap does NOT apply here) at 15-minute cadence;
placing readings on the 5-minute grid leaves 10-minute holes that the schema's
≤30-min interpolation bridges, so no separate resampler is needed.

Insulin comes by two routes that coexist across the cohort:
  * CSII (pump) patients: numeric ``CSII - bolus insulin (Novolin R, IU)`` and
    ``CSII - basal insulin (Novolin R, IU / H)`` columns.
  * MDI patients: free-text ``Insulin dose - s.c.`` like ``"Novolin R, 6 IU"``.
    The short-acting "R"egulars (Novolin/Humulin/Gansulin R, aspart, lispro)
    are treated as boluses; long-acting analogues (glargine, detemir, degludec)
    are spread over 24 h into the basal rate.

Carbohydrate comes from the multi-line ``Dietary intake`` text, which lists FOOD
weights, not carb grams.  ``_CARB_FRACTION`` maps common foods (bilingual) to a
coarse carb-by-weight fraction; this is an approximation, the lowest-fidelity
channel in the whole harness, and is documented as such.  Several files are
``.xls``/``.xlsx`` interchangeably (extension unreliable) — readers are chosen by
magic bytes — and Excel lock files (``~$``) are skipped.
"""
from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timedelta

import numpy as np
import openpyxl
import xlrd

from .schema import GRID_MIN, Segment, segment_grid, lay_on_grid

_CGM_COL = 'CGM (mg / dl)'
_DIET_COL = 'Dietary intake'
_SC_COL = 'Insulin dose - s.c.'
_BOLUS_COL = 'CSII - bolus insulin (Novolin R, IU)'
_BASAL_COL = 'CSII - basal insulin (Novolin R, IU / H)'

# Long-acting (basal) insulin markers; everything else parsed from s.c. is bolus.
_LONG_ACTING = ('glargine', 'detemir', 'degludec', 'lantus', 'levemir', 'tresiba',
                'toujeo', '甘精', '地特', '德谷')

# Coarse carbohydrate-by-weight fractions for the recurring Shanghai foods.
# Approximate — food weight → carb grams; documented as low-fidelity.
_CARB_FRACTION: list[tuple[tuple[str, ...], float]] = [
    (('glucose', '葡萄糖'), 1.00),
    (('sugar', '蔗糖', '白糖'), 1.00),
    (('biscuit', 'cracker', '饼干'), 0.70),
    (('steamed bread', 'bread', 'mantou', '馒头', '面包'), 0.50),
    (('bun', 'baozi', '包子'), 0.40),
    (('noodle', 'noodles', '面条', '面'), 0.25),
    (('rice', 'congee', 'porridge', '米饭', '米', '粥'), 0.25),
    (('dumpling', 'jiaozi', '饺子', '馄饨'), 0.22),
    (('corn', '玉米'), 0.20),
    (('potato', '土豆', '红薯', '马铃薯'), 0.18),
    (('fruit', 'apple', 'banana', 'orange', 'pear', '水果', '苹果', '香蕉'), 0.12),
    (('soy', 'bean', 'tofu', '豆'), 0.10),
    (('milk', 'yogurt', '牛奶', '酸奶'), 0.05),
    (('vegetable', 'cabbage', '蔬菜', '青菜', '菜'), 0.04),
    (('egg', '鸡蛋', '蛋'), 0.01),
    (('meat', 'pork', 'beef', 'chicken', 'fish', '肉', '鱼', '鸡'), 0.00),
]
_DEFAULT_CARB_FRACTION = 0.15
_FOOD_RE = re.compile(r'([^\d\n]+?)\s*([\d.]+)\s*g', re.IGNORECASE)


def _sniff(path: str) -> str:
    with open(path, 'rb') as f:
        sig = f.read(8)
    if sig[:2] == b'PK':
        return 'xlsx'
    if sig[:4] == b'\xd0\xcf\x11\xe0':
        return 'xls'
    return 'unknown'


def _read(path: str) -> tuple[list, list[list], int]:
    """Return (header, rows, xls_datemode) choosing the reader by magic bytes."""
    kind = _sniff(path)
    if kind == 'xlsx':
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
        rs = list(ws.iter_rows(values_only=True))
        return list(rs[0]), [list(r) for r in rs[1:]], 0
    wb = xlrd.open_workbook(path)
    ws = wb.sheet_by_index(0)
    return ws.row_values(0), [ws.row_values(i) for i in range(1, ws.nrows)], wb.datemode


def _as_datetime(v, datemode: int) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if isinstance(v, (int, float)) and v > 0:
        return xlrd.xldate.xldate_as_datetime(v, datemode)
    return None


def _carbs_from_text(text: str) -> float:
    """Sum food-weight × carb-fraction over a multi-line dietary-intake cell."""
    total = 0.0
    for name, grams in _FOOD_RE.findall(text):
        low = name.lower()
        frac = next((f for keys, f in _CARB_FRACTION if any(k in low for k in keys)),
                    _DEFAULT_CARB_FRACTION)
        total += float(grams) * frac
    return total


def parse_workbook(path: str) -> list[Segment]:
    """Parse one Shanghai workbook into gap-split Segments."""
    header, rows, datemode = _read(path)
    H = {h: i for i, h in enumerate(header)}
    if _CGM_COL not in H:
        return []

    def cell(r, name):
        i = H.get(name)
        return r[i] if i is not None and i < len(r) else None

    recs = []
    for r in rows:
        ts = _as_datetime(cell(r, 'Date'), datemode)
        cg = cell(r, _CGM_COL)
        if ts is None:
            continue
        recs.append((ts, r, cg))
    recs = [x for x in recs if x[0] is not None]
    if len(recs) < 2:
        return []
    recs.sort(key=lambda x: x[0])
    base = recs[0][0]
    n = int(round((recs[-1][0] - base).total_seconds() / (GRID_MIN * 60))) + 1

    cgm = np.full(n, np.nan)
    carb_events, bolus_events, basal_pw, longact = [], [], [], []
    for ts, r, cg in recs:
        idx = int(round((ts - base).total_seconds() / (GRID_MIN * 60)))
        if isinstance(cg, (int, float)) and cg and 0 <= idx < n:
            cgm[idx] = float(cg)
        diet = cell(r, _DIET_COL)
        if isinstance(diet, str) and diet.strip() not in ('', 'data not available', '未记录', '/'):
            g = _carbs_from_text(diet)
            if g > 0:
                carb_events.append((ts, g))
        b = cell(r, _BOLUS_COL)
        if isinstance(b, (int, float)) and b:
            bolus_events.append((ts, float(b)))
        ba = cell(r, _BASAL_COL)
        if isinstance(ba, (int, float)) and ba:
            basal_pw.append((ts, float(ba)))
        sc = cell(r, _SC_COL)
        if isinstance(sc, str) and sc.strip():
            for dose_ts, dose, is_long in _parse_sc(ts, sc):
                (longact if is_long else bolus_events).append((dose_ts, dose))

    carb_grams = lay_on_grid(base, n, carb_events)
    bolus_units = lay_on_grid(base, n, bolus_events)
    basal_rate = _basal_rate(base, n, basal_pw, longact)
    exercise = np.zeros(n, dtype=np.float64)

    pid = os.path.basename(path).split('_')[0]
    return segment_grid('shanghai', pid, base, cgm,
                        carb_grams, bolus_units, basal_rate, exercise)


def _parse_sc(ts: datetime, text: str) -> list[tuple[datetime, float, bool]]:
    """Parse an ``Insulin dose - s.c.`` cell into (ts, IU, is_long_acting) entries."""
    out = []
    for chunk in re.split(r'[;\n]', text):
        m = re.search(r'([\d.]+)\s*IU', chunk, re.IGNORECASE)
        if not m:
            continue
        dose = float(m.group(1))
        low = chunk.lower()
        is_long = any(k in low for k in _LONG_ACTING)
        out.append((ts, dose, is_long))
    return out


def _basal_rate(base: datetime, n: int, csii: list[tuple[datetime, float]],
                longact: list[tuple[datetime, float]]) -> np.ndarray:
    """CSII basal (piecewise IU/h) plus MDI long-acting doses spread over 24 h."""
    out = np.zeros(n, dtype=np.float64)
    if csii:
        csii.sort(key=lambda x: x[0])
        times = np.array([(t - base).total_seconds() for t, _ in csii])
        vals = np.array([v for _, v in csii])
        step_secs = np.arange(n) * (GRID_MIN * 60)
        pos = np.clip(np.searchsorted(times, step_secs, side='right') - 1, 0, len(vals) - 1)
        out = vals[pos].astype(np.float64)
    # Long-acting analogue: each dose contributes dose/24 IU/h for 24 h from injection.
    span = 24 * 60 // GRID_MIN
    for ts, dose in longact:
        lo = int(round((ts - base).total_seconds() / (GRID_MIN * 60)))
        out[max(0, lo):min(n, lo + span)] += dose / 24.0
    return out


def load(root_dir: str = '../T1DMSIM/datasets/ShanghaiT1DM/Shanghai_T1DM') -> list[Segment]:
    """Load every ShanghaiT1DM workbook under ``root_dir`` (skipping ~$ lock files)."""
    segs: list[Segment] = []
    for path in sorted(glob.glob(os.path.join(root_dir, '*.xls*'))):
        if os.path.basename(path).startswith('~$'):
            continue
        segs.extend(parse_workbook(path))
    return segs
