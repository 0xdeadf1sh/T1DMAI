"""Convert a T1DMDROID ``.t1dmbak`` backup into a finetune cache, its last N days held out as test.

Pool ``phone`` in ``finetune_data``'s format. BG: measured, NORMAL-flagged grid samples. Carb and
insulin: rebuilt from logged events per ``SPEC/invariants.md`` §5, custom curves verbatim. Exercise:
the grams the phone laid into each bucket.

The grid carries only the sensor that was authoritative at the time. Every sensor's own NORMAL
readings form one more subject each, over that sensor's span cut at the held-out boundary, so the
grid alone is scored; overlapping sensors are separate subjects, never merged."""

import argparse
import base64
import gzip
import json
import os

import numpy as np

import finetune_data as fd
from config import MIN_CONTEXT_PATCHES, PATCH_SIZE, PREDICTION_PATCHES
from T1DMSIM.simulator import basal_curve, gamma_curve

ARCHIVE_FORMAT = 't1dm.archive'
STEP_MS = fd.STEP_S * 1000
STEPS_PER_DAY = 24 * 60 // fd.DT_MINUTES
TEST_DAYS = 7


def held_out_start(n: int, test_days: int) -> int:
    """First held-out step of an ``n``-step record; negative when the record is shorter."""
    return n - test_days * STEPS_PER_DAY


def read_archive(path: str) -> dict[str, list[dict]]:
    """Records by kind. A file without its end record is truncated and refused."""
    kinds: dict[str, list[dict]] = {}
    with gzip.open(path, 'rt') as f:
        header = json.loads(f.readline())
        if header.get('format') != ARCHIVE_FORMAT:
            raise SystemExit(f'{path}: format {header.get("format")!r}, not {ARCHIVE_FORMAT!r}')
        for line in f:
            o = json.loads(line)
            kinds.setdefault(o.get('t'), []).append(o)
    if 'end' not in kinds:
        raise SystemExit(f'{path}: no end record, the backup is truncated')
    return kinds


def custom_curve(o: dict) -> np.ndarray | None:
    """The row's ``customCurve``: little-endian f64 per 5-min step."""
    return None if 'cc' not in o else np.frombuffer(base64.b64decode(o['cc']), dtype='<f8')


def meal_curve(o: dict) -> np.ndarray:
    cc = custom_curve(o)
    if cc is not None:
        return cc
    k, theta, _ = fd.gi_gamma_params(float(o.get('gi', fd.CARB_GI_DEFAULT)))
    return gamma_curve(float(o['g']), float(o.get('k', k)), float(o.get('th', theta)),
                       float(o['dm']))


def dose_curve(o: dict) -> np.ndarray:
    cc = custom_curve(o)
    if cc is not None:
        return cc
    if o['kd'] == 'BASAL':
        rates = {name: float(o[key]) for key, name in (('ka', 'ka_per_hour'), ('ke', 'ke_per_hour'))
                 if key in o}
        return basal_curve(float(o['u']), float(o['dm']), **rates)
    if 'k' in o and 'th' in o:
        return gamma_curve(float(o['u']), float(o['k']), float(o['th']), float(o['dm']))
    # The phone's exponential fallback exists only in t1dm-core; skipping would drop insulin.
    raise SystemExit(f'bolus {o["cid"]} has neither a custom curve nor gamma parameters')


def live_events(kinds: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], int]:
    """Meals and doses a tombstone has not retired, and how many it did."""
    retired = {t['cid']: t['ua'] for t in kinds.get('tombstone', [])}
    events = {kind: [o for o in kinds.get(kind, [])
                     if o['cid'] not in retired or o['ua'] > retired[o['cid']]]
              for kind in ('meal', 'dose')}
    return events, sum(len(kinds.get(k, [])) - len(v) for k, v in events.items())


def sensor_traces(kinds: dict[str, list[dict]], t0_ms: int, n: int) -> list[tuple[str, np.ndarray]]:
    """Per sensor, ordered by first reading: bg mg/dL per step (NaN unmeasured), newest rx wins."""
    traces: dict[str, np.ndarray] = {}
    for r in sorted(kinds.get('reading', []), key=lambda r: r['rx']):
        if r.get('pv') != 'MEASURED' or r.get('fl') != 'NORMAL':
            continue
        i = int(round((r['ts'] - t0_ms) / STEP_MS))
        if 0 <= i < n:
            traces.setdefault(r['s'], np.full(n, np.nan, dtype=np.float32))[i] = r['bg']
    return sorted(traces.items(), key=lambda kv: int(np.flatnonzero(np.isfinite(kv[1]))[0]))


def _lay(dst: np.ndarray, curve: np.ndarray, start: int) -> None:
    """Add ``curve`` from grid step ``start``; steps outside the grid fall away, as on the phone."""
    lo, hi = max(0, -start), min(len(curve), len(dst) - start)
    if hi > lo:
        dst[start + lo:start + hi] += curve[lo:hi]


def record_channels(kinds: dict[str, list[dict]]) -> dict:
    """Per 5-min step: bg mg/dL (NaN unmeasured), carb g, insulin U, exercise g."""
    samples = kinds.get('sample', [])
    if not samples:
        raise SystemExit('the backup has no samples')
    events, n_deleted = live_events(kinds)
    first = min(samples, key=lambda s: s['ts'])
    t0_ms = first['ts']
    n = (max(s['ts'] for s in samples) - t0_ms) // STEP_MS + 1
    bg = np.full(n, np.nan, dtype=np.float32)
    exercise = np.zeros(n, dtype=np.float64)
    for s in samples:
        i = (s['ts'] - t0_ms) // STEP_MS
        if 'bg' in s and s.get('pv') == 'MEASURED' and s.get('fl') == 'NORMAL':
            bg[i] = s['bg']
        exercise[i] = float(s.get('exg', 0.0))

    carb, insulin = np.zeros(n), np.zeros(n)
    at = lambda ts: int(round((ts - t0_ms) / STEP_MS))  # noqa: E731 — bucketize's rounding
    for o in events['meal']:
        _lay(carb, meal_curve(o), at(o['ts']))
    for o in events['dose']:
        _lay(insulin, dose_curve(o), at(o['ts']))
    return {'t0_ms': t0_ms, 'tz_min': int(first['tz']), 'bg': bg, 'carb': carb,
            'insulin': insulin, 'exercise': np.nan_to_num(exercise),
            'n_meals': len(events['meal']), 'n_doses': len(events['dose']), 'n_deleted': n_deleted}


def convert(path: str, out_dir: str, test_days: int, source: str, sid: str,
            grid_only: bool = False) -> None:
    if test_days < 1:
        raise SystemExit('--test-days must be at least 1')
    kinds = read_archive(path)
    r = record_channels(kinds)
    n = len(r['bg'])
    test_start = held_out_start(n, test_days)
    min_train = (MIN_CONTEXT_PATCHES + PREDICTION_PATCHES) * PATCH_SIZE
    if test_start < min_train:
        raise SystemExit(f'{test_days} test days leave {max(test_start, 0)} train steps; '
                         f'a window needs {min_train}')
    is_test = np.zeros(n, dtype=np.uint8)
    is_test[test_start:] = 1

    app = fd._ChannelAppender(out_dir)
    index: list[dict] = []
    # Local wall-clock seconds: the time probe reads the hour straight off t0.
    t0 = r['t0_ms'] // 1000 + r['tz_min'] * 60
    carb, insulin = r['carb'].astype(np.float32), r['insulin'].astype(np.float32)
    exercise = r['exercise'].astype(np.float32)

    def add(src: str, subject: str, bg: np.ndarray, lo: int, hi: int) -> None:
        start = app.append({'bg': bg[lo:hi], 'carb': carb[lo:hi], 'insulin': insulin[lo:hi],
                            'exercise': exercise[lo:hi], 'is_test': is_test[lo:hi]})
        ts = test_start - lo
        index.append({
            'key': f'phone:{src}:{subject}', 'pool': 'phone', 'source': src, 'sid': subject,
            'start': start, 'n': int(hi - lo), 't0': t0 + lo * fd.STEP_S,
            'test_start': -1 if ts >= hi - lo else max(int(ts), 0),
        })

    add(source, sid, r['bg'], 0, n)
    print(f'{path}: {n / STEPS_PER_DAY:.1f} days, {int(np.isfinite(r["bg"]).sum())} measured BG '
          f'over {n} steps; {r["n_meals"]} meals, {r["n_doses"]} doses, {r["n_deleted"]} deleted; '
          f'train {test_start} steps, test {n - test_start} ({test_days} days)', flush=True)
    if not grid_only:
        for k, (sensor, bg) in enumerate(sensor_traces(kinds, r['t0_ms'], n), 1):
            fin = np.flatnonzero(np.isfinite(bg))
            lo, hi = int(fin[0]), min(int(fin[-1]) + 1, test_start)
            if hi <= lo:
                print(f'  {sid}-s{k}: {sensor}, held-out days only, skipped', flush=True)
                continue
            add(source, f'{sid}-s{k}', bg, lo, hi)
            print(f'  {sid}-s{k}: {sensor}, steps {lo}-{hi}, '
                  f'{int(np.isfinite(bg[lo:hi]).sum())} measured', flush=True)
    fd.finish_cache(app, index, out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('backup', help='.t1dmbak file exported by T1DMDROID')
    p.add_argument('--test-days', type=int, default=TEST_DAYS,
                   help='trailing days held out as test')
    p.add_argument('--out', default=None, help='cache directory; default datasets/t1dmdroid/<stem>')
    p.add_argument('--source', default='t1dmdroid', help='sub-dataset name in the cache')
    p.add_argument('--sid', default=None, help='subject id; default the backup file stem')
    p.add_argument('--grid-only', action='store_true',
                   help='skip the per-sensor subjects; the authoritative grid alone')
    a = p.parse_args()
    stem = os.path.splitext(os.path.basename(a.backup))[0]
    convert(a.backup, a.out or os.path.join('datasets', 't1dmdroid', stem), a.test_days,
            a.source, a.sid or stem, a.grid_only)


if __name__ == '__main__':
    main()
