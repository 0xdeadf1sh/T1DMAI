"""MetaboNet + DiaData merge into a finetuning cache; gap-aware datasets over it.

``python finetune_data.py build`` writes bg/carb/insulin/exercise channels via the
event-to-curve rules of ``T1DMCOMMON/SPEC/invariants.md`` §5; a gap masks the head
slot, never the loss. ``_normalize_features`` mirrors ``data._build_sample``'s transform."""

import argparse
import io
import json
import os
import zipfile
from typing import Any

import numpy as np
import torch

# Env-patches config before ``from config import``: forkserver workers reimport this module fresh.
import config as _config
_PATCH_ENV = 'T1DMAI_FINETUNE_CONFIG_PATCH'
if os.environ.get(_PATCH_ENV):
    for _k, _v in json.loads(os.environ[_PATCH_ENV]).items():
        setattr(_config, _k, tuple(_v) if isinstance(_v, list) else _v)

from config import (
    PATCH_SIZE, N_INPUT_FEATURES, PATCH_DIM, PREDICTION_PATCHES,
    MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES, MAX_MASKED_PATCHES,
    NON_MASKABLE_FEATS,
)
from normalization import CHANNEL_NAMES, RISK_SPACE_CHANNELS, SPARSE_LOG1P_CHANNELS
from data import sample_mask_spans, _mask_slots, BG_MASKED_FEAT
import utils
from utils import kovatchev_f_np
from T1DMSIM.simulator import (
    DT_MINUTES, BG_CLAMP_MIN, BG_CLAMP_MAX,
    gamma_curve, basal_curve, bolus_pk_for_dose,
    BOLUS_VARIANTS, BASAL_VARIANTS,
    EXERCISE_GAMMA_K, EXERCISE_GAMMA_THETA, EXERCISE_CARB_EQUIV_PER_MIN,
)

STEP_S = DT_MINUTES * 60
CACHE_CHANNELS = ('bg', 'carb', 'insulin', 'exercise', 'is_test')
CHANNEL_DTYPES = {c: np.float32 for c in CACHE_CHANNELS} | {'is_test': np.uint8}
CACHE_VERSION = 'finetune-cache-v2'

# Carb entry error above this, not a meal (train.parquet max 855 g).
CARB_EVENT_MAX_G = 300.0
# Workout row above this isn't a session (max 8218 min; unbounded peaks at 367 g/step, z=+32).
EXERCISE_SESSION_MAX_MIN = 240.0
# Bouts within this of the prior bout's end join one session, else disposal runs 40x too high.
EXERCISE_SESSION_JOIN_MIN = 30.0
# Below this, an MDI-labeled row is a per-slot pump rate; at/above, a long-acting injection.
MDI_INJECTION_MIN_U = 1.5

# No residual IOB in the first two days (empty pre-record history); windows start past them.
LEAD_IN_STEPS = 48 * 60 // DT_MINUTES

EVAL_GAP_BUDGET = 0.2  # gap-patch cap over the trailing MIN_CONTEXT_PATCHES

# 30/60/90/120 min = the challenge horizons; step index within the forecast zone.
HORIZON_MINUTES = (30, 60, 90, 120)
HORIZON_STEPS = tuple(m // (DT_MINUTES) - 1 for m in HORIZON_MINUTES)
assert PREDICTION_PATCHES * PATCH_SIZE * DT_MINUTES == HORIZON_MINUTES[-1], (
    "the forecast zone must span exactly 120 min for the challenge metrics"
)

# DiaData Database labels duplicating MetaboNet source_files.
DIADATA_DROP_SOURCES = frozenset({'RBG', 'DLCP3', 'PEDAP', 'HUPA-UCM', 'ShanghaiT1D'})
DIADATA_CSV = 'DiaData_V3.0/5_min_sampling/raw/raw/MDB_5min_sampling_raw/MDB_5min_sampling_raw.csv'

METABONET_COLUMNS = [
    'id', 'source_file', 'date', 'CGM', 'basal', 'bolus', 'insulin', 'carbs',
    'workout_duration', 'insulin_delivery_modality',
    'insulin_type_basal', 'insulin_type_bolus',
]


def gi_gamma_params(gi: float) -> tuple[float, float, float]:
    """SPEC invariants.md §5 carb gamma: (k, theta, duration_minutes) for a GI."""
    g = min(max(gi, 0.0), 100.0) / 100.0
    k = 4.5 + (2.0 - 4.5) * g
    theta = 30.0 + (15.0 - 30.0) * g
    dur = min(max(k * theta * 4.0, 120.0), 360.0)
    return k, theta, dur


CARB_GI_DEFAULT = 50.0


# Regular human insulin is not a rapid analogue: peak 2-3 h, DIA 6-8 h.
_REGULAR_VARIANT = {'gamma_k': 3.0, 'gamma_theta': 75.0, 'dia_base_hours': 7.0}


def bolus_variant(name: str | None) -> dict[str, float]:
    s = (name or '').lower()
    if 'regular' in s or 'novolin' in s or 'humulin r' in s or 'gansulin' in s:
        return _REGULAR_VARIANT
    if 'lispro' in s or 'humalog' in s:
        return BOLUS_VARIANTS['lispro']
    # aspart, novolog/novalog, fiasp, apidra/glulisine, unknown -> aspart-shaped
    return BOLUS_VARIANTS['aspart']


# Detemir: half-life 5-7 h, duration 12-24 h — neither T1DMSIM analogue fits.
_DETEMIR_VARIANT = {'ka': 0.30, 'ke': 0.09, 'action_hours': 20.0,
                    'tail_clip_hours': 4.0}


def basal_variant(name: str | None) -> dict[str, float]:
    s = (name or '').lower()
    if 'degludec' in s or 'tresiba' in s:
        return BASAL_VARIANTS['degludec']
    if 'detemir' in s or 'levemir' in s:
        return _DETEMIR_VARIANT
    return BASAL_VARIANTS['glargine']


def _add_curve(dst: np.ndarray, curve: np.ndarray, start: int) -> None:
    n = min(len(curve), len(dst) - start)
    if n > 0:
        dst[start:start + n] += curve[:n]


def _events_to_curves(
    n: int,
    idx: np.ndarray,
    carbs: np.ndarray,
    basal: np.ndarray,
    bolus: np.ndarray,
    insulin: np.ndarray,
    workout: np.ndarray,
    is_mdi: np.ndarray,
    bolus_type: str | None,
    basal_type: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(carb_curve, insulin_curve, exercise_curve), each (n,) float32 amount/step.

    ``is_mdi`` is PER ROW: ShanghaiT1DM mixes MDI injections with pump slots in one
    record; a 20 U injection through the rapid kernel would peak 10x too high."""
    carb_out = np.zeros(n, dtype=np.float64)
    ins_out = np.zeros(n, dtype=np.float64)
    ex_out = np.zeros(n, dtype=np.float64)

    ck, ct, cdur = gi_gamma_params(CARB_GI_DEFAULT)
    ev = np.flatnonzero(np.nan_to_num(carbs) > 0.0)
    for j in ev:
        g = min(float(carbs[j]), CARB_EVENT_MAX_G)
        _add_curve(carb_out, gamma_curve(g, ck, ct, cdur), int(idx[j]))

    bv = bolus_variant(bolus_type)
    ev = np.flatnonzero(np.nan_to_num(bolus) > 0.0)
    for j in ev:
        k, theta, dur = bolus_pk_for_dose(
            float(bolus[j]), bv['gamma_k'], bv['gamma_theta'], bv['dia_base_hours'])
        _add_curve(ins_out, gamma_curve(float(bolus[j]), k, theta, dur), int(idx[j]))

    b = np.nan_to_num(basal)
    av = basal_variant(basal_type)
    # Both MDI label and dose size required: label alone double-spreads HUPA-UCM per-slot rates.
    inj = (b >= MDI_INJECTION_MIN_U) & is_mdi
    for j in np.flatnonzero(inj):
        _add_curve(
            ins_out,
            basal_curve(float(basal[j]), av['action_hours'] * 60.0,
                        av['ka'], av['ke'], av['tail_clip_hours']),
            int(idx[j]),
        )
    basal_slot = np.zeros(n, dtype=np.float64)
    sel = (b > 0.0) & ~inj
    np.add.at(basal_slot, idx[sel], b[sel])
    # Merged-``insulin``-only rows: injection-sized doses take bolus PK, rest per-slot pump.
    only_total = np.isnan(basal) & np.isnan(bolus) & (np.nan_to_num(insulin) > 0.0)
    for j in np.flatnonzero(only_total & (np.nan_to_num(insulin) >= MDI_INJECTION_MIN_U)):
        k, theta, dur = bolus_pk_for_dose(
            float(insulin[j]), bv['gamma_k'], bv['gamma_theta'], bv['dia_base_hours'])
        _add_curve(ins_out, gamma_curve(float(insulin[j]), k, theta, dur), int(idx[j]))
    small = only_total & (np.nan_to_num(insulin) < MDI_INJECTION_MIN_U)
    np.add.at(basal_slot, idx[small], insulin[small])
    if basal_slot.any():
        kern = gamma_curve(1.0, bv['gamma_k'], bv['gamma_theta'],
                           bv['dia_base_hours'] * 60.0)
        ins_out += np.convolve(basal_slot, kern)[:n]

    # Bouts joined into sessions (idx sorted) before the gamma spread, per SPEC §5.
    sessions: list[list[float]] = []  # [start_slot, duration_min, end_min]
    for j in np.flatnonzero(np.nan_to_num(workout) > 0.0):
        t_min = float(idx[j]) * DT_MINUTES
        dur = float(workout[j])
        if sessions and t_min - sessions[-1][2] < EXERCISE_SESSION_JOIN_MIN:
            sessions[-1][1] += dur
            sessions[-1][2] = max(sessions[-1][2], t_min + dur)
        else:
            sessions.append([float(idx[j]), dur, t_min + dur])
    for slot, dur, _end in sessions:
        dur = min(dur, EXERCISE_SESSION_MAX_MIN)
        mag = dur * EXERCISE_CARB_EQUIV_PER_MIN
        _add_curve(
            ex_out,
            gamma_curve(mag, EXERCISE_GAMMA_K, EXERCISE_GAMMA_THETA, dur + 90.0),
            int(slot),
        )

    return (carb_out.astype(np.float32), ins_out.astype(np.float32),
            ex_out.astype(np.float32))


class _ChannelAppender:
    """Flat binary per channel; count tracked by the caller through ``offset``."""

    def __init__(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.files = {c: open(os.path.join(out_dir, f'{c}.bin'), 'wb')
                      for c in CACHE_CHANNELS}
        self.offset = 0

    def append(self, arrays: dict[str, np.ndarray]) -> int:
        n = len(arrays['bg'])
        for c in CACHE_CHANNELS:
            a = arrays[c]
            assert a.dtype == CHANNEL_DTYPES[c] and len(a) == n
            self.files[c].write(a.tobytes())
        start = self.offset
        self.offset += n
        return start

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def _process_metabonet_subject(
    app: _ChannelAppender,
    index: list[dict[str, Any]],
    source: str,
    sid: str,
    train_raw: dict[str, list[np.ndarray]] | None,
    test_raw: dict[str, list[np.ndarray]] | None,
) -> None:
    parts = [r for r in (train_raw, test_raw) if r is not None]
    ts = np.concatenate([np.concatenate(p['ts']) for p in parts])
    cols = {c: np.concatenate([np.concatenate(p[c]) for p in parts])
            for c in ('CGM', 'basal', 'bolus', 'insulin', 'carbs',
                      'workout_duration', 'is_mdi', 'is_test')}
    order = np.argsort(ts, kind='stable')
    ts = ts[order]
    cols = {c: v[order] for c, v in cols.items()}

    t0 = int(ts[0])
    idx = np.rint((ts - t0) / STEP_S).astype(np.int64)
    n = int(idx[-1]) + 1
    if n < (MIN_CONTEXT_PATCHES + PREDICTION_PATCHES) * PATCH_SIZE and test_raw is None:
        return

    meta_src = train_raw if train_raw is not None else test_raw
    assert meta_src is not None
    bolus_type = meta_src['bolus_type']
    basal_type = meta_src['basal_type']

    bg = np.full(n, np.nan, dtype=np.float32)
    cgm = cols['CGM']
    have = np.isfinite(cgm)
    bg[idx[have]] = cgm[have]

    carb, ins, ex = _events_to_curves(
        n, idx, cols['carbs'], cols['basal'], cols['bolus'], cols['insulin'],
        cols['workout_duration'], cols['is_mdi'].astype(bool),
        bolus_type, basal_type,
    )

    # Per-step test flag: CTR3 rows interleave train/test; one boundary index can't say which.
    is_test_grid = np.zeros(n, dtype=np.uint8)
    trow = cols['is_test'].astype(bool)
    is_test_grid[idx[trow]] = 1
    # Off the sorted grid: parquet chunk order isn't preserved; ShanghaiT1DM blocks violate it.
    test_start = int(idx[trow].min()) if trow.any() else -1

    start = app.append({'bg': bg, 'carb': carb, 'insulin': ins, 'exercise': ex,
                        'is_test': is_test_grid})
    index.append({
        'key': f'metabonet:{source}:{sid}', 'pool': 'metabonet', 'source': source,
        'sid': sid, 'start': start, 'n': n, 't0': t0, 'test_start': test_start,
    })


def _new_acc() -> dict:
    return {
        'ts': [], 'CGM': [], 'basal': [], 'bolus': [], 'insulin': [],
        'carbs': [], 'workout_duration': [], 'is_mdi': [], 'is_test': [],
        'bolus_type': None, 'basal_type': None,
    }


def _collect_parquet(path: str, keep: dict[tuple[str, str], dict] | None,
                     on_source_end, limit_subjects: int | None,
                     is_test: bool) -> None:
    """Stream a MetaboNet parquet grouped by contiguous source blocks.

    ``keep``: accumulate every subject (test pass, in RAM). ``on_source_end(source,
    subjects)``: called per finished source (train pass). ``is_test`` stamps each row."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cur_source: str | None = None
    acc: dict[str, dict] = {}

    def flush() -> None:
        if cur_source is not None and on_source_end is not None:
            on_source_end(cur_source, acc)
        acc.clear()

    test_flag = np.uint8(1 if is_test else 0)
    for batch in pf.iter_batches(columns=METABONET_COLUMNS, batch_size=500_000):
        df = batch.to_pandas()
        ts = df['date'].to_numpy().astype('datetime64[s]').astype(np.int64)
        num = {c: df[c].to_numpy(dtype=np.float32, na_value=np.nan)
               for c in ('CGM', 'basal', 'bolus', 'insulin', 'carbs', 'workout_duration')}
        mdi = df['insulin_delivery_modality'].eq('MDI').to_numpy(
            dtype=bool, na_value=False).astype(np.uint8)
        srcs = df['source_file'].to_numpy(dtype=object)
        ids = df['id'].to_numpy(dtype=object)
        # \x1f not \x00: <U dtype strips trailing NUL, so ('Loop','2x')/('Loop2','x') would collide.
        group_key = np.char.add(np.char.add(srcs.astype(str), '\x1f'), ids.astype(str))
        boundaries = np.flatnonzero(group_key[1:] != group_key[:-1]) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(df)]])
        for s, e in zip(starts, ends):
            source, sid = str(srcs[s]), str(ids[s])
            if source != cur_source:
                flush()
                cur_source = source
            if keep is not None:
                a = keep.setdefault((source, sid), _new_acc())
            else:
                if limit_subjects is not None and sid not in acc and len(acc) >= limit_subjects:
                    continue
                a = acc.setdefault(sid, _new_acc())
            a['ts'].append(ts[s:e])
            for c, v in num.items():
                a[c].append(v[s:e])
            a['is_mdi'].append(mdi[s:e])
            a['is_test'].append(np.full(e - s, test_flag, dtype=np.uint8))
            if a['bolus_type'] is None:
                a['bolus_type'] = _first_str(df['insulin_type_bolus'], s, e)
            if a['basal_type'] is None:
                a['basal_type'] = _first_str(df['insulin_type_basal'], s, e)
    flush()


def _first_str(col, s: int, e: int) -> str | None:
    v = col.iloc[s:e].dropna()
    return str(v.iloc[0]) if len(v) else None


def _build_metabonet(app: _ChannelAppender, index: list[dict[str, Any]],
                     metabonet_dir: str, limit_subjects: int | None) -> None:
    test_raw: dict[tuple[str, str], dict] = {}
    _collect_parquet(os.path.join(metabonet_dir, 'test.parquet'),
                     keep=test_raw, on_source_end=None, limit_subjects=None,
                     is_test=True)

    done: set[tuple[str, str]] = set()

    def on_source_end(source: str, subjects: dict[str, dict]) -> None:
        for sid, raw in subjects.items():
            t = test_raw.get((source, sid))
            _process_metabonet_subject(app, index, source, sid, raw, t)
            done.add((source, sid))
        print(f'  metabonet {source}: {len(subjects)} subjects', flush=True)

    _collect_parquet(os.path.join(metabonet_dir, 'train.parquet'),
                     keep=None, on_source_end=on_source_end,
                     limit_subjects=limit_subjects, is_test=False)

    # Test-only subjects (no train-period rows).
    n_test_only = 0
    per_source_test_only: dict[str, int] = {}
    for (source, sid), raw in test_raw.items():
        if (source, sid) in done:
            continue
        if (limit_subjects is not None
                and per_source_test_only.get(source, 0) >= limit_subjects):
            continue
        _process_metabonet_subject(app, index, source, sid, None, raw)
        per_source_test_only[source] = per_source_test_only.get(source, 0) + 1
        n_test_only += 1
    print(f'  metabonet test-only subjects: {n_test_only}', flush=True)


def _build_diadata(app: _ChannelAppender, index: list[dict[str, Any]],
                   archive: str, limit_subjects: int | None) -> None:
    import pandas as pd
    zf = zipfile.ZipFile(archive)
    acc: dict[str, dict[str, list]] = {}
    per_source_count: dict[str, int] = {}
    min_len = (MIN_CONTEXT_PATCHES + PREDICTION_PATCHES) * PATCH_SIZE

    def flush(pid: str) -> None:
        a = acc.pop(pid)
        source = a['source']
        ts = np.concatenate(a['ts'])
        cgm = np.concatenate(a['cgm'])
        order = np.argsort(ts, kind='stable')
        ts, cgm = ts[order], cgm[order]
        t0 = int(ts[0])
        idx = np.rint((ts - t0) / STEP_S).astype(np.int64)
        n = int(idx[-1]) + 1
        if n < min_len or n > 400_000:
            return
        if limit_subjects is not None and per_source_count.get(source, 0) >= limit_subjects:
            return
        bg = np.full(n, np.nan, dtype=np.float32)
        have = np.isfinite(cgm)
        bg[idx[have]] = cgm[have]
        z = np.zeros(n, dtype=np.float32)
        start = app.append({'bg': bg, 'carb': z, 'insulin': z, 'exercise': z,
                            'is_test': np.zeros(n, dtype=np.uint8)})
        index.append({
            'key': f'diadata:{source}:{pid}', 'pool': 'diadata', 'source': source,
            'sid': pid, 'start': start, 'n': n, 't0': t0, 'test_start': -1,
        })
        per_source_count[source] = per_source_count.get(source, 0) + 1

    with zf.open(DIADATA_CSV) as fh:
        reader = pd.read_csv(io.TextIOWrapper(fh, encoding='utf-8'),
                             chunksize=2_000_000)
        prev_pid: str | None = None
        for chunk in reader:
            keep = ~chunk['Database'].isin(DIADATA_DROP_SOURCES)
            chunk = chunk[keep]
            if not len(chunk):
                continue
            ts = pd.to_datetime(chunk['ts']).to_numpy().astype('datetime64[s]').astype(np.int64)
            cgm = chunk['GlucoseCGM'].to_numpy(dtype=np.float64, na_value=np.nan)
            pids = chunk['PtID'].to_numpy(dtype=object)
            dbs = chunk['Database'].to_numpy(dtype=object)
            boundaries = np.flatnonzero(pids[1:] != pids[:-1]) + 1
            starts = np.concatenate([[0], boundaries])
            ends = np.concatenate([boundaries, [len(chunk)]])
            for s, e in zip(starts, ends):
                pid = str(pids[s])
                if prev_pid is not None and pid != prev_pid and prev_pid in acc:
                    flush(prev_pid)
                a = acc.setdefault(pid, {'ts': [], 'cgm': [], 'source': str(dbs[s])})
                a['ts'].append(ts[s:e])
                a['cgm'].append(cgm[s:e])
                prev_pid = pid
    for pid in list(acc):
        flush(pid)
    print(f'  diadata: {sum(per_source_count.values())} subjects '
          f'{dict(sorted(per_source_count.items()))}', flush=True)


def build_cache(metabonet_dir: str, out_dir: str, skip_diadata: bool = False,
                limit_subjects: int | None = None) -> None:
    app = _ChannelAppender(out_dir)
    index: list[dict[str, Any]] = []
    print('building metabonet...', flush=True)
    _build_metabonet(app, index, metabonet_dir, limit_subjects)
    if not skip_diadata:
        print('building diadata...', flush=True)
        _build_diadata(app, index, os.path.join(metabonet_dir, 'archive.zip'),
                       limit_subjects)
    app.close()
    keys = [r['key'] for r in index]
    assert len(keys) == len(set(keys)), (
        'duplicate subject keys — a source or PtID block reappeared '
        'non-contiguously and was processed twice')
    meta = {
        'cache_version': CACHE_VERSION, 'dt_minutes': DT_MINUTES,
        'channels': CACHE_CHANNELS, 'total_steps': app.offset,
        'n_subjects': len(index),
    }
    with open(os.path.join(out_dir, 'index.json'), 'w') as f:
        json.dump({'meta': meta, 'subjects': index}, f)
    fit_cache_stats(out_dir)
    print(f'done: {len(index)} subjects, {app.offset} steps -> {out_dir}', flush=True)


def fit_cache_stats(cache_dir: str) -> None:
    """Fit and write ``<cache>/normalization_stats.json`` from the built cache."""
    from normalization import save_normalization_stats
    stats = compute_cache_stats(FinetuneCache(cache_dir))
    save_normalization_stats(stats, os.path.join(cache_dir, 'normalization_stats.json'))


class FinetuneCache:
    """Memmapped view of a built cache. Opened lazily so DataLoader forks re-open."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, 'index.json')) as f:
            d = json.load(f)
        if d['meta'].get('cache_version') != CACHE_VERSION:
            raise ValueError(f"cache at {cache_dir!r} has version "
                             f"{d['meta'].get('cache_version')!r}, expected {CACHE_VERSION}")
        self.meta = d['meta']
        self.subjects: list[dict[str, Any]] = d['subjects']
        self._mm: dict[str, np.memmap] | None = None

    def __getstate__(self) -> dict[str, Any]:
        # memmap pickles as a materialized ndarray (whole file); drop handles, re-open lazily.
        return {**self.__dict__, '_mm': None}

    def _maps(self) -> dict[str, np.memmap]:
        if self._mm is None:
            total = int(self.meta['total_steps'])
            self._mm = {
                c: np.memmap(os.path.join(self.cache_dir, f'{c}.bin'),
                             dtype=CHANNEL_DTYPES[c], mode='r', shape=(total,))
                for c in CACHE_CHANNELS
            }
        return self._mm

    def channels(self, rec: dict[str, Any], lo: int, hi: int) -> dict[str, np.ndarray]:
        mm = self._maps()
        s = int(rec['start'])
        return {c: np.asarray(mm[c][s + lo:s + hi]) for c in CACHE_CHANNELS}

    def train_len(self, rec: dict[str, Any]) -> int:
        ts = int(rec['test_start'])
        return int(rec['n']) if ts < 0 else min(int(rec['n']), ts)


_STATS_CHANNEL = dict(zip(CHANNEL_NAMES, ('bg', 'carb', 'insulin', 'exercise')))
STATS_FIT_WINDOWS = 2000
STATS_FIT_SEED = 0
# Std floor: near-zero data else pushes one event past z=+40; caps channel max at this sigma.
STATS_SPARSE_Z_MAX = 12.0


def compute_cache_stats(cache: 'FinetuneCache') -> dict[str, dict[str, float]]:
    """Per-channel mean/std fit over SAMPLER-DRAWN training windows.

    Raw-step fit overweighs DiaData (43% of steps vs ~20% sampler draw), so this
    replays the training draw law (STATS_FIT_WINDOWS windows) and pools steps in
    ``_normalize_features`` space; test-period steps never enter."""
    ds = FinetuneTrainDataset(cache, stats=None, seed=STATS_FIT_SEED,
                              total_steps=1, batch_size=1)
    rng = np.random.default_rng(STATS_FIT_SEED)
    acc = {name: [0.0, 0.0, 0] for name in CHANNEL_NAMES}  # sum, sumsq, n
    for _ in range(STATS_FIT_WINDOWS):
        w = ds._draw_from(rng, raw=True)
        for name in CHANNEL_NAMES:
            x = np.asarray(w[_STATS_CHANNEL[name]], dtype=np.float64)
            if name in RISK_SPACE_CHANNELS:
                x = x[np.isfinite(x)]
                if not len(x):
                    continue
                x = kovatchev_f_np(np.clip(x, BG_CLAMP_MIN, BG_CLAMP_MAX))
            elif name in SPARSE_LOG1P_CHANNELS:
                x = np.log1p(np.maximum(x, 0.0))
            a = acc[name]
            a[0] += float(x.sum())
            a[1] += float((x * x).sum())
            a[2] += len(x)
    stats: dict[str, dict[str, float]] = {}
    for name, (s, sq, n) in acc.items():
        assert n > 0, f'no fitted data for channel {name}'
        mean = s / n
        var = max(sq / n - mean * mean, 0.0)
        std = float(np.sqrt(var))
        if name in SPARSE_LOG1P_CHANNELS:
            mm = cache._maps()[_STATS_CHANNEL[name]]
            ch_max = float(np.log1p(max(float(mm.max()), 0.0)))
            std = max(std, (ch_max - mean) / STATS_SPARSE_Z_MAX)
        assert std > 0.0, f'zero variance for channel {name}'
        stats[name] = {'mean': float(mean), 'std': std}
    return stats


def _interp_short_gaps(bg: np.ndarray, max_steps: int) -> np.ndarray:
    """Linearly fill NaN runs of length <= max_steps bracketed by measurements."""
    if max_steps <= 0 or not np.isnan(bg).any():
        return bg
    out = bg.copy()
    isn = np.isnan(out)
    d = np.diff(np.concatenate([[0], isn.view(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s <= max_steps and s > 0 and e < len(out):
            out[s:e] = np.interp(np.arange(s, e), [s - 1, e], [out[s - 1], out[e]])
    return out


def _normalize_features(bg: np.ndarray, carb: np.ndarray, insulin: np.ndarray,
                        exercise: np.ndarray,
                        stats: dict[str, dict[str, float]]) -> np.ndarray:
    """(N, N_INPUT_FEATURES) z-space feature stack; NaN bg must be pre-filled."""
    bg_c = np.clip(bg, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    feats = np.stack([
        bg_c,
        np.maximum(carb, 0.0).astype(np.float32),
        np.maximum(insulin, 0.0).astype(np.float32),
        np.maximum(exercise, 0.0).astype(np.float32),
        np.zeros_like(bg_c),
    ], axis=-1)
    for c, name in enumerate(CHANNEL_NAMES):
        mean, std = stats[name]['mean'], stats[name]['std']
        col = feats[:, c]
        if name in RISK_SPACE_CHANNELS:
            col = kovatchev_f_np(col)
        elif name in SPARSE_LOG1P_CHANNELS:
            col = np.log1p(np.maximum(col, 0.0))
        feats[:, c] = (col - mean) / (std + 1e-8)
    return feats


_BG_GAP_FILL_MGDL = 120.0  # written only into masked patches, whose bg input is zeroed


def _assemble_sample(feats: np.ndarray, bg: np.ndarray, spans: list[tuple[int, int]],
                     gap_patches: np.ndarray, seq_len: int,
                     n_ctx: int) -> dict[str, Any]:
    """Shared tail of the train/eval sample builders. ``bg`` raw mg/dL with NaN gaps."""
    mask_idx, valid, mask_d, anchor_step = _mask_slots(spans, seq_len)
    span_patches = np.concatenate(
        [np.arange(s, s + L, dtype=np.int64) for s, L in spans])
    masked_all = np.union1d(span_patches, gap_patches)

    patches_t = torch.from_numpy(
        feats[:seq_len * PATCH_SIZE].reshape(seq_len, PATCH_DIM).copy())
    rows = torch.from_numpy(masked_all)
    for feat_idx in NON_MASKABLE_FEATS:
        patches_t[rows, feat_idx::N_INPUT_FEATURES] = 0.0
    patches_t[rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0

    bg_filled = np.clip(np.nan_to_num(bg, nan=_BG_GAP_FILL_MGDL),
                        BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    bg_patches = bg_filled[:seq_len * PATCH_SIZE].reshape(seq_len, PATCH_SIZE)
    targets_t = torch.from_numpy(bg_patches[mask_idx].copy())

    last_visible = np.flatnonzero(np.isfinite(bg[:n_ctx * PATCH_SIZE]))
    assert len(last_visible), "window has no measured context bg"
    last_bg = float(np.clip(bg[last_visible[-1]], BG_CLAMP_MIN, BG_CLAMP_MAX))
    anchor_bg = np.full(MAX_MASKED_PATCHES, last_bg, dtype=np.float32)
    anchor_bg[valid] = bg_filled[anchor_step[valid]]
    assert np.isfinite(anchor_bg).all()

    return {
        'patches': patches_t.float(),
        'targets': targets_t.float(),
        'n_context_patches': n_ctx,
        'mask_idx': mask_idx,
        'valid': valid,
        'anchor_bg': anchor_bg,
        'masked_all': masked_all,
    }


class FinetuneTrainDataset(torch.utils.data.Dataset):
    """Tempered-source sampling over the cache's train-period steps.

    Draw per index: pool (diadata w.p. ``diadata_frac``), source ``p ∝
    n_subjects^source_alpha``, subject uniform, window rejection-sampled under the
    gap budget, retried WITHIN the drawn (pool, source). Deterministic in ``(seed, idx)``."""

    def __init__(self, cache: FinetuneCache, stats: dict[str, dict[str, float]] | None,
                 seed: int, total_steps: int, batch_size: int,
                 source_alpha: float = 0.5, diadata_frac: float = 0.2,
                 gap_budget: float = 0.2, max_interp_steps: int = 1,
                 no_carbs: bool = False) -> None:
        self.cache = cache
        self.stats = stats
        self.seed = seed
        self.total = total_steps * batch_size
        self.gap_budget = gap_budget
        self.max_interp_steps = max_interp_steps
        self.no_carbs = no_carbs

        min_len = (MIN_CONTEXT_PATCHES + PREDICTION_PATCHES) * PATCH_SIZE
        w_patches = MIN_CONTEXT_PATCHES + PREDICTION_PATCHES
        need_vis = int(np.ceil((1.0 - gap_budget) * w_patches))
        pools: dict[str, dict[str, list[int]]] = {'metabonet': {}, 'diadata': {}}
        excluded: dict[str, int] = {}
        for i, rec in enumerate(cache.subjects):
            tl = cache.train_len(rec)
            if tl < min_len:
                continue
            # Some min-length window must clear the gap budget, else every draw here retries.
            bg = _interp_short_gaps(
                np.asarray(cache.channels(rec, 0, tl)['bg'], dtype=np.float32),
                max_interp_steps)
            vis = np.isfinite(bg[:(tl // PATCH_SIZE) * PATCH_SIZE]) \
                .reshape(-1, PATCH_SIZE).all(axis=1)
            cs = np.concatenate([[0], np.cumsum(vis.astype(np.int64))])
            if len(cs) <= w_patches or (cs[w_patches:] - cs[:-w_patches]).max() < need_vis:
                excluded[rec['source']] = excluded.get(rec['source'], 0) + 1
                continue
            pools[rec['pool']].setdefault(rec['source'], []).append(i)
        if excluded:
            print(f'finetune sampler: excluded {sum(excluded.values())} subjects '
                  f'with no viable window {dict(sorted(excluded.items()))}',
                  flush=True)
        self.pool_sources: dict[str, tuple[list[str], np.ndarray]] = {}
        for pool, srcs in pools.items():
            if not srcs:
                continue
            names = sorted(srcs)
            w = np.array([len(srcs[s]) for s in names], dtype=np.float64) ** source_alpha
            self.pool_sources[pool] = (names, w / w.sum())
        # Keyed by (pool, source): the same study name could exist in both pools.
        self.subjects_by_source = {(p, s): v for p, srcs in pools.items()
                                   for s, v in srcs.items()}
        self.diadata_frac = diadata_frac if 'diadata' in self.pool_sources else 0.0
        assert 'metabonet' in self.pool_sources, "no trainable metabonet subjects"

    def __len__(self) -> int:
        return self.total

    def _draw_window(self, rng: np.random.Generator, rec: dict[str, Any],
                     raw: bool = False) -> dict[str, Any] | None:
        train_len = self.cache.train_len(rec)

        n_ctx = int(rng.integers(MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES + 1))
        seq_len = n_ctx + PREDICTION_PATCHES
        need = seq_len * PATCH_SIZE
        if train_len < need:
            n_ctx = MIN_CONTEXT_PATCHES
            seq_len = n_ctx + PREDICTION_PATCHES
            need = seq_len * PATCH_SIZE
            if train_len < need:
                return None
        # Skip the dose-history lead-in when the record leaves room for it.
        lead = LEAD_IN_STEPS if train_len - need >= LEAD_IN_STEPS else 0
        max_start_patch = (train_len - need - lead) // PATCH_SIZE
        start = lead + int(rng.integers(max_start_patch + 1)) * PATCH_SIZE

        ch = self.cache.channels(rec, start, start + need)
        bg = _interp_short_gaps(ch['bg'].astype(np.float32), self.max_interp_steps)
        visible = np.isfinite(bg).reshape(seq_len, PATCH_SIZE).all(axis=1)
        gap_patches = np.flatnonzero(~visible).astype(np.int64)
        if len(gap_patches) > self.gap_budget * seq_len:
            return None

        spans = None
        for _ in range(20):
            cand = sample_mask_spans(seq_len, rng)
            ok = all(
                visible[s:s + L].all()
                and (s == 0 or visible[s - 1])
                and (s + L == seq_len or visible[s + L])
                for s, L in cand
            )
            if ok:
                spans = cand
                break
        if spans is None:
            zone = seq_len - PREDICTION_PATCHES
            if visible[zone:].all() and visible[zone - 1]:
                spans = [(zone, PREDICTION_PATCHES)]
            else:
                return None

        if raw:
            return {'bg': bg, 'carb': ch['carb'], 'insulin': ch['insulin'],
                    'exercise': ch['exercise'], 'start': start, 'seq_len': seq_len}
        assert self.stats is not None, 'stats=None dataset is for raw draws only'
        carb = np.zeros_like(ch['carb']) if self.no_carbs else ch['carb']
        feats = _normalize_features(
            np.nan_to_num(bg, nan=_BG_GAP_FILL_MGDL),
            carb, ch['insulin'], ch['exercise'], self.stats)
        sample = _assemble_sample(feats, bg, spans, gap_patches, seq_len, n_ctx)
        # Time-probe hour/slot: slot j reads patch mask_idx[j], step start+mask_idx[j]*PATCH_SIZE.
        abs_s = int(rec['t0']) + (start + sample['mask_idx'] * PATCH_SIZE) * STEP_S
        sample['slot_hour'] = ((abs_s % 86400) / 3600.0).astype(np.float32)
        return sample

    def _draw_from(self, rng: np.random.Generator,
                   raw: bool = False) -> dict[str, Any]:
        """One accepted window under the documented draw law."""
        for _ in range(8):
            pool = ('diadata' if rng.random() < self.diadata_frac else 'metabonet')
            names, p = self.pool_sources[pool]
            source = names[int(rng.choice(len(names), p=p))]
            subs = self.subjects_by_source[(pool, source)]
            # Retries stay inside the drawn (pool, source): a global redraw favors dense sources.
            for _ in range(32):
                rec = self.cache.subjects[subs[int(rng.integers(len(subs)))]]
                s = self._draw_window(rng, rec, raw=raw)
                if s is not None:
                    return s
        raise RuntimeError('finetune sampler: window draws exhausted — '
                           'gap budget or cache too restrictive')

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng((self.seed, idx))
        return self._draw_from(rng)


def finetune_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Left-padded batch; the attention mask covers the FULL masked set (gaps included)."""
    B = len(samples)
    M = MAX_MASKED_PATCHES
    seq_lens = [s['n_context_patches'] + PREDICTION_PATCHES for s in samples]
    max_T = max(seq_lens)
    n_pads = [max_T - sl for sl in seq_lens]

    patches = torch.zeros(B, max_T, PATCH_DIM, dtype=torch.float32)
    targets = torch.stack([s['targets'] for s in samples])
    valid = torch.from_numpy(np.stack([s['valid'] for s in samples]))
    mask_idx = torch.zeros(B, M, dtype=torch.long)
    anchor_bg = torch.from_numpy(np.stack([s['anchor_bg'] for s in samples]))
    is_pad = torch.zeros(B, max_T, dtype=torch.bool)
    masked = torch.zeros(B, max_T, dtype=torch.bool)

    for i, s in enumerate(samples):
        n_pad = n_pads[i]
        patches[i, n_pad:, :] = s['patches']
        is_pad[i, :n_pad] = True
        row_valid = valid[i]
        idx = torch.from_numpy(s['mask_idx']) + n_pad
        mask_idx[i, row_valid] = idx[row_valid]
        masked[i, torch.from_numpy(s['masked_all']) + n_pad] = True

    attn_mask = utils.create_attention_mask_from_visible(~masked, is_pad)
    assert attn_mask.any(dim=-1).all(), "an all-False attention row NaNs softmax"
    _bit = patches[..., BG_MASKED_FEAT::N_INPUT_FEATURES]
    assert torch.equal(_bit[..., 0], masked.to(_bit.dtype)), (
        "feat 4 does not reproduce the masked set that built attn_mask")

    out = {
        'patches': patches, 'targets': targets, 'attn_mask': attn_mask,
        'mask_idx': mask_idx, 'valid': valid, 'anchor_bg': anchor_bg,
    }
    if 'slot_hour' in samples[0]:
        out['slot_hour'] = torch.from_numpy(
            np.stack([s['slot_hour'] for s in samples]))
    if 'true_bg_horizon' in samples[0]:
        out['true_bg_horizon'] = torch.from_numpy(
            np.stack([s['true_bg_horizon'] for s in samples]))
    return out


def build_eval_windows(cache: FinetuneCache, n_windows: int,
                       seed: int) -> list[tuple[int, int]]:
    """(subject_index, origin_step) pairs over the test-period steps.

    Origin must: be ``is_test``-flagged, have >= MIN_CONTEXT_PATCHES history with
    its anchor patch fully measured, truth at >= 1 of the four horizon steps, and
    clear EVAL_GAP_BUDGET over trailing context. Stricter than the leaderboard template."""
    assert n_windows > 0, 'n_windows must be positive (0 would score millions)'
    cands_subj: list[np.ndarray] = []
    cands_step: list[np.ndarray] = []
    zone = PREDICTION_PATCHES * PATCH_SIZE
    min_hist = MIN_CONTEXT_PATCHES * PATCH_SIZE
    max_gaps = EVAL_GAP_BUDGET * MIN_CONTEXT_PATCHES
    for i, rec in enumerate(cache.subjects):
        if int(rec['test_start']) < 0:
            continue
        n = int(rec['n'])
        if n < min_hist + zone:
            continue
        ch = cache.channels(rec, 0, n)
        finite = np.isfinite(ch['bg'])
        origins = np.arange(min_hist, n - zone + 1)
        keep = np.asarray(ch['is_test'][origins], dtype=bool)
        # fin6[p]: steps p..p+5 all measured — one prospective patch.
        c6 = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
        fin6 = (c6[PATCH_SIZE:] - c6[:-PATCH_SIZE]) == PATCH_SIZE
        keep &= fin6[origins - PATCH_SIZE]
        any_target = np.zeros(len(origins), dtype=bool)
        for hs in HORIZON_STEPS:
            any_target |= finite[origins + hs]
        keep &= any_target
        # Gap budget per phase: patch k covers o-6k..o-6k+5; count fin6 by residue prefix sum.
        gap_ok = np.zeros(len(origins), dtype=bool)
        fin6_i = fin6.astype(np.int64)
        for r in range(PATCH_SIZE):
            cs = np.cumsum(fin6_i[r::PATCH_SIZE])
            sel = (origins % PATCH_SIZE) == r
            if not sel.any():
                continue
            o = origins[sel]
            j_hi = (o - PATCH_SIZE - r) // PATCH_SIZE
            j_lo = j_hi - MIN_CONTEXT_PATCHES
            vis = cs[j_hi] - np.where(j_lo >= 0, cs[np.maximum(j_lo, 0)], 0)
            gap_ok[sel] = (MIN_CONTEXT_PATCHES - vis) <= max_gaps
        keep &= gap_ok
        kept = origins[keep]
        if len(kept):
            cands_subj.append(np.full(len(kept), i, dtype=np.int64))
            cands_step.append(kept.astype(np.int64))
    assert cands_subj, "no eval origins survived the filters"
    subj = np.concatenate(cands_subj)
    step = np.concatenate(cands_step)
    rng = np.random.default_rng(seed)
    if n_windows and n_windows < len(subj):
        pick = rng.choice(len(subj), size=n_windows, replace=False)
        subj, step = subj[pick], step[pick]
    return list(zip(subj.tolist(), step.tolist()))


class FinetuneEvalDataset(torch.utils.data.Dataset):
    """Right-edge forecast windows at fixed test origins; truth rides beside."""

    def __init__(self, cache: FinetuneCache, stats: dict[str, dict[str, float]],
                 windows: list[tuple[int, int]], max_interp_steps: int = 1,
                 no_carbs: bool = False) -> None:
        self.cache = cache
        self.stats = stats
        self.windows = windows
        self.max_interp_steps = max_interp_steps
        self.no_carbs = no_carbs

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        subj, origin = self.windows[i]
        rec = self.cache.subjects[subj]
        n_ctx = min(MAX_CONTEXT_PATCHES, origin // PATCH_SIZE)
        seq_len = n_ctx + PREDICTION_PATCHES
        lo = origin - n_ctx * PATCH_SIZE
        hi = origin + PREDICTION_PATCHES * PATCH_SIZE
        ch = self.cache.channels(rec, lo, hi)
        bg = _interp_short_gaps(ch['bg'].astype(np.float32), self.max_interp_steps)
        true_horizon = ch['bg'][n_ctx * PATCH_SIZE:].astype(np.float32)

        visible = np.isfinite(bg).reshape(seq_len, PATCH_SIZE).all(axis=1)
        visible[n_ctx:] = False
        gap_patches = np.flatnonzero(~visible[:n_ctx]).astype(np.int64)
        spans = [(n_ctx, PREDICTION_PATCHES)]

        carb = np.zeros_like(ch['carb']) if self.no_carbs else ch['carb']
        feats = _normalize_features(
            np.nan_to_num(bg, nan=_BG_GAP_FILL_MGDL),
            carb, ch['insulin'], ch['exercise'], self.stats)
        # Zone gaps filled so targets stay finite; context keeps NaNs for honest anchor reads.
        bg_for_sample = bg.copy()
        zone_lo = n_ctx * PATCH_SIZE
        bg_for_sample[zone_lo:] = np.nan_to_num(bg[zone_lo:], nan=_BG_GAP_FILL_MGDL)
        sample = _assemble_sample(feats, bg_for_sample, spans, gap_patches,
                                  seq_len, n_ctx)
        sample['true_bg_horizon'] = true_horizon
        return sample


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='build the merged finetuning cache')
    b.add_argument('--metabonet-dir', default='metabonet')
    b.add_argument('--out', default='metabonet/cache_finetune')
    b.add_argument('--skip-diadata', action='store_true')
    b.add_argument('--limit-subjects', type=int, default=None,
                   help='per source, for a quick smoke build')
    f = sub.add_parser('fit-stats',
                       help='fit normalization_stats.json over a built cache')
    f.add_argument('--cache', default='metabonet/cache_finetune')
    a = p.parse_args()
    if a.cmd == 'build':
        build_cache(a.metabonet_dir, a.out, a.skip_diadata, a.limit_subjects)
    elif a.cmd == 'fit-stats':
        fit_cache_stats(a.cache)


if __name__ == '__main__':
    main()
