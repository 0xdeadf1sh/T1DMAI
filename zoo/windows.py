"""Right-edge forecast windows over a finetune cache, in mg/dL, units and grams.

A window's context is the ``ctx`` steps before ``origin``; its zone is the ``ZONE`` steps from
``origin`` on, so the row's own reading is ``bg[:, -1]`` and ``pred_30`` is zone step 5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finetune_data import HORIZON_MINUTES, HORIZON_STEPS, FinetuneCache, _interp_short_gaps

ZONE = HORIZON_MINUTES[-1] // 5
MAX_INTERP_STEPS = 6  # the leaderboard admits CGM gaps up to 30 min


@dataclass
class Windows:
    bg: np.ndarray        # (N, ctx) mg/dL, gaps filled; last column measured
    insulin: np.ndarray   # (N, ctx + ZONE) U per step, bolus + basal; zone part as announced
    carb: np.ndarray      # (N, ctx + ZONE) g per step
    target: np.ndarray    # (N, ZONE) mg/dL, NaN where unmeasured; all-NaN for template rows
    source: np.ndarray    # (N,) sub-dataset name
    row: np.ndarray       # (N,) template row, -1 for training windows

    def __len__(self) -> int:
        return len(self.bg)

    def take(self, idx: np.ndarray) -> 'Windows':
        return Windows(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))


def _subject_arrays(cache: FinetuneCache, rec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ch = cache.channels(rec, 0, int(rec['n']))
    bg = _interp_short_gaps(np.asarray(ch['bg'], dtype=np.float32), MAX_INTERP_STEPS)
    ins = np.nan_to_num(np.asarray(ch['bolus_u'], np.float32) + np.asarray(ch['basal_u'], np.float32))
    return bg, ins, np.nan_to_num(np.asarray(ch['carb_g'], np.float32))


def _fill_context(bg: np.ndarray) -> np.ndarray:
    """Carry each row's last reading forward, then its first back; rows are (N, ctx)."""
    n, L = bg.shape
    ok = np.isfinite(bg)
    last = np.maximum.accumulate(np.where(ok, np.arange(L), -1), axis=1)
    out = np.take_along_axis(bg, np.maximum(last, 0), axis=1)
    first = np.argmax(ok, axis=1)
    lead = last < 0
    out[lead] = np.broadcast_to(bg[np.arange(n), first][:, None], (n, L))[lead]
    return out


def _cut(bg, ins, carb, origins: np.ndarray, ctx: int, with_target: bool):
    c = origins[:, None] + np.arange(-ctx, 0)[None, :]
    z = origins[:, None] + np.arange(ZONE)[None, :]
    n = len(bg)
    zc = np.clip(z, 0, n - 1)
    dose = np.concatenate([c, zc], axis=1)
    pad = (z >= n)
    ins_w = ins[dose]
    carb_w = carb[dose]
    ins_w[:, ctx:][pad] = 0.0
    carb_w[:, ctx:][pad] = 0.0
    target = bg[zc].astype(np.float32) if with_target else np.full(z.shape, np.nan, np.float32)
    target[pad] = np.nan
    return _fill_context(bg[c]), ins_w, carb_w, target


def _concat(parts: list[tuple], ctx: int) -> Windows:
    if not parts:
        e = np.empty((0, ctx), np.float32)
        return Windows(e, np.empty((0, ctx + ZONE), np.float32), np.empty((0, ctx + ZONE), np.float32),
                       np.empty((0, ZONE), np.float32), np.empty(0, object), np.empty(0, np.int64))
    return Windows(*(np.concatenate([p[i] for p in parts]) for i in range(6)))


def train_windows(cache: FinetuneCache, n: int, ctx: int, seed: int = 0) -> Windows:
    """``n`` windows drawn uniformly over train-period steps whose context and zone fit."""
    rng = np.random.default_rng(seed)
    lens = np.array([max(cache.train_len(r) - ctx - ZONE, 0) for r in cache.subjects], np.int64)
    p = lens / lens.sum()
    counts = rng.multinomial(n, p)
    parts = []
    for s in np.flatnonzero(counts):
        rec = cache.subjects[s]
        bg, ins, carb = _subject_arrays(cache, rec)
        origins = ctx + rng.integers(0, lens[s], counts[s])
        origins = origins[np.isfinite(bg[origins - 1])]
        if not len(origins):
            continue
        b, i, c, t = _cut(bg, ins, carb, origins, ctx, True)
        parts.append((b, i, c, t, np.full(len(b), rec['source'], object), np.full(len(b), -1)))
    return _concat(parts, ctx)


def template_windows(cache: FinetuneCache, rows_by_subject: dict[int, list[int]],
                     t_epoch: np.ndarray, ctx: int) -> Windows:
    """One window per anchorable template row: its reading measured and its context in range."""
    parts = []
    for s, rows in rows_by_subject.items():
        rec = cache.subjects[s]
        bg, ins, carb = _subject_arrays(cache, rec)
        idx = np.asarray(rows, np.int64)
        origin = np.rint((t_epoch[idx] - int(rec['t0'])) / 300.0).astype(np.int64) + 1
        keep = (origin >= 1) & (origin <= len(bg))
        idx, origin = idx[keep], origin[keep]
        keep = np.isfinite(bg[origin - 1])
        idx, origin = idx[keep], origin[keep]
        if not len(idx):
            continue
        lo = np.maximum(origin - ctx, 0)
        short = origin - lo < ctx
        if short.any():
            pre = np.full(ctx, np.nan, np.float32)
            bg = np.concatenate([pre, bg])
            ins = np.concatenate([np.zeros(ctx, np.float32), ins])
            carb = np.concatenate([np.zeros(ctx, np.float32), carb])
            origin = origin + ctx
        b, i, c, t = _cut(bg, ins, carb, origin, ctx, False)
        parts.append((b, i, c, t, np.full(len(b), rec['source'], object), idx))
    return _concat(parts, ctx)


def drop_zone_doses(w: Windows) -> Windows:
    """Zero insulin and carbohydrate after the origin: what a live forecast actually knows."""
    ctx = w.bg.shape[1]
    w.insulin[:, ctx:] = 0.0
    w.carb[:, ctx:] = 0.0
    return w


def horizon_columns(zone_pred: np.ndarray) -> np.ndarray:
    """(N, ZONE) zone forecast -> (N, len(HORIZON_MINUTES)) leaderboard columns."""
    return zone_pred[:, list(HORIZON_STEPS)]
