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
STEP_S = 300
DAY_S = 86_400
HOLDOUT_DAYS = 10
CAL_DAYS = 5
BLOCKS = ('fit', 'val', 'cal')


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


def _subject_arrays(cache: FinetuneCache, rec: dict) -> tuple[np.ndarray, ...]:
    """``(measured bg, gap-filled bg, insulin, carb)`` over the whole record."""
    ch = cache.channels(rec, 0, int(rec['n']))
    raw = np.asarray(ch['bg'], dtype=np.float32)
    bg = _interp_short_gaps(raw, MAX_INTERP_STEPS)
    ins = np.nan_to_num(np.asarray(ch['bolus_u'], np.float32) + np.asarray(ch['basal_u'], np.float32))
    return raw, bg, ins, np.nan_to_num(np.asarray(ch['carb_g'], np.float32))


def person_of(rec: dict) -> str:
    """A phone cache holds one person's grid and per-sensor records; other records stand alone."""
    return f"phone:{rec['source']}" if rec.get('pool') == 'phone' else rec['key']


def block_spans(cache: FinetuneCache) -> np.ndarray:
    """(subjects, 3, 2) record steps ``[lo, hi)`` of the fit, val and cal blocks.

    Per person, the last ``HOLDOUT_DAYS`` of calendar time before the train period ends leave the
    fit: the older part selects epochs, the last ``CAL_DAYS`` calibrate the bands.
    """
    tl = np.array([cache.train_len(r) for r in cache.subjects], np.int64)
    t0 = np.array([int(r['t0']) for r in cache.subjects], np.int64)
    person = [person_of(r) for r in cache.subjects]
    end: dict[str, int] = {}
    for p, e in zip(person, (t0 + tl * STEP_S).tolist()):
        end[p] = max(end.get(p, e), e)
    e = np.array([end[p] for p in person], np.int64)
    cuts = np.stack([e - HOLDOUT_DAYS * DAY_S, e - CAL_DAYS * DAY_S], axis=1)
    steps = np.clip(-((t0[:, None] - cuts) // STEP_S), 0, tl[:, None])
    edges = np.concatenate([np.zeros_like(tl)[:, None], steps, tl[:, None]], axis=1)
    return np.stack([edges[:, :-1], edges[:, 1:]], axis=-1)


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


def _cut(bg, ins, carb, origins: np.ndarray, ctx: int, measured: np.ndarray | None):
    """Context from the gap-filled ``bg``; target from ``measured``, all-NaN when it is None."""
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
    target = (measured[zc].astype(np.float32) if measured is not None
              else np.full(z.shape, np.nan, np.float32))
    target[pad] = np.nan
    return _fill_context(bg[c]), ins_w, carb_w, target


def _concat(parts: list[tuple], ctx: int) -> Windows:
    if not parts:
        e = np.empty((0, ctx), np.float32)
        return Windows(e, np.empty((0, ctx + ZONE), np.float32), np.empty((0, ctx + ZONE), np.float32),
                       np.empty((0, ZONE), np.float32), np.empty(0, object), np.empty(0, np.int64))
    return Windows(*(np.concatenate([p[i] for p in parts]) for i in range(6)))


def train_windows(cache: FinetuneCache, n: int, ctx: int, seed: int = 0,
                  block: str = 'fit') -> Windows:
    """Up to ``n`` windows drawn uniformly over origins inside one of ``block_spans``' blocks.

    The last context reading and the whole zone lie in the block; the context may reach back
    before it. Origins whose last reading was not measured are dropped; targets are measured only.
    """
    rng = np.random.default_rng(seed)
    span = block_spans(cache)[:, BLOCKS.index(block)]
    first = np.maximum(span[:, 0] + 1, ctx)
    lens = np.maximum(span[:, 1] - ZONE - first + 1, 0)
    if not lens.sum():
        return _concat([], ctx)
    counts = rng.multinomial(n, lens / lens.sum())
    parts = []
    for s in np.flatnonzero(counts):
        rec = cache.subjects[s]
        raw, bg, ins, carb = _subject_arrays(cache, rec)
        origins = first[s] + rng.integers(0, lens[s], counts[s])
        origins = origins[np.isfinite(raw[origins - 1])]
        if not len(origins):
            continue
        b, i, c, t = _cut(bg, ins, carb, origins, ctx, raw)
        parts.append((b, i, c, t, np.full(len(b), rec['source'], object), np.full(len(b), -1)))
    return _concat(parts, ctx)


def template_windows(cache: FinetuneCache, rows_by_subject: dict[int, list[int]],
                     t_epoch: np.ndarray, ctx: int) -> Windows:
    """One window per anchorable template row: its reading measured and its context in range."""
    parts = []
    for s, rows in rows_by_subject.items():
        rec = cache.subjects[s]
        raw, bg, ins, carb = _subject_arrays(cache, rec)
        idx = np.asarray(rows, np.int64)
        origin = np.rint((t_epoch[idx] - int(rec['t0'])) / 300.0).astype(np.int64) + 1
        keep = (origin >= 1) & (origin <= len(bg))
        idx, origin = idx[keep], origin[keep]
        keep = np.isfinite(raw[origin - 1])
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
        b, i, c, t = _cut(bg, ins, carb, origin, ctx, None)
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
