"""Tests for the personal-record adapter (realdata/personal.py) and the day-holdout
split policy (finetune/finetune_personal.py).

Every test builds its own synthetic SQLite record in a tmpdir, so the suite runs on a
fresh clone and never reads a real patient database.

The properties worth pinning are the ones whose failure is silent rather than loud:
a stored curve quietly re-derived from the parameters beside it, an event's action
lost at the ``skip_hours`` boundary, a pre-resolved channel left at full length by a
slice, and a training window overlapping the day the model is scored on.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from realdata.schema import GRID_MIN, MIN_SEGMENT_STEPS, Segment, segment_grid
from realdata.features import segment_to_channels
from realdata.run_eval import _slice
from realdata.personal import load as load_personal
from T1DMSIM.simulator import gamma_curve, basal_curve

STEP_MS = GRID_MIN * 60 * 1000
TZ = 240                                  # minutes east of UTC (the record's offset)
DAY_STEPS = 24 * 60 // GRID_MIN           # 288

_SCHEMA = """
CREATE TABLE samples (
    ts INTEGER PRIMARY KEY, tz_offset INTEGER NOT NULL DEFAULT 0,
    bg REAL, hr REAL, steps REAL, sleep REAL, exercise REAL, mood INTEGER,
    updated_at INTEGER NOT NULL, received_at INTEGER NOT NULL);
CREATE TABLE meal_event (
    id INTEGER PRIMARY KEY, client_id TEXT NOT NULL, ts INTEGER NOT NULL,
    tz_offset INTEGER NOT NULL DEFAULT 0, grams REAL NOT NULL,
    gi REAL, k REAL, theta REAL, duration_min REAL NOT NULL,
    custom_curve TEXT, note TEXT, updated_at INTEGER NOT NULL, received_at INTEGER NOT NULL);
CREATE TABLE dose_event (
    id INTEGER PRIMARY KEY, client_id TEXT NOT NULL, ts INTEGER NOT NULL,
    tz_offset INTEGER NOT NULL DEFAULT 0, kind TEXT NOT NULL, units REAL NOT NULL,
    duration_min REAL NOT NULL, k REAL, theta REAL, ka_per_hour REAL, ke_per_hour REAL,
    custom_curve TEXT, note TEXT, updated_at INTEGER NOT NULL, received_at INTEGER NOT NULL);
"""

_T0 = 1_784_129_400_000                   # on the grid; the record's first sample


def _utc_naive(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).replace(tzinfo=None)


def _build_db(
    path, n_steps=MIN_SEGMENT_STEPS * 8, meals=(), doses=(), bg=None,
    tz=TZ, t0=_T0, bg_nan_at=(),
):
    """Write a synthetic record. ``meals``/``doses`` are dicts of column values."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    trace = (np.full(n_steps, 120.0) if bg is None else np.asarray(bg, dtype=np.float64))
    assert trace.size == n_steps
    for i in range(n_steps):
        ts = t0 + i * STEP_MS
        v = None if i in bg_nan_at else float(trace[i])
        conn.execute("INSERT INTO samples (ts, tz_offset, bg, updated_at, received_at) "
                     "VALUES (?,?,?,?,?)", (ts, tz, v, ts, ts))
    for j, m in enumerate(meals):
        conn.execute(
            "INSERT INTO meal_event (id, client_id, ts, tz_offset, grams, gi, k, theta, "
            "duration_min, custom_curve, updated_at, received_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (j + 1, f"m{j}", m['ts'], tz, m['grams'], m.get('gi'), m.get('k'),
             m.get('theta'), m['duration_min'],
             None if m.get('curve') is None else json.dumps(list(m['curve'])),
             m['ts'], m['ts']))
    for j, d in enumerate(doses):
        conn.execute(
            "INSERT INTO dose_event (id, client_id, ts, tz_offset, kind, units, "
            "duration_min, k, theta, ka_per_hour, ke_per_hour, custom_curve, "
            "updated_at, received_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (j + 1, f"d{j}", d['ts'], tz, d['kind'], d['units'], d['duration_min'],
             d.get('k'), d.get('theta'), d.get('ka_per_hour'), d.get('ke_per_hour'),
             None if d.get('curve') is None else json.dumps(list(d['curve'])),
             d['ts'], d['ts']))
    conn.commit()
    conn.close()
    return str(path)


# --------------------------- curve resolution ----------------------------- #
def test_parametric_meal_matches_simulator_gamma(tmp_path):
    """A meal without a stored curve is shaped by the simulator's own gamma."""
    grams, k, theta, dur = 40.0, 2.0, 15.0, 120.0
    at = 20                                                    # step index of the meal
    db = _build_db(tmp_path / 'r.db',
                   meals=[{'ts': _T0 + at * STEP_MS, 'grams': grams, 'gi': 100.0,
                           'k': k, 'theta': theta, 'duration_min': dur}])
    seg = load_personal(db, skip_hours=0.0)[0]
    want = np.asarray(gamma_curve(grams, k, theta, dur))
    got = seg.carb_curve[at:at + want.size]
    print(f"[DUMP] gamma: sum want {want.sum():.6f} got {got.sum():.6f} "
          f"max|Δ| {np.abs(got - want).max():.3e}")
    assert np.allclose(got, want, atol=1e-12)
    assert seg.carb_curve[:at].sum() == 0.0                    # nothing before the meal


def test_basal_matches_simulator_bateman(tmp_path):
    """A basal dose is a Bateman curve at the simulator's default tail clip."""
    units, dur, ka, ke = 12.0, 1440.0, 0.3, 0.07
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS * 2,
                   doses=[{'ts': _T0, 'kind': 'basal', 'units': units,
                           'duration_min': dur, 'ka_per_hour': ka, 'ke_per_hour': ke}])
    seg = load_personal(db, skip_hours=0.0)[0]
    want = np.asarray(basal_curve(units, dur, ka, ke))
    got = seg.insulin_curve[:want.size]
    print(f"[DUMP] bateman: sum want {want.sum():.6f} got {got.sum():.6f} "
          f"max|Δ| {np.abs(got - want).max():.3e}")
    assert np.allclose(got, want, atol=1e-12)


def test_custom_curve_is_never_re_derived(tmp_path):
    """A stored curve is laid down as-is, even where the parameters beside it disagree.

    The curve here is deliberately nothing a gamma would produce (flat), and k/theta
    are populated with values that would generate something quite different — so a
    re-derivation would be unmistakable.
    """
    dur, n = 60.0, 12
    flat = [2.0] * n                                   # sums to 24 g, uniformly
    db = _build_db(tmp_path / 'r.db',
                   meals=[{'ts': _T0 + 10 * STEP_MS, 'grams': 24.0, 'k': 2.0,
                           'theta': 15.0, 'duration_min': dur, 'curve': flat}])
    seg = load_personal(db, skip_hours=0.0)[0]
    got = seg.carb_curve[10:10 + n]
    print(f"[DUMP] custom: {got}")
    assert np.allclose(got, flat, atol=1e-12)
    assert not np.allclose(got, np.asarray(gamma_curve(24.0, 2.0, 15.0, dur)))


def test_curve_not_summing_to_total_is_rejected(tmp_path):
    """A stored series that does not account for its event is a defect, not input."""
    db = _build_db(tmp_path / 'r.db',
                   meals=[{'ts': _T0 + 10 * STEP_MS, 'grams': 30.0,
                           'duration_min': 60.0, 'curve': [1.0] * 12}])   # sums to 12
    with pytest.raises(AssertionError, match="curve sums to"):
        load_personal(db, skip_hours=0.0)


def test_bolus_without_curve_or_parameters_is_rejected(tmp_path):
    """No shape is invented for a dose that carries neither a curve nor gamma params."""
    db = _build_db(tmp_path / 'r.db',
                   doses=[{'ts': _T0 + 10 * STEP_MS, 'kind': 'bolus', 'units': 5.0,
                           'duration_min': 360.0}])
    with pytest.raises(AssertionError, match="refusing to invent"):
        load_personal(db, skip_hours=0.0)


def test_bolus_and_basal_share_one_insulin_channel(tmp_path):
    """Both dose kinds sum into ``insulin_curve`` — the model reads one channel."""
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS * 2,
                   doses=[{'ts': _T0, 'kind': 'basal', 'units': 10.0,
                           'duration_min': 1440.0, 'ka_per_hour': 0.3, 'ke_per_hour': 0.07},
                          {'ts': _T0 + 20 * STEP_MS, 'kind': 'bolus', 'units': 4.0,
                           'duration_min': 360.0, 'k': 3.0, 'theta': 25.0}])
    seg = load_personal(db, skip_hours=0.0)[0]
    total = float(seg.insulin_curve.sum())
    print(f"[DUMP] insulin laid down {total:.4f} U of 14.0 U logged")
    assert total == pytest.approx(14.0, abs=1e-6)


# ------------------------------ the skip cut ------------------------------ #
def test_pre_cut_event_keeps_its_tail(tmp_path):
    """An event before the cut still delivers the action that reaches past it.

    This is the whole reason the cut is applied after the curves are laid down. A
    basal taken at hour 0 with a 24 h duration must still be acting at hour 12.
    """
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS * 3,
                   doses=[{'ts': _T0, 'kind': 'basal', 'units': 12.0,
                           'duration_min': 1440.0, 'ka_per_hour': 0.3, 'ke_per_hour': 0.07}])
    full = load_personal(db, skip_hours=0.0)[0]
    cut = load_personal(db, skip_hours=12.0)[0]
    half = DAY_STEPS // 2
    print(f"[DUMP] insulin at first retained step: {cut.insulin_curve[0]:.6f} U/step "
          f"(full grid step {half}: {full.insulin_curve[half]:.6f})")
    assert cut.insulin_curve[0] > 0.0
    assert np.allclose(cut.insulin_curve, full.insulin_curve[half:], atol=1e-12)
    assert len(cut) == len(full) - half


def test_skip_advances_t0_and_trims_every_channel(tmp_path):
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS * 2)
    full = load_personal(db, skip_hours=0.0)[0]
    cut = load_personal(db, skip_hours=6.0)[0]
    steps = 6 * 60 // GRID_MIN
    assert cut.t0 == full.t0 + timedelta(hours=6)
    for name in ('cgm', 'carb_grams', 'bolus_units', 'basal_rate', 'exercise',
                 'carb_curve', 'insulin_curve'):
        arr = getattr(cut, name)
        assert arr.shape == (len(full) - steps,), f"{name} not trimmed"


def test_skip_consuming_the_record_is_rejected(tmp_path):
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS)
    with pytest.raises(AssertionError, match="discards all"):
        load_personal(db, skip_hours=48.0)


# ------------------------------- exclusion -------------------------------- #
def test_exclude_splits_the_record_and_keeps_event_action(tmp_path):
    """An excluded range reads as a CGM outage; events inside it keep their curves.

    The dose sits INSIDE the hole, so a filter that dropped event rows rather than
    readings would erase its action from the segment that follows — the same miscount
    the ``skip_hours`` tail rule exists to prevent.
    """
    n = DAY_STEPS * 3
    db = _build_db(tmp_path / 'r.db', n_steps=n, doses=[
        {'ts': _T0 + (DAY_STEPS + 30) * STEP_MS, 'kind': 'basal', 'units': 14.0,
         'duration_min': 1440.0, 'ka_per_hour': 0.3, 'ke_per_hour': 0.07}])
    t0_local = _utc_naive(_T0) + timedelta(minutes=TZ)
    lo = t0_local + timedelta(days=1)
    hi = lo + timedelta(days=1)

    segs = load_personal(db, skip_hours=0.0, exclude=[(lo, hi)])
    assert len(segs) == 2, "a day-long hole must break the record in two"
    assert segs[0].t0 == t0_local
    assert segs[1].t0 == hi
    assert segs[1].insulin_curve.sum() > 0.0, "the excluded dose lost its tail"


def test_exclude_leaves_retained_readings_bit_identical(tmp_path):
    """Excluding a range must not perturb a single retained sample or channel value."""
    n = DAY_STEPS * 3
    rng = np.random.default_rng(0)
    bg = 120.0 + 30.0 * np.sin(np.arange(n) / 17.0) + rng.normal(0, 2.0, n)
    db = _build_db(tmp_path / 'r.db', n_steps=n, bg=bg, meals=[
        {'ts': _T0 + 5 * STEP_MS, 'grams': 40.0, 'k': 2.0, 'theta': 15.0,
         'duration_min': 120.0}])
    t0_local = _utc_naive(_T0) + timedelta(minutes=TZ)
    lo = t0_local + timedelta(days=2)

    full = load_personal(db, skip_hours=0.0)[0]
    cut = load_personal(db, skip_hours=0.0, exclude=[(lo, lo + timedelta(days=1))])[0]
    assert cut.t0 == full.t0
    for name in ('cgm', 'carb_curve', 'insulin_curve'):
        kept = getattr(cut, name)
        np.testing.assert_array_equal(kept, getattr(full, name)[:len(kept)])


def test_exclude_consuming_the_record_is_rejected(tmp_path):
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS)
    t0_local = _utc_naive(_T0) + timedelta(minutes=TZ)
    with pytest.raises(AssertionError, match="no record left"):
        load_personal(db, skip_hours=0.0, exclude=[
            (t0_local - timedelta(days=1), t0_local + timedelta(days=2))])


def test_inverted_exclude_range_is_rejected(tmp_path):
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS)
    t0_local = _utc_naive(_T0) + timedelta(minutes=TZ)
    with pytest.raises(AssertionError, match="empty or inverted"):
        load_personal(db, skip_hours=0.0,
                      exclude=[(t0_local + timedelta(hours=2), t0_local)])


# ------------------------------- timestamps ------------------------------- #
def test_t0_is_local_wall_clock(tmp_path):
    """``t0`` (and so the time-of-day probe target) carries the record's tz_offset."""
    db = _build_db(tmp_path / 'r.db', tz=TZ)
    seg = load_personal(db, skip_hours=0.0)[0]
    utc = _utc_naive(_T0)
    print(f"[DUMP] utc {utc}  ->  local t0 {seg.t0}  (tz_offset {TZ} min)")
    assert seg.t0 == utc + timedelta(minutes=TZ)
    assert seg.hour_of_day()[0] == pytest.approx(seg.t0.hour + seg.t0.minute / 60.0)


def test_mixed_tz_offset_is_rejected(tmp_path):
    """A record whose offset changes cannot be stamped by a single local t0."""
    db = _build_db(tmp_path / 'r.db')
    conn = sqlite3.connect(db)
    conn.execute("UPDATE samples SET tz_offset = 180 WHERE ts > ?", (_T0 + 50 * STEP_MS,))
    conn.commit()
    conn.close()
    with pytest.raises(AssertionError, match="more than one tz_offset"):
        load_personal(db, skip_hours=0.0)


def test_off_grid_sample_is_rejected(tmp_path):
    db = _build_db(tmp_path / 'r.db')
    conn = sqlite3.connect(db)
    conn.execute("UPDATE samples SET ts = ts + 60000 WHERE ts = ?", (_T0 + 5 * STEP_MS,))
    conn.commit()
    conn.close()
    with pytest.raises(AssertionError, match="not on the"):
        load_personal(db, skip_hours=0.0)


def test_null_bg_rows_do_not_anchor_the_grid(tmp_path):
    """The grid starts at the first BG-BEARING sample, not the first row."""
    db = _build_db(tmp_path / 'r.db', bg_nan_at=(0, 1))
    seg = load_personal(db, skip_hours=0.0)[0]
    assert seg.t0 == _utc_naive(_T0) + timedelta(minutes=TZ + 2 * GRID_MIN)


# ------------------------- the pre-resolved channels ---------------------- #
def test_features_prefers_pre_resolved_curves(tmp_path):
    """``segment_to_channels`` returns the stored channels instead of convolving."""
    db = _build_db(tmp_path / 'r.db',
                   meals=[{'ts': _T0 + 10 * STEP_MS, 'grams': 30.0, 'k': 2.0,
                           'theta': 15.0, 'duration_min': 120.0}])
    seg = load_personal(db, skip_hours=0.0)[0]
    ch = segment_to_channels(seg)
    assert np.allclose(ch['carb'], seg.carb_curve, atol=1e-12)
    assert np.allclose(ch['insulin'], seg.insulin_curve, atol=1e-12)


def test_published_cohort_path_still_convolves():
    """A Segment without pre-resolved curves takes the kernel path, unchanged."""
    n = MIN_SEGMENT_STEPS * 2
    carb = np.zeros(n); carb[10] = 50.0
    seg = Segment('t', 'p', datetime(2024, 1, 1), np.full(n, 120.0), carb,
                  np.zeros(n), np.zeros(n), np.zeros(n))
    assert seg.carb_curve is None
    ch = segment_to_channels(seg)
    print(f"[DUMP] kernel path: peak {ch['carb'].max():.4f} at step {ch['carb'].argmax()}")
    assert ch['carb'][10] < 50.0                  # spread by the kernel, not a spike
    assert ch['carb'].sum() == pytest.approx(50.0, rel=1e-6)


def test_slice_keeps_curves_aligned(tmp_path):
    """``_slice`` must trim the pre-resolved channels with everything else.

    Left un-named in ``replace``, they would survive at full length and silently
    misalign against the sliced CGM — every downstream feature stack would then read
    carbohydrate and insulin from the wrong instants.
    """
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS * 2,
                   meals=[{'ts': _T0 + 100 * STEP_MS, 'grams': 30.0, 'k': 2.0,
                           'theta': 15.0, 'duration_min': 120.0}])
    seg = load_personal(db, skip_hours=0.0)[0]
    a, b = 60, 300
    sub = _slice(seg, a, b)
    print(f"[DUMP] sliced {len(sub)} steps; curve len {sub.carb_curve.shape[0]}")
    assert sub.carb_curve.shape == (b - a,)
    assert sub.insulin_curve.shape == (b - a,)
    assert np.allclose(sub.carb_curve, seg.carb_curve[a:b], atol=1e-12)


def test_segment_rejects_one_sided_curves():
    n = MIN_SEGMENT_STEPS
    with pytest.raises(AssertionError, match="supplied together"):
        Segment('t', 'p', datetime(2024, 1, 1), np.full(n, 120.0), np.zeros(n),
                np.zeros(n), np.zeros(n), np.zeros(n), carb_curve=np.zeros(n))


def test_segment_rejects_misaligned_curve_length():
    n = MIN_SEGMENT_STEPS
    with pytest.raises(AssertionError, match="!= cgm length"):
        Segment('t', 'p', datetime(2024, 1, 1), np.full(n, 120.0), np.zeros(n),
                np.zeros(n), np.zeros(n), np.zeros(n),
                carb_curve=np.zeros(n - 1), insulin_curve=np.zeros(n - 1))


def test_segment_grid_slices_curves_per_segment():
    """A gap-split record carries the right slice of each pre-resolved channel."""
    n = MIN_SEGMENT_STEPS * 4
    cgm = np.full(n, 120.0)
    cgm[n // 2:n // 2 + 20] = np.nan                       # > MAX_INTERP_GAP_MIN -> split
    ramp = np.arange(n, dtype=np.float64)
    segs = segment_grid('t', 'p', datetime(2024, 1, 1), cgm, np.zeros(n), np.zeros(n),
                        np.zeros(n), np.zeros(n),
                        carb_curve=ramp, insulin_curve=ramp * 2.0)
    assert len(segs) == 2
    for s in segs:
        assert s.carb_curve.shape == (len(s),)
        assert np.allclose(s.insulin_curve, s.carb_curve * 2.0)
    print(f"[DUMP] split -> {[len(s) for s in segs]}, "
          f"curve starts {[float(s.carb_curve[0]) for s in segs]}")
    assert segs[0].carb_curve[0] == 0.0
    assert segs[1].carb_curve[0] == float(n // 2 + 20)


def test_augment_refuses_a_curve_bearing_segment(tmp_path):
    """Augmentation rewrites raw events, which pre-resolved curves would override."""
    from metrics.augmented.augment import augment_segment
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS * 2)
    seg = load_personal(db, skip_hours=0.0)[0]
    with pytest.raises(AssertionError, match="pre-resolved curves"):
        augment_segment(seg)


# --------------------------- the day-holdout split ------------------------ #
def _personal_module():
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'finetune'))
    import finetune_personal
    return finetune_personal


def _six_day_db(tmp_path, sds):
    """A record of ``len(sds)`` whole local days, each with a prescribed BG spread.

    ``t0`` is placed at local midnight so day boundaries land on exact multiples of
    ``DAY_STEPS``, keeping the expected spans easy to state.
    """
    midnight_utc = 1_784_073_600_000 - TZ * 60_000       # local 2026-07-15 00:00
    rng = np.random.default_rng(0)
    trace = np.concatenate([120.0 + rng.normal(0.0, sd, DAY_STEPS) for sd in sds])
    return _build_db(tmp_path / 'days.db', n_steps=len(sds) * DAY_STEPS,
                     bg=trace, t0=midnight_utc)


def test_day_spans_are_whole_local_days(tmp_path):
    P = _personal_module()
    db = _six_day_db(tmp_path, [10.0] * 4)
    seg = load_personal(db, skip_hours=0.0)[0]
    spans = P.day_spans(seg)
    print(f"[DUMP] spans {[(str(d), lo, hi) for d, lo, hi in spans]}")
    assert [(lo, hi) for _d, lo, hi in spans] == [
        (i * DAY_STEPS, (i + 1) * DAY_STEPS) for i in range(4)]
    assert spans[0][0] == date(2026, 7, 15)


def test_partial_days_are_not_listed(tmp_path):
    """Half a day at the end is no day at all."""
    P = _personal_module()
    db = _six_day_db(tmp_path, [10.0] * 3)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM samples WHERE ts > (SELECT MIN(ts) FROM samples) + ?",
                 ((3 * DAY_STEPS - 40) * STEP_MS,))
    conn.commit()
    conn.close()
    seg = load_personal(db, skip_hours=0.0)[0]
    assert len(P.day_spans(seg)) == 2


def test_median_day_is_chosen_not_the_calmest(tmp_path):
    """The held-out day sits nearest the record's median variability.

    Six days of engineered spread: the choice must be neither the flattest nor the
    most volatile, and must have two full days before it.
    """
    P = _personal_module()
    sds = [4.0, 40.0, 8.0, 30.0, 16.0, 22.0]           # day index 0..5
    db = _six_day_db(tmp_path, sds)
    segs = load_personal(db, skip_hours=0.0)
    sp = P.build_splits(segs, 'cv')
    ranked = sp['ranked']
    chaos = {d: c for d, _lo, _hi, c in ranked}
    med = float(np.median(list(chaos.values())))
    print(f"[DUMP] cv per day {[(str(d), round(c, 2)) for d, c in chaos.items()]}")
    print(f"[DUMP] median {med:.2f} -> val_day {sp['val_day']} cal_day {sp['cal_day']}")
    eligible = [d for d, lo, _hi, _c in ranked if lo >= 2 * DAY_STEPS]
    assert sp['val_day'] in eligible
    best = min(eligible, key=lambda d: abs(chaos[d] - med))
    assert sp['val_day'] == best
    assert sp['val_day'] != min(chaos, key=chaos.__getitem__)   # not the calmest
    assert sp['val_day'] != max(chaos, key=chaos.__getitem__)   # not the most chaotic
    assert sp['cal_day'] == sp['val_day'] - timedelta(days=1)


def test_reservation_never_overlaps_training(tmp_path):
    """No step the model trains on is read by a calibration or scoring window.

    The reserved set is each held-out day plus the day before it, which its windows
    consume as context; the training spans are the complement.
    """
    P = _personal_module()
    db = _six_day_db(tmp_path, [4.0, 40.0, 8.0, 30.0, 16.0, 22.0])
    segs = load_personal(db, skip_hours=0.0)
    sp = P.build_splits(segs, 'cv')
    reserved = set()
    for lo, hi in (sp['cal_block'], sp['val_block']):
        reserved |= set(range(lo, hi))
    train = set()
    for a, b in sp['train_spans']:
        train |= set(range(a, b))
    print(f"[DUMP] reserved {len(reserved)} train {len(train)} "
          f"overlap {len(reserved & train)} of {sp['host_steps']}")
    assert not (reserved & train)
    assert len(sp['cal'][0]) == 2 * DAY_STEPS
    assert len(sp['test'][0]) == 2 * DAY_STEPS


def test_eval_segments_end_on_their_named_day(tmp_path):
    """The scored windows' prediction zones fall inside the held-out day itself."""
    P = _personal_module()
    db = _six_day_db(tmp_path, [4.0, 40.0, 8.0, 30.0, 16.0, 22.0])
    segs = load_personal(db, skip_hours=0.0)
    sp = P.build_splits(segs, 'cv')
    for role, day in (('cal', sp['cal_day']), ('test', sp['val_day'])):
        s = sp[role][0]
        last = (s.t0 + timedelta(minutes=GRID_MIN * (len(s) - 1))).date()
        first = s.t0.date()
        print(f"[DUMP] {role} segment {first} .. {last} (named {day})")
        assert last == day
        assert first == day - timedelta(days=1)     # the context day precedes it


def test_val_day_override_requires_two_leading_days(tmp_path):
    P = _personal_module()
    db = _six_day_db(tmp_path, [10.0] * 5)
    segs = load_personal(db, skip_hours=0.0)
    with pytest.raises(AssertionError, match="two preceding full days"):
        P.build_splits(segs, 'cv', val_day=date(2026, 7, 16))


def test_unknown_val_day_is_reported_with_the_candidates(tmp_path):
    P = _personal_module()
    db = _six_day_db(tmp_path, [10.0] * 5)
    segs = load_personal(db, skip_hours=0.0)
    with pytest.raises(SystemExit, match="complete local day"):
        P.build_splits(segs, 'cv', val_day=date(2030, 1, 1))


def test_too_short_a_record_cannot_hold_out_a_day(tmp_path):
    P = _personal_module()
    db = _six_day_db(tmp_path, [10.0] * 2)
    segs = load_personal(db, skip_hours=0.0)
    with pytest.raises(AssertionError, match="leak-free reservation"):
        P.build_splits(segs, 'cv')


def test_chaos_metrics_rank_a_flat_day_below_a_volatile_one(tmp_path):
    P = _personal_module()
    rng = np.random.default_rng(1)
    flat = 120.0 + rng.normal(0.0, 2.0, DAY_STEPS)
    wild = 120.0 + rng.normal(0.0, 45.0, DAY_STEPS)
    for metric in P.CHAOS_METRICS:
        lo, hi = P.day_chaos(flat, metric), P.day_chaos(wild, metric)
        print(f"[DUMP] {metric}: flat {lo:.3f} < wild {hi:.3f}")
        assert lo < hi


def test_unknown_chaos_metric_is_rejected():
    P = _personal_module()
    with pytest.raises(AssertionError, match="unknown chaos metric"):
        P.day_chaos(np.full(DAY_STEPS, 120.0), 'entropy')


def test_cal_day_after_val_day_is_rejected(tmp_path):
    """A later calibration day would read the scored day as its own context."""
    P = _personal_module()
    db = _six_day_db(tmp_path, [4.0, 40.0, 8.0, 30.0, 16.0, 22.0])
    segs = load_personal(db, skip_hours=0.0)
    with pytest.raises(AssertionError, match="must precede"):
        P.build_splits(segs, 'cv', val_day=date(2026, 7, 18), cal_day=date(2026, 7, 20))


# ------------------- the median spans the record, not one run -------------- #
def _split_db(tmp_path, runs):
    """A record broken into separate segments by outages longer than the bridgeable gap.

    ``runs`` is a list of per-run lists of daily BG standard deviations. Runs are parted
    by a whole missing day, well past MAX_INTERP_GAP_MIN, so each becomes its own Segment.
    """
    midnight_utc = 1_784_073_600_000 - TZ * 60_000
    rng = np.random.default_rng(3)
    trace, nan_at, i = [], [], 0
    for r_i, run in enumerate(runs):
        if r_i > 0:
            nan_at.extend(range(i, i + DAY_STEPS))          # a whole day of dropout
            trace.append(np.full(DAY_STEPS, 120.0))
            i += DAY_STEPS
        for sd in run:
            trace.append(120.0 + rng.normal(0.0, sd, DAY_STEPS))
            i += DAY_STEPS
    flat = np.concatenate(trace)
    return _build_db(tmp_path / 'split.db', n_steps=flat.size, bg=flat,
                     t0=midnight_utc, bg_nan_at=tuple(nan_at))


def test_outage_splits_the_record_into_several_segments(tmp_path):
    db = _split_db(tmp_path, [[10.0] * 3, [10.0] * 4])
    segs = load_personal(db, skip_hours=0.0)
    print(f"[DUMP] segments {[len(s) for s in segs]}")
    assert len(segs) == 2


def test_median_counts_days_outside_the_longest_run(tmp_path):
    """A day hidden in a shorter segment still shapes what counts as average.

    The long run is uniformly turbulent and the short one calm. Ranking only the long
    run would put the median inside the turbulent band and hold out a turbulent day;
    the record's true median sits far lower.
    """
    P = _personal_module()
    db = _split_db(tmp_path, [[2.0] * 3, [50.0, 50.0, 20.0, 45.0]])
    segs = load_personal(db, skip_hours=0.0)
    sp = P.build_splits(segs, 'cv')
    host = max(segs, key=len)
    host_only = float(np.median([c for _d, _lo, _hi, c in P.rank_days(host, 'cv')]))
    print(f"[DUMP] host-only median {host_only:.2f} vs record median {sp['median']:.2f}")
    print(f"[DUMP] off-host days {[(str(d), round(c, 2)) for _i, d, c in sp['off_host']]}")
    assert len(sp['off_host']) == 3                       # the calm run is scored
    assert sp['median'] < host_only                       # and pulls the median down
    # The chosen day is the eligible host day nearest the RECORD median, not the host's.
    ranked = P.rank_days(host, 'cv')
    eligible = [(d, c) for d, lo, _hi, c in ranked if lo >= 2 * DAY_STEPS]
    assert sp['val_day'] == min(eligible, key=lambda t: (abs(t[1] - sp['median']), t[0]))[0]


def test_off_host_days_are_training_only(tmp_path):
    """Days outside the host are never reserved; their whole segment trains."""
    P = _personal_module()
    db = _split_db(tmp_path, [[2.0] * 3, [50.0, 50.0, 20.0, 45.0]])
    segs = load_personal(db, skip_hours=0.0)
    sp = P.build_splits(segs, 'cv')
    host = max(segs, key=len)
    assert sp['n_other_segs'] == 1
    for s in sp['cal'] + sp['test']:
        assert len(s) == 2 * DAY_STEPS
    non_host_lens = sorted(len(s) for s in segs if s is not host)
    assert non_host_lens[0] in sorted(len(s) for s in sp['train'])


# --------------------------- basal schedule XOR --------------------------- #
_SCHEDULE_TABLE = """
CREATE TABLE basal_schedule_dose (
    id INTEGER PRIMARY KEY, client_id TEXT NOT NULL, schedule_id TEXT NOT NULL,
    label TEXT NOT NULL, time_of_day_min INTEGER NOT NULL, dose_u REAL NOT NULL,
    duration_min REAL NOT NULL, ka_per_hour REAL NOT NULL, ke_per_hour REAL NOT NULL,
    tz_offset INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL,
    updated_at INTEGER NOT NULL, received_at INTEGER NOT NULL);
"""


def _add_schedule(db, active, dose_u=14.0, tod_min=1320):
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEDULE_TABLE)
    conn.execute(
        "INSERT INTO basal_schedule_dose (id, client_id, schedule_id, label, "
        "time_of_day_min, dose_u, duration_min, ka_per_hour, ke_per_hour, tz_offset, "
        "active, updated_at, received_at) VALUES (1,'s','sched','night',?,?,1440,0.3,0.07,?,?,0,0)",
        (tod_min, dose_u, TZ, 1 if active else 0))
    conn.commit()
    conn.close()


def test_active_basal_schedule_is_rejected(tmp_path):
    """A scheduled basal background must not be silently dropped.

    The producer prefers an active schedule over the logged injections on its presence
    alone, so reading the injections would reconstruct a channel it never fed the model —
    and where the user stopped logging injections, almost no basal at all. Neither
    miscount shows up in the output, so this fails closed.
    """
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS * 2,
                   doses=[{'ts': _T0, 'kind': 'bolus', 'units': 5.0,
                           'duration_min': 360.0, 'k': 3.0, 'theta': 25.0}])
    _add_schedule(db, active=True)
    with pytest.raises(AssertionError, match="active basal_schedule_dose"):
        load_personal(db, skip_hours=0.0)


def test_inactive_basal_schedule_is_allowed(tmp_path):
    """A superseded schedule is not the live representation and does not block loading."""
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS * 2,
                   doses=[{'ts': _T0, 'kind': 'basal', 'units': 10.0,
                           'duration_min': 1440.0, 'ka_per_hour': 0.3, 'ke_per_hour': 0.07}])
    _add_schedule(db, active=False)
    seg = load_personal(db, skip_hours=0.0)[0]
    assert seg.insulin_curve.sum() == pytest.approx(10.0, abs=1e-6)


def test_absent_schedule_table_is_allowed(tmp_path):
    """An older record with no schedule concept at all loads unchanged."""
    db = _build_db(tmp_path / 'r.db', n_steps=DAY_STEPS * 2)
    seg = load_personal(db, skip_hours=0.0)[0]
    assert len(seg) > 0


# ------------------------- the exported model card ------------------------ #
def test_model_card_does_not_call_a_personal_finetune_a_cohort_eval():
    """A held-out DAY of one record must not be labelled a cohort eval with patients withheld."""
    from exporters.descriptor import _finetune_reference_metrics
    best = {'30': {'rmse_point': 20.3, 'mard': 12.7},
            '60': {'rmse_point': 30.7, 'mard': 21.9},
            '120': {'rmse_point': 39.1, 'mard': 27.8}}
    cohort = _finetune_reference_metrics(best, {'dataset': 'ohiot1dm', 'holdout': '591'})
    personal = _finetune_reference_metrics(best, {'dataset': 'personal',
                                                 'val_day': '2026-07-21'})
    print(f"[DUMP] cohort source={cohort['source']!r} personal source={personal['source']!r}")
    assert cohort['source'] == 'real-heldout'
    assert personal['source'] == 'personal-heldout-day'
    assert 'patients withheld' not in personal['note']
    assert 'one day of one record' in personal['note']
    assert personal['rmse_mgdl'] == cohort['rmse_mgdl']        # same numbers, honest label


def test_model_card_reports_the_held_out_day_as_the_holdout():
    """Without the fallback the card would report holdout: null for a personal finetune."""
    import torch
    from exporters.descriptor import build_model_card
    from model import T1DMAI
    ckpt = {'finetune_meta': {'dataset': 'personal', 'mode': 'personalize',
                              'val_day': '2026-07-21', 'cal_day': '2026-07-20',
                              'total_steps': 400, 'lr_scale': 0.05,
                              'best_heldout': {'60': {'rmse_point': 30.7, 'mard': 21.9}}},
            'step': 400}
    with torch.no_grad():
        card = build_model_card(T1DMAI(), ckpt)
    print(f"[DUMP] finetune block {card['finetune']}")
    assert card['finetune']['holdout'] == '2026-07-21'
    assert card['reference_metrics']['source'] == 'personal-heldout-day'
