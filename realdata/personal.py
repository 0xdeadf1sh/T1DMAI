"""
Personal-record adapter — parses a T1DMSERVER sync database into canonical Segments.

This is the one adapter whose source already holds the model's channels.  The three
published cohorts log bare amounts (grams eaten, units injected), so
``features.segment_to_channels`` has to reconstruct absorption and action by
convolving them with mean-population kernels.  A phone-authored record instead
stores, per event, the resolved per-five-minute series the phone itself fed the
model: a `custom_curve` where the user built or mixed one, otherwise the parameters
that generate it.  Those series are carried through verbatim on the Segment's
``carb_curve`` / ``insulin_curve`` fields and bypass the kernels entirely, so a
fine-tune here sees the same input distribution the device produces at inference.

Tables read (all read-only, and nothing is ever written):

    samples      ts (epoch ms, on the grid), bg (mg/dL), tz_offset (minutes)
    meal_event   ts, grams, k, theta, duration_min, custom_curve
    dose_event   ts, kind ('bolus'|'basal'), units, duration_min, k, theta,
                 ka_per_hour, ke_per_hour, custom_curve

**Basal is schedule-XOR-discrete, and only the discrete half is implemented.**  The
phone derives its basal contribution from the active daily schedule when one is
configured, and from the logged injections only otherwise — the schedule wins on its
mere presence, not on whether it produced anything in the window.  This adapter reads
the logged injections, so a record carrying an active schedule would lose its entire
background insulin: the largest term in the channel, absent with nothing to show for
it.  Such a record is rejected rather than loaded thinner than it is.  Extending it
means tiling each scheduled slot per local day the way the phone does, against a
record that exercises it.

Curve resolution, in precedence order — an explicit ``custom_curve`` wins and is
never re-derived from the parameters stored beside it:

    custom_curve present  ->  used as stored
    meal, parametric      ->  ``simulator.gamma_curve(grams, k, theta, duration)``
    dose kind='bolus'     ->  ``simulator.gamma_curve(units, k, theta, duration)``
    dose kind='basal'     ->  ``simulator.basal_curve(units, duration, ka, ke)``

Both generators come from the simulator the model was pretrained against, at their
default step and tail-clip, which is also what the phone's own curve engine
reproduces — so no curve mathematics is restated here.

**Timestamps are converted to local wall clock.**  ``Segment.t0`` and therefore
``Segment.hour_of_day`` (the time-of-day probe target) must be local, as they are
for every other adapter, so each sample's UTC epoch is shifted by ``tz_offset``.  A
record whose ``tz_offset`` is not constant is rejected rather than stamped wrong.

**``skip_hours`` is applied after the curves are laid down, not before.**  An event
that precedes the cut still has action reaching past it — a long-acting basal runs a
full day, a rapid bolus some six hours — so the channels are built across the whole
grid and only then trimmed.  Dropping the rows first would open the retained window
with no insulin or carbohydrate on board.

**``exclude`` drops readings, not rows.**  A range listed there removes the CGM
samples inside it and nothing else: the meal and dose rows stay, so their curves are
still laid down and still reach past the cut, exactly as a pre-``skip_hours`` event
keeps its tail.  The hole then reads as an ordinary CGM outage and ``segment_grid``
applies its own rules to it — bridged if it is short enough, a split if it is not,
and the runt either side dropped.  This is how a stretch whose *event log* is wrong
is kept out of training: a long-acting basal taken but never recorded leaves the
insulin channel reading near-zero while the patient was in fact covered, so those
hours cannot be trusted as targets even though every CGM reading in them is sound.
Excluding them states that; nothing about the record is rewritten to hide it.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import numpy as np

from T1DMSIM.simulator import gamma_curve, basal_curve, DT_MINUTES

from .schema import GRID_MIN, Segment, segment_grid

STEP_MS: int = GRID_MIN * 60 * 1000
DEFAULT_SKIP_HOURS: float = 24.0
DEFAULT_PATIENT: str = 'self'
CURVE_SUM_RTOL: float = 1e-6          # a curve must sum to its event total (relative)

assert DT_MINUTES == GRID_MIN, (
    f"simulator step {DT_MINUTES} min != real-data grid {GRID_MIN} min; the curve "
    "generators would emit a series at the wrong resolution"
)


def _local_naive(epoch_ms: int, tz_offset_min: int) -> datetime:
    """Epoch milliseconds -> naive local wall-clock datetime at ``tz_offset_min``."""
    shifted = (epoch_ms + tz_offset_min * 60_000) / 1000.0
    return datetime.fromtimestamp(shifted, tz=timezone.utc).replace(tzinfo=None)


def _resolve_curve(
    total: float, duration_min: float, custom_curve: str | None,
    k: float | None, theta: float | None,
    ka_per_hour: float | None, ke_per_hour: float | None,
    is_basal: bool, what: str,
) -> np.ndarray:
    """Resolve one event to its per-step rate series (length ``duration_min / GRID_MIN``).

    ``what`` is a caller-supplied row label used only in assertion messages.

    Returns:
        (L,) float64 rates whose sum is ``total`` — grams/step for a meal, IU/step
        for a dose.
    """
    if custom_curve is not None:
        values = np.asarray(json.loads(custom_curve), dtype=np.float64)
        assert values.ndim == 1 and values.size > 0, f"{what}: empty custom_curve"
        expected = int(round(duration_min / GRID_MIN))
        assert abs(values.size - expected) <= 1, (
            f"{what}: custom_curve has {values.size} points but duration_min "
            f"{duration_min} implies {expected} five-minute buckets — the stored series "
            "is not on this grid, so laying it down would misplace its tail"
        )
    elif is_basal:
        assert ka_per_hour is not None and ke_per_hour is not None, (
            f"{what}: basal dose has neither a custom_curve nor ka/ke rates"
        )
        values = np.asarray(
            basal_curve(total, duration_min, ka_per_hour, ke_per_hour), dtype=np.float64)
    else:
        assert k is not None and theta is not None, (
            f"{what}: no custom_curve and no gamma parameters (k, theta) — refusing to "
            "invent a shape for it"
        )
        values = np.asarray(gamma_curve(total, k, theta, duration_min), dtype=np.float64)

    assert np.isfinite(values).all(), f"{what}: non-finite value in the resolved curve"
    got = float(values.sum())
    assert abs(got - total) <= CURVE_SUM_RTOL * max(1.0, abs(total)), (
        f"{what}: curve sums to {got} but the event total is {total} — a rate series "
        "must account for exactly its event"
    )
    return values


def _accumulate(grid: np.ndarray, start_idx: int, values: np.ndarray) -> None:
    """Add ``values`` onto ``grid`` from ``start_idx``, clipping to the grid at both ends.

    ``start_idx`` may be negative: an event that began before the grid still
    contributes the part of its curve that reaches into it.
    """
    n = grid.size
    lo = max(0, start_idx)
    hi = min(n, start_idx + values.size)
    if lo < hi:
        grid[lo:hi] += values[lo - start_idx:hi - start_idx]


def _query(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    return conn.execute(sql).fetchall()


def _assert_no_active_basal_schedule(conn: sqlite3.Connection) -> None:
    """Refuse a record whose basal comes from a schedule rather than logged injections.

    Where an active schedule exists the phone builds its basal contribution by tiling
    that schedule and ignores the logged injections entirely, so reading the injections
    would reconstruct a channel the device never fed the model — and where the user
    stopped logging injections once the schedule was set, it would reconstruct almost no
    basal at all. Neither miscount is visible in the result, which is why this fails
    closed instead of warning.
    """
    present = _query(
        conn, "SELECT name FROM sqlite_master WHERE type='table' AND name='basal_schedule_dose'")
    if not present:
        return                                     # older schema: no schedule concept
    rows = _query(conn, "SELECT COUNT(*) FROM basal_schedule_dose WHERE active = 1")
    n_active = int(rows[0][0]) if rows else 0
    assert n_active == 0, (
        f"record carries {n_active} active basal_schedule_dose row(s). The producer treats "
        "basal as schedule-XOR-discrete and prefers the schedule on its presence alone, so "
        "the logged basal injections this adapter reads are not what the model was fed — the "
        "scheduled background would be missing from the insulin channel. Reading a schedule "
        "requires tiling each slot per local day; until then such a record is not loadable."
    )


def load(
    root_dir: str,
    skip_hours: float = DEFAULT_SKIP_HOURS,
    patient: str = DEFAULT_PATIENT,
    exclude: Sequence[tuple[datetime, datetime]] | None = None,
) -> list[Segment]:
    """Load a personal sync database into Segments carrying pre-resolved curves.

    Args:
        root_dir: path to the SQLite file (named for the ``load_dataset`` signature,
            which passes a location; this adapter's location is a file, not a
            directory).
        skip_hours: hours discarded from the START of the record, applied after the
            curves are laid down so pre-cut events keep contributing their tails.
        patient: provenance label written onto every Segment.
        exclude: half-open ``[start, end)`` LOCAL wall-clock ranges whose CGM
            readings are dropped before the grid is built. Events are untouched, so
            the ranges read as CGM outages and are resolved by ``segment_grid``.

    Returns:
        List of Segments (possibly empty), gap-split by the canonical rules, each
        carrying ``carb_curve`` (g/step) and ``insulin_curve`` (IU/step).
    """
    assert os.path.exists(root_dir), f"personal database not found: {root_dir}"
    assert skip_hours >= 0.0, f"skip_hours must be non-negative, got {skip_hours}"
    for lo, hi in (exclude or ()):
        assert lo < hi, f"exclude range {lo} .. {hi} is empty or inverted"

    conn = sqlite3.connect(f"file:{os.path.abspath(root_dir)}?mode=ro", uri=True)
    try:
        _assert_no_active_basal_schedule(conn)
        samples = _query(conn, "SELECT ts, bg, tz_offset FROM samples ORDER BY ts")
        meals = _query(conn, """
            SELECT id, ts, grams, k, theta, duration_min, custom_curve
            FROM meal_event ORDER BY ts""")
        doses = _query(conn, """
            SELECT id, ts, kind, units, duration_min, k, theta,
                   ka_per_hour, ke_per_hour, custom_curve
            FROM dose_event ORDER BY ts""")
    finally:
        conn.close()

    bearing = [(int(ts), float(bg), int(tz)) for ts, bg, tz in samples if bg is not None]
    if len(bearing) < 2:
        return []

    offsets = {tz for _, _, tz in bearing}
    assert len(offsets) == 1, (
        f"record spans more than one tz_offset {sorted(offsets)}; a single local t0 "
        "cannot stamp it, and the time-of-day probe target would be wrong after the "
        "change — split the record per offset before loading"
    )
    tz_offset = bearing[0][2]

    if exclude:
        kept = [row for row in bearing
                if not any(lo <= _local_naive(row[0], tz_offset) < hi for lo, hi in exclude)]
        assert len(kept) >= 2, (
            f"the excluded ranges leave {len(kept)} reading(s) of {len(bearing)}; there "
            "is no record left to lay on a grid"
        )
        bearing = kept

    grid_t0_ms = min(ts for ts, _, _ in bearing)
    grid_end_ms = max(ts for ts, _, _ in bearing)
    for ts, _, _ in bearing:
        assert ts % STEP_MS == 0, (
            f"sample ts {ts} is not on the {GRID_MIN}-minute grid; the record cannot be "
            "laid onto a uniform grid without moving it"
        )
    n = (grid_end_ms - grid_t0_ms) // STEP_MS + 1

    cgm = np.full(n, np.nan, dtype=np.float64)
    for ts, bg, _ in bearing:
        cgm[(ts - grid_t0_ms) // STEP_MS] = bg

    carb_curve = np.zeros(n, dtype=np.float64)
    insulin_curve = np.zeros(n, dtype=np.float64)
    carb_grams = np.zeros(n, dtype=np.float64)
    bolus_units = np.zeros(n, dtype=np.float64)
    basal_rate = np.zeros(n, dtype=np.float64)
    exercise = np.zeros(n, dtype=np.float64)

    for mid, ts, grams, k, theta, duration_min, custom in meals:
        values = _resolve_curve(
            float(grams), float(duration_min), custom, k, theta, None, None,
            is_basal=False, what=f"meal_event id={mid}")
        idx = (int(ts) - grid_t0_ms) // STEP_MS
        _accumulate(carb_curve, idx, values)
        if 0 <= idx < n:
            carb_grams[idx] += float(grams)

    for did, ts, kind, units, duration_min, k, theta, ka, ke, custom in doses:
        is_basal = (kind == 'basal')
        values = _resolve_curve(
            float(units), float(duration_min), custom, k, theta, ka, ke,
            is_basal=is_basal, what=f"dose_event id={did} kind={kind}")
        idx = (int(ts) - grid_t0_ms) // STEP_MS
        _accumulate(insulin_curve, idx, values)
        if is_basal:
            # Raw-channel fallback for the kernel path and the dose figures: a
            # long-acting dose has no logged rate, so spread it flat over its own
            # duration. The pre-resolved insulin_curve above is what the model reads.
            hours = max(float(duration_min) / 60.0, 1e-9)
            _accumulate(basal_rate, idx,
                        np.full(int(round(float(duration_min) / GRID_MIN)),
                                float(units) / hours))
        elif 0 <= idx < n:
            bolus_units[idx] += float(units)

    skip_steps = int(round(skip_hours * 60.0 / GRID_MIN))
    assert skip_steps < n, (
        f"skip_hours={skip_hours} discards all {n} grid steps of the record"
    )
    if skip_steps > 0:
        sl = slice(skip_steps, n)
        cgm, carb_curve, insulin_curve = cgm[sl], carb_curve[sl], insulin_curve[sl]
        carb_grams, bolus_units = carb_grams[sl], bolus_units[sl]
        basal_rate, exercise = basal_rate[sl], exercise[sl]

    t0_local = (_local_naive(grid_t0_ms, tz_offset)
                + timedelta(minutes=GRID_MIN * skip_steps))

    return segment_grid(
        'personal', patient, t0_local, cgm,
        carb_grams, bolus_units, basal_rate, exercise,
        carb_curve=carb_curve, insulin_curve=insulin_curve,
    )
