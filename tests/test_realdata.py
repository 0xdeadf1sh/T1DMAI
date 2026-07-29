"""Tests for the real-data adapters (realdata/).

Two layers:
  * dataset-independent unit tests of the schema's gap-interpolate / gap-split
    and event-binning core (always run);
  * per-dataset invariant checks that load each corpus and assert physiological
    plausibility (skipped when the dataset directory is absent).
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest

from realdata.schema import (
    GRID_MIN, MAX_INTERP_GAP_MIN, MIN_SEGMENT_STEPS, Segment,
    segment_grid, lay_on_grid,
)

_DATA_ROOT = '../T1DMSIM/datasets'


# ----------------------------- schema core ------------------------------- #
def _const(n, v=0.0):
    return np.full(n, v, dtype=np.float64)


def test_small_gap_interpolated():
    """A gap ≤ MAX_INTERP_GAP_MIN is bridged inline, yielding one segment."""
    n = MIN_SEGMENT_STEPS * 2
    cgm = np.linspace(100, 200, n)
    hole = MAX_INTERP_GAP_MIN // GRID_MIN          # exactly the bridgeable limit
    lo = n // 2
    cgm[lo:lo + hole] = np.nan
    segs = segment_grid('t', 'p', datetime(2024, 1, 1), cgm,
                        _const(n), _const(n), _const(n), _const(n))
    print(f"[DUMP] small-gap -> {len(segs)} segment(s), len {len(segs[0]) if segs else 0}")
    assert len(segs) == 1
    assert np.isfinite(segs[0].cgm).all()
    # interpolated midpoint should sit between its finite neighbours
    assert 100 < segs[0].cgm[lo] < 200


def test_large_gap_splits():
    """A gap > MAX_INTERP_GAP_MIN breaks the record into two segments."""
    n = MIN_SEGMENT_STEPS * 3
    cgm = np.linspace(100, 200, n)
    hole = MAX_INTERP_GAP_MIN // GRID_MIN + 5      # un-bridgeable
    lo = n // 2
    cgm[lo:lo + hole] = np.nan
    segs = segment_grid('t', 'p', datetime(2024, 1, 1), cgm,
                        _const(n), _const(n), _const(n), _const(n))
    print(f"[DUMP] large-gap -> {len(segs)} segments, lens {[len(s) for s in segs]}")
    assert len(segs) == 2
    assert all(np.isfinite(s.cgm).all() for s in segs)


def test_runt_segment_dropped():
    """Segments shorter than MIN_SEGMENT_STEPS are discarded."""
    n = MIN_SEGMENT_STEPS + 10
    cgm = np.full(n, 120.0)
    cgm[5:n] = np.nan          # only 5 good steps survive — a runt
    segs = segment_grid('t', 'p', datetime(2024, 1, 1), cgm,
                        _const(n), _const(n), _const(n), _const(n))
    print(f"[DUMP] runt -> {len(segs)} segments")
    assert segs == []


def test_lay_on_grid_bins_events():
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    # two events into step 0, one into step 3, one out of range
    ev = [(t0, 10.0), (datetime(2024, 1, 1, 0, 2, 0), 5.0),
          (datetime(2024, 1, 1, 0, 15, 0), 7.0), (datetime(2023, 12, 31), 99.0)]
    grid = lay_on_grid(t0, 6, ev)
    print(f"[DUMP] grid {grid.tolist()}")
    assert grid[0] == 15.0 and grid[3] == 7.0 and grid.sum() == 22.0


def test_lay_on_grid_rounds_offgrid_events():
    """An OFF-grid event snaps to the NEAREST 5-min step (round), not the floored
    step — so events share the CGM placement convention. This is what the basic
    binning test above cannot catch: its timestamps (0:02, 0:15) land in the same
    cell under both round and floor. 0:13 is 2.6 steps: round → step 3, while the
    old floor would have mis-binned it into step 2 (a 5-min lag vs the CGM)."""
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    grid = lay_on_grid(t0, 6, [(datetime(2024, 1, 1, 0, 13, 0), 7.0)])
    print(f"[DUMP] offgrid grid {grid.tolist()}")
    assert grid[3] == 7.0, "0:13 (2.6 steps) must round to the nearest step 3"
    assert grid[2] == 0.0, "must NOT floor the off-grid event into step 2"
    assert grid.sum() == 7.0
    # A second off-grid event rounding DOWN to the nearer step (0:08 = 1.6 → 2).
    grid2 = lay_on_grid(t0, 6, [(datetime(2024, 1, 1, 0, 8, 0), 3.0)])
    assert grid2[2] == 3.0 and grid2[1] == 0.0, "0:08 (1.6 steps) must round to step 2"


def test_segment_invariants_enforced():
    with pytest.raises(AssertionError):
        Segment('t', 'p', datetime(2024, 1, 1), np.array([100.0, np.nan]),
                _const(2), _const(2), _const(2), _const(2))   # NaN cgm rejected


# --------------------------- per-dataset checks --------------------------- #
def _assert_segment_sane(s: Segment):
    n = len(s)
    assert n >= MIN_SEGMENT_STEPS
    assert np.isfinite(s.cgm).all()
    assert (s.cgm >= 10).all() and (s.cgm <= 600).all(), "CGM out of mg/dL range"
    for ch in (s.carb_grams, s.bolus_units, s.basal_rate, s.exercise):
        assert ch.shape == (n,) and (ch >= 0).all()


@pytest.mark.skipif(not os.path.isdir(f'{_DATA_ROOT}/ohiot1dm'), reason='OhioT1DM absent')
def test_ohio_loads():
    from realdata import ohio
    segs = ohio.load()
    pats = sorted({s.patient for s in segs})
    print(f"[DUMP] Ohio {len(segs)} segs, patients {pats}")
    assert pats == ['559', '563', '570', '575', '588', '591']
    for s in segs:
        _assert_segment_sane(s)
    cgm = np.concatenate([s.cgm for s in segs])
    print(f"[DUMP] Ohio pooled CGM mean {cgm.mean():.0f}, TBR {100*(cgm<70).mean():.1f}%")
    assert 120 < cgm.mean() < 200


@pytest.mark.skipif(not os.path.isdir(f'{_DATA_ROOT}/AZT1D'), reason='AZT1D absent')
def test_azt1d_loads():
    from realdata import azt1d
    segs = azt1d.load()
    print(f"[DUMP] AZT1D {len(segs)} segs, {len({s.patient for s in segs})} subjects")
    assert len(segs) > 0
    for s in segs:
        _assert_segment_sane(s)
        assert s.basal_rate.max() <= azt1d.BASAL_CLIP_IU_PER_H + 1e-6


@pytest.mark.skipif(not os.path.isdir(f'{_DATA_ROOT}/ShanghaiT1DM'), reason='Shanghai absent')
def test_shanghai_loads():
    from realdata import shanghai
    segs = shanghai.load()
    print(f"[DUMP] Shanghai {len(segs)} segs, {len({s.patient for s in segs})} patients")
    assert len(segs) > 0
    for s in segs:
        _assert_segment_sane(s)
    cgm = np.concatenate([s.cgm for s in segs])
    assert 100 < cgm.mean() < 220
