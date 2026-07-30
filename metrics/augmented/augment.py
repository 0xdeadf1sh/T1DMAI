"""
Reconstruct the meals and boluses that real CGM records omit, and inject them
into a Segment's raw event channels so the announced-event evaluation has a
complete driver record.

Real cohorts log only a fraction of the events that drive their excursions:
roughly half of hyperglycemic onsets carry no logged meal, and many
hypoglycemic onsets carry no logged bolus (``scratch/realdata_completeness.py``).
Where a clinical excursion has no logged cause, this module injects a synthetic
one — physiology says a hyper onset is meal-driven and a hypo onset is
insulin-driven — sized from the observed CGM swing:

    hyper onset (BG crosses above HYPER) with no logged carb in the prior 2 h
        -> a carbohydrate event LEAD steps before the onset, grams proportional
           to the subsequent rise (CARB_MGDL_PER_GRAM), clamped to a meal range.
    hypo onset (BG crosses below HYPO) with no logged bolus in the prior 2 h
        -> a bolus LEAD steps before the onset, units proportional to the
           subsequent fall (INSULIN_MGDL_PER_UNIT), clamped to a bolus range.

The injected amount is read off the same CGM the model is later scored against,
so an evaluation on the augmented record is an UPPER BOUND on the announced-event
regime (the score attainable if every meal and dose were logged and declared) —
not a deployment number. The raw channels are the only thing changed; CGM is
untouched.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD
from realdata.schema import Segment, GRID_MIN

STEPS_PER_HOUR = 60 // GRID_MIN
LOOKBACK_STEPS = 2 * STEPS_PER_HOUR              # 2 h "is this excursion already explained?" window
ABSORB_STEPS = 2 * STEPS_PER_HOUR               # window over which the rise/fall magnitude is measured

# Population dose→BG factors used to size an injected event from the CGM swing it
# must explain (clinical rules of thumb: ~5 mg/dL per gram of carbohydrate, ~50
# mg/dL per unit of rapid insulin). Clamps keep an injected event in a plausible
# single-meal / single-bolus range.
CARB_MGDL_PER_GRAM = 5.0
INSULIN_MGDL_PER_UNIT = 50.0
CARB_GRAMS_RANGE = (15.0, 90.0)
BOLUS_UNITS_RANGE = (0.5, 8.0)

# An ingested meal precedes its CGM rise; an injected dose precedes its fall.
CARB_LEAD_STEPS = 3                             # 15 min
BOLUS_LEAD_STEPS = 6                            # 30 min


def _onsets(cgm: np.ndarray, thr: float, below: bool) -> np.ndarray:
    """Indices where ``cgm`` crosses ``thr`` (downward if ``below`` else upward)."""
    c = np.asarray(cgm)
    if below:
        return np.where((c[1:] < thr) & (c[:-1] >= thr))[0] + 1
    return np.where((c[1:] > thr) & (c[:-1] <= thr))[0] + 1


def _clamp(x: float, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    return float(min(hi, max(lo, x)))


def augment_segment(seg: Segment) -> Segment:
    """Inject reconstructed meals/boluses into a Segment's raw event channels.

    Returns a new Segment with ``carb_grams`` / ``bolus_units`` augmented (every
    other field shared with the input). The check "is this excursion already
    explained?" reads the RUNNING channel (logged + previously injected events),
    so onsets within ``LOOKBACK_STEPS`` of each other are not double-filled.

    Augmentation is expressed in the RAW event channels, so it only reaches the
    model through the bridge's kernels. A Segment carrying pre-resolved
    ``carb_curve`` / ``insulin_curve`` bypasses those kernels entirely, which would
    make an injected event invisible and yield a report labelled augmented that is
    not — rejected here rather than discovered in the numbers.

    Args:
        seg: a real-data Segment (length ``N``) with no pre-resolved curves.

    Returns:
        Segment with the same ``cgm`` and an augmented ``carb_grams`` (``N,``) /
        ``bolus_units`` (``N,``).
    """
    assert seg.carb_curve is None and seg.insulin_curve is None, (
        f"cannot augment {seg.dataset}/{seg.patient}: the Segment carries pre-resolved "
        "curves, which take precedence over the raw events augmentation rewrites"
    )
    n = len(seg)
    cgm = seg.cgm
    carb = seg.carb_grams.copy()
    bolus = seg.bolus_units.copy()

    for t in _onsets(cgm, BG_HYPER_THRESHOLD, below=False):
        if carb[max(0, t - LOOKBACK_STEPS):t].sum() > 0:
            continue
        rise = float(np.max(cgm[t:t + ABSORB_STEPS]) - cgm[max(0, t - 1)])
        if rise <= 0:
            continue
        grams = _clamp(rise / CARB_MGDL_PER_GRAM, CARB_GRAMS_RANGE)
        carb[max(0, t - CARB_LEAD_STEPS)] += grams

    for t in _onsets(cgm, BG_HYPO_THRESHOLD, below=True):
        if bolus[max(0, t - LOOKBACK_STEPS):t].sum() > 0:
            continue
        fall = float(cgm[max(0, t - 1)] - np.min(cgm[t:t + ABSORB_STEPS]))
        if fall <= 0:
            continue
        units = _clamp(fall / INSULIN_MGDL_PER_UNIT, BOLUS_UNITS_RANGE)
        bolus[max(0, t - BOLUS_LEAD_STEPS)] += units

    assert carb.shape == (n,) and bolus.shape == (n,)
    return replace(seg, carb_grams=carb, bolus_units=bolus)


def _summary(name: str) -> None:
    """Print per-dataset injection counts (how much the record was completed)."""
    from realdata import load_dataset
    segs = load_dataset(name)
    days = sum(len(s) for s in segs) / (24 * STEPS_PER_HOUR)
    carb_before = sum(int((s.carb_grams > 0).sum()) for s in segs)
    bolus_before = sum(int((s.bolus_units > 0).sum()) for s in segs)
    aug = [augment_segment(s) for s in segs]
    carb_after = sum(int((s.carb_grams > 0).sum()) for s in aug)
    bolus_after = sum(int((s.bolus_units > 0).sum()) for s in aug)
    g_inj = sum(float(a.carb_grams.sum() - s.carb_grams.sum()) for a, s in zip(aug, segs))
    u_inj = sum(float(a.bolus_units.sum() - s.bolus_units.sum()) for a, s in zip(aug, segs))
    print(f"\n=== {name} ({days:.0f} patient-days) ===")
    print(f"  carb events {carb_before} -> {carb_after}  (+{carb_after - carb_before}, "
          f"{g_inj:.0f} g injected, {g_inj/max(days,1):.0f} g/day)")
    print(f"  bolus events {bolus_before} -> {bolus_after}  (+{bolus_after - bolus_before}, "
          f"{u_inj:.1f} U injected, {u_inj/max(days,1):.1f} U/day)")


if __name__ == '__main__':
    for ds in ('ohiot1dm', 'azt1d', 'shanghai'):
        _summary(ds)
