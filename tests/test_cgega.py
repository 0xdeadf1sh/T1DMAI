"""Tests for the CG-EGA clinical-accuracy metric (cg_ega.py).

The load-bearing correctness test is perfect-prediction => 100% AP everywhere.
A port-equivalence test re-evaluates the exact boolean zone conditions
transcribed in ``scratch/_cgega_reference.py`` as an independent oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

import cg_ega


# --------------------------------------------------------------------------- #
# Oracle: the EXACT zone conditions copied from scratch/_cgega_reference.py.
# These re-implement P-EGA / R-EGA independently of cg_ega.py so a divergence
# in either direction is caught.
# --------------------------------------------------------------------------- #
def _oracle_p_marks(y_true, y_pred, dy_true):
    mod = np.zeros_like(y_true, dtype=np.float64)
    mod[((dy_true > -2) & (dy_true <= -1)) | ((dy_true >= 1) & (dy_true < 2))] = 10
    mod[(dy_true <= -2) | (dy_true >= 2)] = 20
    A = (((y_pred <= 70 + mod) & (y_true <= 70))
         | ((y_pred <= y_true * 6 / 5 + mod) & (y_pred >= y_true * 4 / 5 - mod)))
    E = (((y_true > 180) & (y_pred < 70 - mod))
         | ((y_pred > 180 + mod) & (y_true <= 70)))
    D = (((y_pred > 70 + mod) & (y_pred > y_true * 6 / 5 + mod)
          & (y_true <= 70) & (y_pred <= 180 + mod))
         | ((y_true > 240) & (y_pred < 180 - mod) & (y_pred >= 70 - mod)))
    C = (((y_true > 70) & (y_pred > y_true * 22 / 17 + (180 - 70 * 22 / 17) + mod))
         | ((y_true <= 180) & (y_pred < y_true * 7 / 5 - 182 - mod)))
    B = ~(A | C | D | E)
    return np.argmax(np.stack([A, B, C, D, E], axis=0), axis=0)


def _oracle_r_marks(dy_true, dy_pred):
    A = (((dy_pred >= dy_true - 1) & (dy_pred <= dy_true + 1))
         | ((dy_pred <= dy_true / 2) & (dy_pred >= dy_true * 2))
         | ((dy_pred <= dy_true * 2) & (dy_pred >= dy_true / 2)))
    B = (~A) & (
        ((dy_pred <= -1) & (dy_true <= -1))
        | ((dy_pred <= dy_true + 2) & (dy_pred >= dy_true - 2))
        | ((dy_pred >= 1) & (dy_true >= 1)))
    uC = (dy_true < 1) & (dy_true >= -1) & (dy_pred > dy_true + 2)
    lC = (dy_true <= 1) & (dy_true > -1) & (dy_pred < dy_true - 2)
    uD = (dy_pred <= 1) & (dy_pred >= -1) & (dy_pred > dy_true + 2)
    lD = (dy_pred <= 1) & (dy_pred >= -1) & (dy_pred < dy_true - 2)
    uE = (dy_pred > 1) & (dy_true < -1)
    lE = (dy_pred < -1) & (dy_true > 1)
    return np.argmax(np.stack([A, B, uC, lC, uD, lD, uE, lE], axis=0), axis=0)


def _realistic_traj(seed: int = 0, n: int = 12, t: int = 24):
    """A small batch of plausible BG trajectories spanning hypo/eu/hyper."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(55.0, 260.0, size=(n, 1))
    drift = np.cumsum(rng.normal(0.0, 4.0, size=(n, t)), axis=1)
    y = np.clip(base + drift, 40.0, 400.0)
    last = np.clip(base[:, 0] + rng.normal(0.0, 6.0, size=n), 40.0, 400.0)
    return y, last


def test_perfect_prediction_is_all_accurate():
    """y_pred == y_true => dy match (R-EGA A) and value match (P-EGA A) => AP."""
    y, last = _realistic_traj(seed=1)
    counts = cg_ega.cg_ega_counts(y, y, last)
    fr = cg_ega.cg_ega_fractions(counts)
    populated = 0
    for reg in ("hypo", "eu", "hyper"):
        total = counts[f"ap_{reg}"] + counts[f"be_{reg}"] + counts[f"ep_{reg}"]
        if total == 0:
            continue
        populated += 1
        assert fr[f"ap_{reg}"] == pytest.approx(1.0), \
            f"{reg}: AP not 100% under perfect prediction (counts={counts})"
        assert counts[f"be_{reg}"] == 0, f"{reg}: nonzero BE under perfect pred"
        assert counts[f"ep_{reg}"] == 0, f"{reg}: nonzero EP under perfect pred"
    assert populated >= 2, "test trajectory should populate multiple regions"
    print(f"[DUMP] perfect-pred counts={counts}")


def test_fractions_sum_to_one_per_region():
    y, last = _realistic_traj(seed=2)
    yp, _ = _realistic_traj(seed=99)
    yp = yp[: y.shape[0]]
    counts = cg_ega.cg_ega_counts(y, yp, last)
    fr = cg_ega.cg_ega_fractions(counts)
    for reg in ("hypo", "eu", "hyper"):
        total = counts[f"ap_{reg}"] + counts[f"be_{reg}"] + counts[f"ep_{reg}"]
        if total == 0:
            assert fr[f"ap_{reg}"] is None
            continue
        s = fr[f"ap_{reg}"] + fr[f"be_{reg}"] + fr[f"ep_{reg}"]
        assert s == pytest.approx(1.0), f"{reg}: AP+BE+EP={s} != 1"
    print(f"[DUMP] mismatched-pred fractions={fr}")


def test_opposite_rate_is_erroneous():
    """Truth falling fast (−2 mg/dL/min), prediction rising fast (+2) => the
    rate is catastrophically wrong (R-EGA uE) => EP regardless of region.

    Hold the true series strictly above 70 mg/dL so every point is euglycemic
    (region is assigned by the TRUE value), isolating the rate axis."""
    T = 6
    last = np.array([180.0])
    y_true = (180.0 + np.arange(1, T + 1) * (-10.0)).reshape(1, T)  # −2 mg/dL/min, 170..120
    y_pred = (180.0 + np.arange(1, T + 1) * (10.0)).reshape(1, T)   # +2 mg/dL/min
    marks = cg_ega.cg_ega_marks(y_true, y_pred, last)
    # every step has dy_true=-2 (<-1) & dy_pred=+2 (>1) => uE (index 6).
    assert np.all(marks["r_mark"] == 6), \
        f"expected R-EGA uE for opposite fast rates, got {marks['r_mark']}"
    # all true values 120..170 are euglycemic.
    assert np.all(marks["region"] == "eu"), f"region: {marks['region']}"
    counts = cg_ega.cg_ega_counts(y_true, y_pred, last)
    assert counts["ap_eu"] == 0 and counts["be_eu"] == 0, \
        f"opposite-rate points must be EP, got {counts}"
    assert counts["ep_eu"] == T, f"expected all {T} EP, got {counts}"
    print(f"[DUMP] opposite-rate counts={counts}")


def test_large_point_error_is_erroneous():
    """A STABLE point with a huge value error => P-EGA E => EP. True hyper (300),
    prediction deep hypo (50): P-EGA E (y_true>180 & y_pred<70). Both flat
    (dy≈0 => R-EGA A), so the EP verdict is driven by the POINT axis alone."""
    T = 6
    last_t = np.array([300.0])
    y_true = np.full((1, T), 300.0)   # stable hyper, true
    y_pred = np.full((1, T), 50.0)    # stable but wildly low prediction
    marks = cg_ega.cg_ega_marks(y_true, y_pred, last_t)
    pm = cg_ega._p_ega_marks(np.full(T, 300.0), np.full(T, 50.0), np.zeros(T))
    # P-EGA E: (y_true>180) & (y_pred < 70 - mod), mod=0 here. index 4 for EVERY
    # step — the point axis alone forces EP irrespective of the rate mark.
    assert np.all(pm == 4), f"true-hyper/pred-hypo should be P-EGA E, got {pm}"
    assert np.all(marks["p_mark"] == 4), f"pipeline p_mark: {marks['p_mark']}"
    counts = cg_ega.cg_ega_counts(y_true, y_pred, last_t)
    # hyper filters: P column E is all-EP in filter_EP_hyper -> AP/BE both 0,
    # so the (R,P)=(*, E) cell is EP for every R-mark (incl. the t=0 lC transient
    # from the prediction's jump off the shared anchor).
    assert counts["ap_hyper"] == 0 and counts["be_hyper"] == 0, \
        f"large point error must be EP, got {counts}"
    assert counts["ep_hyper"] == T, f"expected all EP, got {counts}"
    print(f"[DUMP] large-point-error counts={counts}  pmarks={pm}  rmarks={marks['r_mark']}")


def test_counts_are_asymmetric_in_the_two_trajectories():
    """Swapping the two trajectories must change the table.

    ``cg_ega_counts(y_true, y_pred, ...)`` takes the FIRST argument as the reference on
    every axis — the glycemic region, the ±20% acceptance denominator, the zone-D
    excursion gates and the rate-dependent ``mod`` widening. Nothing else in this file
    can see that: the load-bearing perfect-prediction case passes ``(y, y)`` and is
    invariant under the swap, and every other test calls in the documented order. A
    caller that reverses the two arguments therefore produces a well-formed table of a
    different statistic and no test notices. Pin the asymmetry itself.
    """
    T = 8
    last = np.array([120.0, 300.0])
    a = np.stack([np.linspace(115.0, 60.0, T),        # eu -> hypo
                  np.full(T, 300.0)])                 # stable hyper
    b = np.stack([np.full(T, 190.0),                  # stable hyper
                  np.linspace(295.0, 150.0, T)])      # hyper -> eu
    fwd = cg_ega.cg_ega_counts(a, b, last)
    rev = cg_ega.cg_ega_counts(b, a, last)

    def _tot(c, reg):
        return c[f"ap_{reg}"] + c[f"be_{reg}"] + c[f"ep_{reg}"]

    assert fwd != rev, f"transposing the trajectories left the table unchanged: {fwd}"
    assert any(_tot(fwd, r) != _tot(rev, r) for r in ("hypo", "eu", "hyper")), (
        f"region DENOMINATORS did not move under transposition: "
        f"{[(r, _tot(fwd, r), _tot(rev, r)) for r in ('hypo', 'eu', 'hyper')]}"
    )
    # the point count is conserved either way — only its distribution over regions moves
    assert sum(fwd.values()) == sum(rev.values()) == a.size
    print(f"[DUMP] asymmetry: fwd={fwd}\n[DUMP]            rev={rev}")


def test_port_equivalence_against_reference_conditions():
    """For a random batch, the cg_ega P/R marks must equal the oracle marks
    re-derived straight from the reference boolean conditions."""
    y_true, last_t = _realistic_traj(seed=7, n=16, t=20)
    y_pred, _ = _realistic_traj(seed=8, n=16, t=20)
    # anchor pred at a perturbed true-anchor so the dy_pred[t=0] is well-defined
    last_p = last_t + np.random.default_rng(8).normal(0, 5, size=last_t.shape)

    dy_true = cg_ega.cg_ega_rates(y_true, last_t).reshape(-1)
    dy_pred = cg_ega.cg_ega_rates(y_pred, last_p).reshape(-1)
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    p_ref = _oracle_p_marks(yt, yp, dy_true)
    r_ref = _oracle_r_marks(dy_true, dy_pred)
    p_got = cg_ega._p_ega_marks(yt, yp, dy_true)
    r_got = cg_ega._r_ega_marks(dy_true, dy_pred)

    assert np.array_equal(p_got, p_ref), "P-EGA marks diverge from reference"
    assert np.array_equal(r_got, r_ref), "R-EGA marks diverge from reference"

    # And the full per-point AP/BE/EP classification reproduces a hand-rolled
    # reference lookup over the same marks (exercise the filter matrices).
    region = np.where(yt <= 70, "hypo", np.where(yt <= 180, "eu", "hyper"))
    ap_ref = np.zeros(yt.shape, dtype=bool)
    be_ref = np.zeros(yt.shape, dtype=bool)
    f_ap = {"hypo": _ref_AP_hypo, "eu": _ref_AP_eu, "hyper": _ref_AP_hyper}
    f_be = {"hypo": _ref_BE_hypo, "eu": _ref_BE_eu, "hyper": _ref_BE_hyper}
    p_cols = {"hypo": [0, 3, 4], "eu": [0, 1, 2], "hyper": [0, 1, 2, 3, 4]}
    for i in range(yt.shape[0]):
        reg = region[i]
        cols = p_cols[reg]
        pm = p_got[i]
        if pm in cols:
            j = cols.index(pm)
            ap_ref[i] = bool(f_ap[reg][r_got[i]][j])
            be_ref[i] = (not ap_ref[i]) and bool(f_be[reg][r_got[i]][j])
        # pm not in region cols => neither AP nor BE => EP
    # cg_ega's vectorized classification:
    counts_per = []
    for i in range(yt.shape[0]):
        reg = region[i]
        lbl = cg_ega._LABEL_TABLE[reg][r_got[i], p_got[i]]
        counts_per.append(lbl)
    counts_per = np.array(counts_per)
    assert np.array_equal(counts_per == cg_ega._LABEL_AP, ap_ref), \
        "AP classification diverges from reference filter lookup"
    assert np.array_equal(counts_per == cg_ega._LABEL_BE, be_ref), \
        "BE classification diverges from reference filter lookup"
    print(f"[DUMP] port-equivalence OK on {yt.shape[0]} points")


# Reference filter matrices (VERBATIM from scratch/_cgega_reference.py) as plain
# python lists, used by the port-equivalence oracle above.
_ref_AP_hypo = [[1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
_ref_BE_hypo = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]]
_ref_AP_eu = [[1, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
_ref_BE_eu = [[0, 0, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]]
_ref_AP_hyper = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
_ref_BE_hyper = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
