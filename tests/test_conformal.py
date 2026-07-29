"""Split-conformal quantile recalibration (``conformal.py``).

Deterministic synthetic checks: the invariants (median fixed, monotone fan,
zero-delta identity) and that an asymmetric miscalibration is corrected back to
nominal per-side coverage — harder on the side that was worse.
"""
import numpy as np

from conformal import (
    fit_quantile_conformal, apply_quantile_conformal, _conformal_offset,
    band_coverage,
)

LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
MED = LEVELS.index(0.5)


def _make_fan(true, half):
    """Build an (N, S, K) ascending fan centred on ``true`` with per-level offsets
    proportional to (τ − 0.5) scaled by ``half`` (a deliberately too-narrow band)."""
    N, S = true.shape
    K = len(LEVELS)
    off = np.array([(t - 0.5) for t in LEVELS])
    return true[:, :, None] + half * off[None, None, :]


def test_zero_delta_is_identity():
    rng = np.random.default_rng(0)
    q = np.sort(rng.standard_normal((5, 3, 7)), axis=-1)
    out = apply_quantile_conformal(q, np.zeros((3, 7)), MED)
    assert np.array_equal(out, q)
    print("\n[DUMP] zero-delta identity ✓")


def test_median_fixed_and_monotone():
    rng = np.random.default_rng(1)
    q = np.sort(rng.standard_normal((8, 4, 7)), axis=-1)
    delta = rng.standard_normal((4, 7)) * 5.0
    delta[:, MED] = 0.0
    out = apply_quantile_conformal(q, delta, MED)
    assert np.allclose(out[..., MED], q[..., MED]), "median column must be untouched"
    assert (np.diff(out, axis=-1) >= -1e-9).all(), "calibrated fan must stay ascending"
    print("[DUMP] median fixed + monotone under arbitrary delta ✓")


def test_band_coverage_known_fan():
    # Hand-built fan: 4 windows, 2 steps, 7 levels. Edges LO=0.05 -> col0, HI=0.95 -> col6.
    # Put the [lo,hi] band at [90,150] everywhere; truths chosen so the count is exact.
    LO, HI = LEVELS.index(0.05), LEVELS.index(0.95)
    N, S, K = 4, 2, len(LEVELS)
    q = np.zeros((N, S, K))
    # ascending fan per window/step centred on 120 with band [90,150]
    base = np.array([90, 100, 110, 120, 130, 140, 150.0])
    q[...] = base[None, None, :]
    true = np.array([[120, 80],     # in,  below-lo  (out)
                     [90, 200],     # on-edge(in), above-hi(out)
                     [150, 120],    # on-edge(in), in
                     [149, 91]])    # in, in
    cov = band_coverage(q, true, LO, HI)
    # step0: 4/4 in; step1: 2/4 in (window2 & window3)
    assert np.allclose(cov, [1.0, 0.5]), cov
    print(f"\n[DUMP] band_coverage known fan -> {cov.tolist()} (expect [1.0, 0.5]) ✓")


def test_conformal_offset_finite_sample():
    r = np.arange(100.0)                       # 0..99
    # UPPER edge (tau>=0.5) uses ceil: ceil((100+1)*0.9)-1 = 91-1 = 90 -> order stat 90.0
    assert _conformal_offset(r, 0.9) == 90.0
    assert _conformal_offset(r, 0.95) == r[int(np.ceil(101 * 0.95)) - 1]
    # LOWER edge (tau<0.5) uses floor (conservative for a lower bound) — one order
    # statistic below the ceil choice, so the lower band edge sits lower and the
    # hypo edge stays conservative at small N.
    assert _conformal_offset(r, 0.05) == r[int(np.floor(101 * 0.05)) - 1]
    assert _conformal_offset(r, 0.1) == r[int(np.floor(101 * 0.1)) - 1]
    print("[DUMP] conformal offset order-statistic (floor lower / ceil upper) ✓")


def test_lower_edge_conservative_small_n():
    """At small calibration N the lower (hypo) edge must stay conservative.

    With the anti-conservative ceil order statistic the tau=0.05/0.10 lower edges
    escaped ~2x nominal at n=20; the floor choice restores P(true<=edge) <= tau.
    Monte-Carlo over many cal/test draws (single step, qk=0 so residual==truth).
    """
    rng = np.random.default_rng(11)
    for tau, n in [(0.05, 20), (0.10, 20)]:
        esc = []
        for _ in range(4000):
            cal = rng.standard_normal(n)
            d = _conformal_offset(cal, tau)        # lower-edge offset
            t = rng.standard_normal()
            esc.append(t <= d)                     # escape below the lower edge
        rate = float(np.mean(esc))
        print(f"[DUMP] lower-edge escape tau={tau} n={n} -> {rate:.3f} (target<= {tau})")
        # conservative: at or below nominal (small slack for MC noise)
        assert rate <= tau + 0.02, (tau, n, rate)


def test_recovers_nominal_coverage_asymmetric():
    rng = np.random.default_rng(7)
    S = 2
    n_cal, n_te = 4000, 4000
    # Truth ~ N(0,1) per step; the model emits a TOO-NARROW, DOWN-SHIFTED fan, so the
    # band under-covers and the lower side escapes far more than the upper.
    cal_true = rng.standard_normal((n_cal, S))
    te_true = rng.standard_normal((n_te, S))
    # Fan centred on a biased, narrow estimate of 0 (predict ~ -0.4, half-width 0.8 —
    # far narrower than the true ~N(0,1) needs for 90%, and shifted down).
    def fan(true_like):
        n = true_like.shape[0]
        base = np.full((n, S), -0.4)
        return _make_fan(base, half=0.8)
    cal_q = fan(cal_true)
    te_q = fan(te_true)
    LO, HI, H10 = LEVELS.index(0.05), LEVELS.index(0.95), LEVELS.index(0.1)

    def below(q, true, k):
        return float(np.mean(true < q[:, :, k]))
    def cov(q, true):
        return float(np.mean((true >= q[:, :, LO]) & (true <= q[:, :, HI])))

    raw_cov = cov(te_q, te_true)
    raw_lo = below(te_q, te_true, LO)            # should be >> 0.05 (under-covers below)
    delta = fit_quantile_conformal(cal_q, cal_true, LEVELS, MED)
    te_cal = apply_quantile_conformal(te_q, delta, MED)
    cal_cov = cov(te_cal, te_true)
    cal_lo = below(te_cal, te_true, LO)
    cal_lo10 = below(te_cal, te_true, H10)
    print(f"\n[DUMP] 90% coverage raw={raw_cov:.3f} -> calibrated={cal_cov:.3f} (target 0.90)")
    print(f"[DUMP] P(true<lower05) raw={raw_lo:.3f} -> calibrated={cal_lo:.3f} (target 0.05)")
    print(f"[DUMP] P(true<tau0.10) calibrated={cal_lo10:.3f} (target 0.10)")
    assert raw_cov < 0.80, "synthetic band should start under-covering"
    assert abs(cal_cov - 0.90) < 0.03, "conformal must restore ~90% coverage"
    assert abs(cal_lo - 0.05) < 0.02 and abs(cal_lo10 - 0.10) < 0.03, "per-level edges calibrated"
    # The per-level corrections are non-trivial and asymmetric (each side fit from its
    # own residuals), while the median is left exactly untouched.
    assert np.all(delta[:, MED] == 0.0), "median delta must be 0"
    assert abs(delta[:, LO]).mean() > 1e-3 and abs(delta[:, HI]).mean() > 1e-3, "edges corrected"
    assert not np.allclose(abs(delta[:, LO]).mean(), abs(delta[:, HI]).mean()), "asymmetric correction"
