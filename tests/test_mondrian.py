"""Region-binned (Mondrian) conformal: the fit, the grouped apply, and the floor rule.

What these pin, in the order the module is used:

* the region axis — one edge near 110 mg/dL, never at a clinical threshold, read
  off where the forecast is HEADING, and invariant under the correction itself
  (``conformal`` holds the median fixed, so a window's bin cannot move);
* the ARITHMETIC FLOOR as a stated rule — a bin with fewer than
  ``MIN_N_OWN_FIT = 39`` calibration windows takes the marginal delta and records
  that it did, because below 39 its own τ=0.05 offset IS the sample minimum;
* per-bin conditional coverage: with two regions whose residual scales differ, the
  marginal fit under-covers one and over-covers the other and the binned fit does
  neither;
* the grouped apply — one ``conformal.apply_quantile_conformal`` call per bin with
  that bin's ``(S, K)`` slice, never a gathered ``(N, S, K)`` delta, which the
  function's 2-D assert still refuses;
* the infill fit is coarse, loud about its fallbacks, and stamped ``shipped=False``.
"""
import numpy as np
import pytest

import conformal
import mondrian
from config import PREDICTION_PATCHES, PATCH_SIZE, QUANTILE_LEVELS

S = PREDICTION_PATCHES * PATCH_SIZE
K = len(QUANTILE_LEVELS)
MED = QUANTILE_LEVELS.index(0.5)
LO, HI = QUANTILE_LEVELS.index(0.05), QUANTILE_LEVELS.index(0.95)

# Gaussian z at each τ, so a fan built with a given half-scale has a KNOWN
# coverage against Gaussian truth of that same scale.
_Z = {0.05: -1.6449, 0.1: -1.2816, 0.25: -0.6745, 0.5: 0.0,
      0.75: 0.6745, 0.9: 1.2816, 0.95: 1.6449}


def _fan(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(N, S, K) ascending Gaussian fan: center + z_tau * scale."""
    z = np.array([_Z[t] for t in QUANTILE_LEVELS])
    return center[:, :, None] + scale[:, None, None] * z[None, None, :]


def _cohort(rng, n: int, dest: float, model_scale: float, true_scale: float):
    """n windows whose median line sits flat at ``dest`` with a fan of
    ``model_scale`` and truth actually drawn at ``true_scale``."""
    center = np.full((n, S), float(dest)) + rng.standard_normal((n, 1)) * 0.5
    q = _fan(center, np.full(n, float(model_scale)))
    true = center + rng.standard_normal((n, S)) * true_scale
    return q, true


def test_region_edge_is_not_a_clinical_threshold():
    assert 70.0 not in mondrian.REGION_EDGES, \
        "a bin edge at the hypo threshold splits the alarm decision across two fits"
    assert mondrian.REGION_EDGES == (110.0,)
    assert mondrian.N_REGION_BINS == 2
    b = mondrian.region_bin(np.array([40.0, 109.999, 110.0, 400.0]))
    assert b.tolist() == [0, 0, 1, 1], b
    assert mondrian.MIN_N_OFFSET_EXISTS == 19 and mondrian.MIN_N_OWN_FIT == 39
    # The floors are arithmetic, not tuned: they are where the order statistic
    # floor((n+1)*0.05) first reaches 1 and 2.
    assert int(np.floor((19 + 1) * 0.05)) == 1 and int(np.floor((18 + 1) * 0.05)) == 0
    assert int(np.floor((39 + 1) * 0.05)) == 2 and int(np.floor((38 + 1) * 0.05)) == 1
    print(f"\n[DUMP] region edges {mondrian.REGION_EDGES} mg/dL; bins "
          f"{[mondrian.bin_label(i) for i in range(mondrian.N_REGION_BINS)]}; "
          f"floors {mondrian.MIN_N_OFFSET_EXISTS}/{mondrian.MIN_N_OWN_FIT} ✓")


def test_destination_is_invariant_under_the_correction():
    """The bin is read off the median line, which conformal never moves."""
    rng = np.random.default_rng(0)
    q, _ = _cohort(rng, 60, 130.0, 10.0, 30.0)
    delta = rng.standard_normal((mondrian.N_REGION_BINS, S, K)) * 5.0
    delta[:, :, MED] = 0.0
    bin_before = mondrian.region_bin(mondrian.forecast_destination(q, MED))
    qc = mondrian.apply_mondrian(q, delta, bin_before, MED)
    bin_after = mondrian.region_bin(mondrian.forecast_destination(qc, MED))
    assert np.array_equal(bin_before, bin_after)
    assert np.allclose(qc[:, :, MED], q[:, :, MED]), "median moved"
    assert np.all(np.diff(qc, axis=-1) >= -1e-9), "fan crossed"
    print(f"[DUMP] destination bin invariant under apply; median fixed; fan monotone ✓")


def test_thin_bin_takes_the_marginal_delta_and_says_so():
    """Ohio's low bin (32 calibration windows at edge 110) is the case this rule is for."""
    rng = np.random.default_rng(1)
    q_lo, t_lo = _cohort(rng, 32, 95.0, 8.0, 30.0)     # < 110, under the floor
    q_hi, t_hi = _cohort(rng, 112, 150.0, 8.0, 12.0)   # >= 110, over it
    q = np.concatenate([q_lo, q_hi]); true = np.concatenate([t_lo, t_hi])
    b = mondrian.region_bin(mondrian.forecast_destination(q, MED))
    assert (b == 0).sum() == 32 and (b == 1).sum() == 112

    delta, marginal, meta = mondrian.fit_mondrian(q, true, b, QUANTILE_LEVELS, MED)
    low, high = meta['bins'][0], meta['bins'][1]
    assert low['own_fit'] is False and low['n'] == 32
    assert str(mondrian.MIN_N_OWN_FIT) in low['fallback_reason']
    assert np.array_equal(delta[0], marginal), "thin bin must take the MARGINAL delta"
    assert high['own_fit'] is True and not np.array_equal(delta[1], marginal)
    assert meta['n_fallback_bins'] == 1
    print(f"[DUMP] thin bin n=32 -> marginal, reason: {low['fallback_reason']}")
    print(f"[DUMP] bin n={[r['n'] for r in meta['bins']]}, own_fit="
          f"{[r['own_fit'] for r in meta['bins']]} ✓")


def test_binned_fit_restores_conditional_coverage_the_marginal_one_loses():
    """Two regions, two residual scales: marginal is right on average and wrong in both."""
    rng = np.random.default_rng(2)
    # The model's fan is the SAME width everywhere; the truth is far noisier in the
    # low region. A single pooled delta therefore over-widens one and under-widens
    # the other.
    cal = [_cohort(rng, 200, 90.0, 10.0, 35.0), _cohort(rng, 200, 160.0, 10.0, 8.0)]
    test = [_cohort(rng, 400, 90.0, 10.0, 35.0), _cohort(rng, 400, 160.0, 10.0, 8.0)]
    cq = np.concatenate([c[0] for c in cal]); ct = np.concatenate([c[1] for c in cal])
    tq = np.concatenate([t[0] for t in test]); tt = np.concatenate([t[1] for t in test])
    cb = mondrian.region_bin(mondrian.forecast_destination(cq, MED))
    tb = mondrian.region_bin(mondrian.forecast_destination(tq, MED))

    delta, marginal, meta = mondrian.fit_mondrian(cq, ct, cb, QUANTILE_LEVELS, MED)
    assert all(r['own_fit'] for r in meta['bins']), meta['bins']
    t_marg = conformal.apply_quantile_conformal(tq, marginal, MED)
    t_mond = mondrian.apply_mondrian(tq, delta, tb, MED)

    rep = mondrian.bin_report({'marginal': t_marg, 'binned': t_mond}, tt, tb, LO, HI)
    for rec in rep:
        gm = abs(rec['arms']['marginal']['cov'] - 0.90)
        gb = abs(rec['arms']['binned']['cov'] - 0.90)
        assert gb < gm, (rec['label'], rec['arms'])
        print(f"[DUMP] region {rec['label']:>12} n={rec['n']:<4} cov marginal "
              f"{rec['arms']['marginal']['cov']:.3f} (width "
              f"{rec['arms']['marginal']['width']:.1f}) -> binned "
              f"{rec['arms']['binned']['cov']:.3f} (width "
              f"{rec['arms']['binned']['width']:.1f})")
    # The pooled figure is the one that hides it: marginal looks fine overall.
    pooled_m = float(np.mean(conformal.band_coverage(t_marg, tt, LO, HI)))
    pooled_b = float(np.mean(conformal.band_coverage(t_mond, tt, LO, HI)))
    print(f"[DUMP] pooled cov marginal {pooled_m:.3f} vs binned {pooled_b:.3f} — "
          f"the pooled number is what conceals the per-region gap ✓")

    # Every bin also reports per d, the only axis a masked-BG metric may be binned
    # on; the right-edge forecast span puts patch p at one-sided d = p+1.
    groups = mondrian.forecast_d_step_groups(PREDICTION_PATCHES, PATCH_SIZE)
    assert list(groups) == [f'd{i}' for i in range(1, PREDICTION_PATCHES + 1)]
    assert sum(len(v) for v in groups.values()) == S
    rep_d = mondrian.bin_report({'binned': t_mond}, tt, tb, LO, HI, step_groups=groups)
    for rec in rep_d:
        assert list(rec['by_d']) == list(groups)
        for lbl, blk in rec['by_d'].items():
            assert blk['n_steps'] == PATCH_SIZE
            assert blk['arms']['binned']['cov'] is not None
    print(f"[DUMP] per-d rows present for every bin: "
          f"{ {l: round(rep_d[0]['by_d'][l]['arms']['binned']['cov'], 3) for l in groups} } ✓")


def test_grouped_apply_matches_per_group_and_the_2d_assert_is_intact():
    rng = np.random.default_rng(3)
    q = np.concatenate([_cohort(rng, 50, 90.0, 9.0, 20.0)[0],
                        _cohort(rng, 50, 150.0, 9.0, 20.0)[0]])
    b = mondrian.region_bin(mondrian.forecast_destination(q, MED))
    delta = rng.standard_normal((mondrian.N_REGION_BINS, S, K)) * 3.0
    delta[:, :, MED] = 0.0

    got = mondrian.apply_mondrian(q, delta, b, MED)
    want = np.empty_like(q)
    for bi in range(mondrian.N_REGION_BINS):
        rows = np.flatnonzero(b == bi)
        want[rows] = conformal.apply_quantile_conformal(q[rows], delta[bi], MED)
    assert np.array_equal(got, want)

    # A gathered per-window delta is exactly what apply must keep refusing, and the
    # refusal is the guard against a 1-D (K,) delta broadcasting across every step.
    with pytest.raises(AssertionError):
        conformal.apply_quantile_conformal(q, delta[b], MED)
    with pytest.raises(AssertionError):
        conformal.apply_quantile_conformal(q, delta[0, 0, :], MED)

    # Zero delta is still the identity through the binned path.
    z = np.zeros((mondrian.N_REGION_BINS, S, K))
    assert np.array_equal(mondrian.apply_mondrian(q, z, b, MED), q)
    print(f"[DUMP] grouped apply == per-group apply; (N,S,K) and (K,) deltas both "
          f"rejected by conformal's 2-D assert; zero delta identity ✓")


def test_infill_fit_is_coarse_loud_and_never_shipped(capsys):
    rng = np.random.default_rng(4)
    q_lo, t_lo = _cohort(rng, 10, 95.0, 8.0, 15.0)     # deliberately under the floor
    q_hi, t_hi = _cohort(rng, 120, 150.0, 8.0, 15.0)
    q = np.concatenate([q_lo, q_hi]); true = np.concatenate([t_lo, t_hi])
    b = mondrian.region_bin(mondrian.forecast_destination(q, MED))
    pats = [f"p{i % 7}" for i in range(len(q))]

    delta, marginal, meta = mondrian.fit_infill_conformal(
        q, true, b, QUANTILE_LEVELS, MED, patients=pats)
    out = capsys.readouterr().out
    assert 'INFILL-CONFORMAL' in out and 'MARGINAL FALLBACK' in out, out
    assert meta['shipped'] is False and meta['protocol'] == 'infill'
    assert meta['bins'][0]['own_fit'] is False
    assert np.array_equal(delta[0], marginal)
    # Every bin reports its distinct-patient count beside its n.
    assert all(r['n_patients'] is not None for r in meta['bins'])
    assert delta.shape == (mondrian.N_REGION_BINS, S, K)
    print(f"\n[DUMP] infill fit shipped={meta['shipped']}, bins "
          f"{[(r['n'], r['n_patients'], r['own_fit']) for r in meta['bins']]}; "
          f"fallback announced on stdout ✓")
