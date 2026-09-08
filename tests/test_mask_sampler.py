"""Tests for ``data.sample_mask_spans`` — the masked-span sampler.

Every reference figure is ENUMERATED from the live config knobs, never written down. Placement
is uniform CONDITIONAL on (n_spans, vector) per branch; the MARGINAL is not flat.
"""

from collections import Counter
from fractions import Fraction
from itertools import product
from math import comb

import numpy as np
import pytest

from config import (MASK_MAX_SPANS, MASK_RIGHT_EDGE_QUOTA, MASK_SPAN_LENGTHS,
                    MAX_CONTEXT_PATCHES, MAX_MASKED_PATCHES,
                    MIN_CONTEXT_PATCHES, PATCH_SIZE, PREDICTION_PATCHES)
import data
from data import sample_mask_spans, _mask_slots
# d grouping/per-d shares live in d_balance, reused never re-derived: avoids a second drift.
from d_balance import N_D_GROUPS, _group, d_distribution


# How far a frequency may sit from its reference, in binomial/delta-method standard errors.
_Z_TOL = 5.0            # pooled statistics: hundreds of z, all at high counts
_COND_Z_TOL = 6.0       # per-bucket conditional check: ~2500 z at low counts

# The live quota as an EXACT rational, so the closed form below stays exact.
_QUOTA = Fraction(str(MASK_RIGHT_EDGE_QUOTA))


def _length_vectors() -> list[tuple[Fraction, tuple[int, ...]]]:
    """Every ``(probability, length vector)`` the sampler can draw.

    n_spans uniform over 1..MASK_MAX_SPANS; length vector uniform over vectors of that arity
    fitting MAX_MASKED_PATCHES (over-budget rejects the WHOLE vector). Quota-independent.
    """
    out: list[tuple[Fraction, tuple[int, ...]]] = []
    for n in range(1, MASK_MAX_SPANS + 1):
        vecs = _feasible_vectors(n)
        for v in vecs:
            out.append((Fraction(1, MASK_MAX_SPANS) * Fraction(1, len(vecs)), v))
    return out


def _feasible_vectors(n: int) -> list[tuple[int, ...]]:
    """The length vectors of arity ``n`` the budget admits, in a fixed order."""
    return [v for v in product(MASK_SPAN_LENGTHS, repeat=n)
            if sum(v) <= MAX_MASKED_PATCHES]


def _gap_shares(m: int, slack: int):
    """``(i, g, P(G_i = g))`` for ``m`` spans over ``slack`` in ``m + 1`` gaps.

    ``G_i`` is the sum of the first ``i+1`` gaps; span ``i`` starts at ``sum(v[:i])+i+G_i``.
    Compositions with ``G_i=g``: ``C(g+i,i)*C(slack-g+m-i-1,m-i-1)`` out of ``C(slack+m,m)``.
    """
    total = comb(slack + m, m)
    for i in range(m):
        for g in range(slack + 1):
            ways = comb(g + i, i) * comb(slack - g + m - i - 1, m - i - 1)
            if ways:
                yield i, g, Fraction(ways, total)


def _vector_placements(T: int, v: tuple[int, ...], q: Fraction):
    """``(probability, start, length)`` for every span the sampler can place.

    Both branches, mixed at ``q``. Right-edge pins the last span at ``T - v[-1]``, composing
    the other ``n-1`` over the prefix (same ``slack``: the trailing gap is fixed at 0).
    """
    n = len(v)
    slack = T - sum(v) - (n - 1)
    assert slack >= 0, f"window of {T} patches cannot hold {v}"
    if q < 1:
        for i, g, share in _gap_shares(n, slack):
            yield (1 - q) * share, sum(v[:i]) + i + g, v[i]
    if q > 0:
        yield q, T - v[-1], v[-1]
        for i, g, share in _gap_shares(n - 1, slack):
            yield q * share, sum(v[:i]) + i + g, v[i]


def _conditional_marginal(T: int, v: tuple[int, ...],
                          q: Fraction = _QUOTA) -> list[Fraction]:
    """``P(patch p is masked | length vector v)``, exactly."""
    p = [Fraction(0)] * T
    for prob, start, length in _vector_placements(T, v, q):
        for k in range(length):
            p[start + k] += prob
    return p


def _position_marginal(T: int, q: Fraction = _QUOTA) -> list[float]:
    """``P(patch p is masked)`` for every p — conditional marginals mixed over the length law.

    Accumulated in ``Fraction`` so the quota-0 left/right symmetry below is bit-exact.
    """
    p = [Fraction(0)] * T
    for prob, v in _length_vectors():
        for j, x in enumerate(_conditional_marginal(T, v, q)):
            p[j] += prob * x
    return [float(x) for x in p]


def _flush_right_probability(T: int, q: Fraction = _QUOTA) -> float:
    """``P(some span ends at patch T-1)``, exactly.

    At most one span can end there, so the placements that do are disjoint events. Exceeds
    the quota by the uniform branch's own right-edge landings.
    """
    total = Fraction(0)
    for prob, v in _length_vectors():
        for share, start, length in _vector_placements(T, v, q):
            if start + length == T:
                total += prob * share
    return float(total)


def _plateau(T: int) -> tuple[int, int]:
    """Half-open bounds of the interior plateau, past both edge structures.

    Left taper is at most ``L-1`` positions (span CEILING). Right is one patch wider (the
    quota's pinned span plus its mandatory separator). Both derived from MASK_SPAN_LENGTHS[-1].
    """
    w = min(MASK_SPAN_LENGTHS[-1], (T - 2) // 2)
    assert T - 2 * w - 1 >= 2, f"no interior left at T={T}"
    return w, T - w - 1


def _masked_patch_mean() -> Fraction:
    """``E[sum(L)]`` under the sampler, exactly."""
    return sum(prob * sum(v) for prob, v in _length_vectors())


class _CountingRng:
    """A ``Generator`` proxy that counts ``random()`` calls and forwards the rest.

    ``sample_mask_spans`` draws the branch with a single ``rng.random()``, so the count IS
    how many times the branch draw was consumed.
    """

    def __init__(self, rng: "np.random.Generator") -> None:
        self._rng = rng
        self.n_random = 0

    def random(self, *args, **kwargs):
        self.n_random += 1
        return self._rng.random(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._rng, name)


@pytest.mark.parametrize("T", [20, 36, 52])
def test_mask_placement_interior_is_flat(T):
    """Placement is uniform CONDITIONAL on (n_spans, length vector) within each branch;
    the marginal that leaves has a flat interior.

    Measured on the LIVE sampler against the exact marginal; asserts shape, never magnitude.
    """
    marg = _position_marginal(T)
    lo, hi = _plateau(T)
    interior = marg[lo:hi]

    # (a) Marginal integrates to E[#masked]; quota-independent, so quota moves mass, never adds it.
    assert abs(sum(marg) - float(_masked_patch_mean())) < 1e-9, \
        f"sum of the marginal {sum(marg):.9f} != E[#masked] {float(_masked_patch_mean()):.9f}"

    # (b) Interior is a plateau: spread small vs edge-taper deficit, small in absolute terms too.
    mean = sum(interior) / len(interior)
    spread = (max(interior) - min(interior)) / mean
    drop = (mean - marg[0]) / mean
    assert drop > 0.1, f"no edge taper at all: outermost is {marg[0] / mean:.4f}x the interior"
    assert spread < 0.05, f"T={T}: interior spread {spread * 100:.3f}% is not a plateau"
    assert spread < 0.15 * drop, (
        f"T={T}: interior spread {spread * 100:.3f}% is not small against the "
        f"{drop * 100:.1f}% edge deficit — the taper has leaked into the interior")

    # (c) Right edge is LIFTED, not tapered, by at least the quota: every right-edge draw masks T-1.
    assert marg[T - 1] >= float(MASK_RIGHT_EDGE_QUOTA), (
        f"T={T}: patch T-1 is masked on {marg[T - 1]:.4f} of draws, under the "
        f"{MASK_RIGHT_EDGE_QUOTA} quota that pins a span there")
    assert all(x > mean for x in marg[T - lo:]), (
        f"T={T}: the flush-right block is not above the interior plateau")

    # (d) At quota 0 the closed form is exactly symmetric — the property the quota exists to break.
    flat = _position_marginal(T, Fraction(0))
    assert all(abs(flat[k] - flat[T - 1 - k]) < 1e-12 for k in range(T)), \
        "the quota-0 marginal is not left/right symmetric — placement is biased"

    # (e) The guarantee itself: per-position frequency must match each vector's exact marginal.

    # Positions the vector can't avoid are asserted deterministically; rest by z against the sem.
    rng = np.random.default_rng(20250813 + T)
    n_draws = 40_000
    counts: dict[tuple[int, ...], np.ndarray] = {}
    drawn: Counter = Counter()
    for _ in range(n_draws):
        spans = sample_mask_spans(T, rng)
        v = tuple(L for _s, L in spans)
        c = counts.get(v)
        if c is None:
            c = counts[v] = np.zeros(T, dtype=np.int64)
        for start, length in spans:
            c[start:start + length] += 1
        drawn[v] += 1

    checked, checked_multi, worst, worst_at = 0, 0, 0.0, None
    for v, c in counts.items():
        nb = drawn[v]
        if nb < 200:                      # too thin to say anything either way
            continue
        exact = _conditional_marginal(T, v)
        certain = [k for k, q in enumerate(exact) if q == 1]
        assert all(int(c[k]) == nb for k in certain), (
            f"positions {certain} are masked by every placement of {v} but were "
            "left visible")
        keep = [k for k, q in enumerate(exact)
                if q != 1 and float(q) * nb >= 25.0]
        if len(keep) < 4:
            continue
        e = np.asarray([float(exact[k]) for k in keep])
        z = (c[keep] / nb - e) / np.sqrt(e * (1.0 - e) / nb)
        if float(np.abs(z).max()) > worst:
            worst, worst_at = float(np.abs(z).max()), (v, int(np.abs(z).argmax()))
        checked += 1
        checked_multi += len(v) > 1
    assert checked >= 4, f"only {checked} length vectors were dense enough to check"
    assert MASK_MAX_SPANS < 2 or checked_multi >= 1, \
        "no multi-span length vector was checked — the composition is untested"
    assert worst < _COND_Z_TOL, (
        f"T={T}: placement does not match the two-branch closed form conditional "
        f"on the length vector — {worst:.2f}σ at {worst_at}, tolerance {_COND_Z_TOL}σ")

    print(f"\n[DUMP] placement T={T} | interior [{lo}, {hi}) mean={mean:.6f}; spread "
          f"{spread * 100:.4f}% against a {drop * 100:.2f}% edge deficit; patch T-1 "
          f"masked {marg[T - 1]:.4f} (quota {MASK_RIGHT_EDGE_QUOTA}); conditional "
          f"uniformity over {checked} length vectors ({checked_multi} multi-span), "
          f"worst {worst:.2f}σ ✓")


@pytest.mark.parametrize("T", [20, 36, 52])
def test_mask_placement_edge_taper(T):
    """The marginal tapers at the LEFT end and climbs monotonically inward; the right end
    carries the quota's ramp instead of the mirror taper.

    Taper width and magnitude are derived from MASK_SPAN_LENGTHS[-1] and printed, never pinned.
    """
    marg = _position_marginal(T)
    lo, hi = _plateau(T)
    mean = sum(marg[lo:hi]) / (hi - lo)

    # The outermost position is strictly the thinnest at the LEFT edge.
    assert marg[0] < 0.9 * mean, \
        f"T={T}: outermost density {marg[0] / mean:.4f}x the interior — the taper is gone"
    assert marg[0] < min(marg[lo:hi]), "the outermost position is not below the plateau"

    # Monotone inward over the first half of the taper.
    probe = max(1, MASK_SPAN_LENGTHS[-1] // 2)
    for k in range(probe):
        assert marg[k] < marg[k + 1], \
            f"T={T}: taper not increasing inward at position {k} ({marg[k]:.6f} >= {marg[k + 1]:.6f})"
        assert marg[k] < mean, \
            f"T={T}: position {k} is not below the interior plateau"

    # And by the plateau bound the taper has arrived: within 5% of the interior.
    assert abs(marg[lo] / mean - 1.0) < 0.05, (
        f"T={T}: position {lo} is {marg[lo] / mean:.4f}x the interior — the taper "
        f"is wider than the span ceiling {MASK_SPAN_LENGTHS[-1]}")

    # Right edge: the pinned span's mandatory separator is the last position below the plateau.
    assert marg[hi] <= mean, (
        f"T={T}: position {hi} is above the interior plateau — it is the separator "
        f"a flush-right span of the full ceiling charges and can never cover")
    assert marg[T - 1] > marg[0], \
        "the right edge tapers like the left — the quota is not placing anything"

    print(f"\n[DUMP] taper T={T} | " + "  ".join(
        f"p{k}={marg[k] / mean:.4f}x" for k in range(min(lo + 1, T // 2))) +
        f"  (plateau [{lo}, {hi}), sep {marg[hi] / mean:.4f}x, edge "
        f"{marg[T - 1] / mean:.4f}x) ✓")


def test_length_distribution_is_whole_vector_rejection():
    """Conditional on ``n_spans``, EVERY feasible length vector is equally likely — the
    signature of redrawing the WHOLE vector on overflow.

    Redrawing one element leaves probabilities uneven though the feasible set is unchanged.
    """
    shares = {L: Fraction(0) for L in MASK_SPAN_LENGTHS}
    n_spans_mean = Fraction(0)
    masked_mean = _masked_patch_mean()
    p_full = Fraction(0)
    for prob, v in _length_vectors():
        n_spans_mean += prob * len(v)
        if sum(v) == MAX_MASKED_PATCHES:
            p_full += prob
        for L in v:
            shares[L] += prob
    total = sum(shares.values())
    pct = {L: float(shares[L] / total) * 100.0 for L in MASK_SPAN_LENGTHS}

    # Whether the budget can reject: MASK_MAX_SPANS*max(L)<=MAX_MASKED_PATCHES means never.
    rejects = any(len(_feasible_vectors(n)) < len(MASK_SPAN_LENGTHS) ** n
                  for n in range(1, MASK_MAX_SPANS + 1))
    assert abs(sum(pct.values()) - 100.0) < 1e-9
    assert abs(float(n_spans_mean) - (MASK_MAX_SPANS + 1) / 2) < 1e-9, \
        f"mean spans per sample {float(n_spans_mean):.4f}, expected uniform over 1..{MASK_MAX_SPANS}"
    # Longer spans are what the budget rejects: shares must DECREASE with length when it bites.
    assert all(pct[a] >= pct[b] for a, b in zip(MASK_SPAN_LENGTHS, MASK_SPAN_LENGTHS[1:])), \
        f"span-length shares are not non-increasing in L: {pct}"
    assert (pct[MASK_SPAN_LENGTHS[0]] > pct[MASK_SPAN_LENGTHS[-1]]) == rejects, \
        ("the budget rejects but every length is equally likely" if rejects else
         "no vector is over budget, so the lengths must be equally likely")

    # Against the live sampler, at the longest window (every vector fits).
    T = MAX_CONTEXT_PATCHES + PREDICTION_PATCHES
    n_draws = 60_000
    rng = np.random.default_rng(8675309)
    by_arity: dict[int, Counter] = {n: Counter() for n in range(1, MASK_MAX_SPANS + 1)}
    masked = np.zeros(n_draws)
    for i in range(n_draws):
        spans = sample_mask_spans(T, rng)
        v = tuple(L for _s, L in spans)
        by_arity[len(v)][v] += 1
        masked[i] = sum(v)

    p_arity = 1.0 / MASK_MAX_SPANS
    for n in range(1, MASK_MAX_SPANS + 1):
        drawn = sum(by_arity[n].values())
        z = (drawn / n_draws - p_arity) / np.sqrt(p_arity * (1 - p_arity) / n_draws)
        assert abs(z) < _Z_TOL, f"n_spans={n} drawn at {z:.2f}σ from uniform"

        vecs = _feasible_vectors(n)
        assert set(by_arity[n]) <= set(vecs), \
            f"the sampler drew an over-budget vector at arity {n}: " \
            f"{set(by_arity[n]) - set(vecs)}"
        e = drawn / len(vecs)
        chi2 = sum((by_arity[n][v] - e) ** 2 / e for v in vecs)
        dof = len(vecs) - 1
        z_chi2 = (chi2 - dof) / np.sqrt(2 * dof)
        assert z_chi2 < _Z_TOL, (
            f"arity {n}: the {len(vecs)} feasible length vectors are not equiprobable "
            f"(chi2={chi2:.1f}, dof={dof}, {z_chi2:.2f}σ) — the over-budget rejection "
            "is redrawing one element, not the whole vector")

    z_mean = (masked.mean() - float(masked_mean)) / (masked.std(ddof=1) / np.sqrt(n_draws))
    assert abs(z_mean) < _Z_TOL, (
        f"mean masked patches {masked.mean():.4f} vs the enumerated "
        f"{float(masked_mean):.6f} ({z_mean:.2f}σ)")
    emp_full = float((masked == MAX_MASKED_PATCHES).mean())
    pf = float(p_full)
    z_full = (emp_full - pf) / np.sqrt(max(pf * (1 - pf), 1e-12) / n_draws)
    assert abs(z_full) < _Z_TOL, (
        f"P(#masked == {MAX_MASKED_PATCHES}) = {emp_full * 100:.3f}% vs the "
        f"enumerated {pf * 100:.3f}% ({z_full:.2f}σ)")

    # E[#masked] sets the padded fraction of M slots; e.g. 41.76% at (1,2,3,4), MAX=8.
    pad_pct = (MAX_MASKED_PATCHES - float(masked_mean)) / MAX_MASKED_PATCHES * 100.0
    assert 0.0 <= pad_pct < 100.0
    print(f"\n[DUMP] lengths | shares {{{', '.join(f'{L}: {pct[L]:.2f}%' for L in MASK_SPAN_LENGTHS)}}}; "
          f"E[spans]={float(n_spans_mean):.3f} E[#masked]={float(masked_mean):.4f}; "
          f"P(full)={pf * 100:.2f}%; padded slots {pad_pct:.2f}%; "
          f"{[len(_feasible_vectors(n)) for n in range(1, MASK_MAX_SPANS + 1)]} feasible "
          f"vectors per arity (budget rejects: {rejects}), all equiprobable to {_Z_TOL}σ ✓")


@pytest.mark.parametrize("T", [20, 52])
def test_flush_right_frequency_matches_the_enumeration(T):
    """A span ends at patch T-1 as often as the two-branch enumeration says.

    Frequency is NOT the quota: uniform placement lands on the right edge too, so the
    figure is q + (1-q)*P_uniform. The reference is enumerated, never assumed as the quota.
    """
    exact = _flush_right_probability(T)
    unif = _flush_right_probability(T, Fraction(0))
    assert exact > float(MASK_RIGHT_EDGE_QUOTA) >= 0.0

    n_draws = 60_000
    rng = np.random.default_rng(515151 + T)
    hits = 0
    for _ in range(n_draws):
        spans = sample_mask_spans(T, rng)
        hits += (spans[-1][0] + spans[-1][1] == T)
    emp = hits / n_draws
    z = (emp - exact) / np.sqrt(exact * (1 - exact) / n_draws)
    assert abs(z) < _Z_TOL, (
        f"T={T}: a span ends at T-1 on {emp * 100:.3f}% of draws against the "
        f"enumerated {exact * 100:.3f}% ({z:.2f}σ) — the quota is not being "
        f"applied at the rate config declares")
    print(f"\n[DUMP] flush-right T={T} | {emp * 100:.2f}% vs enumerated "
          f"{exact * 100:.2f}% ({z:+.2f}σ); quota {MASK_RIGHT_EDGE_QUOTA}, uniform "
          f"branch alone would give {unif * 100:.2f}% ✓")


def test_quota_zero_never_consumes_the_placement_draw(monkeypatch):
    """At quota 0 the branch draw is short-circuited, so the mask stream is not merely
    distributed the same as the pre-quota sampler's, it IS that stream.

    Lose the short circuit and every mask after the first shifts by one draw, undetected here.
    """
    T = MAX_CONTEXT_PATCHES + PREDICTION_PATCHES
    n_draws = 500

    monkeypatch.setattr(data, 'MASK_RIGHT_EDGE_QUOTA', 0.0)
    counting = _CountingRng(np.random.default_rng(24680))
    zero_spans = [sample_mask_spans(T, counting) for _ in range(n_draws)]
    assert counting.n_random == 0, (
        f"the placement draw was consumed {counting.n_random} times at quota 0 — "
        f"the short circuit is gone and the mask stream has shifted")

    # The proxy is transparent: a plain Generator on the same seed draws the same masks.
    plain = np.random.default_rng(24680)
    assert zero_spans == [sample_mask_spans(T, plain) for _ in range(n_draws)]

    # at the live quota the draw IS consumed, once per call
    monkeypatch.setattr(data, 'MASK_RIGHT_EDGE_QUOTA', float(MASK_RIGHT_EDGE_QUOTA))
    counting = _CountingRng(np.random.default_rng(24680))
    live_spans = [sample_mask_spans(T, counting) for _ in range(n_draws)]
    assert counting.n_random == n_draws, (
        f"{counting.n_random} placement draws over {n_draws} calls — the branch is "
        f"not drawn exactly once per sample")
    assert live_spans != zero_spans, "the quota changed no mask at all"
    print(f"\n[DUMP] rng stream | quota 0: 0 random() draws over {n_draws} samples, "
          f"masks identical to a plain Generator; quota {MASK_RIGHT_EDGE_QUOTA}: "
          f"{counting.n_random} draws ✓")


@pytest.mark.parametrize("T", [20, 36, 52])
def test_right_edge_draws_are_flush_and_still_separated(T):
    """A flush-right draw pins the LAST span at ``T - L`` and leaves a visible patch before it.

    The separator could be dropped here since the span sits outside the composition; an
    abutting span reads as one longer span to utils._span_layout, merging anchors and buckets.
    """
    rng = np.random.default_rng(777_000 + T)
    flush = multi = 0
    for _ in range(20_000):
        spans = sample_mask_spans(T, rng)
        # The branch pins the LAST span only; an interior span reaching T-1 overran its prefix.
        assert all(s + L < T for s, L in spans[:-1]), \
            f"a span other than the last reaches the window edge: {spans} at T={T}"
        start, length = spans[-1]
        if start + length != T:
            continue
        flush += 1
        if len(spans) > 1:
            multi += 1
            prev_s, prev_L = spans[-2]
            assert prev_s + prev_L < start, (
                f"the pinned span abuts its neighbour: {spans} at T={T}")
            assert prev_s >= 0
    assert flush > 0 and multi > 0, (
        f"T={T}: {flush} flush draws, {multi} with a neighbour — the branch is "
        f"untested at the arity where the separator matters")
    print(f"\n[DUMP] right-edge structure T={T} | {flush} flush draws, {multi} with "
          f"a preceding span, every one pinned at T-L and separated ✓")


@pytest.mark.parametrize("T", [20, 36, 52])
def test_spans_never_abut_and_fit_the_budget(T):
    """At most MASK_MAX_SPANS spans, lengths in MASK_SPAN_LENGTHS, budget-capped, in the window.

    Dropped, the separator merges an abutting span into one longer span downstream.
    """
    rng = np.random.default_rng(20250813 + T)
    seen_lengths = set()
    seen_counts = set()
    max_masked = 0
    for _ in range(4000):
        spans = sample_mask_spans(T, rng)
        assert 1 <= len(spans) <= MASK_MAX_SPANS, f"{len(spans)} spans at T={T}"
        seen_counts.add(len(spans))
        total = 0
        for start, length in spans:
            assert length in MASK_SPAN_LENGTHS, f"illegal span length {length}"
            assert 0 <= start and start + length <= T, \
                f"span ({start}, {length}) outside [0, {T})"
            seen_lengths.add(length)
            total += length
        assert total <= MAX_MASKED_PATCHES, \
            f"{total} masked patches exceeds MAX_MASKED_PATCHES={MAX_MASKED_PATCHES}"
        max_masked = max(max_masked, total)
        for (s0, l0), (s1, _l1) in zip(spans, spans[1:]):
            assert s1 > s0 + l0, f"abutting (or overlapping) spans {spans} at T={T}"
    assert seen_lengths == set(MASK_SPAN_LENGTHS), \
        f"lengths {sorted(seen_lengths)} do not cover MASK_SPAN_LENGTHS"
    assert seen_counts == set(range(1, MASK_MAX_SPANS + 1)), \
        f"span counts {sorted(seen_counts)} do not cover 1..{MASK_MAX_SPANS}"
    print(f"\n[DUMP] structure T={T} | 4000 draws: counts {sorted(seen_counts)}, "
          f"lengths {sorted(seen_lengths)}, max masked {max_masked} <= "
          f"{MAX_MASKED_PATCHES}, no abutting pair ✓")


def test_mask_slots_pad_and_distance():
    """_mask_slots expands spans into fixed M slots: valid ascending, padded gather patch 0.
    d is distance to nearest visible EITHER side.

    d disagrees with the anchor: a slot can sit at d=1 while anchoring far to the left.
    """
    T = 20
    spans = [(0, 2), (5, 4), (12, 1)]        # backcast edge, interior, interior
    mask_idx, valid, d, anchor_step = _mask_slots(spans, T)

    assert mask_idx.shape == valid.shape == d.shape == (MAX_MASKED_PATCHES,)
    assert int(valid.sum()) == 7
    assert mask_idx[valid].tolist() == [0, 1, 5, 6, 7, 8, 12]
    assert (mask_idx[~valid] == 0).all(), "padded slots must gather patch 0"
    # Span(0,2) d=2,1 (no left neighbour); span(5,4) d=1,2,2,1; span(12,1) d=1.
    assert d[valid].tolist() == [2, 1, 1, 2, 2, 1, 1], f"d = {d[valid].tolist()}"
    # Anchor one-sided: whole span shares one step; patch-0 span reads right neighbour's FIRST step.
    from config import PATCH_SIZE
    assert anchor_step[valid].tolist() == [
        2 * PATCH_SIZE, 2 * PATCH_SIZE,                          # span at patch 0
        5 * PATCH_SIZE - 1, 5 * PATCH_SIZE - 1,
        5 * PATCH_SIZE - 1, 5 * PATCH_SIZE - 1,                  # span at patch 5
        12 * PATCH_SIZE - 1,                                     # span at patch 12
    ]
    # Last slot of the 4-patch span: d=1 (right neighbour) while the anchor sits 4 patches left.
    assert d[valid][5] == 1 and anchor_step[valid][5] == 5 * PATCH_SIZE - 1
    print(f"\n[DUMP] slots | mask_idx={mask_idx.tolist()} valid={valid.sum()} "
          f"d={d[valid].tolist()}; anchor one-sided (slot 5: d=1 right, anchor 4 "
          f"patches left) ✓")


def test_anchor_is_farther_than_the_nearest_evidence_for_a_third_of_slots():
    """Some anchors sit FARTHER than the nearest visible evidence: right half of two-sided spans.

    j+1 vs d=min(j+1,L-j) part when 2j+1>L; MECHANISM asserted, never the share.
    """
    lengths_T = [n + PREDICTION_PATCHES
                 for n in range(MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES + 1)]
    p_T = 1.0 / len(lengths_T)
    supervised = 0.0
    farther = 0.0
    for T in lengths_T:
        for prob, v in _length_vectors():
            for share_v, start, L in _vector_placements(T, v, _QUOTA):
                share = p_T * float(prob) * float(share_v)
                has_left, has_right = start > 0, start + L < T
                for j in range(L):
                    supervised += share
                    anchor = (j + 1) if has_left else (L - j)
                    d = min(j + 1 if has_left else L + T,
                            L - j if has_right else L + T)
                    if anchor > d:
                        # The mechanism, asserted per contributing slot.
                        assert has_left and has_right, \
                            f"a one-sided span anchored farther than d: {v} slot {j}"
                        assert 2 * j + 1 > L, \
                            f"anchor beat d in the LEFT half of a span: L={L}, j={j}"
                        farther += share
                    else:
                        assert anchor == d, \
                            f"anchor {anchor} < d {d}: the anchor cannot beat the nearest patch"
    pct = farther / supervised * 100.0
    assert pct > 0.0, \
        "no slot anchors farther than the nearest evidence — the anchor is no longer one-sided"
    assert pct < 50.0, \
        "over half the slots anchor farther — only the right half of two-sided spans can"

    # The enumeration is a model of `_mask_slots`; hold it against the real thing.
    n_draws = 40_000
    rng = np.random.default_rng(11235)
    per_draw_far = np.zeros(n_draws)
    per_draw_sup = np.zeros(n_draws)
    for i in range(n_draws):
        T = int(lengths_T[rng.integers(len(lengths_T))])
        spans = sample_mask_spans(T, rng)
        mask_idx, valid, d, anchor_step = _mask_slots(spans, T)
        idx = mask_idx[valid].tolist()
        dd = d[valid].tolist()
        aa = anchor_step[valid].tolist()
        slot = 0
        for start, L in spans:
            two_sided = start > 0 and start + L < T
            for j in range(L):
                per_draw_sup[i] += 1
                # The anchor's patch, read back out of the step index the code emits.
                anchor_patches = abs(idx[slot] - aa[slot] // PATCH_SIZE)
                if anchor_patches > dd[slot]:
                    assert two_sided and 2 * j + 1 > L, (
                        f"slot {j} of span ({start},{L}) at T={T} anchors farther "
                        "than d outside the right half of a two-sided span")
                    per_draw_far[i] += 1
                else:
                    assert anchor_patches == dd[slot], \
                        f"anchor {anchor_patches} closer than d {dd[slot]}"
                slot += 1
    emp = per_draw_far.sum() / per_draw_sup.sum()
    # Slots inside one draw are correlated, so sem is delta-method on the ratio, not binomial.
    resid = per_draw_far - (pct / 100.0) * per_draw_sup
    sem = resid.std(ddof=1) / np.sqrt(n_draws) / per_draw_sup.mean()
    z = (emp - pct / 100.0) / sem
    assert abs(z) < _Z_TOL, (
        f"the live sampler anchors farther on {emp * 100:.3f}% of slots against the "
        f"enumerated {pct:.3f}% ({z:.2f}σ) — the enumeration and `_mask_slots` disagree")
    print(f"\n[DUMP] anchor vs d | {pct:.2f}% of supervised slots anchor FARTHER "
          f"than the nearest visible patch (one-sided, left-preferring); live sampler "
          f"{emp * 100:.2f}% ({z:+.2f}σ) ✓")


def test_sampler_agrees_with_the_closed_form_coarsely():
    """The live sampler against both closed forms it feeds: marginal at T=52, d_balance's shares.

    COARSE by necessity: at 60000 draws the per-position 1sigma is ~1% of the mean. Catches
    a placement rule wrong by a lot — a bad quota rate, a curriculum, a missing separator.
    """
    T = 52
    n_draws = 60_000
    rng = np.random.default_rng(4242)
    counts = np.zeros(T, dtype=np.int64)
    masked_per_draw = np.zeros(n_draws)
    for i in range(n_draws):
        for start, length in sample_mask_spans(T, rng):
            counts[start:start + length] += 1
            masked_per_draw[i] += length
    empirical = counts / n_draws
    exact = np.asarray(_position_marginal(T))

    sem = np.sqrt(exact * (1.0 - exact) / n_draws)
    z = (empirical - exact) / sem
    assert float(np.abs(z).max()) < _Z_TOL, (
        f"empirical marginal is {float(np.abs(z).max()):.2f}σ off the exact one at "
        f"position {int(np.abs(z).argmax())} — a placement bug, not sampling noise")
    mean_masked = float(masked_per_draw.mean())
    exact_masked = float(_masked_patch_mean())
    z_masked = ((mean_masked - exact_masked)
                / (masked_per_draw.std(ddof=1) / np.sqrt(n_draws)))
    assert abs(z_masked) < _Z_TOL, (
        f"mean masked patches {mean_masked:.4f} vs the exact {exact_masked:.6f} "
        f"({z_masked:.2f}σ) — the length distribution moved (per-element rejection?)")

    # Per-d shares against d_balance's enumeration; SAMPLER_REFERENCE comes from these numbers.
    ref = np.asarray(d_distribution(
        tuple(MASK_SPAN_LENGTHS), int(MAX_MASKED_PATCHES), int(MASK_MAX_SPANS),
        int(MIN_CONTEXT_PATCHES), int(MAX_CONTEXT_PATCHES), int(PREDICTION_PATCHES),
        float(MASK_RIGHT_EDGE_QUOTA)))
    assert abs(ref.sum() - 1.0) < 1e-9
    lengths_T = np.arange(MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES + 1) + PREDICTION_PATCHES
    rng = np.random.default_rng(90210)
    group_counts = np.zeros((n_draws, N_D_GROUPS))
    per_draw = np.zeros(n_draws)
    for i in range(n_draws):
        Ti = int(lengths_T[rng.integers(len(lengths_T))])
        _mask_idx, valid, d, _anchor = _mask_slots(sample_mask_spans(Ti, rng), Ti)
        for dv in d[valid].tolist():
            group_counts[i, _group(dv) - 1] += 1
            per_draw[i] += 1
    emp = group_counts.sum(0) / per_draw.sum()
    zs = []
    for g in range(N_D_GROUPS):
        resid = group_counts[:, g] - ref[g] * per_draw
        sem_g = resid.std(ddof=1) / np.sqrt(n_draws) / per_draw.mean()
        zg = (emp[g] - ref[g]) / sem_g
        zs.append(zg)
        assert abs(zg) < _Z_TOL, (
            f"d group {g + 1}: sampler gives {emp[g] * 100:.3f}% against "
            f"d_balance's enumerated {ref[g] * 100:.3f}% ({zg:.2f}σ) — every "
            "d-binned metric is being read against a mixture the sampler does not "
            "draw")
    # SAMPLER_REFERENCE is a FROZEN copy; sampler_reference_applies() checks only the knobs.
    from metrics.protocols import SAMPLER_REFERENCE
    for g in range(N_D_GROUPS):
        frozen = SAMPLER_REFERENCE['share_pct'][g + 1] / 100.0
        assert abs(frozen - ref[g]) < 1e-5, (
            f"SAMPLER_REFERENCE['share_pct'][{g + 1}] is {frozen * 100:.3f}% against "
            f"d_balance's {ref[g] * 100:.3f}% — the frozen copy has gone stale; "
            f"re-enumerate it at the live knobs")
    exact_mean = float(_masked_patch_mean())
    assert abs(SAMPLER_REFERENCE['mean_masked'] - exact_mean) < 1e-3, (
        f"SAMPLER_REFERENCE['mean_masked'] {SAMPLER_REFERENCE['mean_masked']} vs the "
        f"enumerated {exact_mean:.4f}")
    for g in range(N_D_GROUPS):
        frozen_pps = SAMPLER_REFERENCE['patches_per_sample'][g + 1]
        assert abs(frozen_pps - ref[g] * exact_mean) < 1e-3, (
            f"SAMPLER_REFERENCE['patches_per_sample'][{g + 1}] {frozen_pps} vs the "
            f"enumerated {ref[g] * exact_mean:.4f}")

    print(f"\n[DUMP] sampler vs closed form | T={T}, {n_draws} draws: worst position "
          f"{float(np.abs(z).max()):.2f}σ (tol {_Z_TOL}σ); mean masked {mean_masked:.4f} "
          f"vs {exact_masked:.4f}; d groups "
          f"{[f'{e * 100:.2f}%' for e in emp]} vs {[f'{r * 100:.2f}%' for r in ref]} "
          f"(worst {max(abs(x) for x in zs):.2f}σ) ✓")
