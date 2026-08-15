"""The exact ``d`` histogram of the masked-BG sampler.

``d`` -- a masked patch's distance in patches to the nearest visible evidence on
either side -- is the only axis a masked-BG metric bins on, and the mixture the
sampler draws is what those bins are read against.  This module enumerates that
mixture exactly from the sampler's own knobs; ``metrics.protocols.SAMPLER_REFERENCE``
is produced from it and checked against it.  ``python d_balance.py`` prints the
live figures.

The enumeration covers BOTH of ``data.sample_mask_spans``' placement branches.
Uniform placement alone leaves the deployed one-sided forecast case at ~3% of
masked slots, which is what ``config.MASK_RIGHT_EDGE_QUOTA`` corrects; a change
to the sampler that is not mirrored here silently moves every ``d``-binned
figure's reference.

The tail is POOLED, not binned: ``d = 1, 2, 3`` and ``d >= 4``.  ``d >
N_D_GROUPS`` needs an edge-touching span longer than ``N_D_GROUPS`` patches, and
it lies beyond the deployed horizon.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import comb

from config import (
    MASK_MAX_SPANS,
    MASK_RIGHT_EDGE_QUOTA,
    MASK_SPAN_LENGTHS,
    MAX_CONTEXT_PATCHES,
    MAX_MASKED_PATCHES,
    MIN_CONTEXT_PATCHES,
    PREDICTION_PATCHES,
)

# Groups the histogram reports.  The last is a pooled tail: every d at or above
# it shares one bin.
N_D_GROUPS = 4


def _group(d: int) -> int:
    """1-based group index; everything at or above N_D_GROUPS pools into it."""
    return min(d, N_D_GROUPS)


def _accumulate_span(mass: list[float], weight: float, S: int, Li: int, T: int) -> float:
    """Add one span's ``Li`` masked patches to ``mass`` at their own ``d``.

    ``d`` is the distance to the nearest visible evidence on EITHER side, so a
    span touching an edge is one-sided there and counts outward from the other.
    """
    left_edge, right_edge = S == 0, S + Li == T
    for o in range(Li):
        if left_edge and not right_edge:
            d = Li - o
        elif right_edge and not left_edge:
            d = o + 1
        else:
            d = min(o + 1, Li - o)
        mass[_group(d) - 1] += weight
    return weight * Li


@lru_cache(maxsize=8)
def d_distribution(
    span_lengths: tuple[int, ...],
    max_masked: int,
    max_spans: int,
    min_ctx: int,
    max_ctx: int,
    pred: int,
    right_edge_quota: float = 0.0,
) -> tuple[float, ...]:
    """Exact share of masked patches in each ``d`` group, by enumeration.

    Enumerated rather than sampled: at 1e5 draws the per-position 1-sigma is
    about 0.99 % of the mean, several times the effect being measured, so a
    correct sampler still reports the wrong answer in every replicate.

    Both of ``data.sample_mask_spans``' placement branches are enumerated. Under
    the right-edge branch the LAST span is pinned at ``T - L_last`` and the rest
    are composed over the prefix that clears it and its separator, which holds
    the same ``slack`` the uniform branch spreads over one more gap. Both branches
    draw ``n_spans`` and the lengths identically, so the quota moves this
    histogram and nothing else.

    Returns a tuple of ``N_D_GROUPS`` shares summing to 1.
    """
    vecs: list[tuple[float, tuple[int, ...]]] = []
    for n in range(1, max_spans + 1):
        cand = [v for v in product(span_lengths, repeat=n) if sum(v) <= max_masked]
        if not cand:
            continue
        for v in cand:
            vecs.append(((1.0 / max_spans) / len(cand), v))

    Ts = list(range(min_ctx + pred, max_ctx + pred + 1))
    pT = 1.0 / len(Ts)
    mass = [0.0] * N_D_GROUPS
    total = 0.0
    q = float(right_edge_quota)

    def compose(m: int, slack: int, weight: float, lengths: tuple[int, ...],
                T: int) -> float:
        """``m`` spans over ``slack`` free patches in ``m + 1`` ordered gaps."""
        acc = 0.0
        allc = comb(slack + m, m)
        for i in range(1, m + 1):
            base = sum(lengths[: i - 1]) + (i - 1)
            Li = lengths[i - 1]
            for k in range(slack + 1):
                pk = comb(k + i - 1, i - 1) * comb(slack - k + m - i, m - i) / allc
                if pk == 0.0:
                    continue
                acc += _accumulate_span(mass, weight * pk, base + k, Li, T)
        return acc

    for T in Ts:
        for w, L in vecs:
            n = len(L)
            slack = T - sum(L) - (n - 1)
            if slack < 0:
                continue
            last = L[-1]
            w_right = q * w
            w_unif = w - w_right

            if w_unif > 0.0:
                total += compose(n, slack, pT * w_unif, L, T)

            if w_right > 0.0:
                wr = pT * w_right
                # The pinned span first, then the prefix composition (if any).
                total += _accumulate_span(mass, wr, T - last, last, T)
                if n > 1:
                    total += compose(n - 1, slack, wr, L[:-1], T)

    return tuple(m / total for m in mass)


def _current_distribution() -> tuple[float, ...]:
    return d_distribution(
        tuple(MASK_SPAN_LENGTHS),
        int(MAX_MASKED_PATCHES),
        int(MASK_MAX_SPANS),
        int(MIN_CONTEXT_PATCHES),
        int(MAX_CONTEXT_PATCHES),
        int(PREDICTION_PATCHES),
        float(MASK_RIGHT_EDGE_QUOTA),
    )


def summary() -> str:
    p = _current_distribution()
    rows = "  ".join(
        f"d{'>=' if g == N_D_GROUPS - 1 else '='}{g + 1}: p={pi * 100:.3f}%"
        for g, pi in enumerate(p)
    )
    return (
        f"MASK_SPAN_LENGTHS={tuple(MASK_SPAN_LENGTHS)} "
        f"MAX_MASKED_PATCHES={MAX_MASKED_PATCHES} "
        f"MASK_RIGHT_EDGE_QUOTA={MASK_RIGHT_EDGE_QUOTA}\n  {rows}"
    )


if __name__ == "__main__":
    print(summary())
