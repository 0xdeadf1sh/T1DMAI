"""
Per-``d`` supervision weights for the masked-BG objective.

Uniform mask PLACEMENT does not give uniform supervision by DIFFICULTY.  Under
the sampler in ``config``, ``d`` -- the distance in patches to the nearest
visible evidence on either side -- is distributed roughly 71 / 27 / 1.3 / 0.6 %
over ``d = 1..4`` at arm ``a``.  The deployed 2 h forecast needs one patch at
each of ``d = 1..4``, so its far half is supervised 16x and 35x less than its
near half.  ``python d_balance.py`` prints the live figures for the active arm.

This module reweights the LOSS so each ``d`` group carries equal mass.  It does
not touch placement: the sampler is unchanged and stays uniform, and span length
is the lever the plan sanctions for moving the ``d`` histogram.

Two design points worth stating, because both are load-bearing:

  * **The tail is pooled, not binned.**  Weights flatten ``d = 1, 2, 3`` and
    ``d >= 4`` -- four groups.  ``d > N_D_GROUPS`` needs an edge-touching span
    longer than ``N_D_GROUPS`` patches, so it is reachable at every arm whose
    span ceiling exceeds 4; it lies beyond the deployed horizon, and giving it
    its own bin would hand a very rare group a very large weight.
  * **Widening comes first, reweighting second.**  Flattening by weights alone
    is possible at any span ceiling, but the cost is not.  At arm ``a``'s ceiling
    of 4 the ``d >= 4`` weight is 41x and the effective sample size falls to
    6.5 % -- a 15x variance inflation that makes the far horizons noise.  At the
    ceiling of 8 and 12 slots of arms ``b`` and ``c`` the same target costs 3.3x
    and keeps 64 % ESS.  Report ``effective_sample_size()`` beside any run that
    enables this.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import comb

from config import (
    D_BALANCED_LOSS,
    MASK_MAX_SPANS,
    MASK_SPAN_LENGTHS,
    MAX_CONTEXT_PATCHES,
    MAX_MASKED_PATCHES,
    MIN_CONTEXT_PATCHES,
    PREDICTION_PATCHES,
    SAMPLER_ARM,
)

# Groups the weights flatten.  The last is a pooled tail: every d at or above it
# shares one weight.
N_D_GROUPS = 4


def _group(d: int) -> int:
    """1-based group index; everything at or above N_D_GROUPS pools into it."""
    return min(d, N_D_GROUPS)


@lru_cache(maxsize=8)
def d_distribution(
    span_lengths: tuple[int, ...],
    max_masked: int,
    max_spans: int,
    min_ctx: int,
    max_ctx: int,
    pred: int,
) -> tuple[float, ...]:
    """Exact share of masked patches in each ``d`` group, by enumeration.

    Enumerated rather than sampled: at 1e5 draws the per-position 1-sigma is
    about 0.99 % of the mean, several times the effect being measured, so a
    correct sampler still reports the wrong answer in every replicate.

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

    for T in Ts:
        for w, L in vecs:
            n = len(L)
            slack = T - sum(L) - (n - 1)
            if slack < 0:
                continue
            allc = comb(slack + n, n)
            for i in range(1, n + 1):
                base = sum(L[: i - 1]) + (i - 1)
                Li = L[i - 1]
                for k in range(slack + 1):
                    pk = comb(k + i - 1, i - 1) * comb(slack - k + n - i, n - i) / allc
                    if pk == 0.0:
                        continue
                    S = base + k
                    left_edge, right_edge = S == 0, S + Li == T
                    for o in range(Li):
                        if left_edge and not right_edge:
                            d = Li - o
                        elif right_edge and not left_edge:
                            d = o + 1
                        else:
                            d = min(o + 1, Li - o)
                        mass[_group(d) - 1] += pT * w * pk
                        total += pT * w * pk

    return tuple(m / total for m in mass)


def _current_distribution() -> tuple[float, ...]:
    return d_distribution(
        tuple(MASK_SPAN_LENGTHS),
        int(MAX_MASKED_PATCHES),
        int(MASK_MAX_SPANS),
        int(MIN_CONTEXT_PATCHES),
        int(MAX_CONTEXT_PATCHES),
        int(PREDICTION_PATCHES),
    )


def d_weights() -> tuple[float, ...]:
    """Per-group multiplier making each ``d`` group carry equal loss mass.

    ``w[g] = (1/N_D_GROUPS) / p[g]``, normalised so ``E[w] == 1`` under the
    sampler -- i.e. the weighted loss keeps the same scale as the unweighted one
    and ``log_sigma_Q`` is not handed a level shift to absorb.
    """
    p = _current_distribution()
    w = [(1.0 / N_D_GROUPS) / max(pi, 1e-12) for pi in p]
    mean = sum(pi * wi for pi, wi in zip(p, w))
    return tuple(wi / mean for wi in w)


def effective_sample_size() -> float:
    """``ESS/N`` under the weights, in ``(0, 1]``.

    ``(E[w])^2 / E[w^2]``.  1.0 means the weights are uniform and cost nothing;
    0.065 means 15x the variance of an unweighted batch.  Quote it with any
    result produced under weighting -- a per-``d`` figure computed at 6 % ESS is
    not comparable with one computed at 64 %.
    """
    p = _current_distribution()
    w = d_weights()
    ew = sum(pi * wi for pi, wi in zip(p, w))
    ew2 = sum(pi * wi * wi for pi, wi in zip(p, w))
    return (ew * ew) / ew2


def summary() -> str:
    p, w = _current_distribution(), d_weights()
    ess = effective_sample_size()
    rows = "  ".join(
        f"d{'>=' if g == N_D_GROUPS - 1 else '='}{g + 1}: p={pi * 100:.2f}% w={wi:.2f}"
        for g, (pi, wi) in enumerate(zip(p, w))
    )
    return (
        f"arm {SAMPLER_ARM}: MASK_SPAN_LENGTHS={tuple(MASK_SPAN_LENGTHS)} "
        f"MAX_MASKED_PATCHES={MAX_MASKED_PATCHES} D_BALANCED_LOSS={D_BALANCED_LOSS}\n"
        f"  {rows}\n  ESS = {ess * 100:.1f}%  max weight = {max(w):.2f}x"
    )


if __name__ == "__main__":
    print(summary())
