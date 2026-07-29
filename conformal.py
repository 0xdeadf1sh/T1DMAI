"""Split-conformal quantile recalibration for the BG quantile fan.

The model's quantile bands are *over-confident and asymmetric* in exactly the
excursion windows that matter: at an excursion peak the realized BG escapes the
band far more than the nominal rate — and the lower (hypo) side worse than the
upper. On the verified full run, reserved-partition split-conformal lifts
excursion-peak coverage from ≈ 0.77 toward ≈ 0.84 (target 0.90) and pulls the
τ=0.10 hypo-edge escape rate from ≈ 0.20 down toward ≈ 0.12.

This module fits a PER-HORIZON, PER-QUANTILE additive correction ``delta[s, k]``
on a held-out calibration set so each calibrated quantile achieves its nominal
coverage, then applies it to fresh forecasts. The fit is asymmetric by
construction (each level k is recalibrated independently from its own residuals),
which is what corrects the hypo side harder than the hyper side.

Invariants:
  * The MEDIAN (τ=0.5) is held FIXED — ``delta`` is forced to 0 there, so the
    point forecast is untouched (conformal moves only the band edges).
  * The applied fan stays MONOTONE non-decreasing in τ (no quantile crossing):
    after adding ``delta`` the edges are clamped outward from the fixed median.
  * ``delta`` of all zeros ⇒ ``apply`` returns the input unchanged.

The fit needs only point quantiles + truth (no MC sampling), works identically on
simulator and real-CGM data, and must be RE-FIT per target distribution (sim,
each real cohort) — its validity rests on calibration/test exchangeability.
"""
from __future__ import annotations

import numpy as np


def _conformal_offset(residuals: np.ndarray, tau: float) -> float:
    """The finite-sample-valid empirical ``tau``-quantile of calibration residuals.

    ``tau`` is the quantile LEVEL of the band edge being calibrated (e.g. 0.05 for
    the lower 90% edge, 0.95 for the upper). The conformal order statistic differs
    by side:

      * UPPER edge (``tau >= 0.5``): ``ceil((n+1)·tau)`` (1-indexed) — the
        split-conformal valid choice for an upper bound.
      * LOWER edge (``tau < 0.5``): ``floor((n+1)·tau)`` — the symmetric valid
        choice for a *lower* bound. Using ``ceil`` here (one order statistic too
        high) sits the lower edge too high and lets realized BG escape BELOW it
        more than the nominal rate — anti-conservative on exactly the clinically
        load-bearing hypo edge, and biased most at small calibration N.

    The index is clamped into range so ``q_k + offset`` covers at the nominal level.
    """
    s = np.sort(residuals)
    n = len(s)
    if tau < 0.5:
        idx = int(np.floor((n + 1) * tau)) - 1
    else:
        idx = int(np.ceil((n + 1) * tau)) - 1
    idx = min(max(idx, 0), n - 1)
    return float(s[idx])


def fit_quantile_conformal(cal_q: np.ndarray, cal_true: np.ndarray,
                           levels: tuple[float, ...], median_idx: int) -> np.ndarray:
    """Fit per-horizon, per-quantile additive corrections on a calibration set.

    Args:
        cal_q:    ``(N, S, K)`` calibration quantile forecasts (mg/dL), ascending in K.
        cal_true: ``(N, S)`` calibration true BG (mg/dL).
        levels:   the ``K`` quantile levels τ (ascending), ``QUANTILE_LEVELS``.
        median_idx: index of the τ=0.5 median in ``levels`` (held fixed: delta=0).

    Returns:
        ``delta`` ``(S, K)`` additive mg/dL corrections; ``delta[:, median_idx] == 0``.
        The calibrated quantile is ``q + delta`` (before monotonicity enforcement).
    """
    assert cal_q.ndim == 3 and cal_true.ndim == 2, (cal_q.shape, cal_true.shape)
    N, S, K = cal_q.shape
    assert cal_true.shape == (N, S) and K == len(levels), (cal_true.shape, K, len(levels))
    delta = np.zeros((S, K), dtype=np.float64)
    for s in range(S):
        for k, tau in enumerate(levels):
            if k == median_idx:
                continue                      # median held fixed
            # Residual true − q_k: we want q_k + delta to be the empirical τ-quantile
            # of the truth, i.e. delta = τ-quantile of (true − q_k).
            r = cal_true[:, s] - cal_q[:, s, k]
            r = r[np.isfinite(r)]   # drop non-finite residuals: np.sort sends NaN to
            if r.size == 0:         # the high-tau tail, poisoning the upper deltas
                continue            # no finite calibration data at this (step,level)
            delta[s, k] = _conformal_offset(r, tau)
    return delta


def band_coverage(q: np.ndarray, true: np.ndarray, lo_idx: int, hi_idx: int) -> np.ndarray:
    """Per-step empirical coverage of the ``[lo_idx, hi_idx]`` band over a window batch.

    Args:
        q:    ``(N, S, K)`` quantile fan (mg/dL), ascending in K.
        true: ``(N, S)`` true BG (mg/dL).
        lo_idx, hi_idx: column indices of the band's lower / upper edge.

    Returns:
        ``(S,)`` fraction of windows whose truth lands in ``[q_lo, q_hi]`` at each step.
        Pure numpy; touches none of the fit/apply internals, so the conformal
        invariants are unaffected.
    """
    assert q.ndim == 3 and true.ndim == 2, (q.shape, true.shape)
    N, S, K = q.shape
    assert true.shape == (N, S), (true.shape, (N, S))
    inside = (true >= q[:, :, lo_idx]) & (true <= q[:, :, hi_idx])   # (N, S)
    return inside.mean(axis=0).astype(np.float64)


def apply_quantile_conformal(q: np.ndarray, delta: np.ndarray, median_idx: int) -> np.ndarray:
    """Apply a fitted ``delta`` to a quantile fan, keeping it monotone + median-fixed.

    Args:
        q:     ``(..., S, K)`` quantile fan (mg/dL), ascending in K.
        delta: ``(S, K)`` corrections from :func:`fit_quantile_conformal`.
        median_idx: index of the τ=0.5 median (its delta is 0; left untouched).

    Returns:
        ``(..., S, K)`` calibrated fan, ascending in K with the median column intact.
        ``delta`` all-zero ⇒ returns ``q`` unchanged.
    """
    assert delta.ndim == 2 and delta.shape == (q.shape[-2], q.shape[-1]), (
        delta.shape, q.shape)   # (S, K) only; a 1-D (K,) delta would broadcast
                                # silently across every horizon step otherwise
    qc = q + delta                            # broadcast (S, K) over leading dims
    qc[..., median_idx] = q[..., median_idx]  # median exactly preserved
    K = q.shape[-1]
    # Enforce monotonicity outward from the fixed median (no quantile crossing):
    # lower edges must not exceed the next-higher edge; upper edges not fall below
    # the next-lower edge.
    for k in range(median_idx - 1, -1, -1):
        qc[..., k] = np.minimum(qc[..., k], qc[..., k + 1])
    for k in range(median_idx + 1, K):
        qc[..., k] = np.maximum(qc[..., k], qc[..., k - 1])
    return qc
