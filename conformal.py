"""Split-conformal quantile recalibration for the BG quantile fan, in mg/dL.

Raw bands under-cover at excursion peaks, the hypo edge worst: ≈0.77 against a 0.90
target, with the τ=0.10 edge escaped ≈0.20 of the time. This module fits a PER-STEP,
PER-QUANTILE additive correction ``delta[s, k]`` on a held-out calibration set and
applies it to fresh forecasts. Each level is fit from its own residuals, which is what
corrects the hypo side harder than the hyper side.

Load-bearing invariants:
  * the MEDIAN (τ=0.5) is held FIXED — ``delta`` is forced to 0 there;
  * the applied fan stays MONOTONE non-decreasing in τ;
  * an all-zero ``delta`` makes ``apply`` the identity.

Must be RE-FIT per target distribution — validity rests on calibration/test
exchangeability.
"""
from __future__ import annotations

import numpy as np


def _conformal_offset(residuals: np.ndarray, tau: float) -> float:
    """Finite-sample-valid empirical ``tau``-quantile of the calibration residuals.

    ``tau`` is the LEVEL of the band edge being calibrated. UPPER edge
    (``tau >= 0.5``) takes order statistic ``ceil((n+1)·tau)``, 1-indexed; LOWER edge
    (``tau < 0.5``) takes ``floor((n+1)·tau)``. ``ceil`` on a lower edge is one order
    statistic too high — it sits the edge too high and is anti-conservative on exactly
    the load-bearing hypo edge, worst at small calibration N. The index is clamped
    into range.
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
    """Fit per-step, per-quantile additive mg/dL corrections on a calibration set.

    Args:
        cal_q: ``(N, S, K)`` calibration quantile forecasts (mg/dL), ascending in K.
        cal_true: ``(N, S)`` calibration true BG (mg/dL).
        levels: the ``K`` quantile levels τ, ascending — ``QUANTILE_LEVELS``.
        median_idx: index of τ=0.5 in ``levels``; held fixed at delta 0.

    Returns:
        ``delta`` ``(S, K)`` mg/dL; the calibrated quantile is ``q + delta``, before
        monotonicity enforcement.
    """
    assert cal_q.ndim == 3 and cal_true.ndim == 2, (cal_q.shape, cal_true.shape)
    N, S, K = cal_q.shape
    assert cal_true.shape == (N, S) and K == len(levels), (cal_true.shape, K, len(levels))
    delta = np.zeros((S, K), dtype=np.float64)
    for s in range(S):
        for k, tau in enumerate(levels):
            if k == median_idx:
                continue                      # median held fixed
            # delta = τ-quantile of (true − q_k), so q_k + delta lands on the
            # empirical τ-quantile of the truth.
            r = cal_true[:, s] - cal_q[:, s, k]
            r = r[np.isfinite(r)]   # np.sort sends NaN to the high-tau tail
            if r.size == 0:
                continue
            delta[s, k] = _conformal_offset(r, tau)
    return delta


def band_coverage(q: np.ndarray, true: np.ndarray, lo_idx: int, hi_idx: int) -> np.ndarray:
    """``(S,)`` per-step empirical coverage of the ``[lo_idx, hi_idx]`` band.

    q: ``(N, S, K)`` quantile fan (mg/dL), ascending in K. true: ``(N, S)`` mg/dL.
    lo_idx / hi_idx: column indices of the band's lower / upper edge.
    """
    assert q.ndim == 3 and true.ndim == 2, (q.shape, true.shape)
    N, S, K = q.shape
    assert true.shape == (N, S), (true.shape, (N, S))
    inside = (true >= q[:, :, lo_idx]) & (true <= q[:, :, hi_idx])   # (N, S)
    return inside.mean(axis=0).astype(np.float64)


def apply_quantile_conformal(q: np.ndarray, delta: np.ndarray, median_idx: int) -> np.ndarray:
    """Apply a fitted ``delta``, keeping the fan monotone and the median fixed.

    Args:
        q: ``(..., S, K)`` quantile fan (mg/dL), ascending in K.
        delta: ``(S, K)`` corrections from :func:`fit_quantile_conformal`.
        median_idx: index of τ=0.5; that column is left untouched.

    Returns:
        ``(..., S, K)`` calibrated fan; an all-zero ``delta`` returns ``q`` unchanged.
    """
    # (S, K) only: a 1-D (K,) delta would broadcast silently across every step.
    assert delta.ndim == 2 and delta.shape == (q.shape[-2], q.shape[-1]), (
        delta.shape, q.shape)
    qc = q + delta
    qc[..., median_idx] = q[..., median_idx]
    K = q.shape[-1]
    # Clamp outward from the fixed median — no quantile crossing.
    for k in range(median_idx - 1, -1, -1):
        qc[..., k] = np.minimum(qc[..., k], qc[..., k + 1])
    for k in range(median_idx + 1, K):
        qc[..., k] = np.maximum(qc[..., k], qc[..., k - 1])
    return qc
