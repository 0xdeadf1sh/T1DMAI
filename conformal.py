"""Split-conformal quantile recalibration for the BG quantile fan, in mg/dL.
Fits a per-step, per-quantile additive delta[s,k] on held-out calibration residuals.
Median (tau=0.5) held FIXED; applied fan stays MONOTONE; delta=0 is the identity.
Must be RE-FIT per target distribution — validity rests on cal/test exchangeability.
"""
from __future__ import annotations

import numpy as np


def _conformal_offset(residuals: np.ndarray, tau: float) -> float:
    """Finite-sample-valid empirical tau-quantile of the calibration residuals.

    Upper edge (tau>=0.5): order statistic ceil((n+1)*tau). Lower edge (tau<0.5):
    floor((n+1)*tau) — ceil there is anti-conservative on the hypo edge. Index clamped.
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

    cal_q (N,S,K) mg/dL ascending in K; cal_true (N,S) mg/dL; levels (K,) tau ascending;
    median_idx held fixed at delta 0. Returns delta (S,K) mg/dL, q+delta before monotonicity.
    """
    assert cal_q.ndim == 3 and cal_true.ndim == 2, (cal_q.shape, cal_true.shape)
    N, S, K = cal_q.shape
    assert cal_true.shape == (N, S) and K == len(levels), (cal_true.shape, K, len(levels))
    delta = np.zeros((S, K), dtype=np.float64)
    for s in range(S):
        for k, tau in enumerate(levels):
            if k == median_idx:
                continue                      # median held fixed
            # delta = tau-quantile of (true - q_k), so q_k+delta lands on the empirical quantile.
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
    """Apply a fitted delta, keeping the fan monotone and the median fixed.

    q (...,S,K) mg/dL ascending in K; delta (S,K) from fit_quantile_conformal; median_idx
    column left untouched. Returns (...,S,K); an all-zero delta returns q unchanged.
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
