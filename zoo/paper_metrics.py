"""Per-horizon forecast metrics after Jacobs et al., IEEE RBME 2024.

Accuracy, bias, glucose-specific error, clinical grids, hypo detection, time gain, paired tests.
Inputs are mg/dL arrays."""
from __future__ import annotations

import numpy as np

HYPO_MGDL = 70.0


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gmse_penalty(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Del Favero et al. 2012 gMSE penalty: overestimates in hypo and underestimates in hyper."""
    a_l, a_h, t_l, t_h, b_l, b_h, g_l, g_h = 1.5, 1.0, 85.0, 155.0, 30.0, 100.0, 10.0, 20.0
    low = 1.0 - _smoothstep(true, t_l - b_l, t_l)
    high = _smoothstep(true, t_h, t_h + b_h)
    over = _smoothstep(pred - true, 0.0, g_l)
    under = _smoothstep(true - pred, 0.0, g_h)
    return 1.0 + a_l * low * over + a_h * high * under


def clarke_zones(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Clarke zone index 0..4 = A..E."""
    from metrics.core.suite import _clarke
    z = np.full(len(true), 1, np.int8)
    for i, m in enumerate(_clarke(pred, true)):
        z[m] = i
    return z


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float | None:
    d = float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return None if d == 0 else float((tp * tn - fp * fn) / np.sqrt(d))


def point_metrics(true: np.ndarray, pred: np.ndarray, origin: np.ndarray, horizon_min: int) -> dict:
    """One horizon's metrics over rows where truth, prediction and the origin reading are finite."""
    import dts_grid
    from cg_ega import cg_ega_counts, cg_ega_fractions
    ok = np.isfinite(true) & np.isfinite(pred) & (true > 0) & np.isfinite(origin)
    t, p, o = true[ok], pred[ok], origin[ok]
    e = p - t
    out: dict = {'n': int(ok.sum())}
    if not len(t):
        return out
    dts = np.bincount(dts_grid.dts_zones(np.clip(t, 40.0, None), p), minlength=5) / len(t)
    clk = np.bincount(clarke_zones(t, p), minlength=5) / len(t)
    th, ph = t < HYPO_MGDL, p < HYPO_MGDL
    tp, fp = int((th & ph).sum()), int((~th & ph).sum())
    fn, tn = int((th & ~ph).sum()), int((~th & ~ph).sum())
    cg = cg_ega_fractions(cg_ega_counts(t[:, None], p[:, None], o, freq_min=float(horizon_min)))
    out |= {
        'rmse': float(np.sqrt(np.mean(e ** 2))),
        'mae': float(np.mean(np.abs(e))),
        'mard': float(100 * np.mean(np.abs(e) / t)),
        'me': float(np.mean(e)),
        'mre': float(100 * np.mean(e / t)),
        'grmse': float(np.sqrt(np.mean(e ** 2 * gmse_penalty(t, p)))),
        'rmse_hypo': float(np.sqrt(np.mean(e[th] ** 2))) if th.any() else None,
        'me_hypo': float(np.mean(e[th])) if th.any() else None,
        'dts_a': float(100 * dts[0]), 'dts_ab': float(100 * dts[:2].sum()),
        'clarke_a': float(100 * clk[0]), 'clarke_ab': float(100 * clk[:2].sum()),
        'hypo_sens': tp / (tp + fn) if tp + fn else None,
        'hypo_spec': tn / (tn + fp) if tn + fp else None,
        'hypo_ppv': tp / (tp + fp) if tp + fp else None,
        'hypo_mcc': _mcc(tp, fp, fn, tn),
        'hypo_n': tp + fn,
    }
    out |= {f'cgega_{k}': (None if v is None else float(100 * v)) for k, v in cg.items()}
    return out


def time_gain(ts: np.ndarray, pred: np.ndarray, truth_ts: np.ndarray, truth: np.ndarray,
              horizon_min: int) -> float | None:
    """Horizon minus the lag (5-min steps, 0..horizon) of peak forecast-CGM correlation.

    ``ts`` origin epoch s of each forecast; ``truth_ts``/``truth`` the measured CGM."""
    target = ts + horizon_min * 60
    order = np.argsort(truth_ts)
    tts, tv = truth_ts[order], truth[order]
    best, best_k = -np.inf, None
    for k in range(horizon_min // 5 + 1):
        q = target - k * 300
        i = np.clip(np.searchsorted(tts, q), 0, len(tts) - 1)
        y = np.where(tts[i] == q, tv[i], np.nan)
        ok = np.isfinite(y) & np.isfinite(pred)
        if ok.sum() < 100:
            continue
        r = float(np.corrcoef(pred[ok], y[ok])[0, 1])
        if r > best:
            best, best_k = r, k
    return None if best_k is None else float(horizon_min - 5 * best_k)


def wilcoxon_abs_error(true: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank on paired |error|: (mean |a| - |b| mg/dL, p)."""
    from scipy.stats import wilcoxon
    ok = np.isfinite(true) & np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[ok] - true[ok]) - np.abs(b[ok] - true[ok])
    if not np.any(d):
        return 0.0, 1.0
    return float(np.mean(d)), float(wilcoxon(d, zero_method='wilcox').pvalue)
