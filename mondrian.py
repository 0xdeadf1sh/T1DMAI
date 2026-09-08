"""Region-binned (Mondrian) split-conformal recalibration of the BG quantile fan.

Bins on where the forecast is HEADING (median line's last-patch mean), not last observed BG.
Below n=39 a bin takes the MARGINAL delta (n>=19 for any offset); only FORECAST ships.
"""
from __future__ import annotations

import numpy as np

import conformal
from config import PATCH_SIZE

REGION_EDGES: tuple[float, ...] = (110.0,)
N_REGION_BINS: int = len(REGION_EDGES) + 1

# Order-statistic floors of ``conformal._conformal_offset`` at tau = 0.05.
MIN_N_OFFSET_EXISTS: int = 19
MIN_N_OWN_FIT: int = 39

# The final patch, not the last step: one 5-min grid point is noisier than the 30 min it closes on.
DESTINATION_STEPS: int = PATCH_SIZE


def bin_label(b: int) -> str:
    """Human-readable interval of region bin ``b`` in mg/dL."""
    lo = '-inf' if b == 0 else f"{REGION_EDGES[b - 1]:.0f}"
    hi = 'inf' if b == len(REGION_EDGES) else f"{REGION_EDGES[b]:.0f}"
    return f"[{lo}, {hi})"


def forecast_destination(q: np.ndarray, median_idx: int) -> np.ndarray:
    """Where each window's forecast is HEADING, mg/dL — the region variable.

    ``q`` (N,S,K) mg/dL fan ascending in K; mean of the ``median_idx`` column over the last
    ``DESTINATION_STEPS`` steps. Invariant under conformal correction, which holds the median fixed.
    """
    assert q.ndim == 3, q.shape
    return q[:, -DESTINATION_STEPS:, median_idx].mean(axis=1).astype(np.float64)


def region_bin(destination: np.ndarray) -> np.ndarray:
    """``(N,)`` region-bin index in ``[0, N_REGION_BINS)`` for each destination."""
    d = np.asarray(destination, dtype=np.float64)
    assert np.isfinite(d).all(), "non-finite forecast destination — cannot bin"
    return np.searchsorted(np.asarray(REGION_EDGES, dtype=np.float64), d,
                           side='right').astype(np.int64)


def _distinct_patients(patients, rows: np.ndarray):
    """Distinct-patient count over ``rows``, or ``None`` when no ids were given."""
    if patients is None:
        return None
    return int(len({patients[i] for i in rows}))


def fit_mondrian(cal_q: np.ndarray, cal_true: np.ndarray, cal_bin: np.ndarray,
                 levels: tuple[float, ...], median_idx: int,
                 patients=None, protocol: str = 'forecast',
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit one conformal correction per region bin, plus the marginal baseline.

    Gives delta (N_REGION_BINS,S,K), marginal (S,K), and a meta dict with n/n_patients/own_fit.
    Marginal is both the thin-bin fallback and pre-Mondrian baseline, compared within one run.
    """
    assert cal_q.ndim == 3 and cal_true.ndim == 2, (cal_q.shape, cal_true.shape)
    N, S, K = cal_q.shape
    assert cal_true.shape == (N, S), (cal_true.shape, (N, S))
    cal_bin = np.asarray(cal_bin, dtype=np.int64)
    assert cal_bin.shape == (N,), (cal_bin.shape, N)
    assert ((cal_bin >= 0) & (cal_bin < N_REGION_BINS)).all(), "bin index out of range"
    assert protocol in ('forecast', 'infill'), protocol

    marginal = conformal.fit_quantile_conformal(cal_q, cal_true, levels, median_idx)

    delta = np.zeros((N_REGION_BINS, S, K), dtype=np.float64)
    bins: list[dict] = []
    for b in range(N_REGION_BINS):
        rows = np.flatnonzero(cal_bin == b)
        n = int(rows.size)
        if n >= MIN_N_OWN_FIT:
            delta[b] = conformal.fit_quantile_conformal(
                cal_q[rows], cal_true[rows], levels, median_idx)
            own, reason = True, None
        else:
            delta[b] = marginal
            own = False
            if n == 0:
                reason = "no calibration rows fell in this bin"
            else:
                reason = (f"n={n} < {MIN_N_OWN_FIT}: own tau=0.05 offset would be the "
                          f"calibration-sample minimum"
                          + ("" if n >= MIN_N_OFFSET_EXISTS else
                             f" (and n < {MIN_N_OFFSET_EXISTS}, below which no offset "
                             f"index exists at all)"))
        rec = {'bin': b, 'label': bin_label(b), 'n': n,
               'n_patients': _distinct_patients(patients, rows),
               'own_fit': own, 'fallback_reason': reason}
        bins.append(rec)
        if verbose:
            tag = 'own fit' if own else 'MARGINAL FALLBACK'
            print(f"[{protocol.upper()}-CONFORMAL] region {rec['label']:>12} "
                  f"n={n:5d} patients={rec['n_patients']}  {tag}"
                  + ("" if own else f"  <- {reason}"))

    meta = {
        'layout': 'mondrian',
        'protocol': protocol,
        'shipped': protocol == 'forecast',
        'region_edges': list(REGION_EDGES),
        'region_variable': f"median forecast, mean over the last {DESTINATION_STEPS} steps",
        'min_n_own_fit': MIN_N_OWN_FIT,
        'min_n_offset_exists': MIN_N_OFFSET_EXISTS,
        'n_cal': int(N), 'horizon_steps': int(S), 'levels': list(levels),
        'median_idx': int(median_idx),
        'bins': bins,
        'n_fallback_bins': int(sum(0 if r['own_fit'] else 1 for r in bins)),
    }
    return delta, marginal, meta


def fit_infill_conformal(cal_q: np.ndarray, cal_true: np.ndarray, cal_bin: np.ndarray,
                         levels: tuple[float, ...], median_idx: int,
                         patients=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """The INFILL protocol's own coarse fit — never part of the shipped band.

    Same region axis as forecast, nothing finer (crossing with d/sidedness would multiply
    bins by six against the n>=19 floor). Every marginal fallback is announced on stdout.
    """
    delta, marginal, meta = fit_mondrian(
        cal_q, cal_true, cal_bin, levels, median_idx,
        patients=patients, protocol='infill', verbose=True)
    assert meta['shipped'] is False
    return delta, marginal, meta


def apply_mondrian(q: np.ndarray, delta: np.ndarray, bin_idx: np.ndarray,
                   median_idx: int) -> np.ndarray:
    """Apply fit_mondrian's (N_REGION_BINS,S,K) delta to a (N,S,K) mg/dL fan.

    Windows GROUPED by bin_idx (N,), each group passed that bin's (S,K) slice; median column
    untouched. A gathered per-window delta is never formed: apply_quantile_conformal asserts 2-D.
    """
    assert q.ndim == 3, q.shape
    N, S, K = q.shape
    assert delta.ndim == 3 and delta.shape == (N_REGION_BINS, S, K), (delta.shape, q.shape)
    bin_idx = np.asarray(bin_idx, dtype=np.int64)
    assert bin_idx.shape == (N,), (bin_idx.shape, N)
    assert ((bin_idx >= 0) & (bin_idx < N_REGION_BINS)).all(), "bin index out of range"

    out = np.empty_like(q, dtype=np.float64)
    covered = 0
    for b in range(N_REGION_BINS):
        rows = np.flatnonzero(bin_idx == b)
        if rows.size == 0:
            continue
        out[rows] = conformal.apply_quantile_conformal(q[rows], delta[b], median_idx)
        covered += int(rows.size)
    assert covered == N, f"{covered} of {N} windows were assigned a bin"
    return out


def mean_band_width(q: np.ndarray, lo_idx: int, hi_idx: int) -> float:
    """Mean ``q_hi - q_lo`` (mg/dL) over windows and horizon steps."""
    if q.shape[0] == 0:
        return float('nan')
    return float(np.mean(q[:, :, hi_idx] - q[:, :, lo_idx]))


def forecast_d_step_groups(n_pred_patches: int, patch_size: int) -> dict:
    """``{'d1': steps, ...}`` for the right-edge FORECAST protocol.

    No right neighbour, so patch ``p`` sits at one-sided ``d = p + 1`` and every
    step of that patch inherits it.
    """
    return {f"d{p + 1}": np.arange(p * patch_size, (p + 1) * patch_size)
            for p in range(n_pred_patches)}


def d_step_groups(d_per_step) -> dict:
    """``{'d<k>': steps}`` from a per-step ``d`` vector (any protocol)."""
    d_per_step = np.asarray(d_per_step)
    return {f"d{int(v)}": np.flatnonzero(d_per_step == v)
            for v in sorted(set(d_per_step.tolist()))}


def bin_report(arms: dict, true: np.ndarray, bin_idx: np.ndarray,
               lo_idx: int, hi_idx: int, patients=None,
               step_groups: dict | None = None) -> list[dict]:
    """Per-bin coverage WITH its n, distinct-patient count and mean width, per arm.

    ``arms`` {name: (N,S,K) fan}, all scored on the same windows. ``step_groups`` adds a
    per-``d`` ``by_d`` block: pooling over ``d`` mixes difficulties and isn't a selection metric.
    """
    bin_idx = np.asarray(bin_idx, dtype=np.int64)
    rows_out: list[dict] = []
    for b in range(N_REGION_BINS):
        rows = np.flatnonzero(bin_idx == b)
        rec: dict = {'bin': b, 'label': bin_label(b), 'n': int(rows.size),
                     'n_patients': _distinct_patients(patients, rows), 'arms': {}}
        for name, q in arms.items():
            if rows.size == 0:
                rec['arms'][name] = {'cov': None, 'width': None}
                continue
            cov = conformal.band_coverage(q[rows], true[rows], lo_idx, hi_idx)  # (S,)
            rec['arms'][name] = {'cov': float(np.mean(cov)),
                                 'width': mean_band_width(q[rows], lo_idx, hi_idx)}
        if step_groups:
            by_d: dict = {}
            for label, steps in step_groups.items():
                steps = np.asarray(steps)
                by_d[label] = {'n_steps': int(steps.size), 'arms': {}}
                for name, q in arms.items():
                    if rows.size == 0:
                        by_d[label]['arms'][name] = {'cov': None, 'width': None}
                        continue
                    qs = q[np.ix_(rows, steps)]
                    ts = true[np.ix_(rows, steps)]
                    by_d[label]['arms'][name] = {
                        'cov': float(np.mean((ts >= qs[:, :, lo_idx])
                                             & (ts <= qs[:, :, hi_idx]))),
                        'width': mean_band_width(qs, lo_idx, hi_idx)}
            rec['by_d'] = by_d
        rows_out.append(rec)
    return rows_out


def print_bin_report(report: list[dict], target: float, title: str) -> None:
    """Render :func:`bin_report` as a table — coverage never without n and width.

    A ``by_d`` block prints under its bin; the row above is a summary, never selection.
    """
    arms = list(report[0]['arms']) if report else []
    head = f"{'region':>12} {'d':>4} {'n':>6} {'pats':>5} " + " ".join(
        f"{a[:9] + ' cov':>13} {a[:9] + ' wid':>13}" for a in arms)
    print(f"\n{title} (target coverage {target:.2f})")
    print(head)

    def _cells(src: dict) -> str:
        out = []
        for a in arms:
            v = src[a]
            out.append(f"{'--':>13} {'--':>13}" if v['cov'] is None
                       else f"{100.0 * v['cov']:>12.1f}% {v['width']:>13.1f}")
        return " ".join(out)

    for rec in report:
        pats = '--' if rec['n_patients'] is None else str(rec['n_patients'])
        print(f"{rec['label']:>12} {'all':>4} {rec['n']:>6} {pats:>5} "
              + _cells(rec['arms']))
        for label, blk in (rec.get('by_d') or {}).items():
            print(f"{'':>12} {label:>4} {rec['n']:>6} {pats:>5} " + _cells(blk['arms']))
