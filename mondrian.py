"""Region-binned (Mondrian) split-conformal recalibration of the BG quantile fan.

``conformal.py`` fits ONE ``delta[s, k]`` over the whole calibration set — a
MARGINAL correction, valid on average and wrong in both directions on the two
regimes that matter: a fan headed for hypoglycaemia and a fan headed for
hyperglycaemia have different residual distributions, and the pooled offset
under-covers one while over-covering the other. This module re-fits that same
correction once per REGION bin, which restores coverage conditional on the
region instead of only on average.

WHAT THE REGION IS. The bin is a function of the WINDOW, taken from where the
forecast is HEADING: the median line's mean over the final patch of the horizon
(``forecast_destination``). It is deliberately not the last observed BG — a
correction that ships is applied to a forecast, and the forecast's own
destination is what selects its regime. ``REGION_EDGES`` is ``(110.0,)``: one
edge, in the euglycaemic band, well clear of the hypo threshold. A bin edge at a
CLINICAL threshold (70) is the one placement to avoid — it puts the boundary
exactly where the alarm decision is made, so the windows that decide the alarm
are split across two separately-fit corrections, and it starves the low bin (3 /
a handful of calibration windows against the n >= 19 floor
below).

Because ``conformal`` holds the median FIXED, a window's region is identical
before and after correction. The binning is therefore not circular and needs no
second pass.

THE ARITHMETIC FLOOR. ``conformal._conformal_offset`` takes order statistic
``floor((n+1)*tau)`` on the lower edge, so at tau = 0.05:

  * ``n >= 19`` — the index reaches 1, i.e. an offset exists at all;
  * ``n >= 39`` — the index reaches 2, i.e. the offset is no longer the sample
    MINIMUM of the calibration residuals.

Below 39 a bin's own tau = 0.05 offset IS its most extreme residual: an estimate
with no data behind it that moves with a single outlier. Such a bin therefore
takes the MARGINAL delta, and ``fit_mondrian`` records that it did. The fallback
is a stated rule, not an accident of small numbers.

Both quantities are reported per bin beside every coverage figure — ``n``, the
DISTINCT PATIENT count, and the MEAN BAND WIDTH. Coverage alone is not
interpretable: it is bought with width, and n windows drawn from two patients is
not n independent observations.

TWO PROTOCOLS, ONE OF WHICH SHIPS. The masked-BG objective supervises forecast,
backcast and infill, and their residuals are not exchangeable — most infill
supervision sits at small d, where the answer is bracketed by visible evidence on
both sides. Conditioning on (region, d, sidedness) would multiply the bins by
six against a floor already needing n >= 19, so the split is by REMIT instead:

  * the FORECAST protocol is calibrated as specified above and is what ships;
  * INFILL gets its own coarse fit through :func:`fit_infill_conformal`, which
    announces every fallback on stdout, and whose meta carries
    ``shipped = False``. Its easier residuals never enter the shipped band.

``conformal.py`` is untouched by all of this. ``fit_quantile_conformal`` is
called once per bin on that bin's own rows and the results are stacked;
``apply_quantile_conformal`` is called once per bin GROUP with that bin's
``(S, K)`` slice, never with a gathered per-window delta (which its ``ndim == 2``
assert refuses, and that assert is load-bearing — it is what stops a 1-D ``(K,)``
delta from broadcasting silently across every horizon step).
"""
from __future__ import annotations

import numpy as np

import conformal
from config import PATCH_SIZE

# --------------------------------------------------------------------------- #
# The region axis
# --------------------------------------------------------------------------- #
REGION_EDGES: tuple[float, ...] = (110.0,)
N_REGION_BINS: int = len(REGION_EDGES) + 1

# Order-statistic floors of ``conformal._conformal_offset`` at tau = 0.05 (see
# the module docstring). MIN_N_OWN_FIT is the rule the fallback is stated on.
MIN_N_OFFSET_EXISTS: int = 19
MIN_N_OWN_FIT: int = 39

# The destination is read over the final patch of the horizon rather than off the
# single last step: one 5-minute grid point of a median line is a noisier
# statement of "where this is heading" than the 30 minutes it closes on.
DESTINATION_STEPS: int = PATCH_SIZE


def bin_label(b: int) -> str:
    """Human-readable interval of region bin ``b`` in mg/dL."""
    lo = '-inf' if b == 0 else f"{REGION_EDGES[b - 1]:.0f}"
    hi = 'inf' if b == len(REGION_EDGES) else f"{REGION_EDGES[b]:.0f}"
    return f"[{lo}, {hi})"


def forecast_destination(q: np.ndarray, median_idx: int) -> np.ndarray:
    """Where each window's forecast is HEADING, in mg/dL — the region variable.

    Args:
        q: ``(N, S, K)`` mg/dL quantile fan, ascending in K.
        median_idx: column of the tau=0.5 median.

    Returns:
        ``(N,)`` mean of the median line over the last ``DESTINATION_STEPS`` steps.
        Invariant under conformal correction, which holds the median fixed.
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


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def fit_mondrian(cal_q: np.ndarray, cal_true: np.ndarray, cal_bin: np.ndarray,
                 levels: tuple[float, ...], median_idx: int,
                 patients=None, protocol: str = 'forecast',
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit one conformal correction per region bin, plus the marginal baseline.

    ``conformal.fit_quantile_conformal`` has no bin argument, so it is called
    ONCE PER BIN on that bin's calibration rows and the results are stacked. The
    marginal fit over ALL calibration rows is returned alongside — it is both the
    fallback for a thin bin and the pre-Mondrian baseline, measured here so the
    two arms are always compared within one run.

    Args:
        cal_q:    ``(N, S, K)`` calibration fans (mg/dL), ascending in K.
        cal_true: ``(N, S)`` calibration truth (mg/dL).
        cal_bin:  ``(N,)`` region-bin index per calibration window.
        levels:   the K quantile levels (ascending).
        median_idx: index of the tau=0.5 median (held fixed).
        patients: optional per-window patient id sequence, for the distinct count.
        protocol: ``'forecast'`` (ships) or ``'infill'`` (never ships).
        verbose:  print one line per bin, including every marginal fallback.

    Returns:
        ``(delta, marginal, meta)`` — ``delta`` ``(N_REGION_BINS, S, K)``,
        ``marginal`` ``(S, K)``, and a meta dict whose ``bins`` list carries per
        bin: ``n``, ``n_patients``, ``own_fit`` and ``fallback_reason``.
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
            # STATED RULE, not an accident: below MIN_N_OWN_FIT this bin's own
            # tau=0.05 offset would be the sample minimum of its residuals, so it
            # takes the marginal delta and says so.
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

    Same region axis as the forecast fit and nothing finer: infill residuals are
    additionally conditioned on ``d`` and on sidedness, and crossing those with
    the region would multiply the bins by six against the n >= 19 floor. Infill is
    therefore calibrated coarsely, REPORTED per ``d``, and kept in its own slot.

    Every marginal fallback is announced on stdout, because a fallback here is
    the expected case rather than the exception, and a silent one would read as a
    fitted correction.
    """
    delta, marginal, meta = fit_mondrian(
        cal_q, cal_true, cal_bin, levels, median_idx,
        patients=patients, protocol='infill', verbose=True)
    assert meta['shipped'] is False
    return delta, marginal, meta


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #
def apply_mondrian(q: np.ndarray, delta: np.ndarray, bin_idx: np.ndarray,
                   median_idx: int) -> np.ndarray:
    """Apply a per-bin ``delta`` stack to a fan, one ``conformal.apply`` call per bin.

    The windows are GROUPED by bin and each group is passed that bin's ``(S, K)``
    slice, then written back into place. A gathered per-window ``(N, S, K)`` delta
    is never formed: ``conformal.apply_quantile_conformal`` asserts its delta is
    2-D, and that assert stops a 1-D ``(K,)`` delta broadcasting across every
    horizon step.

    Args:
        q:       ``(N, S, K)`` mg/dL fan, ascending in K.
        delta:   ``(N_REGION_BINS, S, K)`` from :func:`fit_mondrian`.
        bin_idx: ``(N,)`` region bin of each window.
        median_idx: index of the tau=0.5 median.

    Returns:
        ``(N, S, K)`` calibrated fan; the median column is untouched.
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


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def mean_band_width(q: np.ndarray, lo_idx: int, hi_idx: int) -> float:
    """Mean ``q_hi - q_lo`` (mg/dL) over windows and horizon steps."""
    if q.shape[0] == 0:
        return float('nan')
    return float(np.mean(q[:, :, hi_idx] - q[:, :, lo_idx]))


def forecast_d_step_groups(n_pred_patches: int, patch_size: int) -> dict:
    """``{'d1': steps, 'd2': steps, ...}`` for the right-edge FORECAST protocol.

    The span has no right neighbour, so patch ``p`` of it sits at one-sided
    ``d = p + 1`` and every step of that patch inherits it.
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

    Args:
        arms: ``{arm_name: (N, S, K) fan}`` — e.g. raw / marginal / mondrian, all
            scored on the same windows so the comparison is within one run.
        true: ``(N, S)`` truth (mg/dL).
        bin_idx: ``(N,)`` region bin per window.
        lo_idx, hi_idx: the band edges coverage and width are read on.
        patients: optional per-window patient ids.
        step_groups: ``{label: step indices}`` — the ``d`` groups of the protocol
            (:func:`forecast_d_step_groups` / :func:`d_step_groups`). Given, each
            bin also reports per ``d``, which is the only axis a masked-BG metric
            may be binned on; a figure pooled over ``d`` mixes regimes of very
            different difficulty and is not a selection metric.

    Returns:
        one dict per bin: ``bin``, ``label``, ``n``, ``n_patients``, per-arm
        ``cov``/``width`` over the whole horizon, and (with ``step_groups``)
        ``by_d`` carrying the same pair per ``d`` group.
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

    A ``by_d`` block, when present, is printed under its bin: ``d`` is the axis
    the masked-BG metrics are read on, and the whole-horizon row above it is a
    summary, never the figure to select on.
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
