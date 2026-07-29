"""
Actual-vs-predicted BG comparison figures for the real-data report.

Pure plotting (matplotlib); each function takes numpy arrays and writes a PNG.
The driver (``realdata.report.build_figures``) supplies the model's risk-space
median BG forecast and the true CGM per test window.

Figures, per dataset:
  * trajectory_grid  — example windows: real CGM (context+future) vs prediction
  * parity_scatter   — predicted vs true at 30/60/120 min, with the identity line
  * clarke_grid      — Clarke Error Grid (zones drawn) at 30/60/120 min

The former per-channel ``channel_anatomy_grid`` / ``day_channel_trajectory``
panels were dropped: the risk-space redesign removed the model's carb / insulin /
IS / HGO dynamics outputs, so there are no per-channel μ ±σ trajectories to plot.
"""
from __future__ import annotations

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _OK = True
except Exception:
    _OK = False

from config import BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD
from .horizons import FIGURE_HORIZONS, FIGURE_HORIZON_IDX
from .schema import GRID_MIN  # canonical CGM grid resolution (minutes per step)
from clock_face import draw_clock_axis


def _hlabel(h_min: int) -> str:
    """Compact horizon label: '30 min', '1 h', '2 h', … (whole hours as hours)."""
    return f'{h_min} min' if h_min % 60 else f'{h_min // 60} h'


def _hhmm(hour: float) -> str:
    """Format a float hour-of-day in ``[0, 24)`` as a 'HH:MM' wall-clock string."""
    h = int(hour) % 24
    m = int(round((hour - int(hour)) * 60))
    if m == 60:
        m = 0
        h = (h + 1) % 24
    return f'{h:02d}:{m:02d}'


def _annotate_tod(ax, ex: dict) -> None:
    """Overlay the time-of-day probe clock readout on a trajectory panel.

    Guarded no-op unless the example carries a finite ``pred_hour`` — the probe
    being disabled (``TIME_PROBE_ENABLED`` off) leaves the key absent or NaN.
    Renders 'pred HH:MM (R…) / true HH:MM', tinted grey→blue by the confidence
    ``tod_R`` (the resultant length R of the per-bin softmax belief) so a low-``R``
    origin reads as tentative.
    """
    ph = ex.get('pred_hour')
    if ph is None or not np.isfinite(ph):
        return
    th = ex.get('true_hour')
    R = ex.get('tod_R')
    have_R = R is not None and np.isfinite(R)
    parts = [f'pred {_hhmm(float(ph))}']
    if have_R:
        parts.append(f'(R{float(R):.1f})')
    if th is not None and np.isfinite(th):
        parts.append(f'/ true {_hhmm(float(th))}')
    r = float(np.clip(R, 0.0, 2.0)) / 2.0 if have_R else 0.0
    col = tuple((1.0 - r) * np.array([0.55, 0.55, 0.55]) + r * np.array([0.12, 0.30, 0.90]))
    ax.text(0.02, 0.97, ' '.join(parts), transform=ax.transAxes, fontsize=6.5,
            va='top', ha='left', color=col, fontweight='bold')


def _covered_horizons(n_steps: int):
    """The (horizons, step-indices) from FIGURE_HORIZONS that an ``n_steps``-long
    forecast array actually reaches; lets the same plotters take a
    ``PREDICTION_HORIZON_HOURS`` single-pass array or a ``NIGHT_LONG_HORIZON_HOURS``
    rolled one without reading past the end."""
    hs = [h for h in FIGURE_HORIZONS if FIGURE_HORIZON_IDX[h] < n_steps]
    return hs, [FIGURE_HORIZON_IDX[h] for h in hs]


def _draw_clarke(ax):
    """Draw canonical Clarke Error Grid zone boundaries (mg/dL, 0..400)."""
    ax.plot([0, 400], [0, 400], 'k:', lw=0.8)
    ax.plot([0, 175 / 3], [70, 70], 'k-', lw=0.6)
    ax.plot([175 / 3, 400 / 1.2], [70, 400], 'k-', lw=0.6)
    ax.plot([70, 70], [84, 400], 'k-', lw=0.6)
    ax.plot([0, 70], [180, 180], 'k-', lw=0.6)
    ax.plot([70, 290], [180, 400], 'k-', lw=0.6)
    ax.plot([70, 70], [0, 56], 'k-', lw=0.6)
    ax.plot([70, 400], [56, 320], 'k-', lw=0.6)
    ax.plot([180, 180], [0, 70], 'k-', lw=0.6)
    ax.plot([180, 400], [70, 70], 'k-', lw=0.6)
    ax.plot([240, 240], [70, 180], 'k-', lw=0.6)
    ax.plot([240, 400], [180, 180], 'k-', lw=0.6)
    ax.plot([130, 180], [0, 70], 'k-', lw=0.6)
    for x, y, t in [(30, 15, 'A'), (370, 260, 'B'), (280, 370, 'B'), (160, 370, 'C'),
                    (160, 15, 'C'), (30, 140, 'D'), (370, 120, 'D'), (30, 370, 'E'),
                    (370, 15, 'E')]:
        ax.text(x, y, t, fontsize=9, fontweight='bold', color='gray')
    ax.set_xlim(0, 400); ax.set_ylim(0, 400); ax.set_aspect('equal')


def _clarke_AB(pred, true):
    pb = np.clip(pred, 1, None); tb = np.clip(true, 1, None)
    rel = np.abs(pb - tb) / tb
    A = (rel <= 0.20) | ((pb <= 70) & (tb <= 70))
    E = ((pb <= 70) & (tb >= 180)) | ((pb >= 180) & (tb <= 70))
    c_up = (tb >= 70) & (tb <= 290) & (pb >= tb + 110)
    c_lo = (tb >= 130) & (tb <= 180) & (pb <= (7.0 / 5.0) * tb - 182.0)
    C = (~A) & (~E) & (c_up | c_lo)
    D = (~A) & (~E) & (~C) & ((tb <= 70) | (tb >= 240)) & (pb >= 70) & (pb <= 180)
    B = (~A) & (~E) & (~C) & (~D)
    return float(100 * A.mean()), float(100 * (A | B).mean())


def trajectory_grid(examples: list, path: str, title: str, ncols: int = 3):
    """examples: list of dicts {ctx_tail, true_future, pred_future, label}.

    Optional per-example keys are rendered when present: ``band_lo``/``band_hi``
    (calibrated 90% band edges, each ``(len(pred_future),)`` mg/dL); the
    time-of-day probe readout ``pred_hour``/``true_hour`` (hour-of-day in
    ``[0, 24)``) plus ``tod_R`` (probe confidence ``R`` — the resultant length of the per-bin softmax belief); and
    ``time_probs`` (``(P, TIME_PROBE_N_BINS)`` per-patch softmax belief) rendered as
    a top-strip row of ``P`` native clock-face histograms (no rotation — one clock
    per prediction patch). The TOD keys are absent when the probe is disabled
    (``TIME_PROBE_ENABLED`` off), so every render stays guarded and a probe-off run
    draws the exact band/forecast panel unchanged."""
    if not _OK or not examples:
        return
    n = len(examples); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), squeeze=False)
    for i, ex in enumerate(examples):
        ax = axes[i // ncols][i % ncols]
        ct = ex['ctx_tail']; tf = ex['true_future']; pf = ex['pred_future']
        t_ctx = np.arange(-len(ct), 0) * GRID_MIN
        t_fut = (np.arange(len(tf)) + 1) * GRID_MIN
        ax.plot(t_ctx, ct, color='0.4', lw=1.2)
        # Optional calibrated 90% band ribbon (sim figure path). Absent ex.band_lo
        # ⇒ the exact band-less figure. Each edge is (len(pred_future),) mg/dL.
        if ex.get('band_lo') is not None and ex.get('band_hi') is not None:
            bl = np.asarray(ex['band_lo']); bh = np.asarray(ex['band_hi'])
            ax.fill_between(np.r_[0, t_fut[:len(bl)]], np.r_[ct[-1], bl], np.r_[ct[-1], bh],
                            color='C3', alpha=0.15, label='90% band')
        ax.plot(np.r_[0, t_fut], np.r_[ct[-1], tf], color='C0', lw=1.6, label='true CGM')
        ax.plot(np.r_[0, t_fut], np.r_[ct[-1], pf], color='C3', lw=1.6, ls='--', label='predicted')
        ax.axvline(0, color='0.8', lw=0.8)
        ax.axhline(BG_HYPO_THRESHOLD, color='0.85', lw=0.6)
        ax.axhline(BG_HYPER_THRESHOLD, color='0.85', lw=0.6)
        ax.set_title(ex.get('label', ''), fontsize=8)
        _annotate_tod(ax, ex)
        # Per-patch time-of-day clock strip: one native clock per prediction patch
        # (no rotation) across the panel's top band. Absent ``time_probs`` (probe
        # off) => no strip, exact band/forecast panel unchanged.
        tp = ex.get('time_probs')
        if tp is not None:
            tp = np.asarray(tp)
            P = tp.shape[0]; gap = 0.01; w = (1.0 - (P + 1) * gap) / P
            for p in range(P):
                cax = ax.inset_axes([gap + p * (w + gap), 0.76, w, 0.22])
                draw_clock_axis(cax, tp[p])
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc='best')
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle(title, fontsize=11)
    fig.supxlabel('minutes from prediction start', fontsize=9)
    fig.supylabel('BG (mg/dL)', fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=360); plt.close(fig)


def parity_scatter(pred: np.ndarray, true: np.ndarray, path: str, title: str, ncols: int = 5):
    """pred/true: (N, S). Scatter pred vs true at each FIGURE horizon the array
    reaches, one panel per hour-by-hour horizon (wrapped into a grid). True may
    carry trailing NaN where a rolled window ran past its segment; those points
    are dropped per horizon."""
    if not _OK or len(pred) == 0:
        return
    hs, idx = _covered_horizons(pred.shape[1])
    if not hs:
        return
    ncols = min(ncols, len(hs)); nrows = (len(hs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.9 * nrows), squeeze=False)
    for j, (h, k) in enumerate(zip(hs, idx)):
        ax = axes[j // ncols][j % ncols]
        t, p = true[:, k], pred[:, k]
        m = np.isfinite(t) & np.isfinite(p); t, p = t[m], p[m]
        if len(t) == 0:
            ax.axis('off'); continue
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        ax.scatter(t, p, s=8, alpha=0.35, color='C0', edgecolors='none')
        lim = [40, max(400, float(np.nanmax([t.max(), p.max()])) + 10)]
        ax.plot(lim, lim, 'k-', lw=0.8); ax.plot(lim, [1.2 * x for x in lim], 'k:', lw=0.6)
        ax.plot(lim, [0.8 * x for x in lim], 'k:', lw=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_title(f'{_hlabel(h)}  (RMSE {rmse:.1f}, n={len(t)})', fontsize=9)
        if j // ncols == nrows - 1:
            ax.set_xlabel('true CGM (mg/dL)', fontsize=8)
        if j % ncols == 0:
            ax.set_ylabel('predicted BG (mg/dL)', fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(hs), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=360); plt.close(fig)


def clarke_grid(pred: np.ndarray, true: np.ndarray, path: str, title: str, ncols: int = 5):
    """Clarke Error Grid at each FIGURE horizon the array reaches, one panel per
    hour-by-hour horizon (wrapped into a grid). Trailing-NaN true (rolled past the
    segment) is dropped per horizon."""
    if not _OK or len(pred) == 0:
        return
    hs, idx = _covered_horizons(pred.shape[1])
    if not hs:
        return
    ncols = min(ncols, len(hs)); nrows = (len(hs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.9 * nrows), squeeze=False)
    for j, (h, k) in enumerate(zip(hs, idx)):
        ax = axes[j // ncols][j % ncols]
        t, p = true[:, k], pred[:, k]
        m = np.isfinite(t) & np.isfinite(p); t, p = t[m], p[m]
        _draw_clarke(ax)
        if len(t) == 0:
            ax.set_title(f'{_hlabel(h)}  (n=0)', fontsize=9); continue
        ax.scatter(t, p, s=8, alpha=0.4, color='C0', edgecolors='none', zorder=3)
        a, ab = _clarke_AB(p, t)
        ax.set_title(f'{_hlabel(h)}  (A {a:.0f}%, A+B {ab:.0f}%)', fontsize=9)
        if j // ncols == nrows - 1:
            ax.set_xlabel('reference CGM (mg/dL)', fontsize=8)
        if j % ncols == 0:
            ax.set_ylabel('predicted BG (mg/dL)', fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(hs), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=360); plt.close(fig)
