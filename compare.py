#!/usr/bin/env python3
"""
Cross-model comparison over the trained checkpoints under ``models/``.

Compares the three capacities (nano, small, medium) on every horizon and probe the
per-model metric directories carry, and writes high-resolution figures and
machine-readable JSON. No markdown is produced.

Reporting basis
---------------
``metrics/core/suite.py`` scores each horizon twice and the two are NOT interchangeable:

``median_line``   the genuine point forecast ``f_inv(median)``. The basis published
                  numbers use, and the basis this script reports as headline accuracy
                  (RMSE, MAE, MARD, Clarke, skill).
``band-scored``   the top-level block, computed on ``pred_eff = clip(true, q_lo, q_hi)``
                  — zero error wherever the truth falls inside the 50% band. Useful as
                  a band-geometry diagnostic, not comparable to a point forecast.

Every emitted record carries a ``basis`` field, and every accessor here is named for
the basis it reads, so the two can never be confused at a call site. Two families have
no median-line counterpart and are reported band-scored, labelled as such: CG-EGA
(scored on ``pred_eff``) and hypo/hyper detection (keyed off the τ alarm band edges).

Colour discipline
-----------------
Palette and chrome come from the suite's single copy at ``metrics/figstyle.py``; this
script adds no second copy. Per that module's rule a mark encodes exactly one job, so
each figure carries exactly one hue job: capacity rides one ordinal blue ladder, and
the evaluation source keeps its fixed categorical hue.

Usage
-----
    python compare.py                 # writes comparison/{figures,data}
    python compare.py --out somewhere
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
# The trained-checkpoint tree, one directory per capacity. The GUI discovers its
# checkpoint from the same root — a second root is how the two start disagreeing
# about which checkpoint "medium" names.
MODEL_ROOT = os.path.join(ROOT, 'models')

sys.path.insert(0, os.path.join(ROOT, 'metrics'))
import figstyle as fs                                     # type: ignore[import]  # noqa: E402

import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402

# ---------------------------------------------------------------------------- #
# The grid.
# ---------------------------------------------------------------------------- #
SIZES = ('nano', 'small', 'medium')
# One training variant and one evaluation source. Both axes are kept as tuples
# rather than inlined: every figure below iterates them, and a second source
# would otherwise mean re-threading a loop through forty call sites.
VARIANTS = ('sim',)
COHORTS = ('sim',)
HORIZONS = ('30', '60', '120')
SWEEP_HORIZONS = ('30', '60', '120', '180', '240', '300', '360', '420', '480')

VARIANT_LABEL = {'sim': 'sim — trained on T1DMSIM'}
VARIANT_SHORT = {'sim': 'sim'}
SIZE_LABEL = {s: s for s in SIZES}

# Capacity is an ordered magnitude, so it rides one hue light-to-dark.
SIZE_COLOR = dict(zip(SIZES, fs.DOSE_CARB[1:1 + len(SIZES)]))
# Variant is categorical. Used only where the source is not also a hue.
VARIANT_COLOR = dict(zip(VARIANTS, fs.SERIES))
# The simulator source is the reference, per figstyle.
COHORT_COLOR = dict(fs.COHORT, sim=fs.REFERENCE)
COHORT_LABEL = dict(fs.COHORT_LABEL, sim='T1DMSIM')



# ---------------------------------------------------------------------------- #
# Small helpers.
# ---------------------------------------------------------------------------- #
def _f(v: Any) -> float:
    """A metric as float, with a missing or unrepresentable value as NaN."""
    if v is None:
        return math.nan
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def _jsonable(v: Any) -> Any:
    """NaN-free JSON: a non-finite float is a null, not a bare NaN token."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def median_line(row: dict, *path: str) -> float:
    """A value on the MEDIAN-LINE basis — the genuine point forecast."""
    d: Any = (row or {}).get('median_line') or {}
    for p in path:
        if not isinstance(d, dict):
            return math.nan
        d = d.get(p)
    return _f(d)


def band_scored(row: dict, *path: str) -> float:
    """A value on the BAND-SCORED basis — clip(true, q_lo, q_hi). Not a point forecast."""
    d: Any = row or {}
    for p in path:
        if not isinstance(d, dict):
            return math.nan
        d = d.get(p)
    return _f(d)


def read_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def read_csv(path: str) -> dict[str, np.ndarray]:
    """A CSV of numbers as column arrays; an empty or non-numeric cell becomes NaN."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return {}
    head, body = rows[0], rows[1:]
    cols: dict[str, np.ndarray] = {}
    for j, name in enumerate(head):
        vals = []
        for r in body:
            cell = r[j] if j < len(r) else ''
            try:
                vals.append(float(cell))
            except ValueError:
                vals.append(math.nan)
        cols[name] = np.asarray(vals, dtype=float)
    return cols


def pooled_rmse(vals: list[float], weights: list[float]) -> float:
    """Window-weighted RMSE pooling — quadratic, not a mean of RMSEs."""
    v = np.asarray(vals, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return math.nan
    return float(np.sqrt(np.sum(w[ok] * v[ok] ** 2) / np.sum(w[ok])))


def pooled_mean(vals: list[float], weights: list[float]) -> float:
    v = np.asarray(vals, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return math.nan
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok]))


def loglog_fit(x: list[float], y: list[float]) -> dict[str, float]:
    """Power-law fit y ~ a * x**b, returned with the fraction of variance explained."""
    xa = np.asarray(x, float)
    ya = np.asarray(y, float)
    ok = np.isfinite(xa) & np.isfinite(ya) & (xa > 0) & (ya > 0)
    if ok.sum() < 3:
        return {'exponent': math.nan, 'coefficient': math.nan, 'r2': math.nan, 'n': int(ok.sum())}
    lx, ly = np.log(xa[ok]), np.log(ya[ok])
    b, a = np.polyfit(lx, ly, 1)
    resid = ly - (a + b * lx)
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else math.nan
    return {'exponent': float(b), 'coefficient': float(np.exp(a)), 'r2': r2, 'n': int(ok.sum())}


# ---------------------------------------------------------------------------- #
# Loading.
# ---------------------------------------------------------------------------- #
def load_params() -> dict[str, dict[str, Any]]:
    """Parameter count and architecture provenance, read from the checkpoints."""
    import torch
    out: dict[str, dict[str, Any]] = {}
    for size in SIZES:
        rec: dict[str, Any] = {'n_params': None}
        path = os.path.join(MODEL_ROOT, size, 'checkpoints', 't1dmai_best.pt')
        if os.path.exists(path):
            ck = torch.load(path, map_location='cpu', weights_only=False)
            sd = ck.get('model_state_dict') or {}
            rec['n_params'] = int(sum(int(v.numel()) for v in sd.values()))
            rec['arch_version'] = ck.get('arch_version')
            rec['loss_schema'] = ck.get('loss_schema')
            del ck
        out[size] = rec
    return out


def load_all() -> dict[str, Any]:
    """Every input this comparison reads, indexed by size and variant."""
    M: dict[str, Any] = {'size': {}, 'cell': {}}
    for size in SIZES:
        base = os.path.join(MODEL_ROOT, size)
        M['size'][size] = {
            'summary': read_json(os.path.join(base, 'figures', 'summary.json')),
            'training_summary': read_json(os.path.join(base, 'logs', 'training_summary.json')),
            'resolved_config': read_json(os.path.join(base, 'logs', 'resolved_config.json')),
            'training_log': read_csv(os.path.join(base, 'logs', 'training_log.csv')),
            'validation_log': read_csv(os.path.join(base, 'logs', 'validation_log.csv')),
        }
        for variant in VARIANTS:
            mdir = os.path.join(base, f'metrics_{variant}')
            cell = {
                'sim': read_json(os.path.join(mdir, 'sim', 'stats.json')),
                'whatif': read_json(os.path.join(mdir, 'whatif.json')),
            }
            M['cell'][(size, variant)] = cell
    M['params'] = load_params()
    return M


def cohort_stats(M: dict, size: str, variant: str, cohort: str) -> dict:
    """The per-source stats block from that cell's evaluation report."""
    cell = M['cell'][(size, variant)]
    return (cell.get(cohort) or {}).get(cohort) or {}


def horizon_row(M: dict, size: str, variant: str, cohort: str, h: str) -> dict:
    return ((cohort_stats(M, size, variant, cohort).get('metrics') or {}).get(h)) or {}


def n_params(M: dict, size: str) -> float:
    return _f((M['params'].get(size) or {}).get('n_params'))


# ---------------------------------------------------------------------------- #
# Figure chrome.
# ---------------------------------------------------------------------------- #
FIGDIR = ''
DATADIR = ''
WRITTEN: list[str] = []


def save(fig, name: str) -> str:
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    plt.close(fig)
    WRITTEN.append(os.path.relpath(path, ROOT))
    print(f'  figure  {os.path.relpath(path, ROOT)}')
    return path


def dump(obj: Any, name: str) -> str:
    os.makedirs(DATADIR, exist_ok=True)
    path = os.path.join(DATADIR, name)
    with open(path, 'w') as fh:
        json.dump(_jsonable(obj), fh, indent=1, sort_keys=False)
        fh.write('\n')
    WRITTEN.append(os.path.relpath(path, ROOT))
    print(f'  data    {os.path.relpath(path, ROOT)}')
    return path


# Header geometry is measured in INCHES, not figure fractions: the fonts are in
# points, so a fractional offset that clears the title on a tall figure collides
# with it on a short one.
HEADER_IN = 0.86          # reserved band above the panels
TITLE_IN = 0.17           # title baseline, from the top edge
SUBTITLE_IN = 0.42        # subtitle baseline, from the top edge
LEGEND_IN = 0.13          # legend top, from the top edge


# Below this width the title and the legend cannot share a line, so the legend
# drops under the subtitle and the reserved band grows to hold it. The ladder
# collapsed to a single evaluation source, which is what made these figures narrow.
NARROW_IN = 11.0
MIN_FIG_IN = 8.0          # floor on figure width, so a title has room to sit


def _is_narrow(fig) -> bool:
    return fig.get_figwidth() < NARROW_IN


def _frac_from_top(fig, inches: float) -> float:
    return 1.0 - inches / max(fig.get_figheight(), 1e-6)


def suptitle(fig, title: str, subtitle: str = '') -> None:
    fig.suptitle(title, x=0.006, ha='left', va='top', fontsize=13, color=fs.INK,
                 y=_frac_from_top(fig, TITLE_IN))
    if subtitle:
        fig.text(0.006, _frac_from_top(fig, SUBTITLE_IN), subtitle, ha='left', va='top',
                 fontsize=8.5, color=fs.INK2)


def _fig_legend(fig, handles, ncol: int, fontsize: float = 8.5):
    if _is_narrow(fig):
        return fig.legend(handles=handles, loc='upper left',
                          bbox_to_anchor=(0.006, _frac_from_top(fig, SUBTITLE_IN + 0.24)),
                          ncol=ncol, frameon=False, fontsize=fontsize, labelcolor=fs.INK2)
    return fig.legend(handles=handles, loc='upper right',
                      bbox_to_anchor=(0.997, _frac_from_top(fig, LEGEND_IN)),
                      ncol=ncol, frameon=False, fontsize=fontsize, labelcolor=fs.INK2)


def size_legend(fig) -> None:
    _fig_legend(fig, [Line2D([], [], color=SIZE_COLOR[s], lw=2.4, label=s) for s in SIZES],
                len(SIZES))


def variant_legend(fig) -> None:
    _fig_legend(fig, [Line2D([], [], color=VARIANT_COLOR[v], lw=2.4, label=VARIANT_LABEL[v])
                      for v in VARIANTS], len(VARIANTS))


def cohort_legend(fig, cohorts=COHORTS) -> None:
    _fig_legend(fig, [Patch(facecolor=COHORT_COLOR[c], edgecolor=fs.SURFACE,
                            label=COHORT_LABEL[c]) for c in cohorts], len(cohorts))


def size_axis(ax) -> None:
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels(SIZES)
    ax.set_xlim(-0.35, len(SIZES) - 0.65)


def grid(nrows: int, ncols: int, w: float = 3.6, h: float = 2.7):
    fig_w = max(ncols * w, MIN_FIG_IN)
    header = HEADER_IN + (0.34 if fig_w < NARROW_IN else 0.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, nrows * h + header),
                             squeeze=False)
    fig.subplots_adjust(top=_frac_from_top(fig, header))
    return fig, axes


def blank(ax) -> None:
    ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes,
            fontsize=8.5, color=fs.MUTED)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


# ---------------------------------------------------------------------------- #
# A. Capacity, cost and training dynamics.
# ---------------------------------------------------------------------------- #
def fig_capacity(M: dict) -> None:
    fig, axes = grid(1, 4, w=3.5, h=3.0)
    ax = axes[0][0]
    vals = [n_params(M, s) for s in SIZES]
    bars = ax.bar(range(len(SIZES)), vals, color=[SIZE_COLOR[s] for s in SIZES], width=0.62)
    fs.bar_labels(ax, bars, fmt='{:,.0f}')
    ax.set_yscale('log'); ax.set_title('parameters'); ax.set_ylabel('count')
    size_axis(ax); fs.ygrid(ax)

    ax = axes[0][1]
    hours = [_f((M['size'][s]['training_summary'].get('progress') or {}).get('elapsed_hours'))
             for s in SIZES]
    bars = ax.bar(range(len(SIZES)), hours, color=[SIZE_COLOR[s] for s in SIZES], width=0.62)
    fs.bar_labels(ax, bars, fmt='{:.2f}')
    ax.set_title('pretrain wall-clock'); ax.set_ylabel('hours')
    size_axis(ax); fs.ygrid(ax)

    ax = axes[0][2]
    sps = [_f((M['size'][s]['training_summary'].get('progress') or {}).get('steps_per_second'))
           for s in SIZES]
    bars = ax.bar(range(len(SIZES)), sps, color=[SIZE_COLOR[s] for s in SIZES], width=0.62)
    fs.bar_labels(ax, bars, fmt='{:.2f}')
    ax.set_title('throughput'); ax.set_ylabel('steps / second')
    size_axis(ax); fs.ygrid(ax)

    ax = axes[0][3]
    mem = [_f((M['size'][s]['training_summary'].get('hardware') or {}).get('gpu_peak_memory_mb'))
           for s in SIZES]
    bars = ax.bar(range(len(SIZES)), mem, color=[SIZE_COLOR[s] for s in SIZES], width=0.62)
    fs.bar_labels(ax, bars, fmt='{:.0f}')
    ax.set_title('peak GPU memory'); ax.set_ylabel('MB')
    size_axis(ax); fs.ygrid(ax)

    labels = ' · '.join(f'{s}: {M["size"][s]["summary"].get("arch_label", "?")}' for s in SIZES)
    suptitle(fig, 'Capacity ladder and pretraining cost', labels)
    save(fig, 'fig01_capacity_and_cost.png')


def fig_pretrain(M: dict) -> None:
    fig, axes = grid(2, 3, w=3.9, h=2.9)
    panels = [
        ('training_log', 'step', 'loss_ema', 'training loss (EMA)', 'loss', True),
        ('validation_log', 'step', 'val_loss_total', 'validation loss', 'loss', True),
        ('validation_log', 'step', 'overfit_ratio', 'overfit ratio', 'train / val', False),
        ('validation_log', 'step', 'bg_rmse_60', 'validation BG RMSE @60m', 'mg/dL', False),
        ('training_log', 'step', 'grad_norm', 'gradient norm', 'norm', True),
        ('training_log', 'step', 'step_time_seconds', 'step time', 'seconds', True),
    ]
    for k, (src, xk, yk, title, ylab, logy) in enumerate(panels):
        ax = axes[k // 3][k % 3]
        drew = False
        for s in SIZES:
            cols = M['size'][s][src]
            if xk not in cols or yk not in cols:
                continue
            x, y = cols[xk], cols[yk]
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            ax.plot(x[ok], y[ok], color=SIZE_COLOR[s], lw=1.6, label=s)
            drew = True
        if not drew:
            blank(ax); continue
        if logy:
            ax.set_yscale('log')
        ax.set_title(title); ax.set_xlabel('step'); ax.set_ylabel(ylab)
    size_legend(fig)
    suptitle(fig, 'Training dynamics on the simulator corpus')
    save(fig, 'fig02_pretrain_dynamics.png')


# ---------------------------------------------------------------------------- #
# B. Accuracy, on the median line.
# ---------------------------------------------------------------------------- #
def _accuracy_panel(M: dict, ax, cohort: str, h: str, key: str) -> bool:
    drew = False
    for v in VARIANTS:
        y = [median_line(horizon_row(M, s, v, cohort, h), key) for s in SIZES]
        if not np.isfinite(y).any():
            continue
        ax.plot(range(len(SIZES)), y, color=VARIANT_COLOR[v], lw=2.0, marker='o',
                markersize=5, markeredgecolor=fs.SURFACE, markeredgewidth=1.5)
        drew = True
    size_axis(ax)
    fs.ygrid(ax)
    return drew


def fig_accuracy(M: dict, key: str, unit: str, title: str, fname: str) -> None:
    fig, axes = grid(len(HORIZONS), len(COHORTS), w=3.3, h=2.5)
    for i, h in enumerate(HORIZONS):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            if not _accuracy_panel(M, ax, c, h, key):
                blank(ax)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(f'{h} min\n{unit}')
    variant_legend(fig)
    suptitle(fig, title, 'Median-line basis — the genuine point forecast f_inv(median). '
                         'One row per forecast horizon.')
    save(fig, fname)


def fig_skill(M: dict) -> None:
    fig, axes = grid(len(HORIZONS), len(COHORTS), w=3.3, h=2.5)
    for i, h in enumerate(HORIZONS):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            if not _accuracy_panel(M, ax, c, h, 'skill_point'):
                blank(ax)
            ax.axhline(0.0, color=fs.AXIS, lw=1.0, zorder=1)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(f'{h} min\nskill vs persistence')
    variant_legend(fig)
    suptitle(fig, 'Skill against a last-value persistence baseline',
             'Median-line basis. Zero is persistence; above zero beats holding the last CGM '
             'reading flat.')
    save(fig, 'fig05_skill_vs_persistence.png')


def fig_rmse_vs_horizon(M: dict) -> None:
    fig, axes = grid(len(VARIANTS), len(COHORTS), w=3.3, h=2.5)
    xs = [int(h) for h in SWEEP_HORIZONS]
    for i, v in enumerate(VARIANTS):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            drew = False
            for s in SIZES:
                by_hour = cohort_stats(M, s, v, c).get('rmse_by_hour') or {}
                y = [_f((by_hour.get(h) or {}).get('rmse_point_median')) for h in SWEEP_HORIZONS]
                if not np.isfinite(y).any():
                    continue
                ax.plot(xs, y, color=SIZE_COLOR[s], lw=1.9, marker='o', markersize=3.4,
                        markeredgecolor=fs.SURFACE, markeredgewidth=1.0)
                drew = True
            by_hour = cohort_stats(M, SIZES[-1], v, c).get('rmse_by_hour') or {}
            p = [_f((by_hour.get(h) or {}).get('rmse_persist_point')) for h in SWEEP_HORIZONS]
            if np.isfinite(p).any():
                ax.plot(xs, p, color=fs.MUTED, lw=1.3, ls=(0, (4, 3)), zorder=1)
                drew = True
            if not drew:
                blank(ax); continue
            ax.set_xticks([30, 120, 240, 360, 480])
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(f'{VARIANT_SHORT[v]}\nRMSE (mg/dL)')
            if i == len(VARIANTS) - 1:
                ax.set_xlabel('horizon (min)')
    size_legend(fig)
    suptitle(fig, 'Median-line point RMSE across the full horizon sweep',
             'Dashed muted rule is last-value persistence.')
    save(fig, 'fig06_rmse_vs_horizon.png')


def fig_basis_gap(M: dict) -> None:
    """How far the band-scored headline sits below the median line it is not comparable to."""
    fig, axes = grid(len(HORIZONS), len(COHORTS), w=3.3, h=2.6)
    rows_lbl = [f'{s}/{v}' for s in SIZES for v in VARIANTS]
    for i, h in enumerate(HORIZONS):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            before, after = [], []
            for s in SIZES:
                for v in VARIANTS:
                    row = horizon_row(M, s, v, c, h)
                    after.append(band_scored(row, 'rmse_point'))
                    before.append(median_line(row, 'rmse_point'))
            if not (np.isfinite(before).any() and np.isfinite(after).any()):
                blank(ax); continue
            for y, (a, b) in enumerate(zip(before, after)):
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                ax.plot([b, a], [y, y], color=fs.PAIR[0], lw=1.4, zorder=2)
                ax.scatter([b], [y], s=26, color=fs.PAIR[0], zorder=3,
                           edgecolor=fs.SURFACE, linewidth=1.2)
                ax.scatter([a], [y], s=26, color=fs.PAIR[1], zorder=3,
                           edgecolor=fs.SURFACE, linewidth=1.2)
            fs.rows(ax, rows_lbl if j == 0 else [''] * len(rows_lbl))
            ax.tick_params(axis='y', labelsize=6.5)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if i == len(HORIZONS) - 1:
                ax.set_xlabel('RMSE (mg/dL)')
            if j == 0:
                ax.set_ylabel(f'{h} min')
    handles = [Line2D([], [], marker='o', linestyle='none', markersize=7, markerfacecolor=c,
                      markeredgecolor=fs.SURFACE, markeredgewidth=2, label=t)
               for c, t in zip(fs.PAIR, ('band-scored', 'median line'))]
    _fig_legend(fig, handles, 2)
    suptitle(fig, 'Reporting basis: band-scored headline against the median line',
             'The band-scored figure charges zero error wherever truth falls inside the 50% band, '
             'so it is a band-geometry diagnostic and not comparable to a published point forecast.')
    save(fig, 'fig07_reporting_basis_gap.png')


def fig_scaling(M: dict, scaling: dict) -> None:
    fig, axes = grid(len(HORIZONS), len(COHORTS), w=3.3, h=2.5)
    xs = [n_params(M, s) for s in SIZES]
    for i, h in enumerate(HORIZONS):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            drew = False
            for v in VARIANTS:
                y = [median_line(horizon_row(M, s, v, c, h), 'rmse_point') for s in SIZES]
                if not np.isfinite(y).any():
                    continue
                ax.plot(xs, y, color=VARIANT_COLOR[v], lw=1.8, marker='o', markersize=5,
                        markeredgecolor=fs.SURFACE, markeredgewidth=1.5, linestyle='none')
                fit = (scaling.get(c, {}).get(v, {}).get(h) or {})
                b, a = _f(fit.get('exponent')), _f(fit.get('coefficient'))
                if math.isfinite(b) and math.isfinite(a):
                    gx = np.geomspace(min(xs), max(xs), 32)
                    ax.plot(gx, a * gx ** b, color=VARIANT_COLOR[v], lw=1.2, alpha=0.75)
                    ax.annotate(f'b={b:+.3f}', xy=(0.03, 0.06 + 0.11 * VARIANTS.index(v)),
                                xycoords='axes fraction', fontsize=7, color=VARIANT_COLOR[v])
                drew = True
            if not drew:
                blank(ax); continue
            ax.set_xscale('log'); ax.set_yscale('log')
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(f'{h} min\nRMSE (mg/dL)')
            if i == len(HORIZONS) - 1:
                ax.set_xlabel('parameters')
    variant_legend(fig)
    suptitle(fig, 'Median-line RMSE against capacity, fitted as a power law',
             'Log-log axes; b is the fitted exponent of RMSE ~ params^b. A negative b means '
             'accuracy improves with capacity.')
    save(fig, 'fig08_scaling_law.png')


# ---------------------------------------------------------------------------- #
# C. Clinical grids.
# ---------------------------------------------------------------------------- #
def fig_clarke(M: dict) -> None:
    keys = [('clarke_A', 'zone A (%)'), ('clarke_AB', 'zone A+B (%)'), ('clarke_D', 'zone D (%)')]
    fig, axes = grid(len(keys), len(COHORTS), w=3.3, h=2.5)
    for i, (key, ylab) in enumerate(keys):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            drew = False
            for v in VARIANTS:
                y = [median_line(horizon_row(M, s, v, c, '60'), key) for s in SIZES]
                if not np.isfinite(y).any():
                    continue
                ax.plot(range(len(SIZES)), y, color=VARIANT_COLOR[v], lw=2.0, marker='o',
                        markersize=5, markeredgecolor=fs.SURFACE, markeredgewidth=1.5)
                drew = True
            if not drew:
                blank(ax); continue
            size_axis(ax); fs.ygrid(ax)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(ylab)
    variant_legend(fig)
    suptitle(fig, 'Clarke Error Grid zones at the 60-minute horizon',
             'Median-line basis. Zone A+B is clinically acceptable; zone D is a dangerous '
             'failure to detect.')
    save(fig, 'fig09_clarke_zones.png')


def fig_cgega(M: dict) -> None:
    regions = [('hypo', 'hypoglycaemic range'), ('eu', 'euglycaemic range'),
               ('hyper', 'hyperglycaemic range')]
    fig, axes = grid(len(regions), len(COHORTS), w=3.3, h=2.6)
    labels = [f'{s}/{v}' for s in SIZES for v in VARIANTS]
    for i, (reg, title) in enumerate(regions):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            ap, be, ep = [], [], []
            for s in SIZES:
                for v in VARIANTS:
                    cg = cohort_stats(M, s, v, c).get('cgega') or {}
                    ap.append(_f(cg.get(f'ap_{reg}')))
                    be.append(_f(cg.get(f'be_{reg}')))
                    ep.append(_f(cg.get(f'ep_{reg}')))
            if not np.isfinite(ap).any():
                blank(ax); continue
            y = np.arange(len(labels))
            ap_a = np.nan_to_num(np.asarray(ap, float))
            be_a = np.nan_to_num(np.asarray(be, float))
            ep_a = np.nan_to_num(np.asarray(ep, float))
            ax.barh(y, ap_a, color=fs.SEQ[4], height=0.66, label='accurate')
            ax.barh(y, be_a, left=ap_a, color=fs.SEQ[2], height=0.66, label='benign')
            ax.barh(y, ep_a, left=ap_a + be_a, color=fs.SERIES[1], height=0.66, label='erroneous')
            fs.rows(ax, labels if j == 0 else [''] * len(labels))
            ax.tick_params(axis='y', labelsize=6.5)
            ax.set_xlim(0, 100); ax.margins(x=0)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(title)
            if i == len(regions) - 1:
                ax.set_xlabel('% of readings')
    handles = [Patch(facecolor=col, edgecolor=fs.SURFACE, label=lab)
               for col, lab in ((fs.SEQ[4], 'accurate'), (fs.SEQ[2], 'benign error'),
                                (fs.SERIES[1], 'erroneous'))]
    _fig_legend(fig, handles, 3)
    suptitle(fig, 'CG-EGA by glycaemic region',
             'Band-scored basis — CG-EGA scores clip(true, q_lo, q_hi) and has no median-line '
             'counterpart. Read as a band diagnostic, not as a point-forecast result.')
    save(fig, 'fig10_cgega.png')


# ---------------------------------------------------------------------------- #
# D. Excursion detection, off the alarm band edges.
# ---------------------------------------------------------------------------- #
def fig_detection(M: dict, event: str, fname: str) -> None:
    fig, axes = grid(2, len(COHORTS), w=3.3, h=2.6)
    for i, stat in enumerate(('recall', 'precision')):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            drew = False
            for v in VARIANTS:
                for hi, h in enumerate(HORIZONS):
                    y = [band_scored(horizon_row(M, s, v, c, h), event, stat) for s in SIZES]
                    if not np.isfinite(y).any():
                        continue
                    ax.plot(range(len(SIZES)), y, color=VARIANT_COLOR[v], lw=1.7,
                            alpha=(0.42, 0.7, 1.0)[hi], marker=('^', 'o', 's')[hi],
                            markersize=4.4, markeredgecolor=fs.SURFACE, markeredgewidth=1.1)
                    drew = True
            if not drew:
                blank(ax); continue
            size_axis(ax); fs.ygrid(ax); ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(stat)
    handles = [Line2D([], [], color=VARIANT_COLOR[v], lw=2.4, label=VARIANT_LABEL[v])
               for v in VARIANTS]
    handles += [Line2D([], [], color=fs.MUTED, lw=1.7, marker=m, markersize=4.4,
                       label=f'{h} min') for m, h in zip(('^', 'o', 's'), HORIZONS)]
    _fig_legend(fig, handles, 6, fontsize=8)
    suptitle(fig, f'{event.capitalize()}glycaemia detection',
             f'Keyed off the τ alarm band edge, not the median line. Marker shape is the '
             f'horizon; hue is the training variant.')
    save(fig, fname)


def fig_threshold_sweep(M: dict) -> None:
    fig, axes = grid(len(VARIANTS), len(COHORTS), w=3.3, h=2.6)
    for i, v in enumerate(VARIANTS):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            drew = False
            for s in SIZES:
                tc = ((cohort_stats(M, s, v, c).get('threshold_curves') or {})
                      .get('hypo') or {}).get('60') or []
                pts = [(_f(p.get('recall')), _f(p.get('precision'))) for p in tc]
                pts = [(r, p) for r, p in pts if math.isfinite(r) and math.isfinite(p)]
                if not pts:
                    continue
                r = [p[0] for p in pts]; pr = [p[1] for p in pts]
                ax.plot(r, pr, color=SIZE_COLOR[s], lw=1.6, marker='o', markersize=2.6,
                        markeredgecolor='none')
                drew = True
            if not drew:
                blank(ax); continue
            ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(f'{VARIANT_SHORT[v]}\nprecision')
            if i == len(VARIANTS) - 1:
                ax.set_xlabel('recall')
    size_legend(fig)
    suptitle(fig, 'Hypoglycaemia precision-recall traced by the alarm offset sweep, 60 min',
             'Each curve walks the alarm threshold offset; capacity rides the ordinal ladder.')
    save(fig, 'fig13_threshold_sweep.png')


def fig_night_onset(M: dict) -> None:
    fig, axes = grid(2, len(COHORTS), w=3.3, h=2.6)
    for i, event in enumerate(('hypo', 'hyper')):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            rec, prec, cols = [], [], []
            for s in SIZES:
                for v in VARIANTS:
                    no = cohort_stats(M, s, v, c).get('night_onset') or {}
                    blk = no.get(event) or {}
                    rec.append(_f(blk.get('recall')))
                    prec.append(_f(blk.get('precision')))
                    cols.append(SIZE_COLOR[s])
            if not np.isfinite(rec).any():
                blank(ax); continue
            ax.scatter(rec, prec, s=46, c=cols, edgecolor=fs.SURFACE, linewidth=1.4, zorder=3)
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('recall')
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(f'overnight {event}\nprecision')
    size_legend(fig)
    suptitle(fig, 'Overnight excursion onset',
             'One mark per model; position is the recall-precision pair over whole nights.')
    save(fig, 'fig14_night_onset.png')




# ---------------------------------------------------------------------------- #
# E. Uncertainty and calibration.
# ---------------------------------------------------------------------------- #
def fig_conformal(M: dict) -> None:
    fig, axes = grid(2, len(COHORTS), w=3.3, h=2.6)
    for j, c in enumerate(COHORTS):
        for i, (key, ylab) in enumerate((('half_width', '90% interval half-width (mg/dL)'),
                                         ('coverage', 'realized coverage'))):
            ax = axes[i][j]
            drew = False
            for v in VARIANTS:
                for hi, h in enumerate(HORIZONS):
                    y = [_f(((cohort_stats(M, s, v, c).get('conformal') or {}).get(h) or {})
                            .get(key)) for s in SIZES]
                    if not np.isfinite(y).any():
                        continue
                    ax.plot(range(len(SIZES)), y, color=VARIANT_COLOR[v], lw=1.7,
                            alpha=(0.42, 0.7, 1.0)[hi], marker=('^', 'o', 's')[hi],
                            markersize=4.4, markeredgecolor=fs.SURFACE, markeredgewidth=1.1)
                    drew = True
            if not drew:
                blank(ax); continue
            size_axis(ax); fs.ygrid(ax)
            if key == 'coverage':
                fs.threshold(ax, 0.90, 'nominal 0.90')
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(ylab)
    handles = [Line2D([], [], color=VARIANT_COLOR[v], lw=2.4, label=VARIANT_LABEL[v])
               for v in VARIANTS]
    handles += [Line2D([], [], color=fs.MUTED, lw=1.7, marker=m, markersize=4.4,
                       label=f'{h} min') for m, h in zip(('^', 'o', 's'), HORIZONS)]
    _fig_legend(fig, handles, 6, fontsize=8)
    suptitle(fig, 'Split-conformal 90% interval: width and realized coverage',
             'Calibrated on the held-out calibration split, measured on the test split.')
    save(fig, 'fig16_conformal.png')


def fig_cqr(M: dict) -> None:
    fig, axes = grid(2, len(COHORTS), w=3.7, h=2.9)
    labels = [f'{s}/{v}' for s in SIZES for v in VARIANTS]
    panels = [('cov90', 'raw_cov90', 'cal_cov90', '90% coverage'),
              ('escape', 'raw_hypo_escape', 'cal_hypo_escape', 'hypo escape rate')]
    for i, (_, rk, ck, ylab) in enumerate(panels):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            before, after = [], []
            for s in SIZES:
                for v in VARIANTS:
                    blk = ((cohort_stats(M, s, v, c).get('conformal_cqr') or {}).get('60')) or {}
                    before.append(_f(blk.get(rk)))
                    after.append(_f(blk.get(ck)))
            if not np.isfinite(before).any():
                blank(ax); continue
            fs.dumbbell_rows(ax, labels if j == 0 else [''] * len(labels),
                             before, after, fmt='{:.2f}')
            ax.tick_params(axis='y', labelsize=6.5)
            if ylab == '90% coverage':
                ax.axvline(0.90, color=fs.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(ylab)
    fs.pair_legend(fig, 'raw quantile fan', 'CQR-calibrated',
                   loc='upper right',
                   bbox_to_anchor=(0.997, _frac_from_top(fig, LEGEND_IN)), ncol=2)
    suptitle(fig, 'Conformalized quantile regression at 60 minutes',
             'Coverage should move toward the dashed nominal rule and the hypoglycaemia escape '
             'rate should fall.')
    save(fig, 'fig17_cqr_calibration.png')


def fig_band_geometry(M: dict) -> None:
    fig, axes = grid(2, len(COHORTS), w=3.3, h=2.6)
    for i, (key, ylab, ref) in enumerate((('band_cov50', 'realized 50% band coverage', 0.50),
                                          ('band_width', 'mean band width (mg/dL)', None))):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            drew = False
            for v in VARIANTS:
                for hi, h in enumerate(HORIZONS):
                    y = [band_scored(horizon_row(M, s, v, c, h), key) for s in SIZES]
                    if not np.isfinite(y).any():
                        continue
                    ax.plot(range(len(SIZES)), y, color=VARIANT_COLOR[v], lw=1.7,
                            alpha=(0.42, 0.7, 1.0)[hi], marker=('^', 'o', 's')[hi],
                            markersize=4.4, markeredgecolor=fs.SURFACE, markeredgewidth=1.1)
                    drew = True
            if not drew:
                blank(ax); continue
            size_axis(ax); fs.ygrid(ax)
            if ref is not None:
                fs.threshold(ax, ref, 'nominal 0.50')
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(ylab)
    handles = [Line2D([], [], color=VARIANT_COLOR[v], lw=2.4, label=VARIANT_LABEL[v])
               for v in VARIANTS]
    handles += [Line2D([], [], color=fs.MUTED, lw=1.7, marker=m, markersize=4.4,
                       label=f'{h} min') for m, h in zip(('^', 'o', 's'), HORIZONS)]
    _fig_legend(fig, handles, 6, fontsize=8)
    suptitle(fig, 'Geometry of the 50% quantile band',
             'The band the band-scored headline is projected onto: how often it contains the '
             'truth, and how wide it has to be to do so.')
    save(fig, 'fig18_band_geometry.png')






def fig_whatif_dose(M: dict) -> None:
    fig, axes = grid(2, len(COHORTS), w=3.5, h=2.7)
    for i, chan in enumerate(('carb', 'insulin')):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            drew = False
            for s in SIZES:
                blk = ((M['cell'][(s, VARIANTS[0])]['whatif'] or {}).get(c) or {}).get(chan) or {}
                doses = [_f(d) for d in (blk.get('doses') or [])]
                y = [_f(x) for x in ((blk.get('mean_dbg') or {}).get('120') or [])]
                if not doses or not np.isfinite(y).any():
                    continue
                ax.plot(doses, y, color=SIZE_COLOR[s], lw=1.9, marker='o', markersize=4,
                        markeredgecolor=fs.SURFACE, markeredgewidth=1.2)
                drew = True
            if not drew:
                blank(ax); continue
            fs.zeroline(ax)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(f'{chan} response\nΔBG at 120 min (mg/dL)')
            ax.set_xlabel('carbohydrate (g)' if chan == 'carb' else 'insulin (U)')
    size_legend(fig)
    suptitle(fig, 'Counterfactual dose response',
             'Injecting a dose into an otherwise quiet context and reading the forecast shift. '
             'Carbohydrate should raise BG, insulin should lower it.')
    save(fig, 'fig21_whatif_dose_response.png')


def fig_whatif_quality(M: dict) -> None:
    rows_spec = [
        ('carb', 'carb — correct sign fraction', True),
        ('insulin', 'insulin — correct sign fraction', True),
        ('carb', 'carb — terminal monotonicity', False),
        ('insulin', 'insulin — terminal monotonicity', False),
    ]
    fig, axes = grid(len(rows_spec), len(COHORTS), w=3.5, h=2.4)
    for i, (chan, ylab, from_curve) in enumerate(rows_spec):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            drew = False
            for v in VARIANTS:
                y = []
                for s in SIZES:
                    blk = ((M['cell'][(s, v)]['whatif'] or {}).get(c) or {}).get(chan) or {}
                    if from_curve:
                        seq = [_f(x) for x in ((blk.get('correct_sign_frac') or {})
                                               .get('120') or [])]
                        seq = [x for x in seq if math.isfinite(x)]
                        y.append(float(np.mean(seq)) if seq else math.nan)
                    else:
                        y.append(_f((blk.get('monotone_frac') or {}).get('terminal')))
                if not np.isfinite(y).any():
                    continue
                ax.plot(range(len(SIZES)), y, color=VARIANT_COLOR[v], lw=2.0, marker='o',
                        markersize=5, markeredgecolor=fs.SURFACE, markeredgewidth=1.5)
                drew = True
            if not drew:
                blank(ax); continue
            size_axis(ax); fs.ygrid(ax); ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
    variant_legend(fig)
    suptitle(fig, 'Counterfactual response quality',
             'Sign correctness averaged over the dose ladder at 120 minutes, and the fraction '
             'of windows whose terminal response orders monotonically with dose.')
    save(fig, 'fig22_whatif_quality.png')


def _empty_future_probed(M: dict) -> bool:
    """Did any run actually probe the empty-future arm?

    It needs raw carb/bolus EVENTS to strip, and a simulator segment carries
    pre-resolved channels only — ``whatif.run`` marks both arms ``not_probed``
    rather than reporting the null arm relabelled. Drawing the figure anyway
    ships three "no data" panels, which read as a measured zero.
    """
    for size in SIZES:
        for v in VARIANTS:
            ef = (((M['cell'][(size, v)]['whatif'] or {}).get(VARIANTS[0]) or {})
                  .get('empty_future') or {})
            for arm in ('all', 'quiet'):
                blk = ef.get(arm) or {}
                if blk.get('n') and not blk.get('not_probed'):
                    return True
    return False


def fig_empty_future(M: dict) -> None:
    keys = [('mean_turning_points', 'mean turning points'),
            ('monotone_frac', 'monotone fraction'),
            ('net_drift_mgdl', 'net drift (mg/dL)')]
    fig, axes = grid(len(keys), len(COHORTS), w=3.5, h=2.5)
    for i, (key, ylab) in enumerate(keys):
        for j, c in enumerate(COHORTS):
            ax = axes[i][j]
            drew = False
            for v in VARIANTS:
                y = [_f((((M['cell'][(s, v)]['whatif'] or {}).get(c) or {})
                         .get('empty_future') or {}).get('quiet', {}).get(key)) for s in SIZES]
                if not np.isfinite(y).any():
                    continue
                ax.plot(range(len(SIZES)), y, color=VARIANT_COLOR[v], lw=2.0, marker='o',
                        markersize=5, markeredgecolor=fs.SURFACE, markeredgewidth=1.5)
                drew = True
            if not drew:
                blank(ax); continue
            size_axis(ax); fs.ygrid(ax)
            if key == 'net_drift_mgdl':
                fs.zeroline(ax)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(ylab)
    variant_legend(fig)
    suptitle(fig, 'Shape of the forecast under a quiet, event-free future',
             'With nothing logged ahead, how much structure the trajectory still invents.')
    save(fig, 'fig23_empty_future.png')








def fig_ranking(rankings: dict) -> None:
    fig, axes = grid(1, len(HORIZONS), w=4.6, h=4.0)
    for j, h in enumerate(HORIZONS):
        ax = axes[0][j]
        order = rankings.get('pooled', {}).get(h, [])
        if not order:
            blank(ax); continue
        names = [r['model'] for r in order]
        vals = [_f(r['rmse_point']) for r in order]
        cols = [SIZE_COLOR[r['size']] for r in order]
        y = np.arange(len(names))
        bars = ax.barh(y, vals, color=cols, height=0.68)
        for b, v in zip(bars, vals):
            if math.isfinite(v):
                ax.annotate(f'{v:.1f}', xy=(v, b.get_y() + b.get_height() / 2),
                            xytext=(4, 0), textcoords='offset points', ha='left',
                            va='center', fontsize=7.5, color=fs.INK2)
        fs.rows(ax, names)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.set_title(f'{h} min'); ax.set_xlabel('pooled RMSE (mg/dL)')
    size_legend(fig)
    suptitle(fig, 'Every capacity ranked by pooled accuracy',
             'Median-line point RMSE, window-weighted. Shorter is better.')
    save(fig, 'fig27_ranking.png')


# ---------------------------------------------------------------------------- #
# JSON products.
# ---------------------------------------------------------------------------- #
def build_models(M: dict) -> dict:
    out = {}
    for size in SIZES:
        sz = M['size'][size]
        ts, cfg = sz['training_summary'] or {}, sz['resolved_config'] or {}
        prog, hw = ts.get('progress') or {}, ts.get('hardware') or {}
        rec = {
            'size': size,
            'arch_label': (sz['summary'] or {}).get('arch_label'),
            'n_params': (M['params'].get(size) or {}).get('n_params'),
            'arch_version': ts.get('arch_version'),
            'loss_schema': ts.get('loss_schema'),
            'architecture': {k: cfg.get(k) for k in
                             ('d_model', 'n_layers', 'n_heads', 'ffn_dim', 'patch_size',
                              'max_context_patches', 'min_context_patches',
                              'prediction_patches', 'prediction_horizon_hours',
                              'night_long_horizon_hours')},
            'optimization': {k: cfg.get(k) for k in
                             ('total_steps', 'batch_size', 'muon_lr', 'muon_momentum',
                              'adam_lr', 'adam_weight_decay', 'warmup_steps', 'lr_min_ratio',
                              'gradient_clip_norm', 'ema_decay', 'master_seed')},
            'pretraining': {
                'elapsed_hours': prog.get('elapsed_hours'),
                'steps_per_second': prog.get('steps_per_second'),
                'gpu_peak_memory_mb': hw.get('gpu_peak_memory_mb'),
                'device': hw.get('device'),
                'best_val_loss': (ts.get('validation') or {}).get('best_val_loss'),
                'best_val_step': (ts.get('validation') or {}).get('best_val_step'),
            },
            'variants': {},
        }
        for v in VARIANTS:
            rec['variants'][v] = {
                'label': VARIANT_LABEL[v],
                'evaluated_step': ((M['cell'][(size, v)]['sim'] or {}).get('_meta') or {})
                                  .get('step'),
            }
        out[size] = rec
    return out


def build_metrics_long(M: dict) -> list[dict]:
    """Tidy long-form records. Every row names the basis it was measured on."""
    rows: list[dict] = []
    for size in SIZES:
        for v in VARIANTS:
            for c in COHORTS:
                st = cohort_stats(M, size, v, c)
                n_test = st.get('n_test_windows')
                n_pat = st.get('n_patients')
                for h in HORIZONS:
                    row = horizon_row(M, size, v, c, h)
                    if not row:
                        continue
                    common = {'size': size, 'variant': v, 'cohort': c, 'horizon_min': int(h),
                              'n_test_windows': n_test, 'n_patients': n_pat}
                    for key in ('rmse_point', 'mae_point', 'rmse_winmean', 'mae_winmean',
                                'rmse_macro', 'mard', 'clarke_A', 'clarke_AB', 'clarke_D',
                                'clarke_E', 'skill_point'):
                        rows.append({**common, 'basis': 'median_line', 'metric': key,
                                     'value': median_line(row, key)})
                        rows.append({**common, 'basis': 'band_scored', 'metric': key,
                                     'value': band_scored(row, key)})
                    for key in ('band_cov50', 'band_width'):
                        rows.append({**common, 'basis': 'band_geometry', 'metric': key,
                                     'value': band_scored(row, key)})
                    for key in ('rmse_persist_point', 'rmse_persist_winmean'):
                        rows.append({**common, 'basis': 'persistence', 'metric': key,
                                     'value': band_scored(row, key)})
                    for ev in ('hypo', 'hyper'):
                        for stat in ('recall', 'precision', 'n_true', 'n_pred'):
                            rows.append({**common, 'basis': 'alarm_band_edge',
                                         'metric': f'{ev}_{stat}',
                                         'value': band_scored(row, ev, stat)})
                cg = st.get('cgega') or {}
                for key in ('ap_hypo', 'be_hypo', 'ep_hypo', 'ap_eu', 'be_eu', 'ep_eu',
                            'ap_hyper', 'be_hyper', 'ep_hyper'):
                    rows.append({'size': size, 'variant': v, 'cohort': c, 'horizon_min': None,
                                 'n_test_windows': n_test, 'n_patients': n_pat,
                                 'basis': 'band_scored', 'metric': f'cgega_{key}',
                                 'value': _f(cg.get(key))})
                for h in SWEEP_HORIZONS:
                    blk = (st.get('rmse_by_hour') or {}).get(h) or {}
                    common = {'size': size, 'variant': v, 'cohort': c, 'horizon_min': int(h),
                              'n_test_windows': blk.get('n'), 'n_patients': n_pat}
                    rows.append({**common, 'basis': 'median_line', 'metric': 'sweep_rmse_point',
                                 'value': _f(blk.get('rmse_point_median'))})
                    rows.append({**common, 'basis': 'band_scored', 'metric': 'sweep_rmse_point',
                                 'value': _f(blk.get('rmse_point'))})
                    rows.append({**common, 'basis': 'persistence',
                                 'metric': 'sweep_rmse_persist_point',
                                 'value': _f(blk.get('rmse_persist_point'))})
    return rows


def build_pooled(M: dict) -> dict:
    """Window-weighted pooling across the evaluation sources, on the median line."""
    out: dict = {}
    for size in SIZES:
        for v in VARIANTS:
            key = f'{size}/{v}'
            out[key] = {}
            for h in HORIZONS:
                wts = [_f(cohort_stats(M, size, v, c).get('n_test_windows'))
                       for c in COHORTS]
                rec = {'n_test_windows': float(np.nansum(wts))}
                for m in ('rmse_point', 'rmse_winmean', 'rmse_macro'):
                    rec[m] = pooled_rmse(
                        [median_line(horizon_row(M, size, v, c, h), m) for c in COHORTS], wts)
                for m in ('mae_point', 'mard', 'clarke_A', 'clarke_AB', 'clarke_D',
                          'skill_point'):
                    rec[m] = pooled_mean(
                        [median_line(horizon_row(M, size, v, c, h), m) for c in COHORTS], wts)
                for ev in ('hypo', 'hyper'):
                    for stat in ('recall', 'precision'):
                        rec[f'{ev}_{stat}'] = pooled_mean(
                            [band_scored(horizon_row(M, size, v, c, h), ev, stat)
                             for c in COHORTS], wts)
                rec['rmse_persist_point'] = pooled_rmse(
                    [band_scored(horizon_row(M, size, v, c, h), 'rmse_persist_point')
                     for c in COHORTS], wts)
                out[key][h] = rec
    return out


def build_rankings(M: dict, pooled: dict) -> dict:
    out: dict = {'pooled': {}}
    for h in HORIZONS:
        recs = []
        for size in SIZES:
            for v in VARIANTS:
                val = _f(pooled[f'{size}/{v}'][h].get('rmse_point'))
                recs.append({'model': f'{size}/{v}', 'size': size, 'variant': v,
                             'rmse_point': val,
                             'mard': _f(pooled[f'{size}/{v}'][h].get('mard')),
                             'clarke_AB': _f(pooled[f'{size}/{v}'][h].get('clarke_AB')),
                             'skill_point': _f(pooled[f'{size}/{v}'][h].get('skill_point'))})
        recs = [r for r in recs if math.isfinite(r['rmse_point'])]
        recs.sort(key=lambda r: r['rmse_point'])
        for i, r in enumerate(recs, 1):
            r['rank'] = i
        out['pooled'][h] = recs
    for c in COHORTS:
        out[c] = {}
        for h in HORIZONS:
            recs = []
            for size in SIZES:
                for v in VARIANTS:
                    val = median_line(horizon_row(M, size, v, c, h), 'rmse_point')
                    if math.isfinite(val):
                        recs.append({'model': f'{size}/{v}', 'size': size, 'variant': v,
                                     'rmse_point': val})
            recs.sort(key=lambda r: r['rmse_point'])
            for i, r in enumerate(recs, 1):
                r['rank'] = i
            out[c][h] = recs
    return out


def build_scaling(M: dict) -> dict:
    xs = [n_params(M, s) for s in SIZES]
    out: dict = {}
    for c in COHORTS:
        out[c] = {}
        for v in VARIANTS:
            out[c][v] = {}
            for h in HORIZONS:
                ys = [median_line(horizon_row(M, s, v, c, h), 'rmse_point') for s in SIZES]
                out[c][v][h] = dict(loglog_fit(xs, ys), basis='median_line',
                                    metric='rmse_point')
    return out




def build_uncertainty(M: dict) -> dict:
    out: dict = {}
    for size in SIZES:
        for v in VARIANTS:
            key = f'{size}/{v}'
            out[key] = {}
            for c in COHORTS:
                st = cohort_stats(M, size, v, c)
                rec: dict = {'conformal': {}, 'conformal_cqr': {}, 'band': {}}
                for h in HORIZONS:
                    cf = (st.get('conformal') or {}).get(h) or {}
                    rec['conformal'][h] = {'half_width': _f(cf.get('half_width')),
                                           'coverage': _f(cf.get('coverage'))}
                    cq = (st.get('conformal_cqr') or {}).get(h) or {}
                    rec['conformal_cqr'][h] = {k: _f(cq.get(k)) for k in
                                               ('raw_cov90', 'cal_cov90', 'raw_hypo_escape',
                                                'cal_hypo_escape', 'n_cal', 'n_test')}
                    row = horizon_row(M, size, v, c, h)
                    rec['band'][h] = {'band_cov50': band_scored(row, 'band_cov50'),
                                      'band_width': band_scored(row, 'band_width')}
                out[key][c] = rec
    return out






def build_index() -> dict:
    return {
        'generated_by': 'compare.py',
        'grid': {'sizes': list(SIZES), 'variants': list(VARIANTS),
                 'variant_labels': VARIANT_LABEL,
                 'cohorts': list(COHORTS),
                 'horizons_min': [int(h) for h in HORIZONS],
                 'sweep_horizons_min': [int(h) for h in SWEEP_HORIZONS]},
        'reporting_basis': {
            'median_line': 'Point forecast f_inv(median). Headline accuracy basis; the basis '
                           'published point-forecast numbers use.',
            'band_scored': 'Computed on pred_eff = clip(true, q_lo, q_hi), which charges zero '
                           'error wherever truth lies inside the 50% band. A band diagnostic, '
                           'not comparable to a point forecast.',
            'alarm_band_edge': 'Excursion detection keys off the tau-lower and tau-upper alarm '
                               'band edges; it has no median-line counterpart.',
            'band_geometry': 'Properties of the 50% quantile band itself.',
            'persistence': 'Last-value baseline, carried for skill.',
        },
        'files': {
            # index.json names itself; it is written last, so WRITTEN does not yet hold it.
            'data': sorted([p for p in WRITTEN if '/data/' in p.replace(os.sep, '/')]
                           + [os.path.relpath(os.path.join(DATADIR, 'index.json'), ROOT)]),
            'figures': sorted(p for p in WRITTEN if '/figures/' in p.replace(os.sep, '/')),
        },
    }


# ---------------------------------------------------------------------------- #
def main() -> None:
    global FIGDIR, DATADIR
    ap = argparse.ArgumentParser(description=(__doc__ or 'model comparison').strip().split('\n')[0])
    ap.add_argument('--out', default=os.path.join(ROOT, 'comparison'),
                    help='output directory (default: comparison/)')
    ap.add_argument('--dpi', type=int, default=300, help='figure resolution (default: 300)')
    args = ap.parse_args()

    FIGDIR = os.path.join(args.out, 'figures')
    DATADIR = os.path.join(args.out, 'data')

    fs.style()
    plt.rcParams.update({'savefig.dpi': args.dpi, 'figure.dpi': 110})

    print('loading ...')
    M = load_all()

    print('deriving ...')
    models = build_models(M)
    long_rows = build_metrics_long(M)
    pooled = build_pooled(M)
    rankings = build_rankings(M, pooled)
    scaling = build_scaling(M)
    uncertainty = build_uncertainty(M)

    print('figures ...')
    fig_capacity(M)
    fig_pretrain(M)
    fig_accuracy(M, 'rmse_point', 'RMSE (mg/dL)',
                 'Median-line point RMSE across the grid', 'fig03_rmse.png')
    fig_accuracy(M, 'mard', 'MARD (%)',
                 'Median-line MARD across the grid', 'fig04_mard.png')
    fig_skill(M)
    fig_rmse_vs_horizon(M)
    fig_basis_gap(M)
    fig_scaling(M, scaling)
    fig_clarke(M)
    fig_cgega(M)
    fig_detection(M, 'hypo', 'fig11_hypo_detection.png')
    fig_detection(M, 'hyper', 'fig12_hyper_detection.png')
    fig_threshold_sweep(M)
    fig_night_onset(M)
    fig_conformal(M)
    fig_cqr(M)
    fig_band_geometry(M)
    fig_whatif_dose(M)
    fig_whatif_quality(M)
    if _empty_future_probed(M):
        fig_empty_future(M)
    else:
        print('  skipped fig23_empty_future — no run probed the empty-future arm '
              '(a simulator segment carries no raw events to strip)')
    fig_ranking(rankings)

    print('data ...')
    dump(models, 'models.json')
    dump(long_rows, 'metrics_long.json')
    dump(pooled, 'metrics_pooled.json')
    dump(rankings, 'rankings.json')
    dump(scaling, 'scaling.json')
    dump(uncertainty, 'uncertainty.json')
    dump(build_index(), 'index.json')

    print(f'\n{len(WRITTEN)} files under {os.path.relpath(args.out, ROOT)}/')


if __name__ == '__main__':
    main()
