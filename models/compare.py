#!/usr/bin/env python3
"""
Cross-model comparison over the trained checkpoints under ``models/``.

Compares the full grid — three capacities (nano, small, medium) crossed with
three training variants (sim, ohio, multi) — on every evaluation cohort, horizon and
probe the per-model metric directories carry, and writes high-resolution figures and
machine-readable JSON. No markdown is produced.

Reporting basis
---------------
``realdata/metrics.py`` scores each horizon twice and the two are NOT interchangeable:

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

The fine-tune CSVs and the ``finetune_meta.baseline_heldout`` blocks are already on the
median line — ``finetune/finetune.py:_metric`` prefers it — so transfer deltas compare
like with like.

Excluded by request
-------------------
The CGM 15-minute shift probe (``shift15.json``) and everything augmentation-related
(``augexp.json``, the ``augmented/`` evaluation regime) are not read. ``index.json``
records the exclusion.

Colour discipline
-----------------
Palette and chrome come from the suite's single copy at ``metrics/figstyle.py``; this
script adds no second copy. Per that module's rule a mark encodes exactly one job, so
each figure carries exactly one hue job: capacity rides one ordinal blue ladder,
cohort keeps its fixed categorical hue, and variant takes the categorical trio only in
figures where cohort is carried by the panel title instead of by colour.

Usage
-----
    python compare.py                 # writes models/comparison/{figures,data}
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, os.path.join(ROOT, 'metrics'))
import figstyle as fs                                     # type: ignore[import]  # noqa: E402

import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402

# ---------------------------------------------------------------------------- #
# The grid.
# ---------------------------------------------------------------------------- #
SIZES = ('nano', 'small', 'medium')
VARIANTS = ('sim', 'ohio', 'multi')
REAL_COHORTS = ('ohiot1dm', 'azt1d', 'shanghai')
COHORTS = REAL_COHORTS + ('sim',)
HORIZONS = ('30', '60', '120')
SWEEP_HORIZONS = ('30', '60', '120', '180', '240', '300', '360', '420', '480')

VARIANT_LABEL = {
    'sim': 'sim — pretrain only',
    'ohio': 'ohio — fine-tuned on OhioT1DM',
    'multi': 'multi — fine-tuned on 3 cohorts',
}
VARIANT_SHORT = {'sim': 'sim', 'ohio': 'ohio', 'multi': 'multi'}
SIZE_LABEL = {s: s for s in SIZES}

# Capacity is an ordered magnitude, so it rides one hue light-to-dark.
SIZE_COLOR = dict(zip(SIZES, fs.DOSE_CARB[1:1 + len(SIZES)]))
# Variant is categorical. Used only where cohort is not also a hue.
VARIANT_COLOR = dict(zip(VARIANTS, fs.SERIES))
# The simulator cohort is the reference, per figstyle.
COHORT_COLOR = dict(fs.COHORT, sim=fs.REFERENCE)
COHORT_LABEL = dict(fs.COHORT_LABEL, sim='T1DMSIM')

EXCLUDED_INPUTS = {
    'shift15.json': 'CGM 15-minute shift probe — excluded by request',
    'augexp.json': 'augmentation experiment — excluded by request',
    'augmented/': 'augmented evaluation regime — excluded by request',
}


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
    """Parameter count and fine-tune provenance, read from the checkpoints."""
    import torch
    out: dict[str, dict[str, Any]] = {}
    for size in SIZES:
        rec: dict[str, Any] = {'n_params': None, 'finetune': {}}
        for variant in ('ohio', 'multi'):
            path = os.path.join(HERE, size, f'weights_{variant}.pt')
            if not os.path.exists(path):
                continue
            ck = torch.load(path, map_location='cpu', weights_only=False)
            if rec['n_params'] is None:
                sd = ck.get('model_state_dict') or {}
                rec['n_params'] = int(sum(int(v.numel()) for v in sd.values()))
                rec['arch_version'] = ck.get('arch_version')
                rec['loss_schema'] = ck.get('loss_schema')
            meta = ck.get('finetune_meta') or {}
            rec['finetune'][variant] = {
                'step': ck.get('step'),
                'dataset': meta.get('dataset'),
                'datasets': meta.get('datasets'),
                'mode': meta.get('mode'),
                'holdout': meta.get('holdout'),
                'total_steps': meta.get('total_steps'),
                'lr_scale': meta.get('lr_scale'),
                'base_checkpoint': meta.get('base_checkpoint'),
                'baseline_heldout': meta.get('baseline_heldout'),
                'best_heldout': meta.get('best_heldout'),
            }
            del ck
        out[size] = rec
    return out


def load_all() -> dict[str, Any]:
    """Every input this comparison reads, indexed by size and variant."""
    M: dict[str, Any] = {'size': {}, 'cell': {}}
    for size in SIZES:
        base = os.path.join(HERE, size)
        M['size'][size] = {
            'summary': read_json(os.path.join(base, 'figures', 'summary.json')),
            'training_summary': read_json(os.path.join(base, 'logs', 'training_summary.json')),
            'resolved_config': read_json(os.path.join(base, 'logs', 'resolved_config.json')),
            'training_log': read_csv(os.path.join(base, 'logs', 'training_log.csv')),
            'validation_log': read_csv(os.path.join(base, 'logs', 'validation_log.csv')),
            'finetune_log': read_csv(os.path.join(base, 'finetune_log.csv')),
            'finetune_multi_log': read_csv(os.path.join(base, 'finetune_multi_log.csv')),
        }
        for variant in VARIANTS:
            mdir = os.path.join(base, f'metrics_{variant}')
            cell = {
                'real': read_json(os.path.join(mdir, 'real', 'stats.json')),
                'sim': read_json(os.path.join(mdir, 'sim', 'stats.json')),
                'amp_var': read_json(os.path.join(mdir, 'amp_var.json')),
                'time': read_json(os.path.join(mdir, 'time.json')),
                'whatif': read_json(os.path.join(mdir, 'whatif.json')),
                'q2': read_json(os.path.join(mdir, 'q2_event_response.json')),
                'logging_stats': read_json(os.path.join(mdir, 'logging_stats.json')),
            }
            M['cell'][(size, variant)] = cell
    M['meal_stats'] = read_json(os.path.join(HERE, SIZES[0], 'metrics_sim', 'meal_stats.json'))
    M['params'] = load_params()
    return M


def cohort_stats(M: dict, size: str, variant: str, cohort: str) -> dict:
    """The per-cohort stats block, from whichever evaluation regime holds that cohort."""
    cell = M['cell'][(size, variant)]
    regime = 'sim' if cohort == 'sim' else 'real'
    return (cell.get(regime) or {}).get(cohort) or {}


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
    WRITTEN.append(os.path.relpath(path, HERE))
    print(f'  figure  {os.path.relpath(path, HERE)}')
    return path


def dump(obj: Any, name: str) -> str:
    os.makedirs(DATADIR, exist_ok=True)
    path = os.path.join(DATADIR, name)
    with open(path, 'w') as fh:
        json.dump(_jsonable(obj), fh, indent=1, sort_keys=False)
        fh.write('\n')
    WRITTEN.append(os.path.relpath(path, HERE))
    print(f'  data    {os.path.relpath(path, HERE)}')
    return path


# Header geometry is measured in INCHES, not figure fractions: the fonts are in
# points, so a fractional offset that clears the title on a tall figure collides
# with it on a short one.
HEADER_IN = 0.86          # reserved band above the panels
TITLE_IN = 0.17           # title baseline, from the top edge
SUBTITLE_IN = 0.42        # subtitle baseline, from the top edge
LEGEND_IN = 0.13          # legend top, from the top edge


def _frac_from_top(fig, inches: float) -> float:
    return 1.0 - inches / max(fig.get_figheight(), 1e-6)


def suptitle(fig, title: str, subtitle: str = '') -> None:
    fig.suptitle(title, x=0.006, ha='left', va='top', fontsize=13, color=fs.INK,
                 y=_frac_from_top(fig, TITLE_IN))
    if subtitle:
        fig.text(0.006, _frac_from_top(fig, SUBTITLE_IN), subtitle, ha='left', va='top',
                 fontsize=8.5, color=fs.INK2)


def _fig_legend(fig, handles, ncol: int, fontsize: float = 8.5):
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
    fig_h = nrows * h + HEADER_IN
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * w, fig_h), squeeze=False)
    fig.subplots_adjust(top=_frac_from_top(fig, HEADER_IN))
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
    suptitle(fig, 'Pretraining dynamics on the simulator corpus',
             'Shared base for all three variants; the ohio and multi checkpoints fine-tune from it.')
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
                         'Rows are forecast horizons; columns are evaluation cohorts.')
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
             'Dashed muted rule is last-value persistence. Rows are training variants; '
             'columns are evaluation cohorts.')
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


def fig_q2(M: dict) -> None:
    fig, axes = grid(2, len(REAL_COHORTS), w=3.7, h=2.8)
    labels = [f'{s}/{v}' for s in SIZES for v in VARIANTS]
    for i, event in enumerate(('hypo', 'hyper')):
        for j, c in enumerate(REAL_COHORTS):
            ax = axes[i][j]
            with_, without = [], []
            for s in SIZES:
                for v in VARIANTS:
                    blk = ((M['cell'][(s, v)]['q2'] or {}).get(c) or {}).get(event) or {}
                    with_.append(_f((blk.get('with') or {}).get('recall')))
                    without.append(_f((blk.get('without') or {}).get('recall')))
            if not np.isfinite(with_).any():
                blank(ax); continue
            fs.dumbbell_rows(ax, labels if j == 0 else [''] * len(labels),
                             without, with_, fmt='{:.0f}')
            ax.tick_params(axis='y', labelsize=6.5)
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(f'{event} recall (%)')
            if i == 1:
                ax.set_xlabel('recall (%)')
    fs.pair_legend(fig, 'no logged event in context', 'logged event in context',
                   loc='upper right',
                   bbox_to_anchor=(0.997, _frac_from_top(fig, LEGEND_IN)), ncol=2)
    suptitle(fig, 'Excursion recall with and without a logged event in context',
             'Whether the forecast leans on the carbohydrate and bolus log or on the CGM trace '
             'alone.')
    save(fig, 'fig15_event_dependence.png')


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


# ---------------------------------------------------------------------------- #
# F. Behavioural probes.
# ---------------------------------------------------------------------------- #
def fig_amplitude(M: dict) -> None:
    keys = [('amp_ratio', 'amplitude ratio  std(pred) / std(true)', 1.0),
            ('direction_corr', 'direction correlation', None),
            ('g_star', 'g*  optimal trend gain', 1.0)]
    fig, axes = grid(len(keys), len(REAL_COHORTS), w=3.5, h=2.6)
    for i, (key, ylab, ref) in enumerate(keys):
        for j, c in enumerate(REAL_COHORTS):
            ax = axes[i][j]
            drew = False
            for v in VARIANTS:
                y = [_f((((M['cell'][(s, v)]['amp_var'] or {}).get(c) or {})
                         .get('bg_per_patch_dbg') or {}).get(key)) for s in SIZES]
                if not np.isfinite(y).any():
                    continue
                ax.plot(range(len(SIZES)), y, color=VARIANT_COLOR[v], lw=2.0, marker='o',
                        markersize=5, markeredgecolor=fs.SURFACE, markeredgewidth=1.5)
                drew = True
            if not drew:
                blank(ax); continue
            size_axis(ax); fs.ygrid(ax)
            if ref is not None:
                fs.threshold(ax, ref, 'unity')
            if i == 0:
                ax.set_title(COHORT_LABEL[c])
            if j == 0:
                ax.set_ylabel(ylab)
    variant_legend(fig)
    suptitle(fig, 'Trend amplitude and direction fidelity',
             'Whether the forecast moves as far as the truth does, and in the same direction. '
             'A ratio below unity is a flattened, over-smoothed trajectory.')
    save(fig, 'fig19_amplitude_and_trend.png')


def fig_time_of_day(M: dict) -> None:
    keys = [('tod_mae_h', 'clock-phase MAE (hours)', 'mae_h'),
            ('tod_acc_2h', 'within 2 hours (%)', 'acc_2h'),
            ('tod_conf', 'mean confidence', None),
            ('tod_gross_rate', 'gross error rate (%)', 'gross_rate')]
    fig, axes = grid(2, 2, w=4.4, h=3.0)
    for k, (key, ylab, chance_key) in enumerate(keys):
        ax = axes[k // 2][k % 2]
        drew = False
        for v in VARIANTS:
            y = [_f(((M['cell'][(s, v)]['time'] or {}).get('pooled') or {}).get(key))
                 for s in SIZES]
            if not np.isfinite(y).any():
                continue
            ax.plot(range(len(SIZES)), y, color=VARIANT_COLOR[v], lw=2.0, marker='o',
                    markersize=5, markeredgecolor=fs.SURFACE, markeredgewidth=1.5)
            drew = True
        if not drew:
            blank(ax); continue
        if chance_key:
            ch = _f((((M['cell'][(SIZES[0], VARIANTS[0])]['time'] or {}).get('_meta') or {})
                     .get('chance') or {}).get(chance_key))
            if math.isfinite(ch):
                fs.threshold(ax, ch, 'chance')
        size_axis(ax); fs.ygrid(ax)
        ax.set_title(ylab)
    variant_legend(fig)
    suptitle(fig, 'Time-of-day probe, pooled over the real cohorts',
             'What the model infers about clock phase from the trace alone, against the '
             'uniform-guess rule.')
    save(fig, 'fig20_time_of_day.png')


def fig_whatif_dose(M: dict) -> None:
    fig, axes = grid(2, len(REAL_COHORTS), w=3.5, h=2.7)
    for i, chan in enumerate(('carb', 'insulin')):
        for j, c in enumerate(REAL_COHORTS):
            ax = axes[i][j]
            drew = False
            for s in SIZES:
                blk = ((M['cell'][(s, 'multi')]['whatif'] or {}).get(c) or {}).get(chan) or {}
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
    suptitle(fig, 'Counterfactual dose response of the multi-cohort models',
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
    fig, axes = grid(len(rows_spec), len(REAL_COHORTS), w=3.5, h=2.4)
    for i, (chan, ylab, from_curve) in enumerate(rows_spec):
        for j, c in enumerate(REAL_COHORTS):
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


def fig_empty_future(M: dict) -> None:
    keys = [('mean_turning_points', 'mean turning points'),
            ('monotone_frac', 'monotone fraction'),
            ('net_drift_mgdl', 'net drift (mg/dL)')]
    fig, axes = grid(len(keys), len(REAL_COHORTS), w=3.5, h=2.5)
    for i, (key, ylab) in enumerate(keys):
        for j, c in enumerate(REAL_COHORTS):
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


# ---------------------------------------------------------------------------- #
# G. Transfer and generalization.
# ---------------------------------------------------------------------------- #
def fig_finetune(M: dict) -> None:
    fig, axes = grid(2, 3, w=3.9, h=2.9)
    specs = [('finetune_log', 'rmse_point_30', 'ohio — RMSE @30m'),
             ('finetune_log', 'rmse_point_60', 'ohio — RMSE @60m'),
             ('finetune_log', 'rmse_point_120', 'ohio — RMSE @120m'),
             ('finetune_multi_log', 'rmse_point_30', 'multi — RMSE @30m'),
             ('finetune_multi_log', 'rmse_point_60', 'multi — RMSE @60m'),
             ('finetune_multi_log', 'rmse_point_120', 'multi — RMSE @120m')]
    for k, (src, key, title) in enumerate(specs):
        ax = axes[k // 3][k % 3]
        drew = False
        for s in SIZES:
            cols = M['size'][s][src]
            if 'step' not in cols or key not in cols:
                continue
            x, y = cols['step'], cols[key]
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            ax.plot(x[ok], y[ok], color=SIZE_COLOR[s], lw=1.7)
            drew = True
        if not drew:
            blank(ax); continue
        ax.set_title(title); ax.set_xlabel('fine-tune step'); ax.set_ylabel('mg/dL')
    size_legend(fig)
    suptitle(fig, 'Fine-tuning trajectories on the held-out patient',
             'Median-line RMSE — finetune.py records the median line, so these are directly '
             'comparable to the headline accuracy figures.')
    save(fig, 'fig24_finetune_curves.png')


def fig_transfer_delta(transfer: dict) -> None:
    fig, axes = grid(1, len(HORIZONS), w=4.4, h=3.4)
    labels = [f'{s}/{v}' for s in SIZES for v in ('ohio', 'multi')]
    for j, h in enumerate(HORIZONS):
        ax = axes[0][j]
        before, after = [], []
        for s in SIZES:
            for v in ('ohio', 'multi'):
                rec = transfer.get(f'{s}/{v}', {}).get(h, {})
                before.append(_f(rec.get('baseline_rmse_point')))
                after.append(_f(rec.get('finetuned_rmse_point')))
        if not np.isfinite(before).any():
            blank(ax); continue
        fs.dumbbell_rows(ax, labels if j == 0 else [''] * len(labels),
                         before, after, fmt='{:.1f}')
        ax.tick_params(axis='y', labelsize=7.5)
        ax.set_title(f'{h} min'); ax.set_xlabel('RMSE (mg/dL)')
    fs.pair_legend(fig, 'before fine-tuning', 'after fine-tuning',
                   loc='upper right',
                   bbox_to_anchor=(0.997, _frac_from_top(fig, LEGEND_IN)), ncol=2)
    suptitle(fig, 'What fine-tuning bought, on the held-out patient',
             'Median-line RMSE before and after transfer on the identical held-out split, both '
             'read from the checkpoint. The cohort-wide evaluation is a larger set and is not '
             'mixed in here.')
    save(fig, 'fig25_transfer_delta.png')


def fig_reality_gap(M: dict) -> None:
    fig, axes = grid(1, len(HORIZONS), w=4.4, h=3.2)
    for j, h in enumerate(HORIZONS):
        ax = axes[0][j]
        drew = False
        for v in VARIANTS:
            sim = [median_line(horizon_row(M, s, v, 'sim', h), 'rmse_point') for s in SIZES]
            real = []
            for s in SIZES:
                vals, wts = [], []
                for c in REAL_COHORTS:
                    vals.append(median_line(horizon_row(M, s, v, c, h), 'rmse_point'))
                    wts.append(_f(cohort_stats(M, s, v, c).get('n_test_windows')))
                real.append(pooled_rmse(vals, wts))
            if not (np.isfinite(sim).any() and np.isfinite(real).any()):
                continue
            ax.plot(sim, real, color=VARIANT_COLOR[v], lw=1.5, marker='o', markersize=6,
                    markeredgecolor=fs.SURFACE, markeredgewidth=1.6)
            for s, xx, yy in zip(SIZES, sim, real):
                if math.isfinite(xx) and math.isfinite(yy):
                    ax.annotate(s, xy=(xx, yy), xytext=(4, 4), textcoords='offset points',
                                fontsize=7, color=fs.INK2)
            drew = True
        if not drew:
            blank(ax); continue
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=fs.AXIS, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.set_title(f'{h} min')
        ax.set_xlabel('RMSE on the simulator (mg/dL)')
        if j == 0:
            ax.set_ylabel('pooled RMSE on real cohorts (mg/dL)')
    variant_legend(fig)
    suptitle(fig, 'The reality gap',
             'Median-line RMSE on synthetic data against the window-weighted pooling over the '
             'three real cohorts. The dashed diagonal is parity.')
    save(fig, 'fig26_reality_gap.png')


def fig_ranking(rankings: dict) -> None:
    fig, axes = grid(1, len(HORIZONS), w=4.6, h=4.0)
    for j, h in enumerate(HORIZONS):
        ax = axes[0][j]
        order = rankings.get('real_pooled', {}).get(h, [])
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
    suptitle(fig, 'All twelve models ranked by pooled real-cohort accuracy',
             'Median-line point RMSE, window-weighted across OhioT1DM, AZT1D and ShanghaiT1DM. '
             'Shorter is better.')
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
            ft = ((M['params'].get(size) or {}).get('finetune') or {}).get(v) or {}
            rec['variants'][v] = {
                'label': VARIANT_LABEL[v],
                'evaluated_step': ((M['cell'][(size, v)]['real'] or {}).get('_meta') or {})
                                  .get('step'),
                'finetune_dataset': ft.get('dataset'),
                'finetune_datasets': ft.get('datasets'),
                'finetune_mode': ft.get('mode'),
                'finetune_holdout': ft.get('holdout'),
                'finetune_total_steps': ft.get('total_steps'),
                'finetune_lr_scale': ft.get('lr_scale'),
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
    """Window-weighted pooling across the three real cohorts, on the median line."""
    out: dict = {}
    for size in SIZES:
        for v in VARIANTS:
            key = f'{size}/{v}'
            out[key] = {}
            for h in HORIZONS:
                wts = [_f(cohort_stats(M, size, v, c).get('n_test_windows'))
                       for c in REAL_COHORTS]
                rec = {'n_test_windows': float(np.nansum(wts))}
                for m in ('rmse_point', 'rmse_winmean', 'rmse_macro'):
                    rec[m] = pooled_rmse(
                        [median_line(horizon_row(M, size, v, c, h), m) for c in REAL_COHORTS], wts)
                for m in ('mae_point', 'mard', 'clarke_A', 'clarke_AB', 'clarke_D',
                          'skill_point'):
                    rec[m] = pooled_mean(
                        [median_line(horizon_row(M, size, v, c, h), m) for c in REAL_COHORTS], wts)
                for ev in ('hypo', 'hyper'):
                    for stat in ('recall', 'precision'):
                        rec[f'{ev}_{stat}'] = pooled_mean(
                            [band_scored(horizon_row(M, size, v, c, h), ev, stat)
                             for c in REAL_COHORTS], wts)
                rec['rmse_persist_point'] = pooled_rmse(
                    [band_scored(horizon_row(M, size, v, c, h), 'rmse_persist_point')
                     for c in REAL_COHORTS], wts)
                out[key][h] = rec
    return out


def build_rankings(M: dict, pooled: dict) -> dict:
    out: dict = {'real_pooled': {}}
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
        out['real_pooled'][h] = recs
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


def build_transfer(M: dict) -> dict:
    """What fine-tuning bought, measured before and after on the SAME held-out split.

    Both blocks come from the checkpoint's own ``finetune_meta`` — ``baseline_heldout``
    is the pre-transfer model and ``best_heldout`` the selected post-transfer one, each
    scored on the identical held-out patient, calibration split and median-line basis.
    The cohort-wide numbers under ``metrics_*/`` are a different, larger evaluation set
    and are deliberately NOT used here: pairing them would compare one patient against
    six.
    """
    metrics = ('rmse_point', 'rmse_winmean', 'mard', 'clarke_AB', 'skill_point',
               'hypo_recall', 'hyper_recall')
    out: dict = {}
    for size in SIZES:
        for v in ('ohio', 'multi'):
            ft = ((M['params'].get(size) or {}).get('finetune') or {}).get(v) or {}
            base = ft.get('baseline_heldout') or {}
            best = ft.get('best_heldout') or {}
            key = f'{size}/{v}'
            out[key] = {
                'holdout': ft.get('holdout'),
                'n_test_windows': base.get('n_test_windows'),
                'n_cal_windows': base.get('n_cal_windows'),
                'n_patients': base.get('n_patients'),
                'basis': 'median_line',
                'split': 'held-out patient, identical before and after',
            }
            for h in HORIZONS:
                b, a = (base.get(h) or {}), (best.get(h) or {})
                rec: dict = {}
                for m in metrics:
                    bv, av = _f(b.get(m)), _f(a.get(m))
                    rec[f'baseline_{m}'] = bv
                    rec[f'finetuned_{m}'] = av
                    rec[f'delta_{m}'] = (av - bv if math.isfinite(av) and math.isfinite(bv)
                                         else math.nan)
                bv, av = rec['baseline_rmse_point'], rec['finetuned_rmse_point']
                rec['rmse_point_pct_change'] = (100.0 * (av - bv) / bv
                                                if math.isfinite(av) and math.isfinite(bv)
                                                and bv > 0 else math.nan)
                out[key][h] = rec
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


def build_behaviour(M: dict) -> dict:
    out: dict = {}
    for size in SIZES:
        for v in VARIANTS:
            cell = M['cell'][(size, v)]
            key = f'{size}/{v}'
            rec: dict = {'amplitude': {}, 'time_of_day': {}, 'counterfactual': {},
                         'event_dependence': {}}
            for c in REAL_COHORTS:
                amp = (cell['amp_var'] or {}).get(c) or {}
                rec['amplitude'][c] = {
                    'bg_per_patch_dbg': amp.get('bg_per_patch_dbg'),
                    'inter_patch_variability': amp.get('inter_patch_variability'),
                }
                w = (cell['whatif'] or {}).get(c) or {}
                cf: dict = {'n_windows': w.get('n_windows'), 'n_quiet': w.get('n_quiet'),
                            'empty_future': w.get('empty_future'),
                            'null_rail': w.get('null_rail')}
                for chan in ('carb', 'insulin'):
                    blk = w.get(chan) or {}
                    cf[chan] = {k: blk.get(k) for k in
                                ('n', 'doses', 'mean_dbg', 'correct_sign_frac', 'mean_peak_dbg',
                                 'slope', 'ref_dose', 'mean_adverse_dbg', 'onset_min',
                                 'peak_min', 'monotone_frac', 'mean_terminal_inversions',
                                 'rescue')}
                rec['counterfactual'][c] = cf
                q = (cell['q2'] or {}).get(c) or {}
                rec['event_dependence'][c] = q
            for c in list(REAL_COHORTS) + ['pooled']:
                t = (cell['time'] or {}).get(c)
                if t:
                    rec['time_of_day'][c] = t
            rec['time_of_day']['_chance'] = ((cell['time'] or {}).get('_meta') or {}).get('chance')
            out[key] = rec
    return out


def build_corpus(M: dict) -> dict:
    """Corpus properties that do not vary with the model, verified invariant here."""
    ref = M['cell'][(SIZES[0], VARIANTS[0])]['logging_stats'] or {}
    invariant = True
    for size in SIZES:
        for v in VARIANTS:
            other = M['cell'][(size, v)]['logging_stats'] or {}
            for c in REAL_COHORTS:
                if (ref.get(c) or {}) != (other.get(c) or {}):
                    invariant = False
    return {
        'note': 'Properties of the evaluation corpora, not of any model.',
        'logging_density_is_model_invariant': invariant,
        'logging_density': {c: (ref.get(c) or {}) for c in REAL_COHORTS},
        'meal_statistics': M['meal_stats'],
    }


def build_index() -> dict:
    return {
        'generated_by': 'models/compare.py',
        'grid': {'sizes': list(SIZES), 'variants': list(VARIANTS),
                 'variant_labels': VARIANT_LABEL,
                 'cohorts': list(COHORTS), 'real_cohorts': list(REAL_COHORTS),
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
        'excluded_inputs': EXCLUDED_INPUTS,
        'files': {
            # index.json names itself; it is written last, so WRITTEN does not yet hold it.
            'data': sorted([p for p in WRITTEN if '/data/' in p.replace(os.sep, '/')]
                           + [os.path.relpath(os.path.join(DATADIR, 'index.json'), HERE)]),
            'figures': sorted(p for p in WRITTEN if '/figures/' in p.replace(os.sep, '/')),
        },
    }


# ---------------------------------------------------------------------------- #
def main() -> None:
    global FIGDIR, DATADIR
    ap = argparse.ArgumentParser(description=(__doc__ or 'model comparison').strip().split('\n')[0])
    ap.add_argument('--out', default=os.path.join(HERE, 'comparison'),
                    help='output directory (default: models/comparison)')
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
    transfer = build_transfer(M)
    uncertainty = build_uncertainty(M)
    behaviour = build_behaviour(M)
    corpus = build_corpus(M)

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
    fig_q2(M)
    fig_conformal(M)
    fig_cqr(M)
    fig_band_geometry(M)
    fig_amplitude(M)
    fig_time_of_day(M)
    fig_whatif_dose(M)
    fig_whatif_quality(M)
    fig_empty_future(M)
    fig_finetune(M)
    fig_transfer_delta(transfer)
    fig_reality_gap(M)
    fig_ranking(rankings)

    print('data ...')
    dump(models, 'models.json')
    dump(long_rows, 'metrics_long.json')
    dump(pooled, 'metrics_pooled.json')
    dump(rankings, 'rankings.json')
    dump(scaling, 'scaling.json')
    dump(transfer, 'transfer.json')
    dump(uncertainty, 'uncertainty.json')
    dump(behaviour, 'behaviour.json')
    dump(corpus, 'corpus.json')
    dump(build_index(), 'index.json')

    print(f'\n{len(WRITTEN)} files under {os.path.relpath(args.out, HERE)}/')


if __name__ == '__main__':
    main()
