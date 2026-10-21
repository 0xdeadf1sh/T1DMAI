#!/usr/bin/env python3
"""Benchmark every zoo model and T1DMAI checkpoint on MetaboNet, blind and given future inputs.

python benchmark.py --cache DIR --template template.parquet --truth test.parquet [--out benchmark]
Blind zeroes insulin and carbohydrate after the origin; given passes them as logged.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TRACKS = ('blind', 'given')
TRACK_COLOR = {'blind': '#2a78d6', 'given': '#eb6834'}
TRACK_MARKER = {'blind': 'o', 'given': 'D'}
HIGHLIGHT_COLORS = ('#2a78d6', '#eb6834', '#1baf7a')
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

LABEL = {
    'persistence': 'Persistence', 'linear_extrapolation': 'Linear extrapolation',
    'ridge_ar': 'AR (ridge)', 'ridge_arx': 'ARX (ridge)', 'kalman': 'Kalman filter',
    'svr': 'SVR', 'gp': 'Sparse GP', 'xgboost': 'XGBoost', 'lstm': 'LSTM',
    'glunet': 'GluNet', 'patchtst': 'PatchTST', 'nhits': 'N-HiTS',
    'chronos_bolt_tiny_zs': 'Chronos-Bolt tiny (zero-shot)',
    'chronos_bolt_mini_zs': 'Chronos-Bolt mini (zero-shot)',
    'chronos_bolt_tiny_ft': 'Chronos-Bolt tiny (fine-tuned)',
    'chronos_bolt_mini_ft': 'Chronos-Bolt mini (fine-tuned)',
}
FAMILY = {
    'persistence': 'Baseline', 'linear_extrapolation': 'Baseline', 'ridge_ar': 'Classical',
    'ridge_arx': 'Classical', 'kalman': 'Classical', 'svr': 'Classical', 'gp': 'Classical',
    'xgboost': 'Classical', 'lstm': 'Deep', 'glunet': 'Deep', 'patchtst': 'Deep',
    'nhits': 'Deep',
}
FAMILY_ORDER = ('Baseline', 'Classical', 'Deep', 'Foundation', 'T1DMAI')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cache', required=True, help='finetune cache built by finetune_data.py')
    p.add_argument('--template', required=True, help='leaderboard template.parquet')
    p.add_argument('--truth', required=True, help='MetaboNet test.parquet')
    p.add_argument('--out', default=os.path.join(ROOT, 'benchmark'))
    p.add_argument('--runs', default=os.path.join(ROOT, 'zoo_runs'), help='zoo fits to reuse')
    p.add_argument('--models', default='all', help="comma list of zoo models, 'all' or 'none'")
    p.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH',
                   help='a T1DMAI checkpoint to add; repeatable')
    p.add_argument('--no-discover', action='store_true',
                   help='skip models/*/checkpoints/t1dmai_best.pt and checkpoints_finetune/')
    p.add_argument('--tracks', default='blind,given')
    p.add_argument('--every', type=int, default=1, metavar='N',
                   help='keep every Nth template row; for quick runs only')
    p.add_argument('--train-windows', type=int, default=500_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--refit', action='store_true', help='refit zoo models even when fitted')
    p.add_argument('--rescore', action='store_true', help='recompute cached predictions')
    p.add_argument('--bootstrap', type=int, default=1000, help='subject-level resamples for CIs')
    p.add_argument('--figures-only', action='store_true', help='redraw from results.json')
    p.add_argument('--figure-width', type=float, default=3.5, help='single column, inches')
    p.add_argument('--text-width', type=float, default=7.16, help='double column, inches')
    p.add_argument('--highlight', default='',
                   help='comma list of entries drawn in the horizon figure; default picks three')
    return p.parse_args()


# --- entries ------------------------------------------------------------------------------

def label_of(name: str) -> str:
    if name.startswith('t1dmai_'):
        return 'T1DMAI ' + name.removeprefix('t1dmai_').replace('_', ' ')
    return LABEL.get(name, name.replace('_', ' '))


def family_of(name: str) -> str:
    if name.startswith('t1dmai_'):
        return 'T1DMAI'
    if name.startswith('chronos'):
        return 'Foundation'
    return FAMILY.get(name, 'Classical')


def entries(args) -> list[dict]:
    from zoo.registry import models
    zoo = sorted(models())
    if args.models == 'none':
        zoo = []
    elif args.models != 'all':
        wanted = args.models.split(',')
        unknown = set(wanted) - set(zoo)
        if unknown:
            sys.exit(f'unknown zoo models: {sorted(unknown)}')
        zoo = wanted
    out = [{'name': k, 'kind': 'zoo', 'key': k} for k in zoo]
    ckpts: dict[str, str] = {}
    if not args.no_discover:
        for path in sorted(glob.glob(os.path.join(ROOT, 'models', '*', 'checkpoints', 't1dmai_best.pt'))):
            ckpts['t1dmai_' + os.path.basename(os.path.dirname(os.path.dirname(path)))] = path
        ft = os.path.join(ROOT, 'checkpoints_finetune', 'finetune_best.pt')
        if os.path.isfile(ft):
            ckpts['t1dmai_finetune'] = ft
    for spec in args.checkpoint:
        name, _, path = spec.partition('=')
        if not path:
            sys.exit(f'--checkpoint {spec!r}: expected NAME=PATH')
        ckpts['t1dmai_' + name.removeprefix('t1dmai_')] = os.path.abspath(path)
    out += [{'name': n, 'kind': 'checkpoint', 'path': p} for n, p in ckpts.items()]
    return out


# --- forecasting ----------------------------------------------------------------------------

def zoo_predictions(entry, tracks, args, cache, tpl, pred_dir) -> dict:
    from export_submission import _write
    from finetune_data import HORIZON_MINUTES
    from zoo.run import fit_or_load, predict_template
    from zoo.registry import models
    reads = models()[entry['key']]().reads_doses
    done = {}
    fitted = {}
    for track in tracks:
        path = os.path.join(pred_dir, f"{entry['name']}__{track}.parquet")
        if os.path.isfile(path) and not args.rescore:
            done[track] = path
            continue
        # A model that ignores doses gives one forecast for both tracks: fit once, copy.
        causal = track == 'blind' and reads
        twin = next((p for t, p in done.items() if not reads), None)
        if twin is not None:
            shutil.copyfile(twin, path)
            done[track] = path
            continue
        if causal not in fitted:
            fitted[causal] = fit_or_load(entry['key'], causal, cache, args.runs,
                                         args.train_windows, args.seed, refit=args.refit)
        model = fitted[causal]
        t = time.time()
        pred, anchored = predict_template(model, cache, tpl, causal)
        _write(tpl.table, pred, tpl.n_rows, HORIZON_MINUTES, path)
        print(f"  {entry['name']} {track}: {anchored}/{tpl.n_rows} anchored "
              f"({time.time() - t:.0f} s)", flush=True)
        done[track] = path
    return done


def checkpoint_predictions(entry, tracks, args, pred_dir) -> dict:
    done = {}
    for track in tracks:
        path = os.path.join(pred_dir, f"{entry['name']}__{track}.parquet")
        if not (os.path.isfile(path) and not args.rescore):
            cmd = [sys.executable, os.path.join(ROOT, 'export_submission.py'),
                   '--checkpoint', entry['path'], '--template', args.template, '--cache', args.cache,
                   '--out', path, '--future-doses', 'zero' if track == 'blind' else 'announced',
                   '--fill', 'ladder', '--every', str(args.every), '--truth', os.devnull + '.absent']
            print(f"  {entry['name']} {track}: export_submission.py", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.isfile(path):
                print(r.stdout[-2000:], r.stderr[-2000:], sep='\n', file=sys.stderr)
                print(f"  {entry['name']} {track}: FAILED, skipped", flush=True)
                continue
        done[track] = path
    return done


# --- scoring --------------------------------------------------------------------------------

def subject_bootstrap_rmse(err2: np.ndarray, subj: np.ndarray, b: int, rng) -> tuple[float, float]:
    """95% percentile CI of RMSE, resampling subjects with replacement."""
    if b <= 0 or len(err2) == 0:
        return float('nan'), float('nan')
    uniq, inv = np.unique(subj, return_inverse=True)
    sse = np.bincount(inv, weights=err2, minlength=len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    draws = rng.integers(0, len(uniq), size=(b, len(uniq)))
    rmse = np.sqrt(sse[draws].sum(axis=1) / np.maximum(cnt[draws].sum(axis=1), 1.0))
    lo, hi = np.percentile(rmse, [2.5, 97.5])
    return float(lo), float(hi)


def score(path: str, truth: np.ndarray, subj: np.ndarray, source: np.ndarray, b: int, seed: int) -> dict:
    import pyarrow.parquet as pq

    from finetune_data import HORIZON_MINUTES
    from score_submission import horizon_metrics
    t = pq.read_table(path)
    rng = np.random.default_rng(seed)
    out: dict = {'overall': {}, 'by_source': {}}
    for h, m in enumerate(HORIZON_MINUTES):
        pred = t.column(f'pred_{m}').to_numpy(zero_copy_only=False).astype(np.float64)
        tr = truth[:, h]
        ok = np.isfinite(pred) & np.isfinite(tr) & (tr > 0.0)
        r = horizon_metrics(pred, tr)
        e = pred[ok] - tr[ok]
        r['mae'] = float(np.mean(np.abs(e))) if len(e) else None
        r['rmse_lo'], r['rmse_hi'] = subject_bootstrap_rmse(e ** 2, subj[ok], b, rng)
        out['overall'][str(m)] = r
        for s in np.unique(source[ok]):
            sel = ok & (source == s)
            out['by_source'].setdefault(str(s), {})[str(m)] = {
                'n': int(sel.sum()), 'rmse': float(np.sqrt(np.mean((pred[sel] - tr[sel]) ** 2)))}
    return out


# --- output ---------------------------------------------------------------------------------

def tex(s: str) -> str:
    return s.replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def order_rows(results: list[dict], track: str, horizon: str = '30') -> list[dict]:
    def key(r):
        v = r['tracks'].get(track, {}).get('overall', {}).get(horizon, {}).get('rmse')
        return (FAMILY_ORDER.index(r['family']), v if v is not None else float('inf'))
    return sorted(results, key=key)


def write_tables(results: list[dict], tracks, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    hs = ('30', '60', '120')

    def cell(r, track, h, metric, fmt, best):
        v = r['tracks'].get(track, {}).get('overall', {}).get(h, {}).get(metric)
        if v is None:
            return '--'
        s = fmt.format(v)
        return rf'\textbf{{{s}}}' if best.get((track, h, metric)) == r['name'] else s

    def best_of(metric, lower):
        best = {}
        for track in tracks:
            for h in hs:
                vals = [(r['tracks'].get(track, {}).get('overall', {}).get(h, {}).get(metric), r['name'])
                        for r in results]
                vals = [v for v in vals if v[0] is not None]
                if vals:
                    best[(track, h, metric)] = (min if lower else max)(vals)[1]
        return best

    rows = order_rows(results, tracks[0])
    best = best_of('rmse', True)
    cols = 'l' + 'r' * (len(hs) * len(tracks))
    head = ' & '.join(rf'\multicolumn{{{len(hs)}}}{{c}}{{{t.capitalize()}}}' for t in tracks)
    rules = ' '.join(rf'\cmidrule(lr){{{2 + i * len(hs)}-{1 + (i + 1) * len(hs)}}}'
                     for i in range(len(tracks)))
    lines = [rf'\begin{{tabular}}{{{cols}}}', r'\toprule',
             rf'Model & {head} \\', rules,
             ' & ' + ' & '.join(f'{h}\\,min' for _ in tracks for h in hs) + r' \\', r'\midrule']
    fam = None
    for r in rows:
        if r['family'] != fam:
            if fam is not None:
                lines.append(r'\addlinespace')
            lines.append(rf'\multicolumn{{{1 + len(hs) * len(tracks)}}}{{l}}{{\emph{{{r["family"]}}}}} \\')
            fam = r['family']
        cells = []
        for track in tracks:
            for h in hs:
                c = cell(r, track, h, 'rmse', '{:.2f}', best)
                lo = r['tracks'].get(track, {}).get('overall', {}).get(h, {}).get('rmse_lo')
                hi = r['tracks'].get(track, {}).get('overall', {}).get(h, {}).get('rmse_hi')
                if lo is not None and np.isfinite(lo):
                    c += rf'\,{{\scriptsize({lo:.1f}--{hi:.1f})}}'
                cells.append(c)
        lines.append(rf'\quad {tex(r["label"])} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(out, 'rmse.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    best_m, best_a = best_of('mard', True), best_of('dts_a', False)
    best_c = {**best_m, **best_a}
    lines = [r'\begin{tabular}{l' + 'rr' * len(tracks) * 2 + '}', r'\toprule',
             'Model & ' + ' & '.join(rf'\multicolumn{{4}}{{c}}{{{t.capitalize()}}}' for t in tracks) + r' \\',
             ' '.join(rf'\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}' for i in range(len(tracks))),
             ' & ' + ' & '.join(r'MARD$_{30}$ & MARD$_{60}$ & DTS-A$_{30}$ & DTS-A$_{60}$'
                                for _ in tracks) + r' \\', r'\midrule']
    for r in rows:
        cells = []
        for track in tracks:
            cells += [cell(r, track, h, 'mard', '{:.1f}', best_c) for h in ('30', '60')]
            cells += [cell(r, track, h, 'dts_a', '{:.1f}', best_c) for h in ('30', '60')]
        lines.append(f'{tex(r["label"])} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(out, 'clinical.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_csv(results: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'label', 'family', 'track', 'horizon_min', 'n', 'rmse', 'rmse_lo',
                    'rmse_hi', 'mae', 'mard', 'dts_a'])
        for r in results:
            for track, s in r['tracks'].items():
                for h, m in s['overall'].items():
                    w.writerow([r['name'], r['label'], r['family'], track, h, m['n'], m['rmse'],
                                m['rmse_lo'], m['rmse_hi'], m['mae'], m['mard'], m['dts_a']])


def figure_style(size: float) -> None:
    import matplotlib
    matplotlib.use('pgf')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
        'font.family': 'serif', 'font.size': size, 'axes.labelsize': size,
        'axes.titlesize': size, 'legend.fontsize': size - 1, 'xtick.labelsize': size - 1,
        'ytick.labelsize': size - 1, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK,
        'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'lines.linewidth': 1.4, 'legend.frameon': False, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pgf.preamble': r'\usepackage[T1]{fontenc}\usepackage{lmodern}',
    })


def save(fig, out: str, name: str) -> None:
    for ext in ('pgf', 'pdf'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'))


def write_figures(results: list[dict], tracks, args, out: str) -> None:
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    figure_style(8)

    def val(r, track, h, k='rmse'):
        v = r['tracks'].get(track, {}).get('overall', {}).get(h, {}).get(k)
        return np.nan if v is None else v

    # Forest plot: RMSE with subject-bootstrap CIs, blind and given, at 30 and 60 min.
    rows = sorted(results, key=lambda r: val(r, tracks[0], '30'), reverse=True)
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(args.text_width, 0.16 * len(rows) + 0.8), sharey=True)
    for ax, h in zip(axes, ('30', '60')):
        for i, track in enumerate(tracks):
            off = (i - (len(tracks) - 1) / 2) * 0.22
            v = np.array([val(r, track, h) for r in rows])
            lo = np.array([val(r, track, h, 'rmse_lo') for r in rows])
            hi = np.array([val(r, track, h, 'rmse_hi') for r in rows])
            err = np.vstack([v - lo, hi - v])
            err = np.where(np.isfinite(err), err, 0.0)
            ax.errorbar(v, y + off, xerr=err, fmt=TRACK_MARKER[track], ms=3.2, lw=0.8,
                        color=TRACK_COLOR[track], capsize=0, label=track.capitalize())
        ax.set_xlabel(f'RMSE at {h} min (mg/dL)')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(y, [tex(r['label']) for r in rows])
    axes[0].tick_params(axis='y', length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.6, 1.0), ncol=len(tracks),
               handletextpad=0.3, columnspacing=1.2)
    save(fig, out, 'rmse_forest')
    plt.close(fig)

    # RMSE versus horizon for three highlighted entries; solid blind, dashed given.
    from finetune_data import HORIZON_MINUTES
    hl = [n for n in args.highlight.split(',') if n] or default_highlight(results, tracks[0])
    by_name = {r['name']: r for r in results}
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(args.figure_width, 2.3))
    ends = []
    for name, color in zip(hl[:len(HIGHLIGHT_COLORS)], HIGHLIGHT_COLORS):
        r = by_name.get(name)
        if r is None:
            continue
        for track in tracks:
            v = [val(r, track, str(m)) for m in HORIZON_MINUTES]
            ax.plot(HORIZON_MINUTES, v, color=color, ls='-' if track == 'blind' else '--',
                    marker=TRACK_MARKER[track], ms=3)
        ends.append([val(r, tracks[0], str(HORIZON_MINUTES[-1])), tex(r['label'])])
    lo, hi = ax.get_ylim()
    gap = 0.07 * (hi - lo)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for yv, text in ends:
        ax.annotate(text, (HORIZON_MINUTES[-1], yv), xytext=(5, 0), textcoords='offset points',
                    va='center', color=INK, fontsize=7, annotation_clip=False)
    ax.set_xticks(HORIZON_MINUTES)
    ax.set_xlabel('Prediction horizon (min)')
    ax.set_ylabel('RMSE (mg/dL)')
    style = [Line2D([], [], color=INK_2, ls='-' if t == 'blind' else '--', marker=TRACK_MARKER[t],
                    ms=3, label=t.capitalize()) for t in tracks]
    ax.legend(handles=style, loc='upper left')
    save(fig, out, 'rmse_horizon')
    plt.close(fig)

    # What the future inputs are worth: blind minus given RMSE, dose-reading entries only.
    if set(TRACKS) <= set(tracks):
        gap = [(r, [val(r, 'blind', h) - val(r, 'given', h) for h in ('30', '60', '120')])
               for r in results if r.get('reads_doses', True)]
        gap = [g for g in gap if np.isfinite(g[1][1])]
        gap.sort(key=lambda g: g[1][1])
        if gap:
            fig, axes = plt.subplots(1, 3, figsize=(args.text_width, 0.16 * len(gap) + 0.7),
                                     sharey=True)
            yy = np.arange(len(gap))
            for ax, (j, h) in zip(axes, enumerate(('30', '60', '120'))):
                ax.barh(yy, [g[1][j] for g in gap], height=0.6, color=TRACK_COLOR['given'])
                ax.axvline(0, color=INK_2, lw=0.6)
                ax.set_xlabel(rf'$\Delta$RMSE at {h} min (mg/dL)')
                ax.grid(axis='y', visible=False)
            axes[0].set_yticks(yy, [tex(g[0]['label']) for g in gap])
            axes[0].tick_params(axis='y', length=0)
            save(fig, out, 'future_input_gain')
            plt.close(fig)

    # Per-dataset RMSE at 30 min, blind track: sequential single hue, values printed.
    track = tracks[0]
    sources = sorted({s for r in results for s in r['tracks'].get(track, {}).get('by_source', {})})
    rows = order_rows(results, track)
    if sources:
        grid = np.array([[r['tracks'].get(track, {}).get('by_source', {}).get(s, {}).get('30', {}).get('rmse', np.nan)
                          for s in sources] for r in rows])
        fig, ax = plt.subplots(figsize=(args.text_width, 0.2 * len(rows) + 1.0))
        # pcolormesh, not imshow: the pgf backend writes an image as a side-car PNG.
        from matplotlib.colors import BoundaryNorm
        edges = np.linspace(np.nanmin(grid), np.nanmax(grid), 9)
        norm = BoundaryNorm(edges, plt.get_cmap('Blues').N)
        im = ax.pcolormesh(np.ma.masked_invalid(grid), cmap='Blues', norm=norm, edgecolors='white',
                           linewidth=1.0)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(sources)) + 0.5, [tex(s) for s in sources], rotation=45,
                      ha='right')
        ax.set_yticks(np.arange(len(rows)) + 0.5, [tex(r['label']) for r in rows])
        ax.grid(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        hi = np.nanmax(grid)
        for (i, j), v in np.ndenumerate(grid):
            if np.isfinite(v):
                ax.text(j + 0.5, i + 0.5, f'{v:.0f}', ha='center', va='center', fontsize=5.5,
                        color='white' if v > 0.6 * hi else INK)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label='RMSE at 30 min (mg/dL)',
                     format='%.0f')
        save(fig, out, f'rmse_by_dataset_{track}')
        plt.close(fig)


def default_highlight(results: list[dict], track: str) -> list[str]:
    def best(fam):
        c = [r for r in results if r['family'] in fam and
             r['tracks'].get(track, {}).get('overall', {}).get('30', {}).get('rmse') is not None]
        return min(c, key=lambda r: r['tracks'][track]['overall']['30']['rmse'])['name'] if c else None
    return [n for n in (best({'T1DMAI'}), best({'Classical', 'Deep', 'Foundation'}),
                        'persistence') if n]


# --- main ----------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    tracks = tuple(t for t in args.tracks.split(',') if t)
    if not set(tracks) <= set(TRACKS):
        sys.exit(f'--tracks must be from {TRACKS}')
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, 'results.json')

    if not args.figures_only:
        from finetune_data import FinetuneCache
        from score_submission import truth_at_horizons
        from zoo.run import Template
        cache = FinetuneCache(args.cache)
        tpl = Template(args.template, args.every, cache)
        pred_dir = os.path.join(args.out, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        print(f'template: {tpl.n_rows} rows (every {args.every})', flush=True)
        truth = truth_at_horizons(tpl.table, args.truth)
        src = np.asarray(tpl.table.column('source_file').to_pylist(), dtype=object).astype(str)
        sid = np.asarray([str(x) for x in tpl.table.column('id').to_pylist()], dtype=object).astype(str)
        subj = np.char.add(np.char.add(src, '\x1f'), sid)

        results = []
        for e in entries(args):
            print(f"== {e['name']}", flush=True)
            try:
                files = (zoo_predictions(e, tracks, args, cache, tpl, pred_dir) if e['kind'] == 'zoo'
                         else checkpoint_predictions(e, tracks, args, pred_dir))
            except Exception as exc:  # one broken model must not sink the run
                print(f"  {e['name']}: FAILED ({exc!r}), skipped", flush=True)
                continue
            reads = True
            if e['kind'] == 'zoo':
                from zoo.registry import models
                reads = models()[e['key']]().reads_doses
            if not files:
                continue
            results.append({
                'name': e['name'], 'label': label_of(e['name']), 'family': family_of(e['name']),
                'kind': e['kind'], 'reads_doses': reads,
                'tracks': {t: score(p, truth, subj, src, args.bootstrap, args.seed)
                           for t, p in files.items()},
            })
        meta = {'template': os.path.abspath(args.template), 'truth': os.path.abspath(args.truth),
                'every': args.every, 'rows': tpl.n_rows, 'bootstrap': args.bootstrap,
                'tracks': list(tracks)}
        with open(results_path, 'w') as f:
            json.dump({'meta': meta, 'results': results}, f, indent=1)
        write_csv(results, os.path.join(args.out, 'results.csv'))
    else:
        with open(results_path) as f:
            results = json.load(f)['results']

    write_tables(results, tracks, os.path.join(args.out, 'tables'))
    write_figures(results, tracks, args, os.path.join(args.out, 'figures'))
    print(f'wrote {args.out}/results.json, tables/*.tex, figures/*.pgf|pdf', flush=True)


if __name__ == '__main__':
    main()
