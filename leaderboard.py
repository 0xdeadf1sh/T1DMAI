#!/usr/bin/env python3
"""Score forecast files with zoo/paper_metrics.py and render the README's tables and figures.

python leaderboard.py MANIFEST.json [--out results] [--figures screenshots]
MANIFEST: {"tracks": {NAME: {"title", "truth", "time_gain", "entries": [{"key", "label",
"family", "path"}]}}}; every entry of a track fills the same template rows."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HORIZONS = (30, 60, 90, 120)
FAMILIES = ('Baseline', 'Classical', 'Deep', 'Foundation', 'T1DMAI')
THEMES = {
    'dark': {'bg': '#141414', 'ink': '#ececec', 'ink2': '#b4b4b4', 'grid': '#333333',
             'fam': {'Baseline': '#8f8e89', 'Classical': '#3987e5', 'Deep': '#199e70',
                     'Foundation': '#9085e9', 'T1DMAI': '#d95926'},
             'cat': ('#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'),
             'seq': ('#184f95', '#cde2fb'), 'sig': {'best': '#86b6ef', 'tie': '#256abf',
                                                    'worse': '#333333'}},
}
STYLE = {
    'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage[T1]{fontenc}',
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'lines.linewidth': 1.2, 'lines.markersize': 4.5,
    'legend.frameon': False, 'savefig.dpi': 200,
}
MARKERS = ('o', 's', '^', 'D', 'v')


def tex(s: str) -> str:
    """Escape LaTeX specials in a plain label."""
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                 ('#', r'\#'), ('_', r'\_'), ('<', r'$<$'), ('>', r'$>$'), ('·', r'$\cdot$')):
        s = s.replace(a, b)
    return s


def _model_ticks(ax, rows, th, axis='y'):
    """Model names as tick labels; T1DMAI rows bold in the family color."""
    names = [rf"\textbf{{{tex(r['label'])}}}" if r['family'] == 'T1DMAI' else tex(r['label'])
             for r in rows]
    (ax.set_yticks if axis == 'y' else ax.set_xticks)(range(len(rows)), names)
    for t, r in zip(ax.get_yticklabels() if axis == 'y' else ax.get_xticklabels(), rows):
        t.set_color(th['fam']['T1DMAI'] if r['family'] == 'T1DMAI' else th['ink'])


def load_track(spec: dict) -> dict:
    """Predictions ``(models, rows, horizons)``, truth ``(rows, horizons)``, origin reading."""
    import pyarrow.parquet as pq

    from score_submission import _epoch_s, _nearest, _subject_keys, load_truth, truth_at_horizons
    first = pq.read_table(spec['entries'][0]['path'], columns=['source_file', 'id', 'date'])
    keys, ts = _subject_keys(first), _epoch_s(first.column('date'))
    truth = truth_at_horizons(first, spec['truth'])
    cgm = load_truth(spec['truth'], set(keys.tolist()))
    origin = np.full(len(ts), np.nan)
    for k, (tts, v) in cgm.items():
        sel = keys == k
        origin[sel] = _nearest(tts, v, ts[sel])
    preds = []
    for e in spec['entries']:
        t = pq.read_table(e['path'])
        if len(t) != len(ts) or not np.array_equal(_epoch_s(t.column('date')), ts):
            raise SystemExit(f"{e['path']}: rows differ from {spec['entries'][0]['path']}")
        preds.append(np.stack([t.column(f'pred_{m}').to_numpy(zero_copy_only=False)
                               .astype(np.float64) for m in HORIZONS], axis=1))
    return {'pred': np.stack(preds), 'truth': truth, 'origin': origin, 'ts': ts, 'keys': keys,
            'cgm': cgm}


def score_track(spec: dict, d: dict) -> list[dict]:
    from zoo import paper_metrics as pm
    rows = []
    for i, e in enumerate(spec['entries']):
        r = {'key': e['key'], 'label': e['label'], 'family': e['family'], 'h': {}}
        for j, m in enumerate(HORIZONS):
            r['h'][m] = pm.point_metrics(d['truth'][:, j], d['pred'][i, :, j], d['origin'], m)
            if spec.get('time_gain'):
                tg = [pm.time_gain(d['ts'][d['keys'] == k], d['pred'][i, d['keys'] == k, j],
                                   tts, v, m) for k, (tts, v) in d['cgm'].items()]
                r['h'][m]['time_gain'] = tg[0] if len(tg) == 1 else None
        rows.append(r)
    alpha = 0.05 / max(len(rows) - 1, 1)
    for j, m in enumerate(HORIZONS):
        best = min(range(len(rows)), key=lambda i: rows[i]['h'][m].get('rmse', np.inf))
        for i, r in enumerate(rows):
            if i == best:
                r['h'][m] |= {'vs_best': 'best', 'p_vs_best': None, 'dmae_vs_best': 0.0}
                continue
            dm, p = pm.wilcoxon_abs_error(d['truth'][:, j], d['pred'][i, :, j],
                                          d['pred'][best, :, j])
            r['h'][m] |= {'vs_best': 'tie' if p >= alpha or dm <= 0 else 'worse',
                          'p_vs_best': p, 'dmae_vs_best': dm}
    return rows


def _fmt(v, nd=1) -> str:
    return '—' if v is None else f'{v:.{nd}f}'


def tables(name: str, spec: dict, rows: list[dict]) -> str:
    """Markdown tables: accuracy, clinical, hypo; rows sorted by RMSE at 60 min."""
    rows = sorted(rows, key=lambda r: r['h'][60]['rmse'])
    star = lambda r, m: ' ◆' if r['h'][m]['vs_best'] in ('best', 'tie') else ''
    out = [f'#### Accuracy — RMSE / MARD / bias (mg/dL, %)\n',
           '| Model | Family | RMSE 30 | RMSE 60 | RMSE 90 | RMSE 120 | gRMSE 60 | MARD 30 | '
           'MARD 60 | MARD 120 | ME 60 | MRE 60 % |' + (' TG 30 | TG 60 |' if spec.get(
               'time_gain') else ''),
           '| --- | --- |' + ' ---: |' * (10 + (2 if spec.get('time_gain') else 0))]
    for r in rows:
        h = r['h']
        cells = [f"{_fmt(h[m]['rmse'])}{star(r, m)}" for m in HORIZONS]
        cells += [_fmt(h[60]['grmse']), _fmt(h[30]['mard']), _fmt(h[60]['mard']),
                  _fmt(h[120]['mard']), f"{h[60]['me']:+.1f}", f"{h[60]['mre']:+.1f}"]
        if spec.get('time_gain'):
            cells += [_fmt(h[30].get('time_gain'), 0), _fmt(h[60].get('time_gain'), 0)]
        out.append(f"| {r['label']} | {r['family']} | " + ' | '.join(cells) + ' |')
    out += ['', '#### Clinical — DTS / Clarke / CG-EGA (% of points)\n',
            '| Model | DTS A 30 | DTS A 60 | DTS A 120 | DTS A+B 120 | Clarke A 60 | '
            'Clarke A+B 120 | CG-EGA hypo AP 30 | hypo EP 60 | eu EP 60 | hyper EP 60 |',
            '| --- |' + ' ---: |' * 10]
    for r in rows:
        h = r['h']
        out.append(f"| {r['label']} | " + ' | '.join([
            _fmt(h[30]['dts_a']), _fmt(h[60]['dts_a']), _fmt(h[120]['dts_a']),
            _fmt(h[120]['dts_ab']), _fmt(h[60]['clarke_a']), _fmt(h[120]['clarke_ab']),
            _fmt(h[30].get('cgega_ap_hypo')), _fmt(h[60].get('cgega_ep_hypo')),
            _fmt(h[60].get('cgega_ep_eu')), _fmt(h[60].get('cgega_ep_hyper'))]) + ' |')
    out += ['', '#### Hypoglycaemia (< 70 mg/dL at the horizon)\n',
            '| Model | Sens 30 | Sens 60 | Spec 60 | PPV 60 | MCC 30 | MCC 60 | '
            'RMSE<70 30 | ME<70 30 | ME<70 60 |',
            '| --- |' + ' ---: |' * 9]
    for r in rows:
        h = r['h']
        out.append(f"| {r['label']} | " + ' | '.join([
            _fmt(h[30]['hypo_sens'], 2), _fmt(h[60]['hypo_sens'], 2), _fmt(h[60]['hypo_spec'], 3),
            _fmt(h[60]['hypo_ppv'], 2), _fmt(h[30]['hypo_mcc'], 2), _fmt(h[60]['hypo_mcc'], 2),
            _fmt(h[30]['rmse_hypo']), f"{h[30]['me_hypo']:+.1f}" if h[30]['me_hypo'] is not None
            else '—', f"{h[60]['me_hypo']:+.1f}" if h[60]['me_hypo'] is not None else '—'])
            + ' |')
    n = rows[0]['h'][60]['n']
    out.append(f"\nn = {n} forecasts at 60 min; ◆ = best or not significantly worse "
               f"(Wilcoxon on |error|, Bonferroni α = 0.05/{len(rows) - 1}).")
    return '\n'.join(out) + '\n'


def _axes(fig, ax, th):
    fig.patch.set_facecolor(th['bg'])
    ax.set_facecolor(th['bg'])
    for s in ax.spines.values():
        s.set_color(th['ink2'])
    ax.tick_params(which='both', colors=th['ink2'], labelcolor=th['ink'])
    ax.grid(True, axis='x', color=th['grid'], lw=0.5, ls=':')
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(th['ink'])
    ax.yaxis.label.set_color(th['ink'])
    ax.title.set_color(th['ink'])


def _title(ax, s, **kw):
    ax.set_title(tex(s), loc='left', color=THEMES['dark']['ink'], **kw)


def _legend(obj, th, **kw):
    leg = obj.legend(**kw)
    for t in leg.get_texts():
        t.set_color(th['ink'])
    return leg


def _save(fig, path_stem: str, theme: str):
    fig.savefig(f'{path_stem}.png', bbox_inches='tight', facecolor=fig.get_facecolor())


def fig_leaderboard(rows, title, stem, metric='rmse', xlabel='RMSE (mg/dL)', reverse=False):
    """Dot plot, one row per model, markers at 30/60/120 min, sorted by the 60-min value."""
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r['h'][60][metric], reverse=reverse)
    for theme, th in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(rows) + 1.2))
        _axes(fig, ax, th)
        y = np.arange(len(rows))
        vals = np.array([[r['h'][m][metric] for m in (30, 60, 120)] for r in rows])
        cap_lo, cap_hi = np.sort(vals.ravel())[[1, -2]]
        span = cap_hi - cap_lo
        edge = (cap_lo - 0.06 * span, cap_hi + 0.06 * span)
        shown = np.clip(vals, *edge)
        for k, m in enumerate((30, 60, 120)):
            ax.scatter(shown[:, k], y, s=22, marker=MARKERS[k], color=th['cat'][k],
                       edgecolor=th['bg'], linewidth=0.6, zorder=3, label=f'{m} min')
        for i in range(len(rows)):
            ax.plot([shown[i].min(), shown[i].max()], [i, i], color=th['ink2'], lw=0.6,
                    zorder=2)
            for k in range(3):
                if vals[i, k] != shown[i, k]:
                    ax.annotate(f'{vals[i, k]:.0f}', (shown[i, k], i), xytext=(0, 5),
                                textcoords='offset points', ha='center', fontsize=7,
                                color=th['ink2'])
        ax.set_xlim(edge[0] - 0.02 * span, edge[1] + 0.02 * span)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        _model_ticks(ax, rows, th)
        ax.set_xlabel(tex(xlabel))
        _legend(ax, th, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncols=3,
                borderaxespad=0.2, handletextpad=0.2)
        _title(ax, title, pad=20)
        _save(fig, stem, theme)
        plt.close(fig)


def auto_keys(rows: list[dict]) -> list[str]:
    """Best of each of Classical, Deep, Foundation, T1DMAI by 60-min RMSE, then persistence."""
    out = []
    for fam in ('Classical', 'Deep', 'Foundation', 'T1DMAI'):
        c = [r for r in rows if r['family'] == fam]
        if c:
            out.append(min(c, key=lambda r: r['h'][60]['rmse'])['key'])
    return out + ['persistence']


def fig_horizon(rows, keys, metric, ylabel, title, stem, ylim=None):
    """Lines over the four horizons for up to five named models."""
    import matplotlib.pyplot as plt
    if keys == 'auto':
        keys = auto_keys(rows)
    sel = [next(r for r in rows if r['key'] == k) for k in keys]
    for theme, th in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        _axes(fig, ax, th)
        ax.grid(True, axis='y', color=th['grid'], lw=0.5, ls=':')
        for k, r in enumerate(sel):
            v = [r['h'][m][metric] for m in HORIZONS]
            ax.plot(HORIZONS, v, color=th['fam'][r['family']], marker=MARKERS[k],
                    markeredgecolor=th['bg'], markeredgewidth=0.6, label=tex(r['label']))
        ax.set_xticks(HORIZONS)
        ax.set_xlabel('Horizon (min)')
        ax.set_ylabel(tex(ylabel[:1].upper() + ylabel[1:]))
        if ylim:
            ax.set_ylim(*ylim)
        _title(ax, title)
        _legend(ax, th, handlelength=2.2)
        _save(fig, stem, theme)
        plt.close(fig)


def fig_heatmap(rows, metric, title, stem, fmt='{:.0f}', higher_better=True):
    """Models x horizons, one-hue sequential, values printed."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    rows = sorted(rows, key=lambda r: r['h'][60][metric] or 0, reverse=higher_better)
    v = np.array([[np.nan if r['h'][m].get(metric) is None else r['h'][m][metric]
                   for m in HORIZONS] for r in rows])
    for theme, th in THEMES.items():
        cmap = LinearSegmentedColormap.from_list('seq', th['seq'])
        fig, ax = plt.subplots(figsize=(5.6, 0.26 * len(rows) + 1.1))
        _axes(fig, ax, th)
        ax.grid(False)
        vv = v if higher_better else -v
        ax.imshow(vv, aspect='auto', cmap=cmap)
        lo, hi = np.nanmin(vv), np.nanmax(vv)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isfinite(v[i, j]):
                    dark_cell = (vv[i, j] - lo) / max(hi - lo, 1e-9) > 0.55
                    ink = ('#ffffff' if theme == 'light' else '#0b0b0b') if dark_cell else th['ink']
                    s = fmt.format(v[i, j])
                    ax.text(j, i, s.lstrip('-') if float(s) == 0 else s, ha='center', va='center', fontsize=8,
                            color=ink)
        ax.set_xticks(range(len(HORIZONS)), [f'{m} min' for m in HORIZONS])
        _model_ticks(ax, rows, th)
        ax.tick_params(length=0, top=False, right=False, labeltop=True, labelbottom=False)
        _title(ax, title, pad=18)
        _save(fig, stem, theme)
        plt.close(fig)


def fig_significance(rows, title, stem):
    """Models x horizons: best, not significantly worse, significantly worse."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    rows = sorted(rows, key=lambda r: r['h'][60]['rmse'])
    code = {'best': 0, 'tie': 1, 'worse': 2}
    v = np.array([[code[r['h'][m]['vs_best']] for m in HORIZONS] for r in rows])
    for theme, th in THEMES.items():
        s = th['sig']
        fig, ax = plt.subplots(figsize=(5.2, 0.26 * len(rows) + 1.4))
        _axes(fig, ax, th)
        ax.grid(False)
        ax.imshow(v, aspect='auto', cmap=ListedColormap([s['best'], s['tie'], s['worse']]),
                  vmin=0, vmax=2)
        for i in range(v.shape[0] + 1):
            ax.axhline(i - 0.5, color=th['bg'], lw=2)
        for j in range(v.shape[1] + 1):
            ax.axvline(j - 0.5, color=th['bg'], lw=2)
        ax.set_xticks(range(len(HORIZONS)), [f'{m} min' for m in HORIZONS])
        _model_ticks(ax, rows, th)
        ax.tick_params(length=0, top=False, right=False, labeltop=True, labelbottom=False)
        _legend(ax, th, handles=[Patch(color=s['best'], label='Lowest RMSE'),
                                 Patch(color=s['tie'], label='Not significantly worse'),
                                 Patch(color=s['worse'], label='Significantly worse')],
                loc='upper center', bbox_to_anchor=(0.5, -0.01), ncols=3,
                handlelength=1.2, columnspacing=1.0)
        _title(ax, title, pad=18)
        _save(fig, stem, theme)
        plt.close(fig)


def fig_paths(rows, groups, paths, ref_key, title, stem):
    """Our architectures x training paths, RMSE at 30/60/120; dashed line = best zoo model."""
    import matplotlib.pyplot as plt
    by = {r['key']: r for r in rows}
    ref = by[ref_key]
    for theme, th in THEMES.items():
        fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=False)
        for ax, m in zip(axs, (30, 60, 120)):
            _axes(fig, ax, th)
            ax.grid(True, axis='y', color=th['grid'], lw=0.5, ls=':')
            ax.grid(False, axis='x')
            w = 0.8 / len(paths)
            for k, (p, plabel) in enumerate(paths):
                x = np.arange(len(groups)) + (k - (len(paths) - 1) / 2) * w
                v = [by[f'{g}_{p}']['h'][m]['rmse'] if f'{g}_{p}' in by else np.nan
                     for g, _ in groups]
                ax.scatter(x, v, s=32, marker=MARKERS[k], color=th['cat'][k], label=tex(plabel),
                           edgecolor=th['bg'], linewidth=0.6, zorder=3)
            ax.axhline(ref['h'][m]['rmse'], color=th['ink2'], lw=0.8, ls='--')
            ax.text(-0.45, ref['h'][m]['rmse'], tex(ref['label']), fontsize=7,
                    color=th['ink2'], ha='left', va='bottom')
            ax.set_xticks(range(len(groups)), [tex(gl) for _, gl in groups])
            ax.tick_params(axis='x', top=False)
            ax.set_xlim(-0.5, len(groups) - 0.5)
            ax.set_title(f'{m} min')
            lo = np.nanmin([by[f'{g}_{p}']['h'][m]['rmse'] for g, _ in groups for p, _ in paths
                            if f'{g}_{p}' in by] + [ref['h'][m]['rmse']])
            ax.set_ylim(lo * 0.85, None)
        axs[0].set_ylabel('RMSE (mg/dL)')
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        h, lab = axs[0].get_legend_handles_labels()
        _legend(fig, th, handles=h, labels=lab, loc='upper right', bbox_to_anchor=(0.99, 0.97),
                ncols=len(paths), handletextpad=0.2)
        fig.suptitle(tex(title), color=th['ink'], fontsize=10, x=0.01, ha='left', y=0.97)
        _save(fig, stem, theme)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    p.add_argument('manifest')
    p.add_argument('--out', default='results')
    p.add_argument('--figures', default='screenshots')
    p.add_argument('--figures-only', action='store_true', help='reuse OUT/<track>.json')
    a = p.parse_args()
    with open(a.manifest) as f:
        man = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.figures, exist_ok=True)
    scored = {}
    for name, spec in man['tracks'].items():
        path = os.path.join(a.out, f'{name}.json')
        if a.figures_only:
            with open(path) as f:
                rows = json.load(f)
            for r in rows:
                r['h'] = {int(k): v for k, v in r['h'].items()}
        else:
            rows = score_track(spec, load_track(spec))
            with open(path, 'w') as f:
                json.dump(rows, f, indent=1)
            with open(os.path.join(a.out, f'{name}.md'), 'w') as f:
                f.write(tables(name, spec, rows))
            print(f'{name}: {len(rows)} models scored')
        scored[name] = rows
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update(STYLE)
    kinds = {'leaderboard': fig_leaderboard, 'horizon': fig_horizon, 'heatmap': fig_heatmap,
             'significance': fig_significance, 'paths': fig_paths}
    for fig in man.get('figures', []):
        fig = dict(fig)
        kind, track, stem = fig.pop('kind'), fig.pop('track'), fig.pop('stem')
        kinds[kind](scored[track], stem=os.path.join(a.figures, stem), **fig)
        print(f'figure {stem}')


if __name__ == '__main__':
    main()
