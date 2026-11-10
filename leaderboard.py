#!/usr/bin/env python3
"""Score forecast files with zoo/paper_metrics.py and render the README's tables and figures.

python leaderboard.py MANIFEST.json [--out results] [--figures screenshots]
MANIFEST: {"cost"?, "figures", "tracks": {NAME: {"title", "truth", "time_gain", "cache"?,
"entries": [{"key", "label", "family", "path", "bands"? (a --quantiles export)}]}}}."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HORIZONS = (30, 60, 90, 120)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
MEAL_MIN_G, MEAL_WINDOW_MIN = 5.0, 120
NIGHT_HOURS = range(0, 6)
N_BOOT = 2000
MIN_SUBJECT_FORECASTS = 100
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
                 ('#', r'\#'), ('_', r'\_'), ('<', r'$<$'), ('>', r'$>$')):
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
    preds, bands = [], {}
    for e in spec['entries']:
        t = pq.read_table(e['path'])
        if len(t) != len(ts) or not np.array_equal(_epoch_s(t.column('date')), ts):
            raise SystemExit(f"{e['path']}: rows differ from {spec['entries'][0]['path']}")
        preds.append(np.stack([t.column(f'pred_{m}').to_numpy(zero_copy_only=False)
                               .astype(np.float64) for m in HORIZONS], axis=1))
        if e.get('bands'):
            b = pq.read_table(e['bands'])
            if not np.array_equal(_epoch_s(b.column('date')), ts):
                raise SystemExit(f"{e['bands']}: rows differ from {e['path']}")
            bands[e['key']] = np.stack([np.stack(
                [b.column(f'q{round(100 * q):02d}_{m}').to_numpy(zero_copy_only=False)
                 .astype(np.float64) for q in QUANTILES], axis=1) for m in HORIZONS], axis=1)
    src = first.column('source_file').to_numpy(zero_copy_only=False).astype(str)
    return {'pred': np.stack(preds), 'truth': truth, 'origin': origin, 'ts': ts, 'keys': keys,
            'cgm': cgm, 'src': src, 'bands': bands, 'hour': (ts // 3600) % 24,
            'day': ts // 86400, 'meal': _meal_flag(spec.get('cache'), first, ts)}


def _meal_flag(cache_dir, table, ts) -> np.ndarray | None:
    """True where a carb entry (>= MEAL_MIN_G) sits in the MEAL_WINDOW_MIN before the origin."""
    if not cache_dir:
        return None
    from finetune_data import FinetuneCache
    cache = FinetuneCache(cache_dir)
    sid = table.column('id').to_numpy(zero_copy_only=False).astype(str)
    src = table.column('source_file').to_numpy(zero_copy_only=False).astype(str)
    by_key = {(r['source'], r['sid']): r for r in cache.subjects}
    out = np.zeros(len(ts), bool)
    w = MEAL_WINDOW_MIN // 5
    for key in set(zip(src.tolist(), sid.tolist())):
        rec = by_key.get(key)
        if rec is None:
            continue
        sel = np.flatnonzero((src == key[0]) & (sid == key[1]))
        n = int(rec['n'])
        carb = np.nan_to_num(np.asarray(cache.channels(rec, 0, n)['carb_g'], np.float64))
        c = np.concatenate([[0.0], np.cumsum(carb >= MEAL_MIN_G)])
        k = np.clip(np.rint((ts[sel] - int(rec['t0'])) / 300.0).astype(np.int64), 0, n - 1)
        out[sel] = c[k + 1] - c[np.maximum(k + 1 - w, 0)] > 0
    return out


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


def dataset_stats(spec: dict, d: dict) -> list[dict]:
    """Per source and overall: subjects, days, CGM coverage, test-CGM glucose distribution."""
    groups: dict[str, list] = {}
    src_of = dict(zip(d['keys'].tolist(), d['src'].tolist()))
    for k, (tts, v) in d['cgm'].items():
        groups.setdefault(src_of[k], []).append((tts, v))
    rows = []
    for name, parts in sorted(groups.items()) + [('All', [p for g in groups.values() for p in g])]:
        v = np.concatenate([p[1] for p in parts])
        span = sum((p[0][-1] - p[0][0]) / 300 + 1 for p in parts if len(p[0]))
        g = v[np.isfinite(v)]
        rows.append({'source': name, 'subjects': len(parts), 'days': span / 288,
                     'forecasts': int(np.isin(d['src'], list(groups) if name == 'All'
                                              else [name]).sum()),
                     'coverage': 100 * len(g) / max(span, 1), 'mean': g.mean(),
                     'cv': 100 * g.std() / g.mean(), 'tbr54': 100 * (g < 54).mean(),
                     'tbr70': 100 * (g < 70).mean(),
                     'tir': 100 * ((g >= 70) & (g <= 180)).mean(),
                     'tar180': 100 * (g > 180).mean(), 'tar250': 100 * (g > 250).mean()})
    rows[-1]['hist'] = np.histogram(g, bins=np.arange(40, 405, 5))[0].tolist()
    return rows


def _clusters(d: dict) -> np.ndarray:
    """Resampling unit for the bootstrap: the subject, or the day when there is one subject."""
    by = d['keys'] if len(np.unique(d['keys'])) > 1 else d['day']
    return np.unique(by, return_inverse=True)[1]


def _rmse_me(t, p, sel) -> dict:
    ok = sel & np.isfinite(t) & np.isfinite(p)
    e = p[ok] - t[ok]
    return {'n': int(ok.sum()), 'rmse': float(np.sqrt(np.mean(e ** 2))) if ok.any() else None,
            'me': float(e.mean()) if ok.any() else None}


def extras(spec: dict, d: dict) -> dict:
    """Bootstrap CIs, strata, per-subject/per-source/per-day spread, band calibration."""
    rng = np.random.default_rng(0)
    cl = _clusters(d)
    n_cl = cl.max() + 1
    boot = rng.multinomial(n_cl, np.full(n_cl, 1 / n_cl), size=N_BOOT).astype(np.float64)
    t = d['truth']
    strata = {'hypo': lambda tj: tj < 70, 'eu': lambda tj: (tj >= 70) & (tj <= 180),
              'hyper': lambda tj: tj > 180}
    night = np.isin(d['hour'], list(NIGHT_HOURS))
    per_subject = len(np.unique(d['keys'])) > 1
    subj_n = dict(zip(*np.unique(d['keys'], return_counts=True)))
    big = np.array([subj_n[k] >= MIN_SUBJECT_FORECASTS for k in d['keys']])
    unit = d['keys'] if per_subject else d['day']
    out = {'cluster': 'subject' if per_subject else 'day', 'clusters': int(n_cl), 'models': {}}
    for i, e in enumerate(spec['entries']):
        r = {'ci': {}, 'strata': {}}
        for j, m in enumerate(HORIZONS):
            tj, pj = t[:, j], d['pred'][i, :, j]
            ok = np.isfinite(tj) & np.isfinite(pj)
            se = np.bincount(cl[ok], (pj[ok] - tj[ok]) ** 2, n_cl)
            cnt = np.bincount(cl[ok], minlength=n_cl).astype(np.float64)
            b = np.sqrt((boot @ se) / np.maximum(boot @ cnt, 1))
            r['ci'][m] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            s = {name: _rmse_me(tj, pj, f(tj)) for name, f in strata.items()}
            s['night'], s['day'] = _rmse_me(tj, pj, night), _rmse_me(tj, pj, ~night)
            if d['meal'] is not None:
                s['meal'], s['no_meal'] = _rmse_me(tj, pj, d['meal']), _rmse_me(tj, pj, ~d['meal'])
            r['strata'][m] = s
        t60, p60 = t[:, 1], d['pred'][i, :, 1]
        ok = np.isfinite(t60) & np.isfinite(p60)
        sel = ok & big if per_subject else ok
        r['spread60'] = [_rmse_me(t60, p60, sel & (unit == u))['rmse']
                         for u in np.unique(unit[sel])]
        r['source60'] = {s: _rmse_me(t60, p60, ok & (d['src'] == s))['rmse']
                         for s in np.unique(d['src'])}
        if e['key'] in d['bands']:
            r['bands'] = _band_scores(t, d['bands'][e['key']])
        out['models'][e['key']] = r
    return out


def traces(spec: dict, d: dict, rows: list[dict]) -> list[dict]:
    """Three example origins: a fall into a low, a rise after a meal, a quiet night."""
    by = {r['key']: r for r in rows}
    ours = [k for k in d['bands'] if by[k]['family'] == 'T1DMAI']
    best = min(ours, key=lambda k: by[k]['h'][60]['rmse'])
    show = [best] + [k for k in ('xgboost', 'persistence') if k in by]
    idx = {e['key']: i for i, e in enumerate(spec['entries'])}
    t, o = d['truth'], d['origin']
    full = np.isfinite(t).all(axis=1) & np.isfinite(o)
    night = np.isin(d['hour'], list(NIGHT_HOURS))
    meal = d['meal'] if d['meal'] is not None else np.zeros(len(o), bool)
    picks = {}
    sane = full & (t < 390).all(axis=1) & (t > 45).all(axis=1) & (o < 390)
    low = np.flatnonzero(sane & (t[:, 1] < 70) & (o >= 110))
    if len(low):
        picks['A fall into a low'] = _median_case(low, o[low] - t[low, 1])
    rise = np.flatnonzero(sane & meal & (o < 180) & (t[:, 1] - o > 40))
    if len(rise):
        picks['A rise after a meal'] = _median_case(rise, t[rise, 1] - o[rise])
    calm = np.flatnonzero(full & night & (np.abs(t - o[:, None]).max(axis=1) < 15))
    if len(calm):
        picks['Quiet night'] = calm[len(calm) // 2]
    out = []
    for title, r in picks.items():
        tts, v = d['cgm'][d['keys'][r]]
        sel = (tts >= d['ts'][r] - 3 * 3600) & (tts <= d['ts'][r] + 2 * 3600)
        out.append({'title': title, 'source': str(d['src'][r]),
                    'cgm_min': ((tts[sel] - d['ts'][r]) / 60).tolist(),
                    'cgm': np.where(np.isfinite(v[sel]), v[sel], np.nan).tolist(),
                    'models': [{'key': k, 'label': by[k]['label'], 'family': by[k]['family'],
                                'pred': d['pred'][idx[k], r].tolist(),
                                'band': d['bands'][k][r].tolist() if k in d['bands'] else None}
                               for k in show]})
    return out


def _day_rmse(d: dict, i: int) -> list[tuple[float, int]]:
    """(RMSE at 60 min, day) for every test day with at least 250 forecasts, sorted."""
    t60, p60 = d['truth'][:, 1], d['pred'][i, :, 1]
    out = []
    for day in np.unique(d['day']):
        sel = (d['day'] == day) & np.isfinite(t60) & np.isfinite(p60)
        if sel.sum() >= 250:
            out.append((float(np.sqrt(np.mean((p60[sel] - t60[sel]) ** 2))), int(day)))
    return sorted(out)


def day_blocks(spec: dict, d: dict, n_blocks: int = 4, every_h: int = 2) -> dict:
    """Per sim-only model: n_blocks pairs of consecutive test days spread evenly over the test."""
    if len(np.unique(d['keys'])) != 1:
        return {}
    idx = {e['key']: i for i, e in enumerate(spec['entries'])}
    tts, v = next(iter(d['cgm'].values()))
    out = {}
    for zs in sorted(k for k in d['bands'] if k.endswith('_zs')):
        full = {day: e for e, day in _day_rmse(d, idx[zs])}
        starts = [day for day in sorted(full) if day + 1 in full]
        picks = [starts[int(round(p))] for p in np.linspace(0, len(starts) - 1, n_blocks)]
        blocks = []
        for b, first in enumerate(picks):
            days = []
            for day in (first, first + 1):
                t0 = day * 86400
                cs = (tts >= t0) & (tts <= t0 + 86400 + 7200)
                org = np.flatnonzero((d['day'] == day) & (d['ts'] % (every_h * 3600) == 0)
                                     & np.isfinite(d['origin']))
                days.append({'day': int(day - min(full)) + 1, 'rmse': full[day],
                             'cgm_h': ((tts[cs] - t0) / 3600).tolist(), 'cgm': v[cs].tolist(),
                             'origins': [{'h': float((d['ts'][r] - t0) / 3600),
                                          'bg': float(d['origin'][r]),
                                          'pred': d['pred'][idx[zs], r].tolist(),
                                          'band': d['bands'][zs][r].tolist()} for r in org]})
            blocks.append(days)
        out[zs.removesuffix('_zs')] = blocks
    return out


def _median_case(idx: np.ndarray, size: np.ndarray) -> int:
    """The qualifying origin with the median effect size: typical, not extreme."""
    return int(idx[np.argsort(size)[len(idx) // 2]])


def _band_scores(t: np.ndarray, q: np.ndarray) -> dict:
    """Coverage, width and pinball loss of a (rows, horizons, quantiles) band."""
    lv = np.array(QUANTILES)
    out = {}
    for j, m in enumerate(HORIZONS):
        ok = np.isfinite(t[:, j]) & np.isfinite(q[:, j]).all(axis=1)
        y, qq = t[ok, j], q[ok, j]
        diff = y[:, None] - qq
        out[m] = {'n': int(ok.sum()),
                  'cover50': float(100 * ((y >= qq[:, 2]) & (y <= qq[:, 4])).mean()),
                  'cover90': float(100 * ((y >= qq[:, 0]) & (y <= qq[:, 6])).mean()),
                  'width50': float((qq[:, 4] - qq[:, 2]).mean()),
                  'width90': float((qq[:, 6] - qq[:, 0]).mean()),
                  'pinball': float(np.maximum(lv * diff, (lv - 1) * diff).mean()),
                  'below': (100 * (diff < 0).mean(axis=0)).tolist()}
    return out


def _fmt(v, nd=1) -> str:
    return '-' if v is None else f'{v:.{nd}f}'


def tables(name: str, spec: dict, rows: list[dict]) -> str:
    """Markdown tables: accuracy, clinical, hypo; rows sorted by RMSE at 60 min."""
    rows = sorted(rows, key=lambda r: r['h'][60]['rmse'])
    star = lambda r, m: ' *' if r['h'][m]['vs_best'] in ('best', 'tie') else ''
    out = [f'#### Accuracy: RMSE, MARD, bias (mg/dL, %)\n',
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
    out += ['', '#### Clinical: DTS, Clarke, CG-EGA (% of points)\n',
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
            else '-', f"{h[60]['me_hypo']:+.1f}" if h[60]['me_hypo'] is not None else '-'])
            + ' |')
    n = rows[0]['h'][60]['n']
    out.append(f"\n{n} forecasts at 60 min. * means lowest RMSE, or not significantly worse than "
               f"it (Wilcoxon test on absolute errors, alpha 0.05/{len(rows) - 1}).")
    return '\n'.join(out) + '\n'


def _ci(lo_hi) -> str:
    return f'{lo_hi[0]:.1f} to {lo_hi[1]:.1f}'


def extra_tables(spec: dict, rows: list[dict], ex: dict, ds: list[dict],
                 cost: dict | None) -> dict[str, str]:
    """Markdown snippets keyed by name: dataset, ci, strata, bands, sources, cost."""
    rows = sorted(rows, key=lambda r: r['h'][60]['rmse'])
    m_ = ex['models']
    out = {}
    t = ['| Source | Subjects | Person-days | Forecasts | CGM coverage % | Mean mg/dL | CV % | '
         '< 54 % | < 70 % | 70-180 % | > 180 % | > 250 % |', '| --- |' + ' ---: |' * 11]
    for s in ds if len(ds) > 2 else ds[:1]:
        t.append(f"| {s['source']} | {s['subjects']} | {s['days']:.0f} | {s['forecasts']} | "
                 f"{s['coverage']:.0f} | {s['mean']:.0f} | {s['cv']:.0f} | {s['tbr54']:.1f} | "
                 f"{s['tbr70']:.1f} | {s['tir']:.0f} | {s['tar180']:.0f} | {s['tar250']:.0f} |")
    out['dataset'] = '\n'.join(t)
    t = ['| Model | RMSE 30 | 95% CI | RMSE 60 | 95% CI | RMSE 120 | 95% CI |',
         '| --- |' + ' ---: |' * 6]
    for r in rows:
        c = m_[r['key']]['ci']
        t.append(f"| {r['label']} | " + ' | '.join(
            f"{r['h'][m]['rmse']:.1f} | {_ci(c[m])}" for m in (30, 60, 120)) + ' |')
    out['ci'] = '\n'.join(t)
    has_meal = 'meal' in m_[rows[0]['key']]['strata'][60]
    cols = [('hypo', '< 70'), ('eu', '70-180'), ('hyper', '> 180'), ('night', 'night'),
            ('day', 'day')] + ([('meal', 'meal in last 2 h'), ('no_meal', 'no meal')] if has_meal
                               else [])
    t = ['| Model | ' + ' | '.join(c for _, c in cols) + ' |', '| --- |' + ' ---: |' * len(cols)]
    for r in rows:
        s = m_[r['key']]['strata'][60]
        t.append(f"| {r['label']} | " + ' | '.join(
            '-' if s[k]['rmse'] is None else f"{s[k]['rmse']:.1f} ({s[k]['me']:+.0f})"
            for k, _ in cols) + ' |')
    n = m_[rows[0]['key']]['strata'][60]
    out['strata'] = '\n'.join(t) + '\n\nForecasts per group: ' + ', '.join(
        f'{c} {n[k]["n"]}' for k, c in cols) + '.'
    t = ['| Model | 50% band hits (30 / 60 / 120) | 90% band hits (30 / 60 / 120) | '
         '90% band width mg/dL (30 / 60 / 120) | Pinball loss (30 / 60 / 120) |',
         '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        b = m_[r['key']].get('bands')
        if b:
            t.append(f"| {r['label']} | " + ' | '.join(
                ' / '.join(f'{b[m][k]:.{nd}f}' for m in (30, 60, 120))
                for k, nd in (('cover50', 0), ('cover90', 0), ('width90', 0), ('pinball', 1)))
                + ' |')
    out['bands'] = '\n'.join(t)
    srcs = sorted(m_[rows[0]['key']]['source60'])
    if len(srcs) > 1:
        t = ['| Model | ' + ' | '.join(srcs) + ' |', '| --- |' + ' ---: |' * len(srcs)]
        for r in rows:
            s = m_[r['key']]['source60']
            t.append(f"| {r['label']} | " + ' | '.join(_fmt(s[k]) for k in srcs) + ' |')
        out['sources'] = '\n'.join(t)
    if cost:
        t = ['| Model | Numbers in the model | Microseconds per forecast | History used |',
             '| --- | ---: | ---: | ---: |']
        lab = {r['key']: r['label'] for r in rows}
        for k, c in sorted(cost.items(), key=lambda kv: kv[1]['ms_per_forecast']):
            name = lab.get(k) or lab.get(k.removeprefix('t1dmai_') + '_lora', k)
            us, minutes = 1e3 * c['ms_per_forecast'], 5 * c['ctx_steps']
            t.append(f"| {name.replace(' (sim + LoRA)', '')} | {c['numbers']:,} | "
                     f"{f'{us:,.0f}' if us >= 100 else f'{us:.2g}'} | "
                     f"{f'{minutes} min' if minutes < 60 else f'{minutes / 60:g} h'} |")
        out['cost'] = '\n'.join(t)
    return out


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


def _pick(rows, keys):
    by = {r['key']: r for r in rows}
    return [by[k] for k in (auto_keys(rows) if keys == 'auto' else keys) if k in by]


def fig_hist(ds, title, stem):
    """Test-CGM histogram with the 70 and 180 mg/dL lines."""
    import matplotlib.pyplot as plt
    th = THEMES['dark']
    h = np.array(ds[-1]['hist'], float)
    edges = np.arange(40, 405, 5)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    _axes(fig, ax, th)
    ax.grid(False)
    col = np.where(edges[:-1] < 70, th['fam']['T1DMAI'],
                   np.where(edges[:-1] >= 180, th['cat'][3], th['cat'][0]))
    ax.bar(edges[:-1], 100 * h / h.sum(), width=5, align='edge', color=col, linewidth=0)
    for x in (70, 180):
        ax.axvline(x, color=th['ink2'], lw=0.8, ls='--')
    s = ds[-1]
    ax.text(0.99, 0.95, tex(f"below 70: {s['tbr70']:.1f}%,  70-180: {s['tir']:.0f}%,  "
                            f"above 180: {s['tar180']:.0f}%"), transform=ax.transAxes,
            ha='right', va='top', color=th['ink'], fontsize=8)
    ax.set_xlim(40, 400)
    ax.set_xlabel('CGM glucose (mg/dL)')
    ax.set_ylabel(tex('Share of readings (%)'))
    _title(ax, title)
    _save(fig, stem, 'dark')
    plt.close(fig)


def fig_ci(rows, ex, title, stem):
    """RMSE at 60 min with the 95% bootstrap interval, one row per model."""
    import matplotlib.pyplot as plt
    th = THEMES['dark']
    rows = sorted(rows, key=lambda r: r['h'][60]['rmse'])
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(rows) + 1.0))
    _axes(fig, ax, th)
    for i, r in enumerate(rows):
        lo, hi = ex['models'][r['key']]['ci'][60]
        c = th['fam'][r['family']]
        ax.plot([lo, hi], [i, i], color=c, lw=1.2, solid_capstyle='butt')
        ax.scatter([r['h'][60]['rmse']], [i], s=18, color=c, edgecolor=th['bg'],
                   linewidth=0.6, zorder=3)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    _model_ticks(ax, rows, th)
    ax.set_xlabel(tex('RMSE at 60 min (mg/dL), dot = value, bar = 95% interval'))
    _title(ax, title)
    _save(fig, stem, 'dark')
    plt.close(fig)


def fig_spread(rows, ex, keys, title, stem):
    """Box plot of 60-min RMSE per subject (or per day) for a few models."""
    import matplotlib.pyplot as plt
    th = THEMES['dark']
    sel = sorted(_pick(rows, keys), key=lambda r: np.median(ex['models'][r['key']]['spread60']))
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(sel) + 1.1))
    _axes(fig, ax, th)
    data = [ex['models'][r['key']]['spread60'] for r in sel]
    bp = ax.boxplot(data, orientation='horizontal', widths=0.5, patch_artist=True, showfliers=True,
                    flierprops={'marker': '.', 'markersize': 3, 'markeredgecolor': th['ink2']},
                    medianprops={'color': th['ink'], 'lw': 1.0},
                    whiskerprops={'color': th['ink2'], 'lw': 0.6},
                    capprops={'color': th['ink2'], 'lw': 0.6})
    for patch, r in zip(bp['boxes'], sel):
        patch.set(facecolor=th['fam'][r['family']], edgecolor=th['ink2'], lw=0.6, alpha=0.85)
    ax.set_yticks(range(1, len(sel) + 1))
    ax.set_ylim(len(sel) + 0.5, 0.5)
    _model_ticks_at(ax, sel, th, start=1)
    unit = 'subject' if ex['cluster'] == 'subject' else 'test day'
    ax.set_xlabel(tex(f'RMSE at 60 min per {unit} (mg/dL)'))
    _title(ax, title)
    _save(fig, stem, 'dark')
    plt.close(fig)


def _model_ticks_at(ax, rows, th, start=0):
    names = [rf"\textbf{{{tex(r['label'])}}}" if r['family'] == 'T1DMAI' else tex(r['label'])
             for r in rows]
    ax.set_yticks(range(start, start + len(rows)), names)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(th['fam']['T1DMAI'] if r['family'] == 'T1DMAI' else th['ink'])


def fig_strata(rows, ex, keys, title, stem):
    """RMSE at 60 min split by glucose range, time of day and meals."""
    import matplotlib.pyplot as plt
    th = THEMES['dark']
    sel = _pick(rows, keys)
    s0 = ex['models'][sel[0]['key']]['strata'][60]
    panels = [('Glucose at the target time', [('hypo', 'below 70'), ('eu', '70-180'),
                                               ('hyper', '$>$180')]),
              ('Time of day', [('night', 'midnight-6am'), ('day', '6am-midnight')])]
    if 'meal' in s0:
        panels.append(('Meals', [('meal', 'ate in last 2h'), ('no_meal', 'other')]))
    fig, axs = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.0), sharey=True,
                            gridspec_kw={'width_ratios': [len(g) for _, g in panels]})
    for ax, (pt, groups) in zip(np.atleast_1d(axs), panels):
        _axes(fig, ax, th)
        ax.grid(True, axis='y', color=th['grid'], lw=0.5, ls=':')
        ax.grid(False, axis='x')
        w = 0.8 / len(sel)
        for k, r in enumerate(sel):
            s = ex['models'][r['key']]['strata'][60]
            x = np.arange(len(groups)) + (k - (len(sel) - 1) / 2) * w
            ax.scatter(x, [s[g]['rmse'] for g, _ in groups], s=22, marker=MARKERS[k],
                       color=th['fam'][r['family']], edgecolor=th['bg'], linewidth=0.6,
                       zorder=3, label=tex(r['label']))
        ax.set_xticks(range(len(groups)), [gl for _, gl in groups])
        ax.tick_params(axis='x', top=False)
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_title(tex(pt), fontsize=9)
    np.atleast_1d(axs)[0].set_ylabel('RMSE at 60 min (mg/dL)')
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    h, lab = np.atleast_1d(axs)[0].get_legend_handles_labels()
    _legend(fig, th, handles=h, labels=lab, loc='upper left', bbox_to_anchor=(0.0, 0.92),
            ncols=3, handletextpad=0.2, columnspacing=1.0)
    fig.suptitle(tex(title), color=th['ink'], fontsize=10, x=0.01, ha='left', y=0.99)
    _save(fig, stem, 'dark')
    plt.close(fig)


def fig_calibration(rows, ex, title, stem, horizon=60):
    """Observed share below each forecast quantile against the promised share."""
    import matplotlib.pyplot as plt
    th = THEMES['dark']
    best: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: r['h'][60]['rmse']):
        if 'bands' in ex['models'][r['key']]:
            best.setdefault(r['key'].split('_')[-1].split('-')[0], r)
    sel = list(best.values())
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    _axes(fig, ax, th)
    ax.grid(True, axis='y', color=th['grid'], lw=0.5, ls=':')
    lv = 100 * np.array(QUANTILES)
    ax.plot([0, 100], [0, 100], color=th['ink2'], lw=0.8, ls='--', label='perfect')
    for k, r in enumerate(sel):
        b = ex['models'][r['key']]['bands'][horizon]['below']
        c = th['cat'][k]
        ax.plot(lv, b, marker=MARKERS[k], color=c, markeredgecolor=th['bg'],
                markeredgewidth=0.6, label=tex(r['label']))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(tex('Promised share below the quantile (%)'))
    ax.set_ylabel(tex('Observed share below (%)'))
    _title(ax, title)
    _legend(ax, th, loc='upper left', fontsize=7, handlelength=1.6)
    _save(fig, stem, 'dark')
    plt.close(fig)


def fig_traces(ex, title, stem):
    """Example forecasts: CGM, three models at 30/60/90/120 min, T1DMAI 50% and 90% bands."""
    import matplotlib.pyplot as plt
    th = THEMES['dark']
    tr = ex.get('traces', [])
    if not tr:
        return
    fig, axs = plt.subplots(1, len(tr), figsize=(3.3 * len(tr), 3.0))
    for ax, t in zip(np.atleast_1d(axs), tr):
        _axes(fig, ax, th)
        ax.grid(True, axis='y', color=th['grid'], lw=0.5, ls=':')
        x, v = np.array(t['cgm_min']), np.array(t['cgm'], float)
        past = x <= 0
        ax.plot(x[past], v[past], color=th['ink'], lw=1.0, label='CGM before the forecast')
        ax.plot(x[~past], v[~past], color=th['ink2'], lw=1.0, ls=':', label='CGM afterwards')
        ax.axvspan(0, 120, color=th['grid'], alpha=0.35, lw=0)
        for k, mdl in enumerate(t['models']):
            xs = [0] + list(HORIZONS)
            o = v[past][-1]
            c = th['fam'][mdl['family']] if mdl['key'] != 'persistence' else th['fam']['Baseline']
            if mdl['band']:
                b = np.array(mdl['band'])
                ax.fill_between(xs, [o] + list(b[:, 0]), [o] + list(b[:, 6]), color=c,
                                alpha=0.18, lw=0)
                ax.fill_between(xs, [o] + list(b[:, 2]), [o] + list(b[:, 4]), color=c,
                                alpha=0.30, lw=0)
            ax.plot(xs, [o] + mdl['pred'], color=c, marker=MARKERS[k], ms=3.5,
                    markeredgecolor=th['bg'], markeredgewidth=0.5,
                    label=tex(mdl['label']))
        for y in (70, 180):
            ax.axhline(y, color=th['ink2'], lw=0.5, ls='--')
        ax.set_xlim(-180, 120)
        ax.set_xticks([-180, -120, -60, 0, 60, 120])
        ax.set_xlabel('Minutes from the forecast')
        ax.set_title(tex(t['title']), fontsize=9)
    np.atleast_1d(axs)[0].set_ylabel('Glucose (mg/dL)')
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    h, lab = np.atleast_1d(axs)[0].get_legend_handles_labels()
    _legend(fig, th, handles=h, labels=lab, loc='upper left', bbox_to_anchor=(0.0, 0.92),
            ncols=3, handletextpad=0.3, columnspacing=1.0)
    fig.suptitle(tex(title), color=th['ink'], fontsize=10, x=0.01, ha='left', y=0.99)
    _save(fig, stem, 'dark')
    plt.close(fig)


def fig_days(ex, arch, block, title, stem):
    """Two consecutive test days: CGM and the sim-only 2-hour forecasts with 50%/90% bands."""
    import matplotlib.pyplot as plt
    th = THEMES['dark']
    blocks = ex.get('blocks', {}).get(arch)
    if not blocks:
        return
    days = blocks[block]
    c = th['fam']['T1DMAI']
    fig, axs = plt.subplots(len(days), 1, figsize=(9.6, 2.6 * len(days)), sharex=True)
    for ax, dd in zip(np.atleast_1d(axs), days):
        _axes(fig, ax, th)
        ax.grid(True, axis='both', color=th['grid'], lw=0.5, ls=':')
        for y in (70, 180):
            ax.axhline(y, color=th['ink2'], lw=0.5, ls='--')
        ax.plot(dd['cgm_h'], dd['cgm'], color=th['ink'], lw=1.0, label='CGM')
        for k, o in enumerate(dd['origins']):
            xs = o['h'] + np.array([0] + list(HORIZONS)) / 60
            b = np.array(o['band'])
            ax.fill_between(xs, [o['bg']] + list(b[:, 0]), [o['bg']] + list(b[:, 6]), color=c,
                            alpha=0.15, lw=0, label=tex('90% band') if k == 0 else None)
            ax.fill_between(xs, [o['bg']] + list(b[:, 2]), [o['bg']] + list(b[:, 4]), color=c,
                            alpha=0.30, lw=0, label=tex('50% band') if k == 0 else None)
            ax.plot(xs, [o['bg']] + o['pred'], color=c, lw=1.2, marker='o', ms=2.5,
                    label='2 hour forecast' if k == 0 else None)
            ax.scatter([o['h']], [o['bg']], s=10, color=th['ink'], zorder=4)
        ax.set_xlim(0, 26)
        ax.set_ylabel('Glucose (mg/dL)')
        ax.set_title(tex(f"Test day {dd['day']}. RMSE at 60 min that day: {dd['rmse']:.0f} mg/dL"),
                     loc='left', fontsize=9, color=th['ink'])
    last = np.atleast_1d(axs)[-1]
    last.set_xticks(range(0, 27, 3))
    last.set_xlabel('Hour of the day')
    h, lab = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _legend(fig, th, handles=h, labels=lab, loc='upper right', bbox_to_anchor=(0.99, 0.995),
            ncols=4, handletextpad=0.4)
    fig.suptitle(tex(title), color=th['ink'], fontsize=10, x=0.01, ha='left', y=0.995)
    _save(fig, stem, 'dark')
    plt.close(fig)


def fig_cost(rows, cost, title, stem, offsets=None):
    """Accuracy against CPU time per forecast, log scale; offsets: {key: [dx, dy]} in points."""
    import matplotlib.pyplot as plt
    th = THEMES['dark']
    by = {r['key']: r for r in rows}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _axes(fig, ax, th)
    ax.grid(True, axis='both', which='major', color=th['grid'], lw=0.5, ls=':')
    offsets = offsets or {}
    for k, c in cost.items():
        r = by.get(k) or by.get(k.removeprefix('t1dmai_') + '_lora')
        if r is None:
            continue
        x, y = 1e3 * c['ms_per_forecast'], r['h'][60]['rmse']
        ax.scatter([x], [y], s=26, color=th['fam'][r['family']], edgecolor=th['bg'],
                   linewidth=0.6, zorder=3)
        dx, dy = offsets.get(k, (4, 2))
        ax.annotate(tex(r['label'].replace('T1DMAI ', '').replace(' (sim + LoRA)', '')),
                    (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=6.5,
                    color=th['fam']['T1DMAI'] if r['family'] == 'T1DMAI' else th['ink2'],
                    ha='right' if dx < 0 else 'left')
    for fam in FAMILIES:
        ax.scatter([], [], s=26, color=th['fam'][fam], label=fam)
    _legend(ax, th, loc='upper right', fontsize=7, handletextpad=0.2)
    ax.set_xscale('log')
    lo = min(r['h'][60]['rmse'] for r in rows)
    ax.set_ylim(lo - 3, None)
    ax.set_xlabel('Microseconds per forecast on one CPU thread (log scale)')
    ax.set_ylabel('RMSE at 60 min (mg/dL)')
    _title(ax, title)
    _save(fig, stem, 'dark')
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
    cost = None
    if man.get('cost'):
        with open(man['cost']) as f:
            cost = json.load(f)
        with open(os.path.join(a.out, 'cost.json'), 'w') as f:
            json.dump(cost, f, indent=1)
    scored, extra = {}, {}
    for name, spec in man['tracks'].items():
        path = os.path.join(a.out, f'{name}.json')
        xpath = os.path.join(a.out, f'{name}-extra.json')
        if a.figures_only:
            with open(path) as f:
                rows = json.load(f)
            with open(xpath) as f:
                ex = json.load(f)
        else:
            d = load_track(spec)
            rows = score_track(spec, d)
            ex = extras(spec, d) | {'dataset': dataset_stats(spec, d),
                                    'traces': traces(spec, d, rows),
                                    'blocks': day_blocks(spec, d)}
            with open(path, 'w') as f:
                json.dump(rows, f, indent=1)
            with open(xpath, 'w') as f:
                json.dump(ex, f, indent=1)
            with open(os.path.join(a.out, f'{name}.md'), 'w') as f:
                f.write(tables(name, spec, rows))
            print(f'{name}: {len(rows)} models scored')
        _int_keys(rows, ex)
        for snip, md in extra_tables(spec, rows, ex, ex['dataset'],
                                     cost if name == 'personal' else None).items():
            with open(os.path.join(a.out, f'{name}-{snip}.md'), 'w') as f:
                f.write(md + '\n')
        scored[name], extra[name] = rows, ex
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update(STYLE)
    kinds = {'leaderboard': fig_leaderboard, 'horizon': fig_horizon, 'heatmap': fig_heatmap,
             'significance': fig_significance, 'paths': fig_paths}
    with_ex = {'ci': fig_ci, 'spread': fig_spread, 'strata': fig_strata,
               'calibration': fig_calibration}
    for fig in man.get('figures', []):
        fig = dict(fig)
        kind, track, stem = fig.pop('kind'), fig.pop('track'), fig.pop('stem')
        out = os.path.join(a.figures, stem)
        if kind in with_ex:
            with_ex[kind](scored[track], extra[track], stem=out, **fig)
        elif kind == 'hist':
            fig_hist(extra[track]['dataset'], stem=out, **fig)
        elif kind == 'traces':
            fig_traces(extra[track], stem=out, **fig)
        elif kind == 'days':
            fig_days(extra[track], stem=out, **fig)
        elif kind == 'cost':
            fig_cost(scored[track], cost, stem=out, **fig)
        else:
            kinds[kind](scored[track], stem=out, **fig)
        print(f'figure {stem}')


def _int_keys(rows: list[dict], ex: dict) -> None:
    """JSON turns horizon keys into strings; turn them back."""
    for r in rows:
        r['h'] = {int(k): v for k, v in r['h'].items()}
    for r in ex['models'].values():
        for f in ('ci', 'strata', 'bands'):
            if f in r:
                r[f] = {int(k): v for k, v in r[f].items()}


if __name__ == '__main__':
    main()
