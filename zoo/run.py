"""Fit a zoo model, fill the leaderboard template, score it, and rebuild the zoo leaderboard.

python -m zoo.run MODEL --cache DIR --template template.parquet --truth test.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from zoo.registry import models

MODELS = models()
from zoo.windows import drop_zone_doses, horizon_columns, template_windows, train_windows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('model', choices=sorted(MODELS))
    p.add_argument('--cache', required=True)
    p.add_argument('--template', required=True)
    p.add_argument('--truth', required=True)
    p.add_argument('--out', default=os.path.join(ROOT, 'zoo_runs'))
    p.add_argument('--train-windows', type=int, default=500_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--causal', action='store_true',
                   help='zero doses after the origin, in training and prediction')
    p.add_argument('--every', type=int, default=1, metavar='N',
                   help='keep every Nth template row; scoring only, never a file to submit')
    return p.parse_args()


def rebuild_leaderboard(out: str) -> None:
    """``leaderboard.json`` and ``leaderboard.md`` from every ``<out>/<model>/scores.json``."""
    rows = []
    for name in sorted(os.listdir(out)):
        path = os.path.join(out, name, 'scores.json')
        if os.path.isfile(path):
            with open(path) as f:
                rows.append({'model': name, **json.load(f)})
    rows.sort(key=lambda r: r['overall']['30']['rmse'] or float('inf'))
    with open(os.path.join(out, 'leaderboard.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    hs = ('30', '60', '90', '120')
    lines = ['| Model | ' + ' | '.join(f'RMSE {h}' for h in hs) + ' | '
             + ' | '.join(f'DTS-A {h}' for h in hs) + ' |',
             '| --- |' + ' ---: |' * (2 * len(hs))]
    for r in rows:
        o = r['overall']
        lines.append(f"| {r['model']} | " + ' | '.join(f"{o[h]['rmse']:.2f}" for h in hs) + ' | '
                     + ' | '.join(f"{o[h]['dts_a']:.1f}" for h in hs) + ' |')
    with open(os.path.join(out, 'leaderboard.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def run_name(key: str, causal: bool) -> str:
    return key + ('_causal' if causal else '')


def fit_or_load(key: str, causal: bool, cache, runs: str, n_windows: int, seed: int,
                refit: bool = False):
    """The model ``zoo_runs/<run>/params.npz`` holds, fitted and saved there first if absent."""
    model = MODELS[key]()
    model.causal = causal
    run_dir = os.path.join(runs, run_name(key, causal))
    path = os.path.join(run_dir, 'params.npz')
    if os.path.isfile(path) and not refit:
        with np.load(path) as z:
            model.load({k: z[k] for k in z.files})
        return model
    os.makedirs(run_dir, exist_ok=True)
    t = time.time()
    tw = train_windows(cache, n_windows, model.ctx, seed)
    if causal:
        tw = drop_zone_doses(tw)
    print(f'{run_name(key, causal)}: {len(tw)} train windows, ctx {model.ctx}  '
          f'({time.time() - t:.0f} s)', flush=True)
    model.fit(tw)
    np.savez(path, **model.state())
    return model


class Template:
    """Leaderboard rows, each resolved to its cache subject."""

    def __init__(self, path: str, every: int, cache) -> None:
        from export_submission import _read_template
        from finetune_data import HORIZON_MINUTES
        self.table, t_src, t_sid, self.t_epoch = _read_template(path, every, HORIZON_MINUTES)
        self.n_rows = len(self.t_epoch)
        by_key = {(r['source'], r['sid']): i for i, r in enumerate(cache.subjects)}
        self.rows_by_subject: dict[int, list[int]] = {}
        for row in range(self.n_rows):
            s = by_key.get((str(t_src[row]), str(t_sid[row])))
            if s is not None:
                self.rows_by_subject.setdefault(s, []).append(row)


def predict_template(model, cache, tpl: Template, causal: bool) -> tuple[np.ndarray, int]:
    """``(rows, horizons)`` mg/dL, unanchorable rows ladder-filled; plus the anchored count."""
    from export_submission import _apply_ladder
    from finetune_data import HORIZON_MINUTES
    from T1DMSIM.simulator import BG_CLAMP_MAX, BG_CLAMP_MIN
    pred = np.full((tpl.n_rows, len(HORIZON_MINUTES)), np.nan)
    anchored = 0
    for s, rows in tpl.rows_by_subject.items():
        w = template_windows(cache, {s: rows}, tpl.t_epoch, model.ctx)
        if causal:
            w = drop_zone_doses(w)
        if len(w):
            pred[w.row] = horizon_columns(model.predict(w))
            anchored += len(w)
    _apply_ladder(pred, cache, tpl.rows_by_subject, tpl.t_epoch)
    np.clip(pred, BG_CLAMP_MIN, BG_CLAMP_MAX, out=pred)
    return pred, anchored


def main() -> None:
    args = parse_args()
    import pyarrow.parquet as pq

    from export_submission import _write
    from finetune_data import HORIZON_MINUTES, FinetuneCache
    from score_submission import score_table

    cache = FinetuneCache(args.cache)
    run_dir = os.path.join(args.out, run_name(args.model, args.causal))
    model = fit_or_load(args.model, args.causal, cache, args.out, args.train_windows, args.seed,
                        refit=True)
    tpl = Template(args.template, args.every, cache)
    t = time.time()
    pred, anchored = predict_template(model, cache, tpl, args.causal)
    n_rows = tpl.n_rows
    print(f'anchored {anchored}/{n_rows} template rows  ({time.time() - t:.0f} s)', flush=True)
    out_parquet = os.path.join(run_dir, 'predictions.parquet')
    if _write(tpl.table, pred, n_rows, HORIZON_MINUTES, out_parquet) == 0:
        sys.exit('no horizon fully populated')

    scores = score_table(pq.read_table(out_parquet), args.truth, by_source=True)
    by_source = scores.pop('by_source')
    with open(os.path.join(run_dir, 'scores.json'), 'w') as f:
        json.dump({'overall': {str(m): v for m, v in scores.items()},
                   'by_source': {s: {str(m): v for m, v in r.items()}
                                 for s, r in by_source.items()},
                   'anchored': anchored, 'rows': n_rows, 'every': args.every,
                   'causal': args.causal,
                   'train_windows': args.train_windows, 'seed': args.seed}, f, indent=1)
    rebuild_leaderboard(args.out)


if __name__ == '__main__':
    main()
