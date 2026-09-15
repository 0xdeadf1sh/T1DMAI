"""Compare two checkpoints' counterfactual dose response on identical windows; prints one table.

whatif.py's probe, on fresh simulator test patients and on the patient's own record from a backup,
with the insulin ladder injected twice: the simulator's gamma and the phone's logged bolus shape.
Every figure is a response against that model's own no-dose baseline."""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ROOT, 'metrics'), ROOT]

import numpy as np
import torch

import whatif
import sim_data
from config import PATCH_SIZE
from day_curves import load_model
from metrics.core.horizons import HORIZONS
from metrics.core.schema import segment_grid
from t1dmdroid_converter import custom_curve, live_events, read_archive, record_channels

SIDES = (('carb', 'hypo'), ('insulin', 'hyper'), ('exercise', 'hyper'))


def phone_bolus_kernel(kinds: dict) -> tuple[np.ndarray, int]:
    """Unit-total mean of the logged boluses' stored curves, and how many there were."""
    events, _ = live_events(kinds)
    curves = [c / c.sum() for o in events['dose'] if o['kd'] == 'BOLUS'
              for c in [custom_curve(o)] if c is not None and c.sum() > 0]
    if not curves:
        raise SystemExit('the backup holds no bolus with a stored curve')
    kernel = np.zeros(max(len(c) for c in curves))
    for c in curves:
        kernel[:len(c)] += c
    return kernel / kernel.sum(), len(curves)


def record_segments(kinds: dict) -> list:
    """The record's gap-free stretches; carb and insulin ride as pre-resolved curves."""
    r = record_channels(kinds)
    t0 = (datetime.fromtimestamp(r['t0_ms'] / 1000, timezone.utc).replace(tzinfo=None)
          + timedelta(minutes=r['tz_min']))
    z = np.zeros(len(r['bg']))
    return segment_grid('phone', 'record', t0, r['bg'].astype(np.float64), z, z, z,
                        r['exercise'], carb_curve=r['carb'], insulin_curve=r['insulin'])


def rows(r: dict, sides: tuple[str, ...]) -> list[tuple[str, float | None]]:
    """(label, value) for the table; labels carry no per-model count, so the two align."""
    out = []
    for side, arm in SIDES:
        if side not in sides:
            continue
        b = r[side]
        if not b.get('n'):
            out.append((f'{side}: not probed', None))
            continue
        u, ref = b['unit'], whatif.REF_IDX
        for d, v in zip(b['doses'][1:], b['mean_dbg'][str(HORIZONS[-1])][1:]):
            out.append((f'{side} Δ@{HORIZONS[-1]}m {d:g} {u} (mg/dL)', v))
        for h in HORIZONS:
            out.append((f'{side} right sign @{h}m, {b["doses"][ref]:g} {u}',
                        b['correct_sign_frac'][str(h)][ref]))
        out += [
            (f'{side} slope (mg/dL per {u})', b['slope']['median']),
            (f'{side} monotone, terminal', b['monotone_frac']['terminal']),
            (f'{side} monotone, pointwise', b['monotone_frac']['pointwise']),
            (f'{side} onset (min)', b['onset_min']['median']),
            (f'{side} onset reached', b['onset_min']['reached_frac']),
            (f'{side} peak (min)', b['peak_min']['median']),
            (f'{side} wrong-way push (mg/dL)', b['mean_adverse_dbg']),
            (f'{side} {arm} windows to rescue', b['rescue']['n']),
            (f'{side} {arm} rescued, {b["doses"][ref]:g} {u}',
             None if b['rescue']['frac_by_dose'] is None else b['rescue']['frac_by_dose'][ref]),
        ]
    return out


def print_table(title: str, names: tuple[str, str], a: list, b: list) -> None:
    w = max(len(label) for label, _ in a)
    cols = max(12, *(len(n) for n in names))
    print(f'\n== {title}')
    print(f'{"":{w}}  {names[0]:>{cols}}  {names[1]:>{cols}}  {"Δ":>8}')
    for (label, va), (_, vb) in zip(a, b):
        cell = lambda v: f'{"—":>{cols}}' if v is None else f'{v:>{cols}.3g}'  # noqa: E731
        diff = '' if va is None or vb is None else f'{vb - va:>+8.3g}'
        print(f'{label:{w}}  {cell(va)}  {cell(vb)}  {diff}')


def model_name(path: str) -> str:
    m = re.search(r'models/([^/]+)/checkpoints/[^/]+$', os.path.abspath(path).replace(os.sep, '/'))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('pretrained', help='checkpoint A')
    p.add_argument('finetuned', help='checkpoint B')
    p.add_argument('--backup', required=True, help='.t1dmbak: the record cohort and bolus shape')
    p.add_argument('--sim-seeds', type=int, default=len(sim_data.TEST_SEEDS))
    p.add_argument('--sim-hours', type=float, default=sim_data.DEFAULT_HOURS)
    p.add_argument('--stride-patches', type=int, default=whatif.STRIDE // PATCH_SIZE,
                   help='simulator window stride')
    p.add_argument('--cap', type=int, default=whatif.CAP, help='simulator windows per patient')
    p.add_argument('--record-stride-patches', type=int, default=1)
    p.add_argument('--record-cap', type=int, default=100_000, help='record windows per stretch')
    a = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = (model_name(a.pretrained), model_name(a.finetuned))
    models = [load_model(device, path)[:2] for path in (a.pretrained, a.finetuned)]
    kinds = read_archive(a.backup)
    kernel, n_bolus = phone_bolus_kernel(kinds)
    cohorts = [
        ('simulator', sim_data.make_sim_segments(sim_data.TEST_SEEDS[:max(1, a.sim_seeds)],
                                                 a.sim_hours),
         a.stride_patches * PATCH_SIZE, a.cap),
        ('record', record_segments(kinds), a.record_stride_patches * PATCH_SIZE, a.record_cap),
    ]
    for cohort, segs, stride, cap in cohorts:
        res = {}
        for shape, bolus in (('gamma', None), ('phone', kernel)):
            print(f'probing {cohort} with the {shape} bolus...', file=sys.stderr, flush=True)
            res[shape] = [whatif.run(m.to(device).eval(), stats, device, segs, stride=stride,
                                     cap=cap, bolus_kernel=bolus) for m, stats in models]
        n = res['gamma'][0]['n_windows']
        rail = ' / '.join(f'{r["null_rail"]["max_abs_dbg"]}' for r in res['gamma'])
        print(f'\n#### {cohort}: {n} windows, null-rail max |Δ| {rail} mg/dL')
        print_table(f'{cohort} — carb and exercise', names,
                    *[rows(r, ('carb', 'exercise')) for r in res['gamma']])
        print_table(f'{cohort} — insulin, simulator gamma bolus', names,
                    *[rows(r, ('insulin',)) for r in res['gamma']])
        print_table(f'{cohort} — insulin, phone bolus (mean of {n_bolus} logged curves)', names,
                    *[rows(r, ('insulin',)) for r in res['phone']])


if __name__ == '__main__':
    main()
