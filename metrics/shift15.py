"""
CGM-lag experiment: the simulator emits BG with no interstitial sensor lag, but
real CGM lags plasma BG by ~15 min, so a sim-trained model sees every real
event→BG response arriving 3 steps (15 min) late. Advance the real CGM by 3
steps — bg_aligned[t] = cgm[t+3] — to re-align the response to its cause, and
re-measure direction/amplitude (amp_var) AND the headline suite, baseline
(shift=0) vs shifted (shift=3). The suite is the shared
``realdata.metrics.compute_suite``, fed the window fan, so its level metrics are
BAND-SCORED (the truth projected onto the τ=``METRIC_BAND_TAU_LO``/``_HI`` band);
the amp_var block stays on the median line. The model is always conditioned —
each window announces its true future carbs/insulin.

Same split / strides / caps as run.py so shift=0 reproduces the headline suite.
Writes metrics/shift15.json.
"""
from __future__ import annotations
import os, sys, json, dataclasses
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
torch.set_num_threads(8)

from realdata import load_dataset
from realdata.calibrate import collect_windows, forecast_windows, forecast_bands
from realdata.metrics import compute_suite
from curves import load_model, split_segments
import amp_var as AV
import figstyle as F
from figstyle import plt

F.style()

DATASETS = ('ohiot1dm', 'azt1d', 'shanghai')
CAL_STRIDE, CAL_CAP = 8, 24
TEST_STRIDE, TEST_CAP = 4, 60
SHIFT_STEPS = 3                       # 15 min on the 5-min grid


def shift_cgm_back(seg, k: int):
    """Advance CGM by k steps (bg[t]=cgm[t+k]); truncate per-step channels to match.
    Events keep their original index → the BG response moves k steps earlier
    relative to its causal carb/insulin."""
    if k == 0:
        return seg
    n = len(seg.cgm) - k
    return dataclasses.replace(
        seg, cgm=seg.cgm[k:].copy(),
        carb_grams=seg.carb_grams[:n].copy(), bolus_units=seg.bolus_units[:n].copy(),
        basal_rate=seg.basal_rate[:n].copy(), exercise=seg.exercise[:n].copy())


def headline(model, stats, device, cal, test):
    # Always conditioned: each window announces its true future carbs/insulin.
    collect_windows(model, stats, cal, device, stride_patches=CAL_STRIDE,
                    max_per_patient=CAL_CAP, conditional=True)
    test_w = collect_windows(model, stats, test, device, stride_patches=TEST_STRIDE,
                             max_per_patient=TEST_CAP, conditional=True)
    pred, true, last_bg, pats = forecast_windows(test_w)
    m = compute_suite(pred, true, last_bg, pats, bands=forecast_bands(test_w))   # keyed by int horizon
    return {'n_test': len(test_w),
            'h': {h: {'rmse': m[int(h)]['rmse_point'], 'skill': 100 * m[int(h)]['skill_point'],
                      'mard': m[int(h)]['mard'], 'clarkeA': m[int(h)]['clarke_A'],
                      'hypoR': m[int(h)]['hypo']['recall'], 'hypoP': m[int(h)]['hypo']['precision'],
                      'hyperR': m[int(h)]['hyper']['recall'], 'hyperP': m[int(h)]['hyper']['precision']}
                  for h in ('30', '60', '120')}}


SHAPE_KEYS = (('direction_corr', 'direction corr'), ('g_star', 'g*'),
              ('amp_ratio', 'amplitude ratio'))


def _figure(out: dict, step) -> str:
    """Baseline against CGM-advanced, one row per cohort × metric.

    A dumbbell is the before→after form: one hue in two shades, both ends labelled,
    so the reader sees the size AND the direction of each move without decoding a
    pair of bars.
    """
    dss = [d for d in DATASETS if d in out]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, rows, title, xlabel, fmt in (
            (axes[0],
             [(f'{F.label(d)} · {name}', out[d]['base']['amp'][key], out[d]['shift15']['amp'][key])
              for d in dss for key, name in SHAPE_KEYS],
             'per-patch ΔBG shape', 'dimensionless', '{:.2f}'),
            (axes[1],
             [(f'{F.label(d)} · @{h}m', out[d]['base']['headline']['h'][h]['rmse'],
               out[d]['shift15']['headline']['h'][h]['rmse'])
              for d in dss for h in ('30', '60', '120')],
             'forecast error', 'RMSE (mg/dL)', '{:.1f}')):
        F.dumbbell_rows(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], fmt)
        ax.set_title(title); ax.set_xlabel(xlabel)
    fig.suptitle(f'CGM {SHIFT_STEPS * 5}-min lag shift probe (step {step})',
                 x=0.006, ha='left', fontsize=12, color=F.INK)
    F.pair_legend(fig, 'as recorded', f'advanced {SHIFT_STEPS * 5} min',
                  loc='upper right', bbox_to_anchor=(0.995, 0.995), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return F.save(fig, 'shift15.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = load_model(device)
    model = model.to(device); model.eval()
    print(f"[shift15] model step={step} device={device}  shift={SHIFT_STEPS} steps (15 min)")
    out = {'_meta': {'step': step, 'shift_steps': SHIFT_STEPS}}
    for ds in DATASETS:
        segs = load_dataset(ds)
        cal, test = split_segments(segs, ds)
        res_ds = {}
        for tag, k in (('base', 0), ('shift15', SHIFT_STEPS)):
            cal_k = [shift_cgm_back(s, k) for s in cal]
            test_k = [shift_cgm_back(s, k) for s in test]
            amp = AV.run(model, stats, device, test_k)['bg_per_patch_dbg']
            hl = headline(model, stats, device, cal_k, test_k)
            res_ds[tag] = {'amp': amp, 'headline': hl}
        out[ds] = res_ds

        b, s = res_ds['base'], res_ds['shift15']
        ba, sa = b['amp'], s['amp']
        print(f"\n== {ds} ==")
        print(f"  per-patch ΔBG   dir_corr: {ba['direction_corr']:.2f} -> {sa['direction_corr']:.2f}   "
              f"g*: {ba['g_star']:.2f} -> {sa['g_star']:.2f}   "
              f"amp_ratio: {ba['amp_ratio']:.2f} -> {sa['amp_ratio']:.2f}")
        for h in ('30', '60', '120'):
            bh, sh = b['headline']['h'][h], s['headline']['h'][h]
            fr = lambda x: '—' if x is None else f"{x:.2f}"
            print(f"  @{h:>3}m RMSE {bh['rmse']:.1f}->{sh['rmse']:.1f}  "
                  f"skill {bh['skill']:+.1f}%->{sh['skill']:+.1f}%  "
                  f"hyperR {fr(bh['hyperR'])}->{fr(sh['hyperR'])}  "
                  f"hypoR {fr(bh['hypoR'])}->{fr(sh['hypoR'])}")
    with open(os.path.join(HERE, 'shift15.json'), 'w') as f:
        json.dump(out, f, indent=2)
    path = _figure(out, step)
    print(f"\n[shift15] wrote shift15.json and {path}")


if __name__ == '__main__':
    main()
