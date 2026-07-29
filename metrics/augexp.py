"""
What if the real datasets are AUGMENTED to account for their orphan excursions?

metrics/augmented/augment.py reconstructs the unlogged meal/bolus behind every
hyper/hypo onset that has no logged cause, SIZED FROM THE CGM SWING IT MUST
EXPLAIN — so an announced-event eval on the augmented record is the ORACLE UPPER
BOUND of the announced regime (the score attainable if every event were logged
and declared), not a deployment number.

We compare, all recomputed here on the live best checkpoint (the model is always
conditioned — each window announces its true future carbs/insulin):
  real      (the unaugmented segments)
  aug       (augment_segment'd segments)
plus the per-patch ΔBG direction/amplitude (amp_var) for real vs aug, the cleanest
read of whether announcing the reconstructed cause recovers the forecast direction
the orphan excursions denied it.

Writes metrics/augexp.json.
"""
from __future__ import annotations
import os, sys, json
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
from metrics.augmented.augment import augment_segment
from curves import load_model, split_segments
import amp_var as AV
import figstyle as F
from figstyle import plt

F.style()

DATASETS = ('ohiot1dm', 'azt1d', 'shanghai')
CAL_STRIDE, CAL_CAP = 8, 24
TEST_STRIDE, TEST_CAP = 4, 60
HS = ('30', '60', '120')


def headline(model, stats, device, cal, test):
    # cal split is collected for parity with the deployment harness, but there is
    # no per-patient calibration to fit on cal under the risk-space design, so the
    # suite is computed directly on the test windows. The fan is passed, so the
    # level metrics are BAND-SCORED (truth projected onto the
    # τ=METRIC_BAND_TAU_LO/_HI band), as in the three formal reports. The model is
    # always conditioned: each window announces its true future carbs/insulin.
    collect_windows(model, stats, cal, device, stride_patches=CAL_STRIDE,
                    max_per_patient=CAL_CAP, conditional=True)
    test_w = collect_windows(model, stats, test, device, stride_patches=TEST_STRIDE,
                             max_per_patient=TEST_CAP, conditional=True)
    pred, true, last_bg, pats = forecast_windows(test_w)
    m = compute_suite(pred, true, last_bg, pats, bands=forecast_bands(test_w))   # keyed by int horizon
    return {h: {'rmse': m[int(h)]['rmse_point'], 'skill': 100 * m[int(h)]['skill_point'],
                'mard': m[int(h)]['mard'], 'clarkeA': m[int(h)]['clarke_A'],
                'hypoR': m[int(h)]['hypo']['recall'], 'hypoP': m[int(h)]['hypo']['precision'],
                'hyperR': m[int(h)]['hyper']['recall'], 'hyperP': m[int(h)]['hyper']['precision']}
            for h in HS}


def _fr(x):
    return '—' if x is None else f"{x:.2f}"


SHAPE_KEYS = (('direction_corr', 'direction corr'), ('g_star', 'g*'),
              ('amp_ratio', 'amplitude ratio'))


def _figure(out: dict, step) -> str:
    """As-logged against reconstructed, one row per cohort × metric.

    The oracle bound is a before→after on the same windows, so it takes the same
    dumbbell form as the lag probe and reads on the same axis conventions.
    """
    dss = [d for d in DATASETS if d in out]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for ax, rows, title, xlabel, fmt in (
            (axes[0],
             [(f'{F.label(d)} · {name}', out[d]['direction']['real_cond'][key],
               out[d]['direction']['aug_cond'][key])
              for d in dss for key, name in SHAPE_KEYS],
             'per-patch ΔBG shape', 'dimensionless', '{:.2f}'),
            (axes[1],
             [(f'{F.label(d)} · @{h}m', out[d]['headline']['real_cond'][h]['rmse'],
               out[d]['headline']['aug_cond'][h]['rmse'])
              for d in dss for h in HS],
             'forecast error', 'RMSE (mg/dL)', '{:.1f}'),
            (axes[2],
             [(f'{F.label(d)} · {name}', out[d]['inject'][key][0], out[d]['inject'][key][1])
              for d in dss for key, name in (('carb_events', 'carb events'),
                                             ('bolus_events', 'bolus events'))],
             'what reconstruction injected', 'events in the test split', '{:.0f}')):
        F.dumbbell_rows(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], fmt)
        ax.set_title(title); ax.set_xlabel(xlabel)
    fig.suptitle(f'Augmentation oracle upper bound (step {step})',
                 x=0.005, ha='left', fontsize=12, color=F.INK)
    F.pair_legend(fig, 'as logged', 'reconstructed',
                  loc='upper right', bbox_to_anchor=(0.995, 0.995), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return F.save(fig, 'augexp.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = load_model(device)
    model = model.to(device); model.eval()
    print(f"[augexp] model step={step} device={device}")
    out = {'_meta': {'step': step}}
    for ds in DATASETS:
        segs = load_dataset(ds)
        cal, test = split_segments(segs, ds)
        cal_a = [augment_segment(s) for s in cal]
        test_a = [augment_segment(s) for s in test]
        # how much was injected into the test split
        c_b = sum(int((s.carb_grams > 0).sum()) for s in test)
        c_a = sum(int((s.carb_grams > 0).sum()) for s in test_a)
        b_b = sum(int((s.bolus_units > 0).sum()) for s in test)
        b_a = sum(int((s.bolus_units > 0).sum()) for s in test_a)

        real_c = headline(model, stats, device, cal, test)
        aug_c = headline(model, stats, device, cal_a, test_a)

        dir_real_c = AV.run(model, stats, device, test)['bg_per_patch_dbg']
        dir_aug_c = AV.run(model, stats, device, test_a)['bg_per_patch_dbg']

        out[ds] = {'inject': {'carb_events': [c_b, c_a], 'bolus_events': [b_b, b_a]},
                   'headline': {'real_cond': real_c, 'aug_cond': aug_c},
                   'direction': {'real_cond': dir_real_c, 'aug_cond': dir_aug_c}}

        print(f"\n== {ds} ==  carb events {c_b}->{c_a}  bolus events {b_b}->{b_a} (test split)")
        print(f"  per-patch ΔBG dir_corr (cond): real {dir_real_c['direction_corr']:.2f} -> aug {dir_aug_c['direction_corr']:.2f}"
              f"   g*: {dir_real_c['g_star']:.2f} -> {dir_aug_c['g_star']:.2f}"
              f"   amp_ratio: {dir_real_c['amp_ratio']:.2f} -> {dir_aug_c['amp_ratio']:.2f}")
        print(f"  {'horizon':>8} | {'RMSE real_c->aug_c':>22} | {'hyperR real_c->aug_c':>22} | {'hypoR real_c->aug_c':>22}")
        for h in HS:
            rc, ac = real_c[h], aug_c[h]
            print(f"  {'@'+h+'m':>8} | {rc['rmse']:8.1f} -> {ac['rmse']:8.1f}      | "
                  f"{_fr(rc['hyperR'])} -> {_fr(ac['hyperR'])}            | "
                  f"{_fr(rc['hypoR'])} -> {_fr(ac['hypoR'])}")
    with open(os.path.join(HERE, 'augexp.json'), 'w') as f:
        json.dump(out, f, indent=2)
    path = _figure(out, step)
    print(f"\n[augexp] wrote augexp.json and {path}")


if __name__ == '__main__':
    main()
