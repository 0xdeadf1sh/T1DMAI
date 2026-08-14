"""
Q2 — when an excursion IS explained by a logged event, does the model respond?

For each 2 h window over the test split, split TRUE excursions by whether the
causal input is logged (hyper↔carb, hypo↔bolus), and measure the model's
detection recall (does the forecast band edge cross the same threshold anywhere
in the window). The model is always conditioned: the window's true future
carbs+insulin are announced (overrides).

Detection fires off the forecast band edges of the same ``predict`` call — the
τ=``METRIC_BAND_TAU_HI`` upper edge for hyper, the τ=``METRIC_BAND_TAU_LO`` lower
edge for hypo — matching train.py's band-edge hypo/hyper detectors rather than a
median crossing. The TRUE-excursion side stays on the clamped CGM.

The contrast (event vs no-event subset) shows whether the model's forecast band
actually reacts to the declared event.
Writes metrics/q2_event_response.json.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)                              # curves (local load_model/split_segments)
torch.set_num_threads(8)

from config import (PATCH_SIZE, PREDICTION_PATCHES, MAX_CONTEXT_PATCHES,
                    BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD, CHANNEL_TO_FEAT,
                    QUANTILE_LEVELS, N_QUANTILES,
                    METRIC_BAND_TAU_LO, METRIC_BAND_TAU_HI)
from realdata import load_dataset
from realdata.features import build_feature_stack, context_window
from realdata.calibrate import _future_overrides
from curves import load_model, split_segments
from inference import predict
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
import figstyle as F
from figstyle import plt

F.style()

# Every window runs the FORECAST protocol: one masked span of PREDICTION_PATCHES
# patches ending at the window's last patch, with the whole context visible — the
# forecast case of the masked-BG objective, not a mode of its own. ``PRED`` is that
# span in steps. Detection scans the whole span, so the recall reported here pools
# d = 1..PREDICTION_PATCHES one-sided (``realdata.run_eval.horizon_d_patches`` is
# the single derivation of d): it is a per-window "does the band cross anywhere"
# call, not a per-horizon number, and it is never split by span length or by arm.
PRED = PREDICTION_PATCHES * PATCH_SIZE
CTX = MAX_CONTEXT_PATCHES * PATCH_SIZE
DATASETS = ('ohiot1dm', 'azt1d', 'shanghai')
STRIDE = 8 * PATCH_SIZE
CAP = 60
ANNOUNCE = (0, 1, 2)                       # carb, insulin, exercise
# Every announceable channel is announced, and that is checked rather than left to
# read correctly: an announced set short of ``CHANNEL_TO_FEAT`` leaves the dropped
# slot at ``normalize(0)``, which for exercise_equiv is a legal "no session" value
# (−0.139 z on the balanced pool), so the recall contrast is measured in a regime
# training never saw.
assert ANNOUNCE == tuple(CHANNEL_TO_FEAT), (
    f"announced set {ANNOUNCE} != announceable set {tuple(CHANNEL_TO_FEAT)}")
_BAND_LO_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_LO)
_BAND_HI_IDX = QUANTILE_LEVELS.index(METRIC_BAND_TAU_HI)


def _rec(d):
    return None if d['n'] == 0 else round(100 * d['hit'] / d['n'], 1)


def run(model, stats, device, segs):
    # recall buckets: [side][has_event?] -> {hit,n}; always conditioned.
    B = {side: {ev: {'hit': 0, 'n': 0} for ev in ('with', 'without')}
         for side in ('hyper', 'hypo')}
    for seg in segs:
        n = (len(seg) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX + PRED:
            continue
        feats = build_feature_stack(seg, stats)
        # Scored ground-truth BG (the true-excursion side) is the raw (bg-clamped)
        # CGM — one space. Logged carb/bolus presence stays raw.
        cgm_s = np.clip(seg.cgm, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(float)
        cnt = 0
        for ps in range(CTX, n - PRED + 1, STRIDE):
            if cnt >= CAP:
                break
            cnt += 1
            ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)
            fut = cgm_s[ps:ps + PRED]
            carb_in = float(np.sum(seg.carb_grams[ps:ps + PRED])) > 0
            bol_in = float(np.sum(seg.bolus_units[ps:ps + PRED])) > 0
            true_hyper = bool(np.any(fut > BG_HYPER_THRESHOLD))
            true_hypo = bool(np.any(fut < BG_HYPO_THRESHOLD))
            if not (true_hyper or true_hypo):
                continue
            out_p = predict(model, ctx, normalization_stats=stats, device=device,
                            overrides=_future_overrides(feats, ps, ANNOUNCE))
            bands = out_p['bands'].reshape(-1, N_QUANTILES).cpu().numpy()   # (PRED, K) mg/dL
            assert bands.shape == (PRED, N_QUANTILES), bands.shape
            p_hyper = bool(np.any(bands[:, _BAND_HI_IDX] > BG_HYPER_THRESHOLD))
            p_hypo = bool(np.any(bands[:, _BAND_LO_IDX] < BG_HYPO_THRESHOLD))
            if true_hyper:
                b = B['hyper']['with' if carb_in else 'without']
                b['n'] += 1; b['hit'] += int(p_hyper)
            if true_hypo:
                b = B['hypo']['with' if bol_in else 'without']
                b['n'] += 1; b['hit'] += int(p_hypo)
    out = {}
    for side in ('hyper', 'hypo'):
        out[side] = {ev: {'n': B[side][ev]['n'], 'recall': _rec(B[side][ev])}
                     for ev in ('with', 'without')}
    return out


def _figure(res: dict, step) -> str:
    """Detection recall by side, split on whether the causal event was logged.

    Cohort owns the hue; the event/no-event contrast is carried by position, so a
    reader compares within a pair by proximity rather than by colour.
    """
    dss = [d for d in DATASETS if d in res]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    w = 0.2
    for ax, side, cause in ((axes[0], 'hyper', 'carbohydrate'), (axes[1], 'hypo', 'bolus')):
        for k, d in enumerate(dss):
            vals, ns = [], []
            for ev in ('with', 'without'):
                cell = res[d][side][ev]
                vals.append(cell['recall'] if cell['recall'] is not None else 0.0)
                ns.append(cell['n'])
            bars = ax.bar(np.arange(2) + (k - 1) * w, vals, width=w - 0.04,
                          color=F.cohort_color(d), edgecolor=F.SURFACE, linewidth=1.2)
            for b, v, n in zip(bars, vals, ns):
                ax.annotate(f"{v:.0f}%\nn={n}" if n else '—',
                            xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=7.5, color=F.INK2, linespacing=1.25)
        F.ygrid(ax)
        ax.set_title(f'{side} excursions detected')
        ax.set_ylabel('detection recall (%)'); ax.set_ylim(0, 118)
        ax.set_xticks(range(2))
        ax.set_xticklabels([f'{cause} logged', f'no {cause} logged'])
    fig.suptitle(f'Event-conditioned excursion response (step {step})',
                 x=0.007, ha='left', fontsize=12, color=F.INK)
    F.cohort_legend(fig, dss, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return F.save(fig, 'q2_event_response.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = load_model(device)
    model = model.to(device); model.eval()
    print(f"[q2] model step={step} device={device}")
    # The bin the recalls below are taken over, recorded beside them: a right-edge
    # masked span, scanned whole, so d runs 1..PREDICTION_PATCHES one-sided.
    res = {'_meta': {'step': step,
                     'protocol': 'forecast (masked span at the last patch)',
                     'd_patches_pooled': list(range(1, PREDICTION_PATCHES + 1)),
                     'one_sided': True}}
    for ds in DATASETS:
        segs = load_dataset(ds)
        _, test = split_segments(segs, ds)
        r = run(model, stats, device, test)
        res[ds] = r
        print(f"\n== {ds} ==")
        for side in ('hyper', 'hypo'):
            cause = 'carb' if side == 'hyper' else 'bolus'
            for ev in ('with', 'without'):
                d = r[side][ev]
                print(f"  {side:5} {ev:7}{cause:5}: n={d['n']:4}  recall={d['recall']}")
    with open(os.path.join(HERE, 'q2_event_response.json'), 'w') as f:
        json.dump(res, f, indent=2)
    path = _figure(res, step)
    print(f"\n[q2] wrote q2_event_response.json and {path}")


if __name__ == '__main__':
    main()
