"""
Time-of-day probe on the real cohorts — does the model read the wall-clock from
the glucose/carb/insulin trajectory shape alone?

The model carries an auxiliary per-patch time-of-day probe (``TIME_PROBE_*``): a
head off every prediction-patch hidden state that classifies the hour-of-day into
``TIME_PROBE_N_BINS`` circular bins.  It is diagnostic-only (never feeds the BG
loss or checkpoint selection) but with ``TIME_PROBE_DETACH=False`` it co-trains
the shared trunk, so its accuracy is a readout of how much absolute circadian
phase the forecast representation actually encodes.

Training reports the probe on the (in-domain) simulator validation set; this
script measures the SAME headline metrics on the three real cohorts, where the
Segment's wall-clock ``t0`` supplies the true origin hour.  Each window is scored
conditioned (its true future carbs/insulin/exercise announced), matching the rest of the
real-data eval and the always-conditioned training-validation regime the probe
metrics are defined on.

Metric definitions mirror ``train._run_validation`` byte-for-byte (same ``utils``
circular helpers, patch-0 decode compared to the origin hour).  ``tod_xwin_jump_h``
(the cross-window phase-advance witness) comes for free: consecutive windows are
strided exactly one prediction horizon apart, so each adjacent pair is a
teacher-forced cross-window pair.

Runs on the live best checkpoint (checkpoints/t1dmai_best.pt).  GPU if available.
Writes metrics/time.json and metrics/time_figures/{time_<ds>.png, time_summary.png}.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    PATCH_SIZE, PREDICTION_PATCHES, MAX_CONTEXT_PATCHES, PREDICTION_HORIZON_HOURS,
    TIME_PROBE_ENABLED, TIME_PROBE_N_BINS, TIME_PROBE_BIN_HOURS, CHANNEL_TO_FEAT,
)
from realdata import load_dataset
from realdata.features import build_feature_stack, context_window
from realdata.calibrate import _future_overrides
from curves import load_model, split_segments
from inference import predict
from utils import (
    time_of_day_decode_bins, time_inter_patch_jump_hours, time_cross_window_jump_hours,
    aggregate_origin_belief, _hour_to_unit,
    circular_hour_error, circular_hour_residual, circular_bias_hours, circular_std_hours,
)
from clock_face import draw_clock_axis

# Every window runs the FORECAST protocol: one masked span of PREDICTION_PATCHES
# patches ending at the window's last patch, with the whole context visible — the
# forecast case of the masked-BG objective, not a mode of its own. The probe emits
# one row per MASKED patch, so row j is the patch at d = j + 1 patches from the
# nearest visible evidence, one-sided; the inter-patch clock advance below is the
# spacing of those patches, not of a fixed zone. Row 0 is the forecast origin.
PRED = PREDICTION_PATCHES * PATCH_SIZE                # steps in one masked forecast span
CTX = MAX_CONTEXT_PATCHES * PATCH_SIZE
STRIDE = PREDICTION_PATCHES * PATCH_SIZE              # one span → adjacent windows are a cross-window pair
CAP = 120                                            # windows/patient
ANNOUNCE = (0, 1, 2)                                  # conditioned: carb + insulin + exercise
# Every announceable channel is announced, and that is checked rather than left to
# read correctly: an announced set short of ``CHANNEL_TO_FEAT`` leaves the dropped
# slot at ``normalize(0)``, which for exercise_equiv is a legal "no session" value
# (−0.139 z on the balanced pool), so the probe is scored in a regime training
# never saw.
assert ANNOUNCE == tuple(CHANNEL_TO_FEAT), (
    f"announced set {ANNOUNCE} != announceable set {tuple(CHANNEL_TO_FEAT)}")
DATASETS = ('ohiot1dm', 'azt1d', 'shanghai')
ADV = PATCH_SIZE * 5.0 / 60.0                         # inter-patch clock advance (0.5 h)
FIGDIR = os.path.join(HERE, 'time_figures')

# Uniform-clock chance references (residual ~ U(-12, 12]).
CHANCE = dict(mae_h=6.0, acc_1h=100.0 * 2.0 / 24.0, acc_2h=100.0 * 4.0 / 24.0,
              acc_bin=25.0, p90_h=0.9 * 12.0, gross_rate=100.0 * 18.0 / 24.0)

DS_LABEL = {'ohiot1dm': 'OhioT1DM', 'azt1d': 'AZT1D', 'shanghai': 'ShanghaiT1DM'}


class Accum:
    """Per-sample time-of-day readouts for one cohort (arrays, finalized once)."""

    def __init__(self) -> None:
        self.pred_hour: list[float] = []
        self.true_hour: list[float] = []
        self.R: list[float] = []
        self.jump: list[float] = []          # per-patch inter-patch advance deviation (h)
        self.xwin: list[float] = []          # cross-window advance deviation (h)
        self.beliefs: list[np.ndarray] = []  # (n_bins,) fused origin belief per window

    def add(self, pred_hour: float, true_hour: float, R: float, jump: float,
            belief: np.ndarray) -> None:
        self.pred_hour.append(pred_hour)
        self.true_hour.append(true_hour)
        self.R.append(R)
        self.jump.append(jump)
        self.beliefs.append(belief)

    def __len__(self) -> int:
        return len(self.pred_hour)


def collect(model, stats, device, segs) -> Accum:
    """Slide one-horizon windows over each test segment, reading the probe per window."""
    acc = Accum()
    for seg in segs:
        n = (len(seg) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX + PRED:
            continue
        feats = build_feature_stack(seg, stats)
        hod = seg.hour_of_day()
        cnt = 0
        prev_ps = None
        prev_tp = None
        for ps in range(CTX, n - PRED + 1, STRIDE):
            if cnt >= CAP:
                break
            cnt += 1
            ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)
            ov = _future_overrides(feats, ps, ANNOUNCE)
            out = predict(model, ctx, normalization_stats=stats, device=device,
                          overrides=ov, return_time=True)
            # One row per MASKED patch, in mask order — not a fixed trailing zone.
            tp = out.get('time_pred')                    # (masked patches, n_bins) bin logits
            if tp is None:                               # probe disabled ⇒ nothing to score
                return acc
            tp = tp.detach().cpu()
            # Row 0 is the FIRST masked patch, the forecast origin under this protocol.
            h0, r0 = time_of_day_decode_bins(tp[0:1, :], TIME_PROBE_N_BINS)
            jump = float(time_inter_patch_jump_hours(
                tp.unsqueeze(0), TIME_PROBE_N_BINS, ADV).item())
            probs = torch.softmax(tp, dim=-1).numpy()    # (P, n_bins)
            belief = aggregate_origin_belief(probs, ADV, TIME_PROBE_BIN_HOURS)
            acc.add(float(h0.item()), float(hod[ps]), float(r0.item()), jump, belief)
            # Cross-window witness: the immediately-preceding window sits exactly one
            # horizon earlier (STRIDE == PRED), so (prev, cur) is a teacher-forced pair.
            if prev_tp is not None and prev_ps is not None and ps - prev_ps == PRED:
                acc.xwin.append(float(time_cross_window_jump_hours(
                    prev_tp.unsqueeze(0), tp.unsqueeze(0),
                    TIME_PROBE_N_BINS, PREDICTION_HORIZON_HOURS).item()))
            prev_ps, prev_tp = ps, tp
    return acc


def finalize(acc: Accum) -> dict:
    """Headline probe metrics from accumulated arrays (mirrors train._run_validation)."""
    if len(acc) == 0:
        return {'n': 0}
    ph = torch.tensor(acc.pred_hour, dtype=torch.float32)
    th = torch.tensor(acc.true_hour, dtype=torch.float32)
    R = torch.tensor(acc.R, dtype=torch.float32)
    ae = circular_hour_error(ph, th)                     # (N,) in [0, 12]
    pb = (ph // 6).long() % 4
    tb = (th // 6).long() % 4
    hi = R >= R.median()
    out = {
        'n': len(acc),
        'tod_mae_h': float(ae.mean()),
        'tod_acc_1h': float(100.0 * (ae <= 1.0).float().mean()),
        'tod_acc_2h': float(100.0 * (ae <= 2.0).float().mean()),
        'tod_acc_bin': float(100.0 * (pb == tb).float().mean()),
        'tod_conf': float(R.mean()),
        'tod_bias_h': float(circular_bias_hours(ph, th)),
        'tod_std_h': float(circular_std_hours(ph, th)),
        'tod_p90_h': float(torch.quantile(ae, 0.9)),
        'tod_gross_rate': float(100.0 * (ae > 3.0).float().mean()),
        'tod_mae_hiconf': float(ae[hi].mean()) if bool(hi.any()) else float(ae.mean()),
        'tod_jump_h': float(torch.tensor(acc.jump).mean()) if acc.jump else None,
        'tod_xwin_jump_h': float(torch.tensor(acc.xwin).mean()) if acc.xwin else None,
    }
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _fig_dataset(ds: str, acc: Accum, m: dict, path: str) -> None:
    """Per-cohort figure: pred-vs-true density, residual histogram, reliability,
    and a gallery of aggregated origin-belief clock dials."""
    ph = np.asarray(acc.pred_hour)
    th = np.asarray(acc.true_hour)
    R = np.asarray(acc.R)
    resid = ((ph - th + 12.0) % 24.0) - 12.0             # signed circular residual (h)
    ae = np.minimum(np.abs(ph - th) % 24.0, 24.0 - np.abs(ph - th) % 24.0)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.28)
    fig.suptitle(f"{DS_LABEL[ds]} — time-of-day probe on real CGM  "
                 f"(n={m['n']}, MAE={m['tod_mae_h']:.2f} h, R={m['tod_conf']:.2f})",
                 fontsize=13, y=0.98)

    # (0,0) predicted vs true origin hour — density + identity diagonal.
    ax = fig.add_subplot(gs[0, 0])
    hb = ax.hexbin(th, ph, gridsize=24, extent=(0, 24, 0, 24), cmap='viridis', mincnt=1)
    ax.plot([0, 24], [0, 24], color='0.7', lw=1.0, ls='--', zorder=3)
    ax.set_xlim(0, 24); ax.set_ylim(0, 24)
    ax.set_xticks(range(0, 25, 6)); ax.set_yticks(range(0, 25, 6))
    ax.set_xlabel('true origin hour'); ax.set_ylabel('predicted origin hour')
    ax.set_title('predicted vs true clock')
    fig.colorbar(hb, ax=ax, shrink=0.8, label='windows')

    # (0,1) signed circular residual histogram — bias & spread.
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(resid, bins=np.linspace(-12, 12, 49), color='C0', alpha=0.85)
    ax.axvline(0.0, color='0.6', lw=1.0, ls='--')
    ax.axvline(m['tod_bias_h'], color='C3', lw=1.4,
               label=f"bias {m['tod_bias_h']:+.2f} h\nsd {m['tod_std_h']:.2f} h")
    ax.set_xlim(-12, 12)
    ax.set_xlabel('predicted − true (h, circular)'); ax.set_ylabel('windows')
    ax.set_title('signed clock residual'); ax.legend(fontsize=9, loc='upper right')

    # (0,2) reliability: MAE by confidence (R) decile — is R a usable trust gate?
    ax = fig.add_subplot(gs[0, 2])
    order = np.argsort(R)
    if len(R) >= 10:
        chunks = np.array_split(order, 10)
        xr = [float(R[c].mean()) for c in chunks]
        ym = [float(ae[c].mean()) for c in chunks]
        ax.plot(xr, ym, marker='o', color='C2')
    else:
        ax.scatter(R, ae, s=8, color='C2', alpha=0.5)
    ax.axhline(CHANCE['mae_h'], color='0.6', lw=1.0, ls='--', label='chance (6 h)')
    ax.set_xlabel('confidence R (resultant length)'); ax.set_ylabel('MAE (h)')
    ax.set_title('reliability: error vs confidence'); ax.legend(fontsize=9)
    ax.set_ylim(bottom=0.0)

    # (1, :) clock-dial gallery of the fused origin belief, spanning the day.
    ncol = 8
    gg = gs[1, :].subgridspec(1, ncol, wspace=0.05)
    if len(acc) > 0:
        # pick windows whose TRUE origin hour tiles the 24 h clock most evenly
        targets = (np.arange(ncol) + 0.5) * (24.0 / ncol)
        idx = [int(np.argmin(np.minimum(np.abs(th - t), 24.0 - np.abs(th - t))))
               for t in targets]
        for j, i in enumerate(idx):
            cax = fig.add_subplot(gg[0, j])
            draw_clock_axis(cax, acc.beliefs[i], show_hand=True, wedge_color='C0')
            tv = _hour_to_unit(np.array(th[i]))          # red true-hour tick (y-up unit)
            cax.plot([0.86 * tv[0], 1.02 * tv[0]], [0.86 * tv[1], 1.02 * tv[1]],
                     color='C3', lw=1.6, zorder=4)
            cax.set_title(f"t={th[i]:.1f}\np={ph[i]:.1f}", fontsize=6, pad=1)
    fig.text(0.5, 0.015, 'aggregated origin-belief dials (blue wedges = belief, grey hand = '
             'resultant, red tick = true hour); hour 0 at top, clockwise',
             ha='center', fontsize=8, color='0.4')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _fig_summary(results: dict, path: str) -> None:
    """Cross-cohort headline bars vs uniform-clock chance."""
    dss = [d for d in DATASETS if results.get(d, {}).get('n', 0) > 0]
    if not dss:
        return
    fig, (axh, axp) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(dss))

    # hour-scale metrics
    hkeys = [('tod_mae_h', 'MAE'), ('tod_std_h', 'sd'), ('tod_p90_h', 'p90')]
    w = 0.8 / len(hkeys)
    for k, (key, lab) in enumerate(hkeys):
        axh.bar(x + (k - (len(hkeys) - 1) / 2) * w,
                [results[d][key] for d in dss], w, label=lab)
    axh.axhline(CHANCE['mae_h'], color='0.5', ls='--', lw=1.0, label='chance MAE (6 h)')
    axh.set_xticks(x); axh.set_xticklabels([DS_LABEL[d] for d in dss])
    axh.set_ylabel('hours'); axh.set_title('clock error (lower is better)')
    axh.legend(fontsize=9)

    # percentage metrics
    pkeys = [('tod_acc_1h', '±1 h'), ('tod_acc_2h', '±2 h'), ('tod_acc_bin', '6 h-bin')]
    w = 0.8 / len(pkeys)
    for k, (key, lab) in enumerate(pkeys):
        axp.bar(x + (k - (len(pkeys) - 1) / 2) * w,
                [results[d][key] for d in dss], w, label=lab)
    for lvl, lab in ((CHANCE['acc_1h'], '±1 h'), (CHANCE['acc_2h'], '±2 h'),
                     (CHANCE['acc_bin'], 'bin')):
        axp.axhline(lvl, color='0.6', ls=':', lw=0.9)
    axp.set_xticks(x); axp.set_xticklabels([DS_LABEL[d] for d in dss])
    axp.set_ylabel('%'); axp.set_title('accuracy vs chance (dotted)')
    axp.legend(fontsize=9)

    fig.suptitle('Time-of-day probe on real CGM — headline metrics', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = load_model(device)
    model = model.to(device); model.eval()
    print(f"[time] model step={step} device={device}  probe_enabled={TIME_PROBE_ENABLED} "
          f"n_bins={TIME_PROBE_N_BINS}")
    if not TIME_PROBE_ENABLED:
        print("[time] TIME_PROBE_ENABLED is False — the probe head is not built; nothing to score.")

    os.makedirs(FIGDIR, exist_ok=True)
    out = {'_meta': {
        'step': step, 'regime': 'conditioned', 'probe_enabled': bool(TIME_PROBE_ENABLED),
        'n_bins': TIME_PROBE_N_BINS, 'bin_hours': TIME_PROBE_BIN_HOURS,
        'advance_hours': ADV, 'horizon_hours': PREDICTION_HORIZON_HOURS,
        'stride_patches': PREDICTION_PATCHES, 'cap_per_patient': CAP,
        'chance': CHANCE,
        # The protocol and the bin, recorded beside the numbers: a right-edge
        # masked span, scored off its rows, so d runs 1..PREDICTION_PATCHES
        # one-sided. The scalar clock is read at row 0, d = 1.
        'protocol': 'forecast (masked span at the last patch)',
        'd_patches': list(range(1, PREDICTION_PATCHES + 1)),
        'one_sided': True,
    }}
    accs: dict[str, Accum] = {}
    for ds in DATASETS:
        segs = load_dataset(ds)
        _, test = split_segments(segs, ds)
        acc = collect(model, stats, device, test)
        accs[ds] = acc
        out[ds] = finalize(acc)
        m = out[ds]
        print(f"\n== {ds} ==  (n={m['n']})")
        if m['n']:
            fr = lambda v: '—' if v is None else f"{v:.2f}"
            print(f"  MAE={m['tod_mae_h']:.2f} h  acc±1h={m['tod_acc_1h']:.1f}%  "
                  f"acc±2h={m['tod_acc_2h']:.1f}%  4-bin={m['tod_acc_bin']:.1f}%  R={m['tod_conf']:.2f}")
            print(f"  bias={m['tod_bias_h']:+.2f} h  sd={m['tod_std_h']:.2f} h  "
                  f"p90={m['tod_p90_h']:.2f} h  gross>3h={m['tod_gross_rate']:.1f}%  "
                  f"hi-conf MAE={m['tod_mae_hiconf']:.2f} h")
            print(f"  jump={fr(m['tod_jump_h'])} h  xwin jump={fr(m['tod_xwin_jump_h'])} h")
            _fig_dataset(ds, acc, m, os.path.join(FIGDIR, f"time_{ds}.png"))

    # pooled across cohorts
    pooled = Accum()
    for acc in accs.values():
        pooled.pred_hour += acc.pred_hour; pooled.true_hour += acc.true_hour
        pooled.R += acc.R; pooled.jump += acc.jump; pooled.xwin += acc.xwin
        pooled.beliefs += acc.beliefs
    out['pooled'] = finalize(pooled)
    if out['pooled'].get('n'):
        pm = out['pooled']
        print(f"\n== pooled ==  (n={pm['n']})  MAE={pm['tod_mae_h']:.2f} h  "
              f"acc±1h={pm['tod_acc_1h']:.1f}%  R={pm['tod_conf']:.2f}")

    _fig_summary(out, os.path.join(FIGDIR, 'time_summary.png'))
    with open(os.path.join(HERE, 'time.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[time] wrote time.json and figures under {os.path.relpath(FIGDIR, ROOT)}/")


if __name__ == '__main__':
    main()
