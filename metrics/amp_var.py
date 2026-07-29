"""
Quantitative backing for the realeval write-up:

Q3 — amplitude of the model's BG predictions vs the real signal, scored on the
     per-PATCH (30-min) ΔBG (the granularity the eval uses). The model is always
     conditioned — each window announces its true future carbs/insulin.

Q4 — inter-patch variability: the typical 30-min patch-to-patch BG move
     (std of true per-patch ΔBG), pooled and as a per-patient distribution
     (the inter-patient spread of that variability).

Risk-space redesign note: the per-channel carb/insulin amplitude analysis (the
old ``mu_raw.reshape(-1, 4)`` predicted-dynamics-channel std vs the true
absorption/action curves) was DROPPED — carbs/insulin/IS/HGO are no longer model
outputs. This script now scores BG amplitude only.

Runs on the live best checkpoint (checkpoints/t1dmai_best.pt). GPU if available.
Writes metrics/amp_var.json.
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

from config import PATCH_SIZE, PREDICTION_PATCHES
from realdata import load_dataset
from realdata.features import build_feature_stack, context_window
from realdata.calibrate import _future_overrides
from curves import load_model, split_segments
from inference import predict
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
from config import MAX_CONTEXT_PATCHES
import figstyle as F
from figstyle import plt

F.style()

PRED = PREDICTION_PATCHES * PATCH_SIZE
CTX = MAX_CONTEXT_PATCHES * PATCH_SIZE
DATASETS = ('ohiot1dm', 'azt1d', 'shanghai')
STRIDE = 4 * PATCH_SIZE
CAP = 80                                   # windows/patient


def _std(sumsq, s, n):
    return float(np.sqrt(max(sumsq / n - (s / n) ** 2, 0.0)))


def run(model, stats, device, segs, announce=(0, 1), pairs_out=None):
    """Pooled per-patch ΔBG statistics over ``segs``.

    ``pairs_out``, when a list is passed, additionally collects the raw
    ``(true_dbg, pred_dbg)`` arrays the figure scatters — kept off the return value
    so the callers that consume only the summary (shift15, augexp) are untouched,
    and out of the JSON, which stays a summary.
    """
    # pooled accumulators for per-patch ΔBG (BG amplitude + direction). The model
    # is always conditioned — each window announces its true future carbs/insulin.
    bg = dict(pp=0.0, pt=0.0, tt=0.0, p=0.0, t=0.0, n=0)
    per_patient_dbg: dict[str, list] = {}
    for seg in segs:
        n = (len(seg) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX + PRED:
            continue
        feats = build_feature_stack(seg, stats)
        # Scored ground-truth BG is the raw (bg-clamped) CGM (one space) — both the
        # ``last_bg`` anchor and the scored future.
        cgm_s = np.clip(seg.cgm, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(float)
        cnt = 0
        for ps in range(CTX, n - PRED + 1, STRIDE):
            if cnt >= CAP:
                break
            cnt += 1
            ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)
            ov = _future_overrides(feats, ps, announce)
            out = predict(model, ctx, normalization_stats=stats, device=device, overrides=ov)
            last_bg = float(cgm_s[ps - 1])
            cgm = cgm_s[ps:ps + PRED]
            pred_bg = out['median_bg'].flatten().cpu().numpy()   # headline forecast
            # per-patch (30-min) ΔBG, anchored at last_bg
            te = cgm[PATCH_SIZE - 1::PATCH_SIZE]
            pe = pred_bg[PATCH_SIZE - 1::PATCH_SIZE]
            tprev = np.concatenate([[last_bg], te[:-1]])
            pprev = np.concatenate([[last_bg], pe[:-1]])
            td, pd = te - tprev, pe - pprev
            bg['pp'] += float(np.sum(pd * pd)); bg['tt'] += float(np.sum(td * td))
            bg['pt'] += float(np.sum(pd * td)); bg['p'] += float(np.sum(pd)); bg['t'] += float(np.sum(td))
            bg['n'] += len(td)
            per_patient_dbg.setdefault(seg.patient, []).extend(td.tolist())
            if pairs_out is not None:
                pairs_out.append((td, pd))
    # finalize
    mp, mt = bg['p'] / bg['n'], bg['t'] / bg['n']
    sp = _std(bg['pp'], bg['p'], bg['n']); st = _std(bg['tt'], bg['t'], bg['n'])
    cov = bg['pt'] / bg['n'] - mp * mt
    corr = cov / (sp * st) if sp > 0 and st > 0 else float('nan')
    gstar = bg['pt'] / bg['pp'] if bg['pp'] > 0 else float('nan')   # OLS slope
    bg_out = dict(std_true=st, std_pred=sp, amp_ratio=sp / st if st else float('nan'),
                  direction_corr=corr, g_star=gstar, n_patch=bg['n'])
    # inter-patient spread of inter-patch ΔBG variability
    pstd = sorted(float(np.std(v)) for v in per_patient_dbg.values() if len(v) >= 8)
    arr = np.array(pstd)
    inter_patient = dict(
        n_patients=len(pstd),
        per_patient_std_dbg_mgdl=dict(
            min=float(arr.min()), p25=float(np.percentile(arr, 25)),
            median=float(np.median(arr)), p75=float(np.percentile(arr, 75)),
            max=float(arr.max()), across_patient_std=float(arr.std())),
    )
    return dict(bg_per_patch_dbg=bg_out, inter_patch_variability=inter_patient)


def _panel_scatter(ax, ds: str, pairs: list, b: dict) -> None:
    """Predicted against true per-patch ΔBG, as a density — 4k points would over-plot.

    Density is a magnitude, so it rides one hue light→dark; the identity line and
    the RMSE-optimal fit are drawn over it in ink rather than in a series colour,
    since neither is a series.
    """
    t = np.concatenate([p[0] for p in pairs]); p = np.concatenate([p[1] for p in pairs])
    lim = float(np.percentile(np.abs(np.concatenate([t, p])), 99.5))
    ax.hexbin(t, p, gridsize=34, extent=(-lim, lim, -lim, lim), mincnt=1,
              cmap=F.density_cmap(), linewidths=0.0)
    ax.plot([-lim, lim], [-lim, lim], color=F.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.annotate('identity', xy=(lim, lim), xytext=(-6, -14), textcoords='offset points',
                ha='right', va='top', fontsize=7.5, color=F.MUTED)
    g = b['g_star']
    ax.plot([-lim, lim], [-lim * g, lim * g], color=F.INK2, lw=1.4, zorder=4)
    ax.annotate(f"g* = {g:.2f}", xy=(lim, lim * g), xytext=(-3, 4), textcoords='offset points',
                ha='right', fontsize=8, color=F.INK2)
    ax.set_title(f"{F.label(ds)} — corr {b['direction_corr']:.2f}, "
                 f"amplitude {b['amp_ratio']:.2f}×")
    ax.set_xlabel('true 30-min ΔBG (mg/dL)'); ax.set_ylabel('predicted 30-min ΔBG (mg/dL)')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')


def _figure(res: dict, pairs: dict, step) -> str:
    dss = [d for d in DATASETS if d in res]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    for k, d in enumerate(dss):
        _panel_scatter(axes[0, k], d, pairs[d], res[d]['bg_per_patch_dbg'])

    ax = axes[1, 0]
    keys = (('amp_ratio', 'amplitude ratio'), ('direction_corr', 'direction corr'),
            ('g_star', 'g*'))
    w = 0.2
    for k, d in enumerate(dss):
        vals = [res[d]['bg_per_patch_dbg'][key] for key, _ in keys]
        bars = ax.bar(np.arange(len(keys)) + (k - 1) * w, vals, width=w - 0.04,
                      color=F.cohort_color(d), edgecolor=F.SURFACE, linewidth=1.2)
        F.bar_labels(ax, bars)
    F.ygrid(ax)
    ax.set_title('per-patch ΔBG, model vs signal')
    ax.set_ylabel('dimensionless'); ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([lab for _, lab in keys])

    ax = axes[1, 1]
    for k, d in enumerate(dss):
        pp = res[d]['inter_patch_variability']['per_patient_std_dbg_mgdl']
        c = F.cohort_color(d)
        ax.plot([pp['min'], pp['max']], [k, k], color=c, lw=2.0, solid_capstyle='round', zorder=2)
        ax.plot([pp['p25'], pp['p75']], [k, k], color=c, lw=6.0, alpha=0.35,
                solid_capstyle='butt', zorder=2)
        ax.scatter([pp['median']], [k], s=54, color=c, zorder=3,
                   edgecolor=F.SURFACE, linewidth=2)
        ax.annotate(f"{pp['median']:.1f}", xy=(pp['median'], k), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=7.5, color=F.INK2)
    ax.set_title('per-patient volatility (min · IQR · median · max)')
    ax.set_xlabel('std of true 30-min ΔBG (mg/dL)')
    ax.set_yticks(range(len(dss)))
    ax.set_yticklabels([f"{F.label(d)}  (n={res[d]['inter_patch_variability']['n_patients']})"
                        for d in dss])
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_ylim(-0.6, len(dss) - 0.4)

    ax = axes[1, 2]
    w = 0.3
    for k, (key, name, color) in enumerate((('std_true', 'true', F.PAIR[0]),
                                            ('std_pred', 'forecast', F.PAIR[1]))):
        vals = [res[d]['bg_per_patch_dbg'][key] for d in dss]
        bars = ax.bar(np.arange(len(dss)) + (k - 0.5) * w, vals, width=w - 0.04,
                      color=color, edgecolor=F.SURFACE, linewidth=1.2, label=name)
        F.bar_labels(ax, bars, '{:.1f}')
    F.ygrid(ax)
    ax.set_title('30-min ΔBG amplitude')
    ax.set_ylabel('std (mg/dL)'); ax.margins(y=0.2)
    ax.set_xticks(range(len(dss))); ax.set_xticklabels([F.label(d) for d in dss])
    F.legend(ax, loc='upper right')

    fig.suptitle(f'Prediction-vs-true amplitude and inter-patch variability (step {step})',
                 x=0.006, ha='left', fontsize=12, color=F.INK)
    F.cohort_legend(fig, dss, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return F.save(fig, 'amp_var.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = load_model(device)
    model = model.to(device); model.eval()
    print(f"[amp_var] model step={step} device={device}")
    res = {'_meta': {'step': step, 'regime': 'conditioned'}}
    pairs = {}
    for ds in DATASETS:
        segs = load_dataset(ds)
        _, test = split_segments(segs, ds)
        pairs[ds] = []
        r = run(model, stats, device, test, pairs_out=pairs[ds])
        res[ds] = r
        b = r['bg_per_patch_dbg']; iv = r['inter_patch_variability']
        print(f"\n== {ds} ==")
        print(f"  BG ΔBG/patch: std_true={b['std_true']:.1f}  std_pred={b['std_pred']:.1f}  "
              f"amp_ratio={b['amp_ratio']:.2f}  dir_corr={b['direction_corr']:.2f}  g*={b['g_star']:.2f}  (n={b['n_patch']})")
        pp = iv['per_patient_std_dbg_mgdl']
        print(f"  inter-patch ΔBG std (mg/dL): pooled_true={b['std_true']:.1f}  | per-patient "
              f"min/med/max={pp['min']:.1f}/{pp['median']:.1f}/{pp['max']:.1f} "
              f"across-patient std={pp['across_patient_std']:.1f}  (n_pat={iv['n_patients']})")
    with open(os.path.join(HERE, 'amp_var.json'), 'w') as f:
        json.dump(res, f, indent=2)
    path = _figure(res, pairs, step)
    print(f"\n[amp_var] wrote amp_var.json and {path}")


if __name__ == '__main__':
    main()
