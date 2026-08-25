"""48 h BG day figures (2 h and 8 h forecasts, conditioned) on fresh T1DMSIM patients.

No event markers — the simulator's combined insulin carries no discrete events.
Reuses ``day_curves.plot_day``; sim bridge is metrics/sim/sim_data.py. Writes metrics/sim/figures/sim_day{k}.png.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)                                   # day_curves
sys.path.insert(0, os.path.join(ROOT, 'metrics', 'sim'))  # sim_data

import day_curves as CV                  # the shared 48 h figure machinery
import sim_data as SD
import protocols as PR
from config import TIME_PROBE_ENABLED, TIME_PROBE_N_BINS
from metrics.core.features import context_window
from metrics.core.calibrate import _future_overrides
from day_curves import _pick_day_start
from inference import predict, predict_rolling
from utils import time_of_day_decode_bins

# Every forecast runs the FORECAST protocol, masked set from ``metrics.protocols``.
# ``hp // CV.PREDICTION_PATCHES`` is a tile's roll count; past roll 0 the evidence is the previous roll's output.
N_DAYS = 10
SEEDS = list(SD.TEST_SEEDS)[:14]          # held-out sim test patients (≥N_DAYS, one day each)


def sim_bg_sigma(model, stats, runs, horizon_patches,
                 stride_steps=8 * CV.PATCH_SIZE, max_windows=200, report=None):
    """Per-horizon-step ±1σ envelope, mg/dL; each window announces its true future carb/insulin/exercise."""
    H = horizon_patches * CV.PATCH_SIZE
    acc = {'se': np.zeros(H), 'n': np.zeros(H), 'count': 0}
    for pid, d in runs:
        if acc['count'] >= max_windows:
            break
        feats = SD.build_sim_feature_stack(d, stats)
        # scored against the raw bg-clamped CGM
        CV._sigma_accumulate(model, stats, feats, CV.smooth_bg_truth(d['bg_observed']),
                             horizon_patches, stride_steps, acc, max_windows,
                             report=report)
    return np.sqrt(acc['se'] / np.maximum(acc['n'], 1.0))


def sim_day_curves(model, stats, d, ds, pid, sig2=None, sig8=None):
    """BG 2 h/8 h conditioned forecasts tiled across 48 h; events zeroed, so no markers."""
    feats = SD.build_sim_feature_stack(d, stats)
    T = CV.DAY_PATCHES * CV.PATCH_SIZE
    # the plotted truth is the raw bg-clamped CGM
    cgm = CV.smooth_bg_truth(d['bg_observed'])
    hod0 = float(d['hour_of_day'][ds])
    hours = hod0 + np.arange(T) * (5.0 / 60.0)

    ms = PR.forecast_masked_set(CV.MAX_CONTEXT_PATCHES)

    def tile(hp):
        hsteps = hp * CV.PATCH_SIZE
        nr = hp // CV.PREDICTION_PATCHES
        curve = np.full(T, np.nan)
        for c in range(CV.DAY_PATCHES // hp):
            ts = ds + c * hsteps
            ctx = context_window(feats, ts, CV.MAX_CONTEXT_PATCHES)
            if nr == 1:
                ov = _future_overrides(feats, ts, CV.ANNOUNCE)
                out = predict(model, ctx, normalization_stats=stats, overrides=ov,
                              mask_spans=ms.spans)
                bg = CV._protocol_bg(out, ms)
            else:
                fn = CV._make_night_overrides_fn(feats, ts, CV.ANNOUNCE, stats)
                out = predict_rolling(model, ctx, n_rolls=nr, normalization_stats=stats, overrides_fn=fn)
                bg = CV._bg(out)
            curve[c * hsteps:c * hsteps + len(bg)] = bg[:hsteps]
        return curve

    z = np.zeros(T)
    spec = {
        'patient': pid, 'hours': hours, 'true': cgm[ds:ds + T],
        'p2': tile(CV.PREDICTION_PATCHES),
        'p8': tile(CV.H8_PATCHES),
        'tile2_h': [hod0 + c * CV.PRED * (5/60) for c in range(CV.DAY_PATCHES // CV.PREDICTION_PATCHES + 1)],
        'tile8_h': [hod0 + c * CV.H8_PATCHES * CV.PATCH_SIZE * (5/60) for c in range(CV.DAY_PATCHES // CV.H8_PATCHES + 1)],
        'carb_steps': z, 'bolus_steps': z,
        'carb_total': float(np.sum(d['total_carb'][ds:ds + T])),
        'bolus_total': float(np.sum(d['total_insulin'][ds:ds + T])),
    }
    _attach_tod_probe(model, stats, feats, d, ds, spec)
    CV._attach_bg_bands(spec, sig2, sig8)
    return spec


def _attach_tod_probe(model, stats, feats, d, ds, spec):
    """Per-2 h-tile time-of-day probe arrays into ``spec``, aligned with ``spec['tile2_h']``.

    Per-patch ``(P, TIME_PROBE_N_BINS)`` softmax belief; the origin hour decodes the FIRST MASKED patch's row.
    Under ``TIME_PROBE_ENABLED = False`` the keys stay absent and the renderer drops the clock overlay.
    """
    if not TIME_PROBE_ENABLED:
        return
    hsteps = CV.PREDICTION_PATCHES * CV.PATCH_SIZE
    ms = PR.forecast_masked_set(CV.MAX_CONTEXT_PATCHES)
    pred_hour, true_hour, tod_R, bin_probs = [], [], [], []
    for c in range(CV.DAY_PATCHES // CV.PREDICTION_PATCHES):
        ts = ds + c * hsteps
        ctx = context_window(feats, ts, CV.MAX_CONTEXT_PATCHES)
        # Announced, like every other forward here: an un-announced maskable slot takes
        # normalize(0), a legal "no event", so the probe would read a regime the figure never runs.
        ov = _future_overrides(feats, ts, CV.ANNOUNCE)
        out = predict(model, ctx, normalization_stats=stats, overrides=ov,
                      return_time=True, mask_spans=ms.spans)
        tp = out.get('time_pred')
        if tp is None:
            ph, r = float('nan'), float('nan')
            bin_probs.append(None)
        else:
            bin_probs.append(torch.softmax(tp, -1).cpu().numpy())
            h_t, r_t = time_of_day_decode_bins(tp[0:1, :], TIME_PROBE_N_BINS)
            ph, r = float(h_t.item()), float(r_t.item())
        pred_hour.append(ph)
        true_hour.append(float(d['hour_of_day'][ts]))
        tod_R.append(r)
    spec['tile2_pred_hour'] = pred_hour
    spec['tile2_true_hour'] = true_hour
    spec['tile2_R'] = tod_R
    if bin_probs and all(b is not None for b in bin_probs):
        spec['tile2_time_probs'] = np.stack(bin_probs)


class _Seg:
    def __init__(self, d): self._d = d
    def __len__(self): return len(self._d['bg_observed'])
    def hour_of_day(self): return self._d['hour_of_day'].astype(float)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = CV.load_model(device)
    model = model.to(device)
    model.eval()
    print(f"[curves_sim] model step={step}; generating {len(SEEDS)} sim patients (seeds {SEEDS})…", flush=True)
    runs = SD.make_sim_runs(SEEDS, SD.DEFAULT_HOURS)

    report = PR.RunReport(label='curves_sim (T1DMSIM cohort)',
                          n_ctx=CV.MAX_CONTEXT_PATCHES)
    print(PR.context_note(CV.MAX_CONTEXT_PATCHES), flush=True)
    # census at the PINNED 24 h footprint, so a sim run and a real run read on one axis
    print(report.census('t1dmsim', [len(d['bg_observed']) for _pid, d in runs],
                        stride_patches=8).format(), flush=True)

    need = CV.MAX_CONTEXT_PATCHES + CV.DAY_PATCHES         # 168 h context + the 48 h day window
    days = []
    for pid, d in runs:
        if len(d['bg_observed']) // CV.PATCH_SIZE < need:
            continue
        ds = _pick_day_start(_Seg(d))
        if ds is None:
            continue
        days.append((pid, d, ds))
        if len(days) >= N_DAYS:
            break
    print(f"[curves_sim] {len(days)} day windows selected "
          f"(of {len(runs)} patients; the 48 h figure span needs "
          f"{need} patches, beyond the census footprint)", flush=True)

    print("[curves_sim] fitting BG error envelope (2 h + 8 h)…", flush=True)
    sig2 = sim_bg_sigma(model, stats, runs, CV.PREDICTION_PATCHES,
                        stride_steps=8 * CV.PATCH_SIZE, max_windows=200,
                        report=report)
    sig8 = sim_bg_sigma(model, stats, runs, CV.H8_PATCHES,
                        stride_steps=16 * CV.PATCH_SIZE, max_windows=100,
                        report=report)
    print(f"[curves_sim] ±1σ @30/60/120m = {sig2[5]:.0f}/{sig2[11]:.0f}/{sig2[23]:.0f} mg/dL; "
          f"@8h = {sig8[-1]:.0f} mg/dL", flush=True)

    figdir = os.path.join(HERE, 'sim', 'figures'); os.makedirs(figdir, exist_ok=True)
    for i, (pid, d, ds) in enumerate(days, 1):
        spec = sim_day_curves(model, stats, d, ds, pid, sig2=sig2, sig8=sig8)
        out = os.path.join(figdir, f'sim_day{i}.png')
        CV.plot_day(spec, 'T1DMSIM', out)
        print(f"  wrote {out}  ({pid})", flush=True)
    print()
    report.emit()
    print("[curves_sim] DONE")


if __name__ == '__main__':
    main()
