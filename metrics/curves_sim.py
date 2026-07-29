"""
Simulator (in-domain) equivalent of curves.py: the same 48 h BG figures
(2 h/8 h BG, conditioned) on FRESH T1DMSIM patients.

No logged-event markers on the BG panels (the simulator's combined insulin has no
discrete events).

Risk-space redesign note: the model now outputs only a BG quantile forecast, so
the former carb/insulin/IS/HGO channel panels (which overlaid the simulator's TRUE
latents on the predicted dynamics channels) were DROPPED — those are no longer
model outputs. The figure is BG-only.

Reuses curves.plot_day + constants; the sim data bridge is metrics/sim/sim_data.py.
Runs on the live best checkpoint. Writes metrics/sim/figures/sim_day{k}.png.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)                                   # curves
sys.path.insert(0, os.path.join(ROOT, 'metrics', 'sim'))  # sim_data

import curves as CV                       # sets threads + local load_model/split_segments
import sim_data as SD
from config import TIME_PROBE_ENABLED, TIME_PROBE_N_BINS
from realdata.features import context_window
from realdata.calibrate import _future_overrides
from curves import _pick_day_start
from inference import predict, predict_rolling
from utils import time_of_day_decode_bins

N_DAYS = 10
SEEDS = list(SD.TEST_SEEDS)[:14]          # held-out sim test patients (≥N_DAYS, one day each)


def sim_bg_sigma(model, stats, runs, horizon_patches,
                 stride_steps=8 * CV.PATCH_SIZE, max_windows=200):
    """Per-horizon-step ±1σ envelope over the sim cohort (curves.bg_sigma_real analogue).
    Conditioned: each window announces its true future carbs/insulin."""
    H = horizon_patches * CV.PATCH_SIZE
    acc = {'se': np.zeros(H), 'n': np.zeros(H), 'count': 0}
    for pid, d in runs:
        if acc['count'] >= max_windows:
            break
        feats = SD.build_sim_feature_stack(d, stats)
        # Error envelope is scored against the raw (bg-clamped) CGM (one space).
        CV._sigma_accumulate(model, stats, feats, CV.smooth_bg_truth(d['bg_observed']),
                             horizon_patches, stride_steps, acc, max_windows)
    return np.sqrt(acc['se'] / np.maximum(acc['n'], 1.0))


def sim_day_curves(model, stats, d, ds, pid, sig2=None, sig8=None):
    """BG 2 h/8 h conditioned forecasts tiled across 48 h (events zeroed → no markers)."""
    feats = SD.build_sim_feature_stack(d, stats)
    T = CV.DAY_PATCHES * CV.PATCH_SIZE
    # The plotted "true CGM" the forecast is scored against is the raw (bg-clamped)
    # CGM (one space).
    cgm = CV.smooth_bg_truth(d['bg_observed'])
    hod0 = float(d['hour_of_day'][ds])
    hours = hod0 + np.arange(T) * (5.0 / 60.0)

    def tile(hp):
        hsteps = hp * CV.PATCH_SIZE
        nr = hp // CV.PREDICTION_PATCHES
        curve = np.full(T, np.nan)
        for c in range(CV.DAY_PATCHES // hp):
            ts = ds + c * hsteps
            ctx = context_window(feats, ts, CV.MAX_CONTEXT_PATCHES)
            if nr == 1:
                ov = _future_overrides(feats, ts, CV.ANNOUNCE)
                out = predict(model, ctx, normalization_stats=stats, overrides=ov)
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
    """Populate the model's per-2h-tile time-of-day probe arrays into ``spec``.

    Runs a single ``model.forward(..., return_time=True)`` (via ``inference.predict``)
    at each 2 h forecast origin, storing the per-patch ``(P, TIME_PROBE_N_BINS)``
    softmax belief (``spec['tile2_time_probs']``, the native small-multiple clocks
    ``curves.plot_day`` renders) plus the scalar origin hour decoded from patch 0 vs
    the true ``d['hour_of_day'][ts]`` and the confidence ``R``, aligned with the
    ``spec['tile2_h']`` tile origins. Skipped entirely when ``TIME_PROBE_ENABLED`` is
    off (the head is unbuilt and the forward yields no probe output) so a probe-off
    checkpoint still renders — the renderer gates the clock overlay on key presence.
    """
    if not TIME_PROBE_ENABLED:
        return
    hsteps = CV.PREDICTION_PATCHES * CV.PATCH_SIZE
    pred_hour, true_hour, tod_R, bin_probs = [], [], [], []
    for c in range(CV.DAY_PATCHES // CV.PREDICTION_PATCHES):
        ts = ds + c * hsteps
        ctx = context_window(feats, ts, CV.MAX_CONTEXT_PATCHES)
        out = predict(model, ctx, normalization_stats=stats, return_time=True)
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

    need = CV.MAX_CONTEXT_PATCHES + CV.DAY_PATCHES         # MAX_CONTEXT_PATCHES ctx + DAY_PATCHES window (24 h + 48 h)
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
    print(f"[curves_sim] {len(days)} day windows selected", flush=True)

    print("[curves_sim] fitting BG error envelope (2 h + 8 h)…", flush=True)
    sig2 = sim_bg_sigma(model, stats, runs, CV.PREDICTION_PATCHES,
                        stride_steps=8 * CV.PATCH_SIZE, max_windows=200)
    sig8 = sim_bg_sigma(model, stats, runs, CV.H8_PATCHES,
                        stride_steps=16 * CV.PATCH_SIZE, max_windows=100)
    print(f"[curves_sim] ±1σ @30/60/120m = {sig2[5]:.0f}/{sig2[11]:.0f}/{sig2[23]:.0f} mg/dL; "
          f"@8h = {sig8[-1]:.0f} mg/dL", flush=True)

    figdir = os.path.join(HERE, 'sim', 'figures'); os.makedirs(figdir, exist_ok=True)
    for i, (pid, d, ds) in enumerate(days, 1):
        spec = sim_day_curves(model, stats, d, ds, pid, sig2=sig2, sig8=sig8)
        out = os.path.join(figdir, f'sim_day{i}.png')
        CV.plot_day(spec, 'T1DMSIM', out)
        print(f"  wrote {out}  ({pid})", flush=True)
    print("[curves_sim] DONE")


if __name__ == '__main__':
    main()
