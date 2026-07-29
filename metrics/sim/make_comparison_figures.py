"""
In-domain T1DMSIM comparison figures: simulated-CGM-vs-predicted BG across fresh
simulator patients, in the announced-event regime. Writes metrics/sim/figures/
{sim_trajectories,parity,clarke}.png on the current checkpoint. The BG panels are
conditioned on the announced future carbs/insulin.

Risk-space redesign note: the model now outputs only a BG quantile forecast, so
the former 24 h carb/insulin/IS/HGO channel-overlay panels (which compared deleted
dynamics outputs to simulator truth) are dropped — only the BG comparison remains.

Usage:  CUDA_VISIBLE_DEVICES="" python metrics/sim/make_comparison_figures.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from config import QUANTILE_LEVELS
from realdata.report import load_model
from realdata.figures import trajectory_grid, parity_scatter, clarke_grid
from sim_data import make_sim_runs, collect_sim_rows, TEST_SEEDS, DEFAULT_HOURS

_LO_IDX = QUANTILE_LEVELS.index(0.05)
_HI_IDX = QUANTILE_LEVELS.index(0.95)


def _fmt_hour(h: float) -> str:
    """Format a fractional hour-of-day in [0, 24) as 'HH:MM'."""
    h = h % 24.0
    hh = int(h)
    mm = int(round((h - hh) * 60.0))
    if mm == 60:
        hh = (hh + 1) % 24
        mm = 0
    return f"{hh:02d}:{mm:02d}"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = load_model(device)
    print(f"model step={step}; generating sim figures (device={device})…")
    runs = make_sim_runs(TEST_SEEDS, DEFAULT_HOURS)
    figdir = os.path.join(HERE, 'figures')
    os.makedirs(figdir, exist_ok=True)

    # The SIM figure path is where the stored sim delta is valid (sim cal/test
    # exchangeability holds): thread it so the trajectory band ribbon is calibrated.
    rows = collect_sim_rows(model, stats, runs, device, cap=24,
                            conformal_delta=model.conformal_delta)
    pred = np.stack([r['pred'] for r in rows]); true = np.stack([r['true'] for r in rows])
    motion = np.nanmax(true, 1) - np.nanmin(true, 1)   # true has a NaN-padded tail
    order = np.argsort(motion)
    picks = [order[int(q * (len(order) - 1))] for q in
             (0.97, 0.88, 0.78, 0.62, 0.5, 0.38, 0.22, 0.12, 0.04)]
    examples = []
    for i in picks:
        label = f"{rows[i]['patient']} · swing {motion[i]:.0f} mg/dL"
        # Append the model's time-of-day forecast clock when the TOD probe is on
        # (probe-off rows carry NaN pred_hour); the true origin clock is shown alongside.
        pred_hour = rows[i].get('pred_hour', float('nan'))
        true_hour = rows[i].get('true_hour', float('nan'))
        tod_r = rows[i].get('tod_R', float('nan'))
        if pred_hour is not None and np.isfinite(pred_hour):
            label += (f" · clock {_fmt_hour(pred_hour)} (R{tod_r:.1f})"
                      f" / true {_fmt_hour(true_hour)}")
        ex = {'ctx_tail': rows[i]['ctx_tail'], 'true_future': rows[i]['true'],
              'pred_future': rows[i]['pred'], 'label': label,
              'time_probs': rows[i].get('time_probs')}
        # Calibrated 90% band ribbon (single-pass only; rolling rows carry bands=None),
        # sliced to the plotted prediction length.
        b = rows[i].get('bands')
        if b is not None:
            plen = len(rows[i]['pred'])
            ex['band_lo'] = b[:plen, _LO_IDX]
            ex['band_hi'] = b[:plen, _HI_IDX]
        examples.append(ex)
    trajectory_grid(examples, os.path.join(figdir, 'sim_trajectories.png'),
                    'T1DMSIM: simulated CGM vs predicted BG (example test windows)')
    parity_scatter(pred, true, os.path.join(figdir, 'sim_parity.png'),
                   'T1DMSIM: predicted vs true BG (test windows)')
    clarke_grid(pred, true, os.path.join(figdir, 'sim_clarke.png'),
                'T1DMSIM: Clarke Error Grid (test windows)')
    print(f"  {len(rows)} windows → 3 figures")
    print("done")


if __name__ == '__main__':
    main()
