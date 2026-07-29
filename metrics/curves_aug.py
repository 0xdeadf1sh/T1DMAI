"""
Augmented-regime equivalent of curves.py: the same 48 h BG day-curve figures
(2 h/8 h BG, conditioned), but on records whose unlogged meals/boluses have been
reconstructed and injected first (`metrics/augmented/augment.py`). The injected
carb/bolus events therefore appear in the logged-event overlay, so the curves show
the augmented (logged + injected) regime the `metrics/augmented/` report scores —
an upper bound on the announced-event regime.

Risk-space redesign note: the predicted-dynamics-channel panels (carbs/insulin/IS/
HGO) were dropped with curves.py — those are no longer model outputs.

Reuses curves.make_curve_figures with augment.augment_segment. Runs on the live best
checkpoint (GPU if available). Writes metrics/augmented/figures/<ds>_day<k>.png.
"""
from __future__ import annotations
import os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)                                   # curves
sys.path.insert(0, os.path.join(ROOT, 'metrics', 'augmented'))  # augment

import curves as CV                       # sets threads + local load_model/make_curve_figures
from augment import augment_segment


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = CV.load_model(device)
    model = model.to(device)
    model.eval()
    print(f"[curves_aug] model step={step} device={device}")
    figdir = os.path.join(HERE, 'augmented', 'figures')
    CV.make_curve_figures(model, stats, figdir, augment_fn=augment_segment)
    print("\n[curves_aug] DONE")


if __name__ == '__main__':
    main()
