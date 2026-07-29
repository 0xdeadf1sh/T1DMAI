"""
Augmented announced-event actual-vs-predicted BG figures, across OhioT1DM / AZT1D /
ShanghaiT1DM. Reconstructs the unlogged meals/boluses (``augment.augment_segment``),
then writes metrics/augmented/figures/{dataset}_{trajectories,parity,clarke}.png on the
current checkpoint, conditioned on the announced (logged + injected) future
carbs/insulin.

Risk-space redesign note: the model now outputs only a BG quantile forecast, so the
former 24 h carb/insulin/IS/HGO channel-overlay panels (channels_2h/channels_8h) are
dropped — only the BG figures remain.

Usage:  CUDA_VISIBLE_DEVICES="" python metrics/augmented/make_comparison_figures.py
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from realdata.report import build_figures
from augment import augment_segment


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_figures(os.path.join(HERE, 'figures'), device, conditional=True,
                  augment_fn=augment_segment)


if __name__ == '__main__':
    main()
