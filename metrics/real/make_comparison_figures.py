"""
Announced-event actual-vs-predicted BG figures for the real-data report, across
OhioT1DM / AZT1D / ShanghaiT1DM. Writes metrics/real/figures/{dataset}_{trajectories,
parity,clarke}.png on the current checkpoint, conditioned on the announced future
carbs/insulin.

Risk-space redesign note: the model now outputs only a BG quantile forecast, so the
former 24 h carb/insulin/IS/HGO channel-overlay panels (channels_2h/channels_8h) are
dropped — only the BG figures remain.

Usage:  CUDA_VISIBLE_DEVICES="" python metrics/real/make_comparison_figures.py
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from realdata.report import build_figures


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_figures(os.path.join(HERE, 'figures'), device, conditional=True)


if __name__ == '__main__':
    main()
