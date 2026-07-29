"""
Real-data comparison report in the announced-event regime: evaluate T1DMAI on
OhioT1DM / AZT1D / ShanghaiT1DM with each window's future carbohydrate and
insulin announced to the model, and write metrics/real/{stats.json, README.md,
figures/rmse_vs_horizon.png}.

Usage:  python metrics/real/build_report.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from realdata.report import evaluate_all, render_readme, render_figure


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R = evaluate_all(device, conditional=True)
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    with open(os.path.join(HERE, 'stats.json'), 'w') as f:
        json.dump(R, f, indent=2)
    with open(os.path.join(HERE, 'README.md'), 'w') as f:
        f.write(render_readme(R, mode='real'))
    render_figure(R, os.path.join(HERE, 'figures', 'rmse_vs_horizon.png'))
    print("wrote metrics/real/{stats.json, README.md, figures/rmse_vs_horizon.png}")
    for ds in ('ohiot1dm', 'azt1d', 'shanghai'):
        m = R[ds]['metrics']
        print(f"  {ds:10} RMSE point/wm @120: {m['120']['rmse_point']:.1f} / {m['120']['rmse_winmean']:.1f}  "
              f"({R[ds]['n_test_windows']} test win)")


if __name__ == '__main__':
    main()
