"""
Augmented announced-event report: reconstruct the unlogged meals/boluses into the
real records (``augment.augment_segment``), then evaluate T1DMAI with every window's
future carbohydrate and insulin announced. Writes metrics/augmented/{stats.json,
README.md, figures/rmse_vs_horizon.png}.

Because injected amounts are sized from the CGM the forecast is scored against, the
numbers are an upper bound on the announced-event regime (stated in the README).

Usage:  python metrics/augmented/build_report.py
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from realdata.report import evaluate_all, render_readme, render_figure
from augment import augment_segment


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R = evaluate_all(device, conditional=True, augment_fn=augment_segment)
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    with open(os.path.join(HERE, 'stats.json'), 'w') as f:
        json.dump(R, f, indent=2)
    with open(os.path.join(HERE, 'README.md'), 'w') as f:
        f.write(render_readme(R, mode='augmented'))
    render_figure(R, os.path.join(HERE, 'figures', 'rmse_vs_horizon.png'))
    print("wrote metrics/augmented/{stats.json, README.md, figures/rmse_vs_horizon.png}")
    for ds in ('ohiot1dm', 'azt1d', 'shanghai'):
        m = R[ds]['metrics']
        print(f"  {ds:10} RMSE point/wm @120: {m['120']['rmse_point']:.1f} / {m['120']['rmse_winmean']:.1f}  "
              f"({R[ds]['n_test_windows']} test win)")


if __name__ == '__main__':
    main()
