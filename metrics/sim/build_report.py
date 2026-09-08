"""In-domain T1DMSIM report: fresh simulator patients at fixed seeds, future doses announced.

Writes metrics/sim/{stats.json, README.md, figures/rmse_vs_horizon.png}. In-domain reference,
not generalisation, not a peer comparison.
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from config import SIMULATOR_WARMUP_HOURS
from metrics.core.report import load_model, render_sim_readme, render_sim_figure
from metrics.core.run_eval import (evaluate_from_windows, night_onset_from_records,
                               rmse_by_horizon_from_records)
from sim_data import (make_sim_runs, collect_sim_windows, build_sim_feature_stack,
                      _smooth_sim_bg, CAL_SEEDS, TEST_SEEDS, DEFAULT_HOURS)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = load_model(device)
    cal_runs = make_sim_runs(CAL_SEEDS, DEFAULT_HOURS)
    test_runs = make_sim_runs(TEST_SEEDS, DEFAULT_HOURS)
    cal_w = collect_sim_windows(model, stats, cal_runs, device, stride_patches=8,
                                max_per_patient=24)
    test_w = collect_sim_windows(model, stats, test_runs, device, stride_patches=4,
                                 max_per_patient=60)
    res = evaluate_from_windows(cal_w, test_w)
    res['dataset'] = 'T1DMSIM'
    res['conditional'] = True
    # record = (feats, cgm, hod); truth is the raw bg-clamped CGM, never the future
    night_records = ((build_sim_feature_stack(d, stats),
                      _smooth_sim_bg(d['bg_observed']),
                      d['hour_of_day'].astype(float)) for _, d in test_runs)
    res['night_onset'] = night_onset_from_records(model, stats, night_records, device)
    # rolled, announced; figure only, never alters the suite
    rbh_records = ((build_sim_feature_stack(d, stats), _smooth_sim_bg(d['bg_observed']))
                   for _, d in test_runs)
    res['rmse_by_hour'] = rmse_by_horizon_from_records(
        model, stats, rbh_records, device, conditional=True)
    R = {'_meta': {'step': step, 'conditional': True, 'augmented': False,
                   'hours': DEFAULT_HOURS, 'warmup': SIMULATOR_WARMUP_HOURS}, 'sim': res}
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    with open(os.path.join(HERE, 'stats.json'), 'w') as f:
        json.dump(R, f, indent=2)
    with open(os.path.join(HERE, 'README.md'), 'w') as f:
        f.write(render_sim_readme(R))
    render_sim_figure(R, os.path.join(HERE, 'figures', 'rmse_vs_horizon.png'))
    m = res['metrics']
    print("wrote metrics/sim/{stats.json, README.md, figures/rmse_vs_horizon.png}")
    print(f"  T1DMSIM  RMSE point/wm @30/60/120: "
          f"{m['30']['rmse_point']:.1f}/{m['60']['rmse_point']:.1f}/{m['120']['rmse_point']:.1f} pt  "
          f"({res['n_test_windows']} test win, {res['n_patients']} patients)")


if __name__ == '__main__':
    main()
