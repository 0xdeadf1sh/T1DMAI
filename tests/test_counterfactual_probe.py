"""Smoke only: the net is untrained, so response sign is not asserted."""

import math

import torch


def _get_stats():
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def test_counterfactual_probe_smoke():
    from train import _run_counterfactual_probe
    from model import T1DMAI
    from data import T1DMDataset
    from config import BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD

    stats = _get_stats()
    device = torch.device('cpu')

    # probe iterates min(len, VALIDATION_PROBE_N_PATIENTS)
    val_dataset = T1DMDataset(master_seed=777, total_steps=4, batch_size=1,
                              normalization_stats=stats, cache_path=None)

    model = T1DMAI().to(device)
    model.eval()

    result = _run_counterfactual_probe(
        model, val_dataset, stats, device,
        hypo_threshold=BG_HYPO_THRESHOLD, hyper_threshold=BG_HYPER_THRESHOLD,
    )

    expected_keys = {
        'cf_carb_dbg', 'cf_carb_dir', 'cf_insulin_dbg', 'cf_insulin_dir',
        'cf_carb_monotonic', 'cf_insulin_monotonic',
        'cf_hypo_rescue', 'cf_hyper_rescue',
        'cf_n', 'cf_hypo_n', 'cf_hyper_n',
    }
    assert set(result.keys()) == expected_keys, (
        f"cf_* key set mismatch: got {sorted(result.keys())}")
    assert len(expected_keys) == 11

    assert isinstance(result['cf_n'], int) and result['cf_n'] >= 1, (
        f"cf_n must be a positive probe count, got {result['cf_n']!r}")
    assert isinstance(result['cf_hypo_n'], int) and result['cf_hypo_n'] >= 0
    assert isinstance(result['cf_hyper_n'], int) and result['cf_hyper_n'] >= 0

    # None means no baseline hypo/hyper sample existed
    for k, v in result.items():
        if k in ('cf_n', 'cf_hypo_n', 'cf_hyper_n'):
            continue
        if v is None:
            assert k in ('cf_hypo_rescue', 'cf_hyper_rescue'), (
                f"{k} unexpectedly None (only rescue rates may be None)")
            continue
        assert math.isfinite(float(v)), f"cf key {k} is non-finite: {v}"

    for k in ('cf_carb_dir', 'cf_insulin_dir',
              'cf_carb_monotonic', 'cf_insulin_monotonic'):
        assert 0.0 <= float(result[k]) <= 1.0, f"{k}={result[k]} out of [0,1]"

    print(f"\n[DUMP] cf_probe | n={result['cf_n']} "
          f"hypo_n={result['cf_hypo_n']} hyper_n={result['cf_hyper_n']}")
    print(f"[DUMP] cf_probe | carb_dbg={result['cf_carb_dbg']:.3f} "
          f"insulin_dbg={result['cf_insulin_dbg']:.3f} "
          f"carb_dir={result['cf_carb_dir']:.2f} insulin_dir={result['cf_insulin_dir']:.2f} "
          f"carb_mono={result['cf_carb_monotonic']:.2f} "
          f"insulin_mono={result['cf_insulin_monotonic']:.2f}")
    print(f"[DUMP] cf_probe | all 11 cf_* keys present and finite/None ✓")
