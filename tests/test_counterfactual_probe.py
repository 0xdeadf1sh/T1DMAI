"""The injected curves are pinned; the probe itself is a smoke test, since the net is
untrained and its response sign is not assertable."""

import math

import numpy as np
import pytest
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


def test_cf_bolus_curve_is_the_spec_curve_not_a_rectangle():
    from train import _cf_bolus_curve
    from config import (CF_CARB_BOLUS_G, CF_INSULIN_BOLUS_U,
                        PREDICTION_PATCHES, PATCH_SIZE)

    ps = PREDICTION_PATCHES * PATCH_SIZE

    carb = _cf_bolus_curve('carb', CF_CARB_BOLUS_G, ps)
    assert carb.shape == (ps,)
    assert (carb >= 0.0).all()
    assert abs(float(carb.sum()) - CF_CARB_BOLUS_G) < 1e-3, float(carb.sum())
    # GI 100 peaks at (k-1)*theta = 15 min, the fourth 5-min step
    assert int(carb.argmax()) == 3, f"carb peak at step {int(carb.argmax())}"
    assert abs(float(carb[0]) - 1.7907) < 1e-3, float(carb[0])
    # the rectangle this replaced was flat across the first PATCH_SIZE steps
    assert float(carb[0]) < 0.5 * float(carb[3]), (float(carb[0]), float(carb[3]))

    ins = _cf_bolus_curve('insulin', CF_INSULIN_BOLUS_U, ps)
    assert ins.shape == (ps,)
    assert (ins >= 0.0).all()
    assert abs(float(ins.sum()) - CF_INSULIN_BOLUS_U) < 1e-3, float(ins.sum())
    assert int(ins.argmax()) > int(carb.argmax()), (
        f"insulin peaks at step {int(ins.argmax())}, carbs at {int(carb.argmax())}")

    # truncation keeps the head of the curve rather than rescaling it back to the total
    short = _cf_bolus_curve('carb', CF_CARB_BOLUS_G, 10)
    assert short.shape == (10,)
    assert np.allclose(short, carb[:10])
    assert float(short.sum()) < CF_CARB_BOLUS_G

    with pytest.raises(ValueError):
        _cf_bolus_curve('exercise', 1.0, ps)

    print(f"\n[DUMP] cf_curve | carb sum={carb.sum():.3f} peak_step={int(carb.argmax())} "
          f"step0={carb[0]:.4f} | insulin sum={ins.sum():.3f} "
          f"peak_step={int(ins.argmax())}")
