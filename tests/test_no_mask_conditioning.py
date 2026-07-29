"""Pin the post-refactor contract: NO mask bits, NO conditioned/unconditioned
dichotomy.

The refactor removed the per-patch mask-bit tier and the cond/uncond split. The
model is now ALWAYS conditioned on the announced future carb/insulin:

* ``PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES`` (= 18, with the 4 temporal
  features dropped) — no trailing mask bits, and ``config.N_MASK_BITS`` no longer
  exists.
* A built sample carries no ``reveal_mask`` key, and its patch rows are exactly
  ``PATCH_DIM`` wide.
* In the prediction zone bg (feat 0) is ALWAYS zeroed (the model predicts it)
  while carb (feat 1) / insulin (feat 2) carry their TRUE future values (always
  conditioned).
* The collated batch carries no ``reveal_mask`` key.

These guards fail loudly if any of the removed machinery creeps back in.
"""

import numpy as np
import torch


def _get_stats():
    """Load or compute normalization stats for the on-the-fly build."""
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def test_patch_dim_no_mask_bits():
    """(a) PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES, and the N_MASK_BITS
    constant is deleted from config."""
    import config
    from config import PATCH_DIM, PATCH_SIZE, N_INPUT_FEATURES

    assert PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES, (
        f"PATCH_DIM {PATCH_DIM} != PATCH_SIZE*N_INPUT_FEATURES "
        f"{PATCH_SIZE * N_INPUT_FEATURES}")
    assert PATCH_DIM == 18, f"expected PATCH_DIM 18 at the active config, got {PATCH_DIM}"
    assert not hasattr(config, 'N_MASK_BITS'), \
        "config.N_MASK_BITS must be deleted — there are no mask bits"
    # The block-masking / carb-noise tunables are deleted too.
    for gone in ('BLOCK_MASK_PROB', 'CARB_NOISE_AUG_ENABLED', 'CARB_NOISE_AUG_SIGMA'):
        assert not hasattr(config, gone), f"config.{gone} must be deleted"
    # The mask-bit channel map is renamed to a plain feat map.
    assert hasattr(config, 'CHANNEL_TO_FEAT'), "config.CHANNEL_TO_FEAT must exist"
    assert not hasattr(config, 'CHANNEL_TO_FEAT_BIT'), \
        "config.CHANNEL_TO_FEAT_BIT renamed to CHANNEL_TO_FEAT"
    assert config.CHANNEL_TO_FEAT == {0: 1, 1: 2}, \
        f"CHANNEL_TO_FEAT must map carb->feat1, insulin->feat2, got {config.CHANNEL_TO_FEAT}"
    print(f"\n[DUMP] contract | PATCH_DIM={PATCH_DIM} (no mask bits); "
          f"CHANNEL_TO_FEAT={config.CHANNEL_TO_FEAT}; N_MASK_BITS removed ✓")


def test_build_sample_no_reveal_mask_and_width_18():
    """(b) A ``_build_sample`` output carries no 'reveal_mask' key and its patch
    rows are PATCH_DIM (== 18) wide."""
    from data import _build_sample, _make_simulator, simulate_discard_warmup
    from config import PATCH_DIM, PATCH_SIZE, N_INPUT_FEATURES

    stats = _get_stats()
    sim = _make_simulator(patient_seed=321, uniform_skills=False)
    data = simulate_discard_warmup(sim, 55.5)
    icr = float(sim.patient.icr)

    s = _build_sample(data=data, icr=icr, stats=stats,
                      rng=np.random.default_rng(7))

    assert 'reveal_mask' not in s, "a built sample must not carry reveal_mask"
    assert 'loss_mask' not in s, "a built sample must not carry loss_mask"
    last_dim = int(s['patches'].shape[-1])
    assert last_dim == PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES == 18, (
        f"patch last-dim {last_dim} != PATCH_DIM {PATCH_DIM} (18, no mask bits)")
    print(f"\n[DUMP] build_sample | keys={sorted(s.keys())}; patch last-dim={last_dim} ✓")


def test_pred_zone_bg_zeroed_carb_insulin_conditioned():
    """(c) In the prediction zone bg (feat 0) is all-zero while carb (feat 1) /
    insulin (feat 2) are NOT all-zero — the model is ALWAYS conditioned on the
    announced future doses.  Built on a fresh on-the-fly sample via
    ``T1DMDataset(cache_path=None)``."""
    from data import T1DMDataset
    from config import (PATCH_SIZE, N_INPUT_FEATURES, PREDICTION_PATCHES,
                        CHANNEL_TO_FEAT)

    stats = _get_stats()
    dataset = T1DMDataset(master_seed=1234, total_steps=8, batch_size=1,
                          normalization_stats=stats, cache_path=None)

    carb_feat = CHANNEL_TO_FEAT[0]
    insulin_feat = CHANNEL_TO_FEAT[1]
    assert (carb_feat, insulin_feat) == (1, 2)

    # Carb/insulin events are sparse; scan a few samples so the test window is
    # virtually certain to contain at least one announced dose of each.
    saw_carb = saw_insulin = False
    for idx in range(8):
        s = dataset[idx]
        assert 'reveal_mask' not in s, "sample must not carry reveal_mask"
        patches = s['patches'].numpy()
        n_ctx = patches.shape[0] - PREDICTION_PATCHES
        pred = patches[n_ctx:]
        feat_grid = pred.reshape(PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES)

        # bg (feat 0) is the predicted target — ALWAYS zeroed in the pred zone.
        assert (feat_grid[:, :, 0] == 0.0).all(), \
            f"bg feat 0 must be zeroed in pred zone (sample {idx})"
        if np.any(feat_grid[:, :, carb_feat] != 0.0):
            saw_carb = True
        if np.any(feat_grid[:, :, insulin_feat] != 0.0):
            saw_insulin = True
        if saw_carb and saw_insulin:
            break

    assert saw_carb, "carb feat 1 must carry true future doses (always conditioned)"
    assert saw_insulin, "insulin feat 2 must carry true future doses (always conditioned)"
    print(f"\n[DUMP] pred_zone | bg feat0 zeroed; carb feat{carb_feat} & "
          f"insulin feat{insulin_feat} conditioned (non-zero) ✓")


def test_collate_no_reveal_mask():
    """(d) The collated batch carries no 'reveal_mask' (nor 'loss_mask') key, and
    patches are PATCH_DIM wide."""
    from data import T1DMDataset, collate_fn
    from config import PATCH_DIM

    stats = _get_stats()
    dataset = T1DMDataset(master_seed=42, total_steps=8, batch_size=4,
                          normalization_stats=stats, cache_path=None)
    samples = [dataset[i] for i in range(4)]
    batch = collate_fn(samples)

    assert 'reveal_mask' not in batch, "batch must not carry reveal_mask"
    assert 'loss_mask' not in batch, "batch must not carry loss_mask"
    assert int(batch['patches'].shape[-1]) == PATCH_DIM == 18, \
        f"batch patch last-dim {batch['patches'].shape[-1]} != PATCH_DIM {PATCH_DIM}"
    print(f"\n[DUMP] collate | keys={sorted(batch.keys())}; "
          f"patch last-dim={int(batch['patches'].shape[-1])} ✓")
