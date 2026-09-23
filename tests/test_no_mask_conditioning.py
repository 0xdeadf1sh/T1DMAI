"""One announcement bit; no conditioned/unconditioned dichotomy.

``z=0`` in a withheld bg slot decodes to an ordinary reading (~142 mg/dL), not a
sentinel — so the masked set is ANNOUNCED, never inferred from position."""

import numpy as np
import torch


def _get_stats():
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=4)


def test_patch_dim_and_feat_map():
    """bg_masked is in NEITHER feat list: written from the masked set, not announced."""
    import config
    from config import PATCH_DIM, PATCH_SIZE, N_INPUT_FEATURES
    from data import BG_MASKED_FEAT

    assert PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES, (
        f"PATCH_DIM {PATCH_DIM} != PATCH_SIZE*N_INPUT_FEATURES "
        f"{PATCH_SIZE * N_INPUT_FEATURES}")
    assert PATCH_DIM == 24, f"expected PATCH_DIM 24 at the active config, got {PATCH_DIM}"
    assert not hasattr(config, 'N_MASK_BITS'), \
        "config.N_MASK_BITS must be deleted — the trailing mask-bit tier is gone"
    for gone in ('BLOCK_MASK_PROB', 'CARB_NOISE_AUG_ENABLED', 'CARB_NOISE_AUG_SIGMA'):
        assert not hasattr(config, gone), f"config.{gone} must be deleted"
    assert hasattr(config, 'CHANNEL_TO_FEAT'), "config.CHANNEL_TO_FEAT must exist"
    assert not hasattr(config, 'CHANNEL_TO_FEAT_BIT'), \
        "config.CHANNEL_TO_FEAT_BIT renamed to CHANNEL_TO_FEAT"
    assert config.CHANNEL_TO_FEAT == {0: 1, 1: 2}, \
        f"CHANNEL_TO_FEAT must map carb->feat1, insulin->feat2, got {config.CHANNEL_TO_FEAT}"
    assert config.NON_MASKABLE_FEATS == (0,), \
        f"NON_MASKABLE_FEATS must be (0,), got {config.NON_MASKABLE_FEATS}"
    assert config.MASKABLE_FEATS == (1, 2), \
        f"MASKABLE_FEATS must be (1, 2), got {config.MASKABLE_FEATS}"
    assert tuple(config.CHANNEL_TO_FEAT.values()) == config.MASKABLE_FEATS, \
        "CHANNEL_TO_FEAT's image must be exactly MASKABLE_FEATS"
    assert BG_MASKED_FEAT == 3, f"bg_masked must be feat 3, got {BG_MASKED_FEAT}"
    assert BG_MASKED_FEAT not in config.MASKABLE_FEATS, \
        "bg_masked must not be announceable — it is derived from the masked set"
    assert BG_MASKED_FEAT not in config.NON_MASKABLE_FEATS, \
        "bg_masked must not be withheld — it is the announcement itself"
    print(f"\n[DUMP] contract | PATCH_DIM={PATCH_DIM} (bg_masked at feat "
          f"{BG_MASKED_FEAT}, step-major); CHANNEL_TO_FEAT={config.CHANNEL_TO_FEAT}; "
          f"MASKABLE_FEATS={config.MASKABLE_FEATS}; N_MASK_BITS removed ✓")


def test_build_sample_no_reveal_mask_and_patch_width():
    from data import _build_sample, row_trajectory, simulate_row
    from config import PATCH_DIM, PATCH_SIZE, N_INPUT_FEATURES

    stats = _get_stats()
    row, icr = simulate_row(321)

    s = _build_sample(data=row_trajectory(row, 0), icr=icr, stats=stats,
                      rng=np.random.default_rng(7), boundary=True)

    assert 'reveal_mask' not in s, "a built sample must not carry reveal_mask"
    assert 'loss_mask' not in s, "a built sample must not carry loss_mask"
    last_dim = int(s['patches'].shape[-1])
    assert last_dim == PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES == 24, (
        f"patch last-dim {last_dim} != PATCH_DIM {PATCH_DIM} (24)")
    print(f"\n[DUMP] build_sample | keys={sorted(s.keys())}; patch last-dim={last_dim} ✓")


def test_masked_patches_withhold_bg_and_announce_the_bit():
    """The trap: no announce loop writes the bit, so left at its 0.0 init it announces
    a withheld patch as OBSERVED, at the right shapes and with a legal-looking z in
    the bg slot. Assert it REPRODUCES the mask, not that it is somewhere non-zero."""
    from data import T1DMDataset, BG_MASKED_FEAT
    from config import PATCH_SIZE, N_INPUT_FEATURES, CHANNEL_TO_FEAT
    from normalization import CHANNEL_NAMES, normalize

    stats = _get_stats()
    dataset = T1DMDataset(master_seed=1234, total_steps=8, batch_size=1,
                          normalization_stats=stats, cache_path=None)

    carb_feat = CHANNEL_TO_FEAT[0]
    insulin_feat = CHANNEL_TO_FEAT[1]
    assert (carb_feat, insulin_feat) == (1, 2)

    # zero-RAW baseline = log1p(0) z-scored, NOT 0; a no-dose cell carries this value.
    zero_raw_z = normalize(
        np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), stats)[0]
    carb_baseline = float(zero_raw_z[carb_feat])

    saw_carb = saw_insulin = False
    carb_min = np.inf
    n_masked_seen = 0
    for idx in range(8):
        s = dataset[idx]
        assert 'reveal_mask' not in s, "sample must not carry reveal_mask"
        patches = s['patches'].numpy()
        T = patches.shape[0]
        feat_grid = patches.reshape(T, PATCH_SIZE, N_INPUT_FEATURES)

        bfd = s['bg_formula_data']
        masked = np.zeros(T, dtype=bool)
        masked[bfd['mask_idx'][bfd['valid']]] = True
        n_masked_seen += int(masked.sum())

        # per-patch bit, step-major: all PATCH_SIZE columns of the bg_masked feat
        bit = feat_grid[:, :, BG_MASKED_FEAT]
        assert np.array_equal(bit, np.repeat(masked[:, None], PATCH_SIZE, axis=1)
                              .astype(np.float32)), (
            f"feat {BG_MASKED_FEAT} does not reproduce the sampled mask "
            f"(sample {idx}): announced {sorted(np.flatnonzero(bit.any(axis=1)))} "
            f"vs masked {sorted(np.flatnonzero(masked))}")

        assert (feat_grid[masked, :, 0] == 0.0).all(), \
            f"bg feat 0 must be zeroed on every masked patch (sample {idx})"
        # z=0 on a visible patch is an ordinary reading, so withholding there would be undetectable.
        assert not (feat_grid[~masked, :, 0] == 0.0).all(), \
            f"bg feat 0 zeroed on a VISIBLE patch (sample {idx})"

        carb_col = feat_grid[:, :, carb_feat]
        assert not np.all(carb_col == 0.0), (
            f"carb feat {carb_feat} is the literal-0.0 fill (sample "
            f"{idx}) — the announced column was dropped; a no-dose cell must "
            f"carry {carb_baseline:.6f}")
        assert (carb_col >= carb_baseline - 1e-4).all(), (
            f"carb feat {carb_feat} fell below the zero-RAW baseline "
            f"{carb_baseline:.6f} (sample {idx}, min {float(carb_col.min()):.6f}) "
            "— raw floor or log1p encoding missing")
        carb_min = min(carb_min, float(carb_col.min()))
        if np.any(feat_grid[:, :, carb_feat] != 0.0):
            saw_carb = True
        if np.any(feat_grid[:, :, insulin_feat] != 0.0):
            saw_insulin = True

    assert saw_carb, "carb feat 1 must carry true future doses (always conditioned)"
    assert saw_insulin, "insulin feat 2 must carry true future doses (always conditioned)"
    print(f"\n[DUMP] announcement | {n_masked_seen} masked patches over 8 samples; "
          f"the bit reproduces the mask; bg feat0 withheld there; carb feat{carb_feat} "
          f"min={carb_min:.6f} >= baseline {carb_baseline:.6f}; "
          f"insulin feat{insulin_feat} conditioned ✓")


def test_collate_no_reveal_mask():
    """The announcement bit survives the left-pad, rebased onto the padded axis."""
    from data import T1DMDataset, collate_fn, BG_MASKED_FEAT
    from config import PATCH_DIM, N_INPUT_FEATURES, PATCH_SIZE

    stats = _get_stats()
    dataset = T1DMDataset(master_seed=42, total_steps=8, batch_size=4,
                          normalization_stats=stats, cache_path=None)
    samples = [dataset[i] for i in range(4)]
    batch = collate_fn(samples)

    assert 'reveal_mask' not in batch, "batch must not carry reveal_mask"
    assert 'loss_mask' not in batch, "batch must not carry loss_mask"
    assert int(batch['patches'].shape[-1]) == PATCH_DIM == 24, \
        f"batch patch last-dim {batch['patches'].shape[-1]} != PATCH_DIM {PATCH_DIM}"

    bit = batch['patches'][..., BG_MASKED_FEAT::N_INPUT_FEATURES]   # (B, max_T, S)
    assert bit.shape[-1] == PATCH_SIZE
    mask_idx = batch['bg_formula_data']['mask_idx']
    valid = batch['bg_formula_data']['valid']
    for i in range(len(samples)):
        announced = torch.nonzero(bit[i].any(dim=-1)).flatten().tolist()
        expected = sorted(set(mask_idx[i][valid[i]].tolist()))
        assert announced == expected, (
            f"sample {i}: announced patches {announced} != masked set {expected} "
            "on the PADDED axis")
    print(f"\n[DUMP] collate | keys={sorted(batch.keys())}; patch last-dim="
          f"{int(batch['patches'].shape[-1])}; feat {BG_MASKED_FEAT} matches the "
          f"rebased mask_idx ✓")
