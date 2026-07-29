"""Tests for normalization.py and data.py — shapes, normalization, collation.

Risk-space redesign: 3-feature input stack (bg, carb, insulin) — the 4 temporal
sin/cos features are dropped — with 3-channel normalization (bg_absolute in
Kovatchev RISK space, carb_intake / insulin_combined log1p).  Targets are the
smoothed pred-zone BG ``(P, S)`` mg/dL.  The conditioned/unconditioned dichotomy
and the per-patch mask bits are gone — there is no ``loss_mask`` and no
``reveal_mask``.  Patches are ``(T, PATCH_DIM=18)`` (= PATCH_SIZE*N_INPUT_FEATURES,
no trailing mask bits); in the prediction zone bg (feat 0) is zeroed while carb
(feat 1) / insulin (feat 2) ALWAYS carry their true future values (the model is
always conditioned on the announced doses).
"""

import math
import numpy as np
import torch
import pytest


def test_normalization_stats():
    """Normalization statistics are computed and have sane values."""
    from normalization import compute_normalization_stats

    stats = compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)

    print("\n[DUMP] normalization | statistics:")
    for channel, values in stats.items():
        print(f"  {channel}: mean={values['mean']:.4f}, std={values['std']:.4f}")
        assert not math.isnan(values['mean']), f"NaN mean for {channel}"
        assert not math.isnan(values['std']), f"NaN std for {channel}"
        assert values['std'] > 0, f"Zero std for {channel}"


def _get_stats():
    """Helper: compute or load normalization stats for testing."""
    import os
    from normalization import compute_normalization_stats, load_normalization_stats, NORM_STATS_FILE
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def test_channel_names_trimmed_to_three():
    """The redesign trims CHANNEL_NAMES to the 3 input signal channels — bg,
    carb, insulin — with bg_delta / IS / HGO dropped."""
    from normalization import CHANNEL_NAMES
    assert CHANNEL_NAMES == ['bg_absolute', 'carb_intake', 'insulin_combined'], \
        f"CHANNEL_NAMES must be the trimmed 3-channel list, got {CHANNEL_NAMES}"


def test_normalize_denormalize_roundtrip():
    """denormalize(normalize(x)) is the identity for both dense and sparse
    (log1p) channels, and the numpy and torch denormalize branches agree.

    Normalization is the highest bug surface — a broken log1p/expm1 transform or
    a swapped mean/std would corrupt training silently. This pins the round-trip.
    """
    import torch
    from normalization import (normalize, denormalize, CHANNEL_NAMES,
                               SPARSE_LOG1P_CHANNELS)

    stats = _get_stats()
    rng = np.random.default_rng(0)
    # Physically-plausible non-negative raw values — the sparse channels'
    # forward path is log1p(max(x, 0)), so a negative input would not
    # round-trip (and never occurs in real data).  The bg_absolute channel is
    # now RISK-space (Kovatchev f applied before the z-score), and f clamps to
    # the physical band, so bg must sit inside [BG_CLAMP_MIN, BG_CLAMP_MAX] for
    # the round-trip to be identity (a sub-floor bg would clamp, not recover).
    import T1DMSIM.simulator as sim
    raw = rng.uniform(0.0, 50.0, size=(7, len(CHANNEL_NAMES))).astype(np.float32)
    bg_col = CHANNEL_NAMES.index('bg_absolute')
    raw[:, bg_col] = rng.uniform(
        sim.BG_CLAMP_MIN + 5.0, sim.BG_CLAMP_MAX - 5.0, size=raw.shape[0]
    ).astype(np.float32)

    norm = normalize(raw, stats)
    back_np = denormalize(norm, stats)
    np.testing.assert_allclose(back_np, raw, rtol=1e-4, atol=1e-3)

    # The torch branch must match the numpy branch one-to-one.
    back_torch = denormalize(torch.tensor(norm), stats)
    assert isinstance(back_torch, torch.Tensor)
    np.testing.assert_allclose(back_torch.numpy(), back_np, rtol=1e-5, atol=1e-5)

    # The sparse-channel path must actually be exercised by this fixture.
    assert any(n in SPARSE_LOG1P_CHANNELS for n in CHANNEL_NAMES), \
        "no sparse channel present — log1p/expm1 path untested"
    max_err = float(np.abs(back_np - raw).max())
    print(f"\n[DUMP] norm_roundtrip | channels={len(CHANNEL_NAMES)}, "
          f"sparse={sorted(SPARSE_LOG1P_CHANNELS)}, max abs err={max_err:.2e}")


def test_bg_risk_roundtrip_and_last_bg_anchor():
    """Now that ``normalize`` folds Kovatchev ``f`` into the bg channel (feat 0 is
    ``z(f(bg))``), the two crossings must still compose to the identity:

    (a) over a physical mg/dL grid (50..350) with carb/insulin,
        ``denormalize(normalize([[bg,carb,insulin]]))`` recovers the inputs — bg
        via the ``z → f_inv`` path (≈1e-3 relative, the log/exp round-trip), the
        sparse carb/insulin via ``log1p → expm1`` (near-exact); and

    (b) ``last_bg_mgdl_from_context`` recovers the mg/dL anchor from a context
        whose last bg cell carries the risk-space-normalized value — the TRAIN and
        INFERENCE anchors must be computed by the same inverse.
    """
    import torch
    from normalization import normalize, denormalize, CHANNEL_NAMES
    from utils import last_bg_mgdl_from_context
    from config import PATCH_SIZE, N_INPUT_FEATURES

    stats = _get_stats()
    bg_col = CHANNEL_NAMES.index('bg_absolute')
    carb_col = CHANNEL_NAMES.index('carb_intake')
    ins_col = CHANNEL_NAMES.index('insulin_combined')

    # (a) Round-trip over a bg grid with fixed non-negative carb/insulin.
    bg_grid = np.linspace(50.0, 350.0, 16, dtype=np.float32)
    raw = np.zeros((bg_grid.size, len(CHANNEL_NAMES)), dtype=np.float32)
    raw[:, bg_col] = bg_grid
    raw[:, carb_col] = 12.5
    raw[:, ins_col] = 3.25

    back = denormalize(normalize(raw, stats), stats)
    bg_relerr = float(np.abs(back[:, bg_col] - bg_grid).max() / bg_grid.max())
    assert bg_relerr < 5e-3, f"bg z→f_inv round-trip drifted: rel err {bg_relerr}"
    np.testing.assert_allclose(back[:, carb_col], raw[:, carb_col], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(back[:, ins_col], raw[:, ins_col], rtol=1e-4, atol=1e-3)

    # (b) last_bg anchor: build a context whose rightmost bg cell is the
    #     risk-space-normalized value of a chosen anchor, and recover it.
    n_ctx = 4
    for bg_anchor in (72.0, 120.0, 245.0):
        cell = np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32)
        cell[0, bg_col] = bg_anchor
        z_bg = float(normalize(cell, stats)[0, bg_col])
        context = torch.zeros(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)
        context[-1, -1, 0] = z_bg
        recovered = float(last_bg_mgdl_from_context(context, stats).item())
        rel = abs(recovered - bg_anchor) / bg_anchor
        assert rel < 5e-3, f"last_bg anchor drifted at {bg_anchor}: got {recovered} (rel {rel})"

    print(f"\n[DUMP] bg_risk_roundtrip | bg rel err={bg_relerr:.2e}; "
          f"last_bg anchor recovered within 5e-3 over {(72.0, 120.0, 245.0)} ✓")


def test_dataset_shapes():
    """Dataset produces correctly shaped samples: PATCH_DIM-wide patches and an
    (P, S) mg/dL BG target.  The retired per-channel loss_mask AND the
    reveal_mask are both gone — no mask key of any kind ships in a sample."""
    from data import T1DMDataset
    from config import (PREDICTION_PATCHES, PATCH_SIZE, PATCH_DIM,
                        N_INPUT_FEATURES, MIN_CONTEXT_PATCHES)

    assert PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES, \
        "PATCH_DIM must be PATCH_SIZE*N_INPUT_FEATURES (no mask bits)"

    stats = _get_stats()
    dataset = T1DMDataset(master_seed=42, total_steps=10, batch_size=2, normalization_stats=stats)
    sample = dataset[0]

    # Verify all expected keys exist — and that BOTH retired mask keys are gone.
    required_keys = ['patches', 'targets', 'n_context_patches', 'bg_formula_data']
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    assert 'loss_mask' not in sample, "loss_mask is retired"
    assert 'reveal_mask' not in sample, "reveal_mask is retired (no cond/uncond split)"

    patches = sample['patches']
    targets = sample['targets']
    print(f"\n[DUMP] dataset | patches shape: {patches.shape}")
    print(f"[DUMP] dataset | targets shape: {targets.shape}")

    assert patches.shape[1] == PATCH_DIM, f"Expected {PATCH_DIM} patch features, got {patches.shape[1]}"
    assert patches.shape[0] >= MIN_CONTEXT_PATCHES + PREDICTION_PATCHES, "Need at least min_context + prediction patches"

    # BG target is the smoothed pred-zone trajectory (P, S) mg/dL.
    assert targets.shape == (PREDICTION_PATCHES, PATCH_SIZE), \
        f"targets shape {targets.shape} != {(PREDICTION_PATCHES, PATCH_SIZE)}"
    import T1DMSIM.simulator as sim
    assert (targets >= sim.BG_CLAMP_MIN - 1e-3).all(), "BG target must be mg/dL (>= clamp floor)"


def test_pred_zone_always_conditioned():
    """The model is ALWAYS conditioned: in the prediction zone bg (feat 0) is
    zeroed while carb (feat 1) / insulin (feat 2) carry their TRUE future values.
    No reveal mask, no mask bits — patches are exactly PATCH_SIZE*N_INPUT_FEATURES
    wide.

    THE TRAP this guards: the old layout appended mask bits and split the pred
    zone into conditioned/unconditioned regimes. Here every announceable channel
    is unconditionally present, and the pred-zone row width is the bare feature
    flat with nothing trailing.
    """
    from data import _build_sample, _make_simulator, simulate_discard_warmup
    from config import (PATCH_SIZE, N_INPUT_FEATURES, PATCH_DIM,
                        PREDICTION_PATCHES, CHANNEL_TO_FEAT)

    assert CHANNEL_TO_FEAT == {0: 1, 1: 2}, "carb -> feat1, insulin -> feat2"

    stats = _get_stats()
    sim = _make_simulator(patient_seed=321, uniform_skills=False)
    data = simulate_discard_warmup(sim, 55.5)
    icr = float(sim.patient.icr)

    s = _build_sample(data=data, icr=icr, stats=stats,
                      rng=np.random.default_rng(7))
    assert 'reveal_mask' not in s, "a built sample must carry no reveal_mask"
    patches = s['patches'].numpy()
    assert patches.shape[1] == PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES, \
        "pred-zone row is the bare feature flat — no trailing mask bits"

    n_ctx = patches.shape[0] - PREDICTION_PATCHES
    pred = patches[n_ctx:]                                  # (P, PATCH_DIM)
    feat_grid = pred.reshape(PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES)

    carb_feat = CHANNEL_TO_FEAT[0]
    insulin_feat = CHANNEL_TO_FEAT[1]
    # bg (feat 0) is the predicted target — ALWAYS zeroed in the pred zone.
    assert (feat_grid[:, :, 0] == 0.0).all(), "bg feat 0 must be zeroed in pred zone"
    # carb (feat 1) / insulin (feat 2) carry the announced true future doses; on a
    # real trajectory at least some are non-zero (the model is always conditioned).
    assert np.any(feat_grid[:, :, carb_feat] != 0.0), "carb feat 1 must carry true future doses"
    assert np.any(feat_grid[:, :, insulin_feat] != 0.0), "insulin feat 2 must carry true future doses"
    print(f"\n[DUMP] pred_zone | PATCH_DIM={PATCH_DIM} (no mask bits); bg feat0 "
          f"zeroed, carb feat{carb_feat}/insulin feat{insulin_feat} conditioned ✓")


def test_pick_pred_start_step_uniform():
    """Uniform window picker: patch-aligned, deterministic, legal-or-None."""
    from data import _pick_pred_start_step
    from config import PATCH_SIZE, PREDICTION_PATCHES

    n_steps = 600
    n_ctx = 16
    room = PREDICTION_PATCHES * PATCH_SIZE

    starts = set()
    for seed in range(64):
        s = _pick_pred_start_step(n_steps, n_ctx, room, np.random.default_rng(seed))
        assert s is not None
        assert s % PATCH_SIZE == 0, f"start {s} not patch-aligned"
        assert n_ctx * PATCH_SIZE <= s <= n_steps - room, f"start {s} out of legal range"
        # Determinism: the same seed must reproduce the pick bit-for-bit.
        s2 = _pick_pred_start_step(n_steps, n_ctx, room, np.random.default_rng(seed))
        assert s == s2, f"non-deterministic pick at seed {seed}: {s} != {s2}"
        starts.add(s)
    print(f"\n[DUMP] pick_pred_start | {len(starts)} distinct starts over 64 seeds")
    assert len(starts) > 1, "picker should explore more than one window across seeds"

    # No legal window (room + context exceed the trajectory) → None.
    assert _pick_pred_start_step(n_ctx * PATCH_SIZE, n_ctx, room, np.random.default_rng(0)) is None


def test_collation_padding():
    """Batch collation correctly pads variable-length sequences to PATCH_DIM."""
    from data import T1DMDataset, collate_fn
    from config import PREDICTION_PATCHES, PATCH_DIM

    stats = _get_stats()
    dataset = T1DMDataset(master_seed=42, total_steps=10, batch_size=4, normalization_stats=stats)

    samples = [dataset[i] for i in range(4)]
    batch = collate_fn(samples)

    patches = batch['patches']
    attn_mask = batch['attn_mask']

    B = patches.shape[0]
    T = patches.shape[1]
    print(f"\n[DUMP] collation | batch patches shape: {patches.shape}")
    print(f"[DUMP] collation | attn_mask shape: {attn_mask.shape}")

    assert patches.shape == (B, T, PATCH_DIM), f"Bad patches shape: {patches.shape}"
    assert 'loss_mask' not in batch, "batch must not carry the retired loss_mask"
    assert 'reveal_mask' not in batch, "batch must not carry the retired reveal_mask"

    # Padded positions should be all-zero
    for i in range(B):
        n_ctx = samples[i]['n_context_patches']
        n_total = n_ctx + PREDICTION_PATCHES
        n_pad = T - n_total
        if n_pad > 0:
            assert (patches[i, :n_pad] == 0).all(), f"Padding not zeroed for sample {i}"
            # Padded positions blocked in attention, except the diagonal.
            pad_mask = attn_mask[i, :n_pad, :].clone()
            pad_mask[torch.arange(n_pad), torch.arange(n_pad)] = False
            assert not pad_mask.any(), f"Padding rows (off-diagonal) should be blocked for sample {i}"
            col_mask = attn_mask[i, :, :n_pad].clone()
            col_mask[torch.arange(n_pad), torch.arange(n_pad)] = False
            assert not col_mask.any(), f"Padding cols (off-diagonal) should be blocked for sample {i}"


# ---------------------------------------------------------------------------
# Cross-window (paired-window) time-of-day probe input (batch['next_window']).
# ---------------------------------------------------------------------------

def test_next_window_batch_shape_and_space():
    """With the probe on (``TIME_PROBE_ENABLED`` and ``TIME_PROBE_CROSS_WINDOW_WEIGHT
    > 0``) a collated batch carries ``next_window`` — window ``k`` shifted forward one
    horizon, built + normalized ENTIRELY inside data.py (three-spaces rule).  Pin its
    contract: shapes/dtypes, ``valid`` True at default config (room to shift), and the
    FROZEN-index-map invariant that the shifted pred zone keeps bg (feat 0) zeroed so
    no future bg leaks.  The default-off path (weight 0) ships no key at all.
    """
    from data import T1DMDataset, collate_fn
    from config import (PREDICTION_PATCHES, PATCH_DIM, N_INPUT_FEATURES,
                        NON_MASKABLE_FEATS, TIME_PROBE_ENABLED,
                        TIME_PROBE_CROSS_WINDOW_WEIGHT)

    if not (TIME_PROBE_ENABLED and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0):
        pytest.skip("cross-window probe off — data.py ships no next_window key")

    stats = _get_stats()
    B = 3
    dataset = T1DMDataset(master_seed=42, total_steps=10, batch_size=B,
                          normalization_stats=stats)
    samples = [dataset[i] for i in range(B)]

    # Per-sample sub-dict present and well-formed.
    for s in samples:
        assert 'next_window' in s, "probe on => each sample must carry next_window"
        nw = s['next_window']
        assert set(nw) == {'patches', 'last_bg', 'pred_start_hour', 'valid'}, \
            f"unexpected next_window keys {sorted(nw)}"

    batch = collate_fn(samples)
    assert 'next_window' in batch, "collate must stack next_window when samples carry it"
    nw = batch['next_window']
    max_T = batch['patches'].shape[1]

    assert nw['patches'].shape == (B, max_T, PATCH_DIM), \
        f"next_window patches {tuple(nw['patches'].shape)} != {(B, max_T, PATCH_DIM)}"
    assert nw['patches'].dtype == torch.float32
    assert nw['last_bg'].shape == (B,) and nw['last_bg'].dtype == torch.float32
    assert nw['pred_start_hour'].shape == (B,)
    assert nw['valid'].shape == (B,) and nw['valid'].dtype == torch.bool

    # At default config NIGHT_LONG_HORIZON_HOURS (8h) leaves room to shift by one
    # 2h horizon, so every sample's next window is in-range.
    assert bool(nw['valid'].all()), \
        f"all next windows should be valid at default config, got {nw['valid'].tolist()}"

    # last_bg is physical mg/dL (survives the forward's units tripwire).
    import T1DMSIM.simulator as sim
    assert (nw['last_bg'] >= sim.BG_CLAMP_MIN - 1e-3).all() and \
           (nw['last_bg'] <= sim.BG_CLAMP_MAX + 1e-3).all(), \
        "next_window last_bg must be physical mg/dL"

    # FROZEN index map: the shifted window's OWN pred zone (rightmost P patches)
    # keeps every NON_MASKABLE feat (bg, feat 0) zeroed — no future-bg leak.
    grid = nw['patches'].reshape(B, max_T, PATCH_DIM // N_INPUT_FEATURES, N_INPUT_FEATURES)
    pred_grid = grid[:, max_T - PREDICTION_PATCHES:, :, :]      # (B, P, S, F)
    for feat_idx in NON_MASKABLE_FEATS:
        assert (pred_grid[..., feat_idx] == 0.0).all(), \
            f"shifted pred zone leaked a non-zero feat {feat_idx} (bg must stay zeroed)"

    print(f"\n[DUMP] next_window | patches {tuple(nw['patches'].shape)}  "
          f"valid={nw['valid'].tolist()}  last_bg~"
          f"[{float(nw['last_bg'].min()):.0f},{float(nw['last_bg'].max()):.0f}] mg/dL  "
          f"pred-zone feat{NON_MASKABLE_FEATS} zeroed ✓")
