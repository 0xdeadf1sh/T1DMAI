"""normalization.py and data.py — shapes, normalization, collation.
Five input features ``[bg, carb, insulin, exercise, bg_masked]`` over FOUR normalized
channels: ``bg_absolute`` in Kovatchev risk space, other three log1p; feat 4 is a
per-patch bit, no statistics. ``exercise_equiv`` is g/step carb-EQUIVALENT glucose
disposal, carb's encoding, never Kovatchev; targets raw mg/dL BG of ``MAX_MASKED_PATCHES`` slots."""

import math
import numpy as np
import torch
import pytest


def test_normalization_stats():
    from normalization import compute_normalization_stats

    stats = compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)

    print("\n[DUMP] normalization | statistics:")
    for channel, values in stats.items():
        print(f"  {channel}: mean={values['mean']:.4f}, std={values['std']:.4f}")
        assert not math.isnan(values['mean']), f"NaN mean for {channel}"
        assert not math.isnan(values['std']), f"NaN std for {channel}"
        assert values['std'] > 0, f"Zero std for {channel}"


def _get_stats():
    import os
    from normalization import compute_normalization_stats, load_normalization_stats, NORM_STATS_FILE
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def test_channel_names_are_the_four_input_signals():
    """The order pins every channel index in the project, so this is a literal pin,
    not a count — and the two counts, 4 channels against 5 features, pin separately.
    ``exercise_equiv`` is a log1p channel like carb, never a risk-space one: it is
    carbohydrate-equivalent glucose disposal in g/step, not a glucose."""
    from config import N_INPUT_FEATURES
    from data import BG_MASKED_FEAT
    from normalization import (CHANNEL_NAMES, N_CHANNELS, SPARSE_LOG1P_CHANNELS,
                               RISK_SPACE_CHANNELS)
    assert CHANNEL_NAMES == ['bg_absolute', 'carb_intake', 'insulin_combined',
                             'exercise_equiv'], \
        f"CHANNEL_NAMES must be the 4-channel input list, got {CHANNEL_NAMES}"
    assert len(CHANNEL_NAMES) == N_CHANNELS == 4
    assert N_INPUT_FEATURES == 5
    # the normalized channels take the LEADING columns, the bit follows them
    assert BG_MASKED_FEAT == len(CHANNEL_NAMES) == 4
    assert SPARSE_LOG1P_CHANNELS == frozenset(
        {'carb_intake', 'insulin_combined', 'exercise_equiv'}), \
        f"exercise_equiv must be log1p-encoded like carb, got {SPARSE_LOG1P_CHANNELS}"
    assert RISK_SPACE_CHANNELS == frozenset({'bg_absolute'}), \
        ("only bg is a glucose — the Kovatchev transform must never reach "
         f"exercise_equiv, got {RISK_SPACE_CHANNELS}")


# balanced pool's fitted exercise_equiv stats + zero-RAW z; move with POOL, refit if it changes.
_BALANCED_EXERCISE_MEAN = 0.025454530988451768
_BALANCED_EXERCISE_STD = 0.18077422733814857
_BALANCED_EXERCISE_ZERO_Z = -0.1408083886
# largest raw exercise cell over the balanced pool, so the grid spans the whole trained range.
_BALANCED_EXERCISE_MAX_RAW = 22.4029


def _cwd_stats_or_skip() -> dict:
    """The CWD ``normalization_stats.json``: the balanced pool's fit."""
    import os
    from normalization import load_normalization_stats, NORM_STATS_FILE
    if not os.path.exists(NORM_STATS_FILE):
        pytest.skip(f"{NORM_STATS_FILE} required for the fitted-stats gates")
    return load_normalization_stats()


def test_exercise_channel_roundtrip_and_zero_baseline():
    """feat 3 survives normalize→denormalize, and a no-session cell lands on the fitted
    zero-RAW baseline, not on 0.0. Neither failure raises on its own: a Kovatchev
    transform on exercise breaks the round-trip, and a rescaling — or the wrong pool's
    stats — moves the baseline a masked patch fills with, retraining the channel."""
    from normalization import (CHANNEL_NAMES, normalize, denormalize,
                               SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS)

    stats = _cwd_stats_or_skip()
    ex_col = CHANNEL_NAMES.index('exercise_equiv')
    assert ex_col == 3, f"exercise_equiv must be channel 3, got {ex_col}"
    assert 'exercise_equiv' in SPARSE_LOG1P_CHANNELS
    assert 'exercise_equiv' not in RISK_SPACE_CHANNELS

    # the CWD file is the balanced pool's four-key fit
    ex_stats = stats['exercise_equiv']
    assert ex_stats['mean'] == pytest.approx(_BALANCED_EXERCISE_MEAN, abs=1e-15), \
        f"exercise_equiv mean {ex_stats['mean']!r} is not the balanced pool's fit"
    assert ex_stats['std'] == pytest.approx(_BALANCED_EXERCISE_STD, abs=1e-15), \
        f"exercise_equiv std {ex_stats['std']!r} is not the balanced pool's fit"

    # round-trip over the whole trained g/step range
    grid = np.linspace(0.0, _BALANCED_EXERCISE_MAX_RAW, 4096, dtype=np.float32)
    raw = np.zeros((grid.size, len(CHANNEL_NAMES)), dtype=np.float32)
    raw[:, ex_col] = grid
    raw[:, CHANNEL_NAMES.index('bg_absolute')] = 120.0  # legal f argument
    back = denormalize(normalize(raw, stats), stats)
    max_err = float(np.abs(np.asarray(back)[:, ex_col] - grid).max())
    assert max_err < 1e-4, (
        f"exercise_equiv round-trip max abs error {max_err:.3e} >= 1e-4 over "
        f"[0, {_BALANCED_EXERCISE_MAX_RAW}] g/step")

    # the zero-RAW baseline: what a cell announcing no session carries
    zero_z = float(normalize(
        np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), stats)[0, ex_col])
    assert abs(zero_z - _BALANCED_EXERCISE_ZERO_Z) < 1e-9, (
        f"z(raw 0) for exercise_equiv = {zero_z!r}, expected "
        f"{_BALANCED_EXERCISE_ZERO_Z} — wrong statistics or a rescaled channel")
    assert zero_z != 0.0, \
        "a sparse log1p channel's zero-dose baseline must not collapse to z=0"

    # large normalized range is no reason to rescale: insulin reaches a comparable z at its own max.
    z_max = float(normalize(raw, stats)[-1, ex_col])
    print(f"\n[DUMP] exercise_channel | col={ex_col} mean={ex_stats['mean']:.12g} "
          f"std={ex_stats['std']:.12g}; roundtrip max abs err={max_err:.3e}; "
          f"z(0)={zero_z:.10f}; z({_BALANCED_EXERCISE_MAX_RAW})={z_max:.4f}")


def test_normalization_stats_at_load_are_complete_and_nondegenerate():
    """One entry per input channel, each with a strictly positive std.
    A three-key file raises ``KeyError`` in ``data.py``, loud. A four-key file with
    ``std: 0.0`` does not: the pipeline divides by ``0 + 1e-8``, scaling feat 3 by
    ~1e8, and trains to completion behind a plausible validation table."""
    import json
    from normalization import (CHANNEL_NAMES, load_normalization_stats,
                               NORM_STATS_FILE)

    stats = _cwd_stats_or_skip()
    assert set(stats) == set(CHANNEL_NAMES), (
        f"loaded stats keys {sorted(stats)} != CHANNEL_NAMES {sorted(CHANNEL_NAMES)}")
    for name in CHANNEL_NAMES:
        assert set(stats[name]) >= {'mean', 'std'}, f"{name} stats = {stats[name]}"
        assert math.isfinite(stats[name]['mean']), f"non-finite mean for {name}"
        assert math.isfinite(stats[name]['std']), f"non-finite std for {name}"
        assert stats[name]['std'] > 0.0, f"non-positive std for {name}"

    # the loader itself must refuse both shapes, not just this file
    import tempfile
    with open(NORM_STATS_FILE) as f:
        good = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        short = {k: v for k, v in good.items() if k != 'exercise_equiv'}
        short_path = f"{tmp}/short.json"
        with open(short_path, 'w') as f:
            json.dump(short, f)
        with pytest.raises((KeyError, ValueError)):
            load_normalization_stats(short_path)

        degenerate = {k: dict(v) for k, v in good.items()}
        degenerate['exercise_equiv']['std'] = 0.0
        degen_path = f"{tmp}/degenerate.json"
        with open(degen_path, 'w') as f:
            json.dump(degenerate, f)
        with pytest.raises((KeyError, ValueError)):
            load_normalization_stats(degen_path)

    print(f"\n[DUMP] stats_at_load | {len(stats)} keys == CHANNEL_NAMES, all std > 0; "
          f"three-key and std=0 files both rejected at load ✓")


def test_normalize_denormalize_roundtrip():
    """``denormalize(normalize(x))`` is the identity for dense and sparse channels
    alike, and the numpy and torch branches agree."""
    import torch
    from normalization import (normalize, denormalize, CHANNEL_NAMES,
                               SPARSE_LOG1P_CHANNELS)

    stats = _get_stats()
    rng = np.random.default_rng(0)
    # non-negative: sparse path is log1p(max(x,0)); bg must sit in [BG_CLAMP_MIN,MAX], f clamps.
    import T1DMSIM.simulator as sim
    raw = rng.uniform(0.0, 50.0, size=(7, len(CHANNEL_NAMES))).astype(np.float32)
    bg_col = CHANNEL_NAMES.index('bg_absolute')
    raw[:, bg_col] = rng.uniform(
        sim.BG_CLAMP_MIN + 5.0, sim.BG_CLAMP_MAX - 5.0, size=raw.shape[0]
    ).astype(np.float32)

    norm = normalize(raw, stats)
    back_np = denormalize(norm, stats)
    np.testing.assert_allclose(back_np, raw, rtol=1e-4, atol=1e-3)

    # the torch branch must match the numpy branch one-to-one
    back_torch = denormalize(torch.tensor(norm), stats)
    assert isinstance(back_torch, torch.Tensor)
    np.testing.assert_allclose(back_torch.numpy(), back_np, rtol=1e-5, atol=1e-5)

    # the fixture must actually exercise the sparse path
    assert any(n in SPARSE_LOG1P_CHANNELS for n in CHANNEL_NAMES), \
        "no sparse channel present — log1p/expm1 path untested"
    max_err = float(np.abs(back_np - raw).max())
    print(f"\n[DUMP] norm_roundtrip | channels={len(CHANNEL_NAMES)}, "
          f"sparse={sorted(SPARSE_LOG1P_CHANNELS)}, max abs err={max_err:.2e}")


def test_bg_risk_roundtrip_and_last_bg_anchor():
    """feat 0 is ``z(f(bg))``, so the two crossings must compose to the identity.
    bg recovers through ``z → f_inv`` to ≈1e-3 relative, carb/insulin through
    ``log1p → expm1`` near-exactly. ``last_bg_mgdl_from_context`` is the same inverse,
    and the train and inference anchors must both go through it."""
    import torch
    from normalization import normalize, denormalize, CHANNEL_NAMES
    from utils import last_bg_mgdl_from_context
    from config import PATCH_SIZE, N_INPUT_FEATURES

    stats = _get_stats()
    bg_col = CHANNEL_NAMES.index('bg_absolute')
    carb_col = CHANNEL_NAMES.index('carb_intake')
    ins_col = CHANNEL_NAMES.index('insulin_combined')

    # round-trip over a bg grid at fixed non-negative carb/insulin
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

    # a context whose rightmost bg cell is a chosen anchor's normalized value
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
    """``(M, S)`` mg/dL targets, one row per HEAD SLOT, not per horizon patch."""
    from data import T1DMDataset
    from config import (PREDICTION_PATCHES, PATCH_SIZE, PATCH_DIM,
                        N_INPUT_FEATURES, MIN_CONTEXT_PATCHES,
                        MAX_MASKED_PATCHES)

    assert PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES, \
        "PATCH_DIM must be PATCH_SIZE*N_INPUT_FEATURES (the bit is feat 4, inside it)"

    stats = _get_stats()
    dataset = T1DMDataset(master_seed=42, total_steps=10, batch_size=2, normalization_stats=stats)
    sample = dataset[0]

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

    # padded slots gather patch 0 via mask_idx, so row count is fixed at M, never the masked count.
    assert targets.shape == (MAX_MASKED_PATCHES, PATCH_SIZE), \
        f"targets shape {targets.shape} != {(MAX_MASKED_PATCHES, PATCH_SIZE)}"
    import T1DMSIM.simulator as sim
    assert (targets >= sim.BG_CLAMP_MIN - 1e-3).all(), "BG target must be mg/dL (>= clamp floor)"

    bfd = sample['bg_formula_data']
    for key in ('mask_idx', 'valid', 'anchor_bg', 'd', 'slot_hour'):
        assert bfd[key].shape == (MAX_MASKED_PATCHES,), \
            f"bg_formula_data['{key}'] must be (M,), got {bfd[key].shape}"
    assert bool(bfd['valid'].any()), "a sample must carry at least one masked patch"
    assert int(bfd['valid'].sum()) <= MAX_MASKED_PATCHES


def test_masked_set_always_conditioned():
    """On a masked patch bg is zeroed while carb, insulin and exercise keep their true
    or announced values, at every position, masked or visible. The masked set comes
    from ``bg_formula_data``, never position. Exercise sessions are rarer than meals, so
    feat 3 is guarded on never being the literal-0.0 fill, not on being non-zero somewhere."""
    from data import (_build_sample, _make_simulator, simulate_discard_warmup,
                      ON_THE_FLY_SIM_HOURS)
    from config import (PATCH_SIZE, N_INPUT_FEATURES, PATCH_DIM, CHANNEL_TO_FEAT)
    from normalization import CHANNEL_NAMES, normalize

    assert CHANNEL_TO_FEAT == {0: 1, 1: 2, 2: 3}, \
        "carb -> feat1, insulin -> feat2, exercise -> feat3"

    stats = _get_stats()
    sim = _make_simulator(patient_seed=321, uniform_skills=False)
    data = simulate_discard_warmup(sim, ON_THE_FLY_SIM_HOURS)
    icr = float(sim.patient.icr)

    s = _build_sample(data=data, icr=icr, stats=stats,
                      rng=np.random.default_rng(7))
    assert 'reveal_mask' not in s, "a built sample must carry no reveal_mask"
    patches = s['patches'].numpy()
    assert patches.shape[1] == PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES, \
        "the patch row is the bare feature flat — no trailing mask bits"

    T = patches.shape[0]
    feat_grid = patches.reshape(T, PATCH_SIZE, N_INPUT_FEATURES)
    bfd = s['bg_formula_data']
    masked = np.zeros(T, dtype=bool)
    masked[bfd['mask_idx'][bfd['valid']]] = True

    carb_feat = CHANNEL_TO_FEAT[0]
    insulin_feat = CHANNEL_TO_FEAT[1]
    exercise_feat = CHANNEL_TO_FEAT[2]
    # bg is the predicted target, always zeroed on a masked patch
    assert (feat_grid[masked, :, 0] == 0.0).all(), \
        "bg feat 0 must be zeroed on every masked patch"
    # carb/insulin carry announced doses over the WHOLE window; some non-zero on a real trajectory.
    assert np.any(feat_grid[masked, :, carb_feat] != 0.0), \
        "carb feat 1 must carry true doses on the masked patches"
    assert np.any(feat_grid[masked, :, insulin_feat] != 0.0), \
        "insulin feat 2 must carry true doses on the masked patches"
    # no-session exercise cell is normalize(0), NOT 0.0; all-zero column means gather dropped it.
    exercise_baseline = float(normalize(
        np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), stats)[0, exercise_feat])
    ex_col = feat_grid[:, :, exercise_feat]
    assert not np.all(ex_col == 0.0), (
        f"exercise feat {exercise_feat} is the literal-0.0 fill — an unannounced "
        f"cell must carry the zero-RAW baseline {exercise_baseline:.6f}")
    assert (ex_col >= exercise_baseline - 1e-4).all(), (
        f"exercise feat {exercise_feat} below the zero-RAW baseline "
        f"{exercise_baseline:.6f} (min {float(ex_col.min()):.6f})")
    print(f"\n[DUMP] masked_set | PATCH_DIM={PATCH_DIM}; masked patches "
          f"{sorted(np.flatnonzero(masked).tolist())} of {T}; bg feat0 zeroed there, "
          f"carb feat{carb_feat}/insulin feat{insulin_feat}/exercise "
          f"feat{exercise_feat} conditioned (exercise min "
          f"{float(ex_col.min()):.6f} vs baseline {exercise_baseline:.6f}) ✓")


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
        # the same seed must reproduce the pick bit for bit
        s2 = _pick_pred_start_step(n_steps, n_ctx, room, np.random.default_rng(seed))
        assert s == s2, f"non-deterministic pick at seed {seed}: {s} != {s2}"
        starts.add(s)
    print(f"\n[DUMP] pick_pred_start | {len(starts)} distinct starts over 64 seeds")
    assert len(starts) > 1, "picker should explore more than one window across seeds"

    # no legal window (room + context exceed the trajectory) => None
    assert _pick_pred_start_step(n_ctx * PATCH_SIZE, n_ctx, room, np.random.default_rng(0)) is None


def test_collation_padding():
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

    for i in range(B):
        n_ctx = samples[i]['n_context_patches']
        n_total = n_ctx + PREDICTION_PATCHES
        n_pad = T - n_total
        if n_pad > 0:
            assert (patches[i, :n_pad] == 0).all(), f"Padding not zeroed for sample {i}"
            # pad positions are blocked in attention, except the diagonal
            pad_mask = attn_mask[i, :n_pad, :].clone()
            pad_mask[torch.arange(n_pad), torch.arange(n_pad)] = False
            assert not pad_mask.any(), f"Padding rows (off-diagonal) should be blocked for sample {i}"
            col_mask = attn_mask[i, :, :n_pad].clone()
            col_mask[torch.arange(n_pad), torch.arange(n_pad)] = False
            assert not col_mask.any(), f"Padding cols (off-diagonal) should be blocked for sample {i}"


def test_next_window_batch_shape_and_space():
    """``next_window`` is window ``k`` shifted forward one horizon, built and normalized
    entirely inside data.py. Its pred zone must keep bg zeroed, or future bg leaks.
    At weight 0 the key is absent altogether."""
    from data import T1DMDataset, collate_fn
    from config import (PREDICTION_PATCHES, PATCH_DIM, N_INPUT_FEATURES,
                        NON_MASKABLE_FEATS, MAX_MASKED_PATCHES,
                        TIME_PROBE_ENABLED, TIME_PROBE_CROSS_WINDOW_WEIGHT)

    if not (TIME_PROBE_ENABLED and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0):
        pytest.skip("cross-window probe off — data.py ships no next_window key")

    stats = _get_stats()
    B = 3
    dataset = T1DMDataset(master_seed=42, total_steps=10, batch_size=B,
                          normalization_stats=stats)
    samples = [dataset[i] for i in range(B)]

    # window k+1 carries ONE masked span (its forecast zone): per-slot arrays plus (B,) scalars.
    for s in samples:
        assert 'next_window' in s, "probe on => each sample must carry next_window"
        nw = s['next_window']
        assert set(nw) == {'patches', 'mask_idx', 'valid_slots', 'anchor_bg', 'd',
                           'slot_hour', 'last_bg', 'pred_start_hour', 'valid'}, \
            f"unexpected next_window keys {sorted(nw)}"

    batch = collate_fn(samples)
    assert 'next_window' in batch, "collate must stack next_window when samples carry it"
    nw = batch['next_window']
    max_T = batch['patches'].shape[1]
    M = MAX_MASKED_PATCHES

    assert nw['patches'].shape == (B, max_T, PATCH_DIM), \
        f"next_window patches {tuple(nw['patches'].shape)} != {(B, max_T, PATCH_DIM)}"
    assert nw['patches'].dtype == torch.float32
    assert nw['last_bg'].shape == (B,) and nw['last_bg'].dtype == torch.float32
    assert nw['pred_start_hour'].shape == (B,)
    assert nw['valid'].shape == (B,) and nw['valid'].dtype == torch.bool
    # the masked span is the right-edge zone: exactly PREDICTION_PATCHES valid slots
    assert nw['mask_idx'].shape == (B, M) and nw['mask_idx'].dtype == torch.int64
    assert nw['valid_slots'].shape == (B, M) and nw['valid_slots'].dtype == torch.bool
    assert nw['anchor_bg'].shape == (B, M)
    assert nw['d'].shape == (B, M)
    assert nw['slot_hour'].shape == (B, M)
    assert int(nw['valid_slots'].sum(dim=1).min()) == PREDICTION_PATCHES, \
        "the shifted window's masked set is its own right-edge forecast span"
    # padded slot still needs a legal mg/dL anchor: units tripwire reads all M, valid discards them.
    import T1DMSIM.simulator as sim
    assert (nw['anchor_bg'] >= sim.BG_CLAMP_MIN - 1e-3).all() and \
           (nw['anchor_bg'] <= sim.BG_CLAMP_MAX + 1e-3).all(), \
        "every next_window anchor, padded slots included, must be physical mg/dL"

    # NIGHT_LONG_HORIZON_HOURS (8h) leaves room for one 2h shift, so every next window is in range.
    assert bool(nw['valid'].all()), \
        f"all next windows should be valid at default config, got {nw['valid'].tolist()}"

    # last_bg is physical mg/dL, so it clears the units tripwire
    assert (nw['last_bg'] >= sim.BG_CLAMP_MIN - 1e-3).all() and \
           (nw['last_bg'] <= sim.BG_CLAMP_MAX + 1e-3).all(), \
        "next_window last_bg must be physical mg/dL"

    # the shifted window's own pred zone keeps every NON_MASKABLE feat zeroed
    grid = nw['patches'].reshape(B, max_T, PATCH_DIM // N_INPUT_FEATURES, N_INPUT_FEATURES)
    pred_grid = grid[:, max_T - PREDICTION_PATCHES:, :, :]      # (B, P, S, F)
    for feat_idx in NON_MASKABLE_FEATS:
        assert (pred_grid[..., feat_idx] == 0.0).all(), \
            f"shifted pred zone leaked a non-zero feat {feat_idx} (bg must stay zeroed)"

    print(f"\n[DUMP] next_window | patches {tuple(nw['patches'].shape)}  "
          f"valid={nw['valid'].tolist()}  last_bg~"
          f"[{float(nw['last_bg'].min()):.0f},{float(nw['last_bg'].max()):.0f}] mg/dL  "
          f"pred-zone feat{NON_MASKABLE_FEATS} zeroed ✓")
