"""Tests for normalization.py and data.py — shapes, normalization, collation.

Risk-space redesign: 5-feature input stack
[bg, carb, insulin, exercise, bg_masked] — the 4 temporal sin/cos features are
dropped — over 4 NORMALIZED channels (bg_absolute in Kovatchev RISK space;
carb_intake / insulin_combined / exercise_equiv log1p).  Feat 4 is a per-patch
BIT and is deliberately not a channel: it has no statistics and never sees the
z-score.  ``exercise_equiv`` is carbohydrate-EQUIVALENT glucose disposal in
g/step, so it takes carb's encoding and never the Kovatchev transform.

The prediction zone is gone.  A sample is a window of ``T`` patches, each visible
or masked; targets are the raw mg/dL BG of the ``MAX_MASKED_PATCHES`` head slots,
``(M, S)``.  A masked patch withholds bg (feat 0) and announces itself in feat 4
while carb / insulin / exercise ALWAYS carry their true or announced values (the
model is always conditioned).  The conditioned/unconditioned dichotomy and the
trailing mask-bit tier are gone — there is no ``loss_mask`` and no
``reveal_mask``.  Patches are ``(T, PATCH_DIM=30)`` = PATCH_SIZE*N_INPUT_FEATURES.
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


def test_channel_names_are_the_four_input_signals():
    """CHANNEL_NAMES is the 4 input SIGNAL channels in input-feature order — bg,
    carb, insulin, exercise — with bg_delta / IS / HGO dropped.  The order pins
    every channel index in the project, so this is a literal pin, not a count.

    It stays at FOUR against ``N_INPUT_FEATURES == 5``: ``bg_masked`` (feat 4) is
    a BIT announcing that feat 0 is withheld, not a measured signal, so it carries
    no mean, no std and no log1p encoding and gets no entry here.  The old
    ``len(CHANNEL_NAMES) == N_INPUT_FEATURES`` identity is exactly what breaks
    first, so the two counts are pinned separately.

    ``exercise_equiv`` is a log1p channel like carb, NOT a risk-space one: it is
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
    # The normalized channels occupy the LEADING columns and the bit follows them.
    assert BG_MASKED_FEAT == len(CHANNEL_NAMES) == 4
    assert SPARSE_LOG1P_CHANNELS == frozenset(
        {'carb_intake', 'insulin_combined', 'exercise_equiv'}), \
        f"exercise_equiv must be log1p-encoded like carb, got {SPARSE_LOG1P_CHANNELS}"
    assert RISK_SPACE_CHANNELS == frozenset({'bg_absolute'}), \
        ("only bg is a glucose — the Kovatchev transform must never reach "
         f"exercise_equiv, got {RISK_SPACE_CHANNELS}")


# --- The exercise input channel (feat 3) ------------------------------------
# The balanced pool's fitted exercise_equiv statistics, and the normalized value
# a zero-RAW (no session) cell takes under them.  Both are exact: the fit is a
# full pass over the pool, and the baseline is the float32 z the input pipeline
# actually writes.  They move with the POOL — refit against
# ``normalization_stats.json`` whenever the cache geometry changes.  The channel is g/step carbohydrate-EQUIVALENT glucose
# disposal — rescaling it to an intensity trains a different quantity.
_BALANCED_EXERCISE_MEAN = 0.025454530988451768
_BALANCED_EXERCISE_STD = 0.18077422733814857
_BALANCED_EXERCISE_ZERO_Z = -0.1408083886
# Largest raw exercise cell over the balanced pool, so the round-trip grid below
# spans the whole trained range rather than a comfortable middle.
_BALANCED_EXERCISE_MAX_RAW = 22.4029


def _cwd_stats_or_skip() -> dict:
    """Load the CWD ``normalization_stats.json`` — the balanced pool's fit."""
    import os
    from normalization import load_normalization_stats, NORM_STATS_FILE
    if not os.path.exists(NORM_STATS_FILE):
        pytest.skip(f"{NORM_STATS_FILE} required for the fitted-stats gates")
    return load_normalization_stats()


def test_exercise_channel_roundtrip_and_zero_baseline():
    """§2.4 gate: feat 3 survives normalize→denormalize, and a no-session cell
    lands on the fitted zero-RAW baseline rather than on 0.0.

    Two failures this catches, neither of which raises on its own: a Kovatchev
    transform applied to exercise (it is a carb-equivalent disposal rate, not a
    glucose) would break the round-trip outright, and a rescaling to an
    intensity — or the wrong pool's statistics — would move the baseline a
    masked patch is filled with, retraining the channel on a different quantity
    at the same shapes.
    """
    from normalization import (CHANNEL_NAMES, normalize, denormalize,
                               SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS)

    stats = _cwd_stats_or_skip()
    ex_col = CHANNEL_NAMES.index('exercise_equiv')
    assert ex_col == 3, f"exercise_equiv must be channel 3, got {ex_col}"
    assert 'exercise_equiv' in SPARSE_LOG1P_CHANNELS
    assert 'exercise_equiv' not in RISK_SPACE_CHANNELS

    # The CWD file is the balanced pool's four-key fit (§2.0's stats swap).
    ex_stats = stats['exercise_equiv']
    assert ex_stats['mean'] == pytest.approx(_BALANCED_EXERCISE_MEAN, abs=1e-15), \
        f"exercise_equiv mean {ex_stats['mean']!r} is not the balanced pool's fit"
    assert ex_stats['std'] == pytest.approx(_BALANCED_EXERCISE_STD, abs=1e-15), \
        f"exercise_equiv std {ex_stats['std']!r} is not the balanced pool's fit"

    # (a) Round-trip over the whole trained g/step range.
    grid = np.linspace(0.0, _BALANCED_EXERCISE_MAX_RAW, 4096, dtype=np.float32)
    raw = np.zeros((grid.size, len(CHANNEL_NAMES)), dtype=np.float32)
    raw[:, ex_col] = grid
    raw[:, CHANNEL_NAMES.index('bg_absolute')] = 120.0  # legal f argument
    back = denormalize(normalize(raw, stats), stats)
    max_err = float(np.abs(np.asarray(back)[:, ex_col] - grid).max())
    assert max_err < 1e-4, (
        f"exercise_equiv round-trip max abs error {max_err:.3e} >= 1e-4 over "
        f"[0, {_BALANCED_EXERCISE_MAX_RAW}] g/step")

    # (b) The zero-RAW baseline: what a cell announcing no session carries.
    zero_z = float(normalize(
        np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), stats)[0, ex_col])
    assert abs(zero_z - _BALANCED_EXERCISE_ZERO_Z) < 1e-9, (
        f"z(raw 0) for exercise_equiv = {zero_z!r}, expected "
        f"{_BALANCED_EXERCISE_ZERO_Z} — wrong statistics or a rescaled channel")
    assert zero_z != 0.0, \
        "a sparse log1p channel's zero-dose baseline must not collapse to z=0"

    # The normalized range is large and that is not a reason to rescale: insulin
    # already reaches a comparable z at its own maximum.
    z_max = float(normalize(raw, stats)[-1, ex_col])
    print(f"\n[DUMP] exercise_channel | col={ex_col} mean={ex_stats['mean']:.12g} "
          f"std={ex_stats['std']:.12g}; roundtrip max abs err={max_err:.3e}; "
          f"z(0)={zero_z:.10f}; z({_BALANCED_EXERCISE_MAX_RAW})={z_max:.4f}")


def test_normalization_stats_at_load_are_complete_and_nondegenerate():
    """§2.4 gate: the loaded statistics carry one entry per input channel and a
    strictly positive std for each.

    A three-key file against the four-CHANNEL fit raises ``KeyError`` in
    ``data.py`` — loud.  A FOUR-key file carrying ``std: 0.0`` does not: the
    input pipeline divides by ``0 + 1e-8`` and feat 3 is multiplied by ~1e8,
    which trains to completion and reports a plausible validation table.  Both
    shapes must be refused where they enter, at load.
    """
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

    # The loader itself must refuse both degenerate shapes, not just this file.
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
    ``(M, S)`` mg/dL BG target, one row per HEAD SLOT (not per horizon patch).
    The retired per-channel loss_mask AND the reveal_mask are both gone — no mask
    key of any kind ships in a sample."""
    from data import T1DMDataset
    from config import (PREDICTION_PATCHES, PATCH_SIZE, PATCH_DIM,
                        N_INPUT_FEATURES, MIN_CONTEXT_PATCHES,
                        MAX_MASKED_PATCHES)

    assert PATCH_DIM == PATCH_SIZE * N_INPUT_FEATURES, \
        "PATCH_DIM must be PATCH_SIZE*N_INPUT_FEATURES (the bit is feat 4, inside it)"

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

    # BG target is the raw mg/dL trajectory of each of the M head slots.  Padded
    # slots gather patch 0 exactly as mask_idx does, so the row count is FIXED at
    # M and never the masked-patch count.
    assert targets.shape == (MAX_MASKED_PATCHES, PATCH_SIZE), \
        f"targets shape {targets.shape} != {(MAX_MASKED_PATCHES, PATCH_SIZE)}"
    import T1DMSIM.simulator as sim
    assert (targets >= sim.BG_CLAMP_MIN - 1e-3).all(), "BG target must be mg/dL (>= clamp floor)"

    # The per-slot arrays are all (M,) and agree with the targets.
    bfd = sample['bg_formula_data']
    for key in ('mask_idx', 'valid', 'anchor_bg', 'd', 'slot_hour'):
        assert bfd[key].shape == (MAX_MASKED_PATCHES,), \
            f"bg_formula_data['{key}'] must be (M,), got {bfd[key].shape}"
    assert bool(bfd['valid'].any()), "a sample must carry at least one masked patch"
    assert int(bfd['valid'].sum()) <= MAX_MASKED_PATCHES


def test_masked_set_always_conditioned():
    """The model is ALWAYS conditioned: on every MASKED patch bg (feat 0) is
    zeroed while carb (feat 1) / insulin (feat 2) / exercise (feat 3) carry their
    TRUE or announced values.  No reveal mask, no trailing mask bits — patches are
    exactly PATCH_SIZE*N_INPUT_FEATURES wide.

    THE TRAP this guards: the old layout appended mask bits and split the pred
    zone into conditioned/unconditioned regimes.  Here every announceable channel
    is unconditionally present at every position, masked or visible, and the row
    width is the bare feature flat with nothing trailing.

    The masked set comes from ``bg_formula_data``, not from position: a span may
    end at ``T-1``, start at patch 0, or sit between visible patches.

    Exercise sessions are far rarer than meals, so a single window need not
    contain one; feat 3 is guarded on never being the literal-0.0 fill (the shape
    a dropped column takes) rather than on being non-zero somewhere.
    """
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
    # bg (feat 0) is the predicted target — ALWAYS zeroed on a masked patch.
    assert (feat_grid[masked, :, 0] == 0.0).all(), \
        "bg feat 0 must be zeroed on every masked patch"
    # carb (feat 1) / insulin (feat 2) carry the announced true doses over the
    # WHOLE window; on a real trajectory at least some are non-zero.
    assert np.any(feat_grid[masked, :, carb_feat] != 0.0), \
        "carb feat 1 must carry true doses on the masked patches"
    assert np.any(feat_grid[masked, :, insulin_feat] != 0.0), \
        "insulin feat 2 must carry true doses on the masked patches"
    # exercise (feat 3): a no-session cell is normalize(0) for a log1p channel,
    # which is NOT 0.0 — an all-zero column means the gather dropped it.
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
                        NON_MASKABLE_FEATS, MAX_MASKED_PATCHES,
                        TIME_PROBE_ENABLED, TIME_PROBE_CROSS_WINDOW_WEIGHT)

    if not (TIME_PROBE_ENABLED and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0):
        pytest.skip("cross-window probe off — data.py ships no next_window key")

    stats = _get_stats()
    B = 3
    dataset = T1DMDataset(master_seed=42, total_steps=10, batch_size=B,
                          normalization_stats=stats)
    samples = [dataset[i] for i in range(B)]

    # Per-sample sub-dict present and well-formed.  Window k+1 carries ONE masked
    # span — its own right-edge forecast zone — so it ships the same per-slot
    # arrays the main window does, plus the (B,) scalars.
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
    # The per-slot arrays are (B, M) and its masked span is the right-edge zone:
    # exactly PREDICTION_PATCHES valid slots, the rest padded.
    assert nw['mask_idx'].shape == (B, M) and nw['mask_idx'].dtype == torch.int64
    assert nw['valid_slots'].shape == (B, M) and nw['valid_slots'].dtype == torch.bool
    assert nw['anchor_bg'].shape == (B, M)
    assert nw['d'].shape == (B, M)
    assert nw['slot_hour'].shape == (B, M)
    assert int(nw['valid_slots'].sum(dim=1).min()) == PREDICTION_PATCHES, \
        "the shifted window's masked set is its own right-edge forecast span"
    # Padded slots must still carry a legal mg/dL anchor — the forward's (B, M)
    # units tripwire reads all M, valid is what discards them downstream.
    import T1DMSIM.simulator as sim
    assert (nw['anchor_bg'] >= sim.BG_CLAMP_MIN - 1e-3).all() and \
           (nw['anchor_bg'] <= sim.BG_CLAMP_MAX + 1e-3).all(), \
        "every next_window anchor, padded slots included, must be physical mg/dL"

    # At default config NIGHT_LONG_HORIZON_HOURS (8h) leaves room to shift by one
    # 2h horizon, so every sample's next window is in-range.
    assert bool(nw['valid'].all()), \
        f"all next windows should be valid at default config, got {nw['valid'].tolist()}"

    # last_bg is physical mg/dL (survives the forward's units tripwire).
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
