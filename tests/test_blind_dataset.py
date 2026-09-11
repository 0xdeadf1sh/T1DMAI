"""The ``blind`` masked-channel policy, at the dataset boundary.

``blind=True`` withholds feats 1-3 on a masked patch too, at ``zero_dose_fill``'s
``normalize(0)``. ``train_blind.py`` is the only caller; ``blind=False`` is what ships.
"""

import numpy as np
import torch

from config import MASKABLE_FEATS, N_INPUT_FEATURES, PATCH_DIM, PATCH_SIZE

BLIND_SEED = 20260815


def _get_stats():
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def _build(blind: bool, stats, seed: int = BLIND_SEED) -> dict:
    """One sample at a fixed seed; same patient and same rng, so only ``blind`` differs."""
    from data import (_build_sample, _make_simulator, simulate_discard_warmup,
                      ON_THE_FLY_SIM_HOURS)
    sim = _make_simulator(seed, uniform_skills=False)
    data = simulate_discard_warmup(sim, ON_THE_FLY_SIM_HOURS)
    return _build_sample(
        data=data, icr=float(sim.patient.icr), stats=stats,
        rng=np.random.default_rng(seed ^ 0xDEADBEEF), blind=blind,
    )


def test_blind_withholds_every_dose_cell_of_a_masked_patch_and_nothing_else():
    """The flag's whole footprint: feats 1-3 of masked patches, every cell, at exactly the fill.

    A leak is silent: an announced dose surviving a masked patch still trains and validates,
    under a blind name. The masked SET must match too, or the tables aren't comparable.
    """
    from data import BG_MASKED_FEAT, zero_dose_fill

    stats = _get_stats()
    plain = _build(blind=False, stats=stats)
    blind = _build(blind=True, stats=stats)
    fill = zero_dose_fill(stats)

    p, b = plain['patches'], blind['patches']
    assert p.shape == b.shape, f"{tuple(p.shape)} vs {tuple(b.shape)}"
    assert p.shape[-1] == PATCH_DIM

    # Announcement bit IS the masked set; require agreement before comparing anything keyed on it.
    bit_p = p[:, BG_MASKED_FEAT::N_INPUT_FEATURES][:, 0] > 0.5
    bit_b = b[:, BG_MASKED_FEAT::N_INPUT_FEATURES][:, 0] > 0.5
    assert torch.equal(bit_p, bit_b), "the blind flag moved the sampled mask"
    assert torch.equal(
        torch.from_numpy(plain['bg_formula_data']['mask_idx']),
        torch.from_numpy(blind['bg_formula_data']['mask_idx'])), "mask_idx moved"
    n_masked = int(bit_b.sum())
    assert n_masked > 0, "the fixed seed drew no masked patch — pick another"

    print(f"\n[DUMP] seq_len={p.shape[0]} masked={n_masked} "
          f"fill={{{', '.join(f'{f}: {z:+.4f}' for f, z in sorted(fill.items()))}}}")

    for feat_idx in MASKABLE_FEATS:
        cells = b[bit_b, feat_idx::N_INPUT_FEATURES]              # (n_masked, S)
        assert cells.shape[-1] == PATCH_SIZE, (
            f"feat {feat_idx} stride gave {cells.shape[-1]} columns, expected "
            f"PATCH_SIZE={PATCH_SIZE} — the step-major layout is broken")
        expected = torch.full_like(cells, float(fill[feat_idx]))
        assert torch.equal(cells, expected), (
            f"feat {feat_idx} on a masked patch is not the fill: "
            f"max|delta| {float((cells - expected).abs().max()):.6g}")
        announced = p[bit_p, feat_idx::N_INPUT_FEATURES]
        print(f"[DUMP]   feat {feat_idx}: announced range "
              f"[{float(announced.min()):+.4f}, {float(announced.max()):+.4f}] "
              f"-> {float(fill[feat_idx]):+.4f}")

    # A masked span with no dose trivially satisfies the above; across all three feats it cannot.
    dose_cols = [f for feat in MASKABLE_FEATS
                 for f in range(feat, PATCH_DIM, N_INPUT_FEATURES)]
    assert not torch.equal(p[bit_p][:, dose_cols], b[bit_b][:, dose_cols]), (
        "no masked dose cell moved — this seed announces nothing in its masked "
        "span, so the assertions above have no subject; pick another")

    for feat_idx in (0, BG_MASKED_FEAT):
        assert torch.equal(p[bit_p, feat_idx::N_INPUT_FEATURES],
                           b[bit_b, feat_idx::N_INPUT_FEATURES]), (
            f"the blind flag moved feat {feat_idx} on a masked patch")

    assert torch.equal(p[~bit_p], b[~bit_b]), (
        "the blind flag changed a VISIBLE patch — it must only reach the "
        "patches feat 4 announces")

    for k in sorted(plain['bg_formula_data']):
        a = np.asarray(plain['bg_formula_data'][k])
        c = np.asarray(blind['bg_formula_data'][k])
        assert np.array_equal(a, c), f"bg_formula_data[{k!r}] moved under blind"
    assert torch.equal(plain['targets'], blind['targets'])


def test_the_dataset_honours_the_flag_end_to_end():
    """The THREADING, through ``__getitem__``: a dataset that stored the flag and never passed
    it on satisfies every other test here. Both policies are checked on the same index.
    """
    from data import BG_MASKED_FEAT, T1DMDataset, zero_dose_fill

    stats = _get_stats()
    kw = dict(master_seed=BLIND_SEED, total_steps=4, batch_size=1,
              normalization_stats=stats, patient_uniform_sample_prob=0.0)
    plain = T1DMDataset(**kw)[0]
    blind = T1DMDataset(**kw, blind=True)[0]
    fill = zero_dose_fill(stats)

    p, b = plain['patches'], blind['patches']
    bit = b[:, BG_MASKED_FEAT::N_INPUT_FEATURES][:, 0] > 0.5
    assert int(bit.sum()) > 0, "the dataset drew no masked patch at this index"
    print(f"\n[DUMP] dataset index 0: {int(bit.sum())} masked patches of {b.shape[0]}")

    for feat_idx in MASKABLE_FEATS:
        cells = b[bit, feat_idx::N_INPUT_FEATURES]
        assert torch.equal(cells, torch.full_like(cells, float(fill[feat_idx]))), (
            f"T1DMDataset(blind=True) did not blind feat {feat_idx} — the flag is "
            "stored but not threaded to the sample builder")
    # Three dose channels TOGETHER: a sparse channel with no event already carries normalize(0).
    dose_cols = [f for feat in MASKABLE_FEATS
                 for f in range(feat, PATCH_DIM, N_INPUT_FEATURES)]
    assert not torch.equal(p[bit][:, dose_cols], b[bit][:, dose_cols]), (
        "the DEFAULT dataset produced the same masked dose cells as the blind one "
        "— either blind=False is not the announced policy, or this index announces "
        "nothing at all and the assertions above have no subject")


def test_next_window_is_blinded_too():
    """``next_window`` carries its own right-edge span. Left announced it hands the
    trunk true doses the primary window withholds — two conventions, one step."""
    from config import PREDICTION_PATCHES
    from data import BG_MASKED_FEAT, zero_dose_fill

    stats = _get_stats()
    blind = _build(blind=True, stats=stats)
    nw = blind.get('next_window')
    if nw is None:
        import pytest
        pytest.skip("cross-window time probe is off (TIME_PROBE_CROSS_WINDOW_WEIGHT)")

    fill = zero_dose_fill(stats)
    q = nw['patches']
    bit = q[:, BG_MASKED_FEAT::N_INPUT_FEATURES][:, 0] > 0.5
    assert int(bit.sum()) == PREDICTION_PATCHES, (
        f"next_window masks {int(bit.sum())} patches, expected the "
        f"{PREDICTION_PATCHES}-patch forecast span")
    print(f"\n[DUMP] next_window valid={nw['valid']} masked={int(bit.sum())}")
    for feat_idx in MASKABLE_FEATS:
        cells = q[bit, feat_idx::N_INPUT_FEATURES]
        expected = torch.full_like(cells, float(fill[feat_idx]))
        assert torch.equal(cells, expected), (
            f"next_window feat {feat_idx} on a masked patch is not the fill")


def test_the_fill_is_the_no_dose_baseline_inference_already_writes():
    """``zero_dose_fill`` == what ``inference`` puts in an un-announced slot, so a blind model
    rolls in-distribution with no ``inference.py`` change. Measured off ``_build_patches_tensor``'s
    output, never recomputed: a second copy of the formula would only agree with itself.
    """
    from inference import _build_patches_tensor
    from data import zero_dose_fill

    stats = _get_stats()
    fill = zero_dose_fill(stats)
    n_ctx = 8
    ctx = torch.zeros(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)
    patches, _ = _build_patches_tensor(ctx, None, stats, None)
    pred = patches[n_ctx:]                             # the un-announced zone

    for feat_idx in MASKABLE_FEATS:
        cells = pred[:, feat_idx::N_INPUT_FEATURES]
        expected = torch.full_like(cells, float(fill[feat_idx]))
        assert torch.equal(cells, expected), (
            f"feat {feat_idx}: blind fill {fill[feat_idx]:.6f} is not what "
            f"inference writes ({float(cells.flatten()[0]):.6f})")

    # not z = 0: on a log1p channel z = 0 inverts to expm1(mean), a phantom dose
    from normalization import CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS
    for feat_idx, z in sorted(fill.items()):
        name = CHANNEL_NAMES[feat_idx]
        assert name in SPARSE_LOG1P_CHANNELS, (
            f"{name} is not a sparse log1p channel — the phantom-dose argument "
            "below does not apply to it and this test needs rewriting")
        phantom = float(np.expm1(stats[name]['mean']))
        assert abs(z) > 1e-3, (
            f"{name}: the fill is z={z:.6g}, indistinguishable from the z=0 "
            "sentinel — the two would decode to the same value")
        print(f"[DUMP] {name}: fill z={z:+.4f}, and z=0 would decode to "
              f"{phantom:.4f} per step")
