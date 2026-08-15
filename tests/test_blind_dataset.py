"""The ``blind`` masked-channel policy, at the dataset boundary.

``data.T1DMDataset(blind=True)`` withholds the dose channels on a masked patch as
well as bg: feats 1-3 take ``data.zero_dose_fill``'s per-channel ``normalize(0)``
instead of the announced value.  ``train_blind.py`` is the only caller; the
default ``blind=False`` is the announced policy every shipped checkpoint was
trained under.

Three claims, and each is one a plausible implementation could get wrong:

* the flag touches the masked patches' dose cells and NOTHING else — not the
  visible patches, not feat 0, not feat 4, and not the sampler's draw;
* ``blind=False`` is byte-identical to the pre-flag builder, pinned by a digest
  taken at the commit before the flag existed (``tests/test_bitident.py`` pins
  the forward, not the sampler; this closes that gap for the dose cells);
* the fill IS the no-dose baseline ``inference`` writes into an un-announced
  slot, which is what lets a blind model roll long-horizon in-distribution with
  no change to ``inference.py``.
"""

import hashlib

import numpy as np
import torch

from config import MASKABLE_FEATS, N_INPUT_FEATURES, PATCH_DIM, PATCH_SIZE

# Fixed seed for every sample built here.  The digest below is a property of this
# seed, the live config geometry and the loaded normalization stats.
BLIND_SEED = 20260815

# sha256 over the whole sample — patches, targets, n_context_patches, every
# bg_formula_data entry and the whole next_window dict — of the ``blind=False``
# build at seed BLIND_SEED, computed at e644d63, the commit BEFORE the blind flag
# was added.  See ``test_the_default_path_is_byte_identical_through_the_flag``
# for what a mismatch means and when it is legitimate to restamp.
DEFAULT_PATH_DIGEST = (
    '6b6b97543f5bd0760115f4c47c6445716be4b54b1cbcd0855ce37ad3e35e5c89')


def _get_stats():
    """Load or compute normalization stats for the on-the-fly build."""
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def _build(blind: bool, stats, seed: int = BLIND_SEED) -> dict:
    """One on-the-fly sample at a fixed seed, under either policy.

    Everything but ``blind`` is held equal — the same simulator patient, the same
    window-selection rng — so the two builds differ in exactly the thing under
    test.
    """
    from data import (_build_sample, _make_simulator, simulate_discard_warmup,
                      ON_THE_FLY_SIM_HOURS)
    sim = _make_simulator(seed, uniform_skills=False)
    data = simulate_discard_warmup(sim, ON_THE_FLY_SIM_HOURS)
    return _build_sample(
        data=data, icr=float(sim.patient.icr), stats=stats,
        rng=np.random.default_rng(seed ^ 0xDEADBEEF), blind=blind,
    )


def _sample_digest(sample: dict) -> str:
    """sha256 over every array and scalar the sample carries.

    Order-stable (sorted keys), dtype- and shape-tagged, and taken over raw bytes
    — a float that moved in its last bit changes the digest.
    """
    h = hashlib.sha256()

    def feed(name, obj):
        h.update(name.encode())
        if isinstance(obj, (int, float, bool)):
            h.update(np.asarray(obj, dtype=np.float64).tobytes())
            return
        arr = obj.detach().cpu().numpy() if hasattr(obj, 'detach') else np.asarray(obj)
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())

    feed('patches', sample['patches'])
    feed('targets', sample['targets'])
    feed('n_context_patches', sample['n_context_patches'])
    for k in sorted(sample['bg_formula_data']):
        feed(k, sample['bg_formula_data'][k])
    nw = sample.get('next_window')
    if nw is not None:
        for k in sorted(nw):
            feed(f'next_window.{k}', nw[k])
    return h.hexdigest()


def test_blind_withholds_every_dose_cell_of_a_masked_patch_and_nothing_else():
    """The flag's whole footprint: feats 1-3 of the masked patches, all
    ``PATCH_SIZE`` step-major cells of each, at exactly the fill.

    Everything else must be byte-identical to the announced build from the same
    seed — every cell of every VISIBLE patch, and feat 0 / feat 4 of the masked
    ones.  A leak here is silent: an announced dose surviving on a masked patch
    still trains and still validates, and the blind run would simply be measuring
    a partly-conditioned model under a blind name.

    The masked SET must also be identical.  The fill is written after the sampler
    has drawn, so a flag that moved the draw would make the conditioned and blind
    validation tables incomparable — which is the whole point of the fork.
    """
    from data import BG_MASKED_FEAT, zero_dose_fill

    stats = _get_stats()
    plain = _build(blind=False, stats=stats)
    blind = _build(blind=True, stats=stats)
    fill = zero_dose_fill(stats)

    p, b = plain['patches'], blind['patches']
    assert p.shape == b.shape, f"{tuple(p.shape)} vs {tuple(b.shape)}"
    assert p.shape[-1] == PATCH_DIM

    # The announcement bit IS the masked set; read it off both and require they
    # agree before comparing anything keyed on it.
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

    # (a) every dose cell of every masked patch is the fill, exactly.
    for feat_idx in MASKABLE_FEATS:
        cells = b[bit_b, feat_idx::N_INPUT_FEATURES]              # (n_masked, S)
        assert cells.shape[-1] == PATCH_SIZE, (
            f"feat {feat_idx} stride gave {cells.shape[-1]} columns, expected "
            f"PATCH_SIZE={PATCH_SIZE} — the step-major layout is broken")
        expected = torch.full_like(cells, float(fill[feat_idx]))
        assert torch.equal(cells, expected), (
            f"feat {feat_idx} on a masked patch is not the fill: "
            f"max|delta| {float((cells - expected).abs().max()):.6g}")
        # And it is not what the announced build had there — otherwise the
        # assertion above would pass on a window with no doses in it.
        announced = p[bit_p, feat_idx::N_INPUT_FEATURES]
        print(f"[DUMP]   feat {feat_idx}: announced range "
              f"[{float(announced.min()):+.4f}, {float(announced.max()):+.4f}] "
              f"-> {float(fill[feat_idx]):+.4f}")

    # A seed whose masked span happens to hold no dose at all would satisfy every
    # assertion above without the flag doing anything, so require that something
    # moved.  Per-feat this is not guaranteed — a window may genuinely have no
    # exercise — but across the three it is, at any seed worth testing on.
    dose_cols = [f for feat in MASKABLE_FEATS
                 for f in range(feat, PATCH_DIM, N_INPUT_FEATURES)]
    assert not torch.equal(p[bit_p][:, dose_cols], b[bit_b][:, dose_cols]), (
        "no masked dose cell moved — this seed announces nothing in its masked "
        "span, so the assertions above have no subject; pick another")

    # (b) feat 0 and feat 4 of the masked patches are untouched by the flag.
    for feat_idx in (0, BG_MASKED_FEAT):
        assert torch.equal(p[bit_p, feat_idx::N_INPUT_FEATURES],
                           b[bit_b, feat_idx::N_INPUT_FEATURES]), (
            f"the blind flag moved feat {feat_idx} on a masked patch")

    # (c) every cell of every visible patch is byte-identical.
    assert torch.equal(p[~bit_p], b[~bit_b]), (
        "the blind flag changed a VISIBLE patch — it must only reach the "
        "patches feat 4 announces")

    # (d) nothing else the sample carries moves: the targets, the anchors, the
    # trajectories and the announced-future arrays are all the same window.
    for k in sorted(plain['bg_formula_data']):
        a = np.asarray(plain['bg_formula_data'][k])
        c = np.asarray(blind['bg_formula_data'][k])
        assert np.array_equal(a, c), f"bg_formula_data[{k!r}] moved under blind"
    assert torch.equal(plain['targets'], blind['targets'])


def test_the_dataset_honours_the_flag_end_to_end():
    """``T1DMDataset(blind=True)[i]`` is blind — through ``__getitem__``, not
    through ``_build_sample`` called by hand.

    Every other test here builds its sample by calling the private builder
    directly with ``blind=`` passed explicitly, which tests the builder and says
    nothing about the THREADING: a ``T1DMDataset`` that stored the flag and never
    passed it on, or passed ``blind=False``, would satisfy all of them. That
    threading is the fork's delta 2 — every dataset the blind trainer builds is
    ``blind=True`` — so it needs a subject of its own.

    Both directions are checked on the same index, because "the blind dataset
    produced blind samples" is only evidence if the plain dataset did not.
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
    # The default is still the announced policy on the same index, asserted over
    # the three dose channels TOGETHER rather than one at a time: a sparse channel
    # with no event in this span legitimately carries normalize(0) already — feat
    # 3 does here — so per-feature the two policies coincide by accident.
    dose_cols = [f for feat in MASKABLE_FEATS
                 for f in range(feat, PATCH_DIM, N_INPUT_FEATURES)]
    assert not torch.equal(p[bit][:, dose_cols], b[bit][:, dose_cols]), (
        "the DEFAULT dataset produced the same masked dose cells as the blind one "
        "— either blind=False is not the announced policy, or this index announces "
        "nothing at all and the assertions above have no subject")


def test_next_window_is_blinded_too():
    """The cross-window time probe's second window follows the same policy.

    ``next_window`` carries its own right-edge forecast span, built inside
    ``_build_sample`` from its own masked rows.  Left announced it would hand the
    trunk the true doses of a span the primary window withholds — one model, two
    conventions, in the same optimizer step.
    """
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


def test_the_default_path_is_byte_identical_through_the_flag():
    """``blind=False`` reproduces the pre-flag builder bit for bit.

    A conditioned run must be unaffected by the blind trainer landing, and
    nothing else checks the SAMPLER's output: ``tests/test_bitident.py`` pins the
    forward against a frozen input, so a builder that started writing a different
    dose cell would pass it.

    A mismatch is legitimate only when something deliberately moved the sample —
    the sampler constants, the context window, the channel transforms or the
    normalization pool.  Restamp then, and never to make this test pass.
    """
    stats = _get_stats()
    got = _sample_digest(_build(blind=False, stats=stats))
    print(f"\n[DUMP] default-path digest {got}")
    assert got == DEFAULT_PATH_DIGEST, (
        f"the blind=False sample moved: {got} != {DEFAULT_PATH_DIGEST}. Nothing "
        "in this change may touch it — see the docstring before restamping.")


def test_the_fill_is_the_no_dose_baseline_inference_already_writes():
    """``zero_dose_fill`` == what ``inference`` puts in an un-announced slot.

    This identity is why the blind model needs no change to ``inference.py``: a
    rolling forecast with no ``overrides_fn`` seeds its dose channels from the
    same ``normalize(0)``, so the blind model rolls on the distribution it was
    trained on rather than an out-of-distribution one.

    Measured off ``inference._build_patches_tensor``'s output rather than
    recomputed from the stats — the claim is about the two paths agreeing, and a
    second copy of the formula would agree with itself.
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

    # And it is not z = 0, which is what makes the distinction load-bearing: on a
    # log1p channel z = 0 inverts to expm1(mean), a phantom dose.
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
