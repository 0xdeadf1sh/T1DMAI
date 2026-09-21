"""The training-only bolus-only cap and the paired-arm sample builder (``data.py``).
A tiny ``--events`` cache gives every row its ``tail_dose_`` arrays, which is what the
cap reads the intended boundary bolus from. Nothing here touches the training loop:
the cap is a draw, and the draw is what these pin."""

from __future__ import annotations

import collections

import numpy as np
import pytest
import torch

from config import MAX_CONTEXT_PATCHES, PATCH_SIZE, PREDICTION_PATCHES
from T1DMSIM.cache_simulator import build_cache
from data import (
    N_TAIL_ARMS, TAIL_ARMS, TAIL_ARM_BOLUS, TAIL_ARM_CARBS, TAIL_ARM_NONE,
    UNCAPPED_TAIL_ARMS, T1DMDataset,
)
from normalization import load_normalization_stats

_POOL = 8
_N = 240
_SEED = 4242
# Every drawn bolus is log-uniform 0.5..20 U, so this cap spares a minority and this one none.
_CAP_MID = 2.0
_CAP_ALL = 0.5


@pytest.fixture(scope='module')
def events_cache(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One tiny cache carrying the event channels, and so the boundary doses."""
    out_dir = str(tmp_path_factory.mktemp('bolus_cap_cache') / 'cache')
    build_cache(
        out_dir=out_dir, pool_size=_POOL, n_jobs=1,
        context_steps=MAX_CONTEXT_PATCHES * PATCH_SIZE,
        tail_steps=PREDICTION_PATCHES * PATCH_SIZE,
        dataset_md=out_dir + '.dataset.md', baseline_stats=None,
        force=True, events=True,
    )
    return out_dir


def _dataset(cache: str, cap: float | None = None, partition: str = 'train') -> T1DMDataset:
    return T1DMDataset(
        master_seed=_SEED, total_steps=_N, batch_size=1,
        normalization_stats=load_normalization_stats(),
        cache_path=cache, cache_partition=partition, max_bolus_only_u=cap)


def _arms(ds: T1DMDataset, n: int = _N) -> "list[tuple[int, float]]":
    """``(arm, intended boundary bolus U)`` per sample, in index order."""
    out = []
    for i in range(n):
        bf = ds[i]['bg_formula_data']
        out.append((int(bf['tail_arm']), float(bf['arm_bolus_u'])))
    return out


def test_no_capped_training_sample_keeps_an_oversized_bolus_only_arm(events_cache):
    base = _arms(_dataset(events_cache))
    capped = _arms(_dataset(events_cache, _CAP_MID))
    over = [i for i, (a, u) in enumerate(base)
            if a == TAIL_ARM_BOLUS and u > _CAP_MID]
    assert over, 'no baseline draw exceeds the cap, so this test has no subject'
    for i, (arm, u) in enumerate(capped):
        assert not (arm == TAIL_ARM_BOLUS and u > _CAP_MID), (
            f'sample {i} kept arm bolus at {u:.2f} U above the {_CAP_MID} U cap')
    print(f"[DUMP] bolus cap | {len(over)}/{_N} oversized bolus-only draws redrawn ✓")


def test_the_capped_mass_lands_uniformly_on_the_other_three_arms(events_cache):
    """Only the oversized bolus-only draws move, and they move nowhere else."""
    base = _arms(_dataset(events_cache))
    capped = _arms(_dataset(events_cache, _CAP_ALL))
    moved = [(b[0], c[0]) for b, c in zip(base, capped) if b[0] != c[0]]
    assert moved, 'the cap moved nothing'
    assert all(src == TAIL_ARM_BOLUS for src, _ in moved), (
        'an arm other than bolus-only was redrawn')
    assert all(dst in UNCAPPED_TAIL_ARMS for _, dst in moved), (
        f'a redraw landed outside {UNCAPPED_TAIL_ARMS}')
    assert {dst for _, dst in moved} == set(UNCAPPED_TAIL_ARMS), (
        'the redraw is not spread over all three remaining arms')
    cb = collections.Counter(a for a, _ in base)
    cc = collections.Counter(a for a, _ in capped)
    assert sum(cb.values()) == sum(cc.values()) == _N
    assert cc[TAIL_ARM_BOLUS] < cb[TAIL_ARM_BOLUS]
    for arm in UNCAPPED_TAIL_ARMS:
        assert cc[arm] >= cb[arm], f'{TAIL_ARMS[arm]} lost windows to a bolus-only cap'
    print(f"[DUMP] cap spread | {len(moved)} redraws, "
          f"{ {TAIL_ARMS[a]: cc[a] - cb[a] for a in UNCAPPED_TAIL_ARMS} } ✓")


def test_the_uncapped_draw_is_the_pre_cap_rng_stream(events_cache):
    """``None`` must spend no draw: the arm is still the one substream's first integer."""
    ds = _dataset(events_cache)
    from utils import compute_patient_seed
    for i in range(48):
        seed = compute_patient_seed(_SEED, i // ds.batch_size, i % ds.batch_size)
        want = int(np.random.default_rng(seed ^ 0xA12_0F5E1).integers(N_TAIL_ARMS))
        got = int(ds[i]['bg_formula_data']['tail_arm'])
        assert got == want, f'sample {i} drew arm {got}, the pre-cap stream gives {want}'
    print("[DUMP] default stream | 48 samples draw the pre-cap arm ✓")


def test_a_cap_below_every_dose_still_leaves_the_window_untouched(events_cache):
    """A redrawn arm changes the horizon; a spared one must be byte-for-byte the old sample."""
    base = _dataset(events_cache)
    capped = _dataset(events_cache, _CAP_MID)
    spared = 0
    for i in range(48):
        a, b = base[i], capped[i]
        if a['bg_formula_data']['tail_arm'] != b['bg_formula_data']['tail_arm']:
            continue
        assert torch.equal(a['patches'], b['patches'])
        assert torch.equal(a['targets'], b['targets'])
        spared += 1
    assert spared, 'every sample was redrawn, so nothing pins the spared path'
    print(f"[DUMP] spared samples | {spared}/48 identical under the cap ✓")


def test_the_held_out_slabs_refuse_the_cap(events_cache):
    """Runs with and without the cap must score identical windows, so val/cal never cap."""
    for partition in ('val', 'cal'):
        with pytest.raises(ValueError, match='training-only'):
            _dataset(events_cache, _CAP_MID, partition=partition)
    # An uncapped val dataset is what train.py builds, and it draws its own arms as before.
    val = _dataset(events_cache, None, partition='val')
    assert {int(a) for a, _ in _arms(val, 32)} <= set(range(N_TAIL_ARMS))
    print("[DUMP] held-out slabs | val and cal refuse the cap ✓")


def test_a_cap_needs_the_boundary_dose_the_cache_carries():
    with pytest.raises(ValueError, match='tail_dose_'):
        T1DMDataset(master_seed=_SEED, total_steps=4, batch_size=1,
                    normalization_stats=load_normalization_stats(),
                    cache_path=None, max_bolus_only_u=_CAP_MID)


def test_paired_arms_of_one_row_share_everything_but_the_tail(events_cache):
    """The paired-arm helper the low-rescue reading needs: one row, two horizons."""
    ds = _dataset(events_cache, None, partition='val')
    n_ctx = None
    for i in range(8):
        a = ds.sample_for_arm(i, TAIL_ARM_NONE)
        b = ds.sample_for_arm(i, TAIL_ARM_CARBS)
        assert a['n_context_patches'] == b['n_context_patches']
        n_ctx = int(a['n_context_patches'])
        assert torch.equal(a['patches'][:n_ctx], b['patches'][:n_ctx])
        assert int(a['bg_formula_data']['tail_arm']) == TAIL_ARM_NONE
        assert int(b['bg_formula_data']['tail_arm']) == TAIL_ARM_CARBS
        assert float(b['bg_formula_data']['arm_bolus_u']) == 0.0
        assert not np.array_equal(a['bg_formula_data']['true_bg_trajectory'],
                                  b['bg_formula_data']['true_bg_trajectory'])
    print(f"[DUMP] paired arms | 8 rows share a {n_ctx}-patch context, differ in tail ✓")
