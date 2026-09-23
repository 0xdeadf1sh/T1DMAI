"""The blosc2-backed simulator cache (T1DMSIM/cache_simulator.py + data.py). Builder lives in the
EXTERNAL T1DMSIM repo (``T1DMSIM.cache_simulator``), reached via the T1DMAI/T1DMSIM symlink; no
causal smoothing, stats fit on RAW transformed channels over context AND tails. ``pool_size=4``,
``n_jobs=1``, the contract geometry so ``T1DMDataset`` needs no monkey-patch."""

from __future__ import annotations

import json
import os
import pathlib
import shutil

import numpy as np
import pytest
import torch

import config as _cfg
from config import (
    MAX_CONTEXT_PATCHES,
    MAX_MASKED_PATCHES,
    MIN_CONTEXT_PATCHES,
    PATCH_DIM,
    PATCH_SIZE,
    PREDICTION_PATCHES,
)
from T1DMSIM.cache_simulator import (
    CACHE_FORMAT_VERSION,
    CHANNEL_NAMES,
    DEFAULT_ROWS_PER_CHUNK,
    N_TAIL_ARMS,
    SKILLS_FILE,
    TAIL_ARMS,
    _BLOSC2_MAX_CHUNK_BYTES,
    _resolve_rows_per_chunk,
    build_cache,
    tail_channel_names,
)
from T1DMSIM.simulator import BG_CLAMP_MAX, BG_CLAMP_MIN

_TINY_WARMUP_HOURS = _cfg.SIMULATOR_WARMUP_HOURS
_TINY_POOL = 4
_TINY_UNIFORM_PROB = 0.0
_CONTEXT_STEPS = MAX_CONTEXT_PATCHES * PATCH_SIZE
_TAIL_STEPS = PREDICTION_PATCHES * PATCH_SIZE

# the top-level meta.json keys the T1DMDataset loader hard-checks
_REQUIRED_META_KEYS = (
    'pool_size', 'n_timesteps', 'sim_hours', 'simulator_warmup_hours',
    'patient_uniform_sample_prob', 'dt_minutes', 'channels', 'cache_format',
    'context_steps', 'tail_steps', 'tail_arms', 'tail_channels',
)
# {mean, std} keys the builder emits for the curves layout, pinned here.
_BUILDER_NORM_STATS_KEYS = frozenset({
    'bg_absolute', 'carb_intake', 'insulin_combined',
})


def _build_tiny_cache(out_dir: str, rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK) -> None:
    """``baseline_stats=None`` skips the diff/stats.json dependency, and ``dataset_md``
    next to ``out_dir`` keeps the DATASET.md render out of the read-only T1DMSIM repo."""
    build_cache(
        out_dir=out_dir,
        pool_size=_TINY_POOL,
        warmup_hours=_TINY_WARMUP_HOURS,
        n_jobs=1,
        rows_per_chunk=rows_per_chunk,
        dataset_md=out_dir + '.dataset.md',
        baseline_stats=None,
        force=True,
    )


@pytest.fixture(scope='module')
def blosc2_cache(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A single tiny blosc2 cache, built once and shared by the read-only tests."""
    out_dir = str(tmp_path_factory.mktemp('blosc2_cache') / 'cache')
    _build_tiny_cache(out_dir)
    return out_dir


def _cache_stats(cache_dir: str) -> dict:
    """The builder's emitted normalization_stats.json, verbatim."""
    with open(os.path.join(cache_dir, 'normalization_stats.json')) as f:
        return json.load(f)


def _model_stats(cache_dir: str) -> dict:
    """Fit all N_CHANNELS off the cache itself: milliseconds on a 4-patient pool, and
    it keeps the fixture honest about which channels the input pipeline requires."""
    from normalization import compute_normalization_stats_from_cache
    return compute_normalization_stats_from_cache(cache_dir)


def test_cache_layout(blosc2_cache: str) -> None:
    out_dir = blosc2_cache

    files = set(os.listdir(out_dir))
    expected = (
        {f'{name}.b2nd' for name in CHANNEL_NAMES}
        | {f'tail_{name}.b2nd' for name in tail_channel_names(False)}
        | {'icr.npy', SKILLS_FILE, 'meta.json', 'normalization_stats.json'}
    )
    assert files == expected, f"Cache directory mismatch: {files} != {expected}"

    with open(os.path.join(out_dir, 'meta.json')) as f:
        meta = json.load(f)
    print(f"\n[DUMP] cache_layout | meta subset = "
          f"{ {k: meta.get(k) for k in _REQUIRED_META_KEYS} }")

    missing = [k for k in _REQUIRED_META_KEYS if k not in meta]
    assert not missing, f"meta.json missing required keys: {missing}"
    assert meta['cache_format'] == CACHE_FORMAT_VERSION
    assert meta['pool_size'] == _TINY_POOL
    assert meta['n_timesteps'] == _CONTEXT_STEPS == meta['context_steps']
    assert meta['tail_steps'] == _TAIL_STEPS
    assert tuple(meta['tail_arms']) == TAIL_ARMS
    assert abs(meta['simulator_warmup_hours'] - _TINY_WARMUP_HOURS) < 1e-6
    assert meta['patient_uniform_sample_prob'] == _TINY_UNIFORM_PROB
    assert tuple(meta['channels']) == CHANNEL_NAMES

    # the builder's emitted {mean, std} must COVER the model's input stack, nothing left over
    from normalization import CHANNEL_NAMES as NORM_CHANNEL_NAMES
    stats = _cache_stats(out_dir)
    assert set(stats) == _BUILDER_NORM_STATS_KEYS, f"norm stats keys = {set(stats)}"
    for name, mv in stats.items():
        assert set(mv) == {'mean', 'std'}, f"{name} stats = {mv}"
        # std > 0 is load-time contract: no-event channel fits std=0, normalize divides by std+1e-8
        assert np.isfinite(mv['mean']) and np.isfinite(mv['std']) and mv['std'] > 0
    # no gap either way: an omitted channel is a KeyError, an invented one fits something unread
    assert set(stats) == set(NORM_CHANNEL_NAMES), (
        f"builder-emitted stats {sorted(stats)} vs model channels "
        f"{sorted(NORM_CHANNEL_NAMES)}: the builder must fit every channel and "
        f"only those")
    print(f"[DUMP] cache_layout | norm stats = {stats}")


def test_tail_arrays_have_the_contract_shape(blosc2_cache: str) -> None:
    """One (pool, 4, tail_steps) array per tail channel; arm 0 is the no-dose continuation."""
    import blosc2

    for name in tail_channel_names(False):
        arr = blosc2.open(os.path.join(blosc2_cache, f'tail_{name}.b2nd'), mode='r')
        assert isinstance(arr, blosc2.NDArray)
        assert tuple(arr.shape) == (_TINY_POOL, N_TAIL_ARMS, _TAIL_STEPS), name

    carb = np.asarray(blosc2.open(
        os.path.join(blosc2_cache, 'tail_total_carb.b2nd'), mode='r')[:])
    bolus = np.asarray(blosc2.open(
        os.path.join(blosc2_cache, 'tail_bolus_insulin.b2nd'), mode='r')[:])
    # arms 2/3 carry the boundary carbohydrate, arms 1/3 the boundary bolus
    assert (carb[:, 2, 0] > carb[:, 0, 0]).all()
    assert (bolus[:, 1, 0] > bolus[:, 0, 0]).all()
    print(f"\n[DUMP] tails | arms={TAIL_ARMS} steps={_TAIL_STEPS}")


def test_cache_compression_shrinks_disk(blosc2_cache: str) -> None:
    """Byte-shuffle plus zstd over smooth biological signals gives >=1.5x even on a
    4-patient pool; the floor is loose on purpose, to survive codec tweaks."""
    out_dir = blosc2_cache
    with open(os.path.join(out_dir, 'meta.json')) as f:
        meta = json.load(f)
    pool_size = meta['pool_size']
    n_timesteps = meta['n_timesteps']

    raw_total = 0
    compressed_total = 0
    for name in CHANNEL_NAMES:
        itemsize = 4  # every channel is 4-byte, float32 or int32
        raw_total += pool_size * n_timesteps * itemsize
        compressed_total += os.path.getsize(os.path.join(out_dir, f'{name}.b2nd'))

    ratio = raw_total / max(compressed_total, 1)
    print(f"\n[DUMP] cache_compression | raw={raw_total} "
          f"compressed={compressed_total} ratio={ratio:.2f}x")
    assert ratio >= 1.5, (
        f"Compression ratio {ratio:.2f}x is below the floor — codec or "
        "filter regression?"
    )


def test_cache_reads_back_via_dataset(blosc2_cache: str) -> None:
    from data import T1DMDataset

    out_dir = blosc2_cache
    stats = _model_stats(out_dir)
    dataset = T1DMDataset(
        master_seed=0,
        total_steps=2,
        batch_size=4,
        normalization_stats=stats,
        patient_uniform_sample_prob=_TINY_UNIFORM_PROB,
        simulator_warmup_hours=_TINY_WARMUP_HOURS,
        cache_path=out_dir,
    )
    assert len(dataset) == 8

    for i in range(len(dataset)):
        sample = dataset[i]
        assert set(sample) >= {'patches', 'targets', 'n_context_patches',
                               'bg_formula_data'}
        patches = sample['patches']
        targets = sample['targets']
        n_ctx = sample['n_context_patches']

        assert MIN_CONTEXT_PATCHES <= n_ctx <= MAX_CONTEXT_PATCHES
        assert patches.shape == (n_ctx + PREDICTION_PATCHES, PATCH_DIM)
        # one target row per HEAD SLOT: padded slots gather patch 0, ``valid`` discards them
        assert targets.shape == (MAX_MASKED_PATCHES, PATCH_SIZE)
        assert patches.dtype == torch.float32 and targets.dtype == torch.float32
        assert torch.isfinite(patches).all(), f"non-finite patches at {i}"
        assert torch.isfinite(targets).all(), f"non-finite targets at {i}"

        # the target is the true BG label, in physical mg/dL
        tnp = targets.numpy()
        assert (tnp >= BG_CLAMP_MIN - 1e-3).all() and (tnp <= BG_CLAMP_MAX + 1e-3).all(), (
            f"targets out of physical range at {i}: "
            f"[{tnp.min():.1f}, {tnp.max():.1f}]"
        )
        last_bg = sample['bg_formula_data']['last_bg']
        assert BG_CLAMP_MIN - 1e-3 <= last_bg <= BG_CLAMP_MAX + 1e-3

    s0 = dataset[0]
    print(f"\n[DUMP] cache_read | sample0 keys={sorted(s0.keys())} "
          f"patches={tuple(s0['patches'].shape)} targets={tuple(s0['targets'].shape)} "
          f"n_ctx={s0['n_context_patches']} last_bg={s0['bg_formula_data']['last_bg']:.1f}")


def test_horizon_bg_is_the_sampled_arms_tail(blosc2_cache: str) -> None:
    """The forecast target is the chosen arm's tail BG, read straight off the cache."""
    import blosc2
    from data import T1DMDataset

    stats = _model_stats(blosc2_cache)
    ds = T1DMDataset(master_seed=0, total_steps=4, batch_size=1,
                     normalization_stats=stats, cache_path=blosc2_cache)
    tail_bg = np.asarray(blosc2.open(
        os.path.join(blosc2_cache, 'tail_bg_observed.b2nd'), mode='r')[:])
    for i in range(4):
        s = ds[i]
        true_bg = np.asarray(s['bg_formula_data']['true_bg_trajectory'])
        # one of the four arms, clamped to the physical range by _build_sample
        arms = np.clip(tail_bg[:, :, :], BG_CLAMP_MIN, BG_CLAMP_MAX)
        assert any(np.allclose(true_bg, arms[r, a], atol=1e-4)
                   for r in range(_TINY_POOL) for a in range(N_TAIL_ARMS)), i


def test_tail_is_never_visible_model_input(blosc2_cache: str) -> None:
    """Every cached sample masks all PREDICTION_PATCHES tail patches and zeroes their bg.

    The tail is behaviour-off: visible, the model reads a counterfactual as observed
    history and gets the forecast handed to it.
    """
    from config import N_INPUT_FEATURES, PREDICTION_PATCHES
    from data import T1DMDataset, BG_MASKED_FEAT

    stats = _model_stats(blosc2_cache)
    ds = T1DMDataset(master_seed=3, total_steps=12, batch_size=2,
                     normalization_stats=stats, cache_path=blosc2_cache)
    for i in range(24):
        s = ds[i]
        n_ctx = int(s['n_context_patches'])
        tail = slice(n_ctx, n_ctx + PREDICTION_PATCHES)
        bit = s['patches'][tail, BG_MASKED_FEAT::N_INPUT_FEATURES]
        bg = s['patches'][tail, 0::N_INPUT_FEATURES]
        assert bool((bit == 1.0).all()), f"sample {i} leaves a tail patch announced-visible"
        assert bool((bg == 0.0).all()), f"sample {i} feeds tail bg to the model"
        fd = s['bg_formula_data']
        scored = set(np.asarray(fd['mask_idx'])[np.asarray(fd['valid'])].tolist())
        assert set(range(n_ctx, n_ctx + PREDICTION_PATCHES)) <= scored


def test_stats_missing_a_channel_are_refused_by_the_loader(blosc2_cache: str) -> None:
    """A missing channel must FAIL rather than leave that channel untrained.
    Input gather walks ``CHANNEL_NAMES`` and indexes ``stats[name]``, so a
    ``.get(name, {'mean': 0, 'std': 1})`` fallback anywhere turns a missing fit into an
    untrained channel that trains to completion."""
    from data import T1DMDataset

    stats = {k: v for k, v in _cache_stats(blosc2_cache).items()
             if k != 'insulin_combined'}
    assert 'insulin_combined' not in stats
    dataset = T1DMDataset(
        master_seed=0,
        total_steps=1,
        batch_size=1,
        normalization_stats=stats,
        patient_uniform_sample_prob=_TINY_UNIFORM_PROB,
        simulator_warmup_hours=_TINY_WARMUP_HOURS,
        cache_path=blosc2_cache,
    )
    with pytest.raises(KeyError, match='insulin_combined'):
        dataset[0]
    print("\n[DUMP] short_stats | raises KeyError at the input gather ✓")


def test_rows_per_chunk_does_not_perturb_values(tmp_path: pathlib.Path) -> None:
    """Chunking is a pure storage knob: leaking into the decoded values would train a
    different distribution silently."""
    import blosc2

    out_a = str(tmp_path / 'cache_chunk_a')
    out_b = str(tmp_path / 'cache_chunk_b')
    _build_tiny_cache(out_a, rows_per_chunk=1)
    _build_tiny_cache(out_b, rows_per_chunk=4)

    for name in CHANNEL_NAMES:
        a = blosc2.open(os.path.join(out_a, f'{name}.b2nd'), mode='r')
        b = blosc2.open(os.path.join(out_b, f'{name}.b2nd'), mode='r')
        assert isinstance(a, blosc2.NDArray)
        assert isinstance(b, blosc2.NDArray)
        a_full = np.asarray(a[:])
        b_full = np.asarray(b[:])
        print(
            f"[DUMP] chunking_invariance | {name:<20} "
            f"chunks_a={a.chunks} chunks_b={b.chunks} "
            f"equal={np.array_equal(a_full, b_full)}"
        )
        assert np.array_equal(a_full, b_full), (
            f"Channel {name!r} differs between rows_per_chunk=1 and =4 — "
            "chunking is leaking into the decoded values."
        )


def test_missing_required_meta_key_is_rejected(
    blosc2_cache: str, tmp_path: pathlib.Path
) -> None:
    """A meta.json missing a required key must be rejected with a clear error."""
    from data import T1DMDataset

    bad_dir = str(tmp_path / 'cache_missing_key')
    shutil.copytree(blosc2_cache, bad_dir)
    with open(os.path.join(bad_dir, 'meta.json')) as f:
        meta = json.load(f)
    meta.pop('tail_arms', None)
    with open(os.path.join(bad_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    stats = _model_stats(blosc2_cache)
    with pytest.raises(ValueError, match='missing keys'):
        T1DMDataset(
            master_seed=0, total_steps=1, batch_size=1,
            normalization_stats=stats,
            patient_uniform_sample_prob=_TINY_UNIFORM_PROB,
            simulator_warmup_hours=_TINY_WARMUP_HOURS,
            cache_path=bad_dir,
        )


def test_unsupported_cache_format_is_rejected(
    blosc2_cache: str, tmp_path: pathlib.Path
) -> None:
    """A meta.json with an unknown cache_format must be rejected with a clear error."""
    from data import T1DMDataset

    bad_dir = str(tmp_path / 'cache_bad_format')
    shutil.copytree(blosc2_cache, bad_dir)
    with open(os.path.join(bad_dir, 'meta.json')) as f:
        meta = json.load(f)
    meta['cache_format'] = 'blosc2-ndarray-v999'  # present but unsupported
    with open(os.path.join(bad_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    stats = _model_stats(blosc2_cache)
    with pytest.raises(ValueError, match='not supported'):
        T1DMDataset(
            master_seed=0, total_steps=1, batch_size=1,
            normalization_stats=stats,
            patient_uniform_sample_prob=_TINY_UNIFORM_PROB,
            simulator_warmup_hours=_TINY_WARMUP_HOURS,
            cache_path=bad_dir,
        )


def test_wrong_tail_geometry_is_rejected(
    blosc2_cache: str, tmp_path: pathlib.Path
) -> None:
    """The horizon IS the tail: a tail of another length cannot be consumed."""
    from data import T1DMDataset

    bad_dir = str(tmp_path / 'cache_bad_tail')
    shutil.copytree(blosc2_cache, bad_dir)
    with open(os.path.join(bad_dir, 'meta.json')) as f:
        meta = json.load(f)
    meta['tail_steps'] = _TAIL_STEPS + 6
    with open(os.path.join(bad_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    stats = _model_stats(blosc2_cache)
    with pytest.raises(ValueError, match='tail_steps'):
        T1DMDataset(
            master_seed=0, total_steps=1, batch_size=1,
            normalization_stats=stats,
            cache_path=bad_dir,
        )


def test_resolve_rows_per_chunk_clamps() -> None:
    """Clamped to min(user_input, pool_size, byte_cap)."""
    # within all caps: unchanged
    assert _resolve_rows_per_chunk(32, 50_000, 666) == 32
    # more rows than the pool: clamp to pool_size
    assert _resolve_rows_per_chunk(10_000, 4, 666) == 4
    # enough rows to pass blosc2's 2 GiB limit, ~800k at 666 x 4 B: clamp to the cap
    huge = 10_000_000
    out = _resolve_rows_per_chunk(huge, huge, 666)
    chunk_bytes = out * 666 * 4
    assert chunk_bytes <= _BLOSC2_MAX_CHUNK_BYTES
    # and one more row would exceed it
    assert (out + 1) * 666 * 4 > _BLOSC2_MAX_CHUNK_BYTES

    with pytest.raises(ValueError, match='rows_per_chunk'):
        _resolve_rows_per_chunk(0, 100, 666)
    with pytest.raises(ValueError, match='chunk limit'):
        # n_timesteps so large that even a single-row chunk overflows
        _resolve_rows_per_chunk(1, 100, _BLOSC2_MAX_CHUNK_BYTES)


def test_no_partial_dir_left_after_success(blosc2_cache: str) -> None:
    """The .partial staging dir must not survive a successful build."""
    assert os.path.isdir(blosc2_cache)
    assert not os.path.exists(blosc2_cache + '.partial'), (
        "Staging directory leaked into the final tree — atomic rename "
        "logic regressed."
    )


def _oracle_cache_stats(cache_dir: str, n_rows: int | None = None) -> dict:
    """A naive oracle for the streaming Welford in ``compute_normalization_stats_from_cache``.
    Reads each signal's context AND tails in full, applies the fit's own forward transform —
    ``kovatchev_f_np`` on bg, ``log1p(max(x, 0))`` on the sparse pair — to RAW post-noise
    values, then plain ``np.mean``/``np.std(ddof=1)``."""
    import blosc2
    from data import SIM_CHANNEL
    from normalization import CHANNEL_NAMES as NORM_CHANNEL_NAMES
    from utils import kovatchev_f_np

    def _read(name: str) -> np.ndarray:
        arr = blosc2.open(os.path.join(cache_dir, f'{name}.b2nd'), mode='r')
        full = np.asarray(arr[:], dtype=np.float64)
        return full if n_rows is None else full[:n_rows]

    out = {}
    for name in NORM_CHANNEL_NAMES:
        src = SIM_CHANNEL[name]
        raw = np.concatenate([_read(src).ravel(), _read(f'tail_{src}').ravel()])
        if name == 'bg_absolute':
            vals = kovatchev_f_np(raw).ravel()
        else:
            vals = np.log1p(np.maximum(raw, 0.0)).ravel()
        out[name] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1))}
    return out


def test_normalization_stats_from_cache(blosc2_cache: str) -> None:
    """Matches the oracle, honours ``sample_rows``, and reproduces the builder's own
    emitted ``normalization_stats.json``."""
    from normalization import (
        CHANNEL_NAMES as NORM_CHANNEL_NAMES,
        compute_normalization_stats_from_cache,
    )

    blosc2_dir = blosc2_cache

    # batched Welford over the full pool == the naive direct computation
    got = compute_normalization_stats_from_cache(blosc2_dir, batch_rows=2)
    oracle = _oracle_cache_stats(blosc2_dir)
    for name in NORM_CHANNEL_NAMES:
        assert got[name]['mean'] == pytest.approx(oracle[name]['mean'], rel=1e-6, abs=1e-6), name
        assert got[name]['std'] == pytest.approx(oracle[name]['std'], rel=1e-6, abs=1e-6), name

    # sample_rows reads only the leading rows
    got_2 = compute_normalization_stats_from_cache(blosc2_dir, sample_rows=2, batch_rows=2)
    oracle_2 = _oracle_cache_stats(blosc2_dir, n_rows=2)
    for name in NORM_CHANNEL_NAMES:
        assert got_2[name]['mean'] == pytest.approx(oracle_2[name]['mean'], rel=1e-6, abs=1e-6), name

    # builder fits during transcode with same transform/Kovatchev constants, must agree here.
    emitted = _cache_stats(blosc2_dir)
    assert set(emitted) == _BUILDER_NORM_STATS_KEYS
    for name in sorted(_BUILDER_NORM_STATS_KEYS):
        assert emitted[name]['mean'] == pytest.approx(got[name]['mean'], rel=1e-6, abs=1e-6), name
        assert emitted[name]['std'] == pytest.approx(got[name]['std'], rel=1e-6, abs=1e-6), name

    print(f"\n[DUMP] norm_from_cache | oracle match + sample_rows honored across "
          f"{len(NORM_CHANNEL_NAMES)} channels; emitted==recomputed")
