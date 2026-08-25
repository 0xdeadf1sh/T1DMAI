"""The blosc2-backed simulator cache (T1DMSIM/cache_simulator.py + data.py).

The builder lives in the EXTERNAL T1DMSIM repo, reachable as
``T1DMSIM.cache_simulator`` through the ``T1DMAI/T1DMSIM`` symlink. There is no
causal smoothing anywhere, so stats are fit on the RAW transformed channels.

FOUR is the count of NORMALIZED channels; the input stack is five features wide,
the fifth being the ``bg_masked`` bit, which carries no statistics.

``pool_size=4``, ``n_jobs=1`` and the real ``ON_THE_FLY_SIM_HOURS`` /
``SIMULATOR_WARMUP_HOURS``, so ``T1DMDataset`` accepts the cache without
monkey-patching its trajectory-length validation; a full build is ~0.4 s.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil

import numpy as np
import pytest
import torch

import config as _cfg
import data as _data
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
    _BLOSC2_MAX_CHUNK_BYTES,
    _resolve_rows_per_chunk,
    build_cache,
)
from T1DMSIM.simulator import BG_CLAMP_MAX, BG_CLAMP_MIN


# match the on-the-fly window and warmup, which T1DMDataset compares against config
# alongside dt and uniform_prob. One full-length patient is ~90 ms, so pool_size=4
# keeps every build under a second.
_TINY_SIM_HOURS = _data.ON_THE_FLY_SIM_HOURS
_TINY_WARMUP_HOURS = _cfg.SIMULATOR_WARMUP_HOURS
_TINY_POOL = 4
_TINY_UNIFORM_PROB = 0.0

# the eight top-level meta.json keys the T1DMDataset loader hard-checks
_REQUIRED_META_KEYS = (
    'pool_size', 'n_timesteps', 'sim_hours', 'simulator_warmup_hours',
    'patient_uniform_sample_prob', 'dt_minutes', 'channels', 'cache_format',
)
# the {mean, std} contract in the builder-emitted normalization_stats.json: all four
# of ``normalization.CHANNEL_NAMES``, pinned so a builder that starts or stops
# emitting one is visible here rather than in a training run
_BUILDER_NORM_STATS_KEYS = frozenset({
    'bg_absolute', 'carb_intake', 'insulin_combined', 'exercise_equiv',
})


def _build_tiny_cache(out_dir: str, rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK) -> None:
    """``baseline_stats=None`` skips the diff/stats.json dependency, and ``dataset_md``
    next to ``out_dir`` keeps the DATASET.md render out of the read-only T1DMSIM repo."""
    build_cache(
        out_dir=out_dir,
        pool_size=_TINY_POOL,
        sim_hours=_TINY_SIM_HOURS,
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


def _transcode_to_npy_memmap(blosc2_dir: str, npy_dir: str) -> None:
    """Rewrite a built blosc2 cache as an ``npy-memmap-v1`` cache, same data.

    The external builder only emits blosc2, so the reader's npy path is exercised
    through this synthesis.
    """
    import blosc2

    os.makedirs(npy_dir, exist_ok=True)
    for name in CHANNEL_NAMES:
        arr = blosc2.open(os.path.join(blosc2_dir, f'{name}.b2nd'), mode='r')
        assert isinstance(arr, blosc2.NDArray)
        np.save(os.path.join(npy_dir, f'{name}.npy'), np.asarray(arr[:]))
    shutil.copy(
        os.path.join(blosc2_dir, 'icr.npy'), os.path.join(npy_dir, 'icr.npy')
    )
    with open(os.path.join(blosc2_dir, 'meta.json')) as f:
        meta = json.load(f)
    meta['cache_format'] = 'npy-memmap-v1'
    meta.pop('rows_per_chunk', None)
    meta.pop('zstd_clevel', None)
    with open(os.path.join(npy_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)


def test_cache_layout(blosc2_cache: str) -> None:
    out_dir = blosc2_cache

    files = set(os.listdir(out_dir))
    expected = (
        {f'{name}.b2nd' for name in CHANNEL_NAMES}
        | {'icr.npy', 'meta.json', 'normalization_stats.json'}
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
    assert meta['n_timesteps'] > 0
    assert abs(meta['sim_hours'] - _TINY_SIM_HOURS) < 1e-6
    assert abs(meta['simulator_warmup_hours'] - _TINY_WARMUP_HOURS) < 1e-6
    assert meta['patient_uniform_sample_prob'] == _TINY_UNIFORM_PROB
    assert tuple(meta['channels']) == CHANNEL_NAMES

    # the builder's emitted {mean, std} must COVER the model's input stack with
    # nothing left over
    from normalization import CHANNEL_NAMES as NORM_CHANNEL_NAMES
    stats = _cache_stats(out_dir)
    assert set(stats) == _BUILDER_NORM_STATS_KEYS, f"norm stats keys = {set(stats)}"
    for name, mv in stats.items():
        assert set(mv) == {'mean', 'std'}, f"{name} stats = {mv}"
        # std > 0 is the load-time contract: a channel whose window held no event fits
        # std = 0, and normalize divides by std + 1e-8
        assert np.isfinite(mv['mean']) and np.isfinite(mv['std']) and mv['std'] > 0
    # no gap either way: an omitted channel reaches the input gather as a KeyError, an
    # invented one is a fit for something the model never reads
    assert set(stats) == set(NORM_CHANNEL_NAMES), (
        f"builder-emitted stats {sorted(stats)} vs model channels "
        f"{sorted(NORM_CHANNEL_NAMES)}: the builder must fit every channel and "
        f"only those")
    print(f"[DUMP] cache_layout | norm stats = {stats} (builder: 3 keys; model "
          f"needs {len(NORM_CHANNEL_NAMES)}, exercise_equiv hand-merged)")


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
        assert set(sample) >= {'patches', 'targets', 'n_context_patches', 'bg_formula_data'}
        patches = sample['patches']
        targets = sample['targets']
        n_ctx = sample['n_context_patches']

        assert MIN_CONTEXT_PATCHES <= n_ctx <= MAX_CONTEXT_PATCHES
        assert patches.shape == (n_ctx + PREDICTION_PATCHES, PATCH_DIM)
        # one target row per HEAD SLOT, not per horizon patch: padded slots gather
        # patch 0 and ``valid`` discards them
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


def test_stats_missing_a_channel_are_refused_by_the_loader(blosc2_cache: str) -> None:
    """A missing channel must FAIL rather than leave that channel untrained.

    The input gather walks ``CHANNEL_NAMES`` and indexes ``stats[name]``, so a
    ``.get(name, {'mean': 0, 'std': 1})`` fallback anywhere on this path turns a
    missing fit into an untrained channel that trains to completion. The short file is
    CONSTRUCTED here, so a builder that stops emitting one cannot retire this path.
    """
    from data import T1DMDataset

    stats = {k: v for k, v in _cache_stats(blosc2_cache).items()
             if k != 'exercise_equiv'}
    assert 'exercise_equiv' not in stats
    dataset = T1DMDataset(
        master_seed=0,
        total_steps=1,
        batch_size=1,
        normalization_stats=stats,
        patient_uniform_sample_prob=_TINY_UNIFORM_PROB,
        simulator_warmup_hours=_TINY_WARMUP_HOURS,
        cache_path=blosc2_cache,
    )
    with pytest.raises(KeyError, match='exercise_equiv'):
        dataset[0]
    print("\n[DUMP] three_key_stats | builder file raises KeyError('exercise_equiv') "
          "at the input gather — no silent-default fallback ✓")


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
    meta.pop('pool_size', None)
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


def test_npy_memmap_cache_reads_back_identically(
    blosc2_cache: str, tmp_path: pathlib.Path
) -> None:
    """The uncompressed layout is a read convenience for pools shaped elsewhere, so it
    must decode to exactly the same per-channel data as the compressed one."""
    from data import CACHE_CHANNEL_NAMES, CACHE_FORMAT_NPY, T1DMDataset

    blosc2_dir = blosc2_cache
    npy_dir = str(tmp_path / 'cache_npy')
    _transcode_to_npy_memmap(blosc2_dir, npy_dir)

    files = set(os.listdir(npy_dir))
    expected = {f'{name}.npy' for name in CHANNEL_NAMES} | {'icr.npy', 'meta.json'}
    assert files == expected, f"npy cache dir mismatch: {files} != {expected}"
    with open(os.path.join(npy_dir, 'meta.json')) as f:
        assert json.load(f)['cache_format'] == CACHE_FORMAT_NPY

    stats = _model_stats(blosc2_dir)
    kwargs = dict(
        master_seed=0,
        total_steps=2,
        batch_size=4,
        normalization_stats=stats,
        patient_uniform_sample_prob=_TINY_UNIFORM_PROB,
        simulator_warmup_hours=_TINY_WARMUP_HOURS,
    )
    ds_b2 = T1DMDataset(cache_path=blosc2_dir, **kwargs)
    ds_npy = T1DMDataset(cache_path=npy_dir, **kwargs)

    # raw cache rows must be byte-identical across the two on-disk formats
    arrays_b2, icr_b2 = ds_b2._load_cache()
    arrays_npy, icr_npy = ds_npy._load_cache()
    assert np.array_equal(icr_b2, icr_npy), "icr differs between formats"
    for name in CACHE_CHANNEL_NAMES:
        a = np.asarray(arrays_b2[name][:])
        b = np.asarray(arrays_npy[name][:])
        print(f"\n[DUMP] npy_parity | {name:<20} equal={np.array_equal(a, b)}")
        assert np.array_equal(a, b), f"channel {name!r} differs between formats"

    sample = ds_npy[0]
    assert torch.isfinite(sample['patches']).all()
    assert torch.isfinite(sample['targets']).all()


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

    Reads all four signals in full, applies the model fit's own forward transform —
    ``kovatchev_f_np`` on bg, ``log1p(max(x, 0))`` on the sparse three — to the RAW
    post-noise values, then a plain ``np.mean`` / ``np.std(ddof=1)``. ``total_exercise``
    is read at its cached g/step scale: not a glucose, so log1p, never the risk transform.
    """
    import blosc2
    from normalization import CHANNEL_NAMES as NORM_CHANNEL_NAMES
    from utils import kovatchev_f_np

    def _read(name: str) -> np.ndarray:
        arr = blosc2.open(os.path.join(cache_dir, f'{name}.b2nd'), mode='r')
        full = np.asarray(arr[:], dtype=np.float64)
        return full if n_rows is None else full[:n_rows]

    # model channel -> the cached channel it is fit from
    _SOURCE_CHANNEL = {
        'bg_absolute': 'bg_observed',
        'carb_intake': 'total_carb',
        'insulin_combined': 'total_insulin',
        'exercise_equiv': 'total_exercise',
    }
    assert set(_SOURCE_CHANNEL) == set(NORM_CHANNEL_NAMES), (
        f"oracle covers {sorted(_SOURCE_CHANNEL)} but the model fits "
        f"{sorted(NORM_CHANNEL_NAMES)} — an uncovered channel silently drops out")

    out = {}
    for name in NORM_CHANNEL_NAMES:
        raw = _read(_SOURCE_CHANNEL[name])
        if name == 'bg_absolute':
            vals = kovatchev_f_np(raw).ravel()
        else:
            vals = np.log1p(np.maximum(raw, 0.0)).ravel()
        out[name] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1))}
    return out


def test_normalization_stats_from_cache(
    blosc2_cache: str, tmp_path: pathlib.Path
) -> None:
    """Matches the oracle, is format-agnostic, honours ``sample_rows``, and reproduces
    the builder's own emitted ``normalization_stats.json``."""
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

    # the same data as npy-memmap-v1 must fit identical stats
    npy_dir = str(tmp_path / 'cache_np')
    _transcode_to_npy_memmap(blosc2_dir, npy_dir)
    got_npy = compute_normalization_stats_from_cache(npy_dir, batch_rows=3)
    for name in NORM_CHANNEL_NAMES:
        assert got_npy[name]['mean'] == pytest.approx(got[name]['mean'], rel=1e-6, abs=1e-6), name
        assert got_npy[name]['std'] == pytest.approx(got[name]['std'], rel=1e-6, abs=1e-6), name

    # sample_rows reads only the leading rows
    got_2 = compute_normalization_stats_from_cache(blosc2_dir, sample_rows=2, batch_rows=2)
    oracle_2 = _oracle_cache_stats(blosc2_dir, n_rows=2)
    for name in NORM_CHANNEL_NAMES:
        assert got_2[name]['mean'] == pytest.approx(oracle_2[name]['mean'], rel=1e-6, abs=1e-6), name

    # the builder fits during transcode with the same transform and the same Kovatchev
    # constants, so its emitted stats must agree with the recomputed ones
    emitted = _cache_stats(blosc2_dir)
    assert set(emitted) == _BUILDER_NORM_STATS_KEYS
    for name in sorted(_BUILDER_NORM_STATS_KEYS):
        assert emitted[name]['mean'] == pytest.approx(got[name]['mean'], rel=1e-6, abs=1e-6), name
        assert emitted[name]['std'] == pytest.approx(got[name]['std'], rel=1e-6, abs=1e-6), name
    assert 'exercise_equiv' in got and got['exercise_equiv']['std'] > 0, (
        "the cache fitter must produce exercise_equiv even though the builder "
        "does not emit it")

    print(f"\n[DUMP] norm_from_cache | oracle match + npy==blosc2 + sample_rows "
          f"honored across {len(NORM_CHANNEL_NAMES)} channels; "
          f"emitted==recomputed over the builder's {len(_BUILDER_NORM_STATS_KEYS)} "
          f"(exercise_equiv fit but not emitted)")
