"""Tests for the blosc2-backed simulator cache (T1DMSIM/cache_simulator.py + data.py).

The pool builder now lives in the EXTERNAL T1DMSIM repo (reachable as
``T1DMSIM.cache_simulator`` through the ``T1DMAI/T1DMSIM`` symlink); the old
in-repo ``cache_simulator.py`` is gone. Causal smoothing was likewise removed
project-wide, so normalization stats are fit on the RAW (transformed) channels
with NO smoothing.

Builds tiny caches end-to-end and asserts:

* The on-disk layout matches the documented format — one ``.b2nd`` per channel,
  ``icr.npy``, ``meta.json`` with the eight loader-required fields, and the
  builder-emitted ``normalization_stats.json`` (the 3-channel {mean, std}
  contract the model consumes).
* ``T1DMDataset(cache_path=...)`` opens the cache and returns samples whose
  shapes / dtypes / value ranges match the spec.
* Re-building the **same** pool with a different ``rows_per_chunk`` value
  produces byte-identical per-row reads — chunking is a storage-only knob
  and must not perturb the cached signal.
* A ``meta.json`` missing a required key, or carrying an unsupported
  ``cache_format``, is rejected by ``T1DMDataset`` with a clear error.
* A synthesized ``npy-memmap-v1`` cache (transcoded from the built ``.b2nd``
  channels) loads and decodes byte-identically to the compressed layout,
  keeping data.py's uncompressed read path covered.
* ``compute_normalization_stats_from_cache`` matches a direct no-smoothing
  oracle (Kovatchev-``f`` on bg, ``log1p`` on the sparse channels), agrees
  across on-disk formats, honors ``sample_rows``, and reproduces the builder's
  own emitted ``normalization_stats.json``.

These tests use ``pool_size=4`` with ``n_jobs=1`` and the real
``ON_THE_FLY_SIM_HOURS`` / ``SIMULATOR_WARMUP_HOURS`` (so ``T1DMDataset``
accepts the cache without monkey-patching its trajectory-length validation);
a full build is ~0.4 s, so the whole suite stays well under a few seconds.
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


# Match the on-the-fly window / warmup so the cache validates cleanly inside
# T1DMDataset (it compares sim_hours / warmup / dt / uniform_prob against config;
# uniform_prob is 0.0 on both sides). One full-length patient is ~90 ms, so
# pool_size=4 keeps every build well under a second.
_TINY_SIM_HOURS = _data.ON_THE_FLY_SIM_HOURS
_TINY_WARMUP_HOURS = _cfg.SIMULATOR_WARMUP_HOURS
_TINY_POOL = 4
_TINY_UNIFORM_PROB = 0.0

# The eight required top-level meta.json keys the T1DMDataset loader hard-checks.
_REQUIRED_META_KEYS = (
    'pool_size', 'n_timesteps', 'sim_hours', 'simulator_warmup_hours',
    'patient_uniform_sample_prob', 'dt_minutes', 'channels', 'cache_format',
)
# The 3-channel {mean, std} contract in the builder-emitted normalization_stats.json.
_NORM_STATS_KEYS = frozenset({'bg_absolute', 'carb_intake', 'insulin_combined'})


def _build_tiny_cache(out_dir: str, rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK) -> None:
    """Build a tiny blosc2 cache via the EXTERNAL builder, with no side effects.

    ``baseline_stats=None`` skips the diff/stats.json dependency, and pointing
    ``dataset_md`` next to ``out_dir`` (inside the test tmp tree) keeps the
    DATASET.md render out of the read-only T1DMSIM repo.
    """
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
    """Load the cache's own emitted normalization_stats.json (3-channel contract)."""
    with open(os.path.join(cache_dir, 'normalization_stats.json')) as f:
        return json.load(f)


def _transcode_to_npy_memmap(blosc2_dir: str, npy_dir: str) -> None:
    """Rewrite a built blosc2 cache as an ``npy-memmap-v1`` cache (same data).

    Decompresses each ``.b2nd`` channel in full and writes it as a raw
    ``<channel>.npy``, copies ``icr.npy`` verbatim, and flips
    ``meta['cache_format']`` to ``'npy-memmap-v1'`` (dropping the blosc2-only
    ``rows_per_chunk`` / ``zstd_clevel`` keys). This is the on-disk shape an
    externally-generated uncompressed pool has — the external builder itself
    only emits blosc2, so the reader's npy path is exercised via this synthesis.
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
    """Built cache has the expected files, the 8 required meta fields, and stats."""
    out_dir = blosc2_cache

    # .b2nd per channel + icr.npy + meta.json + normalization_stats.json.
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

    # normalization_stats.json: exactly the 3-channel {mean, std} contract.
    stats = _cache_stats(out_dir)
    assert set(stats) == _NORM_STATS_KEYS, f"norm stats keys = {set(stats)}"
    for name, mv in stats.items():
        assert set(mv) == {'mean', 'std'}, f"{name} stats = {mv}"
        assert np.isfinite(mv['mean']) and np.isfinite(mv['std']) and mv['std'] > 0
    print(f"[DUMP] cache_layout | norm stats = {stats}")


def test_cache_compression_shrinks_disk(blosc2_cache: str) -> None:
    """Compressed .b2nd files are strictly smaller than the raw float32 size.

    Smooth biological signals + byte-shuffle + zstd should give >=1.5x even on
    a tiny 4-patient pool — the floor is intentionally loose so this stays
    robust to future codec / chunking tweaks.
    """
    out_dir = blosc2_cache
    with open(os.path.join(out_dir, 'meta.json')) as f:
        meta = json.load(f)
    pool_size = meta['pool_size']
    n_timesteps = meta['n_timesteps']

    raw_total = 0
    compressed_total = 0
    for name in CHANNEL_NAMES:
        itemsize = 4  # all current channels are 4-byte (float32 or int32).
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
    """T1DMDataset(cache_path=...) opens the cache and returns sane samples."""
    from data import T1DMDataset

    out_dir = blosc2_cache
    stats = _cache_stats(out_dir)
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
        assert targets.shape == (PREDICTION_PATCHES, PATCH_SIZE)
        assert patches.dtype == torch.float32 and targets.dtype == torch.float32
        assert torch.isfinite(patches).all(), f"non-finite patches at {i}"
        assert torch.isfinite(targets).all(), f"non-finite targets at {i}"

        # Targets are the true BG forecast label in physical mg/dL.
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


def test_rows_per_chunk_does_not_perturb_values(tmp_path: pathlib.Path) -> None:
    """Same simulator inputs + different chunking -> byte-identical per-row reads.

    Chunking is a pure storage knob; if it ever started affecting the decoded
    data we'd silently train on a different distribution. Pin it.
    """
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
    meta.pop('pool_size', None)  # drop a required key
    with open(os.path.join(bad_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    stats = _cache_stats(blosc2_cache)
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

    stats = _cache_stats(blosc2_cache)
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
    """An ``npy-memmap-v1`` cache loads and yields byte-identical rows vs blosc2.

    The uncompressed layout is a read convenience for pools shaped elsewhere; it
    must decode to exactly the same per-channel data as the compressed layout the
    reader's other path uses.
    """
    from data import CACHE_CHANNEL_NAMES, CACHE_FORMAT_NPY, T1DMDataset

    blosc2_dir = blosc2_cache
    npy_dir = str(tmp_path / 'cache_npy')
    _transcode_to_npy_memmap(blosc2_dir, npy_dir)

    # Layout: one raw .npy per channel + icr.npy + meta.json.
    files = set(os.listdir(npy_dir))
    expected = {f'{name}.npy' for name in CHANNEL_NAMES} | {'icr.npy', 'meta.json'}
    assert files == expected, f"npy cache dir mismatch: {files} != {expected}"
    with open(os.path.join(npy_dir, 'meta.json')) as f:
        assert json.load(f)['cache_format'] == CACHE_FORMAT_NPY

    stats = _cache_stats(blosc2_dir)
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

    # Raw cache rows must be byte-identical across the two on-disk formats.
    arrays_b2, icr_b2 = ds_b2._load_cache()
    arrays_npy, icr_npy = ds_npy._load_cache()
    assert np.array_equal(icr_b2, icr_npy), "icr differs between formats"
    for name in CACHE_CHANNEL_NAMES:
        a = np.asarray(arrays_b2[name][:])
        b = np.asarray(arrays_npy[name][:])
        print(f"\n[DUMP] npy_parity | {name:<20} equal={np.array_equal(a, b)}")
        assert np.array_equal(a, b), f"channel {name!r} differs between formats"

    # The assembled sample from the npy cache is finite.
    sample = ds_npy[0]
    assert torch.isfinite(sample['patches']).all()
    assert torch.isfinite(sample['targets']).all()


def test_resolve_rows_per_chunk_clamps() -> None:
    """rows_per_chunk is clamped to min(user_input, pool_size, byte_cap)."""
    # Within all caps: returned unchanged.
    assert _resolve_rows_per_chunk(32, 50_000, 666) == 32
    # User asks for more rows than the pool — clamp to pool_size.
    assert _resolve_rows_per_chunk(10_000, 4, 666) == 4
    # User asks for so many rows the chunk would exceed blosc2's 2 GiB limit
    # (~800k rows at 666 x 4 B); clamp to the byte cap.
    huge = 10_000_000
    out = _resolve_rows_per_chunk(huge, huge, 666)
    chunk_bytes = out * 666 * 4
    assert chunk_bytes <= _BLOSC2_MAX_CHUNK_BYTES
    # …and the next-larger row count would exceed it.
    assert (out + 1) * 666 * 4 > _BLOSC2_MAX_CHUNK_BYTES

    # Out-of-band inputs are rejected loudly.
    with pytest.raises(ValueError, match='rows_per_chunk'):
        _resolve_rows_per_chunk(0, 100, 666)
    with pytest.raises(ValueError, match='chunk limit'):
        # n_timesteps so large that even a single-row chunk overflows.
        _resolve_rows_per_chunk(1, 100, _BLOSC2_MAX_CHUNK_BYTES)


def test_no_partial_dir_left_after_success(blosc2_cache: str) -> None:
    """The .partial staging dir must not survive a successful build."""
    assert os.path.isdir(blosc2_cache)
    assert not os.path.exists(blosc2_cache + '.partial'), (
        "Staging directory leaked into the final tree — atomic rename "
        "logic regressed."
    )


def _oracle_cache_stats(cache_dir: str, n_rows: int | None = None) -> dict:
    """Independent per-channel stats over a (tiny) blosc2 cache — NO smoothing.

    A deliberately naive oracle for cross-checking the streaming Welford
    accumulation in ``compute_normalization_stats_from_cache``: read the three
    input signals in full and apply the SAME forward transform the model fit
    uses — Kovatchev ``f`` on bg (``utils.kovatchev_f_np``, which clamps to the
    physical range), ``log1p(max(x, 0))`` on the sparse carb/insulin channels —
    directly on the RAW post-noise values (causal smoothing was removed
    project-wide), then take a plain ``np.mean`` / sample ``np.std(ddof=1)`` over
    the flattened values.
    """
    import blosc2
    from normalization import CHANNEL_NAMES as NORM_CHANNEL_NAMES
    from utils import kovatchev_f_np

    def _read(name: str) -> np.ndarray:
        arr = blosc2.open(os.path.join(cache_dir, f'{name}.b2nd'), mode='r')
        full = np.asarray(arr[:], dtype=np.float64)
        return full if n_rows is None else full[:n_rows]

    src = {
        'bg_absolute': _read('bg_observed'),
        'carb_intake': _read('total_carb'),
        'insulin_combined': _read('total_insulin'),
    }
    out = {}
    for name in NORM_CHANNEL_NAMES:
        raw = src[name]
        if name == 'bg_absolute':
            vals = kovatchev_f_np(raw).ravel()
        else:
            vals = np.log1p(np.maximum(raw, 0.0)).ravel()
        out[name] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1))}
    return out


def test_normalization_stats_from_cache(
    blosc2_cache: str, tmp_path: pathlib.Path
) -> None:
    """``compute_normalization_stats_from_cache`` matches a no-smoothing oracle,
    is format-agnostic (blosc2 == npy-memmap), honors ``sample_rows``, and
    reproduces the builder's own emitted ``normalization_stats.json``."""
    from normalization import (
        CHANNEL_NAMES as NORM_CHANNEL_NAMES,
        compute_normalization_stats_from_cache,
    )

    blosc2_dir = blosc2_cache

    # (1) Batched Welford over the full pool == naive direct computation.
    got = compute_normalization_stats_from_cache(blosc2_dir, batch_rows=2)
    oracle = _oracle_cache_stats(blosc2_dir)
    for name in NORM_CHANNEL_NAMES:
        assert got[name]['mean'] == pytest.approx(oracle[name]['mean'], rel=1e-6, abs=1e-6), name
        assert got[name]['std'] == pytest.approx(oracle[name]['std'], rel=1e-6, abs=1e-6), name

    # (2) Same data as an npy-memmap-v1 cache → identical stats (format-agnostic).
    npy_dir = str(tmp_path / 'cache_np')
    _transcode_to_npy_memmap(blosc2_dir, npy_dir)
    got_npy = compute_normalization_stats_from_cache(npy_dir, batch_rows=3)
    for name in NORM_CHANNEL_NAMES:
        assert got_npy[name]['mean'] == pytest.approx(got[name]['mean'], rel=1e-6, abs=1e-6), name
        assert got_npy[name]['std'] == pytest.approx(got[name]['std'], rel=1e-6, abs=1e-6), name

    # (3) sample_rows reads only the leading rows.
    got_2 = compute_normalization_stats_from_cache(blosc2_dir, sample_rows=2, batch_rows=2)
    oracle_2 = _oracle_cache_stats(blosc2_dir, n_rows=2)
    for name in NORM_CHANNEL_NAMES:
        assert got_2[name]['mean'] == pytest.approx(oracle_2[name]['mean'], rel=1e-6, abs=1e-6), name

    # (4) The builder emitted the same 3-channel stats it fits during transcode
    #     (both are no-smoothing + identical forward transform + [40, 400]
    #     Kovatchev constants), so they agree with the recomputed stats.
    emitted = _cache_stats(blosc2_dir)
    for name in NORM_CHANNEL_NAMES:
        assert emitted[name]['mean'] == pytest.approx(got[name]['mean'], rel=1e-6, abs=1e-6), name
        assert emitted[name]['std'] == pytest.approx(got[name]['std'], rel=1e-6, abs=1e-6), name

    print(f"\n[DUMP] norm_from_cache | oracle match + npy==blosc2 + sample_rows "
          f"honored + emitted==recomputed across {len(NORM_CHANNEL_NAMES)} channels")
