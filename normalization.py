"""
Per-channel normalization statistics: fit, load, apply.

Run once before training::

    python normalization.py

The fit simulates a pool of independent patients and accumulates Welford stats
into ``normalization_stats.json``, read back by ``data.py``, ``train.py``,
``inference.py`` and ``gui.py``.

Four normalized signal channels, and they are the whole input feature stack —
there are no temporal columns:

- ``bg_absolute`` — z-scored in Kovatchev RISK space: ``f`` before the z-fit, and
  the sole bg input path.  ``f_inv`` after the un-z-score on the way back.  The
  head emits BG in risk space too, anchored and inverted to mg/dL downstream.
- ``carb_intake`` / ``insulin_combined`` / ``exercise_equiv`` — sparse: at zero
  most of the time with heavy-tailed spikes (meals, boluses, exercise), so
  ``log1p`` runs before the mean/std fit.  ``log1p(x) ≈ x`` near zero leaves the
  dense baseline put while compressing spikes into the bulk; the inverse is
  ``expm1`` clamped ``≥ 0``, these channels being physically non-negative.
  ``exercise_equiv`` is carbohydrate-EQUIVALENT glucose disposal in g/step, so it
  takes carb's encoding exactly and never a glucose's.

``SPARSE_LOG1P_CHANNELS`` and ``RISK_SPACE_CHANNELS`` are the single sources of
truth for which channel takes which branch; every normalize/denormalize call site
in the project consults them, and changing either requires re-running this script.
"""

from __future__ import annotations
import json
import math
from typing import Any
import numpy as np

# The normalization pool is ``master_seed + 1_000_000 + i``.  Training hashes its
# own patient seeds, so the two are not disjoint *ranges* — a collision between
# this offset band and a hashed training seed is astronomically unlikely, and that
# is what keeps the fit off training samples.
from config import (
    NORM_N_PATIENTS, NORM_STATS_FILE,
    MASTER_SEED,
    PATIENT_UNIFORM_SAMPLE_PROB, SIMULATOR_WARMUP_HOURS,
)

# Order pins every channel's integer index project-wide and the order Welford's
# accumulator iterates in.  Reordering invalidates every saved checkpoint.
CHANNEL_NAMES = [
    'bg_absolute',         # observed CGM glucose, mg/dL, post-CGM-noise
    'carb_intake',         # carbohydrate absorption, g/step, post-absorption-noise
    'insulin_combined',    # basal+bolus insulin action, U/step, post-absorption-noise
    'exercise_equiv',      # exercise disposal as a carbohydrate EQUIVALENT, g/step
]

N_CHANNELS = len(CHANNEL_NAMES)

# Pinned at 4 and deliberately NOT tied to ``config.N_INPUT_FEATURES``, which is
# 5: input feat 4 (``bg_masked``) is a per-patch BIT announcing that feat 0 is
# withheld, carries no statistics, and is neither normalized nor denormalized here.
assert N_CHANNELS == 4, (
    f"CHANNEL_NAMES has {N_CHANNELS} entries; N_CHANNELS is pinned at 4 "
    "(bg_absolute, carb_intake, insulin_combined, exercise_equiv). Adding or "
    "removing a normalized signal channel invalidates every saved checkpoint "
    "and every saved normalization_stats.json, so change this deliberately."
)

SPARSE_LOG1P_CHANNELS: frozenset[str] = frozenset({
    'carb_intake', 'insulin_combined', 'exercise_equiv',
})

# Disjoint from SPARSE_LOG1P_CHANNELS; only a glucose ever belongs here.
RISK_SPACE_CHANNELS: frozenset[str] = frozenset({'bg_absolute'})


def _forward_transform(x: np.ndarray, name: str) -> np.ndarray:
    """The pre-normalization transform of one channel; dense channels pass through."""
    if name in RISK_SPACE_CHANNELS:
        # ``kovatchev_f_np`` clamps the raw bg to the physical range internally,
        # so f is always well-defined and no smoothing is needed.
        from utils import kovatchev_f_np
        return kovatchev_f_np(x)
    if name in SPARSE_LOG1P_CHANNELS:
        # max(x, 0) absorbs tiny negative float drift.
        return np.log1p(np.maximum(x, 0.0))
    return x


def _welford_batch_update(
    counts: np.ndarray, means: np.ndarray, M2s: np.ndarray, c: int, vals: np.ndarray,
) -> None:
    """Merge a batch of channel-``c`` values into the running Welford accumulators.

    Chan et al.'s parallel-variance formula, mutating ``counts``/``means``/``M2s``
    in place at index ``c``.  ``vals`` may be any shape.
    """
    n_b = vals.size
    if n_b == 0:
        return
    mean_b = vals.mean()
    M2_b = ((vals - mean_b) ** 2).sum()
    n_a = counts[c]
    n_ab = n_a + n_b
    delta = mean_b - means[c]
    means[c] += delta * (n_b / n_ab)
    M2s[c] += M2_b + delta * delta * (n_a * n_b / n_ab)
    counts[c] = n_ab


def _finalize_welford_stats(
    counts: np.ndarray, means: np.ndarray, M2s: np.ndarray,
) -> dict[str, dict[str, float]]:
    """The Welford accumulators as ``{name: {mean, std}}``.

    Std is the sample standard deviation ``sqrt(M2 / (n - 1))``; the denominator
    clamp guards a zero-observation channel.
    """
    denom = np.maximum(counts - 1, 1)
    stds = np.sqrt(M2s / denom)
    stats: dict[str, dict[str, float]] = {}
    for c, name in enumerate(CHANNEL_NAMES):
        stats[name] = {'mean': float(means[c]), 'std': float(stds[c])}
    return stats


def compute_normalization_stats(
    master_seed: int = MASTER_SEED,
    n_patients: int = NORM_N_PATIENTS,
    n_hours: float | None = None,
    patient_uniform_sample_prob: float = PATIENT_UNIFORM_SAMPLE_PROB,
    simulator_warmup_hours: float = SIMULATOR_WARMUP_HOURS,
) -> dict[str, dict[str, float]]:
    """
    Per-channel ``{name: {mean, std}}`` from ``n_patients`` independent simulations.

    The pool is drawn from the SAME generative process as the cache / on-the-fly
    training data — the ``patient_uniform_sample_prob`` skill mix and the
    ``ON_THE_FLY_SIM_HOURS`` post-warmup window (what ``n_hours=None`` resolves
    to), the ``simulator_warmup_hours`` cold-start discard — so the saved stats
    match the distribution the model sees.  Fitting on a different distribution
    (100% normal-skill patients over a 720 h run, say) silently mis-scales every
    normalized input.

    Seeds are ``master_seed + 1_000_000 + i``.  Training derives its patient seeds
    by hashing, so this band and the training seeds are not disjoint *ranges*; a
    collision is astronomically unlikely, so the fit effectively never sees a
    training sample.
    """
    # float64 because this sums tens of millions of values; float32 accumulates
    # visible rounding error over a stream this long.
    counts = np.zeros(N_CHANNELS, dtype=np.float64)
    means = np.zeros(N_CHANNELS, dtype=np.float64)
    M2s = np.zeros(N_CHANNELS, dtype=np.float64)

    # Lazy: ``data.py`` imports from this module, so a top-level import is
    # circular.  Reusing data.py's ``_make_simulator`` and ``ON_THE_FLY_SIM_HOURS``
    # is what keeps the pool on the same generative process as training.
    from data import simulate_discard_warmup, _make_simulator, ON_THE_FLY_SIM_HOURS
    if n_hours is None:
        n_hours = ON_THE_FLY_SIM_HOURS

    print(f"Computing normalization statistics from {n_patients} patients × "
          f"{n_hours}h (uniform-skill prob {patient_uniform_sample_prob}, "
          f"warmup {simulator_warmup_hours}h)...")

    for i in range(n_patients):
        # mod 2^31-1 keeps the seed inside ``np.random.default_rng``'s legal range.
        seed = (master_seed + 1_000_000 + i) % (2**31 - 1)
        # Same XOR constant as data.py's substream, so the skill mix matches the
        # cache / training at the same ``patient_uniform_sample_prob``.
        use_uniform = bool(
            np.random.default_rng(seed ^ 0x5A17_5EEDD).random() < patient_uniform_sample_prob
        )
        sim = _make_simulator(seed, uniform_skills=use_uniform)
        # Drops the artificial cold-start window (no IOB, fresh basal).
        data = simulate_discard_warmup(sim, n_hours, warmup_hours=simulator_warmup_hours)

        # ``bg_observed`` is post-CGM-noise: the input pipeline normalizes the
        # observed BG, never the clean ``data['bg']``.
        bg = data['bg_observed'].astype(np.float64)
        carb = data['total_carb'].astype(np.float64)
        insulin = data['total_insulin'].astype(np.float64)
        exercise = data['total_exercise'].astype(np.float64)

        # Order MUST match CHANNEL_NAMES so the stats keys line up.  A short list
        # truncates the zip below in silence: that channel finalizes to
        # {mean: 0.0, std: 0.0}, which the input pipeline divides by.
        raw_channels = [bg, carb, insulin, exercise]
        assert len(raw_channels) == len(CHANNEL_NAMES), (
            f"{len(raw_channels)} raw channels against {len(CHANNEL_NAMES)} "
            "CHANNEL_NAMES; every named channel needs its own array."
        )

        # Transformed BEFORE the Welford update, on the RAW post-noise channels
        # with no smoothing, so the saved stats live in the space the model is fed.
        channels = [
            _forward_transform(arr, name)
            for arr, name in zip(raw_channels, CHANNEL_NAMES)
        ]

        for c, vals in enumerate(channels):
            _welford_batch_update(counts, means, M2s, c, vals)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_patients} patients")

    return _finalize_welford_stats(counts, means, M2s)


def compute_normalization_stats_from_cache(
    cache_path: str,
    sample_rows: int | None = None,
    batch_rows: int = 4096,
) -> dict[str, dict[str, float]]:
    """
    Per-channel ``{name: {mean, std}}`` read DIRECTLY from a ``simulator_cache`` pool.

    Where :func:`compute_normalization_stats` re-simulates the matching
    distribution, this reads the trajectories the model actually trains on, so the
    result is exact against the pool.  The transform is identical, so the output
    is a drop-in replacement.  Rows are streamed in ``batch_rows`` slices, so the
    full pool never resides in RAM.

    ``cache_path`` is a ``T1DMSIM/cache_simulator.py`` directory: ``meta.json``
    plus per-channel ``.npy`` (``npy-memmap-v1``) or ``.b2nd``
    (``blosc2-ndarray-v1``) arrays.  ``sample_rows=None`` reads every row and is
    exact; an int reads only the leading block, which is as representative as a
    random one — each cache row is an i.i.d. draw keyed on its own index.
    """
    import os
    # Lazy: dodges the data ↔ normalization circular import.
    from data import CACHE_CHANNEL_NAMES, CACHE_FORMAT_NPY, CACHE_FORMAT_BLOSC2

    meta_path = os.path.join(cache_path, 'meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No meta.json under {cache_path!r}; not a cache directory "
            "(build one with T1DMSIM/cache_simulator.py)."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    pool_size = int(meta['pool_size'])
    n_timesteps = int(meta['n_timesteps'])
    cache_format = str(meta['cache_format'])
    if tuple(meta['channels']) != CACHE_CHANNEL_NAMES:
        raise ValueError(
            f"Cache channels {tuple(meta['channels'])} disagree with expected "
            f"{CACHE_CHANNEL_NAMES}; rebuild the cache."
        )
    # Temporal, IS and HGO channels are not normalized.  The per-channel read in
    # the loop must follow CHANNEL_NAMES order, NOT this tuple's.
    needed = ('bg_observed', 'total_carb', 'total_insulin', 'total_exercise')

    arrays: dict[str, Any] = {}
    if cache_format == CACHE_FORMAT_NPY:
        for name in needed:
            arrays[name] = np.load(
                os.path.join(cache_path, f'{name}.npy'), mmap_mode='r')
    elif cache_format == CACHE_FORMAT_BLOSC2:
        import blosc2
        for name in needed:
            # Deliberately NOT mmap_mode='r' (see data.py's _load_cache): a mapped
            # .b2nd accumulates every chunk it touches with no way to release
            # them, and this pass reads the WHOLE pool.
            arrays[name] = blosc2.open(
                os.path.join(cache_path, f'{name}.b2nd'), mode='r')
    else:
        raise ValueError(
            f"Unsupported cache_format {cache_format!r} (expected "
            f"{CACHE_FORMAT_NPY!r} or {CACHE_FORMAT_BLOSC2!r})."
        )

    n_rows = pool_size if sample_rows is None else min(int(sample_rows), pool_size)

    counts = np.zeros(N_CHANNELS, dtype=np.float64)
    means = np.zeros(N_CHANNELS, dtype=np.float64)
    M2s = np.zeros(N_CHANNELS, dtype=np.float64)

    print(
        f"Computing normalization statistics from cache {cache_path!r}: "
        f"{n_rows:,}/{pool_size:,} rows × {n_timesteps} steps "
        f"(format {cache_format}, uniform-skill prob "
        f"{meta.get('patient_uniform_sample_prob')}, sim_hours "
        f"{meta.get('sim_hours')}, warmup {meta.get('simulator_warmup_hours')})..."
    )

    for start in range(0, n_rows, batch_rows):
        stop = min(start + batch_rows, n_rows)
        bg = np.asarray(arrays['bg_observed'][start:stop], dtype=np.float64)  # (B, T)
        carb = np.asarray(arrays['total_carb'][start:stop], dtype=np.float64)
        insulin = np.asarray(arrays['total_insulin'][start:stop], dtype=np.float64)
        exercise = np.asarray(arrays['total_exercise'][start:stop], dtype=np.float64)

        # Order MUST match CHANNEL_NAMES so the stats keys line up.  A short list
        # truncates the zip in silence: that channel finalizes to
        # {mean: 0.0, std: 0.0}.
        raw_channels = [bg, carb, insulin, exercise]
        assert len(raw_channels) == len(CHANNEL_NAMES), (
            f"{len(raw_channels)} raw channels against {len(CHANNEL_NAMES)} "
            "CHANNEL_NAMES; every named channel needs its own array."
        )
        for c, (arr, name) in enumerate(zip(raw_channels, CHANNEL_NAMES)):
            # Each (B, T) block transformed with NO smoothing, matching
            # data._build_sample, so the stats live in the space the model is fed.
            _welford_batch_update(counts, means, M2s, c, _forward_transform(arr, name))

        if start == 0 or stop % (batch_rows * 25) < batch_rows or stop == n_rows:
            print(f"  Processed {stop:,}/{n_rows:,} rows", flush=True)

    return _finalize_welford_stats(counts, means, M2s)


def save_normalization_stats(
    stats: dict[str, dict[str, float]],
    path: str = NORM_STATS_FILE,
) -> None:
    """Persist normalization statistics to a JSON file."""
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization statistics to {path}")


def load_normalization_stats(path: str = NORM_STATS_FILE) -> dict[str, dict[str, float]]:
    """Load ``{name: {mean, std}}``, raising on a missing channel or a bad statistic.

    Validated on the way in, because nothing downstream distinguishes a malformed
    file from a well-formed one.  A missing channel reaches ``normalize`` through
    a ``.get`` default and trains an untrained channel; a ``std`` of ``0.0``
    divides by ``0 + 1e-8`` and scales its channel by ~1e8.  Neither raises, and
    both train to completion behind a plausible validation table.
    """
    with open(path, 'r') as f:
        stats = json.load(f)

    missing = [name for name in CHANNEL_NAMES if name not in stats]
    if missing:
        raise ValueError(
            f"{path}: no statistics for {missing}. One entry per CHANNEL_NAMES "
            f"channel is required ({list(CHANNEL_NAMES)}); the file carries "
            f"{sorted(stats)}. A file fitted before a channel was added still "
            f"yields a fully formed sample against the current pool."
        )
    for name in CHANNEL_NAMES:
        entry = stats[name]
        mean, std = float(entry['mean']), float(entry['std'])
        if not (math.isfinite(mean) and math.isfinite(std)):
            raise ValueError(
                f"{path}: channel {name!r} has non-finite statistics "
                f"(mean={mean}, std={std})."
            )
        if std <= 0.0:
            raise ValueError(
                f"{path}: channel {name!r} has std={std!r}. normalize divides by "
                f"std + 1e-8, so this scales the channel by ~1e8. "
                f"A zero std means the fitter recorded no observations for it — "
                f"most often a zip() truncated between the raw-channel arrays and "
                f"CHANNEL_NAMES."
            )
    return stats


def normalize(
    data: np.ndarray,
    stats: dict[str, dict[str, float]],
    channel_names: list[str] | None = None,
) -> np.ndarray:
    """Normalize a ``(..., N_CHANNELS)`` raw array; float32, same shape out.

    ``channel_names`` must be in the same order as ``data``'s last axis and
    defaults to ``CHANNEL_NAMES``.  Must stay symmetric to ``denormalize`` — the
    round trip is identity to within float precision.
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    # Copied so the caller's array is not mutated, float32 to match what the model
    # and the DataLoader expect.
    result = data.copy().astype(np.float32)
    for c, name in enumerate(channel_names):
        mean = stats[name]['mean']
        std = stats[name]['std']
        x = result[..., c]
        if name in RISK_SPACE_CHANNELS:
            from utils import kovatchev_f_np
            x = kovatchev_f_np(x)
        elif name in SPARSE_LOG1P_CHANNELS:
            x = np.log1p(np.maximum(x, 0.0))
        # The 1e-8 floor defends against a saved std of zero.
        result[..., c] = (x - mean) / (std + 1e-8)
    return result


def denormalize(
    data: np.ndarray | 'torch.Tensor',  # type: ignore[name-defined]
    stats: dict[str, dict[str, float]],
    channel_names: list[str] | None = None,
) -> 'np.ndarray | torch.Tensor':  # type: ignore[name-defined]
    """The inverse of ``normalize`` over a ``(..., N_CHANNELS)`` array or tensor.

    Same type and shape out.  The torch branch runs entirely in autograd-tracked
    ops, so a caller needing the gradient through ``expm1`` keeps a clean backward
    path.
    """
    # Lazy: the GUI / inference paths import this module at process startup and
    # must not drag torch into a numpy-only context.
    import torch
    if channel_names is None:
        channel_names = CHANNEL_NAMES

    if isinstance(data, torch.Tensor):
        # Cloned so the caller's tensor is not mutated in place; fp32 regardless
        # of any upstream autocast dtype.
        result = data.clone().float()
        for c, name in enumerate(channel_names):
            mean = stats[name]['mean']
            std = stats[name]['std']
            x = result[..., c] * (std + 1e-8) + mean
            if name in RISK_SPACE_CHANNELS:
                # Un-z-scoring yielded a RISK value; invert back to mg/dL.
                from utils import kovatchev_f_inv
                x = kovatchev_f_inv(x)
            elif name in SPARSE_LOG1P_CHANNELS:
                # clamp(min=0) absorbs float round-off; the channel is physically
                # non-negative.
                x = torch.expm1(x).clamp(min=0.0)
            result[..., c] = x
        return result
    else:
        # numpy branch — mirrors the torch branch one-to-one.
        result = data.copy().astype(np.float32)
        for c, name in enumerate(channel_names):
            mean = stats[name]['mean']
            std = stats[name]['std']
            x = result[..., c] * (std + 1e-8) + mean
            if name in RISK_SPACE_CHANNELS:
                from utils import kovatchev_f_inv_np
                x = kovatchev_f_inv_np(x)
            elif name in SPARSE_LOG1P_CHANNELS:
                x = np.maximum(np.expm1(x), 0.0)
            result[..., c] = x
        return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute per-channel normalization statistics and write '
                    f'them to {NORM_STATS_FILE}.'
    )
    parser.add_argument(
        '--from-cache', type=str, default=None, metavar='DIR',
        help='Compute stats directly from a simulator_cache directory (exact '
             'against the pool) instead of re-simulating the distribution.'
    )
    parser.add_argument(
        '--cache-sample-rows', type=int, default=None, metavar='N',
        help='With --from-cache, read only the first N rows (a representative '
             'i.i.d. subsample) rather than the whole pool. Omit for exact.'
    )
    parser.add_argument(
        '--out', type=str, default=NORM_STATS_FILE,
        help=f'Output JSON path (default: {NORM_STATS_FILE}).'
    )
    args = parser.parse_args()

    if args.from_cache is not None:
        stats = compute_normalization_stats_from_cache(
            args.from_cache, sample_rows=args.cache_sample_rows,
        )
    else:
        _uniform_prob = PATIENT_UNIFORM_SAMPLE_PROB
        _warmup = SIMULATOR_WARMUP_HOURS
        print(
            f"Resolved patient_uniform_sample_prob={_uniform_prob}, "
            f"simulator_warmup_hours={_warmup} (config.py)"
        )
        stats = compute_normalization_stats(
            patient_uniform_sample_prob=_uniform_prob,
            simulator_warmup_hours=_warmup,
        )

    print("\nNormalization statistics:")
    for name, vals in stats.items():
        print(f"  {name}: mean={vals['mean']:.6f}, std={vals['std']:.6f}")
    save_normalization_stats(stats, args.out)
