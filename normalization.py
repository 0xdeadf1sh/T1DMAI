"""
Per-channel normalization statistics: fit, load, apply. Run once: python normalization.py
SPARSE_LOG1P_CHANNELS / RISK_SPACE_CHANNELS are the single source of truth for the branch.
"""

from __future__ import annotations
import json
import math
from typing import Any
import numpy as np

# Pool seed = master_seed + 1_000_000 + i; a training-seed collision is astronomically unlikely.
from config import (
    INPUT_LAYOUT, INPUT_LAYOUTS,
    NORM_N_PATIENTS, NORM_STATS_FILE,
    MASTER_SEED,
    PATIENT_UNIFORM_SAMPLE_PROB, SIMULATOR_WARMUP_HOURS,
)

# Order pins each channel's index project-wide; reordering invalidates every saved checkpoint.
CHANNEL_NAMES = list(INPUT_LAYOUTS[INPUT_LAYOUT])

N_CHANNELS = len(CHANNEL_NAMES)

# Pinned per layout: a changed count invalidates every checkpoint and stats file of that layout.
assert N_CHANNELS == {'curves': 3, 'events': 9}[INPUT_LAYOUT], (
    f"layout {INPUT_LAYOUT!r} has {N_CHANNELS} normalized channels; the count is pinned"
)

# Curves: g/step, U/step. Events: g, U, U at the onset slot.
SPARSE_LOG1P_CHANNELS: frozenset[str] = frozenset({
    'carb_intake', 'insulin_combined',
    'carb_g', 'bolus_u', 'basal_u',
})

# Disjoint from SPARSE_LOG1P_CHANNELS; only a glucose ever belongs here.
RISK_SPACE_CHANNELS: frozenset[str] = frozenset({'bg_absolute'})

# Dose descriptors: ln(x / ref) on a dosed slot, 0 elsewhere; stats fixed at mean 0, std 1.
LOG_RATIO_REFS: dict[str, float] = {
    'carb_gi': 50.0,
    'bolus_peak_min': 60.0, 'bolus_dur_h': 5.0,
    'basal_peak_min': 60.0, 'basal_dur_h': 5.0,
}
LOG_RATIO_STATS = {'mean': 0.0, 'std': 1.0}


def log_ratio(x: np.ndarray, ref: float) -> np.ndarray:
    """ln(x / ref) where x > 0, else 0."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0.0, np.log(np.maximum(x, 1e-6) / ref), 0.0)


def _forward_transform(x: np.ndarray, name: str) -> np.ndarray:
    """The pre-normalization transform of one channel; dense channels pass through."""
    if name in RISK_SPACE_CHANNELS:
        # kovatchev_f_np clamps raw bg internally, so f is always well-defined; no smoothing needed.
        from utils import kovatchev_f_np
        return kovatchev_f_np(x)
    if name in SPARSE_LOG1P_CHANNELS:
        # max(x, 0) absorbs tiny negative float drift.
        return np.log1p(np.maximum(x, 0.0))
    if name in LOG_RATIO_REFS:
        return log_ratio(x, LOG_RATIO_REFS[name])
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
    patient_uniform_sample_prob: float = PATIENT_UNIFORM_SAMPLE_PROB,
    simulator_warmup_hours: float = SIMULATOR_WARMUP_HOURS,
) -> dict[str, dict[str, float]]:
    """
    Per-channel ``{name: {mean, std}}`` from ``n_patients`` rows off T1DMSIM's row builder.

    Fitted over each row's context and all four tails, the span a sample draws from.
    """
    # float64: sums tens of millions of values; float32 accumulates visible rounding error here.
    counts = np.zeros(N_CHANNELS, dtype=np.float64)
    means = np.zeros(N_CHANNELS, dtype=np.float64)
    M2s = np.zeros(N_CHANNELS, dtype=np.float64)

    # Lazy: data.py imports this module (circular).
    from data import simulate_row, SIM_CHANNEL

    print(f"Computing normalization statistics from {n_patients} rows "
          f"(uniform-skill prob {patient_uniform_sample_prob}, "
          f"warmup {simulator_warmup_hours}h)...")

    for i in range(n_patients):
        # mod 2^31-1 keeps the seed inside ``np.random.default_rng``'s legal range.
        seed = (master_seed + 1_000_000 + i) % (2**31 - 1)
        row, _icr, _skills = simulate_row(seed)

        # bg_observed is post-CGM-noise; the pipeline normalizes it, never the clean row['bg'].
        for c, name in enumerate(CHANNEL_NAMES):
            src = SIM_CHANNEL[name]
            vals = np.concatenate(
                [np.asarray(row[src]).ravel(),
                 np.asarray(row[f'tail_{src}']).ravel()]).astype(np.float64)
            # Transformed before Welford, on raw post-noise values; stats live in model space.
            _welford_batch_update(counts, means, M2s, c, _forward_transform(vals, name))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_patients} patients")

    return _finalize_welford_stats(counts, means, M2s)


def compute_normalization_stats_from_cache(
    cache_path: str,
    sample_rows: int | None = None,
    batch_rows: int = 4096,
) -> dict[str, dict[str, float]]:
    """
    Per-channel ``{name: {mean, std}}`` read directly from a simulator_cache pool (exact, no resim).

    sample_rows=None reads every row exactly; an int samples the leading i.i.d. block.
    """
    import os
    # Lazy: dodges the data ↔ normalization circular import.
    from data import CACHE_CHANNEL_NAMES, CACHE_FORMAT_BLOSC2, SIM_CHANNEL

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
    if INPUT_LAYOUT != 'curves':
        raise ValueError(
            "the events layout takes the cache's own normalization_stats_events.json "
            "(cache_simulator.py --events); only the curves layout is fit here."
        )
    if tuple(meta['channels'])[:len(CACHE_CHANNEL_NAMES)] != CACHE_CHANNEL_NAMES:
        raise ValueError(
            f"Cache channels {tuple(meta['channels'])} disagree with expected "
            f"{CACHE_CHANNEL_NAMES}; rebuild the cache."
        )
    if cache_format != CACHE_FORMAT_BLOSC2:
        raise ValueError(
            f"Unsupported cache_format {cache_format!r} (expected "
            f"{CACHE_FORMAT_BLOSC2!r})."
        )
    # Temporal, IS and HGO channels are not normalized; the loop follows CHANNEL_NAMES order.
    needed = [SIM_CHANNEL[name] for name in CHANNEL_NAMES]

    import blosc2
    arrays: dict[str, Any] = {}
    for name in needed + [f'tail_{n}' for n in needed]:
        # Not mmap: a mapped .b2nd never releases touched chunks, and this reads the whole pool.
        arrays[name] = blosc2.open(
            os.path.join(cache_path, f'{name}.b2nd'), mode='r')

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
        # Context and all four tails, the span a sample draws from.
        for c, name in enumerate(CHANNEL_NAMES):
            src = needed[c]
            block = np.concatenate([
                np.asarray(arrays[src][start:stop], dtype=np.float64).ravel(),
                np.asarray(arrays[f'tail_{src}'][start:stop], dtype=np.float64).ravel(),
            ])
            # Transformed with no smoothing, matching data._build_sample's space.
            _welford_batch_update(counts, means, M2s, c, _forward_transform(block, name))

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

    A missing channel silently trains untrained; std=0.0 scales the channel by ~1e8.
    Neither raises — both complete training behind a plausible table.
    """
    with open(path, 'r') as f:
        stats = json.load(f)
    # A dose descriptor has no fitted statistic; a cache's file may leave it out.
    for name in CHANNEL_NAMES:
        if name in LOG_RATIO_REFS:
            stats.setdefault(name, dict(LOG_RATIO_STATS))

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

    channel_names order matches data's last axis; must stay symmetric with denormalize.
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    # Copied so the caller's array isn't mutated; float32 to match the model/DataLoader.
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
        elif name in LOG_RATIO_REFS:
            x = log_ratio(x, LOG_RATIO_REFS[name])
        # The 1e-8 floor defends against a saved std of zero.
        result[..., c] = (x - mean) / (std + 1e-8)
    return result


def denormalize(
    data: np.ndarray | 'torch.Tensor',  # type: ignore[name-defined]
    stats: dict[str, dict[str, float]],
    channel_names: list[str] | None = None,
) -> 'np.ndarray | torch.Tensor':  # type: ignore[name-defined]
    """The inverse of ``normalize`` over a ``(..., N_CHANNELS)`` array or tensor.

    Same type and shape out; the torch branch is autograd-tracked, so grad flows through expm1.
    """
    # Lazy: GUI/inference import this module at startup and must not drag in torch.
    import torch
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    # ln(x / ref) maps an empty slot and a dose at ref both to 0: no inverse exists.
    lossy = [n for n in channel_names if n in LOG_RATIO_REFS]
    if lossy:
        raise ValueError(f"denormalize: no inverse for dose-descriptor channels {lossy}")

    if isinstance(data, torch.Tensor):
        # Cloned so the caller's tensor isn't mutated; fp32 regardless of upstream autocast dtype.
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
                # clamp(min=0) absorbs float round-off; the channel is physically non-negative.
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
