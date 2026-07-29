"""
T1DMAI Normalization — compute and load per-channel statistics.
================================================================

What this module is for
-----------------------
The model only ever sees normalized inputs (zero mean, unit variance) for the
three signal channels — the full input feature stack, with no temporal columns.
The mean and standard deviation needed to
perform that normalization are computed *once*, before training begins, by
simulating a large pool of independent patients and accumulating per-channel
statistics with Welford's online algorithm.  The resulting stats live in
``normalization_stats.json`` and are re-loaded by ``data.py``, ``train.py``,
``inference.py`` and ``gui.py`` whenever they need to convert between raw and
normalized space.

Run once before training::

    python normalization.py

Channels
--------
There are three channels that get normalized — the whole input stack.  Two of them are
"sparse" — they sit at zero most of the time and spike occasionally (carbs at
mealtimes, insulin at boluses).  Their distribution is heavy-tailed, so taking
the raw mean/std would be dominated by rare events and the dense baseline
(e.g. zero-carb windows) would land far below the normalized zero.

To fix that, the sparse channels are passed through ``log1p`` *before* the
mean/std fit.  ``log1p(x) ≈ x`` near zero, so the dense baseline barely moves,
but rare large spikes are compressed into the bulk of the distribution.  The
denormalization path inverts the transform with ``expm1`` and clamps back to
``≥ 0`` (these channels are physically non-negative).

The set of log1p-scaled channels is ``SPARSE_LOG1P_CHANNELS`` below — the
single source of truth for which channels live in log space.  Every
normalize/denormalize call site in the project consults this set, so changing
its membership requires re-running this script.

Channel roles
-------------
- Inputs (all three normalized signals): bg_absolute, carb_intake,
  insulin_combined.  These are the entire input feature stack — there are no
  temporal columns.  bg_absolute is z-scored in Kovatchev risk space (``f``
  applied before the z-fit, its sole input path); carb/insulin keep log1p+z.
  The model does not directly predict BG: the quantile head emits BG forecasts
  in Kovatchev risk space, anchored on the last context BG and inverted back to
  mg/dL downstream.
"""

from __future__ import annotations
import json
from typing import Any
import numpy as np

# All shared constants live in config.py.  The training-time master seed is
# combined with a +1_000_000 offset below to derive the normalization patient
# pool.  Training draws its own patient seeds by hashing (``compute_patient_seed``),
# so the two pools are not literally disjoint *ranges* — but a collision between
# this small offset band and a hashed training seed is astronomically unlikely,
# which is what prevents the "stats fitted on training samples" leakage class.
from config import (
    NORM_N_PATIENTS, NORM_STATS_FILE,
    MASTER_SEED, N_INPUT_FEATURES,
    PATIENT_UNIFORM_SAMPLE_PROB, SIMULATOR_WARMUP_HOURS,
)

# === Channel metadata ===
# Order matters: it pins (a) the integer index of every channel everywhere in
# the project and (b) the order Welford's accumulator iterates over.  Reordering
# this list invalidates every previously-saved checkpoint.
CHANNEL_NAMES = [
    'bg_absolute',         # Index 0: observed CGM blood glucose (mg/dL, post-CGM-noise)
    'carb_intake',         # Index 1: carbohydrate absorption curve (grams/step, post-absorption-noise)
    'insulin_combined',    # Index 2: combined basal+bolus insulin action (units/step, post-absorption-noise)
]

# Every input feature column is a normalized signal channel: the input stack is
# exactly [bg_absolute, carb_intake, insulin_combined] with no temporal columns,
# so CHANNEL_NAMES must line up one-to-one with the model's input features.
assert len(CHANNEL_NAMES) == N_INPUT_FEATURES, (
    f"CHANNEL_NAMES has {len(CHANNEL_NAMES)} entries but N_INPUT_FEATURES="
    f"{N_INPUT_FEATURES}; the normalized signal channels and the input features "
    "must agree."
)

# Convenience constant — number of channels to normalize.  Used by Welford's
# accumulator below to size the running-stats arrays.
N_CHANNELS = len(CHANNEL_NAMES)

# Sparse channels: non-negative, heavy-tailed event spikes (meal carbs, insulin
# boluses).  log1p compresses spikes into the bulk distribution while leaving
# the dense ≈0 baseline almost untouched, so the resulting normalized space
# is much better behaved than a plain z-score.
# Frozenset because membership is the only operation we ever do with it and
# treating it as immutable prevents accidental mutation at runtime.
SPARSE_LOG1P_CHANNELS: frozenset[str] = frozenset({
    'carb_intake', 'insulin_combined',
})

# Risk-space channels: the dense ``bg_absolute`` INPUT channel is fed through the
# Kovatchev risk transform ``f`` BEFORE the z-score (and inverted with ``f_inv``
# on the way back), so the fitted bg mean/std live in risk space.  This is the
# SOLE bg-input path — the single membership set every transform consults,
# mirroring ``SPARSE_LOG1P_CHANNELS``.  The two sets are disjoint (bg is
# dense/risk, carb+insulin are sparse/log1p).
RISK_SPACE_CHANNELS: frozenset[str] = frozenset({'bg_absolute'})


def _forward_transform(x: np.ndarray, name: str) -> np.ndarray:
    """Apply the pre-normalization transform for a channel.

    For sparse channels we clamp negative values to zero (they should never be
    negative, but float drift or simulator quirks can produce tiny negatives)
    and then take ``log1p`` so the spike distribution is compressed before the
    mean/std fit downstream.  The risk-space bg channel takes the Kovatchev ``f``
    instead (the sole bg-input path), so its fitted mean/std live in risk space.
    Dense channels pass through unchanged.
    """
    if name in RISK_SPACE_CHANNELS:
        # Risk transform BEFORE the z-fit, so the saved mean/std live in risk
        # space.  ``kovatchev_f_np`` clamps the raw bg to the physical range
        # internally, so f is always well-defined (no smoothing needed).
        from utils import kovatchev_f_np
        return kovatchev_f_np(x)
    if name in SPARSE_LOG1P_CHANNELS:
        # log1p(x) = log(1 + x).  Near zero this is ≈ x (so the dense baseline
        # stays put) but for x >> 1 it grows logarithmically (so spikes don't
        # blow out the std).  np.maximum guards against tiny negative drift.
        return np.log1p(np.maximum(x, 0.0))
    return x


def _welford_batch_update(
    counts: np.ndarray, means: np.ndarray, M2s: np.ndarray, c: int, vals: np.ndarray,
) -> None:
    """Merge a batch of channel-``c`` values into the running Welford accumulators.

    Chan et al.'s parallel-variance formula: fold the batch's own
    ``(n_b, mean_b, M2_b)`` into the running ``(counts[c], means[c], M2s[c])``
    in one vectorized step, mutating the three arrays in place at index ``c``.
    Equivalent to streaming each value through the scalar recurrence but orders
    of magnitude faster. ``vals`` may be any shape — it is reduced via ``.size``
    / ``.mean()`` / elementwise ops.
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
    """Turn the Welford accumulators into the canonical ``{name: {mean, std}}`` dict.

    Std is the sample standard deviation ``sqrt(M2 / (n - 1))``; the denominator
    clamp guards the (training-scale-impossible) zero-observation channel.
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
    Compute per-channel mean and std from ``n_patients`` independent simulations.

    The pool is drawn from the SAME generative process as the cache / on-the-fly
    training data — the ``patient_uniform_sample_prob`` skill mix and the
    ``ON_THE_FLY_SIM_HOURS`` post-warmup window — so the saved stats match the
    distribution the model actually sees. Fitting on a different distribution
    (e.g. 100% normal-skill patients over a long 720 h run) silently mis-scales
    the model's normalized inputs; see ``scratch/compare_norm_stats.py``.

    Seeds used: ``master_seed + 1_000_000 + i`` for ``i in range(n_patients)``.
    Training derives its patient seeds by hashing (``compute_patient_seed``), so
    this offset band and the training seeds are not disjoint *ranges*; the point
    is that a collision between the two is astronomically unlikely, so the
    normalization fit effectively never sees a training-distribution sample
    (no data leakage).

    Args:
        master_seed: Master training seed (used to derive disjoint normalization seeds).
        n_patients: Number of patients to simulate.
        n_hours: Post-warmup hours to generate per patient. ``None`` (default)
            resolves to ``data.ON_THE_FLY_SIM_HOURS`` so the window matches the
            cache / on-the-fly training data.
        patient_uniform_sample_prob: Per-patient probability of drawing
            uniformly-sampled skills (tail oversampling), matching the cache /
            training mix. Resolve it the same way the cache was built.
        simulator_warmup_hours: Hours discarded from the start of every run
            (cold-start window), matching the cache.

    Returns:
        stats: dict mapping channel_name → {'mean': float, 'std': float}.
    """
    # Welford's online algorithm — numerically stable single-pass mean/variance.
    # We need three running tensors per channel:
    #   counts[c] = number of samples seen for channel c
    #   means[c]  = running mean
    #   M2s[c]    = running sum of squared deltas from the running mean
    # All in float64 because we are summing tens of millions of values; float32
    # would accumulate visible rounding error over a stream this long.
    counts = np.zeros(N_CHANNELS, dtype=np.float64)
    means = np.zeros(N_CHANNELS, dtype=np.float64)
    M2s = np.zeros(N_CHANNELS, dtype=np.float64)

    # Lazy imports — keep a circular-import door closed. ``data.py`` imports
    # from ``normalization`` for ``CHANNEL_NAMES`` etc., so importing it at
    # module top would deadlock on first import. We reuse data.py's own
    # ``_make_simulator`` (uniform-skill aware) and ``ON_THE_FLY_SIM_HOURS`` so
    # the pool is drawn from the SAME generative process as the cache /
    # on-the-fly training data — same skill mix, same post-warmup window.
    from data import simulate_discard_warmup, _make_simulator, ON_THE_FLY_SIM_HOURS
    if n_hours is None:
        n_hours = ON_THE_FLY_SIM_HOURS

    print(f"Computing normalization statistics from {n_patients} patients × "
          f"{n_hours}h (uniform-skill prob {patient_uniform_sample_prob}, "
          f"warmup {simulator_warmup_hours}h)...")

    for i in range(n_patients):
        # Seed offset (1M) keeps the normalization pool disjoint from training.
        # The mod 2^31-1 keeps the seed within ``np.random``'s legal range —
        # the simulator passes this to ``np.random.default_rng`` internally.
        seed = (master_seed + 1_000_000 + i) % (2**31 - 1)
        # Per-patient uniform-skill decision, mirroring data.py's substream
        # (same XOR constant) so the skill mix matches the cache / training at
        # the same ``patient_uniform_sample_prob``.
        use_uniform = bool(
            np.random.default_rng(seed ^ 0x5A17_5EEDD).random() < patient_uniform_sample_prob
        )
        sim = _make_simulator(seed, uniform_skills=use_uniform)
        # ``simulate_discard_warmup`` runs n_hours + warmup_hours of simulation
        # and drops the warmup window so we never fit stats on the artificial
        # cold-start period (no IOB, fresh basal).  See data.py.
        data = simulate_discard_warmup(sim, n_hours, warmup_hours=simulator_warmup_hours)

        # Pull every channel out of the simulator dict and cast to float64 so
        # all downstream Welford updates happen at the same precision. Use
        # ``bg_observed`` (post-CGM-noise) so the stats live in the same space
        # the model sees at training time — the input pipeline normalizes
        # observed BG, not the underlying clean ``bg``.
        bg = data['bg_observed'].astype(np.float64)
        carb = data['total_carb'].astype(np.float64)
        insulin = data['total_insulin'].astype(np.float64)

        # Order MUST match CHANNEL_NAMES so the stats dict keys line up below.
        raw_channels = [bg, carb, insulin]

        # Apply the per-channel forward transform (bg → Kovatchev f, which clamps to
        # the physical BG range; sparse carb/insulin → log1p(max(x,0))) directly on
        # the RAW post-noise channels — no smoothing — BEFORE the Welford update, so
        # the saved stats live in the same raw + transformed space the model is fed.
        channels = [
            _forward_transform(arr, name)
            for arr, name in zip(raw_channels, CHANNEL_NAMES)
        ]

        # Fold each channel's per-patient values into the running accumulators.
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
    Compute per-channel stats DIRECTLY from a ``simulator_cache`` pool.

    Where :func:`compute_normalization_stats` re-simulates the matching
    distribution, this reads the cached trajectories the model actually trains
    on, so the result is exact against the pool — no sampling residual from a
    separate simulation. The transform is identical (``kovatchev_f_np`` on
    ``RISK_SPACE_CHANNELS`` — bg fit in Kovatchev risk space — and ``log1p`` on
    ``SPARSE_LOG1P_CHANNELS``), so the output is a drop-in replacement for the
    re-simulated stats. Rows are streamed in mmap-backed batches, so the full
    pool never resides in RAM.

    Args:
        cache_path: Directory produced by ``T1DMSIM/cache_simulator.py`` (a
            ``meta.json`` plus per-channel ``.npy`` (``npy-memmap-v1``) or ``.b2nd``
            (``blosc2-ndarray-v1``) arrays).
        sample_rows: ``None`` (default) reads every row — exact. An int reads
            only the first ``sample_rows`` rows: a valid representative
            subsample, since each cache row is an i.i.d. draw keyed on its own
            index, so the leading block is as representative as a random one, at
            a fraction of the I/O.
        batch_rows: Rows per mmap slice — larger means fewer, larger reads.

    Returns:
        stats: dict mapping channel_name → {'mean': float, 'std': float}.
    """
    import os
    # Lazy import dodges the data ↔ normalization circular import and reuses
    # data.py's cache constants as the single source of truth.
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
    # The cached channels we need (temporal, exercise, IS and HGO channels are
    # not normalized). Order of the per-channel read in the loop must match
    # CHANNEL_NAMES, NOT this set.
    needed = ('bg_observed', 'total_carb', 'total_insulin')

    arrays: dict[str, Any] = {}
    if cache_format == CACHE_FORMAT_NPY:
        for name in needed:
            arrays[name] = np.load(
                os.path.join(cache_path, f'{name}.npy'), mmap_mode='r')
    elif cache_format == CACHE_FORMAT_BLOSC2:
        import blosc2
        for name in needed:
            arrays[name] = blosc2.open(
                os.path.join(cache_path, f'{name}.b2nd'),
                mode='r', mmap_mode='r')  # type: ignore[arg-type]
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

        # Order MUST match CHANNEL_NAMES so stats keys line up.
        raw_channels = [bg, carb, insulin]
        for c, (arr, name) in enumerate(zip(raw_channels, CHANNEL_NAMES)):
            # Forward-transform each (B, T) block directly — bg → Kovatchev f
            # (clamped to the physical range), sparse → log1p(max(x,0)) — with NO
            # smoothing, matching data._build_sample, so the cache-derived stats
            # live in the same raw + transformed space the model is fed.
            _welford_batch_update(counts, means, M2s, c, _forward_transform(arr, name))

        if start == 0 or stop % (batch_rows * 25) < batch_rows or stop == n_rows:
            print(f"  Processed {stop:,}/{n_rows:,} rows", flush=True)

    return _finalize_welford_stats(counts, means, M2s)


def save_normalization_stats(
    stats: dict[str, dict[str, float]],
    path: str = NORM_STATS_FILE,
) -> None:
    """
    Persist normalization statistics to a JSON file.

    Args:
        stats: dict from ``compute_normalization_stats``.
        path: Output file path (default: ``normalization_stats.json``).
    """
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization statistics to {path}")


def load_normalization_stats(path: str = NORM_STATS_FILE) -> dict[str, dict[str, float]]:
    """
    Load normalization statistics from a JSON file.

    Args:
        path: Path to ``normalization_stats.json``.

    Returns:
        stats: dict mapping channel_name → {'mean': float, 'std': float}.
    """
    with open(path, 'r') as f:
        return json.load(f)


def normalize(
    data: np.ndarray,
    stats: dict[str, dict[str, float]],
    channel_names: list[str] | None = None,
) -> np.ndarray:
    """
    Normalize data using precomputed statistics.

    For sparse channels the forward path is ``(log1p(max(x, 0)) - mu) / std``;
    for dense channels it's the plain z-score ``(x - mu) / std``.  The two paths
    must stay symmetric to ``denormalize`` so the round-trip is identity to
    within float precision.

    Args:
        data: (..., N_CHANNELS) array of raw values.
        stats: Normalization statistics dict.
        channel_names: Channel names in the same order as ``data``'s last
            dimension.  Defaults to ``CHANNEL_NAMES``.

    Returns:
        normalized: same shape as ``data``.
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES
    # ``copy().astype(float32)`` so we don't mutate the caller's array and so
    # the result matches the dtype the model and DataLoader pipeline expect.
    result = data.copy().astype(np.float32)
    for c, name in enumerate(channel_names):
        mean = stats[name]['mean']
        std = stats[name]['std']
        # x is a *view* into result[..., c] — we write the normalized result
        # back to the same slot at the end of the iteration.
        x = result[..., c]
        if name in RISK_SPACE_CHANNELS:
            from utils import kovatchev_f_np
            x = kovatchev_f_np(x)
        elif name in SPARSE_LOG1P_CHANNELS:
            x = np.log1p(np.maximum(x, 0.0))
        # 1e-8 floor on std defends against any pathological saved-stats value
        # of zero (e.g. a channel with no events in the normalization pool).
        result[..., c] = (x - mean) / (std + 1e-8)
    return result


def denormalize(
    data: np.ndarray | 'torch.Tensor',  # type: ignore[name-defined]
    stats: dict[str, dict[str, float]],
    channel_names: list[str] | None = None,
) -> 'np.ndarray | torch.Tensor':  # type: ignore[name-defined]
    """
    Denormalize data back to raw units.

    The inverse of ``normalize``.  For sparse channels the path is
    ``max(expm1(x_norm * std + mu), 0)``; for dense channels it is
    ``x_norm * std + mu``.  Accepts both numpy arrays and torch tensors; the
    torch branch runs entirely in autograd-tracked tensor ops so any caller that
    needs the gradient through ``expm1`` (e.g. differentiable carb/insulin
    denormalization) keeps a clean backward path.

    Args:
        data: (..., N_CHANNELS) normalized array or tensor.
        stats: Normalization statistics dict.
        channel_names: Channel names. Defaults to ``CHANNEL_NAMES``.

    Returns:
        raw: same type and shape as input.
    """
    # Lazy torch import — the GUI / inference paths import this module at
    # process startup and we don't want to drag torch into a numpy-only context.
    import torch
    if channel_names is None:
        channel_names = CHANNEL_NAMES

    if isinstance(data, torch.Tensor):
        # Clone so backward gradients flow through this branch but the caller's
        # tensor isn't mutated in place.  ``.float()`` because losses always
        # want fp32 here regardless of the autocast dtype upstream.
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
                # expm1 is the exact inverse of log1p.  The clamp(min=0) absorbs
                # any tiny negative drift from float round-off — the underlying
                # channels are physically non-negative.
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
    # CLI entry point. Two ways to fit the stats, both writing the same JSON:
    #   * default — re-simulate the matching distribution (skill mix + warmup
    #     read from config.py, as train.py does) so the stats track the cache
    #     without needing it on disk.
    #   * --from-cache DIR — read the stats straight from a cache pool's bytes
    #     (exact, no sampling residual; faster than simulation).
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
        # All configuration lives in config.py now (no JSON tier).
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
