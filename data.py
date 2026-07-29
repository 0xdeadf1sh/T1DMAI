"""
T1DMAI Data Pipeline — on-the-fly dataset generation from T1DMSIM.
==================================================================

What this module produces
-------------------------
A ``T1DMDataset`` returns one training sample per ``__getitem__`` call.
A "sample" is everything the model needs for one optimization step's slice
of one batch:

* ``patches``           — (T, PATCH_DIM) float tensor.  T = n_ctx + PREDICTION_PATCHES.
                          Each patch is ``PATCH_SIZE × N_INPUT_FEATURES = PATCH_DIM`` raw
                          values — there are NO mask bits.  The model is always
                          conditioned on the future carb/insulin; in the prediction
                          zone bg (feat 0) is zeroed (it is predicted) while carb
                          (feat 1) / insulin (feat 2) carry their true values.
* ``targets``           — (PREDICTION_PATCHES, PATCH_SIZE) float tensor of
                          ground-truth BG (mg/dL).  This is the RAW prediction-zone
                          ``bg_observed`` (the SAME raw signal fed as the model
                          input — no smoothing).  NOT f-transformed in the batch;
                          the Kovatchev risk transform is applied exactly once at
                          the top of the loss.
* ``n_context_patches`` — int, length of the variable-size context window.
* ``bg_formula_data``   — dict of per-sample scalars/trajectories consumed by the
                          validation / inference paths: ``last_bg`` (the raw
                          last-context BG, mg/dL), ``true_bg_trajectory`` and
                          ``extended_true_bg_trajectory`` (mg/dL ground truth over
                          the prediction zone and the long horizon),
                          ``pred_start_hour``, and the announced future
                          ``extended_carb_*`` / ``extended_insulin_*`` (normalized
                          + raw) used by the conditioned rolling override and the
                          counterfactual probes.

One raw post-noise space (no smoothing)
---------------------------------------
Every signal channel — bg, carb, AND insulin — is fed to the model RAW: there is no
causal smoother on inputs or on the forecast target.  The SAME raw bg is the model
input, the forecast TARGET and ``last_bg`` (bg only clamped to the physical
[BG_CLAMP_MIN, BG_CLAMP_MAX] range; carb/insulin floored at 0).  There is no
input/target asymmetry: the model lives entirely in this raw post-noise space
(inputs, target, loss, metrics).  Deployment realism is intrinsic — the live
CGM/dose stream is consumed exactly as-is, and the autoregressive roll re-feeds the
model's own raw output, so the train and inference input distributions match.

Conventions and gotchas
-----------------------
* Context windows are sampled uniformly in [MIN_CONTEXT_PATCHES,
  MAX_CONTEXT_PATCHES]; when MIN == MAX the context is fixed-length. The
  collate function left-pads shorter sequences so the prediction horizon
  is always at the right edge.
* The input feature stack is exactly [bg_absolute, carb, insulin] (3
  features); there are no temporal sin/cos features. bg (feat 0) enters in
  Kovatchev risk space — z(f(bg)) — while carb/insulin keep log1p+z.
* There are no mask bits and no conditioned/unconditioned dichotomy.  In the
  prediction horizon carb (feat 1) and insulin (feat 2) ALWAYS carry their true
  future values (the model is always conditioned on the announced doses), while
  bg (feat 0) is ALWAYS zeroed there (its input slot is blanked so the model
  cannot copy the ground-truth signal it is asked to predict).

GPU-starvation fix (April 2026)
-------------------------------
The simulator is Python-bound: ``T1DMSimulator.generate_hours(N)`` runs an
explicit Python loop over N timesteps.  The on-the-fly path requests
``ON_THE_FLY_SIM_HOURS`` of post-warmup simulation — enough for a
full 24 h prediction-start hour-of-day jitter window on top of the context +
long-horizon footprint.  See the constant's own docstring near its definition for
the exact arithmetic that justifies its value.
"""

import json
import mmap
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any

from config import (
    PATCH_SIZE, N_INPUT_FEATURES, PATCH_DIM,
    CHANNEL_TO_FEAT, NON_MASKABLE_FEATS,
    MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES, PREDICTION_PATCHES,
    PATIENT_UNIFORM_SAMPLE_PROB,
    SIMULATOR_WARMUP_HOURS, NIGHT_LONG_HORIZON_PATCHES, CACHE_MADVISE_DONTNEED,
    TIME_PROBE_ENABLED, TIME_PROBE_CROSS_WINDOW_WEIGHT,
)
from utils import compute_patient_seed, create_attention_mask, kovatchev_f_np
from normalization import CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS

# === On-the-fly simulator request size ===
# Each training sample needs at most
#   MAX_CONTEXT_PATCHES + max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)
#   patches × PATCH_SIZE timesteps/patch × DT_MINUTES = 5 min/timestep
# of raw trajectory.  With MAX_CONTEXT_PATCHES context patches (24 h) plus the
# NIGHT_LONG_HORIZON_PATCHES (8 h) long-horizon window that is 32 h minimum.
#
# Crucially, the post-warmup trajectory always starts at midnight (warmup
# discards an integer number of hours, so warmup_hours % 24 == 0 → trajectory
# step 0 is midnight).  The pred_start_step jitter is therefore *also* an
# hour-of-day range, and ``_pick_pred_start_step`` only samples patch-aligned
# positions in [first, last] where:
#   first = n_ctx · PATCH_SIZE                                 = 288
#   last  = floor((N - n_long_horizon_steps) / PATCH_SIZE) · PATCH_SIZE
# Candidates land at 30-min spacing (PATCH_SIZE × STEP_MINUTES = 30 min).
# For *exactly* uniform hour-of-day coverage we want MAX_CONTEXT_PATCHES
# candidates spanning 23.5 h between first and last (so they land on distinct
# 30-min slots after mod 24).  With MAX_CONTEXT_PATCHES = 48,
# NIGHT_LONG_HORIZON_PATCHES = 16 and PATCH_SIZE = 6 that gives:
#   first = MAX_CONTEXT_PATCHES · PATCH_SIZE                    = 288
#   last  = first + (MAX_CONTEXT_PATCHES - 1) · PATCH_SIZE      = 570
#   N     = last + NIGHT_LONG_HORIZON_PATCHES · PATCH_SIZE      = 666 steps = 55.5 h
# Anything larger creates duplicates at midnight (one extra candidate at
# hour 0).  Anything smaller cuts coverage off short of 23.5 h.
ON_THE_FLY_SIM_HOURS: float = 55.5

# === Conformal-calibration partition (RESERVED) ===
# A disjoint seed band for the conformal-calibration partition, kept clear of
# both the training hashed seeds and normalization's ``+1_000_000`` band.  The
# i-th calibration patient draws ``master_seed + CALIBRATION_SEED_OFFSET + i``.
# This partition backs split-conformal band recalibration (``conformal.py`` via
# ``calibrate_conformal.py``); it still feeds neither the training loop nor the
# headline validation metrics — only the post-hoc conformal fit.
CALIBRATION_SEED_OFFSET: int = 2_000_000


# === Cache-pool partitioning (train / val / cal disjointness) ===
# In cache mode the index→row map is ``cache_idx = patient_seed % pool_size``.
# The validation (``master_seed + 10_000_000``) and conformal-calibration
# (``seed_offset = CALIBRATION_SEED_OFFSET``) seed bands are distinct in seed
# space, but ``sha256(seed) % pool_size`` reprojects each band INDEPENDENTLY and
# uniformly over ``[0, pool_size)`` — so a val/cal sample can land on the exact
# cache row a train sample also uses, leaking held-out trajectories into
# training.  (The on-the-fly path is immune: distinct seed bands hash to distinct
# 63-bit seeds with astronomically low collision probability and never touch a
# shared finite pool.)
#
# Fix: carve the pool into three DISJOINT slabs keyed by a ``cache_partition``
# tag.  ``val`` and ``cal`` take small reserved tail bands (only a handful of
# patients are ever drawn from them); ``train`` takes the whole remaining head.
# Each partition maps ``cache_idx = slab_start + (patient_seed % slab_size)``, so
# no row is shared across partitions for ANY master seed.  The reserve sizes are
# fixed structural constants (not training tunables): generous enough to keep
# val/cal row diversity high, negligible against the multi-million-row pool.
CACHE_PARTITIONS: tuple[str, ...] = ('train', 'val', 'cal')
CACHE_VAL_SLAB_ROWS: int = 100_000   # reserved tail rows for the validation bands
CACHE_CAL_SLAB_ROWS: int = 100_000   # reserved tail rows for the calibration band


def _cache_slab_geometry(pool_size: int, partition: str) -> tuple[int, int]:
    """
    Half-open ``[slab_start, slab_start + slab_size)`` cache-row band for a
    partition, carved so the three bands are pairwise DISJOINT and cover
    ``[0, pool_size)``.

    Layout (so train keeps the large contiguous head):
        train : ``[0, pool_size - val_slab - cal_slab)``
        cal   : ``[pool_size - val_slab - cal_slab, pool_size - val_slab)``
        val   : ``[pool_size - val_slab, pool_size)``

    The val/cal reserves are clamped so they never starve train on a tiny test
    pool: each is at most a third of the pool, and at least 1 row.

    Args:
        pool_size: number of rows in the cache pool (``meta['pool_size']``).
        partition: one of ``CACHE_PARTITIONS``.

    Returns:
        ``(slab_start, slab_size)`` with ``slab_size >= 1``.
    """
    assert partition in CACHE_PARTITIONS, partition
    assert pool_size >= 3, f"cache pool_size={pool_size} too small to partition"
    third = pool_size // 3
    val_slab = max(1, min(CACHE_VAL_SLAB_ROWS, third))
    cal_slab = max(1, min(CACHE_CAL_SLAB_ROWS, third))
    train_slab = pool_size - val_slab - cal_slab
    assert train_slab >= 1, (
        f"train slab empty: pool_size={pool_size} val={val_slab} cal={cal_slab}")
    if partition == 'train':
        return 0, train_slab
    if partition == 'cal':
        return train_slab, cal_slab
    return train_slab + cal_slab, val_slab  # 'val'


class _UniformSkillRngProxy:
    """
    Proxy around a numpy ``Generator`` that overrides only ``multivariate_normal``
    to return raw skill values drawn from a uniform distribution.

    The T1DMSIM patient sampler draws patient skills via a single
    ``rng.multivariate_normal`` call, sigmoids the result, then clips to
    ``[SKILL_MIN, SKILL_MAX]``.  The normal sampler squeezes most patients
    near the centre, so "extreme" (very skilled / very unskilled) patients
    are rare.  This proxy short-circuits that one call so the resulting
    skills, after the sigmoid the simulator applies, land *uniformly*
    across ``[SKILL_MIN, SKILL_MAX]`` — oversampling tail patients.

    Every other rng method (``normal``, ``uniform``, ``integers``, …) is
    forwarded unchanged so the patient's non-skill parameters keep their
    usual distributions.
    """

    def __init__(self, rng: np.random.Generator, skill_min: float, skill_max: float) -> None:
        self._rng = rng
        self._skill_min = skill_min
        self._skill_max = skill_max

    def __getattr__(self, name: str):
        # Forward every attribute access to the wrapped generator.  This
        # makes the proxy behave like the real rng for everything except
        # the one method we override below.
        return getattr(self._rng, name)

    def multivariate_normal(self, mean, _cov, *_args, **_kwargs):
        n = len(mean)
        # Stay strictly inside (0, 1) so ``log(skills/(1-skills))`` is finite —
        # the simulator applies a sigmoid downstream so we work in logit space.
        lo = max(self._skill_min, 1e-3)
        hi = min(self._skill_max, 1.0 - 1e-3)
        skills = self._rng.uniform(lo, hi, size=n)
        # Inverse-sigmoid the uniform draw so that, after the simulator's
        # sigmoid, we get back to a uniform skill distribution.
        return np.log(skills / (1.0 - skills))


def _make_simulator(patient_seed: int, uniform_skills: bool):
    """
    Construct a fresh ``T1DMSimulator``, optionally with uniform-skill sampling.

    The returned simulator is stateful — ``generate_hours`` advances its
    internal clock — so we never cache instances across calls. A cache hit
    on a previously-used seed would otherwise hand back a simulator whose
    state has already advanced past the warmup window, silently corrupting
    any sample drawn at that seed (validation reusing a training seed,
    retried/refetched batches, etc.).
    """
    from T1DMSIM.simulator import T1DMSimulator
    if not uniform_skills:
        return T1DMSimulator(seed=patient_seed)

    # Uniform-skill mode: monkey-patch the module-level ``generate_patient``
    # for the duration of the simulator constructor.  This avoids
    # duplicating the simulator's patient-generation code, and inherits any
    # future changes to it for free.  We restore the original immediately
    # so other simulator instances are unaffected.
    from T1DMSIM import simulator as _sim_mod

    original = _sim_mod.generate_patient

    def _patched(rng):
        proxy: Any = _UniformSkillRngProxy(rng, _sim_mod.SKILL_MIN, _sim_mod.SKILL_MAX)
        return original(proxy)

    _sim_mod.generate_patient = _patched
    try:
        return T1DMSimulator(seed=patient_seed)
    finally:
        _sim_mod.generate_patient = original


def simulate_discard_warmup(sim, hours: float, warmup_hours: float = SIMULATOR_WARMUP_HOURS) -> dict:
    """
    Run ``sim.generate_hours(hours + warmup_hours)`` and drop the first
    ``warmup_hours`` from every returned array.

    The simulator starts from an empty meal / insulin history,
    so the first day has unrealistic dynamics (no prior-day IOB, no
    residual carb-on-board, fresh basal state).  Every non-test caller
    routes through this wrapper instead of calling ``generate_hours``
    directly, so training, normalization, inference and the GUI all see
    the same cold-start-free window.
    """
    from T1DMSIM.simulator import DT_MINUTES
    raw = sim.generate_hours(hours + warmup_hours)
    # Number of timesteps in the warmup window.  We slice every per-channel
    # array by ``[n_warmup:]`` so the returned dict starts at the first
    # post-warmup timestep.
    n_warmup = int(warmup_hours * 60 / DT_MINUTES)
    return {k: v[n_warmup:] for k, v in raw.items()}


def _pick_pred_start_step(
    n_steps: int,
    n_ctx: int,
    n_pred_steps: int,
    rng: np.random.Generator,
) -> int | None:
    """
    Pick a patch-aligned pred-zone start anywhere in the trajectory, requiring
    only that the preceding ``n_ctx`` patches are available as context and
    that ``n_pred_steps`` more timesteps follow.

    Returns ``None`` if no valid window exists in this trajectory.

    Args:
        n_steps: Total number of simulator timesteps available.
        n_ctx: Number of context patches.
        n_pred_steps: Length of the room required ahead of the start in raw
            timesteps. Callers pass the long-horizon footprint
            (NIGHT_LONG_HORIZON_PATCHES * PATCH_SIZE) so the trailing GT slice
            fits, even though the supervised prediction zone is shorter.
        rng: numpy Generator for the random pick.
    """
    n_ctx_steps = n_ctx * PATCH_SIZE

    # Earliest legal pred-zone start: needs n_ctx_steps of context behind it.
    # Latest legal pred-zone start: leaves room for n_pred_steps ahead.
    earliest = n_ctx_steps
    latest = n_steps - n_pred_steps
    if latest < earliest:
        return None

    # Patch-aligned candidates in [earliest, latest].
    first = ((earliest + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    last = (latest // PATCH_SIZE) * PATCH_SIZE
    if first > last:
        return None
    n_candidates = (last - first) // PATCH_SIZE + 1

    return int(first + PATCH_SIZE * int(rng.integers(0, n_candidates)))


def _pick_pred_start_step_at_hour(
    hour_of_day: np.ndarray,
    n_ctx: int,
    n_pred_steps: int,
    target_hour: float,
    rng: np.random.Generator,
    tol_hours: float = 0.5,
) -> int | None:
    """Patch-aligned pred-zone start whose hour-of-day is nearest ``target_hour``
    (circular distance), with the SAME context/horizon room requirement as
    ``_pick_pred_start_step``.

    Used for the night-onset validation: the prediction origin is forced to the
    bedtime hour (``NOCTURNAL_START_HOUR`` ≈ 22:00) so the rolled forecast spans
    the whole night. Among candidates within ``tol_hours`` of the target (one per
    day on a multi-day trajectory) one is chosen at random for variety; if none
    qualify, the single nearest candidate is used. Returns ``None`` when no legal
    window exists (mirrors ``_pick_pred_start_step``).

    Args:
        hour_of_day: (N,) per-step hour-of-day for the (trimmed) trajectory.
        n_ctx: context patches required behind the origin.
        n_pred_steps: timesteps of room required ahead (the long-horizon footprint).
        target_hour: desired origin hour-of-day (e.g. ``NOCTURNAL_START_HOUR``).
        rng: numpy Generator.
        tol_hours: candidates within this many hours of the target are eligible
            for the random pick.
    """
    n_steps = len(hour_of_day)
    earliest = n_ctx * PATCH_SIZE
    latest = n_steps - n_pred_steps
    if latest < earliest:
        return None
    first = ((earliest + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    last = (latest // PATCH_SIZE) * PATCH_SIZE
    if first > last:
        return None
    cands = np.arange(first, last + 1, PATCH_SIZE)
    d = np.abs(hour_of_day[cands] - float(target_hour)) % 24.0
    circ = np.minimum(d, 24.0 - d)
    near = cands[circ <= tol_hours]
    if len(near) > 0:
        return int(near[int(rng.integers(0, len(near)))])
    return int(cands[int(np.argmin(circ))])


CACHE_CHANNEL_NAMES = (
    'bg_observed',
    'total_carb',
    'total_insulin',
    'insulin_resistance',
    'hgo',
    'total_exercise',
    'hour_of_day',
    'day',
)


# Supported on-disk cache formats (see T1DMSIM/cache_simulator.py). ``blosc2`` is the
# compressed format; ``npy`` stores each channel as a raw uncompressed ``.npy``
# memmap read directly by the dataloader. Both carry the same ``meta.json``
# fields and per-row read semantics.
CACHE_FORMAT_BLOSC2 = 'blosc2-ndarray-v1'
CACHE_FORMAT_NPY = 'npy-memmap-v1'
SUPPORTED_CACHE_FORMATS = (CACHE_FORMAT_BLOSC2, CACHE_FORMAT_NPY)


class T1DMDataset(Dataset):
    """
    T1DM training dataset.

    Each index maps to a unique ``(step, position)`` pair, which deterministically
    derives a unique patient seed via ``compute_patient_seed``.

    With ``cache_path=None`` (default) the simulator is run on demand inside
    the worker — convenient but the dominant per-batch cost.  With
    ``cache_path`` pointing at a directory produced by ``T1DMSIM/cache_simulator.py``
    the dataset skips the simulator entirely and pulls pre-generated
    trajectories from disk, eliminating the GPU-starvation bottleneck.  In
    cache mode the index → cache row mapping is
    ``patient_seed % cache_pool_size`` so different ``master_seed``s draw
    different mixes from the same pool while every ``idx`` remains
    deterministic.

    Args:
        master_seed: Master seed for deterministic data generation.
        total_steps: Total number of training steps.
        batch_size: Samples per step (determines dataset length).
        normalization_stats: Dict from ``load_normalization_stats``.
        patient_uniform_sample_prob: Per-sample probability that this sample's
            patient is drawn with skills sampled uniformly across
            [SKILL_MIN, SKILL_MAX] instead of the simulator's default
            multivariate-normal sampler.  Used to oversample tail / extreme
            patients.  ``0.0`` disables the feature.  Ignored when
            ``cache_path`` is set — the cache bakes the uniform-skill mix in
            at generation time.
        simulator_warmup_hours: Hours discarded from the start of every
            simulator run (cold-start window).  Default
            ``SIMULATOR_WARMUP_HOURS``.  Ignored when ``cache_path`` is set
            (the cache's own warmup is fixed at generation time, and a
            mismatch raises at load).
        cache_path: Optional path to a simulator cache directory.  When set,
            simulator output is read from disk instead of regenerated.
        seed_offset: Added to ``master_seed`` before deriving every patient
            seed, so a disjoint dataset (e.g. the reserved conformal-calibration
            partition, ``CALIBRATION_SEED_OFFSET``) draws a seed band disjoint
            from the training and normalization bands.  Default 0 (training).
        force_pred_start_hour: when set, the prediction origin is forced to the
            patch-aligned step nearest this hour-of-day (validation-only).
        cache_partition: which DISJOINT cache-pool slab this dataset draws rows
            from in cache mode — one of ``CACHE_PARTITIONS`` (``'train'`` |
            ``'val'`` | ``'cal'``).  ``'val'`` and ``'cal'`` map to small reserved
            tail bands; ``'train'`` maps to the remaining head.  This is what
            keeps the +10M val / +2M cal seed bands from collapsing onto train
            rows via ``patient_seed % pool_size``.  No effect on the on-the-fly
            path (distinct seed bands already never collide there).
    """

    def __init__(
        self,
        master_seed: int,
        total_steps: int,
        batch_size: int,
        normalization_stats: dict[str, dict[str, float]],
        patient_uniform_sample_prob: float = PATIENT_UNIFORM_SAMPLE_PROB,
        simulator_warmup_hours: float = SIMULATOR_WARMUP_HOURS,
        cache_path: str | None = None,
        seed_offset: int = 0,
        force_pred_start_hour: float | None = None,
        cache_partition: str = 'train',
    ) -> None:
        self.master_seed = master_seed
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.stats = normalization_stats
        self.seed_offset = seed_offset
        # When set, the prediction origin is forced to the patch-aligned step
        # nearest this hour-of-day (e.g. NOCTURNAL_START_HOUR for the night-onset
        # validation) instead of a uniform-random origin. Validation-only.
        self.force_pred_start_hour = force_pred_start_hour
        self.patient_uniform_sample_prob = patient_uniform_sample_prob
        self.simulator_warmup_hours = simulator_warmup_hours
        self.cache_path = cache_path
        if cache_partition not in CACHE_PARTITIONS:
            raise ValueError(
                f"cache_partition={cache_partition!r} must be one of "
                f"{CACHE_PARTITIONS}."
            )
        self.cache_partition = cache_partition
        # Filled in by the first ``_load_cache`` once pool_size is known:
        # (slab_start, slab_size) for this partition. None in on-the-fly mode.
        self._cache_slab: tuple[int, int] | None = None

        # Lazy state — populated on first access inside the worker so the
        # blosc2 NDArray handles (mmap-backed) aren't pickled across the
        # DataLoader fork boundary.
        self._cache_arrays: dict[str, Any] | None = None
        self._cache_icr: np.ndarray | None = None
        self._cache_pool_size: int | None = None
        self._cache_n_timesteps: int | None = None
        self._cache_meta: dict[str, Any] | None = None
        # Per-channel madvise metadata for the npy-memmap format:
        # name -> (mmap_obj, data_offset_bytes, row_bytes). None for the blosc2
        # format (different memory model) or when MADV_DONTNEED is unavailable.
        self._cache_mmaps: dict[str, tuple[Any, int, int]] | None = None
        self._madv_dontneed: int | None = (
            getattr(mmap, 'MADV_DONTNEED', None) if CACHE_MADVISE_DONTNEED else None
        )

        if cache_path is not None:
            from T1DMSIM.simulator import DT_MINUTES as _DT_MINUTES
            meta_path = os.path.join(cache_path, 'meta.json')
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"Cache path {cache_path!r} is missing meta.json — "
                    "did you run T1DMSIM/cache_simulator.py to populate it, or did "
                    "the build crash mid-way? Re-run T1DMSIM/cache_simulator.py."
                )
            with open(meta_path) as f:
                meta = json.load(f)

            required_keys = (
                'pool_size', 'n_timesteps', 'sim_hours',
                'simulator_warmup_hours', 'patient_uniform_sample_prob',
                'dt_minutes', 'channels', 'cache_format',
            )
            missing = [k for k in required_keys if k not in meta]
            if missing:
                raise ValueError(
                    f"Cache meta.json at {cache_path!r} is missing keys "
                    f"{missing}. Cache was built by an older cache_simulator "
                    "— rebuild it with T1DMSIM/cache_simulator.py."
                )

            # Two formats are supported: the compressed blosc2 layout and the
            # uncompressed per-channel .npy memmap layout. The legacy .npy
            # caches (no cache_format key at all) are still rejected — they
            # are caught by the required-keys check above.
            cache_format = str(meta['cache_format'])
            if cache_format not in SUPPORTED_CACHE_FORMATS:
                raise ValueError(
                    f"Cache cache_format={cache_format!r} is not supported "
                    f"by this version of data.py (expected one of "
                    f"{SUPPORTED_CACHE_FORMATS}). Rebuild the cache with the "
                    "current T1DMSIM/cache_simulator.py."
                )

            # Every value the cache silently bakes into the trajectories
            # must match the dataset's runtime assumptions. Anything that
            # disagrees changes what the model sees vs the on-the-fly path,
            # so we fail loudly rather than train on quietly-divergent data.
            cache_warmup = float(meta['simulator_warmup_hours'])
            if abs(cache_warmup - float(simulator_warmup_hours)) > 1e-6:
                raise ValueError(
                    f"Cache simulator_warmup_hours={cache_warmup} disagrees with "
                    f"dataset simulator_warmup_hours={simulator_warmup_hours}. "
                    "Rebuild the cache with the matching warmup or change the "
                    "dataset/training config."
                )
            cache_sim_hours = float(meta['sim_hours'])
            if abs(cache_sim_hours - float(ON_THE_FLY_SIM_HOURS)) > 1e-6:
                raise ValueError(
                    f"Cache sim_hours={cache_sim_hours} disagrees with "
                    f"ON_THE_FLY_SIM_HOURS={ON_THE_FLY_SIM_HOURS}. "
                    "Rebuild the cache or change ON_THE_FLY_SIM_HOURS in data.py."
                )
            cache_dt = float(meta['dt_minutes'])
            if abs(cache_dt - float(_DT_MINUTES)) > 1e-6:
                raise ValueError(
                    f"Cache dt_minutes={cache_dt} disagrees with simulator "
                    f"DT_MINUTES={_DT_MINUTES}. Rebuild the cache."
                )
            cache_uniform = float(meta['patient_uniform_sample_prob'])
            if abs(cache_uniform - float(patient_uniform_sample_prob)) > 1e-6:
                raise ValueError(
                    f"Cache patient_uniform_sample_prob={cache_uniform} disagrees "
                    f"with dataset patient_uniform_sample_prob="
                    f"{patient_uniform_sample_prob}. The uniform-skill mix is "
                    "baked into cache rows at build time — rebuild the cache or "
                    "change the dataset/training config."
                )
            cache_channels = tuple(meta['channels'])
            if cache_channels != CACHE_CHANNEL_NAMES:
                raise ValueError(
                    f"Cache channels={cache_channels} disagrees with expected "
                    f"{CACHE_CHANNEL_NAMES}. Rebuild the cache."
                )

            self._cache_pool_size = int(meta['pool_size'])
            self._cache_n_timesteps = int(meta['n_timesteps'])
            self._cache_meta = meta
            # Carve this partition's disjoint row band out of the pool now that
            # pool_size is known (train head, val/cal reserved tails) so the
            # held-out seed bands can never reproject onto a train cache row.
            self._cache_slab = _cache_slab_geometry(
                self._cache_pool_size, self.cache_partition)

            # Heads-up: with pool_size < total samples drawn we silently cycle
            # the pool. The on-the-fly path has effectively-zero collision
            # over the 64-bit seed space; cache mode trades that for disk
            # locality. Surfacing this here so it's not invisible.
            n_samples = total_steps * batch_size
            if self._cache_pool_size < n_samples:
                reuse = n_samples / max(self._cache_pool_size, 1)
                print(
                    f"[T1DMDataset] cache pool_size={self._cache_pool_size} "
                    f"< total_steps*batch_size={n_samples}; each cache row is "
                    f"drawn ~{reuse:.1f}x. This is benign: _build_sample takes a "
                    "fresh random context+horizon window per draw (each "
                    f"{self._cache_n_timesteps}-step trajectory admits ~2112 "
                    "distinct patch-aligned windows), so every reuse is a "
                    "DIFFERENT training window — a pool far smaller than "
                    "total_steps*batch_size is fine.",
                    flush=True,
                )

    def __len__(self) -> int:
        # One dataset index per (step, position) pair — exactly enough work
        # to feed a full training run with no repeats.
        return self.total_steps * self.batch_size

    def _load_cache(self) -> tuple[dict[str, Any], np.ndarray]:
        """
        Open cache arrays on first use in this process.

        Returns the per-channel array dict (memory-mapped) and the
        per-patient ICR array (tiny, loaded fully into RAM).

        Two on-disk formats are supported, distinguished by
        ``meta['cache_format']``:

        * ``'blosc2-ndarray-v1'`` — each channel is a chunked, byte-shuffle +
          zstd ``.b2nd``; a per-row read decompresses exactly one chunk.
        * ``'npy-memmap-v1'`` — each channel is a raw uncompressed ``.npy``
          memmap; a per-row read faults in only the touched pages.

        In both cases ``mmap_mode='r'`` lets DataLoader workers share the
        kernel page cache, and ``arr[i:i+1]`` returns a fresh ndarray row, so
        ``__getitem__`` is identical across formats.
        """
        if self._cache_arrays is None:
            assert self.cache_path is not None
            assert self._cache_pool_size is not None
            assert self._cache_n_timesteps is not None
            assert self._cache_meta is not None
            expected_shape = (self._cache_pool_size, self._cache_n_timesteps)
            cache_format = str(self._cache_meta['cache_format'])
            arrays: dict[str, Any] = {}

            if cache_format == CACHE_FORMAT_NPY:
                mmaps: dict[str, tuple[Any, int, int]] = {}
                for name in CACHE_CHANNEL_NAMES:
                    arr = np.load(
                        os.path.join(self.cache_path, f'{name}.npy'),
                        mmap_mode='r',
                    )
                    if tuple(arr.shape) != expected_shape:
                        raise ValueError(
                            f"Cache channel {name!r} has shape {tuple(arr.shape)}, "
                            f"expected {expected_shape} from meta.json. The cache "
                            "directory is corrupt or partially-written — rebuild it."
                        )
                    arrays[name] = arr
                    # Random access over a multi-TB cache: suppress the kernel's
                    # 128 KB readahead so a fault pulls only the pages the row
                    # actually touches. Without this the per-row MADV_DONTNEED
                    # below drops only the row's ~2 pages and leaves the ~30
                    # readahead pages resident, so page cache grows ~128 KB per
                    # read (~0.5 GB/step) and floods the shared unified-memory
                    # pool. Best-effort; access here is 100% random so there is
                    # no sequential throughput to trade away.
                    _madv_random = getattr(mmap, 'MADV_RANDOM', None)
                    if _madv_random is not None:
                        try:
                            arr._mmap.madvise(_madv_random)
                        except (OSError, ValueError, AttributeError):
                            pass
                    # Record the mapping + per-row byte geometry so __getitem__
                    # can MADV_DONTNEED exactly the pages it faults in.
                    mmaps[name] = (
                        arr._mmap, int(arr.offset),
                        int(arr.shape[1] * arr.dtype.itemsize),
                    )
                if self._madv_dontneed is not None:
                    self._cache_mmaps = mmaps
            else:
                import blosc2
                for name in CACHE_CHANNEL_NAMES:
                    # blosc2's stub mistypes **kwargs as dict per kwarg;
                    # mmap_mode='r' is the documented API and works at runtime.
                    arr = blosc2.open(
                        os.path.join(self.cache_path, f'{name}.b2nd'),
                        mode='r',
                        mmap_mode='r',  # type: ignore[arg-type]
                    )
                    if not isinstance(arr, blosc2.NDArray):
                        raise ValueError(
                            f"Cache channel {name!r} is not a blosc2 NDArray "
                            f"(got {type(arr).__name__}). The cache directory "
                            "is corrupt or built by a different tool — rebuild it."
                        )
                    if tuple(arr.shape) != expected_shape:
                        raise ValueError(
                            f"Cache channel {name!r} has shape {tuple(arr.shape)}, "
                            f"expected {expected_shape} from meta.json. The cache "
                            "directory is corrupt or partially-written — rebuild it."
                        )
                    arrays[name] = arr
            self._cache_arrays = arrays
            icr = np.load(os.path.join(self.cache_path, 'icr.npy'))
            if icr.shape != (self._cache_pool_size,):
                raise ValueError(
                    f"Cache icr.npy has shape {icr.shape}, expected "
                    f"({self._cache_pool_size},). Rebuild the cache."
                )
            self._cache_icr = icr
        assert self._cache_arrays is not None and self._cache_icr is not None
        return self._cache_arrays, self._cache_icr

    def _madvise_row(self, cache_idx: int) -> None:
        """Reclaim the page-cache pages just read for row ``cache_idx``.

        For the npy-memmap cache format every row read faults a page or two of
        the multi-TB channel files into this worker's mapping; over the cache
        pool the access barely repeats, so those pages would otherwise
        accumulate as unbounded page cache. ``madvise(MADV_DONTNEED)`` on the
        page-aligned range covering the row drops them immediately (a re-read
        re-faults from the file). Best-effort: a no-op when the format is
        blosc2 or ``MADV_DONTNEED`` is unavailable, and any per-call failure is
        swallowed so a platform quirk can never break data loading.

        Args:
            cache_idx: Row index within the cache pool that was just read.
        """
        mmaps = self._cache_mmaps
        advice = self._madv_dontneed
        if mmaps is None or advice is None:
            return
        page = mmap.PAGESIZE
        for mm, data_offset, row_bytes in mmaps.values():
            start = data_offset + cache_idx * row_bytes
            aligned = start - (start % page)
            length = (start + row_bytes) - aligned
            length += (-length) % page  # round up to a whole number of pages
            try:
                mm.madvise(advice, aligned, length)
            except (OSError, ValueError, AttributeError):
                pass

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Generate one training sample for the given dataset index.

        Args:
            idx: Dataset index in ``[0, total_steps * batch_size)``.

        Returns:
            sample dict — see ``_build_sample`` for the full key list.
        """
        # Decompose the flat index back into the (step, position) pair the
        # training loop is on.  Determinism: the same idx always produces
        # the same patient_seed regardless of which worker handles it.  The
        # ``seed_offset`` shifts the whole band (0 = training; the reserved
        # conformal-calibration band uses ``CALIBRATION_SEED_OFFSET``).
        step = idx // self.batch_size
        position = idx % self.batch_size
        patient_seed = compute_patient_seed(
            self.master_seed + self.seed_offset, step, position,
        )

        if self.cache_path is not None:
            cache_arrays, cache_icr = self._load_cache()
            assert self._cache_pool_size is not None
            assert self._cache_slab is not None
            # Map into this partition's DISJOINT row band: a held-out (val/cal)
            # seed can only resolve to a reserved tail row, never a train row.
            slab_start, slab_size = self._cache_slab
            cache_idx = slab_start + int(patient_seed % slab_size)
            if self._cache_mmaps is not None:
                # npy-memmap format: ``[idx:idx+1]`` is a *view* into the shared
                # mmap, so copy each row out (np.array) to sever it from the
                # mapping, then MADV_DONTNEED the touched pages. Without the
                # copy the sample would alias the very pages we drop and
                # re-fault them on use. ``[idx:idx+1]`` rather than ``[idx]``
                # keeps the call inside the slice-indexing path the stubs annotate.
                data = {
                    name: np.array(cache_arrays[name][cache_idx:cache_idx + 1])[0]
                    for name in CACHE_CHANNEL_NAMES
                }
                self._madvise_row(cache_idx)
            else:
                # blosc2 format: indexing decompresses exactly the touched chunk
                # into a fresh writable ndarray, so no defensive copy is needed
                # and there is no resident mapping to advise away.
                data = {
                    name: np.asarray(cache_arrays[name][cache_idx:cache_idx + 1])[0]
                    for name in CACHE_CHANNEL_NAMES
                }
            icr = float(cache_icr[cache_idx])
        else:
            # With probability ``patient_uniform_sample_prob``, oversample a tail
            # patient by drawing skills uniformly.  The mode-decision rng is
            # keyed off ``patient_seed`` so the same idx always resolves the same
            # way — preserving full dataset determinism.  XOR with a magic
            # constant gives a different but still deterministic substream.
            if self.patient_uniform_sample_prob > 0.0:
                mode_rng = np.random.default_rng(patient_seed ^ 0x5A17_5EEDD)
                use_uniform = bool(mode_rng.random() < self.patient_uniform_sample_prob)
            else:
                use_uniform = False

            sim = _make_simulator(patient_seed, uniform_skills=use_uniform)
            # On-the-fly: simulate ON_THE_FLY_SIM_HOURS post-warmup, not the full
            # 720h normalization cohort runs.  This is the GPU-starvation fix —
            # we used to simulate ~12× more data than we needed per sample.
            data = simulate_discard_warmup(
                sim, ON_THE_FLY_SIM_HOURS, warmup_hours=self.simulator_warmup_hours
            )
            icr = float(sim.patient.icr)

        # Window-selection rng — separate seed substream so the mode rng above
        # doesn't influence window selection.
        rng = np.random.default_rng(patient_seed ^ 0xDEADBEEF)
        return _build_sample(
            data=data,
            icr=icr,
            stats=self.stats,
            rng=rng,
            force_pred_start_hour=self.force_pred_start_hour,
        )


def make_calibration_dataset(
    master_seed: int,
    n_patients: int,
    batch_size: int,
    normalization_stats: dict[str, dict[str, float]],
    cache_path: str | None = None,
) -> T1DMDataset:
    """
    Construct the conformal-calibration dataset.

    Draws its patient seeds from the band ``master_seed + CALIBRATION_SEED_OFFSET
    + i`` — disjoint from the training hashed seeds and from normalization's
    ``+1_000_000`` band — so the conformal recalibration pass scores patients
    neither the loss nor the headline validation ever saw.  This partition feeds
    split-conformal band recalibration only, never the training loop or the
    headline validation metrics.

    Args:
        master_seed: the same master seed the training run uses.
        n_patients: number of calibration patients (the dataset length).
        batch_size: collation batch size (``len == n_patients`` when this is 1).
        normalization_stats: the SAME stats dict training uses.
        cache_path: optional simulator cache directory.

    Returns:
        A ``T1DMDataset`` over the calibration seed band (no block masking).
    """
    return T1DMDataset(
        master_seed=master_seed,
        total_steps=n_patients,
        batch_size=batch_size,
        normalization_stats=normalization_stats,
        cache_path=cache_path,
        seed_offset=CALIBRATION_SEED_OFFSET,
        cache_partition='cal',
    )


# ---------------------------------------------------------------------------
# Shared post-simulator sample-building logic.
# ---------------------------------------------------------------------------
def _build_sample(
    data: dict[str, np.ndarray],
    icr: float,
    stats: dict[str, dict[str, float]],
    rng: np.random.Generator,
    force_pred_start_hour: float | None = None,
) -> dict[str, Any]:
    """
    Convert a raw simulator output dict into one training sample.

    Everything after the simulator call lives here so callers reuse the same
    feature pipeline.  The prediction window can start at any patch-aligned
    position in the trajectory — a single model is trained on both day-time
    and night-time windows simultaneously.

    One raw post-noise space (no smoothing): every signal channel (bg / carb /
    insulin) AND the BG target / ``last_bg`` are the RAW ``bg_observed`` /
    ``total_carb`` / ``total_insulin`` (bg clamped to the physical range,
    carb/insulin floored at 0).  There is no input/target asymmetry and no filter.

    Args:
        data: Dict with keys ``bg_observed``, ``total_carb``, ``total_insulin``,
              ``hour_of_day``, ``day`` (plus the unused-by-the-input-stack
              ``insulin_resistance``, ``hgo``, ``total_exercise``) — each a 1-D
              array of length N.  The model is fed the RAW ``bg_observed`` /
              ``total_carb`` / ``total_insulin``; the same raw bg is the target and
              last_bg.
        icr: Patient's insulin-to-carb ratio (accepted for caller compatibility;
              not consumed — no physics reconstruction in the redesign).
        stats: Normalization statistics from ``load_normalization_stats``.
        rng: numpy Generator used for window selection.
        force_pred_start_hour: force the origin near this hour-of-day.

    Returns:
        Dict with keys ``patches``, ``targets``, ``n_context_patches``,
        ``bg_formula_data``.
    """
    # Cast every consumed channel to float32 up front.  ``bg_observed`` is the
    # post-CGM-noise read; ``total_carb`` / ``total_insulin`` carry the simulator's
    # per-step AR(1) absorption noise.  There is NO smoothing: the RAW post-noise
    # signals are fed to the model AND used as the forecast target, so the whole
    # pipeline lives in one raw post-noise space.
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    bg_raw = data['bg_observed'].astype(np.float32)
    carb_raw = data['total_carb'].astype(np.float32)
    insulin_raw = data['total_insulin'].astype(np.float32)
    hour_of_day = data['hour_of_day'].astype(np.float32)
    day_index = data['day'].astype(np.int32)

    N = len(bg_raw)

    # No smoother. bg is only clamped to the physical BG range so it is a legal
    # Kovatchev-f / last_bg argument (the cache is already rail-filtered into
    # [41, 399], but on-the-fly generation and edge reads still need the guard);
    # the sparse carb/insulin channels are floored at 0. ``bg`` is BOTH the model
    # input bg and the forecast target source / last_bg — one raw signal.
    bg = np.clip(bg_raw, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.maximum(carb_raw, 0.0).astype(np.float32)
    insulin = np.maximum(insulin_raw, 0.0).astype(np.float32)

    # Stack into the canonical N_INPUT_FEATURES-feature input layout
    # [bg_absolute, carb, insulin]:
    #   0 bg_absolute, 1 carb, 2 insulin
    # All three signal channels are the RAW (clamped/floored) post-noise values.
    # hour_of_day is retained above for prediction-start selection / nocturnal
    # metadata; neither it nor day_index is a model input feature any longer.
    features = np.stack([
        bg, carb, insulin,
    ], axis=-1)  # (N, N_INPUT_FEATURES=3)
    assert features.shape[-1] == N_INPUT_FEATURES == len(CHANNEL_NAMES), (
        f"feature stack has {features.shape[-1]} cols, expected "
        f"N_INPUT_FEATURES={N_INPUT_FEATURES} == len(CHANNEL_NAMES)={len(CHANNEL_NAMES)}"
    )

    # Normalize every signal channel in place.  bg (feat 0) enters in Kovatchev
    # risk space — z(f(bg)) — the sole BG input path; the sparse carb / insulin
    # channels get log1p first so heavy-tailed event spikes don't distort the
    # z-score.  Column order == CHANNEL_NAMES order, so the gather never reads a
    # dropped channel.
    for c, name in enumerate(CHANNEL_NAMES):
        mean = stats[name]['mean']
        std = stats[name]['std']
        col = features[:, c]
        if name in RISK_SPACE_CHANNELS:
            # bg fed as z( f(bg) ).  bg is raw post-noise, physically clamped
            # to [BG_CLAMP_MIN, BG_CLAMP_MAX] above, so f is well-defined.
            col = kovatchev_f_np(col)
        elif name in SPARSE_LOG1P_CHANNELS:
            col = np.log1p(np.maximum(col, 0.0))
        features[:, c] = (col - mean) / (std + 1e-8)

    # === Random window selection ===
    # Each training sample is a single random window into this patient's
    # trajectory.  ``n_ctx`` is variable (8-24 h of context); the prediction
    # horizon length is fixed and may start at any patch-aligned position.
    n_ctx = int(rng.integers(MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES + 1))
    long_horizon_patches = max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)
    total_patches_needed = n_ctx + long_horizon_patches
    total_steps_needed = total_patches_needed * PATCH_SIZE

    # Trim N down to a multiple of PATCH_SIZE so reshape() below has clean
    # dimensions.  If even the trimmed length isn't enough for the chosen
    # n_ctx, fall back to the minimum context size.
    N_trimmed = (N // PATCH_SIZE) * PATCH_SIZE
    if N_trimmed < total_steps_needed:
        n_ctx = MIN_CONTEXT_PATCHES
        total_patches_needed = n_ctx + long_horizon_patches
        total_steps_needed = total_patches_needed * PATCH_SIZE

    n_pred_steps = PREDICTION_PATCHES * PATCH_SIZE
    n_long_horizon_steps = long_horizon_patches * PATCH_SIZE
    # The room requirement is the long horizon so the trailing GT slice fits.
    if force_pred_start_hour is not None:
        # Night-onset validation: force the origin to ~the bedtime hour so the
        # rolled forecast spans the whole night. Falls back to a uniform-random
        # origin if no candidate near the target hour leaves enough room.
        pred_start_step = _pick_pred_start_step_at_hour(
            hour_of_day[:N_trimmed], n_ctx, n_long_horizon_steps,
            float(force_pred_start_hour), rng,
        )
        if pred_start_step is None:
            pred_start_step = _pick_pred_start_step(
                N_trimmed, n_ctx, n_long_horizon_steps, rng,
            )
    else:
        pred_start_step = _pick_pred_start_step(
            N_trimmed, n_ctx, n_long_horizon_steps, rng,
        )
    if pred_start_step is None:
        # No valid window in this trajectory: skip this sample by raising —
        # DataLoader will retry the next index.  Only happens on pathologically
        # short simulator outputs.
        raise RuntimeError(
            f"No prediction window found; trajectory length {N_trimmed}, "
            f"n_ctx={n_ctx}, n_pred={n_pred_steps}"
        )
    start_step = pred_start_step - n_ctx * PATCH_SIZE
    end_step = start_step + total_steps_needed

    # Slice the (already-normalized) feature stack and the raw BG out of this
    # random window.  ``bg_window`` covers the full long-horizon range so the
    # validation rolling pass has raw ground-truth BG for every horizon; the model
    # only consumes the first PREDICTION_PATCHES patches of the prediction zone.
    window = features[start_step:end_step]
    bg_window = bg[start_step:end_step]
    # Normalized carb/insulin windows — used to build the announced future-input
    # overrides for the rolling validation and the counterfactual probes (carb=feat
    # 1, insulin=feat 2 in the normalized stack). ``carb`` / ``insulin`` here are the
    # RAW (floored) per-step signals, so the ``*_raw_window`` names are now literal.
    # Both consumers (predict_rolling discards the raw side; the counterfactual probe
    # re-normalizes through the same log1p-z stats on baseline and perturbed sides)
    # are self-consistent with the signal. Sliced to the long horizon below.
    _carb_feat = CHANNEL_TO_FEAT[0]
    _insulin_feat = CHANNEL_TO_FEAT[1]
    carb_norm_window = features[start_step:end_step, _carb_feat]
    insulin_norm_window = features[start_step:end_step, _insulin_feat]
    carb_raw_window = carb[start_step:end_step]
    insulin_raw_window = insulin[start_step:end_step]

    # Reshape into (n_patches, PATCH_SIZE, N_INPUT_FEATURES) so we can split
    # context vs prediction patches by axis-0 slicing.
    patches_3d = window.reshape(total_patches_needed, PATCH_SIZE, N_INPUT_FEATURES)
    # Leading-axis slices of a C-contiguous array are themselves contiguous, so
    # the reshape below is a view and the sole materializing ``.copy()`` happens
    # at the ``torch.from_numpy`` boundary — no redundant intermediate copy here.
    ctx_patches = patches_3d[:n_ctx]

    # Only the first PREDICTION_PATCHES patches of the prediction zone are
    # exposed to the model; patches beyond that exist in ``window`` only so the
    # extended BG trajectory carries the long-horizon ground truth.
    pred_patches = patches_3d[n_ctx:n_ctx + PREDICTION_PATCHES]

    # Flatten (PATCH_SIZE, N_INPUT_FEATURES) → PATCH_DIM.  There are NO mask bits:
    # the model is always conditioned on the future carb/insulin, which keep their
    # true values in the prediction zone.
    ctx_flat = ctx_patches.reshape(n_ctx, PATCH_SIZE * N_INPUT_FEATURES)

    # === BG target ===
    # The forecast target is the RAW prediction-zone BG (mg/dL), NOT f-transformed
    # in the batch (the risk transform is applied once in the loss).
    pred_start_in_window = n_ctx * PATCH_SIZE
    target_bg = bg_window[
        pred_start_in_window:pred_start_in_window + PREDICTION_PATCHES * PATCH_SIZE
    ]
    targets_t = torch.from_numpy(
        target_bg.reshape(PREDICTION_PATCHES, PATCH_SIZE).copy()
    )  # (P, S) mg/dL

    # Prediction-zone inputs: bg (feat 0, the only NON_MASKABLE_FEATS entry) is
    # what the model predicts, so it is ALWAYS zeroed; carb (feat 1) and insulin
    # (feat 2) keep their true future values — the model is always conditioned on
    # the announced doses (no block masking, no reveal mask).
    pred_flat_t = torch.from_numpy(
        pred_patches.reshape(PREDICTION_PATCHES, PATCH_SIZE * N_INPUT_FEATURES).copy()
    )
    for feat_idx in NON_MASKABLE_FEATS:
        pred_flat_t[:, feat_idx::N_INPUT_FEATURES] = 0.0

    ctx_tensor = torch.from_numpy(ctx_flat.copy())
    all_patches_t = torch.cat([ctx_tensor, pred_flat_t], dim=0)
    assert all_patches_t.shape[-1] == PATCH_DIM, (
        f"patch row width {all_patches_t.shape[-1]} != PATCH_DIM {PATCH_DIM}"
    )

    # === BG trajectories for validation / inference ===
    # ``true_bg_traj`` / ``extended_true_bg_traj`` are the RAW prediction-zone BG
    # (same raw signal as the target), sourced canonically here (never from a
    # left-padded context[-1,-1,0] at train time).  The pred zone is the rightmost
    # PREDICTION_PATCHES patches by construction.
    true_bg_traj = bg_window[
        pred_start_in_window:pred_start_in_window + PREDICTION_PATCHES * PATCH_SIZE
    ]
    extended_true_bg_traj = bg_window[
        pred_start_in_window:pred_start_in_window + n_long_horizon_steps
    ]
    # ``last_bg`` is the last-context RAW BG the model is ANCHORED on at the
    # context→prediction boundary — simply the last context value of ``bg`` (clamped
    # to the physical range above, so it is always a legal Kovatchev-f argument),
    # equal to the last context BG INPUT the model sees (inference reads the same
    # value, no re-derivation).
    if pred_start_step > 0:
        last_bg = float(bg[pred_start_step - 1])
    else:
        last_bg = float(bg[0])

    # Announced future carbs/insulin over the long horizon (normalized + raw),
    # for the conditioned night-onset rolling override.
    _lh = slice(pred_start_in_window, pred_start_in_window + n_long_horizon_steps)
    extended_carb_norm = carb_norm_window[_lh]
    extended_insulin_norm = insulin_norm_window[_lh]
    extended_carb_raw = carb_raw_window[_lh]
    extended_insulin_raw = insulin_raw_window[_lh]

    # Hour-of-day at the prediction-zone start for nocturnal metric filtering.
    # Indexed with the ABSOLUTE step ``pred_start_step``.
    pred_start_hour = float(hour_of_day[pred_start_step])

    bg_formula_data = {
        'last_bg': last_bg,
        'true_bg_trajectory': true_bg_traj.copy(),
        'extended_true_bg_trajectory': extended_true_bg_traj.copy(),
        'pred_start_hour': pred_start_hour,
        'extended_carb_norm': extended_carb_norm.copy(),
        'extended_insulin_norm': extended_insulin_norm.copy(),
        'extended_carb_raw': extended_carb_raw.copy(),
        'extended_insulin_raw': extended_insulin_raw.copy(),
    }

    sample = {
        'patches': all_patches_t.float(),
        'targets': targets_t.float(),
        'n_context_patches': n_ctx,
        'bg_formula_data': bg_formula_data,
    }

    # === Cross-window (paired-window) time-of-day probe input (window k+1) ===
    # Window k+1 is window k shifted forward by exactly PREDICTION_PATCHES: its
    # context ends at pred_start + P patches and it predicts [pred_start+P,
    # pred_start+2P].  TEACHER-FORCED on the SAME already-normalized raw
    # ``features`` (a pure re-slice — the sole (a)<->(b) normalize crossing at
    # ~L954-964 stays authoritative).  Its own prediction zone keeps feat 0 (bg)
    # zeroed (FROZEN NON_MASKABLE_FEATS) so no future bg leaks; carb/insulin stay
    # announced.  SAME n_ctx as window k => identical seq_len / left-pad / attn_mask
    # (train.py reuses ``batch['attn_mask']`` for the 2nd forward).  Diagnostic-only;
    # consumed by the cross-window consistency penalty.  Gated so the default-off
    # path ships nothing.
    if TIME_PROBE_ENABLED and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
        seq_len = n_ctx + PREDICTION_PATCHES
        next_end_patch = n_ctx + 2 * PREDICTION_PATCHES
        next_valid = next_end_patch <= total_patches_needed   # in-range on patches_3d
        if next_valid:
            next_ctx = patches_3d[PREDICTION_PATCHES:n_ctx + PREDICTION_PATCHES]   # (n_ctx, S, F)
            next_pred = patches_3d[n_ctx + PREDICTION_PATCHES:next_end_patch]      # (P, S, F)
            next_ctx_flat = next_ctx.reshape(n_ctx, PATCH_SIZE * N_INPUT_FEATURES)
            next_pred_flat_t = torch.from_numpy(
                next_pred.reshape(PREDICTION_PATCHES, PATCH_SIZE * N_INPUT_FEATURES).copy()
            )
            for feat_idx in NON_MASKABLE_FEATS:
                next_pred_flat_t[:, feat_idx::N_INPUT_FEATURES] = 0.0
            next_ctx_tensor = torch.from_numpy(next_ctx_flat.copy())
            next_patches_t = torch.cat([next_ctx_tensor, next_pred_flat_t], dim=0)  # (n_ctx+P, PATCH_DIM)
            next_pred_start_step = pred_start_step + PREDICTION_PATCHES * PATCH_SIZE
            next_last_bg = float(bg[next_pred_start_step - 1])          # raw mg/dL (clamped, physical)
            next_pred_start_hour = float(hour_of_day[next_pred_start_step])
        else:
            # Room only under NIGHT_LONG_HORIZON_HOURS == PREDICTION_HORIZON_HOURS.
            # Ship a finite placeholder (masked out downstream); last_bg reuses window
            # k's valid mg/dL so the forward's units tripwire never fires.
            next_patches_t = torch.zeros(seq_len, PATCH_DIM, dtype=torch.float32)
            next_last_bg = last_bg
            next_pred_start_hour = pred_start_hour
        assert next_patches_t.shape == (seq_len, PATCH_DIM)
        sample['next_window'] = {
            'patches': next_patches_t.float(),
            'last_bg': float(next_last_bg),
            'pred_start_hour': float(next_pred_start_hour),
            'valid': bool(next_valid),
        }

    return sample


def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate variable-length samples into a padded batch.

    Left-pads shorter context sequences so the last ``PREDICTION_PATCHES``
    tokens are always the prediction horizon.  Padding positions are blocked
    by the per-batch attention mask (and the diagonal is forced True so
    softmax doesn't NaN on all-False rows).

    Args:
        samples: List of dicts from ``T1DMDataset.__getitem__``.

    Returns:
        batch dict with keys::

          patches:           (B, max_T, PATCH_DIM)             float32
          targets:           (B, P, PATCH_SIZE)                float32 (mg/dL)
          attn_mask:         (B, max_T, max_T)                 bool
          bg_formula_data:   batched dict (see _build_sample)
          n_context_patches: (B,)                              long
          next_window:       optional batched dict (cross-window time-of-day
                             probe input; present iff samples carry it)
    """
    B = len(samples)
    n_contexts = [s['n_context_patches'] for s in samples]
    seq_lens = [n + PREDICTION_PATCHES for n in n_contexts]
    max_T = max(seq_lens)

    # Pre-allocate the padded outputs.  Left-padding zeros are fine — the
    # attention mask is what actually disables padding positions.
    patches_batch = torch.zeros(B, max_T, PATCH_DIM, dtype=torch.float32)
    attn_masks = torch.zeros(B, max_T, max_T, dtype=torch.bool)
    targets_batch = torch.stack([s['targets'] for s in samples])           # (B, P, PATCH_SIZE)
    n_ctx_tensor = torch.tensor(n_contexts, dtype=torch.long)

    for i, s in enumerate(samples):
        n_ctx = n_contexts[i]
        seq_len = seq_lens[i]
        n_pad = max_T - seq_len

        # Left-pad: place actual data at positions [n_pad:max_T] so the
        # prediction horizon is at the right edge for every sample in the
        # batch.  The model relies on this convention.
        patches_batch[i, n_pad:, :] = s['patches']

        # Build the per-sample mask using the un-padded length, then embed
        # it into the padded mask at the same offset so context-vs-pred
        # boundaries line up with the data placement above.
        base_mask = create_attention_mask(n_ctx, PREDICTION_PATCHES)  # (seq_len, seq_len)

        attn_masks[i, n_pad:, n_pad:] = base_mask

    # Force the diagonal True for ALL positions, including padding.  An
    # all-False row would make softmax produce NaN (exp(-inf) / 0).
    # Self-attention on padding is harmless — these positions are never read by
    # the model (prediction positions can't attend to padding by construction).
    diag = torch.arange(max_T)
    attn_masks[:, diag, diag] = True

    # Cross-window (paired-window) time-of-day probe input, present only when the
    # samples carry it (probe on + cross-window weight > 0). Window k+1 shares
    # window k's n_ctx, so it reuses the SAME left-pad ``n_pad`` and the SAME
    # ``attn_mask`` — no separate mask is shipped.
    next_window_batched = None
    if 'next_window' in samples[0]:
        nw_patches = torch.zeros(B, max_T, PATCH_DIM, dtype=torch.float32)
        for i, s in enumerate(samples):
            n_pad = max_T - seq_lens[i]              # identical to window k's n_pad
            nw_patches[i, n_pad:, :] = s['next_window']['patches']
        next_window_batched = {
            'patches': nw_patches,                                                   # (B, max_T, PATCH_DIM)
            'last_bg': torch.tensor(
                [s['next_window']['last_bg'] for s in samples], dtype=torch.float32),  # (B,) mg/dL
            'pred_start_hour': torch.tensor(
                [s['next_window']['pred_start_hour'] for s in samples], dtype=torch.float32),  # (B,)
            'valid': torch.tensor(
                [s['next_window']['valid'] for s in samples], dtype=torch.bool),      # (B,)
        }

    # Pack the per-sample BG trajectories / scalars used by validation and
    # inference.  ``last_bg`` and ``pred_start_hour`` are per-sample scalars;
    # the trajectories are raw mg/dL ground truth.  The ``extended_carb/
    # insulin_*`` announced-future arrays are NOT stacked here — they are
    # consumed only from UN-COLLATED samples by the night-onset validation
    # override (``train._run_night_onset_validation`` iterates the dataset
    # directly), so they ride only on each sample's own ``bg_formula_data``.
    bg_formula_batched: dict[str, Any] = {
        'last_bg': torch.tensor(
            [s['bg_formula_data']['last_bg'] for s in samples], dtype=torch.float32),
        'true_bg_trajectory': torch.tensor(
            np.stack([s['bg_formula_data']['true_bg_trajectory'] for s in samples]),
            dtype=torch.float32,
        ),
        'extended_true_bg_trajectory': torch.tensor(
            np.stack([s['bg_formula_data']['extended_true_bg_trajectory'] for s in samples]),
            dtype=torch.float32,
        ),
        'pred_start_hour': torch.tensor(
            [s['bg_formula_data']['pred_start_hour'] for s in samples], dtype=torch.float32),
    }

    return {
        'patches': patches_batch,
        'targets': targets_batch,
        'attn_mask': attn_masks,
        'bg_formula_data': bg_formula_batched,
        'n_context_patches': n_ctx_tensor,
        **({'next_window': next_window_batched} if next_window_batched is not None else {}),
    }
