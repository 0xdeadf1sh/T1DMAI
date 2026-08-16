"""
T1DMAI Data Pipeline — on-the-fly dataset generation from T1DMSIM.
==================================================================

What this module produces
-------------------------
A ``T1DMDataset`` returns one training sample per ``__getitem__`` call.
A "sample" is everything the model needs for one optimization step's slice
of one batch:

* ``patches``           — (T, PATCH_DIM) float tensor.  T = n_ctx + PREDICTION_PATCHES.
                          Each patch is ``PATCH_SIZE × N_INPUT_FEATURES = PATCH_DIM``
                          values and is either VISIBLE or MASKED.  A masked patch
                          withholds bg (feat 0) and announces itself through the
                          bg_masked bit (feat 4, written into all PATCH_SIZE of its
                          step-major columns); carb (feat 1) / insulin (feat 2) /
                          exercise (feat 3) keep their true or announced values
                          everywhere, unless the dataset is ``blind`` (below).
* ``targets``           — (MAX_MASKED_PATCHES, PATCH_SIZE) float tensor of
                          ground-truth BG (mg/dL), one row per head slot.  This is
                          the RAW ``bg_observed`` (the SAME raw signal fed as the
                          model input — no smoothing).  NOT f-transformed in the
                          batch; the Kovatchev risk transform is applied exactly
                          once at the top of the loss.
* ``n_context_patches`` — int, length of the variable-size context window.
* ``bg_formula_data``   — dict of per-sample arrays/scalars consumed by the loss,
                          validation and inference paths.  The masked set, all
                          (MAX_MASKED_PATCHES,): ``mask_idx`` (patch index per head
                          slot), ``valid``, ``anchor_bg`` (mg/dL), ``d`` (patches to
                          the nearest visible evidence on either side) and
                          ``slot_hour`` (true hour of day).  Plus ``last_bg`` (the
                          raw last-context BG, mg/dL), ``true_bg_trajectory`` and
                          ``extended_true_bg_trajectory`` (mg/dL ground truth from
                          the context edge over the horizon and the long horizon),
                          ``pred_start_hour``, and the announced future
                          ``extended_carb_*`` / ``extended_insulin_*`` /
                          ``extended_exercise_*`` (normalized + raw) used by the
                          conditioned rolling override and the counterfactual
                          probes.

One raw post-noise space (no smoothing)
---------------------------------------
Every signal channel — bg, carb, insulin AND exercise — is fed to the model RAW:
there is no causal smoother on inputs or on the forecast target.  The SAME raw bg is
the model input, the forecast TARGET and ``last_bg`` (bg only clamped to the physical
[BG_CLAMP_MIN, BG_CLAMP_MAX] range; carb/insulin/exercise floored at 0).  There is no
input/target asymmetry: the model lives entirely in this raw post-noise space
(inputs, target, loss, metrics).  Deployment realism is intrinsic — the live
CGM/dose stream is consumed exactly as-is, and the autoregressive roll re-feeds the
model's own raw output, so the train and inference input distributions match.

The masked-BG objective
-----------------------
There is no prediction zone.  ``sample_mask_spans`` draws 1-3 non-abutting
spans anywhere in the window and the model emits a quantile fan for every
masked patch: a span ending at patch T-1 is a FORECAST, one starting at patch 0
a BACKCAST, anything else INFILL.  The old trailing forecast is the right-edge
case of the general objective, not a separate mode.

Conventions and gotchas
-----------------------
* Context windows are sampled uniformly in [MIN_CONTEXT_PATCHES,
  MAX_CONTEXT_PATCHES]; when MIN == MAX the context is fixed-length. The
  collate function left-pads shorter sequences to the BATCH maximum.
* The input feature stack is exactly [bg_absolute, carb, insulin,
  exercise_equiv, bg_masked]; there are no temporal sin/cos features. bg
  (feat 0) enters in Kovatchev risk space — z(f(bg)) — while carb/insulin/
  exercise keep log1p+z.  Exercise is the simulator's carbohydrate-EQUIVALENT
  glucose-disposal curve in g/step, fed at that scale, never rescaled to an
  intensity and never risk-transformed.  bg_masked (feat 4) is a BIT and is
  never normalized, so the feature count and the normalized-channel count are
  no longer the same number.
* carb (feat 1), insulin (feat 2) and exercise (feat 3) carry their true or
  announced values, masked patches included; bg (feat 0) is zeroed exactly at
  the masked patches, so the model cannot copy the signal it is asked to emit.
  Feats 1-3 are PLAN channels: nothing but what the patient announced is ever
  written into them.
* ``blind=True`` is the ONE departure, and it is off by default.  It withholds
  feats 1-3 on masked patches too, filling them with the per-channel zero-RAW
  ``normalize(0)`` (``zero_dose_fill``), which is the same "no dose" baseline
  ``inference.predict_rolling`` writes when nothing is announced.  It exists so
  ``train_blind.py`` can measure the model with no conditioning at all; a model
  only ever sees one convention, so feat 4 still announces the withholding and
  there is no second bit.
* Masking is NOT inferable from position, and z = 0 in a withheld bg slot
  decodes to an ordinary reading (~142 mg/dL on the balanced pool), not a
  sentinel — which is why feat 4 announces the masked set explicitly.

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
    CHANNEL_TO_FEAT, NON_MASKABLE_FEATS, MASKABLE_FEATS,
    MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES, PREDICTION_PATCHES,
    MASK_MAX_SPANS, MASK_RIGHT_EDGE_QUOTA, MASK_SPAN_LENGTHS, MAX_MASKED_PATCHES,
    PATIENT_UNIFORM_SAMPLE_PROB,
    SIMULATOR_WARMUP_HOURS, NIGHT_LONG_HORIZON_PATCHES, CACHE_MADVISE_DONTNEED,
    TIME_PROBE_ENABLED, TIME_PROBE_CROSS_WINDOW_WEIGHT,
)
import utils
from utils import compute_patient_seed, kovatchev_f_np
from normalization import (
    CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS, normalize,
)

# === The bg_masked announcement column ===
# The input stack is [*CHANNEL_NAMES, bg_masked].  The leading
# ``len(CHANNEL_NAMES)`` columns are normalized signal channels; the trailing
# column is a per-PATCH 0/1 bit and is never normalized, so the two counts are
# no longer the same number and must not be asserted equal.  The index is
# derived here rather than restated: the bit is whatever column follows the
# channels.
BG_MASKED_FEAT = len(CHANNEL_NAMES)
assert N_INPUT_FEATURES == len(CHANNEL_NAMES) + 1, (
    f"N_INPUT_FEATURES={N_INPUT_FEATURES} should be len(CHANNEL_NAMES)="
    f"{len(CHANNEL_NAMES)} normalized channels plus the bg_masked bit"
)

# === What a masked patch withholds ===
# Two conventions, and a model only ever sees one of them.  Under ``announced`` a
# masked patch withholds bg alone and the announced carb / insulin / exercise
# ride through it; under ``blind`` it withholds those three as well.  The string
# is the checkpoint's provenance — no parameter shape depends on it, so a strict
# state-dict load accepts weights trained under either — and
# ``finetune/finetune.py`` compares it against the policy it implements.  An
# ABSENT key means ``announced``: the key was introduced with the blind trainer,
# so every checkpoint written before it was conditioned.
MASKED_CHANNEL_POLICY_ANNOUNCED = 'announced'
MASKED_CHANNEL_POLICY_BLIND = 'blind'


def masked_channel_policy(blind: bool) -> str:
    """The masked-channel policy name a ``blind`` flag selects.

    Args:
        blind: True when masked patches withhold the dose channels too.

    Returns:
        ``MASKED_CHANNEL_POLICY_BLIND`` or ``MASKED_CHANNEL_POLICY_ANNOUNCED``.
    """
    return MASKED_CHANNEL_POLICY_BLIND if blind else MASKED_CHANNEL_POLICY_ANNOUNCED


def stored_masked_channel_policy(training_config: dict[str, Any] | None) -> str:
    """The masked-channel policy a checkpoint's ``training_config`` records.

    The ONE reader of the absent-key convention.  An absent key reads as
    ``announced`` unconditionally — not as "unknown": the key was introduced WITH
    the blind trainer and a blind run always stamps it, so no checkpoint that
    lacks it can be a blind one.  Every consumer — ``finetune/finetune.py``'s
    policy guard, ``gui.py``'s load, ``make_card.py``, ``model_health.py`` — goes
    through here, so a second convention cannot appear.

    Args:
        training_config: a checkpoint's ``training_config``, or None.

    Returns:
        ``MASKED_CHANNEL_POLICY_BLIND`` or ``MASKED_CHANNEL_POLICY_ANNOUNCED``.
    """
    tc = training_config or {}
    return str(tc.get('masked_channel_policy', masked_channel_policy(blind=False)))


def checkpoint_masked_channel_policy(ckpt: dict[str, Any] | None) -> str:
    """``stored_masked_channel_policy`` over a whole checkpoint dict.

    Args:
        ckpt: a loaded checkpoint, or None.

    Returns:
        ``MASKED_CHANNEL_POLICY_BLIND`` or ``MASKED_CHANNEL_POLICY_ANNOUNCED``.
    """
    return stored_masked_channel_policy((ckpt or {}).get('training_config'))


def zero_dose_fill(stats: dict[str, dict[str, float]]) -> dict[int, float]:
    """Per-feat ``normalize(0)`` for every announceable channel, in z-space.

    What a blind masked patch carries in feats 1-3.  NOT ``z = 0``: the sparse
    channels are log1p'd before the z-score, so ``z = 0`` inverts to
    ``expm1(mean)`` — a phantom dose of ~0.47 g/step of carb on the balanced
    pool, not an absence of one.  ``normalize(0)`` is the channel's ``-mean/std``
    and means "no dose"; it is also the baseline ``inference.predict_rolling``
    writes into an un-overridden slot, so a blind roll is in-distribution with no
    change to the inference path.

    Derived from the loaded stats every time rather than written down: the values
    are properties of the pool the stats were fit on.

    Args:
        stats: the run's normalization statistics.

    Returns:
        ``{feat_idx: z}`` over ``MASKABLE_FEATS``.
    """
    zero_raw = normalize(
        np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), stats,
    )[0]                                              # (n_channels,) z-space
    return {feat: float(zero_raw[feat]) for feat in MASKABLE_FEATS}


def blind_masked_doses(
    patches: torch.Tensor,
    masked: torch.Tensor,
    fill: dict[int, float],
) -> None:
    """Withhold the dose channels of every masked patch, IN PLACE.

    The one place the blind convention is implemented.  ``data._build_sample``
    applies it to the sample's own masked set and to ``next_window``'s;
    ``train_blind.py``'s two protocol forwards apply it to the sets they mask, so
    validation scores the task the model was trained on.  Feat 0 and feat 4 are
    untouched — the bg withholding and the announcement bit are the same under
    either policy.

    Args:
        patches: ``(..., T, PATCH_DIM)`` step-major patch rows, modified in place.
        masked: ``(..., T)`` bool, True at the patches that withhold.
        fill: ``zero_dose_fill``'s ``{feat_idx: z}``.
    """
    assert patches.shape[-1] == PATCH_DIM, (
        f"patch row width {patches.shape[-1]} != PATCH_DIM {PATCH_DIM}")
    assert masked.shape == patches.shape[:-1], (
        f"masked {tuple(masked.shape)} does not index patches "
        f"{tuple(patches.shape)}")
    withheld = masked.unsqueeze(-1)
    for feat_idx, z in fill.items():
        block = patches[..., feat_idx::N_INPUT_FEATURES]
        patches[..., feat_idx::N_INPUT_FEATURES] = block.masked_fill(withheld, z)

# === On-the-fly simulator request size ===
# Each training sample needs at most
#   MAX_CONTEXT_PATCHES + max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)
#   patches × PATCH_SIZE timesteps/patch × DT_MINUTES = 5 min/timestep
# of raw trajectory.  With MAX_CONTEXT_PATCHES context patches (24 h) plus the
# NIGHT_LONG_HORIZON_PATCHES (8 h) long-horizon window that is 32 h minimum.
#
# The post-warmup trajectory always starts at midnight (warmup discards an
# integer number of hours, so warmup_hours % 24 == 0 → trajectory step 0 is
# midnight).  The pred_start_step jitter is therefore *also* an hour-of-day
# range.  ``_pick_pred_start_step`` picks uniformly among the patch-aligned
# starts, of which there are
#   n_candidates = N / PATCH_SIZE
#                  - max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)
#                  - n_ctx + 1
# and it returns None once that falls below 1.  The max() is the operative
# term: it equals NIGHT_LONG_HORIZON_PATCHES only because 16 > 4 today, and
# becomes PREDICTION_PATCHES as soon as the supervised horizon outgrows the
# nocturnal one.  Candidates sit at 30-min spacing (PATCH_SIZE × DT_MINUTES),
# so a day holds 48 slots and hour-of-day coverage is exactly uniform at a
# fixed n_ctx iff n_candidates % 48 == 0.
#
# At N = 1242 steps, n_candidates = 192 - n_ctx, which is exact at
# n_ctx ∈ {48, 96, 144} — the 24 h, 48 h and 72 h context widths — and nowhere
# else.  Training draws n_ctx uniformly (``_build_sample``), so the pooled
# hour-of-day histogram is a mixture over widths and is never flat.  Exactly
# enumerated, the most-drawn 30-min slot over the least is 1.3221 over
# n_ctx 16-48, 1.4158 with the ceiling at 96, and 1.5972 at 144.  The peak is
# always slot 47 (23:30) and the trough slot 0, so the residual tilt runs away
# from midnight rather than piling onto it.  This length buys exactness at the
# deployed context widths and a flatter mixture everywhere else.
#
# For whole-day context widths one length condition remains, N ≡ 90 (mod 288),
# and 1242 steps = 103.5 h is the smallest such length that leaves a 72 h
# context any candidate at all.  Any other length puts an extra candidate on
# slot 0.
ON_THE_FLY_SIM_HOURS: float = 103.5

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

    Reached through ``force_pred_start_hour``, for an evaluation that needs its
    origins pinned to one hour-of-day — a bedtime origin
    (``NOCTURNAL_START_HOUR`` ≈ 22:00), say, so a rolled forecast spans the whole
    night. Among candidates within ``tol_hours`` of the target (one per
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
        blind: withhold the dose channels on masked patches too, filling feats
            1-3 with ``zero_dose_fill``.  Default False — the announced policy,
            byte-identical to what this dataset produced before the flag existed.
            ``train_blind.py`` sets it; nothing else does.
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
        blind: bool = False,
    ) -> None:
        self.master_seed = master_seed
        self.blind = blind
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.stats = normalization_stats
        self.seed_offset = seed_offset
        # When set, the prediction origin is forced to the patch-aligned step
        # nearest this hour-of-day instead of a uniform-random origin, for an
        # evaluation that needs its origins pinned. Validation-only.
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
        # open cache handles aren't pickled across the DataLoader fork boundary.
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

            # With pool_size < total_steps * batch_size the pool cycles, and
            # that is benign rather than a shortfall to warn about:
            # ``_build_sample`` takes a fresh random context+horizon window per
            # draw, and one cached trajectory admits many distinct
            # patch-aligned windows, so every reuse is a DIFFERENT training
            # window. The on-the-fly path has effectively-zero collision over
            # the 64-bit seed space; cache mode trades that for disk locality.

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

        Only the npy format is mapped. Workers share the kernel page cache
        either way, and ``arr[i:i+1]`` returns a fresh ndarray row, so
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
                    # Deliberately NOT mmap_mode='r'. A mapped .b2nd faults each
                    # touched chunk's compressed pages into this process and
                    # nothing can drop them again — blosc2 exposes no mapping to
                    # madvise, so the npy path's MADV_DONTNEED trick has no
                    # counterpart here. Random access over the pool then grows
                    # RssFile ~100 KB per channel per row (~400 MB/step at
                    # BATCH_SIZE=512) until the whole cache is resident. Plain
                    # file reads leave the pages in ordinary page cache, which is
                    # charged to nobody and reclaimed under pressure.
                    arr = blosc2.open(
                        os.path.join(self.cache_path, f'{name}.b2nd'),
                        mode='r',
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
        blosc2 (not mapped at all, precisely because there would be no mapping
        left to advise) or ``MADV_DONTNEED`` is unavailable, and any per-call
        failure is swallowed so a platform quirk can never break data loading.

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
            blind=self.blind,
        )


def make_calibration_dataset(
    master_seed: int,
    n_patients: int,
    batch_size: int,
    normalization_stats: dict[str, dict[str, float]],
    cache_path: str | None = None,
    blind: bool = False,
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
        blind: the SAME masked-channel policy the run trained under — split
            conformal is only valid on the distribution the model runs under, so
            a blind checkpoint must be calibrated on blind windows.

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
        blind=blind,
    )


# ---------------------------------------------------------------------------
# Masked-BG span sampler.
# ---------------------------------------------------------------------------
def sample_mask_spans(
    seq_len: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Draw the masked spans of one sample, left to right.

    A masked patch withholds bg (feat 0) — and, under ``blind``, the dose
    channels with it — and the model emits a quantile fan for it.  A span ending
    at patch ``seq_len - 1`` is a forecast, one starting at patch 0 a backcast,
    anything else infill — one objective, three cases.  Placement is the same
    draw under either policy: nothing here reads the flag.

    Procedure::

        n_spans      ~ U{1 .. MASK_MAX_SPANS}
        each L_i     ~ U(MASK_SPAN_LENGTHS), independently
        if sum(L) > MAX_MASKED_PATCHES: resample the WHOLE length vector
        with prob MASK_RIGHT_EDGE_QUOTA: last span flush right, rest over the
                                         prefix by stars-and-bars
        otherwise:                       stars-and-bars over n_spans + 1 gaps

    The over-budget rejection redraws the whole vector, never one element:
    per-element redrawing yields a different length distribution and hence a
    different ``d`` histogram.  Placement is a uniform composition of the slack
    over the gaps, with a MANDATORY visible patch between neighbouring spans —
    there is no rejection loop on placement, and no curriculum or annealing.

    ``MASK_RIGHT_EDGE_QUOTA`` is the one departure from uniform placement, and it
    changes ONLY where the last span lands: ``n_spans`` and the length law are
    drawn identically in both branches, so the span-length histogram is
    quota-independent and only the ``d`` histogram moves.  Under uniform
    placement a FORECAST — the case the model is deployed as — is an accident
    worth ~3% of windows, and the band it emits there decays with training while
    every selection scalar improves.  ``config.MASK_RIGHT_EDGE_QUOTA`` carries
    the paired evidence for the value.

    Both branches are enumerated exactly in ``d_balance.d_distribution``; a change
    here that is not mirrored there silently moves every ``d``-binned figure's
    reference.

    Two masked spans never abut.  The separator is what makes the anchor, the
    per-span median basis and the DILATE length bucket well defined per span;
    two spans with nothing between them are one longer span.

    Args:
        seq_len: T, the number of patches in the sample's window.
        rng: numpy Generator.

    Returns:
        List of ``(start_patch, length)`` pairs, strictly increasing in
        ``start_patch``, with at least one visible patch between consecutive
        spans and ``sum(length) <= MAX_MASKED_PATCHES``.
    """
    lengths_pool = np.asarray(MASK_SPAN_LENGTHS, dtype=np.int64)
    n_spans = int(rng.integers(1, MASK_MAX_SPANS + 1))

    # Rejection is on the LENGTH VECTOR as a whole.
    while True:
        span_lengths = rng.choice(lengths_pool, size=n_spans, replace=True)
        if int(span_lengths.sum()) <= MAX_MASKED_PATCHES:
            break

    total_masked = int(span_lengths.sum())
    # One mandatory visible patch per interior boundary is charged up front, so
    # the slack is what remains to spread freely over the n_spans + 1 gaps.
    slack = seq_len - total_masked - (n_spans - 1)
    if slack < 0:
        raise RuntimeError(
            f"window of {seq_len} patches cannot hold {n_spans} spans of "
            f"total length {total_masked} with separators"
        )

    def _compose(slack_: int, n_gaps_: int) -> np.ndarray:
        """Stars and bars: ``slack_`` indistinguishable stars into ``n_gaps_``
        ordered bins, uniform over compositions.  Choosing the ``n_gaps_ - 1``
        bar positions out of ``slack_ + n_gaps_ - 1`` slots without replacement
        is exactly that uniform draw."""
        if n_gaps_ == 1:
            return np.array([slack_], dtype=np.int64)
        bars_ = np.sort(rng.choice(slack_ + n_gaps_ - 1, size=n_gaps_ - 1,
                                   replace=False))
        g = np.empty(n_gaps_, dtype=np.int64)
        g[0] = bars_[0]
        g[1:-1] = bars_[1:] - bars_[:-1] - 1
        g[-1] = slack_ + n_gaps_ - 2 - bars_[-1]
        return g

    spans: list[tuple[int, int]] = []
    # The right-edge branch reserves the LAST span at the window's final patch and
    # composes the rest over the prefix that stays clear of it and its separator.
    # That prefix has exactly ``slack`` free patches — pinning the last span is the
    # uniform arrangement with its trailing gap fixed at 0, and fixing a gap frees
    # the separator it no longer needs — so the branch is feasible whenever the
    # window holds the spans at all, and needs no feasibility test of its own.
    last = int(span_lengths[-1])
    right_edge = MASK_RIGHT_EDGE_QUOTA > 0.0 and rng.random() < MASK_RIGHT_EDGE_QUOTA

    if right_edge and n_spans == 1:
        spans.append((seq_len - last, last))
    elif right_edge:
        gaps = _compose(slack, n_spans)          # n_spans-1 spans -> n_spans gaps
        pos = 0
        for i in range(n_spans - 1):
            pos += int(gaps[i])
            spans.append((pos, int(span_lengths[i])))
            pos += int(span_lengths[i]) + 1
        spans.append((seq_len - last, last))
    else:
        gaps = _compose(slack, n_spans + 1)
        pos = 0
        for i in range(n_spans):
            pos += int(gaps[i])
            spans.append((pos, int(span_lengths[i])))
            pos += int(span_lengths[i])
            if i < n_spans - 1:
                pos += 1                  # the mandatory visible separator
        assert pos + int(gaps[-1]) == seq_len

    assert all(
        spans[i][0] > spans[i - 1][0] + spans[i - 1][1] for i in range(1, n_spans)
    ), f"abutting masked spans {spans} at seq_len={seq_len}"
    assert spans[0][0] >= 0 and spans[-1][0] + spans[-1][1] <= seq_len, (
        f"masked spans {spans} fall outside a {seq_len}-patch window")
    return spans


def _anchor_step_for_span(start_patch: int, length: int) -> int:
    """
    Step index (window-relative) of the anchor of one masked span.

    ONE-SIDED, LEFT-PREFERRING: every slot of the span takes the LAST step of
    the left neighbour patch, or — when the span starts at patch 0 and there is
    no left neighbour — the FIRST step of the right neighbour patch.  Every slot
    of a contiguous span therefore gets the SAME value.

    This is the single anchor rule.  A span starting at patch 0 is the only
    no-left-neighbour case there is, so it also covers what used to be a
    separate ``pred_start_step == 0`` fallback on ``last_bg``.

    Read at the raw mg/dL array on the training path: with the span at the right
    edge, ``start_patch = n_ctx`` gives ``bg_window[n_ctx * PATCH_SIZE - 1]``,
    byte-identical to the old ``bg[pred_start_step - 1]``.
    """
    if start_patch > 0:
        return start_patch * PATCH_SIZE - 1
    return (start_patch + length) * PATCH_SIZE


def _mask_slots(
    spans: list[tuple[int, int]],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand spans into the head's fixed ``MAX_MASKED_PATCHES`` slots.

    Returns ``(mask_idx, valid, d, anchor_step)``, each of length
    ``MAX_MASKED_PATCHES`` (= M).  Padded slots gather patch 0 and are discarded
    by ``valid``.

    ``d`` is the distance in patches to the nearest visible evidence on EITHER
    side — never the span length, which confounds one-sided and two-sided cases
    at equal difficulty, and never the arm.

    ``d`` and the anchor disagree by construction, and any metric binned on
    ``d`` is reporting a distance the anchor did not use.  THE ANCHOR IGNORES
    THE NEAR SIDE: it is one-sided and left-preferring, so the last slot of a
    two-sided 4-patch span is at ``d = 1`` off its right neighbour while
    anchoring 4 patches to the left.  A third of supervision anchors farther
    than the nearest visible evidence; ``tests/test_mask_sampler.py`` enumerates
    the share exactly over ``(T, n_spans, length vector, placement branch, gap
    composition)`` and prints it, rather than anyone writing it down.
    That costs no information — masked rows attend to everything — it only
    makes the head's offset parameterisation work harder.
    """
    M = MAX_MASKED_PATCHES
    mask_idx = np.zeros(M, dtype=np.int64)
    valid = np.zeros(M, dtype=bool)
    d = np.zeros(M, dtype=np.int64)
    anchor_step = np.zeros(M, dtype=np.int64)

    slot = 0
    for start_patch, length in spans:
        has_left = start_patch > 0
        has_right = start_patch + length < seq_len
        step = _anchor_step_for_span(start_patch, length)
        for j in range(length):
            mask_idx[slot] = start_patch + j
            valid[slot] = True
            anchor_step[slot] = step
            left = j + 1 if has_left else length + seq_len
            right = length - j if has_right else length + seq_len
            d[slot] = min(left, right)
            slot += 1
    assert slot <= M, f"{slot} masked patches exceeds MAX_MASKED_PATCHES={M}"
    return mask_idx, valid, d, anchor_step


# ---------------------------------------------------------------------------
# Shared post-simulator sample-building logic.
# ---------------------------------------------------------------------------
def _build_sample(
    data: dict[str, np.ndarray],
    icr: float,
    stats: dict[str, dict[str, float]],
    rng: np.random.Generator,
    force_pred_start_hour: float | None = None,
    blind: bool = False,
) -> dict[str, Any]:
    """
    Convert a raw simulator output dict into one training sample.

    Everything after the simulator call lives here so callers reuse the same
    feature pipeline.  The prediction window can start at any patch-aligned
    position in the trajectory — a single model is trained on both day-time
    and night-time windows simultaneously.

    One raw post-noise space (no smoothing): every signal channel (bg / carb /
    insulin / exercise) AND the BG target / ``last_bg`` are the RAW
    ``bg_observed`` / ``total_carb`` / ``total_insulin`` / ``total_exercise``
    (bg clamped to the physical range, the sparse channels floored at 0).  There
    is no input/target asymmetry and no filter.

    Args:
        data: Dict with keys ``bg_observed``, ``total_carb``, ``total_insulin``,
              ``total_exercise``, ``hour_of_day``, ``day`` (plus the
              unused-by-the-input-stack ``insulin_resistance``, ``hgo``) — each a
              1-D array of length N.  The model is fed the RAW ``bg_observed`` /
              ``total_carb`` / ``total_insulin`` / ``total_exercise``; the same
              raw bg is the target and last_bg.
        icr: Patient's insulin-to-carb ratio (accepted for caller compatibility;
              not consumed — no physics reconstruction in the redesign).
        stats: Normalization statistics from ``load_normalization_stats``.
        rng: numpy Generator used for window selection.
        force_pred_start_hour: force the origin near this hour-of-day.
        blind: withhold the dose channels on masked patches too (``blind_masked_doses``).

    Returns:
        Dict with keys ``patches``, ``targets``, ``n_context_patches``,
        ``bg_formula_data``.
    """
    # Cast every consumed channel to float32 up front.  ``bg_observed`` is the
    # post-CGM-noise read; ``total_carb`` / ``total_insulin`` carry the simulator's
    # per-step AR(1) absorption noise.  ``total_exercise`` is the carbohydrate-
    # EQUIVALENT glucose-disposal curve in g/step (the simulator subtracts it from
    # the appearance term), so it is fed at its trained g/step scale and never
    # rescaled to an intensity.  There is NO smoothing: the RAW post-noise
    # signals are fed to the model AND used as the forecast target, so the whole
    # pipeline lives in one raw post-noise space.
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    bg_raw = data['bg_observed'].astype(np.float32)
    carb_raw = data['total_carb'].astype(np.float32)
    insulin_raw = data['total_insulin'].astype(np.float32)
    exercise_raw = data['total_exercise'].astype(np.float32)
    hour_of_day = data['hour_of_day'].astype(np.float32)
    day_index = data['day'].astype(np.int32)

    N = len(bg_raw)

    # No smoother. bg is only clamped to the physical BG range so it is a legal
    # Kovatchev-f / last_bg argument (the cache is already rail-filtered into the
    # open interval BG_CLAMP_MIN + 1 … BG_CLAMP_MAX - 1, which T1DMSIM derives
    # the same way at cache_simulator.py:180-181 so a clamp change cannot
    # desynchronise the two; on-the-fly generation and edge reads still need the
    # guard); the sparse carb/insulin/exercise channels are floored at 0. ``bg``
    # is BOTH the model input bg and the forecast target source / last_bg — one
    # raw signal.
    bg = np.clip(bg_raw, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.maximum(carb_raw, 0.0).astype(np.float32)
    insulin = np.maximum(insulin_raw, 0.0).astype(np.float32)
    exercise = np.maximum(exercise_raw, 0.0).astype(np.float32)

    # Stack into the canonical N_INPUT_FEATURES-feature input layout
    # [bg_absolute, carb, insulin, exercise, bg_masked]:
    #   0 bg_absolute, 1 carb, 2 insulin, 3 exercise (g/step carb-equivalent),
    #   4 bg_masked (per-PATCH announcement bit, written per window below).
    # The four signal channels are the RAW (clamped/floored) post-noise values.
    # hour_of_day is retained above for prediction-start selection / nocturnal
    # metadata; neither it nor day_index is a model input feature any longer.
    features = np.stack([
        bg, carb, insulin, exercise,
        np.zeros_like(bg),
    ], axis=-1)  # (N, N_INPUT_FEATURES)
    # Two separate facts, and they are no longer the same number: the feature
    # array carries N_INPUT_FEATURES columns, of which only the LEADING
    # len(CHANNEL_NAMES) are normalized signal channels.  The trailing
    # bg_masked column is a bit, not a signal, and never sees the z-score.
    assert features.shape[-1] == N_INPUT_FEATURES, (
        f"feature stack has {features.shape[-1]} cols, expected "
        f"N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )
    assert len(CHANNEL_NAMES) == BG_MASKED_FEAT < N_INPUT_FEATURES, (
        f"normalized channels must occupy columns 0..{BG_MASKED_FEAT - 1} with "
        f"bg_masked at {BG_MASKED_FEAT}; got len(CHANNEL_NAMES)="
        f"{len(CHANNEL_NAMES)}, N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )

    # Normalize every signal channel in place.  bg (feat 0) enters in Kovatchev
    # risk space — z(f(bg)) — the sole BG input path; the sparse carb / insulin /
    # exercise channels get log1p first so heavy-tailed event spikes don't distort
    # the z-score.  Exercise is a carb-equivalent disposal rate, not a glucose, so
    # it takes the log1p branch and never the risk transform.  Column order ==
    # CHANNEL_NAMES order, so the gather never reads a dropped channel.
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
        # Pin the origin to ~one hour-of-day — a bedtime origin, say, so a rolled
        # forecast spans the whole night. Falls back to a uniform-random origin
        # if no candidate near the target hour leaves enough room.
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
    # Normalized carb/insulin/exercise windows — used to build the announced
    # future-input overrides for the rolling validation and the counterfactual
    # probes (carb=feat 1, insulin=feat 2, exercise=feat 3 in the normalized
    # stack). ``carb`` / ``insulin`` / ``exercise`` here are the RAW (floored)
    # per-step signals, so the ``*_raw_window`` names are now literal.
    # Both consumers (predict_rolling discards the raw side; the counterfactual probe
    # re-normalizes through the same log1p-z stats on baseline and perturbed sides)
    # are self-consistent with the signal. Sliced to the long horizon below.
    _carb_feat = CHANNEL_TO_FEAT[0]
    _insulin_feat = CHANNEL_TO_FEAT[1]
    _exercise_feat = CHANNEL_TO_FEAT[2]
    carb_norm_window = features[start_step:end_step, _carb_feat]
    insulin_norm_window = features[start_step:end_step, _insulin_feat]
    exercise_norm_window = features[start_step:end_step, _exercise_feat]
    carb_raw_window = carb[start_step:end_step]
    insulin_raw_window = insulin[start_step:end_step]
    exercise_raw_window = exercise[start_step:end_step]

    # Reshape into (n_patches, PATCH_SIZE, N_INPUT_FEATURES).  Leading-axis
    # slices of a C-contiguous array are themselves contiguous, so the reshape
    # below is a view and the sole materializing ``.copy()`` happens at the
    # ``torch.from_numpy`` boundary — no redundant intermediate copy here.
    patches_3d = window.reshape(total_patches_needed, PATCH_SIZE, N_INPUT_FEATURES)

    # === Masked-BG objective ===
    # The window the model sees is seq_len patches long; each is visible or
    # masked.  Only the first PREDICTION_PATCHES patches past the context are
    # exposed — patches beyond that exist in ``window`` solely so the extended
    # BG trajectory carries the long-horizon ground truth.
    seq_len = n_ctx + PREDICTION_PATCHES
    spans = sample_mask_spans(seq_len, rng)
    masked_patches = np.concatenate(
        [np.arange(s, s + L, dtype=np.int64) for s, L in spans]
    )
    mask_idx, valid, mask_d, anchor_step = _mask_slots(spans, seq_len)

    # Flatten (PATCH_SIZE, N_INPUT_FEATURES) → PATCH_DIM.  The row layout is
    # STEP-MAJOR, so a feature's columns are the ``f::N_INPUT_FEATURES`` stride.
    all_patches_t = torch.from_numpy(
        patches_3d[:seq_len].reshape(seq_len, PATCH_SIZE * N_INPUT_FEATURES).copy()
    )
    masked_rows = torch.from_numpy(masked_patches)
    # A masked patch withholds bg (feat 0, the only NON_MASKABLE_FEATS entry):
    # that is what the model is asked to emit.  carb (feat 1), insulin (feat 2)
    # and exercise (feat 3) keep their true or announced values everywhere — the
    # model is always conditioned on the announced plan.  The loop touches
    # NON_MASKABLE_FEATS only, so every MASKABLE_FEATS column passes through
    # untouched whatever N_INPUT_FEATURES is.
    for feat_idx in NON_MASKABLE_FEATS:
        all_patches_t[masked_rows, feat_idx::N_INPUT_FEATURES] = 0.0
    # Under the blind policy the same patches withhold their doses as well, at
    # the zero-RAW "no dose" fill rather than at z = 0.  Off by default, and the
    # loop above is what the default path runs — feats 1-3 pass through untouched.
    blind_fill = zero_dose_fill(stats) if blind else None
    unblinded_dose_rows = None
    unblinded_dose_patches = None
    if blind_fill is not None:
        blind_flags = torch.zeros(seq_len, dtype=torch.bool)
        blind_flags[masked_rows] = True
        # Keep what the patient ACTUALLY did on the withheld patches, before the
        # fill overwrites it in place.  The blind policy is a property of the
        # OBJECTIVE — of which patches the model may read — not of the history,
        # and a caller that un-masks has to be able to un-blind with it.  The
        # long-horizon roll is that caller: its context is observed CGM history,
        # so restoring bg while leaving the fill in feats 1-3 hands the roll a
        # history asserting that a meal and a bolus did not happen.  Saved for
        # the masked rows only (at most MAX_MASKED_PATCHES of them) rather than
        # for the whole window.
        unblinded_dose_rows = np.asarray(masked_rows, dtype=np.int64).copy()
        unblinded_dose_patches = all_patches_t[masked_rows].clone()
        blind_masked_doses(all_patches_t, blind_flags, blind_fill)
    # Announce the masked set explicitly.  Masking is not inferable from
    # position any more, and z = 0 in a withheld bg slot decodes to an ordinary
    # reading (~142 mg/dL on the balanced pool), not a sentinel.  The bit is per
    # PATCH and the layout step-major, so it is written into ALL PATCH_SIZE
    # columns of feat 4 — that is why PATCH_DIM grows by PATCH_SIZE to carry one
    # bit.  Putting it anywhere outside the step-major block would break
    # PATCH_DIM = PATCH_SIZE * N_INPUT_FEATURES and the stride idiom above.
    all_patches_t[masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0
    assert all_patches_t.shape[-1] == PATCH_DIM, (
        f"patch row width {all_patches_t.shape[-1]} != PATCH_DIM {PATCH_DIM}"
    )

    # === BG targets, one per head slot ===
    # RAW BG (mg/dL) at each masked patch, NOT f-transformed in the batch (the
    # risk transform is applied once in the loss).  Padded slots gather patch 0
    # exactly as ``mask_idx`` does; ``valid`` is what discards them.
    pred_start_in_window = n_ctx * PATCH_SIZE
    bg_patches = bg_window[:seq_len * PATCH_SIZE].reshape(seq_len, PATCH_SIZE)
    targets_t = torch.from_numpy(bg_patches[mask_idx].copy())   # (M, S) mg/dL

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
    # === Anchors ===
    # ONE-SIDED, LEFT-PREFERRING, read off the RAW mg/dL array — no decode round
    # trip.  ``_anchor_step_for_span`` is the single rule: last step of the left
    # neighbour, or first step of the right neighbour when the span starts at
    # patch 0.  "Span starts at patch 0" is the only no-left-neighbour case
    # there is, which is why the old ``pred_start_step == 0`` fallback on
    # ``last_bg`` is gone rather than sitting beside it as a second rule.
    #
    # Padded slots hold ``last_bg`` — an arbitrary but LEGAL mg/dL from this
    # window, so the forward's (B, M) units tripwire never fires on a slot
    # ``valid`` is about to discard.
    #
    # ``last_bg`` itself is the right-edge case of the same rule: the span of
    # PREDICTION_PATCHES starting at patch n_ctx anchors on
    # ``bg_window[n_ctx * PATCH_SIZE - 1]``, byte-identical to the old
    # ``bg[pred_start_step - 1]``.  It stays because the rolling validation and
    # the inference paths still forecast from the context edge and read it.
    last_bg = float(bg_window[_anchor_step_for_span(n_ctx, PREDICTION_PATCHES)])
    anchor_bg = np.full(MAX_MASKED_PATCHES, last_bg, dtype=np.float32)
    anchor_bg[valid] = bg_window[anchor_step[valid]]

    # Per-slot TRUE hour of day, at the masked patch's own first step.  The time
    # probe's target is a per-slot clock: derived instead as
    # ``pred_start_hour + 0.5 * j`` it is off by
    # ``(mask_idx[j] - n_ctx - j) * 0.5`` h under the general masked set, with
    # every shape still matching.  A right-edge span reproduces
    # ``pred_start_hour + 0.5 * j`` exactly.
    slot_hour = hour_of_day[start_step + mask_idx * PATCH_SIZE].astype(np.float32)

    # Announced future carbs/insulin/exercise over the long horizon (normalized +
    # raw), for the conditioned rolled-forecast override.  Exercise is a PLAN
    # channel exactly like the doses: what rides here is what the patient
    # announced, never an inferred session.
    _lh = slice(pred_start_in_window, pred_start_in_window + n_long_horizon_steps)
    extended_carb_norm = carb_norm_window[_lh]
    extended_insulin_norm = insulin_norm_window[_lh]
    extended_exercise_norm = exercise_norm_window[_lh]
    extended_carb_raw = carb_raw_window[_lh]
    extended_insulin_raw = insulin_raw_window[_lh]
    extended_exercise_raw = exercise_raw_window[_lh]

    # Hour-of-day at the prediction-zone start for nocturnal metric filtering.
    # Indexed with the ABSOLUTE step ``pred_start_step``.
    pred_start_hour = float(hour_of_day[pred_start_step])

    bg_formula_data = {
        # The masked set and everything keyed to it, all (M,) with M =
        # MAX_MASKED_PATCHES.  Padded slots gather patch 0 and carry a legal
        # anchor; ``valid`` is the only thing that discards them, so any loss or
        # metric path that drops it supervises those slots against patch 0.
        'mask_idx': mask_idx,          # (M,) int64  patch index per head slot
        'valid': valid,                # (M,) bool
        'anchor_bg': anchor_bg,        # (M,) float32 mg/dL
        'd': mask_d,                   # (M,) int64  patches to nearest visible, EITHER side
        'slot_hour': slot_hour,        # (M,) float32 true hour of day per slot
        'last_bg': last_bg,
        'true_bg_trajectory': true_bg_traj.copy(),
        'extended_true_bg_trajectory': extended_true_bg_traj.copy(),
        'pred_start_hour': pred_start_hour,
        'extended_carb_norm': extended_carb_norm.copy(),
        'extended_insulin_norm': extended_insulin_norm.copy(),
        'extended_exercise_norm': extended_exercise_norm.copy(),
        'extended_carb_raw': extended_carb_raw.copy(),
        'extended_insulin_raw': extended_insulin_raw.copy(),
        'extended_exercise_raw': extended_exercise_raw.copy(),
    }
    if unblinded_dose_rows is not None:
        # BLIND ONLY, and UN-COLLATED ONLY.  ``collate_fn`` builds its batched
        # dict from an explicit allowlist and does not carry these, exactly as it
        # does not carry the ``extended_*`` arrays above; both are consumed from a
        # single ``dataset[i]`` sample by the validation roll.
        #
        # ABSENT rather than None under the announced policy, and that is
        # load-bearing: feats 1-3 were never overwritten there so there is nothing
        # to undo, and the announced sample has to stay byte-identical through the
        # flag — ``tests/test_blind_dataset.py`` digests it and compares the two
        # policies key by key.
        bg_formula_data['unblinded_dose_rows'] = unblinded_dose_rows
        bg_formula_data['unblinded_dose_patches'] = unblinded_dose_patches

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
    # ~L967-983 stays authoritative).  It carries ONE masked span, the right-edge
    # forecast zone: feat 0 (bg) withheld there (FROZEN NON_MASKABLE_FEATS) so no
    # future bg leaks, feat 4 announcing it, carb/insulin/exercise announced
    # throughout.  Window k's general masked set is deliberately NOT reused — the
    # probe compares two forecasts one horizon apart.  SAME n_ctx as window k =>
    # identical seq_len and left-pad, so the two windows share the PADDING
    # geometry — and nothing else.  They do NOT share a masked set (window k's
    # comes from the uniform span sampler, this one is always the right-edge
    # forecast span), and the attention mask is a function of the masked set, so
    # each window needs its own; ``collate_fn`` builds it.  Diagnostic-only;
    # consumed by the cross-window consistency penalty.  Gated so the default-off
    # path ships nothing.
    if TIME_PROBE_ENABLED and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
        next_end_patch = n_ctx + 2 * PREDICTION_PATCHES
        next_valid = next_end_patch <= total_patches_needed   # in-range on patches_3d
        next_spans = [(n_ctx, PREDICTION_PATCHES)]
        next_mask_idx, next_slot_valid, next_d, next_anchor_step = _mask_slots(
            next_spans, seq_len
        )
        next_masked_rows = torch.arange(n_ctx, n_ctx + PREDICTION_PATCHES)
        # Window k+1 sits one horizon further along the SAME window, so its own
        # step 0 is at PREDICTION_PATCHES * PATCH_SIZE in ``bg_window``.
        next_offset = PREDICTION_PATCHES * PATCH_SIZE
        if next_valid:
            next_patches_t = torch.from_numpy(
                patches_3d[PREDICTION_PATCHES:next_end_patch]
                .reshape(seq_len, PATCH_SIZE * N_INPUT_FEATURES).copy()
            )
            for feat_idx in NON_MASKABLE_FEATS:
                next_patches_t[next_masked_rows, feat_idx::N_INPUT_FEATURES] = 0.0
            next_patches_t[next_masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0
            next_pred_start_step = pred_start_step + next_offset
            next_last_bg = float(
                bg_window[next_offset + _anchor_step_for_span(n_ctx, PREDICTION_PATCHES)]
            )                                                # raw mg/dL (clamped, physical)
            next_anchor_bg = np.full(MAX_MASKED_PATCHES, next_last_bg, dtype=np.float32)
            next_anchor_bg[next_slot_valid] = bg_window[
                next_offset + next_anchor_step[next_slot_valid]
            ]
            next_pred_start_hour = float(hour_of_day[next_pred_start_step])
            next_slot_hour = hour_of_day[
                start_step + next_offset + next_mask_idx * PATCH_SIZE
            ].astype(np.float32)
        else:
            # Room only under NIGHT_LONG_HORIZON_HOURS == PREDICTION_HORIZON_HOURS.
            # Ship a finite placeholder (masked out downstream); last_bg reuses window
            # k's valid mg/dL so the forward's units tripwire never fires.  The
            # announcement bit is still written so the placeholder is not a window
            # claiming every patch is observed.
            next_patches_t = torch.zeros(seq_len, PATCH_DIM, dtype=torch.float32)
            next_patches_t[next_masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0
            next_last_bg = last_bg
            next_anchor_bg = np.full(MAX_MASKED_PATCHES, last_bg, dtype=np.float32)
            next_pred_start_hour = pred_start_hour
            next_slot_hour = np.full(MAX_MASKED_PATCHES, pred_start_hour, dtype=np.float32)
        # Window k+1's own masked span withholds its doses under the blind policy
        # too — both windows of the pair are drawn under one convention, and the
        # placeholder branch's masked rows take the fill for the same reason.
        if blind_fill is not None:
            next_blind_flags = torch.zeros(seq_len, dtype=torch.bool)
            next_blind_flags[next_masked_rows] = True
            blind_masked_doses(next_patches_t, next_blind_flags, blind_fill)
        assert next_patches_t.shape == (seq_len, PATCH_DIM)
        sample['next_window'] = {
            'patches': next_patches_t.float(),
            'mask_idx': next_mask_idx,
            'valid_slots': next_slot_valid,
            'anchor_bg': next_anchor_bg,
            'd': next_d,
            'slot_hour': next_slot_hour,
            'last_bg': float(next_last_bg),
            'pred_start_hour': float(next_pred_start_hour),
            'valid': bool(next_valid),
        }

    return sample


def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate variable-length samples into a padded batch.

    Left-pads to ``max_T = max(seq_lens)`` — the BATCH maximum, never
    ``MAX_SEQ_LEN``, which appears nowhere in this module.  A forward that
    asserts ``patches.shape[1] == MAX_SEQ_LEN`` fires on most validation
    batches; write every forward as ``(B, T, .)`` with ``T <= MAX_SEQ_LEN``.
    Padding positions are blocked by the per-batch attention mask (and the
    diagonal is forced True so softmax doesn't NaN on all-False rows).

    Head-slot indices are rebased onto the PADDED axis here: a sample's masked
    patch ``p`` sits at ``n_pad + p``.  Padded slots keep index 0 — the padded
    tensor's position 0 — and are discarded by ``valid``.

    Args:
        samples: List of dicts from ``T1DMDataset.__getitem__``.

    Returns:
        batch dict with keys::

          patches:           (B, max_T, PATCH_DIM)             float32
          targets:           (B, M, PATCH_SIZE)                float32 (mg/dL)
          attn_mask:         (B, max_T, max_T)                 bool
          bg_formula_data:   batched dict (see _build_sample)
          n_context_patches: (B,)                              long
          next_window:       optional batched dict (cross-window time-of-day
                             probe input; present iff samples carry it).  It
                             carries its OWN ``attn_mask``: the two windows
                             share n_ctx and so the padding geometry, but not
                             the masked set the mask is built from.
    """
    B = len(samples)
    M = MAX_MASKED_PATCHES
    n_contexts = [s['n_context_patches'] for s in samples]
    seq_lens = [n + PREDICTION_PATCHES for n in n_contexts]
    max_T = max(seq_lens)
    n_pads = [max_T - sl for sl in seq_lens]

    # Pre-allocate the padded outputs.  Left-padding zeros are fine — the
    # attention mask is what actually disables padding positions.
    patches_batch = torch.zeros(B, max_T, PATCH_DIM, dtype=torch.float32)
    targets_batch = torch.stack([s['targets'] for s in samples])           # (B, M, PATCH_SIZE)
    n_ctx_tensor = torch.tensor(n_contexts, dtype=torch.long)

    valid_batch = torch.from_numpy(
        np.stack([s['bg_formula_data']['valid'] for s in samples])
    )                                                                      # (B, M) bool
    mask_idx_batch = torch.zeros(B, M, dtype=torch.long)
    is_pad = torch.zeros(B, max_T, dtype=torch.bool)
    masked = torch.zeros(B, max_T, dtype=torch.bool)

    for i, s in enumerate(samples):
        n_pad = n_pads[i]

        # Left-pad: place actual data at positions [n_pad:max_T] so the
        # prediction horizon is at the right edge for every sample in the
        # batch.  The model relies on this convention.
        patches_batch[i, n_pad:, :] = s['patches']

        is_pad[i, :n_pad] = True
        row_valid = valid_batch[i]
        idx = torch.from_numpy(s['bg_formula_data']['mask_idx']) + n_pad
        mask_idx_batch[i, row_valid] = idx[row_valid]
        masked[i, mask_idx_batch[i, row_valid]] = True

    # The whole mask, vectorised: visible rows see visible columns, masked rows
    # see everything real, pad rows and columns are blocked, and the diagonal is
    # forced True at EVERY position including padding.  That last step is what
    # keeps a pad row — blocked as a row and as a column — from being all-False
    # and NaN-ing the softmax on the direct-to-SDPA bool path.  It lives inside
    # ``create_attention_mask_from_visible`` as line 4 of its four-line rule; the
    # assert below is this call site's guard that the guarantee still holds.
    attn_masks = utils.create_attention_mask_from_visible(~masked, is_pad)
    assert attn_masks.any(dim=-1).all(), "an all-False attention row NaNs softmax"

    # Feat 4 must agree with the masked set that built the attention mask, on
    # THIS path as well as the inference one.  Nothing else catches a
    # disagreement: z = 0 in a withheld bg slot decodes to an ordinary reading
    # (~142 mg/dL on the balanced pool), so a patch announced as visible while
    # its bg is withheld teaches the model to read a fabricated observation, and
    # every loss and metric downstream stays finite and plausible.  The bit is
    # per patch and the layout step-major, so all PATCH_SIZE columns of the
    # feature carry it identically.
    _bit = patches_batch[..., BG_MASKED_FEAT::N_INPUT_FEATURES]   # (B, max_T, PATCH_SIZE)
    assert _bit.shape[-1] == PATCH_SIZE, (
        f"feat {BG_MASKED_FEAT} stride slice gave {_bit.shape[-1]} columns, "
        f"expected PATCH_SIZE={PATCH_SIZE} — the step-major layout is broken"
    )
    assert bool((_bit == _bit[..., :1]).all()), (
        "the bg_masked bit differs across a patch's step-major columns"
    )
    assert torch.equal(_bit[..., 0], masked.to(_bit.dtype)), (
        "feat 4 does not reproduce the sampled mask that built attn_mask"
    )

    # Cross-window (paired-window) time-of-day probe input, present only when the
    # samples carry it (probe on + cross-window weight > 0). Window k+1 shares
    # window k's n_ctx, so it shares the left-pad ``n_pad`` and therefore the
    # padding geometry — ``is_pad`` below is window k's. The masked set is NOT
    # shared: window k+1 masks its own right-edge forecast span, window k
    # whatever its sampler drew. The attention mask is built from the masked set,
    # so window k+1 gets its own, shipped at ``next_window['attn_mask']``.
    next_window_batched = None
    if 'next_window' in samples[0]:
        nw_patches = torch.zeros(B, max_T, PATCH_DIM, dtype=torch.float32)
        nw_valid_slots = torch.from_numpy(
            np.stack([s['next_window']['valid_slots'] for s in samples])
        )                                                                            # (B, M) bool
        nw_mask_idx = torch.zeros(B, M, dtype=torch.long)
        nw_masked = torch.zeros(B, max_T, dtype=torch.bool)
        for i, s in enumerate(samples):
            n_pad = n_pads[i]                        # identical to window k's n_pad
            nw_patches[i, n_pad:, :] = s['next_window']['patches']
            row_valid = nw_valid_slots[i]
            idx = torch.from_numpy(s['next_window']['mask_idx']) + n_pad
            nw_mask_idx[i, row_valid] = idx[row_valid]
            nw_masked[i, nw_mask_idx[i, row_valid]] = True

        nw_attn_masks = utils.create_attention_mask_from_visible(~nw_masked, is_pad)
        assert nw_attn_masks.any(dim=-1).all(), "an all-False attention row NaNs softmax"

        # The same feat-4-versus-mask agreement the primary patches get above,
        # against THIS window's masked set. A patch announced masked while the
        # attention mask still offers it as evidence contradicts itself, and
        # nothing downstream notices: the withheld bg is zeroed either way and
        # every loss and metric stays finite.
        _nw_bit = nw_patches[..., BG_MASKED_FEAT::N_INPUT_FEATURES]   # (B, max_T, PATCH_SIZE)
        assert _nw_bit.shape[-1] == PATCH_SIZE, (
            f"feat {BG_MASKED_FEAT} stride slice gave {_nw_bit.shape[-1]} columns, "
            f"expected PATCH_SIZE={PATCH_SIZE} — the step-major layout is broken"
        )
        assert bool((_nw_bit == _nw_bit[..., :1]).all()), (
            "the next_window bg_masked bit differs across a patch's step-major columns"
        )
        assert torch.equal(_nw_bit[..., 0], nw_masked.to(_nw_bit.dtype)), (
            "next_window feat 4 does not reproduce the mask that built its attn_mask"
        )

        next_window_batched = {
            'patches': nw_patches,                                                   # (B, max_T, PATCH_DIM)
            'attn_mask': nw_attn_masks,                                              # (B, max_T, max_T) bool
            'mask_idx': nw_mask_idx,                                                 # (B, M) long, padded axis
            'valid_slots': nw_valid_slots,                                           # (B, M) bool
            'anchor_bg': torch.from_numpy(
                np.stack([s['next_window']['anchor_bg'] for s in samples])),         # (B, M) mg/dL
            'd': torch.from_numpy(
                np.stack([s['next_window']['d'] for s in samples])),                 # (B, M) long
            'slot_hour': torch.from_numpy(
                np.stack([s['next_window']['slot_hour'] for s in samples])),         # (B, M) hours
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
    # insulin/exercise_*`` announced-future arrays are NOT stacked here — they are
    # consumed only from UN-COLLATED samples, by the rolled-forecast override
    # (``train._make_long_horizon_overrides_fn``, whose callers iterate the
    # dataset directly), so they ride only on each sample's own ``bg_formula_data``.
    bg_formula_batched: dict[str, Any] = {
        # The masked set, on the PADDED patch axis.  Every one of these is (B, M).
        'mask_idx': mask_idx_batch,
        'valid': valid_batch,
        'anchor_bg': torch.from_numpy(
            np.stack([s['bg_formula_data']['anchor_bg'] for s in samples])),   # mg/dL
        'd': torch.from_numpy(
            np.stack([s['bg_formula_data']['d'] for s in samples])),
        'slot_hour': torch.from_numpy(
            np.stack([s['bg_formula_data']['slot_hour'] for s in samples])),   # hours
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
