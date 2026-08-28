"""
T1DMAI Data Pipeline — on-the-fly and cached sample generation from T1DMSIM.

``T1DMDataset.__getitem__`` returns one sample:

* ``patches``           — (T, PATCH_DIM) float.  T = n_ctx + PREDICTION_PATCHES.
                          Each patch is PATCH_SIZE × N_INPUT_FEATURES values and
                          is either VISIBLE or MASKED.  A masked patch withholds
                          bg (feat 0) and announces itself through the bg_masked
                          bit (feat 4, written into all PATCH_SIZE of its
                          step-major columns); carb (feat 1) / insulin (feat 2) /
                          exercise (feat 3) keep their true or announced values
                          everywhere, unless the dataset is ``blind`` (below).
* ``targets``           — (MAX_MASKED_PATCHES, PATCH_SIZE) float, ground-truth BG
                          in mg/dL, one row per head slot.  The RAW
                          ``bg_observed`` — the SAME signal fed as input, no
                          smoothing — and NOT f-transformed in the batch: the
                          Kovatchev risk transform is applied once at the top of
                          the loss.
* ``n_context_patches`` — int, length of the variable-size context window.
* ``bg_formula_data``   — per-sample arrays consumed by the loss, validation and
                          inference.  The masked set, all (MAX_MASKED_PATCHES,):
                          ``mask_idx`` (patch index per head slot), ``valid``,
                          ``anchor_bg`` (mg/dL), ``d`` (patches to the nearest
                          visible evidence on either side) and ``slot_hour``
                          (true hour of day).  Plus ``last_bg`` (raw last-context
                          BG, mg/dL), ``true_bg_trajectory`` and
                          ``extended_true_bg_trajectory`` (mg/dL ground truth from
                          the context edge over the horizon and the long horizon),
                          ``pred_start_hour``, and the announced future
                          ``extended_{carb,insulin,exercise}_{norm,raw}`` used by
                          the conditioned rolling override and the counterfactual
                          probes.

One raw post-noise space, no smoothing.  Every signal channel — bg, carb, insulin
AND exercise — is fed RAW, and there is no causal smoother on inputs or on the
forecast target.  The SAME raw bg is the model input, the forecast TARGET and
``last_bg`` (bg clamped only to [BG_CLAMP_MIN, BG_CLAMP_MAX]; carb / insulin /
exercise floored at 0).  Deployment realism is intrinsic: the live CGM/dose stream
is consumed as-is and the autoregressive roll re-feeds the model's own raw output,
so train and inference input distributions match.

There is no prediction zone.  ``sample_mask_spans`` draws 1-3 non-abutting spans
anywhere in the window and the model emits a quantile fan for every masked patch:
a span ending at patch T-1 is a FORECAST, one starting at patch 0 a BACKCAST,
anything else INFILL — one objective, three cases.

* Context windows are sampled uniformly in [MIN_CONTEXT_PATCHES,
  MAX_CONTEXT_PATCHES]; ``collate_fn`` left-pads to the BATCH maximum.
* The input feature stack is exactly [bg_absolute, carb, insulin, exercise_equiv,
  bg_masked]; there are no temporal sin/cos features.  bg (feat 0) enters in
  Kovatchev risk space — z(f(bg)) — while carb/insulin/exercise keep log1p+z.
  Exercise is the simulator's carbohydrate-EQUIVALENT glucose-disposal curve in
  g/step, fed at that scale, never rescaled to an intensity and never
  risk-transformed.  bg_masked (feat 4) is a BIT and is never normalized, so the
  feature count and the normalized-channel count are not the same number.
* Feats 1-3 are PLAN channels: nothing but what the patient announced is ever
  written into them, masked patches included.  bg (feat 0) is zeroed exactly at
  the masked patches, so the model cannot copy the signal it is asked to emit.
* ``blind=True`` is the ONE departure, and it is off by default.  It withholds
  feats 1-3 on masked patches too, at the per-channel zero-RAW ``normalize(0)``
  (``zero_dose_fill``) — the same "no dose" baseline
  ``inference.predict_rolling`` writes when nothing is announced — so
  ``train_blind.py`` can measure the model with no conditioning at all.  A model
  only ever sees one convention, so feat 4 still announces the withholding and
  there is no second bit.
* Masking is NOT inferable from position, and z = 0 in a withheld bg slot decodes
  to an ordinary reading (~142 mg/dL on the balanced pool), not a sentinel —
  which is why feat 4 announces the masked set explicitly.
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

# The input stack is [*CHANNEL_NAMES, bg_masked]: the leading
# ``len(CHANNEL_NAMES)`` columns are normalized signal channels, the trailing one
# a per-PATCH 0/1 bit that is never normalized.  The two counts differ and must
# not be asserted equal.  Derived, never restated.
BG_MASKED_FEAT = len(CHANNEL_NAMES)
assert N_INPUT_FEATURES == len(CHANNEL_NAMES) + 1, (
    f"N_INPUT_FEATURES={N_INPUT_FEATURES} should be len(CHANNEL_NAMES)="
    f"{len(CHANNEL_NAMES)} normalized channels plus the bg_masked bit"
)

# What a masked patch withholds; a model only ever sees one of the two.
# ``announced`` withholds bg alone and lets the announced carb / insulin /
# exercise ride through; ``blind`` withholds those three as well.  The string is
# checkpoint provenance only — no parameter shape depends on it, so a strict
# state-dict load accepts weights trained under either — and
# ``calibrate_conformal.py`` compares it against the policy it implements.  An
# ABSENT key means ``announced``.
MASKED_CHANNEL_POLICY_ANNOUNCED = 'announced'
MASKED_CHANNEL_POLICY_BLIND = 'blind'


def masked_channel_policy(blind: bool) -> str:
    """The masked-channel policy name a ``blind`` flag selects."""
    return MASKED_CHANNEL_POLICY_BLIND if blind else MASKED_CHANNEL_POLICY_ANNOUNCED


def stored_masked_channel_policy(training_config: dict[str, Any] | None) -> str:
    """The masked-channel policy a checkpoint's ``training_config`` records.

    The ONE reader of the absent-key convention, so a second convention cannot
    appear: every consumer goes through here.  An absent key reads as
    ``announced`` unconditionally, never as "unknown" — a blind run always stamps
    the key, so a checkpoint lacking it cannot be a blind one.
    """
    tc = training_config or {}
    return str(tc.get('masked_channel_policy', masked_channel_policy(blind=False)))


def checkpoint_masked_channel_policy(ckpt: dict[str, Any] | None) -> str:
    """``stored_masked_channel_policy`` over a whole checkpoint dict."""
    return stored_masked_channel_policy((ckpt or {}).get('training_config'))


def zero_dose_fill(stats: dict[str, dict[str, float]]) -> dict[int, float]:
    """``{feat_idx: z}`` over ``MASKABLE_FEATS`` — per-feat ``normalize(0)``, z-space.

    What a blind masked patch carries in feats 1-3.  NOT ``z = 0``: the sparse
    channels are log1p'd before the z-score, so ``z = 0`` inverts to
    ``expm1(mean)`` — a phantom ~0.47 g/step of carb on the balanced pool, not an
    absence of one.  ``normalize(0)`` is the channel's ``-mean/std``, means "no
    dose", and is the same baseline ``inference.predict_rolling`` writes into an
    un-overridden slot, so a blind roll is in-distribution.
    Derived from the loaded stats every time, never written down: the values are
    properties of the pool the stats were fit on.
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

    ``patches`` ``(..., T, PATCH_DIM)`` step-major rows, ``masked`` ``(..., T)``
    bool, ``fill`` from ``zero_dose_fill``.  The one place the blind convention is
    implemented: ``_build_sample`` applies it to the sample's own masked set and
    to ``next_window``'s, ``train_blind.py``'s two protocol forwards to the sets
    they mask, so validation scores the task the model trained on.  Feat 0 and
    feat 4 are untouched — the bg withholding and the announcement bit are the
    same under either policy.
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

# Post-warmup hours of raw trajectory per sample.  A sample needs at most
# MAX_CONTEXT_PATCHES + max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)
# patches, read off config so a resize carries this with it.
#
# Warmup discards a whole number of hours, so step 0 is midnight and the
# pred_start_step jitter is also an hour-of-day range.  ``_pick_pred_start_step``
# picks uniformly among the patch-aligned starts, of which there are
#   n_candidates = N / PATCH_SIZE
#                  - max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)
#                  - n_ctx + 1
# returning None once that falls below 1.  The max() equals
# NIGHT_LONG_HORIZON_PATCHES only because 16 > 4 today, and becomes
# PREDICTION_PATCHES as soon as the supervised horizon outgrows the nocturnal one.
# Candidates sit at 30-min spacing, so a day holds 48 slots and hour-of-day
# coverage is uniform at a fixed n_ctx iff n_candidates % 48 == 0, i.e.
# N ≡ 90 (mod 288); any other length puts an extra candidate on slot 0.
#
# At N = 2394 steps, n_candidates = 384 - n_ctx, exact at every whole-day width
# up to the ceiling (n_ctx ∈ {48, 96, ..., 336}, 24 h through 7 days), and 2394 is
# the SMALLEST congruent length leaving a 336-patch context any candidate at all:
# that needs (336 + 16) × 6 = 2112 steps and 2106 is the previous congruent
# length.  At the ceiling it leaves exactly 48 candidates, one full day.
#
# n_ctx is drawn uniformly, so the pooled hour-of-day histogram is a mixture over
# widths and never flat; the residual peak sits at slot 47 (23:30) and the trough
# at slot 0, so the tilt runs away from midnight rather than onto it.
#
# PAIRED with T1DMSIM/cache_simulator.py's --sim-hours: ``T1DMDataset.__init__``
# rejects a cache whose meta['sim_hours'] differs.  Raising it lengthens every
# on-the-fly simulator request in proportion.
ON_THE_FLY_SIM_HOURS: float = 199.5

# The i-th calibration patient draws ``master_seed + CALIBRATION_SEED_OFFSET + i``
# — a band clear of both the training hashed seeds and normalization's
# ``+1_000_000``.  It backs split-conformal recalibration only, never the training
# loop or the headline validation metrics.
CALIBRATION_SEED_OFFSET: int = 2_000_000


# Cache-pool partitioning, train / val / cal disjointness.  ``sha256(seed) %
# pool_size`` reprojects the validation (``master_seed + 10_000_000``) and
# calibration (``CALIBRATION_SEED_OFFSET``) seed bands INDEPENDENTLY and uniformly
# over ``[0, pool_size)``, so without this a val/cal sample lands on the exact
# cache row a train sample uses and leaks a held-out trajectory into training.
# (The on-the-fly path is immune — distinct seed bands hash to distinct 63-bit
# seeds and never touch a shared finite pool.)
#
# So the pool is carved into three DISJOINT slabs keyed by ``cache_partition``,
# each mapping ``cache_idx = slab_start + (patient_seed % slab_size)``: no row is
# shared across partitions for ANY master seed.  The reserves are structural
# constants, not training tunables.
CACHE_PARTITIONS: tuple[str, ...] = ('train', 'val', 'cal')
CACHE_VAL_SLAB_ROWS: int = 100_000   # reserved tail rows for the validation bands
CACHE_CAL_SLAB_ROWS: int = 100_000   # reserved tail rows for the calibration band


def _cache_slab_geometry(pool_size: int, partition: str) -> tuple[int, int]:
    """``(slab_start, slab_size)`` cache-row band of a partition, half-open, ``slab_size >= 1``.

    The three bands are pairwise DISJOINT and cover ``[0, pool_size)``, train
    keeping the large contiguous head::

        train : [0, pool_size - val_slab - cal_slab)
        cal   : [pool_size - val_slab - cal_slab, pool_size - val_slab)
        val   : [pool_size - val_slab, pool_size)

    The val/cal reserves are clamped to at most a third of the pool and at least
    one row, so a tiny test pool cannot starve train.
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
    """A numpy ``Generator`` proxy overriding only ``multivariate_normal``.

    The T1DMSIM patient sampler draws skills in one ``rng.multivariate_normal``
    call, sigmoids, then clips to ``[SKILL_MIN, SKILL_MAX]``; the normal sampler
    squeezes most patients to the centre and makes extreme ones rare.  Short-
    circuiting that one call lands the post-sigmoid skills UNIFORMLY across the
    range, oversampling the tails.  Every other method is forwarded unchanged, so
    the non-skill parameters keep their usual distributions.
    """

    def __init__(self, rng: np.random.Generator, skill_min: float, skill_max: float) -> None:
        self._rng = rng
        self._skill_min = skill_min
        self._skill_max = skill_max

    def __getattr__(self, name: str):
        return getattr(self._rng, name)

    def multivariate_normal(self, mean, _cov, *_args, **_kwargs):
        n = len(mean)
        # Strictly inside (0, 1) so the logit below is finite.
        lo = max(self._skill_min, 1e-3)
        hi = min(self._skill_max, 1.0 - 1e-3)
        skills = self._rng.uniform(lo, hi, size=n)
        # Inverse-sigmoid, so the simulator's own sigmoid restores the uniform.
        return np.log(skills / (1.0 - skills))


def _make_simulator(patient_seed: int, uniform_skills: bool):
    """A fresh ``T1DMSimulator``, optionally with uniform-skill sampling.

    Never cached across calls: the simulator is stateful — ``generate_hours``
    advances its clock — so a cache hit on a used seed hands back an instance
    already past the warmup window and silently corrupts every sample drawn at
    that seed.
    """
    from T1DMSIM.simulator import T1DMSimulator
    if not uniform_skills:
        return T1DMSimulator(seed=patient_seed)

    # Monkey-patch the module-level ``generate_patient`` for the duration of the
    # constructor rather than duplicating the simulator's patient generation;
    # restored immediately so other instances are unaffected.
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
    """``sim.generate_hours(hours + warmup_hours)`` less the first ``warmup_hours``.

    The simulator starts from an empty meal / insulin history, so the first day
    has unrealistic dynamics — no prior-day IOB, no residual carb-on-board, fresh
    basal.  Every non-test caller routes through here, so training, normalization,
    inference and the GUI all see the same cold-start-free window.
    """
    from T1DMSIM.simulator import DT_MINUTES
    raw = sim.generate_hours(hours + warmup_hours)
    n_warmup = int(warmup_hours * 60 / DT_MINUTES)
    return {k: v[n_warmup:] for k, v in raw.items()}


def _pick_pred_start_step(
    n_steps: int,
    n_ctx: int,
    n_pred_steps: int,
    rng: np.random.Generator,
) -> int | None:
    """A patch-aligned pred-zone start anywhere in the trajectory, or ``None``.

    Requires only ``n_ctx`` patches of context behind it and ``n_pred_steps`` raw
    timesteps ahead.  Callers pass the long-horizon footprint
    (NIGHT_LONG_HORIZON_PATCHES * PATCH_SIZE) as the room, so the trailing
    ground-truth slice fits even though the supervised zone is shorter.
    """
    n_ctx_steps = n_ctx * PATCH_SIZE

    earliest = n_ctx_steps
    latest = n_steps - n_pred_steps
    if latest < earliest:
        return None

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
    """A patch-aligned start whose hour-of-day is nearest ``target_hour``, circular.

    ``hour_of_day`` is the (N,) per-step clock of the trimmed trajectory; the
    context / horizon room requirement is ``_pick_pred_start_step``'s, and so is
    the ``None`` return.  Reached through ``force_pred_start_hour``, for an
    evaluation with its origins pinned to one hour — a bedtime origin
    (``NOCTURNAL_START_HOUR`` ≈ 22:00), say, so a rolled forecast spans the whole
    night.  One candidate within ``tol_hours`` of the target is picked at random
    for variety (there is about one per day); with none eligible, the single
    nearest.
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


# On-disk cache formats (T1DMSIM/cache_simulator.py): ``blosc2`` compressed,
# ``npy`` a raw uncompressed per-channel memmap.  Same ``meta.json`` fields and
# per-row read semantics either way.
CACHE_FORMAT_BLOSC2 = 'blosc2-ndarray-v1'
CACHE_FORMAT_NPY = 'npy-memmap-v1'
SUPPORTED_CACHE_FORMATS = (CACHE_FORMAT_BLOSC2, CACHE_FORMAT_NPY)


class T1DMDataset(Dataset):
    """T1DM training dataset; length ``total_steps * batch_size``.

    Each index maps to a unique ``(step, position)`` pair, deterministically
    deriving a patient seed via ``compute_patient_seed``.  With ``cache_path=None``
    the simulator runs on demand inside the worker and dominates the per-batch
    cost; with a ``T1DMSIM/cache_simulator.py`` directory it reads pre-generated
    trajectories instead, mapping ``patient_seed % slab_size`` within this
    partition's slab, so different ``master_seed``s draw different mixes from one
    pool and every ``idx`` stays deterministic.

    ``patient_uniform_sample_prob`` is the per-sample probability of drawing
    skills uniformly across [SKILL_MIN, SKILL_MAX] rather than from the
    simulator's multivariate normal, oversampling tail patients; ``0.0`` disables
    it.  It and ``simulator_warmup_hours`` are IGNORED in cache mode — both are
    baked in at generation time, and a warmup mismatch raises at load.

    ``seed_offset`` shifts the whole patient-seed band (0 = training;
    ``CALIBRATION_SEED_OFFSET`` for the reserved conformal partition).
    ``cache_partition`` picks the DISJOINT cache slab — ``'val'`` and ``'cal'``
    take the reserved tails, ``'train'`` the head — which is what keeps the +10M
    val / +2M cal bands from collapsing onto train rows; it has no effect on the
    on-the-fly path, where distinct seed bands never collide anyway.
    ``force_pred_start_hour`` pins the prediction origin to the patch-aligned step
    nearest that hour (validation only).  ``blind`` withholds the dose channels on
    masked patches too, at ``zero_dose_fill``; ``train_blind.py`` is its only
    caller.
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
        # (slab_start, slab_size) for this partition; None in on-the-fly mode.
        self._cache_slab: tuple[int, int] | None = None

        # Lazy — populated on first access inside the worker, so no open cache
        # handle is pickled across the DataLoader fork boundary.
        self._cache_arrays: dict[str, Any] | None = None
        self._cache_icr: np.ndarray | None = None
        self._cache_pool_size: int | None = None
        self._cache_n_timesteps: int | None = None
        self._cache_meta: dict[str, Any] | None = None
        # npy-memmap madvise metadata, name -> (mmap, data_offset_bytes,
        # row_bytes).  None under blosc2 (different memory model) or when
        # MADV_DONTNEED is unavailable.
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

            # A cache with no cache_format key at all is already rejected by the
            # required-keys check above.
            cache_format = str(meta['cache_format'])
            if cache_format not in SUPPORTED_CACHE_FORMATS:
                raise ValueError(
                    f"Cache cache_format={cache_format!r} is not supported "
                    f"by this version of data.py (expected one of "
                    f"{SUPPORTED_CACHE_FORMATS}). Rebuild the cache with the "
                    "current T1DMSIM/cache_simulator.py."
                )

            # Every value baked into the trajectories must match the dataset's
            # runtime assumptions: a disagreement changes what the model sees
            # against the on-the-fly path, so it fails loudly instead.
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
            # Carve this partition's disjoint band now that pool_size is known, so
            # a held-out seed can never reproject onto a train cache row.
            self._cache_slab = _cache_slab_geometry(
                self._cache_pool_size, self.cache_partition)

            # A pool smaller than total_steps * batch_size cycles, which is
            # benign: ``_build_sample`` draws a fresh random context+horizon window
            # each time and one trajectory admits many patch-aligned windows, so
            # every reuse is a DIFFERENT training window.

    def __len__(self) -> int:
        return self.total_steps * self.batch_size

    def _load_cache(self) -> tuple[dict[str, Any], np.ndarray]:
        """Open the cache arrays on first use in this process.

        Gives the per-channel array dict and the per-patient ICR array (tiny,
        fully in RAM).  Two formats, by ``meta['cache_format']``:

        * ``'blosc2-ndarray-v1'`` — chunked byte-shuffle + zstd ``.b2nd``; a
          per-row read decompresses exactly one chunk.
        * ``'npy-memmap-v1'`` — raw uncompressed ``.npy`` memmap; a per-row read
          faults in only the touched pages.

        Only the npy format is mapped.  Workers share the kernel page cache either
        way and ``arr[i:i+1]`` returns a fresh row, so ``__getitem__`` is identical
        across formats.
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
                    # Suppress the kernel's 128 KB readahead so a fault pulls only
                    # the pages the row touches.  Without it the per-row
                    # MADV_DONTNEED below drops the row's ~2 pages and leaves ~30
                    # readahead pages resident, growing page cache ~128 KB per read
                    # (~0.5 GB/step).  Best-effort; access is 100% random, so there
                    # is no sequential throughput to trade away.
                    _madv_random = getattr(mmap, 'MADV_RANDOM', None)
                    if _madv_random is not None:
                        try:
                            arr._mmap.madvise(_madv_random)
                        except (OSError, ValueError, AttributeError):
                            pass
                    # Per-row byte geometry, so __getitem__ can MADV_DONTNEED
                    # exactly the pages it faults in.
                    mmaps[name] = (
                        arr._mmap, int(arr.offset),
                        int(arr.shape[1] * arr.dtype.itemsize),
                    )
                if self._madv_dontneed is not None:
                    self._cache_mmaps = mmaps
            else:
                import blosc2
                for name in CACHE_CHANNEL_NAMES:
                    # Deliberately NOT mmap_mode='r'.  A mapped .b2nd faults each
                    # touched chunk's compressed pages into this process and
                    # nothing can drop them again — blosc2 exposes no mapping to
                    # madvise.  Random access then grows RssFile ~100 KB per channel
                    # per row (~400 MB/step at BATCH_SIZE=512) until the whole cache
                    # is resident.  Plain file reads leave the pages in ordinary
                    # page cache, charged to nobody and reclaimed under pressure.
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

        Under npy-memmap every row read faults a page or two of the multi-TB
        channel files into this worker's mapping, and the access barely repeats,
        so they would otherwise accumulate as unbounded page cache.
        ``madvise(MADV_DONTNEED)`` over the page-aligned range drops them at once;
        a re-read re-faults from the file.  Best-effort: a no-op under blosc2
        (never mapped) or without ``MADV_DONTNEED``, and any per-call failure is
        swallowed so a platform quirk cannot break data loading.
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
            length += (-length) % page  # whole pages
            try:
                mm.madvise(advice, aligned, length)
            except (OSError, ValueError, AttributeError):
                pass

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """One training sample for ``idx`` in ``[0, total_steps * batch_size)``.

        Keys are ``_build_sample``'s.  The same ``idx`` always resolves to the same
        patient seed, whichever worker handles it.
        """
        step = idx // self.batch_size
        position = idx % self.batch_size
        patient_seed = compute_patient_seed(
            self.master_seed + self.seed_offset, step, position,
        )

        if self.cache_path is not None:
            cache_arrays, cache_icr = self._load_cache()
            assert self._cache_pool_size is not None
            assert self._cache_slab is not None
            # This partition's DISJOINT band: a held-out (val/cal) seed can only
            # resolve to a reserved tail row, never a train row.
            slab_start, slab_size = self._cache_slab
            cache_idx = slab_start + int(patient_seed % slab_size)
            if self._cache_mmaps is not None:
                # npy-memmap: ``[idx:idx+1]`` is a VIEW into the shared mmap, so
                # each row is copied out to sever it before MADV_DONTNEED drops
                # the pages — without the copy the sample aliases the very pages
                # dropped and re-faults them on use.  ``[idx:idx+1]`` rather than
                # ``[idx]`` keeps the call on the slice path the stubs annotate.
                data = {
                    name: np.array(cache_arrays[name][cache_idx:cache_idx + 1])[0]
                    for name in CACHE_CHANNEL_NAMES
                }
                self._madvise_row(cache_idx)
            else:
                # blosc2: indexing decompresses the touched chunk into a fresh
                # writable ndarray — no defensive copy needed, no mapping to advise.
                data = {
                    name: np.asarray(cache_arrays[name][cache_idx:cache_idx + 1])[0]
                    for name in CACHE_CHANNEL_NAMES
                }
            icr = float(cache_icr[cache_idx])
        else:
            # Keyed off ``patient_seed`` so the same idx always resolves the same
            # way; the XOR is a separate deterministic substream.
            if self.patient_uniform_sample_prob > 0.0:
                mode_rng = np.random.default_rng(patient_seed ^ 0x5A17_5EEDD)
                use_uniform = bool(mode_rng.random() < self.patient_uniform_sample_prob)
            else:
                use_uniform = False

            sim = _make_simulator(patient_seed, uniform_skills=use_uniform)
            data = simulate_discard_warmup(
                sim, ON_THE_FLY_SIM_HOURS, warmup_hours=self.simulator_warmup_hours
            )
            icr = float(sim.patient.icr)

        # Separate substream, so the mode rng above cannot influence window
        # selection.
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
    """The conformal-calibration dataset over the reserved seed band.

    Patient seeds are ``master_seed + CALIBRATION_SEED_OFFSET + i``, disjoint from
    the training hashed seeds and from normalization's ``+1_000_000`` band, so the
    recalibration pass scores patients neither the loss nor the headline
    validation ever saw.  ``master_seed`` and ``normalization_stats`` must be the
    training run's own, and ``blind`` its masked-channel policy: split conformal
    is valid only on the distribution the model runs under, so a blind checkpoint
    must be calibrated on blind windows.
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


def sample_mask_spans(
    seq_len: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """The masked spans of one sample over a ``seq_len``-patch window, left to right.

    ``(start_patch, length)`` pairs, strictly increasing in ``start_patch``, with
    at least one visible patch between consecutive spans and
    ``sum(length) <= MAX_MASKED_PATCHES``.  A span ending at patch
    ``seq_len - 1`` is a forecast, one starting at patch 0 a backcast, anything
    else infill.  The draw is the same under either masked-channel policy —
    nothing here reads the flag.

    Procedure::

        n_spans      ~ U{1 .. MASK_MAX_SPANS}
        each L_i     ~ U(MASK_SPAN_LENGTHS), independently
        if sum(L) > MAX_MASKED_PATCHES: resample the WHOLE length vector
        with prob MASK_RIGHT_EDGE_QUOTA: last span flush right, rest over the
                                         prefix by stars-and-bars
        otherwise:                       stars-and-bars over n_spans + 1 gaps

    The over-budget rejection redraws the whole vector, never one element:
    per-element redrawing yields a different length distribution and so a
    different ``d`` histogram.  Placement is a uniform composition of the slack
    over the gaps, with a MANDATORY visible patch between neighbouring spans; no
    rejection loop on placement, no curriculum, no annealing.

    ``MASK_RIGHT_EDGE_QUOTA`` is the one departure from uniform placement and
    changes ONLY where the last span lands: ``n_spans`` and the length law are
    drawn identically in both branches, so the span-length histogram is
    quota-independent and only the ``d`` histogram moves.  Under uniform placement
    a FORECAST — the deployed case — is an accident worth ~3% of windows, and the
    band it emits there decays with training while every selection scalar
    improves.  ``config.MASK_RIGHT_EDGE_QUOTA`` carries the paired evidence.

    Both branches are enumerated exactly in ``d_balance.d_distribution``; a change
    here not mirrored there moves every ``d``-binned figure's reference.

    Two masked spans never abut: the separator is what makes the anchor, the
    spline's node sequence and the DILATE length bucket well defined per span, and
    two spans with nothing between them are one longer span.
    """
    lengths_pool = np.asarray(MASK_SPAN_LENGTHS, dtype=np.int64)
    n_spans = int(rng.integers(1, MASK_MAX_SPANS + 1))

    # Rejection is on the LENGTH VECTOR as a whole.
    while True:
        span_lengths = rng.choice(lengths_pool, size=n_spans, replace=True)
        if int(span_lengths.sum()) <= MAX_MASKED_PATCHES:
            break

    total_masked = int(span_lengths.sum())
    # One mandatory visible patch per interior boundary, charged up front; the
    # slack is what remains to spread freely over the n_spans + 1 gaps.
    slack = seq_len - total_masked - (n_spans - 1)
    if slack < 0:
        raise RuntimeError(
            f"window of {seq_len} patches cannot hold {n_spans} spans of "
            f"total length {total_masked} with separators"
        )

    def _compose(slack_: int, n_gaps_: int) -> np.ndarray:
        """``slack_`` stars into ``n_gaps_`` ordered bins, uniform over compositions.

        Choosing the ``n_gaps_ - 1`` bar positions out of ``slack_ + n_gaps_ - 1``
        slots without replacement is exactly that uniform draw.
        """
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
    # The right-edge branch pins the LAST span at the final patch and composes the
    # rest over the prefix clear of it and its separator.  That prefix holds
    # exactly ``slack`` free patches — pinning is the uniform arrangement with the
    # trailing gap fixed at 0, and fixing a gap frees the separator it no longer
    # needs — so the branch needs no feasibility test of its own.
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
    """Window-relative step index of one masked span's anchor.

    ONE-SIDED and LEFT-PREFERRING: the LAST step of the left neighbour patch, or
    the FIRST step of the right neighbour when the span starts at patch 0 — the
    only no-left-neighbour case there is.  Every slot of a contiguous span gets
    the SAME value.  The single anchor rule.

    Read at the raw mg/dL array on the training path: a right-edge span,
    ``start_patch = n_ctx``, gives ``bg_window[n_ctx * PATCH_SIZE - 1]``.
    """
    if start_patch > 0:
        return start_patch * PATCH_SIZE - 1
    return (start_patch + length) * PATCH_SIZE


def _mask_slots(
    spans: list[tuple[int, int]],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spans expanded into the head's fixed ``MAX_MASKED_PATCHES`` (= M) slots.

    ``(mask_idx, valid, d, anchor_step)``, each of length M.  Padded slots gather
    patch 0 and are discarded by ``valid``.

    ``d`` is the distance in patches to the nearest visible evidence on EITHER
    side — never the span length, which confounds one-sided and two-sided cases at
    equal difficulty, and never the arm.

    ``d`` and the anchor disagree by construction, so any metric binned on ``d``
    reports a distance the anchor did not use.  THE ANCHOR IGNORES THE NEAR SIDE,
    being one-sided and left-preferring: the last slot of a two-sided 4-patch span
    sits at ``d = 1`` off its right neighbour while anchoring 4 patches left.  A
    third of supervision anchors farther than the nearest visible evidence;
    ``tests/test_mask_sampler.py`` enumerates the share exactly over ``(T,
    n_spans, length vector, placement branch, gap composition)`` and prints it
    rather than anyone writing it down.  It costs no information — masked rows
    attend to everything — only a harder job for the head's offset
    parameterisation.
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


def _build_sample(
    data: dict[str, np.ndarray],
    icr: float,
    stats: dict[str, dict[str, float]],
    rng: np.random.Generator,
    force_pred_start_hour: float | None = None,
    blind: bool = False,
) -> dict[str, Any]:
    """One training sample from a raw simulator output dict.

    Everything after the simulator call lives here, so every caller reuses one
    feature pipeline.  Keys out: ``patches``, ``targets``, ``n_context_patches``,
    ``bg_formula_data``.  The prediction window may start at any patch-aligned
    position — one model covers day and night windows alike.

    ``data`` carries 1-D length-N ``bg_observed``, ``total_carb``,
    ``total_insulin``, ``total_exercise``, ``hour_of_day``, ``day`` (plus
    ``insulin_resistance`` and ``hgo``, which the input stack does not use).
    ``icr`` is accepted for caller compatibility and not consumed.

    One raw post-noise space, no smoothing: those four RAW signals are the model
    input, and the same raw bg is the BG target and ``last_bg`` (clamped to the
    physical range; the sparse channels floored at 0).  No input/target asymmetry
    and no filter.
    """
    # ``total_exercise`` is the carbohydrate-EQUIVALENT glucose-disposal curve in
    # g/step — what the simulator subtracts from the appearance term — fed at that
    # trained scale and never rescaled to an intensity.
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    bg_raw = data['bg_observed'].astype(np.float32)
    carb_raw = data['total_carb'].astype(np.float32)
    insulin_raw = data['total_insulin'].astype(np.float32)
    exercise_raw = data['total_exercise'].astype(np.float32)
    hour_of_day = data['hour_of_day'].astype(np.float32)
    day_index = data['day'].astype(np.int32)

    N = len(bg_raw)

    # No smoother.  bg is clamped only so it is a legal Kovatchev-f / last_bg
    # argument: the cache is already rail-filtered into the open interval
    # BG_CLAMP_MIN + 1 … BG_CLAMP_MAX - 1, derived the same way at
    # T1DMSIM/cache_simulator.py:180-181 so a clamp change cannot desynchronise
    # the two, but on-the-fly generation and edge reads still need the guard.
    bg = np.clip(bg_raw, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.maximum(carb_raw, 0.0).astype(np.float32)
    insulin = np.maximum(insulin_raw, 0.0).astype(np.float32)
    exercise = np.maximum(exercise_raw, 0.0).astype(np.float32)

    # The canonical input layout: 0 bg_absolute, 1 carb, 2 insulin, 3 exercise
    # (g/step carb-equivalent), 4 bg_masked (the per-PATCH announcement bit,
    # written per window below).  hour_of_day and day_index are metadata for
    # prediction-start selection, not input features.
    features = np.stack([
        bg, carb, insulin, exercise,
        np.zeros_like(bg),
    ], axis=-1)  # (N, N_INPUT_FEATURES)
    # Two different numbers: N_INPUT_FEATURES columns, of which only the LEADING
    # len(CHANNEL_NAMES) are normalized signal channels.  The trailing bg_masked
    # column is a bit and never sees the z-score.
    assert features.shape[-1] == N_INPUT_FEATURES, (
        f"feature stack has {features.shape[-1]} cols, expected "
        f"N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )
    assert len(CHANNEL_NAMES) == BG_MASKED_FEAT < N_INPUT_FEATURES, (
        f"normalized channels must occupy columns 0..{BG_MASKED_FEAT - 1} with "
        f"bg_masked at {BG_MASKED_FEAT}; got len(CHANNEL_NAMES)="
        f"{len(CHANNEL_NAMES)}, N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )

    # Column order == CHANNEL_NAMES order, so the gather never reads a dropped
    # channel.  Exercise is a carb-equivalent disposal rate, not a glucose, so it
    # takes the log1p branch and never the risk transform.
    for c, name in enumerate(CHANNEL_NAMES):
        mean = stats[name]['mean']
        std = stats[name]['std']
        col = features[:, c]
        if name in RISK_SPACE_CHANNELS:
            # bg fed as z(f(bg)), the sole BG input path; clamped above, so f is
            # well-defined.
            col = kovatchev_f_np(col)
        elif name in SPARSE_LOG1P_CHANNELS:
            col = np.log1p(np.maximum(col, 0.0))
        features[:, c] = (col - mean) / (std + 1e-8)

    # One random window per sample: ``n_ctx`` variable, the horizon length fixed
    # and free to start at any patch-aligned position.
    n_ctx = int(rng.integers(MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES + 1))
    long_horizon_patches = max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)
    total_patches_needed = n_ctx + long_horizon_patches
    total_steps_needed = total_patches_needed * PATCH_SIZE

    # A multiple of PATCH_SIZE, so the reshape below is clean; too short for the
    # drawn n_ctx falls back to the minimum context.
    N_trimmed = (N // PATCH_SIZE) * PATCH_SIZE
    if N_trimmed < total_steps_needed:
        n_ctx = MIN_CONTEXT_PATCHES
        total_patches_needed = n_ctx + long_horizon_patches
        total_steps_needed = total_patches_needed * PATCH_SIZE

    n_pred_steps = PREDICTION_PATCHES * PATCH_SIZE
    n_long_horizon_steps = long_horizon_patches * PATCH_SIZE
    # The room requirement is the long horizon, so the trailing GT slice fits.
    if force_pred_start_hour is not None:
        # Falls back to a uniform-random origin when no candidate near the target
        # hour leaves enough room.
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
        # No valid window: raising skips the sample and the DataLoader retries the
        # next index.  Only a pathologically short simulator output reaches here.
        raise RuntimeError(
            f"No prediction window found; trajectory length {N_trimmed}, "
            f"n_ctx={n_ctx}, n_pred={n_pred_steps}"
        )
    start_step = pred_start_step - n_ctx * PATCH_SIZE
    end_step = start_step + total_steps_needed

    # ``bg_window`` covers the whole long-horizon range so the validation rolling
    # pass has raw ground-truth BG at every horizon; the model consumes only the
    # first PREDICTION_PATCHES patches of the prediction zone.
    window = features[start_step:end_step]
    bg_window = bg[start_step:end_step]
    # The announced future-input overrides for the rolling validation and the
    # counterfactual probes, sliced to the long horizon below.  ``predict_rolling``
    # discards the raw side; the counterfactual probe re-normalizes through the
    # same log1p-z stats on the baseline and perturbed sides alike.
    _carb_feat = CHANNEL_TO_FEAT[0]
    _insulin_feat = CHANNEL_TO_FEAT[1]
    _exercise_feat = CHANNEL_TO_FEAT[2]
    carb_norm_window = features[start_step:end_step, _carb_feat]
    insulin_norm_window = features[start_step:end_step, _insulin_feat]
    exercise_norm_window = features[start_step:end_step, _exercise_feat]
    carb_raw_window = carb[start_step:end_step]
    insulin_raw_window = insulin[start_step:end_step]
    exercise_raw_window = exercise[start_step:end_step]

    # A leading-axis slice of a C-contiguous array is contiguous, so this reshape
    # is a view and the sole materializing ``.copy()`` is at the
    # ``torch.from_numpy`` boundary.
    patches_3d = window.reshape(total_patches_needed, PATCH_SIZE, N_INPUT_FEATURES)

    # The model sees seq_len patches, each visible or masked.  Only the first
    # PREDICTION_PATCHES past the context are exposed; anything beyond exists in
    # ``window`` solely to carry the long-horizon ground truth.
    seq_len = n_ctx + PREDICTION_PATCHES
    spans = sample_mask_spans(seq_len, rng)
    masked_patches = np.concatenate(
        [np.arange(s, s + L, dtype=np.int64) for s, L in spans]
    )
    mask_idx, valid, mask_d, anchor_step = _mask_slots(spans, seq_len)

    # Flattened (PATCH_SIZE, N_INPUT_FEATURES) → PATCH_DIM, STEP-MAJOR: a
    # feature's columns are the ``f::N_INPUT_FEATURES`` stride.
    all_patches_t = torch.from_numpy(
        patches_3d[:seq_len].reshape(seq_len, PATCH_SIZE * N_INPUT_FEATURES).copy()
    )
    masked_rows = torch.from_numpy(masked_patches)
    # A masked patch withholds bg — feat 0, the only NON_MASKABLE_FEATS entry —
    # because that is what the model is asked to emit.  The loop touches
    # NON_MASKABLE_FEATS only, so every MASKABLE_FEATS column passes through
    # untouched whatever N_INPUT_FEATURES is.
    for feat_idx in NON_MASKABLE_FEATS:
        all_patches_t[masked_rows, feat_idx::N_INPUT_FEATURES] = 0.0
    # Under the blind policy the same patches withhold their doses too, at the
    # zero-RAW "no dose" fill rather than at z = 0.
    blind_fill = zero_dose_fill(stats) if blind else None
    unblinded_dose_rows = None
    unblinded_dose_patches = None
    if blind_fill is not None:
        blind_flags = torch.zeros(seq_len, dtype=torch.bool)
        blind_flags[masked_rows] = True
        # What the patient ACTUALLY did on the withheld patches, kept before the
        # fill overwrites it in place.  The blind policy is a property of the
        # OBJECTIVE — of which patches the model may read — not of the history, so
        # a caller that un-masks must be able to un-blind with it.  The
        # long-horizon roll is that caller: its context is observed CGM history, so
        # restoring bg while leaving the fill in feats 1-3 hands the roll a history
        # asserting that a meal and a bolus did not happen.
        unblinded_dose_rows = np.asarray(masked_rows, dtype=np.int64).copy()
        unblinded_dose_patches = all_patches_t[masked_rows].clone()
        blind_masked_doses(all_patches_t, blind_flags, blind_fill)
    # The masked set is announced explicitly: masking is not inferable from
    # position, and z = 0 in a withheld bg slot decodes to an ordinary reading
    # (~142 mg/dL on the balanced pool), not a sentinel.  The bit is per PATCH and
    # the layout step-major, so it goes into ALL PATCH_SIZE columns of feat 4 —
    # which is why PATCH_DIM grows by PATCH_SIZE to carry one bit.  Anywhere
    # outside the step-major block breaks PATCH_DIM and the stride idiom above.
    all_patches_t[masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0
    assert all_patches_t.shape[-1] == PATCH_DIM, (
        f"patch row width {all_patches_t.shape[-1]} != PATCH_DIM {PATCH_DIM}"
    )

    # One target per head slot: RAW BG (mg/dL) at each masked patch, NOT
    # f-transformed in the batch — the risk transform is applied once in the loss.
    # Padded slots gather patch 0 exactly as ``mask_idx`` does, and ``valid`` is
    # what discards them.
    pred_start_in_window = n_ctx * PATCH_SIZE
    bg_patches = bg_window[:seq_len * PATCH_SIZE].reshape(seq_len, PATCH_SIZE)
    targets_t = torch.from_numpy(bg_patches[mask_idx].copy())   # (M, S) mg/dL

    # RAW prediction-zone BG for validation / inference — the same raw signal as
    # the target, sourced here and never from a left-padded ``context[-1, -1, 0]``.
    true_bg_traj = bg_window[
        pred_start_in_window:pred_start_in_window + PREDICTION_PATCHES * PATCH_SIZE
    ]
    extended_true_bg_traj = bg_window[
        pred_start_in_window:pred_start_in_window + n_long_horizon_steps
    ]
    # Anchors: ONE-SIDED and LEFT-PREFERRING, read off the RAW mg/dL array with no
    # decode round trip.  ``_anchor_step_for_span`` is the single rule.
    #
    # Padded slots hold ``last_bg`` — an arbitrary but LEGAL mg/dL from this window
    # — so the forward's (B, M) units tripwire never fires on a slot ``valid`` is
    # about to discard.
    #
    # ``last_bg`` is the right-edge case of the same rule: the span of
    # PREDICTION_PATCHES starting at patch n_ctx anchors on
    # ``bg_window[n_ctx * PATCH_SIZE - 1]``.  It stays because the rolling
    # validation and the inference paths forecast from the context edge and read it.
    last_bg = float(bg_window[_anchor_step_for_span(n_ctx, PREDICTION_PATCHES)])
    anchor_bg = np.full(MAX_MASKED_PATCHES, last_bg, dtype=np.float32)
    anchor_bg[valid] = bg_window[anchor_step[valid]]

    # Per-slot TRUE hour of day, at the masked patch's own first step.  Derived
    # instead as ``pred_start_hour + 0.5 * j`` it is off by
    # ``(mask_idx[j] - n_ctx - j) * 0.5`` h under a general masked set, with every
    # shape still matching; a right-edge span reproduces it exactly.
    slot_hour = hour_of_day[start_step + mask_idx * PATCH_SIZE].astype(np.float32)

    # Announced future carbs / insulin / exercise over the long horizon, for the
    # conditioned rolled-forecast override.  Exercise is a PLAN channel exactly
    # like the doses: what the patient announced, never an inferred session.
    _lh = slice(pred_start_in_window, pred_start_in_window + n_long_horizon_steps)
    extended_carb_norm = carb_norm_window[_lh]
    extended_insulin_norm = insulin_norm_window[_lh]
    extended_exercise_norm = exercise_norm_window[_lh]
    extended_carb_raw = carb_raw_window[_lh]
    extended_insulin_raw = insulin_raw_window[_lh]
    extended_exercise_raw = exercise_raw_window[_lh]

    # For nocturnal metric filtering; indexed with the ABSOLUTE step.
    pred_start_hour = float(hour_of_day[pred_start_step])

    bg_formula_data = {
        # The masked set, all (M,) with M = MAX_MASKED_PATCHES.  Padded slots
        # gather patch 0 and carry a legal anchor; ``valid`` is the ONLY thing
        # that discards them, so a loss or metric path that drops it supervises
        # those slots against patch 0.
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
        # BLIND ONLY and UN-COLLATED ONLY: ``collate_fn`` builds its batched dict
        # from an allowlist and carries neither these nor the ``extended_*`` arrays
        # above, both being consumed from a single ``dataset[i]`` by the validation
        # roll.  ABSENT rather than None under the announced policy, which is
        # load-bearing — nothing was overwritten there to undo, and the announced
        # sample must stay byte-identical through the flag
        # (``tests/test_blind_dataset.py`` digests it key by key).
        bg_formula_data['unblinded_dose_rows'] = unblinded_dose_rows
        bg_formula_data['unblinded_dose_patches'] = unblinded_dose_patches

    sample = {
        'patches': all_patches_t.float(),
        'targets': targets_t.float(),
        'n_context_patches': n_ctx,
        'bg_formula_data': bg_formula_data,
    }

    # Cross-window time-of-day probe input, window k+1: window k shifted forward
    # by exactly PREDICTION_PATCHES, so its context ends at pred_start + P and it
    # predicts [pred_start+P, pred_start+2P].  TEACHER-FORCED on the SAME
    # already-normalized ``features`` — a pure re-slice, the normalize crossing
    # above stays the authoritative one.  It carries ONE masked span, the
    # right-edge forecast zone: feat 0 withheld there so no future bg leaks, feat
    # 4 announcing it, carb/insulin/exercise announced throughout.  Window k's
    # general masked set is deliberately NOT reused — the probe compares two
    # forecasts one horizon apart.  Same n_ctx, so the two windows share the
    # PADDING geometry and nothing else: the masked sets differ, and the attention
    # mask is a function of the masked set, so each window gets its own from
    # ``collate_fn``.  Diagnostic only.
    if TIME_PROBE_ENABLED and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
        next_end_patch = n_ctx + 2 * PREDICTION_PATCHES
        next_valid = next_end_patch <= total_patches_needed   # in-range on patches_3d
        next_spans = [(n_ctx, PREDICTION_PATCHES)]
        next_mask_idx, next_slot_valid, next_d, next_anchor_step = _mask_slots(
            next_spans, seq_len
        )
        next_masked_rows = torch.arange(n_ctx, n_ctx + PREDICTION_PATCHES)
        # One horizon further along the SAME window, so window k+1's own step 0 is
        # at PREDICTION_PATCHES * PATCH_SIZE in ``bg_window``.
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
            # Room exists only under NIGHT_LONG_HORIZON_HOURS ==
            # PREDICTION_HORIZON_HOURS.  A finite placeholder, masked out
            # downstream; last_bg reuses window k's legal mg/dL so the forward's
            # units tripwire never fires, and the announcement bit is still written
            # so the placeholder is not a window claiming every patch is observed.
            next_patches_t = torch.zeros(seq_len, PATCH_DIM, dtype=torch.float32)
            next_patches_t[next_masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0
            next_last_bg = last_bg
            next_anchor_bg = np.full(MAX_MASKED_PATCHES, last_bg, dtype=np.float32)
            next_pred_start_hour = pred_start_hour
            next_slot_hour = np.full(MAX_MASKED_PATCHES, pred_start_hour, dtype=np.float32)
        # Both windows of the pair are drawn under one convention, the placeholder
        # branch's masked rows included.
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
    """Collate variable-length samples into a padded batch::

          patches:           (B, max_T, PATCH_DIM)             float32
          targets:           (B, M, PATCH_SIZE)                float32 (mg/dL)
          attn_mask:         (B, max_T, max_T)                 bool
          bg_formula_data:   batched dict (see _build_sample)
          n_context_patches: (B,)                              long
          next_window:       optional batched dict, present iff the samples carry
                             it, with its OWN ``attn_mask``

    Left-pads to ``max_T = max(seq_lens)``, the BATCH maximum, never
    ``MAX_SEQ_LEN``, which appears nowhere in this module: a forward asserting
    ``patches.shape[1] == MAX_SEQ_LEN`` fires on most validation batches, so every
    forward takes ``(B, T, .)`` with ``T <= MAX_SEQ_LEN``.  Padding positions are
    blocked by the per-batch attention mask, whose diagonal is forced True so
    softmax cannot NaN on an all-False row.

    Head-slot indices are rebased onto the PADDED axis here: a sample's masked
    patch ``p`` sits at ``n_pad + p``.  Padded slots keep index 0 — the padded
    tensor's position 0 — and are discarded by ``valid``.
    """
    B = len(samples)
    M = MAX_MASKED_PATCHES
    n_contexts = [s['n_context_patches'] for s in samples]
    seq_lens = [n + PREDICTION_PATCHES for n in n_contexts]
    max_T = max(seq_lens)
    n_pads = [max_T - sl for sl in seq_lens]

    # Left-padding zeros are fine: the attention mask is what disables padding
    # positions.
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

        # Data at [n_pad:max_T], so the prediction horizon sits at the right edge
        # for every sample in the batch.  The model relies on this.
        patches_batch[i, n_pad:, :] = s['patches']

        is_pad[i, :n_pad] = True
        row_valid = valid_batch[i]
        idx = torch.from_numpy(s['bg_formula_data']['mask_idx']) + n_pad
        mask_idx_batch[i, row_valid] = idx[row_valid]
        masked[i, mask_idx_batch[i, row_valid]] = True

    # Visible rows see visible columns, masked rows see everything real, pad rows
    # and columns are blocked, and the diagonal is forced True at EVERY position
    # including padding — that last step is what keeps a pad row, blocked as a row
    # and as a column, from being all-False and NaN-ing the softmax on the
    # direct-to-SDPA bool path.  The assert is this call site's guard on it.
    attn_masks = utils.create_attention_mask_from_visible(~masked, is_pad)
    assert attn_masks.any(dim=-1).all(), "an all-False attention row NaNs softmax"

    # Feat 4 must agree with the masked set that built the attention mask, and
    # nothing else catches a disagreement: z = 0 in a withheld bg slot decodes to
    # an ordinary reading (~142 mg/dL on the balanced pool), so a patch announced
    # visible while its bg is withheld teaches the model to read a fabricated
    # observation and every loss and metric downstream stays finite and plausible.
    # The bit is per patch and the layout step-major, so all PATCH_SIZE columns
    # carry it identically.
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

    # Window k+1 shares window k's n_ctx, so it shares the left-pad ``n_pad`` and
    # the padding geometry — ``is_pad`` below is window k's.  The masked set is NOT
    # shared (k+1 masks its own right-edge forecast span, k whatever its sampler
    # drew), and the attention mask is a function of the masked set, so k+1 gets
    # its own at ``next_window['attn_mask']``.
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

        # The same feat-4-versus-mask agreement as above, against THIS window's
        # masked set.  A patch announced masked while the attention mask still
        # offers it as evidence contradicts itself and nothing downstream notices:
        # the withheld bg is zeroed either way and every loss and metric stays
        # finite.
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

    # The trajectories are raw mg/dL ground truth; ``last_bg`` and
    # ``pred_start_hour`` are per-sample scalars.  The announced-future
    # ``extended_*`` arrays are deliberately NOT stacked here: they are consumed
    # only from UN-COLLATED samples by the rolled-forecast override
    # (``train._make_long_horizon_overrides_fn``, whose callers iterate the dataset
    # directly), so they ride on each sample's own ``bg_formula_data``.
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
