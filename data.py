"""
T1DMAI data pipeline: on-the-fly and cached sample generation from T1DMSIM.
Sample dict contract lives on ``T1DMDataset.__getitem__``; space/objective rules
are in this repo's CLAUDE.md.
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any

from config import (
    PATCH_SIZE, N_INPUT_FEATURES, PATCH_DIM, INPUT_LAYOUT, INPUT_LAYOUTS,
    CHANNEL_TO_FEAT, NON_MASKABLE_FEATS, MASKABLE_FEATS,
    MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES, PREDICTION_PATCHES,
    MASK_MAX_SPANS, MASK_RIGHT_EDGE_QUOTA, MASK_SPAN_LENGTHS, MAX_MASKED_PATCHES,
    PATIENT_UNIFORM_SAMPLE_PROB, N_SKILLS, SKILL_NAMES,
    SIMULATOR_WARMUP_HOURS, NIGHT_LONG_HORIZON_PATCHES,
    TIME_PROBE_ENABLED, TIME_PROBE_CROSS_WINDOW_WEIGHT,
)
import utils
from utils import compute_patient_seed, kovatchev_f_np
from normalization import (
    CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS, LOG_RATIO_REFS,
    log_ratio, normalize,
)

# [*CHANNEL_NAMES, bg_masked]; the bit is never normalized, so the two counts differ on purpose.
BG_MASKED_FEAT = len(CHANNEL_NAMES)
assert N_INPUT_FEATURES == len(CHANNEL_NAMES) + 1, (
    f"N_INPUT_FEATURES={N_INPUT_FEATURES} should be len(CHANNEL_NAMES)="
    f"{len(CHANNEL_NAMES)} normalized channels plus the bg_masked bit"
)

# Checkpoint provenance only, no parameter shape depends on it; an absent key means 'announced'.
MASKED_CHANNEL_POLICY_ANNOUNCED = 'announced'
MASKED_CHANNEL_POLICY_BLIND = 'blind'


def masked_channel_policy(blind: bool) -> str:
    """The masked-channel policy name a ``blind`` flag selects."""
    return MASKED_CHANNEL_POLICY_BLIND if blind else MASKED_CHANNEL_POLICY_ANNOUNCED


def stored_masked_channel_policy(training_config: dict[str, Any] | None) -> str:
    """The masked-channel policy a checkpoint's ``training_config`` records.

    Sole reader of the absent-key convention: absent means ``announced``, never "unknown".
    """
    tc = training_config or {}
    return str(tc.get('masked_channel_policy', masked_channel_policy(blind=False)))


def checkpoint_masked_channel_policy(ckpt: dict[str, Any] | None) -> str:
    """``stored_masked_channel_policy`` over a whole checkpoint dict."""
    return stored_masked_channel_policy((ckpt or {}).get('training_config'))


def zero_dose_fill(stats: dict[str, dict[str, float]]) -> dict[int, float]:
    """``{feat_idx: z}`` over ``MASKABLE_FEATS`` — per-feat ``normalize(0)``, z-space.

    NOT ``z=0``: log1p channels invert z=0 to ``expm1(mean)``, a phantom dose, not an absence.
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

    ``patches`` (..., T, PATCH_DIM) step-major, ``masked`` (..., T) bool. Feats 0/4 untouched.
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

# Clear of training hashed seeds and normalization's +1_000_000 band; backs split-conformal only.
CALIBRATION_SEED_OFFSET: int = 2_000_000


# 3 DISJOINT slabs (cache_idx = slab_start + seed%slab_size) so val/cal never land on a train row.
CACHE_PARTITIONS: tuple[str, ...] = ('train', 'val', 'cal')
CACHE_VAL_SLAB_ROWS: int = 100_000   # reserved tail rows for the validation bands
CACHE_CAL_SLAB_ROWS: int = 100_000   # reserved tail rows for the calibration band


def _cache_slab_geometry(pool_size: int, partition: str) -> tuple[int, int]:
    """``(slab_start, slab_size)`` cache-row band of a partition, half-open, ``slab_size >= 1``.

    Order: train head, then cal, then val tail. Reserves clamped to a third of the pool, min 1 row.
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

    Short-circuits skill sampling to land post-sigmoid skills UNIFORMLY, oversampling tails.
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

    Never cached: stateful, a reused seed's instance is past warmup.
    """
    from T1DMSIM.simulator import T1DMSimulator
    if not uniform_skills:
        return T1DMSimulator(seed=patient_seed)

    # Monkey-patches generate_patient for the constructor only; restored in finally below.
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

    Discards the cold-start day (no prior IOB/COB). Every non-test caller routes through here.
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

    Callers pass the long-horizon footprint as room, so the ground-truth slice fits.
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

    Used for pinned-hour eval (e.g. bedtime). Random within ``tol_hours``, else nearest.
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


# Owned by T1DMSIM: the cache's own channel lists, arm order, format string and skills file.
from T1DMSIM.cache_simulator import (  # noqa: E402
    CACHE_FORMAT_VERSION as CACHE_FORMAT_BLOSC2,
    CHANNEL_NAMES as CACHE_CHANNEL_NAMES,
    N_TAIL_ARMS, SKILLS_FILE, TAIL_ARMS,
)

# Normalized channel -> simulator channel; a point-dose channel keeps its name.
SIM_CHANNEL = {'bg_absolute': 'bg_observed', 'carb_intake': 'total_carb',
               'insulin_combined': 'total_insulin'}
SIM_CHANNEL |= {c: c for c in INPUT_LAYOUTS['events'][1:]}
# What this layout reads from a cache; one built with --events serves both layouts.
READ_CACHE_CHANNELS = CACHE_CHANNEL_NAMES + tuple(
    SIM_CHANNEL[n] for n in CHANNEL_NAMES if SIM_CHANNEL[n] not in CACHE_CHANNEL_NAMES)
# Per-step tail channels a sample needs: the input channels plus the clock the slots read.
READ_TAIL_CHANNELS = tuple(SIM_CHANNEL[n] for n in CHANNEL_NAMES) + ('hour_of_day', 'day')

# The tails end the row, so a sample's context is the last n_ctx patches before the boundary.
CONTEXT_STEPS = MAX_CONTEXT_PATCHES * PATCH_SIZE
TAIL_STEPS = PREDICTION_PATCHES * PATCH_SIZE
SUPPORTED_CACHE_FORMATS = (CACHE_FORMAT_BLOSC2,)


def _row_config():
    """T1DMSIM's ``RowConfig`` for an on-the-fly row, at the geometry this model consumes."""
    from T1DMSIM.cache_simulator import RowConfig, DEFAULT_WARMUP_OFFSET_STEPS
    from T1DMSIM.simulator import DT_MINUTES
    return RowConfig(
        warmup_steps=int(SIMULATOR_WARMUP_HOURS * 60 / DT_MINUTES),
        context_steps=CONTEXT_STEPS,
        max_attempts=1, rail_high=float('inf'), rail_low=float('-inf'),
        hypo_prob=0.0, hypo_min_frac=0.0, hypo_threshold=0.0, seed_salt=0,
        event_refractory_steps=1, events=INPUT_LAYOUT == 'events',
        tail_steps=TAIL_STEPS, warmup_offset_steps=DEFAULT_WARMUP_OFFSET_STEPS,
    )


def simulate_row(patient_seed: int) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
    """One un-cached row through T1DMSIM's own builder: arrays, ICR, skills.

    The cache is the same builder run ahead of time, so the two paths cannot drift.
    """
    from T1DMSIM.cache_simulator import simulate_row as _sim_row
    arrays, stats = _sim_row(int(patient_seed), _row_config())
    return arrays, float(stats['icr']), np.asarray(stats['skills'], dtype=np.float32)


def row_trajectory(row: dict[str, Any], arm: int) -> dict[str, np.ndarray]:
    """Context plus one arm's tail per channel, so the boundary is the trajectory's end."""
    return {
        name: np.concatenate(
            [np.asarray(row[name]), np.asarray(row[f'tail_{name}'][arm])])
        for name in READ_TAIL_CHANNELS
    }


class T1DMDataset(Dataset):
    """T1DM training dataset; length ``total_steps * batch_size``.

    ``seed_offset``/``cache_partition`` pick the seed band and slab; ``blind`` withholds doses too.
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

        # Lazy: populated on first access, so no open cache handle is pickled across the fork.
        self._cache_arrays: dict[str, Any] | None = None
        self._cache_icr: np.ndarray | None = None
        self._cache_skills: np.ndarray | None = None
        self._cache_pool_size: int | None = None
        self._cache_n_timesteps: int | None = None
        self._cache_meta: dict[str, Any] | None = None

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
                'context_steps', 'tail_steps', 'tail_arms', 'tail_channels',
            )
            missing = [k for k in required_keys if k not in meta]
            if missing:
                raise ValueError(
                    f"Cache meta.json at {cache_path!r} is missing keys "
                    f"{missing}. Cache was built by an older cache_simulator "
                    "— rebuild it with T1DMSIM/cache_simulator.py."
                )

            # A missing cache_format key is already rejected by the required-keys check above.
            cache_format = str(meta['cache_format'])
            if cache_format not in SUPPORTED_CACHE_FORMATS:
                raise ValueError(
                    f"Cache cache_format={cache_format!r} is not supported "
                    f"by this version of data.py (expected one of "
                    f"{SUPPORTED_CACHE_FORMATS}). Rebuild the cache with the "
                    "current T1DMSIM/cache_simulator.py."
                )

            # Baked-in values must match runtime assumptions, or the model sees a different input.
            cache_warmup = float(meta['simulator_warmup_hours'])
            if abs(cache_warmup - float(simulator_warmup_hours)) > 1e-6:
                raise ValueError(
                    f"Cache simulator_warmup_hours={cache_warmup} disagrees with "
                    f"dataset simulator_warmup_hours={simulator_warmup_hours}. "
                    "Rebuild the cache with the matching warmup or change the "
                    "dataset/training config."
                )
            # The geometry is read off the cache, never off a simulated-hours scalar held here.
            context_steps = int(meta['context_steps'])
            if (context_steps != int(meta['n_timesteps'])
                    or context_steps % PATCH_SIZE
                    or context_steps < MIN_CONTEXT_PATCHES * PATCH_SIZE):
                raise ValueError(
                    f"Cache context_steps={context_steps} must equal n_timesteps="
                    f"{meta['n_timesteps']}, be a multiple of PATCH_SIZE={PATCH_SIZE} "
                    f"and hold at least MIN_CONTEXT_PATCHES={MIN_CONTEXT_PATCHES} "
                    "patches. Rebuild the cache."
                )
            if int(meta['tail_steps']) != TAIL_STEPS:
                raise ValueError(
                    f"Cache tail_steps={meta['tail_steps']} disagrees with "
                    f"PREDICTION_PATCHES*PATCH_SIZE={TAIL_STEPS}; the horizon is the "
                    "tail. Rebuild the cache with --tail-steps to match."
                )
            if tuple(meta['tail_arms']) != TAIL_ARMS:
                raise ValueError(
                    f"Cache tail_arms={tuple(meta['tail_arms'])} disagrees with "
                    f"{TAIL_ARMS}; the index is the arm's identity. Rebuild the cache."
                )
            tail_absent = [c for c in READ_TAIL_CHANNELS
                           if c not in tuple(meta['tail_channels'])]
            if tail_absent:
                raise ValueError(
                    f"Cache tail_channels lack {tail_absent}, which the "
                    f"{INPUT_LAYOUT!r} layout reads at the horizon "
                    "(cache_simulator.py --events). Rebuild the cache."
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
            n_base = len(CACHE_CHANNEL_NAMES)
            absent = [c for c in READ_CACHE_CHANNELS if c not in cache_channels]
            if cache_channels[:n_base] != CACHE_CHANNEL_NAMES or absent:
                raise ValueError(
                    f"Cache channels={cache_channels} must start with "
                    f"{CACHE_CHANNEL_NAMES}; the {INPUT_LAYOUT!r} layout also needs {absent} "
                    "(cache_simulator.py --events). Rebuild the cache."
                )

            self._cache_pool_size = int(meta['pool_size'])
            self._cache_n_timesteps = int(meta['n_timesteps'])
            self._cache_meta = meta
            # Carved now pool_size is known, so a held-out seed never reprojects onto a train row.
            self._cache_slab = _cache_slab_geometry(
                self._cache_pool_size, self.cache_partition)

            # A smaller-than-batch pool cycles; benign since each reuse draws a fresh random window.

    def __len__(self) -> int:
        return self.total_steps * self.batch_size

    def _load_cache(self) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """Open the cache arrays on first use in this process.

        Per-channel array dict — context plus ``tail_`` arms — with the ICR and skill tables.
        """
        if self._cache_arrays is None:
            assert self.cache_path is not None
            assert self._cache_pool_size is not None
            assert self._cache_n_timesteps is not None
            assert self._cache_meta is not None
            pool = self._cache_pool_size
            import blosc2
            arrays: dict[str, Any] = {}
            wanted = [(n, (pool, self._cache_n_timesteps)) for n in READ_CACHE_CHANNELS]
            wanted += [(f'tail_{n}', (pool, N_TAIL_ARMS, TAIL_STEPS))
                       for n in READ_TAIL_CHANNELS]
            for name, expected_shape in wanted:
                # Not mmap_mode='r': blosc2 has no madvise, so mapped pages never drop.
                arr = blosc2.open(
                    os.path.join(self.cache_path, f'{name}.b2nd'), mode='r')
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
            if icr.shape != (pool,):
                raise ValueError(
                    f"Cache icr.npy has shape {icr.shape}, expected "
                    f"({pool},). Rebuild the cache."
                )
            self._cache_icr = icr
            skills = np.load(os.path.join(self.cache_path, SKILLS_FILE))
            if skills.shape != (pool, N_SKILLS):
                raise ValueError(
                    f"Cache {SKILLS_FILE} has shape {skills.shape}, expected "
                    f"({pool}, {N_SKILLS}) for {list(SKILL_NAMES)}. Rebuild the cache."
                )
            self._cache_skills = skills.astype(np.float32)
        assert self._cache_arrays is not None and self._cache_icr is not None
        assert self._cache_skills is not None
        return self._cache_arrays, self._cache_icr, self._cache_skills

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

        # Separate substream, so the arm draw cannot influence window selection.
        arm = int(np.random.default_rng(patient_seed ^ 0xA12_0F5E1).integers(N_TAIL_ARMS))

        if self.cache_path is not None:
            cache_arrays, cache_icr, cache_skills = self._load_cache()
            assert self._cache_pool_size is not None
            assert self._cache_slab is not None
            # DISJOINT band: a held-out (val/cal) seed can only resolve to a reserved tail row.
            slab_start, slab_size = self._cache_slab
            cache_idx = slab_start + int(patient_seed % slab_size)
            # blosc2 indexing decompresses into a fresh array; no copy or advise needed.
            row = {
                name: np.asarray(cache_arrays[name][cache_idx:cache_idx + 1])[0]
                for name in READ_CACHE_CHANNELS
            }
            row |= {
                f'tail_{name}': np.asarray(
                    cache_arrays[f'tail_{name}'][cache_idx:cache_idx + 1])[0]
                for name in READ_TAIL_CHANNELS
            }
            icr = float(cache_icr[cache_idx])
            skills = cache_skills[cache_idx]
        else:
            row, icr, skills = simulate_row(patient_seed)

        rng = np.random.default_rng(patient_seed ^ 0xDEADBEEF)
        return _build_sample(
            data=row_trajectory(row, arm),
            icr=icr,
            stats=self.stats,
            rng=rng,
            force_pred_start_hour=self.force_pred_start_hour,
            blind=self.blind,
            boundary=True,
            skills=skills,
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

    Seeds are ``master_seed + CALIBRATION_SEED_OFFSET + i``, disjoint from train.
    ``blind`` must match the checkpoint's policy.
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
    pin_right: int | None = None,
) -> list[tuple[int, int]]:
    """The masked spans of one sample over a ``seq_len``-patch window, left to right.

    ``(start_patch, length)`` pairs; semantics in this repo's CLAUDE.md, mirrored in
    ``d_balance``. ``pin_right`` fixes the last span flush right at that length.
    """
    assert pin_right is None or 0 < pin_right <= MAX_MASKED_PATCHES, (
        f"pin_right={pin_right} must fit MAX_MASKED_PATCHES={MAX_MASKED_PATCHES}")
    lengths_pool = np.asarray(MASK_SPAN_LENGTHS, dtype=np.int64)
    n_spans = int(rng.integers(1, MASK_MAX_SPANS + 1))

    # Rejection is on the LENGTH VECTOR as a whole.
    while True:
        span_lengths = rng.choice(lengths_pool, size=n_spans, replace=True)
        if pin_right is not None:
            span_lengths[-1] = pin_right
        if int(span_lengths.sum()) <= MAX_MASKED_PATCHES:
            break

    total_masked = int(span_lengths.sum())
    # One mandatory visible patch per interior boundary, charged up front.
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
    # Pins the LAST span at the final patch; rest compose over the prefix, trailing gap fixed at 0.
    last = int(span_lengths[-1])
    right_edge = pin_right is not None or (
        MASK_RIGHT_EDGE_QUOTA > 0.0 and rng.random() < MASK_RIGHT_EDGE_QUOTA)

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

    ONE-SIDED, LEFT-PREFERRING: last step of the left neighbour, or first of the right at patch 0.
    """
    if start_patch > 0:
        return start_patch * PATCH_SIZE - 1
    return (start_patch + length) * PATCH_SIZE


def _mask_slots(
    spans: list[tuple[int, int]],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spans expanded into the head's fixed ``MAX_MASKED_PATCHES`` (= M) slots.

    ``(mask_idx, valid, d, anchor_step)``, length M; padded slots gather patch 0, valid=False.
    ``d`` (nearest-side) and the anchor (ONE-SIDED left-preferring) disagree by construction.
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
    boundary: bool = False,
    skills: np.ndarray | None = None,
) -> dict[str, Any]:
    """One training sample from a raw simulator output dict.

    Keys out: ``patches``, ``targets``, ``n_context_patches``, ``bg_formula_data``, ``icr``,
    ``skills``. ``boundary`` pins the horizon to the trajectory's last PREDICTION_PATCHES.
    """
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    bg_raw = data['bg_observed'].astype(np.float32)
    hour_of_day = data['hour_of_day'].astype(np.float32)

    N = len(bg_raw)

    # No smoother; clamp only makes bg a legal Kovatchev-f/last_bg argument (edge-read guard).
    bg = np.clip(bg_raw, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    doses = [np.maximum(data[SIM_CHANNEL[name]], 0.0).astype(np.float32)
             for name in CHANNEL_NAMES[1:]]

    # [bg_absolute, *dose channels of the layout, bg_masked bit written per window below].
    features = np.stack([bg, *doses, np.zeros_like(bg)], axis=-1)  # (N, N_INPUT_FEATURES)
    dose_raw = features[:, 1:BG_MASKED_FEAT].copy()
    # Only the LEADING len(CHANNEL_NAMES) columns are normalized; the trailing bg_masked is a bit.
    assert features.shape[-1] == N_INPUT_FEATURES, (
        f"feature stack has {features.shape[-1]} cols, expected "
        f"N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )
    assert len(CHANNEL_NAMES) == BG_MASKED_FEAT < N_INPUT_FEATURES, (
        f"normalized channels must occupy columns 0..{BG_MASKED_FEAT - 1} with "
        f"bg_masked at {BG_MASKED_FEAT}; got len(CHANNEL_NAMES)="
        f"{len(CHANNEL_NAMES)}, N_INPUT_FEATURES={N_INPUT_FEATURES}"
    )

    # A dose channel is a rate or a point dose, not a glucose: log1p branch, never risk.
    for c, name in enumerate(CHANNEL_NAMES):
        mean = stats[name]['mean']
        std = stats[name]['std']
        col = features[:, c]
        if name in RISK_SPACE_CHANNELS:
            # bg fed as z(f(bg)), the sole BG input path; clamped above so f is well-defined.
            col = kovatchev_f_np(col)
        elif name in SPARSE_LOG1P_CHANNELS:
            col = np.log1p(np.maximum(col, 0.0))
        elif name in LOG_RATIO_REFS:
            col = log_ratio(col, LOG_RATIO_REFS[name])
        features[:, c] = (col - mean) / (std + 1e-8)

    n_pred_steps = PREDICTION_PATCHES * PATCH_SIZE
    # Multiple of PATCH_SIZE for a clean reshape.
    N_trimmed = (N // PATCH_SIZE) * PATCH_SIZE

    if boundary:
        # The tails end the row, so the horizon is fixed and the context is what precedes it.
        long_horizon_patches = PREDICTION_PATCHES
        pred_start_step = N_trimmed - n_pred_steps
        ctx_avail = pred_start_step // PATCH_SIZE
        if ctx_avail < MIN_CONTEXT_PATCHES:
            raise RuntimeError(
                f"boundary row holds {ctx_avail} context patches, below "
                f"MIN_CONTEXT_PATCHES={MIN_CONTEXT_PATCHES}"
            )
        n_ctx = int(rng.integers(
            MIN_CONTEXT_PATCHES, min(MAX_CONTEXT_PATCHES, ctx_avail) + 1))
    else:
        # One random window per sample: n_ctx variable, horizon fixed, patch-aligned start.
        n_ctx = int(rng.integers(MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES + 1))
        long_horizon_patches = max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)
        # Too short for the drawn n_ctx falls back to the minimum.
        if N_trimmed < (n_ctx + long_horizon_patches) * PATCH_SIZE:
            n_ctx = MIN_CONTEXT_PATCHES
        n_long_horizon_steps = long_horizon_patches * PATCH_SIZE
        # The room requirement is the long horizon, so the trailing GT slice fits.
        if force_pred_start_hour is not None:
            # Falls back to a uniform-random origin when no candidate near the target fits.
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
            # Raising skips the sample; DataLoader retries the next index.
            raise RuntimeError(
                f"No prediction window found; trajectory length {N_trimmed}, "
                f"n_ctx={n_ctx}, n_pred={n_pred_steps}"
            )

    total_patches_needed = n_ctx + long_horizon_patches
    total_steps_needed = total_patches_needed * PATCH_SIZE
    n_long_horizon_steps = long_horizon_patches * PATCH_SIZE
    start_step = pred_start_step - n_ctx * PATCH_SIZE
    end_step = start_step + total_steps_needed

    # bg_window covers the long-horizon range; the model consumes only PREDICTION_PATCHES of it.
    window = features[start_step:end_step]
    bg_window = bg[start_step:end_step]
    # Announced future-input overrides for rolling validation and counterfactual probes.
    _dose_feats = [CHANNEL_TO_FEAT[ch] for ch in sorted(CHANNEL_TO_FEAT)]
    dose_norm_window = features[start_step:end_step][:, _dose_feats]
    dose_raw_window = dose_raw[start_step:end_step]

    # A leading-axis slice of a C-contiguous array stays contiguous, so this reshape is a view.
    patches_3d = window.reshape(total_patches_needed, PATCH_SIZE, N_INPUT_FEATURES)

    # Only the first PREDICTION_PATCHES past context are exposed; the rest is GT-only.
    seq_len = n_ctx + PREDICTION_PATCHES
    # The tail is a behaviour-off counterfactual, so it is the horizon and never model input.
    spans = sample_mask_spans(
        seq_len, rng, pin_right=PREDICTION_PATCHES if boundary else None)
    masked_patches = np.concatenate(
        [np.arange(s, s + L, dtype=np.int64) for s, L in spans]
    )
    mask_idx, valid, mask_d, anchor_step = _mask_slots(spans, seq_len)

    # step-major PATCH_DIM: a feature's columns are the f::N_INPUT_FEATURES stride.
    all_patches_t = torch.from_numpy(
        patches_3d[:seq_len].reshape(seq_len, PATCH_SIZE * N_INPUT_FEATURES).copy()
    )
    masked_rows = torch.from_numpy(masked_patches)
    # A masked patch withholds bg (feat 0, the only NON_MASKABLE_FEATS entry).
    for feat_idx in NON_MASKABLE_FEATS:
        all_patches_t[masked_rows, feat_idx::N_INPUT_FEATURES] = 0.0
    # Under blind, the same patches withhold doses too, at zero-RAW fill rather than z=0.
    blind_fill = zero_dose_fill(stats) if blind else None
    unblinded_dose_rows = None
    unblinded_dose_patches = None
    if blind_fill is not None:
        blind_flags = torch.zeros(seq_len, dtype=torch.bool)
        blind_flags[masked_rows] = True
        # Kept before the fill overwrites it: the long-horizon roll un-blinds bg from history.
        unblinded_dose_rows = np.asarray(masked_rows, dtype=np.int64).copy()
        unblinded_dose_patches = all_patches_t[masked_rows].clone()
        blind_masked_doses(all_patches_t, blind_flags, blind_fill)
    # Masking is not inferable from position (z=0 decodes to an ordinary reading), hence the bit.
    all_patches_t[masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0
    assert all_patches_t.shape[-1] == PATCH_DIM, (
        f"patch row width {all_patches_t.shape[-1]} != PATCH_DIM {PATCH_DIM}"
    )

    # RAW BG (mg/dL) per head slot, NOT f-transformed here; the risk transform is in the loss.
    pred_start_in_window = n_ctx * PATCH_SIZE
    bg_patches = bg_window[:seq_len * PATCH_SIZE].reshape(seq_len, PATCH_SIZE)
    targets_t = torch.from_numpy(bg_patches[mask_idx].copy())   # (M, S) mg/dL

    # RAW prediction-zone BG, sourced here and never from a left-padded context[-1, -1, 0].
    true_bg_traj = bg_window[
        pred_start_in_window:pred_start_in_window + PREDICTION_PATCHES * PATCH_SIZE
    ]
    extended_true_bg_traj = bg_window[
        pred_start_in_window:pred_start_in_window + n_long_horizon_steps
    ]
    # Padded slots hold last_bg (a LEGAL mg/dL), so the forward's units tripwire never fires.
    last_bg = float(bg_window[_anchor_step_for_span(n_ctx, PREDICTION_PATCHES)])
    anchor_bg = np.full(MAX_MASKED_PATCHES, last_bg, dtype=np.float32)
    anchor_bg[valid] = bg_window[anchor_step[valid]]

    # Per-slot TRUE hour of day, at the masked patch's own first step (not derived/interpolated).
    slot_hour = hour_of_day[start_step + mask_idx * PATCH_SIZE].astype(np.float32)

    # Announced future carbs and insulin for the conditioned rolled-forecast override.
    _lh = slice(pred_start_in_window, pred_start_in_window + n_long_horizon_steps)
    # (steps, dose channels), columns in CHANNEL_TO_FEAT order.
    extended_dose_norm = dose_norm_window[_lh].copy()
    extended_dose_raw = dose_raw_window[_lh].copy()

    # For nocturnal metric filtering; indexed with the ABSOLUTE step.
    pred_start_hour = float(hour_of_day[pred_start_step])

    bg_formula_data = {
        # (M,) with M=MAX_MASKED_PATCHES; padded slots gather patch 0, valid is what drops them.
        'mask_idx': mask_idx,          # (M,) int64  patch index per head slot
        'valid': valid,                # (M,) bool
        'anchor_bg': anchor_bg,        # (M,) float32 mg/dL
        'd': mask_d,                   # (M,) int64  patches to nearest visible, EITHER side
        'slot_hour': slot_hour,        # (M,) float32 true hour of day per slot
        'last_bg': last_bg,
        'true_bg_trajectory': true_bg_traj.copy(),
        'extended_true_bg_trajectory': extended_true_bg_traj.copy(),
        'pred_start_hour': pred_start_hour,
        'extended_dose_norm': extended_dose_norm,
        'extended_dose_raw': extended_dose_raw,
    }
    if INPUT_LAYOUT == 'curves':
        # Named views of the same columns, for the curve-shaped what-if and probe tooling.
        for ch, key in enumerate(('carb', 'insulin')):
            bg_formula_data[f'extended_{key}_norm'] = extended_dose_norm[:, ch].copy()
            bg_formula_data[f'extended_{key}_raw'] = extended_dose_raw[:, ch].copy()
    if unblinded_dose_rows is not None:
        # Blind-only, un-collated; absent (not None) under announced (tests/test_blind_dataset.py).
        bg_formula_data['unblinded_dose_rows'] = unblinded_dose_rows
        bg_formula_data['unblinded_dose_patches'] = unblinded_dose_patches

    sample = {
        'patches': all_patches_t.float(),
        'targets': targets_t.float(),
        'n_context_patches': n_ctx,
        'bg_formula_data': bg_formula_data,
        'icr': float(icr),
        'skills': None if skills is None else np.asarray(skills, dtype=np.float32),
    }

    # Cross-window time-of-day probe: the paired window, teacher-forced, one right-edge span.
    if TIME_PROBE_ENABLED and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
        # A boundary row has nothing past its horizon, so the pair shifts back into the context.
        next_offset = PREDICTION_PATCHES * PATCH_SIZE
        if n_ctx + 2 * PREDICTION_PATCHES > total_patches_needed:
            next_offset = -next_offset
        next_start = start_step + next_offset
        next_valid = next_start >= 0 and next_start + seq_len * PATCH_SIZE <= N_trimmed
        next_spans = [(n_ctx, PREDICTION_PATCHES)]
        next_mask_idx, next_slot_valid, next_d, next_anchor_step = _mask_slots(
            next_spans, seq_len
        )
        next_masked_rows = torch.arange(n_ctx, n_ctx + PREDICTION_PATCHES)
        if next_valid:
            next_bg_window = bg[next_start:next_start + seq_len * PATCH_SIZE]
            next_patches_t = torch.from_numpy(
                features[next_start:next_start + seq_len * PATCH_SIZE]
                .reshape(seq_len, PATCH_SIZE * N_INPUT_FEATURES).copy()
            )
            for feat_idx in NON_MASKABLE_FEATS:
                next_patches_t[next_masked_rows, feat_idx::N_INPUT_FEATURES] = 0.0
            next_patches_t[next_masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0
            next_pred_start_step = pred_start_step + next_offset
            next_last_bg = float(
                next_bg_window[_anchor_step_for_span(n_ctx, PREDICTION_PATCHES)]
            )                                                # raw mg/dL (clamped, physical)
            next_anchor_bg = np.full(MAX_MASKED_PATCHES, next_last_bg, dtype=np.float32)
            next_anchor_bg[next_slot_valid] = next_bg_window[
                next_anchor_step[next_slot_valid]
            ]
            next_pred_start_hour = float(hour_of_day[next_pred_start_step])
            next_slot_hour = hour_of_day[
                next_start + next_mask_idx * PATCH_SIZE
            ].astype(np.float32)
        else:
            # Finite placeholder, masked out downstream; reuses window k's legal last_bg.
            next_patches_t = torch.zeros(seq_len, PATCH_DIM, dtype=torch.float32)
            next_patches_t[next_masked_rows, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0
            next_last_bg = last_bg
            next_anchor_bg = np.full(MAX_MASKED_PATCHES, last_bg, dtype=np.float32)
            next_pred_start_hour = pred_start_hour
            next_slot_hour = np.full(MAX_MASKED_PATCHES, pred_start_hour, dtype=np.float32)
        # Both windows of the pair are drawn under one convention, placeholder branch included.
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
    """Collate variable-length samples into a padded batch.

    Left-pads to ``max_T = max(seq_lens)``, the BATCH max, never ``MAX_SEQ_LEN``. Head-slot
    ``p`` rebases to ``n_pad + p``; padded slots keep index 0, discarded by ``valid``.
    """
    B = len(samples)
    M = MAX_MASKED_PATCHES
    n_contexts = [s['n_context_patches'] for s in samples]
    seq_lens = [n + PREDICTION_PATCHES for n in n_contexts]
    max_T = max(seq_lens)
    n_pads = [max_T - sl for sl in seq_lens]

    # Left-padding zeros are fine: the attention mask is what disables padding positions.
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

        # Data at [n_pad:max_T], so the prediction horizon sits at the right edge for every sample.
        patches_batch[i, n_pad:, :] = s['patches']

        is_pad[i, :n_pad] = True
        row_valid = valid_batch[i]
        idx = torch.from_numpy(s['bg_formula_data']['mask_idx']) + n_pad
        mask_idx_batch[i, row_valid] = idx[row_valid]
        masked[i, mask_idx_batch[i, row_valid]] = True

    # Diagonal is forced True at EVERY position (padding included) so no row is all-False.
    attn_masks = utils.create_attention_mask_from_visible(~masked, is_pad)
    assert attn_masks.any(dim=-1).all(), "an all-False attention row NaNs softmax"

    # The skill head pools these patches alone: real readings the model was allowed to see.
    pool_mask = (~masked) & (~is_pad)
    skills_valid = torch.tensor(
        [s.get('skills') is not None for s in samples], dtype=torch.bool)
    skills_batch = torch.zeros(B, N_SKILLS, dtype=torch.float32)
    for i, s in enumerate(samples):
        if skills_valid[i]:
            skills_batch[i] = torch.from_numpy(
                np.asarray(s['skills'], dtype=np.float32))

    # Feat 4 must agree with the masked set that built attn_mask; nothing else catches drift.
    _bit = patches_batch[..., BG_MASKED_FEAT::N_INPUT_FEATURES]   # (B, max_T, PATCH_SIZE)
    assert _bit.shape[-1] == PATCH_SIZE, (
        f"feat {BG_MASKED_FEAT} stride slice gave {_bit.shape[-1]} columns, "
        f"expected PATCH_SIZE={PATCH_SIZE} — the step-major layout is broken"
    )
    assert bool((_bit == _bit[..., :1]).all()), (
        "the bg_masked bit differs across a patch's step-major columns"
    )
    assert torch.equal(_bit[..., 0], masked.to(_bit.dtype)), (
        "the bg_masked feat does not reproduce the sampled mask that built attn_mask"
    )

    # Window k+1 shares k's n_ctx/n_pad but not the masked set, so it gets its own attn_mask.
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

        # Same feat-4-versus-mask agreement as above, against THIS window's masked set.
        _nw_bit = nw_patches[..., BG_MASKED_FEAT::N_INPUT_FEATURES]   # (B, max_T, PATCH_SIZE)
        assert _nw_bit.shape[-1] == PATCH_SIZE, (
            f"feat {BG_MASKED_FEAT} stride slice gave {_nw_bit.shape[-1]} columns, "
            f"expected PATCH_SIZE={PATCH_SIZE} — the step-major layout is broken"
        )
        assert bool((_nw_bit == _nw_bit[..., :1]).all()), (
            "the next_window bg_masked bit differs across a patch's step-major columns"
        )
        assert torch.equal(_nw_bit[..., 0], nw_masked.to(_nw_bit.dtype)), (
            "the next_window bit does not reproduce the mask that built its attn_mask"
        )

        next_window_batched = {
            'patches': nw_patches,  # (B, max_T, PATCH_DIM)
            'attn_mask': nw_attn_masks,  # (B, max_T, max_T) bool
            'mask_idx': nw_mask_idx,  # (B, M) long, padded axis
            'valid_slots': nw_valid_slots,  # (B, M) bool
            'anchor_bg': torch.from_numpy(
                np.stack([s['next_window']['anchor_bg'] for s in samples])),  # (B, M) mg/dL
            'd': torch.from_numpy(
                np.stack([s['next_window']['d'] for s in samples])),  # (B, M) long
            'slot_hour': torch.from_numpy(
                np.stack([s['next_window']['slot_hour'] for s in samples])),  # (B, M) hours
            'last_bg': torch.tensor(
                [s['next_window']['last_bg'] for s in samples], dtype=torch.float32),  # (B,) mg/dL
            'pred_start_hour': torch.tensor(  # (B,)
                [s['next_window']['pred_start_hour'] for s in samples], dtype=torch.float32),
            'valid': torch.tensor(
                [s['next_window']['valid'] for s in samples], dtype=torch.bool),  # (B,)
        }

    # extended_* arrays are deliberately NOT stacked: only consumed from UN-COLLATED samples.
    bg_formula_batched: dict[str, Any] = {
        # The masked set, on the PADDED patch axis. Every one of these is (B, M).
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
        'pool_mask': pool_mask,
        'skills': skills_batch,
        'skills_valid': skills_valid,
        'bg_formula_data': bg_formula_batched,
        'n_context_patches': n_ctx_tensor,
        **({'next_window': next_window_batched} if next_window_batched is not None else {}),
    }
