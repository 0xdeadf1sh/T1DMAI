"""
T1DMAI data pipeline: on-the-fly and cached sample generation from T1DMSIM.
Sample dict contract lives on ``T1DMDataset.__getitem__``; space/objective rules
are in this repo's CLAUDE.md.
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

# N≡90 mod 288 for exact hour-of-day coverage to a 336-patch context; must match cache_simulator.py.
ON_THE_FLY_SIM_HOURS: float = 199.5

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


# blosc2: compressed. npy: raw uncompressed memmap. Same meta.json fields and read semantics.
CACHE_FORMAT_BLOSC2 = 'blosc2-ndarray-v1'
CACHE_FORMAT_NPY = 'npy-memmap-v1'
SUPPORTED_CACHE_FORMATS = (CACHE_FORMAT_BLOSC2, CACHE_FORMAT_NPY)


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
        self._cache_pool_size: int | None = None
        self._cache_n_timesteps: int | None = None
        self._cache_meta: dict[str, Any] | None = None
        # name -> (mmap, data_offset_bytes, row_bytes); None under blosc2 or w/o MADV_DONTNEED.
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
            # Carved now pool_size is known, so a held-out seed never reprojects onto a train row.
            self._cache_slab = _cache_slab_geometry(
                self._cache_pool_size, self.cache_partition)

            # A smaller-than-batch pool cycles; benign since each reuse draws a fresh random window.

    def __len__(self) -> int:
        return self.total_steps * self.batch_size

    def _load_cache(self) -> tuple[dict[str, Any], np.ndarray]:
        """Open the cache arrays on first use in this process.

        Per-channel array dict plus the per-patient ICR array. Only the npy format is mapped.
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
                    # Suppresses 128 KB readahead; access is 100% random, so it buys nothing.
                    _madv_random = getattr(mmap, 'MADV_RANDOM', None)
                    if _madv_random is not None:
                        try:
                            arr._mmap.madvise(_madv_random)
                        except (OSError, ValueError, AttributeError):
                            pass
                    # Per-row byte geometry, for __getitem__'s MADV_DONTNEED.
                    mmaps[name] = (
                        arr._mmap, int(arr.offset),
                        int(arr.shape[1] * arr.dtype.itemsize),
                    )
                if self._madv_dontneed is not None:
                    self._cache_mmaps = mmaps
            else:
                import blosc2
                for name in CACHE_CHANNEL_NAMES:
                    # Not mmap_mode='r': blosc2 has no madvise, so mapped pages never drop.
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

        Best-effort: a no-op under blosc2 (never mapped) or without ``MADV_DONTNEED``.
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
            # DISJOINT band: a held-out (val/cal) seed can only resolve to a reserved tail row.
            slab_start, slab_size = self._cache_slab
            cache_idx = slab_start + int(patient_seed % slab_size)
            if self._cache_mmaps is not None:
                # Copies the row out before MADV_DONTNEED, else it aliases the dropped pages.
                data = {
                    name: np.array(cache_arrays[name][cache_idx:cache_idx + 1])[0]
                    for name in CACHE_CHANNEL_NAMES
                }
                self._madvise_row(cache_idx)
            else:
                # blosc2 indexing decompresses into a fresh array; no copy or advise needed.
                data = {
                    name: np.asarray(cache_arrays[name][cache_idx:cache_idx + 1])[0]
                    for name in CACHE_CHANNEL_NAMES
                }
            icr = float(cache_icr[cache_idx])
        else:
            # Keyed off patient_seed so the same idx always resolves the same way.
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

        # Separate substream, so the mode rng above cannot influence window selection.
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
) -> list[tuple[int, int]]:
    """The masked spans of one sample over a ``seq_len``-patch window, left to right.

    ``(start_patch, length)`` pairs; sampler semantics per this repo's CLAUDE.md
    (masked-BG objective section). Mirror a placement change in ``d_balance.d_distribution``.
    """
    lengths_pool = np.asarray(MASK_SPAN_LENGTHS, dtype=np.int64)
    n_spans = int(rng.integers(1, MASK_MAX_SPANS + 1))

    # Rejection is on the LENGTH VECTOR as a whole.
    while True:
        span_lengths = rng.choice(lengths_pool, size=n_spans, replace=True)
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
) -> dict[str, Any]:
    """One training sample from a raw simulator output dict.

    Keys out: ``patches``, ``targets``, ``n_context_patches``, ``bg_formula_data``. ``icr`` is
    accepted for caller compatibility and not consumed. Raw post-noise space, no smoothing.
    """
    # total_exercise is a carb-EQUIVALENT glucose-disposal curve in g/step; never rescaled.
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    bg_raw = data['bg_observed'].astype(np.float32)
    carb_raw = data['total_carb'].astype(np.float32)
    insulin_raw = data['total_insulin'].astype(np.float32)
    exercise_raw = data['total_exercise'].astype(np.float32)
    hour_of_day = data['hour_of_day'].astype(np.float32)
    day_index = data['day'].astype(np.int32)

    N = len(bg_raw)

    # No smoother; clamp only makes bg a legal Kovatchev-f/last_bg argument (edge-read guard).
    bg = np.clip(bg_raw, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.maximum(carb_raw, 0.0).astype(np.float32)
    insulin = np.maximum(insulin_raw, 0.0).astype(np.float32)
    exercise = np.maximum(exercise_raw, 0.0).astype(np.float32)

    # [bg_absolute, carb, insulin, exercise g/step, bg_masked bit written per window below].
    features = np.stack([
        bg, carb, insulin, exercise,
        np.zeros_like(bg),
    ], axis=-1)  # (N, N_INPUT_FEATURES)
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

    # Exercise is a carb-equivalent disposal rate, not a glucose: log1p branch, never risk.
    for c, name in enumerate(CHANNEL_NAMES):
        mean = stats[name]['mean']
        std = stats[name]['std']
        col = features[:, c]
        if name in RISK_SPACE_CHANNELS:
            # bg fed as z(f(bg)), the sole BG input path; clamped above so f is well-defined.
            col = kovatchev_f_np(col)
        elif name in SPARSE_LOG1P_CHANNELS:
            col = np.log1p(np.maximum(col, 0.0))
        features[:, c] = (col - mean) / (std + 1e-8)

    # One random window per sample: n_ctx variable, horizon fixed, patch-aligned start.
    n_ctx = int(rng.integers(MIN_CONTEXT_PATCHES, MAX_CONTEXT_PATCHES + 1))
    long_horizon_patches = max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)
    total_patches_needed = n_ctx + long_horizon_patches
    total_steps_needed = total_patches_needed * PATCH_SIZE

    # Multiple of PATCH_SIZE for a clean reshape; too short for drawn n_ctx falls back to minimum.
    N_trimmed = (N // PATCH_SIZE) * PATCH_SIZE
    if N_trimmed < total_steps_needed:
        n_ctx = MIN_CONTEXT_PATCHES
        total_patches_needed = n_ctx + long_horizon_patches
        total_steps_needed = total_patches_needed * PATCH_SIZE

    n_pred_steps = PREDICTION_PATCHES * PATCH_SIZE
    n_long_horizon_steps = long_horizon_patches * PATCH_SIZE
    # The room requirement is the long horizon, so the trailing GT slice fits.
    if force_pred_start_hour is not None:
        # Falls back to a uniform-random origin when no candidate near the target hour fits.
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
    start_step = pred_start_step - n_ctx * PATCH_SIZE
    end_step = start_step + total_steps_needed

    # bg_window covers the long-horizon range; the model consumes only PREDICTION_PATCHES of it.
    window = features[start_step:end_step]
    bg_window = bg[start_step:end_step]
    # Announced future-input overrides for rolling validation and counterfactual probes.
    _carb_feat = CHANNEL_TO_FEAT[0]
    _insulin_feat = CHANNEL_TO_FEAT[1]
    _exercise_feat = CHANNEL_TO_FEAT[2]
    carb_norm_window = features[start_step:end_step, _carb_feat]
    insulin_norm_window = features[start_step:end_step, _insulin_feat]
    exercise_norm_window = features[start_step:end_step, _exercise_feat]
    carb_raw_window = carb[start_step:end_step]
    insulin_raw_window = insulin[start_step:end_step]
    exercise_raw_window = exercise[start_step:end_step]

    # A leading-axis slice of a C-contiguous array stays contiguous, so this reshape is a view.
    patches_3d = window.reshape(total_patches_needed, PATCH_SIZE, N_INPUT_FEATURES)

    # Only the first PREDICTION_PATCHES past context are exposed; the rest is GT-only.
    seq_len = n_ctx + PREDICTION_PATCHES
    spans = sample_mask_spans(seq_len, rng)
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

    # Announced future carbs/insulin/exercise for the conditioned rolled-forecast override.
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
        'extended_carb_norm': extended_carb_norm.copy(),
        'extended_insulin_norm': extended_insulin_norm.copy(),
        'extended_exercise_norm': extended_exercise_norm.copy(),
        'extended_carb_raw': extended_carb_raw.copy(),
        'extended_insulin_raw': extended_insulin_raw.copy(),
        'extended_exercise_raw': extended_exercise_raw.copy(),
    }
    if unblinded_dose_rows is not None:
        # Blind-only, un-collated; absent (not None) under announced (tests/test_blind_dataset.py).
        bg_formula_data['unblinded_dose_rows'] = unblinded_dose_rows
        bg_formula_data['unblinded_dose_patches'] = unblinded_dose_patches

    sample = {
        'patches': all_patches_t.float(),
        'targets': targets_t.float(),
        'n_context_patches': n_ctx,
        'bg_formula_data': bg_formula_data,
    }

    # Cross-window time-of-day probe: window k+1, teacher-forced, one right-edge span; diagnostic.
    if TIME_PROBE_ENABLED and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
        next_end_patch = n_ctx + 2 * PREDICTION_PATCHES
        next_valid = next_end_patch <= total_patches_needed   # in-range on patches_3d
        next_spans = [(n_ctx, PREDICTION_PATCHES)]
        next_mask_idx, next_slot_valid, next_d, next_anchor_step = _mask_slots(
            next_spans, seq_len
        )
        next_masked_rows = torch.arange(n_ctx, n_ctx + PREDICTION_PATCHES)
        # Window k+1's own step 0 is at PREDICTION_PATCHES*PATCH_SIZE in bg_window.
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
        "feat 4 does not reproduce the sampled mask that built attn_mask"
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
            "next_window feat 4 does not reproduce the mask that built its attn_mask"
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
        'bg_formula_data': bg_formula_batched,
        'n_context_patches': n_ctx_tensor,
        **({'next_window': next_window_batched} if next_window_batched is not None else {}),
    }
