"""
T1DMAI Training Loop — Muon + AdamW (risk-space BG redesign).
=============================================================

What this file does
-------------------
End-to-end training driver.  Reads CLI flags over the ``config.py`` constants,
builds the model + data
pipeline + optimizers, runs a streaming training loop with EMA validation, and
writes per-step CSV logs plus periodic checkpoints.

Loss (every step) — see ``risk_loss.risk_total_loss``:

    L_total = learned Kendall-Gal combine of
                L_Q  (quantile pinball in Kovatchev RISK space)
                L_D  (DILATE = DILATE_ALPHA·shape + (1-DILATE_ALPHA)·TDI, on the
                      RISK-space median)
              weighted by two learned log-σ on a small ``KendallGalWeighting``
              module (its own AdamW group, weight_decay 0, excluded from the EMA).

The model emits per-step quantiles ``q_tau`` and a ``median`` in RISK space; the
forecast target stays raw mg/dL in the batch and is f-transformed exactly once
inside ``risk_total_loss``.  The headline BG forecast at validation /
inference is ``kovatchev_f_inv(median)``.

Numerical care taken (fp32-native — no autocast / bf16):
    * NaN / Inf guards on both the loss and the post-backward gradient norm skip
      the optimizer step and decay momentum buffers by ½ instead of poisoning
      state (soft-DTW at low γ can still Inf in fp32).
    * EMA shadow weights swapped in around validation; live weights restored
      before training continues.

Logs written:
    ``logs/training_log.csv``       — per-step (loss_total, loss_Q, loss_D[+
                                       shape/tdi], log_sigma_Q, log_sigma_D, …).
    ``logs/validation_log.csv``     — every validation step.
    ``logs/training_summary.json``  — periodic snapshot of progress.
    ``logs/resolved_config.json``   — the resolved CLI > config.py config.

Usage::

    python train.py
    python train.py --master-seed 42 --total-steps 100000 --batch-size 512
"""

import argparse
import contextlib
import csv
import json
import math
import os
import random
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from config import (
    MASTER_SEED, DETERMINISTIC, TOTAL_STEPS, BATCH_SIZE, NUM_WORKERS,
    MUON_LR, MUON_MOMENTUM, MUON_NS_ITERATIONS, MUON_WEIGHT_DECAY,
    ADAM_LR, ADAM_BETAS, ADAM_WEIGHT_DECAY, ADAM_EPS,
    WARMUP_STEPS, LR_MIN_RATIO, WEIGHT_DECAY_SCHEDULE_CORRECTION, GRADIENT_CLIP_NORM,
    PREDICTION_PATCHES,
    MAX_CONTEXT_PATCHES, MIN_CONTEXT_PATCHES,
    LOG_INTERVAL, CHECKPOINT_INTERVAL, VALIDATION_INTERVAL,
    VALIDATION_N_PATIENTS, NORM_STATS_FILE, PATCH_SIZE,
    N_INPUT_FEATURES,
    PATIENT_UNIFORM_SAMPLE_PROB, SIMULATOR_WARMUP_HOURS,
    EMA_DECAY,
    CF_CARB_BOLUS_G, CF_INSULIN_BOLUS_U,
    BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD,
    HYPO_ALARM_QUANTILE_TAU, HYPER_ALARM_QUANTILE_TAU,
    EXCURSION_PRECISION_TOLERANCE_MGDL,
    NOCTURNAL_START_HOUR, NOCTURNAL_END_HOUR,
    PREDICTION_HORIZON_HOURS, NIGHT_LONG_HORIZON_HOURS, NIGHT_LONG_HORIZON_PATCHES,
    QUANTILE_LEVELS, N_QUANTILES,
    TIME_PROBE_LOSS_WEIGHT, TIME_PROBE_N_BINS,
    TIME_PROBE_LABEL_SMOOTH_BINS, TIME_PROBE_CROSS_WINDOW_WEIGHT, TIME_PROBE_CROSS_WINDOW_FRACTION,
)

# Schema-version stamps recorded as checkpoint provenance metadata: they are
# written into every checkpoint dict so the arch / loss schema a checkpoint was
# produced under is self-describing. config.py is the authoritative single source
# of truth and defines these; the local sentinels remain only as a defensive
# fallback so an older config that predates the stamps still imports. When config
# defines them (the norm), config wins.
try:
    from config import ARCH_VERSION as _CFG_ARCH_VERSION  # type: ignore[attr-defined]
except ImportError:
    _CFG_ARCH_VERSION = None
try:
    from config import LOSS_SCHEMA as _CFG_LOSS_SCHEMA  # type: ignore[attr-defined]
except ImportError:
    _CFG_LOSS_SCHEMA = None
ARCH_VERSION = _CFG_ARCH_VERSION if _CFG_ARCH_VERSION is not None else 'risk-v3'
LOSS_SCHEMA = _CFG_LOSS_SCHEMA if _CFG_LOSS_SCHEMA is not None else 'kendall-pinball-dilate-v3'

from utils import (
    ModelEMA, kovatchev_f_inv,
    time_of_day_bin_ce, time_of_day_decode_bins,
    time_cross_window_consistency_loss, time_cross_window_jump_hours, time_inter_patch_jump_hours,
    circular_hour_error, circular_bias_hours, circular_std_hours,
)

from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

# Central τ → ascending-quantile-axis lookup. The model emits q_tau on the
# ascending QUANTILE_LEVELS axis (index-for-index); these locate the band edges
# the marginal-coverage / pinball diagnostics read.
_TAU_LO_IDX = QUANTILE_LEVELS.index(0.05)
_TAU_HI_IDX = QUANTILE_LEVELS.index(0.95)
# Inner-50% band edges (τ.25/.75) for the diagnostic inner50_cov metric.
_TAU_INNER_LO_IDX = QUANTILE_LEVELS.index(0.25)
_TAU_INNER_HI_IDX = QUANTILE_LEVELS.index(0.75)
# Clinical hypo/hyper ALARM band edges. Detection keys off the band edges, not
# the median: hypo off the τ=HYPO_ALARM_QUANTILE_TAU LOWER edge (the conservative
# low-envelope call), hyper off the τ=HYPER_ALARM_QUANTILE_TAU UPPER edge. Indices
# via QUANTILE_LEVELS.index of the config taus — never a bare literal.
_HYPO_BAND_IDX = QUANTILE_LEVELS.index(HYPO_ALARM_QUANTILE_TAU)
_HYPER_BAND_IDX = QUANTILE_LEVELS.index(HYPER_ALARM_QUANTILE_TAU)

# Per-horizon excursion-detection DISPLAY targets ``(base@30min, slope_per_30min,
# floor)`` for the `Excursions by Horizon` table section. DISPLAY-ONLY (they set
# row colour / Target text, never the loss / CSV / checkpoint selection), so they
# live in train.py rather than config.py, since they are display-only. The target
# declines with horizon (near-term detection is largely fixed by insulin-on-board;
# long-horizon is information-limited).
EXCURSION_TARGET_HYPO_RECALL = (90.0, 10.0, 50.0)
EXCURSION_TARGET_HYPO_PRECISION = (75.0, 8.0, 45.0)
EXCURSION_TARGET_HYPER_RECALL = (85.0, 8.0, 55.0)
EXCURSION_TARGET_HYPER_PRECISION = (85.0, 8.0, 55.0)


class _OffsetSampler(Sampler):
    """Sequential sampler that starts from a given offset.

    A fresh run always uses offset 0; the offset parameter is retained as a
    generic capability (skip the first ``offset`` dataset indices) and is not
    exercised by the standard training path.
    """

    def __init__(self, total_len: int, offset: int = 0) -> None:
        self.total_len = total_len
        self.offset = offset

    def __iter__(self):
        return iter(range(self.offset, self.total_len))

    def __len__(self) -> int:
        return self.total_len - self.offset


def setup_determinism(seed: int) -> None:
    """Pin every RNG and disable nondeterministic kernels for reproducibility.

    Gated on ``config.DETERMINISTIC`` by the caller. Seeds python/numpy/torch
    (CPU + all CUDA devices), forces cuDNN into deterministic mode, disables
    TF32 on both matmul and cuDNN, and asks PyTorch to use deterministic
    algorithms (``warn_only`` so an op lacking a deterministic CUDA kernel —
    e.g. the SDPA flash path — warns instead of raising).

    Caveat: the SDPA flash / memory-efficient attention BACKWARD has no
    deterministic CUDA kernel, so GPU training is reproducible only to within
    numerical noise — not bit-exact. The data stream (every RNG + the per-worker
    DataLoader seeding) IS fully reproducible, which is the dominant source of
    run-to-run variance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def _worker_init_fn(worker_id: int) -> None:
    """Seed a DataLoader worker's global numpy + random RNGs.

    The dominant variance source — the simulated data stream — is thus fully
    seeded: the per-sample simulator RNG is already keyed deterministically on
    ``compute_patient_seed``, and this closes the incidental global-numpy /
    random gap (e.g. carb-noise jitter) so each worker is reproducible.
    ``torch.initial_seed()`` is PyTorch's per-worker base (already derived from
    the base seed + worker id), so deriving from it keeps workers distinct.
    """
    s = torch.initial_seed() % 2 ** 31
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


from model import T1DMAI
from muon import Muon
from normalization import (
    load_normalization_stats, compute_normalization_stats,
    save_normalization_stats, CHANNEL_NAMES,
)
from data import T1DMDataset, collate_fn
from risk_loss import risk_total_loss, KendallGalWeighting
import cg_ega

# EMA smoothing factor for loss trend (higher = slower/smoother)
LOSS_EMA_ALPHA = 0.98

# Canonical step length in minutes. One patch (PATCH_SIZE steps) spans 30 min,
# so each step is 30 / PATCH_SIZE = 5 min. Used as the fallback wherever the
# batched ``bg_formula_data`` does not carry a ``dt_minutes`` scalar — collate_fn
# packs only {last_bg, true_bg_trajectory, extended_true_bg_trajectory,
# pred_start_hour}, NOT dt_minutes.
STEP_MINUTES = 30.0 / PATCH_SIZE  # == 5.0


def _dt_minutes(bg_formula_data: dict[str, Any]) -> float:
    """Per-step minutes from ``bg_formula_data`` if present, else the canonical
    ``STEP_MINUTES`` (collate_fn does not pack ``dt_minutes``)."""
    v = bg_formula_data.get('dt_minutes')
    return float(v) if v is not None else float(STEP_MINUTES)


# ============================================================================
# BG-zone constants (shared by validation metrics)
# ============================================================================

BG_HORIZONS_MIN: tuple[int, ...] = (30, 60, 120, 180, 360, 480)
# Horizons at which per-horizon (un-pooled) Clarke-A and MARD are reported, so
# the metrics can be read against the single-horizon published SOTA bars.
EVALFIX_CLARKE_MARD_HORIZONS_MIN: tuple[int, ...] = (30, 60, 120)
# Marginal coverage horizons: the per-(h, τ) empirical coverage of the central
# 90% band (τ.05/.95). MARGINAL — per-step, NOT joint-over-horizon.
COVERAGE_HORIZONS_MIN: tuple[int, ...] = (30, 60, 120)

BG_TARGET_LO = 70.0
BG_TARGET_HI = 180.0

# Excursion-amplitude validation diagnostic (the over/under-dispersion the
# per-PATCH trend_amp_ratio misses). Per window the NET peak deviation from
# last_bg over the whole horizon is taken at the TRUE peak index; a window
# counts only if |true peak deviation| > EXC_AMP_MIN_MGDL (a real excursion —
# avoids the small-denominator peak-ratio blow-up). exc_amp_ratio =
# std(pred_exc)/std(true_exc) (target 1.0; >1 over-disperses/overshoots, <1
# damps). over/undershoot fractions count windows whose pred/true peak ratio
# sits beyond these bounds.
EXC_AMP_MIN_MGDL = 15.0
EXC_AMP_OVERSHOOT_RATIO = 1.25
EXC_AMP_UNDERSHOOT_RATIO = 0.75


def _excursion_bucket_horizons(active_patches: int) -> list[int]:
    """Disjoint per-patch (30-min) bucket end-horizons, in minutes, for the
    per-horizon hypo/hyper recall & precision metrics. Patch ``p`` (0-indexed)
    covers steps ``[p·S, (p+1)·S)`` — minutes ``[p·30, (p+1)·30)`` at the
    canonical 5-min step — and is labelled by its end horizon ``(p+1)·30``. For
    the default 2 h horizon this is ``[30, 60, 90, 120]``. Shared by the metric
    emitter (compute_learning_metrics) and finalizer (_run_validation) so the
    ``@{h}`` count keys line up."""
    return [(p + 1) * PATCH_SIZE * 5 for p in range(active_patches)]


def _excursion_target(spec: tuple[float, float, float], horizon_min: int) -> float:
    """Per-horizon excursion-detection DISPLAY target (percent).

    ``spec`` is ``(base@30min, slope_per_30min, floor)``; the target declines
    linearly with horizon and floors:
    ``max(floor, base - slope * (horizon_min/30 - 1))``. DISPLAY-ONLY — colours
    the ``Excursions by Horizon`` rows; never feeds the loss or checkpoint
    selection. Rationale + values: the module-local EXCURSION_TARGET_* constants.
    """
    base, slope, floor = spec
    return max(floor, base - slope * (horizon_min / 30.0 - 1.0))


# ============================================================================
# Validation: forecast → mg/dL bridge
# ============================================================================

def _median_to_mgdl(median_risk: torch.Tensor) -> torch.Tensor:
    """RISK-space per-step median ``(B, P, S)`` → mg/dL ``(B, P*S)``.

    ``kovatchev_f_inv`` clamps the risk input first, then forms / exps the base,
    then clamps the mg/dL output to ``[BG_CLAMP_MIN, BG_CLAMP_MAX]`` — the SOLE
    (risk)→(mg/dL) crossing for the headline forecast. The flattened result is
    the SOLE ``pred_bg`` consumed by every BG metric below.

    Args:
        median_risk: ``(B, P, S)`` per-step median in Kovatchev risk space.

    Returns:
        pred_bg: ``(B, P*S)`` mg/dL, asserted within the physical clamp band.
    """
    B, P, S = median_risk.shape
    pred_bg = kovatchev_f_inv(median_risk.reshape(B, P * S))
    assert bool(((pred_bg >= BG_CLAMP_MIN - 1e-3) & (pred_bg <= BG_CLAMP_MAX + 1e-3)).all()), (
        "pred_bg out of physical clamp band — f_inv must clamp to "
        f"[{BG_CLAMP_MIN}, {BG_CLAMP_MAX}]"
    )
    return pred_bg


def compute_learning_metrics(
    pred_bg: torch.Tensor,            # (B, P*S) mg/dL — f_inv(median)
    q_mgdl: dict[str, torch.Tensor],  # 'lo'/'hi' band edges (B, P*S) mg/dL, or {}
    bg_formula_data: dict[str, Any],
    active_patches: int,
    hypo_threshold: float = BG_HYPO_THRESHOLD,
    hyper_threshold: float = BG_HYPER_THRESHOLD,
) -> dict[str, float]:
    """Diagnostic BG-forecast metrics in mg/dL space (sums + counts for clean
    batch aggregation), all from the SINGLE headline forecast ``pred_bg`` =
    ``f_inv(median)``.

    Args:
        pred_bg: ``(B, P*S)`` headline BG forecast (mg/dL).
        q_mgdl: mg/dL band-edge dict. ``'lo'/'hi'`` (B, P*S) = the central 90%
            interval (τ.05/.95) for marginal coverage; ``'inner_lo'/'inner_hi'``
            (τ.25/.75) for inner50_cov; ``'hypo_lo'`` (τ=HYPO_ALARM_QUANTILE_TAU
            lower edge) and ``'hyper_hi'`` (τ=HYPER_ALARM_QUANTILE_TAU upper edge)
            — the band edges the clinical hypo/hyper recall+precision key off (NOT
            the median). ``{}`` skips the coverage diagnostics.
        bg_formula_data: dict carrying ``true_bg_trajectory`` ``(B, ≥P*S)`` and
            ``last_bg`` ``(B,)`` (mg/dL) and ``dt_minutes`` (scalar).
        active_patches: number of prediction patches scored (PREDICTION_PATCHES).
        hypo_threshold / hyper_threshold: clinical crossing thresholds (mg/dL).

    Returns:
        dict of metric sums / counts (finalized in ``_run_validation``).
    """
    assert pred_bg.ndim == 2, f"pred_bg must be (B, P*S), got {tuple(pred_bg.shape)}"
    B = pred_bg.shape[0]
    P = active_patches
    S = PATCH_SIZE
    dt = _dt_minutes(bg_formula_data)
    last_bg = bg_formula_data['last_bg'].float()                       # (B,)
    true_bg = bg_formula_data['true_bg_trajectory'][:, :P * S].float()  # (B, P*S)
    last_bg_col = last_bg.unsqueeze(1)                                  # (B, 1)
    total_steps_h = P * S

    out: dict[str, float] = {}

    # Stage every on-device scalar reduction here (as a 0-dim tensor) and flush
    # the whole batch to the host with a SINGLE ``torch.stack(...).tolist()`` sync
    # at the end — one cuda->cpu transfer instead of ~90 per-metric ``.item()``
    # calls. Bit-identical: the reductions themselves are unchanged (same op, same
    # device); only the host transfer is batched. Bool/int count-sums are cast to
    # fp32 (counts << 2**24, so exact) so they stack with the float sums.
    _g: dict[str, torch.Tensor] = {}

    def _stage(key: str, t: torch.Tensor) -> None:
        """Defer a 0-dim reduction ``t`` for the single batched host transfer."""
        _g[key] = t if t.dtype == torch.float32 else t.float()

    for h_min in BG_HORIZONS_MIN:
        h_idx = (h_min // int(dt)) - 1
        if 0 <= h_idx < total_steps_h:
            diff = pred_bg[:, h_idx] - true_bg[:, h_idx]
            _stage(f'bg_rmse_{h_min}_sq_sum', diff.pow(2).sum())
            out[f'bg_rmse_{h_min}_cnt'] = float(B)
            _stage(f'bg_mae_{h_min}_abs_sum', diff.abs().sum())
            out[f'bg_mae_{h_min}_cnt'] = float(B)
        else:
            out[f'bg_rmse_{h_min}_sq_sum'] = 0.0
            out[f'bg_rmse_{h_min}_cnt'] = 0.0
            out[f'bg_mae_{h_min}_abs_sum'] = 0.0
            out[f'bg_mae_{h_min}_cnt'] = 0.0

    abs_rel = (pred_bg - true_bg).abs() / true_bg.clamp(min=1.0)
    _stage('mard_sum', abs_rel.sum())
    out['mard_cnt'] = float(abs_rel.numel())

    pred_in = ((pred_bg >= BG_TARGET_LO) & (pred_bg <= BG_TARGET_HI)).float()
    true_in = ((true_bg >= BG_TARGET_LO) & (true_bg <= BG_TARGET_HI)).float()
    pred_frac = pred_in.mean(dim=1)
    true_frac = true_in.mean(dim=1)
    _stage('tir_err_sum', (pred_frac - true_frac).abs().sum())
    out['tir_err_cnt'] = float(B)
    _stage('pred_tir_sum', pred_frac.sum())
    _stage('true_tir_sum', true_frac.sum())

    pred_below = (pred_bg < BG_TARGET_LO).float().mean(dim=1)
    true_below = (true_bg < BG_TARGET_LO).float().mean(dim=1)
    pred_above = (pred_bg > BG_TARGET_HI).float().mean(dim=1)
    true_above = (true_bg > BG_TARGET_HI).float().mean(dim=1)
    _stage('tbr_err_sum', (pred_below - true_below).abs().sum())
    _stage('tar_err_sum', (pred_above - true_above).abs().sum())

    # Excursion detection counts. Denominators are the strict clinical
    # crossings; recall and precision are the plain confusion-matrix quantities
    # off the band edges (no tolerance).
    #
    # The clinical hypo/hyper detectors key off the BAND EDGES, not the median:
    # hypo fires when the τ=HYPO_ALARM_QUANTILE_TAU LOWER edge (``pred_lo``) dips
    # below the threshold (the conservative low-envelope call); hyper fires when
    # the τ=HYPER_ALARM_QUANTILE_TAU UPPER edge (``pred_hi``) rises above the
    # threshold. Truth stays off the TRUE bg (unchanged).
    pred_lo = q_mgdl['hypo_lo']    # (B, P*S) mg/dL, τ=HYPO_ALARM_QUANTILE_TAU lower band edge
    pred_hi = q_mgdl['hyper_hi']   # (B, P*S) mg/dL, τ=HYPER_ALARM_QUANTILE_TAU upper band edge
    assert bool(((pred_lo >= BG_CLAMP_MIN - 1e-3) & (pred_lo <= BG_CLAMP_MAX + 1e-3)).all()), (
        f"pred_lo out of physical clamp band [{BG_CLAMP_MIN}, {BG_CLAMP_MAX}]"
    )
    assert bool(((pred_hi >= BG_CLAMP_MIN - 1e-3) & (pred_hi <= BG_CLAMP_MAX + 1e-3)).all()), (
        f"pred_hi out of physical clamp band [{BG_CLAMP_MIN}, {BG_CLAMP_MAX}]"
    )

    # RECALL is strict TP (the conservative band edge is deliberate). PRECISION
    # forgives a predicted excursion whose band edge is within
    # EXCURSION_PRECISION_TOLERANCE_MGDL of the true value (near-threshold CGM noise
    # is not a false alarm); tolerance 0.0 recovers the strict count.
    true_hypo = true_bg < hypo_threshold
    pred_hypo = pred_lo < hypo_threshold
    hypo_tp = true_hypo & pred_hypo
    close_prec_hypo = (pred_lo - true_bg).abs() <= EXCURSION_PRECISION_TOLERANCE_MGDL
    _stage('hypo_true', true_hypo.sum())
    _stage('hypo_pred', pred_hypo.sum())
    _stage('hypo_recall_hit', hypo_tp.sum())
    _stage('hypo_prec_hit', (pred_hypo & (true_hypo | close_prec_hypo)).sum())

    true_hyper = true_bg > hyper_threshold
    pred_hyper = pred_hi > hyper_threshold
    hyper_tp = true_hyper & pred_hyper
    close_prec_hyper = (pred_hi - true_bg).abs() <= EXCURSION_PRECISION_TOLERANCE_MGDL
    _stage('hyper_true', true_hyper.sum())
    _stage('hyper_pred', pred_hyper.sum())
    _stage('hyper_recall_hit', hyper_tp.sum())
    _stage('hyper_prec_hit', (pred_hyper & (true_hyper | close_prec_hyper)).sum())

    # Per-horizon (disjoint 30-min patch buckets) excursion counts.
    for _p, _h in enumerate(_excursion_bucket_horizons(P)):
        _s0, _s1 = _p * S, (_p + 1) * S
        _th, _ph = true_hypo[:, _s0:_s1], pred_hypo[:, _s0:_s1]
        _cph = close_prec_hypo[:, _s0:_s1]
        _stage(f'hypo_true@{_h}', _th.sum())
        _stage(f'hypo_pred@{_h}', _ph.sum())
        _stage(f'hypo_recall_hit@{_h}', (_th & _ph).sum())
        _stage(f'hypo_prec_hit@{_h}', (_ph & (_th | _cph)).sum())
        _yt, _yp = true_hyper[:, _s0:_s1], pred_hyper[:, _s0:_s1]
        _cpy = close_prec_hyper[:, _s0:_s1]
        _stage(f'hyper_true@{_h}', _yt.sum())
        _stage(f'hyper_pred@{_h}', _yp.sum())
        _stage(f'hyper_recall_hit@{_h}', (_yt & _yp).sum())
        _stage(f'hyper_prec_hit@{_h}', (_yp & (_yt | _cpy)).sum())

    # CG-EGA (Kovatchev 2004; dotXem adaptation to PREDICTION). Per-region
    # AP/BE/EP counts, accumulated across batches then finalized to fractions.
    cg = cg_ega.cg_ega_counts(
        pred_bg.detach().cpu().numpy(),
        true_bg.detach().cpu().numpy(),
        last_bg.detach().cpu().numpy(),
        freq_min=dt,
    )
    for _ck, _cv in cg.items():
        out[f'cgega_{_ck}'] = float(_cv)

    # Clarke Error Grid (Clarke et al. 1987), reference = true_bg, pred = pred_bg.
    pb = pred_bg.clamp(min=1.0)
    tb = true_bg.clamp(min=1.0)
    rel_err = (pb - tb).abs() / tb
    in_A = (rel_err <= 0.20) | ((pb <= 70.0) & (tb <= 70.0))
    zone_E = ((pb <= 70.0) & (tb >= 180.0)) | ((pb >= 180.0) & (tb <= 70.0))
    c_upper = (tb >= 70.0) & (tb <= 290.0) & (pb >= tb + 110.0)
    c_lower = (tb >= 130.0) & (tb <= 180.0) & (pb <= (7.0 / 5.0) * tb - 182.0)
    zone_C = (~in_A) & (~zone_E) & (c_upper | c_lower)
    zone_D = (
        (~in_A) & (~zone_E) & (~zone_C)
        & ((tb <= 70.0) | (tb >= 240.0))
        & (pb >= 70.0) & (pb <= 180.0)
    )
    zone_B = (~in_A) & (~zone_E) & (~zone_C) & (~zone_D)
    _stage('clarke_A', in_A.sum())
    _stage('clarke_B', zone_B.sum())
    _stage('clarke_C', zone_C.sum())
    _stage('clarke_D', zone_D.sum())
    _stage('clarke_E', zone_E.sum())
    out['clarke_total'] = float(in_A.numel())

    # Per-horizon Clarke-A and MARD (read-only diagnostic).
    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        h_idx = (h_min // int(dt)) - 1
        if 0 <= h_idx < total_steps_h:
            _stage(f'evalfix_clarke_A@{h_min}', in_A[:, h_idx].sum())
            out[f'evalfix_clarke_A@{h_min}_cnt'] = float(B)
            _stage(f'evalfix_mard@{h_min}_sum', abs_rel[:, h_idx].sum())
            out[f'evalfix_mard@{h_min}_cnt'] = float(B)
        else:
            out[f'evalfix_clarke_A@{h_min}'] = 0.0
            out[f'evalfix_clarke_A@{h_min}_cnt'] = 0.0
            out[f'evalfix_mard@{h_min}_sum'] = 0.0
            out[f'evalfix_mard@{h_min}_cnt'] = 0.0

    # roc_corr / roc_rmse / trend_gain_beta / trend_amp_ratio on the per-PATCH
    # (30-min) ΔBG (mean-collapse detectors — more important now). Five-sum
    # accumulation, finalized in _run_validation.
    pred_patch_end = pred_bg.reshape(B, P, S)[:, :, -1]      # (B, P)
    true_patch_end = true_bg.reshape(B, P, S)[:, :, -1]      # (B, P)
    pred_patch_anchored = torch.cat([last_bg_col, pred_patch_end], dim=1)  # (B, P+1)
    true_patch_anchored = torch.cat([last_bg_col, true_patch_end], dim=1)  # (B, P+1)
    x = pred_patch_anchored[:, 1:] - pred_patch_anchored[:, :-1]  # (B, P)
    y = true_patch_anchored[:, 1:] - true_patch_anchored[:, :-1]  # (B, P)
    _stage('roc_sq_sum', ((x - y).pow(2)).sum())
    out['roc_cnt'] = float(x.numel())
    _stage('roc_sum_x', x.sum())
    _stage('roc_sum_y', y.sum())
    _stage('roc_sum_xx', (x * x).sum())
    _stage('roc_sum_yy', (y * y).sum())
    _stage('roc_sum_xy', (x * y).sum())

    # Excursion-magnitude amplitude (NET peak deviation from last_bg over the
    # whole horizon — captures the over/under-dispersion the per-PATCH ΔBG
    # trend_amp_ratio above structurally misses, since the global-basis median
    # can damp per-step slope while overshooting the net excursion). Per window:
    # peak index = argmax|true − last_bg|; (pred_exc, true_exc) = the deviation
    # there. Real excursions only (|true_exc| > EXC_AMP_MIN_MGDL). Streaming
    # five-sum + over/under-shoot counts, finalized in _run_validation.
    pred_dev_h = pred_bg - last_bg_col              # (B, P*S)
    true_dev_h = true_bg - last_bg_col              # (B, P*S)
    peak_idx = true_dev_h.abs().argmax(dim=1, keepdim=True)  # (B, 1)
    true_exc = true_dev_h.gather(1, peak_idx).squeeze(1)     # (B,)
    pred_exc = pred_dev_h.gather(1, peak_idx).squeeze(1)     # (B,)
    exc_mask = true_exc.abs() > EXC_AMP_MIN_MGDL
    if bool(exc_mask.any()):
        te = true_exc[exc_mask]
        pe = pred_exc[exc_mask]
        ratio = pe / te                             # both relative to last_bg
        out['exc_cnt'] = float(te.numel())
        _stage('exc_sum_pe', pe.sum())
        _stage('exc_sum_te', te.sum())
        _stage('exc_sum_pe2', (pe * pe).sum())
        _stage('exc_sum_te2', (te * te).sum())
        _stage('exc_sum_pete', (pe * te).sum())
        _stage('exc_over', (ratio > EXC_AMP_OVERSHOOT_RATIO).sum())
        _stage('exc_under', (ratio < EXC_AMP_UNDERSHOOT_RATIO).sum())
    else:
        for _ek in ('exc_cnt', 'exc_sum_pe', 'exc_sum_te', 'exc_sum_pe2',
                    'exc_sum_te2', 'exc_sum_pete', 'exc_over', 'exc_under'):
            out[_ek] = 0.0

    # BG curve match: anchor-relative Pearson r of (pred − last_bg) vs
    # (true − last_bg) — scores curve SHAPE, not trivial level agreement.
    xb = (pred_bg - last_bg_col).reshape(-1)
    yb = (true_bg - last_bg_col).reshape(-1)
    out['bgcurve_n'] = float(xb.numel())
    _stage('bgcurve_sx', xb.sum())
    _stage('bgcurve_sy', yb.sum())
    _stage('bgcurve_sxx', (xb * xb).sum())
    _stage('bgcurve_syy', (yb * yb).sum())
    _stage('bgcurve_sxy', (xb * yb).sum())

    # Marginal per-(h, τ) coverage of the central 90% band (τ.05/.95). MARGINAL
    # (per-step empirical hit-rate, target 0.90), NOT joint-over-horizon.
    if q_mgdl:
        lo = q_mgdl['lo']
        hi = q_mgdl['hi']
        for h_min in COVERAGE_HORIZONS_MIN:
            h_idx = (h_min // int(dt)) - 1
            if 0 <= h_idx < total_steps_h:
                covered = ((true_bg[:, h_idx] >= lo[:, h_idx])
                           & (true_bg[:, h_idx] <= hi[:, h_idx])).float()
                _stage(f'coverage90@{h_min}_hit', covered.sum())
                out[f'coverage90@{h_min}_cnt'] = float(B)
            else:
                out[f'coverage90@{h_min}_hit'] = 0.0
                out[f'coverage90@{h_min}_cnt'] = 0.0

    # Diagnostic: sign_balance@h (fraction of true BG strictly below the median
    # forecast, target 0.5 — detects a median that systematically sits above or
    # below truth) and inner50_cov@h (empirical coverage of the [τ.25, τ.75]
    # interval, target 0.5 — calibration of the inner band).
    inner_lo = q_mgdl.get('inner_lo') if q_mgdl else None
    inner_hi = q_mgdl.get('inner_hi') if q_mgdl else None
    for h_min in COVERAGE_HORIZONS_MIN:
        h_idx = (h_min // int(dt)) - 1
        if 0 <= h_idx < total_steps_h:
            below = (true_bg[:, h_idx] < pred_bg[:, h_idx]).float()
            _stage(f'sign_balance@{h_min}_below', below.sum())
            out[f'sign_balance@{h_min}_cnt'] = float(B)
            if inner_lo is not None and inner_hi is not None:
                in_inner = ((true_bg[:, h_idx] >= inner_lo[:, h_idx])
                            & (true_bg[:, h_idx] <= inner_hi[:, h_idx])).float()
                _stage(f'inner50_cov@{h_min}_hit', in_inner.sum())
                out[f'inner50_cov@{h_min}_cnt'] = float(B)
            else:
                out[f'inner50_cov@{h_min}_hit'] = 0.0
                out[f'inner50_cov@{h_min}_cnt'] = 0.0
        else:
            out[f'sign_balance@{h_min}_below'] = 0.0
            out[f'sign_balance@{h_min}_cnt'] = 0.0
            out[f'inner50_cov@{h_min}_hit'] = 0.0
            out[f'inner50_cov@{h_min}_cnt'] = 0.0

    # Single batched host transfer for every staged on-device reduction.
    if _g:
        _keys = list(_g)
        _vals = torch.stack([_g[k] for k in _keys]).tolist()
        for _k, _v in zip(_keys, _vals):
            out[_k] = _v

    return out


# ============================================================================
# ANSI pretty-printing for validation metrics
# ============================================================================

_ANSI_RED = '\033[91m'
_ANSI_YELLOW = '\033[93m'
_ANSI_GREEN = '\033[92m'
_ANSI_CYAN = '\033[96m'
_ANSI_GRAY = '\033[90m'
_ANSI_BOLD = '\033[1m'
_ANSI_RESET = '\033[0m'


def _tier_color(value: float, thresholds: tuple[float, float], higher_is_better: bool) -> str:
    """Return ANSI color code for a value given (red_edge, green_edge) thresholds."""
    red_edge, green_edge = thresholds
    if higher_is_better:
        if value >= green_edge:
            return _ANSI_GREEN
        if value >= red_edge:
            return _ANSI_YELLOW
        return _ANSI_RED
    else:
        if value <= green_edge:
            return _ANSI_GREEN
        if value <= red_edge:
            return _ANSI_YELLOW
        return _ANSI_RED


def _tier_band(value: float, good_lo: float, good_hi: float, warn_lo: float, warn_hi: float) -> str:
    """Tier for band metrics (both too-low and too-high are bad)."""
    if good_lo <= value <= good_hi:
        return _ANSI_GREEN
    if warn_lo <= value <= warn_hi:
        return _ANSI_YELLOW
    return _ANSI_RED


def _strip_ansi(s: str) -> str:
    """Strip ANSI escape sequences for width calculation."""
    import re
    return re.sub(r'\x1b\[[0-9;]*m', '', s)


def _render_validation_table(
    step: int,
    val_metrics: dict[str, Any],
    prev_metrics: dict[str, Any] | None = None,
) -> str:
    """Render a SOTA-target validation table with ANSI colors.

    Four columns: ``Metric | Value | Prev | Target``. The model is always
    conditioned (the prediction-zone carbs+insulin are announced), so there is a
    single validation pass and one value per metric — colored by SOTA tier with a
    trend arrow against the previous validation. Section-title rows span the full
    inner width and are excluded from column-width measurement.
    """

    def _fmt(fmt: str, val: float, suffix: str = '') -> str:
        return f"{fmt.format(val)}{suffix}"

    def _colored(text: str, code: str) -> str:
        return f"{code}{text}{_ANSI_RESET}"

    def _pad(s: str, width: int, align: str = 'l') -> str:
        visible_len = len(_strip_ansi(s))
        spaces = max(0, width - visible_len)
        if align == 'r':
            return ' ' * spaces + s
        if align == 'c':
            l = spaces // 2
            r = spaces - l
            return ' ' * l + s + ' ' * r
        return s + ' ' * spaces

    def _prev_val(metric_key: str, scale: float = 1.0) -> float | None:
        if prev_metrics is None or metric_key is None:
            return None
        v = prev_metrics.get(metric_key)
        if not isinstance(v, (int, float)):
            return None
        return float(v) * scale

    _TREND_REL_TOL = 0.005
    _TREND_EPS = 1e-6

    def _trend_cell(curr: float | None, prev: float | None,
                    direction: str, band_mid: float | None = None) -> str:
        if direction == 'none':
            return ''
        if curr is None or prev is None:
            return _colored('—', _ANSI_GRAY)
        denom = max(abs(prev), abs(curr), _TREND_EPS)
        if abs(curr - prev) / denom < _TREND_REL_TOL:
            return _colored('•', _ANSI_GRAY)
        rose = curr > prev
        if direction == 'lower':
            return _colored('↑', _ANSI_RED) if rose else _colored('↓', _ANSI_GREEN)
        if direction == 'higher':
            return _colored('↑', _ANSI_GREEN) if rose else _colored('↓', _ANSI_RED)
        if direction == 'band':
            if band_mid is None:
                return _colored('—', _ANSI_GRAY)
            improved = abs(curr - band_mid) < abs(prev - band_mid)
            sym = '↑' if rose else '↓'
            return _colored(sym, _ANSI_GREEN) if improved else _colored(sym, _ANSI_RED)
        return _colored('—', _ANSI_GRAY)

    rows: list[tuple[str, str, str, str, str, str]] = []
    # (metric_name, value_colored, prev_colored, trend_colored, target_text,
    #  unit). The value carries NO unit; the unit is appended once in the layout
    # after the value + trend.

    def _section(title: str) -> None:
        rows.append((_colored(title, _ANSI_BOLD + _ANSI_CYAN), '', '', '', '', ''))

    def _blank() -> None:
        rows.append(('', '', '', '', '', ''))

    def _prev_cell(prev_key: str | None, prev_scale: float, fmt: str, unit: str) -> str:
        prev = _prev_val(prev_key, prev_scale) if prev_key else None
        if prev is None:
            return _colored('—', _ANSI_GRAY)
        return _colored(f"{_fmt(fmt, prev)}{unit}", _ANSI_GRAY)

    def info_row(metric: str, val: float | None, fmt: str = '{:+.4f}',
                 unit: str = '', target: str = 'Minimize',
                 prev_key: str | None = None,
                 prev_scale: float = 1.0,
                 direction: str = 'lower') -> None:
        if val is None:
            return
        prev = _prev_val(prev_key, prev_scale) if prev_key else None
        prev_cell = _prev_cell(prev_key, prev_scale, fmt, unit)
        trend = _trend_cell(val, prev, direction)
        rows.append((metric, _colored(_fmt(fmt, val), _ANSI_CYAN),
                     prev_cell, trend, target, unit))

    def lower_row(metric: str, val: float | None, sota: float,
                  fmt: str = '{:.2f}', unit: str = '',
                  warn_mult: float = 1.5,
                  prev_key: str | None = None,
                  prev_scale: float = 1.0) -> None:
        target = f"<{_fmt(fmt, sota)}{unit}"
        if val is None:
            return
        prev = _prev_val(prev_key, prev_scale) if prev_key else None
        prev_cell = _prev_cell(prev_key, prev_scale, fmt, unit)
        warn_edge = sota * warn_mult if sota > 0 else sota + 1.0
        color = _tier_color(val, (warn_edge, sota), higher_is_better=False)
        trend = _trend_cell(val, prev, 'lower')
        rows.append((metric, _colored(_fmt(fmt, val), color),
                     prev_cell, trend, target, unit))

    def higher_row(metric: str, val: float | None, sota: float,
                   fmt: str = '{:.2f}', unit: str = '',
                   warn_gap: float | None = None,
                   prev_key: str | None = None,
                   prev_scale: float = 1.0) -> None:
        target = f">{_fmt(fmt, sota)}{unit}"
        if val is None:
            return
        prev = _prev_val(prev_key, prev_scale) if prev_key else None
        prev_cell = _prev_cell(prev_key, prev_scale, fmt, unit)
        gap = warn_gap if warn_gap is not None else max(sota * 0.10, 1.0)
        warn_edge = sota - gap
        color = _tier_color(val, (warn_edge, sota), higher_is_better=True)
        trend = _trend_cell(val, prev, 'higher')
        rows.append((metric, _colored(_fmt(fmt, val), color),
                     prev_cell, trend, target, unit))

    def band_row(metric: str, val: float | None, lo: float, hi: float,
                 fmt: str = '{:.3f}', unit: str = '',
                 warn_pad: float | None = None,
                 prev_key: str | None = None,
                 prev_scale: float = 1.0) -> None:
        target = f"{_fmt(fmt, lo)}–{_fmt(fmt, hi)}{unit}"
        if val is None:
            return
        prev = _prev_val(prev_key, prev_scale) if prev_key else None
        prev_cell = _prev_cell(prev_key, prev_scale, fmt, unit)
        band_mid = 0.5 * (lo + hi)
        pad = warn_pad if warn_pad is not None else 0.5 * (hi - lo)
        color = _tier_band(val, lo, hi, lo - pad, hi + pad)
        rows.append((metric, _colored(_fmt(fmt, val), color),
                     prev_cell, _trend_cell(val, prev, 'band', band_mid),
                     target, unit))

    # ============================================================
    # 1. Training & Internal Losses
    # ============================================================
    _section('Training & Internal Losses')
    info_row('val_loss_total', val_metrics.get('val_loss_total'),
             prev_key='val_loss_total')
    info_row('val_loss_Q', val_metrics.get('val_loss_Q'),
             prev_key='val_loss_Q')
    info_row('pinball (diag)', val_metrics.get('val_pinball'),
             prev_key='val_pinball')
    info_row('dilate (val)', val_metrics.get('val_loss_D'),
             prev_key='val_loss_D', direction='none')
    info_row('log_sigma_Q', val_metrics.get('log_sigma_Q'),
             fmt='{:+.4f}', target='Kendall-Gal (diag)',
             prev_key='log_sigma_Q', direction='none')
    info_row('log_sigma_D', val_metrics.get('log_sigma_D'),
             fmt='{:+.4f}', target='Kendall-Gal (diag)',
             prev_key='log_sigma_D', direction='none')
    info_row('train_ema', val_metrics.get('train_loss_ema'),
             prev_key='train_loss_ema')
    band_row('overfit_ratio', val_metrics.get('overfit_ratio'),
             0.400, 0.600, fmt='{:.3f}', warn_pad=0.10,
             prev_key='overfit_ratio')
    _blank()

    # ============================================================
    # 2. Formula-Reconstructed BG (RMSE)
    # ============================================================
    _section('BG Forecast (RMSE)')
    bg_rmse_sota = {30: 15.0, 60: 25.0, 120: 36.0}
    night_bg_rmse_sota = {180: 50.0, 360: 62.0, 480: 72.0}
    for h_min in (30, 60, 120):
        lower_row(f'bg_rmse @{h_min}m', val_metrics.get(f'bg_rmse_{h_min}'),
                  bg_rmse_sota[h_min], fmt='{:.1f}', unit=' mg/dL', warn_mult=1.5,
                  prev_key=f'bg_rmse_{h_min}')
    _blank()

    _section('BG Forecast (RMSE) — Night Only @ 180+')
    for h_min in (180, 360, 480):
        lower_row(f'night_bg_rmse @{h_min}m', val_metrics.get(f'night_bg_rmse_{h_min}'),
                  night_bg_rmse_sota[h_min], fmt='{:.1f}', unit=' mg/dL', warn_mult=1.5,
                  prev_key=f'night_bg_rmse_{h_min}')
    _blank()

    # ============================================================
    # 3. Calibration (marginal coverage)
    # ============================================================
    _section('Quantile Calibration (marginal coverage of 90% band)')
    for h_min in COVERAGE_HORIZONS_MIN:
        v = val_metrics.get(f'coverage90@{h_min}')
        band_row(f'coverage90 @{h_min}m',
                 (v * 100.0) if v is not None else None,
                 88.0, 92.0, fmt='{:.2f}', unit='%', warn_pad=5.0,
                 prev_key=f'coverage90@{h_min}', prev_scale=100.0)
    # Diagnostic (uncoloured): sign_balance / inner50_cov, both target 0.50.
    for h_min in COVERAGE_HORIZONS_MIN:
        v = val_metrics.get(f'sign_balance@{h_min}')
        info_row(f'sign_balance @{h_min}m',
                 (v * 100.0) if v is not None else None,
                 fmt='{:.2f}', unit='%', target='≈ 50% (diag)',
                 prev_key=f'sign_balance@{h_min}', prev_scale=100.0, direction='none')
    for h_min in COVERAGE_HORIZONS_MIN:
        v = val_metrics.get(f'inner50_cov@{h_min}')
        info_row(f'inner50_cov @{h_min}m',
                 (v * 100.0) if v is not None else None,
                 fmt='{:.2f}', unit='%', target='≈ 50% (diag)',
                 prev_key=f'inner50_cov@{h_min}', prev_scale=100.0, direction='none')
    _blank()

    # ============================================================
    # 4. Relative Error & Derivative Tracking
    # ============================================================
    _section('Relative Error & Derivative Tracking')
    mard_sota = {30: 7.0, 60: 12.0, 120: 19.0}
    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        lower_row(f'mard @{h_min}m', val_metrics.get(f'evalfix_mard@{h_min}'),
                  mard_sota[h_min], fmt='{:.2f}', unit='%', warn_mult=2.0,
                  prev_key=f'evalfix_mard@{h_min}')
    lower_row('roc_rmse', val_metrics.get('roc_rmse'),
              14.0, fmt='{:.1f}', unit=' mg/dL', warn_mult=1.5,
              prev_key='roc_rmse')
    higher_row('direction (roc_corr)', val_metrics.get('roc_corr'),
               0.650, fmt='{:+.3f}', warn_gap=0.20,
               prev_key='roc_corr')
    higher_row('amplitude (trend_amp_ratio)', val_metrics.get('trend_amp_ratio'),
               0.600, fmt='{:.3f}', warn_gap=0.30,
               prev_key='trend_amp_ratio')
    higher_row('dir×amp (trend_gain_beta)', val_metrics.get('trend_gain_beta'),
               0.800, fmt='{:+.3f}', warn_gap=0.30,
               prev_key='trend_gain_beta')
    higher_row('bg_curve_corr', val_metrics.get('bg_curve_corr'),
               0.700, fmt='{:+.3f}', warn_gap=0.20,
               prev_key='bg_curve_corr')
    # Excursion-magnitude amplitude (NET peak deviation — what trend_amp_ratio
    # above misses). exc_amp_ratio is the headline knob for the DILATE_ALPHA / G
    # recalibration: green ≈ 1.0, red when it over-disperses (>1) or damps (<1).
    band_row('amp@peak (exc_amp_ratio)', val_metrics.get('exc_amp_ratio'),
             0.90, 1.10, fmt='{:.3f}', warn_pad=0.20,
             prev_key='exc_amp_ratio')
    band_row('exc_gain_beta', val_metrics.get('exc_gain_beta'),
             0.90, 1.10, fmt='{:+.3f}', warn_pad=0.25,
             prev_key='exc_gain_beta')
    higher_row('exc_corr', val_metrics.get('exc_corr'),
               0.650, fmt='{:+.3f}', warn_gap=0.20,
               prev_key='exc_corr')
    lower_row('exc_overshoot_frac', val_metrics.get('exc_overshoot_frac'),
              0.150, fmt='{:.3f}', warn_mult=2.0,
              prev_key='exc_overshoot_frac')
    lower_row('exc_undershoot_frac', val_metrics.get('exc_undershoot_frac'),
              0.150, fmt='{:.3f}', warn_mult=2.0,
              prev_key='exc_undershoot_frac')
    info_row('exc_n (windows)', val_metrics.get('exc_n'),
             fmt='{:.0f}', target='count (diag)', direction='none')
    # Conformal coverage probe: does split-conformal recalibration restore the band
    # coverage the raw fan loses at excursion peaks? RAW = the model's bands, CAL =
    # after per-(step,quantile) conformal (median untouched). Fit/eval on a held-out
    # val split (small sample — directional). cov90 → 0.90; hypo escape (truth below
    # the τ=0.10 edge) → 0.10.
    if val_metrics.get('conf_cov90_raw') is not None:
        info_row('conf cov90@peak RAW', val_metrics.get('conf_cov90_raw') * 100.0,
                 fmt='{:.1f}', unit='%', target='ref 90 (diag)', direction='none')
        info_row('conf cov90@peak CAL', val_metrics.get('conf_cov90_cal') * 100.0,
                 fmt='{:.1f}', unit='%', target='ref 90 (diag)', direction='none')
        info_row('conf hypo-escape RAW', val_metrics.get('conf_hypo_esc_raw') * 100.0,
                 fmt='{:.1f}', unit='%', target='ref 10 (diag)', direction='none')
        info_row('conf hypo-escape CAL', val_metrics.get('conf_hypo_esc_cal') * 100.0,
                 fmt='{:.1f}', unit='%', target='ref 10 (diag)', direction='none')
        info_row('conf_n (exc windows)', val_metrics.get('conf_n'),
                 fmt='{:.0f}', target='count (diag)', direction='none')
    _blank()

    # ============================================================
    # 5. Clinical Error Grid Analysis (Clarke)
    # ============================================================
    _section('Clinical Error Grid Analysis (Clarke)')
    clarke_A_sota = {30: 95.0, 60: 85.0, 120: 72.0}
    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        higher_row(f'clarke_A @{h_min}m', val_metrics.get(f'evalfix_clarke_A@{h_min}'),
                   clarke_A_sota[h_min], fmt='{:.2f}', unit='%', warn_gap=2.0,
                   prev_key=f'evalfix_clarke_A@{h_min}')
    higher_row('clarke_A+B', val_metrics.get('clarke_AB_pct'),
               98.0, fmt='{:.2f}', unit='%', warn_gap=2.0,
               prev_key='clarke_AB_pct')
    lower_row('clarke_D', val_metrics.get('clarke_D_pct'),
              0.50, fmt='{:.3f}', unit='%', warn_mult=4.0,
              prev_key='clarke_D_pct')
    _blank()

    # ============================================================
    # 5b. Clinical Accuracy (CG-EGA)
    # ============================================================
    _section('Clinical Accuracy (CG-EGA)')
    for _reg, _ap_sota in (('hypo', 80.0), ('eu', 90.0), ('hyper', 85.0)):
        _ap = val_metrics.get(f'cgega_ap_{_reg}')
        higher_row(f'cgega_AP @{_reg}',
                   (_ap * 100.0) if _ap is not None else None,
                   _ap_sota, fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'cgega_ap_{_reg}', prev_scale=100.0)
    for _reg, _ep_sota in (('hypo', 10.0), ('eu', 2.0), ('hyper', 5.0)):
        _ep = val_metrics.get(f'cgega_ep_{_reg}')
        lower_row(f'cgega_EP @{_reg}',
                  (_ep * 100.0) if _ep is not None else None,
                  _ep_sota, fmt='{:.2f}', unit='%', warn_mult=2.0,
                  prev_key=f'cgega_ep_{_reg}', prev_scale=100.0)
    _blank()

    # ============================================================
    # 6. Longitudinal Excursions & TIR
    # ============================================================
    _section('Longitudinal Excursions & TIR')
    pred_tir = val_metrics.get('pred_tir')
    true_tir = val_metrics.get('true_tir')
    info_row('pred_tir', (pred_tir * 100.0) if pred_tir is not None else None,
             fmt='{:.2f}', unit='%', target='≈ true_tir',
             prev_key='pred_tir', prev_scale=100.0, direction='none')
    info_row('true_tir', (true_tir * 100.0) if true_tir is not None else None,
             fmt='{:.2f}', unit='%', target='ground truth',
             prev_key='true_tir', prev_scale=100.0, direction='none')
    tir_e = val_metrics.get('tir_err')
    lower_row('tir_abs_err', (tir_e * 100.0) if tir_e is not None else None,
              5.0, fmt='{:.2f}', unit='%', warn_mult=2.0,
              prev_key='tir_err', prev_scale=100.0)
    # Precision rows carry a ±k suffix flagging the forgiveness band (recall is strict).
    _ptol = EXCURSION_PRECISION_TOLERANCE_MGDL
    _ptol_sfx = f" ±{_ptol:g}" if _ptol else ""
    hr = val_metrics.get('hypo_recall')
    higher_row(f"hypo_recall({val_metrics.get('hypo_n_steps', 0)}st)",
               (hr * 100.0) if hr is not None else None,
               90.0, fmt='{:.2f}', unit='%', warn_gap=10.0,
               prev_key='hypo_recall', prev_scale=100.0)
    hp = val_metrics.get('hypo_precision')
    higher_row(f'hypo_precision{_ptol_sfx}',
               (hp * 100.0) if hp is not None else None,
               75.0, fmt='{:.2f}', unit='%', warn_gap=10.0,
               prev_key='hypo_precision', prev_scale=100.0)
    yr = val_metrics.get('hyper_recall')
    higher_row(f"hyper_recall({val_metrics.get('hyper_n_steps', 0)}st)",
               (yr * 100.0) if yr is not None else None,
               85.0, fmt='{:.2f}', unit='%', warn_gap=10.0,
               prev_key='hyper_recall', prev_scale=100.0)
    yp = val_metrics.get('hyper_precision')
    higher_row(f'hyper_precision{_ptol_sfx}',
               (yp * 100.0) if yp is not None else None,
               85.0, fmt='{:.2f}', unit='%', warn_gap=10.0,
               prev_key='hyper_precision', prev_scale=100.0)
    _blank()

    # Per-horizon excursion detection (disjoint 30-min buckets).
    _section('Excursions by Horizon  (announced carbs+insulin)')
    _eh = _excursion_bucket_horizons(PREDICTION_PATCHES)
    for _h in _eh:
        v = val_metrics.get(f'hypo_recall@{_h}')
        higher_row(f"hypo_recall@{_h}m({val_metrics.get(f'hypo_n_steps@{_h}', 0)}st)",
                   (v * 100.0) if v is not None else None,
                   _excursion_target(EXCURSION_TARGET_HYPO_RECALL, _h),
                   fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'hypo_recall@{_h}', prev_scale=100.0)
    for _h in _eh:
        v = val_metrics.get(f'hypo_precision@{_h}')
        higher_row(f'hypo_precision@{_h}m{_ptol_sfx}',
                   (v * 100.0) if v is not None else None,
                   _excursion_target(EXCURSION_TARGET_HYPO_PRECISION, _h),
                   fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'hypo_precision@{_h}', prev_scale=100.0)
    for _h in _eh:
        v = val_metrics.get(f'hyper_recall@{_h}')
        higher_row(f"hyper_recall@{_h}m({val_metrics.get(f'hyper_n_steps@{_h}', 0)}st)",
                   (v * 100.0) if v is not None else None,
                   _excursion_target(EXCURSION_TARGET_HYPER_RECALL, _h),
                   fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'hyper_recall@{_h}', prev_scale=100.0)
    for _h in _eh:
        v = val_metrics.get(f'hyper_precision@{_h}')
        higher_row(f'hyper_precision@{_h}m{_ptol_sfx}',
                   (v * 100.0) if v is not None else None,
                   _excursion_target(EXCURSION_TARGET_HYPER_PRECISION, _h),
                   fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'hyper_precision@{_h}', prev_scale=100.0)
    _blank()

    # ============================================================
    # 7. Nocturnal Validation Metrics
    # ============================================================
    _section('Nocturnal Validation Metrics')
    for h_min in (30, 60, 120):
        lower_row(f'night_bg_rmse @{h_min}m',
                  val_metrics.get(f'night_bg_rmse_{h_min}'),
                  bg_rmse_sota[h_min], fmt='{:.1f}', unit=' mg/dL', warn_mult=1.5,
                  prev_key=f'night_bg_rmse_{h_min}')
    night_hr = val_metrics.get('night_hypo_recall')
    higher_row(f"night_hypo_recall({val_metrics.get('night_hypo_n_steps', 0)}st)",
               (night_hr * 100.0) if night_hr is not None else None,
               90.0, fmt='{:.2f}', unit='%', warn_gap=10.0,
               prev_key='night_hypo_recall', prev_scale=100.0)
    night_hp = val_metrics.get('night_hypo_precision')
    higher_row(f'night_hypo_precision{_ptol_sfx}',
               (night_hp * 100.0) if night_hp is not None else None,
               75.0, fmt='{:.2f}', unit='%', warn_gap=10.0,
               prev_key='night_hypo_precision', prev_scale=100.0)
    night_yr = val_metrics.get('night_hyper_recall')
    higher_row(f"night_hyper_recall({val_metrics.get('night_hyper_n_steps', 0)}st)",
               (night_yr * 100.0) if night_yr is not None else None,
               85.0, fmt='{:.2f}', unit='%', warn_gap=10.0,
               prev_key='night_hyper_recall', prev_scale=100.0)
    night_yp = val_metrics.get('night_hyper_precision')
    higher_row(f'night_hyper_precision{_ptol_sfx}',
               (night_yp * 100.0) if night_yp is not None else None,
               85.0, fmt='{:.2f}', unit='%', warn_gap=10.0,
               prev_key='night_hyper_precision', prev_scale=100.0)
    no_n = val_metrics.get('night_onset_n_nights', 0)
    no_hr = val_metrics.get('night_onset_hypo_recall')
    higher_row(f'night-onset hypo recall ({no_n}n)',
               (no_hr * 100.0) if no_hr is not None else None,
               70.0, fmt='{:.2f}', unit='%', warn_gap=15.0,
               prev_key='night_onset_hypo_recall', prev_scale=100.0)
    no_hp = val_metrics.get('night_onset_hypo_precision')
    higher_row('night-onset hypo precision',
               (no_hp * 100.0) if no_hp is not None else None,
               50.0, fmt='{:.2f}', unit='%', warn_gap=15.0,
               prev_key='night_onset_hypo_precision', prev_scale=100.0)
    no_yr = val_metrics.get('night_onset_hyper_recall')
    higher_row(f'night-onset hyper recall ({no_n}n)',
               (no_yr * 100.0) if no_yr is not None else None,
               70.0, fmt='{:.2f}', unit='%', warn_gap=15.0,
               prev_key='night_onset_hyper_recall', prev_scale=100.0)
    no_yp = val_metrics.get('night_onset_hyper_precision')
    higher_row('night-onset hyper precision',
               (no_yp * 100.0) if no_yp is not None else None,
               60.0, fmt='{:.2f}', unit='%', warn_gap=15.0,
               prev_key='night_onset_hyper_precision', prev_scale=100.0)
    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        higher_row(f'night_clarke_A @{h_min}m', val_metrics.get(f'night_clarke_A@{h_min}'),
                   clarke_A_sota[h_min], fmt='{:.2f}', unit='%', warn_gap=2.0,
                   prev_key=f'night_clarke_A@{h_min}')
    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        lower_row(f'night_mard @{h_min}m', val_metrics.get(f'night_mard@{h_min}'),
                  mard_sota[h_min], fmt='{:.2f}', unit='%', warn_mult=2.0,
                  prev_key=f'night_mard@{h_min}')
    _blank()

    # ============================================================
    # 8. Counterfactual dose-response probe (uncoloured diagnostics)
    # ============================================================
    _cf_n = int(val_metrics.get('cf_n', 0) or 0)
    _cf_hypo_n = int(val_metrics.get('cf_hypo_n', 0) or 0)
    _cf_hyper_n = int(val_metrics.get('cf_hyper_n', 0) or 0)
    _section(f'Counterfactual Dose-Response ({_cf_n} samples)')
    _cc = val_metrics.get('cf_carb_dbg')
    info_row('carb→BG (+bolus)', _cc, fmt='{:+.2f}', unit=' mg/dL',
             target='> 0 (carbs raise)', prev_key='cf_carb_dbg', direction='none')
    _ccd = val_metrics.get('cf_carb_dir')
    info_row('carb→BG direction', (_ccd * 100.0) if _ccd is not None else None,
             fmt='{:.1f}', unit='%', target='≈ 100% (diag)',
             prev_key='cf_carb_dir', prev_scale=100.0, direction='none')
    _ci = val_metrics.get('cf_insulin_dbg')
    info_row('insulin→BG (+bolus)', _ci, fmt='{:+.2f}', unit=' mg/dL',
             target='< 0 (insulin lowers)', prev_key='cf_insulin_dbg', direction='none')
    _cid = val_metrics.get('cf_insulin_dir')
    info_row('insulin→BG direction', (_cid * 100.0) if _cid is not None else None,
             fmt='{:.1f}', unit='%', target='≈ 100% (diag)',
             prev_key='cf_insulin_dir', prev_scale=100.0, direction='none')
    _ccm = val_metrics.get('cf_carb_monotonic')
    info_row('carb monotonic', (_ccm * 100.0) if _ccm is not None else None,
             fmt='{:.1f}', unit='%', target='≈ 100% (diag)',
             prev_key='cf_carb_monotonic', prev_scale=100.0, direction='none')
    _cim = val_metrics.get('cf_insulin_monotonic')
    info_row('insulin monotonic', (_cim * 100.0) if _cim is not None else None,
             fmt='{:.1f}', unit='%', target='≈ 100% (diag)',
             prev_key='cf_insulin_monotonic', prev_scale=100.0, direction='none')
    _chr = val_metrics.get('cf_hypo_rescue')
    info_row(f'hypo rescue (carb, {_cf_hypo_n}n)',
             (_chr * 100.0) if _chr is not None else None,
             fmt='{:.1f}', unit='%', target='higher (diag)',
             prev_key='cf_hypo_rescue', prev_scale=100.0, direction='none')
    _cyr = val_metrics.get('cf_hyper_rescue')
    info_row(f'hyper rescue (insulin, {_cf_hyper_n}n)',
             (_cyr * 100.0) if _cyr is not None else None,
             fmt='{:.1f}', unit='%', target='higher (diag)',
             prev_key='cf_hyper_rescue', prev_scale=100.0, direction='none')
    _blank()

    # ============================================================
    # Time-of-day probe (tier-coloured; co-trains the trunk but never feeds loss
    # or selection). Thresholds are clock-usability judgements, NOT external
    # SOTA: green ~ a usable clock, red ~ near random-phase chance (chance is
    # mae 6 h, acc +/-1h 8%, +/-2h 17%, 4-bin 25%).
    # ============================================================
    _section('Time-of-day probe (diagnostic)')
    lower_row('tod mae', val_metrics.get('tod_mae_h'), 1.5,
              fmt='{:.2f}', unit=' h', warn_mult=2.0, prev_key='tod_mae_h')
    higher_row('tod acc +/-1h', val_metrics.get('tod_acc_1h'), 60.0,
               fmt='{:.1f}', unit='%', warn_gap=30.0, prev_key='tod_acc_1h')
    higher_row('tod acc +/-2h', val_metrics.get('tod_acc_2h'), 80.0,
               fmt='{:.1f}', unit='%', warn_gap=35.0, prev_key='tod_acc_2h')
    higher_row('tod acc 4-bin', val_metrics.get('tod_acc_bin'), 70.0,
               fmt='{:.1f}', unit='%', warn_gap=30.0, prev_key='tod_acc_bin')
    higher_row('tod confidence R', val_metrics.get('tod_conf'), 0.80,
               fmt='{:.2f}', warn_gap=0.40, prev_key='tod_conf')
    # Clock reliability: is the probe usable as an actual clock?
    band_row('tod bias', val_metrics.get('tod_bias_h'), -1.0, 1.0,
             fmt='{:+.2f}', unit=' h', warn_pad=2.0, prev_key='tod_bias_h')
    lower_row('tod precision sd', val_metrics.get('tod_std_h'), 1.5,
              fmt='{:.2f}', unit=' h', warn_mult=2.0, prev_key='tod_std_h')
    lower_row('tod p90 err', val_metrics.get('tod_p90_h'), 3.0,
              fmt='{:.2f}', unit=' h', warn_mult=2.0, prev_key='tod_p90_h')
    lower_row('tod gross >3h', val_metrics.get('tod_gross_rate'), 10.0,
              fmt='{:.1f}', unit='%', warn_mult=2.5, prev_key='tod_gross_rate')
    lower_row('tod mae hi-conf', val_metrics.get('tod_mae_hiconf'), 1.0,
              fmt='{:.2f}', unit=' h', warn_mult=2.0, prev_key='tod_mae_hiconf')
    lower_row('tod jump', val_metrics.get('tod_jump_h'), 0.25,
              fmt='{:.2f}', unit=' h', warn_mult=3.0, prev_key='tod_jump_h')
    lower_row('tod xwin jump', val_metrics.get('tod_xwin_jump_h'), 0.5,
              fmt='{:.2f}', unit=' h', warn_mult=3.0, prev_key='tod_xwin_jump_h')
    _blank()

    # ============================================================
    # Prune orphaned section headers + collapse blank runs.
    # ============================================================
    def _is_section_row(r: tuple) -> bool:
        return bool(r[0]) and not r[1]

    def _is_blank_row(r: tuple) -> bool:
        return not any(r)

    pruned: list[tuple] = []
    for i, r in enumerate(rows):
        if _is_section_row(r):
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            if nxt is None or _is_section_row(nxt) or _is_blank_row(nxt):
                continue
        if _is_blank_row(r) and (not pruned or _is_blank_row(pruned[-1])):
            continue
        pruned.append(r)
    while pruned and _is_blank_row(pruned[-1]):
        pruned.pop()
    rows = pruned

    # ============================================================
    # Layout — four columns: Metric | Value | Prev | Target.
    # ============================================================
    def _with_unit(colored: str, unit: str) -> str:
        if not unit:
            return colored
        if colored.endswith(_ANSI_RESET):
            return colored[: -len(_ANSI_RESET)] + unit + _ANSI_RESET
        return colored + unit

    def _value_cell(value: str, trend: str, unit: str) -> str:
        return f"{_with_unit(value, unit)}{trend}"

    headers = ['Metric', 'Value', 'Prev', 'Target']
    col_w = [len(h) for h in headers]
    for r in rows:
        if _is_section_row(r) or _is_blank_row(r):
            continue
        metric, value, prev, trend, target, unit = r
        cells = (metric, _value_cell(value, trend, unit), prev, target)
        for i, cell in enumerate(cells):
            col_w[i] = max(col_w[i], len(_strip_ansi(cell)))

    c1, c2, c3, c4 = col_w
    inner_w = c1 + c2 + c3 + c4 + 9

    max_title_w = 0
    for r in rows:
        if _is_section_row(r):
            max_title_w = max(max_title_w, len(_strip_ansi(r[0])))
    if max_title_w > inner_w:
        c1 += max_title_w - inner_w
        inner_w = max_title_w

    top_edge = f"├─{'─' * c1}─┬─{'─' * c2}─┬─{'─' * c3}─┬─{'─' * c4}─┤"
    mid_edge = f"├─{'─' * c1}─┼─{'─' * c2}─┼─{'─' * c3}─┼─{'─' * c4}─┤"
    bot_edge = f"└─{'─' * c1}─┴─{'─' * c2}─┴─{'─' * c3}─┴─{'─' * c4}─┘"
    sep_line = f"│ {' ' * c1} │ {' ' * c2} │ {' ' * c3} │ {' ' * c4} │"

    title_top = f"┌─{'─' * inner_w}─┐"
    title_lines = [
        f"Validation @ step {step} — {PREDICTION_HORIZON_HOURS}h window",
        "Conditioned (announced carbs+insulin)",
        "SOTA Target Comparison",
    ]
    title_inner = [
        f"│ {_ANSI_BOLD}{_ANSI_CYAN}{_pad(t, inner_w, 'c')}{_ANSI_RESET} │"
        for t in title_lines
    ]

    header_row = (
        f"│ {_ANSI_BOLD}{_pad('Metric', c1)}{_ANSI_RESET} "
        f"│ {_ANSI_BOLD}{_pad('Value', c2)}{_ANSI_RESET} "
        f"│ {_ANSI_BOLD}{_pad('Prev', c3)}{_ANSI_RESET} "
        f"│ {_ANSI_BOLD}{_pad('Target', c4)}{_ANSI_RESET} │"
    )

    lines = [title_top, *title_inner, top_edge, header_row, mid_edge]
    for r in rows:
        if _is_blank_row(r):
            lines.append(sep_line)
            continue
        metric, value, prev, trend, target, unit = r
        if _is_section_row(r):
            title = _strip_ansi(metric)
            lines.append(
                f"│ {_ANSI_BOLD}{_ANSI_CYAN}{_pad(title, inner_w)}{_ANSI_RESET} │"
            )
            continue
        vcell = _value_cell(value, trend, unit)
        lines.append(
            f"│ {_pad(metric, c1)} "
            f"│ {_pad(vcell, c2)} "
            f"│ {_pad(prev, c3)} "
            f"│ {_colored(_pad(target, c4), _ANSI_GRAY) if target else _pad('', c4)} │"
        )
    lines.append(bot_edge)
    return '\n'.join(lines)


# ============================================================================
# Optimizer utilities
# ============================================================================

def _build_optimizers(
    model: T1DMAI,
    weighting: KendallGalWeighting,
    muon_lr: float,
    adam_lr: float,
    muon_momentum: float,
    adam_weight_decay: float = ADAM_WEIGHT_DECAY,
) -> tuple[Muon, torch.optim.AdamW]:
    """
    Split parameters into Muon (≥2D) and AdamW (≤1D) groups.

    Muon owns the 2D weight matrices; AdamW owns 1D parameters, biases, norms.
    Muon's matrices are further split into a normalized group (every matrix that
    feeds a norm — gets the AdamC ``gamma_t/gamma_max`` schedule-aware weight-decay
    correction) and an output group (``bg_head[-1]`` / ``time_head[-1]``, kept at
    constant decay since they are not followed by a norm). The two scalar
    Kendall-Gal log-σ parameters (on ``weighting``) go into their OWN AdamW group
    with ``weight_decay=0`` — they are scalars, never Muon (a log-variance must
    not decay toward 0), and live off ``model`` so the weight EMA never touches
    them.

    Returns:
        muon_opt, adam_opt.
    """
    # Output projections are NOT followed by a normalization, so the AdamC
    # steady-state analysis (<g, x> = 0) does not hold for them — they are
    # excluded from the schedule-aware weight-decay correction (paper section 6,
    # "excluding the output layer of the network"). Identify them by object
    # identity: the BG head's final Linear and the time probe's final Linear
    # (if the probe is built). Everything else 2D is a normalized matrix.
    output_weight_ids = {id(model.bg_head[-1].weight)}
    if getattr(model, "time_head", None) is not None:
        output_weight_ids.add(id(model.time_head[-1].weight))

    muon_normalized = []   # normalized matrices — get the gamma_t/gamma_max decay correction
    muon_output = []       # output projections — plain (uncorrected) decay
    adam_params = []
    for _name, param in model.named_parameters():
        if param.ndim >= 2:
            if id(param) in output_weight_ids:
                muon_output.append(param)
            else:
                muon_normalized.append(param)
        else:
            adam_params.append(param)

    kendall_params = []
    for _name, param in weighting.named_parameters():
        assert param.ndim < 2, (
            f"Kendall-Gal weighting param {_name} must be scalar/1D, "
            f"got ndim={param.ndim}"
        )
        kendall_params.append(param)

    # Two Muon groups. The normalized group is flagged wd_corrected=True and
    # carries base_weight_decay so _update_lr can rescale its weight_decay to
    # base*(gamma_t/gamma_max) each step (AdamC); the output group keeps constant
    # decay. With WEIGHT_DECAY_SCHEDULE_CORRECTION=False _update_lr leaves both
    # groups at the constant MUON_WEIGHT_DECAY, reducing exactly to the old
    # single-group behaviour (two constant-decay groups are numerically identical
    # to one; momentum buffers are per-parameter, not per-group).
    muon_groups = [
        {"params": muon_normalized, "weight_decay": MUON_WEIGHT_DECAY,
         "base_weight_decay": MUON_WEIGHT_DECAY, "wd_corrected": True},
    ]
    if muon_output:
        muon_groups.append(
            {"params": muon_output, "weight_decay": MUON_WEIGHT_DECAY,
             "base_weight_decay": MUON_WEIGHT_DECAY, "wd_corrected": False}
        )
    muon_opt = Muon(muon_groups, lr=muon_lr, momentum=muon_momentum,
                    ns_iterations=MUON_NS_ITERATIONS, weight_decay=MUON_WEIGHT_DECAY)
    adam_opt = torch.optim.AdamW(
        [
            {'params': adam_params},
            {'params': kendall_params, 'weight_decay': 0.0},
        ],
        lr=adam_lr, betas=ADAM_BETAS, weight_decay=adam_weight_decay, eps=ADAM_EPS,
    )
    return muon_opt, adam_opt


def _update_lr(
    muon_opt: Muon,
    adam_opt: torch.optim.AdamW,
    step: int,
    peak_muon_lr: float,
    peak_adam_lr: float,
    warmup_steps: int,
    total_steps: int,
    lr_min_ratio: float,
    wd_correction: bool = WEIGHT_DECAY_SCHEDULE_CORRECTION,
) -> None:
    """Apply the warmup + cosine decay LR schedule to both optimizers.

    When ``wd_correction`` is set, the AdamC schedule-aware weight-decay
    correction (arXiv 2506.02285) is also applied: every Muon group flagged
    ``wd_corrected`` has its ``weight_decay`` rescaled to
    ``base_weight_decay * ratio`` where ``ratio = gamma_t/gamma_max`` is the same
    schedule multiplier used for the LR (gamma_max = peak LR). Combined with the
    optimizer's own p*(1 - lr*wd) step this realizes the gamma_t^2/gamma_max decay
    of Algorithm 1 on the normalized matrices; the output-head group
    (wd_corrected False) and AdamW keep constant decay.
    """
    if step < warmup_steps:
        ratio = step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        ratio = lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))
    for group in muon_opt.param_groups:
        group['lr'] = peak_muon_lr * ratio
        if group.get('wd_corrected', False):
            # Authoritative (idempotent) assignment: correction on -> the AdamC
            # base*(gamma_t/gamma_max); off -> the base constant. Writing the
            # value in both branches means a corrected group never retains a
            # stale scaled decay if wd_correction is ever toggled on a live
            # optimizer (production keeps it constant per run, so this is a
            # robustness guard, not a behaviour change).
            group['weight_decay'] = (
                group['base_weight_decay'] * ratio if wd_correction
                else group['base_weight_decay']
            )
    for group in adam_opt.param_groups:
        group['lr'] = peak_adam_lr * ratio


# ============================================================================
# Statistics persistence
# ============================================================================

def _write_training_summary(
    log_dir: str,
    step: int,
    total_steps: int,
    loss_history: list[float],
    best_val_loss: float,
    best_val_step: int,
    training_config: dict,
    train_start_time: float,
    val_history: list[dict],
    device: torch.device,
) -> None:
    """Write a comprehensive training summary JSON (at each checkpoint + end)."""
    now = time.time()
    elapsed_hours = (now - train_start_time) / 3600.0
    steps_done = step + 1
    pct_complete = 100.0 * steps_done / max(total_steps, 1)

    if len(loss_history) > 500:
        stride = len(loss_history) // 500
        sampled_losses = loss_history[::stride]
    else:
        sampled_losses = list(loss_history)

    recent = [v for v in loss_history[-100:] if v == v]
    recent_mean = sum(recent) / len(recent) if recent else None
    recent_min = min(recent) if recent else None

    finite_history = [v for v in loss_history if v == v]
    all_min = min(finite_history) if finite_history else None
    all_min_step = loss_history.index(all_min) if all_min is not None else None

    overfit_summary = None
    if len(val_history) >= 2:
        latest_val = val_history[-1]
        overfit_summary = {
            'latest_step': latest_val.get('step'),
            'val_loss': latest_val.get('val_loss_total'),
            'train_loss_at_val': latest_val.get('train_loss_ema'),
            'overfit_ratio': latest_val.get('overfit_ratio'),
            'best_val_loss': best_val_loss,
            'best_val_step': best_val_step,
        }

    summary = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
        'arch_version': ARCH_VERSION,
        'loss_schema': LOSS_SCHEMA,
        'progress': {
            'step': step,
            'total_steps': total_steps,
            'pct_complete': round(pct_complete, 2),
            'elapsed_hours': round(elapsed_hours, 3),
            'steps_per_second': round(steps_done / max(now - train_start_time, 1e-9), 2),
        },
        'loss': {
            'current': loss_history[-1] if loss_history else None,
            'recent_mean_100': round(recent_mean, 6) if recent_mean is not None else None,
            'recent_min_100': round(recent_min, 6) if recent_min is not None else None,
            'all_time_min': round(all_min, 6) if all_min is not None else None,
            'all_time_min_step': all_min_step,
            'sampled_history': [round(v, 6) for v in sampled_losses],
        },
        'validation': overfit_summary,
        'validation_history': val_history,
        'prediction_window': {
            'horizon_hours': PREDICTION_HORIZON_HOURS,
            'prediction_patches': PREDICTION_PATCHES,
        },
        'hardware': {
            'device': str(device),
            'gpu_peak_memory_mb': (
                round(torch.cuda.max_memory_allocated() / 1e6, 1)
                if device.type == 'cuda' else 0
            ),
        },
        'config': training_config,
    }

    path = os.path.join(log_dir, 'training_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# Validation
# ============================================================================

VAL_BATCH_SIZE = 8


def _reconstruct_context_from_patch(patches: torch.Tensor) -> torch.Tensor:
    """Reshape (T, PATCH_DIM) → (T, PATCH_SIZE, N_INPUT_FEATURES) for the
    rolling-validation / counterfactual context. ``PATCH_DIM = PATCH_SIZE *
    N_INPUT_FEATURES`` (= 18) — there are no mask bits to strip."""
    feat_cols = PATCH_SIZE * N_INPUT_FEATURES
    assert patches.shape[1] == feat_cols, (
        f"patches must be (T, {feat_cols}), got {tuple(patches.shape)}"
    )
    return patches.reshape(patches.shape[0], PATCH_SIZE, N_INPUT_FEATURES)


def _is_nocturnal(hour: float) -> bool:
    if NOCTURNAL_START_HOUR <= NOCTURNAL_END_HOUR:
        return NOCTURNAL_START_HOUR <= hour < NOCTURNAL_END_HOUR
    else:
        return hour >= NOCTURNAL_START_HOUR or hour < NOCTURNAL_END_HOUR


def _make_long_horizon_overrides_fn(bf: dict):
    """Per-roll announced carb(0)/insulin(1) overrides for ``predict_rolling``, built
    from the sample's shipped future doses (``extended_{carb,insulin}_{norm,raw}``).

    The long-horizon (nocturnal) roll is the KNOWN-PLAN regime: the future insulin
    curve (basal + boluses) and carb curve are announced each roll, so the rolling
    forecast is conditioned on them (mask bits flipped to 1 by the inference bridge)
    while ``predict_rolling`` stays BG-AUTOREGRESSIVE — it still re-feeds the model's
    OWN median BG, since no true BG is observed yet. Returns ``None`` past the end of
    the shipped future, so that roll falls back to its unconditional prediction.

    Returns ``None`` (the whole builder) when the sample lacks the extended-dose
    keys, so the caller can fall back to an unconditioned roll.
    """
    keys = ('extended_carb_norm', 'extended_insulin_norm',
            'extended_carb_raw', 'extended_insulin_raw')
    if not all(k in bf for k in keys):
        return None
    _ps = PREDICTION_PATCHES * PATCH_SIZE
    cn = np.asarray(bf['extended_carb_norm'], dtype=np.float32)
    inn = np.asarray(bf['extended_insulin_norm'], dtype=np.float32)
    cr = np.asarray(bf['extended_carb_raw'], dtype=np.float32)
    ir = np.asarray(bf['extended_insulin_raw'], dtype=np.float32)

    def fn(roll_idx, mu_np, abs_n_ctx):
        a = roll_idx * _ps
        b = a + _ps
        if b > cn.shape[0]:
            return None

        def rs(x):
            return x[a:b].reshape(PREDICTION_PATCHES, PATCH_SIZE)
        return ({0: rs(cn), 1: rs(inn)}, {0: rs(cr), 1: rs(ir)})
    return fn


def _accumulate_long_horizon_bg_metrics(
    model: T1DMAI,
    samples: list[dict[str, Any]],
    norm_stats: dict,
    device: torch.device,
    n_rolls: int,
    agg: dict[str, float],
    night_agg: dict[str, float] | None = None,
) -> None:
    """Run rolling prediction per validation sample and accumulate
    ``bg_rmse_{h}_*`` / ``bg_mae_{h}_*`` for horizons not covered by a single
    forward pass. ``predict_rolling`` is BG-autoregressive: its ``pred_bg`` is
    ``f_inv(median)`` carried across rolls (no physics constants needed).

    The long-horizon roll is CONDITIONED on each sample's announced future
    insulin (basal+boluses) + carb curve via ``_make_long_horizon_overrides_fn``
    — the known-plan regime the rolling forecast is built for (the nocturnal-hypo
    use case knows the programmed basal ahead of time). BG stays autoregressive
    (the model re-feeds its own median); only the doses are announced."""
    from inference import predict_rolling

    if n_rolls <= 0:
        return

    single_pass_steps = PREDICTION_PATCHES * PATCH_SIZE
    dt = 0

    for sample in samples:
        n_ctx = int(sample['n_context_patches'])
        patches = sample['patches']
        ctx_only_patches = patches[:n_ctx]
        context = _reconstruct_context_from_patch(ctx_only_patches)
        bf = sample['bg_formula_data']
        if dt == 0:
            dt = int(_dt_minutes(bf))

        result = predict_rolling(
            model, context, patient_seed=None, n_rolls=n_rolls,
            normalization_stats=norm_stats,
            device=device,
            overrides_fn=_make_long_horizon_overrides_fn(bf),
        )
        pred_bg = result['pred_bg'].detach().cpu()
        true_bg_extended = bf['extended_true_bg_trajectory']
        if not isinstance(true_bg_extended, torch.Tensor):
            true_bg_extended = torch.from_numpy(np.asarray(true_bg_extended)).float()
        else:
            true_bg_extended = true_bg_extended.float().cpu()

        usable = min(pred_bg.shape[0], true_bg_extended.shape[0])
        if usable == 0:
            continue
        pb = pred_bg[:usable]
        tb = true_bg_extended[:usable]

        is_night = _is_nocturnal(float(bf.get('pred_start_hour', 0.0)))

        for h_min in BG_HORIZONS_MIN:
            h_idx = (h_min // dt) - 1
            if h_idx < single_pass_steps:
                continue
            if 0 <= h_idx < usable:
                diff = float(pb[h_idx]) - float(tb[h_idx])
                agg[f'bg_rmse_{h_min}_sq_sum'] = agg.get(f'bg_rmse_{h_min}_sq_sum', 0.0) + diff * diff
                agg[f'bg_rmse_{h_min}_cnt'] = agg.get(f'bg_rmse_{h_min}_cnt', 0.0) + 1.0
                agg[f'bg_mae_{h_min}_abs_sum'] = agg.get(f'bg_mae_{h_min}_abs_sum', 0.0) + abs(diff)
                agg[f'bg_mae_{h_min}_cnt'] = agg.get(f'bg_mae_{h_min}_cnt', 0.0) + 1.0
                if night_agg is not None and is_night:
                    night_agg[f'night_bg_rmse_{h_min}_sq_sum'] = night_agg.get(f'night_bg_rmse_{h_min}_sq_sum', 0.0) + diff * diff
                    night_agg[f'night_bg_rmse_{h_min}_cnt'] = night_agg.get(f'night_bg_rmse_{h_min}_cnt', 0.0) + 1.0
                    night_agg[f'night_bg_mae_{h_min}_abs_sum'] = night_agg.get(f'night_bg_mae_{h_min}_abs_sum', 0.0) + abs(diff)
                    night_agg[f'night_bg_mae_{h_min}_cnt'] = night_agg.get(f'night_bg_mae_{h_min}_cnt', 0.0) + 1.0


def _run_night_onset_validation(
    model: T1DMAI,
    dataset: T1DMDataset,
    norm_stats: dict,
    device: torch.device,
    hypo_threshold: float = BG_HYPO_THRESHOLD,
    hyper_threshold: float = BG_HYPER_THRESHOLD,
) -> dict[str, Any]:
    """Night-onset excursion prediction. Each sample's origin is forced to the
    bedtime hour (``NOCTURNAL_START_HOUR``); autoregressively roll the forecast
    (``predict_rolling``, BG-autoregressive) across the night to
    ``NOCTURNAL_END_HOUR`` and emit a PER-NIGHT binary excursion call. The
    predicted call keys off the BAND EDGES (like every other clinical hypo/hyper
    metric): hypo off the τ=HYPO_ALARM_QUANTILE_TAU lower edge, hyper off the
    τ=HYPER_ALARM_QUANTILE_TAU upper edge (``predict_rolling`` returns the mg/dL
    band fan in ``result['bands']``); truth stays off the TRUE bg. Returns
    ``night_onset_{hypo,hyper}_{recall,precision}`` (+ counts).

    The roll is always conditioned on the night's ANNOUNCED carbs+insulin via an
    ``overrides_fn`` (keys are output-channel space {0:carb, 1:insulin}; the
    inference bridge maps them onto feats 1/2) — the model is always conditioned.
    """
    from inference import predict_rolling

    n_rolls = math.ceil(NIGHT_LONG_HORIZON_HOURS / PREDICTION_HORIZON_HOURS)
    if n_rolls <= 1:
        return {}

    night_len_h = (NOCTURNAL_END_HOUR - NOCTURNAL_START_HOUR) % 24.0
    if night_len_h == 0.0:
        night_len_h = 24.0

    c = {'hypo_true': 0, 'hypo_pred': 0, 'hypo_tp': 0,
         'hyper_true': 0, 'hyper_pred': 0, 'hyper_tp': 0, 'n': 0}
    n_val = min(len(dataset), VALIDATION_N_PATIENTS)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(n_val):
            sample = dataset[i]
            n_ctx = int(sample['n_context_patches'])
            context = _reconstruct_context_from_patch(sample['patches'][:n_ctx])
            bf = sample['bg_formula_data']
            dt = int(_dt_minutes(bf))
            night_steps = int(round(night_len_h * 60.0 / dt))
            result = predict_rolling(
                model, context, patient_seed=None, n_rolls=n_rolls,
                normalization_stats=norm_stats, device=device,
                overrides_fn=_make_long_horizon_overrides_fn(bf),
            )
            pred_bg = result['pred_bg'].detach().cpu()
            bands = result['bands'].detach().cpu()             # (rolls*P, S, N_QUANTILES) mg/dL
            bands = bands.reshape(-1, bands.shape[-1])         # -> (T, N_QUANTILES) per-step, aligned with pred_bg
            true_bg = bf['extended_true_bg_trajectory']
            true_bg = (true_bg.float().cpu() if isinstance(true_bg, torch.Tensor)
                       else torch.from_numpy(np.asarray(true_bg)).float())
            usable = min(pred_bg.shape[0], true_bg.shape[0], night_steps)
            if usable == 0:
                continue
            tb = true_bg[:usable]
            # Band-edge detectors: hypo off the τ=HYPO_ALARM_QUANTILE_TAU LOWER edge,
            # hyper off the τ=HYPER_ALARM_QUANTILE_TAU UPPER edge; truth stays off TRUE bg.
            pred_lo = bands[:usable, _HYPO_BAND_IDX]
            pred_hi = bands[:usable, _HYPER_BAND_IDX]
            c['n'] += 1
            for side, t_ex, p_ex in (
                    ('hypo', bool((tb < hypo_threshold).any().item()),
                             bool((pred_lo < hypo_threshold).any().item())),
                    ('hyper', bool((tb > hyper_threshold).any().item()),
                              bool((pred_hi > hyper_threshold).any().item()))):
                c[f'{side}_true'] += int(t_ex)
                c[f'{side}_pred'] += int(p_ex)
                c[f'{side}_tp'] += int(t_ex and p_ex)

    model.train(was_training)
    out: dict[str, Any] = {'night_onset_n_nights': c['n']}
    for side in ('hypo', 'hyper'):
        tr, pr, tp = c[f'{side}_true'], c[f'{side}_pred'], c[f'{side}_tp']
        out[f'night_onset_{side}_recall'] = (tp / tr) if tr > 0 else None
        out[f'night_onset_{side}_precision'] = (tp / pr) if pr > 0 else None
        out[f'night_onset_{side}_n_true'] = tr
    return out


def _run_counterfactual_probe(
    model: T1DMAI,
    val_dataset: T1DMDataset,
    norm_stats: dict,
    device: torch.device,
    hypo_threshold: float = BG_HYPO_THRESHOLD,
    hyper_threshold: float = BG_HYPER_THRESHOLD,
    samples: list[dict] | None = None,
) -> dict[str, Any]:
    """Counterfactual dose-response probe.

    For up to ``VALIDATION_N_PATIENTS`` validation samples, forecast a BASELINE
    on each sample's TRUE announced pred-zone carbs+insulin, then perturb a single
    dose (a RAW bolus added to the FIRST pred-zone patch, spread across its
    PATCH_SIZE steps, then renormalized through the channel's log1p z-transform)
    and re-forecast. Reports, over the probed samples:

    * ``cf_carb_dbg`` / ``cf_insulin_dbg`` — mean over samples of the
      mean-over-horizon ΔBG (mg/dL) from a ``+CF_CARB_BOLUS_G`` carb /
      ``+CF_INSULIN_BOLUS_U`` insulin bolus vs baseline. Carbs should raise BG
      (>0); insulin should lower it (<0).
    * ``cf_carb_dir`` / ``cf_insulin_dir`` — fraction of samples with the
      clinically-correct sign (carb mean ΔBG > 0; insulin mean ΔBG < 0). Target ~1.
    * ``cf_carb_monotonic`` / ``cf_insulin_monotonic`` — fraction of samples for
      which the horizon-peak (carb: max BG, non-decreasing) / horizon-min
      (insulin: min BG, non-increasing) moves monotonically across bolus levels
      ``[0, B/2, B]``.
    * ``cf_hypo_rescue`` — among samples whose baseline MIN predicted BG dips below
      ``hypo_threshold``, the fraction where ``+CF_CARB_BOLUS_G`` carb lifts the min
      predicted BG to ``>= hypo_threshold``. ``cf_hypo_n`` counts those samples.
    * ``cf_hyper_rescue`` — among samples whose baseline MAX predicted BG exceeds
      ``hyper_threshold``, the fraction where ``+CF_INSULIN_BOLUS_U`` insulin lowers
      the max predicted BG to ``<= hyper_threshold``. ``cf_hyper_n`` counts those.

    All forecasts go through ``inference.predict`` (the SOLE risk->mg/dL bridge);
    every dose is announced via the ``overrides`` dict (output-channel space
    {0: carb, 1: insulin}). Runs under ``model.eval()`` + ``torch.no_grad()``.

    ``samples`` optionally supplies the already-materialized ``val_dataset[i]``
    dicts (index-ordered) from the main validation loop; when given they are
    reused verbatim (samples are deterministic in ``i``) instead of re-indexing
    the dataset — which would re-run the simulator. ``None`` falls back to
    per-index ``val_dataset[i]``.

    Returns:
        dict with the ``cf_*`` keys above plus ``cf_n`` (samples probed).
    """
    from inference import predict

    _ps = PREDICTION_PATCHES * PATCH_SIZE
    carb_m = float(norm_stats['carb_intake']['mean'])
    carb_s = float(norm_stats['carb_intake']['std'])
    ins_m = float(norm_stats['insulin_combined']['mean'])
    ins_s = float(norm_stats['insulin_combined']['std'])
    carb_B = float(CF_CARB_BOLUS_G)
    ins_B = float(CF_INSULIN_BOLUS_U)

    def _renorm(raw: np.ndarray, m: float, s: float) -> torch.Tensor:
        """Raw per-step (P*S,) → normalized (P, S) torch tensor via log1p z."""
        norm = (np.log1p(np.maximum(raw, 0.0)) - m) / (s + 1e-8)
        return torch.from_numpy(
            norm.reshape(PREDICTION_PATCHES, PATCH_SIZE).astype(np.float32))

    def _perturb_raw(raw: np.ndarray, bolus: float) -> np.ndarray:
        """Add ``bolus`` to the first pred-zone patch, spread across its steps."""
        out = raw.copy()
        out[:PATCH_SIZE] = out[:PATCH_SIZE] + bolus / float(PATCH_SIZE)
        return out

    def _forecast(carb_t: torch.Tensor, ins_t: torch.Tensor) -> np.ndarray:
        res = predict(model, context, normalization_stats=norm_stats,
                      device=device, overrides={0: carb_t, 1: ins_t})
        return res['median_bg'].detach().cpu().numpy()      # (P*S,) mg/dL

    n_val = min(len(val_dataset), VALIDATION_N_PATIENTS)
    if samples is not None:
        n_val = min(n_val, len(samples))

    carb_dbg_sum = 0.0
    carb_dir_hits = 0
    ins_dbg_sum = 0.0
    ins_dir_hits = 0
    carb_mono_hits = 0
    ins_mono_hits = 0
    hypo_n = 0
    hypo_rescue_hits = 0
    hyper_n = 0
    hyper_rescue_hits = 0
    n_probed = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(n_val):
            sample = samples[i] if samples is not None else val_dataset[i]
            n_ctx = int(sample['n_context_patches'])
            context = _reconstruct_context_from_patch(sample['patches'][:n_ctx])
            bf = sample['bg_formula_data']

            keys = ('extended_carb_norm', 'extended_insulin_norm',
                    'extended_carb_raw', 'extended_insulin_raw')
            if not all(k in bf for k in keys):
                continue

            carb_norm = np.asarray(bf['extended_carb_norm'], dtype=np.float32)[:_ps]
            ins_norm = np.asarray(bf['extended_insulin_norm'], dtype=np.float32)[:_ps]
            carb_raw = np.asarray(bf['extended_carb_raw'], dtype=np.float32)[:_ps]
            ins_raw = np.asarray(bf['extended_insulin_raw'], dtype=np.float32)[:_ps]
            if carb_norm.shape[0] < _ps or ins_norm.shape[0] < _ps:
                continue

            carb_true_t = torch.from_numpy(
                carb_norm.reshape(PREDICTION_PATCHES, PATCH_SIZE))
            ins_true_t = torch.from_numpy(
                ins_norm.reshape(PREDICTION_PATCHES, PATCH_SIZE))

            baseline = _forecast(carb_true_t, ins_true_t)               # (P*S,)

            # +full carb bolus, insulin at truth.
            carb_full = _renorm(_perturb_raw(carb_raw, carb_B), carb_m, carb_s)
            carb_pert = _forecast(carb_full, ins_true_t)
            carb_dbg = float(np.mean(carb_pert - baseline))
            carb_dbg_sum += carb_dbg
            carb_dir_hits += int(carb_dbg > 0.0)

            # +full insulin bolus, carb at truth.
            ins_full = _renorm(_perturb_raw(ins_raw, ins_B), ins_m, ins_s)
            ins_pert = _forecast(carb_true_t, ins_full)
            ins_dbg = float(np.mean(ins_pert - baseline))
            ins_dbg_sum += ins_dbg
            ins_dir_hits += int(ins_dbg < 0.0)

            # Monotonicity across [0, B/2, B].
            carb_half = _renorm(_perturb_raw(carb_raw, carb_B / 2.0), carb_m, carb_s)
            carb_peaks = [
                float(baseline.max()),
                float(_forecast(carb_half, ins_true_t).max()),
                float(carb_pert.max()),
            ]
            carb_mono_hits += int(
                carb_peaks[1] >= carb_peaks[0] - 1e-6
                and carb_peaks[2] >= carb_peaks[1] - 1e-6)

            ins_half = _renorm(_perturb_raw(ins_raw, ins_B / 2.0), ins_m, ins_s)
            ins_mins = [
                float(baseline.min()),
                float(_forecast(carb_true_t, ins_half).min()),
                float(ins_pert.min()),
            ]
            ins_mono_hits += int(
                ins_mins[1] <= ins_mins[0] + 1e-6
                and ins_mins[2] <= ins_mins[1] + 1e-6)

            # Hypo rescue: baseline-hypo samples lifted out by +carb.
            if float(baseline.min()) < hypo_threshold:
                hypo_n += 1
                hypo_rescue_hits += int(float(carb_pert.min()) >= hypo_threshold)

            # Hyper rescue: baseline-hyper samples brought down by +insulin.
            if float(baseline.max()) > hyper_threshold:
                hyper_n += 1
                hyper_rescue_hits += int(float(ins_pert.max()) <= hyper_threshold)

            n_probed += 1

    model.train(was_training)

    nz = max(n_probed, 1)
    return {
        'cf_carb_dbg': carb_dbg_sum / nz if n_probed else None,
        'cf_carb_dir': carb_dir_hits / nz if n_probed else None,
        'cf_insulin_dbg': ins_dbg_sum / nz if n_probed else None,
        'cf_insulin_dir': ins_dir_hits / nz if n_probed else None,
        'cf_carb_monotonic': carb_mono_hits / nz if n_probed else None,
        'cf_insulin_monotonic': ins_mono_hits / nz if n_probed else None,
        'cf_hypo_rescue': (hypo_rescue_hits / hypo_n) if hypo_n > 0 else None,
        'cf_hyper_rescue': (hyper_rescue_hits / hyper_n) if hyper_n > 0 else None,
        'cf_n': n_probed,
        'cf_hypo_n': hypo_n,
        'cf_hyper_n': hyper_n,
    }


_CONF_MEDIAN_IDX = QUANTILE_LEVELS.index(0.5)
_CONF_LO_IDX = QUANTILE_LEVELS.index(0.05)
_CONF_HI_IDX = QUANTILE_LEVELS.index(0.95)
_CONF_HYPO_IDX = QUANTILE_LEVELS.index(0.10)


def _conformal_val_probe(bands: np.ndarray, true: np.ndarray, last: np.ndarray) -> dict:
    """Raw-vs-calibrated band-coverage probe on a held-out split of the val windows.

    A live witness that split-conformal recalibration (``conformal.py``) would restore
    the band coverage the raw fan loses at excursions — without changing the median.
    Fits per-(step, quantile) corrections on a deterministic 60% of the collected val
    windows and measures EXCURSION-PEAK coverage on the disjoint 40%. The DEPLOYABLE
    delta is fit on the disjoint reserved partition by ``calibrate_conformal.py``; this
    is only the in-training signal, and the val sample is small (~100 windows), so read
    it as directional, not exact.

    Args:
        bands: ``(M, H, K)`` mg/dL quantile fans; true: ``(M, H)`` mg/dL; last: ``(M,)``.

    Returns:
        ``{conf_cov90_raw, conf_cov90_cal, conf_hypo_esc_raw, conf_hypo_esc_cal, conf_n}``
        (empty if too few excursion windows to be meaningful).
    """
    import conformal
    M = bands.shape[0]
    if M < 50:
        return {}
    perm = np.random.default_rng(0).permutation(M)
    ncal = int(0.6 * M)
    ci, ti = perm[:ncal], perm[ncal:]
    delta = conformal.fit_quantile_conformal(
        bands[ci], true[ci], QUANTILE_LEVELS, _CONF_MEDIAN_IDX)
    bt, tt, lt = bands[ti], true[ti], last[ti]
    j = np.argmax(np.abs(tt - lt[:, None]), axis=1)            # per-window true-peak step
    idx = np.where((tt.max(1) - tt.min(1)) > 25.0)[0]          # excursion windows only
    if len(idx) < 20:
        return {}

    def _stats(B: np.ndarray) -> tuple[float, float]:
        pt = tt[idx, j[idx]]                                   # true at the peak
        cov = np.mean((B[idx, j[idx], _CONF_LO_IDX] <= pt)
                      & (pt <= B[idx, j[idx], _CONF_HI_IDX]))
        hypo = np.mean(pt < B[idx, j[idx], _CONF_HYPO_IDX])    # truth below the τ=0.10 hypo edge
        return float(cov), float(hypo)

    bt_cal = conformal.apply_quantile_conformal(bt, delta, _CONF_MEDIAN_IDX)
    cov_raw, hypo_raw = _stats(bt)
    cov_cal, hypo_cal = _stats(bt_cal)
    return {'conf_cov90_raw': cov_raw, 'conf_cov90_cal': cov_cal,
            'conf_hypo_esc_raw': hypo_raw, 'conf_hypo_esc_cal': hypo_cal,
            'conf_n': float(len(idx))}


def _run_validation(
    model: T1DMAI,
    val_dataset: T1DMDataset,
    norm_stats: dict,
    device: torch.device,
    weighting: KendallGalWeighting,
    bg_hypo_threshold: float = BG_HYPO_THRESHOLD,
    bg_hyper_threshold: float = BG_HYPER_THRESHOLD,
) -> dict[str, Any]:
    """Run validation on a fixed set of patients, processing in batches.

    ``pred_bg = f_inv(median)`` is the SOLE BG forecast into the metric suite.
    Returns ``val_loss_total`` (= ``risk_total_loss``), ``val_loss_Q``,
    ``val_pinball`` (diagnostic), per-(h,τ) marginal coverage, plus the kept
    BG/Clarke/CG-EGA/roc/nocturnal/rolling diagnostics.

    ``risk_total_loss`` runs the DILATE soft-DTW DP at the validation batch size
    (VAL_BATCH_SIZE=8), which is cheap; ``val_loss_total`` is kept literally
    equal to ``risk_total_loss`` and ``val_loss_D`` is surfaced directly from the
    validation loss components (no separate train running-mean).
    """
    model.eval()
    totals: dict[str, float] = {'loss_total': 0.0, 'loss_Q': 0.0, 'loss_D': 0.0, 'pinball': 0.0}
    n_samples = 0

    agg: dict[str, float] = {}
    night_agg: dict[str, float] = {}

    n_val = min(len(val_dataset), VALIDATION_N_PATIENTS)

    # Per-window full mg/dL fans + truth + anchor, collected for the conformal
    # coverage probe (raw-vs-calibrated band coverage, see _conformal_val_probe).
    conf_bands_list: list[np.ndarray] = []
    conf_true_list: list[np.ndarray] = []
    conf_last_list: list[np.ndarray] = []

    # Materialized (index-ordered) val samples, reused by the counterfactual probe
    # instead of re-indexing the dataset (which would re-run the simulator).
    val_samples_ordered: list[dict] = []

    # Time-of-day probe: collect per-sample decoded hour + confidence across the
    # WHOLE val set. Clock-reliability stats (bias, precision, p90 tail, gross-error
    # rate, confidence-selective MAE) need the full residual distribution, not the
    # running sums the point metrics could get away with. The headline tod_* metrics
    # stay comparable to pred_start_hour, so they decode PATCH 0; tod_jump_vals
    # collects the per-sample inter-patch phase-advance deviation (the no-jumping
    # witness) over ALL patches.
    tod_adv = PATCH_SIZE * STEP_MINUTES / 60.0        # 30 min per patch = 0.5 h (PATCH_SIZE-invariant)
    tod_pred_hours: list[torch.Tensor] = []
    tod_true_hours: list[torch.Tensor] = []
    tod_conf_vals: list[torch.Tensor] = []
    tod_jump_vals: list[torch.Tensor] = []
    tod_xwin_vals: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_start in range(0, n_val, VAL_BATCH_SIZE):
            batch_end = min(batch_start + VAL_BATCH_SIZE, n_val)
            samples = [val_dataset[i] for i in range(batch_start, batch_end)]
            val_samples_ordered.extend(samples)
            batch = collate_fn(samples)

            patches = batch['patches'].to(device, non_blocking=True)
            attn_mask = batch['attn_mask'].to(device, non_blocking=True)
            bg_formula = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                          for k, v in batch['bg_formula_data'].items()}
            last_bg = bg_formula['last_bg'].float()                       # (B,)
            # risk_total_loss requires the target shaped (B, P, S) to match
            # ``median``; reshape the flat (B, P*S) raw trajectory.
            true_bg_full = (
                bg_formula['true_bg_trajectory'][:, :PREDICTION_PATCHES * PATCH_SIZE]
                .float().reshape(-1, PREDICTION_PATCHES, PATCH_SIZE)
            )

            q_tau, median, time_pred = model(patches, attn_mask, last_bg, return_time=True)
            q_tau = q_tau.float()
            median = median.float()
            B_batch = median.shape[0]

            # Loss in fp32. f is applied to the target exactly once inside
            # risk_total_loss; pinball + DILATE are combined by the learned
            # Kendall-Gal weighting.
            loss_total, parts = risk_total_loss(
                q_tau, median, true_bg_full, weighting,
            )

            # pred_bg: the SOLE headline forecast.
            pred_bg = _median_to_mgdl(median)                            # (B, P*S)

            # Median forecast roughness (risk space): mean |Δ²median| over the
            # patch-major flat horizon (pooled) and over the last patch (far) —
            # the anti-oscillation witness the headline RMSE structurally masks.
            m_flat = median.reshape(B_batch, -1)                         # (B, P*S) patch-major
            d2 = m_flat[:, 2:] - 2.0 * m_flat[:, 1:-1] + m_flat[:, :-2]  # (B, P*S-2)
            agg['median_rough_abs_sum'] = agg.get('median_rough_abs_sum', 0.0) + float(d2.abs().sum())
            agg['median_rough_cnt'] = agg.get('median_rough_cnt', 0.0) + float(d2.numel())
            _far0 = (PREDICTION_PATCHES - 1) * PATCH_SIZE - 1            # first Δ² centred in the last patch
            d2_far = d2[:, _far0:]
            agg['median_rough_far_abs_sum'] = agg.get('median_rough_far_abs_sum', 0.0) + float(d2_far.abs().sum())
            agg['median_rough_far_cnt'] = agg.get('median_rough_far_cnt', 0.0) + float(d2_far.numel())

            # Band edges in mg/dL: the 90% interval (τ.05/.95) for coverage and
            # the inner-50% edges (τ.25/.75) for inner50_cov. f_inv is elementwise,
            # so invert the whole fan ONCE and index the edges out of it (identical
            # to inverting each slice, one pass instead of five).
            conf_full = kovatchev_f_inv(q_tau)                           # (B, P, S, 7) mg/dL
            q_lo = conf_full[..., _TAU_LO_IDX].reshape(B_batch, -1)
            q_hi = conf_full[..., _TAU_HI_IDX].reshape(B_batch, -1)
            q_inner_lo = conf_full[..., _TAU_INNER_LO_IDX].reshape(B_batch, -1)
            q_inner_hi = conf_full[..., _TAU_INNER_HI_IDX].reshape(B_batch, -1)
            # Clinical hypo/hyper ALARM band edges (τ=HYPO/HYPER_ALARM_QUANTILE_TAU lower/upper).
            q_hypo_lo = conf_full[..., _HYPO_BAND_IDX].reshape(B_batch, -1)
            q_hyper_hi = conf_full[..., _HYPER_BAND_IDX].reshape(B_batch, -1)
            q_mgdl = {'lo': q_lo, 'hi': q_hi,
                      'inner_lo': q_inner_lo, 'inner_hi': q_inner_hi,
                      'hypo_lo': q_hypo_lo, 'hyper_hi': q_hyper_hi}

            # Collect the FULL mg/dL fan + truth + anchor for the conformal probe.
            conf_bands_list.append(
                conf_full.reshape(B_batch, -1, N_QUANTILES).detach().cpu().numpy())
            conf_true_list.append(
                true_bg_full.reshape(B_batch, -1).detach().cpu().numpy())
            conf_last_list.append(last_bg.detach().cpu().numpy())

            learn = compute_learning_metrics(
                pred_bg, q_mgdl, bg_formula, PREDICTION_PATCHES,
                hypo_threshold=bg_hypo_threshold,
                hyper_threshold=bg_hyper_threshold,
            )
            for k, v in learn.items():
                agg[k] = agg.get(k, 0.0) + v

            # Long-horizon rolling pass (night-only accumulation inside).
            n_rolls = math.ceil(NIGHT_LONG_HORIZON_HOURS / PREDICTION_HORIZON_HOURS)
            if n_rolls > 1:
                _accumulate_long_horizon_bg_metrics(
                    model, samples, norm_stats, device, n_rolls, agg,
                    night_agg=night_agg,
                )

            # Nocturnal subset (no second forward).
            pred_start_hours = batch['bg_formula_data'].get('pred_start_hour')
            if pred_start_hours is not None:
                night_idx = [j for j, h in enumerate(pred_start_hours)
                             if _is_nocturnal(float(h))]
                if night_idx:
                    nidx = torch.tensor(night_idx, device=device, dtype=torch.long)
                    night_bg_formula = {
                        k: (v.index_select(0, nidx)
                            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B_batch
                            else v)
                        for k, v in bg_formula.items()
                    }
                    night_learn = compute_learning_metrics(
                        pred_bg.index_select(0, nidx),
                        {'lo': q_lo.index_select(0, nidx), 'hi': q_hi.index_select(0, nidx),
                         'inner_lo': q_inner_lo.index_select(0, nidx),
                         'inner_hi': q_inner_hi.index_select(0, nidx),
                         'hypo_lo': q_hypo_lo.index_select(0, nidx),
                         'hyper_hi': q_hyper_hi.index_select(0, nidx)},
                        night_bg_formula, PREDICTION_PATCHES,
                        hypo_threshold=bg_hypo_threshold,
                        hyper_threshold=bg_hyper_threshold,
                    )
                    for k, v in night_learn.items():
                        night_agg[k] = night_agg.get(k, 0.0) + v

            totals['loss_total'] += float(loss_total) * B_batch
            totals['loss_Q'] += float(parts.get('loss_Q', float('nan'))) * B_batch
            totals['loss_D'] += float(parts.get('loss_D', float('nan'))) * B_batch
            totals['pinball'] += float(parts.get('pinball', parts.get('loss_Q', float('nan')))) * B_batch

            # Time-of-day probe (diagnostic only — never enters the loss totals or
            # checkpoint selection). Decode PATCH 0's bin logits to a wall-clock hour
            # + resultant-length confidence R (the patch aligned to pred_start_hour);
            # collect per-sample hour + R for the clock-reliability stats, plus the
            # per-sample inter-patch advance-deviation for the no-jumping witness.
            _psh = batch['bg_formula_data'].get('pred_start_hour')
            if time_pred is not None and _psh is not None:
                hours0, R0 = time_of_day_decode_bins(time_pred[:, 0, :], TIME_PROBE_N_BINS)  # (B,)
                tod_pred_hours.append(hours0.detach().cpu())
                tod_true_hours.append(_psh.float().reshape(-1).detach().cpu())
                tod_conf_vals.append(R0.detach().cpu())
                tod_jump_vals.append(
                    time_inter_patch_jump_hours(time_pred, TIME_PROBE_N_BINS, tod_adv).detach().cpu()
                )
                # Cross-window no-jump witness (diagnostic): a 2nd forward on window k+1
                # (batch['next_window'], teacher-forced true shifted trajectory); measure
                # |clock_{k+1,origin} - clock_{k,origin} - H| per valid sample. Full batch
                # (no fraction subsample in validation). Reuses attn_mask (same n_ctx).
                if TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
                    _nw = batch.get('next_window')
                    if _nw is not None and bool(_nw['valid'].any()):
                        _nw_patches = _nw['patches'].to(device, non_blocking=True)
                        _nw_last_bg = _nw['last_bg'].to(device, non_blocking=True).float()
                        _nw_valid = _nw['valid'].to(device, non_blocking=True)
                        _, _, _time_pred_next = model(
                            _nw_patches, attn_mask, _nw_last_bg, return_time=True,
                        )
                        _xwin = time_cross_window_jump_hours(
                            time_pred, _time_pred_next, TIME_PROBE_N_BINS, PREDICTION_HORIZON_HOURS,
                        )                                                    # (B,)
                        tod_xwin_vals.append(_xwin[_nw_valid].detach().cpu())
            n_samples += B_batch

    model.train()
    n = max(n_samples, 1)

    result: dict[str, Any] = {
        'val_loss_total': totals['loss_total'] / n,
        'val_loss_Q': totals['loss_Q'] / n,
        'val_loss_D': totals['loss_D'] / n,
        'val_pinball': totals['pinball'] / n,
        'log_sigma_Q': float(weighting.log_sigma_Q.detach()),
        'log_sigma_D': float(weighting.log_sigma_D.detach()),
    }

    # ---- Finalize time-of-day probe (diagnostic; NOT in any loss total) ----
    # Point accuracy PLUS clock-reliability stats over the full val distribution:
    # bias/precision decomposition, the p90 tail, the gross-error (blunder) rate,
    # and the confidence-selective MAE (is the magnitude-R confidence actually usable
    # as a "trust the reading" gate).
    if tod_pred_hours:
        _ph = torch.cat(tod_pred_hours)
        _th = torch.cat(tod_true_hours)
        _R = torch.cat(tod_conf_vals)
        _ae = circular_hour_error(_ph, _th)                       # (N,) in [0, 12]
        _pb = (_ph // 6).long() % 4
        _tb = (_th // 6).long() % 4
        result['tod_mae_h'] = float(_ae.mean())
        result['tod_acc_1h'] = float(100.0 * (_ae <= 1.0).float().mean())
        result['tod_acc_2h'] = float(100.0 * (_ae <= 2.0).float().mean())
        result['tod_acc_bin'] = float(100.0 * (_pb == _tb).float().mean())
        result['tod_conf'] = float(_R.mean())
        result['tod_bias_h'] = float(circular_bias_hours(_ph, _th))
        result['tod_std_h'] = float(circular_std_hours(_ph, _th))
        result['tod_p90_h'] = float(torch.quantile(_ae, 0.9))
        result['tod_gross_rate'] = float(100.0 * (_ae > 3.0).float().mean())
        # Confidence-selective reliability: MAE on the top-half-confidence readings.
        _hi = _R >= _R.median()
        result['tod_mae_hiconf'] = (
            float(_ae[_hi].mean()) if bool(_hi.any()) else float(_ae.mean())
        )
        # No-jumping witness: mean |inter-patch advance deviation| in hours (~0 means
        # the predicted clock marches forward one patch = tod_adv hours per step).
        if tod_jump_vals:
            result['tod_jump_h'] = float(torch.cat(tod_jump_vals).mean())
        if tod_xwin_vals:
            result['tod_xwin_jump_h'] = float(torch.cat(tod_xwin_vals).mean())

    # ---- Finalize BG metrics ----
    for h_min in BG_HORIZONS_MIN:
        cnt = agg.get(f'bg_rmse_{h_min}_cnt', 0.0)
        if cnt > 0:
            result[f'bg_rmse_{h_min}'] = math.sqrt(agg[f'bg_rmse_{h_min}_sq_sum'] / cnt)
            result[f'bg_mae_{h_min}'] = agg[f'bg_mae_{h_min}_abs_sum'] / cnt
        else:
            result[f'bg_rmse_{h_min}'] = None
            result[f'bg_mae_{h_min}'] = None

    _rc = agg.get('median_rough_cnt', 0.0)
    result['median_roughness'] = (agg['median_rough_abs_sum'] / _rc) if _rc > 0 else None
    _rfc = agg.get('median_rough_far_cnt', 0.0)
    result['median_roughness_far'] = (agg['median_rough_far_abs_sum'] / _rfc) if _rfc > 0 else None

    result['tir_err'] = agg.get('tir_err_sum', 0.0) / max(agg.get('tir_err_cnt', 0.0), 1.0)
    result['pred_tir'] = agg.get('pred_tir_sum', 0.0) / max(agg.get('tir_err_cnt', 0.0), 1.0)
    result['true_tir'] = agg.get('true_tir_sum', 0.0) / max(agg.get('tir_err_cnt', 0.0), 1.0)
    result['tbr_err'] = agg.get('tbr_err_sum', 0.0) / max(agg.get('tir_err_cnt', 0.0), 1.0)
    result['tar_err'] = agg.get('tar_err_sum', 0.0) / max(agg.get('tir_err_cnt', 0.0), 1.0)

    result['hypo_recall'] = (
        agg.get('hypo_recall_hit', 0.0) / agg['hypo_true'] if agg.get('hypo_true', 0.0) > 0 else None)
    result['hypo_precision'] = (
        agg.get('hypo_prec_hit', 0.0) / agg['hypo_pred'] if agg.get('hypo_pred', 0.0) > 0 else None)
    result['hypo_n_steps'] = int(agg.get('hypo_true', 0.0))
    result['hyper_recall'] = (
        agg.get('hyper_recall_hit', 0.0) / agg['hyper_true'] if agg.get('hyper_true', 0.0) > 0 else None)
    result['hyper_precision'] = (
        agg.get('hyper_prec_hit', 0.0) / agg['hyper_pred'] if agg.get('hyper_pred', 0.0) > 0 else None)
    result['hyper_n_steps'] = int(agg.get('hyper_true', 0.0))

    for _h in _excursion_bucket_horizons(PREDICTION_PATCHES):
        _ht = agg.get(f'hypo_true@{_h}', 0.0)
        _hp = agg.get(f'hypo_pred@{_h}', 0.0)
        result[f'hypo_recall@{_h}'] = (
            agg.get(f'hypo_recall_hit@{_h}', 0.0) / _ht if _ht > 0 else None)
        result[f'hypo_precision@{_h}'] = (
            agg.get(f'hypo_prec_hit@{_h}', 0.0) / _hp if _hp > 0 else None)
        result[f'hypo_n_steps@{_h}'] = int(_ht)
        _yt = agg.get(f'hyper_true@{_h}', 0.0)
        _yp = agg.get(f'hyper_pred@{_h}', 0.0)
        result[f'hyper_recall@{_h}'] = (
            agg.get(f'hyper_recall_hit@{_h}', 0.0) / _yt if _yt > 0 else None)
        result[f'hyper_precision@{_h}'] = (
            agg.get(f'hyper_prec_hit@{_h}', 0.0) / _yp if _yp > 0 else None)
        result[f'hyper_n_steps@{_h}'] = int(_yt)

    _cgega_counts = {k: agg.get(f'cgega_{k}', 0.0) for k in (
        'ap_hypo', 'be_hypo', 'ep_hypo',
        'ap_eu', 'be_eu', 'ep_eu',
        'ap_hyper', 'be_hyper', 'ep_hyper')}
    for _k, _v in cg_ega.cg_ega_fractions(_cgega_counts).items():
        result[f'cgega_{_k}'] = _v

    clarke_total = max(agg.get('clarke_total', 0.0), 1.0)
    result['clarke_AB_pct'] = 100.0 * (agg.get('clarke_A', 0.0) + agg.get('clarke_B', 0.0)) / clarke_total
    result['clarke_D_pct'] = 100.0 * agg.get('clarke_D', 0.0) / clarke_total
    result['clarke_E_pct'] = 100.0 * agg.get('clarke_E', 0.0) / clarke_total

    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        c_cnt = agg.get(f'evalfix_clarke_A@{h_min}_cnt', 0.0)
        result[f'evalfix_clarke_A@{h_min}'] = (
            100.0 * agg.get(f'evalfix_clarke_A@{h_min}', 0.0) / c_cnt if c_cnt > 0 else None)
        m_cnt = agg.get(f'evalfix_mard@{h_min}_cnt', 0.0)
        result[f'evalfix_mard@{h_min}'] = (
            100.0 * agg.get(f'evalfix_mard@{h_min}_sum', 0.0) / m_cnt if m_cnt > 0 else None)

    # Marginal per-(h, τ) coverage of the 90% band.
    for h_min in COVERAGE_HORIZONS_MIN:
        c_cnt = agg.get(f'coverage90@{h_min}_cnt', 0.0)
        result[f'coverage90@{h_min}'] = (
            agg.get(f'coverage90@{h_min}_hit', 0.0) / c_cnt if c_cnt > 0 else None)

    # Diagnostics: sign_balance (target 0.5) and inner50_cov (target 0.5).
    for h_min in COVERAGE_HORIZONS_MIN:
        s_cnt = agg.get(f'sign_balance@{h_min}_cnt', 0.0)
        result[f'sign_balance@{h_min}'] = (
            agg.get(f'sign_balance@{h_min}_below', 0.0) / s_cnt if s_cnt > 0 else None)
        i_cnt = agg.get(f'inner50_cov@{h_min}_cnt', 0.0)
        result[f'inner50_cov@{h_min}'] = (
            agg.get(f'inner50_cov@{h_min}_hit', 0.0) / i_cnt if i_cnt > 0 else None)

    roc_n = agg.get('roc_cnt', 0.0)
    if roc_n > 0:
        result['roc_rmse'] = math.sqrt(agg['roc_sq_sum'] / roc_n)
        sx = agg['roc_sum_x']; sy = agg['roc_sum_y']
        sxx = agg['roc_sum_xx']; syy = agg['roc_sum_yy']; sxy = agg['roc_sum_xy']
        var_x = sxx - (sx * sx) / roc_n
        var_y = syy - (sy * sy) / roc_n
        cov_xy = sxy - (sx * sy) / roc_n
        denom = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
        result['roc_corr'] = cov_xy / denom if denom > 1e-9 else 0.0
        result['trend_gain_beta'] = cov_xy / var_y if var_y > 1e-9 else None
        result['trend_amp_ratio'] = (
            math.sqrt(max(var_x, 0.0) / var_y) if var_y > 1e-9 else None)
    else:
        result['roc_rmse'] = None
        result['roc_corr'] = None
        result['trend_gain_beta'] = None
        result['trend_amp_ratio'] = None

    # Excursion-magnitude amplitude (net peak deviation), finalized from the
    # streaming five-sum + over/under counts. exc_amp_ratio = std(pred)/std(true)
    # at the true peak (target 1.0); exc_gain_beta = slope of pred on true
    # (target 1.0); exc_corr = their correlation; the over/under fractions are
    # the per-window inconsistency witnesses (re-centring trades one for the
    # other, variance-reduction lowers both).
    exc_n = agg.get('exc_cnt', 0.0)
    if exc_n > 0:
        s_pe = agg.get('exc_sum_pe', 0.0)
        s_te = agg.get('exc_sum_te', 0.0)
        var_pe = max(agg.get('exc_sum_pe2', 0.0) - s_pe * s_pe / exc_n, 0.0)
        var_te = max(agg.get('exc_sum_te2', 0.0) - s_te * s_te / exc_n, 0.0)
        cov_pt = agg.get('exc_sum_pete', 0.0) - s_pe * s_te / exc_n
        result['exc_amp_ratio'] = (
            math.sqrt(var_pe / var_te) if var_te > 1e-9 else None)
        result['exc_gain_beta'] = cov_pt / var_te if var_te > 1e-9 else None
        result['exc_corr'] = (
            cov_pt / math.sqrt(var_pe * var_te)
            if var_pe > 1e-9 and var_te > 1e-9 else None)
        result['exc_overshoot_frac'] = agg.get('exc_over', 0.0) / exc_n
        result['exc_undershoot_frac'] = agg.get('exc_under', 0.0) / exc_n
        result['exc_n'] = exc_n
    else:
        result['exc_amp_ratio'] = None
        result['exc_gain_beta'] = None
        result['exc_corr'] = None
        result['exc_overshoot_frac'] = None
        result['exc_undershoot_frac'] = None
        result['exc_n'] = 0.0

    # Conformal coverage probe (raw-vs-calibrated band coverage on a held-out val split).
    if conf_bands_list:
        result.update(_conformal_val_probe(
            np.concatenate(conf_bands_list, axis=0),
            np.concatenate(conf_true_list, axis=0),
            np.concatenate(conf_last_list, axis=0),
        ))

    def _curve_corr(prefix: str) -> float | None:
        n_c = agg.get(f'{prefix}_n', 0.0)
        if n_c <= 1:
            return None
        sx = agg.get(f'{prefix}_sx', 0.0); sy = agg.get(f'{prefix}_sy', 0.0)
        sxx = agg.get(f'{prefix}_sxx', 0.0); syy = agg.get(f'{prefix}_syy', 0.0)
        sxy = agg.get(f'{prefix}_sxy', 0.0)
        var_x = sxx - (sx * sx) / n_c
        var_y = syy - (sy * sy) / n_c
        cov_xy = sxy - (sx * sy) / n_c
        denom = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
        return cov_xy / denom if denom > 1e-9 else None
    result['bg_curve_corr'] = _curve_corr('bgcurve')

    # ---- Nocturnal metrics ----
    for h_min in BG_HORIZONS_MIN:
        cnt = night_agg.get(f'night_bg_rmse_{h_min}_cnt', 0.0)
        if cnt > 0:
            result[f'night_bg_rmse_{h_min}'] = math.sqrt(
                night_agg[f'night_bg_rmse_{h_min}_sq_sum'] / cnt)
        else:
            cnt_single = night_agg.get(f'bg_rmse_{h_min}_cnt', 0.0)
            if cnt_single > 0:
                result[f'night_bg_rmse_{h_min}'] = math.sqrt(
                    night_agg[f'bg_rmse_{h_min}_sq_sum'] / cnt_single)
            else:
                result[f'night_bg_rmse_{h_min}'] = None

    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        m_cnt = night_agg.get(f'evalfix_mard@{h_min}_cnt', 0.0)
        result[f'night_mard@{h_min}'] = (
            100.0 * night_agg.get(f'evalfix_mard@{h_min}_sum', 0.0) / m_cnt if m_cnt > 0 else None)

    night_hypo_true = night_agg.get('hypo_true', 0.0)
    result['night_hypo_recall'] = (
        night_agg.get('hypo_recall_hit', 0.0) / night_hypo_true if night_hypo_true > 0 else None)
    result['night_hypo_n_steps'] = int(night_hypo_true)
    night_hypo_pred = night_agg.get('hypo_pred', 0.0)
    result['night_hypo_precision'] = (
        night_agg.get('hypo_prec_hit', 0.0) / night_hypo_pred if night_hypo_pred > 0 else None)

    night_hyper_true = night_agg.get('hyper_true', 0.0)
    result['night_hyper_recall'] = (
        night_agg.get('hyper_recall_hit', 0.0) / night_hyper_true if night_hyper_true > 0 else None)
    result['night_hyper_n_steps'] = int(night_hyper_true)
    night_hyper_pred = night_agg.get('hyper_pred', 0.0)
    result['night_hyper_precision'] = (
        night_agg.get('hyper_prec_hit', 0.0) / night_hyper_pred if night_hyper_pred > 0 else None)

    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        c_cnt = night_agg.get(f'evalfix_clarke_A@{h_min}_cnt', 0.0)
        result[f'night_clarke_A@{h_min}'] = (
            100.0 * night_agg.get(f'evalfix_clarke_A@{h_min}', 0.0) / c_cnt if c_cnt > 0 else None)

    _night_cgega_counts = {k: night_agg.get(f'cgega_{k}', 0.0) for k in (
        'ap_hypo', 'be_hypo', 'ep_hypo',
        'ap_eu', 'be_eu', 'ep_eu',
        'ap_hyper', 'be_hyper', 'ep_hyper')}
    for _k, _v in cg_ega.cg_ega_fractions(_night_cgega_counts).items():
        result[f'night_cgega_{_k}'] = _v

    # Counterfactual dose-response probe (diagnostic; one pass over the same
    # validation samples, conditioned on the announced doses).
    result.update(_run_counterfactual_probe(
        model, val_dataset, norm_stats, device,
        hypo_threshold=bg_hypo_threshold, hyper_threshold=bg_hyper_threshold,
        samples=val_samples_ordered,
    ))

    return result


# ============================================================================
# Checkpoint build
# ============================================================================

def _build_checkpoint(
    model: T1DMAI,
    weighting: KendallGalWeighting,
    muon_opt: Muon,
    adam_opt: torch.optim.AdamW,
    step: int,
    loss_history: list[float],
    training_config: dict,
    normalization_stats: dict,
    master_seed: int,
    val_history: list[dict],
    best_val_loss: float,
    best_val_step: int,
    loss_ema: float | None,
    ema: "ModelEMA | None" = None,
) -> dict:
    """Build a serializable checkpoint dict.

    ``arch_version`` / ``loss_schema`` are stamped as checkpoint provenance
    metadata so the arch / loss schema a checkpoint was produced under is
    self-describing. The two Kendall-Gal log-σ parameters live off ``model`` (on
    ``weighting``), so they are serialized separately as ``weighting_state_dict``.
    """
    ckpt = {
        'arch_version': ARCH_VERSION,
        'loss_schema': LOSS_SCHEMA,
        'step': step,
        'model_state_dict': model.state_dict(),
        'weighting_state_dict': weighting.state_dict(),
        'muon_optimizer_state_dict': muon_opt.state_dict(),
        'adam_optimizer_state_dict': adam_opt.state_dict(),
        'training_config': training_config,
        'normalization_stats': normalization_stats,
        'master_seed': master_seed,
        'loss_history': loss_history,
        'val_history': val_history,
        'best_val_loss': best_val_loss,
        'best_val_step': best_val_step,
        'loss_ema': loss_ema,
    }
    if ema is not None:
        ckpt['model_ema_state_dict'] = ema.state_dict()
    return ckpt


# ============================================================================
# Main training loop
# ============================================================================

def train(
    total_steps: int = TOTAL_STEPS,
    batch_size: int = BATCH_SIZE,
    master_seed: int = MASTER_SEED,
    num_workers: int = NUM_WORKERS,
    log_interval: int = LOG_INTERVAL,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    validation_interval: int = VALIDATION_INTERVAL,
    device: torch.device | None = None,
    muon_lr: float = MUON_LR,
    muon_momentum: float = MUON_MOMENTUM,
    adam_lr: float = ADAM_LR,
    warmup_steps: int = WARMUP_STEPS,
    lr_min_ratio: float = LR_MIN_RATIO,
    gradient_clip_norm: float = GRADIENT_CLIP_NORM,
    patient_uniform_sample_prob: float = PATIENT_UNIFORM_SAMPLE_PROB,
    adam_weight_decay: float = ADAM_WEIGHT_DECAY,
    weight_decay_schedule_correction: bool = WEIGHT_DECAY_SCHEDULE_CORRECTION,
    simulator_warmup_hours: float = SIMULATOR_WARMUP_HOURS,
    ema_decay: float = EMA_DECAY,
    bg_hypo_threshold: float = BG_HYPO_THRESHOLD,
    bg_hyper_threshold: float = BG_HYPER_THRESHOLD,
    cache_path: str | None = None,
) -> list[float]:
    """Run the T1DMAI training loop. Returns the per-step total-loss history."""
    # Full determinism (gated on config) must run before any model / optimizer /
    # dataloader is constructed so every downstream RNG draw is reproducible.
    if DETERMINISTIC:
        setup_determinism(master_seed)
        print(f"Determinism enabled (seed={master_seed}; TF32 off, cuDNN deterministic)")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f"Training on: {device}")

    if device.type == 'cuda' and not DETERMINISTIC:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    train_start_time = time.time()

    # ------------------------------------------------------------------ #
    # Normalization statistics
    # ------------------------------------------------------------------ #
    if os.path.exists(NORM_STATS_FILE):
        norm_stats = load_normalization_stats()
        print(f"Loaded normalization stats from {NORM_STATS_FILE}")
    else:
        print("Computing normalization statistics (this takes ~2-5 minutes)...")
        norm_stats = compute_normalization_stats(
            master_seed=master_seed,
            patient_uniform_sample_prob=patient_uniform_sample_prob,
            simulator_warmup_hours=simulator_warmup_hours,
        )
        save_normalization_stats(norm_stats)

    # ------------------------------------------------------------------ #
    # Model and optimizers
    # ------------------------------------------------------------------ #
    # When DETERMINISTIC, setup_determinism() already seeded torch + all CUDA
    # devices at the top of train(); seed here only on the non-deterministic path
    # so model init is still reproducible without double-seeding.
    if not DETERMINISTIC:
        torch.manual_seed(master_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(master_seed)

    model = T1DMAI().to(device)

    # The two learned Kendall-Gal log-σ live on this separate tiny module — NOT
    # on ``model`` — so the weight EMA (which wraps only ``model``) never touches
    # them; they get their own weight_decay=0 AdamW group in _build_optimizers.
    weighting = KendallGalWeighting().to(device)

    muon_opt, adam_opt = _build_optimizers(
        model, weighting, muon_lr, adam_lr, muon_momentum,
        adam_weight_decay=adam_weight_decay,
    )

    # Weight EMA for evaluation stability. It wraps only ``model`` — the
    # Kendall-Gal log-σ (on ``weighting``) are EMA-excluded by living off-model.
    ema: ModelEMA | None = None
    if ema_decay > 0.0:
        ema = ModelEMA(model, decay=ema_decay).to(device)
        print(f"Weight EMA enabled (decay={ema_decay})")

    start_step = 0
    loss_history: list[float] = []
    val_history: list[dict] = []
    best_val_loss = float('inf')
    best_val_step = -1
    loss_ema: float | None = None
    consecutive_nan = 0
    prev_val_metrics: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Dataset and DataLoader
    # ------------------------------------------------------------------ #
    dataset = T1DMDataset(
        master_seed=master_seed,
        total_steps=total_steps,
        batch_size=batch_size,
        normalization_stats=norm_stats,
        patient_uniform_sample_prob=patient_uniform_sample_prob,
        simulator_warmup_hours=simulator_warmup_hours,
        cache_path=cache_path,
    )
    val_dataset = T1DMDataset(
        master_seed=master_seed + 10_000_000,
        total_steps=VALIDATION_N_PATIENTS,
        batch_size=1,
        normalization_stats=norm_stats,
        # The model is always conditioned: the prediction-zone carbs+insulin are
        # announced (the future doses ride on each sample's bg_formula_data and
        # are fed through the inference override path).
        patient_uniform_sample_prob=patient_uniform_sample_prob,
        simulator_warmup_hours=simulator_warmup_hours,
        cache_path=cache_path,
        cache_partition='val',
    )
    val_dataset_night_onset = T1DMDataset(
        master_seed=master_seed + 10_000_000,
        total_steps=VALIDATION_N_PATIENTS, batch_size=1,
        normalization_stats=norm_stats,
        force_pred_start_hour=NOCTURNAL_START_HOUR,
        patient_uniform_sample_prob=patient_uniform_sample_prob,
        simulator_warmup_hours=simulator_warmup_hours, cache_path=cache_path,
        cache_partition='val',
    )

    sampler = _OffsetSampler(len(dataset), offset=start_step * batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True if num_workers > 0 else False,
        # 2 (not 8): prefetch_factor * num_workers batches sit buffered in
        # anonymous RAM (~10 GB at 8*20=160 on the shared unified-memory pool),
        # and each refill is a page-cache burst. 2*20=40 batches keeps the GPU
        # fed without hoarding the pool.
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )

    # ------------------------------------------------------------------ #
    # Training config (save once)
    # ------------------------------------------------------------------ #
    from config import (
        D_MODEL as _CFG_D_MODEL, N_LAYERS as _CFG_N_LAYERS,
        N_HEADS as _CFG_N_HEADS, FFN_DIM as _CFG_FFN_DIM,
    )
    training_config = {
        'arch_version': ARCH_VERSION, 'loss_schema': LOSS_SCHEMA,
        'master_seed': master_seed, 'total_steps': total_steps, 'batch_size': batch_size,
        'num_workers': num_workers,
        'd_model': _CFG_D_MODEL, 'n_layers': _CFG_N_LAYERS, 'n_heads': _CFG_N_HEADS,
        'ffn_dim': _CFG_FFN_DIM, 'patch_size': PATCH_SIZE, 'max_context_patches': MAX_CONTEXT_PATCHES,
        'min_context_patches': MIN_CONTEXT_PATCHES, 'prediction_patches': PREDICTION_PATCHES,
        'prediction_horizon_hours': PREDICTION_HORIZON_HOURS,
        'night_long_horizon_hours': NIGHT_LONG_HORIZON_HOURS,
        'muon_lr': muon_lr, 'muon_momentum': muon_momentum, 'adam_lr': adam_lr,
        'adam_weight_decay': adam_weight_decay, 'warmup_steps': warmup_steps,
        'lr_min_ratio': lr_min_ratio,
        'weight_decay_schedule_correction': weight_decay_schedule_correction,
        'gradient_clip_norm': gradient_clip_norm, 'checkpoint_interval': checkpoint_interval,
        'validation_interval': validation_interval, 'log_interval': log_interval,
        'patient_uniform_sample_prob': patient_uniform_sample_prob,
        'simulator_warmup_hours': simulator_warmup_hours,
        'ema_decay': ema_decay,
        'bg_hypo_threshold': bg_hypo_threshold,
        'bg_hyper_threshold': bg_hyper_threshold,
        'cache_path': cache_path,
    }
    with open('logs/resolved_config.json', 'w') as f:
        json.dump(training_config, f, indent=2)

    # ------------------------------------------------------------------ #
    # Training log CSV — re-keyed to the new loss components.
    # Columns: step, loss_total, loss_ema, loss_Q, loss_D, loss_D_shape,
    #          loss_D_tdi, loss_tod, loss_tod_xwin, log_sigma_Q, log_sigma_D,
    #          grad_norm, lr_muon, lr_adam, step_time_seconds, gpu_memory_mb
    # ------------------------------------------------------------------ #
    _eh = _excursion_bucket_horizons(PREDICTION_PATCHES)
    train_log_path = 'logs/training_log.csv'
    # A run always starts fresh, so always write a fresh header.
    train_log_exists = False
    train_log_file = open(train_log_path, 'a' if train_log_exists else 'w', newline='')
    train_log_writer = csv.writer(train_log_file)
    if not train_log_exists:
        train_log_writer.writerow([
            'step', 'loss_total', 'loss_ema',
            'loss_Q', 'loss_D', 'loss_D_shape', 'loss_D_tdi',
            'loss_tod', 'loss_tod_xwin',
            'log_sigma_Q', 'log_sigma_D',
            'grad_norm', 'lr_muon', 'lr_adam',
            'step_time_seconds', 'gpu_memory_mb',
        ])

    # ------------------------------------------------------------------ #
    # Validation log CSV — rewritten header (risk schema). A run always starts
    # fresh, so the header is always (re)written.
    # ------------------------------------------------------------------ #
    val_log_path = 'logs/validation_log.csv'
    _val_header = [
        'step',
        'val_loss_total', 'val_loss_Q', 'val_loss_D', 'val_pinball',
        'train_loss_ema', 'overfit_ratio',
        *[f'coverage90@{h}' for h in COVERAGE_HORIZONS_MIN],
        *[f'sign_balance@{h}' for h in COVERAGE_HORIZONS_MIN],
        *[f'inner50_cov@{h}' for h in COVERAGE_HORIZONS_MIN],
        *[f'bg_rmse_{h}' for h in BG_HORIZONS_MIN],
        *[f'evalfix_mard@{h}' for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
        'pred_tir', 'true_tir', 'tir_err',
        'hypo_recall', 'hypo_precision', 'hypo_n_steps',
        'hyper_recall', 'hyper_precision', 'hyper_n_steps',
        'cgega_ap_hypo', 'cgega_ep_hypo',
        'cgega_ap_eu', 'cgega_ep_eu',
        'cgega_ap_hyper', 'cgega_ep_hyper',
        *[f'evalfix_clarke_A@{h}' for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
        'clarke_AB_pct', 'clarke_D_pct', 'clarke_E_pct',
        'roc_rmse', 'roc_corr', 'trend_gain_beta', 'trend_amp_ratio',
        'bg_curve_corr',
        # Excursion-magnitude amplitude (NET peak deviation vs last_bg — the
        # over/under-dispersion the per-PATCH trend_amp_ratio misses).
        'exc_amp_ratio', 'exc_gain_beta', 'exc_corr',
        'exc_overshoot_frac', 'exc_undershoot_frac', 'exc_n',
        # Conformal coverage probe (raw vs calibrated band coverage at excursion peaks).
        'conf_cov90_raw', 'conf_cov90_cal',
        'conf_hypo_esc_raw', 'conf_hypo_esc_cal', 'conf_n',
        # Median forecast roughness (risk-space mean |Δ²median|): the
        # anti-oscillation witness the headline RMSE structurally masks. Pooled
        # over the horizon and over the last patch (where the zigzag concentrated).
        'median_roughness', 'median_roughness_far',
        # Counterfactual dose-response probe (diagnostic).
        'cf_carb_dbg', 'cf_carb_dir', 'cf_insulin_dbg', 'cf_insulin_dir',
        'cf_carb_monotonic', 'cf_insulin_monotonic',
        'cf_hypo_rescue', 'cf_hyper_rescue',
        'cf_n', 'cf_hypo_n', 'cf_hyper_n',
        # Time-of-day probe (point accuracy + clock reliability + no-jumping witness).
        'tod_mae_h', 'tod_acc_1h', 'tod_acc_2h', 'tod_acc_bin', 'tod_conf',
        'tod_bias_h', 'tod_std_h', 'tod_p90_h', 'tod_gross_rate', 'tod_mae_hiconf',
        'tod_jump_h', 'tod_xwin_jump_h',
        # Per-horizon excursion buckets.
        *[f'hypo_recall@{_h}' for _h in _eh],
        *[f'hypo_precision@{_h}' for _h in _eh],
        *[f'hypo_n_steps@{_h}' for _h in _eh],
        *[f'hyper_recall@{_h}' for _h in _eh],
        *[f'hyper_precision@{_h}' for _h in _eh],
        *[f'hyper_n_steps@{_h}' for _h in _eh],
        # Nocturnal.
        *[f'night_bg_rmse_{_h}' for _h in BG_HORIZONS_MIN],
        'night_hypo_recall', 'night_hypo_precision', 'night_hypo_n_steps',
        'night_hyper_recall', 'night_hyper_precision', 'night_hyper_n_steps',
        *[f'night_clarke_A@{_h}' for _h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
        *[f'night_mard@{_h}' for _h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
        # Night-onset.
        'night_onset_n_nights',
        'night_onset_hypo_recall', 'night_onset_hypo_precision', 'night_onset_hypo_n_true',
        'night_onset_hyper_recall', 'night_onset_hyper_precision', 'night_onset_hyper_n_true',
    ]
    # A run always starts fresh, so always write a fresh header.
    val_log_exists = False
    val_log_file = open(val_log_path, 'a' if val_log_exists else 'w', newline='')
    val_log_writer = csv.writer(val_log_file)
    if not val_log_exists:
        val_log_writer.writerow(_val_header)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    data_iter = iter(loader)
    step = start_step

    _interrupted = False
    _prev_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig: int, frame: object) -> None:
        nonlocal _interrupted
        if not _interrupted:
            print("\n  [Interrupted] Ctrl+C received — will save after this step completes.")
            _interrupted = True
        else:
            signal.signal(signal.SIGINT, _prev_sigint)
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    while step < total_steps:
        t0 = time.perf_counter()
        # Force train mode every iteration: validation flips the model to eval().
        model.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        patches = batch['patches'].to(device, non_blocking=True)
        attn_mask = batch['attn_mask'].to(device, non_blocking=True)
        bg_formula = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch['bg_formula_data'].items()
        }
        last_bg = bg_formula['last_bg'].float()                       # (B,)
        # risk_total_loss requires the target shaped (B, P, S) to match
        # ``median``; reshape the flat (B, P*S) raw trajectory.
        true_bg_full = (
            bg_formula['true_bg_trajectory'][:, :PREDICTION_PATCHES * PATCH_SIZE]
            .float().reshape(-1, PREDICTION_PATCHES, PATCH_SIZE)
        )

        # Cross-window probe input (window k+1) — TIME-PROBE-only overhead, fully
        # skipped when the penalty is off (data.py ships the key iff enabled+weight>0).
        next_window = None
        if TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
            _nw = batch.get('next_window')
            if _nw is not None:
                next_window = {
                    'patches': _nw['patches'].to(device, non_blocking=True),
                    'last_bg': _nw['last_bg'].to(device, non_blocking=True).float(),
                    'valid': _nw['valid'].to(device, non_blocking=True),
                }

        # Resilience closures (kept verbatim from the prior loop).
        def _halve_optimizer_state() -> None:
            for p_group in muon_opt.param_groups:
                for p in p_group['params']:
                    st = muon_opt.state.get(p, {})
                    if 'momentum_buffer' in st:
                        st['momentum_buffer'].mul_(0.5)
            for p_group in adam_opt.param_groups:
                for p in p_group['params']:
                    st = adam_opt.state.get(p, {})
                    if 'exp_avg' in st:
                        st['exp_avg'].mul_(0.5)

        def _maybe_restore_from_ema(reason: str) -> bool:
            nonlocal consecutive_nan
            if consecutive_nan >= 10 and ema is not None:
                model.load_state_dict(ema.state_dict(), strict=False)
                # The restored weights are NOT the ones the optimizer moments were
                # accumulated against, so clear Muon/AdamW state — otherwise the
                # first post-recovery step re-applies stale moments to the
                # rolled-back weights and can immediately re-diverge.
                muon_opt.state.clear()
                adam_opt.state.clear()
                print(f"  [RECOVERY] {consecutive_nan} consecutive {reason} — restored model from EMA shadow weights (optimizer state cleared)")
                consecutive_nan = 0
                return True
            return False

        def _skip_nonfinite_step(reason: str) -> None:
            """Skip backward+optimizer for a non-finite forward/loss/backward and
            decay optimizer moments by ½ (NEVER poison state). A non-finite
            median/cost PROPAGATES to the loss rather than tripping a deep assert,
            so a NaN here is expected, not fatal — it must route through this
            guard, not crash the run."""
            nonlocal consecutive_nan, loss_ema
            consecutive_nan += 1
            print(f"  [WARNING] {reason} at step {step} (consecutive: {consecutive_nan}) — skipping backward+optimizer step")
            _maybe_restore_from_ema(reason)
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            _halve_optimizer_state()
            loss_history.append(float('nan'))
            if loss_ema is None:
                loss_ema = 1.0
            else:
                loss_ema = 0.98 * loss_ema + 0.02 * 1.0

        # Forward + loss + backward, wrapped so a non-finite loss (caught by the
        # isfinite guard) OR an exception raised inside forward/loss/backward
        # (e.g. a CUDA fault from propagated NaN) both route to the same skip /
        # EMA-restore resilience path instead of aborting the run.
        try:
            # Forward (fp32-native — no autocast).
            q_tau, median, time_pred = model(patches, attn_mask, last_bg, return_time=True)
            q_tau = q_tau.float()
            median = median.float()

            # Loss in fp32. f applied to the target exactly once inside the loss.
            loss_total, parts = risk_total_loss(
                q_tau, median, true_bg_full, weighting,
            )

            # Time-of-day probe loss — added to the BACKWARD tensor ONLY. It never
            # touches loss_total or parts (logging/EMA/CSV/checkpoint selection stay
            # on BG accuracy), but with TIME_PROBE_DETACH=False its gradient DOES
            # shape the shared trunk (a representation-shaping auxiliary task). The
            # probe emits per-patch hour-of-day bin logits; the loss is a soft
            # circular cross-entropy (per-patch CE only — within a 2 h window the P
            # patches share one forward/context and are already clock-consistent), plus
            # a teacher-forced cross-window phase-advance penalty coupling window k to
            # window k+1 (two INDEPENDENT forwards) so the rolling clock advances by
            # exactly one horizon across the seam.
            _tod_extra = loss_total.new_zeros(())
            _tod_loss_val = float('nan')    # per-patch CE (logged as loss_tod)
            _tod_xwin_val = float('nan')    # cross-window penalty alone (logged as loss_tod_xwin)
            _psh = batch['bg_formula_data'].get('pred_start_hour')
            if time_pred is not None and _psh is not None:
                P = time_pred.shape[1]
                adv = PATCH_SIZE * STEP_MINUTES / 60.0                                # 30 min per patch = 0.5 h (PATCH_SIZE-invariant)
                base = _psh.to(time_pred.device).float().reshape(-1)                   # (B,)
                offs = torch.arange(P, device=time_pred.device, dtype=torch.float32) * adv  # (P,)
                tgt_hours = (base[:, None] + offs[None, :]) % 24.0                      # (B,P)
                _tod_ce = time_of_day_bin_ce(
                    time_pred, tgt_hours, TIME_PROBE_N_BINS, TIME_PROBE_LABEL_SMOOTH_BINS
                )
                _tod_loss_val = float(_tod_ce.detach())
                _tod_loss = _tod_ce
                # Cross-window (paired-window) phase-advance penalty. A SECOND forward on
                # window k+1 (true trajectory shifted one horizon, built by data.py as
                # batch['next_window']) couples the two INDEPENDENT clocks so the rolling
                # clock advances by exactly PREDICTION_HORIZON_HOURS across the seam.
                # Diagnostic-only: rides _tod_extra (never loss_total/parts/val/selection),
                # co-trains the trunk via the shared backward, masked by the validity flag,
                # subsampled by the fraction knob. INSIDE this try => non-finite propagates.
                if (next_window is not None
                        and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0
                        and bool(next_window['valid'].any())):
                    B_nw = next_window['patches'].shape[0]
                    n_sub = (B_nw if TIME_PROBE_CROSS_WINDOW_FRACTION >= 1.0
                             else max(1, math.ceil(TIME_PROBE_CROSS_WINDOW_FRACTION * B_nw)))
                    nw_valid_s = next_window['valid'][:n_sub]
                    if bool(nw_valid_s.any()):
                        nw_mask = attn_mask[:n_sub] if attn_mask.dim() == 3 else attn_mask
                        _, _, time_pred_next = model(
                            next_window['patches'][:n_sub], nw_mask,
                            next_window['last_bg'][:n_sub], return_time=True,
                        )
                        _tod_xwin = time_cross_window_consistency_loss(
                            time_pred[:n_sub], time_pred_next, TIME_PROBE_N_BINS,
                            PREDICTION_HORIZON_HOURS, valid=nw_valid_s,
                        )
                        if torch.isfinite(_tod_xwin):
                            _tod_loss = _tod_loss + TIME_PROBE_CROSS_WINDOW_WEIGHT * _tod_xwin
                            _tod_xwin_val = float(_tod_xwin.detach())
                if torch.isfinite(_tod_loss):
                    _tod_extra = TIME_PROBE_LOSS_WEIGHT * _tod_loss
            loss_backward = loss_total + _tod_extra

            if not torch.isfinite(loss_backward):
                _skip_nonfinite_step("NaN/Inf total loss")
                step += 1
                continue

            loss_backward.backward()
        except RuntimeError as exc:
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            _skip_nonfinite_step(f"forward/loss/backward RuntimeError ({exc})")
            step += 1
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(weighting.parameters()),
            gradient_clip_norm, error_if_nonfinite=False,
        )

        gn_val = float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm)
        is_finite = torch.isfinite(grad_norm) if torch.is_tensor(grad_norm) else (gn_val == gn_val and gn_val != float('inf'))

        if is_finite:
            _update_lr(muon_opt, adam_opt, step, muon_lr, adam_lr, warmup_steps, total_steps, lr_min_ratio,
                       wd_correction=weight_decay_schedule_correction)
            muon_opt.step()
            adam_opt.step()
            if ema is not None:
                ema.update(model)
            consecutive_nan = 0
        else:
            consecutive_nan += 1
            print(f"  [WARNING] NaN/Inf gradient at step {step} (consecutive: {consecutive_nan}), skipping optimizer step")
            _maybe_restore_from_ema("NaN gradients")
            _halve_optimizer_state()
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)

        step_time = time.perf_counter() - t0
        loss_val = float(loss_total.item())
        loss_history.append(loss_val)

        if loss_ema is None:
            loss_ema = loss_val
        else:
            loss_ema = LOSS_EMA_ALPHA * loss_ema + (1.0 - LOSS_EMA_ALPHA) * loss_val

        # ---- Logging ----
        if step % log_interval == 0:
            cur_lr_muon = muon_opt.param_groups[0]['lr']
            cur_lr_adam = adam_opt.param_groups[0]['lr']
            gpu_mb = (torch.cuda.memory_allocated(device) / 1e6
                      if device.type == 'cuda' else 0.0)
            grad_norm_val = float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm)

            loss_q = float(parts.get('loss_Q', float('nan')))
            loss_d = float(parts.get('loss_D', float('nan')))
            loss_d_shape = float(parts.get('loss_D_shape', float('nan')))
            loss_d_tdi = float(parts.get('loss_D_tdi', float('nan')))
            log_sigma_q = float(parts.get('log_sigma_Q', float('nan')))
            log_sigma_d = float(parts.get('log_sigma_D', float('nan')))
            loss_tod = _tod_loss_val
            loss_tod_xwin = _tod_xwin_val

            print(
                f"Step {step:>6}/{total_steps} | "
                f"Loss: {loss_val:.4f} (ema={loss_ema:.4f}) | "
                f"L_Q: {loss_q:.4f}  L_D: {loss_d:.4f} "
                f"(sh={loss_d_shape:.4f} tdi={loss_d_tdi:.4f}) | "
                f"logσ: Q={log_sigma_q:+.4f} D={log_sigma_d:+.4f} | "
                f"L_tod: {loss_tod:.4f} (xwin {loss_tod_xwin:.4f}) | "
                f"Grad: {grad_norm_val:.3f} | "
                f"LR_muon: {cur_lr_muon:.6f} | LR_adam: {cur_lr_adam:.6f} | "
                f"Time: {step_time:.2f}s"
            )

            train_log_writer.writerow([
                step,
                round(loss_val, 6), round(loss_ema, 6),
                round(loss_q, 6), round(loss_d, 6),
                round(loss_d_shape, 6), round(loss_d_tdi, 6),
                round(loss_tod, 6), round(loss_tod_xwin, 6),
                round(log_sigma_q, 6), round(log_sigma_d, 6),
                round(grad_norm_val, 6),
                round(cur_lr_muon, 8), round(cur_lr_adam, 8),
                round(step_time, 4), round(gpu_mb, 1),
            ])
            train_log_file.flush()

        # ---- Validation ----
        is_final_step = step == total_steps - 1
        if validation_interval < 999999 and step > 0 and (
            step % validation_interval == 0 or is_final_step
        ):
            eval_ctx = ema.apply_to(model) if ema is not None else contextlib.nullcontext()
            with eval_ctx:
                val_metrics = _run_validation(
                    model, val_dataset, norm_stats, device, weighting,
                    bg_hypo_threshold=bg_hypo_threshold,
                    bg_hyper_threshold=bg_hyper_threshold,
                )
                val_metrics.update(_run_night_onset_validation(
                    model, val_dataset_night_onset, norm_stats, device,
                    hypo_threshold=bg_hypo_threshold, hyper_threshold=bg_hyper_threshold))

            val_total = val_metrics['val_loss_total']
            train_ema = loss_ema if loss_ema is not None else loss_val
            overfit_ratio = 1.0 / (1.0 + math.exp(-(val_total - train_ema)))
            val_metrics['train_loss_ema'] = train_ema
            val_metrics['overfit_ratio'] = overfit_ratio

            print()
            print(_render_validation_table(step, val_metrics, prev_val_metrics))
            print()
            prev_val_metrics = dict(val_metrics)

            def _cv(key: str, n: int = 4):
                v = val_metrics.get(key)
                if isinstance(v, bool):       # bool BEFORE int (bool ⊂ int)
                    return v
                if isinstance(v, (int, float)):
                    return round(v, n)
                return ''

            val_log_writer.writerow([
                step,
                round(val_metrics['val_loss_total'], 6),
                round(val_metrics['val_loss_Q'], 6),
                round(val_metrics['val_loss_D'], 6),
                round(val_metrics['val_pinball'], 6),
                round(train_ema, 6),
                round(overfit_ratio, 6),
                *[_cv(f'coverage90@{h}') for h in COVERAGE_HORIZONS_MIN],
                *[_cv(f'sign_balance@{h}') for h in COVERAGE_HORIZONS_MIN],
                *[_cv(f'inner50_cov@{h}') for h in COVERAGE_HORIZONS_MIN],
                *[_cv(f'bg_rmse_{h}') for h in BG_HORIZONS_MIN],
                *[_cv(f'evalfix_mard@{h}') for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
                _cv('pred_tir'), _cv('true_tir'), _cv('tir_err'),
                _cv('hypo_recall'), _cv('hypo_precision'), _cv('hypo_n_steps'),
                _cv('hyper_recall'), _cv('hyper_precision'), _cv('hyper_n_steps'),
                *[_cv(f'cgega_{m}_{r}') for r in ('hypo', 'eu', 'hyper') for m in ('ap', 'ep')],
                *[_cv(f'evalfix_clarke_A@{h}') for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
                _cv('clarke_AB_pct'), _cv('clarke_D_pct'), _cv('clarke_E_pct'),
                _cv('roc_rmse', 6), _cv('roc_corr'), _cv('trend_gain_beta'), _cv('trend_amp_ratio'),
                _cv('bg_curve_corr'),
                _cv('exc_amp_ratio'), _cv('exc_gain_beta'), _cv('exc_corr'),
                _cv('exc_overshoot_frac'), _cv('exc_undershoot_frac'), _cv('exc_n'),
                _cv('conf_cov90_raw'), _cv('conf_cov90_cal'),
                _cv('conf_hypo_esc_raw'), _cv('conf_hypo_esc_cal'), _cv('conf_n'),
                _cv('median_roughness'), _cv('median_roughness_far'),
                _cv('cf_carb_dbg'), _cv('cf_carb_dir'),
                _cv('cf_insulin_dbg'), _cv('cf_insulin_dir'),
                _cv('cf_carb_monotonic'), _cv('cf_insulin_monotonic'),
                _cv('cf_hypo_rescue'), _cv('cf_hyper_rescue'),
                _cv('cf_n'), _cv('cf_hypo_n'), _cv('cf_hyper_n'),
                _cv('tod_mae_h'), _cv('tod_acc_1h'), _cv('tod_acc_2h'),
                _cv('tod_acc_bin'), _cv('tod_conf'),
                _cv('tod_bias_h'), _cv('tod_std_h'), _cv('tod_p90_h'),
                _cv('tod_gross_rate'), _cv('tod_mae_hiconf'),
                _cv('tod_jump_h'), _cv('tod_xwin_jump_h'),
                *[_cv(f'hypo_recall@{_h}') for _h in _eh],
                *[_cv(f'hypo_precision@{_h}') for _h in _eh],
                *[_cv(f'hypo_n_steps@{_h}') for _h in _eh],
                *[_cv(f'hyper_recall@{_h}') for _h in _eh],
                *[_cv(f'hyper_precision@{_h}') for _h in _eh],
                *[_cv(f'hyper_n_steps@{_h}') for _h in _eh],
                *[_cv(f'night_bg_rmse_{_h}') for _h in BG_HORIZONS_MIN],
                _cv('night_hypo_recall'), _cv('night_hypo_precision'), _cv('night_hypo_n_steps'),
                _cv('night_hyper_recall'), _cv('night_hyper_precision'), _cv('night_hyper_n_steps'),
                *[_cv(f'night_clarke_A@{_h}') for _h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
                *[_cv(f'night_mard@{_h}') for _h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
                _cv('night_onset_n_nights'),
                _cv('night_onset_hypo_recall'), _cv('night_onset_hypo_precision'), _cv('night_onset_hypo_n_true'),
                _cv('night_onset_hyper_recall'), _cv('night_onset_hyper_precision'), _cv('night_onset_hyper_n_true'),
            ])
            val_log_file.flush()

            def _f4(x: Any) -> str:
                return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"

            print(
                f"  [Val | median roughness |Δ²| risk] "
                f"all={_f4(val_metrics.get('median_roughness'))}  "
                f"far(last patch)={_f4(val_metrics.get('median_roughness_far'))}"
            )

            def _r(x: float | None, n: int = 4) -> float | None:
                return round(x, n) if isinstance(x, (int, float)) else None

            val_record = {
                'step': step,
                'val_loss_total': round(val_total, 6),
                'val_loss_Q': round(val_metrics['val_loss_Q'], 6),
                'val_loss_D': round(val_metrics['val_loss_D'], 6),
                'val_pinball': round(val_metrics['val_pinball'], 6),
                'train_loss_ema': round(train_ema, 6),
                'overfit_ratio': round(overfit_ratio, 4),
                **{f'coverage90@{h}': _r(val_metrics.get(f'coverage90@{h}'))
                   for h in COVERAGE_HORIZONS_MIN},
                **{f'sign_balance@{h}': _r(val_metrics.get(f'sign_balance@{h}'))
                   for h in COVERAGE_HORIZONS_MIN},
                **{f'inner50_cov@{h}': _r(val_metrics.get(f'inner50_cov@{h}'))
                   for h in COVERAGE_HORIZONS_MIN},
                **{f'evalfix_mard@{h}': _r(val_metrics.get(f'evalfix_mard@{h}'))
                   for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN},
                'pred_tir': _r(val_metrics['pred_tir']),
                'true_tir': _r(val_metrics['true_tir']),
                'tir_err': _r(val_metrics['tir_err']),
                'hypo_recall': _r(val_metrics.get('hypo_recall')),
                'hypo_precision': _r(val_metrics.get('hypo_precision')),
                'hypo_n_steps': val_metrics['hypo_n_steps'],
                'hyper_recall': _r(val_metrics.get('hyper_recall')),
                'hyper_precision': _r(val_metrics.get('hyper_precision')),
                'hyper_n_steps': val_metrics['hyper_n_steps'],
                **{f'cgega_{m}_{r}': _r(val_metrics.get(f'cgega_{m}_{r}'))
                   for r in ('hypo', 'eu', 'hyper') for m in ('ap', 'be', 'ep')},
                'clarke_AB_pct': _r(val_metrics.get('clarke_AB_pct')),
                'clarke_D_pct': _r(val_metrics.get('clarke_D_pct')),
                'clarke_E_pct': _r(val_metrics.get('clarke_E_pct')),
                'roc_rmse': _r(val_metrics.get('roc_rmse'), 6),
                'roc_corr': _r(val_metrics.get('roc_corr')),
                'trend_gain_beta': _r(val_metrics.get('trend_gain_beta')),
                'trend_amp_ratio': _r(val_metrics.get('trend_amp_ratio')),
                'bg_curve_corr': _r(val_metrics.get('bg_curve_corr')),
                'exc_amp_ratio': _r(val_metrics.get('exc_amp_ratio')),
                'exc_gain_beta': _r(val_metrics.get('exc_gain_beta')),
                'exc_corr': _r(val_metrics.get('exc_corr')),
                'exc_overshoot_frac': _r(val_metrics.get('exc_overshoot_frac')),
                'exc_undershoot_frac': _r(val_metrics.get('exc_undershoot_frac')),
                'exc_n': val_metrics.get('exc_n'),
                'conf_cov90_raw': _r(val_metrics.get('conf_cov90_raw')),
                'conf_cov90_cal': _r(val_metrics.get('conf_cov90_cal')),
                'conf_hypo_esc_raw': _r(val_metrics.get('conf_hypo_esc_raw')),
                'conf_hypo_esc_cal': _r(val_metrics.get('conf_hypo_esc_cal')),
                'conf_n': val_metrics.get('conf_n'),
                'median_roughness': _r(val_metrics.get('median_roughness'), 6),
                'median_roughness_far': _r(val_metrics.get('median_roughness_far'), 6),
                'cf_carb_dbg': _r(val_metrics.get('cf_carb_dbg')),
                'cf_carb_dir': _r(val_metrics.get('cf_carb_dir')),
                'cf_insulin_dbg': _r(val_metrics.get('cf_insulin_dbg')),
                'cf_insulin_dir': _r(val_metrics.get('cf_insulin_dir')),
                'cf_carb_monotonic': _r(val_metrics.get('cf_carb_monotonic')),
                'cf_insulin_monotonic': _r(val_metrics.get('cf_insulin_monotonic')),
                'cf_hypo_rescue': _r(val_metrics.get('cf_hypo_rescue')),
                'cf_hyper_rescue': _r(val_metrics.get('cf_hyper_rescue')),
                'cf_n': val_metrics.get('cf_n'),
                'cf_hypo_n': val_metrics.get('cf_hypo_n'),
                'cf_hyper_n': val_metrics.get('cf_hyper_n'),
                'tod_mae_h': _r(val_metrics.get('tod_mae_h')),
                'tod_acc_1h': _r(val_metrics.get('tod_acc_1h')),
                'tod_acc_2h': _r(val_metrics.get('tod_acc_2h')),
                'tod_acc_bin': _r(val_metrics.get('tod_acc_bin')),
                'tod_conf': _r(val_metrics.get('tod_conf')),
                'tod_bias_h': _r(val_metrics.get('tod_bias_h')),
                'tod_std_h': _r(val_metrics.get('tod_std_h')),
                'tod_p90_h': _r(val_metrics.get('tod_p90_h')),
                'tod_gross_rate': _r(val_metrics.get('tod_gross_rate')),
                'tod_mae_hiconf': _r(val_metrics.get('tod_mae_hiconf')),
                'tod_jump_h': _r(val_metrics.get('tod_jump_h')),
                'tod_xwin_jump_h': _r(val_metrics.get('tod_xwin_jump_h')),
            }
            for h in BG_HORIZONS_MIN:
                val_record[f'bg_rmse_{h}'] = _r(val_metrics.get(f'bg_rmse_{h}'))
            for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
                val_record[f'evalfix_clarke_A@{h}'] = _r(val_metrics.get(f'evalfix_clarke_A@{h}'))
            val_history.append(val_record)

            # best.pt = min val_loss_total (= risk_total_loss).
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_val_step = step
                torch.save(
                    _build_checkpoint(model, weighting, muon_opt, adam_opt, step,
                                      loss_history, training_config, norm_stats,
                                      master_seed, val_history, best_val_loss, best_val_step,
                                      loss_ema, ema=ema),
                    'checkpoints/t1dmai_best.pt'
                )
                print(f"  [Checkpoint] saved best model (val_loss={val_total:.4f})")

        # ---- Checkpointing ----
        if checkpoint_interval < 999999 and step % checkpoint_interval == 0 and step > 0:
            path = f'checkpoints/t1dmai_step_{step}.pt'
            torch.save(
                _build_checkpoint(model, weighting, muon_opt, adam_opt, step,
                                  loss_history, training_config, norm_stats,
                                  master_seed, val_history, best_val_loss, best_val_step,
                                  loss_ema, ema=ema),
                path
            )
            print(f"  [Checkpoint] saved {path}")

            _write_training_summary(
                log_dir='logs', step=step, total_steps=total_steps,
                loss_history=loss_history, best_val_loss=best_val_loss,
                best_val_step=best_val_step, training_config=training_config,
                train_start_time=train_start_time, val_history=val_history, device=device,
            )

        step += 1

        if _interrupted:
            break

    signal.signal(signal.SIGINT, _prev_sigint)

    train_log_file.close()
    val_log_file.close()

    if _interrupted:
        interrupted_step = step - 1
        path = f'checkpoints/t1dmai_interrupted_step_{interrupted_step}.pt'
        torch.save(
            _build_checkpoint(model, weighting, muon_opt, adam_opt, interrupted_step,
                              loss_history, training_config, norm_stats,
                              master_seed, val_history, best_val_loss, best_val_step,
                              loss_ema, ema=ema),
            path,
        )
        print(f"  [Interrupted] Checkpoint saved → {path}")
    else:
        # The final step always writes a checkpoint (contract: the final-step
        # validation always writes a ckpt + the periodic guarantee here).
        final_step = step - 1
        if checkpoint_interval < 999999 and final_step > 0:
            already_saved = final_step % checkpoint_interval == 0
            if not already_saved:
                path = f'checkpoints/t1dmai_step_{final_step}.pt'
                torch.save(
                    _build_checkpoint(model, weighting, muon_opt, adam_opt, final_step,
                                      loss_history, training_config, norm_stats,
                                      master_seed, val_history, best_val_loss, best_val_step,
                                      loss_ema, ema=ema),
                    path,
                )
                print(f"  [Checkpoint] saved final model → {path}")

    _write_training_summary(
        log_dir='logs', step=step - 1, total_steps=total_steps,
        loss_history=loss_history, best_val_loss=best_val_loss,
        best_val_step=best_val_step, training_config=training_config,
        train_start_time=train_start_time, val_history=val_history, device=device,
    )

    print(f"Training complete. Steps: {total_steps}, final loss: {loss_history[-1]:.4f}")
    return loss_history


class HelpfulParser(argparse.ArgumentParser):
    """argparse.ArgumentParser that prints the full --help on any error."""

    def error(self, message: str):  # type: ignore[override]
        self.print_help(sys.stderr)
        sys.stderr.write(f'\nerror: {message}\n')
        sys.exit(2)


if __name__ == '__main__':
    parser = HelpfulParser(
        description='Train T1DMAI. Parameters are resolved in this order: '
                    'CLI args > config.py.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--master-seed', type=int, default=None)
    parser.add_argument('--total-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=None)
    parser.add_argument('--validation-interval', type=int, default=None)
    parser.add_argument('--muon-lr', type=float, default=None)
    parser.add_argument('--muon-momentum', type=float, default=None,
                        help='Muon momentum coefficient.')
    parser.add_argument('--adam-lr', type=float, default=None)
    parser.add_argument('--bg-hypo-threshold', type=float, default=None,
                        help='BG (mg/dL) below which a step counts as hypo (drives hypo_recall).')
    parser.add_argument('--bg-hyper-threshold', type=float, default=None,
                        help='BG (mg/dL) above which a step counts as hyper (drives hyper_recall).')
    parser.add_argument('--warmup-steps', type=int, default=None)
    parser.add_argument('--lr-min-ratio', type=float, default=None)
    parser.add_argument('--gradient-clip-norm', type=float, default=None)
    parser.add_argument('--patient-uniform-sample-prob', type=float, default=None,
                        help='Probability that a sample draws its patient with uniformly-sampled '
                             'skills (oversamples tail patients). 0 disables.')
    parser.add_argument('--adam-weight-decay', type=float, default=None,
                        help='AdamW weight decay applied to embeddings and 1D parameters.')
    parser.add_argument('--no-wd-correction', dest='wd_correction', action='store_false', default=None,
                        help='Disable the AdamC schedule-aware weight-decay correction on the normalized (Muon) matrices; restores plain decoupled decay.')
    parser.add_argument('--simulator-warmup-hours', type=float, default=None,
                        help='Hours discarded from the start of every simulator run.')
    parser.add_argument('--ema-decay', type=float, default=None,
                        help='Decay factor for the weight-EMA shadow used at validation. 0 disables.')
    parser.add_argument('--cache-path', type=str, default=None,
                        help='Path to a simulator cache directory produced by T1DMSIM/cache_simulator.py.')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Layer 1: code defaults (from config.py)
    # ------------------------------------------------------------------ #
    resolved = {
        'master_seed': MASTER_SEED,
        'total_steps': TOTAL_STEPS,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'log_interval': LOG_INTERVAL,
        'checkpoint_interval': CHECKPOINT_INTERVAL,
        'validation_interval': VALIDATION_INTERVAL,
        'muon_lr': MUON_LR,
        'muon_momentum': MUON_MOMENTUM,
        'adam_lr': ADAM_LR,
        'warmup_steps': WARMUP_STEPS,
        'lr_min_ratio': LR_MIN_RATIO,
        'gradient_clip_norm': GRADIENT_CLIP_NORM,
        'patient_uniform_sample_prob': PATIENT_UNIFORM_SAMPLE_PROB,
        'adam_weight_decay': ADAM_WEIGHT_DECAY,
        'weight_decay_schedule_correction': WEIGHT_DECAY_SCHEDULE_CORRECTION,
        'simulator_warmup_hours': SIMULATOR_WARMUP_HOURS,
        'ema_decay': EMA_DECAY,
        'prediction_horizon_hours': PREDICTION_HORIZON_HOURS,
        'night_long_horizon_hours': NIGHT_LONG_HORIZON_HOURS,
        'bg_hypo_threshold': BG_HYPO_THRESHOLD,
        'bg_hyper_threshold': BG_HYPER_THRESHOLD,
        'cache_path': None,
    }
    sources = {k: 'config.py' for k in resolved}

    # ------------------------------------------------------------------ #
    # Layer 2: explicit CLI arguments (only override if not None)
    # ------------------------------------------------------------------ #
    cli_map = {
        'master_seed': args.master_seed,
        'total_steps': args.total_steps,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'log_interval': args.log_interval,
        'checkpoint_interval': args.checkpoint_interval,
        'validation_interval': args.validation_interval,
        'muon_lr': args.muon_lr,
        'muon_momentum': args.muon_momentum,
        'adam_lr': args.adam_lr,
        'warmup_steps': args.warmup_steps,
        'lr_min_ratio': args.lr_min_ratio,
        'gradient_clip_norm': args.gradient_clip_norm,
        'patient_uniform_sample_prob': args.patient_uniform_sample_prob,
        'adam_weight_decay': args.adam_weight_decay,
        'weight_decay_schedule_correction': args.wd_correction,
        'simulator_warmup_hours': args.simulator_warmup_hours,
        'ema_decay': args.ema_decay,
        'bg_hypo_threshold': args.bg_hypo_threshold,
        'bg_hyper_threshold': args.bg_hyper_threshold,
        'cache_path': args.cache_path,
    }
    for key, cli_val in cli_map.items():
        if cli_val is not None:
            resolved[key] = cli_val
            sources[key] = 'CLI'

    # Horizon is fixed at config import (PREDICTION_HORIZON_HOURS /
    # NIGHT_LONG_HORIZON_HOURS are plain constants in config.py), so model / data
    # / inference already imported the correct PREDICTION_PATCHES directly — no
    # runtime propagation needed.

    # ------------------------------------------------------------------ #
    # Dump resolved config
    # ------------------------------------------------------------------ #
    rows = [(key, str(value), sources[key]) for key, value in resolved.items()]
    rows.append(('prediction_patches', str(PREDICTION_PATCHES), 'derived'))
    rows.append(('arch_version', str(ARCH_VERSION), 'config.py'))
    rows.append(('loss_schema', str(LOSS_SCHEMA), 'config.py'))
    key_w = max(len(k) for k, _, _ in rows)
    val_w = max(len(v) for _, v, _ in rows)
    body = [f"  {k:<{key_w}}  {v:<{val_w}}  [{s}]" for k, v, s in rows]
    header = "  T1DMAI — Resolved training configuration"
    cfg_line = "  Config: config.py"
    width = max(len(line) for line in (*body, header, cfg_line))
    print("=" * width)
    print(header)
    print(cfg_line)
    print("=" * width)
    for line in body:
        print(line)
    print("=" * width)
    print()

    train(
        total_steps=resolved['total_steps'],
        batch_size=resolved['batch_size'],
        master_seed=resolved['master_seed'],
        num_workers=resolved['num_workers'],
        log_interval=resolved['log_interval'],
        checkpoint_interval=resolved['checkpoint_interval'],
        validation_interval=resolved['validation_interval'],
        muon_lr=resolved['muon_lr'],
        muon_momentum=resolved['muon_momentum'],
        adam_lr=resolved['adam_lr'],
        warmup_steps=resolved['warmup_steps'],
        lr_min_ratio=resolved['lr_min_ratio'],
        gradient_clip_norm=resolved['gradient_clip_norm'],
        patient_uniform_sample_prob=resolved['patient_uniform_sample_prob'],
        adam_weight_decay=resolved['adam_weight_decay'],
        weight_decay_schedule_correction=resolved['weight_decay_schedule_correction'],
        simulator_warmup_hours=resolved['simulator_warmup_hours'],
        ema_decay=resolved['ema_decay'],
        bg_hypo_threshold=resolved['bg_hypo_threshold'],
        bg_hyper_threshold=resolved['bg_hyper_threshold'],
        cache_path=resolved['cache_path'],
    )
