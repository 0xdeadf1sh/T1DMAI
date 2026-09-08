"""Training driver for the masked-BG objective: Muon + AdamW, loss in Kovatchev risk space."""

import argparse
import contextlib
import csv
import json
import math
import os
import random
import signal
import sys
import textwrap
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from config import (                                           # noqa: E402
    MASTER_SEED, DETERMINISTIC, TOTAL_STEPS, BATCH_SIZE, NUM_WORKERS,
    MUON_LR, MUON_MOMENTUM, MUON_NS_ITERATIONS, MUON_WEIGHT_DECAY,
    ADAM_LR, ADAM_BETAS, ADAM_WEIGHT_DECAY, ADAM_EPS,
    WARMUP_STEPS, LR_MIN_RATIO, WEIGHT_DECAY_SCHEDULE_CORRECTION, GRADIENT_CLIP_NORM,
    PREDICTION_PATCHES,
    MAX_CONTEXT_PATCHES, MIN_CONTEXT_PATCHES,
    LOG_INTERVAL, CHECKPOINT_INTERVAL, VALIDATION_INTERVAL,
    VALIDATION_N_PATIENTS, VALIDATION_PROBE_N_PATIENTS, NORM_STATS_FILE, PATCH_SIZE,
    N_INPUT_FEATURES, CHANNEL_TO_FEAT, NON_MASKABLE_FEATS,
    MASK_SPAN_LENGTHS, MAX_MASKED_PATCHES, MASK_RIGHT_EDGE_QUOTA, MSE_ALPHA,
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

from config import ARCH_VERSION, LOSS_SCHEMA

from utils import (
    ModelEMA, kovatchev_f_inv, create_attention_mask_from_visible,
    time_of_day_bin_ce, time_of_day_decode_bins, time_of_day_resultant,
    circular_hour_error, circular_hour_residual, circular_bias_hours, circular_std_hours,
)

from T1DMSIM.simulator import (
    BG_CLAMP_MIN, BG_CLAMP_MAX, bolus_pk_for_dose, gamma_curve,
)

# Band-edge indices on the ascending QUANTILE_LEVELS axis, never a bare literal.
_TAU_LO_IDX = QUANTILE_LEVELS.index(0.05)
_TAU_HI_IDX = QUANTILE_LEVELS.index(0.95)
_TAU_INNER_LO_IDX = QUANTILE_LEVELS.index(0.25)
_TAU_INNER_HI_IDX = QUANTILE_LEVELS.index(0.75)
# Clinical alarms key off these BAND EDGES, not the median: hypo tau=HYPO, hyper tau=HYPER.
_HYPO_BAND_IDX = QUANTILE_LEVELS.index(HYPO_ALARM_QUANTILE_TAU)
_HYPER_BAND_IDX = QUANTILE_LEVELS.index(HYPER_ALARM_QUANTILE_TAU)

# (base@30min, slope_per_30min, floor) per excursion row; DISPLAY-ONLY, never loss/CSV/selection.

# Bar declines with horizon: near-term is fixed by insulin-on-board, far horizon is info-limited.
EXCURSION_TARGET_HYPO_RECALL = (90.0, 10.0, 50.0)
EXCURSION_TARGET_HYPO_PRECISION = (75.0, 8.0, 45.0)
EXCURSION_TARGET_HYPER_RECALL = (85.0, 8.0, 55.0)
EXCURSION_TARGET_HYPER_PRECISION = (85.0, 8.0, 55.0)


class _OffsetSampler(Sampler):
    """Sequential sampler skipping the first ``offset`` indices; training always passes 0."""

    def __init__(self, total_len: int, offset: int = 0) -> None:
        self.total_len = total_len
        self.offset = offset

    def __iter__(self):
        return iter(range(self.offset, self.total_len))

    def __len__(self) -> int:
        return self.total_len - self.offset


def setup_determinism(seed: int) -> None:
    """Pin every RNG and disable nondeterministic kernels. Gated on config.DETERMINISTIC.

    SDPA's flash/memory-efficient BACKWARD has no deterministic CUDA kernel, so GPU training
    is reproducible only to numerical noise; the data stream itself is fully reproducible.
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
    """Seed a DataLoader worker's global numpy + random RNGs. ``torch.initial_seed()`` is
    already per-worker, so deriving from it keeps the workers distinct."""
    s = torch.initial_seed() % 2 ** 31
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


from model import T1DMAI
from muon import Muon
from normalization import (
    load_normalization_stats, compute_normalization_stats,
    save_normalization_stats, CHANNEL_NAMES, normalize, denormalize,
)
from data import (
    T1DMDataset, collate_fn, BG_MASKED_FEAT, masked_channel_policy,
    checkpoint_masked_channel_policy,
)
from risk_loss import risk_total_loss, KendallGalWeighting
import cg_ega
import dts_grid

LOSS_EMA_ALPHA = 0.98

# One patch = 30min, step = 5min; fallback when batched bg_formula_data has no dt_minutes.
STEP_MINUTES = 30.0 / PATCH_SIZE  # == 5.0


def _dt_minutes(bg_formula_data: dict[str, Any]) -> float:
    v = bg_formula_data.get('dt_minutes')
    return float(v) if v is not None else float(STEP_MINUTES)


_PATCH_HOURS = PATCH_SIZE * STEP_MINUTES / 60.0


BG_HORIZONS_MIN: tuple[int, ...] = (30, 60, 120, 180, 360, 480)
# Where Clarke-A and MARD are reported un-pooled, to read against single-horizon SOTA bars.
EVALFIX_CLARKE_MARD_HORIZONS_MIN: tuple[int, ...] = (30, 60, 120)
# MARGINAL per-(h,tau) coverage of the central 90% band, per-step not joint; lines up with d=1..4.
COVERAGE_HORIZONS_MIN: tuple[int, ...] = (30, 60, 90, 120)

BG_TARGET_LO = 70.0
BG_TARGET_HI = 180.0

# bg holds the same index in the feature stack and CHANNEL_NAMES, serving both stride and lookup.
_BG_FEAT = 0
_BG_CHANNEL = CHANNEL_NAMES[_BG_FEAT]
assert tuple(NON_MASKABLE_FEATS) == (_BG_FEAT,), (
    f"NON_MASKABLE_FEATS is {tuple(NON_MASKABLE_FEATS)}; the infill protocol "
    f"restores exactly the withheld feature, which must be bg (feat {_BG_FEAT})"
)


def _assert_mask_is_this_window(patches: torch.Tensor, attn_mask: torch.Tensor,
                                where: str) -> None:
    """The mask a forward runs under must be the one THAT input's feat 4 announces.

    attn_mask (B,T,T) bool blocks a masked column from every visible row. Paired windows
    share n_ctx and mask SHAPE, so a swapped mask raises nothing, only reads withheld evidence.
    """
    masked = patches[..., BG_MASKED_FEAT::N_INPUT_FEATURES][..., 0] > 0.5   # (B, T)
    assert not bool((attn_mask & masked.unsqueeze(1) & ~masked.unsqueeze(2)).any()), (
        f"{where}: a visible row attends a masked column — the attention mask is "
        f"not the one this window's patches announce"
    )


# Fixed, so a moving infill_* column is the model, not the draw; placement is metrics.protocols'.
INFILL_PROTOCOL_SEED = 0

# Excursion-amplitude diagnostic: over/under-dispersion the per-patch trend_amp_ratio misses.

# Counted only when |dev| > EXC_AMP_MIN_MGDL; exc_amp_ratio = std(pred)/std(true), target 1.
EXC_AMP_MIN_MGDL = 15.0
EXC_AMP_OVERSHOOT_RATIO = 1.25
EXC_AMP_UNDERSHOOT_RATIO = 0.75


def _excursion_bucket_horizons(active_patches: int) -> list[int]:
    """Disjoint per-patch bucket END-horizons in minutes: patch ``p`` covers
    ``[p·30, (p+1)·30)`` at the 5-min step and is labelled ``(p+1)·30``. Shared by the metric
    emitter and the finalizer so the ``@{h}`` count keys line up."""
    return [(p + 1) * PATCH_SIZE * 5 for p in range(active_patches)]


def _excursion_target(spec: tuple[float, float, float], horizon_min: int) -> float:
    """Per-horizon excursion-detection DISPLAY target, in percent:
    ``max(floor, base - slope * (horizon_min/30 - 1))`` from a
    ``(base@30min, slope_per_30min, floor)`` spec. Colours rows only — never the loss or
    checkpoint selection."""
    base, slope, floor = spec
    return max(floor, base - slope * (horizon_min / 30.0 - 1.0))


def _median_to_mgdl(median_risk: torch.Tensor) -> torch.Tensor:
    """RISK-space per-step median ``(B, P, S)`` -> mg/dL ``(B, P*S)``, the SOLE risk->mg/dL
    crossing for the headline forecast and the SOLE ``pred_bg`` every BG metric reads.
    ``kovatchev_f_inv`` clamps the risk input, exps, then clamps the output to
    ``[BG_CLAMP_MIN, BG_CLAMP_MAX]``."""
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
    """Diagnostic BG metrics in mg/dL, sums+counts for batch aggregation, off pred_bg=f_inv(median).

    q_mgdl (B,P*S) mg/dL: lo/hi the central 90% band, inner_lo/hi (.25/.75), hypo_lo/hyper_hi
    the clinical band edges; {} skips coverage. bg_formula_data carries true_bg_trajectory/last_bg.
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

    # Stage every reduction as 0-dim, flush with ONE torch.stack.tolist(), not ~90 .item() calls.

    # Bit-identical; count sums cast to fp32 (counts << 2**24, exact) to stack with the rest.
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

    # Detectors key off BAND EDGES: hypo tau=HYPO, hyper tau=HYPER; truth is true bg.
    pred_lo = q_mgdl['hypo_lo']    # (B, P*S) mg/dL, τ=HYPO_ALARM_QUANTILE_TAU lower band edge
    pred_hi = q_mgdl['hyper_hi']   # (B, P*S) mg/dL, τ=HYPER_ALARM_QUANTILE_TAU upper band edge
    assert bool(((pred_lo >= BG_CLAMP_MIN - 1e-3) & (pred_lo <= BG_CLAMP_MAX + 1e-3)).all()), (
        f"pred_lo out of physical clamp band [{BG_CLAMP_MIN}, {BG_CLAMP_MAX}]"
    )
    assert bool(((pred_hi >= BG_CLAMP_MIN - 1e-3) & (pred_hi <= BG_CLAMP_MAX + 1e-3)).all()), (
        f"pred_hi out of physical clamp band [{BG_CLAMP_MIN}, {BG_CLAMP_MAX}]"
    )

    # RECALL is strict TP; PRECISION forgives within EXCURSION_PRECISION_TOLERANCE_MGDL.
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

    # CG-EGA (Kovatchev 2004); true_bg FIRST, the reference on every axis.
    _true_np = true_bg.detach().cpu().numpy()
    _pred_np = pred_bg.detach().cpu().numpy()
    cg = cg_ega.cg_ega_counts(
        _true_np,
        _pred_np,
        last_bg.detach().cpu().numpy(),
        freq_min=dt,
    )
    for _ck, _cv in cg.items():
        out[f'cgega_{_ck}'] = float(_cv)

    # DTS Error Grid (Klonoff 2024), a SECOND grid beside Clarke: penalises overestimate harder.

    # true_bg FIRST — nothing catches a transpose, both args are legal mg/dL; per zone, never A+B.
    _dts_zones = dts_grid.dts_zones(_true_np, _pred_np)          # (B, P*S) 0..4
    for _zi, _zn in enumerate(dts_grid.ZONE_NAMES):
        out[f'dts_{_zn}'] = float((_dts_zones == _zi).sum())
    out['dts_total'] = float(_dts_zones.size)

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
    _clarke_masks = {'A': in_A, 'B': zone_B, 'C': zone_C, 'D': zone_D, 'E': zone_E}
    for _z, _m in _clarke_masks.items():
        _stage(f'clarke_{_z}', _m.sum())
    out['clarke_total'] = float(in_A.numel())

    # Per-horizon zone shares (both grids, MARD): pooled mixes a 5-min error with a 2-hour one.

    # @{h} is the SINGLE step at that horizon, every zone — else A-for-B at 30min hides in the pool.
    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        h_idx = (h_min // int(dt)) - 1
        _live = 0 <= h_idx < total_steps_h
        for _z, _m in _clarke_masks.items():
            if _live:
                _stage(f'evalfix_clarke_{_z}@{h_min}', _m[:, h_idx].sum())
                out[f'evalfix_clarke_{_z}@{h_min}_cnt'] = float(B)
            else:
                out[f'evalfix_clarke_{_z}@{h_min}'] = 0.0
                out[f'evalfix_clarke_{_z}@{h_min}_cnt'] = 0.0
        for _zi, _zn in enumerate(dts_grid.ZONE_NAMES):
            if _live:
                out[f'dts_{_zn}@{h_min}'] = float((_dts_zones[:, h_idx] == _zi).sum())
                out[f'dts_{_zn}@{h_min}_cnt'] = float(B)
            else:
                out[f'dts_{_zn}@{h_min}'] = 0.0
                out[f'dts_{_zn}@{h_min}_cnt'] = 0.0
        if _live:
            _stage(f'evalfix_mard@{h_min}_sum', abs_rel[:, h_idx].sum())
            out[f'evalfix_mard@{h_min}_cnt'] = float(B)
        else:
            out[f'evalfix_mard@{h_min}_sum'] = 0.0
            out[f'evalfix_mard@{h_min}_cnt'] = 0.0

    # roc/trend on per-patch (30min) delta-BG; five-sum, finalized later.
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

    # Excursion amplitude: NET peak deviation from last_bg, missed by per-patch trend_amp_ratio.

    # Peak = argmax|true-last_bg|; real excursions only (|true_exc|>EXC_AMP_MIN_MGDL).
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

    # BG curve match: anchor-relative Pearson r of (pred-last) vs (true-last), SHAPE not level.
    xb = (pred_bg - last_bg_col).reshape(-1)
    yb = (true_bg - last_bg_col).reshape(-1)
    out['bgcurve_n'] = float(xb.numel())
    _stage('bgcurve_sx', xb.sum())
    _stage('bgcurve_sy', yb.sum())
    _stage('bgcurve_sxx', (xb * xb).sum())
    _stage('bgcurve_syy', (yb * yb).sum())
    _stage('bgcurve_sxy', (xb * yb).sum())

    # MARGINAL per-step coverage of the central 90% band (τ.05/.95), target 0.90 — NOT joint.
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

    # sign_balance@h: fraction below median, target 0.5; inner50_cov@h: coverage of [tau.25,.75].
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
    import re
    return re.sub(r'\x1b\[[0-9;]*m', '', s)


def _render_validation_table(
    step: int,
    val_metrics: dict[str, Any],
    prev_metrics: dict[str, Any] | None = None,
) -> str:
    """Render the validation table with ANSI colours: Metric | Value | Prev.

    A READING surface, not the record: validation_log.csv carries every metric unchanged;
    rows are dropped here where console cost outweighs the 1000-step cadence's value.
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
    # (metric, value, prev, trend, target, unit); value carries NO unit, layout appends it once.


    def _section(title: str) -> None:
        rows.append((_colored(title, _ANSI_BOLD + _ANSI_CYAN), '', '', '', '', ''))

    def _blank() -> None:
        rows.append(('', '', '', '', '', ''))

    def _prev_cell(prev_key: str | None, prev_scale: float, fmt: str, unit: str) -> str:
        prev = _prev_val(prev_key, prev_scale) if prev_key else None
        if prev is None:
            return _colored('—', _ANSI_GRAY)
        return _colored(f"{_fmt(fmt, prev)}{unit}", _ANSI_GRAY)

    def _absent_cell() -> str:
        return _colored('—', _ANSI_GRAY)

    def _absent_row(metric: str, prev_key: str | None, prev_scale: float,
                    fmt: str, unit: str, target: str) -> None:
        """An empty bin renders "-", never dropped and never 0.

        A vanished row reads as uncomputed, 0 as measured. The previous figure renders beside
        it, showing whether the bin has just emptied.
        """
        rows.append((metric, _absent_cell(),
                     _prev_cell(prev_key, prev_scale, fmt, unit),
                     '', target, ''))

    def _pct(v: float | None) -> float | None:
        """A [0, 1] fraction as a percentage; None stays None, since an unmeasured bin is not
        a measurement of zero."""
        return None if v is None else v * 100.0

    def info_row(metric: str, val: float | None, fmt: str = '{:+.4f}',
                 unit: str = '', target: str = 'Minimize',
                 prev_key: str | None = None,
                 prev_scale: float = 1.0,
                 direction: str = 'lower',
                 show_absent: bool = False) -> None:
        if val is None:
            if show_absent:
                _absent_row(metric, prev_key, prev_scale, fmt, unit, target)
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
                  prev_scale: float = 1.0,
                  show_absent: bool = False) -> None:
        target = f"<{_fmt(fmt, sota)}{unit}"
        if val is None:
            if show_absent:
                _absent_row(metric, prev_key, prev_scale, fmt, unit, target)
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
                   prev_scale: float = 1.0,
                   show_absent: bool = False) -> None:
        target = f">{_fmt(fmt, sota)}{unit}"
        if val is None:
            if show_absent:
                _absent_row(metric, prev_key, prev_scale, fmt, unit, target)
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
                 prev_scale: float = 1.0,
                 show_absent: bool = False) -> None:
        target = f"{_fmt(fmt, lo)}–{_fmt(fmt, hi)}{unit}"
        if val is None:
            if show_absent:
                _absent_row(metric, prev_key, prev_scale, fmt, unit, target)
            return
        prev = _prev_val(prev_key, prev_scale) if prev_key else None
        prev_cell = _prev_cell(prev_key, prev_scale, fmt, unit)
        band_mid = 0.5 * (lo + hi)
        pad = warn_pad if warn_pad is not None else 0.5 * (hi - lo)
        color = _tier_band(val, lo, hi, lo - pad, hi + pad)
        rows.append((metric, _colored(_fmt(fmt, val), color),
                     prev_cell, _trend_cell(val, prev, 'band', band_mid),
                     target, unit))

    def cov_sharp_row(metric: str, cov: float | None, width: float | None,
                      lo: float, hi: float, warn_pad: float | None = None,
                      prev_key: str | None = None,
                      n: float | None = None, n_unit: str = 'st',
                      target: str | None = None,
                      trend: str = 'band') -> None:
        """Coverage AND its width, always together; renders absent on both halves, never 0.

        trend: MARGINAL nominal is INSIDE ('band'); JOINT bounded above, use 'higher' there.
        """
        label = f"{metric}({int(n)}{n_unit})" if n is not None else metric
        tgt = target if target is not None else f"{lo:.0f}–{hi:.0f}% + width"
        prev_cell = _prev_cell(prev_key, 100.0, '{:.2f}', '%')
        if cov is None:
            rows.append((label, _absent_cell(), prev_cell, '', tgt, ''))
            return
        pad = warn_pad if warn_pad is not None else 0.5 * (hi - lo)
        color = _tier_band(cov, lo, hi, lo - pad, hi + pad)
        w = f" @ w {width:.1f} mg/dL" if width is not None else " @ w —"
        prev = _prev_val(prev_key, 100.0) if prev_key else None
        rows.append((label,
                     _colored(f"{cov:.2f}%", color) + _colored(w, _ANSI_GRAY),
                     prev_cell, _trend_cell(cov, prev, trend, 0.5 * (lo + hi)),
                     tgt, ''))

    def text_row(metric: str, cell: str | None, target: str = '') -> None:
        """A row whose value is a composite the numeric builders cannot carry."""
        rows.append((metric, cell if cell else _absent_cell(),
                     _absent_cell(), '', target, ''))

    _section('Training & Internal Losses')
    info_row('val_loss_total', val_metrics.get('val_loss_total'),
             prev_key='val_loss_total')
    info_row('val_loss_Q', val_metrics.get('val_loss_Q'),
             prev_key='val_loss_Q')
    # val_loss_total is the SELECTION scalar, on each sample's own mask (97% two-sided).

    # Improved while the one-sided band decayed — no proxy for calibration; other terms CSV-only.
    band_row('overfit_ratio', val_metrics.get('overfit_ratio'),
             0.400, 0.600, fmt='{:.3f}', warn_pad=0.10,
             prev_key='overfit_ratio')
    _blank()

    _section('BG Forecast (RMSE / MAE)')
    bg_rmse_sota = {30: 15.0, 60: 25.0, 120: 36.0}
    # MAE targets are RMSE x 0.8 (Gaussian E|e|=sqrt(2/pi)*sigma), a reading aid, not published.
    night_bg_rmse_sota = {180: 50.0, 360: 62.0, 480: 72.0}
    for h_min in (30, 60, 120):
        lower_row(f'bg_rmse @{h_min}m', val_metrics.get(f'bg_rmse_{h_min}'),
                  bg_rmse_sota[h_min], fmt='{:.1f}', unit=' mg/dL', warn_mult=1.5,
                  prev_key=f'bg_rmse_{h_min}', show_absent=True)
    for h_min in (30, 60, 120):
        lower_row(f'bg_mae  @{h_min}m', val_metrics.get(f'bg_mae_{h_min}'),
                  0.8 * bg_rmse_sota[h_min], fmt='{:.1f}', unit=' mg/dL',
                  warn_mult=1.5, prev_key=f'bg_mae_{h_min}', show_absent=True)
    _blank()

    # Only rows whose context isn't n_ctx: the roll starts from the visible run to origin.

    # Two sample sets: night RMSEs score NOCTURNAL subset, mean context/roll every rolled sample.
    _roll_ctx = val_metrics.get('roll_ctx_patches')
    _roll_n = int(val_metrics.get('roll_n', 0) or 0)
    _roll_skipped = int(val_metrics.get('roll_skipped', 0) or 0)
    _roll_seen = _roll_n + _roll_skipped
    _night_roll_n = int(val_metrics.get('night_roll_n', 0) or 0)
    _night_roll_skipped = int(val_metrics.get('night_roll_skipped', 0) or 0)
    _night_seen = _night_roll_n + _night_roll_skipped
    _section('BG Forecast (RMSE) — Night Only @ 180+')
    for h_min in (180, 360, 480):
        _n_h = int(val_metrics.get(f'night_bg_rmse_{h_min}_n', 0) or 0)
        lower_row(f'night_bg_rmse @{h_min}m ({_n_h}n)',
                  val_metrics.get(f'night_bg_rmse_{h_min}'),
                  night_bg_rmse_sota[h_min], fmt='{:.1f}', unit=' mg/dL', warn_mult=1.5,
                  prev_key=f'night_bg_rmse_{h_min}', show_absent=True)
    if _roll_seen:
        # show_absent: all-skipped has no mean context, renders '-', explains the absent RMSEs.

        # Both counters stay so skipping isn't silent; each denominator rides in the row LABEL.
        info_row(f'roll context (mean of {_roll_n} rolled)', _roll_ctx,
                 fmt='{:.1f}', unit=' patches',
                 target=f'{MIN_CONTEXT_PATCHES}–{MAX_CONTEXT_PATCHES} at full n_ctx',
                 prev_key='roll_ctx_patches', direction='none', show_absent=True)
        info_row(f'night roll skipped, short window (of {_night_seen} nocturnal)',
                 float(_night_roll_skipped),
                 fmt='{:.0f}', unit=' samples',
                 target=f'of {_night_seen} nocturnal',
                 prev_key='night_roll_skipped', direction='lower', show_absent=True)
        info_row(f'roll skipped, short window (of {_roll_seen} seen)',
                 val_metrics.get('roll_skipped'),
                 fmt='{:.0f}', unit=' samples', target=f'of {_roll_seen} seen',
                 prev_key='roll_skipped', direction='lower', show_absent=True)
    _blank()

    # Every coverage carries its band width beside it; alone, coverage is bought by widening.

    # Below is binned on d, distance to visible evidence; @30/60/90/120 IS d=1..4 one-sided.
    _fan_eh = _excursion_bucket_horizons(PREDICTION_PATCHES)

    def _fan_n(h: int, key: str = '_fan_n') -> float | None:
        v = val_metrics.get(f'{key}@{h}')
        return float(v) if isinstance(v, (int, float)) else None


    _section('Quantile Calibration (marginal coverage of 90% band, at one step)')
    for h_min in COVERAGE_HORIZONS_MIN:
        v = val_metrics.get(f'coverage90@{h_min}')
        cov_sharp_row(f'coverage90 @{h_min}m',
                      (v * 100.0) if v is not None else None,
                      val_metrics.get(f'sharp90@{h_min}'),
                      88.0, 92.0, warn_pad=5.0,
                      prev_key=f'coverage90@{h_min}', target='88–92% + width')
    for h_min in COVERAGE_HORIZONS_MIN:
        v = val_metrics.get(f'inner50_cov@{h_min}')
        cov_sharp_row(f'inner50_cov @{h_min}m',
                      (v * 100.0) if v is not None else None,
                      val_metrics.get(f'sharp50@{h_min}'),
                      45.0, 55.0, warn_pad=10.0,
                      prev_key=f'inner50_cov@{h_min}', target='≈ 50% + width')
    # One directional-bias witness: far horizon is where median drift shows first. Rest CSV-only.
    _sb_h = COVERAGE_HORIZONS_MIN[-1]
    _sb = val_metrics.get(f'sign_balance@{_sb_h}')
    info_row(f'sign_balance @{_sb_h}m',
             (_sb * 100.0) if _sb is not None else None,
             fmt='{:.2f}', unit='%', target='≈ 50% (diag)',
             prev_key=f'sign_balance@{_sb_h}', prev_scale=100.0, direction='none')

    # ONE-SIDED vs TWO-SIDED at matched d: forecast is one-sided, infill two-sided (0.82% at d=1).

    # BOTH ARMS SAME ESTIMATOR (coverage_sharpness_by_d); coverage90@h above is one step, not this.
    _os = val_metrics.get(f'_fan_cov90@{_fan_eh[0]}') if _fan_eh else None
    _ts = val_metrics.get(_infill_column('marginal90_cov', 1))
    info_row('  ↳ one-sided cov90 @d1', (_os * 100.0) if _os is not None else None,
             fmt='{:.2f}', unit='%', target='vs two-sided below',
             prev_key=f'_fan_cov90@{_fan_eh[0]}' if _fan_eh else None,
             prev_scale=100.0,
             direction='none', show_absent=True)
    info_row('  ↳ two-sided cov90 @d1', (_ts * 100.0) if _ts is not None else None,
             fmt='{:.2f}', unit='%', target='gap ⇒ sidedness starved',
             prev_key=_infill_column('marginal90_cov', 1), prev_scale=100.0,
             direction='none', show_absent=True)

    # CRPS/Winkler/per-d cov-sharp CSV-only: fixing d=1 deficit moves Winkler -1.7%, CRPS -0.3%.

    # joint90 stays: whole path in-band AT ONCE, bounded by smallest marginal, fell 0.69->0.58/6k.
    _joint_h = _fan_eh[-1] if _fan_eh else None
    if _joint_h is not None:
        _j = val_metrics.get(f'joint_cov90@{_joint_h}')
        # trend='higher', unlike other rows: 70-92 acceptable but midpoint 81 isn't the aim.

        # Bounded above by the smallest marginal, so 82->88 is recovery; vs 81 renders a red arrow.
        cov_sharp_row(f'joint90 whole path ≤{_joint_h}m',
                      (_j * 100.0) if _j is not None else None,
                      val_metrics.get(f'_fan_joint_width@{_joint_h}'),
                      70.0, 92.0, warn_pad=15.0,
                      n=_fan_n(_joint_h, '_fan_joint_n'), n_unit=' windows',
                      prev_key=f'joint_cov90@{_joint_h}',
                      target='≤ marginal (simultaneous)', trend='higher')
    _blank()


    _section('Relative Error & Derivative Tracking')
    mard_sota = {30: 7.0, 60: 12.0, 120: 19.0}
    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        lower_row(f'mard @{h_min}m', val_metrics.get(f'evalfix_mard@{h_min}'),
                  mard_sota[h_min], fmt='{:.2f}', unit='%', warn_mult=2.0,
                  prev_key=f'evalfix_mard@{h_min}', show_absent=True)
    lower_row('roc_rmse', val_metrics.get('roc_rmse'),
              14.0, fmt='{:.1f}', unit=' mg/dL', warn_mult=1.5,
              prev_key='roc_rmse')
    higher_row('direction (roc_corr)', val_metrics.get('roc_corr'),
               0.650, fmt='{:+.3f}', warn_gap=0.20,
               prev_key='roc_corr')
    # The anti-oscillation witness the headline RMSE structurally masks.
    lower_row('median_roughness', val_metrics.get('median_roughness'),
              0.010, fmt='{:.6f}', warn_mult=2.0, prev_key='median_roughness')
    lower_row('  ↳ far (last patch)', val_metrics.get('median_roughness_far'),
              0.010, fmt='{:.6f}', warn_mult=2.0, prev_key='median_roughness_far')
    _blank()

    # Mean-collapse detectors, none a selection metric: a mean forecaster scores well, ratio~0.
    _section('Amplitude & Excursion Shape')
    band_row('trend_amp_ratio', val_metrics.get('trend_amp_ratio'),
             0.80, 1.20, fmt='{:.3f}', warn_pad=0.30, prev_key='trend_amp_ratio')
    band_row('trend_gain_beta', val_metrics.get('trend_gain_beta'),
             0.80, 1.20, fmt='{:.3f}', warn_pad=0.30, prev_key='trend_gain_beta')
    higher_row('bg_curve_corr', val_metrics.get('bg_curve_corr'),
               0.700, fmt='{:+.3f}', warn_gap=0.20, prev_key='bg_curve_corr')
    _exc_n = val_metrics.get('exc_n')
    band_row(f"exc_amp_ratio({int(_exc_n)}exc)" if _exc_n else 'exc_amp_ratio',
             val_metrics.get('exc_amp_ratio'), 0.80, 1.20, fmt='{:.3f}',
             warn_pad=0.30, prev_key='exc_amp_ratio')
    band_row('exc_gain_beta', val_metrics.get('exc_gain_beta'),
             0.80, 1.20, fmt='{:.3f}', warn_pad=0.30, prev_key='exc_gain_beta')
    higher_row('exc_corr', val_metrics.get('exc_corr'),
               0.700, fmt='{:+.3f}', warn_gap=0.20, prev_key='exc_corr')
    # Two halves of the amplitude error the ratio can't separate: over=under reports ratio~1.
    _eo = val_metrics.get('exc_overshoot_frac')
    info_row('exc_overshoot_frac', (_eo * 100.0) if _eo is not None else None,
             fmt='{:.2f}', unit='%', target='≈ 50% against undershoot',
             prev_key='exc_overshoot_frac', prev_scale=100.0, direction='none')
    _eu = val_metrics.get('exc_undershoot_frac')
    info_row('exc_undershoot_frac', (_eu * 100.0) if _eu is not None else None,
             fmt='{:.2f}', unit='%', target='≈ 50% against overshoot',
             prev_key='exc_undershoot_frac', prev_scale=100.0, direction='none')
    _blank()

    # Raw vs region-binned split-conformal at excursion PEAKS, held-out 40% of val windows.

    # conf_hypo_esc is a WIDTH figure: half-width sweep takes it 0.250->0.079 as cov 0.888->0.981.
    _conf_n = val_metrics.get('conf_n')
    _section(f"Conformal Probe @ excursion peaks"
             f"{f' ({int(_conf_n)} windows)' if _conf_n else ''}")
    cov_sharp_row('conf cov90 raw', _pct(val_metrics.get('conf_cov90_raw')),
                  val_metrics.get('conf_width_raw'), 85.0, 95.0,
                  prev_key='conf_cov90_raw', target='≈ 90% + width')
    cov_sharp_row('conf cov90 binned', _pct(val_metrics.get('conf_cov90_cal')),
                  val_metrics.get('conf_width_cal'), 85.0, 95.0,
                  prev_key='conf_cov90_cal', target='≈ 90% + width')
    band_row('conf hypo-escape raw', _pct(val_metrics.get('conf_hypo_esc_raw')),
             5.0, 15.0, fmt='{:.2f}', unit='%', warn_pad=5.0,
             prev_key='conf_hypo_esc_raw', prev_scale=100.0)
    band_row('conf hypo-escape binned', _pct(val_metrics.get('conf_hypo_esc_cal')),
             5.0, 15.0, fmt='{:.2f}', unit='%', warn_pad=5.0,
             prev_key='conf_hypo_esc_cal', prev_scale=100.0)
    _blank()

    # Every zone, literature's A+B, then zone A/horizon; A+B hides which side moved, D/E dangerous.
    _section('Clinical Error Grid Analysis (Clarke)')
    higher_row('clarke_A', val_metrics.get('clarke_A_pct'),
               90.0, fmt='{:.2f}', unit='%', warn_gap=5.0, prev_key='clarke_A_pct')
    info_row('clarke_B (benign)', val_metrics.get('clarke_B_pct'),
             fmt='{:.2f}', unit='%', target='the A+B remainder',
             prev_key='clarke_B_pct', direction='none')
    lower_row('clarke_C', val_metrics.get('clarke_C_pct'),
              1.0, fmt='{:.2f}', unit='%', warn_mult=2.0, prev_key='clarke_C_pct')
    lower_row('clarke_D (dangerous)', val_metrics.get('clarke_D_pct'),
              1.0, fmt='{:.2f}', unit='%', warn_mult=2.0, prev_key='clarke_D_pct')
    lower_row('clarke_E (dangerous)', val_metrics.get('clarke_E_pct'),
              0.1, fmt='{:.2f}', unit='%', warn_mult=2.0, prev_key='clarke_E_pct')
    higher_row('clarke_A+B', val_metrics.get('clarke_AB_pct'),
               98.0, fmt='{:.2f}', unit='%', warn_gap=2.0,
               prev_key='clarke_AB_pct')
    # Every zone at every horizon: pooled shares average the decay, hiding a near/far D-vs-A trade.
    for _h in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        for _z, _lo, _hi in (('A', 90.0, None), ('B', None, None), ('C', None, 1.0),
                             ('D', None, 1.0), ('E', None, 0.1)):
            _v = val_metrics.get(f'evalfix_clarke_{_z}@{_h}')
            _k = f'evalfix_clarke_{_z}@{_h}'
            if _lo is not None:
                higher_row(f'  ↳ clarke_{_z} @{_h}m', _v, _lo,
                           fmt='{:.2f}', unit='%', warn_gap=5.0, prev_key=_k,
                           show_absent=True)
            elif _hi is not None:
                lower_row(f'  ↳ clarke_{_z} @{_h}m', _v, _hi,
                          fmt='{:.2f}', unit='%', warn_mult=2.0, prev_key=_k,
                          show_absent=True)
            else:
                info_row(f'  ↳ clarke_{_z} @{_h}m', _v, fmt='{:.2f}', unit='%',
                         target='the A+B remainder', prev_key=_k, direction='none',
                         show_absent=True)
    _blank()

    # Klonoff 2024, kept whole not reduced to one row: zones are contours, so SHAPE is the reading.

    # UNCOLOURED, no acceptance threshold; pZA 90%~MARD 10% rides in zone A's target. No A+B row.
    _section('Clinical Error Grid Analysis (DTS)')
    _DTS_ROWS = (
        ('a', 'pZA — no risk', 'pZA 90% ≈ MARD 10% (no threshold published)'),
        ('b', 'zone B — mild risk', 'minimize'),
        ('c', 'zone C — moderate risk', 'minimize'),
        ('d', 'zone D — high risk', 'minimize'),
        ('e', 'zone E — extreme risk', 'minimize'),
    )
    for _zn, _label, _target in _DTS_ROWS:
        _zv = val_metrics.get(f'dts_{_zn}_pct')
        info_row(_label, _zv, fmt='{:.2f}', unit='%', target=_target,
                 prev_key=f'dts_{_zn}_pct', direction='none', show_absent=True)
    # Per horizon, same axis as Clarke: pooled shares mix a 5-min error with a 2-hour one.
    for _h in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        for _zn in dts_grid.ZONE_NAMES:
            info_row(f'  ↳ dts_{_zn.upper()} @{_h}m',
                     val_metrics.get(f'dts_{_zn}@{_h}'), fmt='{:.2f}', unit='%',
                     target='no threshold published',
                     prev_key=f'dts_{_zn}@{_h}', direction='none', show_absent=True)
    _blank()

    # Kept whole: only thing scoring RATE OF CHANGE by region (Clarke/roc_corr don't).

    # At VALIDATION_N_PATIENTS=100 that bin held ~11 units; 1 unit moved cgega_ap_hypo 0.09.
    _section('Clinical Accuracy (CG-EGA)')
    for _reg, _ap_sota in (('hypo', 80.0), ('eu', 90.0), ('hyper', 85.0)):
        _ap = val_metrics.get(f'cgega_ap_{_reg}')
        higher_row(f'cgega_AP @{_reg}',
                   (_ap * 100.0) if _ap is not None else None,
                   _ap_sota, fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'cgega_ap_{_reg}', prev_scale=100.0, show_absent=True)
    # AP+BE+EP=1: rising BE against flat AP is error moving into the harmless bucket, not improving.
    for _reg in ('hypo', 'eu', 'hyper'):
        _be = val_metrics.get(f'cgega_be_{_reg}')
        info_row(f'cgega_BE @{_reg}',
                 (_be * 100.0) if _be is not None else None,
                 fmt='{:.2f}', unit='%', target='benign remainder',
                 prev_key=f'cgega_be_{_reg}', prev_scale=100.0, direction='none',
                 show_absent=True)
    for _reg, _ep_sota in (('hypo', 10.0), ('eu', 2.0), ('hyper', 5.0)):
        _ep = val_metrics.get(f'cgega_ep_{_reg}')
        lower_row(f'cgega_EP @{_reg}',
                  (_ep * 100.0) if _ep is not None else None,
                  _ep_sota, fmt='{:.2f}', unit='%', warn_mult=2.0,
                  prev_key=f'cgega_ep_{_reg}', prev_scale=100.0, show_absent=True)
    _blank()

    _section('Longitudinal Excursions & TIR')
    # pred_tir / true_tir are CSV-only: tir_abs_err is their difference and the figure that moves.
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

    # Per horizon, DISJOINT 30-min buckets: pooled pair hides a 30-min hit missed at 120-min.

    # Bar DECLINES with horizon (_excursion_target): near fixed by IOB, far horizon info-limited.
    for _h in _excursion_bucket_horizons(PREDICTION_PATCHES):
        _hn = val_metrics.get(f'hypo_n_steps@{_h}')
        higher_row(f'  ↳ hypo_recall @{_h}m'
                   + (f'({int(_hn)}st)' if _hn else ''),
                   _pct(val_metrics.get(f'hypo_recall@{_h}')),
                   _excursion_target(EXCURSION_TARGET_HYPO_RECALL, _h),
                   fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'hypo_recall@{_h}', prev_scale=100.0, show_absent=True)
        higher_row(f'  ↳ hypo_precision @{_h}m{_ptol_sfx}',
                   _pct(val_metrics.get(f'hypo_precision@{_h}')),
                   _excursion_target(EXCURSION_TARGET_HYPO_PRECISION, _h),
                   fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'hypo_precision@{_h}', prev_scale=100.0, show_absent=True)
    for _h in _excursion_bucket_horizons(PREDICTION_PATCHES):
        _yn = val_metrics.get(f'hyper_n_steps@{_h}')
        higher_row(f'  ↳ hyper_recall @{_h}m'
                   + (f'({int(_yn)}st)' if _yn else ''),
                   _pct(val_metrics.get(f'hyper_recall@{_h}')),
                   _excursion_target(EXCURSION_TARGET_HYPER_RECALL, _h),
                   fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'hyper_recall@{_h}', prev_scale=100.0, show_absent=True)
        higher_row(f'  ↳ hyper_precision @{_h}m{_ptol_sfx}',
                   _pct(val_metrics.get(f'hyper_precision@{_h}')),
                   _excursion_target(EXCURSION_TARGET_HYPER_PRECISION, _h),
                   fmt='{:.2f}', unit='%', warn_gap=10.0,
                   prev_key=f'hyper_precision@{_h}', prev_scale=100.0, show_absent=True)
    _blank()

    # BOTH sides: nocturnal hypo is the roll's use case, low-everywhere trades away hyper.

    # Night RMSEs live in rolled @180+; night_clarke/mard/cgega duplicate on a subset, CSV-only.
    _section('Nocturnal Validation Metrics')
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
    _blank()

    _cf_n = int(val_metrics.get('cf_n', 0) or 0)
    # Two directions + harder monotonicity check; magnitudes/carb-monotonicity/rescue are CSV-only.

    # These check the model responds to dose AT ALL — risk-v3 scored 0.56 on insulin, a coin flip.
    _section(f'Counterfactual Dose-Response ({_cf_n} samples)')
    _ccd = val_metrics.get('cf_carb_dir')
    info_row('carb→BG direction', (_ccd * 100.0) if _ccd is not None else None,
             fmt='{:.1f}', unit='%', target='≈ 100% (diag)',
             prev_key='cf_carb_dir', prev_scale=100.0, direction='none')
    _cid = val_metrics.get('cf_insulin_dir')
    info_row('insulin→BG direction', (_cid * 100.0) if _cid is not None else None,
             fmt='{:.1f}', unit='%', target='≈ 100% (diag)',
             prev_key='cf_insulin_dir', prev_scale=100.0, direction='none')
    _cim = val_metrics.get('cf_insulin_monotonic')
    info_row('insulin monotonic', (_cim * 100.0) if _cim is not None else None,
             fmt='{:.1f}', unit='%', target='≈ 100% (diag)',
             prev_key='cf_insulin_monotonic', prev_scale=100.0, direction='none')
    _blank()

    # Diagnostic, co-trains trunk, feeds no loss; thresholds are clock-usability, not external SOTA.

    # Chance at random phase: mae 6h, acc +-1h 8%, +-2h 17%, 4-bin 25%; can't split bias/std/miss.
    _section('Time-of-day probe (diagnostic)')
    lower_row('tod mae', val_metrics.get('tod_mae_h'), 1.5,
              fmt='{:.2f}', unit=' h', warn_mult=2.0, prev_key='tod_mae_h')
    lower_row('  ↳ mae @high confidence', val_metrics.get('tod_mae_hiconf'), 1.0,
              fmt='{:.2f}', unit=' h', warn_mult=2.0, prev_key='tod_mae_hiconf')
    # tod_acc_*/tod_gross_rate are ALREADY percentages (_run_validation scales); no _pct/prev_scale.

    # Either would double-scale: a 16.3% clock would render 1630.00% and tier GREEN above 60% bar.
    higher_row('tod acc ±1h', val_metrics.get('tod_acc_1h'),
               60.0, fmt='{:.2f}', unit='%', warn_gap=20.0,
               prev_key='tod_acc_1h')
    higher_row('tod acc ±2h', val_metrics.get('tod_acc_2h'),
               80.0, fmt='{:.2f}', unit='%', warn_gap=20.0,
               prev_key='tod_acc_2h')
    higher_row('tod acc (bin)', val_metrics.get('tod_acc_bin'),
               70.0, fmt='{:.2f}', unit='%', warn_gap=20.0,
               prev_key='tod_acc_bin')
    # Bias is signed, centred on zero: a band, not a minimand; 2h fast is as broken as 2h slow.
    band_row('tod bias', val_metrics.get('tod_bias_h'), -0.50, 0.50,
             fmt='{:+.3f}', unit=' h', warn_pad=1.0, prev_key='tod_bias_h')
    lower_row('tod std', val_metrics.get('tod_std_h'), 2.0,
              fmt='{:.2f}', unit=' h', warn_mult=2.0, prev_key='tod_std_h')
    lower_row('tod p90 (tail)', val_metrics.get('tod_p90_h'), 4.0,
              fmt='{:.2f}', unit=' h', warn_mult=1.5, prev_key='tod_p90_h')
    lower_row('tod gross-error rate', val_metrics.get('tod_gross_rate'),
              10.0, fmt='{:.2f}', unit='%', warn_mult=2.0,
              prev_key='tod_gross_rate')
    info_row('tod confidence (R)', val_metrics.get('tod_conf'),
             fmt='{:.3f}', target='resultant length, 0–1',
             prev_key='tod_conf', direction='none')
    # Within one window's slots, across paired windows a horizon apart; zero right, either wrong.
    lower_row('tod jump (within window)', val_metrics.get('tod_jump_h'), 1.0,
              fmt='{:.3f}', unit=' h', warn_mult=2.0, prev_key='tod_jump_h')
    lower_row('tod jump (cross-window)', val_metrics.get('tod_xwin_jump_h'), 1.0,
              fmt='{:.3f}', unit=' h', warn_mult=2.0, prev_key='tod_xwin_jump_h')
    _blank()

    # Prune orphaned section headers, collapse blank runs.
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

    # Layout: Metric | Value | Prev. The target column is carried but never rendered.
    def _with_unit(colored: str, unit: str) -> str:
        if not unit:
            return colored
        if colored.endswith(_ANSI_RESET):
            return colored[: -len(_ANSI_RESET)] + unit + _ANSI_RESET
        return colored + unit

    def _value_cell(value: str, trend: str, unit: str) -> str:
        return f"{_with_unit(value, unit)}{trend}"

    headers = ['Metric', 'Value', 'Prev']
    col_w = [len(h) for h in headers]
    for r in rows:
        if _is_section_row(r) or _is_blank_row(r):
            continue
        metric, value, prev, trend, _target, unit = r
        cells = (metric, _value_cell(value, trend, unit), prev)
        for i, cell in enumerate(cells):
            col_w[i] = max(col_w[i], len(_strip_ansi(cell)))

    c1, c2, c3 = col_w
    inner_w = c1 + c2 + c3 + 6

    max_title_w = 0
    for r in rows:
        if _is_section_row(r):
            max_title_w = max(max_title_w, len(_strip_ansi(r[0])))
    if max_title_w > inner_w:
        c1 += max_title_w - inner_w
        inner_w = max_title_w

    top_edge = f"├─{'─' * c1}─┬─{'─' * c2}─┬─{'─' * c3}─┤"
    mid_edge = f"├─{'─' * c1}─┼─{'─' * c2}─┼─{'─' * c3}─┤"
    bot_edge = f"└─{'─' * c1}─┴─{'─' * c2}─┴─{'─' * c3}─┘"
    sep_line = f"│ {' ' * c1} │ {' ' * c2} │ {' ' * c3} │"

    title_top = f"┌─{'─' * inner_w}─┐"
    title_lines = [
        f"Validation @ step {step} — {PREDICTION_HORIZON_HOURS}h window",
        "Conditioned (announced carbs+insulin+exercise)",
    ]
    title_inner = [
        f"│ {_ANSI_BOLD}{_ANSI_CYAN}{_pad(t, inner_w, 'c')}{_ANSI_RESET} │"
        for t in title_lines
    ]

    header_row = (
        f"│ {_ANSI_BOLD}{_pad('Metric', c1)}{_ANSI_RESET} "
        f"│ {_ANSI_BOLD}{_pad('Value', c2)}{_ANSI_RESET} "
        f"│ {_ANSI_BOLD}{_pad('Prev', c3)}{_ANSI_RESET} │"
    )

    lines = [title_top, *title_inner, top_edge, header_row, mid_edge]
    for r in rows:
        if _is_blank_row(r):
            lines.append(sep_line)
            continue
        metric, value, prev, trend, _target, unit = r
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
            f"│ {_pad(prev, c3)} │"
        )
    lines.append(bot_edge)
    return '\n'.join(lines)


def _build_optimizers(
    model: T1DMAI,
    weighting: KendallGalWeighting,
    muon_lr: float,
    adam_lr: float,
    muon_momentum: float,
    adam_weight_decay: float = ADAM_WEIGHT_DECAY,
) -> tuple[Muon, torch.optim.AdamW]:
    """Split params: Muon owns >=2D matrices, AdamW owns 1D/biases/norms.
    Normalized group gets AdamC's gamma_t/gamma_max decay correction; output group
    (bg_head[-1]/time_head[-1]) keeps constant decay. Kendall-Gal log-sigma own an AdamW
    group at wd=0, off model so the EMA never touches them.
    """
    # No norm follows output projections, so AdamC's <g,x>=0 analysis excludes them (sec 6).

    # Identified by object identity; everything else 2D is a normalized matrix.
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

    # Normalized group carries base_weight_decay+wd_corrected so _update_lr can rescale each step.

    # Correction off: both equal MUON_WEIGHT_DECAY, same as one group (momentum bufs are per-param).
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
    """Warmup + cosine LR schedule for both optimizers.
    With wd_correction, wd_corrected Muon groups get weight_decay=base_weight_decay*ratio
    (ratio=gamma_t/gamma_max, same as LR), realizing Algorithm 1 gamma_t^2/gamma_max via the
    optimizer's p*(1-lr*wd) step; output group and AdamW keep constant decay.
    """
    if step < warmup_steps:
        ratio = step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        ratio = lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))
    for group in muon_opt.param_groups:
        group['lr'] = peak_muon_lr * ratio
        if group.get('wd_corrected', False):
            # Written in BOTH branches so it can't retain a stale decay if toggled live.
            group['weight_decay'] = (
                group['base_weight_decay'] * ratio if wd_correction
                else group['base_weight_decay']
            )
    for group in adam_opt.param_groups:
        group['lr'] = peak_adam_lr * ratio


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
    """Write the training-summary JSON (at each checkpoint and at the end)."""
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


VAL_BATCH_SIZE = 8

# Output channels every eval path announces; a strict subset evaluates a plan never trained under.

# An un-announced maskable slot reads as normalize(0), a legal value, so nothing raises.
_ANNOUNCE_CHANNELS = tuple(sorted(CHANNEL_TO_FEAT))


def _reconstruct_context_from_patch(
    patches: torch.Tensor,
    n_ctx: int,
    mask_idx: "np.ndarray | torch.Tensor",
    valid: "np.ndarray | torch.Tensor",
    min_patches: int = MIN_CONTEXT_PATCHES,
) -> "torch.Tensor | None":
    """VISIBLE rolling/counterfactual context, (C, PATCH_SIZE, N_INPUT_FEATURES).
    A masked patch decodes to ~142 mg/dL (balanced pool), an ordinary value not a sentinel, so
    a positional prefix would fabricate one; returns the longest VISIBLE run ending at n_ctx-1
    instead. None if that run is shorter than min_patches; caller must report the skip.
    """
    feat_cols = PATCH_SIZE * N_INPUT_FEATURES
    assert patches.shape[1] == feat_cols, (
        f"patches must be (T, {feat_cols}), got {tuple(patches.shape)}"
    )
    masked = np.asarray(mask_idx).reshape(-1)[np.asarray(valid).reshape(-1).astype(bool)]
    in_context = masked[masked < n_ctx]
    ctx_start = int(in_context.max()) + 1 if in_context.size else 0
    if n_ctx - ctx_start < max(min_patches, 1):
        return None
    # No masked index may reach the context tensor — one tripwire for all three roll sites.
    assert not bool(np.isin(np.arange(ctx_start, n_ctx), masked).any()), (
        f"masked patch inside the reconstructed context [{ctx_start}, {n_ctx})"
    )
    context = patches[ctx_start:n_ctx].reshape(-1, PATCH_SIZE, N_INPUT_FEATURES).clone()
    context[:, :, BG_MASKED_FEAT] = 0.0
    return context


def _observed_patches(sample: dict[str, Any], norm_stats: dict) -> torch.Tensor:
    """One sample patch rows with every withheld BG written back, (T, PATCH_DIM).
    Rolling context is the OBSERVED CGM history; a holed context drops most of the sample
    (526/600 nano windows, 88%; 171/203 nocturnal). Not leakage: restored patches precede the
    forecast origin; RMSE unchanged paired (52.66 vs 51.94 at 360min). Feat 4 left as found.
    """
    bf = sample['bg_formula_data']
    patches = sample['patches']
    if not torch.is_tensor(patches):
        patches = torch.from_numpy(np.asarray(patches)).float()
    _, bg_z = _window_bg_mgdl(
        patches.unsqueeze(0),
        torch.as_tensor(np.asarray(bf['mask_idx'])).long().reshape(1, -1),
        torch.as_tensor(np.asarray(bf['valid'])).bool().reshape(1, -1),
        torch.as_tensor(np.asarray(sample['targets'])).float().unsqueeze(0),
        norm_stats,
    )
    out = patches.clone()
    out[:, _BG_FEAT::N_INPUT_FEATURES] = bg_z[0]
    return out


def _is_nocturnal(hour: float) -> bool:
    if NOCTURNAL_START_HOUR <= NOCTURNAL_END_HOUR:
        return NOCTURNAL_START_HOUR <= hour < NOCTURNAL_END_HOUR
    else:
        return hour >= NOCTURNAL_START_HOUR or hour < NOCTURNAL_END_HOUR


def _make_long_horizon_overrides_fn(bf: dict):
    """Per-roll announced carb(0)/insulin(1)/exercise(2) overrides for predict_rolling.
    Long-horizon roll is KNOWN-PLAN: doses announced each roll while BG stays autoregressive.
    Announced set is exactly tuple(CHANNEL_TO_FEAT); a subset reads as normalize(0), "no session".
    Returns None past shipped future or when extended keys are absent (falls back unconditioned).
    """
    keys = ('extended_carb_norm', 'extended_insulin_norm', 'extended_exercise_norm',
            'extended_carb_raw', 'extended_insulin_raw', 'extended_exercise_raw')
    if not all(k in bf for k in keys):
        return None
    _ps = PREDICTION_PATCHES * PATCH_SIZE
    norm_ch = {
        0: np.asarray(bf['extended_carb_norm'], dtype=np.float32),
        1: np.asarray(bf['extended_insulin_norm'], dtype=np.float32),
        2: np.asarray(bf['extended_exercise_norm'], dtype=np.float32),
    }
    raw_ch = {
        0: np.asarray(bf['extended_carb_raw'], dtype=np.float32),
        1: np.asarray(bf['extended_insulin_raw'], dtype=np.float32),
        2: np.asarray(bf['extended_exercise_raw'], dtype=np.float32),
    }
    assert tuple(sorted(norm_ch)) == _ANNOUNCE_CHANNELS, (
        f"announced set {tuple(sorted(norm_ch))} != CHANNEL_TO_FEAT {_ANNOUNCE_CHANNELS}"
    )
    n_avail = min(int(v.shape[0]) for v in norm_ch.values())

    def fn(roll_idx, mu_np, abs_n_ctx):
        a = roll_idx * _ps
        b = a + _ps
        if b > n_avail:
            return None

        def rs(x):
            return x[a:b].reshape(PREDICTION_PATCHES, PATCH_SIZE)
        return ({ch: rs(v) for ch, v in norm_ch.items()},
                {ch: rs(v) for ch, v in raw_ch.items()})
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
    """Roll each val sample forward for horizons past a single forward pass.
    Accumulates bg_rmse_{h}_*/bg_mae_{h}_*; predict_rolling is BG-autoregressive (pred_bg=
    f_inv(median) carried across rolls), CONDITIONED on announced future insulin/carb/exercise
    (_make_long_horizon_overrides_fn), the known-plan nocturnal-hypo regime.
    """
    from inference import predict_rolling

    if n_rolls <= 0:
        return

    single_pass_steps = PREDICTION_PATCHES * PATCH_SIZE
    dt = 0

    for sample in samples:
        n_ctx = int(sample['n_context_patches'])
        bf = sample['bg_formula_data']
        # Read before the floor: a dropped night sample still needs its own denominator counted.
        is_night = _is_nocturnal(float(bf.get('pred_start_hour', 0.0)))

        # Roll re-feeds its own median; context must be REAL readings reaching the edge, never

        # the positional prefix (masked would roll off a fabricated value); restore BG first.
        context = _reconstruct_context_from_patch(
            _observed_patches(sample, norm_stats), n_ctx,
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool))
        if context is None:
            agg['roll_skipped'] = agg.get('roll_skipped', 0.0) + 1.0
            if night_agg is not None and is_night:
                night_agg['night_roll_skipped'] = night_agg.get('night_roll_skipped', 0.0) + 1.0
            continue
        agg['roll_ctx_sum'] = agg.get('roll_ctx_sum', 0.0) + float(context.shape[0])
        agg['roll_ctx_cnt'] = agg.get('roll_ctx_cnt', 0.0) + 1.0
        if night_agg is not None and is_night:
            night_agg['night_roll_cnt'] = night_agg.get('night_roll_cnt', 0.0) + 1.0
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


def _cf_bolus_curve(channel: str, total: float, n_steps: int) -> np.ndarray:
    """Per-step curve for a counterfactual bolus of total, truncated to n_steps.
    Carb at GI 100, insulin under dose-scaled PK (SPEC/invariants.md §5), so the probe injects
    the shape the model was pretrained on, not a flat block no channel ever carries.
    """
    if channel == 'carb':
        curve = gamma_curve(total, 2.0, 15.0, 120.0)
    elif channel == 'insulin':
        curve = gamma_curve(total, *bolus_pk_for_dose(total))
    else:
        raise ValueError(f"unknown counterfactual channel {channel!r}")
    out = np.zeros(n_steps, dtype=np.float32)
    n = min(n_steps, int(curve.shape[0]))
    out[:n] = curve[:n]
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
    """Counterfactual dose-response probe over up to VALIDATION_PROBE_N_PATIENTS samples.
    Perturbs one RAW bolus (carb: GI-100 gamma; insulin: dose-scaled PK gamma, SPEC/invariants.md
    §5) from the first masked step, log1p-z renormalized; exercise held at its TRUE curve.
    samples reuses val_dataset[i] dicts rather than re-indexing, which would re-run the simulator.
    """
    from inference import predict

    _ps = PREDICTION_PATCHES * PATCH_SIZE
    carb_m = float(norm_stats['carb_intake']['mean'])
    carb_s = float(norm_stats['carb_intake']['std'])
    ins_m = float(norm_stats['insulin_combined']['mean'])
    ins_s = float(norm_stats['insulin_combined']['std'])
    carb_B = float(CF_CARB_BOLUS_G)
    ins_B = float(CF_INSULIN_BOLUS_U)
    carb_curve = _cf_bolus_curve('carb', carb_B, _ps)
    ins_curve = _cf_bolus_curve('insulin', ins_B, _ps)

    def _renorm(raw: np.ndarray, m: float, s: float) -> torch.Tensor:
        """Raw per-step (P*S,) → normalized (P, S) torch tensor via log1p z."""
        norm = (np.log1p(np.maximum(raw, 0.0)) - m) / (s + 1e-8)
        return torch.from_numpy(
            norm.reshape(PREDICTION_PATCHES, PATCH_SIZE).astype(np.float32))

    def _forecast(carb_t: torch.Tensor, ins_t: torch.Tensor,
                  ex_t: torch.Tensor) -> np.ndarray:
        overrides = {0: carb_t, 1: ins_t, 2: ex_t}
        assert tuple(sorted(overrides)) == _ANNOUNCE_CHANNELS, (
            f"announced set {tuple(sorted(overrides))} != CHANNEL_TO_FEAT "
            f"{_ANNOUNCE_CHANNELS}"
        )
        res = predict(model, context, normalization_stats=norm_stats,
                      device=device, overrides=overrides)
        return res['median_bg'].detach().cpu().numpy()      # (P*S,) mg/dL

    # Capped at VALIDATION_PROBE_N_PATIENTS, not VALIDATION_N_PATIENTS: 5 forwards per window.
    n_val = min(len(val_dataset), VALIDATION_PROBE_N_PATIENTS)
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
            bf = sample['bg_formula_data']
            # Baseline/perturbed arms share context; a masked patch moves both by one fake value.

            # No length floor (unlike rolling): a two-arm diff shifts together, not compounding.
            context = _reconstruct_context_from_patch(
                _observed_patches(sample, norm_stats), n_ctx,
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool),
                min_patches=1)
            if context is None:
                continue

            keys = ('extended_carb_norm', 'extended_insulin_norm',
                    'extended_exercise_norm', 'extended_carb_raw',
                    'extended_insulin_raw', 'extended_exercise_raw')
            if not all(k in bf for k in keys):
                continue

            carb_norm = np.asarray(bf['extended_carb_norm'], dtype=np.float32)[:_ps]
            ins_norm = np.asarray(bf['extended_insulin_norm'], dtype=np.float32)[:_ps]
            ex_norm = np.asarray(bf['extended_exercise_norm'], dtype=np.float32)[:_ps]
            carb_raw = np.asarray(bf['extended_carb_raw'], dtype=np.float32)[:_ps]
            ins_raw = np.asarray(bf['extended_insulin_raw'], dtype=np.float32)[:_ps]
            if (carb_norm.shape[0] < _ps or ins_norm.shape[0] < _ps
                    or ex_norm.shape[0] < _ps):
                continue

            carb_true_t = torch.from_numpy(
                carb_norm.reshape(PREDICTION_PATCHES, PATCH_SIZE))
            ins_true_t = torch.from_numpy(
                ins_norm.reshape(PREDICTION_PATCHES, PATCH_SIZE))
            # Exercise never perturbed: held at truth every arm, so probed BG delta is dose alone.
            ex_true_t = torch.from_numpy(
                ex_norm.reshape(PREDICTION_PATCHES, PATCH_SIZE))

            baseline = _forecast(carb_true_t, ins_true_t, ex_true_t)     # (P*S,)

            # +full carb bolus, insulin at truth.
            carb_full = _renorm(carb_raw + carb_curve, carb_m, carb_s)
            carb_pert = _forecast(carb_full, ins_true_t, ex_true_t)
            carb_dbg = float(np.mean(carb_pert - baseline))
            carb_dbg_sum += carb_dbg
            carb_dir_hits += int(carb_dbg > 0.0)

            # +full insulin bolus, carb at truth.
            ins_full = _renorm(ins_raw + ins_curve, ins_m, ins_s)
            ins_pert = _forecast(carb_true_t, ins_full, ex_true_t)
            ins_dbg = float(np.mean(ins_pert - baseline))
            ins_dbg_sum += ins_dbg
            ins_dir_hits += int(ins_dbg < 0.0)

            # Monotonicity across [0, B/2, B].
            carb_half = _renorm(carb_raw + 0.5 * carb_curve, carb_m, carb_s)
            carb_peaks = [
                float(baseline.max()),
                float(_forecast(carb_half, ins_true_t, ex_true_t).max()),
                float(carb_pert.max()),
            ]
            carb_mono_hits += int(
                carb_peaks[1] >= carb_peaks[0] - 1e-6
                and carb_peaks[2] >= carb_peaks[1] - 1e-6)

            ins_half = _renorm(ins_raw + 0.5 * ins_curve, ins_m, ins_s)
            ins_mins = [
                float(baseline.min()),
                float(_forecast(carb_true_t, ins_half, ex_true_t).min()),
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
    Fits split-conformal on a deterministic 60% of collected windows, scores EXCURSION-PEAK
    coverage on the disjoint 40% (region-binned, mondrian.fit_mondrian); ~100 windows, directional.
    bands (M,H,K) mg/dL, true (M,H) mg/dL, last (M,); widths mg/dL, empty if too few excursions.
    """
    import mondrian
    M = bands.shape[0]
    if M < 50:
        return {}
    perm = np.random.default_rng(0).permutation(M)
    ncal = int(0.6 * M)
    ci, ti = perm[:ncal], perm[ncal:]
    # Region is read off the median line (conformal holds it fixed), so a window's bin doesn't move.
    cal_bin = mondrian.region_bin(
        mondrian.forecast_destination(bands[ci], _CONF_MEDIAN_IDX))
    delta, _, _ = mondrian.fit_mondrian(
        bands[ci], true[ci], cal_bin, QUANTILE_LEVELS, _CONF_MEDIAN_IDX,
        patients=[int(i) for i in ci])
    bt, tt, lt = bands[ti], true[ti], last[ti]
    test_bin = mondrian.region_bin(
        mondrian.forecast_destination(bt, _CONF_MEDIAN_IDX))
    j = np.argmax(np.abs(tt - lt[:, None]), axis=1)            # per-window true-peak step
    idx = np.where((tt.max(1) - tt.min(1)) > 25.0)[0]          # excursion windows only
    if len(idx) < 20:
        return {}

    def _stats(B: np.ndarray) -> tuple[float, float, float]:
        pt = tt[idx, j[idx]]                                   # true at the peak
        cov = np.mean((B[idx, j[idx], _CONF_LO_IDX] <= pt)
                      & (pt <= B[idx, j[idx], _CONF_HI_IDX]))
        hypo = np.mean(pt < B[idx, j[idx], _CONF_HYPO_IDX])    # truth below the τ=0.10 hypo edge
        width = np.mean(B[idx, j[idx], _CONF_HI_IDX] - B[idx, j[idx], _CONF_LO_IDX])
        return float(cov), float(hypo), float(width)

    bt_cal = mondrian.apply_mondrian(bt, delta, test_bin, _CONF_MEDIAN_IDX)
    cov_raw, hypo_raw, wid_raw = _stats(bt)
    cov_cal, hypo_cal, wid_cal = _stats(bt_cal)

    return {'conf_cov90_raw': cov_raw, 'conf_cov90_cal': cov_cal,
            'conf_width_raw': wid_raw, 'conf_width_cal': wid_cal,
            'conf_hypo_esc_raw': hypo_raw, 'conf_hypo_esc_cal': hypo_cal,
            'conf_n': float(len(idx))}


def _forecast_protocol(
    patches: torch.Tensor,
    mask_idx: torch.Tensor,
    valid: torch.Tensor,
    n_context_patches: torch.Tensor,
) -> "dict[str, torch.Tensor] | None":
    """FORECAST protocol, rebuilt from a collated batch: trailing PREDICTION_PATCHES masked/scored,
    the deployed 2h forecast case. A batch mask is NOT this (slot j = mask_idx[j], not j patches
    past the edge); left-padding puts the zone at [T-P, T) regardless of n_ctx. Rows whose
    context-edge patch (T-P-1) is masked are DROPPED, else last_bg leaks a withheld true value.
    """
    B, T, _ = patches.shape
    P = PREDICTION_PATCHES
    device = patches.device
    assert T > P, f"window of {T} patches cannot hold a {P}-patch forecast zone"

    masked = torch.zeros(B, T, dtype=torch.bool, device=device)
    v_rows, v_cols = valid.nonzero(as_tuple=True)
    masked[v_rows, mask_idx[v_rows, v_cols]] = True

    keep = ~masked[:, T - P - 1]
    if not bool(keep.any()):
        return None
    rows = keep.nonzero(as_tuple=True)[0]
    n = int(rows.numel())

    fc_patches = patches[rows].clone()
    for feat_idx in NON_MASKABLE_FEATS:
        fc_patches[:, T - P:, feat_idx::N_INPUT_FEATURES] = 0.0
    fc_patches[:, T - P:, BG_MASKED_FEAT::N_INPUT_FEATURES] = 1.0

    fc_masked = masked[rows].clone()
    fc_masked[:, T - P:] = True
    lens = n_context_patches.to(device).reshape(-1) + P
    is_pad = (torch.arange(T, device=device).unsqueeze(0)
              < (T - lens).unsqueeze(1))[rows]
    attn = create_attention_mask_from_visible(~fc_masked, is_pad)

    fc_mask_idx = (torch.arange(T - P, T, device=device, dtype=torch.long)
                   .unsqueeze(0).expand(n, P).contiguous())
    return {'rows': rows, 'patches': fc_patches, 'attn_mask': attn,
            'mask_idx': fc_mask_idx}


def _window_bg_mgdl(
    patches: torch.Tensor,
    mask_idx: torch.Tensor,
    valid: torch.Tensor,
    targets: torch.Tensor,
    norm_stats: dict,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Whole window bg, in mg/dL and normalized z, with masked patches filled.
    A masked patch carries z=0 (~142 mg/dL), legal not a sentinel; withheld values live in
    targets (raw mg/dL), written back to reconstitute the fully-observed window INFILL needs.
    Returns (bg_mgdl, bg_z), both (B,T,PATCH_SIZE); left-pad columns hold junk, never scored.
    """
    device = patches.device
    bg_z = patches[:, :, _BG_FEAT::N_INPUT_FEATURES].clone()          # (B, T, S)
    z_fill = torch.from_numpy(
        normalize(targets.detach().cpu().numpy()[..., None], norm_stats,
                  channel_names=[_BG_CHANNEL])[..., 0]
    ).to(device=device, dtype=bg_z.dtype)                             # (B, M, S)
    rows, slots = valid.nonzero(as_tuple=True)
    bg_z[rows, mask_idx[rows, slots]] = z_fill[rows, slots]

    bg_mgdl = denormalize(bg_z.unsqueeze(-1), norm_stats,
                          channel_names=[_BG_CHANNEL]).squeeze(-1)    # (B, T, S)
    bg_mgdl[rows, mask_idx[rows, slots]] = targets[rows, slots].to(bg_mgdl.dtype)
    return bg_mgdl, bg_z


def _infill_protocol(
    patches: torch.Tensor,
    mask_idx: torch.Tensor,
    valid: torch.Tensor,
    targets: torch.Tensor,
    n_context_patches: torch.Tensor,
    norm_stats: dict,
    rng: "np.random.Generator",
) -> "dict[str, Any] | None":
    """INFILL protocol, rebuilt from a collated batch.
    Mask is sampled INTERIOR spans + the mandatory trailing forecast span; only interior spans
    are scored (spans/d-rule/anchor come from infill_masked_set). Own training mask is REPLACED,
    not kept, so interior spans stay two-sided; rows with too-short context are DROPPED.
    """
    from metrics.protocols import infill_masked_set

    B, T, _ = patches.shape
    S = PATCH_SIZE
    M = MAX_MASKED_PATCHES
    P = PREDICTION_PATCHES
    device = patches.device

    bg_mgdl, bg_z = _window_bg_mgdl(patches, mask_idx, valid, targets, norm_stats)

    keep: list[int] = []
    sets: list[tuple[int, Any, np.ndarray]] = []
    for b in range(B):
        n_ctx = int(n_context_patches.reshape(-1)[b])
        try:
            ms = infill_masked_set(n_ctx, rng)
        except ValueError:
            # Context can't hold MASK_MAX_SPANS spans + separators: no masked set for this row.
            continue
        n_pad = T - ms.seq_len
        assert n_pad >= 0, f"row {b}: n_ctx={n_ctx} exceeds the collated width {T}"
        keep.append(b)
        sets.append((b, ms, bg_mgdl[b, n_pad:n_pad + ms.seq_len]
                     .reshape(-1).detach().cpu().numpy()))
    if not keep:
        return None

    rows = torch.tensor(keep, device=device, dtype=torch.long)
    n = len(keep)

    # Every patch starts VISIBLE with true bg restored; this protocol's own spans then withhold

    # and announce. Feat 4 is rewritten wholesale: training mask's announcement isn't this one.
    inf_patches = patches[rows].clone()
    inf_patches[:, :, _BG_FEAT::N_INPUT_FEATURES] = bg_z[rows]
    inf_patches[:, :, BG_MASKED_FEAT::N_INPUT_FEATURES] = 0.0

    inf_masked = torch.zeros(n, T, dtype=torch.bool, device=device)
    inf_mask_idx = torch.zeros(n, M, dtype=torch.long, device=device)
    inf_anchor = torch.zeros(n, M, dtype=torch.float32, device=device)
    for i, (_b, ms, cgm) in enumerate(sets):
        n_pad = T - ms.seq_len
        patch_cols = torch.from_numpy(ms.mask_idx).to(device) + n_pad    # (M,)
        live = torch.from_numpy(ms.valid).to(device)                     # (M,)
        inf_mask_idx[i] = patch_cols
        inf_masked[i, patch_cols[live]] = True
        # Padded slots anchor from the first patch, as data._build_sample does.

        # MaskedSet.valid discards these padded slots downstream.
        anchor = np.full(M, float(cgm[ms.anchor_step[0]]), dtype=np.float32)
        anchor[ms.valid] = cgm[ms.anchor_step[ms.valid]]
        inf_anchor[i] = torch.from_numpy(anchor).to(device)

    withheld = inf_masked.unsqueeze(-1).expand(n, T, S)
    for feat_idx in NON_MASKABLE_FEATS:
        block = inf_patches[:, :, feat_idx::N_INPUT_FEATURES]
        inf_patches[:, :, feat_idx::N_INPUT_FEATURES] = block.masked_fill(withheld, 0.0)
    announce = inf_patches[:, :, BG_MASKED_FEAT::N_INPUT_FEATURES]
    inf_patches[:, :, BG_MASKED_FEAT::N_INPUT_FEATURES] = announce.masked_fill(
        withheld, 1.0)
    # Masking isn't inferable from position; z=0 in a withheld slot decodes to an ordinary

    # reading, so feat 4 IS the announcement: check it before the forward, not after.
    assert bool(((inf_patches[:, :, BG_MASKED_FEAT::N_INPUT_FEATURES] > 0.5)
                 == withheld).all()), (
        "feat 4 does not reproduce the infill protocol's masked set")

    lens = n_context_patches.to(device).reshape(-1)[rows] + P
    is_pad = (torch.arange(T, device=device).unsqueeze(0)
              < (T - lens).unsqueeze(1))
    attn = create_attention_mask_from_visible(~inf_masked, is_pad)
    return {'patches': inf_patches, 'attn_mask': attn, 'mask_idx': inf_mask_idx,
            'anchor_bg': inf_anchor, 'sets': sets, 'bg_mgdl': bg_mgdl}


# Five metrics.scoring rules + two metrics.protocols protocols, mapped onto header column names.

# Every number is computed there; this only names them, so no rule grows a second definition.

# One tuple each, read by log header, checkpoint record and this mapping; a family added

# to the header and forgotten here reads as an always-empty, unmeasured column.
FAN_SCORE_FAMILIES: tuple[str, ...] = (
    'crps', 'winkler90', 'sharp90', 'sharp50', 'joint_cov90')
INFILL_FAMILIES: tuple[str, ...] = (
    'crps_n', 'rmse', 'rmse_interp', 'crps', 'winkler90',
    'marginal90_cov', 'marginal90_width_mean')


def _infill_column(base: str, d: int) -> str:
    """One infill column name, built by ``metrics.protocols`` and never locally. ``column``
    refuses an infill name without a ``d``, which keeps a pooled masked-BG scalar — a figure
    that improves for free — out of the log and out of the table."""
    from metrics.protocols import INFILL, column
    return column(INFILL, base, d)


def _infill_reachable_d() -> "tuple[int, ...]":
    """The ``d`` bins the infill protocol can populate at the live sampler."""
    from metrics.protocols import INFILL, reachable_d
    return reachable_d(INFILL)


def _absent_if_nan(x: "float | None") -> "float | None":
    """An empty bin reads as absent. NaN is never rendered or logged as 0."""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _nominal_for(lo_idx: int) -> float:
    """The nominal central level whose LOWER edge is fan node ``lo_idx``;
    ``metrics.scoring.central_levels`` derives the fan's own τ pairs, keeping ``1 - 2τ`` in one
    place."""
    # metrics.scoring, not metrics: metrics' lazy __getattr__ recurses on import scoring.
    from metrics.scoring import central_levels
    for nominal, lo, _hi in central_levels():
        if lo == lo_idx:
            return nominal
    raise LookupError(f"fan level {QUANTILE_LEVELS[lo_idx]} has no partner τ")


def _alarm_point_at_tau(curve, tau: float):
    """The swept operating point whose firing set IS the τ band-edge alarm.
    Alarm fires when fan lower edge at τ dips below hypo threshold: q(τ) < thr exactly when
    P(BG <= thr) > τ under scoring.predictive_cdf. Sweep cuts at REALISED score >= c; smallest
    cut above τ selects the score-over-τ set. None when no group clears τ (never fires).
    """
    above = [p for p in curve.points
             if p.score_threshold is not None and p.score_threshold > tau]
    return min(above, key=lambda p: p.score_threshold) if above else None


def _forecast_fan_columns(
    q: np.ndarray,
    true: np.ndarray,
    d: np.ndarray,
    group: np.ndarray,
    observed_days: "float | None",
) -> dict[str, Any]:
    """Forecast protocol scoring-rule columns, per d.
    q (N,PATCH_SIZE,N_QUANTILES) and true (N,PATCH_SIZE) mg/dL, already out of risk space. Patch p
    sits at d=p+1, so @30/60/90/120 IS d=1..4; nothing pooled over d is emitted.
    joint_cov90@h is SIMULTANEOUS path coverage, never interchangeable with coverage90@h.
    """
    from metrics.protocols import FORECAST, reachable_d
    from metrics.scoring import (
        AlarmCurve, alarm_operating_curve, coverage_sharpness_by_d, crps_by_d,
        forecast_lead_minutes, joint_coverage_by_d, winkler_by_d,
    )

    eh = _excursion_bucket_horizons(PREDICTION_PATCHES)
    fc_d = reachable_d(FORECAST)
    lead_min = forecast_lead_minutes(d)
    n_groups = int(np.unique(group).size)

    crps = crps_by_d(q, true, d)
    winkler = winkler_by_d(q, true, d)
    coverage = coverage_sharpness_by_d(q, true, d)
    joint = joint_coverage_by_d(q, true, d, group)
    # An exact sweep: every realised score is a cut, so tau ladder points sit ON the curve.
    alarm = alarm_operating_curve(
        q, true, d, group, lead_min, observed_days, max_points=n_groups + 2)

    n90 = _nominal_for(_TAU_LO_IDX)
    n50 = _nominal_for(_TAU_INNER_LO_IDX)
    out: dict[str, Any] = {}
    for h, dd in zip(eh, fc_d):
        out[f'crps@{h}'] = _absent_if_nan(crps.by_d.get(dd))
        out[f'winkler90@{h}'] = _absent_if_nan(winkler[n90].by_d.get(dd))
        cs90 = coverage[n90].by_d.get(dd)
        cs50 = coverage[n50].by_d.get(dd)
        out[f'sharp90@{h}'] = _absent_if_nan(cs90.mean_width if cs90 else None)
        out[f'sharp50@{h}'] = _absent_if_nan(cs50.mean_width if cs50 else None)
        jt = joint[n90].joint_path_to_d.get(dd)
        out[f'joint_cov90@{h}'] = _absent_if_nan(jt.coverage if jt else None)
        # Beside the columns, table-only: coverage never travels without its width.
        out[f'_fan_cov90@{h}'] = _absent_if_nan(cs90.coverage if cs90 else None)
        out[f'_fan_n@{h}'] = float(crps.n_by_d.get(dd, 0))
        out[f'_fan_joint_width@{h}'] = _absent_if_nan(jt.mean_width if jt else None)
        out[f'_fan_joint_n@{h}'] = float(jt.n if jt else 0)

    def _alarm_columns(curve: "AlarmCurve | None", suffix: str) -> None:
        """One curve's three figures at every swept τ. Detection rate, false alarms per day and
        median lead travel together: a rate bought at a two-minute lead is not a usable alarm
        and the rate alone cannot show it."""
        for tau in _alarm_curve_taus():
            tag = _tau_tag(tau)
            # A ``d`` no scored patch reached has no curve: nothing was measured there.
            det = fa = lead = None
            if curve is not None:
                point = _alarm_point_at_tau(curve, tau)
                if point is None:
                    # Band edge at tau never dipped below threshold: zero det/fa are measurements,

                    # lead time of an alarm that never fired is not, and stays absent.
                    det = 0.0 if curve.deployed.n_events > 0 else None
                    fa = 0.0 if observed_days else None
                else:
                    det = _absent_if_nan(point.detection_rate)
                    fa = _absent_if_nan(point.false_alarms_per_day)
                    lead = _absent_if_nan(point.median_lead_min)
            out[f'alarm_hypo_det@{tag}{suffix}'] = det
            out[f'alarm_hypo_fa_day@{tag}{suffix}'] = fa
            out[f'alarm_hypo_lead_min@{tag}{suffix}'] = lead

    pooled = alarm.pooled
    out['alarm_hypo_n_events'] = float(pooled.deployed.n_events)
    out['_alarm_observed_days'] = observed_days
    # Beside the columns, table-only: pooled-row note comes from the dataclass, verbatim.
    out['_alarm_pooled_note'] = alarm.pooled_note
    _alarm_columns(pooled, '')
    for h, dd in zip(eh, fc_d):
        curve = alarm.by_d.get(dd)
        out[f'alarm_hypo_n_events@{h}'] = (
            None if curve is None else float(curve.deployed.n_events))
        _alarm_columns(curve, f'@{h}')
    return out


def _infill_baseline(masked_set, cgm: np.ndarray) -> np.ndarray:
    """The infill protocol's own baseline over one window's scored steps.
    ``metrics.protocols.baseline_for`` picks it from the protocol, so a caller cannot pair
    infill with persistence by mistake. ``cgm`` IS the window, starting at step 0."""
    from metrics.protocols import baseline_for
    return baseline_for(masked_set, cgm, 0)


def _infill_fan_columns(scores) -> dict[str, Any]:
    """Infill protocol columns, every one named with its d.
    Point errors come from InfillScores, scored against LINEAR INTERPOLATION between
    bracketing visible readings, never persistence (a strawman against a two-sided span).
    Fan figures come from metrics.scoring over the same collected fan.
    """
    from metrics.protocols import INFILL, column, reachable_d
    from metrics.scoring import (
        coverage_sharpness_by_d, crps_by_d, winkler_by_d,
    )

    point = scores.columns()
    q, true, d, _group = scores.fan()
    out: dict[str, Any] = {}
    if d.size == 0:
        return out

    crps = crps_by_d(q, true, d)
    winkler = winkler_by_d(q, true, d)
    coverage = coverage_sharpness_by_d(q, true, d)
    n90 = _nominal_for(_TAU_LO_IDX)
    for dd in reachable_d(INFILL):
        cs = coverage[n90].by_d.get(dd)
        out[column(INFILL, 'crps_n', dd)] = float(crps.n_by_d.get(dd, 0))
        out[column(INFILL, 'rmse', dd)] = _absent_if_nan(
            point.get(column(INFILL, 'rmse', dd)))
        # InfillScores names the baseline column interp_rmse; log header names it rmse_interp.
        out[column(INFILL, 'rmse_interp', dd)] = _absent_if_nan(
            point.get(column(INFILL, 'interp_rmse', dd)))
        out[column(INFILL, 'crps', dd)] = _absent_if_nan(crps.by_d.get(dd))
        out[column(INFILL, 'winkler90', dd)] = _absent_if_nan(
            winkler[n90].by_d.get(dd))
        out[column(INFILL, 'marginal90_cov', dd)] = _absent_if_nan(
            cs.coverage if cs else None)
        out[column(INFILL, 'marginal90_width_mean', dd)] = _absent_if_nan(
            cs.mean_width if cs else None)
    return out


def _slot_jump_hours(
    logits: torch.Tensor,
    mask_idx: torch.Tensor,
    valid: torch.Tensor,
    adv_per_patch: float,
) -> torch.Tensor:
    """Per-sample inter-SLOT clock-advance deviation, in hours (no-jump witness).
    M slots are an arbitrary masked set; expected advance is per PAIR (mask_idx[j+1]-mask_idx[j]
    patches), pairs touching a padded slot dropped. Returns (hours, has_pair), both (B,).
    has_pair is NOT OPTIONAL: a one-slot row has no pair; the 0/0 clamp hands back 0.0, best value.
    """
    if logits.shape[1] < 2:
        return (logits.new_zeros(logits.shape[0]),
                torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device))
    hours, _ = time_of_day_decode_bins(logits, TIME_PROBE_N_BINS)          # (B, M)
    step = circular_hour_residual(hours[:, 1:], hours[:, :-1])             # (B, M-1)
    expected = (mask_idx[:, 1:] - mask_idx[:, :-1]).to(step.dtype) * adv_per_patch
    pair = (valid[:, 1:] & valid[:, :-1]).to(step.dtype)
    dev = circular_hour_residual(step, expected).abs() * pair
    n_pair = pair.sum(dim=-1)
    return dev.sum(dim=-1) / n_pair.clamp(min=1.0), n_pair > 0


def _slot_cross_window_loss(
    logits_k: torch.Tensor,
    logits_next: torch.Tensor,
    advance_hours: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Paired-window phase-advance penalty at a PER-SAMPLE advance.
    Same form as utils.time_cross_window_consistency_loss, but advance is a (B,) tensor: window
    k's slot 0 is its first masked patch (sampler puts it anywhere), k+1 carries the forecast
    span; gap comes from the shipped per-slot true hours. Exactly 0 when no row is valid.
    """
    pk = torch.softmax(logits_k[:, 0, :], dim=-1)
    pn = torch.softmax(logits_next[:, 0, :], dim=-1)
    rk = time_of_day_resultant(pk, TIME_PROBE_N_BINS)                      # (B, 2)
    rn = time_of_day_resultant(pn, TIME_PROBE_N_BINS)                      # (B, 2)
    dtheta = advance_hours.to(rk.dtype) * (2.0 * math.pi / 24.0)           # (B,)
    cos_d, sin_d = torch.cos(dtheta), torch.sin(dtheta)
    rc = rk[:, 0] * cos_d - rk[:, 1] * sin_d
    rs = rk[:, 0] * sin_d + rk[:, 1] * cos_d
    per = (rc - rn[:, 0]) ** 2 + (rs - rn[:, 1]) ** 2                      # (B,)
    vf = valid.to(per.dtype)
    return (per * vf).sum() / vf.sum().clamp(min=1.0)


def _run_validation(
    model: T1DMAI,
    val_dataset: T1DMDataset,
    norm_stats: dict,
    device: torch.device,
    weighting: KendallGalWeighting,
    bg_hypo_threshold: float = BG_HYPO_THRESHOLD,
    bg_hyper_threshold: float = BG_HYPER_THRESHOLD,
) -> dict[str, Any]:
    """Validation on a fixed patient set, in batches: THREE forwards per batch (objective,
    forecast, infill), each scoring different slots. Forecast needs trailing PREDICTION_PATCHES,
    not the objective forward's own masked set (names defined only against a right-edge zone).
    No pooled masked-BG scalar is emitted: an average over the mask distribution improves for free.
    """
    model.eval()
    totals: dict[str, float] = {'loss_total': 0.0, 'loss_Q': 0.0, 'loss_D': 0.0, 'loss_M': 0.0, 'pinball': 0.0}
    n_samples = 0

    agg: dict[str, float] = {}
    night_agg: dict[str, float] = {}

    n_val = min(len(val_dataset), VALIDATION_N_PATIENTS)

    # Per-window mg/dL fans + truth + anchor for the conformal probe (_conformal_val_probe).
    conf_bands_list: list[np.ndarray] = []
    conf_true_list: list[np.ndarray] = []
    conf_last_list: list[np.ndarray] = []

    # Materialized val samples reused by the counterfactual probe (avoids re-running the simulator).
    val_samples_ordered: list[dict] = []

    # INFILL's accumulator (metrics.protocols) and the generator its spans are drawn from.
    from metrics.protocols import InfillScores
    infill_scores = InfillScores()
    infill_rng = np.random.default_rng(INFILL_PROTOCOL_SEED)
    infill_windows = 0

    # Time-of-day probe, over the WHOLE val set: needs the full residual distribution.

    # tod_* decode SLOT 0, scored vs that slot's own true hour (slot 0 = patch mask_idx[0]).
    tod_adv = _PATCH_HOURS                            # phase advance per patch
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
            targets = batch['targets'].to(device, non_blocking=True).float()   # (B, M, S) mg/dL
            mask_idx = bg_formula['mask_idx'].long()                      # (B, M) padded axis
            slot_valid = bg_formula['valid']                              # (B, M) bool
            anchor_bg = bg_formula['anchor_bg'].float()                   # (B, M) mg/dL
            slot_hour = bg_formula['slot_hour'].float()                   # (B, M) hours

            q_tau_obj, median_obj, time_pred = model(
                patches, attn_mask, anchor_bg, mask_idx, return_time=True)
            B_batch = median_obj.shape[0]

            # f applied to target once inside risk_total_loss; valid keeps padded slots (41.8% of

            # head output/sample) from being supervised against patch 0 behind a plausible anchor.
            loss_total, parts = risk_total_loss(
                q_tau_obj.float(), median_obj.float(), targets, weighting,
                valid=slot_valid, mask_idx=mask_idx,
            )

            totals['loss_total'] += float(loss_total) * B_batch
            totals['loss_Q'] += float(parts.get('loss_Q', float('nan'))) * B_batch
            totals['loss_D'] += float(parts.get('loss_D', float('nan'))) * B_batch
            totals['loss_M'] += float(parts.get('loss_M', float('nan'))) * B_batch
            totals['pinball'] += float(parts.get('pinball', parts.get('loss_Q', float('nan')))) * B_batch

            # Time-of-day probe (diagnostic, never in loss/selection), off the OBJECTIVE forward.

            # Decodes slot 0 to hour+confidence R, scored vs its true hour; always valid, no filter.
            if time_pred is not None:
                hours0, R0 = time_of_day_decode_bins(time_pred[:, 0, :], TIME_PROBE_N_BINS)  # (B,)
                tod_pred_hours.append(hours0.detach().cpu())
                tod_true_hours.append(slot_hour[:, 0].detach().cpu())
                tod_conf_vals.append(R0.detach().cpu())
                # Only rows with a consecutive-slot pair (VALIDATION_N_PATIENTS: 45/1000, -4.5%).
                _jump, _jump_pair = _slot_jump_hours(
                    time_pred, mask_idx, slot_valid, tod_adv)
                if bool(_jump_pair.any()):
                    tod_jump_vals.append(_jump[_jump_pair].detach().cpu())
                # Cross-window no-jump witness: 2nd forward on window k+1 (teacher-forced),

                # measures |clock diff - true gap|; shares n_ctx/padding, runs under k+1's OWN mask.
                if TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
                    _nw = batch.get('next_window')
                    if _nw is not None and bool(_nw['valid'].any()):
                        _nw_patches = _nw['patches'].to(device, non_blocking=True)
                        _nw_attn = _nw['attn_mask'].to(device, non_blocking=True)
                        _nw_anchor = _nw['anchor_bg'].to(device, non_blocking=True).float()
                        _nw_mask_idx = _nw['mask_idx'].to(device, non_blocking=True).long()
                        _nw_hour = _nw['slot_hour'].to(device, non_blocking=True).float()
                        _nw_valid = _nw['valid'].to(device, non_blocking=True)
                        _assert_mask_is_this_window(
                            _nw_patches, _nw_attn, 'val cross-window forward')
                        _, _, _time_pred_next = model(
                            _nw_patches, _nw_attn, _nw_anchor, _nw_mask_idx,
                            return_time=True,
                        )
                        # Slots aren't 1 horizon apart; true gap comes off shipped clocks.

                        # BOTH clock diffs: needs a SECOND wrap; +11.9h/-11.9h is 0.2h, not 23.8h.
                        _adv = circular_hour_residual(_nw_hour[:, 0], slot_hour[:, 0])
                        _hk, _ = time_of_day_decode_bins(time_pred[:, 0, :], TIME_PROBE_N_BINS)
                        _hn, _ = time_of_day_decode_bins(_time_pred_next[:, 0, :], TIME_PROBE_N_BINS)
                        _xwin = circular_hour_residual(
                            circular_hour_residual(_hn, _hk), _adv).abs()         # (B,)
                        tod_xwin_vals.append(_xwin[_nw_valid].detach().cpu())
            n_samples += B_batch

            fc = _forecast_protocol(
                patches, mask_idx, slot_valid, batch['n_context_patches'])
            if fc is None:
                continue
            fc_rows = fc['rows']
            q_tau, median = model(
                fc['patches'], fc['attn_mask'], bg_formula['last_bg'].float()[fc_rows]
                .unsqueeze(1).expand(-1, PREDICTION_PATCHES),
                fc['mask_idx'],
            )
            q_tau = q_tau.float()
            median = median.float()
            B_fc = median.shape[0]
            agg['fc_n'] = agg.get('fc_n', 0.0) + float(B_fc)
            # Every metric reads the forecast subset, so bg_formula is narrowed once, not per use.
            bg_formula = {
                k: (v.index_select(0, fc_rows)
                    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B_batch
                    else v)
                for k, v in bg_formula.items()
            }
            last_bg = bg_formula['last_bg'].float()                       # (B_fc,)
            # risk_total_loss needs the target as (B, P, S) to match ``median``.
            true_bg_full = (
                bg_formula['true_bg_trajectory'][:, :PREDICTION_PATCHES * PATCH_SIZE]
                .float().reshape(-1, PREDICTION_PATCHES, PATCH_SIZE)
            )

            # pred_bg: the SOLE headline forecast.
            pred_bg = _median_to_mgdl(median)                            # (B_fc, P*S)

            # Median roughness (risk space): mean |delta^2| over the flat horizon + last patch.
            m_flat = median.reshape(B_fc, -1)                           # (B_fc, P*S) patch-major
            d2 = m_flat[:, 2:] - 2.0 * m_flat[:, 1:-1] + m_flat[:, :-2]  # (B, P*S-2)
            agg['median_rough_abs_sum'] = agg.get('median_rough_abs_sum', 0.0) + float(d2.abs().sum())
            agg['median_rough_cnt'] = agg.get('median_rough_cnt', 0.0) + float(d2.numel())
            _far0 = (PREDICTION_PATCHES - 1) * PATCH_SIZE - 1        # first delta^2 in last patch
            d2_far = d2[:, _far0:]
            agg['median_rough_far_abs_sum'] = agg.get('median_rough_far_abs_sum', 0.0) + float(d2_far.abs().sum())
            agg['median_rough_far_cnt'] = agg.get('median_rough_far_cnt', 0.0) + float(d2_far.numel())

            # Band edges in mg/dL: f_inv is elementwise, so invert the whole fan ONCE and index.
            conf_full = kovatchev_f_inv(q_tau)                           # (B_fc, P, S, 7) mg/dL
            q_lo = conf_full[..., _TAU_LO_IDX].reshape(B_fc, -1)
            q_hi = conf_full[..., _TAU_HI_IDX].reshape(B_fc, -1)
            q_inner_lo = conf_full[..., _TAU_INNER_LO_IDX].reshape(B_fc, -1)
            q_inner_hi = conf_full[..., _TAU_INNER_HI_IDX].reshape(B_fc, -1)
            # Clinical hypo/hyper ALARM band edges (τ=HYPO/HYPER_ALARM_QUANTILE_TAU lower/upper).
            q_hypo_lo = conf_full[..., _HYPO_BAND_IDX].reshape(B_fc, -1)
            q_hyper_hi = conf_full[..., _HYPER_BAND_IDX].reshape(B_fc, -1)
            q_mgdl = {'lo': q_lo, 'hi': q_hi,
                      'inner_lo': q_inner_lo, 'inner_hi': q_inner_hi,
                      'hypo_lo': q_hypo_lo, 'hyper_hi': q_hyper_hi}

            # Collect the FULL mg/dL fan + truth + anchor for the conformal probe.
            conf_bands_list.append(
                conf_full.reshape(B_fc, -1, N_QUANTILES).detach().cpu().numpy())
            conf_true_list.append(
                true_bg_full.reshape(B_fc, -1).detach().cpu().numpy())
            conf_last_list.append(last_bg.detach().cpu().numpy())

            learn = compute_learning_metrics(
                pred_bg, q_mgdl, bg_formula, PREDICTION_PATCHES,
                hypo_threshold=bg_hypo_threshold,
                hyper_threshold=bg_hyper_threshold,
            )
            for k, v in learn.items():
                agg[k] = agg.get(k, 0.0) + v

            # Interior, two-sided spans: scored vs LINEAR INTERPOLATION, not persistence.

            # Right-edge span the inference builder requires rides along masked and UNSCORED.
            infill = _infill_protocol(
                patches, mask_idx, slot_valid, targets,
                batch['n_context_patches'], norm_stats, infill_rng)
            if infill is not None:
                q_inf, median_inf = model(
                    infill['patches'], infill['attn_mask'],
                    infill['anchor_bg'], infill['mask_idx'])
                bands_inf = kovatchev_f_inv(q_inf.float()).detach().cpu().numpy()
                med_inf = kovatchev_f_inv(median_inf.float()).detach().cpu().numpy()
                for i, (_b, ms, cgm) in enumerate(infill['sets']):
                    slots = ms.scored_slot
                    if not slots.any():
                        continue
                    steps = ms.scored_steps()
                    infill_scores.add(
                        ms.scored_d(),
                        med_inf[i][slots],
                        cgm[steps].reshape(-1, PATCH_SIZE),
                        _infill_baseline(ms, cgm),
                        bands_inf[i][slots],
                    )
                    infill_windows += 1

            # Long-horizon rolling pass (night accum inside), capped at VALIDATION_PROBE_N_PATIENTS;

            # cost is per SAMPLE not batch; cap is on window index, windows stay a leading prefix.
            n_rolls = math.ceil(NIGHT_LONG_HORIZON_HOURS / PREDICTION_HORIZON_HOURS)
            probe_end = min(batch_end, VALIDATION_PROBE_N_PATIENTS)
            if n_rolls > 1 and batch_start < probe_end:
                _accumulate_long_horizon_bg_metrics(
                    model, samples[:probe_end - batch_start], norm_stats, device,
                    n_rolls, agg, night_agg=night_agg,
                )

            # Nocturnal subset of the forecast protocol (no third forward).
            pred_start_hours = bg_formula.get('pred_start_hour')
            if pred_start_hours is not None:
                night_idx = [j for j, h in enumerate(pred_start_hours)
                             if _is_nocturnal(float(h))]
                if night_idx:
                    nidx = torch.tensor(night_idx, device=device, dtype=torch.long)
                    night_bg_formula = {
                        k: (v.index_select(0, nidx)
                            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B_fc
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

    model.train()
    n = max(n_samples, 1)

    result: dict[str, Any] = {
        'val_loss_total': totals['loss_total'] / n,
        'val_loss_Q': totals['loss_Q'] / n,
        'val_loss_D': totals['loss_D'] / n,
        'val_loss_M': totals['loss_M'] / n,
        'val_pinball': totals['pinball'] / n,
        'log_sigma_Q': float(weighting.log_sigma_Q.detach()),
        'log_sigma_D': float(weighting.log_sigma_D.detach()),
    }

    # Point accuracy + clock reliability: bias/precision, p90 tail, gross-error rate, hiconf MAE.
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
        # No-jump witness: mean |advance deviation| in hours; ~0 means clock advances tod_adv/patch.
        if tod_jump_vals:
            result['tod_jump_h'] = float(torch.cat(tod_jump_vals).mean())
        if tod_xwin_vals:
            result['tod_xwin_jump_h'] = float(torch.cat(tod_xwin_vals).mean())

    # What each protocol ran on, so a shrinking sample is visible in the log, not just the metric.
    result['fc_n'] = agg.get('fc_n', 0.0)
    _rcc = agg.get('roll_ctx_cnt', 0.0)
    result['roll_ctx_patches'] = (agg.get('roll_ctx_sum', 0.0) / _rcc) if _rcc > 0 else None
    result['roll_n'] = _rcc
    result['roll_skipped'] = agg.get('roll_skipped', 0.0)
    # Same split over nocturnal samples: night_bg_rmse_* is scored on that subset, so the pair

    # above isn't its denominator; it counts the whole val set, nonzero even if night is empty.
    result['night_roll_n'] = night_agg.get('night_roll_cnt', 0.0)
    result['night_roll_skipped'] = night_agg.get('night_roll_skipped', 0.0)

    for h_min in BG_HORIZONS_MIN:
        cnt = agg.get(f'bg_rmse_{h_min}_cnt', 0.0)
        if cnt > 0:
            result[f'bg_rmse_{h_min}'] = math.sqrt(agg[f'bg_rmse_{h_min}_sq_sum'] / cnt)
            result[f'bg_mae_{h_min}'] = agg[f'bg_mae_{h_min}_abs_sum'] / cnt
        else:
            result[f'bg_rmse_{h_min}'] = None
            result[f'bg_mae_{h_min}'] = None
        # Denominator per horizon: past the single forward, accumulated by the rolling probe

        # over VALIDATION_PROBE_N_PATIENTS windows, not VALIDATION_N_PATIENTS.
        result[f'bg_rmse_{h_min}_n'] = cnt

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
    # A+B is kept beside zones because it's what Clarke literature quotes, not A/B interchangeable.
    for _z in ('A', 'B', 'C', 'D', 'E'):
        result[f'clarke_{_z}_pct'] = 100.0 * agg.get(f'clarke_{_z}', 0.0) / clarke_total
    result['clarke_AB_pct'] = 100.0 * (agg.get('clarke_A', 0.0) + agg.get('clarke_B', 0.0)) / clarke_total

    # Every DTS zone: paper's position is pZA alone measures clinical performance; no dts_AB_pct.
    _dts_fr = dts_grid.dts_zone_fractions(
        {k: agg.get(f'dts_{k}', 0.0) for k in (*dts_grid.ZONE_NAMES, 'total')})
    for _zn, _fr in _dts_fr.items():
        result[f'dts_{_zn}_pct'] = None if _fr is None else 100.0 * _fr

    for h_min in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
        for _z in ('A', 'B', 'C', 'D', 'E'):
            c_cnt = agg.get(f'evalfix_clarke_{_z}@{h_min}_cnt', 0.0)
            result[f'evalfix_clarke_{_z}@{h_min}'] = (
                100.0 * agg.get(f'evalfix_clarke_{_z}@{h_min}', 0.0) / c_cnt
                if c_cnt > 0 else None)
        for _zn in dts_grid.ZONE_NAMES:
            d_cnt = agg.get(f'dts_{_zn}@{h_min}_cnt', 0.0)
            result[f'dts_{_zn}@{h_min}'] = (
                100.0 * agg.get(f'dts_{_zn}@{h_min}', 0.0) / d_cnt
                if d_cnt > 0 else None)
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

    # Excursion-magnitude amplitude: exc_amp_ratio=std(pred)/std(true) at true peak (target 1.0),

    # exc_gain_beta=slope of pred on true (target 1.0); over/under fractions trade off on re-centre.
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

    if conf_bands_list:
        # One decoded mg/dL fan over the forecast protocol, read twice.
        fan_mgdl = np.concatenate(conf_bands_list, axis=0)         # (W, P*S, K)
        fan_true = np.concatenate(conf_true_list, axis=0)          # (W, P*S)

        # Conformal coverage probe (raw-vs-calibrated band coverage on a held-out val split).
        result.update(_conformal_val_probe(
            fan_mgdl, fan_true, np.concatenate(conf_last_list, axis=0)))

        # Fan reshaped onto metrics.scoring's unit: one row per MASKED PATCH, (N,S,K), d+window.

        # Protocol lays exactly one patch at each d=1..PREDICTION_PATCHES/window: every bin fills.
        P, S, K = PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES
        q_fan = fan_mgdl.reshape(-1, P, S, K)
        t_fan = fan_true.reshape(-1, P, S)
        n_win = q_fan.shape[0]
        # OBSERVED DAYS: false-alarm denominator, never a default. One decision per window over

        # its whole zone, so exposure is n_win zones; an ISSUANCE rate at 5-min cadence issues more.
        observed_days = n_win * PREDICTION_HORIZON_HOURS / 24.0
        result.update(_forecast_fan_columns(
            q_fan.reshape(n_win * P, S, K),
            t_fan.reshape(n_win * P, S),
            np.tile(np.arange(1, P + 1, dtype=np.int64), n_win),
            np.repeat(np.arange(n_win, dtype=np.int64), P),
            observed_days,
        ))
    result.update(_infill_fan_columns(infill_scores))
    result['_infill_windows'] = float(infill_windows)

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

    # Two families reach this row: rolled (past single forward) and single-pass under it.

    # _n is the count behind whichever supplied the value; row and denominator match.
    for h_min in BG_HORIZONS_MIN:
        cnt = night_agg.get(f'night_bg_rmse_{h_min}_cnt', 0.0)
        if cnt > 0:
            result[f'night_bg_rmse_{h_min}'] = math.sqrt(
                night_agg[f'night_bg_rmse_{h_min}_sq_sum'] / cnt)
        else:
            cnt = night_agg.get(f'bg_rmse_{h_min}_cnt', 0.0)
            if cnt > 0:
                result[f'night_bg_rmse_{h_min}'] = math.sqrt(
                    night_agg[f'bg_rmse_{h_min}_sq_sum'] / cnt)
            else:
                result[f'night_bg_rmse_{h_min}'] = None
        result[f'night_bg_rmse_{h_min}_n'] = cnt

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

    # Counterfactual dose-response probe (diagnostic), over the same validation samples.
    result.update(_run_counterfactual_probe(
        model, val_dataset, norm_stats, device,
        hypo_threshold=bg_hypo_threshold, hyper_threshold=bg_hyper_threshold,
        samples=val_samples_ordered,
    ))

    return result


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
    """Build a serializable checkpoint dict. The two Kendall-Gal log-σ live off ``model`` (on
    ``weighting``), so they serialize separately as ``weighting_state_dict``."""
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


# (training_config key, config.py constant, resize flag); FFN_DIM/BG_HEAD_HIDDEN handled separately.
_RESUME_ARCH_KNOBS = (
    ('d_model', 'D_MODEL', '--d-model'),
    ('n_layers', 'N_LAYERS', '--layers'),
    ('n_heads', 'N_HEADS', '--heads'),
    ('patch_size', 'PATCH_SIZE', '--patch-size'),
    ('min_context_patches', 'MIN_CONTEXT_PATCHES', '--min-context-patches'),
    ('max_context_patches', 'MAX_CONTEXT_PATCHES', '--max-context-patches'),
)


def _check_resume_architecture(ckpt: dict, path: str) -> None:
    """Refuse to resume a checkpoint whose arch, arch_version, masked-channel policy, mse_alpha
    or normalization stats differ from what this process trains under. config.py is the single
    source; a mismatch names the resize_model.py command that aligns it."""
    import config as _cfg
    tc = ckpt.get('training_config', {})
    if ckpt.get('arch_version') != ARCH_VERSION:
        sys.exit(f"--checkpoint {path}: arch_version {ckpt.get('arch_version')!r} != "
                 f"config.py {ARCH_VERSION!r}")
    policy = checkpoint_masked_channel_policy(ckpt)
    if policy != masked_channel_policy(blind=False):
        sys.exit(f"--checkpoint {path}: masked_channel_policy {policy!r}; train.py trains "
                 f"{masked_channel_policy(blind=False)!r} (train_blind.py owns {policy!r})")
    # The restored log_sigma_D was fitted to the D slot this value defines.
    if tc.get('mse_alpha') is not None and float(tc['mse_alpha']) != float(_cfg.MSE_ALPHA):
        sys.exit(f"--checkpoint {path}: mse_alpha {tc['mse_alpha']} != config.py "
                 f"{_cfg.MSE_ALPHA}")
    flags = []
    for key, const, flag in _RESUME_ARCH_KNOBS:
        want = tc.get(key)
        if want is not None and want != getattr(_cfg, const):
            flags.append(f'{flag} {want}')
    d_model = tc.get('d_model', _cfg.D_MODEL)
    ffn = tc.get('ffn_dim')
    if ffn is not None and ffn != _cfg.FFN_DIM:
        flags.append(f'--ffn-mult {ffn // d_model}')
    head0 = ckpt['model_state_dict'].get('bg_head.0.weight')
    if head0 is not None and head0.shape[0] != _cfg.BG_HEAD_HIDDEN:
        flags.append(f'--bg-head-hidden-mult {head0.shape[0] // d_model}')
    if flags:
        sys.exit(f"--checkpoint {path}: architecture differs from config.py; run\n"
                 f"  python resize_model.py {' '.join(flags)}\nthen resume again")


def _restore_weights(
    path: str,
    model: T1DMAI,
    weighting: KendallGalWeighting,
    norm_stats: dict,
) -> dict:
    """Load a checkpoint's live model weights and Kendall-Gal log-σ. Returns the checkpoint
    dict so the caller can load the EMA shadow once the EMA exists. Optimizer state, step,
    histories and best_val_loss are NOT restored: a resumed run starts its own schedule."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    _check_resume_architecture(ckpt, path)
    if ckpt.get('normalization_stats') != norm_stats:
        sys.exit(f"--checkpoint {path}: normalization_stats differ from {NORM_STATS_FILE}")
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    weighting.load_state_dict(ckpt['weighting_state_dict'])
    return ckpt


# Each log has ONE column list, shared by header and row writer, so a metric can't shift the rest.

# val_record (checkpoint) is a THIRD surface, not built from this list; extended by hand.


def _alarm_curve_taus() -> "list[float]":
    """The hypo alarm curve's operating points: the lower-half τ plus the median. The alarm
    fires off a band edge, so sweeping τ IS the operating curve — a lower τ buys detection with
    false alarms. The ladder comes from ``QUANTILE_LEVELS``, so it cannot name a τ the head does
    not emit."""
    return [t for t in QUANTILE_LEVELS if t <= 0.5]


def _tau_tag(tau: float) -> str:
    """CSV-safe τ suffix (``0.05`` -> ``q05``); two digits keeps the columns sorted."""
    return f"q{int(round(tau * 100)):02d}"


def _train_log_columns() -> "list[tuple[str, int]]":
    """logs/training_log.csv columns as (name, decimals).
    DILATE is one call per span-length bucket, not scale-free in H=L*PATCH_SIZE (shape term
    grows with H, TDI does not), so alpha weights differ per bucket, log_sigma_D absorbs it.
    loss_D_L{L}/n_spans_L{L} ride beside the combined value: comparable only at equal mixture.
    """
    return [
        ('step', 0), ('loss_total', 6), ('loss_ema', 6),
        ('loss_Q', 6), ('loss_D', 6), ('loss_D_shape', 6), ('loss_D_tdi', 6), ('loss_M', 6),
        *[(f'loss_D_L{L}', 6) for L in MASK_SPAN_LENGTHS],
        *[(f'n_spans_L{L}', 3) for L in MASK_SPAN_LENGTHS],
        ('n_masked_mean', 3), ('n_spans_mean', 3),
        ('loss_tod', 6), ('loss_tod_xwin', 6),
        ('log_sigma_Q', 6), ('log_sigma_D', 6),
        ('grad_norm', 6), ('lr_muon', 8), ('lr_adam', 8),
        ('step_time_seconds', 4), ('gpu_memory_mb', 1),
    ]


def _val_log_columns() -> "list[tuple[str, int]]":
    """logs/validation_log.csv columns as (name, decimals).
    Three axes, not interchangeable: BG_HORIZONS_MIN columns come off the ROLLING pass (past
    the 2h zone); eh IS the d axis (protocol masks PREDICTION_PATCHES, patch p is p+1 away).
    Infill d caps at ceil(L/2), scored vs LINEAR INTERPOLATION not persistence; pinball IS loss_Q.
    """
    from metrics.protocols import FORECAST, INFILL, column, reachable_d

    eh = _excursion_bucket_horizons(PREDICTION_PATCHES)
    at = _alarm_curve_taus()
    fc_d = reachable_d(FORECAST)
    assert len(eh) == len(fc_d), (
        f"the forecast protocol's per-patch horizons {eh} must be its d axis "
        f"{fc_d} one-for-one — @30/@60/@90/@120 IS d = 1..{len(fc_d)}"
    )
    inf_d = reachable_d(INFILL)
    return [
        ('step', 0),
        ('val_loss_total', 6), ('val_loss_Q', 6), ('val_loss_D', 6), ('val_loss_M', 6),
        ('train_loss_ema', 6), ('overfit_ratio', 6),
        *[(f'coverage90@{h}', 4) for h in COVERAGE_HORIZONS_MIN],
        *[(f'sign_balance@{h}', 4) for h in COVERAGE_HORIZONS_MIN],
        *[(f'inner50_cov@{h}', 4) for h in COVERAGE_HORIZONS_MIN],
        *[(f'bg_rmse_{h}', 4) for h in BG_HORIZONS_MIN],
        *[(f'bg_mae_{h}', 4) for h in BG_HORIZONS_MIN],
        *[(f'bg_rmse_{h}_n', 4) for h in BG_HORIZONS_MIN],
        *[(f'evalfix_mard@{h}', 4) for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
        ('pred_tir', 4), ('true_tir', 4), ('tir_err', 4),
        ('tbr_err', 4), ('tar_err', 4),
        ('hypo_recall', 4), ('hypo_precision', 4), ('hypo_n_steps', 4),
        ('hyper_recall', 4), ('hyper_precision', 4), ('hyper_n_steps', 4),
        # All 3 CG-EGA verdicts/region: AP+BE+EP=1, so AP/EP alone leave benign share derivable.
        *[(f'cgega_{m}_{r}', 4) for r in ('hypo', 'eu', 'hyper') for m in ('ap', 'be', 'ep')],
        *[(f'evalfix_clarke_{z}@{h}', 4)
          for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN for z in ('A', 'B', 'C', 'D', 'E')],
        *[(f'clarke_{z}_pct', 4) for z in ('A', 'B', 'C', 'D', 'E')],
        ('clarke_AB_pct', 4),
        # DTS Error Grid: every zone + pZA/horizon. No A+B column; paper says presenting one is bad.
        *[(f'dts_{z}_pct', 4) for z in dts_grid.ZONE_NAMES],
        *[(f'dts_{z}@{h}', 4)
          for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN for z in dts_grid.ZONE_NAMES],
        ('roc_rmse', 6), ('roc_corr', 4), ('trend_gain_beta', 4), ('trend_amp_ratio', 4),
        ('bg_curve_corr', 4),
        # Excursion-magnitude amplitude: NET peak deviation vs last_bg.
        ('exc_amp_ratio', 4), ('exc_gain_beta', 4), ('exc_corr', 4),
        ('exc_overshoot_frac', 4), ('exc_undershoot_frac', 4), ('exc_n', 4),
        # Conformal coverage probe (raw vs calibrated at excursion peaks); coverage + width, mg/dL.
        ('conf_cov90_raw', 4), ('conf_cov90_cal', 4),
        ('conf_width_raw', 4), ('conf_width_cal', 4),
        ('conf_hypo_esc_raw', 4), ('conf_hypo_esc_cal', 4), ('conf_n', 4),
        # Median forecast roughness (risk-space mean |delta^2|), pooled over horizon + last patch.
        ('median_roughness', 6), ('median_roughness_far', 6),
        # Counterfactual dose-response probe (diagnostic).
        ('cf_carb_dbg', 4), ('cf_carb_dir', 4), ('cf_insulin_dbg', 4), ('cf_insulin_dir', 4),
        ('cf_carb_monotonic', 4), ('cf_insulin_monotonic', 4),
        ('cf_hypo_rescue', 4), ('cf_hyper_rescue', 4),
        ('cf_n', 4), ('cf_hypo_n', 4), ('cf_hyper_n', 4),
        # Time-of-day probe (point accuracy + clock reliability + no-jumping witness).
        ('tod_mae_h', 4), ('tod_acc_1h', 4), ('tod_acc_2h', 4), ('tod_acc_bin', 4), ('tod_conf', 4),
        ('tod_bias_h', 4), ('tod_std_h', 4), ('tod_p90_h', 4), ('tod_gross_rate', 4),
        ('tod_mae_hiconf', 4),
        ('tod_jump_h', 4), ('tod_xwin_jump_h', 4),
        # Protocol coverage: fc_n counts forecast windows (context-edge masked -> dropped).

        # roll_n/skipped split windows OFFERED (leading VALIDATION_PROBE_N_PATIENTS).

        # night_roll_* mirrors that split over the (smaller) nocturnal subset, for comparable rows.
        ('fc_n', 4), ('roll_ctx_patches', 3), ('roll_n', 4), ('roll_skipped', 4),
        ('night_roll_n', 4), ('night_roll_skipped', 4),
        # Strictly-proper scoring per d (=eh): crps mg/dL; winkler90 = interval score at alpha=.10.

        # sharp90/sharp50 = mean band width, mg/dL; joint_cov90 = SIMULTANEOUS, below marginal.
        *[(f'{fam}@{h}', 4) for fam in FAN_SCORE_FAMILIES for h in eh],
        # Hypo alarm curve, swept over tau: det=fraction alarmed, fa_day=false alarms/day,

        # lead_min=MEDIAN lead (mean tail-dominated); pooled (max over zone) + per-d (own counts).
        ('alarm_hypo_n_events', 4),
        *[(f'alarm_hypo_det@{_tau_tag(t)}', 4) for t in at],
        *[(f'alarm_hypo_fa_day@{_tau_tag(t)}', 4) for t in at],
        *[(f'alarm_hypo_lead_min@{_tau_tag(t)}', 2) for t in at],
        *[(f'alarm_hypo_n_events@{h}', 4) for h in eh],
        *[(f'alarm_hypo_det@{_tau_tag(t)}@{h}', 4) for t in at for h in eh],
        *[(f'alarm_hypo_fa_day@{_tau_tag(t)}@{h}', 4) for t in at for h in eh],
        *[(f'alarm_hypo_lead_min@{_tau_tag(t)}@{h}', 2) for t in at for h in eh],
        # Infill protocol, per d (metrics.protocols.column); rmse_interp is the scored baseline.
        *[(column(INFILL, base, d), 4)
          for base in INFILL_FAMILIES for d in inf_d],
        # Per-horizon excursion buckets.
        *[(f'hypo_recall@{h}', 4) for h in eh],
        *[(f'hypo_precision@{h}', 4) for h in eh],
        *[(f'hypo_n_steps@{h}', 4) for h in eh],
        *[(f'hyper_recall@{h}', 4) for h in eh],
        *[(f'hyper_precision@{h}', 4) for h in eh],
        *[(f'hyper_n_steps@{h}', 4) for h in eh],
        # Nocturnal: each RMSE carries its count; the night subset is small, one window moves it.
        *[(f'night_bg_rmse_{h}', 4) for h in BG_HORIZONS_MIN],
        *[(f'night_bg_rmse_{h}_n', 4) for h in BG_HORIZONS_MIN],
        ('night_hypo_recall', 4), ('night_hypo_precision', 4), ('night_hypo_n_steps', 4),
        ('night_hyper_recall', 4), ('night_hyper_precision', 4), ('night_hyper_n_steps', 4),
        *[(f'night_clarke_A@{h}', 4) for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
        *[(f'night_mard@{h}', 4) for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN],
    ]


def _csv_row(columns: "list[tuple[str, int]]", values: dict[str, Any]) -> list:
    """One CSV row from a column spec and a ``{name: value}`` map. A missing or non-numeric
    value writes an empty cell; bools are checked before ints because ``bool`` subclasses
    ``int``."""
    row: list = []
    for name, decimals in columns:
        v = values.get(name)
        if isinstance(v, bool):
            row.append(v)
        elif isinstance(v, (int, float)):
            row.append(round(v, decimals))
        else:
            row.append('')
    return row


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
    checkpoint: str | None = None,
) -> list[float]:
    """Run the T1DMAI training loop. Returns the per-step total-loss history.

    ``checkpoint``: restore that file's model weights, Kendall-Gal log-σ and EMA shadow, then
    train from step 0 with fresh optimizers, schedule, histories and best_val_loss."""
    # Must run before model/optimizer/dataloader construction, so every RNG draw is reproducible.
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

    # setup_determinism already seeded torch+CUDA when DETERMINISTIC; seed here only otherwise.
    if not DETERMINISTIC:
        torch.manual_seed(master_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(master_seed)

    model = T1DMAI().to(device)

    # The two Kendall-Gal log-sigma live on this separate module, not model, so EMA skips them.
    weighting = KendallGalWeighting().to(device)

    resume_ckpt: dict | None = None
    if checkpoint is not None:
        resume_ckpt = _restore_weights(checkpoint, model, weighting, norm_stats)
        print(f"Restored weights from {checkpoint} (saved at step {resume_ckpt.get('step')})")

    muon_opt, adam_opt = _build_optimizers(
        model, weighting, muon_lr, adam_lr, muon_momentum,
        adam_weight_decay=adam_weight_decay,
    )

    ema: ModelEMA | None = None
    if ema_decay > 0.0:
        ema = ModelEMA(model, decay=ema_decay).to(device)
        print(f"Weight EMA enabled (decay={ema_decay})")
        if resume_ckpt is not None and 'model_ema_state_dict' in resume_ckpt:
            shadow = resume_ckpt['model_ema_state_dict']
            if set(shadow) != set(ema.shadow):
                sys.exit(f"--checkpoint {checkpoint}: EMA shadow keys differ from the model")
            ema.load_state_dict(shadow)
            ema.to(device)
    resume_ckpt = None

    start_step = 0
    loss_history: list[float] = []
    val_history: list[dict] = []
    best_val_loss = float('inf')
    best_val_step = -1
    loss_ema: float | None = None
    consecutive_nan = 0
    prev_val_metrics: dict[str, Any] | None = None

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
        # Always conditioned: the future plan rides on bg_formula_data, inference override path.
        patient_uniform_sample_prob=patient_uniform_sample_prob,
        simulator_warmup_hours=simulator_warmup_hours,
        cache_path=cache_path,
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
        # 2, not 8: prefetch_factor*num_workers batches buffer (~10GB at 8*20=160 shared mem).
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )

    from config import (
        D_MODEL as _CFG_D_MODEL, N_LAYERS as _CFG_N_LAYERS,
        N_HEADS as _CFG_N_HEADS, FFN_DIM as _CFG_FFN_DIM,
    )
    training_config = {
        'arch_version': ARCH_VERSION, 'loss_schema': LOSS_SCHEMA, 'mse_alpha': MSE_ALPHA,
        # The sampler constants the run trained under — the provenance a loader compares against.
        'mask_span_lengths': list(MASK_SPAN_LENGTHS),
        'max_masked_patches': MAX_MASKED_PATCHES,
        'mask_right_edge_quota': MASK_RIGHT_EDGE_QUOTA,
        # What a masked patch withheld; train_blind.py stamps 'blind'; absent key reads as this.
        'masked_channel_policy': masked_channel_policy(blind=False),
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
        'resumed_from': checkpoint,
    }
    with open('logs/resolved_config.json', 'w') as f:
        json.dump(training_config, f, indent=2)

    # Training log CSV — header and row from the one shared column spec.
    _train_columns = _train_log_columns()
    train_log_path = 'logs/training_log.csv'
    # A run always starts fresh, so always write a fresh header.
    train_log_exists = False
    train_log_file = open(train_log_path, 'a' if train_log_exists else 'w', newline='')
    train_log_writer = csv.writer(train_log_file)
    if not train_log_exists:
        train_log_writer.writerow([name for name, _ in _train_columns])

    # Validation log CSV.
    val_log_path = 'logs/validation_log.csv'
    _val_columns = _val_log_columns()
    # A run always starts fresh, so always write a fresh header.
    val_log_exists = False
    val_log_file = open(val_log_path, 'a' if val_log_exists else 'w', newline='')
    val_log_writer = csv.writer(val_log_file)
    if not val_log_exists:
        val_log_writer.writerow([name for name, _ in _val_columns])

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
        # Masked set and everything keyed to it, (B,M) padded axis; targets = raw mg/dL per slot.
        mask_idx = bg_formula['mask_idx'].long()                      # (B, M)
        slot_valid = bg_formula['valid']                              # (B, M) bool
        anchor_bg = bg_formula['anchor_bg'].float()                   # (B, M) mg/dL
        slot_hour = bg_formula['slot_hour'].float()                   # (B, M) hours
        targets = batch['targets'].to(device, non_blocking=True).float()   # (B, M, S)

        # Cross-window probe input (window k+1), probe-only overhead, skipped when penalty is off.
        next_window = None
        if TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
            _nw = batch.get('next_window')
            if _nw is not None:
                next_window = {
                    'patches': _nw['patches'].to(device, non_blocking=True),
                    'attn_mask': _nw['attn_mask'].to(device, non_blocking=True),
                    'anchor_bg': _nw['anchor_bg'].to(device, non_blocking=True).float(),
                    'mask_idx': _nw['mask_idx'].to(device, non_blocking=True).long(),
                    'slot_hour': _nw['slot_hour'].to(device, non_blocking=True).float(),
                    'valid': _nw['valid'].to(device, non_blocking=True),
                }

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
                # Restored weights != moments' basis; clear state, else stale moments re-diverge.
                muon_opt.state.clear()
                adam_opt.state.clear()
                print(f"  [RECOVERY] {consecutive_nan} consecutive {reason} — restored model from EMA shadow weights (optimizer state cleared)")
                consecutive_nan = 0
                return True
            return False

        def _skip_nonfinite_step(reason: str) -> None:
            """Skip backward + optimizer for a non-finite forward/loss/backward and decay the
            optimizer moments by ½, never poisoning state. A non-finite median/cost PROPAGATES
            to the loss rather than tripping a deep assert, so a NaN here is expected and must
            route through this guard."""
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

        # Wrapped so a non-finite loss or exception (CUDA fault from NaN) routes to skip/restore.
        try:
            # Forward (fp32-native — no autocast).
            q_tau, median, time_pred = model(
                patches, attn_mask, anchor_bg, mask_idx, return_time=True)
            q_tau = q_tau.float()
            median = median.float()

            # f applied to target once in the loss; valid discards padded slots (gather patch 0);

            # mask_idx groups slots into spans for per-span DILATE buckets and spline nodes.
            loss_total, parts = risk_total_loss(
                q_tau, median, targets, weighting,
                valid=slot_valid, mask_idx=mask_idx,
            )

            # TOD probe loss, on BACKWARD tensor only; loss_total/selection stay on BG accuracy.

            # Target: TRUE hour, not pred_start_hour+.5j (off (mask_idx[j]-n_ctx-j)*.5h).

            # Padded slots are dropped; loss_tod is the only witness, nothing gates on it.
            _tod_extra = loss_total.new_zeros(())
            _tod_loss_val = float('nan')    # per-slot CE (logged as loss_tod)
            _tod_xwin_val = float('nan')    # cross-window penalty alone (logged as loss_tod_xwin)
            if time_pred is not None:
                _tod_ce = time_of_day_bin_ce(
                    time_pred[slot_valid], slot_hour[slot_valid],
                    TIME_PROBE_N_BINS, TIME_PROBE_LABEL_SMOOTH_BINS
                )
                _tod_loss_val = float(_tod_ce.detach())
                _tod_loss = _tod_ce
                # Cross-window phase-advance penalty: 2nd forward on k+1 couples two INDEPENDENT

                # clocks; advance is per SAMPLE (k+1=forecast span, k=wherever sampler put slot 0).

                # Rides _tod_extra only, masked+subsampled; inside try, so a non-finite propagates.
                if (next_window is not None
                        and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0
                        and bool(next_window['valid'].any())):
                    B_nw = next_window['patches'].shape[0]
                    n_sub = (B_nw if TIME_PROBE_CROSS_WINDOW_FRACTION >= 1.0
                             else max(1, math.ceil(TIME_PROBE_CROSS_WINDOW_FRACTION * B_nw)))
                    nw_valid_s = next_window['valid'][:n_sub]
                    if bool(nw_valid_s.any()):
                        # k+1's own mask, not k's: share n_ctx (mask SHAPE), not the masked set.

                        # This grad reaches trunk: wrong mask trains it to read withheld patches.
                        nw_mask = next_window['attn_mask'][:n_sub]
                        _assert_mask_is_this_window(
                            next_window['patches'][:n_sub], nw_mask,
                            'train cross-window forward')
                        _, _, time_pred_next = model(
                            next_window['patches'][:n_sub], nw_mask,
                            next_window['anchor_bg'][:n_sub],
                            next_window['mask_idx'][:n_sub], return_time=True,
                        )
                        _tod_adv = circular_hour_residual(
                            next_window['slot_hour'][:n_sub, 0], slot_hour[:n_sub, 0])
                        _tod_xwin = _slot_cross_window_loss(
                            time_pred[:n_sub], time_pred_next, _tod_adv, nw_valid_s,
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
            loss_m = float(parts.get('loss_M', float('nan')))
            log_sigma_q = float(parts.get('log_sigma_Q', float('nan')))
            log_sigma_d = float(parts.get('log_sigma_D', float('nan')))
            loss_tod = _tod_loss_val
            loss_tod_xwin = _tod_xwin_val

            print(
                f"Step {step:>6}/{total_steps} | "
                f"Loss: {loss_val:.4f} (ema={loss_ema:.4f}) | "
                f"L_Q: {loss_q:.4f}  L_D: {loss_d:.4f} "
                f"(sh={loss_d_shape:.4f} tdi={loss_d_tdi:.4f})  L_M: {loss_m:.4f} | "
                f"logσ: Q={log_sigma_q:+.4f} D={log_sigma_d:+.4f} | "
                f"L_tod: {loss_tod:.4f} (xwin {loss_tod_xwin:.4f}) | "
                f"Grad: {grad_norm_val:.3f} | "
                f"LR_muon: {cur_lr_muon:.6f} | LR_adam: {cur_lr_adam:.6f} | "
                f"Time: {step_time:.2f}s"
            )

            train_log_writer.writerow(_csv_row(_train_columns, {
                'step': step,
                'loss_total': loss_val, 'loss_ema': loss_ema,
                'loss_Q': loss_q, 'loss_D': loss_d,
                'loss_D_shape': loss_d_shape, 'loss_D_tdi': loss_d_tdi, 'loss_M': loss_m,
                # Per-bucket DILATE + span histogram: Q:D balance moves with mixture, even pinned.
                **{k: float(v) for k, v in parts.items()
                   if k.startswith('loss_D_L') or k.startswith('n_spans_L')},
                'n_masked_mean': float(parts.get('n_masked_mean', float('nan'))),
                'n_spans_mean': float(parts.get('n_spans_mean', float('nan'))),
                'loss_tod': loss_tod, 'loss_tod_xwin': loss_tod_xwin,
                'log_sigma_Q': log_sigma_q, 'log_sigma_D': log_sigma_d,
                'grad_norm': grad_norm_val,
                'lr_muon': cur_lr_muon, 'lr_adam': cur_lr_adam,
                'step_time_seconds': step_time, 'gpu_memory_mb': gpu_mb,
            }))
            train_log_file.flush()

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

            val_total = val_metrics['val_loss_total']
            train_ema = loss_ema if loss_ema is not None else loss_val
            overfit_ratio = 1.0 / (1.0 + math.exp(-(val_total - train_ema)))
            val_metrics['train_loss_ema'] = train_ema
            val_metrics['overfit_ratio'] = overfit_ratio

            print()
            print(_render_validation_table(step, val_metrics, prev_val_metrics))
            print()
            prev_val_metrics = dict(val_metrics)

            val_log_writer.writerow(
                _csv_row(_val_columns, {**val_metrics, 'step': step}))
            val_log_file.flush()

            def _r(x: float | None, n: int = 4) -> float | None:
                return round(x, n) if isinstance(x, (int, float)) else None

            # THIRD SURFACE: in every checkpoint's val_history, NOT built from _val_log_columns().

            # Keeps native ints where CSV rounds; extended BY HAND alongside every CSV addition.
            val_record = {
                'step': step,
                'val_loss_total': round(val_total, 6),
                'val_loss_Q': round(val_metrics['val_loss_Q'], 6),
                'val_loss_D': round(val_metrics['val_loss_D'], 6),
                'val_loss_M': round(val_metrics['val_loss_M'], 6),
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
                # All three per region; they sum to 1 by construction.
                **{f'cgega_{m}_{r}': _r(val_metrics.get(f'cgega_{m}_{r}'))
                   for r in ('hypo', 'eu', 'hyper') for m in ('ap', 'be', 'ep')},
                **{f'clarke_{_z}_pct': _r(val_metrics.get(f'clarke_{_z}_pct'))
                   for _z in ('A', 'B', 'C', 'D', 'E')},
                'clarke_AB_pct': _r(val_metrics.get('clarke_AB_pct')),
                **{f'dts_{_z}_pct': _r(val_metrics.get(f'dts_{_z}_pct'))
                   for _z in dts_grid.ZONE_NAMES},
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
                'conf_width_raw': _r(val_metrics.get('conf_width_raw')),
                'conf_width_cal': _r(val_metrics.get('conf_width_cal')),
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
                # How much of the val set each protocol saw.
                'fc_n': val_metrics.get('fc_n'),
                'roll_ctx_patches': _r(val_metrics.get('roll_ctx_patches'), 3),
                'roll_n': val_metrics.get('roll_n'),
                'roll_skipped': val_metrics.get('roll_skipped'),
                'night_roll_n': val_metrics.get('night_roll_n'),
                'night_roll_skipped': val_metrics.get('night_roll_skipped'),
                'tbr_err': _r(val_metrics.get('tbr_err')),
                'tar_err': _r(val_metrics.get('tar_err')),
            }
            for h in BG_HORIZONS_MIN:
                val_record[f'bg_rmse_{h}'] = _r(val_metrics.get(f'bg_rmse_{h}'))
                val_record[f'bg_mae_{h}'] = _r(val_metrics.get(f'bg_mae_{h}'))
                # Denominator travels with the figure; this record is the only copy once logs/ dies.
                val_record[f'bg_rmse_{h}_n'] = val_metrics.get(f'bg_rmse_{h}_n')
            for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
                val_record[f'evalfix_clarke_A@{h}'] = _r(val_metrics.get(f'evalfix_clarke_A@{h}'))
                val_record[f'dts_a@{h}'] = _r(val_metrics.get(f'dts_a@{h}'))

            # Protocol namespaces + d axes from metrics.protocols, same source as _val_log_columns.
            from metrics import protocols as _protocols

            # Families the CSV carried and this didn't: 24 excursion buckets, 12+6 nocturnal.
            _eh_rec = _excursion_bucket_horizons(PREDICTION_PATCHES)
            for h in _eh_rec:
                for _k in ('hypo_recall', 'hypo_precision', 'hypo_n_steps',
                           'hyper_recall', 'hyper_precision', 'hyper_n_steps'):
                    val_record[f'{_k}@{h}'] = _r(val_metrics.get(f'{_k}@{h}'))
            for h in BG_HORIZONS_MIN:
                val_record[f'night_bg_rmse_{h}'] = _r(val_metrics.get(f'night_bg_rmse_{h}'))
                val_record[f'night_bg_rmse_{h}_n'] = val_metrics.get(f'night_bg_rmse_{h}_n')
            for _k in ('night_hypo_recall', 'night_hypo_precision', 'night_hypo_n_steps',
                       'night_hyper_recall', 'night_hyper_precision', 'night_hyper_n_steps'):
                val_record[_k] = _r(val_metrics.get(_k))
            for h in EVALFIX_CLARKE_MARD_HORIZONS_MIN:
                val_record[f'night_clarke_A@{h}'] = _r(val_metrics.get(f'night_clarke_A@{h}'))
                val_record[f'night_mard@{h}'] = _r(val_metrics.get(f'night_mard@{h}'))

            # eh is the forecast protocol's d=1..PREDICTION_PATCHES one-sided; infill axis/names

            # come from metrics.protocols; nothing pooled over d reads as a selection scalar.
            for h in _eh_rec:
                for _k in FAN_SCORE_FAMILIES:
                    val_record[f'{_k}@{h}'] = _r(val_metrics.get(f'{_k}@{h}'))
            # Alarm twice: pooled operating point (deployed decision) + per-d curves under it.

            # Pooled score is a max over the forecast zone, doesn't move when one d stops detecting.
            val_record['alarm_hypo_n_events'] = val_metrics.get('alarm_hypo_n_events')
            for _sfx in ('', *[f'@{h}' for h in _eh_rec]):
                for _t in _alarm_curve_taus():
                    _tg = _tau_tag(_t)
                    val_record[f'alarm_hypo_det@{_tg}{_sfx}'] = _r(
                        val_metrics.get(f'alarm_hypo_det@{_tg}{_sfx}'))
                    val_record[f'alarm_hypo_fa_day@{_tg}{_sfx}'] = _r(
                        val_metrics.get(f'alarm_hypo_fa_day@{_tg}{_sfx}'))
                    val_record[f'alarm_hypo_lead_min@{_tg}{_sfx}'] = _r(
                        val_metrics.get(f'alarm_hypo_lead_min@{_tg}{_sfx}'), 2)
            for h in _eh_rec:
                val_record[f'alarm_hypo_n_events@{h}'] = val_metrics.get(
                    f'alarm_hypo_n_events@{h}')
            for _d in _protocols.reachable_d(_protocols.INFILL):
                for _k in INFILL_FAMILIES:
                    _c = _protocols.column(_protocols.INFILL, _k, _d)
                    val_record[_c] = _r(val_metrics.get(_c))
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
        # The final step always writes a checkpoint.
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
    """argparse.ArgumentParser that prints the full --help on any error. Prefix abbreviation
    is OFF: every flag is written in full."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault('allow_abbrev', False)
        super().__init__(*args, **kwargs)

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
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint whose model weights, Kendall-Gal log-σ and EMA shadow '
                             'start this run; optimizers, schedule, histories and best_val_loss '
                             'start fresh. config.py must match its architecture.')
    args = parser.parse_args()

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
        'checkpoint': None,
    }
    sources = {k: 'config.py' for k in resolved}

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
        'checkpoint': args.checkpoint,
    }
    for key, cli_val in cli_map.items():
        if cli_val is not None:
            resolved[key] = cli_val
            sources[key] = 'CLI'

    rows = [(key, str(value), sources[key]) for key, value in resolved.items()]
    rows.append(('prediction_patches', str(PREDICTION_PATCHES), 'derived'))
    rows.append(('arch_version', str(ARCH_VERSION), 'config.py'))
    rows.append(('loss_schema', str(LOSS_SCHEMA), 'config.py'))
    rows.append(('mse_alpha', str(MSE_ALPHA), 'config.py'))
    # Read back off config: what the run trains with is what config published + checkpoint records.
    rows.append(('mask_span_lengths', str(MASK_SPAN_LENGTHS), 'config.py'))
    rows.append(('max_masked_patches', str(MAX_MASKED_PATCHES), 'config.py'))
    rows.append(('mask_right_edge_quota', str(MASK_RIGHT_EDGE_QUOTA), 'config.py'))
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
        checkpoint=resolved['checkpoint'],
    )
