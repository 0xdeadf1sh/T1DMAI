"""
Model-input bridge — converts a real-data Segment into the model's normalized
3-feature input stack ``[bg_absolute, carbs, insulin]``.

The model's carb/insulin channels are the simulator's absorption/action *curves*,
so raw events must be convolved with simulator-matched kernels (validated in
``scratch/kernel_match.py`` to reproduce the simulator's channels at r=0.94 carb,
0.99 bolus).  The kernels here are the *analytic* forms rebuilt from the
simulator's documented constants (reproducible, no dependence on a captured run):

    carb    mean meal mixture  (fast/med/slow gamma + protein/fat tail)
    insulin gamma k=3, θ=25    (the rapid bolus-action kernel)

Real insulin is fed as one rapid-delivery series — bolus IU plus CSII basal
converted from rate (IU/h → IU/step) — convolved with the rapid kernel.  (MDI
long-acting, which the Shanghai adapter already folded into ``basal_rate`` as a
24-h-spread rate, is approximated as rapid here; a secondary-cohort caveat.)

The risk-space redesign dropped ``bg_delta``, the IS/HGO latent channels, and the
four temporal sin/cos features from the input stack entirely (they were never
observable in real data / the model no longer consumes them), so this bridge emits
exactly the three signal channels ``CHANNEL_NAMES`` retains.  bg (feat 0) is a
Kovatchev risk-space channel (``kovatchev_f`` applied BEFORE the z-score); carb and
insulin keep the log1p+z transform.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from T1DMSIM.simulator import (
    gamma_curve, DT_MINUTES, BOLUS_GAMMA_K, BOLUS_GAMMA_THETA,
    MIXED_MEAL_FAST_K_RANGE, MIXED_MEAL_FAST_THETA_RANGE,
    MIXED_MEAL_MED_K_RANGE, MIXED_MEAL_MED_THETA_RANGE,
    MIXED_MEAL_SLOW_K_RANGE, MIXED_MEAL_SLOW_THETA_RANGE,
    MIXED_MEAL_MED_WEIGHT_BASE, SLOW_CARB_PREFERENCE_BASE,
    PROTEIN_FAT_GAMMA_K, PROTEIN_FAT_GAMMA_THETA, PROTEIN_FAT_FRACTION_OF_CARBS,
)
from config import PATCH_SIZE, N_INPUT_FEATURES
from normalization import CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS
from utils import kovatchev_f_np
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

from .schema import Segment, GRID_MIN

_CARB_KERNEL_MIN = 240
_BOLUS_KERNEL_MIN = 240


def _carb_kernel() -> np.ndarray:
    """Unit-area mean meal mixture: type-weighted gammas + protein/fat tail."""
    fast = (np.mean(MIXED_MEAL_FAST_K_RANGE), np.mean(MIXED_MEAL_FAST_THETA_RANGE))
    med = (np.mean(MIXED_MEAL_MED_K_RANGE), np.mean(MIXED_MEAL_MED_THETA_RANGE))
    slow = (np.mean(MIXED_MEAL_SLOW_K_RANGE), np.mean(MIXED_MEAL_SLOW_THETA_RANGE))
    w = np.array([1.0 - SLOW_CARB_PREFERENCE_BASE, MIXED_MEAL_MED_WEIGHT_BASE,
                  SLOW_CARB_PREFERENCE_BASE])
    w = w / w.sum()
    k = np.zeros(_CARB_KERNEL_MIN // DT_MINUTES)
    for (kk, th), wt in zip((fast, med, slow), w):
        c = gamma_curve(wt, kk, th, _CARB_KERNEL_MIN)
        k[:len(c)] += c[:len(k)]
    pf = gamma_curve(PROTEIN_FAT_FRACTION_OF_CARBS, PROTEIN_FAT_GAMMA_K,
                     PROTEIN_FAT_GAMMA_THETA, _CARB_KERNEL_MIN)
    k[:len(pf)] += pf[:len(k)]
    return k / k.sum()


def _bolus_kernel() -> np.ndarray:
    k = gamma_curve(1.0, BOLUS_GAMMA_K, BOLUS_GAMMA_THETA, _BOLUS_KERNEL_MIN)
    return k / k.sum()


CARB_KERNEL = _carb_kernel()
BOLUS_KERNEL = _bolus_kernel()


def _convolve(amounts: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution of per-step event amounts with a unit-area kernel."""
    n = len(amounts)
    out = np.zeros(n, dtype=np.float64)
    for i in np.nonzero(amounts)[0]:
        end = min(n, i + len(kernel))
        out[i:end] += amounts[i] * kernel[:end - i]
    return out


def segment_to_channels(seg: Segment) -> dict[str, np.ndarray]:
    """Convert a Segment's raw events into the model's absorption/action channels.

    Returns dict with ``carb`` (g/step absorption) and ``insulin`` (IU/step action).
    """
    carb = _convolve(seg.carb_grams, CARB_KERNEL)
    rapid_delivery = seg.bolus_units + seg.basal_rate * (GRID_MIN / 60.0)
    insulin = _convolve(rapid_delivery, BOLUS_KERNEL)
    return {'carb': carb, 'insulin': insulin}


def build_feature_stack(seg: Segment, stats: dict[str, dict[str, float]]) -> np.ndarray:
    """Build the normalized (N, 3) input stack for a whole Segment.

    Channels 0-2 (bg, carb, insulin — the trimmed ``CHANNEL_NAMES``) are
    normalized per ``stats``: bg (feat 0) through the Kovatchev risk transform
    BEFORE the z-score (``RISK_SPACE_CHANNELS``), carb/insulin through log1p
    (``SPARSE_LOG1P_CHANNELS``).  The redesign removed ``bg_delta``, the IS/HGO
    latents, and the temporal sin/cos features from the stack, so there is nothing
    to mean-impute and no temporal tail.
    """
    n = len(seg)
    ch = segment_to_channels(seg)

    # The model consumes RAW post-noise signals: every signal channel (bg feat0,
    # carb feat1, insulin feat2) is used raw BEFORE normalization, mirroring
    # ``data._build_sample`` (same clamps — bg to the physical BG range, the sparse
    # carb/insulin floored at 0; no FIR smoothing).
    bg = np.clip(seg.cgm, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)
    carb = np.clip(ch['carb'], 0.0, None).astype(np.float64)
    insulin = np.clip(ch['insulin'], 0.0, None).astype(np.float64)

    feats = np.zeros((n, N_INPUT_FEATURES), dtype=np.float32)
    raw = {0: bg, 1: carb, 2: insulin}
    assert list(CHANNEL_NAMES) == ['bg_absolute', 'carb_intake', 'insulin_combined'], (
        f"unexpected CHANNEL_NAMES {list(CHANNEL_NAMES)}"
    )
    for c, name in enumerate(CHANNEL_NAMES):
        col = raw[c]
        if name in RISK_SPACE_CHANNELS:
            col = kovatchev_f_np(col)
        elif name in SPARSE_LOG1P_CHANNELS:
            col = np.log1p(np.maximum(col, 0.0))
        feats[:, c] = (col - stats[name]['mean']) / (stats[name]['std'] + 1e-8)
    return feats


def smoothed_cgm(cgm: np.ndarray) -> np.ndarray:
    """RAW CGM (mg/dL), bg-clamped — the comparison ground truth for every metric.

    The model consumes raw post-noise signals, so the truth a forecast is scored
    against (and the persistence/last-context anchors derived from it) is the raw
    CGM, physical-range clamped, that ``build_feature_stack`` feeds the model. Name
    kept for its many call-sites; no FIR smoothing is applied.
    """
    return np.clip(np.asarray(cgm, dtype=np.float64), BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)


def context_window(feats: np.ndarray, pred_start: int, n_ctx_patches: int):
    """Slice the ``n_ctx_patches`` patches ending at ``pred_start`` as (P, S, F)."""
    import torch
    ctx_steps = n_ctx_patches * PATCH_SIZE
    block = feats[pred_start - ctx_steps:pred_start]
    return torch.from_numpy(block.reshape(n_ctx_patches, PATCH_SIZE, N_INPUT_FEATURES).copy())
