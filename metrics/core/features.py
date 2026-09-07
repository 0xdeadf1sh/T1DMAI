"""Model-input bridge: Segment -> the normalized ``N_INPUT_FEATURES`` stack
``[bg_absolute, carbs, insulin, exercise_equiv, bg_masked]``.

bg (feat 0) is Kovatchev risk space — ``kovatchev_f`` BEFORE the z-score; carb, insulin and exercise are log1p+z.
``bg_masked`` is 0.0 throughout: a Segment records what happened, and the masked set is chosen downstream.

The carb/insulin channels are the simulator's absorption/action CURVES, so raw events are convolved with
analytic kernels rebuilt from the simulator's constants (``scratch/kernel_match.py``: r=0.94 carb, 0.99 bolus
against the simulator's own channels):

    carb     mean meal mixture  (fast/med/slow gamma + protein/fat tail)
    insulin  gamma k=3, θ=25    (rapid bolus action)
    exercise gamma k=3, θ=15    (carbohydrate-equivalent disposal)

Insulin is one rapid-delivery series — bolus IU plus CSII basal at IU/h → IU/step; a long-acting analogue
arriving as a 24 h-spread rate is approximated as rapid.

Exercise (feat 3) is g/step carbohydrate-EQUIVALENT disposal, on carb's log1p+z transform, never an intensity.
A source with no activity record writes explicit zeros: an unwritten sparse column sits at z = 0, a phantom
dose, while no session is ``normalize(log1p(0))``, z = -0.1387 under the balanced pool.

``EXERCISE_KERNEL`` is the unit-area shape of ONE announced session, for ``metrics/whatif.py``'s point-event
counterfactual. ``segment_to_channels`` never convolves it — a Segment's ``exercise`` is already per-step.
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
    exercise_curve, EXERCISE_DURATION_MEAN_MIN,
)
from config import PATCH_SIZE, N_INPUT_FEATURES
from data import BG_MASKED_FEAT
from normalization import CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS
from utils import kovatchev_f_np
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

from .schema import Segment, GRID_MIN

# Truncation horizon, min, renormalized: holds 0.99314 of the meal mixture and 0.99616 of the bolus.
_CARB_KERNEL_MIN = 240
_BOLUS_KERNEL_MIN = 240
_EXERCISE_KERNEL_MIN = 240


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


def _exercise_kernel() -> np.ndarray:
    """Unit-area shape of one mean-length session: ``T1DMSIM.simulator.exercise_curve`` (SPEC §5).

    Ramp, plateau over ``EXERCISE_DURATION_MEAN_MIN``, 90-min tail; padded or cut to
    ``_EXERCISE_KERNEL_MIN`` MINUTES and renormalized. Shape only — the caller supplies the grams.
    """
    _n = _EXERCISE_KERNEL_MIN // DT_MINUTES
    k = np.asarray(exercise_curve(1.0, float(EXERCISE_DURATION_MEAN_MIN)), dtype=np.float64)
    k = np.concatenate([k, np.zeros(max(0, _n - len(k)))])[:_n]
    return k / k.sum()


CARB_KERNEL = _carb_kernel()
BOLUS_KERNEL = _bolus_kernel()
EXERCISE_KERNEL = _exercise_kernel()


def _convolve(amounts: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution of per-step event amounts with a unit-area kernel."""
    n = len(amounts)
    out = np.zeros(n, dtype=np.float64)
    for i in np.nonzero(amounts)[0]:
        end = min(n, i + len(kernel))
        out[i:end] += amounts[i] * kernel[:end - i]
    return out


def segment_to_channels(seg: Segment) -> dict[str, np.ndarray]:
    """Raw events -> ``carb`` (g/step absorption), ``insulin`` (IU/step action), ``exercise`` (g/step disposal).

    A Segment carrying pre-resolved ``carb_curve`` / ``insulin_curve`` short-circuits the kernels and returns
    those as-is. ``exercise`` passes through un-convolved on BOTH paths — it is already per-step, and both must
    carry it or the feature stack's raw-column lookup has no feat 3.
    """
    if seg.carb_curve is not None:
        assert seg.insulin_curve is not None, "carb_curve without insulin_curve"
        return {'carb': np.asarray(seg.carb_curve, dtype=np.float64),
                'insulin': np.asarray(seg.insulin_curve, dtype=np.float64),
                'exercise': np.asarray(seg.exercise, dtype=np.float64)}
    carb = _convolve(seg.carb_grams, CARB_KERNEL)
    rapid_delivery = seg.bolus_units + seg.basal_rate * (GRID_MIN / 60.0)
    insulin = _convolve(rapid_delivery, BOLUS_KERNEL)
    return {'carb': carb, 'insulin': insulin,
            'exercise': np.asarray(seg.exercise, dtype=np.float64)}


def build_feature_stack(seg: Segment, stats: dict[str, dict[str, float]]) -> np.ndarray:
    """The normalized ``(N, F)`` input stack for a whole Segment.

    Per ``stats``: bg (feat 0) through the Kovatchev transform BEFORE the z-score (``RISK_SPACE_CHANNELS``),
    carb/insulin/exercise through log1p (``SPARSE_LOG1P_CHANNELS``). Exercise (feat 3) is written explicitly
    even when zero — an unwritten column sits at z = 0, a phantom dose, not ``normalize(log1p(0))``.
    Feat ``BG_MASKED_FEAT`` is the announcement bit: no statistics, no z-score, 0.0 throughout, since every
    step of a Segment is OBSERVED; the masked set is written downstream.
    """
    n = len(seg)
    ch = segment_to_channels(seg)

    # raw post-noise, mirroring ``data._build_sample``: bg to the physical range, the sparse three floored
    # at 0, no smoothing
    bg = np.clip(seg.cgm, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)
    carb = np.clip(ch['carb'], 0.0, None).astype(np.float64)
    insulin = np.clip(ch['insulin'], 0.0, None).astype(np.float64)
    exercise = np.clip(ch['exercise'], 0.0, None).astype(np.float64)

    feats = np.zeros((n, N_INPUT_FEATURES), dtype=np.float32)
    raw = {0: bg, 1: carb, 2: insulin, 3: exercise}
    # every normalized column must be written — an unwritten one is a silent z = 0.
    # Normalized channels are 0..BG_MASKED_FEAT-1, mask bit above: the stack is one column wider.
    assert len(CHANNEL_NAMES) == len(raw) == BG_MASKED_FEAT < N_INPUT_FEATURES, (
        f"{len(CHANNEL_NAMES)} CHANNEL_NAMES, {len(raw)} raw columns, "
        f"BG_MASKED_FEAT {BG_MASKED_FEAT}, N_INPUT_FEATURES {N_INPUT_FEATURES}"
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
    """RAW CGM (mg/dL), bg-clamped — the truth every metric and anchor is scored against.

    No smoothing despite the name, kept for its call sites: the model consumes raw post-noise signals.
    """
    return np.clip(np.asarray(cgm, dtype=np.float64), BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)


def context_window(feats: np.ndarray, pred_start: int, n_ctx_patches: int):
    """Slice the ``n_ctx_patches`` patches ending at ``pred_start`` as (P, S, F)."""
    import torch
    ctx_steps = n_ctx_patches * PATCH_SIZE
    block = feats[pred_start - ctx_steps:pred_start]
    return torch.from_numpy(block.reshape(n_ctx_patches, PATCH_SIZE, N_INPUT_FEATURES).copy())
