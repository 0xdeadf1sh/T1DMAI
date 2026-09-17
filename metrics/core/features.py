"""Model-input bridge: Segment -> normalized (N, N_INPUT_FEATURES) stack, [bg_absolute, carbs,
insulin, exercise_equiv, bg_masked]. carb/insulin are the simulator's absorption/action CURVES: raw
events convolved with kernels rebuilt from simulator constants: the mean-discipline meal GI and a
5 U aspart bolus. Insulin combines bolus IU + basal IU/h into one rapid series;
a 24h long-acting analogue is approximated as rapid. EXERCISE_KERNEL is for whatif.py only."""
from __future__ import annotations

from datetime import datetime

import numpy as np

from T1DMSIM.simulator import (
    gamma_curve, gi_gamma_params, DT_MINUTES, BOLUS_GAMMA_K, BOLUS_GAMMA_THETA, BOLUS_DIA_BASE_HOURS,
    MEAL_GI_MEAN_MAX, MEAL_GI_DISCIPLINE_SPAN,
    EXERCISE_GAMMA_K, EXERCISE_GAMMA_THETA,
)
from config import PATCH_SIZE, N_INPUT_FEATURES
from data import BG_MASKED_FEAT
from normalization import CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS
from utils import kovatchev_f_np
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

from .schema import Segment, GRID_MIN

# Truncation, min; the simulator's own curve lengths, so no kernel is cut short.
_MEAN_MEAL_GI = MEAL_GI_MEAN_MAX - 0.5 * MEAL_GI_DISCIPLINE_SPAN
_BOLUS_KERNEL_MIN = BOLUS_DIA_BASE_HOURS * 60.0
_EXERCISE_KERNEL_MIN = 240


def _carb_kernel() -> np.ndarray:
    """Unit-area meal at the mean-discipline patient's GI."""
    k = gamma_curve(1.0, *gi_gamma_params(_MEAN_MEAL_GI))
    return k / k.sum()


def _bolus_kernel() -> np.ndarray:
    k = gamma_curve(1.0, BOLUS_GAMMA_K, BOLUS_GAMMA_THETA, _BOLUS_KERNEL_MIN)
    return k / k.sum()


def _exercise_kernel() -> np.ndarray:
    """Unit-area exercise gamma (k=3, θ=15), truncated at 240 min and renormalized.

    Shape only; caller supplies grams. NOT ``CARB_KERNEL``: 0.854 of its mass inside 2 h vs 0.986.
    """
    k = gamma_curve(1.0, EXERCISE_GAMMA_K, EXERCISE_GAMMA_THETA, _EXERCISE_KERNEL_MIN)
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
    """Raw events -> carb (g/step absorption), insulin (IU/step action), exercise (g/step disposal).
    A Segment with pre-resolved carb_curve/insulin_curve short-circuits the kernels, returned as-is.
    exercise passes through un-convolved on both paths since it is already per-step; both paths must
    carry it or the feature stack's raw-column lookup has no feat 3."""
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
    """The normalized (N, F) input stack for a whole Segment. Per stats: bg (feat 0) through the
    Kovatchev transform before the z-score (RISK_SPACE_CHANNELS), carb/insulin/exercise through
    log1p (SPARSE_LOG1P_CHANNELS). Exercise is written explicitly even when zero, since unwritten
    sits at z = 0, a phantom dose. Feat BG_MASKED_FEAT is the announcement bit: no stats, 0.0
    throughout, since every step of a Segment is OBSERVED; the masked set is written downstream."""
    n = len(seg)
    ch = segment_to_channels(seg)

    # raw post-noise (mirrors data._build_sample): bg clamped physical, sparse three floored at 0.
    bg = np.clip(seg.cgm, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)
    carb = np.clip(ch['carb'], 0.0, None).astype(np.float64)
    insulin = np.clip(ch['insulin'], 0.0, None).astype(np.float64)
    exercise = np.clip(ch['exercise'], 0.0, None).astype(np.float64)

    feats = np.zeros((n, N_INPUT_FEATURES), dtype=np.float32)
    raw = {0: bg, 1: carb, 2: insulin, 3: exercise}
    # every normalized column must be written; an unwritten one is a silent z = 0, mask bit above.
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

    No smoothing despite the name; kept for its call sites since the model consumes raw signals.
    """
    return np.clip(np.asarray(cgm, dtype=np.float64), BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)


def context_window(feats: np.ndarray, pred_start: int, n_ctx_patches: int):
    """Slice the ``n_ctx_patches`` patches ending at ``pred_start`` as (P, S, F)."""
    import torch
    ctx_steps = n_ctx_patches * PATCH_SIZE
    block = feats[pred_start - ctx_steps:pred_start]
    return torch.from_numpy(block.reshape(n_ctx_patches, PATCH_SIZE, N_INPUT_FEATURES).copy())
