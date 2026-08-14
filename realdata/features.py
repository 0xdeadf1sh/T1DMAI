"""
Model-input bridge — converts a real-data Segment into the model's normalized
``N_INPUT_FEATURES``-feature input stack
``[bg_absolute, carbs, insulin, exercise_equiv, bg_masked]``: four normalized
signal channels plus the per-patch ``bg_masked`` announcement bit, which is 0.0
throughout a stack of observed readings (a Segment is a record of what happened;
the masked set is chosen downstream, by the builder that knows it).

The model's carb/insulin channels are the simulator's absorption/action *curves*,
so raw events must be convolved with simulator-matched kernels (validated in
``scratch/kernel_match.py`` to reproduce the simulator's channels at r=0.94 carb,
0.99 bolus).  The kernels here are the *analytic* forms rebuilt from the
simulator's documented constants (reproducible, no dependence on a captured run):

    carb     mean meal mixture  (fast/med/slow gamma + protein/fat tail)
    insulin  gamma k=3, θ=25    (the rapid bolus-action kernel)
    exercise gamma k=3, θ=15    (the carbohydrate-equivalent disposal curve)

Real insulin is fed as one rapid-delivery series — bolus IU plus CSII basal
converted from rate (IU/h → IU/step) — convolved with the rapid kernel.  (MDI
long-acting, which the Shanghai adapter already folded into ``basal_rate`` as a
24-h-spread rate, is approximated as rapid here; a secondary-cohort caveat.)

Exercise (feat 3) is the simulator's carbohydrate-*equivalent* glucose-disposal
curve in g/step, on the same log1p+z transform as carb.  Every real adapter fills
``Segment.exercise`` with zeros, so the column is structurally present and
identically zero on all five cohorts; it is written explicitly rather than left at
the allocation default, because for a ``SPARSE_LOG1P`` channel an unwritten column
sits at z = 0, which is a phantom dose and not "no session" (no session is
``normalize(log1p(0))``, z = -0.1387 under the balanced pool).  A source that ever
does fill the column must convert its own quantity to g/step carbohydrate-equivalent
first — the trained scale is g/step, not an intensity.

``EXERCISE_KERNEL`` is the appearance shape of ONE announced session, unit area,
for a counterfactual that injects a session as a point event (``metrics/whatif.py``).
It is NOT part of ``segment_to_channels``: a Segment's ``exercise`` field already
holds the resolved per-step curve, so nothing on the input path convolves it.

The risk-space redesign dropped ``bg_delta``, the IS/HGO latent channels, and the
four temporal sin/cos features from the input stack entirely (they were never
observable in real data / the model no longer consumes them), so this bridge emits
exactly the signal channels ``CHANNEL_NAMES`` retains.  bg (feat 0) is a
Kovatchev risk-space channel (``kovatchev_f`` applied BEFORE the z-score); carb,
insulin and exercise keep the log1p+z transform.
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
    EXERCISE_GAMMA_K, EXERCISE_GAMMA_THETA,
)
from config import PATCH_SIZE, N_INPUT_FEATURES
from data import BG_MASKED_FEAT
from normalization import CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS, RISK_SPACE_CHANNELS
from utils import kovatchev_f_np
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

from .schema import Segment, GRID_MIN

# Truncation horizon shared by all three kernels.  The tail past it is folded back
# in by the unit-area renormalization below, so the truncation is a stated
# approximation rather than lost mass: measured against the same kernel built out
# to 2400 min, 240 min retains 0.99314 of the meal mixture, 0.99616 of the bolus
# gamma and 0.99998 of the exercise gamma (k=3, θ=15, mean 45 min).
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
    """Unit-area appearance shape of one exercise session.

    A SINGLE gamma, ``EXERCISE_GAMMA_K`` / ``EXERCISE_GAMMA_THETA``, which is what
    the simulator schedules for a session: it draws
    ``gamma_curve(duration * carb_equiv_per_min, k, θ, duration + 90)`` and adds
    the result into ``total_exercise``.  Two differences from that draw, both
    deliberate:

    * the session's own support is ``duration + 90`` min and varies per draw, so
      it cannot be a kernel constant; this truncates at the fixed
      ``_EXERCISE_KERNEL_MIN`` the other two kernels use and renormalizes to unit
      area, folding the 2e-5 residual tail back in.
    * the magnitude is factored out — the caller supplies the session's grams of
      carbohydrate-equivalent disposal, so this carries shape only.

    NOT the meal mixture: ``CARB_KERNEL`` is a three-way meal-type mixture plus a
    protein/fat tail, and its long tail puts only 0.854 of its mass inside a 2 h
    horizon against this curve's 0.986, so reusing it would announce a session
    the simulator never produces.
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
    """Convert a Segment's raw events into the model's absorption/action channels.

    A Segment carrying pre-resolved ``carb_curve`` / ``insulin_curve`` (a source
    whose events already store the series the producer fed the model) short-circuits
    the kernels and returns those channels as-is — the kernels below exist only to
    reconstruct what such a source already knows.  The three published cohorts leave
    both ``None`` and take the convolution path unchanged.

    Returns dict with ``carb`` (g/step absorption), ``insulin`` (IU/step action) and
    ``exercise`` (g/step carbohydrate-equivalent glucose disposal).  ``exercise``
    passes through un-convolved on both paths: it is already a per-step channel, and
    every adapter fills it with zeros.  Both paths must carry it, or the feature
    stack's raw-column lookup has no entry for feat 3.
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
    """Build the normalized (N, F) input stack for a whole Segment.

    Every channel of ``CHANNEL_NAMES`` (bg, carb, insulin, exercise) is normalized
    per ``stats``: bg (feat 0) through the Kovatchev risk transform BEFORE the
    z-score (``RISK_SPACE_CHANNELS``), carb/insulin/exercise through log1p
    (``SPARSE_LOG1P_CHANNELS``).  The redesign removed ``bg_delta``, the IS/HGO
    latents, and the temporal sin/cos features from the stack, so there is nothing
    to mean-impute and no temporal tail.

    Exercise (feat 3) is zero on every real cohort but is still written explicitly:
    an unwritten column would sit at z = 0, a phantom dose, where the true
    no-session value is ``normalize(log1p(0))``.

    Feat ``BG_MASKED_FEAT`` is the ``bg_masked`` announcement bit, not a signal: it
    carries no normalization statistics and never crosses the z-score.  Every step
    of a Segment is an OBSERVED reading, so the column is 0.0 throughout; the
    masked set is written into the patches downstream, by the builder that knows
    it (``inference._build_patches_tensor`` rewrites BOTH halves of the column, so
    a bit riding in from here could not survive as a phantom announcement either).
    """
    n = len(seg)
    ch = segment_to_channels(seg)

    # The model consumes RAW post-noise signals: every signal channel (bg feat0,
    # carb feat1, insulin feat2, exercise feat3) is used raw BEFORE normalization,
    # mirroring ``data._build_sample`` (same clamps — bg to the physical BG range,
    # the sparse carb/insulin/exercise floored at 0; no FIR smoothing).
    bg = np.clip(seg.cgm, BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float64)
    carb = np.clip(ch['carb'], 0.0, None).astype(np.float64)
    insulin = np.clip(ch['insulin'], 0.0, None).astype(np.float64)
    exercise = np.clip(ch['exercise'], 0.0, None).astype(np.float64)

    feats = np.zeros((n, N_INPUT_FEATURES), dtype=np.float32)
    raw = {0: bg, 1: carb, 2: insulin, 3: exercise}
    # Length check, not a name list: the guard that matters is that every NORMALIZED
    # column gets written, since an unwritten one is a silent z = 0.  The normalized
    # channels occupy columns 0..BG_MASKED_FEAT-1 and the mask bit sits above them,
    # so the stack is exactly one column WIDER than CHANNEL_NAMES.
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
