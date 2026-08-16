"""
Leave-one-patient-out fine-tuning of a pretrained T1DMAI checkpoint on a real
CGM dataset, with the held-out patient scored DURING fine-tuning.

The training machinery (loss geometry, optimizers, LR schedule, EMA) is reused
verbatim from the repo so a fine-tune matches pretraining in everything but data:
the forecast loss is ``risk_loss.risk_total_loss`` under the same Kovatchev risk
transform, the parameter split is Muon(≥2D)/AdamW(≤1D), and validation runs
under the EMA shadow.  The time-of-day probe co-trains the shared trunk here too
(``config.TIME_PROBE_DETACH`` False): its per-patch bin cross-entropy plus
inter-patch consistency loss is added to the backward tensor
only — never to ``loss_ema``, the CSV, or checkpoint selection — so its gradient
shapes the trunk exactly as in pretraining.  Two regimes are supported:

  * ``transfer``    — fine-tune on EVERY OTHER patient, test on the held-out one
                      (cross-patient generalization).
  * ``personalize`` — fine-tune on the held-out patient's own CALIBRATION split,
                      test on its disjoint TEST split (within-patient adaptation).

The held-out number printed here is the apples-to-apples generalization signal to
compare against the all-patients average the ``metrics/`` scripts produce (those
read the hardcoded ``checkpoints/t1dmai_best.pt`` and score ALL patients).

**Selection is a strictly proper scalar under hard admission gates.** A candidate
is admitted only if every gate holds — chiefly the dose-response gate, which
refuses any checkpoint whose insulin correct-sign fraction has fallen below the
pretrained model's — and the admitted candidates are then ranked by CRPS at the
selection horizon, which is weight-free (every shipped τ enters equally) and
strictly proper (the band cannot be widened into a better score). No gate trades
against another and no clinical composite is weighted; RMSE keeps every CSV column
it had and loses only the headline slot and the selection role.

Every eval writes both point metrics to a PER-RUN CSV under ``finetune/logs/``, so
two runs never overwrite each other's curve and ``rmse_point`` and ``rmse_winmean``
can be compared step for step.

The masked set every window here carries is ONE right-edge span of
``PREDICTION_PATCHES`` — the forecast case of the general masked-BG objective,
not a draw from ``data.sample_mask_spans`` — because the held-out score this
script selects on is a forecast at 30/60/120 min.  It runs through the general
machinery all the same: the head gathers ``mask_idx``, the loss discards padded
slots by ``valid``, and the anchor is the shared one-sided rule, whose right-edge
case is the old ``last_bg``.

Runs from either ``finetune/`` or the repo root — ``REPO_ROOT`` is resolved from
this file's location and prepended to ``sys.path``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from model import T1DMAI
from muon import Muon
from risk_loss import risk_total_loss, KendallGalWeighting
from utils import ModelEMA, time_of_day_bin_ce, time_cross_window_consistency_loss
from data import (
    collate_fn, BG_MASKED_FEAT, _anchor_step_for_span, _mask_slots,
    masked_channel_policy, stored_masked_channel_policy,
)
from realdata import load_dataset
from realdata.features import build_feature_stack, smoothed_cgm
from realdata.calibrate import collect_windows, forecast_bands, forecast_windows
from realdata.run_eval import (
    split_segments, evaluate_from_windows, horizon_d_patches, FORECAST_D_PATCHES,
)
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

# ---------------------------------------------------------------------------- #
# Fine-tune tunables (each overridable by the matching CLI flag).
# ---------------------------------------------------------------------------- #
FINETUNE_TOTAL_STEPS: int = 2000          # optimizer steps for the fine-tune
FINETUNE_BATCH_SIZE: int = 128            # samples per step
FINETUNE_WARMUP_STEPS: int = 100          # linear LR warmup before cosine decay
FINETUNE_LR_SCALE: float = 0.1            # peak-LR multiplier vs the pretrain LRs
FINETUNE_LR_MIN_RATIO: float = 0.05       # cosine floor as a fraction of peak
FINETUNE_EMA_DECAY: float = 0.99          # EMA decay (shorter window than pretrain)
FINETUNE_EVAL_INTERVAL: int = 200         # held-out eval cadence (steps)
FINETUNE_TRAIN_STRIDE_PATCHES: int = 2    # pred-start stride for training windows
FINETUNE_EVAL_TEST_STRIDE_PATCHES: int = 4   # window stride for the test split
FINETUNE_EVAL_CAL_STRIDE_PATCHES: int = 8    # window stride for the cal split
FINETUNE_EVAL_MAX_PER_PATIENT: int = 400  # window cap per patient at eval
FINETUNE_SEED: int = 0                    # RNG seed (sampling + shuffle)
FINETUNE_LOG_INTERVAL: int = 50           # training-progress console cadence

# Held-out horizons scored to CSV/console, and the selection horizon/metric.
EVAL_HORIZONS: tuple[int, ...] = (30, 60, 120)
SELECTION_HORIZON: int = 60
# The selection scalar is CRPS at ``SELECTION_HORIZON``: weight-free, since every
# shipped τ enters with the same weight and no clinical quantity is traded against
# another, and strictly proper, so the score is minimized by the true predictive
# distribution and a wider band cannot buy a better one.  A point RMSE is neither —
# it scores one line out of the fan the alarm path actually reads, and a confidently
# wrong forecast can beat a correctly uncertain one.  What replaces the rest of
# RMSE's old job is the admission gates below, not a second scalar.
#
# It is read at ONE horizon, which is ONE ``d`` (the distance in patches to the
# nearest visible evidence — 60 min is d = 2, one-sided, under the forecast
# protocol).  It is never pooled across d: the masked-BG supervision the model
# trains under concentrates at small d, so any average over d improves for free.
SELECTION_METRIC: str = 'crps'
SELECTION_D_PATCHES: int = horizon_d_patches(SELECTION_HORIZON)

# Patience, in EVALS without an admitted improvement, before the fine-tune stops.
# DELIBERATELY UNSET: no run settles it.  The archived personal logs cannot — on the
# large fit rmse_point@60 and rmse_winmean@60 disagree about the best step by 500,
# and the medium fit left no per-step curve at all, having written to a single
# overwritten path with no rmse_winmean column.  None = run the full step budget;
# set --patience once a run under the per-run logging below settles a value.
FINETUNE_PATIENCE_EVALS: int | None = None

# Per-run CSV directory.  ``.gitignore:81`` (``logs/``) covers it at any depth.
FINETUNE_LOG_DIR: str = os.path.join(REPO_ROOT, 'finetune', 'logs')

# --- Admission gates -------------------------------------------------------- #
# A candidate ships only if EVERY gate holds; the strictly proper scalar then ranks
# the admitted candidates.  A gate is a hard yes/no, never a weighted term — the
# hand-weighted clinical composite this replaces was removed deliberately.
DOSE_GATE_MODE: str = 'block'             # 'block' | 'warn' | 'off'
DOSE_GATE_SIGN_TOLERANCE: float = 0.0     # slack below the baseline sign fraction
DOSE_GATE_STRIDE_PATCHES: int = 8         # probe window stride — a compute budget
DOSE_GATE_MAX_WINDOWS_PER_SEG: int = 8    # probe windows per segment — a compute budget
# The exercise sign gate runs on SIMULATOR windows only: every real adapter emits an
# identically zero exercise column, so a probe there measures the response to a
# session that was never announced.  It is off by default because producing those
# windows costs a simulator run; when it is on and either half is missing — the
# Segment-shaped source or ``metrics/whatif.py``'s exercise arm — it names which,
# and under 'block' it refuses rather than passing silently.
EXERCISE_GATE_ENABLED: bool = False
EXERCISE_GATE_SIM_FACTORY: str = 'make_sim_segments'   # in metrics/sim/sim_data.py
EXERCISE_GATE_SIM_HOURS: float = 120.0                 # per simulator patient
EXERCISE_GATE_SIM_PATIENTS: int = 4                    # a compute budget, not a threshold
# Absolute floors, DELIBERATELY UNSET.  No reference pretrain exists to read one
# off, and a guessed floor would silently decide what ships.  None = not enforced.
GATE_MIN_HYPO_RECALL: float | None = None
GATE_MIN_BAND_COV50: float | None = None

# NaN-resilience: consecutive non-finite steps before an EMA-shadow rollback.
CONSECUTIVE_NAN_RESTORE: int = 10

# Dataset-name -> on-disk subpath under REPO_ROOT (the adapter defaults, made
# absolute so the script runs from any cwd).
DATASET_SUBPATHS: dict[str, str] = {
    'ohiot1dm': os.path.join('T1DMSIM', 'datasets', 'ohiot1dm'),
    'azt1d': os.path.join('T1DMSIM', 'datasets', 'AZT1D', 'CGM Records'),
    'shanghai': os.path.join('T1DMSIM', 'datasets', 'ShanghaiT1DM', 'Shanghai_T1DM'),
}

# Derived step counts (NOT tunables — read off the live architecture).
_PRED_STEPS: int = config.PREDICTION_PATCHES * config.PATCH_SIZE
_LONG_STEPS: int = config.NIGHT_LONG_HORIZON_PATCHES * config.PATCH_SIZE
_CTX_MIN_STEPS: int = config.MIN_CONTEXT_PATCHES * config.PATCH_SIZE
_CTX_EVAL_STEPS: int = config.MAX_CONTEXT_PATCHES * config.PATCH_SIZE


# ---------------------------------------------------------------------------- #
# Checkpoint loading + architecture guard.
# ---------------------------------------------------------------------------- #
def _alignment_message(ckpt: dict[str, Any]) -> str:
    """Build the operator-facing message for an arch / state-dict mismatch.

    Reports the checkpoint's stored architecture vs the live ``config.py`` and the
    exact ``resize_model.py`` command that aligns the live config to the
    checkpoint. ``T1DMAI()`` reads the live ``config.py``, so the ``metrics/``
    scripts need the same alignment to load and score the produced weights.
    """
    tc = ckpt.get('training_config', {}) or {}
    d = int(tc.get('d_model', config.D_MODEL))
    n_layers = int(tc.get('n_layers', config.N_LAYERS))
    n_heads = int(tc.get('n_heads', config.N_HEADS))
    ffn = int(tc.get('ffn_dim', config.FFN_DIM))
    ps = int(tc.get('patch_size', config.PATCH_SIZE))
    mcp = int(tc.get('max_context_patches', config.MAX_CONTEXT_PATCHES))
    ffn_mult = max(1, ffn // max(1, d))

    # Build the resize command from the dimensions resize_model.py can actually
    # set, so running it truly aligns the live config to the checkpoint.
    cmd_parts = [
        "python resize_model.py", f"--d-model {d}", f"--layers {n_layers}",
        f"--heads {n_heads}", f"--ffn-mult {ffn_mult}", f"--patch-size {ps}",
        f"--max-context-patches {mcp}",
    ]
    # bg_head_hidden is NOT persisted by the training_config schema, so the
    # checkpoint's true value is unknown unless a future schema carries it; only
    # emit --bg-head-hidden-mult (and report the value) when it is actually present
    # — otherwise it is verified by the strict state-dict load (see _arch_guard).
    if 'bg_head_hidden' in tc:
        bgh = int(tc['bg_head_hidden'])
        cmd_parts.append(f"--bg-head-hidden-mult {max(1, bgh // max(1, d))}")
        bgh_ckpt = str(bgh)
    else:
        bgh_ckpt = "<not in checkpoint; verified by strict load>"
    cmd = " ".join(cmd_parts)

    # prediction_patches is derived from PREDICTION_HORIZON_HOURS; resize_model.py
    # has no flag for it, so a mismatch needs a config.py edit instead.
    pred_note = ""
    ckpt_pp = tc.get('prediction_patches')
    if ckpt_pp is not None and int(ckpt_pp) != config.PREDICTION_PATCHES:
        ph = tc.get('prediction_horizon_hours')
        if ph is not None:
            pred_note = (
                f"\n  ALSO set PREDICTION_HORIZON_HOURS = {ph} in config.py "
                f"(prediction_patches={ckpt_pp} is derived from it; resize_model.py "
                "cannot set it directly)."
            )
        else:
            pred_note = (
                f"\n  ALSO edit PREDICTION_HORIZON_HOURS in config.py so "
                f"PREDICTION_PATCHES == {ckpt_pp} (resize_model.py cannot set "
                "prediction_patches directly)."
            )
    return (
        "Architecture mismatch: the live config.py does not match the checkpoint.\n"
        f"  checkpoint: d_model={d} n_layers={n_layers} n_heads={n_heads} "
        f"ffn_dim={ffn} bg_head_hidden={bgh_ckpt} patch_size={ps} "
        f"max_context_patches={tc.get('max_context_patches')} "
        f"prediction_patches={tc.get('prediction_patches')}\n"
        f"  live:       d_model={config.D_MODEL} n_layers={config.N_LAYERS} "
        f"n_heads={config.N_HEADS} ffn_dim={config.FFN_DIM} "
        f"bg_head_hidden={config.BG_HEAD_HIDDEN} patch_size={config.PATCH_SIZE} "
        f"max_context_patches={config.MAX_CONTEXT_PATCHES} "
        f"prediction_patches={config.PREDICTION_PATCHES}\n"
        "Align the live config.py to the checkpoint, then re-run this script:\n"
        f"  {cmd}" + pred_note + "\n"
        "The metrics/ scripts construct T1DMAI() from the LIVE config.py too, so "
        "they need the same alignment to load and score this fine-tuned checkpoint."
    )


def _arch_guard(ckpt: dict[str, Any]) -> None:
    """Raise ``SystemExit`` with alignment guidance if the arch differs.

    Only keys present in ``ckpt['training_config']`` are compared (the schema does
    not persist ``bg_head_hidden``, so it is verified by the strict state-dict load
    instead).  The mask sampler is ``_sampler_guard``'s; it changes no dimension
    here and needs a different remedy.
    """
    tc = ckpt.get('training_config', {}) or {}
    checks = [
        ('d_model', config.D_MODEL), ('n_layers', config.N_LAYERS),
        ('n_heads', config.N_HEADS), ('ffn_dim', config.FFN_DIM),
        ('patch_size', config.PATCH_SIZE), ('bg_head_hidden', config.BG_HEAD_HIDDEN),
        ('max_context_patches', config.MAX_CONTEXT_PATCHES),
        ('prediction_patches', config.PREDICTION_PATCHES),
    ]
    mismatches = [(k, tc[k], live) for k, live in checks if k in tc and tc[k] != live]
    if mismatches:
        raise SystemExit(_alignment_message(ckpt))


def _sampler_message(tc: dict[str, Any]) -> str:
    """Build the operator-facing message for a mask-sampler mismatch.

    Separate from ``_alignment_message`` because the remedy is different:
    ``resize_model.py`` has no flag for any of these, and the fix is a
    ``config.py`` edit back to the values the checkpoint records.
    """
    lengths = tc.get('mask_span_lengths')
    return (
        "Mask-sampler mismatch: the live sampler is not the one the checkpoint was "
        "trained under.\n"
        f"  checkpoint: mask_span_lengths={tuple(lengths) if lengths is not None else None} "
        f"max_masked_patches={tc.get('max_masked_patches')} "
        f"mask_right_edge_quota={_stored_quota(tc)}\n"
        f"  live:       mask_span_lengths={tuple(config.MASK_SPAN_LENGTHS)} "
        f"max_masked_patches={config.MAX_MASKED_PATCHES} "
        f"mask_right_edge_quota={config.MASK_RIGHT_EDGE_QUOTA}\n"
        "Nothing else catches this: no parameter shape depends on the sampler, so "
        "the strict state-dict load accepts weights trained under any of them. This "
        "script draws no mask from ``data.sample_mask_spans`` — its masked set is one "
        "right-edge span — so what the mismatch says is that the WEIGHTS were shaped "
        "by a different supervision mixture than the live config describes, and "
        "MAX_MASKED_PATCHES is M in this script's slot tensors besides.\n"
        "Restore the checkpoint's values in config.py, or fine-tune a checkpoint "
        "trained under the live ones."
    )


_SAMPLER_KEYS = ('mask_span_lengths', 'max_masked_patches', 'mask_right_edge_quota')


def _stored_quota(tc: dict[str, Any]) -> "float | None":
    """The right-edge quota a checkpoint was trained under, or None if unknowable.

    Within a ``training_config`` that describes a sampler at all, an ABSENT quota
    key is information rather than a reason to skip the comparison: the key was
    introduced with the quota itself, so such a checkpoint was trained under
    uniform placement, which is quota 0.0.  Reading absence as "unknown" would
    exempt precisely the checkpoints that can mismatch, since the pre-quota ones
    are the only ones that do.

    A ``training_config`` that describes no sampler at all is a different case and
    returns None: nothing about it is recoverable, and ``_arch_guard`` skips its
    keys for the same reason.
    """
    if 'mask_right_edge_quota' in tc:
        return float(tc['mask_right_edge_quota'])
    if any(k in tc for k in _SAMPLER_KEYS):
        return 0.0
    return None


def _sampler_guard(ckpt: dict[str, Any]) -> None:
    """Raise ``SystemExit`` if the live mask sampler is not the checkpoint's.

    The recorded sampler CONSTANTS are compared, so the check is true provenance:
    it is those values that shaped the supervision the weights were trained on.

    Keys absent from ``ckpt['training_config']`` are skipped as in ``_arch_guard``,
    with one exception: the right-edge quota, whose absence pins it at 0.0 (see
    ``_stored_quota``).
    """
    tc = ckpt.get('training_config', {}) or {}
    # ``mask_span_lengths`` is stored as a list and bound as a tuple, so both sides
    # go through one normalizer — compared raw, every checkpoint carrying the key
    # would mismatch.
    checks = [
        ('mask_span_lengths', config.MASK_SPAN_LENGTHS, tuple),
        ('max_masked_patches', config.MAX_MASKED_PATCHES, int),
    ]
    mismatched = any(norm(tc[k]) != norm(live) for k, live, norm in checks if k in tc)
    stored_quota = _stored_quota(tc)
    if mismatched or (stored_quota is not None
                      and stored_quota != float(config.MASK_RIGHT_EDGE_QUOTA)):
        raise SystemExit(_sampler_message(tc))

    # A retired knob, checked because the checkpoint still records it and this code
    # can no longer reproduce what it names: under it the pinball loss carried a
    # per-``d`` weight, so the fine-tune would continue those weights' training on a
    # loss that no longer has them. The name is dead in config; it is alive in every
    # checkpoint written before it went.
    if tc.get('d_balanced_loss'):
        raise SystemExit(
            "Retired training objective: the checkpoint records "
            "d_balanced_loss=True, and the per-d loss weights it names no longer "
            "exist — risk_loss.pinball_loss weights every masked patch equally now. "
            "Fine-tuning it would continue those weights' training under a "
            "different objective, which nothing downstream would report.\n"
            "Fine-tune a checkpoint trained under the live loss."
        )


# What a masked patch withholds in the windows THIS script builds: bg alone, with
# the announced carb / insulin / exercise riding through it (``_mask_window``
# touches ``NON_MASKABLE_FEATS`` only). ``train_blind.py`` pretrains the other
# policy.
LIVE_MASKED_CHANNEL_POLICY = masked_channel_policy(blind=False)


def _stored_masked_channel_policy(tc: dict[str, Any]) -> str:
    """The masked-channel policy a checkpoint was trained under.

    ``data.stored_masked_channel_policy`` is the single reader of the absent-key
    convention — it reads as ``announced`` unconditionally, and without the
    sampler-key precondition ``_stored_quota`` needs.
    """
    return stored_masked_channel_policy(tc)


def _policy_guard(ckpt: dict[str, Any]) -> None:
    """Raise ``SystemExit`` if the checkpoint's masked-channel policy is not this
    script's.

    Nothing else catches this. No parameter shape depends on the policy, so the
    strict state-dict load accepts weights trained under either, and both
    directions fine-tune to completion behind a plausible held-out curve:

    * a BLIND checkpoint fine-tuned here would be handed announced doses on the
      patches it must predict — a channel it was trained to read as constant
      suddenly carrying signal, which the fine-tune would spend its budget
      learning rather than adapting to the patient;
    * a conditioned checkpoint fine-tuned blind would lose the conditioning its
      weights were shaped around, and the held-out score would report the loss as
      a property of the patient.
    """
    tc = ckpt.get('training_config', {}) or {}
    stored = _stored_masked_channel_policy(tc)
    if stored != LIVE_MASKED_CHANNEL_POLICY:
        raise SystemExit(
            "Masked-channel policy mismatch: the checkpoint was pretrained under a "
            "different convention for what a masked patch withholds.\n"
            f"  checkpoint: masked_channel_policy={stored!r}"
            + ('' if 'masked_channel_policy' in tc else
               ' (absent — every checkpoint predating the blind trainer)')
            + f"\n  this script: masked_channel_policy={LIVE_MASKED_CHANNEL_POLICY!r}\n"
            "'announced' withholds bg alone and lets the carb / insulin / exercise "
            "plan ride through a masked patch; 'blind' withholds those three as "
            "well, at data.zero_dose_fill's normalize(0). No parameter shape "
            "records which, so the weights load either way and the fine-tune "
            "reports a plausible number for the wrong regime.\n"
            "Fine-tune a checkpoint pretrained under this script's policy."
        )


def load_checkpoint(
    path: str, device: torch.device,
) -> tuple[T1DMAI, dict[str, Any], dict[str, Any]]:
    """Load LIVE weights from a pretrain checkpoint.

    The model is loaded from ``model_state_dict`` (NOT EMA-merged) so fine-tuning
    continues on the un-smoothed parameters, exactly as pretraining did.

    Returns:
        (model, ckpt, stats) — ``stats`` is the checkpoint's own
        ``normalization_stats`` (the z-space the weights live in; never recomputed).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    assert isinstance(ckpt, dict), f"checkpoint at {path} is not a dict"
    assert ckpt.get('normalization_stats') is not None, (
        f"checkpoint at {path} has no normalization_stats — cannot place inputs in "
        "the weights' z-space"
    )
    stats = ckpt['normalization_stats']

    _arch_guard(ckpt)
    _sampler_guard(ckpt)
    _policy_guard(ckpt)

    model = T1DMAI().to(device)
    try:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    except Exception as exc:  # noqa: BLE001 — surface alignment guidance, not a traceback
        raise SystemExit(_alignment_message(ckpt) + f"\n\noriginal error: {exc}")

    return model, ckpt, stats


# ---------------------------------------------------------------------------- #
# Real-segment training samples (replicates data._build_sample for real data).
# ---------------------------------------------------------------------------- #
class RealSegmentDataset(Dataset):
    """Patch-aligned training windows over real CGM segments.

    Per segment the normalized feature stack, the raw (bg-clamped) mg/dL CGM, and
    the hour-of-day are precomputed once; ``__getitem__`` slices a window and
    assembles the same dict ``data._build_sample`` emits, so ``data.collate_fn``
    consumes it unchanged.  All inputs are in normalized z-space (feat 0 is
    z(f(bg)), Kovatchev risk-space then z-score) except feat 4, the per-patch
    ``bg_masked`` bit; the BG targets / anchors are mg/dL.
    """

    def __init__(self, segments: list, stats: dict[str, Any], seed: int) -> None:
        assert len(segments) > 0, "RealSegmentDataset needs at least one segment"
        self._feats: list[np.ndarray] = []
        self._bg_sm: list[np.ndarray] = []
        self._hour: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []
        self._rng = np.random.default_rng(seed)

        stride = FINETUNE_TRAIN_STRIDE_PATCHES * config.PATCH_SIZE
        for seg in segments:
            feats = build_feature_stack(seg, stats)          # (N, N_INPUT_FEATURES) normalized
            bg_sm = smoothed_cgm(seg.cgm)                     # (N,) raw mg/dL, bg-clamped
            hour = np.asarray(seg.hour_of_day())             # (N,)
            n = feats.shape[0]
            assert feats.shape[1] == config.N_INPUT_FEATURES, (
                f"feature stack width {feats.shape[1]} != {config.N_INPUT_FEATURES}"
            )
            assert bg_sm.shape[0] == n and hour.shape[0] == n, "channel length mismatch"
            seg_idx = len(self._feats)
            self._feats.append(feats)
            self._bg_sm.append(bg_sm)
            self._hour.append(hour)
            for pred_start in range(_CTX_MIN_STEPS, n - _PRED_STEPS + 1, stride):
                assert pred_start % config.PATCH_SIZE == 0, (
                    f"pred_start {pred_start} not patch-aligned"
                )
                self._index.append((seg_idx, pred_start))

    def __len__(self) -> int:
        return len(self._index)

    def _mask_bits(self, patches: torch.Tensor, masked_rows: torch.Tensor) -> None:
        """Withhold bg on the masked rows and announce them in feat 4, in place.

        feat 4 is written from the masked set itself on every row — never
        inherited from the feature stack, which carries no notion of a mask, so a
        column left as it arrived announces a masked patch as observed.  The bit
        is per PATCH and the row layout step-major, so it goes into all
        ``PATCH_SIZE`` columns of the feature.
        """
        for f in config.NON_MASKABLE_FEATS:      # bg feat 0 withheld; carb/insulin/exercise kept
            patches[masked_rows, f::config.N_INPUT_FEATURES] = 0.0
        patches[:, BG_MASKED_FEAT::config.N_INPUT_FEATURES] = 0.0
        patches[masked_rows, BG_MASKED_FEAT::config.N_INPUT_FEATURES] = 1.0

    def __getitem__(self, i: int) -> dict[str, Any]:
        """Assemble one training sample.

        Returns a dict with ``patches`` (n_ctx+P, PATCH_DIM) normalized,
        ``targets`` (M, S) mg/dL — one row per head slot — ``n_context_patches``
        int, and ``bg_formula_data`` carrying the masked set (``mask_idx``,
        ``valid``, ``anchor_bg``, ``d``, ``slot_hour``), ``last_bg`` (mg/dL),
        ``true_bg_trajectory`` (PRED_STEPS,) and ``extended_true_bg_trajectory``
        (LONG,) mg/dL, plus ``pred_start_hour``.

        The masked set is ONE right-edge span of ``PREDICTION_PATCHES`` — the
        forecast case of the general masked-BG objective — rather than a draw
        from ``data.sample_mask_spans``: the held-out score this fine-tune
        selects on is a forecast at 30/60/120 min, so the fine-tune distribution
        stays the forecast.  It is expanded onto the head's ``M`` slots by
        ``data._mask_slots`` and anchored by ``data._anchor_step_for_span``, so
        the slot layout, the anchor and ``d`` are the pretraining definitions and
        not a second copy of them.
        """
        seg_idx, pred_start = self._index[i]
        feats = self._feats[seg_idx]
        bg_sm = self._bg_sm[seg_idx]
        hour = self._hour[seg_idx]

        # Variable context: uniform in [MIN, min(MAX, available)] patches.
        hi = min(config.MAX_CONTEXT_PATCHES, pred_start // config.PATCH_SIZE)
        n_ctx = int(self._rng.integers(config.MIN_CONTEXT_PATCHES, hi + 1))
        ctx_steps = n_ctx * config.PATCH_SIZE
        seq_len = n_ctx + config.PREDICTION_PATCHES
        win_start = pred_start - ctx_steps
        win_steps = seq_len * config.PATCH_SIZE

        # The masked set: one span of PREDICTION_PATCHES ending at patch
        # seq_len - 1.  ``anchor_step`` is window-relative, so with the span at
        # the right edge it resolves to bg_window[n_ctx * PATCH_SIZE - 1] —
        # byte-identical to bg_sm[pred_start - 1], the old ``last_bg``.
        spans = [(n_ctx, config.PREDICTION_PATCHES)]
        mask_idx, valid, mask_d, anchor_step = _mask_slots(spans, seq_len)
        masked_rows = torch.arange(n_ctx, seq_len)

        window = feats[win_start:win_start + win_steps]
        patches = torch.from_numpy(
            window.reshape(seq_len, config.PATCH_DIM).copy()).float()
        self._mask_bits(patches, masked_rows)
        assert patches.shape == (seq_len, config.PATCH_DIM), (
            f"patch block {tuple(patches.shape)} != {(seq_len, config.PATCH_DIM)}"
        )
        is_masked = torch.zeros(seq_len, dtype=torch.bool)
        is_masked[masked_rows] = True
        _bit = patches[:, BG_MASKED_FEAT::config.N_INPUT_FEATURES]
        assert torch.equal(_bit, is_masked[:, None].expand_as(_bit).to(_bit.dtype)), (
            "feat 4 does not reproduce the masked set"
        )

        # One target row per head slot, RAW mg/dL (the risk transform is applied
        # once, in the loss).  Padded slots read patch 0 exactly as ``mask_idx``
        # does; ``valid`` is what discards them.
        bg_window = bg_sm[win_start:win_start + win_steps]
        bg_patches = bg_window.reshape(seq_len, config.PATCH_SIZE)
        targets = torch.from_numpy(bg_patches[mask_idx].copy()).float()   # (M, S)

        # Per-slot anchor, one-sided and left-preferring, read off the raw mg/dL
        # array.  Padded slots carry ``last_bg`` — an arbitrary but LEGAL mg/dL
        # from this window, so the forward's (B, M) units tripwire never fires on
        # a slot ``valid`` is about to discard.
        last_bg = float(np.clip(
            bg_window[_anchor_step_for_span(n_ctx, config.PREDICTION_PATCHES)],
            BG_CLAMP_MIN, BG_CLAMP_MAX))
        assert last_bg >= BG_CLAMP_MIN - 1e-3, (
            f"last_bg {last_bg} below BG_CLAMP_MIN — non-mg/dL value in the anchor"
        )
        anchor_bg = np.full(config.MAX_MASKED_PATCHES, last_bg, dtype=np.float32)
        anchor_bg[valid] = np.clip(
            bg_window[anchor_step[valid]], BG_CLAMP_MIN, BG_CLAMP_MAX)

        # Per-slot TRUE hour of day, at the slot's own patch.  Derived instead as
        # pred_start_hour + 0.5 * j it is off by (mask_idx[j] - n_ctx - j) * 0.5 h
        # whenever the masked set is not the right-edge span, with every shape
        # still matching.
        slot_hour = hour[win_start + mask_idx * config.PATCH_SIZE].astype(np.float32)

        true_bg_trajectory = bg_sm[pred_start:pred_start + _PRED_STEPS].astype(np.float32).copy()
        ext = bg_sm[pred_start:pred_start + _LONG_STEPS]
        if len(ext) < _LONG_STEPS:
            ext = np.pad(ext, (0, _LONG_STEPS - len(ext)), mode='edge')
        extended_true_bg_trajectory = ext.astype(np.float32)
        pred_start_hour = float(hour[pred_start])

        sample: dict[str, Any] = {
            'patches': patches,
            'targets': targets,
            'n_context_patches': int(n_ctx),
            'bg_formula_data': {
                # The masked set and everything keyed to it, all (M,).
                'mask_idx': mask_idx,          # (M,) int64  patch index per head slot
                'valid': valid,                # (M,) bool
                'anchor_bg': anchor_bg,        # (M,) float32 mg/dL
                'd': mask_d,                   # (M,) int64  to nearest visible, EITHER side
                'slot_hour': slot_hour,        # (M,) float32 true hour of day per slot
                'last_bg': last_bg,
                'true_bg_trajectory': true_bg_trajectory,
                'extended_true_bg_trajectory': extended_true_bg_trajectory,
                'pred_start_hour': pred_start_hour,
            },
        }

        # === Cross-window (paired-window) time-of-day probe input (window k+1) ===
        # Window k shifted forward by exactly PREDICTION_PATCHES on the SAME
        # raw segment (teacher-forced re-slice, not autoregressive).
        # Same n_ctx => identical seq_len and left-pad as window k. It carries the
        # SAME right-edge span as window k, so the slot layout (mask_idx / valid /
        # d) is shared and only the anchors and the clock move one horizon along;
        # data.collate_fn builds this window its own attention mask from its own
        # masked set all the same, and that is what the 2nd forward runs under.
        # Gated + shape-matched exactly like data._build_sample
        # so the shared collate batches it unchanged; when the future runs off the
        # segment end it ships a masked-out zero placeholder (valid=False).
        if config.TIME_PROBE_ENABLED and config.TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
            next_pred_start = pred_start + _PRED_STEPS
            next_win_start = win_start + _PRED_STEPS
            next_valid = next_pred_start + _PRED_STEPS <= feats.shape[0]
            if next_valid:
                nxt_window = feats[next_win_start:next_win_start + win_steps]
                next_patches = torch.from_numpy(
                    nxt_window.reshape(seq_len, config.PATCH_DIM).copy()).float()
                self._mask_bits(next_patches, masked_rows)
                next_bg_window = bg_sm[next_win_start:next_win_start + win_steps]
                next_last_bg = float(np.clip(
                    next_bg_window[_anchor_step_for_span(n_ctx, config.PREDICTION_PATCHES)],
                    BG_CLAMP_MIN, BG_CLAMP_MAX))
                next_anchor_bg = np.full(
                    config.MAX_MASKED_PATCHES, next_last_bg, dtype=np.float32)
                next_anchor_bg[valid] = np.clip(
                    next_bg_window[anchor_step[valid]], BG_CLAMP_MIN, BG_CLAMP_MAX)
                next_slot_hour = hour[
                    next_win_start + mask_idx * config.PATCH_SIZE].astype(np.float32)
                next_pred_start_hour = float(hour[next_pred_start])
            else:
                # Finite placeholder, masked out downstream by ``valid``. The
                # announcement bit is still written so the placeholder is not a
                # window claiming every patch is observed, and the anchors reuse
                # window k's legal mg/dL so the units tripwire never fires.
                next_patches = torch.zeros(seq_len, config.PATCH_DIM, dtype=torch.float32)
                next_patches[masked_rows, BG_MASKED_FEAT::config.N_INPUT_FEATURES] = 1.0
                next_last_bg = last_bg
                next_anchor_bg = np.full(
                    config.MAX_MASKED_PATCHES, last_bg, dtype=np.float32)
                next_slot_hour = np.full(
                    config.MAX_MASKED_PATCHES, pred_start_hour, dtype=np.float32)
                next_pred_start_hour = pred_start_hour
            assert next_patches.shape == (seq_len, config.PATCH_DIM), (
                f"next_window patch shape {tuple(next_patches.shape)} != {(seq_len, config.PATCH_DIM)}"
            )
            sample['next_window'] = {
                'patches': next_patches,
                'mask_idx': mask_idx,
                'valid_slots': valid,
                'anchor_bg': next_anchor_bg,
                'd': mask_d,
                'slot_hour': next_slot_hour,
                'last_bg': float(next_last_bg),
                'pred_start_hour': float(next_pred_start_hour),
                'valid': bool(next_valid),
            }

        return sample


# ---------------------------------------------------------------------------- #
# Optimizers + LR schedule (mirrors train._build_optimizers / train._update_lr).
# ---------------------------------------------------------------------------- #
def build_optimizers(
    model: T1DMAI, weighting: KendallGalWeighting, muon_lr: float, adam_lr: float,
) -> tuple[Muon, torch.optim.AdamW]:
    """Split parameters into Muon (≥2D) and AdamW (≤1D) groups.

    The model's ≥2D weight matrices go to Muon; its ≤1D params (biases, norms) to
    AdamW.  The two Kendall-Gal log-variance scalars on ``weighting`` are ndim<2,
    so they join AdamW (never Muon) in their OWN ``weight_decay=0`` group — the
    uncertainty weights must not be decayed toward zero.

    Returns:
        (muon_opt, adam_opt).
    """
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adam_params = [p for p in model.parameters() if p.ndim < 2]

    weighting_params = []
    for _name, param in weighting.named_parameters():
        assert param.ndim < 2, (
            f"weighting param {_name} must be a scalar (ndim<2), got ndim={param.ndim}"
        )
        weighting_params.append(param)

    muon_opt = Muon(muon_params, lr=muon_lr, momentum=config.MUON_MOMENTUM,
                    ns_iterations=config.MUON_NS_ITERATIONS,
                    weight_decay=config.MUON_WEIGHT_DECAY)
    adam_opt = torch.optim.AdamW(
        [
            {'params': adam_params, 'weight_decay': config.ADAM_WEIGHT_DECAY},
            {'params': weighting_params, 'weight_decay': 0.0},
        ],
        lr=adam_lr, betas=config.ADAM_BETAS,
        weight_decay=config.ADAM_WEIGHT_DECAY, eps=config.ADAM_EPS,
    )
    return muon_opt, adam_opt


def update_lr(
    muon_opt: Muon, adam_opt: torch.optim.AdamW, step: int,
    peak_muon_lr: float, peak_adam_lr: float,
    warmup_steps: int, total_steps: int, lr_min_ratio: float,
) -> None:
    """Apply the warmup + cosine-decay LR schedule to both optimizers."""
    if step < warmup_steps:
        ratio = step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        ratio = lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))
    for group in muon_opt.param_groups:
        group['lr'] = peak_muon_lr * ratio
    for group in adam_opt.param_groups:
        group['lr'] = peak_adam_lr * ratio


# ---------------------------------------------------------------------------- #
# Metric extraction / selection helpers.
# ---------------------------------------------------------------------------- #
def _metric(res: dict[str, Any], h: int, *path: str) -> float | None:
    """Safely pluck ``res['metrics'][str(h)][path...]``; None if anything is absent.

    Level metrics are read from the MEDIAN-LINE block when the suite carries one
    (``metrics[h]['median_line']``, present once ``compute_suite`` is given the
    quantile fan): a band-scored error shrinks monotonically as the band widens, so
    it cannot serve as a selection objective. Keys absent from that block — the
    band-edge ``hypo``/``hyper`` detectors — fall through to the headline block, whose
    basis is unchanged.
    """
    d: Any = res.get('metrics', {}).get(str(h))
    if isinstance(d, dict) and path:
        ml = d.get('median_line')
        if isinstance(ml, dict) and path[0] in ml:
            d = ml
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    if d is None:
        return None
    try:
        return float(d)
    except (TypeError, ValueError):
        return None


def _summarize_heldout(res: dict[str, Any]) -> dict[str, Any]:
    """Compact per-horizon held-out summary stored in finetune_meta."""
    out: dict[str, Any] = {
        'n_test_windows': res.get('n_test_windows'),
        'n_cal_windows': res.get('n_cal_windows'),
        'n_patients': res.get('n_patients'),
    }
    for h in EVAL_HORIZONS:
        # Both point metrics stay: "retire RMSE" is the headline slot and the
        # selection role, and deletes no column.  ``exporters/descriptor.py``
        # reads ``rmse_point`` and ``mard`` out of this block by name.
        out[str(h)] = {
            SELECTION_METRIC: _metric(res, h, SELECTION_METRIC),
            'd_patches': horizon_d_patches(h),
            'rmse_point': _metric(res, h, 'rmse_point'),
            'mae_point': _metric(res, h, 'mae_point'),
            'rmse_winmean': _metric(res, h, 'rmse_winmean'),
            'mard': _metric(res, h, 'mard'),
            'clarke_AB': _metric(res, h, 'clarke_AB'),
            'skill_point': _metric(res, h, 'skill_point'),
            'band_cov50': _metric(res, h, 'band_cov50'),
            'band_width': _metric(res, h, 'band_width'),
            'hypo_recall': _metric(res, h, 'hypo', 'recall'),
            'hyper_recall': _metric(res, h, 'hyper', 'recall'),
        }
    return out


def _fmt(v: float | None) -> str:
    """Render a possibly-None metric for the console."""
    return 'nan' if v is None else f"{v:.3f}"


# ---------------------------------------------------------------------------- #
# The strictly proper selection scalar.
# ---------------------------------------------------------------------------- #
def _scoring():
    """Import ``metrics/scoring.py`` — the single definition of the proper scoring rules.

    CRPS is defined there and nowhere else: its quadrature over a 7-node fan is not
    a neutral choice, and a second implementation here would be a second answer.
    Lazy, so a driver that never scores a fan never imports it.

    Imported as a plain submodule import.  ``from metrics import scoring`` recurses
    forever through ``metrics/__init__.py``'s lazy ``__getattr__``, because the
    fromlist handler probes ``hasattr(metrics, 'scoring')`` first and that probe is
    what the hook re-enters.
    """
    import metrics.scoring               # noqa: PLC0415 — deliberately lazy
    return metrics.scoring


def fan_order_violations(bands: np.ndarray | None) -> int | None:
    """Count descending steps in the τ fan; None when no window carried one.

    Checked here rather than left to ``metrics.scoring``, whose input contract
    asserts on it: a mid-fine-tune AssertionError would end the run, where a count
    fails the fan-order admission gate and lets the fit continue.
    """
    if bands is None:
        return None
    from realdata.metrics import _FAN_ORDER_TOL_MGDL    # noqa: PLC0415 — the single tolerance
    return int(np.count_nonzero(np.diff(bands, axis=-1) < -_FAN_ORDER_TOL_MGDL))


def crps_by_d_from_windows(
        test_w: list) -> tuple[dict[int, float], dict[int, int], int | None]:
    """Per-``d`` CRPS in mg/dL for the fixed forecast protocol these windows run.

    The scoring unit is a masked PATCH, so each window's ``(PRED_STEPS, K)`` fan is
    reshaped into its ``PREDICTION_PATCHES`` patches and each patch carries the ``d``
    that ``run_eval.FORECAST_D_PATCHES`` derives from ``data._mask_slots`` — one
    definition of the slot layout, not a second.  The right-edge forecast span is
    one-sided, so those are ``d = 1..PREDICTION_PATCHES``.

    CRPS itself is ``metrics.scoring.crps_steps``; nothing about it is restated here.
    The POOLED figure that module also returns is deliberately not read: the
    training sampler's supervision concentrates at small ``d``, so a pooled
    masked-BG scalar improves as the mixture softens and must never select a
    checkpoint.

    A fan the scorer's input contract rejects — non-ascending, non-finite, or in the
    wrong space — is reported and scored as nothing, so that eval selects nothing and
    the fine-tune carries on.  The message is printed rather than swallowed: it is
    the units tripwire.

    Returns:
        ``({d: crps_mgdl}, {d: n_units}, fan_order_violations)`` — the first two empty
        where there are no windows, no fan, or a fan the scorer refused.
    """
    bands = forecast_bands(test_w)
    violations = fan_order_violations(bands)
    if bands is None or bands.shape[0] == 0 or violations:
        return {}, {}, violations          # the fan-order gate reports it; do not score it
    _pred, true, _last, _pats = forecast_windows(test_w)
    n, steps, k = bands.shape
    p, s = config.PREDICTION_PATCHES, config.PATCH_SIZE
    assert steps == p * s, f"window fan has {steps} steps, expected {p} × {s}"
    d = np.tile(np.asarray(FORECAST_D_PATCHES, dtype=np.int64), n)
    try:
        by_d = _scoring().crps_by_d(bands.reshape(n * p, s, k), true.reshape(n * p, s), d)
    except AssertionError as exc:
        print(f"  [eval] metrics.scoring refused this fan: {exc}")
        return {}, {}, violations
    return dict(by_d.by_d), dict(by_d.n_by_d), violations


def attach_proper_scores(res: dict[str, Any], test_w: list) -> dict[str, Any]:
    """Insert the strictly proper per-``d`` scalar into ``res['metrics']``, in place.

    When the metric suite already carries ``SELECTION_METRIC`` at every scored
    horizon, that value is kept and nothing is recomputed — the suite is then the
    single source and this is only the bridge.

    Each horizon's value is its own ``d`` bin's CRPS, and the bin is written beside
    it as ``d_patches``: under the forecast protocol 30 / 60 / 120 min are
    ``d = 1 / 2 / 4`` one-sided, so no reader has to infer the difficulty from a
    horizon label, and nothing is averaged across bins.
    """
    metrics = res.get('metrics', {})
    if all(isinstance(metrics.get(str(h)), dict)
           and metrics[str(h)].get(SELECTION_METRIC) is not None
           for h in EVAL_HORIZONS):
        res.setdefault('proper', {'metric': SELECTION_METRIC, 'basis': 'suite',
                                  'n_test_windows': len(test_w),
                                  'fan_order_violations': None})
        return res

    by_d, n_by_d, violations = crps_by_d_from_windows(test_w)
    for h in EVAL_HORIZONS:
        blk = metrics.get(str(h))
        if not isinstance(blk, dict):
            continue
        v = by_d.get(horizon_d_patches(h))
        blk[SELECTION_METRIC] = (None if v is None or not math.isfinite(v) else float(v))
        blk['d_patches'] = horizon_d_patches(h)
    res['proper'] = {'metric': SELECTION_METRIC,
                     'basis': 'fan-by-d' if by_d else 'unavailable',
                     'n_test_windows': len(test_w),
                     'crps_by_d': {int(k): v for k, v in by_d.items()},
                     'n_by_d': {int(k): v for k, v in n_by_d.items()},
                     'fan_order_violations': violations}
    return res


def _selection_scalar(res: dict[str, Any]) -> float | None:
    """The scalar the selector ranks admitted candidates by: SELECTION_METRIC@SELECTION_HORIZON.

    None when the eval carries no such scalar — and the caller then selects nothing.
    There is deliberately NO fallback to a point error: a fallback would restore the
    selection role RMSE has been retired from, silently, and only on the runs where
    the fan is missing.
    """
    v = _metric(res, SELECTION_HORIZON, SELECTION_METRIC)
    return v if (v is not None and math.isfinite(v)) else None


# ---------------------------------------------------------------------------- #
# Dose-response probe (metrics/whatif.py) and the admission gates.
# ---------------------------------------------------------------------------- #
def _whatif():
    """Import ``metrics/whatif.py`` — the dose-response probe.

    Lazy: it pulls in matplotlib and sets ``torch.set_num_threads(8)`` at module
    scope, and a fine-tune that gates nothing should pay either.  Imported as a
    plain submodule import for the reason ``_scoring`` gives.
    """
    import metrics.whatif               # noqa: PLC0415 — deliberately lazy
    return metrics.whatif


@dataclass
class GateConfig:
    """What the admission gates enforce, and how hard.

    ``mode`` 'block' refuses to ship a candidate whose probe is missing or whose sign
    fraction has fallen; 'warn' records the same verdict and ships anyway; 'off'
    skips the probe entirely.  ``sign_tolerance`` is slack below the pretrained
    model's own measured fraction — 0.0 is the rule as stated ("not below"), and any
    other value is a deliberate loosening.
    """
    mode: str = DOSE_GATE_MODE
    sign_tolerance: float = DOSE_GATE_SIGN_TOLERANCE
    stride_patches: int = DOSE_GATE_STRIDE_PATCHES
    max_windows_per_seg: int = DOSE_GATE_MAX_WINDOWS_PER_SEG
    exercise: bool = EXERCISE_GATE_ENABLED
    exercise_sim_hours: float = EXERCISE_GATE_SIM_HOURS
    exercise_sim_patients: int = EXERCISE_GATE_SIM_PATIENTS
    min_hypo_recall: float | None = GATE_MIN_HYPO_RECALL
    min_band_cov50: float | None = GATE_MIN_BAND_COV50

    def __post_init__(self) -> None:
        assert self.mode in ('block', 'warn', 'off'), (
            f"dose-gate mode {self.mode!r} not one of block/warn/off")


def dose_probe(model: T1DMAI, stats: dict[str, Any], device: torch.device,
               segs: list, cfg: GateConfig) -> dict[str, Any] | None:
    """Run ``metrics/whatif.py``'s counterfactual dose ladder over ``segs``.

    This is the ONLY dose probe usable here.  ``train._run_counterfactual_probe``
    requires the six ``extended_*_{norm,raw}`` keys (``train.py:1929-1931``) that
    ``RealSegmentDataset`` never emits — the two key sets are disjoint — so it hits
    its bare ``continue`` (``:1933``) on every sample and returns ``cf_n = 0`` with
    every ``cf_*`` None, which reads as "ran, found nothing" rather than as a
    failure.

    ``stride``/``cap`` are compute budgets, not thresholds: the gate compares this
    model against the pretrained one under identical settings, so the settings
    cancel.  Returns None when the probe cannot run at all.
    """
    try:
        wf = _whatif()
    except Exception as exc:  # noqa: BLE001 — the gate reports, it does not crash the fit
        print(f"  [gate] metrics/whatif.py did not import ({exc}); dose probe unavailable")
        return None
    if not segs:
        return None
    return wf.run(model, stats, device, segs,
                  stride=cfg.stride_patches * config.PATCH_SIZE,
                  cap=cfg.max_windows_per_seg)


_SIM_GATE_SEGMENTS: dict[tuple[int, float], list] = {}


def sim_gate_segments(cfg: GateConfig) -> tuple[list, str | None]:
    """Simulator segments for the exercise sign gate, or ``([], reason)``.

    The exercise probe is meaningless on a real record — every cohort adapter, and
    the personal one, emits an identically zero exercise column — so this gate runs
    on simulator windows or not at all.  It needs a ``Segment``-shaped simulator
    source; the factory is looked up by name in ``metrics/sim/sim_data.py`` and its
    absence is reported rather than passed over.

    The seeds are that module's own CALIBRATION patients, so the gate never reads
    the simulator patients a report is scored on, and the segments are built once
    per process: running the simulator again at every eval would cost more than the
    fine-tune.
    """
    key = (int(cfg.exercise_sim_patients), float(cfg.exercise_sim_hours))
    if key in _SIM_GATE_SEGMENTS:
        return _SIM_GATE_SEGMENTS[key], None
    try:
        import metrics.sim.sim_data as sim_data       # noqa: PLC0415 — deliberately lazy
    except Exception as exc:  # noqa: BLE001
        return [], f"metrics/sim/sim_data.py did not import ({exc})"
    factory = getattr(sim_data, EXERCISE_GATE_SIM_FACTORY, None)
    if factory is None:
        return [], (f"metrics/sim/sim_data.py has no {EXERCISE_GATE_SIM_FACTORY}() — "
                    "no Segment-shaped simulator source")
    seeds = tuple(sim_data.CAL_SEEDS)[:key[0]]
    segs = list(factory(seeds, hours=cfg.exercise_sim_hours))
    _SIM_GATE_SEGMENTS[key] = segs
    return segs, None


def probe_sign_fracs(probe: dict[str, Any] | None,
                     arm: str) -> tuple[dict[str, float | None], str | None]:
    """Reference-dose correct-sign fraction per horizon, out of a whatif summary.

    The rung read is whatif's own ``REF_IDX`` (the 1.0× rung, i.e. the ``CF_*`` dose)
    and the quantity is whatever that arm's ``sign_gate.metric`` names, so neither is
    a second copy.  ``sign_gate.threshold`` is null there by design — this gate's
    threshold is the pretrained model's own measurement, not a constant.

    Each horizon is a different ``d`` and they are kept apart, never averaged.

    Returns:
        ``(fracs, reason)`` — ``reason`` is why the arm produced nothing, when it did.
    """
    if not probe:
        return {}, 'the dose probe did not run'
    block = probe.get(arm)
    if not isinstance(block, dict):
        return {}, f"the probe carries no {arm} arm"
    if not block.get('n'):
        return {}, block.get('not_probed') or f"the {arm} arm scored no window"
    key = (block.get('sign_gate') or {}).get('metric', 'correct_sign_frac')
    ref = _whatif().REF_IDX
    out: dict[str, float | None] = {}
    for h in EVAL_HORIZONS:
        row = (block.get(key) or {}).get(str(h))
        out[str(h)] = None if row is None else row[ref]
    return out, None


def gate_values(res: dict[str, Any]) -> dict[str, Any]:
    """Everything the gates read, pulled out of one eval result."""
    probe = res.get('dose_probe')
    ex_probe = res.get('exercise_probe')
    insulin, insulin_why = probe_sign_fracs(probe, 'insulin')
    carb, _carb_why = probe_sign_fracs(probe, 'carb')
    exercise, exercise_why = probe_sign_fracs(ex_probe, 'exercise')
    return {
        'insulin_sign': insulin,
        'insulin_reason': insulin_why,
        'carb_sign': carb,
        'exercise_sign': exercise,
        'exercise_reason': (ex_probe or {}).get('unavailable_reason') or exercise_why,
        'hypo_recall': {str(h): _metric(res, h, 'hypo', 'recall') for h in EVAL_HORIZONS},
        'band_cov50': {str(h): _metric(res, h, 'band_cov50') for h in EVAL_HORIZONS},
        'fan_order_violations': (res.get('proper') or {}).get('fan_order_violations'),
        'probe_ran': probe is not None,
    }


def _sign_check(name: str, now: dict[str, float | None], base: dict[str, float | None],
                cfg: GateConfig, reason: str | None = None) -> dict[str, Any]:
    """One dose-sign gate: no horizon may fall below the pretrained model's fraction.

    Compared per horizon, i.e. per ``d``.  A pooled sign fraction would let a gain at
    d = 1 pay for a loss at d = 4, which is the regime the alarm path consumes.
    """
    pairs = [(h, now.get(h), base.get(h)) for h in sorted(set(now) | set(base), key=int)]
    usable = [(h, v, b) for h, v, b in pairs if v is not None and b is not None]
    if not usable:
        return {'status': 'unavailable', 'pass': cfg.mode != 'block',
                'detail': f"{name}: {reason or 'no comparable horizon'}"}
    fails = [(h, v, b) for h, v, b in usable if v < b - cfg.sign_tolerance]
    detail = ' '.join(f"h{h}={v:.3f}/{b:.3f}" for h, v, b in usable)
    if fails:
        worst = ' '.join(f"h{h} {v:.3f} < {b:.3f}" for h, v, b in fails)
        return {'status': 'fail', 'pass': cfg.mode != 'block',
                'detail': f"{name} fell below the pretrained fraction: {worst}"}
    return {'status': 'pass', 'pass': True, 'detail': f"{name} {detail}"}


def _floor_check(name: str, values: dict[str, float | None],
                 floor: float | None) -> dict[str, Any]:
    """An absolute floor, applied per horizon. ``floor`` None = the gate is not set."""
    if floor is None:
        return {'status': 'off', 'pass': True,
                'detail': f"{name}: unset (no run settles a floor)"}
    usable = {h: v for h, v in values.items() if v is not None}
    if not usable:
        return {'status': 'unavailable', 'pass': False,
                'detail': f"{name}: not measured, floor {floor} cannot be checked"}
    fails = {h: v for h, v in usable.items() if v < floor}
    if fails:
        return {'status': 'fail', 'pass': False,
                'detail': f"{name} below {floor}: " +
                          ' '.join(f"h{h}={v:.3f}" for h, v in sorted(fails, key=int))}
    return {'status': 'pass', 'pass': True, 'detail': f"{name} >= {floor}"}


def evaluate_gates(values: dict[str, Any], baseline: dict[str, Any] | None,
                   cfg: GateConfig) -> dict[str, Any]:
    """Apply every admission gate. A candidate ships only if all of them pass.

    Gates are hard and independent — none trades against another, and none carries a
    weight.  Two of them are relative to the PRETRAINED model measured in this same
    run under the same protocol, which is why they need no threshold; the two
    absolute floors are unset by default and simply do not apply.
    """
    base = baseline or {}
    checks: dict[str, dict[str, Any]] = {}

    if cfg.mode == 'off':
        checks['dose_insulin_sign'] = {'status': 'off', 'pass': True,
                                       'detail': 'dose gate disabled (--dose-gate off)'}
    else:
        checks['dose_insulin_sign'] = _sign_check(
            'insulin correct-sign', values.get('insulin_sign', {}),
            base.get('insulin_sign', {}), cfg, values.get('insulin_reason'))

    if not cfg.exercise:
        # Off by default: a real record's exercise column is identically zero, so
        # this gate is only meaningful on simulator windows and costs a simulator
        # run to produce them.
        checks['dose_exercise_sign'] = {
            'status': 'off', 'pass': True,
            'detail': 'exercise sign gate off (--exercise-gate runs it on simulator '
                      'windows)'}
    else:
        checks['dose_exercise_sign'] = _sign_check(
            'exercise correct-sign', values.get('exercise_sign', {}),
            base.get('exercise_sign', {}), cfg, values.get('exercise_reason'))

    viol = values.get('fan_order_violations')
    if viol is None:
        checks['fan_order'] = {'status': 'unavailable', 'pass': True,
                               'detail': 'no quantile fan captured; ordering unchecked'}
    else:
        checks['fan_order'] = {'status': 'pass' if viol == 0 else 'fail',
                               'pass': viol == 0,
                               'detail': f"{viol} descending steps in the τ fan"}

    checks['min_hypo_recall'] = _floor_check(
        'hypo recall', values.get('hypo_recall', {}), cfg.min_hypo_recall)
    checks['min_band_cov50'] = _floor_check(
        'band cov50', values.get('band_cov50', {}), cfg.min_band_cov50)

    blocking = [k for k, c in checks.items() if not c['pass']]
    return {'pass': not blocking, 'blocking': blocking, 'checks': checks,
            'mode': cfg.mode}


def gate_summary(gates: dict[str, Any]) -> str:
    """One console line: the verdict plus every check that did not plainly pass."""
    if gates.get('not_evaluated'):
        return f"not evaluated ({gates['not_evaluated']})"
    if gates.get('pass'):
        noted = [c['detail'] for k, c in gates.get('checks', {}).items()
                 if c['status'] in ('fail', 'unavailable')]
        return 'admitted' + ('' if not noted else ' (' + '; '.join(noted) + ')')
    return 'REFUSED: ' + '; '.join(gates['checks'][k]['detail'] for k in gates['blocking'])


# ---------------------------------------------------------------------------- #
# Selection.
# ---------------------------------------------------------------------------- #
@dataclass
class Verdict:
    """What one eval decided: the scalar, the gates, and whether to keep going."""
    step: int
    scalar: float | None
    gates: dict[str, Any]
    improved: bool
    accepted: bool
    reason: str
    stop: bool = False


@dataclass
class Selector:
    """Rank admitted candidates by the strictly proper scalar; stop on patience.

    ``baseline`` is the PRETRAINED model's own gate values, measured at step 0 of
    this run under exactly the protocol every later candidate is measured under.
    Nothing is hardcoded: no reference number is carried in from another run, and a
    candidate is compared only against a measurement made here.

    ``patience`` counts consecutive evals that did not produce an admitted
    improvement.  None disables stopping.
    """
    cfg: GateConfig = field(default_factory=GateConfig)
    patience: int | None = None
    best: float | None = None
    best_step: int = -1
    best_gates: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    stale_evals: int = 0

    def set_baseline(self, res: dict[str, Any]) -> dict[str, Any]:
        """Freeze the pretrained model's gate values from the step-0 eval."""
        self.baseline = gate_values(res)
        return self.baseline

    def consider(self, step: int, res: dict[str, Any]) -> Verdict:
        """Score one eval: improved? admitted? out of patience?

        The gates are applied only to a candidate that improved on the incumbent,
        because only such a candidate can ship — and the probe they read is run on
        exactly the same condition.  An eval that did not improve records
        ``not_evaluated`` rather than a refusal it was never tested for.
        """
        sel = _selection_scalar(res)
        improved = sel is not None and (self.best is None or sel < self.best)
        gates = (evaluate_gates(gate_values(res), self.baseline, self.cfg) if improved
                 else {'pass': False, 'blocking': [], 'checks': {}, 'mode': self.cfg.mode,
                       'not_evaluated': 'not an improvement on the incumbent'})
        accepted = bool(improved and gates['pass'])
        if accepted:
            self.best, self.best_step, self.best_gates = sel, step, gates
            self.stale_evals = 0
            reason = f"new minimum {SELECTION_METRIC}@{SELECTION_HORIZON}={sel:.4f}"
        else:
            self.stale_evals += 1
            if sel is None:
                reason = (f"no {SELECTION_METRIC}@{SELECTION_HORIZON} in this eval — "
                          "nothing selected (there is no point-error fallback)")
            elif not improved:
                reason = f"{SELECTION_METRIC} {sel:.4f} not below {self.best:.4f}"
            else:
                reason = gate_summary(gates)
        stop = self.patience is not None and self.stale_evals >= self.patience
        return Verdict(step=step, scalar=sel, gates=gates, improved=bool(improved),
                       accepted=accepted, reason=reason, stop=stop)

    def announce_baseline(self) -> None:
        """Say at step 0 where a gate cannot be measured, not at the end.

        Under 'block' an unmeasurable gate refuses every candidate — the safe
        reading, and the one to hear once at the start rather than discover from an
        output that was never written.
        """
        base = self.baseline or {}
        if self.cfg.mode == 'off':
            print("  [gate] dose gate off — no dose-response condition on what ships")
            return
        ins = base.get('insulin_sign') or {}
        if ins:
            print("  [gate] pretrained insulin correct-sign fraction: "
                  + ' '.join(f"h{h}={v:.3f}" for h, v in sorted(ins.items(), key=lambda kv: int(kv[0]))
                             if v is not None))
        else:
            print(f"  [gate] NO baseline insulin sign fraction — "
                  f"{base.get('insulin_reason') or 'the probe produced none'}. Under "
                  f"--dose-gate {self.cfg.mode} "
                  + ("no checkpoint can be admitted until the probe can run."
                     if self.cfg.mode == 'block' else "this is recorded, not enforced."))
        if self.cfg.exercise and not (base.get('exercise_sign') or {}):
            print(f"  [gate] NO baseline exercise sign fraction — "
                  f"{base.get('exercise_reason') or 'the probe produced none'}")

    def meta(self) -> dict[str, Any]:
        """Selection provenance for ``finetune_meta``."""
        return {
            'metric': SELECTION_METRIC,
            'horizon_min': SELECTION_HORIZON,
            'd_patches': SELECTION_D_PATCHES,
            'scalar': self.best,
            'step': self.best_step,
            'patience_evals': self.patience,
            'gate_mode': self.cfg.mode,
            'gate_sign_tolerance': self.cfg.sign_tolerance,
            'gate_min_hypo_recall': self.cfg.min_hypo_recall,
            'gate_min_band_cov50': self.cfg.min_band_cov50,
            'gates': self.best_gates,
            'gate_baseline': self.baseline,
        }


# ---------------------------------------------------------------------------- #
# Per-run logging.
# ---------------------------------------------------------------------------- #
# Every eval logs BOTH point metrics beside the selection scalar, so a later run can
# be asked where each of them was best instead of being asked to trust one.  RMSE
# loses the headline slot and the selection role here, not a column.
_LOG_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (SELECTION_METRIC, (SELECTION_METRIC,)),
    ('rmse_point', ('rmse_point',)),
    ('rmse_winmean', ('rmse_winmean',)),
    ('mae_point', ('mae_point',)),
    ('mae_winmean', ('mae_winmean',)),
    ('mard', ('mard',)),
    ('clarke_AB', ('clarke_AB',)),
    ('skill_point', ('skill_point',)),
    ('hypo_recall', ('hypo', 'recall')),
    ('band_cov50', ('band_cov50',)),
    ('band_width', ('band_width',)),
)


def _slug(text: str) -> str:
    """Filename-safe form of a run label (patient ids carry '#', dates carry ':')."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(text)).strip('_') or 'run'


def run_log_path(driver: str, parts: list[Any]) -> str:
    """A CSV path no other run writes.

    One fixed path per driver is what left the medium personal fit with no per-step
    curve at all: the next run overwrote it.  The name carries the run's own
    identity plus a start timestamp, and the directory is gitignored.
    """
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    name = '-'.join(_slug(p) for p in [driver, *parts, stamp] if p not in (None, ''))
    return os.path.join(FINETUNE_LOG_DIR, name + '.csv')


class RunLog:
    """The per-step CSV for one fine-tune run.

    ``select_split`` adds a mirrored ``sel_*`` block for drivers that report one
    split and select on another (``finetune_personal.py``), so the log records both
    the number that was reported and the number that actually chose the checkpoint.
    """

    def __init__(self, path: str, select_split: bool = False) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.select_split = select_split
        self._fh = open(path, 'w', newline='')
        self._w = csv.writer(self._fh)
        header = ['step', 'loss_ema', 'sel', 'sel_admitted', 'gate_status']
        for name, _p in _LOG_METRICS:
            header += [f'{name}_{h}' for h in EVAL_HORIZONS]
        if select_split:
            for name, _p in _LOG_METRICS:
                header += [f'sel_{name}_{h}' for h in EVAL_HORIZONS]
        for arm in ('insulin', 'exercise'):
            header += [f'{arm}_sign_{h}' for h in EVAL_HORIZONS]
            header += [f'{arm}_sign_base_{h}' for h in EVAL_HORIZONS]
        self._w.writerow(header)
        self._fh.flush()

    @staticmethod
    def _cell(v: float | None) -> str:
        return '' if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.6f}"

    def write(self, step: int, loss_ema: float | None, res: dict[str, Any],
              verdict: Verdict, sel_res: dict[str, Any] | None = None,
              baseline: dict[str, Any] | None = None) -> None:
        """Append one eval's row."""
        gates = verdict.gates or {}
        status = ('pass' if gates.get('pass') else
                  '|'.join(gates.get('blocking', [])) or
                  ('not_evaluated' if gates.get('not_evaluated') else 'unknown'))
        row: list[Any] = [step, self._cell(loss_ema), self._cell(verdict.scalar),
                          int(verdict.accepted), status]
        for _name, path in _LOG_METRICS:
            row += [self._cell(_metric(res, h, *path)) for h in EVAL_HORIZONS]
        if self.select_split:
            src = sel_res if sel_res is not None else res
            for _name, path in _LOG_METRICS:
                row += [self._cell(_metric(src, h, *path)) for h in EVAL_HORIZONS]
        now = gate_values(sel_res if sel_res is not None else res)
        base = baseline or {}
        for arm in ('insulin', 'exercise'):
            key = f'{arm}_sign'
            row += [self._cell(now.get(key, {}).get(str(h))) for h in EVAL_HORIZONS]
            row += [self._cell(base.get(key, {}).get(str(h))) for h in EVAL_HORIZONS]
        self._w.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------- #
# Held-out evaluation (under the EMA shadow).
# ---------------------------------------------------------------------------- #
def probe_for_gates(model: T1DMAI, stats: dict[str, Any], device: torch.device,
                    res: dict[str, Any], gate_segs: list, cfg: GateConfig | None,
                    incumbent: float | None, force: bool) -> None:
    """Attach the dose-response probes to ``res``, in place, if this candidate could ship.

    MUST be called inside the caller's ``with ema.apply_to(model):`` block: the
    exporter ships the EMA weights, so a probe measured outside it gates a different
    model than the one that ships.

    The probe costs eleven forward passes per window, so it runs only where its
    verdict can change anything — the step-0 baseline, and any eval that improves on
    the incumbent scalar.  A candidate that is not an improvement is not selected
    whatever the probe says.
    """
    if cfg is None or cfg.mode == 'off' or not gate_segs:
        return
    sel = _selection_scalar(res)
    if not (force or incumbent is None or (sel is not None and sel < incumbent)):
        return
    res['dose_probe'] = dose_probe(model, stats, device, gate_segs, cfg)
    if not cfg.exercise:
        return
    sim_segs, why = sim_gate_segments(cfg)
    res['exercise_probe'] = ({'unavailable_reason': why} if not sim_segs else
                             dose_probe(model, stats, device, sim_segs, cfg))


def eval_heldout(
    model: T1DMAI, ema: ModelEMA, stats: dict[str, Any],
    cal_segs: list, test_segs: list, device: torch.device,
    gate_cfg: GateConfig | None = None, incumbent: float | None = None,
    force_probe: bool = False,
) -> dict[str, Any]:
    """Score the held-out patient under the EMA shadow.

    Collects cal/test prediction windows with ``collect_windows`` (always
    conditioned on announced doses) and runs the model-free ``evaluate_from_windows``
    suite.  The EMA shadow is swapped in for the duration so the score matches how
    the base models are evaluated by ``realdata.report.load_model``.

    The strictly proper selection scalar and the dose-response probe are produced
    inside that same shadow — the exporter ships EMA weights, so a probe measured
    outside it would gate a different model than the one that ships.  The probe runs
    on the CALIBRATION segments, never on the reported split.

    Returns the ``evaluate_from_windows`` result dict, plus ``proper`` and — where
    the gate probe ran — ``dose_probe`` / ``exercise_probe``.
    """
    with ema.apply_to(model):
        model.eval()
        cal_w = collect_windows(
            model, stats, cal_segs, device,
            stride_patches=FINETUNE_EVAL_CAL_STRIDE_PATCHES,
            max_per_patient=FINETUNE_EVAL_MAX_PER_PATIENT)
        test_w = collect_windows(
            model, stats, test_segs, device,
            stride_patches=FINETUNE_EVAL_TEST_STRIDE_PATCHES,
            max_per_patient=FINETUNE_EVAL_MAX_PER_PATIENT)
        res = evaluate_from_windows(cal_w, test_w)
        attach_proper_scores(res, test_w)
        probe_for_gates(model, stats, device, res, cal_segs, gate_cfg,
                        incumbent, force_probe)
        model.train()
    return res


def _has_eval_window(segments: list) -> bool:
    """True if at least one segment is long enough for an eval window (collect_windows rule)."""
    need = _CTX_EVAL_STEPS + _PRED_STEPS
    return any(((len(s) // config.PATCH_SIZE) * config.PATCH_SIZE) >= need for s in segments)


# ---------------------------------------------------------------------------- #
# Checkpoint output.
# ---------------------------------------------------------------------------- #
def _cpu_state(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach/clone a state dict onto CPU for a portable, crash-safe save."""
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def _output_label(checkpoint_path: str) -> str:
    """Provenance label: parent-dir name when the file is ``weights.pt``, else the stem."""
    abspath = os.path.abspath(checkpoint_path)
    fname = os.path.basename(abspath)
    if fname == 'weights.pt':
        return os.path.basename(os.path.dirname(abspath))
    return os.path.splitext(fname)[0]


# ---------------------------------------------------------------------------- #
# CLI.
# ---------------------------------------------------------------------------- #
def add_selection_args(p: argparse.ArgumentParser) -> None:
    """Stopping-rule and admission-gate flags, shared by all three drivers."""
    p.add_argument('--patience', type=int, default=FINETUNE_PATIENCE_EVALS,
                   help="stop after this many evals with no admitted improvement "
                        "(default: unset — run the full step budget)")
    p.add_argument('--dose-gate', choices=['block', 'warn', 'off'], default=DOSE_GATE_MODE,
                   help="dose-response gate: block refuses to ship a checkpoint whose "
                        "insulin correct-sign fraction fell below the pretrained model's")
    p.add_argument('--gate-sign-tolerance', type=float, default=DOSE_GATE_SIGN_TOLERANCE,
                   help="slack below the pretrained sign fraction (default 0.0: not below)")
    p.add_argument('--gate-stride-patches', type=int, default=DOSE_GATE_STRIDE_PATCHES,
                   help="dose-probe window stride, in patches (a compute budget)")
    p.add_argument('--gate-max-windows', type=int, default=DOSE_GATE_MAX_WINDOWS_PER_SEG,
                   help="dose-probe windows per segment (a compute budget)")
    p.add_argument('--exercise-gate', action='store_true', default=EXERCISE_GATE_ENABLED,
                   help="also gate the exercise sign, on SIMULATOR windows only "
                        "(a real record's exercise column is zeros)")
    p.add_argument('--exercise-gate-patients', type=int, default=EXERCISE_GATE_SIM_PATIENTS,
                   help="simulator patients the exercise gate probes (a compute budget)")
    p.add_argument('--gate-min-hypo-recall', type=float, default=GATE_MIN_HYPO_RECALL,
                   help="absolute hypo-recall floor (default: unset, not enforced)")
    p.add_argument('--gate-min-band-cov50', type=float, default=GATE_MIN_BAND_COV50,
                   help="absolute band-coverage floor (default: unset, not enforced)")


def gate_config_from_args(args: argparse.Namespace) -> GateConfig:
    """Build the gate configuration from the shared flags."""
    return GateConfig(
        mode=args.dose_gate, sign_tolerance=float(args.gate_sign_tolerance),
        stride_patches=int(args.gate_stride_patches),
        max_windows_per_seg=int(args.gate_max_windows),
        exercise=bool(args.exercise_gate),
        exercise_sim_patients=int(args.exercise_gate_patients),
        min_hypo_recall=args.gate_min_hypo_recall,
        min_band_cov50=args.gate_min_band_cov50)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leave-one-patient-out fine-tuning of a T1DMAI checkpoint on real CGM data.")
    p.add_argument('--checkpoint', required=True, help="pretrained checkpoint .pt path")
    p.add_argument('--dataset', required=True, choices=['ohiot1dm', 'azt1d', 'shanghai'])
    p.add_argument('--mode', choices=['transfer', 'personalize'], default='transfer')
    p.add_argument('--holdout', default=None, help="held-out patient id (default: first sorted)")
    p.add_argument('--steps', type=int, default=FINETUNE_TOTAL_STEPS)
    p.add_argument('--batch-size', type=int, default=FINETUNE_BATCH_SIZE)
    p.add_argument('--lr-scale', type=float, default=FINETUNE_LR_SCALE)
    p.add_argument('--warmup', type=int, default=FINETUNE_WARMUP_STEPS)
    p.add_argument('--eval-interval', type=int, default=FINETUNE_EVAL_INTERVAL)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=FINETUNE_SEED)
    p.add_argument('--out', default=None, help="output checkpoint path (default: auto)")
    p.add_argument('--write-best', action='store_true',
                   help="also write checkpoints/t1dmai_best.pt for the metrics/ suite")
    add_selection_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    total_steps = int(args.steps)
    batch_size = int(args.batch_size)
    lr_scale = float(args.lr_scale)
    warmup_steps = int(args.warmup)
    eval_interval = int(args.eval_interval)
    seed = int(args.seed)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Dataset ---------------------------------------------------------- #
    root_dir = os.path.join(REPO_ROOT, DATASET_SUBPATHS[args.dataset])
    segs_all = load_dataset(args.dataset, root_dir=root_dir)
    assert len(segs_all) > 0, f"no segments loaded from {root_dir}"
    patients = sorted({s.patient for s in segs_all})
    holdout = args.holdout if args.holdout is not None else patients[0]
    assert holdout in patients, f"holdout {holdout!r} not in patients {patients}"

    heldout_segs = [s for s in segs_all if s.patient == holdout]
    other_segs = [s for s in segs_all if s.patient != holdout]

    cal_segs, test_segs = split_segments(heldout_segs, args.dataset)
    if args.mode == 'transfer':
        ft_segs = other_segs
    else:  # personalize
        ft_segs = cal_segs

    assert len(ft_segs) > 0, (
        f"no fine-tune segments for mode={args.mode}, holdout={holdout} "
        f"(transfer needs >1 patient; personalize needs a non-empty cal split)"
    )
    if not _has_eval_window(test_segs):
        raise SystemExit(
            f"held-out patient {holdout!r} yields no test window "
            f"(need a contiguous run >= {_CTX_EVAL_STEPS + _PRED_STEPS} steps after the "
            f"split). The patient may be too short, or the cal/test cut left no room — "
            f"pick a different --holdout."
        )

    # --- Model ------------------------------------------------------------ #
    model, ckpt, stats = load_checkpoint(args.checkpoint, device)
    arch_version = ckpt.get('arch_version')
    loss_schema = ckpt.get('loss_schema')
    # The base checkpoint's resolved config — architecture, mask sampler, and what
    # pretraining ran under. The fine-tune changes none of it, and the two guards
    # above have just confirmed the live config agrees on every key it carries, so
    # it travels to the output verbatim rather than being rebuilt from the live
    # config. Without it the provenance ends at this script and the output cannot
    # be guarded, exported or carded. What is specific to THIS run —
    # steps, LR scale, dataset, holdout, selection — is finetune_meta's.
    base_training_config = ckpt.get('training_config', {}) or {}

    # Learned Kendall-Gal uncertainty weighting for the pinball/DILATE combine,
    # mirroring pretraining. Constructed fresh (not resumed from the base
    # checkpoint — train.py likewise has no resume path); its two log-variance
    # scalars live on this SEPARATE module and are EMA-excluded structurally by
    # never being passed to ModelEMA.
    weighting = KendallGalWeighting().to(device)

    # --- Training data ---------------------------------------------------- #
    dataset = RealSegmentDataset(ft_segs, stats, seed=seed)
    assert len(dataset) > 0, (
        f"no training windows from {len(ft_segs)} segments — segments shorter than "
        f"{_CTX_MIN_STEPS + _PRED_STEPS} steps"
    )
    # With drop_last=True a split smaller than the batch yields ZERO batches, which
    # would surface as an unhandled StopIteration in the train loop's re-iter. Clamp
    # the batch to the available windows (realistic in --mode personalize, where the
    # fine-tune set is a single patient's cal split) so it still trains.
    if len(dataset) < batch_size:
        print(f"[finetune] training windows ({len(dataset)}) < batch_size ({batch_size}); "
              f"clamping batch_size to {len(dataset)}")
        batch_size = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=(device.type == 'cuda'))

    # --- Optimizers / schedule / EMA -------------------------------------- #
    muon_lr = config.MUON_LR * lr_scale
    adam_lr = config.ADAM_LR * lr_scale
    muon_opt, adam_opt = build_optimizers(model, weighting, muon_lr, adam_lr)
    # ModelEMA wraps ONLY the model; the two Kendall-Gal log-variance params on
    # `weighting` are EMA-excluded by living off-model (never passed to ModelEMA).
    ema = ModelEMA(model, decay=FINETUNE_EMA_DECAY).to(device)

    # --- Logging ---------------------------------------------------------- #
    log = RunLog(run_log_path('finetune', [_output_label(args.checkpoint), args.dataset,
                                           args.mode, holdout, f"seed{seed}"]))

    # --- Selection -------------------------------------------------------- #
    gate_cfg = gate_config_from_args(args)
    selector = Selector(cfg=gate_cfg, patience=args.patience)

    print(f"[finetune] dataset={args.dataset} mode={args.mode} holdout={holdout} "
          f"device={device.type}")
    print(f"[finetune] ft_segs={len(ft_segs)} windows={len(dataset)} "
          f"cal_segs={len(cal_segs)} test_segs={len(test_segs)} "
          f"steps={total_steps} bs={batch_size} lr_scale={lr_scale}")
    print(f"[finetune] select on {SELECTION_METRIC}@{SELECTION_HORIZON}m "
          f"(d={SELECTION_D_PATCHES}, one-sided)  dose-gate={gate_cfg.mode}  "
          f"patience={args.patience if args.patience is not None else 'unset'}")
    print(f"[finetune] log: {log.path}")

    best_model_sd: dict[str, torch.Tensor] | None = None
    best_ema_sd: dict[str, torch.Tensor] | None = None
    best_weighting_sd: dict[str, torch.Tensor] | None = None
    best_summ: dict[str, Any] | None = None
    baseline_summ: dict[str, Any] | None = None

    # --- Output path / writer --------------------------------------------- #
    label = _output_label(args.checkpoint)
    suffix = '' if args.mode == 'transfer' else '-personalize'
    out_default = os.path.join(REPO_ROOT, 'finetune',
                               f"{label}-finetune-{args.dataset}{suffix}.pt")
    out_path = args.out if args.out is not None else out_default

    def _write_output() -> None:
        assert (best_model_sd is not None and best_ema_sd is not None
                and best_weighting_sd is not None)
        save_dict = {
            'arch_version': arch_version,
            'loss_schema': loss_schema,
            'step': selector.best_step,
            'model_state_dict': best_model_sd,
            'model_ema_state_dict': best_ema_sd,
            'weighting_state_dict': best_weighting_sd,
            'training_config': base_training_config,
            'normalization_stats': stats,
            'finetune_meta': {
                'dataset': args.dataset, 'mode': args.mode, 'holdout': holdout,
                'base_checkpoint': os.path.abspath(args.checkpoint),
                'total_steps': total_steps, 'lr_scale': lr_scale,
                'baseline_heldout': baseline_summ, 'best_heldout': best_summ,
                # How this checkpoint was chosen, and what it had to clear to ship.
                'selection': selector.meta(),
                'log_csv': os.path.basename(log.path),
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        torch.save(save_dict, out_path)
        if args.write_best:
            best_dir = os.path.join(REPO_ROOT, 'checkpoints')
            os.makedirs(best_dir, exist_ok=True)
            torch.save(save_dict, os.path.join(best_dir, 't1dmai_best.pt'))

    loss_ema: float | None = None

    def _report_eval(step: int, res: dict[str, Any], verdict: Verdict) -> None:
        """Log one eval to the per-run CSV and summarize it on the console.

        The headline cell is the strictly proper scalar; both point errors go to the
        CSV every eval, so where each of them was best is a question the log can
        answer instead of one the operator has to trust an answer to.
        """
        log.write(step, loss_ema, res, verdict, baseline=selector.baseline)
        cells = []
        for h in EVAL_HORIZONS:
            cells.append(
                f"h{h}(d{horizon_d_patches(h)}) {SELECTION_METRIC}="
                f"{_fmt(_metric(res, h, SELECTION_METRIC))} "
                f"rmse={_fmt(_metric(res, h, 'rmse_point'))} "
                f"rmseWM={_fmt(_metric(res, h, 'rmse_winmean'))} "
                f"AB={_fmt(_metric(res, h, 'clarke_AB'))}")
        tag = 'baseline' if step == 0 else f"step {step}"
        print(f"[eval {tag}] loss_ema={_fmt(loss_ema)} n_test={res.get('n_test_windows')} "
              f"sel={_fmt(verdict.scalar)} | " + " | ".join(cells))
        print(f"           gates: {gate_summary(verdict.gates)}")

    def _consider(step: int, res: dict[str, Any]) -> Verdict:
        """Rank, gate and (if admitted) snapshot + write this candidate."""
        nonlocal best_model_sd, best_ema_sd, best_weighting_sd, best_summ
        verdict = selector.consider(step, res)
        if verdict.accepted:
            best_model_sd = _cpu_state(model.state_dict())
            best_ema_sd = _cpu_state(ema.state_dict())
            best_weighting_sd = _cpu_state(weighting.state_dict())
            best_summ = _summarize_heldout(res)
            _write_output()
            print(f"  [best] {verdict.reason} at step {step} -> wrote {out_path}")
        elif verdict.improved and verdict.gates.get('blocking'):
            print(f"  [gate] step {step} not selected — {verdict.reason}")
        return verdict

    try:
        # --- Baseline (pre-finetune) eval at step 0 ----------------------- #
        # The step-0 model IS the pretrained checkpoint (the EMA shadow is
        # initialised from it), so this eval measures the gate baseline under
        # exactly the protocol every later candidate is measured under.
        res0 = eval_heldout(model, ema, stats, cal_segs, test_segs, device,
                            gate_cfg=gate_cfg, force_probe=True)
        selector.set_baseline(res0)
        selector.announce_baseline()
        baseline_summ = _summarize_heldout(res0)
        v0 = _consider(0, res0)
        _report_eval(0, res0, v0)

        # --- Fine-tune loop ---------------------------------------------- #
        data_iter = iter(loader)
        consecutive_nan = 0
        step = 0
        stopped_early = False
        last_eval_step = 0
        while step < total_steps:
            model.train()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            patches = batch['patches'].to(device, non_blocking=True)
            attn_mask = batch['attn_mask'].to(device, non_blocking=True)
            _bfd = batch['bg_formula_data']
            # The masked set, all (B, M): the head gathers by index and the loss
            # discards the padded slots by ``slot_valid``. Dropping the latter
            # anywhere on the loss path supervises those slots against patch 0.
            anchor_bg = _bfd['anchor_bg'].to(device, non_blocking=True).float()   # (B, M) mg/dL
            mask_idx = _bfd['mask_idx'].to(device, non_blocking=True)             # (B, M) int64
            slot_valid = _bfd['valid'].to(device, non_blocking=True)              # (B, M) bool
            slot_hour = _bfd['slot_hour'].to(device, non_blocking=True).float()   # (B, M) hours
            # One target row per head slot, mg/dL — not a trailing slice of the
            # BG trajectory, which no longer indexes the supervised patches.
            targets = batch['targets'].to(device, non_blocking=True).float()      # (B, M, S)

            # Cross-window probe input (window k+1) — TIME-PROBE-only overhead,
            # fully skipped when the penalty is off (collate ships the key iff
            # enabled + weight > 0).
            next_window = None
            if config.TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
                _nw = batch.get('next_window')
                if _nw is not None:
                    next_window = {
                        'patches': _nw['patches'].to(device, non_blocking=True),
                        # Window k+1's own attention mask, built by collate from
                        # window k+1's masked set.
                        'attn_mask': _nw['attn_mask'].to(device, non_blocking=True),
                        'anchor_bg': _nw['anchor_bg'].to(device, non_blocking=True).float(),
                        'mask_idx': _nw['mask_idx'].to(device, non_blocking=True),
                        'valid': _nw['valid'].to(device, non_blocking=True),
                    }

            def _halve_optimizer_state() -> None:
                for grp in muon_opt.param_groups:
                    for p in grp['params']:
                        st = muon_opt.state.get(p, {})
                        if 'momentum_buffer' in st:
                            st['momentum_buffer'].mul_(0.5)
                for grp in adam_opt.param_groups:
                    for p in grp['params']:
                        st = adam_opt.state.get(p, {})
                        if 'exp_avg' in st:
                            st['exp_avg'].mul_(0.5)

            def _maybe_restore_from_ema(reason: str) -> bool:
                nonlocal consecutive_nan
                if consecutive_nan >= CONSECUTIVE_NAN_RESTORE:
                    model.load_state_dict(ema.state_dict(), strict=False)
                    muon_opt.state.clear()
                    adam_opt.state.clear()
                    print(f"  [RECOVERY] {consecutive_nan} consecutive {reason} — "
                          f"restored model from EMA shadow (optimizer state cleared)")
                    consecutive_nan = 0
                    return True
                return False

            def _skip_nonfinite_step(reason: str) -> None:
                nonlocal consecutive_nan, loss_ema
                consecutive_nan += 1
                print(f"  [WARNING] {reason} at step {step} "
                      f"(consecutive: {consecutive_nan}) — skipping step")
                _maybe_restore_from_ema(reason)
                muon_opt.zero_grad(set_to_none=True)
                adam_opt.zero_grad(set_to_none=True)
                _halve_optimizer_state()
                # Mirror train.py: a skipped (non-finite) step bumps loss_ema toward a
                # 1.0 penalty so a NaN streak is visible in finetune_log.csv instead of
                # silently freezing at the last good value.
                loss_ema = 1.0 if loss_ema is None else 0.98 * loss_ema + 0.02 * 1.0

            try:
                q_tau, median, time_pred = model(
                    patches, attn_mask, anchor_bg, mask_idx, return_time=True)
                q_tau = q_tau.float()
                median = median.float()
                loss_total, _parts = risk_total_loss(
                    q_tau, median, targets, weighting,
                    valid=slot_valid, mask_idx=mask_idx,
                )

                # Time-of-day probe MSE — added to the BACKWARD tensor ONLY. It never
                # touches loss_total / loss_ema / the CSV / selection (those stay on BG
                # accuracy), but with TIME_PROBE_DETACH=False its gradient shapes the
                # shared trunk, co-training it exactly as pretraining does.
                _tod_extra = loss_total.new_zeros(())
                _tod_loss_val = float('nan')   # raw probe loss, for the per-step log line
                if time_pred is not None:
                    # The target is the slot's OWN clock, shipped per slot: slot j
                    # is patch mask_idx[:, j], so a fixed pred_start_hour + 0.5 * j
                    # is off by (mask_idx[j] - n_ctx - j) * 0.5 h for any masked set
                    # other than the right-edge span, with every shape matching.
                    # Padded slots are dropped rather than trained against patch 0's
                    # hour.
                    _sel = slot_valid.reshape(-1)
                    _tod_ce = time_of_day_bin_ce(
                        time_pred.reshape(-1, config.TIME_PROBE_N_BINS)[_sel],
                        (slot_hour.reshape(-1)[_sel]) % 24.0,
                        config.TIME_PROBE_N_BINS, config.TIME_PROBE_LABEL_SMOOTH_BINS,
                    )
                    _tod_loss = _tod_ce
                    _tod_loss_val = float(_tod_ce.detach())   # logged CE (pre cross-window)

                    # Cross-window (paired-window) phase-advance penalty. A 2nd forward
                    # on window k+1 (batch['next_window'], teacher-forced) couples the
                    # two INDEPENDENT clocks so the rolling clock advances by exactly
                    # PREDICTION_HORIZON_HOURS across the seam — same coupling the
                    # pretraining loop applies. Masked by the validity flag, subsampled
                    # by the fraction knob. Inside this try => non-finite propagates.
                    if (next_window is not None
                            and config.TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0
                            and bool(next_window['valid'].any())):
                        B_nw = next_window['patches'].shape[0]
                        n_sub = (B_nw if config.TIME_PROBE_CROSS_WINDOW_FRACTION >= 1.0
                                 else max(1, math.ceil(config.TIME_PROBE_CROSS_WINDOW_FRACTION * B_nw)))
                        nw_valid_s = next_window['valid'][:n_sub]
                        if bool(nw_valid_s.any()):
                            # The attention mask is a function of the masked set, so
                            # this forward takes the mask collate ships beside window
                            # k+1's own patches. Window k's agrees only while the two
                            # masked sets coincide, and the two windows share n_ctx and
                            # so the mask SHAPE — the wrong window's mask raises nothing
                            # and instead lets the forward attend patches its own feat-4
                            # bit announces as withheld, which this backward carries
                            # into the shared trunk through _tod_extra.
                            _, _, time_pred_next = model(
                                next_window['patches'][:n_sub],
                                next_window['attn_mask'][:n_sub],
                                next_window['anchor_bg'][:n_sub],
                                next_window['mask_idx'][:n_sub], return_time=True,
                            )
                            _tod_xwin = time_cross_window_consistency_loss(
                                time_pred[:n_sub], time_pred_next, config.TIME_PROBE_N_BINS,
                                config.PREDICTION_HORIZON_HOURS, valid=nw_valid_s,
                            )
                            if torch.isfinite(_tod_xwin):
                                _tod_loss = _tod_loss + config.TIME_PROBE_CROSS_WINDOW_WEIGHT * _tod_xwin
                    if torch.isfinite(_tod_loss):
                        _tod_extra = config.TIME_PROBE_LOSS_WEIGHT * _tod_loss
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
                config.GRADIENT_CLIP_NORM, error_if_nonfinite=False)

            if torch.isfinite(grad_norm):
                update_lr(muon_opt, adam_opt, step, muon_lr, adam_lr,
                          warmup_steps, total_steps, FINETUNE_LR_MIN_RATIO)
                muon_opt.step()
                adam_opt.step()
                ema.update(model)
                consecutive_nan = 0
                loss_val = float(loss_total.item())
                loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
            else:
                consecutive_nan += 1
                print(f"  [WARNING] NaN/Inf gradient at step {step} "
                      f"(consecutive: {consecutive_nan}), skipping optimizer step")
                _maybe_restore_from_ema("NaN gradients")
                _halve_optimizer_state()
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            step += 1
            if step % FINETUNE_LOG_INTERVAL == 0:
                lr_now = muon_opt.param_groups[0]['lr']
                print(f"  step {step}/{total_steps} loss_ema={_fmt(loss_ema)} "
                      f"loss_tod={_tod_loss_val:.4f} lr_muon={lr_now:.2e}")
            if step % eval_interval == 0 and step < total_steps:
                res = eval_heldout(model, ema, stats, cal_segs, test_segs, device,
                                   gate_cfg=gate_cfg, incumbent=selector.best)
                verdict = _consider(step, res)
                _report_eval(step, res, verdict)
                last_eval_step = step
                if verdict.stop:
                    stopped_early = True
                    print(f"[finetune] patience {args.patience} exhausted at step {step} "
                          f"(best step {selector.best_step}) — stopping early")
                    break

        # --- Final eval --------------------------------------------------- #
        if not stopped_early and last_eval_step != step:
            res_final = eval_heldout(model, ema, stats, cal_segs, test_segs, device,
                                     gate_cfg=gate_cfg, incumbent=selector.best)
            verdict = _consider(step, res_final)
            _report_eval(step, res_final, verdict)
    finally:
        log.close()

    if best_model_sd is not None:
        _write_output()

    # --- Final report ----------------------------------------------------- #
    print()
    print("=" * 72)
    print(f"output checkpoint: {out_path}" if best_model_sd is not None else
          "NO checkpoint written: no candidate was both an improvement and admitted "
          "by every gate. The base checkpoint is unchanged.")
    print(f"best step: {selector.best_step}  ({SELECTION_METRIC}@{SELECTION_HORIZON}m "
          f"[d={SELECTION_D_PATCHES}]={_fmt(selector.best)})")
    print(f"per-step log: {log.path}")
    print(f"held-out patient: {holdout}  (dataset={args.dataset}, mode={args.mode})")
    print("per-horizon held-out  baseline -> best (Δ):")
    for h in EVAL_HORIZONS:
        for key in (SELECTION_METRIC, 'rmse_point', 'rmse_winmean'):
            base = None if baseline_summ is None else baseline_summ[str(h)].get(key)
            best = None if best_summ is None else best_summ[str(h)].get(key)
            delta = (None if (base is None or best is None) else best - base)
            print(f"  h{h:>3} {key:<12}: {_fmt(base):>8} -> {_fmt(best):>8}"
                  + ("" if delta is None else f"  (Δ {delta:+.3f})"))
    if selector.best_gates is not None:
        print(f"admission gates at the selected step: {gate_summary(selector.best_gates)}")
    print("-" * 72)
    if args.write_best:
        print("checkpoints/t1dmai_best.pt was overwritten. Evaluate ALL patients with:")
        print("  ./metrics/rebuild_all.sh        # run from the repo root")
    else:
        print("To evaluate with the existing all-patients suite (run from the repo root):")
        print(f"  cp {out_path} checkpoints/t1dmai_best.pt && ./metrics/rebuild_all.sh")
    print("rebuild_all.sh reads the hardcoded checkpoints/t1dmai_best.pt and scores ALL "
          "patients; compare that average against the held-out number above for the "
          "apples-to-apples generalization signal.")
    print("=" * 72)


if __name__ == '__main__':
    main()
