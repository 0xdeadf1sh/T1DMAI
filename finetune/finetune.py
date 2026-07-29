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

Runs from either ``finetune/`` or the repo root — ``REPO_ROOT`` is resolved from
this file's location and prepended to ``sys.path``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
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
from data import collate_fn
from realdata import load_dataset
from realdata.features import build_feature_stack, smoothed_cgm
from realdata.calibrate import collect_windows
from realdata.run_eval import split_segments, evaluate_from_windows
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
SELECTION_METRIC: str = 'rmse_winmean'

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
    instead).
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

    Per segment the normalized 3-feature stack, the raw (bg-clamped) mg/dL CGM, and
    the hour-of-day are precomputed once; ``__getitem__`` slices a context +
    prediction window and assembles the same dict ``data._build_sample`` emits, so
    ``data.collate_fn`` consumes it unchanged.  All inputs are in normalized
    z-space (feat 0 is z(f(bg)), Kovatchev risk-space then z-score); the BG targets
    / ``last_bg`` are mg/dL.
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

    def __getitem__(self, i: int) -> dict[str, Any]:
        """Assemble one training sample.

        Returns a dict with ``patches`` (n_ctx+P, PATCH_DIM) normalized,
        ``targets`` (P, S) mg/dL, ``n_context_patches`` int, and ``bg_formula_data``
        carrying ``last_bg`` (mg/dL), ``true_bg_trajectory`` (PRED_STEPS,) and
        ``extended_true_bg_trajectory`` (LONG,) mg/dL, plus ``pred_start_hour``.
        """
        seg_idx, pred_start = self._index[i]
        feats = self._feats[seg_idx]
        bg_sm = self._bg_sm[seg_idx]
        hour = self._hour[seg_idx]

        # Variable context: uniform in [MIN, min(MAX, available)] patches.
        hi = min(config.MAX_CONTEXT_PATCHES, pred_start // config.PATCH_SIZE)
        n_ctx = int(self._rng.integers(config.MIN_CONTEXT_PATCHES, hi + 1))
        ctx_steps = n_ctx * config.PATCH_SIZE

        context = feats[pred_start - ctx_steps:pred_start].reshape(
            n_ctx, config.PATCH_SIZE, config.N_INPUT_FEATURES)
        ctx_flat = context.reshape(n_ctx, config.PATCH_DIM)

        pred = feats[pred_start:pred_start + _PRED_STEPS].reshape(
            config.PREDICTION_PATCHES, config.PATCH_SIZE, config.N_INPUT_FEATURES).copy()
        for f in config.NON_MASKABLE_FEATS:               # bg feat0 zeroed; carb/insulin kept
            pred[:, :, f] = 0.0
        pred_flat = pred.reshape(config.PREDICTION_PATCHES, config.PATCH_DIM)

        patches = torch.from_numpy(
            np.concatenate([ctx_flat, pred_flat], axis=0)).float()
        assert patches.shape[-1] == config.PATCH_DIM, (
            f"patch width {patches.shape[-1]} != PATCH_DIM {config.PATCH_DIM}"
        )

        targets = torch.from_numpy(
            bg_sm[pred_start:pred_start + _PRED_STEPS].reshape(
                config.PREDICTION_PATCHES, config.PATCH_SIZE).copy()).float()

        last_bg = float(min(max(bg_sm[pred_start - 1], BG_CLAMP_MIN), BG_CLAMP_MAX))
        assert last_bg >= BG_CLAMP_MIN - 1e-3, (
            f"last_bg {last_bg} below BG_CLAMP_MIN — non-mg/dL value in the anchor"
        )

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
                'last_bg': last_bg,
                'true_bg_trajectory': true_bg_trajectory,
                'extended_true_bg_trajectory': extended_true_bg_trajectory,
                'pred_start_hour': pred_start_hour,
            },
        }

        # === Cross-window (paired-window) time-of-day probe input (window k+1) ===
        # Window k shifted forward by exactly PREDICTION_PATCHES on the SAME
        # raw segment (teacher-forced re-slice, not autoregressive).
        # Same n_ctx => identical seq_len / left-pad / attn_mask as window k, which
        # data.collate_fn and both training loops reuse for the 2nd forward. Its
        # pred zone keeps bg (feat 0) zeroed so no future bg leaks; carb/insulin
        # stay announced. Gated + shape-matched exactly like data.build_sample so
        # the shared collate batches it unchanged; when the future runs off the
        # segment end it ships a masked-out zero placeholder (valid=False).
        if config.TIME_PROBE_ENABLED and config.TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
            seq_len = n_ctx + config.PREDICTION_PATCHES
            next_pred_start = pred_start + _PRED_STEPS
            next_valid = next_pred_start + _PRED_STEPS <= feats.shape[0]
            if next_valid:
                nxt_ctx = feats[next_pred_start - ctx_steps:next_pred_start].reshape(
                    n_ctx, config.PATCH_SIZE, config.N_INPUT_FEATURES).reshape(
                    n_ctx, config.PATCH_DIM)
                nxt_pred = feats[next_pred_start:next_pred_start + _PRED_STEPS].reshape(
                    config.PREDICTION_PATCHES, config.PATCH_SIZE, config.N_INPUT_FEATURES).copy()
                for f in config.NON_MASKABLE_FEATS:          # bg feat0 zeroed
                    nxt_pred[:, :, f] = 0.0
                nxt_pred_flat = nxt_pred.reshape(config.PREDICTION_PATCHES, config.PATCH_DIM)
                next_patches = torch.from_numpy(
                    np.concatenate([nxt_ctx, nxt_pred_flat], axis=0)).float()
                next_last_bg = float(min(max(bg_sm[next_pred_start - 1], BG_CLAMP_MIN), BG_CLAMP_MAX))
                next_pred_start_hour = float(hour[next_pred_start])
            else:
                next_patches = torch.zeros(seq_len, config.PATCH_DIM, dtype=torch.float32)
                next_last_bg = last_bg
                next_pred_start_hour = pred_start_hour
            assert next_patches.shape == (seq_len, config.PATCH_DIM), (
                f"next_window patch shape {tuple(next_patches.shape)} != {(seq_len, config.PATCH_DIM)}"
            )
            sample['next_window'] = {
                'patches': next_patches,
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
        out[str(h)] = {
            'rmse_point': _metric(res, h, 'rmse_point'),
            'rmse_winmean': _metric(res, h, 'rmse_winmean'),
            'mard': _metric(res, h, 'mard'),
            'clarke_AB': _metric(res, h, 'clarke_AB'),
            'skill_point': _metric(res, h, 'skill_point'),
            'hypo_recall': _metric(res, h, 'hypo', 'recall'),
            'hyper_recall': _metric(res, h, 'hyper', 'recall'),
        }
    return out


def _selection_scalar(res: dict[str, Any]) -> float | None:
    """Checkpoint-selection scalar: SELECTION_HORIZON rmse_winmean, else mean rmse_point."""
    v = _metric(res, SELECTION_HORIZON, SELECTION_METRIC)
    if v is not None and math.isfinite(v):
        return v
    pts = [_metric(res, h, 'rmse_point') for h in EVAL_HORIZONS]
    pts = [p for p in pts if p is not None and math.isfinite(p)]
    return float(sum(pts) / len(pts)) if pts else None


def _fmt(v: float | None) -> str:
    """Render a possibly-None metric for the console."""
    return 'nan' if v is None else f"{v:.3f}"


# ---------------------------------------------------------------------------- #
# Held-out evaluation (under the EMA shadow).
# ---------------------------------------------------------------------------- #
def eval_heldout(
    model: T1DMAI, ema: ModelEMA, stats: dict[str, Any],
    cal_segs: list, test_segs: list, device: torch.device,
) -> dict[str, Any]:
    """Score the held-out patient under the EMA shadow.

    Collects cal/test prediction windows with ``collect_windows`` (always
    conditioned on announced doses) and runs the model-free ``evaluate_from_windows``
    suite.  The EMA shadow is swapped in for the duration so the score matches how
    the base models are evaluated by ``realdata.report.load_model``.

    Returns the ``evaluate_from_windows`` result dict.
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
    log_path = os.path.join(REPO_ROOT, 'finetune', 'finetune_log.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    header = ['step', 'loss_ema']
    for h in EVAL_HORIZONS:
        header += [f'rmse_point_{h}', f'mard_{h}', f'clarke_AB_{h}', f'hypo_recall_{h}']
    log_writer.writerow(header)
    log_file.flush()

    print(f"[finetune] dataset={args.dataset} mode={args.mode} holdout={holdout} "
          f"device={device.type}")
    print(f"[finetune] ft_segs={len(ft_segs)} windows={len(dataset)} "
          f"cal_segs={len(cal_segs)} test_segs={len(test_segs)} "
          f"steps={total_steps} bs={batch_size} lr_scale={lr_scale}")

    best_sel: float | None = None
    best_step: int = -1
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
            'step': best_step,
            'model_state_dict': best_model_sd,
            'model_ema_state_dict': best_ema_sd,
            'weighting_state_dict': best_weighting_sd,
            'normalization_stats': stats,
            'finetune_meta': {
                'dataset': args.dataset, 'mode': args.mode, 'holdout': holdout,
                'base_checkpoint': os.path.abspath(args.checkpoint),
                'total_steps': total_steps, 'lr_scale': lr_scale,
                'baseline_heldout': baseline_summ, 'best_heldout': best_summ,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        torch.save(save_dict, out_path)
        if args.write_best:
            best_dir = os.path.join(REPO_ROOT, 'checkpoints')
            os.makedirs(best_dir, exist_ok=True)
            torch.save(save_dict, os.path.join(best_dir, 't1dmai_best.pt'))

    loss_ema: float | None = None

    def _log_eval(step: int, res: dict[str, Any]) -> None:
        row: list[Any] = [step, '' if loss_ema is None else f"{loss_ema:.4f}"]
        cells: list[str] = []
        for h in EVAL_HORIZONS:
            rmse = _metric(res, h, 'rmse_point')
            mard = _metric(res, h, 'mard')
            cab = _metric(res, h, 'clarke_AB')
            hyp = _metric(res, h, 'hypo', 'recall')
            row += ['' if v is None else f"{v:.6f}" for v in (rmse, mard, cab, hyp)]
            cells.append(f"h{h} rmse={_fmt(rmse)} mard={_fmt(mard)} AB={_fmt(cab)} "
                         f"hypoR={_fmt(hyp)}")
        log_writer.writerow(row)
        log_file.flush()
        sel = _selection_scalar(res)
        tag = 'baseline' if step == 0 else f"step {step}"
        print(f"[eval {tag}] loss_ema={_fmt(loss_ema)} n_test={res.get('n_test_windows')} "
              f"sel={_fmt(sel)} | " + " | ".join(cells))

    def _maybe_select(step: int, res: dict[str, Any]) -> None:
        nonlocal best_sel, best_step, best_model_sd, best_ema_sd, best_weighting_sd, best_summ
        sel = _selection_scalar(res)
        if sel is None:
            return
        if best_sel is None or sel < best_sel:
            best_sel = sel
            best_step = step
            best_model_sd = _cpu_state(model.state_dict())
            best_ema_sd = _cpu_state(ema.state_dict())
            best_weighting_sd = _cpu_state(weighting.state_dict())
            best_summ = _summarize_heldout(res)
            _write_output()
            print(f"  [best] new minimum {SELECTION_METRIC}@{SELECTION_HORIZON}={sel:.3f} "
                  f"at step {step} -> wrote {out_path}")

    try:
        # --- Baseline (pre-finetune) eval at step 0 ----------------------- #
        res0 = eval_heldout(model, ema, stats, cal_segs, test_segs, device)
        baseline_summ = _summarize_heldout(res0)
        _log_eval(0, res0)
        _maybe_select(0, res0)

        # --- Fine-tune loop ---------------------------------------------- #
        data_iter = iter(loader)
        consecutive_nan = 0
        step = 0
        while step < total_steps:
            model.train()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            patches = batch['patches'].to(device, non_blocking=True)
            attn_mask = batch['attn_mask'].to(device, non_blocking=True)
            last_bg = batch['bg_formula_data']['last_bg'].to(device).float()      # (B,)
            true_bg_full = (
                batch['bg_formula_data']['true_bg_trajectory'][:, :_PRED_STEPS]
                .to(device).float()
                .reshape(-1, config.PREDICTION_PATCHES, config.PATCH_SIZE)
            )

            # Cross-window probe input (window k+1) — TIME-PROBE-only overhead,
            # fully skipped when the penalty is off (collate ships the key iff
            # enabled + weight > 0).
            next_window = None
            if config.TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
                _nw = batch.get('next_window')
                if _nw is not None:
                    next_window = {
                        'patches': _nw['patches'].to(device, non_blocking=True),
                        'last_bg': _nw['last_bg'].to(device, non_blocking=True).float(),
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
                q_tau, median, time_pred = model(patches, attn_mask, last_bg, return_time=True)
                q_tau = q_tau.float()
                median = median.float()
                loss_total, _parts = risk_total_loss(q_tau, median, true_bg_full, weighting)

                # Time-of-day probe MSE — added to the BACKWARD tensor ONLY. It never
                # touches loss_total / loss_ema / the CSV / selection (those stay on BG
                # accuracy), but with TIME_PROBE_DETACH=False its gradient shapes the
                # shared trunk, co-training it exactly as pretraining does.
                _tod_extra = loss_total.new_zeros(())
                _tod_loss_val = float('nan')   # raw probe loss, for the per-step log line
                _psh = batch['bg_formula_data'].get('pred_start_hour')
                if time_pred is not None and _psh is not None:
                    P = time_pred.shape[1]
                    adv = config.PATCH_SIZE * 5.0 / 60.0
                    base = _psh.to(time_pred.device).float().reshape(-1)                       # (B,)
                    offs = torch.arange(P, device=time_pred.device, dtype=torch.float32) * adv  # (P,)
                    tgt_hours = (base[:, None] + offs[None, :]) % 24.0                          # (B,P)
                    _tod_ce = time_of_day_bin_ce(
                        time_pred, tgt_hours, config.TIME_PROBE_N_BINS, config.TIME_PROBE_LABEL_SMOOTH_BINS,
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
                            nw_mask = attn_mask[:n_sub] if attn_mask.dim() == 3 else attn_mask
                            _, _, time_pred_next = model(
                                next_window['patches'][:n_sub], nw_mask,
                                next_window['last_bg'][:n_sub], return_time=True,
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
                res = eval_heldout(model, ema, stats, cal_segs, test_segs, device)
                _log_eval(step, res)
                _maybe_select(step, res)

        # --- Final eval --------------------------------------------------- #
        res_final = eval_heldout(model, ema, stats, cal_segs, test_segs, device)
        _log_eval(total_steps, res_final)
        _maybe_select(total_steps, res_final)
    finally:
        log_file.close()

    if best_model_sd is not None:
        _write_output()

    # --- Final report ----------------------------------------------------- #
    print()
    print("=" * 72)
    print(f"output checkpoint: {out_path}")
    print(f"best step: {best_step}  ({SELECTION_METRIC}@{SELECTION_HORIZON}="
          f"{_fmt(best_sel)})")
    print(f"held-out patient: {holdout}  (dataset={args.dataset}, mode={args.mode})")
    print("per-horizon held-out  baseline -> best (Δ):")
    for h in EVAL_HORIZONS:
        base = None if baseline_summ is None else baseline_summ[str(h)]['rmse_point']
        best = None if best_summ is None else best_summ[str(h)]['rmse_point']
        delta = (None if (base is None or best is None) else best - base)
        print(f"  h{h:>3} rmse_point: {_fmt(base)} -> {_fmt(best)}"
              + ("" if delta is None else f"  (Δ {delta:+.3f})"))
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
