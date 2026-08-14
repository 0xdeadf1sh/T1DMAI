"""Shared descriptor emitter.

The descriptor JSON is the SOLE pre/post source for the on-device Rust core (PLAN
§2.4): the app downloads the ``.pte`` (weights baked, graph starts from
already-normalized patches) plus this ``meta`` and NEVER parses the ``.pt`` pickle.
It must therefore carry ``normalization_stats`` and every decode-critical constant
that is absent from the checkpoint.

This module is engine-agnostic; each engine passes its own ``engine`` /
``executorch_version`` / artifact filename.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from typing import Any

import config as cfg
from exporters.modified_forward import NEG_FILL

# The channel order normalization owns — it names the keys of the
# ``normalization_stats`` block below, so the descriptor cannot label them itself.
from normalization import CHANNEL_NAMES

# Kovatchev risk-transform constants and the physical BG clamp. Imported from the
# single sources of truth (utils / the simulator) rather than hardcoded, so the
# descriptor the Rust f/f_inv reproduce can never drift from the model's own
# transform. The checkpoint stores none of these, so they are written into the
# descriptor JSON verbatim below.
from utils import _KOVATCHEV_SCALE, _KOVATCHEV_POWER, _KOVATCHEV_OFFSET
from T1DMSIM.simulator import BG_CLAMP_MIN as _BG_CLAMP_MIN, BG_CLAMP_MAX as _BG_CLAMP_MAX


def build_descriptor(
    *,
    model_id: str,
    engine: str,
    executorch_version: str,
    artifact_filename: str,
    normalization_stats: dict[str, dict[str, float]],
    precision: str = "fp32",
    model_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the descriptor dict. Pure data; no I/O.

    ``model_card`` (optional) is a display-only provenance block the app surfaces in the
    Models panel: parameter count + held-out validation reference metrics. It is OUTSIDE
    the Rust pre/post contract (the on-device `parse_descriptor` ignores it), so adding it
    never perturbs decode; it is stamped verbatim under the top-level ``model_card`` key.
    """
    risk_lo = _KOVATCHEV_SCALE * (math.log(_BG_CLAMP_MIN) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)
    risk_hi = _KOVATCHEV_SCALE * (math.log(_BG_CLAMP_MAX) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)

    T = cfg.MAX_SEQ_LEN
    desc: dict[str, Any] = {
        "schema_version": 1,
        "id": model_id,
        "engine": engine,
        "executorch_version": executorch_version,
        "artifact": artifact_filename,
        "precision": precision,
        "arch_version": cfg.ARCH_VERSION,

        # --- graph I/O contract (PLAN §2.4) ---
        "io": {
            "input_patches": {
                "name": "patches", "shape": [1, T, cfg.PATCH_DIM], "dtype": precision,
                "layout": "step-major: flat = t*N_INPUT_FEATURES + feat",
                "note": "already-normalized; graph starts from z-space patches",
            },
            "input_mask": {
                "name": "attn_mask", "shape": [T, T], "dtype": precision,
                "kind": "additive-float struct",
                "attend": 0.0, "block": NEG_FILL,
                "note": "the sole additive term on the attention logits; position "
                        "enters through RoPE alone, so nothing is pre-combined here",
            },
            "output_head_raw": {
                "name": "head_raw", "output_index": 0,
                "shape": [1, cfg.PREDICTION_PATCHES, cfg.PATCH_SIZE, 1 + 2 * cfg.N_SPREADS],
                "dtype": precision, "space": "kovatchev-risk",
                "note": "(B, P, S, 1+2*N_SPREADS): col0 median delta; 1..3 tau>.5 "
                        "spreads .75/.9/.95; 4..6 tau<.5 spreads .25/.1/.05",
            },
            "output_time_logits": {
                "name": "time_logits", "output_index": 1,
                "shape": [1, cfg.PREDICTION_PATCHES, cfg.TIME_PROBE_N_BINS],
                "dtype": precision, "space": "raw-logits",
                "note": "(B, P, N_BINS): per-prediction-patch hour-of-day bin logits "
                        "from the co-trained time probe; softmax over the N_BINS "
                        "hour-of-day circle downstream (Rust). Present iff the time "
                        "section below is present.",
            },
        },

        # --- time-of-day probe (circadian-phase belief; PLAN §7) ---
        # A co-trained (TIME_PROBE_DETACH=False) auxiliary head that classifies each
        # prediction patch's ABSOLUTE hour-of-day into N_BINS circular bins. It is a
        # belief about what hour-of-day the model thinks it currently is (inferred from
        # the trajectory — there is NO clock input), NOT a per-forecast-step timestamp.
        # The forecast (head_raw) is byte-identical with or without this head.
        "time": {
            "output_index": 1,
            "output_name": "time_logits",
            "shape": [1, cfg.PREDICTION_PATCHES, cfg.TIME_PROBE_N_BINS],
            "n_bins": cfg.TIME_PROBE_N_BINS,
            "bin_hours": cfg.TIME_PROBE_BIN_HOURS,
            "layout": "per prediction-patch: row p = patch p's hour-of-day bin logits",
            "value_kind": "raw logits (softmax over the N_BINS-bin hour-of-day circle)",
            "bin_centers_hours": [
                (k + 0.5) * (24.0 / cfg.TIME_PROBE_N_BINS)
                for k in range(cfg.TIME_PROBE_N_BINS)
            ],
            "circle": "bin k center angle th_k = 2*pi*center_hour_k/24; hour 0 at "
                      "angle 0, increasing with hour",
            # How the app reduces the P per-patch rows to ONE current-hour belief.
            # This is T1DMAI's production reducer (inference.estimate_current_hour /
            # gui._decode_tod): take the FIRST (origin, index 0) prediction patch,
            # softmax it, form the mean resultant vector, read hour from its angle and
            # confidence R from its length (utils.time_of_day_resultant /
            # time_of_day_decode_bins).
            "reduction": "origin_patch",
            "reduction_detail": {
                "patch_index": 0,
                "steps": [
                    "probs = softmax(time_logits[0, 0, :])",
                    "res = sum_k probs[k] * (cos th_k, sin th_k)   # th_k = 2*pi*center_hour_k/24",
                    "hour = (atan2(res.sin, res.cos) mod 2*pi) * 24/(2*pi)",
                    "R = hypot(res.cos, res.sin)   # in [0,1], concentration/confidence",
                ],
                "note": "R in [0,1]: R->1 a concentrated (confident) phase belief, "
                        "R->0 diffuse/ambiguous.",
            },
            # A richer optional fusion T1DMAI's clock-face uses (utils.aggregate_origin
            # _belief): de-rotate patch p by -p*advance_hours (advance = PATCH_SIZE*5min
            # = 24/N_BINS * (PREDICTION_HORIZON_HOURS/PREDICTION_PATCHES) h), average the
            # de-rotated rows, renormalize, then the same resultant read-out. The app's
            # declared reducer is origin_patch (patch 0); this is documented for parity.
            "alt_reduction": {
                "name": "aggregate_origin_belief",
                "advance_hours_per_patch":
                    cfg.PREDICTION_HORIZON_HOURS / cfg.PREDICTION_PATCHES,
            },
            "detach": cfg.TIME_PROBE_DETACH,
            "co_trains_trunk": not cfg.TIME_PROBE_DETACH,
        },

        # --- geometry (also recoverable from the checkpoint, restated for the app) ---
        "geometry": {
            "T": T,
            "PATCH_SIZE": cfg.PATCH_SIZE,
            "N_INPUT_FEATURES": cfg.N_INPUT_FEATURES,
            "PATCH_DIM": cfg.PATCH_DIM,
            "PREDICTION_PATCHES": cfg.PREDICTION_PATCHES,
            "MIN_CONTEXT_PATCHES": cfg.MIN_CONTEXT_PATCHES,
            "MAX_CONTEXT_PATCHES": cfg.MAX_CONTEXT_PATCHES,
            "D_MODEL": cfg.D_MODEL,
            "N_LAYERS": cfg.N_LAYERS,
            "N_HEADS": cfg.N_HEADS,
            "HEAD_DIM": cfg.HEAD_DIM,
            # The normalized SIGNAL channels, in input-feature order — and the keys
            # of `normalization_stats`. There are fewer of them than
            # N_INPUT_FEATURES: the trailing `bg_masked` feat is a per-patch bit
            # carrying no statistics, so it is named here by no channel.
            "CHANNEL_NAMES": list(CHANNEL_NAMES),
            "CHANNEL_TO_FEAT": {str(k): v for k, v in cfg.CHANNEL_TO_FEAT.items()},
            "NON_MASKABLE_FEATS": list(cfg.NON_MASKABLE_FEATS),
            "MASKABLE_FEATS": list(cfg.MASKABLE_FEATS),
        },

        # --- REQUIRED: normalization stats (BG risk-space; carb/insulin log1p) ---
        "normalization_stats": normalization_stats,

        # --- decode-critical constants ABSENT from the checkpoint (PLAN §2.4) ---
        "constants": {
            "ROPE_BASE": cfg.ROPE_BASE,
            "RMSNORM_EPS": 1e-6,
            "NORMALIZE_STD_FLOOR": 1e-8,
            "BG_HEAD_MEDIAN_MODE": cfg.BG_HEAD_MEDIAN_MODE,
            "BG_HEAD_MEDIAN_GLOBAL_DIM": cfg.BG_HEAD_MEDIAN_GLOBAL_DIM,
            "BG_HEAD_STEP_BASIS_TYPE": cfg.BG_HEAD_STEP_BASIS_TYPE,
            "BG_HEAD_STEP_BASIS_DIM": cfg.BG_HEAD_STEP_BASIS_DIM,
            "BG_QUANTILE_SPREAD_MIN": cfg.BG_QUANTILE_SPREAD_MIN,
            "N_SPREADS": cfg.N_SPREADS,
            "N_QUANTILES": cfg.N_QUANTILES,
            "QUANTILE_LEVELS": list(cfg.QUANTILE_LEVELS),
            "MEDIAN_IDX": list(cfg.QUANTILE_LEVELS).index(0.5),
            "neg_fill": NEG_FILL,
            "PREDICTION_HORIZON_HOURS": cfg.PREDICTION_HORIZON_HOURS,
        },

        # --- Kovatchev risk transform (Rust f / f_inv reproduce these) ---
        "kovatchev": {
            "SCALE": _KOVATCHEV_SCALE,
            "POWER": _KOVATCHEV_POWER,
            "OFFSET": _KOVATCHEV_OFFSET,
            "BG_CLAMP_MIN": _BG_CLAMP_MIN,
            "BG_CLAMP_MAX": _BG_CLAMP_MAX,
            "RISK_CLAMP_MIN": risk_lo,
            "RISK_CLAMP_MAX": risk_hi,
        },

        # --- input smoothing REMOVED: the model consumes raw post-noise signals,
        # so the on-device Rust runtime must apply NO input FIR (no "smoother" block). ---

        # --- conformal: OFF for this real-data deployment (PLAN §2.4) ---
        "conformal": {
            "enabled": False,
            "note": "simulator-fit delta omitted for real CGM; raw bands are "
                    "bit-identical (INFERENCE.md §8.4)",
        },
    }
    if model_card is not None:
        desc["model_card"] = model_card
    return desc


_HORIZONS_MIN = [30, 60, 120]


def _round(v: Any) -> float | None:
    return round(float(v), 4) if v is not None else None


def _sim_reference_metrics(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Reference metrics from the LAST ``val_history`` entry (train.py's simulator
    validation suite). A missing key degrades to ``None`` rather than aborting."""
    vh = (checkpoint.get("val_history") or [{}])[-1]

    def r(key: str) -> float | None:
        return _round(vh.get(key))

    return {
        "source": "sim-validation",
        "note": "held-out simulator validation at the exported checkpoint (T1DMAI "
                "train.py); reference only, not on-device realized accuracy",
        "horizons_min": list(_HORIZONS_MIN),
        "rmse_mgdl": [r("bg_rmse_30"), r("bg_rmse_60"), r("bg_rmse_120")],
        "mard_pct": [r("evalfix_mard@30"), r("evalfix_mard@60"), r("evalfix_mard@120")],
        "clarke_a_pct": [r("evalfix_clarke_A@30"), r("evalfix_clarke_A@60"),
                         r("evalfix_clarke_A@120")],
        "coverage90": [r("coverage90@30"), r("coverage90@60"), r("coverage90@120")],
        "clarke_ab_pct": r("clarke_AB_pct"),
        "tod_mae_h": r("tod_mae_h"),
        "tod_mae_hiconf_h": r("tod_mae_hiconf"),
    }


def _finetune_reference_metrics(best: dict[str, Any], fm: dict[str, Any] | None = None
                                ) -> dict[str, Any]:
    """Reference metrics from a finetune checkpoint's ``finetune_meta['best_heldout']``.

    A finetuned checkpoint carries no ``val_history`` (finetune.py evaluates against a
    held-out REAL-cohort patient split instead), so the same per-horizon slots are filled
    from that eval. Only the semantically identical quantities are mapped: ``rmse_point``
    -> ``rmse_mgdl`` and ``mard`` -> ``mard_pct``. ``clarke_a_pct`` / ``coverage90`` /
    the time-probe rows stay ``None`` — the real-cohort eval reports Clarke A+B pooled,
    not zone A alone, so filling them would misstate what was measured.

    A PERSONAL finetune withholds one day of one record rather than patients from a
    cohort, so it is labelled distinctly: the numbers occupy the same slots and no
    consumer reads ``source``, but a card that called a single day's result a
    cohort evaluation with patients withheld would be describing something that did not
    happen.
    """
    def per_h(key: str) -> list[float | None]:
        return [_round((best.get(str(h)) or {}).get(key)) for h in _HORIZONS_MIN]

    if (fm or {}).get("dataset") == 'personal':
        return {
            "source": "personal-heldout-day",
            "note": "held-out-DAY evaluation of a checkpoint finetuned on one person's own "
                    "record (T1DMAI finetune_personal.py); one day of one record, not a "
                    "cohort and no patient withheld; reference only, not on-device realized "
                    "accuracy",
            "horizons_min": list(_HORIZONS_MIN),
            "rmse_mgdl": per_h("rmse_point"),
            "mard_pct": per_h("mard"),
            "clarke_a_pct": [None] * len(_HORIZONS_MIN),
            "coverage90": [None] * len(_HORIZONS_MIN),
            "clarke_ab_pct": None,
            "tod_mae_h": None,
            "tod_mae_hiconf_h": None,
        }

    return {
        "source": "real-heldout",
        "note": "held-out real-cohort evaluation of the finetuned checkpoint (T1DMAI "
                "finetune.py, patients withheld from training); reference only, not "
                "on-device realized accuracy",
        "horizons_min": list(_HORIZONS_MIN),
        "rmse_mgdl": per_h("rmse_point"),
        "mard_pct": per_h("mard"),
        "clarke_a_pct": [None] * len(_HORIZONS_MIN),
        "coverage90": [None] * len(_HORIZONS_MIN),
        "clarke_ab_pct": None,
        "tod_mae_h": None,
        "tod_mae_hiconf_h": None,
        # Kept alongside the flat slots the app reads, so the pooled Clarke A+B and the
        # persistence-skill numbers are not lost.
        "per_horizon_detail": {
            str(h): {k: _round(v) for k, v in (best.get(str(h)) or {}).items()}
            for h in _HORIZONS_MIN
        },
        "n_test_windows": best.get("n_test_windows"),
        "n_patients": best.get("n_patients"),
    }


def build_model_card(model, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Assemble the display-only ``model_card`` (param count + held-out reference
    metrics) from the built ``model`` and its ``checkpoint``.

    Two checkpoint shapes are supported. A pretrained checkpoint carries ``val_history``
    (simulator validation); a finetuned one carries ``finetune_meta`` with a
    ``best_heldout`` real-cohort eval and no ``val_history``. The emitted schema is the
    same either way — only ``reference_metrics['source']`` distinguishes them — plus a
    ``finetune`` provenance block when the checkpoint is a finetune.
    """
    param_count = int(sum(p.numel() for p in model.parameters()))
    fm = checkpoint.get("finetune_meta") or {}
    best = fm.get("best_heldout") or {}

    card: dict[str, Any] = {
        "param_count": param_count,
        "val_step": int(
            (checkpoint.get("val_history") or [{}])[-1].get("step", checkpoint.get("step", 0))
        ),
        "reference_metrics": (
            _finetune_reference_metrics(best, fm) if (best and not checkpoint.get("val_history"))
            else _sim_reference_metrics(checkpoint)
        ),
    }
    if fm:
        card["finetune"] = {
            "dataset": fm.get("dataset"),
            "datasets": fm.get("datasets"),
            "mode": fm.get("mode"),
            # A personal finetune withholds a day, recorded as val_day; without the
            # fallback the card would report holdout: null and drop what was held out.
            "holdout": fm.get("holdout") or fm.get("val_day"),
            "total_steps": fm.get("total_steps"),
            "lr_scale": fm.get("lr_scale"),
        }
    return card


def write_descriptor(descriptor: dict[str, Any], path: str) -> None:
    """Write the descriptor to ``path`` as pretty JSON."""
    with open(path, "w") as f:
        json.dump(descriptor, f, indent=2)
        f.write("\n")


def deploy_to_server(pte_path: str, descriptor: dict[str, Any], deploy_dir: str) -> "tuple[str, str]":
    """Copy ``pte_path`` + its ``descriptor`` into a T1DMSERVER models directory.

    The server registry (``t1dm-store::refresh_models``) hashes every non-``.json`` file
    and pairs it with a SIBLING ``<stem>.json`` sidecar, so the descriptor's filename is
    the artifact's with the extension swapped — ``large-sim.xnnpack.pte`` pairs with
    ``large-sim.xnnpack.json``. The phone's ModelSyncCoordinator then strips the engine
    infix to recover the logical model id (``large-sim.xnnpack`` -> ``large-sim``).

    Returns:
        (deployed artifact path, deployed sidecar path).
    """
    os.makedirs(deploy_dir, exist_ok=True)
    artifact = os.path.join(deploy_dir, os.path.basename(pte_path))
    sidecar = os.path.splitext(artifact)[0] + ".json"
    shutil.copy2(pte_path, artifact)
    write_descriptor(descriptor, sidecar)
    return artifact, sidecar
