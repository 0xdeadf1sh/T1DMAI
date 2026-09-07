"""Shared descriptor emitter, engine-agnostic — each engine passes its own tag, version and filename.

The descriptor JSON is the SOLE pre/post source for the on-device Rust core (PLAN §2.4): the app reads the
artifact plus this and NEVER parses the ``.pt``. It must carry ``normalization_stats`` and every
decode-critical constant the checkpoint lacks.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from typing import Any

import config as cfg
from exporters.modified_forward import NEG_FILL

# normalization owns this order, and it names the keys of the ``normalization_stats`` block below
from normalization import CHANNEL_NAMES

# Kovatchev constants and the physical clamp, imported not hardcoded, so what the Rust f/f_inv reproduce
# cannot drift from the model's transform. The checkpoint stores none of them.
from utils import _KOVATCHEV_SCALE, _KOVATCHEV_POWER, _KOVATCHEV_OFFSET
from T1DMSIM.simulator import BG_CLAMP_MIN as _BG_CLAMP_MIN, BG_CLAMP_MAX as _BG_CLAMP_MAX


def checkpoint_crossing_thresholds(ck: dict[str, Any]) -> tuple[float, float]:
    """The hypo/hyper cutoffs the crossing head was TRAINED on; cfg only if the run stored none."""
    tc = ck.get("training_config") or {}
    return (float(tc.get("bg_hypo_threshold", cfg.BG_HYPO_THRESHOLD)),
            float(tc.get("bg_hyper_threshold", cfg.BG_HYPER_THRESHOLD)))


def build_descriptor(
    *,
    model_id: str,
    engine: str,
    executorch_version: str,
    artifact_filename: str,
    normalization_stats: dict[str, dict[str, float]],
    precision: str = "fp32",
    model_card: dict[str, Any] | None = None,
    head: dict[str, Any] | None = None,
    seq_len: int | None = None,
    crossing_thresholds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Assemble the descriptor dict. Pure data, no I/O.

    ``seq_len`` is the graph's fixed ``T``, default ``cfg.MAX_SEQ_LEN``, and bounds the context the artifact
    accepts (``MAX_CONTEXT_PATCHES = T - PREDICTION_PATCHES``): a shorter export is a shorter memory, not a
    different contract.
    ``head`` is :func:`exporters.head_weights.write_head_weights`'s block; absent, the consumer has the frozen
    graph and no adapter seam. ``model_card`` is display-only and OUTSIDE the Rust contract — the on-device
    ``parse_descriptor`` ignores it, so it can never perturb decode.
    """
    risk_lo = _KOVATCHEV_SCALE * (math.log(_BG_CLAMP_MIN) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)
    risk_hi = _KOVATCHEV_SCALE * (math.log(_BG_CLAMP_MAX) ** _KOVATCHEV_POWER - _KOVATCHEV_OFFSET)

    # The head learned the CHECKPOINT's cutoffs; cfg is only a fallback for a caller with none.
    xh_hypo, xh_hyper = crossing_thresholds or (cfg.BG_HYPO_THRESHOLD, cfg.BG_HYPER_THRESHOLD)

    T = int(seq_len or cfg.MAX_SEQ_LEN)
    P = cfg.PREDICTION_PATCHES
    M = cfg.MAX_MASKED_PATCHES
    max_ctx = T - P
    assert max_ctx >= cfg.MIN_CONTEXT_PATCHES, (
        f"seq_len {T} leaves {max_ctx} context patches, below MIN_CONTEXT_PATCHES "
        f"{cfg.MIN_CONTEXT_PATCHES}"
    )
    # a longer graph would advertise more context than the architecture was trained on, and every
    # consumer takes the descriptor at its word
    assert T <= cfg.MAX_SEQ_LEN, (
        f"seq_len {T} exceeds MAX_SEQ_LEN {cfg.MAX_SEQ_LEN}; the model was never trained on a "
        f"window that long"
    )
    desc: dict[str, Any] = {
        "schema_version": 1,
        "id": model_id,
        "engine": engine,
        "executorch_version": executorch_version,
        "artifact": artifact_filename,
        "precision": precision,
        "arch_version": cfg.ARCH_VERSION,

        # graph I/O contract (PLAN §2.4)
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
            "input_slot_sel": {
                "name": "slot_sel", "shape": [M, T], "dtype": precision,
                "kind": "one-hot selection",
                "note": "row j is one-hot at the patch head slot j reads, in "
                        "ascending patch order; surplus slots repeat patch 0 and "
                        "their outputs are discarded. A selection matmul, not a "
                        "gather, so no int64 tensor crosses the runtime boundary.",
            },
            "output_head_raw": {
                "name": "head_raw", "output_index": 0,
                "shape": [1, M, cfg.PATCH_SIZE, 1 + 2 * cfg.N_SPREADS],
                "dtype": precision, "space": "kovatchev-risk",
                "note": "(B, M, S, 1+2*N_SPREADS): col0 median delta; 1..3 tau>.5 "
                        "spreads .75/.9/.95; 4..6 tau<.5 spreads .25/.1/.05. One row "
                        "per head slot, in slot_sel order.",
            },
            "output_time_logits": {
                "name": "time_logits", "output_index": 1,
                "shape": [1, M, cfg.TIME_PROBE_N_BINS],
                "dtype": precision, "space": "raw-logits",
                "note": "(B, M, N_BINS): per-slot hour-of-day bin logits from the "
                        "co-trained time probe; softmax over the N_BINS hour-of-day "
                        "circle downstream (Rust). Present iff the time section "
                        "below is present.",
            },
            "output_hidden": {
                "name": "hidden", "output_index": 2,
                "shape": [1, T, cfg.D_MODEL],
                "dtype": precision, "space": "trunk-hidden",
                "note": "(B, T, D_MODEL) final-normed hidden state per PATCH — the "
                        "adapter seam. A span's spline nodes are its masked patches "
                        "plus the visible neighbour on each side, so the seam carries "
                        "the whole window: gather the nodes, build the step weights, "
                        "and feed the head block's weights to reproduce head_raw, with "
                        "or without a low-rank adapter.",
            },
            "output_crossing_logits": {
                "name": "crossing_logits", "output_index": 3,
                "shape": [1, M, cfg.PATCH_SIZE, cfg.N_CROSSING],
                "dtype": precision, "space": "raw-logits",
                "note": "(B, M, S, 2): per-step cumulative crossing logits off the same "
                        "step states as head_raw; sigmoid downstream. Thresholds and "
                        "column order in the crossing section below (SPEC/inference.md §8.5).",
            },
        },

        # Crossing head (SPEC/inference.md §8.5): the thresholds are the checkpoint's own.
        "crossing": {
            "output_index": 3,
            "output_name": "crossing_logits",
            "shape": [1, M, cfg.PATCH_SIZE, cfg.N_CROSSING],
            "columns": ["hypo", "hyper"],
            "hypo_mgdl": float(xh_hypo),
            "hyper_mgdl": float(xh_hyper),
            "cumulative": True,
            "value_kind": "raw logits; sigmoid gives the probability",
            "detach": cfg.CROSSING_HEAD_DETACH,
            "co_trains_trunk": not cfg.CROSSING_HEAD_DETACH,
        },

        # Time-of-day probe (PLAN §7): each prediction patch's ABSOLUTE hour-of-day over N_BINS circular
        # bins, inferred from the trajectory — there is NO clock input — not a per-step timestamp.
        # head_raw is byte-identical with or without this head.
        "time": {
            "output_index": 1,
            "output_name": "time_logits",
            "shape": [1, M, cfg.TIME_PROBE_N_BINS],
            "n_bins": cfg.TIME_PROBE_N_BINS,
            "bin_hours": cfg.TIME_PROBE_BIN_HOURS,
            "layout": "per head slot: row j = the hour-of-day bin logits of the "
                      "patch slot_sel row j selected",
            "value_kind": "raw logits (softmax over the N_BINS-bin hour-of-day circle)",
            "bin_centers_hours": [
                (k + 0.5) * (24.0 / cfg.TIME_PROBE_N_BINS)
                for k in range(cfg.TIME_PROBE_N_BINS)
            ],
            "circle": "bin k center angle th_k = 2*pi*center_hour_k/24; hour 0 at "
                      "angle 0, increasing with hour",
            # The P per-patch rows reduced to ONE current-hour belief, as T1DMAI does it
            # (inference.estimate_current_hour / gui._decode_tod).
            "reduction": "origin_slot",
            "reduction_detail": {
                "slot_index": "the slot holding the FIRST forecast patch — with an "
                              "infill span in the masked set that is no longer slot 0",
                "steps": [
                    "probs = softmax(time_logits[0, origin_slot, :])",
                    "res = sum_k probs[k] * (cos th_k, sin th_k)   # th_k = 2*pi*center_hour_k/24",
                    "hour = (atan2(res.sin, res.cos) mod 2*pi) * 24/(2*pi)",
                    "R = hypot(res.cos, res.sin)   # in [0,1], concentration/confidence",
                ],
                "note": "R in [0,1]: R->1 a concentrated (confident) phase belief, "
                        "R->0 diffuse/ambiguous.",
            },
            # The clock-face's fusion (utils.aggregate_origin_belief): de-rotate patch p by -p*advance_hours,
            # average, renormalize, same resultant read-out. The app's declared reducer stays patch 0.
            "alt_reduction": {
                "name": "aggregate_origin_belief",
                "advance_hours_per_patch":
                    cfg.PREDICTION_HORIZON_HOURS / cfg.PREDICTION_PATCHES,
            },
            "detach": cfg.TIME_PROBE_DETACH,
            "co_trains_trunk": not cfg.TIME_PROBE_DETACH,
        },

        # geometry, also recoverable from the checkpoint
        "geometry": {
            "T": T,
            "PATCH_SIZE": cfg.PATCH_SIZE,
            "N_INPUT_FEATURES": cfg.N_INPUT_FEATURES,
            "PATCH_DIM": cfg.PATCH_DIM,
            "PREDICTION_PATCHES": cfg.PREDICTION_PATCHES,
            "MIN_CONTEXT_PATCHES": cfg.MIN_CONTEXT_PATCHES,
            # what THIS artifact accepts, not what the architecture allows
            "MAX_CONTEXT_PATCHES": max_ctx,
            "ARCH_MAX_CONTEXT_PATCHES": cfg.MAX_CONTEXT_PATCHES,
            # head slot count and the cap on a caller's masked set; it sizes no weight, so no shape recovers it
            "MAX_MASKED_PATCHES": M,
            "MASK_MAX_SPANS": cfg.MASK_MAX_SPANS,
            "MASK_SPAN_LENGTHS": list(cfg.MASK_SPAN_LENGTHS),
            "D_MODEL": cfg.D_MODEL,
            "N_LAYERS": cfg.N_LAYERS,
            "N_HEADS": cfg.N_HEADS,
            "HEAD_DIM": cfg.HEAD_DIM,
            # The normalized SIGNAL channels in input-feature order, and the keys of `normalization_stats`.
            # Fewer than N_INPUT_FEATURES: the trailing `bg_masked` bit carries no statistics and no name.
            "CHANNEL_NAMES": list(CHANNEL_NAMES),
            "CHANNEL_TO_FEAT": {str(k): v for k, v in cfg.CHANNEL_TO_FEAT.items()},
            "NON_MASKABLE_FEATS": list(cfg.NON_MASKABLE_FEATS),
            "MASKABLE_FEATS": list(cfg.MASKABLE_FEATS),
        },

        # REQUIRED: BG risk-space, carb/insulin log1p
        "normalization_stats": normalization_stats,

        # decode-critical constants ABSENT from the checkpoint (PLAN §2.4)
        "constants": {
            "ROPE_BASE": cfg.ROPE_BASE,
            "RMSNORM_EPS": 1e-6,
            "NORMALIZE_STD_FLOOR": 1e-8,
            "BG_QUANTILE_SPREAD_MIN": cfg.BG_QUANTILE_SPREAD_MIN,
            "N_SPREADS": cfg.N_SPREADS,
            "N_QUANTILES": cfg.N_QUANTILES,
            "QUANTILE_LEVELS": list(cfg.QUANTILE_LEVELS),
            "MEDIAN_IDX": list(cfg.QUANTILE_LEVELS).index(0.5),
            "neg_fill": NEG_FILL,
            "PREDICTION_HORIZON_HOURS": cfg.PREDICTION_HORIZON_HOURS,
        },

        # Kovatchev risk transform; the Rust f / f_inv reproduce these
        "kovatchev": {
            "SCALE": _KOVATCHEV_SCALE,
            "POWER": _KOVATCHEV_POWER,
            "OFFSET": _KOVATCHEV_OFFSET,
            "BG_CLAMP_MIN": _BG_CLAMP_MIN,
            "BG_CLAMP_MAX": _BG_CLAMP_MAX,
            "RISK_CLAMP_MIN": risk_lo,
            "RISK_CLAMP_MAX": risk_hi,
        },

        # No smoother block by design: the model consumes raw post-noise signals, so the Rust runtime
        # must apply NO input FIR.

        # conformal OFF for this real-data deployment (PLAN §2.4)
        "conformal": {
            "enabled": False,
            "note": "simulator-fit delta omitted for real CGM; raw bands are "
                    "bit-identical (INFERENCE.md §8.4)",
        },
    }
    if head is not None:
        desc["head"] = {**head, "decoder": "bspline-centre-nodes"}
    if model_card is not None:
        desc["model_card"] = model_card
    return desc


_HORIZONS_MIN = [30, 60, 120]


def _round(v: Any) -> float | None:
    return round(float(v), 4) if v is not None else None


def _sim_reference_metrics(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Reference metrics off the LAST ``val_history`` entry; a missing key degrades to ``None``."""
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




def build_model_card(model, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The display-only ``model_card``: param count plus held-out reference metrics.

    The metrics come from the checkpoint's ``val_history``, the simulator validation the run selected on —
    a reference, never on-device realized accuracy.
    """
    return {
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "val_step": int(
            (checkpoint.get("val_history") or [{}])[-1].get("step", checkpoint.get("step", 0))
        ),
        "reference_metrics": _sim_reference_metrics(checkpoint),
    }


def write_descriptor(descriptor: dict[str, Any], path: str) -> None:
    """Write the descriptor to ``path`` as pretty JSON."""
    with open(path, "w") as f:
        json.dump(descriptor, f, indent=2)
        f.write("\n")


def deploy_to_server(pte_path: str, descriptor: dict[str, Any], deploy_dir: str) -> "tuple[str, str]":
    """Copy ``pte_path`` and its ``descriptor`` into a T1DMSERVER models directory -> (artifact, sidecar).

    ``t1dm-store::refresh_models`` hashes every non-``.json`` file and pairs it with a SIBLING ``<stem>.json``,
    so the sidecar is the artifact's name with the extension swapped: ``large-sim.xnnpack.pte`` ->
    ``large-sim.xnnpack.json``. The phone strips the engine infix for the logical id (``large-sim``).
    """
    os.makedirs(deploy_dir, exist_ok=True)
    artifact = os.path.join(deploy_dir, os.path.basename(pte_path))
    sidecar = os.path.splitext(artifact)[0] + ".json"
    shutil.copy2(pte_path, artifact)
    # The head side file does NOT travel this way, so the deployed descriptor must not claim it does:
    # the registry pairs ONE artifact with ONE sidecar and would offer a stray `<id>.head.bin` as a
    # model of its own. A synced phone gets a model it can run and cannot adapt.
    deployed = {k: v for k, v in descriptor.items() if k != "head"}
    write_descriptor(deployed, sidecar)
    return artifact, sidecar
