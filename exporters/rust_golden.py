"""The pipeline golden `T1DMDROID`'s fp64 Rust pre/post is pinned against; regenerate when the contract moves.

Nothing in that reimplementation — masked-patch fill, attention rule, per-slot anchors, quantile
assembly — is exercised by this repository's tests.
Per case: raw four-channel history, masked set, padded patch tensor, per-slot anchors, head output, decoded
fan. Floats travel as values with a tolerance; the boolean attention pattern travels as a digest, exact.
Two cases run the real model (forecast, infill); the ladders feed a DETERMINISTIC synthetic ``head_raw``
through several span layouts, the only way to reach every span length the sampler can draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

import config as cfg
from normalization import normalize, CHANNEL_NAMES
from utils import assemble_quantiles, last_bg_mgdl_from_context
from inference import _build_patches_tensor, _resolve_mask_spans
from exporters.modified_forward import load_model
from exporters.head_weights import head_tensors
from exporters.executorch_xnnpack import _slot_anchor_cells
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX


def raw_history(n_steps: int) -> dict[str, np.ndarray]:
    """A deterministic, physiologically-shaped four-channel history, fp64 throughout.

    The Rust normalizes in fp64 and rounds once at the end; an fp32 reference would disagree in the last bit.
    """
    t = np.arange(n_steps, dtype=np.float64)
    bg = 120.0 + 28.0 * np.sin(2.0 * np.pi * t / 288.0) + 12.0 * np.sin(2.0 * np.pi * t / 47.0)
    for center, amp, width in ((0.22, 52.0, 20.0), (0.55, 44.0, 16.0), (0.81, 36.0, 13.0)):
        bg += amp * np.exp(-0.5 * ((t - center * n_steps) / width) ** 2)
    carb = np.zeros(n_steps)
    insulin = np.full(n_steps, 0.021)
    exercise = np.zeros(n_steps)
    for frac, g, u in ((0.20, 6.5, 0.26), (0.53, 5.5, 0.22), (0.79, 4.0, 0.18)):
        i = int(frac * n_steps)
        carb[i:i + 8] = g
        insulin[i:i + 10] += u
    e = int(0.66 * n_steps)
    exercise[e:e + 14] = 1.4
    return {
        'bg_absolute': np.clip(bg, BG_CLAMP_MIN, BG_CLAMP_MAX),
        'carb_intake': np.maximum(carb, 0.0),
        'insulin_combined': np.maximum(insulin, 0.0),
        'exercise_equiv': np.maximum(exercise, 0.0),
    }


def build_case(name: str, model, stats, n_ctx: int, mask_spans, with_forecast: bool,
               synthetic_head: bool = False) -> dict:
    """One golden case: the raw inputs, the built graph input, and the decoded fan."""
    n_steps = n_ctx * cfg.PATCH_SIZE
    raw = raw_history(n_steps)
    stacked = np.stack([raw[k] for k in CHANNEL_NAMES], axis=-1)          # (N, C) fp64
    feats = normalize(stacked, stats)                                     # fp64 in, fp64 out
    n_ch = len(CHANNEL_NAMES)

    T = cfg.MAX_SEQ_LEN
    p = cfg.PREDICTION_PATCHES if with_forecast else 0
    pad0 = T - n_ctx - p

    # With a future zone the trailing span is mandatory; without one the whole window is observed history.
    if with_forecast:
        # The consumer's builder appends the trailing span itself, so the fixture's `mask_spans` lists only
        # the EXTRA context spans; `_resolve_mask_spans` demands the complete set, so name it here.
        full = sorted([(int(s), int(L)) for s, L in (mask_spans or [])] + [(n_ctx, p)])
        spans = _resolve_mask_spans(full, n_ctx)
    else:
        spans = sorted((int(s), int(L)) for s, L in mask_spans)

    # fp64, rounded once at the boundary, like the Rust
    pt = np.zeros((T, cfg.PATCH_SIZE, cfg.N_INPUT_FEATURES), dtype=np.float64)
    pt[pad0:pad0 + n_ctx, :, :n_ch] = feats.reshape(n_ctx, cfg.PATCH_SIZE, n_ch)
    if p:
        zero_raw = normalize(np.zeros((1, n_ch), dtype=np.float64), stats)[0]
        for feat_idx in cfg.MASKABLE_FEATS:
            pt[pad0 + n_ctx:pad0 + n_ctx + p, :, feat_idx] = float(zero_raw[feat_idx])

    context_spans = [(s, L) for s, L in spans if s < n_ctx]
    extra = [q + pad0 for s, L in context_spans for q in range(s, s + L)]
    visible = torch.ones(T, dtype=torch.bool)
    is_pad = torch.zeros(T, dtype=torch.bool)
    is_pad[:pad0] = True
    if p:
        visible[pad0 + n_ctx:] = False
    for q in extra:
        visible[q] = False
    idx_abs = sorted(extra + list(range(pad0 + n_ctx, T))) if p else sorted(extra)
    for q in idx_abs:
        pt[q, :, 0] = 0.0
        pt[q, :, n_ch] = 1.0

    context = torch.from_numpy(
        np.concatenate([feats, np.zeros((n_steps, 1))], axis=-1)
    ).reshape(n_ctx, cfg.PATCH_SIZE, cfg.N_INPUT_FEATURES).float()
    # The hand-built fp64 tensor above avoids a second fp32 quantisation, but is only safe while it agrees
    # with the shipped builder — else the golden pins this file's layout, not the one inference uses.
    if with_forecast:
        shipped, _ = _build_patches_tensor(
            torch.from_numpy(
                np.concatenate([feats, np.zeros((n_steps, 1))], axis=-1)
            ).reshape(n_ctx, cfg.PATCH_SIZE, cfg.N_INPUT_FEATURES).float(),
            normalization_stats=stats,
            mask_spans=full,
        )
        mine = torch.from_numpy(pt[pad0:].reshape(n_ctx + p, -1)).float()
        d = float((shipped - mine).abs().max())
        assert d < 1e-5, f"the fixture's patch tensor disagrees with _build_patches_tensor by {d:.3e}"

    p_cells, s_cells = _slot_anchor_cells(spans, n_ctx)
    anchors = last_bg_mgdl_from_context(context, stats, p_cells, s_cells)

    m = cfg.MAX_MASKED_PATCHES
    n_masked = len(idx_abs)
    assert n_masked == anchors.numel(), f"{n_masked} slots but {anchors.numel()} anchors"
    mask_idx_t = torch.zeros(1, m, dtype=torch.int64)
    mask_idx_t[0, :n_masked] = torch.tensor(idx_abs, dtype=torch.int64)
    anchor_t = torch.full((1, m), float(anchors[0]), dtype=torch.float32)
    anchor_t[0, :n_masked] = anchors
    valid = torch.zeros(1, m, dtype=torch.bool)
    valid[0, :n_masked] = True

    from utils import create_attention_mask_from_visible
    attend = create_attention_mask_from_visible(visible[None, :], is_pad[None, :])[0]

    hidden = None
    if synthetic_head:
        # fixed pseudo-random head: the decode runs at every span length with no model in the loop
        g = torch.Generator().manual_seed(20260818)
        head_raw = torch.randn(
            1, m, cfg.PATCH_SIZE, 1 + 2 * cfg.N_SPREADS, generator=g, dtype=torch.float32,
        ) * 0.4
    else:
        captured: dict[str, torch.Tensor] = {}
        import model as model_module
        orig = model_module.assemble_quantiles

        def _cap(hr, a, mi=None, vl=None, carry_spread=0.0):
            captured['hr'] = hr.detach().clone()
            return orig(hr, a, mi, vl, carry_spread)

        model_module.assemble_quantiles = _cap
        hook = model.final_norm.register_forward_hook(
            lambda _m, _i, o: captured.__setitem__('hidden', o.detach().clone()))
        try:
            with torch.no_grad():
                model(
                    torch.from_numpy(pt.reshape(T, -1)).float().unsqueeze(0),
                    attend, anchor_t, mask_idx_t,
                )
        finally:
            model_module.assemble_quantiles = orig
            hook.remove()
        head_raw = captured['hr']
        hidden = captured['hidden'][0]                       # (T, D_MODEL)
    # the consumer's node rule per span, so its head_from_hidden path is pinned as well as its decode
    spans_out = []
    j = 0
    while j < n_masked:
        k = j
        while k + 1 < n_masked and idx_abs[k + 1] == idx_abs[k] + 1:
            k += 1
        first, last = idx_abs[j], idx_abs[k]
        spans_out.append({
            'first_slot': int(j), 'length': int(k - j + 1),
            'has_left': bool(first > 0 and attend[first, first - 1]),
            'has_right': bool(last + 1 < T and attend[last, last + 1]),
        })
        j = k + 1

    q_tau, median = assemble_quantiles(
        head_raw.double(), anchor_t.double(), mask_idx_t, valid,
    )

    return {
        "name": name,
        "n_ctx": n_ctx,
        "with_forecast": with_forecast,
        "mask_spans": [[int(s), int(L)] for s, L in (mask_spans or [])],
        "raw_bg": raw['bg_absolute'].tolist(),
        "raw_carb": raw['carb_intake'].tolist(),
        "raw_insulin": raw['insulin_combined'].tolist(),
        "raw_exercise": raw['exercise_equiv'].tolist(),
        "t": T,
        "pad0": int(pad0),
        "n_masked": int(n_masked),
        "slot_patch": [int(i) for i in idx_abs],
        "anchors": [float(x) for x in anchors.tolist()],
        "patches_f32": [float(np.float32(v)) for v in pt.reshape(-1)],
        "attn_sha256": hashlib.sha256(
            attend.numpy().astype(np.uint8).tobytes(order="C")
        ).hexdigest(),
        "head_raw": [float(x) for x in head_raw.double().reshape(-1).tolist()],
        "spans": spans_out,
        "hidden_f32": None if hidden is None else [float(np.float32(v)) for v in hidden.reshape(-1).tolist()],
        "median_risk": [float(x) for x in median[0, :n_masked].reshape(-1).tolist()],
        "q_tau_risk": [float(x) for x in q_tau[0, :n_masked].reshape(-1).tolist()],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="emit the Rust pipeline golden")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, ck = load_model(args.checkpoint)
    stats = ck["normalization_stats"]
    n_ctx = cfg.MIN_CONTEXT_PATCHES

    cases = [
        build_case("forecast", model, stats, n_ctx, None, True),
        build_case("infill", model, stats, n_ctx, [(60, 3), (100, 5)], True),
        # No future zone: a gap repair reads real evidence on BOTH sides of the span.
        build_case("infill_no_forecast", model, stats, n_ctx, [(80, 4)], False),
        # Span lengths 1..4 at once, pinning the decode against the span layout; the sampler draws up to
        # MASK_SPAN_LENGTHS[-1], so a second ladder covers the long end.
        build_case("span_ladder", model, stats, n_ctx,
                   [(10, 1), (20, 2), (40, 3), (70, 4)], False, synthetic_head=True),
        build_case("span_ladder_long", model, stats, n_ctx,
                   [(10, 5), (30, 7)], False, synthetic_head=True),
    ]

    doc = {
        "_comment": "Golden for T1DMDROID's fp64 Rust pre/post pipeline. Regenerate with "
                    "T1DMAI/exporters/rust_golden.py whenever the contract moves.",
        "arch_version": cfg.ARCH_VERSION,
        "patch_size": cfg.PATCH_SIZE,
        "n_input_features": cfg.N_INPUT_FEATURES,
        "max_masked_patches": cfg.MAX_MASKED_PATCHES,
        "prediction_patches": cfg.PREDICTION_PATCHES,
        "normalization_stats": stats,
        # `normalization.normalize` is fp32, the consumer fp64-rounded-once: a few fp32 ulps of
        # `ln(g)^power` apart, about 1e-6 in z. Not bit-identity.
        "tolerances": {"patches": 5e-6, "anchor": 1e-3, "risk": 1e-8, "head_raw_from_hidden": 1e-5},
        # the head file's tensors, so a consumer can pin hidden -> nodes -> spline -> MLP == head_raw on
        # the real-model cases; T1DMAI builds the spline weights in fp32, hence the looser tolerance
        "head": [{"name": n, "shape": list(t.shape), "values": [float(v) for v in t.reshape(-1).tolist()]}
                 for n, t in head_tensors(model)],
        "attn_digest_note": "sha256 over the boolean attend pattern, row-major, one byte "
                            "per cell (1 attend / 0 block)",
        "cases": cases,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f)
        f.write("\n")
    size = os.path.getsize(args.out)
    print(f"[golden] wrote {args.out} ({size} bytes, {len(cases)} cases)")


if __name__ == "__main__":
    main()
