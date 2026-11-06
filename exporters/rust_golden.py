"""The pipeline golden T1DMDROID's fp64 Rust pre/post is pinned against; regenerate on change.
Per case: raw history in the checkpoint's layout, masked set, patch tensor, per-slot anchors,
head output, decoded fan. Floats carry a tolerance; the attention pattern is an exact digest.
Two cases run the real model; ladders feed synthetic head_raw through every span length.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os

import numpy as np
import torch

import config as cfg
from normalization import normalize, CHANNEL_NAMES
from utils import assemble_quantiles, last_bg_mgdl_from_context
from inference import _build_patches_tensor, _resolve_mask_spans
from data import (
    MASKED_CHANNEL_POLICY_BLIND, blind_masked_doses, checkpoint_masked_channel_policy,
    zero_dose_fill,
)
from exporters.modified_forward import load_model
from exporters.head_weights import head_tensors
from exporters.executorch_xnnpack import _slot_anchor_cells
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX


def raw_history(n_steps: int) -> dict[str, np.ndarray]:
    """A deterministic, physiologically-shaped history in the bound layout, fp64 throughout.

    The Rust normalizes in fp64 and rounds once at the end; fp32 would disagree in the last bit.
    """
    t = np.arange(n_steps, dtype=np.float64)
    bg = 120.0 + 28.0 * np.sin(2.0 * np.pi * t / 288.0) + 12.0 * np.sin(2.0 * np.pi * t / 47.0)
    for center, amp, width in ((0.22, 52.0, 20.0), (0.55, 44.0, 16.0), (0.81, 36.0, 13.0)):
        bg += amp * np.exp(-0.5 * ((t - center * n_steps) / width) ** 2)
    out = {'bg_absolute': np.clip(bg, BG_CLAMP_MIN, BG_CLAMP_MAX)}
    meals = ((0.20, 6.5, 0.26, 62.0), (0.53, 5.5, 0.22, 35.0), (0.79, 4.0, 0.18, 50.0))
    if cfg.INPUT_LAYOUT == 'curves':
        carb = np.zeros(n_steps)
        insulin = np.full(n_steps, 0.021)
        for frac, g, u, _gi in meals:
            i = int(frac * n_steps)
            carb[i:i + 8] = g
            insulin[i:i + 10] += u
        out['carb_intake'] = np.maximum(carb, 0.0)
        out['insulin_combined'] = np.maximum(insulin, 0.0)
        return out
    out.update({name: np.zeros(n_steps) for name in CHANNEL_NAMES[1:]})
    for frac, g, u, gi in meals:
        i = int(frac * n_steps)
        out['carb_g'][i], out['carb_gi'][i] = 8.0 * g, gi
        out['bolus_u'][i], out['bolus_peak_min'][i], out['bolus_dur_h'][i] = 10.0 * u, 57.6, 3.6
    for i in range(int(0.07 * n_steps), n_steps, 288):
        out['basal_u'][i], out['basal_peak_min'][i], out['basal_dur_h'][i] = 17.0, 317.0, 73.0
    return out


def build_case(name: str, model, stats, n_ctx: int, mask_spans, with_forecast: bool,
               synthetic_head: bool = False, blind: bool = False) -> dict:
    """One golden case: the raw inputs, the built graph input, and the decoded fan."""
    n_steps = n_ctx * cfg.PATCH_SIZE
    raw = raw_history(n_steps)
    stacked = np.stack([raw[k] for k in CHANNEL_NAMES], axis=-1)          # (N, C) fp64
    feats = normalize(stacked, stats)                                     # fp64 in, fp64 out
    n_ch = len(CHANNEL_NAMES)

    T = cfg.MAX_SEQ_LEN
    p = cfg.PREDICTION_PATCHES if with_forecast else 0
    pad0 = T - n_ctx - p

    # With a future zone the trailing span is mandatory; else the whole window is observed history.
    if with_forecast:
        # Builder appends the trailing span; fixture's mask_spans lists only extra context spans.
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
    fill = zero_dose_fill(stats) if blind else {}
    for q in idx_abs:
        for feat_idx, z in fill.items():
            pt[q, :, feat_idx] = z

    context = torch.from_numpy(
        np.concatenate([feats, np.zeros((n_steps, 1))], axis=-1)
    ).reshape(n_ctx, cfg.PATCH_SIZE, cfg.N_INPUT_FEATURES).float()
    # hand-built fp64 tensor avoids a second fp32 quantisation; must agree with the shipped builder.
    if with_forecast:
        shipped, _ = _build_patches_tensor(
            torch.from_numpy(
                np.concatenate([feats, np.zeros((n_steps, 1))], axis=-1)
            ).reshape(n_ctx, cfg.PATCH_SIZE, cfg.N_INPUT_FEATURES).float(),
            normalization_stats=stats,
            mask_spans=full,
        )
        if fill:
            masked = torch.zeros(n_ctx + p, dtype=torch.bool)
            masked[[q - pad0 for q in idx_abs]] = True
            blind_masked_doses(shipped, masked, fill)
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

        def _cap(hr, *a, **k):
            captured['hr'] = hr.detach().clone()
            return orig(hr, *a, **k)

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
    # consumer's node rule per span, pinning head_from_hidden alongside the decode.
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
        tie_beta=model.spread_tie_beta.double(),
    )

    return {
        "name": name,
        "n_ctx": n_ctx,
        "with_forecast": with_forecast,
        "mask_spans": [[int(s), int(L)] for s, L in (mask_spans or [])],
        "raw_bg": raw['bg_absolute'].tolist(),
        "raw_doses": [raw[k].tolist() for k in CHANNEL_NAMES[1:]],
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


def dose_events_fixture() -> dict:
    """App-shaped dose rows -> the converter's point descriptors and per-step event channels.

    Covers each descriptor rule, a same-step collision, rounding onto the grid and window edges.
    """
    import finetune_data as fd
    import t1dmdroid_converter as conv
    step = conv.STEP_MS
    t0 = step * 5_866_667
    n = 96

    def cc(values: list[float]) -> str:
        return base64.b64encode(np.asarray(values, dtype='<f8').tobytes()).decode()

    meals = [
        {'ts': t0 + 10 * step, 'g': 45.0, 'gi': 70.0, 'dm': 150.0},
        {'ts': t0 + 10 * step + 60_000, 'g': 15.0, 'dm': 180.0},
        {'ts': t0 + 40 * step + 160_000, 'g': 30.0, 'gi': 35.0, 'dm': 30.0,
         'cc': cc([2.0, 8.0, 12.0, 6.0, 2.0])},
        {'ts': t0 - 2 * step, 'g': 20.0, 'gi': 50.0, 'dm': 180.0},
        {'ts': t0 + n * step, 'g': 25.0, 'gi': 50.0, 'dm': 180.0},
    ]
    doses = [
        {'ts': t0 + 10 * step, 'kd': 'BOLUS', 'u': 4.0, 'k': 3.0, 'th': 28.0, 'dm': 230.0},
        {'ts': t0 + 10 * step - 90_000, 'kd': 'BOLUS', 'u': 7.5, 'dm': 300.0},
        {'ts': t0 + 50 * step, 'kd': 'BOLUS', 'u': 2.0, 'dm': 30.0,
         'cc': cc([0.1, 0.5, 0.8, 0.4, 0.2, 0.0])},
        {'ts': t0 + 5 * step, 'kd': 'BASAL', 'u': 18.0, 'ka': 0.187, 'ke': 0.0277,
         'dm': 133 * 60.0},
        {'ts': t0 + 70 * step, 'kd': 'BASAL', 'u': 16.0, 'dm': 73 * 60.0},
        {'ts': t0 + 70 * step, 'kd': 'BASAL', 'u': 4.0, 'dm': 20.0,
         'cc': cc([1.0, 2.0, 1.0, 0.0])},
    ]
    at = lambda ts: int(round((ts - t0) / step))  # noqa: E731 — the converter's own rule
    channels = conv.record_events({'meal': meals, 'dose': doses}, at, n)

    def record(o: dict, kind: str) -> dict:
        r = {'start_ms': o['ts'], 'kind': kind, 'amount': o['g' if kind == 'meal' else 'u'],
             'duration_min': o['dm']}
        for src, dst in (('gi', 'gi'), ('k', 'k'), ('th', 'theta'), ('ka', 'ka_per_hour'),
                         ('ke', 'ke_per_hour')):
            if src in o:
                r[dst] = o[src]
        c = conv.custom_curve(o)
        if c is not None:
            r['custom_curve'] = c.tolist()
        if kind == 'meal':
            r['want'] = {'gi': conv.meal_gi(o)}
        else:
            peak, dur = conv.dose_peak_dur(o)
            r['want'] = {'peak_min': peak, 'dur_h': dur}
        return r

    return {
        "grid_start_ms": t0,
        "n_steps": n,
        "records": [record(o, 'meal') for o in meals]
        + [record(o, o['kd'].lower()) for o in doses],
        "channel_names": list(fd.EVENT_CHANNELS),
        "channels": [channels[c].astype(np.float64).tolist() for c in fd.EVENT_CHANNELS],
        # record_events is fp32; the Rust sums in fp64.
        "rel_tolerance": 1e-6,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="emit the Rust pipeline golden")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, ck = load_model(args.checkpoint)
    stats = ck["normalization_stats"]
    policy = checkpoint_masked_channel_policy(ck)
    blind = policy == MASKED_CHANNEL_POLICY_BLIND
    n_ctx = cfg.MIN_CONTEXT_PATCHES
    # Interior spans sit at fixed fractions of the context, so a short window still holds them.
    at = lambda frac: int(frac * n_ctx)  # noqa: E731

    cases = [
        build_case("forecast", model, stats, n_ctx, None, True, blind=blind),
        build_case("infill", model, stats, n_ctx, [(at(0.36), 3), (at(0.6), 5)], True,
                   blind=blind),
        # No future zone: a gap repair reads real evidence on BOTH sides of the span.
        build_case("infill_no_forecast", model, stats, n_ctx, [(at(0.48), 4)], False,
                   blind=blind),
        # Span lengths 1..4 pin decode vs layout; a second ladder covers MASK_SPAN_LENGTHS[-1].
        build_case("span_ladder", model, stats, n_ctx,
                   [(at(0.06), 1), (at(0.12), 2), (at(0.24), 3), (at(0.42), 4)], False,
                   synthetic_head=True, blind=blind),
        build_case("span_ladder_long", model, stats, n_ctx,
                   [(at(0.06), 5), (at(0.18), 7)], False, synthetic_head=True, blind=blind),
    ]

    doc = {
        "_comment": "Golden for T1DMDROID's fp64 Rust pre/post pipeline. Regenerate with "
                    "T1DMAI/exporters/rust_golden.py whenever the contract moves.",
        "arch_version": cfg.ARCH_VERSION,
        "input_layout": cfg.INPUT_LAYOUT,
        "masked_channel_policy": policy,
        "channel_names": list(CHANNEL_NAMES),
        "patch_size": cfg.PATCH_SIZE,
        "n_input_features": cfg.N_INPUT_FEATURES,
        "max_masked_patches": cfg.MAX_MASKED_PATCHES,
        "prediction_patches": cfg.PREDICTION_PATCHES,
        "normalization_stats": stats,
        # normalize is fp32, consumer fp64-rounded-once: ulps apart in ln(g)^p, not bit-identity.
        "tolerances": {"patches": 5e-6, "anchor": 1e-3, "risk": 1e-8, "head_raw_from_hidden": 1e-5},
        # head file's tensors pin hidden->nodes->spline->MLP==head_raw; fp32 spline, looser tol.
        "head": [{"name": n, "shape": list(t.shape), "values": [float(v) for v in t.reshape(-1).tolist()]}
                 for n, t in head_tensors(model)],
        "attn_digest_note": "sha256 over the boolean attend pattern, row-major, one byte "
                            "per cell (1 attend / 0 block)",
        "cases": cases,
    }
    if cfg.INPUT_LAYOUT == 'events':
        doc["dose_events"] = dose_events_fixture()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f)
        f.write("\n")
    size = os.path.getsize(args.out)
    print(f"[golden] wrote {args.out} ({size} bytes, {len(cases)} cases)")


if __name__ == "__main__":
    main()
