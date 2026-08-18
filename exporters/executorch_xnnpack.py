"""ExecuTorch XNNPACK (CPU fp32) exporter.

Loads the EMA checkpoint, wraps it in the modified forward (external struct mask,
external slot selection, graph cut at ``head_raw``), ``torch.export``s it, lowers
to ExecuTorch XNNPACK -> an fp32 ``<id>.xnnpack.pte``, writes the BG head beside it
as a flat fp32 side file, emits the descriptor, and verifies on host that:

  (1) the modified (struct-mask, slot-selection) forward matches the STOCK
      bool-mask/gather forward's internal ``head_raw``, for a trailing forecast AND
      for a masked set with an infill span in it,
  (2) the lowered ``.pte`` run through the ExecuTorch python runtime matches the
      eager modified forward, on both masked sets, and
  (3) the head side file reproduces ``head_raw`` from the graph's own
      ``slot_hidden`` — the property the on-device adapter path rests on.

CPU fp32 XNNPACK is the reference/authority; the NPU fp16 path is a separate,
deferred engine module.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil

import numpy as np
import torch

import config as cfg
import model as model_module
from data import BG_MASKED_FEAT
from normalization import normalize, CHANNEL_NAMES
from utils import last_bg_mgdl_from_context, time_of_day_decode_bins
from inference import _build_patches_tensor, _resolve_mask_spans
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

from exporters.modified_forward import (
    HeadRawForward, build_slot_selection, build_struct_mask,
    build_struct_mask_from_visible, load_model, window_labels, NEG_FILL,
)
from exporters.descriptor import (
    build_descriptor, build_model_card, deploy_to_server, write_descriptor,
)
from exporters.head_weights import write_head_weights

ENGINE = "executorch_xnnpack_fp32"
VERIFY_TOL = 1e-3
# The head side file is the SAME arithmetic in a different order, so it agrees far
# more tightly than the runtime gate above; a looser bound here would pass a head
# paired with the wrong graph.
HEAD_TOL = 1e-4


class Window:
    """One built export input: the padded patches, the additive mask, the slot
    selection, and the per-slot anchors that name the same masked set to the stock
    forward. Built by :func:`build_representative_input`; the fields are set there
    so the two paths cannot describe different masked sets."""

    patches: torch.Tensor      # (1, T, PATCH_DIM)
    struct: torch.Tensor       # (T, T) additive
    bool_mask: torch.Tensor    # (T, T) bool
    slot_sel: torch.Tensor     # (M, T) one-hot rows
    mask_idx: torch.Tensor     # (1, M) int64; padded slots repeat patch 0
    anchors: torch.Tensor      # (1, M) mg/dL
    spans: list
    n_ctx: int
    T: int
    n_masked: int


def executorch_version() -> str:
    return importlib.metadata.version("executorch")


def _slot_anchor_cells(spans, n_ctx):
    """The ``(patch, step)`` context cell each masked slot anchors on.

    One-sided and left-preferring: the last step of the span's left neighbour, or
    the first step of the right neighbour when the span opens the window. Every slot
    of one span carries the same cell. Only a VISIBLE cell may be named — feat 0 of a
    masked patch is a legal-looking z that decodes to an ordinary mg/dL, so a wrong
    index yields a plausible anchor rather than an error.
    """
    patch_idx: list[int] = []
    step_idx: list[int] = []
    for start, length in spans:
        if start > 0:
            p, s = start - 1, cfg.PATCH_SIZE - 1
        else:
            p, s = start + length, 0
        assert 0 <= p < n_ctx, (
            f"span ({start}, {length}) anchors on patch {p}, outside the {n_ctx} "
            f"context patches"
        )
        patch_idx += [p] * length
        step_idx += [s] * length
    return patch_idx, step_idx


def build_representative_input(
    stats: dict[str, dict[str, float]],
    n_ctx: int | None = None,
    seq_len: int | None = None,
    mask_spans=None,
) -> Window:
    """A plausible normalized fixed-``T`` input from a SYNTHETIC BG series.

    No simulator / sensor needed: a smooth diurnal BG curve with meal-driven
    excursions, a light constant basal, boluses and one exercise bout — passed
    through the exact training preprocessing (clamp bg / floor the rest ->
    normalize; no smoothing) and the shipped ``_build_patches_tensor`` so the
    patches / mask / masked-set bit are construction-identical to a real run.

    The graph is a SINGLE fixed shape ``T``, default
    ``MAX_CONTEXT_PATCHES + PREDICTION_PATCHES``, with the real context LEFT-PADDED
    into the context slots (the future patches at the right edge) — exactly how
    training's ``collate_fn`` pads, so the absolute RoPE positions match training.
    ``_build_patches_tensor`` emits only ``n_ctx + P`` tokens, so we prepend
    ``(T - P) - n_ctx`` zero pad patches here; the struct mask blocks every pad
    COLUMN, so the pad values never reach a masked token.

    ``mask_spans`` is the masked set in UNPADDED window coordinates (what
    ``inference.predict`` takes). None selects the trailing forecast.
    """
    T = int(seq_len or cfg.MAX_SEQ_LEN)
    c = T - cfg.PREDICTION_PATCHES
    n_ctx = int(n_ctx or c)
    n_steps = n_ctx * cfg.PATCH_SIZE
    t = np.arange(n_steps, dtype=np.float64)
    # Diurnal drift 90..170 mg/dL + gaussian meal excursions, one per simulated day.
    bg = 120.0 + 30.0 * np.sin(2.0 * np.pi * t / max(288.0, n_steps / 7.0))
    for center, amp, width in ((0.35 * n_steps, 55.0, 18.0), (0.72 * n_steps, 40.0, 14.0)):
        bg += amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    carb = np.zeros(n_steps, dtype=np.float64)
    carb[int(0.30 * n_steps):int(0.30 * n_steps) + 8] = 6.0     # ~48 g meal appearance
    carb[int(0.68 * n_steps):int(0.68 * n_steps) + 6] = 5.0
    insulin = np.full(n_steps, 0.02, dtype=np.float64)          # basal action
    insulin[int(0.30 * n_steps):int(0.30 * n_steps) + 10] += 0.25
    insulin[int(0.68 * n_steps):int(0.68 * n_steps) + 10] += 0.20
    # Exercise is a carbohydrate-EQUIVALENT disposal curve in g/step, not an
    # intensity: one bout, on the same scale as the meals above.
    exercise = np.zeros(n_steps, dtype=np.float64)
    exercise[int(0.55 * n_steps):int(0.55 * n_steps) + 12] = 1.5

    # No smoothing: raw signals, bg clamped to the physical range, the rest floored
    # at 0 (matching data._build_sample / inference).
    signals = {
        'bg_absolute': np.clip(bg, BG_CLAMP_MIN, BG_CLAMP_MAX),
        'carb_intake': np.maximum(carb, 0.0),
        'insulin_combined': np.maximum(insulin, 0.0),
        'exercise_equiv': np.maximum(exercise, 0.0),
    }
    # Stacked in CHANNEL_NAMES order — normalization owns that order, and a channel
    # added there raises here rather than silently landing in the wrong column.
    raw = np.stack([signals[name] for name in CHANNEL_NAMES], axis=-1).astype(np.float32)
    feats = normalize(raw, stats)                                       # (N, C) z-space
    # Input feats 0..C-1 are the normalized signal channels in CHANNEL_NAMES order;
    # feat BG_MASKED_FEAT is the mask BIT, which carries no statistics and whose only
    # writer on this path is ``_build_patches_tensor``, so the context leaves it 0.
    assert len(CHANNEL_NAMES) == BG_MASKED_FEAT, (
        f"{len(CHANNEL_NAMES)} signal channels but bg_masked sits at feat "
        f"{BG_MASKED_FEAT}; the signal block is no longer feats [0, {BG_MASKED_FEAT})"
    )
    ctx = np.zeros((n_steps, cfg.N_INPUT_FEATURES), dtype=np.float32)
    ctx[:, :len(CHANNEL_NAMES)] = feats
    context = torch.from_numpy(ctx).reshape(n_ctx, cfg.PATCH_SIZE, cfg.N_INPUT_FEATURES)

    spans = _resolve_mask_spans(mask_spans, n_ctx)
    patches, _bool_unpadded = _build_patches_tensor(
        context, normalization_stats=stats, mask_spans=mask_spans,
    )                                                                   # (n_ctx+P, PATCH_DIM)
    pad0 = c - n_ctx
    if pad0 > 0:
        pad = torch.zeros(pad0, cfg.PATCH_DIM, dtype=patches.dtype)
        patches = torch.cat([pad, patches], dim=0)                      # (T, PATCH_DIM)

    # Context-side masked patches, shifted into padded coordinates. The future zone
    # is masked unconditionally by window_labels.
    extra = [p + pad0 for s, L in spans for p in range(s, s + L) if p < n_ctx]
    visible, is_pad, idx_abs = window_labels(n_ctx, extra, T)
    struct = build_struct_mask_from_visible(visible, is_pad, dtype=torch.float32)
    bool_mask = (struct == 0.0)

    p_cells, s_cells = _slot_anchor_cells(spans, n_ctx)
    anchor_v = last_bg_mgdl_from_context(context, stats, p_cells, s_cells)   # (n_masked,)
    m = cfg.MAX_MASKED_PATCHES
    n_masked = len(idx_abs)
    assert n_masked == anchor_v.numel(), (
        f"{n_masked} masked patches but {anchor_v.numel()} anchors"
    )
    # Padded slots read patch 0 and carry a legal mg/dL anchor; ``valid`` discards
    # their outputs downstream.
    mask_idx = torch.zeros(1, m, dtype=torch.int64)
    mask_idx[0, :n_masked] = torch.tensor(idx_abs, dtype=torch.int64)
    anchors = torch.full((1, m), float(anchor_v[0]), dtype=torch.float32)
    anchors[0, :n_masked] = anchor_v

    w = Window.__new__(Window)
    w.patches = patches.unsqueeze(0).float()
    w.struct = struct
    w.bool_mask = bool_mask
    w.mask_idx = mask_idx
    w.anchors = anchors
    w.slot_sel = build_slot_selection(idx_abs, T)
    w.spans = spans
    w.n_ctx = n_ctx
    w.T = T
    w.n_masked = n_masked
    return w


def stock_head_raw(model, w: Window) -> torch.Tensor:
    """Capture the STOCK ``T1DMAI.forward``'s internal ``head_raw`` (bool-mask,
    gather-by-index path) on the same masked set.

    ``head_raw`` is a forward local, so we transiently swap the module-global
    ``assemble_quantiles`` (bound in ``model.py``) for a capturing shim, run the real
    forward, and read the tapped tensor back.
    """
    captured: dict[str, torch.Tensor] = {}
    orig = model_module.assemble_quantiles

    def _cap(head_raw, anchor_bg_mgdl, mask_idx=None, valid=None, carry_spread=0.0):
        captured["head_raw"] = head_raw.detach().clone()
        return orig(head_raw, anchor_bg_mgdl, mask_idx, valid, carry_spread)

    model_module.assemble_quantiles = _cap
    try:
        with torch.no_grad():
            model(w.patches, w.bool_mask, w.anchors, w.mask_idx)
    finally:
        model_module.assemble_quantiles = orig
    return captured["head_raw"]


def eager_time_logits(model, w: Window) -> torch.Tensor:
    """Stock ``T1DMAI.forward(..., return_time=True)`` per-slot time-probe logits —
    the eager reference the exported ``time_logits`` output is validated against."""
    with torch.no_grad():
        _q, _m, time_pred = model(
            w.patches, w.bool_mask, w.anchors, w.mask_idx, return_time=True,
        )
    assert time_pred is not None, "eager forward returned time_pred=None (probe absent)"
    return time_pred


def head_from_hidden(head_path: str, block: dict, slot_hidden: torch.Tensor) -> torch.Tensor:
    """Reproduce ``head_raw`` from ``slot_hidden`` and the flat head side file.

    This is the on-device adapter path in miniature: read the tensors in file order,
    run the same two-hidden-layer SiLU MLP, and expand through the within-patch
    basis. Any disagreement with the graph's own ``head_raw`` means the side file
    and the ``.pte`` are not the same head.
    """
    buf = np.fromfile(head_path, dtype="<f4")
    off = 0
    ten: dict[str, torch.Tensor] = {}
    for spec in block["tensors"]:
        n = int(np.prod(spec["shape"]))
        ten[spec["name"]] = torch.from_numpy(
            buf[off:off + n].astype(np.float32).reshape(spec["shape"]).copy()
        )
        off += n
    assert off == buf.size, f"head file has {buf.size} floats, tensors account for {off}"
    h = slot_hidden
    for name in ("l0", "l1"):
        h = torch.nn.functional.silu(
            torch.nn.functional.linear(h, ten[f"{name}.weight"], ten[f"{name}.bias"])
        )
    out = torch.nn.functional.linear(h, ten["l2.weight"], ten["l2.bias"])
    b, m = out.shape[0], out.shape[1]
    coeff = out.view(b, m, cfg.BG_HEAD_STEP_BASIS_DIM, 1 + 2 * cfg.N_SPREADS)
    return torch.einsum('sk,bmkc->bmsc', ten["step_basis"], coeff)


def export_pte(wrapper, w: Window, out_path: str) -> dict:
    """torch.export the wrapper and lower to ExecuTorch XNNPACK; write ``out_path``.

    Returns a small dict of {delegated, total, non_delegated_ops} for the op-support
    note.
    """
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    # executorch serializes the XNNPACK subgraph via the `flatc` compiler. When the
    # venv is invoked by absolute path (no activation) its bin/ is off PATH, so point
    # FLATC_EXECUTABLE at the bundled binary if the caller hasn't already.
    if not os.environ.get("FLATC_EXECUTABLE"):
        import sys
        import importlib.util
        cands = [os.path.join(sys.prefix, "bin", "flatc")]
        spec = importlib.util.find_spec("executorch")
        for loc in (getattr(spec, "submodule_search_locations", None) or []):
            cands.append(os.path.join(loc, "data", "bin", "flatc"))
        for cand in cands:
            if os.path.isfile(cand):
                os.environ["FLATC_EXECUTABLE"] = os.path.abspath(cand)
                break

    with torch.no_grad():
        ep = torch.export.export(
            wrapper, (w.patches, w.struct, w.slot_sel), strict=False,
        )
    lowered = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    et_prog = lowered.to_executorch()
    with open(out_path, "wb") as f:
        f.write(et_prog.buffer)

    # Op-support introspection: which ops stayed on the portable CPU runtime
    # (i.e. were NOT delegated to XNNPACK).
    info = {"non_delegated_ops": []}
    try:
        edm = lowered.exported_program().graph_module
        delegated = total = 0
        for node in edm.graph.nodes:
            if node.op == "call_function":
                name = getattr(node.target, "_name", str(node.target))
                if "executorch_call_delegate" in str(node.target):
                    delegated += 1
                elif name.startswith("aten") or "aten" in name:
                    total += 1
                    info["non_delegated_ops"].append(name)
        info["delegate_calls"] = delegated
        info["non_delegated_count"] = len(info["non_delegated_ops"])
    except Exception as exc:  # introspection is best-effort only
        info["introspection_error"] = repr(exc)
    return info


def run_pte_outputs(pte_path: str, patches, struct, slot_sel) -> list:
    """Execute the ``.pte`` -> the full ordered list of output tensors
    ``[head_raw, time_logits, slot_hidden]``."""
    args = [patches.contiguous(), struct.contiguous(), slot_sel.contiguous()]
    try:
        from executorch.runtime import Runtime
        rt = Runtime.get()
        program = rt.load_program(pte_path)
        method = program.load_method("forward")
        outs = method.execute(args)
    except Exception:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        module = _load_for_executorch(pte_path)
        outs = module.forward(args)
    return [o if isinstance(o, torch.Tensor) else torch.as_tensor(o) for o in outs]


def write_time_head_golden(model, w: Window, out_path: str) -> None:
    """Emit the Rust decode golden for ``utils.time_of_day_resultant``.

    Rows pair a 12-logit input with T1DMAI's own softmax probs + resultant (hour, R),
    so the Rust port is gated against T1DMAI's geometry core. Mixes the deployed
    model's real per-slot logits with hand-built synthetic distributions (one-hot,
    uniform, bimodal, sharp, wrap-around) to exercise the circular reduction.
    """
    n = cfg.TIME_PROBE_N_BINS
    real = eager_time_logits(model, w)[0]                              # (M, n_bins)

    rows_logits: list[tuple[str, list[float]]] = []
    for p in range(min(cfg.PREDICTION_PATCHES, real.shape[0])):
        rows_logits.append((f"model_slot{p}", real[p].tolist()))

    def onehot(k, hi=8.0):
        v = [0.0] * n
        v[k] = hi
        return v
    rows_logits += [
        # uniform + antipodal are R->0 sentinels: the resultant vanishes so `hour` is
        # pure cancellation noise (NOT a portable target); the Rust test must gate the
        # hour assertion on R >= R_degenerate_eps and only check R there.
        ("uniform_zeros", [0.0] * n),
        ("antipodal_0_6", [3.0 if k in (0, 6) else 0.0 for k in range(n)]),
        ("onehot_bin0", onehot(0)),
        ("onehot_bin6", onehot(6)),
        ("onehot_bin11_wrap", onehot(11)),
        ("bimodal_0_3", [3.0 if k in (0, 3) else 0.0 for k in range(n)]),
        ("adjacent_11_0_wrap", [2.5 if k in (11, 0) else 0.0 for k in range(n)]),
        ("ramp", [0.30 * k for k in range(n)]),
        ("neg_sharp_bin3", [(-4.0 if k != 3 else 4.0) for k in range(n)]),
    ]

    R_DEGEN_EPS = 1e-6
    rows = []
    for name, logits in rows_logits:
        lg = torch.tensor(logits, dtype=torch.float32)
        probs = torch.softmax(lg, dim=-1)
        hour, R = time_of_day_decode_bins(lg, n)
        Rv = float(R.item())
        rows.append({
            "name": name,
            "logits": [float(x) for x in logits],
            "probs": [float(x) for x in probs.tolist()],
            "hour": float(hour.item()),
            "R": Rv,
            # hour is a well-defined circular target only when the resultant is
            # non-degenerate; at R~0 it is FP-noise and must not be asserted.
            "hour_defined": Rv >= R_DEGEN_EPS,
        })

    centers = [(k + 0.5) * (24.0 / n) for k in range(n)]
    doc = {
        "_comment": "Golden for the Rust time-probe decode. Reproduce softmax + "
                    "utils.time_of_day_resultant/decode_bins: probs=softmax(logits); "
                    "res=sum_k probs[k]*(cos th_k, sin th_k), th_k=2*pi*center_hours[k]/24; "
                    "hour=(atan2(sin,cos) mod 2pi)*24/2pi; R=hypot(cos,sin).",
        "n_bins": n,
        "bin_hours": cfg.TIME_PROBE_BIN_HOURS,
        "bin_centers_hours": centers,
        "reduction": "the slot holding the first forecast patch, for the app's "
                     "current-hour belief",
        "hour_tol": 1e-3,
        "R_tol": 1e-4,
        "R_degenerate_eps": R_DEGEN_EPS,
        "hour_note": "assert hour (circular, mod 24) only where hour_defined is true; "
                     "at R<R_degenerate_eps the resultant vanishes and hour is FP-noise.",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="ExecuTorch XNNPACK exporter for T1DMAI")
    ap.add_argument("--checkpoint", required=True, help="path to the trained .pt checkpoint")
    ap.add_argument("--model-id", default="t1dmai_best")
    ap.add_argument("--out-dir", default="exported", help="directory to write the artifact and its descriptor into")
    ap.add_argument("--work-dir", default=None, help="where to write the .pte before copy")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="fixed graph length T (default MAX_CONTEXT_PATCHES + "
                         "PREDICTION_PATCHES). A shorter T is a cheaper artifact with "
                         "a shorter memory; its descriptor reports the context it accepts.")
    ap.add_argument("--deploy-dir", default=None,
                    help="also copy the artifact + a <stem>.json sidecar into a T1DMSERVER "
                         "models directory (e.g. ../T1DMSERVER/data/models)")
    ap.add_argument("--golden", default=None,
                    help="write the Rust time-probe decode golden to this path "
                         "(default: skip — the golden is per-ARCHITECTURE, not per-model, "
                         "so exporting several checkpoints must not keep rewriting it)")
    args = ap.parse_args()

    et_ver = executorch_version()
    print(f"[env] executorch=={et_ver}  torch=={torch.__version__}")

    model, ck = load_model(args.checkpoint)
    stats = ck["normalization_stats"]
    wrapper = HeadRawForward(model).eval()

    T = int(args.seq_len or cfg.MAX_SEQ_LEN)
    c = T - cfg.PREDICTION_PATCHES
    m = cfg.MAX_MASKED_PATCHES

    # Two masked sets over the same geometry: the trailing forecast the app runs
    # every cycle, and one with an infill span in the middle of the context — the
    # case the previous right-edge export could not represent at all.
    w_fc = build_representative_input(stats, seq_len=T)
    infill_spans = [(c // 2, 4), (c, cfg.PREDICTION_PATCHES)]
    w_inf = build_representative_input(stats, seq_len=T, mask_spans=infill_spans)
    print(f"[input] T={T} patches={tuple(w_fc.patches.shape)} "
          f"struct={tuple(w_fc.struct.shape)} slot_sel={tuple(w_fc.slot_sel.shape)}")
    print(f"[input] forecast slots={w_fc.n_masked} infill slots={w_inf.n_masked}")

    # --- struct builder must equal the stock create_attention_mask (no pad) ---
    from utils import create_attention_mask
    stock_bool = create_attention_mask(c, cfg.PREDICTION_PATCHES)
    struct_stock = torch.where(
        stock_bool, torch.zeros(T, T), torch.full((T, T), NEG_FILL)
    )
    assert torch.equal(build_struct_mask(c, T), struct_stock), (
        "build_struct_mask disagrees with the stock create_attention_mask"
    )

    hr_shape = (1, m, cfg.PATCH_SIZE, 1 + 2 * cfg.N_SPREADS)
    tl_shape = (1, m, cfg.TIME_PROBE_N_BINS)
    sh_shape = (1, m, cfg.D_MODEL)

    # --- (1) modified (struct + slot_sel) vs stock (bool + gather) ---
    deltas: dict[str, float] = {}
    for name, w in (("forecast", w_fc), ("infill", w_inf)):
        with torch.no_grad():
            hr_mod, tl_mod, sh_mod = wrapper(w.patches, w.struct, w.slot_sel)
        assert hr_mod.shape == hr_shape, f"{name}: head_raw {tuple(hr_mod.shape)} != {hr_shape}"
        assert tl_mod.shape == tl_shape, f"{name}: time_logits {tuple(tl_mod.shape)} != {tl_shape}"
        assert sh_mod.shape == sh_shape, f"{name}: slot_hidden {tuple(sh_mod.shape)} != {sh_shape}"
        hr_stock = stock_head_raw(model, w)
        # Only the REAL slots are compared: a padded slot repeats patch 0 in both
        # paths, so it agrees, but nothing downstream reads it.
        n = w.n_masked
        d = float((hr_mod[:, :n] - hr_stock[:, :n]).abs().max())
        deltas[f"struct_{name}"] = d
        print(f"[verify] modified vs stock head_raw ({name:8s})     max|Δ| = {d:.3e}")

    # --- export + lower (three inputs, three outputs) ---
    work_dir = args.work_dir or args.out_dir
    os.makedirs(work_dir, exist_ok=True)
    pte_name = f"{args.model_id}.xnnpack.pte"
    pte_work = os.path.join(work_dir, pte_name)
    op_info = export_pte(wrapper, w_fc, pte_work)
    print(f"[export] wrote {pte_work} ({os.path.getsize(pte_work)} bytes)")
    print(f"[export] op-support: {op_info}")

    # --- head side file (the adapter seam) ---
    os.makedirs(args.out_dir, exist_ok=True)
    head_name = f"{args.model_id}.head.bin"
    head_path = os.path.join(args.out_dir, head_name)
    head_block = write_head_weights(model, head_path)
    print(f"[head] wrote {head_path} ({head_block['bytes']} bytes, "
          f"sha256={head_block['sha256'][:12]}…)")

    # --- (2) .pte vs eager modified, on BOTH masked sets ---
    for name, w in (("forecast", w_fc), ("infill", w_inf)):
        outs = run_pte_outputs(pte_work, w.patches, w.struct, w.slot_sel)
        assert len(outs) == 3, (
            f"expected 3 .pte outputs (head_raw, time_logits, slot_hidden), got {len(outs)}"
        )
        hr_pte = outs[0].reshape(hr_shape)
        tl_pte = outs[1].reshape(tl_shape)
        sh_pte = outs[2].reshape(sh_shape)
        with torch.no_grad():
            hr_mod, _tl_mod, _sh_mod = wrapper(w.patches, w.struct, w.slot_sel)
        n = w.n_masked
        deltas[f"pte_{name}"] = float((hr_pte[:, :n] - hr_mod[:, :n]).abs().max())
        print(f"[verify] pte vs eager head_raw     ({name:8s})     max|Δ| = "
              f"{deltas[f'pte_{name}']:.3e}")

        tl_eager = eager_time_logits(model, w)
        deltas[f"time_{name}"] = float((tl_pte[:, :n] - tl_eager[:, :n]).abs().max())
        print(f"[verify] pte vs eager time_logits  ({name:8s})     max|Δ| = "
              f"{deltas[f'time_{name}']:.3e}")

        # --- (3) the head side file reproduces head_raw from the graph's own hidden ---
        hr_head = head_from_hidden(head_path, head_block, sh_pte)
        deltas[f"head_{name}"] = float((hr_head[:, :n] - hr_pte[:, :n]).abs().max())
        print(f"[verify] head file vs pte head_raw ({name:8s})     max|Δ| = "
              f"{deltas[f'head_{name}']:.3e}")

    # --- padded-context sanity: the shortest context the artifact accepts ---
    w16 = build_representative_input(stats, n_ctx=cfg.MIN_CONTEXT_PATCHES, seq_len=T)
    o16 = run_pte_outputs(pte_work, w16.patches, w16.struct, w16.slot_sel)
    finite16 = bool(all(torch.isfinite(o).all() for o in o16))
    print(f"[verify] padded n_ctx={cfg.MIN_CONTEXT_PATCHES} pte outputs finite = {finite16}")

    # --- Rust decode golden for the time probe (opt-in) ---
    golden_path = args.golden
    if golden_path:
        write_time_head_golden(model, w_fc, golden_path)
        print(f"[golden] wrote {golden_path}")

    # --- descriptor ---
    desc = build_descriptor(
        model_id=args.model_id, engine=ENGINE, executorch_version=et_ver,
        artifact_filename=pte_name, normalization_stats=stats, precision="fp32",
        model_card=build_model_card(model, ck), head=head_block, seq_len=T,
    )
    # Named after the artifact, not a fixed "descriptor.json": several models share
    # one out-dir, and the app's ModelStore discovers `*.descriptor.json` either way.
    desc_path = os.path.join(args.out_dir, f"{args.model_id}.xnnpack.descriptor.json")
    write_descriptor(desc, desc_path)
    print(f"[descriptor] wrote {desc_path}")

    # --- copy artifact into out-dir if it was built elsewhere ---
    pte_final = os.path.join(args.out_dir, pte_name)
    if os.path.abspath(pte_work) != os.path.abspath(pte_final):
        shutil.copy2(pte_work, pte_final)
    print(f"[artifact] {pte_final}")

    if args.deploy_dir:
        art, side = deploy_to_server(pte_final, desc, args.deploy_dir)
        print(f"[deploy] {art}\n[deploy] {side}")

    ok = finite16 and all(
        v < (HEAD_TOL if k.startswith("head_") else VERIFY_TOL) for k, v in deltas.items()
    )
    print(f"\nRESULT: {'SUCCESS' if ok else 'FAIL'}")
    print(f"  executorch_version = {et_ver}")
    print(f"  pte                = {pte_final}")
    print(f"  head               = {head_path}")
    print(f"  descriptor         = {desc_path}")
    print(f"  golden             = {golden_path or 'skipped (--golden)'}")
    for k in sorted(deltas):
        print(f"  {k:18s} = {deltas[k]:.3e}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
