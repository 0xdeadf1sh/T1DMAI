"""ExecuTorch XNNPACK (CPU fp32) exporter — spike S1.

Loads the EMA checkpoint, wraps it in the modified ``head_raw`` forward (external
struct mask, in-graph ALiBi, graph cut at ``head_raw``), ``torch.export``s it,
lowers to ExecuTorch XNNPACK -> an fp32 ``<id>.xnnpack.pte``, emits the descriptor,
and verifies on host that:

  (1) the modified (struct-mask) forward matches the STOCK bool-mask forward's
      internal ``head_raw`` (max|Δ|), and
  (2) the lowered ``.pte`` run through the ExecuTorch python runtime matches the
      eager modified forward (max|Δ| < TOL).

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
from normalization import normalize, CHANNEL_NAMES
from utils import last_bg_mgdl_from_context, time_of_day_decode_bins
from inference import _build_patches_tensor
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX

from exporters.modified_forward import (
    HeadRawForward, build_struct_mask, load_model, NEG_FILL,
)
from exporters.descriptor import (
    build_descriptor, build_model_card, deploy_to_server, write_descriptor,
)

ENGINE = "executorch_xnnpack_fp32"
VERIFY_TOL = 1e-3


def executorch_version() -> str:
    return importlib.metadata.version("executorch")


def build_representative_input(
    stats: dict[str, dict[str, float]], n_ctx: int = cfg.MAX_CONTEXT_PATCHES,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]":
    """A plausible normalized ``T=52`` input from a SYNTHETIC 24 h BG series.

    No simulator / sensor needed: a smooth diurnal BG curve with two meal-driven
    excursions, a light constant basal, and two boluses — passed through the exact
    training preprocessing (clamp bg / floor carb-insulin -> normalize; no
    smoothing) and the shipped
    ``_build_patches_tensor`` so the patches / mask are construction-identical to a
    real forecast.

    The graph is a SINGLE fixed shape ``T = MAX_CONTEXT_PATCHES + PREDICTION_PATCHES
    = 52`` with the real context LEFT-PADDED into the 48 context slots (pred at the
    right edge) — exactly how training's ``collate_fn`` pads, so the absolute RoPE
    positions match training. ``_build_patches_tensor`` emits only ``n_ctx + P``
    tokens, so we prepend ``48 - n_ctx`` zero pad patches here; the struct mask blocks
    every pad COLUMN, so the pad values never reach a prediction token.

    Returns:
        (patches (1,52,PATCH_DIM), struct (52,52), bool_mask (52,52), last_bg mg/dL).
    """
    n_steps = n_ctx * cfg.PATCH_SIZE
    t = np.arange(n_steps, dtype=np.float64)
    # Diurnal drift 90..170 mg/dL + two gaussian meal excursions.
    bg = 120.0 + 30.0 * np.sin(2.0 * np.pi * t / n_steps)
    for center, amp, width in ((0.35 * n_steps, 55.0, 18.0), (0.72 * n_steps, 40.0, 14.0)):
        bg += amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    carb = np.zeros(n_steps, dtype=np.float64)
    carb[int(0.30 * n_steps):int(0.30 * n_steps) + 8] = 6.0     # ~48 g meal appearance
    carb[int(0.68 * n_steps):int(0.68 * n_steps) + 6] = 5.0
    insulin = np.full(n_steps, 0.02, dtype=np.float64)          # basal action
    insulin[int(0.30 * n_steps):int(0.30 * n_steps) + 10] += 0.25
    insulin[int(0.68 * n_steps):int(0.68 * n_steps) + 10] += 0.20

    # No smoothing: raw signals, bg clamped to the physical range, carb/insulin
    # floored at 0 (matching data._build_sample / inference).
    bg_s = np.clip(bg, BG_CLAMP_MIN, BG_CLAMP_MAX)
    carb_s = np.maximum(carb, 0.0)
    ins_s = np.maximum(insulin, 0.0)
    raw = np.stack([bg_s, carb_s, ins_s], axis=-1).astype(np.float32)   # (N, 3)
    assert list(CHANNEL_NAMES) == ["bg_absolute", "carb_intake", "insulin_combined"]
    feats = normalize(raw, stats)                                       # (N, 3) z-space
    context = torch.from_numpy(feats).reshape(n_ctx, cfg.PATCH_SIZE, cfg.N_INPUT_FEATURES)

    patches, _mask = _build_patches_tensor(context, normalization_stats=stats)  # (n_ctx+P, PATCH_DIM)
    # Left-pad the CONTEXT to MAX_CONTEXT_PATCHES so the sequence is the fixed T=52
    # (pred stays at the right edge); pad patches are zeros (masked out by struct).
    pad0 = cfg.MAX_CONTEXT_PATCHES - n_ctx
    if pad0 > 0:
        pad = torch.zeros(pad0, cfg.PATCH_DIM, dtype=patches.dtype)
        patches = torch.cat([pad, patches], dim=0)                              # (52, PATCH_DIM)
    struct = build_struct_mask(n_ctx, dtype=torch.float32)                      # (52, 52)
    bool_mask = (struct == 0.0)                                                 # padded bool mask
    last_bg = float(last_bg_mgdl_from_context(context, stats))
    return patches.unsqueeze(0).float(), struct, bool_mask, last_bg


def stock_head_raw(
    model, patches: torch.Tensor, bool_mask: torch.Tensor, last_bg: float,
) -> torch.Tensor:
    """Capture the STOCK ``T1DMAI.forward``'s internal ``head_raw`` (bool-mask path).

    ``head_raw`` is a forward local, so we transiently swap the module-global
    ``assemble_quantiles`` (bound in ``model.py``) for a capturing shim, run the real
    forward, and read the tapped tensor back.
    """
    captured: dict[str, torch.Tensor] = {}
    orig = model_module.assemble_quantiles

    def _cap(head_raw, last_bg_mgdl, carry_spread=0.0):
        captured["head_raw"] = head_raw.detach().clone()
        return orig(head_raw, last_bg_mgdl, carry_spread)

    model_module.assemble_quantiles = _cap
    try:
        last_bg_t = torch.tensor([last_bg], dtype=torch.float32)
        with torch.no_grad():
            model(patches, bool_mask, last_bg_t)
    finally:
        model_module.assemble_quantiles = orig
    return captured["head_raw"]


def export_pte(wrapper, patches: torch.Tensor, struct: torch.Tensor, out_path: str) -> dict:
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
        ep = torch.export.export(wrapper, (patches, struct), strict=False)
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


def run_pte_outputs(pte_path: str, patches: torch.Tensor, struct: torch.Tensor) -> list:
    """Execute the ``.pte`` -> the full ordered list of output tensors.

    A single-output (legacy head_raw-only) ``.pte`` returns ``[head_raw]``; the
    dual-output graph returns ``[head_raw, time_logits]``.
    """
    try:
        from executorch.runtime import Runtime
        rt = Runtime.get()
        program = rt.load_program(pte_path)
        method = program.load_method("forward")
        outs = method.execute([patches.contiguous(), struct.contiguous()])
    except Exception:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        module = _load_for_executorch(pte_path)
        outs = module.forward([patches.contiguous(), struct.contiguous()])
    return [o if isinstance(o, torch.Tensor) else torch.as_tensor(o) for o in outs]


def run_pte(pte_path: str, patches: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
    """Execute the ``.pte`` -> head_raw tensor (output 0)."""
    return run_pte_outputs(pte_path, patches, struct)[0]


def eager_time_logits(
    model, patches: torch.Tensor, bool_mask: torch.Tensor, last_bg: float,
) -> torch.Tensor:
    """Stock ``T1DMAI.forward(..., return_time=True)`` per-patch time-probe logits.

    The eager reference the exported ``time_logits`` output is validated against —
    the time analogue of ``stock_head_raw`` for the forecast head.
    """
    last_bg_t = torch.tensor([last_bg], dtype=torch.float32)
    with torch.no_grad():
        _q, _m, time_pred = model(patches, bool_mask, last_bg_t, return_time=True)
    assert time_pred is not None, "eager forward returned time_pred=None (probe absent)"
    return time_pred


def write_time_head_golden(model, patches, bool_mask, last_bg, out_path: str) -> None:
    """Emit the Rust decode golden for ``utils.time_of_day_resultant``.

    Rows pair a 12-logit input with T1DMAI's own softmax probs + resultant (hour, R),
    so the Rust port is gated against T1DMAI's geometry core. Mixes the deployed
    model's real per-patch logits with hand-built synthetic distributions (one-hot,
    uniform, bimodal, sharp, wrap-around) to exercise the circular reduction.
    """
    n = cfg.TIME_PROBE_N_BINS
    real = eager_time_logits(model, patches, bool_mask, last_bg)[0]   # (P, n_bins)

    rows_logits: list[tuple[str, list[float]]] = []
    for p in range(cfg.PREDICTION_PATCHES):
        rows_logits.append((f"model_patch{p}", real[p].tolist()))

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
        "reduction": "origin_patch (patch index 0) for the app's current-hour belief",
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

    patches, struct, bool_mask, last_bg = build_representative_input(stats)
    print(f"[input] patches={tuple(patches.shape)} struct={tuple(struct.shape)} "
          f"last_bg={last_bg:.2f} mg/dL")

    # --- struct builder must equal the stock create_attention_mask (n_ctx=48, no pad) ---
    from utils import create_attention_mask
    stock_bool_48 = create_attention_mask(cfg.MAX_CONTEXT_PATCHES, cfg.PREDICTION_PATCHES)
    struct_stock_48 = torch.where(
        stock_bool_48, torch.zeros_like(struct), torch.full_like(struct, NEG_FILL)
    )
    assert torch.equal(build_struct_mask(cfg.MAX_CONTEXT_PATCHES), struct_stock_48), (
        "build_struct_mask(48) disagrees with the stock create_attention_mask(48,4)"
    )

    # --- capture the CURRENTLY-DEPLOYED .pte's head_raw BEFORE we overwrite it, so
    #     the regression check (I2) can prove the forecast is byte-identical ---
    hr_shape = (1, cfg.PREDICTION_PATCHES, cfg.PATCH_SIZE, 1 + 2 * cfg.N_SPREADS)
    tl_shape = (1, cfg.PREDICTION_PATCHES, cfg.TIME_PROBE_N_BINS)
    pte_final = os.path.join(args.out_dir, f"{args.model_id}.xnnpack.pte")
    hr_deployed = None
    if os.path.isfile(pte_final):
        try:
            hr_deployed = run_pte(pte_final, patches, struct).reshape(hr_shape).clone()
            print(f"[regress] captured deployed head_raw from {pte_final}")
        except Exception as exc:
            print(f"[regress] could not read deployed .pte ({exc!r}); skipping regression")

    # --- (1) modified (struct) vs stock (bool) head_raw + time_logits ---
    with torch.no_grad():
        hr_mod, tl_mod = wrapper(patches, struct)
    hr_stock = stock_head_raw(model, patches, bool_mask, last_bg)
    d_struct = float((hr_mod - hr_stock).abs().max())
    print(f"[verify] modified(struct) vs stock(bool) head_raw  max|Δ| = {d_struct:.3e}")
    assert hr_mod.shape == hr_shape
    assert tl_mod.shape == tl_shape, f"time_logits shape {tuple(tl_mod.shape)} != {tl_shape}"

    # --- export + lower (two outputs: head_raw, time_logits) ---
    work_dir = args.work_dir or args.out_dir
    os.makedirs(work_dir, exist_ok=True)
    pte_name = f"{args.model_id}.xnnpack.pte"
    pte_work = os.path.join(work_dir, pte_name)
    op_info = export_pte(wrapper, patches, struct, pte_work)
    print(f"[export] wrote {pte_work} ({os.path.getsize(pte_work)} bytes)")
    print(f"[export] op-support: {op_info}")

    # --- (2) .pte vs eager modified head_raw + time_logits ---
    outs = run_pte_outputs(pte_work, patches, struct)
    assert len(outs) == 2, f"expected 2 .pte outputs (head_raw, time_logits), got {len(outs)}"
    hr_pte = outs[0].reshape(hr_shape)
    tl_pte = outs[1].reshape(tl_shape)
    d_pte = float((hr_pte - hr_mod).abs().max())
    print(f"[verify] pte vs eager-modified head_raw            max|Δ| = {d_pte:.3e}")

    # --- (2a) .pte time_logits vs eager stock forward(return_time=True) ---
    tl_eager = eager_time_logits(model, patches, bool_mask, last_bg)  # (1,P,n_bins)
    d_time = float((tl_pte - tl_eager).abs().max())
    print(f"[verify] pte vs eager time_logits                  max|Δ| = {d_time:.3e}")

    # --- (2b) REGRESSION: new head_raw byte-identical to the deployed .pte's ---
    d_regress = None
    if hr_deployed is not None:
        d_regress = float((hr_pte - hr_deployed).abs().max())
        print(f"[regress] new pte head_raw vs deployed pte head_raw max|Δ| = {d_regress:.3e}")

    # --- padded-context sanity: n_ctx=16, ensure finite + shape ---
    p16, s16, _bm16, _lb16 = build_representative_input(stats, n_ctx=cfg.MIN_CONTEXT_PATCHES)
    o16 = run_pte_outputs(pte_work, p16, s16)
    hr16 = o16[0].reshape(hr_shape)
    tl16 = o16[1].reshape(tl_shape)
    finite16 = bool(torch.isfinite(hr16).all() and torch.isfinite(tl16).all())
    print(f"[verify] padded n_ctx={cfg.MIN_CONTEXT_PATCHES} pte head_raw+time finite = {finite16}")

    # --- Rust decode golden for the time probe (opt-in) ---
    golden_path = args.golden
    if golden_path:
        write_time_head_golden(model, patches, bool_mask, last_bg, golden_path)
        print(f"[golden] wrote {golden_path}")

    # --- descriptor ---
    os.makedirs(args.out_dir, exist_ok=True)
    desc = build_descriptor(
        model_id=args.model_id, engine=ENGINE, executorch_version=et_ver,
        artifact_filename=pte_name, normalization_stats=stats, precision="fp32",
        model_card=build_model_card(model, ck),
    )
    # Named after the artifact, not a fixed "descriptor.json": several models share
    # one out-dir, and the app's ModelStore discovers `*.descriptor.json` either way.
    desc_path = os.path.join(args.out_dir, f"{args.model_id}.xnnpack.descriptor.json")
    write_descriptor(desc, desc_path)
    print(f"[descriptor] wrote {desc_path}")

    # --- copy artifact into out-dir if it was built elsewhere ---
    if os.path.abspath(pte_work) != os.path.abspath(pte_final):
        shutil.copy2(pte_work, pte_final)
    print(f"[artifact] {pte_final}")

    if args.deploy_dir:
        art, side = deploy_to_server(pte_final, desc, args.deploy_dir)
        print(f"[deploy] {art}\n[deploy] {side}")

    REGRESS_TOL = 1e-4
    ok = (
        (d_struct < VERIFY_TOL) and (d_pte < VERIFY_TOL)
        and (d_time < VERIFY_TOL) and finite16
        and (d_regress is None or d_regress < REGRESS_TOL)
    )
    print(f"\nRESULT: {'SUCCESS' if ok else 'FAIL'}")
    print(f"  executorch_version = {et_ver}")
    print(f"  pte                = {pte_final}")
    print(f"  descriptor         = {desc_path}")
    print(f"  golden             = {golden_path or 'skipped (--golden)'}")
    print(f"  d_struct           = {d_struct:.3e}")
    print(f"  d_pte              = {d_pte:.3e}")
    print(f"  d_time             = {d_time:.3e}")
    print(f"  d_regress          = {'n/a' if d_regress is None else f'{d_regress:.3e}'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
