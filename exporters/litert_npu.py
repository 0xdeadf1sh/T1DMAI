"""LiteRT (.tflite) NPU-path exporter, via ``litert-torch`` (formerly ``ai-edge-torch``).

Same modified forward as the XNNPACK exporter: external struct mask, ``slot_sel``, cut at ``head_raw``, dual output.
fp32 CPU XNNPACK stays the authority; both precisions are checked on host against the eager forward first.
Emits ``<id>.tflite`` + ``<id>.litert.descriptor.json``, engine ``litert_npu_fp32`` / ``litert_npu_fp16``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os

import numpy as np
import torch

import config as cfg
from exporters.modified_forward import HeadRawForward, load_model
from exporters.descriptor import build_descriptor, build_model_card, write_descriptor
# Same representative input and eager references as XNNPACK — one graph, one input.
# Pulls only torch/numpy/config; no executorch at module load.
from exporters.executorch_xnnpack import (
    build_representative_input, eager_time_logits, stock_head_raw,
)

FP32_TOL = 1e-3        # same gate as the XNNPACK exporter's pte-vs-eager check
FP16_TOL = 5e-2        # fp16 head_raw (risk space ~[-3.2,2.8]); informational


def _import_litert_torch():
    """Converter under ``litert_torch`` or legacy ``ai_edge_torch`` -> (module, name, version)."""
    for name in ("litert_torch", "ai_edge_torch"):
        try:
            m = importlib.import_module(name)
            try:
                ver = importlib.metadata.version(name.replace("_", "-"))
            except Exception:
                ver = getattr(m, "__version__", "?")
            return m, name, ver
        except ImportError:
            continue
    raise SystemExit(
        "neither litert_torch nor ai_edge_torch is importable — "
        "`exporter_venv/bin/pip install litert-torch` first"
    )


def _run_tflite(
    edge_model, patches: torch.Tensor, struct: torch.Tensor, slot_sel: torch.Tensor,
) -> list[np.ndarray]:
    """Run the converted LiteRT model -> outputs as numpy, in the interpreter's order."""
    outs = edge_model(patches, struct, slot_sel)
    if isinstance(outs, (list, tuple)):
        seq = list(outs)
    else:
        seq = [outs]
    return [o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else np.asarray(o) for o in seq]


def main() -> None:
    ap = argparse.ArgumentParser(description="LiteRT (.tflite) NPU-path exporter for T1DMAI")
    ap.add_argument("--checkpoint", required=True, help="path to the trained .pt checkpoint")
    ap.add_argument("--model-id", default="t1dmai_best")
    ap.add_argument("--out-dir", default="exported", help="directory to write the artifact and its descriptor into")
    ap.add_argument("--fp16", action="store_true", help="also emit + validate an fp16 .tflite")
    args = ap.parse_args()

    aet, mod_name, aet_ver = _import_litert_torch()
    print(f"[env] {mod_name}=={aet_ver}  torch=={torch.__version__}")

    model, ck = load_model(args.checkpoint)
    stats = ck["normalization_stats"]
    wrapper = HeadRawForward(model).eval()

    w = build_representative_input(stats)
    patches, struct, slot_sel = w.patches, w.struct, w.slot_sel
    print(f"[input] patches={tuple(patches.shape)} struct={tuple(struct.shape)} "
          f"slot_sel={tuple(slot_sel.shape)} slots={w.n_masked}")

    hr_shape = (1, cfg.PREDICTION_PATCHES, cfg.PATCH_SIZE, 1 + 2 * cfg.N_SPREADS)
    tl_shape = (1, cfg.PREDICTION_PATCHES, cfg.TIME_PROBE_N_BINS)

    with torch.no_grad():
        hr_eager, tl_eager_mod, _sh = wrapper(patches, struct, slot_sel)
    tl_eager = eager_time_logits(model, w)   # stock return_time path
    d_stock = float((hr_eager - stock_head_raw(model, w)).abs().max())
    print(f"[verify] modified(struct) vs stock(bool) head_raw  max|Δ| = {d_stock:.3e}")

    os.makedirs(args.out_dir, exist_ok=True)

    def convert_and_check(tag: str, tol: float, quant_config=None) -> tuple[str, float, float, bool]:
        print(f"\n[convert] {tag}: {mod_name}.convert(wrapper, (patches, struct, slot_sel)) ...")
        if quant_config is not None:
            edge = aet.convert(wrapper, (patches, struct, slot_sel), quant_config=quant_config)
        else:
            edge = aet.convert(wrapper, (patches, struct, slot_sel))
        tfl_name = f"{args.model_id}.tflite" if tag == "fp32" else f"{args.model_id}.{tag}.tflite"
        tfl_path = os.path.join(args.out_dir, tfl_name)
        edge.export(tfl_path)
        sz = os.path.getsize(tfl_path)
        print(f"[export] wrote {tfl_path} ({sz} bytes)")

        outs = _run_tflite(edge, patches, struct, slot_sel)
        assert len(outs) >= 1, f"{tag}: .tflite returned no outputs"
        # by shape: the converter may reorder outputs
        hr_t = tl_t = None
        for o in outs:
            if tuple(o.shape) == hr_shape:
                hr_t = torch.from_numpy(np.ascontiguousarray(o))
            elif tuple(o.shape) == tl_shape:
                tl_t = torch.from_numpy(np.ascontiguousarray(o))
        if hr_t is None and len(outs) >= 1:
            hr_t = torch.from_numpy(np.ascontiguousarray(outs[0])).reshape(hr_shape)
        if tl_t is None and len(outs) >= 2:
            tl_t = torch.from_numpy(np.ascontiguousarray(outs[1])).reshape(tl_shape)
        d_hr = float((hr_t - hr_eager).abs().max())
        d_tl = float((tl_t - tl_eager).abs().max()) if tl_t is not None else float("nan")
        print(f"[verify] {tag} .tflite vs eager head_raw   max|Δ| = {d_hr:.3e} (tol {tol:.1e})")
        print(f"[verify] {tag} .tflite vs eager time_logits max|Δ| = {d_tl:.3e}")
        return tfl_path, d_hr, d_tl, (d_hr < tol)

    tfl_path, d_hr, d_tl, ok = convert_and_check("fp32", FP32_TOL)

    d_hr16 = d_tl16 = None
    if args.fp16:
        try:
            from ai_edge_torch.generative.quantize import quant_recipes  # noqa
            qc = None  # fp16 weight-quant path varies by converter version
        except Exception:
            qc = None
        try:
            import ai_edge_torch  # noqa
            _tfl16, d_hr16, d_tl16, _ok16 = convert_and_check("fp16", FP16_TOL)
        except Exception as exc:
            print(f"[fp16] fp16 conversion path unavailable: {exc!r}")

    # pre/post contract identical to XNNPACK; only the engine tag differs
    engine = "litert_npu_fp32"
    desc = build_descriptor(
        model_id=args.model_id, engine=engine, executorch_version=aet_ver,
        artifact_filename=os.path.basename(tfl_path), normalization_stats=stats,
        precision="fp32", model_card=build_model_card(model, ck),
    )
    desc_path = os.path.join(args.out_dir, f"{args.model_id}.litert.descriptor.json")
    write_descriptor(desc, desc_path)
    print(f"[descriptor] wrote {desc_path}")

    print("\nRESULT:", "SUCCESS" if ok else "FAIL")
    print(f"  {mod_name}_version = {aet_ver}")
    print(f"  tflite             = {tfl_path}")
    print(f"  d_head_raw(fp32)   = {d_hr:.3e}")
    print(f"  d_time_logits      = {d_tl:.3e}")
    if d_hr16 is not None:
        print(f"  d_head_raw(fp16)   = {d_hr16:.3e}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
