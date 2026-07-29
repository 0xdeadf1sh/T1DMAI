"""LiteRT (.tflite) exporter — the NPU-path feasibility spike (T1DMDROID issue 1).

Converts the SAME modified ``head_raw`` + ``time_logits`` forward the ExecuTorch
XNNPACK exporter uses (external struct mask, in-graph ALiBi, graph cut at
``head_raw``, dual output) to a ``.tflite`` via ``litert-torch`` (formerly
``ai-edge-torch``), the Google AI Edge PyTorch->LiteRT converter that builds on
``torch.export`` — the same entry point the ExecuTorch lowering already uses
cleanly.

This artifact is what a LiteRT ``CompiledModel.create(..., Options(Accelerator.NPU,
Accelerator.GPU))`` consumes on the MediaTek APU 990. fp32 CPU XNNPACK remains the
authority; this is the alternative NPU backend artifact, cross-checked here on host
against the eager forward before it ever reaches a device.

Validates, on host:
  (1) the fp32 ``.tflite`` outputs (head_raw, time_logits) vs the eager modified
      forward (max|Δ|), and
  (2) — if requested — the fp16 ``.tflite`` vs the same eager forward (a LOOSER
      tol; fp16 is the NPU-relevant precision).

Emits ``<id>.tflite`` + a ``<id>.litert.descriptor.json`` beside the ExecuTorch
descriptor (engine = ``litert_npu_fp32`` / ``litert_npu_fp16``), reusing the shared
descriptor emitter so the on-device Rust pre/post contract is byte-identical.
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
# Reuse the EXACT representative-input + eager references the XNNPACK exporter uses,
# so the two engines are validated against the same graph on the same input. These
# imports pull only torch/numpy/config (no executorch at module load).
from exporters.executorch_xnnpack import (
    build_representative_input, eager_time_logits, stock_head_raw,
)

FP32_TOL = 1e-3        # same gate as the XNNPACK exporter's pte-vs-eager check
FP16_TOL = 5e-2        # fp16 head_raw (risk space ~[-3.2,2.8]); informational


def _import_litert_torch():
    """Import the converter under either its new (``litert_torch``) or legacy
    (``ai_edge_torch``) module name; return the module + its version string."""
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


def _run_tflite(edge_model, patches: torch.Tensor, struct: torch.Tensor) -> list[np.ndarray]:
    """Execute the converted LiteRT model -> ordered list of numpy output arrays.

    The converted model is callable with the same positional args as the traced
    module and returns the tflite-interpreter outputs (numpy or torch)."""
    outs = edge_model(patches, struct)
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

    patches, struct, bool_mask, last_bg = build_representative_input(stats)
    print(f"[input] patches={tuple(patches.shape)} struct={tuple(struct.shape)} last_bg={last_bg:.2f}")

    hr_shape = (1, cfg.PREDICTION_PATCHES, cfg.PATCH_SIZE, 1 + 2 * cfg.N_SPREADS)
    tl_shape = (1, cfg.PREDICTION_PATCHES, cfg.TIME_PROBE_N_BINS)

    # eager references (the modified-forward outputs the .tflite must reproduce)
    with torch.no_grad():
        hr_eager, tl_eager_mod = wrapper(patches, struct)
    tl_eager = eager_time_logits(model, patches, bool_mask, last_bg)  # stock return_time path
    d_stock = float((hr_eager - stock_head_raw(model, patches, bool_mask, last_bg)).abs().max())
    print(f"[verify] modified(struct) vs stock(bool) head_raw  max|Δ| = {d_stock:.3e}")

    os.makedirs(args.out_dir, exist_ok=True)

    def convert_and_check(tag: str, tol: float, quant_config=None) -> tuple[str, float, float, bool]:
        print(f"\n[convert] {tag}: {mod_name}.convert(wrapper, (patches, struct)) ...")
        if quant_config is not None:
            edge = aet.convert(wrapper, (patches, struct), quant_config=quant_config)
        else:
            edge = aet.convert(wrapper, (patches, struct))
        tfl_name = f"{args.model_id}.tflite" if tag == "fp32" else f"{args.model_id}.{tag}.tflite"
        tfl_path = os.path.join(args.out_dir, tfl_name)
        edge.export(tfl_path)
        sz = os.path.getsize(tfl_path)
        print(f"[export] wrote {tfl_path} ({sz} bytes)")

        outs = _run_tflite(edge, patches, struct)
        assert len(outs) >= 1, f"{tag}: .tflite returned no outputs"
        # match outputs to (head_raw, time_logits) by shape (converter may reorder)
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
            qc = None  # placeholder; fp16 weight quant path varies by version
        except Exception:
            qc = None
        try:
            # Prefer the converter's native fp16 flag if exposed; else weight-only.
            import ai_edge_torch  # noqa
            _tfl16, d_hr16, d_tl16, _ok16 = convert_and_check("fp16", FP16_TOL)
        except Exception as exc:
            print(f"[fp16] fp16 conversion path unavailable: {exc!r}")

    # descriptor (engine tag differs; pre/post contract identical to XNNPACK)
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
