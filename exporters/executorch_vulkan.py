"""ExecuTorch Vulkan (GPU compute) exporter — T1DMDROID issue 20 feasibility spike.

Lowers the SAME modified ``head_raw`` + ``time_logits`` forward the XNNPACK exporter
uses (external struct mask, in-graph ALiBi, graph cut at ``head_raw``, dual output),
but with the ``VulkanPartitioner`` instead of ``XnnpackPartitioner``. It exists to
answer ONE question cheaply, on host, before anyone builds a custom AAR:

    How much of this transformer graph does the ExecuTorch Vulkan backend actually
    accept (delegate to the GPU), and which ops does it reject (fall back to the
    portable CPU kernels)?

The suspects for rejection are the transformer-specific ops the NPU/GPU delegates
tend not to cover: RoPE (complex-ish rotate-half + trig on positions), the SDPA with
a float additive mask, the per-layer ALiBi bias add, and the ``einsum('sk,bpkc->bpsc')``
step-basis projection. If the graph barely delegates, that is a legitimate STOP: report
it and skip the custom-AAR / on-device measurement, because a graph shredded into dozens
of tiny GPU subgraphs interleaved with CPU fallbacks will lose to a clean XNNPACK CPU run
on a ~2.16 M-param model where dispatch overhead already dominates.

fp32 CPU XNNPACK remains the authority (safety rule E); nothing here changes that. This
module only prints a report and, optionally, writes a ``.vulkan.pte`` + descriptor.
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
import logging
import os

import torch

import config as cfg
from exporters.modified_forward import HeadRawForward, load_model
# Reuse the EXACT representative input the XNNPACK exporter traces against, so the
# partition report is about the SAME graph on the SAME input.
from exporters.executorch_xnnpack import build_representative_input

ENGINE = "executorch_vulkan_fp32"
ENGINE_FP16 = "executorch_vulkan_fp16"
VERIFY_TOL = 1e-3


def executorch_version() -> str:
    return importlib.metadata.version("executorch")


def install_vulkan_preprocess_fake_mode_fix() -> None:
    """Cure the ExecuTorch-1.3.1 Vulkan-preprocess ``FakeTensorMode`` mismatch.

    ``VulkanBackend.preprocess`` runs the constant-fold/fuse passes (``FuseBatchNorm``,
    ``FuseClamp``, …) over EACH delegated subgraph via ``_ExportPassBase.call``, which
    re-traces the graph under a fake-tensor mode. To pick that mode it calls
    ``inputs(graph_module)``; the stock ``inputs`` unwraps a lifted-constant placeholder
    to its REAL ``.constant`` tensor. A subgraph whose sole placeholder is such a
    constant then yields ZERO fake inputs, so ``call`` spawns a FRESH ``FakeTensorMode``
    and the re-trace mixes it with the subgraph's own (single) export-time mode — the
    ``AssertionError: fake mode ... doesn't match mode ...`` that blocked serialization.

    The fix returns the placeholder's FAKE ``val`` (never ``.constant``), so every pass
    re-traces under the graph's own single mode. This is numerically exact: the interp
    only needs shape/dtype metadata; real constant values are read by the passes straight
    from the ExportedProgram's state_dict, not from these inputs. (This model has no
    BatchNorm/conv-clamp, so those passes are structural no-ops regardless.)

    Idempotent; safe to call more than once.
    """
    from executorch.exir import pass_base as _pb

    if getattr(_pb._ExportPassBase, "_t1dm_fake_mode_fix", False):
        return

    def _inputs(self, graph_module):  # mirrors the stock fallback, minus `.constant`
        def extract_input(node):
            if "val" in node.meta:
                return node.meta["val"]
            tm = node.meta.get("tensor_meta")
            if tm is not None:
                assert self.fake_tensor_mode is not None
                return _pb.FakeTensor(
                    self.fake_tensor_mode,
                    torch.empty(
                        tm.shape, dtype=tm.dtype, device="meta",
                        requires_grad=tm.requires_grad, memory_format=tm.memory_format,
                    ),
                    torch.device("cpu"),
                )
            if len(node.users) == 0:
                return None
            raise _pb.ExportPassBaseError(
                f"Cannot construct an input for graph module: {graph_module}."
            )

        return [
            extract_input(n) for n in graph_module.graph.nodes if n.op == "placeholder"
        ]

    _pb._ExportPassBase.inputs = _inputs
    _pb._ExportPassBase._t1dm_fake_mode_fix = True


def _ensure_flatc() -> None:
    """Point ``FLATC_EXECUTABLE`` at the bundled flatc (bin/ is off PATH under abs-path
    venv invocation), mirroring the XNNPACK exporter."""
    if os.environ.get("FLATC_EXECUTABLE"):
        return
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


def _target_name_node(n) -> str:
    t = n.target
    return getattr(t, "_name", None) or getattr(t, "__name__", None) or str(t)


def serialize_vulkan_pte(wrapper, patches, struct, out_path: str, force_fp16: bool = False) -> dict:
    """Full ``to_edge_transform_and_lower(Vulkan) -> to_executorch`` -> write ``out_path``.

    Requires ``install_vulkan_preprocess_fake_mode_fix()`` to have run. Returns a census
    of the SERIALIZED program: delegate-subgraph count, ops absorbed inside the delegate
    payloads, and the residual (CPU-fallback) op types left in the top graph.

    When ``force_fp16`` is set the ``VulkanPartitioner`` is built with
    ``compile_options={"force_fp16": True}``: the graph builder's ``get_effective_dtype``
    maps every fp32 GPU tensor (weights + activations) to fp16 storage+compute (constants
    clamped to the fp16 range), while the input/output STAGING buffers keep their original
    fp32 dtype (``get_staging_dtype``) — so the Java ``Tensor.fromBlob(FloatArray)`` I/O
    boundary is unchanged and only the on-GPU math runs at half precision.
    """
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

    _ensure_flatc()
    compile_options = {"force_fp16": True} if force_fp16 else None
    with torch.no_grad():
        ep = torch.export.export(wrapper, (patches, struct), strict=False)
    lowered = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner(compile_options=compile_options)])
    et_prog = lowered.to_executorch()
    with open(out_path, "wb") as f:
        f.write(et_prog.buffer)

    edm = lowered.exported_program().graph_module
    n_deleg = sum(
        1 for n in edm.graph.nodes
        if n.op == "call_function" and "executorch_call_delegate" in _target_name_node(n)
    )
    residual: "collections.Counter[str]" = collections.Counter()
    for n in edm.graph.nodes:
        if n.op != "call_function":
            continue
        nm = _target_name_node(n)
        if "getitem" in nm or "executorch_call_delegate" in nm or nm == "alloc":
            continue
        residual[nm] += 1
    absorbed = 0
    for _name, sub in edm.named_modules():
        if sub.__class__.__name__ == "LoweredBackendModule":
            try:
                om = sub.original_module
                gm = om.graph_module if hasattr(om, "graph_module") else om
                absorbed += sum(
                    1 for n in gm.graph.nodes
                    if n.op == "call_function" and "getitem" not in _target_name_node(n)
                )
            except Exception:
                pass
    return {
        "bytes": os.path.getsize(out_path),
        "delegate_subgraphs": n_deleg,
        "ops_absorbed_in_delegates": absorbed,
        "cpu_fallback_ops": sum(residual.values()),
        "cpu_fallback_op_census": dict(residual.most_common()),
    }


def cpu_faithful_deltas(wrapper, patches, struct) -> dict:
    """Reference numeric check on host: lower the SAME graph to the PORTABLE CPU runtime
    (no delegation) and compare its head_raw + time_logits to the eager modified forward.

    The Vulkan ``.pte`` is the SAME exported graph with ~95% of ops delegated; the pip
    ExecuTorch python runtime carries no ``VulkanBackend`` (it is Android-only in the
    vendored custom AAR), so the GPU-executed numerics delta is measured ON-DEVICE against
    the fp32 XNNPACK authority. This host check proves the exported graph itself is
    faithful (the math the GPU shaders must reproduce)."""
    import config as _cfg
    from executorch.exir import to_edge_transform_and_lower
    from exporters.executorch_xnnpack import run_pte_outputs

    _ensure_flatc()
    hr_shape = (1, _cfg.PREDICTION_PATCHES, _cfg.PATCH_SIZE, 1 + 2 * _cfg.N_SPREADS)
    tl_shape = (1, _cfg.PREDICTION_PATCHES, _cfg.TIME_PROBE_N_BINS)
    with torch.no_grad():
        hr_e, tl_e = wrapper(patches, struct)
        ep = torch.export.export(wrapper, (patches, struct), strict=False)
    lowered_cpu = to_edge_transform_and_lower(ep, partitioner=[])
    et_cpu = lowered_cpu.to_executorch()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".portable.pte", delete=False) as tf:
        tf.write(et_cpu.buffer)
        cpu_path = tf.name
    try:
        outs = run_pte_outputs(cpu_path, patches, struct)
    finally:
        os.remove(cpu_path)
    hr_p = outs[0].reshape(hr_shape)
    tl_p = outs[1].reshape(tl_shape)
    return {
        "head_raw_max_abs_delta": float((hr_p - hr_e).abs().max()),
        "time_logits_max_abs_delta": float((tl_p - tl_e).abs().max()),
    }


def _target_name(node) -> str:
    t = node.target
    return getattr(t, "_name", None) or getattr(t, "__name__", None) or str(t)


def _count_call_functions(gm) -> "collections.Counter[str]":
    """Count call_function targets in a graph module (aten/edge ops only)."""
    c: "collections.Counter[str]" = collections.Counter()
    for node in gm.graph.nodes:
        if node.op == "call_function":
            name = _target_name(node)
            # skip pure symbolic-int / getitem plumbing; keep tensor ops
            if name in ("<built-in function getitem>", "getitem"):
                continue
            c[name] += 1
    return c


def _delegate_op_counts(lowered_gm) -> "tuple[int, collections.Counter[str]]":
    """Walk into each executorch_call_delegate's payload and count the ops it absorbed.

    Returns (n_delegate_subgraphs, Counter of ops that ran on the GPU delegate).
    Best-effort: the LoweredBackendModule stores the delegated subgraph as
    ``original_module`` (an ExportedProgram) whose graph we can re-count.
    """
    n_subgraphs = 0
    absorbed: "collections.Counter[str]" = collections.Counter()
    # named submodules of the top graph module hold the LoweredBackendModules
    lowered_mods = {}
    for name, sub in lowered_gm.named_modules():
        # LoweredBackendModule instances carry an `original_module` ExportedProgram
        if sub.__class__.__name__ == "LoweredBackendModule":
            lowered_mods[name] = sub
    for node in lowered_gm.graph.nodes:
        if node.op == "call_function" and "executorch_call_delegate" in _target_name(node):
            n_subgraphs += 1
    for name, lm in lowered_mods.items():
        try:
            om = lm.original_module
            gm = om.graph_module if hasattr(om, "graph_module") else om
            absorbed += _count_call_functions(gm)
        except Exception:
            pass
    return n_subgraphs, absorbed


class _SkipCapture(logging.Handler):
    """Capture the partitioner's '[no operator implementation], skipping ...' lines —
    they name the exact ops the Vulkan backend has no shader for."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Vulkan Partitioner" in msg or "skipping" in msg.lower() or "subgraphs" in msg:
            self.lines.append(msg)


def partition_report(wrapper, patches, struct) -> dict:
    """torch.export -> to_edge -> VulkanPartitioner().partition() and read the tags.

    We call the partitioner DIRECTLY (not the full to_edge_transform_and_lower) so we
    read the delegation decision — which nodes are tagged for the GPU vs left for the
    portable CPU kernels — WITHOUT the SPIR-V preprocess/serialization step, which
    trips an unrelated torch-2.12 / ExecuTorch-1.3.1 fake-mode bug in a shared pass.
    The partition decision is exactly the feasibility signal Step 1 asks for.
    """
    from executorch.exir import to_edge
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

    cap = _SkipCapture()
    logging.getLogger().addHandler(cap)
    logging.getLogger().setLevel(logging.INFO)

    with torch.no_grad():
        ep = torch.export.export(wrapper, (patches, struct), strict=False)
    edge = to_edge(ep)
    edge_ep = edge.exported_program()

    # full op census of the whole edge graph, before any delegation
    base_counts = _count_call_functions(edge_ep.graph_module)
    total_ops = sum(base_counts.values())

    result = VulkanPartitioner().partition(edge_ep)
    tagged = result.tagged_exported_program
    tags = result.partition_tags or {}

    logging.getLogger().removeHandler(cap)

    absorbed: "collections.Counter[str]" = collections.Counter()
    fallback: "collections.Counter[str]" = collections.Counter()
    tag_ids: set = set()
    for node in tagged.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        name = _target_name(node)
        if name in ("<built-in function getitem>", "getitem"):
            continue
        dtag = node.meta.get("delegation_tag")
        if dtag is not None:
            absorbed[name] += 1
            tag_ids.add(dtag)
        else:
            fallback[name] += 1
    absorbed_total = sum(absorbed.values())
    fallback_total = sum(fallback.values())

    absorbed_types = set(absorbed.keys())
    rejected_types = sorted(t for t in fallback if t not in absorbed_types)

    return {
        "total_edge_ops": total_ops,
        "delegate_subgraphs": len(tag_ids) if tag_ids else len(tags),
        "ops_absorbed_by_gpu": absorbed_total,
        "ops_on_cpu_fallback": fallback_total,
        "delegation_ratio": (absorbed_total / total_ops) if total_ops else 0.0,
        "baseline_op_census": dict(base_counts.most_common()),
        "gpu_absorbed_op_census": dict(absorbed.most_common()),
        "cpu_fallback_op_census": dict(fallback.most_common()),
        "rejected_op_types": rejected_types,
        "partitioner_skip_log": cap.lines,
        "_lowered": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ExecuTorch Vulkan (GPU) partition report for T1DMAI")
    ap.add_argument("--checkpoint", required=True, help="path to the trained .pt checkpoint")
    ap.add_argument("--model-id", default="t1dmai_best")
    ap.add_argument("--out-dir", default="exported", help="directory to write the artifact and its descriptor into")
    ap.add_argument("--write-pte", action="store_true",
                    help="also serialize a .vulkan.pte + descriptor (needs the AAR to load it)")
    ap.add_argument("--fp16", action="store_true",
                    help="build the Vulkan delegate with force_fp16 (fp16 GPU storage+compute); "
                         "engine id -> executorch_vulkan_fp16, precision -> fp16")
    ap.add_argument("--deploy-dir", default=None,
                    help="with --write-pte, also copy the artifact + a <stem>.json sidecar "
                         "into a T1DMSERVER models directory (e.g. ../T1DMSERVER/data/models)")
    ap.add_argument("--report-json", default=None, help="write the partition report as JSON here")
    args = ap.parse_args()

    engine = ENGINE_FP16 if args.fp16 else ENGINE
    precision = "fp16" if args.fp16 else "fp32"

    et_ver = executorch_version()
    print(f"[env] executorch=={et_ver}  torch=={torch.__version__}  engine={engine} precision={precision}")

    model, ck = load_model(args.checkpoint)
    stats = ck["normalization_stats"]
    wrapper = HeadRawForward(model).eval()

    patches, struct, _bool_mask, last_bg = build_representative_input(stats)
    print(f"[input] patches={tuple(patches.shape)} struct={tuple(struct.shape)} last_bg={last_bg:.2f}")

    rep = partition_report(wrapper, patches, struct)
    lowered = rep.pop("_lowered")

    print("\n========== VULKAN PARTITION REPORT ==========")
    print(f"  total edge ops (pre-delegation) : {rep['total_edge_ops']}")
    print(f"  delegated GPU subgraphs         : {rep['delegate_subgraphs']}")
    print(f"  ops absorbed by Vulkan (GPU)    : {rep['ops_absorbed_by_gpu']}")
    print(f"  ops on portable CPU fallback    : {rep['ops_on_cpu_fallback']}")
    print(f"  delegation ratio (GPU ops/total): {rep['delegation_ratio']*100:.1f}%")
    print("\n  --- baseline op census (whole graph) ---")
    for k, v in rep["baseline_op_census"].items():
        print(f"      {v:4d}  {k}")
    print("\n  --- absorbed by GPU delegate ---")
    if rep["gpu_absorbed_op_census"]:
        for k, v in rep["gpu_absorbed_op_census"].items():
            print(f"      {v:4d}  {k}")
    else:
        print("      (none)")
    print("\n  --- left on CPU (portable) fallback ---")
    if rep["cpu_fallback_op_census"]:
        for k, v in rep["cpu_fallback_op_census"].items():
            print(f"      {v:4d}  {k}")
    else:
        print("      (none)")
    print("\n  --- op TYPES rejected by Vulkan (in graph, never delegated) ---")
    for t in rep["rejected_op_types"]:
        print(f"      {t}")
    print("\n  --- partitioner skip log (verbatim) ---")
    for ln in rep.get("partitioner_skip_log", []):
        print(f"      {ln}")
    print("=============================================")

    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"[report] wrote {args.report_json}")

    if args.write_pte:
        install_vulkan_preprocess_fake_mode_fix()
        os.makedirs(args.out_dir, exist_ok=True)
        pte_path = os.path.join(args.out_dir, f"{args.model_id}.vulkan.pte")
        ser = serialize_vulkan_pte(wrapper, patches, struct, pte_path, force_fp16=args.fp16)
        print(f"\n[export] wrote {pte_path} ({ser['bytes']} bytes)")
        print(f"[export] SERIALIZED delegate subgraphs   : {ser['delegate_subgraphs']}")
        print(f"[export] SERIALIZED ops in delegates     : {ser['ops_absorbed_in_delegates']}")
        print(f"[export] SERIALIZED CPU-fallback ops      : {ser['cpu_fallback_ops']} "
              f"{ser['cpu_fallback_op_census']}")

        deltas = cpu_faithful_deltas(wrapper, patches, struct)
        d_hr = deltas["head_raw_max_abs_delta"]
        d_tl = deltas["time_logits_max_abs_delta"]
        print(f"[verify] portable-CPU pte vs eager head_raw    max|Δ| = {d_hr:.3e}")
        print(f"[verify] portable-CPU pte vs eager time_logits max|Δ| = {d_tl:.3e}")
        faithful = (d_hr < VERIFY_TOL) and (d_tl < VERIFY_TOL)
        print(f"[verify] exported-graph faithful (< {VERIFY_TOL:g}) = {faithful}")
        print("[note] the pip ExecuTorch python runtime carries no VulkanBackend; the "
              "GPU-executed numerics delta is a DEVICE measurement (custom AAR), gated "
              "on-device against the fp32 XNNPACK authority (BackendInfo.agreementOk).")

        from exporters.descriptor import (
            build_descriptor, build_model_card, deploy_to_server, write_descriptor,
        )
        desc = build_descriptor(
            model_id=args.model_id, engine=engine, executorch_version=et_ver,
            artifact_filename=os.path.basename(pte_path), normalization_stats=stats,
            precision=precision, model_card=build_model_card(model, ck),
        )
        desc_path = os.path.join(args.out_dir, f"{args.model_id}.vulkan.descriptor.json")
        write_descriptor(desc, desc_path)
        print(f"[descriptor] wrote {desc_path}")

        if args.deploy_dir:
            art, side = deploy_to_server(pte_path, desc, args.deploy_dir)
            print(f"[deploy] {art}\n[deploy] {side}")
        if not faithful:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
