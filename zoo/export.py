"""Export a zoo run to ExecuTorch XNNPACK: .pte, zoo descriptor, and an importable zip.

Runs under .venv-export:  .venv-export/bin/python -m zoo.export RUN --cache DIR [--out exported_zoo]
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile

import numpy as np
import torch

from config import QUANTILE_LEVELS
from zoo.registry import models
from zoo.windows import MAX_INTERP_STEPS, ZONE, drop_zone_doses, train_windows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMAT = 't1dm-zoo-1'
CAL_SEED = 1  # the fit draws seed 0; calibration windows come from a disjoint stream
MEDIAN_INDEX = QUANTILE_LEVELS.index(0.5)


def band_offsets(residual: np.ndarray) -> np.ndarray:
    """(levels, ZONE) mg/dL added to the median; the median's own row is zero."""
    off = np.stack([np.nanquantile(residual, q, axis=0) for q in QUANTILE_LEVELS])
    off[MEDIAN_INDEX] = 0.0
    return np.maximum.accumulate(off, axis=0)


def lower(graph: torch.nn.Module, args: tuple, out_path: str) -> None:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    from exporters.executorch_xnnpack import ensure_flatc
    ensure_flatc()
    with torch.no_grad():
        ep = torch.export.export(graph, args, strict=False)
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    with open(out_path, 'wb') as f:
        f.write(prog.buffer)


def run_pte(path: str, args: tuple) -> np.ndarray:
    from executorch.runtime import Runtime
    method = Runtime.get().load_program(path).load_method('forward')
    return np.asarray(method.execute([a.contiguous() for a in args])[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run', help='a directory name under zoo_runs, e.g. nhits_causal')
    p.add_argument('--cache', required=True)
    p.add_argument('--runs', default=os.path.join(ROOT, 'zoo_runs'))
    p.add_argument('--out', default=os.path.join(ROOT, 'exported_zoo'))
    p.add_argument('--cal-windows', type=int, default=50_000)
    args = p.parse_args()
    from finetune_data import FinetuneCache

    causal = args.run.endswith('_causal')
    key = args.run.removesuffix('_causal')
    model = models()[key]()
    run_dir = os.path.join(args.runs, args.run)
    with np.load(os.path.join(run_dir, 'params.npz')) as z:
        model.load({k: z[k] for k in z.files})

    cal = train_windows(FinetuneCache(args.cache), args.cal_windows, model.ctx, CAL_SEED)
    if causal:
        cal = drop_zone_doses(cal)
    offsets = band_offsets(cal.target - model.predict(cal))

    graph = model.graph().eval()
    ex = tuple(torch.from_numpy(np.ascontiguousarray(x[:1], np.float32))
               for x in (cal.bg, cal.insulin, cal.carb))
    mid = f'zoo-{args.run.replace("_", "-")}'
    os.makedirs(args.out, exist_ok=True)
    pte = os.path.join(args.out, f'{mid}.xnnpack.pte')
    lower(graph, ex, pte)
    worst = 0.0
    for i in range(32):
        a = tuple(torch.from_numpy(np.ascontiguousarray(x[i:i + 1], np.float32))
                  for x in (cal.bg, cal.insulin, cal.carb))
        worst = max(worst, float(np.abs(run_pte(pte, a) - model.predict(cal.take(np.array([i])))).max()))
    print(f'{mid}: .pte vs eager max {worst:.2e} mg/dL', flush=True)
    if worst > 0.05:
        raise SystemExit(f'.pte disagrees with eager by {worst:.3f} mg/dL')

    scores = {}
    path = os.path.join(run_dir, 'scores.json')
    if os.path.isfile(path):
        with open(path) as f:
            scores = json.load(f)['overall']
    ctx = model.ctx
    desc = {
        'format': FORMAT,
        'id': mid,
        'artifact': os.path.basename(pte),
        'engine': 'executorch_xnnpack_fp32',
        'model': key,
        'causal': causal,
        'ctx_steps': ctx,
        'zone_steps': ZONE,
        'gap_fill_interp_max_steps': MAX_INTERP_STEPS,
        'inputs': [{'name': 'bg', 'unit': 'mg/dL', 'shape': [1, ctx]},
                   {'name': 'insulin', 'unit': 'U/step', 'shape': [1, ctx + ZONE]},
                   {'name': 'carb', 'unit': 'g/step', 'shape': [1, ctx + ZONE]}],
        'output': {'name': 'median', 'unit': 'mg/dL', 'shape': [1, ZONE]},
        'quantile_levels': list(QUANTILE_LEVELS),
        'band_offsets_mgdl': np.round(offsets, 3).tolist(),
        'model_card': {'training_data': 'MetaboNet train split + DiaData',
                       'leaderboard': 'MetaboNet live', 'scores': scores},
    }
    dpath = os.path.join(args.out, f'{mid}.descriptor.json')
    with open(dpath, 'w') as f:
        json.dump(desc, f, indent=1)
    with zipfile.ZipFile(os.path.join(args.out, f'{mid}.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(dpath, os.path.basename(dpath))
        z.write(pte, os.path.basename(pte))
    print(f'wrote {mid}.zip  ({os.path.getsize(pte) / 1024:.0f} KB .pte)', flush=True)


if __name__ == '__main__':
    main()
