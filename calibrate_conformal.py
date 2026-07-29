"""Fit split-conformal quantile corrections on the reserved calibration partition.

Post-hoc calibration step (run AFTER training): loads a checkpoint, runs the model
over the disjoint ``CALIBRATION_RESERVE_*`` simulator partition, fits the
per-horizon / per-quantile ``delta`` (``conformal.fit_quantile_conformal``), stores
it back into the checkpoint under ``conformal_delta``, and reports raw-vs-calibrated
excursion-peak coverage on a disjoint test band so the effect is visible.

The corrections live in mg/dL and are SIM-fit — for real cohorts (OhioT1DM, …) re-fit
on a held-out slice of that cohort (coverage validity needs cal/test exchangeability).

Usage:
    python calibrate_conformal.py --checkpoint checkpoints/t1dmai_best.pt [--n-cal 64]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics/sim"))

import config
import conformal
from model import T1DMAI
from inference import predict
from normalization import load_normalization_stats

H = config.PREDICTION_PATCHES * config.PATCH_SIZE
LEVELS = config.QUANTILE_LEVELS
MED = LEVELS.index(0.5)
LO, HI, H10 = LEVELS.index(0.05), LEVELS.index(0.95), LEVELS.index(0.10)


def _collect(model, seeds, stats, device):
    """Run the model over fresh sim patients → (bands (N,H,K), true (N,H), peak_idx, is_exc)."""
    import sim_data as S
    from sim_data import build_sim_feature_stack, _smooth_sim_bg, _future_overrides
    from realdata.features import context_window
    from realdata.calibrate import CTX_STEPS, PRED_STEPS
    Q, T, J, E = [], [], [], []
    for _pid, d in S.make_sim_runs(seeds, 96.0):
        feats = build_sim_feature_stack(d, stats)
        cgm = _smooth_sim_bg(d['bg_observed'])
        n = (len(cgm) // config.PATCH_SIZE) * config.PATCH_SIZE
        for ps in range(CTX_STEPS, n - PRED_STEPS + 1, 4 * config.PATCH_SIZE):
            tr = cgm[ps:ps + H]
            lb = float(cgm[ps - 1])
            if len(tr) < H:
                continue
            out = predict(model, context_window(feats, ps, config.MAX_CONTEXT_PATCHES),
                          normalization_stats=stats, device=device,
                          overrides=_future_overrides(feats, ps, (0, 1)))
            Q.append(out['bands'].detach().cpu().numpy().reshape(H, config.N_QUANTILES))
            T.append(tr)
            J.append(int(np.argmax(np.abs(tr - lb))))
            E.append(tr.max() - tr.min() > 25)
    return np.asarray(Q), np.asarray(T), np.asarray(J), np.asarray(E)


def _peak_coverage(q, true, j, exc):
    c90 = np.mean([q[i, j[i], LO] <= true[i, j[i]] <= q[i, j[i], HI] for i in range(len(q)) if exc[i]])
    below = np.mean([true[i, j[i]] < q[i, j[i], LO] for i in range(len(q)) if exc[i]])
    hypo = np.mean([true[i, j[i]] < q[i, j[i], H10] for i in range(len(q)) if exc[i]])
    return c90, below, hypo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n-cal', type=int, default=64, help='reserved-partition patients to fit on')
    ap.add_argument('--no-write', action='store_true', help='report only; do not modify the checkpoint')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = T1DMAI().to(device)
    sd = ckpt['model_state_dict']
    ema = ckpt.get('model_ema_state_dict')
    model.load_state_dict({k: ema.get(k, v) for k, v in sd.items()} if ema else sd, strict=True)
    model.eval()
    stats = ckpt.get('normalization_stats') or load_normalization_stats()

    ms = config.MASTER_SEED
    cal_seeds = tuple(ms + config.CALIBRATION_RESERVE_SEED_OFFSET + i for i in range(args.n_cal))
    test_seeds = tuple(range(8000, 8012))   # disjoint test band (the sim eval seeds)
    print(f"fitting conformal on {args.n_cal} reserved patients; evaluating on {len(test_seeds)} test patients…")
    cq, ct, _, _ = _collect(model, cal_seeds, stats, device)
    tq, tt, tj, te = _collect(model, test_seeds, stats, device)

    delta = conformal.fit_quantile_conformal(cq, ct, LEVELS, MED)
    tq_cal = conformal.apply_quantile_conformal(tq, delta, MED)

    r = _peak_coverage(tq, tt, tj, te)
    c = _peak_coverage(tq_cal, tt, tj, te)
    print(f"  excursion-peak 90% coverage:   raw {r[0]:.3f} -> calibrated {c[0]:.3f}  (target 0.90)")
    print(f"  truth below lower-05 edge:     raw {r[1]:.3f} -> calibrated {c[1]:.3f}  (target 0.05)")
    print(f"  truth below tau=0.10 (hypo):   raw {r[2]:.3f} -> calibrated {c[2]:.3f}  (target 0.10)")

    if not args.no_write:
        # Store as a TORCH tensor, not a numpy array: torch.load(weights_only=True)
        # (the secure default since PyTorch 2.6, used by load_model and the whole
        # metrics/realdata pipeline) refuses to unpickle numpy's _reconstruct, so a
        # numpy delta makes the checkpoint unloadable on the safe path. Tensors and
        # plain python types (the meta below) are weights_only-safe.
        ckpt['conformal_delta'] = torch.from_numpy(delta.astype(np.float32))
        ckpt['conformal_meta'] = {'n_cal': args.n_cal, 'levels': list(LEVELS),
                                  'median_idx': MED, 'horizon_steps': H, 'space': 'mgdl', 'source': 'sim'}
        torch.save(ckpt, args.checkpoint)
        print(f"  stored conformal_delta {tuple(delta.shape)} in {args.checkpoint}")


if __name__ == '__main__':
    main()
