"""Multi-cohort fine-tune: pool the training patients of several cohorts into ONE
fine-tune, holding out one patient per cohort for an in-loop combined held-out
score. Also usable single-cohort (e.g. ``--datasets uvapadova``).

The validated single-cohort machinery in ``finetune.py`` is reused wholesale — the
dataset builder, optimizers, LR schedule, EMA, resilient step and held-out
evaluator are imported; only the segment pooling across cohorts is new. Defaults
reproduce the Ohio+AZT1D+Shanghai pool with the three average holdouts
(591 / AZ23 / 1003). UVA/Padova is resolved from its simglucose cache.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)              # so ``import finetune`` resolves the sibling

import numpy as np
import torch
from torch.utils.data import DataLoader

import finetune as F                   # also prepends REPO_ROOT to sys.path

DEFAULT_HOLDOUTS = {'ohiot1dm': '591', 'azt1d': 'AZ23', 'shanghai': '1003',
                    'uvapadova': 'adult#001'}


def _load_segs(ds: str):
    """Load one cohort's segments, resolving the UVA/Padova cache specially."""
    if ds == 'uvapadova':
        from realdata.uvapadova import load as _uload
        return _uload()
    return F.load_dataset(ds, root_dir=os.path.join(F.REPO_ROOT, F.DATASET_SUBPATHS[ds]))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-cohort (or single) fine-tune of a T1DMAI checkpoint.")
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--datasets', default='ohiot1dm,azt1d,shanghai', help="comma-separated cohorts to pool")
    p.add_argument('--holdouts', default=None, help="comma-separated ds:patient (default per DEFAULT_HOLDOUTS)")
    p.add_argument('--steps', type=int, default=F.FINETUNE_TOTAL_STEPS)
    p.add_argument('--batch-size', type=int, default=F.FINETUNE_BATCH_SIZE)
    p.add_argument('--lr-scale', type=float, default=F.FINETUNE_LR_SCALE)
    p.add_argument('--warmup', type=int, default=F.FINETUNE_WARMUP_STEPS)
    p.add_argument('--eval-interval', type=int, default=F.FINETUNE_EVAL_INTERVAL)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=F.FINETUNE_SEED)
    p.add_argument('--out', default=None)
    p.add_argument('--write-best', action='store_true')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    total_steps = int(args.steps)
    batch_size = int(args.batch_size)
    lr_scale = float(args.lr_scale)
    warmup_steps = int(args.warmup)
    eval_interval = int(args.eval_interval)
    seed = int(args.seed)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    holdouts = dict(DEFAULT_HOLDOUTS)
    if args.holdouts is not None:
        holdouts = {}
        for tok in args.holdouts.split(','):
            ds, pid = tok.split(':')
            holdouts[ds.strip()] = pid.strip()

    ft_segs: list = []
    cal_pool: list = []
    test_pool: list = []
    holdout_map: dict[str, str] = {}
    for ds in datasets:
        segs = _load_segs(ds)
        pats = sorted({s.patient for s in segs})
        hp = holdouts.get(ds, pats[0])
        assert hp in pats, f"holdout {hp!r} not in {ds} patients {pats}"
        holdout_map[ds] = hp
        heldout = [s for s in segs if s.patient == hp]
        other = [s for s in segs if s.patient != hp]
        cal, test = F.split_segments(heldout, ds)
        ft_segs += other
        cal_pool += cal
        test_pool += test
        print(f"[multi] {ds}: holdout={hp} ft+={len(other)} cal+={len(cal)} test+={len(test)}")

    assert len(ft_segs) > 0, "no fine-tune segments pooled"
    assert F._has_eval_window(test_pool), "pooled held-out yields no eval window"

    model, ckpt, stats = F.load_checkpoint(args.checkpoint, device)
    arch_version = ckpt.get('arch_version')
    loss_schema = ckpt.get('loss_schema')

    dataset = F.RealSegmentDataset(ft_segs, stats, seed=seed)
    assert len(dataset) > 0, "no training windows pooled"
    if len(dataset) < batch_size:
        print(f"[multi] windows ({len(dataset)}) < batch_size ({batch_size}); clamping")
        batch_size = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0, collate_fn=F.collate_fn, pin_memory=(device.type == 'cuda'))

    muon_lr = F.config.MUON_LR * lr_scale
    adam_lr = F.config.ADAM_LR * lr_scale
    # Learned Kendall-Gal weighting for the pinball/DILATE combine; its two log-sigma
    # scalars train in a dedicated AdamW group (build_optimizers) and are EMA-excluded
    # by living off-model (never passed to ModelEMA).
    weighting = F.KendallGalWeighting().to(device)
    muon_opt, adam_opt = F.build_optimizers(model, weighting, muon_lr, adam_lr)
    ema = F.ModelEMA(model, decay=F.FINETUNE_EMA_DECAY).to(device)

    ds_label = 'multi' if len(datasets) > 1 else datasets[0]
    log_path = os.path.join(F.REPO_ROOT, 'finetune', 'finetune_multi_log.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    header = ['step', 'loss_ema']
    for h in F.EVAL_HORIZONS:
        header += [f'rmse_point_{h}', f'mard_{h}', f'clarke_AB_{h}', f'hypo_recall_{h}']
    log_writer.writerow(header)
    log_file.flush()

    print(f"[multi] datasets={datasets} holdouts={holdout_map} device={device.type}")
    print(f"[multi] ft_segs={len(ft_segs)} windows={len(dataset)} cal_pool={len(cal_pool)} "
          f"test_pool={len(test_pool)} steps={total_steps} bs={batch_size} lr_scale={lr_scale}")

    best_sel = None
    best_step = -1
    best_model_sd = None
    best_ema_sd = None
    best_weighting_sd = None
    best_summ = None
    baseline_summ = None

    label = F._output_label(args.checkpoint)
    out_default = os.path.join(F.REPO_ROOT, 'finetune', f"{label}-finetune-{ds_label}.pt")
    out_path = args.out if args.out is not None else out_default

    def _write_output() -> None:
        assert (best_model_sd is not None and best_ema_sd is not None
                and best_weighting_sd is not None)
        save_dict = {
            'arch_version': arch_version, 'loss_schema': loss_schema, 'step': best_step,
            'model_state_dict': best_model_sd, 'model_ema_state_dict': best_ema_sd,
            'weighting_state_dict': best_weighting_sd,
            'normalization_stats': stats,
            'finetune_meta': {
                'dataset': ds_label, 'datasets': datasets, 'mode': 'transfer',
                'holdout': holdout_map, 'base_checkpoint': os.path.abspath(args.checkpoint),
                'total_steps': total_steps, 'lr_scale': lr_scale,
                'baseline_heldout': baseline_summ, 'best_heldout': best_summ,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        torch.save(save_dict, out_path)
        if args.write_best:
            bd = os.path.join(F.REPO_ROOT, 'checkpoints')
            os.makedirs(bd, exist_ok=True)
            torch.save(save_dict, os.path.join(bd, 't1dmai_best.pt'))

    loss_ema = None

    def _log_eval(step, res) -> None:
        row = [step, '' if loss_ema is None else f"{loss_ema:.4f}"]
        cells = []
        for h in F.EVAL_HORIZONS:
            rmse = F._metric(res, h, 'rmse_point'); mard = F._metric(res, h, 'mard')
            cab = F._metric(res, h, 'clarke_AB'); hyp = F._metric(res, h, 'hypo', 'recall')
            row += ['' if v is None else f"{v:.6f}" for v in (rmse, mard, cab, hyp)]
            cells.append(f"h{h} rmse={F._fmt(rmse)} mard={F._fmt(mard)} AB={F._fmt(cab)} hypoR={F._fmt(hyp)}")
        log_writer.writerow(row); log_file.flush()
        sel = F._selection_scalar(res)
        tag = 'baseline' if step == 0 else f"step {step}"
        print(f"[eval {tag}] loss_ema={F._fmt(loss_ema)} n_test={res.get('n_test_windows')} "
              f"sel={F._fmt(sel)} | " + " | ".join(cells))

    def _maybe_select(step, res) -> None:
        nonlocal best_sel, best_step, best_model_sd, best_ema_sd, best_weighting_sd, best_summ
        sel = F._selection_scalar(res)
        if sel is None:
            return
        if best_sel is None or sel < best_sel:
            best_sel = sel; best_step = step
            best_model_sd = F._cpu_state(model.state_dict())
            best_ema_sd = F._cpu_state(ema.state_dict())
            best_weighting_sd = F._cpu_state(weighting.state_dict())
            best_summ = F._summarize_heldout(res)
            _write_output()
            print(f"  [best] new minimum {F.SELECTION_METRIC}@{F.SELECTION_HORIZON}={sel:.3f} "
                  f"at step {step} -> wrote {out_path}")

    try:
        res0 = F.eval_heldout(model, ema, stats, cal_pool, test_pool, device)
        baseline_summ = F._summarize_heldout(res0)
        _log_eval(0, res0)
        _maybe_select(0, res0)

        data_iter = iter(loader)
        consecutive_nan = 0
        step = 0
        while step < total_steps:
            model.train()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            patches = batch['patches'].to(device, non_blocking=True)
            attn_mask = batch['attn_mask'].to(device, non_blocking=True)
            last_bg = batch['bg_formula_data']['last_bg'].to(device).float()
            true_bg_full = (batch['bg_formula_data']['true_bg_trajectory'][:, :F._PRED_STEPS]
                            .to(device).float()
                            .reshape(-1, F.config.PREDICTION_PATCHES, F.config.PATCH_SIZE))

            # Cross-window probe input (window k+1) — skipped when the penalty is off.
            next_window = None
            if F.config.TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
                _nw = batch.get('next_window')
                if _nw is not None:
                    next_window = {
                        'patches': _nw['patches'].to(device, non_blocking=True),
                        'last_bg': _nw['last_bg'].to(device, non_blocking=True).float(),
                        'valid': _nw['valid'].to(device, non_blocking=True),
                    }

            def _halve_optimizer_state() -> None:
                for grp in muon_opt.param_groups:
                    for p in grp['params']:
                        st = muon_opt.state.get(p, {})
                        if 'momentum_buffer' in st:
                            st['momentum_buffer'].mul_(0.5)
                for grp in adam_opt.param_groups:
                    for p in grp['params']:
                        st = adam_opt.state.get(p, {})
                        if 'exp_avg' in st:
                            st['exp_avg'].mul_(0.5)

            def _maybe_restore_from_ema(reason: str) -> bool:
                nonlocal consecutive_nan
                if consecutive_nan >= F.CONSECUTIVE_NAN_RESTORE:
                    model.load_state_dict(ema.state_dict(), strict=False)
                    muon_opt.state.clear()
                    adam_opt.state.clear()
                    print(f"  [RECOVERY] {consecutive_nan} consecutive {reason} — restored from EMA")
                    consecutive_nan = 0
                    return True
                return False

            def _skip_nonfinite_step(reason: str) -> None:
                nonlocal consecutive_nan, loss_ema
                consecutive_nan += 1
                print(f"  [WARNING] {reason} at step {step} (consecutive: {consecutive_nan}) — skipping")
                _maybe_restore_from_ema(reason)
                muon_opt.zero_grad(set_to_none=True)
                adam_opt.zero_grad(set_to_none=True)
                _halve_optimizer_state()
                loss_ema = 1.0 if loss_ema is None else 0.98 * loss_ema + 0.02 * 1.0

            try:
                q_tau, median, time_pred = model(patches, attn_mask, last_bg, return_time=True)
                q_tau = q_tau.float()
                median = median.float()
                loss_total, _parts = F.risk_total_loss(q_tau, median, true_bg_full, weighting)
                # Time-of-day probe co-trains the shared trunk (TIME_PROBE_DETACH=False),
                # mirroring pretraining; isolated from the logged/EMA/selection scalar.
                _tod_extra = loss_total.new_zeros(())
                _tod_loss_val = float('nan')
                _psh = batch['bg_formula_data'].get('pred_start_hour')
                if time_pred is not None and _psh is not None:
                    P = time_pred.shape[1]
                    adv = F.config.PATCH_SIZE * 5.0 / 60.0
                    base = _psh.to(time_pred.device).float().reshape(-1)                       # (B,)
                    offs = torch.arange(P, device=time_pred.device, dtype=torch.float32) * adv  # (P,)
                    tgt_hours = (base[:, None] + offs[None, :]) % 24.0                          # (B,P)
                    _tod_ce = F.time_of_day_bin_ce(
                        time_pred, tgt_hours, F.config.TIME_PROBE_N_BINS, F.config.TIME_PROBE_LABEL_SMOOTH_BINS,
                    )
                    _tod_loss = _tod_ce
                    _tod_loss_val = float(_tod_ce.detach())   # logged CE (pre cross-window)

                    # Cross-window (paired-window) phase-advance penalty (2nd forward
                    # on batch['next_window']) — same coupling pretraining applies.
                    if (next_window is not None
                            and F.config.TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0
                            and bool(next_window['valid'].any())):
                        B_nw = next_window['patches'].shape[0]
                        n_sub = (B_nw if F.config.TIME_PROBE_CROSS_WINDOW_FRACTION >= 1.0
                                 else max(1, math.ceil(F.config.TIME_PROBE_CROSS_WINDOW_FRACTION * B_nw)))
                        nw_valid_s = next_window['valid'][:n_sub]
                        if bool(nw_valid_s.any()):
                            nw_mask = attn_mask[:n_sub] if attn_mask.dim() == 3 else attn_mask
                            _, _, time_pred_next = model(
                                next_window['patches'][:n_sub], nw_mask,
                                next_window['last_bg'][:n_sub], return_time=True,
                            )
                            _tod_xwin = F.time_cross_window_consistency_loss(
                                time_pred[:n_sub], time_pred_next, F.config.TIME_PROBE_N_BINS,
                                F.config.PREDICTION_HORIZON_HOURS, valid=nw_valid_s,
                            )
                            if torch.isfinite(_tod_xwin):
                                _tod_loss = _tod_loss + F.config.TIME_PROBE_CROSS_WINDOW_WEIGHT * _tod_xwin
                    if torch.isfinite(_tod_loss):
                        _tod_extra = F.config.TIME_PROBE_LOSS_WEIGHT * _tod_loss
                loss_backward = loss_total + _tod_extra
                if not torch.isfinite(loss_backward):
                    _skip_nonfinite_step("NaN/Inf total loss")
                    step += 1
                    continue
                loss_backward.backward()
            except RuntimeError as exc:
                muon_opt.zero_grad(set_to_none=True)
                adam_opt.zero_grad(set_to_none=True)
                _skip_nonfinite_step(f"forward/loss/backward RuntimeError ({exc})")
                step += 1
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(weighting.parameters()),
                F.config.GRADIENT_CLIP_NORM, error_if_nonfinite=False)

            if torch.isfinite(grad_norm):
                F.update_lr(muon_opt, adam_opt, step, muon_lr, adam_lr,
                            warmup_steps, total_steps, F.FINETUNE_LR_MIN_RATIO)
                muon_opt.step()
                adam_opt.step()
                ema.update(model)
                consecutive_nan = 0
                loss_val = float(loss_total.item())
                loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
            else:
                consecutive_nan += 1
                print(f"  [WARNING] NaN/Inf gradient at step {step} (consecutive: {consecutive_nan})")
                _maybe_restore_from_ema("NaN gradients")
                _halve_optimizer_state()
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            step += 1
            if step % F.FINETUNE_LOG_INTERVAL == 0:
                print(f"  step {step}/{total_steps} loss_ema={F._fmt(loss_ema)} "
                      f"loss_tod={F._fmt(_tod_loss_val)} "
                      f"lr_muon={muon_opt.param_groups[0]['lr']:.2e}")
            if step % eval_interval == 0 and step < total_steps:
                res = F.eval_heldout(model, ema, stats, cal_pool, test_pool, device)
                _log_eval(step, res)
                _maybe_select(step, res)

        res_final = F.eval_heldout(model, ema, stats, cal_pool, test_pool, device)
        _log_eval(total_steps, res_final)
        _maybe_select(total_steps, res_final)
    finally:
        log_file.close()

    if best_model_sd is not None:
        _write_output()

    print()
    print("=" * 72)
    print(f"output checkpoint: {out_path}")
    print(f"best step: {best_step}  ({F.SELECTION_METRIC}@{F.SELECTION_HORIZON}={F._fmt(best_sel)})")
    print(f"pooled holdouts: {holdout_map}")
    print("per-horizon POOLED held-out  baseline -> best (Δ):")
    for h in F.EVAL_HORIZONS:
        base = None if baseline_summ is None else baseline_summ[str(h)]['rmse_point']
        best = None if best_summ is None else best_summ[str(h)]['rmse_point']
        delta = (None if (base is None or best is None) else best - base)
        print(f"  h{h:>3} rmse_point: {F._fmt(base)} -> {F._fmt(best)}"
              + ("" if delta is None else f"  (Δ {delta:+.3f})"))
    print("=" * 72)


if __name__ == '__main__':
    main()
