"""Finetune a pretrained checkpoint on the merged MetaboNet + DiaData cache.

Selection: highest mean DTS zone-A share over 30/60/90/120 min. Built at the
CHECKPOINT's architecture, not ``config.py``'s — dims come from state-dict shapes.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch


def _apply_checkpoint_dims(ckpt: dict) -> None:
    """Patch ``config`` to the checkpoint's architecture and drop bound modules."""
    import config
    sd = ckpt.get('model_ema_state_dict') or ckpt['model_state_dict']
    tc = ckpt.get('training_config', {})

    d_model, patch_dim = sd['patch_embed.weight'].shape
    n_layers = len({int(k.split('.')[1]) for k in sd if k.startswith('blocks.')})
    head_dim = sd['blocks.0.attn.q_norm.weight'].shape[0]
    assert d_model % head_dim == 0
    config.D_MODEL = int(d_model)
    config.N_HEADS = int(d_model // head_dim)
    config.HEAD_DIM = int(head_dim)
    config.N_LAYERS = int(n_layers)
    config.FFN_DIM = int(sd['blocks.0.ffn.w1.weight'].shape[0])
    config.BG_HEAD_HIDDEN = int(sd['bg_head.0.weight'].shape[0])
    config.TIME_PROBE_ENABLED = 'time_head.0.weight' in sd
    if config.TIME_PROBE_ENABLED:
        config.TIME_PROBE_HIDDEN = int(sd['time_head.0.weight'].shape[0])
    assert patch_dim % config.N_INPUT_FEATURES == 0
    config.PATCH_SIZE = int(patch_dim // config.N_INPUT_FEATURES)
    config.PATCH_DIM = int(patch_dim)
    for k in ('min_context_patches', 'max_context_patches', 'max_masked_patches',
              'mask_right_edge_quota', 'prediction_horizon_hours'):
        if tc.get(k) is not None:
            setattr(config, k.upper(), tc[k])
    if tc.get('mask_span_lengths') is not None:
        config.MASK_SPAN_LENGTHS = tuple(tc['mask_span_lengths'])
    pph = 60 // (config.PATCH_SIZE * 5)
    config._PATCHES_PER_HOUR = pph
    config.PREDICTION_PATCHES = config.PREDICTION_HORIZON_HOURS * pph
    config.MAX_SEQ_LEN = config.MAX_CONTEXT_PATCHES + config.PREDICTION_PATCHES
    config.NIGHT_LONG_HORIZON_PATCHES = config.NIGHT_LONG_HORIZON_HOURS * pph
    # model.py binds these at import; stale, time_head builds at the wrong bin count.
    config.TIME_PROBE_N_BINS = max(1, round(24.0 / config.PREDICTION_HORIZON_HOURS))
    config.TIME_PROBE_BIN_HOURS = 24.0 / config.TIME_PROBE_N_BINS

    # DataLoader workers import modules fresh, so serialize the patch for finetune_data to replay.
    keys = (
        'D_MODEL', 'N_HEADS', 'HEAD_DIM', 'N_LAYERS', 'FFN_DIM', 'BG_HEAD_HIDDEN',
        'TIME_PROBE_ENABLED', 'TIME_PROBE_HIDDEN', 'PATCH_SIZE', 'PATCH_DIM',
        'MIN_CONTEXT_PATCHES', 'MAX_CONTEXT_PATCHES', 'MAX_MASKED_PATCHES',
        'MASK_RIGHT_EDGE_QUOTA', 'MASK_SPAN_LENGTHS', 'PREDICTION_HORIZON_HOURS',
        '_PATCHES_PER_HOUR', 'PREDICTION_PATCHES', 'MAX_SEQ_LEN',
        'NIGHT_LONG_HORIZON_PATCHES', 'TIME_PROBE_N_BINS', 'TIME_PROBE_BIN_HOURS',
    )
    os.environ['T1DMAI_FINETUNE_CONFIG_PATCH'] = json.dumps(
        {k: getattr(config, k) for k in keys})

    for m in ('model', 'data', 'risk_loss', 'inference', 'attribution', 'train',
              'finetune_data'):
        sys.modules.pop(m, None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Finetune a pretrained checkpoint on the merged cache.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', default=None,
                   help='pretrained .pt to start from; omitted = random init at '
                        "config.py's architecture, stats fit on the cache itself")
    p.add_argument('--cache', default='metabonet/cache_finetune')
    p.add_argument('--train-dataset', default=None, metavar='A,B',
                   help='comma-separated sub-datasets to train on; omitted = all')
    p.add_argument('--test-dataset', default=None, metavar='A,B',
                   help='comma-separated sub-datasets to validate on; omitted = all')
    p.add_argument('--out-dir', default='checkpoints_finetune')
    p.add_argument('--total-steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--muon-lr', type=float, default=None,
                   help='default 0.002 finetuning, config.MUON_LR from scratch')
    p.add_argument('--adam-lr', type=float, default=None,
                   help='default 3e-4 finetuning, config.ADAM_LR from scratch')
    p.add_argument('--warmup-steps', type=int, default=2000)
    p.add_argument('--lr-min-ratio', type=float, default=0.01)
    p.add_argument('--ema-decay', type=float, default=None,
                   help='default: config.EMA_DECAY')
    p.add_argument('--validation-interval', type=int, default=1000)
    p.add_argument('--eval-windows', type=int, default=2048)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--eval-seed', type=int, default=1234)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--source-alpha', type=float, default=0.5,
                   help='per-source draw weight n_subjects^alpha')
    p.add_argument('--diadata-frac', type=float, default=0.0)
    p.add_argument('--gap-budget', type=float, default=0.2,
                   help='max fraction of gap patches per training window')
    p.add_argument('--max-interp-steps', type=int, default=1,
                   help='linear-fill CGM gaps up to this many 5-min steps')
    p.add_argument('--no-carbs', action='store_true',
                   help='blank the carb channel, context and zone, train and eval')
    p.add_argument('--log-interval', type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = None
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        _apply_checkpoint_dims(ckpt)

    from config import (
        ARCH_VERSION, LOSS_SCHEMA, PREDICTION_PATCHES, EMA_DECAY, MSE_ALPHA,
        GRADIENT_CLIP_NORM, MUON_LR, ADAM_LR, MUON_MOMENTUM, ADAM_WEIGHT_DECAY,
        MASK_SPAN_LENGTHS, MAX_MASKED_PATCHES, MASK_RIGHT_EDGE_QUOTA,
        TIME_PROBE_N_BINS, TIME_PROBE_LABEL_SMOOTH_BINS, TIME_PROBE_LOSS_WEIGHT,
        QUANTILE_LEVELS, METRIC_BAND_TAU_LO, METRIC_BAND_TAU_HI,
    )
    from model import T1DMAI
    from risk_loss import risk_total_loss, KendallGalWeighting
    from utils import ModelEMA, kovatchev_f_inv, time_of_day_bin_ce
    from data import checkpoint_masked_channel_policy, masked_channel_policy
    from normalization import load_normalization_stats, CHANNEL_NAMES
    from train import _build_optimizers, _update_lr, _worker_init_fn
    from T1DMSIM.simulator import BG_CLAMP_MIN
    import dts_grid
    from metrics.core.suite import band_project
    from finetune_data import (
        FinetuneCache, FinetuneTrainDataset, FinetuneEvalDataset,
        finetune_collate_fn, build_eval_windows, HORIZON_MINUTES, HORIZON_STEPS,
    )
    from torch.utils.data import DataLoader

    if ckpt is not None:
        if ckpt.get('arch_version') != ARCH_VERSION:
            sys.exit(f"--checkpoint {args.checkpoint}: arch_version "
                     f"{ckpt.get('arch_version')!r} != {ARCH_VERSION!r}")
        if checkpoint_masked_channel_policy(ckpt) != masked_channel_policy(blind=False):
            sys.exit(f"--checkpoint {args.checkpoint}: blind masked-channel policy "
                     "— this pipeline conditions on announced doses")
        stats = ckpt['normalization_stats']
        # Same guarantees load_normalization_stats gives a file: a bad channel trains silently.
        for name in CHANNEL_NAMES:
            s = stats.get(name)
            if (s is None or not np.isfinite(s.get('mean', np.nan))
                    or not np.isfinite(s.get('std', np.nan)) or s['std'] <= 0.0):
                sys.exit(f"--checkpoint {args.checkpoint}: bad normalization "
                         f"stats for channel {name!r}: {s}")
    else:
        # Scratch init has no pretrained z-space to honour, so stats are the cache's own.
        stats_path = os.path.join(args.cache, 'normalization_stats.json')
        if not os.path.exists(stats_path):
            sys.exit(f'{stats_path} missing — run: python finetune_data.py '
                     f'fit-stats --cache {args.cache}')
        stats = load_normalization_stats(stats_path)
    if args.ema_decay is None:
        args.ema_decay = EMA_DECAY
    # Finetuning wants ~10x below the from-scratch peaks; scratch wants train.py's.
    if args.muon_lr is None:
        args.muon_lr = 0.002 if ckpt is not None else MUON_LR
    if args.adam_lr is None:
        args.adam_lr = 3e-4 if ckpt is not None else ADAM_LR

    # Seed before construction so a random init is reproducible.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T1DMAI()
    if ckpt is None:
        init_from = 'random'
    elif ckpt.get('model_ema_state_dict') is not None:
        # strict=False only for the EMA shadow's exclusions; anything else is a wrong checkpoint.
        missing, unexpected = model.load_state_dict(
            ckpt['model_ema_state_dict'], strict=False)
        assert not missing and not unexpected, (
            f'EMA state dict mismatch: missing={missing} unexpected={unexpected}')
        init_from = 'model_ema_state_dict'
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        init_from = 'model_state_dict'
    model.to(device).train()
    weighting = KendallGalWeighting().to(device)
    if ckpt is not None and 'weighting_state_dict' in ckpt:
        weighting.load_state_dict(ckpt['weighting_state_dict'])

    all_sources = sorted({r['source'] for r in FinetuneCache(args.cache).subjects})

    def pick(spec: str | None, what: str) -> list[str] | None:
        """Sub-datasets named by a comma-separated spec; None is every one."""
        if spec is None:
            return None
        want = [s.strip() for s in spec.split(',') if s.strip()]
        bad = [s for s in want if s not in all_sources]
        if bad:
            raise SystemExit(f'--{what}-dataset: unknown {bad}; cache has {all_sources}')
        return want

    def view(sources: list[str] | None) -> 'FinetuneCache':
        """A cache over `sources` only; rec['start'] is absolute, so dropping rows is safe."""
        c = FinetuneCache(args.cache)
        if sources is not None:
            c.subjects = [r for r in c.subjects if r['source'] in sources]
            assert c.subjects, f'no subject in the cache belongs to {sources}'
        return c

    train_sources = pick(args.train_dataset, 'train')
    test_sources = pick(args.test_dataset, 'test')
    cache = view(train_sources)
    eval_cache = view(test_sources)
    train_ds = FinetuneTrainDataset(
        cache, stats, seed=args.seed, total_steps=args.total_steps,
        batch_size=args.batch_size, source_alpha=args.source_alpha,
        diadata_frac=args.diadata_frac, gap_budget=args.gap_budget,
        max_interp_steps=args.max_interp_steps,
        no_carbs=args.no_carbs)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=finetune_collate_fn,
        worker_init_fn=_worker_init_fn, pin_memory=(device.type == 'cuda'))

    windows = build_eval_windows(eval_cache, args.eval_windows, args.eval_seed)
    eval_ds = FinetuneEvalDataset(eval_cache, stats, windows,
                                  max_interp_steps=args.max_interp_steps,
                                  no_carbs=args.no_carbs)
    # Workerless: eval forks would copy the CUDA-holding parent on a RAM-tight box.
    eval_loader = DataLoader(
        eval_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=0, collate_fn=finetune_collate_fn)
    # shuffle=False, num_workers=0: batches walk `windows` in order, so position gives the source.
    eval_sources = [eval_cache.subjects[s]['source'] for s, _ in windows]
    eval_source_names = sorted(set(eval_sources))
    print(f'cache: {len(cache.subjects)} train subjects '
          f'({",".join(train_sources) if train_sources else "all"}); '
          f'{len(eval_cache.subjects)} eval subjects '
          f'({",".join(test_sources) if test_sources else "all"}); '
          f'eval windows: {len(windows)} over {len(eval_source_names)} sub-datasets; '
          f'device: {device}', flush=True)

    # A level resolved to a position by lookup, never a literal index.
    band_lo_idx = QUANTILE_LEVELS.index(METRIC_BAND_TAU_LO)
    band_hi_idx = QUANTILE_LEVELS.index(METRIC_BAND_TAU_HI)

    def horizon_metrics(p_, lo_, hi_, t_) -> dict:
        """DTS-A / RMSE / MARD at one horizon, median line and band projection, mg/dL."""
        if not len(t_):
            return {'n': 0, 'dts_a': None, 'rmse': None, 'mard': None,
                    'dts_a_band': None, 'rmse_band': None, 'mard_band': None,
                    'band_cov': None, 'band_width': None}
        rmse = float(np.sqrt(np.mean((p_ - t_) ** 2)))
        mard = float(100.0 * np.mean(np.abs(p_ - t_) / t_))
        # Floor-only: the ceiling half would collapse truth above 400 and score it zone A.
        t_c = np.clip(t_, BG_CLAMP_MIN, None)
        frac = dts_grid.dts_zone_fractions(dts_grid.dts_zone_counts(t_c, p_))
        dts_a = float(frac['a'] * 100.0) if frac['a'] is not None else None
        # Projected twice: each basis onto the truth vector its own metric scores.
        eff = band_project(t_, lo_, hi_)
        eff_c = band_project(t_c, lo_, hi_)
        rmse_b = float(np.sqrt(np.mean((eff - t_) ** 2)))
        mard_b = float(100.0 * np.mean(np.abs(eff - t_) / t_))
        frac_b = dts_grid.dts_zone_fractions(dts_grid.dts_zone_counts(t_c, eff_c))
        dts_a_b = float(frac_b['a'] * 100.0) if frac_b['a'] is not None else None
        # A band figure means nothing without these two: widen enough and its errors go to zero.
        cov = float(100.0 * np.mean((t_ >= lo_) & (t_ <= hi_)))
        width = float(np.mean(hi_ - lo_))
        return {'n': int(len(t_)), 'dts_a': dts_a, 'rmse': rmse, 'mard': mard,
                'dts_a_band': dts_a_b, 'rmse_band': rmse_b, 'mard_band': mard_b,
                'band_cov': cov, 'band_width': width}

    def summarize(cols: dict, source: str | None = None) -> dict:
        """Every horizon plus the two mean DTS-A figures, over one sub-dataset or all."""
        out: dict = {}
        dts_a_vals, dts_a_band_vals = [], []
        for m in HORIZON_MINUTES:
            p_, lo_, hi_, t_, s_ = cols[m]
            sel = np.ones(len(t_), dtype=bool) if source is None else (s_ == source)
            out[m] = horizon_metrics(p_[sel], lo_[sel], hi_[sel], t_[sel])
            if out[m]['dts_a'] is not None:
                dts_a_vals.append(out[m]['dts_a'])
            if out[m]['dts_a_band'] is not None:
                dts_a_band_vals.append(out[m]['dts_a_band'])
        out['mean_dts_a'] = float(np.mean(dts_a_vals)) if dts_a_vals else float('-inf')
        out['mean_dts_a_band'] = (float(np.mean(dts_a_band_vals))
                                  if dts_a_band_vals else float('-inf'))
        return out

    @torch.no_grad()
    def evaluate() -> dict:
        """Pooled metrics, with the same metrics per sub-dataset under 'by_source'."""
        model.eval()
        acc = {m: ([], [], [], [], []) for m in HORIZON_MINUTES}
        pos = 0
        for batch in eval_loader:
            q_tau, median = model(
                batch['patches'].to(device), batch['attn_mask'].to(device),
                batch['anchor_bg'].to(device), batch['mask_idx'].to(device),
            )
            # Eval span is [(n_ctx, PREDICTION_PATCHES)], so slots 0..P-1 are the forecast in order.
            B = median.shape[0]
            pred_mgdl = kovatchev_f_inv(
                median[:, :PREDICTION_PATCHES, :]).reshape(B, -1).cpu().numpy()
            # f_inv is increasing, so the fan stays ascending edge for edge.
            fan = q_tau[:, :PREDICTION_PATCHES, :, :]
            lo_mgdl = kovatchev_f_inv(
                fan[..., band_lo_idx]).reshape(B, -1).cpu().numpy()
            hi_mgdl = kovatchev_f_inv(
                fan[..., band_hi_idx]).reshape(B, -1).cpu().numpy()
            true = batch['true_bg_horizon'].numpy()
            src = np.asarray(eval_sources[pos:pos + B])
            pos += B
            for m, hs in zip(HORIZON_MINUTES, HORIZON_STEPS):
                t = true[:, hs]
                ok = np.isfinite(t)
                a = acc[m]
                a[0].append(pred_mgdl[ok, hs])
                a[1].append(lo_mgdl[ok, hs])
                a[2].append(hi_mgdl[ok, hs])
                a[3].append(t[ok])
                a[4].append(src[ok])
        assert pos == len(windows), f'eval loader covered {pos} of {len(windows)} windows'
        model.train()
        cols = {m: tuple(np.concatenate(v) if v else np.empty(0) for v in acc[m])
                for m in HORIZON_MINUTES}
        out = summarize(cols)
        out['by_source'] = {s: summarize(cols, s) for s in eval_source_names}
        return out

    def print_table(metrics: dict, best: float, best_step: int) -> None:
        # Plain columns are the median line, `b` columns the band projection.
        print('  min    DTS-A%  DTS-A%b     RMSE    RMSEb   MARD%   MARD%b'
              '    cov%   width       n')
        for m in HORIZON_MINUTES:
            r = metrics[m]
            if r['n'] == 0:
                print(f'  {m:<5} —')
                continue
            print(f"  {m:<5} {r['dts_a']:7.2f}  {r['dts_a_band']:7.2f} "
                  f"{r['rmse']:8.2f} {r['rmse_band']:8.2f} "
                  f"{r['mard']:7.2f}  {r['mard_band']:7.2f} "
                  f"{r['band_cov']:7.2f} {r['band_width']:7.2f} {r['n']:7d}")
        print(f"  mean DTS-A {metrics['mean_dts_a']:.3f}  "
              f"(best {best:.3f} @ step {best_step})   band "
              f"{metrics['mean_dts_a_band']:.3f}", flush=True)
        print_by_source(metrics.get('by_source') or {})

    def print_by_source(by: dict) -> None:
        """Median-line DTS-A / RMSE / MARD per sub-dataset; one source repeats the pooled table."""
        if len(by) < 2:
            return
        groups = (('DTS-A%', 'dts_a'), ('RMSE', 'rmse'), ('MARD%', 'mard'))
        w, first = 7, HORIZON_MINUTES[0]
        gw = w * len(HORIZON_MINUTES)
        print(f"  {'':<16}{'':>7} " + ' '.join(f'{g:^{gw}}' for g, _ in groups))
        print(f"  {'sub-dataset':<16}{'n':>7} " + ' '.join(
            ''.join(f'{m:>{w}}' for m in HORIZON_MINUTES) for _ in groups))
        for s in sorted(by, key=lambda k: -by[k][first]['n']):
            r = by[s]
            cells = ' '.join(''.join(
                (f"{r[m][k]:>{w}.2f}" if r[m][k] is not None else f"{'—':>{w}}")
                for m in HORIZON_MINUTES) for _, k in groups)
            print(f'  {s:<16}{r[first]["n"]:>7} {cells}')
        print('', flush=True)

    muon_opt, adam_opt = _build_optimizers(
        model, weighting, args.muon_lr, args.adam_lr, MUON_MOMENTUM,
        ADAM_WEIGHT_DECAY)
    ema = ModelEMA(model, decay=args.ema_decay)
    clip_params = list(model.parameters()) + list(weighting.parameters())

    os.makedirs(args.out_dir, exist_ok=True)
    log_f = open(os.path.join(args.out_dir, 'finetune_log.csv'), 'w', newline='')
    log = csv.writer(log_f)
    log.writerow(['step', 'loss', 'mean_dts_a', 'mean_dts_a_band']
                 + [f'{k}_{m}' for m in HORIZON_MINUTES
                    for k in ('dts_a', 'rmse', 'mard', 'dts_a_band', 'rmse_band',
                              'mard_band', 'band_cov', 'band_width', 'n')])

    def log_row(step: int, loss: float, metrics: dict) -> None:
        row: list = [step, f'{loss:.6f}', f"{metrics['mean_dts_a']:.4f}",
                     f"{metrics['mean_dts_a_band']:.4f}"]
        for m in HORIZON_MINUTES:
            r = metrics[m]
            row += [r['dts_a'], r['rmse'], r['mard'], r['dts_a_band'],
                    r['rmse_band'], r['mard_band'], r['band_cov'],
                    r['band_width'], r['n']]
        log.writerow(row)
        log_f.flush()

    def save(path: str, step: int, metrics: dict) -> None:
        with ema.apply_to(model):
            # clone: on CPU .cpu() returns self, and apply_to's exit restores weights IN PLACE.
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({
            'arch_version': ARCH_VERSION,
            'loss_schema': LOSS_SCHEMA,
            'model_state_dict': sd,
            # The shadow again, under the key train.py's --ema tooling expects.
            'model_ema_state_dict': sd,
            'step': step,
            'weighting_state_dict': weighting.state_dict(),
            'normalization_stats': stats,
            'mask_span_lengths': list(MASK_SPAN_LENGTHS),
            'max_masked_patches': MAX_MASKED_PATCHES,
            'mask_right_edge_quota': MASK_RIGHT_EDGE_QUOTA,
            'training_config': dict(
                ckpt.get('training_config', {}) if ckpt is not None else {},
                finetune=vars(args) | {'init_from': init_from},
                mse_alpha=MSE_ALPHA,
            ),
            'finetune_step': step,
            'finetune_metrics': {
                'mean_dts_a': metrics['mean_dts_a'],
                'mean_dts_a_band': metrics['mean_dts_a_band'],
                **{f'h{m}': metrics[m] for m in HORIZON_MINUTES},
            },
        }, path)

    print(f'baseline eval ({init_from}, {args.checkpoint}):', flush=True)
    with ema.apply_to(model):
        metrics = evaluate()
    best, best_step = metrics['mean_dts_a'], 0
    # Legal best: finetune_best.pt always exists at exit, but never clobber a trained run's best.
    best_path = os.path.join(args.out_dir, 'finetune_best.pt')
    if not os.path.exists(best_path):
        save(best_path, 0, metrics)
    print_table(metrics, best, best_step)
    log_row(0, float('nan'), metrics)

    step = 0
    t_last = time.time()
    n_skipped = 0
    for batch in train_loader:
        step += 1
        _update_lr(muon_opt, adam_opt, step, args.muon_lr, args.adam_lr,
                   args.warmup_steps, args.total_steps, args.lr_min_ratio)

        valid_b = batch['valid'].to(device)
        mask_idx_b = batch['mask_idx'].to(device)
        q_tau, median, time_pred = model(
            batch['patches'].to(device), batch['attn_mask'].to(device),
            batch['anchor_bg'].to(device), mask_idx_b, return_time=True,
        )
        loss, comps = risk_total_loss(
            q_tau, median, batch['targets'].to(device), weighting,
            valid=valid_b, mask_idx=mask_idx_b,
        )
        # Without the probe's CE here, exported time_logits go stale as the trunk moves.
        loss_bw = loss
        if time_pred is not None and 'slot_hour' in batch:
            tod_ce = time_of_day_bin_ce(
                time_pred[valid_b], batch['slot_hour'].to(device)[valid_b],
                TIME_PROBE_N_BINS, TIME_PROBE_LABEL_SMOOTH_BINS)
            loss_bw = loss + TIME_PROBE_LOSS_WEIGHT * tod_ce

        # Guard loss AND grad norm: soft-DTW's fp32 backward can NaN, poisoning every parameter.
        skipped_reason = None
        if torch.isfinite(loss_bw):
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            loss_bw.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                clip_params, GRADIENT_CLIP_NORM, error_if_nonfinite=False)
            if torch.isfinite(grad_norm):
                muon_opt.step()
                adam_opt.step()
                ema.update(model)
            else:
                skipped_reason = 'non-finite gradients'
        else:
            skipped_reason = 'non-finite loss'
        if skipped_reason is not None:
            n_skipped += 1
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)
            print(f'step {step}: {skipped_reason}, step skipped '
                  f'({n_skipped} total)', flush=True)

        if step % args.log_interval == 0:
            dt = time.time() - t_last
            t_last = time.time()
            print(f'step {step}/{args.total_steps}  loss {loss.item():.4f}  '
                  f"Q {comps['loss_Q'].item():.4f}  D {comps['loss_D'].item():.4f}  "
                  f"R {comps['loss_M'].item():.4f}  "
                  f'{dt / args.log_interval:.2f}s/step', flush=True)

        if step % args.validation_interval == 0 or step == args.total_steps:
            with ema.apply_to(model):
                metrics = evaluate()
            if metrics['mean_dts_a'] > best:
                best, best_step = metrics['mean_dts_a'], step
                save(os.path.join(args.out_dir, 'finetune_best.pt'), step, metrics)
            save(os.path.join(args.out_dir, 'finetune_last.pt'), step, metrics)
            print_table(metrics, best, best_step)
            log_row(step, loss.item(), metrics)

    log_f.close()
    print(f'done. best mean DTS-A {best:.3f} @ step {best_step} '
          f'-> {os.path.join(args.out_dir, "finetune_best.pt")}', flush=True)


if __name__ == '__main__':
    main()
