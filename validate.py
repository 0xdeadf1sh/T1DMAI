"""train.py's validation table for one checkpoint: simulator patients, or a backup's test days.

Built at the CHECKPOINT's architecture and masked-channel policy, under its own normalization stats.
Without ``--backup`` the patients are the ones train.py validates on for that master seed."""

import argparse
import sys

import numpy as np
import torch

import config
from finetune import _apply_checkpoint_dims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('--backup', default=None,
                   help='.t1dmbak scored on its test days; omitted = simulator patients')
    p.add_argument('--test-days', type=int, default=None,
                   help="trailing backup days scored; default t1dmdroid_converter's TEST_DAYS")
    p.add_argument('--cache-path', default=None,
                   help='simulator cache whose val partition is drawn; omitted = simulate live')
    p.add_argument('--n-windows', type=int, default=config.VALIDATION_N_PATIENTS)
    p.add_argument('--seed', type=int, default=None,
                   help="master seed; default the checkpoint's, else config.py's")
    p.add_argument('--no-ema', action='store_true', help='live weights, not the EMA shadow')
    p.add_argument('--device', default=None, help='default cuda, then mps, then cpu')
    p.add_argument('--bg-hypo-threshold', type=float, default=config.BG_HYPO_THRESHOLD)
    p.add_argument('--bg-hyper-threshold', type=float, default=config.BG_HYPER_THRESHOLD)
    a = p.parse_args()
    if a.backup is not None and a.cache_path is not None:
        p.error('--backup and --cache-path are exclusive')
    if a.backup is None and a.test_days is not None:
        p.error('--test-days needs --backup')
    if a.test_days is not None and a.test_days < 1:
        p.error('--test-days must be at least 1')
    if a.n_windows < 1:
        p.error('--n-windows must be at least 1')
    return a


def backup_stretches(path: str, test_days: int) -> list[dict[str, np.ndarray]]:
    """The test days' gap-free runs, keyed as ``data._build_sample`` reads a simulator run.

    Hour and day are local, per step from its own sample's ``tz`` (SPEC/invariants.md §2); every
    step of a run holds a measured sample, so every step has one.
    """
    from t1dmdroid_converter import STEP_MS, held_out_start, read_archive, record_channels
    kinds = read_archive(path)
    r = record_channels(kinds)
    n = len(r['bg'])
    tz_min = np.zeros(n, dtype=np.int64)
    for s in kinds['sample']:
        tz_min[(s['ts'] - r['t0_ms']) // STEP_MS] = int(s['tz'])
    local_s = (r['t0_ms'] + np.arange(n, dtype=np.int64) * STEP_MS) // 1000 + tz_min * 60

    scored = np.isfinite(r['bg'])
    scored[:max(held_out_start(n, test_days), 0)] = False
    edges = np.flatnonzero(np.diff(np.concatenate([[0], scored.astype(np.int8), [0]])))
    return [{
        'bg_observed': r['bg'][a:b],
        'total_carb': r['carb'][a:b],
        'total_insulin': r['insulin'][a:b],
        'total_exercise': r['exercise'][a:b],
        'hour_of_day': (local_s[a:b] % 86400) / 3600.0,
        'day': local_s[a:b] // 86400,
    } for a, b in zip(edges[::2], edges[1::2])]


class BackupWindows(torch.utils.data.Dataset):
    """Window ``i``: a stretch drawn by length, then ``_build_sample``'s own draw, both off
    ``default_rng([seed, i])``. Stretches too short for any window are dropped."""

    def __init__(self, stretches: list[dict[str, np.ndarray]], n_windows: int,
                 stats: dict[str, dict[str, float]], seed: int, blind: bool) -> None:
        from data import _build_sample
        self.build = _build_sample
        self.stats = stats
        self.seed = seed
        self.blind = blind
        self.n_windows = n_windows
        self.stretches = [s for s in stretches if self._fits(s)]
        if not self.stretches:
            raise SystemExit('no gap-free test-day stretch is long enough for one window')
        lengths = np.array([len(s['bg_observed']) for s in self.stretches], dtype=np.float64)
        self.p = lengths / lengths.sum()

    def _fits(self, stretch: dict[str, np.ndarray]) -> bool:
        # _build_sample raises iff no origin fits at MIN_CONTEXT_PATCHES, whatever it draws.
        try:
            self.build(stretch, 0.0, self.stats, np.random.default_rng(0), blind=self.blind)
        except RuntimeError:
            return False
        return True

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, i])
        stretch = self.stretches[int(rng.choice(len(self.stretches), p=self.p))]
        return self.build(stretch, 0.0, self.stats, rng, blind=self.blind)


def main() -> None:
    a = parse_args()
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ckpt.get('arch_version') != config.ARCH_VERSION:
        sys.exit(f"{a.checkpoint}: arch_version {ckpt.get('arch_version')!r} != "
                 f"config.py {config.ARCH_VERSION!r}")
    stats = ckpt.get('normalization_stats')
    if stats is None:
        sys.exit(f'{a.checkpoint}: no normalization_stats')
    _apply_checkpoint_dims(ckpt)

    import train
    from data import T1DMDataset, checkpoint_masked_channel_policy, masked_channel_policy
    from model import T1DMAI
    from risk_loss import KendallGalWeighting

    blind = checkpoint_masked_channel_policy(ckpt) == masked_channel_policy(blind=True)
    if blind:
        import train_blind as trainer
    else:
        trainer = train
    trainer.VALIDATION_N_PATIENTS = a.n_windows

    if a.device is not None:
        device = torch.device(a.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    master_seed = a.seed if a.seed is not None else ckpt.get('master_seed')
    if master_seed is None:
        master_seed = config.MASTER_SEED
    if config.DETERMINISTIC:
        train.setup_determinism(master_seed)

    sd = ckpt['model_state_dict']
    ema = None if a.no_ema else ckpt.get('model_ema_state_dict')
    model = T1DMAI().to(device)
    model.load_state_dict({k: ema.get(k, v) for k, v in sd.items()} if ema else sd, strict=True)
    weighting = KendallGalWeighting().to(device)
    weighting.load_state_dict(ckpt['weighting_state_dict'])

    tc = ckpt.get('training_config') or {}
    if a.backup is None:
        dataset = T1DMDataset(
            master_seed=master_seed + train.VAL_SEED_OFFSET,
            total_steps=a.n_windows,
            batch_size=1,
            normalization_stats=stats,
            patient_uniform_sample_prob=tc.get('patient_uniform_sample_prob',
                                               config.PATIENT_UNIFORM_SAMPLE_PROB),
            simulator_warmup_hours=tc.get('simulator_warmup_hours',
                                          config.SIMULATOR_WARMUP_HOURS),
            cache_path=a.cache_path,
            cache_partition='val',
            blind=blind,
        )
    else:
        from t1dmdroid_converter import STEPS_PER_DAY, TEST_DAYS
        test_days = TEST_DAYS if a.test_days is None else a.test_days
        dataset = BackupWindows(backup_stretches(a.backup, test_days), a.n_windows, stats,
                                master_seed, blind)
        steps = sum(len(s['bg_observed']) for s in dataset.stretches)
        print(f'{a.backup}: {test_days} test days, {len(dataset.stretches)} stretches, '
              f'{24 * steps / STEPS_PER_DAY:.1f} h, {a.n_windows} windows',
              file=sys.stderr, flush=True)

    metrics = trainer._run_validation(
        model, dataset, stats, device, weighting,
        bg_hypo_threshold=a.bg_hypo_threshold, bg_hyper_threshold=a.bg_hyper_threshold)
    # The ratio compares against the simulator training stream; a phone record is another domain.
    if a.backup is None and ckpt.get('loss_ema') is not None:
        metrics['train_loss_ema'] = ckpt['loss_ema']
        metrics['overfit_ratio'] = train._overfit_ratio(metrics['val_loss_total'], ckpt['loss_ema'])
    print(trainer._render_validation_table(int(ckpt.get('step') or 0), metrics))


if __name__ == '__main__':
    main()
