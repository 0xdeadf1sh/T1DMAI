"""validate.py's backup windows: test days only, never across a gap, local hour per sample."""

import gzip
import json
import math

import numpy as np
import torch

from config import MIN_CONTEXT_PATCHES, NIGHT_LONG_HORIZON_PATCHES, PATCH_SIZE, PREDICTION_PATCHES
from normalization import load_normalization_stats
from t1dmdroid_converter import STEP_MS, STEPS_PER_DAY
from validate import BackupWindows, backup_stretches

T0_MS = 1_750_032_000_000  # a UTC midnight
NEED = (MIN_CONTEXT_PATCHES + max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES)) * PATCH_SIZE
LONG = NEED + 4 * PATCH_SIZE
SHORT = PATCH_SIZE
PRE = STEPS_PER_DAY
TEST_DAYS = math.ceil((LONG + SHORT + 2) / STEPS_PER_DAY)
N = PRE + TEST_DAYS * STEPS_PER_DAY
GAPS = {PRE + LONG, PRE + LONG + 1 + SHORT}
TZ_SWITCH = PRE + LONG // 2


def _tz(i: int) -> int:
    return 60 if i < TZ_SWITCH else 120


def _backup(tmp_path) -> str:
    path = str(tmp_path / 'synthetic.t1dmbak')
    with gzip.open(path, 'wt') as f:
        f.write(json.dumps({'format': 't1dm.archive'}) + '\n')
        for i in range(N):
            s = {'t': 'sample', 'ts': T0_MS + i * STEP_MS, 'tz': _tz(i)}
            if i not in GAPS:
                s.update(bg=120.0 + 40.0 * math.sin(i / 25.0), pv='MEASURED', fl='NORMAL')
            f.write(json.dumps(s) + '\n')
        f.write(json.dumps({'t': 'end'}) + '\n')
    return path


def _hour(i: int) -> float:
    return ((i * STEP_MS // 1000 + _tz(i) * 60) % 86400) / 3600.0


def test_stretches_are_the_test_days_gap_free_runs(tmp_path):
    stretches = backup_stretches(_backup(tmp_path), TEST_DAYS)
    tail = N - (PRE + LONG + SHORT + 2)
    want = [LONG, SHORT] + ([tail] if tail else [])
    got = [len(s['bg_observed']) for s in stretches]
    print(f'[DUMP] need={NEED} runs want={want} got={got}')
    assert got == want
    assert all(np.isfinite(s['bg_observed']).all() for s in stretches)
    first = stretches[0]['hour_of_day']
    assert first[0] == _hour(PRE)
    assert first[TZ_SWITCH - PRE] == _hour(TZ_SWITCH)
    assert first[TZ_SWITCH - PRE] - first[TZ_SWITCH - PRE - 1] != STEP_MS / 3_600_000


def test_windows_drop_short_stretches_and_repeat_per_index(tmp_path):
    stretches = backup_stretches(_backup(tmp_path), TEST_DAYS)
    ds = BackupWindows(stretches, 6, load_normalization_stats(), seed=3, blind=False)
    assert all(len(s['bg_observed']) >= NEED for s in ds.stretches)
    assert len(ds.stretches) == sum(len(s['bg_observed']) >= NEED for s in stretches)
    assert len(ds) == 6
    for i in range(len(ds)):
        a, b = ds[i], ds[i]
        assert torch.equal(a['patches'], b['patches'])
        assert torch.isfinite(a['targets']).all()
        assert np.isfinite(a['bg_formula_data']['extended_true_bg_trajectory']).all()
