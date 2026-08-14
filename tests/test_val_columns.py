"""Every column the validation header declares is written by the validation run.

§3.6 is a three-way drift between the header (``train._val_log_columns``), the
row writer (``train._csv_row``) and the checkpoint's ``val_record``.
``tests/test_training.py`` already pins the first two to each other by length.
This file pins the surface underneath all three: the ``val_metrics`` dict that
feeds them.

A column no code path ever sets writes an empty cell on every row of every run,
which is indistinguishable from a metric that was measured and came out empty —
and it stays that way silently, because nothing raises and the table still
renders.  Forty-seven columns lived in exactly that state.  The assertion here is
therefore about KEY PRESENCE, not about a value: a bin that is genuinely empty
sets its key to None and that is a measurement of an empty bin, while a key that
is absent is a metric nobody computed.

One validation is run for real — model, dataset, three forwards, both protocols —
because a static check over the column list is what let the gap open in the first
place.

Two rules have no subject in a run that size and are driven from inputs built
here instead: the conformal probe returns nothing under 50 val windows, and the
hypo alarm raises no detection over 12 patients.  An assertion whose subject the
run never produces passes on absence, which is the state the columns above were
already in.
"""
import math
import re

import numpy as np
import pytest
import torch

from config import (
    NOCTURNAL_START_HOUR, PATCH_SIZE, PREDICTION_PATCHES, QUANTILE_LEVELS,
)
from data import T1DMDataset
from model import T1DMAI
from normalization import load_normalization_stats
from risk_loss import KendallGalWeighting

import train

# Small enough to run in a test, large enough that every fixed-protocol bin is
# populated: the forecast protocol lays one patch at each d per window and the
# infill protocol scores every window, so both fill at any N.
N_VAL = 12

# The three keys the caller writes onto the record after _run_validation returns
# (train() computes them from the training-side loss history, not from the
# validation pass).
CALLER_WRITTEN = {'step', 'train_loss_ema', 'overfit_ratio'}

# ``_render_validation_table``'s ``cov_sharp_row`` renders the pair as
# ``<cov>% @ w <width> mg/dL`` and a missing width as ``@ w —``.  Only a width
# VALUE separates the two, so the prefix is not what a pairing check matches on.
_COV_VALUE = re.compile(r'\d+\.\d{2}%')
_COV_WIDTH = re.compile(r'@ w \d+(?:\.\d+)? mg/dL')
_COV_LABELS = ('coverage90', 'inner50_cov', 'cov90 marginal',
               'joint90 whole path', 'infill cov90')

# A conformal probe's output at the scale a real one returns: the correction
# widens the band and lifts coverage toward 90%.  Nothing here is measured — it
# exists to put the two ``conf cov90@peak`` rows on the page, which a run of
# N_VAL windows never does.
SYNTHETIC_CONFORMAL = {
    'conf_cov90_raw': 0.7213, 'conf_width_raw': 44.2,
    'conf_cov90_cal': 0.8967, 'conf_width_cal': 61.5,
    'conf_hypo_esc_raw': 0.191, 'conf_hypo_esc_cal': 0.104,
    'conf_n': 37.0,
}


@pytest.fixture(scope='module')
def val_metrics(monkeypatch_module):
    monkeypatch_module.setattr(train, 'VALIDATION_N_PATIENTS', N_VAL)
    device = torch.device('cpu')
    stats = load_normalization_stats()
    model = T1DMAI().to(device)
    weighting = KendallGalWeighting().to(device)
    kw = dict(master_seed=20_000_017, total_steps=N_VAL, batch_size=1,
              normalization_stats=stats, patient_uniform_sample_prob=0.0)
    metrics = train._run_validation(
        model, T1DMDataset(**kw), stats, device, weighting)
    metrics.update(train._run_night_onset_validation(
        model,
        T1DMDataset(force_pred_start_hour=NOCTURNAL_START_HOUR, **kw),
        stats, device))
    return metrics


@pytest.fixture(scope='module')
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _coverage_rows(table: str) -> "list[str]":
    """The table lines that render a coverage figure — the ones a width rides on.

    Keyed on the rendered value, not on the label alone: a bin with no coverage
    renders ``—`` and has no width to carry.
    """
    return [line for line in table.splitlines()
            if any(k in line for k in _COV_LABELS) and _COV_VALUE.search(line)]


def _expected_coverage_rows() -> int:
    """How many coverage rows the table declares, from the same lists it renders from."""
    return (2 * len(train.COVERAGE_HORIZONS_MIN)                     # coverage90, inner50_cov
            + 2 * len(train._excursion_bucket_horizons(PREDICTION_PATCHES))  # cov90 marginal, joint90
            + len(train._infill_reachable_d()))                      # infill cov90


def _assert_alarm_points_are_operating_points(metrics: dict) -> int:
    """Detection rate, false alarms per day and MEDIAN LEAD travel together.

    A detection rate without its lead time is not an operating point: a rate
    bought at a two-minute lead is not a usable alarm and the rate alone cannot
    show it.  Where the alarm fired on no window the rate is 0 and the lead is
    absent — an alarm that never fired has no lead, and absent is not 0.

    Returns the number of τ that fired, so a caller can pin that the lead
    assertion was reached at all.
    """
    n_events = metrics['alarm_hypo_n_events']
    assert isinstance(n_events, float)
    fired = 0
    for tau in train._alarm_curve_taus():
        tag = train._tau_tag(tau)
        det = metrics[f'alarm_hypo_det@{tag}']
        fa = metrics[f'alarm_hypo_fa_day@{tag}']
        lead = metrics[f'alarm_hypo_lead_min@{tag}']
        if n_events == 0:
            continue
        assert det is not None and 0.0 <= det <= 1.0, f'det@{tag} = {det}'
        assert fa is not None and fa >= 0.0, f'fa_day@{tag} = {fa}'
        if det > 0.0:
            assert lead is not None and lead > 0.0, (
                f'alarm at tau={tau} detected {det:.2%} of events with no lead '
                f'time: a detection rate is not an operating point without it')
            fired += 1
    return fired


def _firing_alarm_columns() -> dict:
    """``train._forecast_fan_columns`` over a fan whose hypo alarm fires at every τ.

    Half the groups descend to 45 mg/dL, so every lower band edge dips under the
    70 mg/dL threshold ahead of the truth and each swept τ carries a detection
    with a lead.  The live fixture raises no detection at any τ, so it reaches
    the lead-time rule on nothing.
    """
    offs = np.linspace(-30.0, 30.0, len(QUANTILE_LEVELS))
    rng = np.random.default_rng(3)
    q, true, d, group = [], [], [], []
    for g in range(40):
        end = 45.0 if g % 2 == 0 else 150.0
        for dd in range(1, PREDICTION_PATCHES + 1):
            centre = np.linspace(120.0, end, PATCH_SIZE)
            q.append(centre[:, None] + offs[None, :])
            true.append(centre + rng.normal(0.0, 2.0, PATCH_SIZE))
            d.append(dd)
            group.append(g)
    return train._forecast_fan_columns(
        np.stack(q), np.stack(true), np.array(d), np.array(group),
        observed_days=5.0)


def test_no_declared_val_column_is_unwritten(val_metrics):
    """No header column is missing from the record the writers read.

    The conformal probe is the one family allowed to be absent: it fits on
    excursion windows only and a validation set that carries none has nothing to
    fit, which is a property of the sample rather than of the wiring.
    """
    declared = [c for c, _ in train._val_log_columns()]
    assert declared, 'the validation header declares no columns at all'
    optional = {c for c in declared if c.startswith('conf_')}
    missing = [c for c in declared
               if c not in val_metrics and c not in CALLER_WRITTEN | optional]
    assert not missing, (
        f"{len(missing)} declared column(s) no validation code path sets, so "
        f"they write empty on every row of every run: {missing}"
    )
    print(f"[DUMP] val columns | {len(declared)} declared, "
          f"{len(declared) - len(optional) - len(CALLER_WRITTEN)} written ✓")


def test_the_five_scoring_rules_carry_values(val_metrics):
    """CRPS, Winkler, sharpness, joint coverage and the alarm curve are measured.

    These are the families ``metrics.scoring`` owns.  Every one of them is
    populated by construction under the two fixed protocols — the forecast
    protocol puts one patch in each ``d`` bin per window — so an empty value here
    is a wiring failure, not an empty bin.
    """
    eh = train._excursion_bucket_horizons(PREDICTION_PATCHES)
    inf_d = train._infill_reachable_d()
    assert eh and train.FAN_SCORE_FAMILIES, 'no forecast-protocol bin to score'
    assert inf_d and train.INFILL_FAMILIES, 'no infill-protocol bin to score'

    checked = 0
    for fam in train.FAN_SCORE_FAMILIES:
        for h in eh:
            v = val_metrics[f'{fam}@{h}']
            assert isinstance(v, float) and math.isfinite(v), f'{fam}@{h} = {v}'
            checked += 1
    for fam in train.INFILL_FAMILIES:
        for d in inf_d:
            col = train._infill_column(fam, d)
            v = val_metrics[col]
            assert isinstance(v, float) and math.isfinite(v), f'{col} = {v}'
            checked += 1
    print(f"[DUMP] scoring rules | {checked} per-d columns carry finite values ✓")


def test_alarm_curve_reports_lead_time_wherever_it_fired(val_metrics):
    """The live run's alarm points carry det and fa/day wherever there are events."""
    n_events = val_metrics['alarm_hypo_n_events']
    fired = _assert_alarm_points_are_operating_points(val_metrics)
    print(f"[DUMP] alarm curve | {int(n_events)} events, "
          f"{len(train._alarm_curve_taus())} operating points, {fired} fired ✓")


def test_a_detection_rate_is_never_reported_without_its_lead_time():
    """The same rule, on a fan built so that every operating point fires.

    N_VAL patients raise no detection, so the test above reaches the lead-time
    assertion on no τ.  This drives the column builder directly, where a lead
    dropped beside a live detection rate fails.
    """
    taus = train._alarm_curve_taus()
    fired = _assert_alarm_points_are_operating_points(_firing_alarm_columns())
    assert fired == len(taus), (
        f'{fired}/{len(taus)} operating points fired on a fan built to fire at '
        f'every τ: the lead-time assertion is reached on nothing')
    print(f"[DUMP] alarm lead | {fired} fired τ, each with a lead time ✓")


def test_the_rendered_table_shows_the_new_rows(val_metrics):
    """The table renders every scoring rule, per d, with sharpness beside coverage."""
    table = train._strip_ansi(train._render_validation_table(1, val_metrics, None))
    for needle in ('Proper Scoring', 'Hypo Alarm Operating Curve',
                   'Infill Protocol', 'crps @30m', 'winkler90 @120m',
                   'joint90 whole path', 'fa/day', 'lead '):
        assert needle in table, f"validation table is missing {needle!r}"
    # Sharpness never travels apart from coverage: a rendered coverage figure
    # carries a width VALUE on the same line.  The row count is pinned too — a
    # coverage the table stops rendering leaves nothing for the loop to check.
    rows = _coverage_rows(table)
    expected = _expected_coverage_rows()
    assert len(rows) == expected, (
        f'{len(rows)} coverage rows rendered, {expected} declared: a coverage '
        f'figure the table drops is checked by nothing')
    for line in rows:
        assert _COV_WIDTH.search(line), (
            f"coverage row without the width that bought it: {line.strip()}")
    print(f"[DUMP] val table | scoring sections rendered, {len(rows)} coverage "
          f"rows each paired with a width ✓")


def test_conformal_coverage_rows_carry_the_width_that_bought_them():
    """``conf cov90@peak`` RAW and CAL each render their own coverage and width.

    The correction moves coverage and width together, so a drop read without the
    width reads a narrowing as a loss of calibration.  Each row is matched
    against the width it was given, which a swapped or dropped width fails.
    """
    table = train._strip_ansi(
        train._render_validation_table(1, dict(SYNTHETIC_CONFORMAL), None))
    for label, cov_key, w_key in (
            ('conf cov90@peak RAW', 'conf_cov90_raw', 'conf_width_raw'),
            ('conf cov90@peak CAL', 'conf_cov90_cal', 'conf_width_cal')):
        line = next((l for l in table.splitlines() if label in l), None)
        assert line is not None, f'{label} is not rendered at all'
        assert f"{SYNTHETIC_CONFORMAL[cov_key] * 100.0:.2f}%" in line, (
            f'{label} does not render its own coverage: {line.strip()}')
        assert f"@ w {SYNTHETIC_CONFORMAL[w_key]:.1f} mg/dL" in line, (
            f'{label} renders a coverage without the width that bought it: '
            f'{line.strip()}')
    print("[DUMP] conformal rows | RAW and CAL each paired with their own width ✓")
