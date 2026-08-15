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
_COV_LABELS = ('coverage90', 'inner50_cov', 'joint90 whole path')

# Families the console table no longer renders, each mapped to one column the
# record must still carry.  The table is a reading surface and the CSV is the
# record, so a trimmed row must still be MEASURED — dropping a row from the page
# and dropping the metric look identical from the page.
CSV_ONLY = {
    'crps @': 'crps@30',
    'winkler90 @': 'winkler90@30',
    'Hypo Alarm': 'alarm_hypo_n_events',
    'fa/day': 'alarm_hypo_fa_day@q25',
    'Infill Protocol': 'infill_rmse@d1',
    'infill crps': 'infill_crps@d1',
    'pred_tir': 'pred_tir',
    'tod acc': 'tod_acc_1h',
    'night_hyper_recall': 'night_hyper_recall',
    'night-onset': 'night_onset_hypo_recall',
}

# Families that WERE trimmed and have since been RESTORED to the page, each pinned
# to a label the table must now render.  Listed rather than simply deleted from
# CSV_ONLY above: where the trim boundary sits is a decision, and a family
# drifting back OFF the page unnoticed is the same defect as one drifting on.
# Every entry here must also still be a declared column, so a restored row cannot
# quietly stop being recorded either.
RESTORED_TO_THE_TABLE = {
    'conf cov90 raw': 'conf_cov90_raw',
    'conf hypo-escape raw': 'conf_hypo_esc_raw',
    'exc_undershoot_frac': 'exc_undershoot_frac',
    'trend_amp_ratio': 'trend_amp_ratio',
    'clarke_A @30m': 'evalfix_clarke_A@30',
    'clarke_C': 'clarke_C_pct',
    'cgega_BE @hypo': 'cgega_be_hypo',
    'median_roughness': 'median_roughness',
    'bg_mae  @30m': 'bg_mae_30',
    'hypo_recall @30m': 'hypo_recall@30',
    'hyper_precision @30m': 'hyper_precision@30',
}

# The conformal probe is the one CSV_ONLY family the fixture cannot produce:
# ``train._conformal_val_probe`` returns nothing under 50 validation windows and
# N_VAL is 12, so its keys are absent from ``val_metrics`` entirely. An absence
# check over a family the run never populates passes on nothing — exactly the
# state the module docstring says this file exists to prevent — so the render
# below is fed these values, at the scale a real probe returns, to give the
# assertion a subject.
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
            + bool(train._excursion_bucket_horizons(PREDICTION_PATCHES)))  # joint90, far horizon only


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


def test_the_rendered_table_shows_the_calibration_rows(val_metrics):
    """The table renders the calibration section, with sharpness beside coverage.

    Calibration is what the page is for: it is the one family the selection scalar
    cannot stand in for, since ``val_loss_total`` improved monotonically across a
    whole run while the deployed one-sided band decayed.  So the coverage rows
    stay, each with the width that bought it, and the one/two-sided pair at the
    same ``d`` stays beside them.
    """
    table = train._strip_ansi(train._render_validation_table(1, val_metrics, None))
    for needle in ('Quantile Calibration', 'coverage90 @30m', 'inner50_cov @30m',
                   'joint90 whole path', 'one-sided cov90 @d1',
                   'two-sided cov90 @d1'):
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
    print(f"[DUMP] val table | calibration section rendered, {len(rows)} coverage "
          f"rows each paired with a width ✓")


def test_rows_dropped_from_the_table_are_still_recorded(val_metrics):
    """Every family the console table stops rendering is still a declared column.

    The table is a reading surface at a 1000-step cadence; ``validation_log.csv``
    is the record.  From the page a trimmed row and a metric nobody computes look
    the same, so each dropped family is pinned here to the column that still
    carries it — and to its absence from the page, so a row quietly restored
    without its width or its ``n`` is caught as well.
    """
    # Every CSV_ONLY family must have a VALUE in the dict the table renders from,
    # or its absence from the page says nothing about the trim.
    metrics = {**val_metrics, **SYNTHETIC_CONFORMAL}
    table = train._strip_ansi(train._render_validation_table(1, metrics, None))
    declared = {c for c, _ in train._val_log_columns()}
    for label, column in CSV_ONLY.items():
        assert column in declared, (
            f"{label!r} was dropped from the table and {column!r} is not a "
            f"declared column either — the metric is gone, not moved")
        assert metrics.get(column) is not None, (
            f"{column!r} has no value in the render input, so {label!r} being "
            f"absent from the table is not evidence of anything")
        assert label not in table, (
            f"{label!r} is back on the validation table; if that is deliberate, "
            f"move it out of CSV_ONLY rather than leaving the two disagreeing")
    print(f"[DUMP] csv-only families | {len(CSV_ONLY)} trimmed families carry a "
          f"value, are still declared as columns, and are absent from the page ✓")


def test_families_restored_to_the_table_are_rendered_and_still_recorded(val_metrics):
    """The other half of the trim boundary: what was put BACK is on the page.

    ``CSV_ONLY`` above pins families to their absence. This pins the families that
    were moved the other way — they must render AND still be declared columns, so
    a restored row cannot quietly fall off the page again, and cannot stop being
    recorded either. Without it, moving an entry out of ``CSV_ONLY`` would be
    enough to satisfy every assertion in this file whether or not the row exists.
    """
    metrics = {**val_metrics, **SYNTHETIC_CONFORMAL}
    table = train._strip_ansi(train._render_validation_table(1, metrics, None))
    declared = {c for c, _ in train._val_log_columns()}
    for label, column in RESTORED_TO_THE_TABLE.items():
        assert column in declared, (
            f"{label!r} is rendered but {column!r} is not a declared column — the "
            f"page would be the only copy")
        assert metrics.get(column) is not None, (
            f"{column!r} has no value in the render input, so finding {label!r} "
            f"on the page would not show the metric is measured")
        assert label in table, (
            f"{label!r} is not on the validation table; if it was trimmed again, "
            f"move it back into CSV_ONLY rather than leaving the two disagreeing")
    overlap = set(CSV_ONLY) & set(RESTORED_TO_THE_TABLE)
    assert not overlap, f"a family is claimed both trimmed and restored: {overlap}"
    print(f"[DUMP] restored families | {len(RESTORED_TO_THE_TABLE)} render on the "
          f"page and are still declared columns ✓")
