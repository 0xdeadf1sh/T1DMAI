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

from config import PATCH_SIZE, PREDICTION_PATCHES, QUANTILE_LEVELS
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
    'clarke_E @120m': 'evalfix_clarke_E@120',
    'dts_C @60m': 'dts_c@60',
    'tod acc ±1h': 'tod_acc_1h',
    'tod gross-error rate': 'tod_gross_rate',
    'tod jump (cross-window)': 'tod_xwin_jump_h',
    'night_hyper_recall': 'night_hyper_recall',
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
    return train._run_validation(
        model, T1DMDataset(**kw), stats, device, weighting)


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


def test_percent_metrics_are_not_scaled_twice_on_the_page(val_metrics):
    """A rate reaches the page once, at the scale its producer emits it.

    ``_run_validation`` stores ``tod_acc_*`` and ``tod_gross_rate`` as
    percentages — the CSV column, ``make_card.py`` and ``models/compare.py`` all
    read them that way — while every other rate key is a fraction the table
    scales itself. Getting that wrong is silent: the row still renders, still
    tiers, and a 16.3% clock printed 1630.00% sat ABOVE its 60% bar and coloured
    green. So the assertion is on the rendered cell, not on the dict.
    """
    percent_rows = {'tod acc ±1h': 'tod_acc_1h',
                    'tod acc ±2h': 'tod_acc_2h',
                    'tod acc (bin)': 'tod_acc_bin',
                    'tod gross-error rate': 'tod_gross_rate'}
    table = train._strip_ansi(train._render_validation_table(1, val_metrics, val_metrics))
    checked = 0
    for label, key in percent_rows.items():
        assert isinstance(val_metrics.get(key), float), f'{key} not measured'
        line = next((ln for ln in table.splitlines() if ln.lstrip('│ ').startswith(label)), None)
        assert line is not None, f'{label!r} is not on the validation table'
        cells = [c.strip() for c in line.strip('│').split('│')]
        for cell in (cells[1], cells[2]):          # Value, then Prev
            pct = re.search(r'(\d+\.\d+)%', cell)
            assert pct is not None, f'{label!r} renders no percentage: {cell!r}'
            assert float(pct.group(1)) == pytest.approx(val_metrics[key], abs=0.01), (
                f'{label!r} renders {pct.group(1)}% for a stored {val_metrics[key]} — '
                f'a percentage scaled a second time on the way to the page')
            assert 0.0 <= float(pct.group(1)) <= 100.0, (
                f'{label!r} renders {pct.group(1)}%, which is not a share of anything')
        checked += 1
    print(f"[DUMP] percent scale | {checked} tod rows render their stored percentage "
          f"once, in Value and Prev alike ✓")


def test_per_horizon_detection_bars_decline_with_horizon(val_metrics):
    """The per-bucket excursion rows are COLOURED against the declining schedule.

    Detection is not horizon-flat — the near call is largely fixed by insulin on
    board and the two-hour call is information-limited — so ``train.py`` keeps a
    per-horizon bar in ``EXCURSION_TARGET_*`` and ``_excursion_target``. Held at
    the 30-minute bar instead, every far bucket reads red on every run and the
    colour stops separating a good far horizon from a bad one.

    The bar itself is asserted in ``tests/test_training.py``; what is pinned here
    is that the TABLE cuts its tier against it. Those are separate failures: the
    schedule sat in the module, unit-tested and correct, while the page passed a
    flat literal — so a value the schedule calls good rendered red. Hence a value
    placed between the far bar and the near one, which the two answer opposite
    ways, and an assertion on the rendered colour rather than on the constants.
    """
    hs = train._excursion_bucket_horizons(PREDICTION_PATCHES)
    assert len(hs) >= 2, 'one bucket cannot show a decline'
    near, far = hs[0], hs[-1]
    cases = (('hypo_recall', train.EXCURSION_TARGET_HYPO_RECALL),
             ('hypo_precision', train.EXCURSION_TARGET_HYPO_PRECISION),
             ('hyper_recall', train.EXCURSION_TARGET_HYPER_RECALL),
             ('hyper_precision', train.EXCURSION_TARGET_HYPER_PRECISION))
    for name, spec in cases:
        bar_near = train._excursion_target(spec, near)
        bar_far = train._excursion_target(spec, far)
        assert bar_far < bar_near, (
            f'{name} does not decline from {near} to {far} min: '
            f'{bar_near} -> {bar_far}')
        # Green under the far bar, red under the near one: ``higher_row``'s
        # default warn gap is 10 points below the bar it is given.
        probe = 0.5 * (bar_far + min(bar_near - 10.0, 100.0))
        assert bar_far <= probe < bar_near - 10.0, (
            f'{name}: no value separates the two bars, so the render below '
            f'cannot tell which one coloured it')
        table = train._render_validation_table(
            1, {**val_metrics, f'{name}@{far}': probe / 100.0}, None)
        line = next(ln for ln in table.splitlines()
                    if f'{name} @{far}m' in train._strip_ansi(ln))
        assert train._ANSI_GREEN in line, (
            f'{name} @{far}m renders {probe:.1f}% uncoloured by its own '
            f'{bar_far:.0f}% bar — the row is tiered against the {near}-minute '
            f'bar instead: {train._strip_ansi(line).strip()}')
    print(f"[DUMP] excursion bars | 4 families decline {near}->{far} min and the "
          f"page tiers the far bucket against the far bar ✓")


def test_joint_coverage_is_trended_toward_its_bound_not_a_band_midpoint():
    """A rise in joint coverage renders as an improvement.

    ``joint90`` is the one coverage row whose colour band and whose improvement
    direction disagree. 70–92 is where a joint figure is acceptable, but its
    midpoint (81) is not a level the metric aims at: joint coverage is bounded
    above by the smallest marginal in scope and higher is better up to that
    bound. Trended as a band, a recovery from 82% to 88% moves AWAY from 81 and
    renders a red arrow while the value itself stays green — the arrow and the
    colour contradicting each other on the row that catches path-level failure.

    ``coverage90 @120m`` is the control: same movement, same builder, and it must
    keep reading green, so a fix that flipped every coverage row would fail here.
    """
    fan = {'_fan_joint_width@120': 60.0, 'sharp90@120': 60.0}

    def arrow(key, label, cur, prev):
        line = next(
            ln for ln in train._render_validation_table(
                1, {**fan, key: cur}, {**fan, key: prev}).splitlines()
            if label in train._strip_ansi(ln))
        sym = '↑' if '↑' in line else ('↓' if '↓' in line else '•')
        hue = ('green' if f'{train._ANSI_GREEN}{sym}' in line
               else 'red' if f'{train._ANSI_RED}{sym}' in line else 'none')
        return sym, hue

    for cur, prev, want in ((0.88, 0.82, 'green'), (0.82, 0.88, 'red')):
        sym, hue = arrow('joint_cov90@120', 'joint90 whole path', cur, prev)
        assert hue == want, (
            f'joint90 {prev:.0%} -> {cur:.0%} renders {sym} {hue}, want {want}: '
            f'the row is scored against a band midpoint (81%) that is not this '
            f"metric's target")
        ctrl_sym, ctrl_hue = arrow('coverage90@120', 'coverage90 @120m', cur, prev)
        assert ctrl_hue == want, (
            f'the marginal control moved too: coverage90 @120m renders '
            f'{ctrl_sym} {ctrl_hue}')
    print("[DUMP] joint90 trend | rise=green, fall=red, marginal control unchanged ✓")


def test_a_sample_with_no_slot_pair_cannot_dilute_the_jump_row():
    """``tod jump (within window)`` is a mean over rows that HAVE a pair.

    ``_slot_jump_hours`` divides by ``pair.sum().clamp(min=1.0)``, so a sample
    with one valid masked slot — no consecutive pair at all — comes back 0.0, the
    best attainable value on a lower-is-better row barred at ``<1.000 h``.
    Averaging those in measures how often the sampler drew one span of one patch,
    not whether the clock is stable. The subject is constructed here rather than
    drawn, because whether a 12-patient fixture happens to contain a no-pair
    sample is luck, and an assertion whose subject the run never produces passes
    on absence.
    """
    B, M, bins = 4, 3, train.TIME_PROBE_N_BINS
    logits = torch.zeros(B, M, bins)
    logits[:, :, 0] = 10.0                       # a pinned clock: deviation is the advance
    mask_idx = torch.arange(M).expand(B, M).contiguous()
    valid = torch.ones(B, M, dtype=torch.bool)
    valid[0, 1:] = False                         # row 0: one slot, therefore no pair
    hours, has_pair = train._slot_jump_hours(logits, mask_idx, valid, train._PATCH_HOURS)

    assert has_pair.tolist() == [False, True, True, True], (
        'the pair flag does not identify the single-slot row')
    assert float(hours[0]) == 0.0, (
        'the 0/0 guard no longer returns 0.0 — this test pins the dilution that '
        'value causes, so it needs the guard to still be there')
    assert float(hours[1:].min()) > 0.0, (
        'the paired rows report no deviation, so including row 0 would not '
        'change the mean and this test has no subject')

    unfiltered = float(hours.mean())
    filtered = float(hours[has_pair].mean())
    assert filtered > unfiltered, 'filtering did not remove the zero'
    assert unfiltered == pytest.approx(filtered * 3 / 4, rel=1e-6), (
        'the dilution is not exactly the no-pair share, so the arithmetic this '
        'test reasons about has changed')
    print(f"[DUMP] jump dilution | 1 of {B} rows has no pair; unfiltered "
          f"{unfiltered:.4f} h vs filtered {filtered:.4f} h ✓")


def test_the_jump_row_a_real_validation_reports_is_the_filtered_mean():
    """And the VALIDATION does the filtering, not just the helper.

    The test above pins ``_slot_jump_hours``' contract; a caller that takes the
    tuple and then averages the whole vector satisfies it completely, which is
    the shape the defect had. So this one runs the real ``_run_validation`` and
    recomputes both candidate means from the same weights and the same batches:
    ``tod_jump_h`` must be the filtered one.

    The subject is asserted before it is used — this seed's 12-sample validation
    contains exactly one window whose sampler drew a single one-patch span, and
    if that ever stops being true the two means coincide and the assertion below
    passes on nothing.
    """
    device = torch.device('cpu')
    stats = load_normalization_stats()
    torch.manual_seed(20_260_815)
    model = T1DMAI().to(device)
    weighting = KendallGalWeighting().to(device)
    kw = dict(master_seed=20_000_017, total_steps=N_VAL, batch_size=1,
              normalization_stats=stats, patient_uniform_sample_prob=0.0)

    ds = T1DMDataset(**kw)
    no_pair = sum(
        1 for i in range(N_VAL)
        if not bool((lambda v: v[1:] & v[:-1])(
            torch.as_tensor(ds[i]['bg_formula_data']['valid']).bool()).any()))
    assert no_pair > 0, (
        'no window in this fixture lacks a slot pair, so the filtered and '
        'unfiltered means are equal and this test has no subject')

    metrics = train._run_validation(model, T1DMDataset(**kw), stats, device, weighting)
    reported = metrics['tod_jump_h']

    # The same forward, batched the same way, so the only difference between the
    # two candidates is which rows they average.
    from data import collate_fn
    every, paired = [], []
    with torch.no_grad():
        for start in range(0, N_VAL, train.VAL_BATCH_SIZE):
            batch = collate_fn([ds[i] for i in range(
                start, min(start + train.VAL_BATCH_SIZE, N_VAL))])
            bf = batch['bg_formula_data']
            mask_idx = bf['mask_idx'].long()
            _, _, time_pred = model(batch['patches'], batch['attn_mask'],
                                    bf['anchor_bg'].float(), mask_idx, return_time=True)
            j, has = train._slot_jump_hours(
                time_pred, mask_idx, bf['valid'], train._PATCH_HOURS)
            every.append(j)
            paired.append(j[has])
    unfiltered = float(torch.cat(every).mean())
    filtered = float(torch.cat(paired).mean())

    assert filtered != pytest.approx(unfiltered, rel=1e-9), (
        'the two candidates coincide on this fixture, so the assertion below '
        'cannot tell them apart')
    assert reported == pytest.approx(filtered, rel=1e-5), (
        f'the validation reports {reported:.4f} h; the filtered mean is '
        f'{filtered:.4f} h and the diluted one {unfiltered:.4f} h — the caller '
        f'is averaging over its own 0/0 guard')
    print(f"[DUMP] jump row | {no_pair}/{N_VAL} windows have no pair; reported "
          f"{reported:.4f} h == filtered {filtered:.4f} h, not {unfiltered:.4f} h ✓")


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
