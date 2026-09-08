"""Every column the validation header declares must be written by validation.
Pins ``val_metrics`` under ``test_training.py``'s header/writer check: an absent
key leaves an empty cell, None means an empty bin. Conformal's subject is built
here since it needs 50+ windows to fire; validation runs for real, not a static
check."""
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

# glycemic-region bins need a window with TRUE BG<70: empty at 12, real at 48.
N_VAL = 48

# written onto the record by the caller, after _run_validation returns
CALLER_WRITTEN = {'step', 'train_loss_ema', 'overfit_ratio'}

# ``cov_sharp_row`` renders ``<cov>% @ w <width> mg/dL``; missing width is ``@ w —``.
_COV_VALUE = re.compile(r'\d+\.\d{2}%')
_COV_WIDTH = re.compile(r'@ w \d+(?:\.\d+)? mg/dL')
_COV_LABELS = ('coverage90', 'inner50_cov', 'joint90 whole path')

# families trimmed from the table, mapped to the column that must still carry them
CSV_ONLY = {
    'crps @': 'crps@30',
    'winkler90 @': 'winkler90@30',
    'Hypo Alarm': 'alarm_hypo_n_events',
    'fa/day': 'alarm_hypo_fa_day@q25',
    'Infill Protocol': 'infill_rmse@d1',
    'infill crps': 'infill_crps@d1',
    'pred_tir': 'pred_tir',
}

# families restored to the page, listed rather than deleted from CSV_ONLY
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

# the conformal probe needs 50+ windows, N_VAL=48 is short, so fake its keys here
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
    """Keyed on the rendered VALUE, not the label: a bin with no coverage renders ``—``
    and has no width to carry."""
    return [line for line in table.splitlines()
            if any(k in line for k in _COV_LABELS) and _COV_VALUE.search(line)]


def _expected_coverage_rows() -> int:
    """From the same lists the table renders from."""
    return (2 * len(train.COVERAGE_HORIZONS_MIN)                     # coverage90, inner50_cov
            + bool(train._excursion_bucket_horizons(PREDICTION_PATCHES)))  # joint90, far horizon


def _assert_alarm_points_are_operating_points(metrics: dict) -> int:
    """Returns the τ count that fired. A rate bought at a two-minute lead is not a
    usable alarm; an alarm that never fired has no lead, and absent is not 0."""
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
    """A fan whose hypo alarm fires at every τ: half the groups descend to 45 mg/dL,
    so every lower edge dips under 70 ahead of the truth."""
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
    """No header column is missing from the record the writers read. The conformal
    probe may be legitimately absent: it fits on excursion windows only."""
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
    """The forecast protocol puts one patch in each ``d`` bin per window, so every
    ``metrics.scoring`` family fills by construction: an empty value is a wiring
    failure, not an empty bin."""
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
    n_events = val_metrics['alarm_hypo_n_events']
    fired = _assert_alarm_points_are_operating_points(val_metrics)
    print(f"[DUMP] alarm curve | {int(n_events)} events, "
          f"{len(train._alarm_curve_taus())} operating points, {fired} fired ✓")


def test_a_detection_rate_is_never_reported_without_its_lead_time():
    """N_VAL patients raise no detection, so the test above reaches the lead-time
    assertion on no τ; this drives the column builder on a fan that fires at every τ."""
    taus = train._alarm_curve_taus()
    fired = _assert_alarm_points_are_operating_points(_firing_alarm_columns())
    assert fired == len(taus), (
        f'{fired}/{len(taus)} operating points fired on a fan built to fire at '
        f'every τ: the lead-time assertion is reached on nothing')
    print(f"[DUMP] alarm lead | {fired} fired τ, each with a lead time ✓")


def test_the_rendered_table_shows_the_calibration_rows(val_metrics):
    """``val_loss_total`` cannot stand in for calibration. Every coverage row stays,
    each with the width that bought it, and the one/two-sided pair at the same
    ``d`` beside them."""
    table = train._strip_ansi(train._render_validation_table(1, val_metrics, None))
    for needle in ('Quantile Calibration', 'coverage90 @30m', 'inner50_cov @30m',
                   'joint90 whole path', 'one-sided cov90 @d1',
                   'two-sided cov90 @d1'):
        assert needle in table, f"validation table is missing {needle!r}"
    # sharpness never travels apart from coverage; the row count is pinned too
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
    """The table is a reading surface; ``validation_log.csv`` is the record. Each
    dropped family is pinned to the column that still carries it, and to its
    absence from the page."""
    # a CSV_ONLY family needs a VALUE in the render input, else its page absence proves nothing
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
    """``tod_acc_*`` / ``tod_gross_rate`` are stored as PERCENTAGES, unlike every
    other rate key. Wrong, a 16.3% clock prints 1630.00% and colours green — the
    assertion is on the rendered cell, not the dict."""
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
    """``EXCURSION_TARGET_*`` declines with horizon; held at the 30-minute bar,
    every far bucket reads red. This pins that the TABLE tiers against the
    schedule, via a probe value the near and far bars answer opposite ways."""
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
        # ``higher_row``'s default warn gap is 10 points below the bar it is given
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
    """``joint90``'s colour band (70-92) disagrees with its direction: bounded
    above by the smallest marginal in scope, higher is better up to that bound, so
    a band midpoint trend renders 82%->88% red. ``coverage90 @120m`` is the
    control and must stay green."""
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
    """A sample with one valid slot and no consecutive pair returns 0.0 (the
    ``pair.sum().clamp(min=1.0)`` 0/0 guard), the best value on a row barred at
    ``<1.000 h``. Constructed, not drawn: a no-pair sample at any N is luck."""
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
    """The VALIDATION does the filtering, not just the helper: both candidate means
    are recomputed from the same weights and batches, and ``tod_jump_h`` must be
    the filtered one, not a caller averaging the whole vector."""
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

    # same forward, same batching: the candidates differ only in which rows they average
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
    """A restored family must render AND stay a declared column, or moving an entry
    out of ``CSV_ONLY`` satisfies every assertion whether the row exists or not."""
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
