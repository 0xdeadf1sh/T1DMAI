import numpy as np

from zoo import paper_metrics as pm


def test_gmse_penalty_targets_hypo_overestimate_and_hyper_underestimate():
    t = np.array([50.0, 50.0, 120.0, 300.0, 300.0])
    p = np.array([90.0, 40.0, 150.0, 250.0, 350.0])
    pen = pm.gmse_penalty(t, p)
    assert np.isclose(pen[0], 2.5) and np.isclose(pen[1], 1.0) and np.isclose(pen[2], 1.0)
    assert np.isclose(pen[3], 2.0) and np.isclose(pen[4], 1.0)


def test_point_metrics_perfect_forecast():
    rng = np.random.default_rng(0)
    t = rng.uniform(50, 300, 2000)
    o = t + rng.normal(0, 5, 2000)
    r = pm.point_metrics(t, t.copy(), o, 30)
    assert r['rmse'] == 0 and r['me'] == 0 and r['dts_a'] == 100 and r['clarke_a'] == 100
    assert r['hypo_sens'] == 1 and r['hypo_spec'] == 1 and np.isclose(r['hypo_mcc'], 1)


def test_time_gain_of_persistence_is_zero_and_of_an_oracle_is_the_horizon():
    ts = np.arange(3000) * 300
    g = 120 + 50 * np.sin(np.arange(3000) / 20.0)
    h = 30
    assert pm.time_gain(ts[:-6], g[:-6], ts, g, h) == 0
    assert pm.time_gain(ts[:-6], g[6:], ts, g, h) == h


def test_wilcoxon_detects_a_strictly_better_model():
    rng = np.random.default_rng(1)
    t = rng.uniform(70, 250, 500)
    good, bad = t + rng.normal(0, 5, 500), t + rng.normal(0, 30, 500)
    d, p = pm.wilcoxon_abs_error(t, bad, good)
    assert d > 0 and p < 1e-6


def _clustered_case():
    """10 days of 288 rows: |a| - |b| is +1 on 6 days and -0.9 on 4, with small row noise."""
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(10), 288)
    shift = np.where(np.isin(day, [0, 2, 3, 5, 7, 8]), 1.0, -0.9)
    t = np.full(len(day), 150.0)
    b = t + 10.0
    a = t + 10.0 + shift + rng.normal(0, 0.1, len(day))
    return t, a, b, day


def test_clustered_wilcoxon_counts_days_not_rows():
    t, a, b, day = _clustered_case()
    _, p_rows = pm.wilcoxon_abs_error(t, a, b)
    d, p_days = pm.wilcoxon_abs_error(t, a, b, day)
    print(f'DUMP row-level p {p_rows:.2e}, per-day p {p_days:.3f}, per-day mean {d:+.3f}')
    assert p_rows < 1e-6 and p_days > 0.05
    assert np.isclose(d, (6 * 1.0 - 4 * 0.9) / 10, atol=0.01)


def test_leaderboard_verdict_uses_the_clustered_test():
    import leaderboard as lb
    t, a, b, day = _clustered_case()
    h = len(lb.HORIZONS)
    d = {'truth': np.repeat(t[:, None], h, 1), 'pred': np.stack([np.repeat(x[:, None], h, 1)
                                                                  for x in (b, a)]),
         'origin': t.copy(), 'keys': np.full(len(t), 'p1'), 'day': day,
         'ts': day * 86400 + np.arange(len(t)) % 288 * 300, 'cgm': {}}
    spec = {'entries': [{'key': k, 'label': k, 'family': 'Baseline'} for k in ('b', 'a')]}
    rows = lb.score_track(spec, d)
    assert rows[0]['h'][60]['vs_best'] == 'best'
    assert rows[1]['h'][60]['vs_best'] == 'tie' and rows[1]['test_unit'] == 'day'
    assert 'per day' in lb.tables('t', spec, rows)
