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
