import json
import os

import numpy as np
import pytest
import torch

import zoo.rocket as rk
import zoo.run as run
from config import QUANTILE_LEVELS
from zoo.registry import models
from zoo.windows import ZONE, Windows

GOLDEN = os.path.join(os.path.dirname(__file__), 'data', 'minirocket_golden.json')
MID = QUANTILE_LEVELS.index(0.5)


@pytest.fixture(scope='module')
def golden():
    """aeon 1.6.0 ``MiniRocket`` fitted on small integer series, with its transform."""
    with open(GOLDEN) as f:
        return json.load(f)


def _params(g: dict, n_channels: int) -> dict:
    if 'channel_indices' in g:
        channels = np.zeros((len(g['n_channels_per_combination']), n_channels), bool)
        at = 0
        for i, k in enumerate(g['n_channels_per_combination']):
            channels[i, g['channel_indices'][at:at + k]] = True
            at += k
    else:
        channels = np.ones((rk.N_KERNELS * len(g['dilations']), 1), bool)
    return {'dilations': np.array(g['dilations']), 'per_dilation': np.array(g['per_dilation']),
            'channels': channels, 'biases': np.array(g['biases'], np.float32)}


def test_dilations_and_quantiles_are_aeons(golden):
    dil, per = rk.fit_dilations(72, 10_000)
    assert [dil.tolist(), per.tolist()] == golden['dilations_72_10000']
    np.testing.assert_array_equal(rk.golden_quantiles(20), np.float32(golden['quantiles_20']))
    for g in (golden['multi'], golden['uni']):
        dil, per = rk.fit_dilations(72, 840)
        assert dil.tolist() == g['dilations'] and per.tolist() == g['per_dilation']


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('track', ['multi', 'uni'])
def test_transform_is_aeons(golden, track, dtype):
    X = np.array(golden['X'], np.float64)
    if track == 'uni':
        X = X[:, :1]
    got = rk.transform(torch.from_numpy(X).to(dtype), _params(golden[track], X.shape[1]))
    np.testing.assert_array_equal(got.numpy().astype(np.float32),
                                  np.float32(golden[track]['features']))


def _devices():
    return ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def test_grid_inputs_give_the_same_features_on_every_device_and_dtype():
    """Flat stretches give a response of exactly 0 against a bias of 0: float noise must not
    decide that comparison."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(64, 3, 72)) * np.array([1.0, 0.3, 0.2])[:, None]
    x[:, 1:, 10:50] = -0.137  # flat basal, no carbs
    x = rk.quantize(x)
    p = rk.fit_kernels(x, 840, np.random.default_rng(0))
    assert (p['biases'] == 0).any()
    ref = rk.transform(torch.from_numpy(x).double(), p).numpy()
    for dev in _devices():
        for dtype in (torch.float32, torch.float64):
            for chunk in (1, 7, 64):
                got = torch.cat([rk.transform(torch.from_numpy(x[i:i + chunk]).to(dev, dtype), p)
                                 for i in range(0, len(x), chunk)])
                np.testing.assert_array_equal(got.cpu().numpy(), ref)


def test_quantize_is_exact_grid():
    q = rk.quantize(np.array([0.1234567, -100.0, 70.0, 1e-5]))
    np.testing.assert_array_equal(q * rk.GRID_STEP, np.round(q * rk.GRID_STEP))
    assert q[1] == -rk.GRID_MAX and q[2] == rk.GRID_MAX and q[3] == 0.0


def _gram(F: np.ndarray, y: np.ndarray):
    Fa = torch.from_numpy(np.concatenate([np.ones((len(F), 1)), F], 1))
    return Fa.T @ Fa, Fa.T @ torch.from_numpy(y)


def _direct(F, y, n_rocket, alpha):
    """Explicit standardized ridge: intercept free, ``alpha`` on rocket, EXTRA_ALPHA on extras."""
    mu, sd = F.mean(0), F.std(0)
    Z = np.concatenate([np.ones((len(F), 1)), (F - mu) / sd], 1)
    pen = np.r_[0.0, np.full(n_rocket, alpha), np.full(F.shape[1] - n_rocket, rk.EXTRA_ALPHA)]
    return np.linalg.solve(Z.T @ Z + np.diag(pen), Z.T @ y)


def test_ridge_path_and_solve_step_are_the_direct_solve():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(200, 7)) * [1, 5, 0.1, 2, 3, 1, 4] + [0, 9, 1, -3, 0, 2, 7]
    y = F @ rng.normal(size=(7, 2)) + rng.normal(size=(200, 2))
    A, R = _gram(F, y)
    mu, inv = rk.standardizer(A)
    for alpha, coef in zip((0.5, 50.0), rk.ridge_path(A, R, mu, inv, 4, (0.5, 50.0))):
        want = _direct(F, y, 4, alpha)
        np.testing.assert_allclose(coef.numpy(), want, rtol=1e-8, atol=1e-10)
        for k in range(2):
            np.testing.assert_allclose(rk.solve_step(A, R[:, k], mu, inv, 4, alpha).numpy(),
                                       want[:, k], rtol=1e-8, atol=1e-10)


def test_standardizer_zeroes_constant_columns():
    F = np.stack([np.arange(10.0), np.full(10, 3.0)], 1)
    mu, inv = rk.standardizer(_gram(F, np.zeros((10, 1)))[0])
    assert inv[1] == 0 and inv[0] > 0 and mu[1] == pytest.approx(3.0)


def _windows(n: int, rng, ctx: int = 72) -> Windows:
    t = np.arange(ctx + ZONE)
    phase = rng.uniform(0, 2 * np.pi, n)[:, None]
    bg = (140 + 40 * np.sin(t / 15 + phase)).astype(np.float32)
    carb = np.where(rng.random((n, ctx + ZONE)) < 0.02, 30.0, 0.0).astype(np.float32)
    ins = np.where(rng.random((n, ctx + ZONE)) < 0.02, 3.0, 0.05).astype(np.float32)
    target = bg[:, ctx:] + rng.normal(0, 2, (n, ZONE)).astype(np.float32)
    target[rng.random((n, ZONE)) < 0.05] = np.nan
    return Windows(bg[:, :ctx], ins, carb, target, np.full(n, 's', object), np.full(n, -1))


def _model(**kw) -> rk.MiniRocket:
    m = rk.MiniRocket(n_features=168, chunk=37, **kw)
    m.device = 'cpu'
    return m


@pytest.fixture(scope='module')
def fitted():
    rng = np.random.default_rng(1)
    w, val, cal = _windows(300, rng), _windows(120, rng), _windows(150, rng)
    m = _model()
    m.fit(w, val)
    m.calibrate(cal)
    return m, w, val, cal


def test_fit_is_per_step_ridge_on_measured_rows(fitted):
    m, w, _, _ = fitted
    F = m._raw(w, np.arange(len(w)))[:, 1:].numpy()
    y = w.target.astype(np.float64) - w.bg[:, -1:]
    ok_all = np.isfinite(y).all(1)
    mu = F[ok_all].mean(0)
    np.testing.assert_allclose(m.mu.numpy(), mu, rtol=1e-9, atol=1e-12)
    keep = m.inv.numpy() > 0
    inv = m.inv.numpy()[keep]
    for k in (0, 7, ZONE - 1):
        ok = np.isfinite(y[:, k])
        Z = np.concatenate([np.ones((ok.sum(), 1)), (F[ok][:, keep] - mu[keep]) * inv], 1)
        pen = np.r_[0.0, np.where(np.arange(F.shape[1]) < m.n_rocket, m.alphas[k],
                                  rk.EXTRA_ALPHA)[keep]]
        want = np.linalg.solve(Z.T @ Z + np.diag(pen), Z.T @ y[ok, k])
        got = m.coef[:, k].numpy()[np.concatenate([[True], keep])]
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)
    assert np.isin(m.alphas / ok_all.sum(), rk.ALPHA_GRID).all()


def test_alpha_is_the_best_grid_point_on_validation(fitted):
    """The path is fit on windows measured at every step; alpha minimizes validation RMSE."""
    m, w, val, _ = fitted
    F = m._raw(w, np.arange(len(w)))[:, 1:].numpy()
    y = w.target.astype(np.float64) - w.bg[:, -1:]
    ok = np.isfinite(y).all(1)
    keep = m.inv.numpy() > 0
    Fk = F[ok][:, keep]
    n_rocket = int(keep[:m.n_rocket].sum())
    Fv = m._raw(val, np.arange(len(val)))[:, 1:].numpy()[:, keep]
    Zv = np.concatenate([np.ones((len(Fv), 1)), (Fv - Fk.mean(0)) / Fk.std(0)], 1)
    truth = val.target.astype(np.float64) - val.bg[:, -1:]
    err = np.array([np.sqrt(np.nanmean((Zv @ _direct(Fk, y[ok], n_rocket, ok.sum() * a)
                                        - truth) ** 2, 0)) for a in rk.ALPHA_GRID])
    np.testing.assert_array_equal(m.alphas, ok.sum() * rk.ALPHA_GRID[err.argmin(0)])


def test_band_is_median_plus_calibration_offsets(fitted):
    m, _, val, cal = fitted
    q = m.predict_quantiles(val)
    assert q.shape == (len(val), ZONE, len(QUANTILE_LEVELS))
    np.testing.assert_array_equal(q[..., MID], m.predict(val))
    assert (np.diff(q, axis=2) >= 0).all()
    res = cal.target - m.predict(cal)
    np.testing.assert_allclose(m.offsets[0], np.minimum(np.nanquantile(res, QUANTILE_LEVELS[0], 0), 0))
    np.testing.assert_allclose(m.offsets[-1], np.maximum(np.nanquantile(res, QUANTILE_LEVELS[-1], 0), 0))
    assert (m.offsets[MID] == 0).all()


def test_band_offsets_keep_a_zero_median_row_and_ascend():
    from zoo.export import band_offsets
    res = np.random.default_rng(4).normal(8, 3, (500, ZONE))
    off = band_offsets(res)
    assert (off[MID] == 0).all() and (np.diff(off, axis=0) >= 0).all()
    assert (off[:MID] == 0).all() and (off[MID + 1:] > 0).all()


def test_fails_closed():
    rng = np.random.default_rng(2)
    w = _windows(80, rng)
    with pytest.raises(ValueError):
        _model().fit(w, None)
    m = _model()
    m.fit(w, _windows(40, rng))
    with pytest.raises(ValueError):
        m.predict_quantiles(w)
    with pytest.raises(ValueError):
        m.calibrate(_windows(5, rng))


def test_state_round_trips(fitted):
    m, _, val, _ = fitted
    m2 = _model()
    m2.load({k: np.asarray(v) for k, v in m.state().items()})
    np.testing.assert_array_equal(m2.predict_quantiles(val), m.predict_quantiles(val))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_gpu_and_cpu_forecasts_agree(fitted):
    m, _, val, _ = fitted
    g = rk.MiniRocket(n_features=168, chunk=37)
    g.device = 'cuda'
    g.load({k: np.asarray(v) for k, v in m.state().items()})
    np.testing.assert_allclose(g.predict_quantiles(val), m.predict_quantiles(val), atol=1e-9)


def test_zone_doses_reach_only_the_zone_bins(fitted):
    m, _, val, _ = fitted
    a = m._raw(val, np.arange(5))
    v2 = val.take(np.arange(len(val)))
    v2.insulin[:, 72:] += 5.0
    v2.carb[:, 72:] += 20.0
    b = m._raw(v2, np.arange(5))
    past = 1 + m.n_rocket + 1 + 5  # through the insulin past bins
    changed = (a != b).any(0).numpy()
    assert not changed[:past].any() and changed[past:past + 4].all()
    assert not changed[past + 4:past + 9].any() and changed[past + 9:].all()


def test_fit_or_load_calibrates_on_the_cal_block(monkeypatch, tmp_path):
    rng = np.random.default_rng(3)
    blocks = []

    def fake_windows(cache, n, ctx, seed, block='fit'):
        blocks.append(block)
        return _windows({'fit': 200, 'val': 60, 'cal': 80}[block], rng)
    monkeypatch.setattr(run, 'train_windows', fake_windows)
    monkeypatch.setitem(run.MODELS, 'minirocket', lambda: _model())
    m = run.fit_or_load('minirocket', True, None, str(tmp_path), 200, 0)
    assert blocks == ['fit', 'val', 'cal'] and m.offsets.shape == (len(QUANTILE_LEVELS), ZONE)
    again = run.fit_or_load('minirocket', True, None, str(tmp_path), 200, 0)
    np.testing.assert_array_equal(again.offsets, m.offsets)


def test_registered():
    assert 'minirocket' in models()
