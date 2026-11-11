import numpy as np
import pytest
import torch

import zoo.boost
from zoo.boost import LightGBM, XGBoost, _lgb_table

BOOSTERS = pytest.mark.parametrize('cls', [XGBoost, LightGBM], ids=['xgboost', 'lightgbm'])
from zoo.classical import Persistence
from zoo.windows import ZONE, Windows


def _windows(bg: np.ndarray, target: np.ndarray, ins=None, carb=None) -> Windows:
    n, ctx = bg.shape
    z = np.zeros((n, ctx + ZONE), np.float32)
    return Windows(bg.astype(np.float32), z if ins is None else ins, z if carb is None else carb,
                   target.astype(np.float32), np.full(n, 'x', object), np.full(n, -1))


def _random(n: int, ctx: int, seed: int, pump: bool) -> Windows:
    rng = np.random.default_rng(seed)
    bg = (130 + np.cumsum(rng.normal(0, 3, (n, ctx)), axis=1)).astype(np.float32)
    carb = np.zeros((n, ctx + ZONE), np.float32)
    carb[:, ctx] = rng.uniform(0, 60, n) * (rng.random(n) < 0.5)
    if pump:
        ins = (rng.integers(0, 40, (n, ctx + ZONE)) * 0.0125).astype(np.float32)
        bg, carb = np.rint(bg), np.rint(carb)
    else:
        ins = rng.exponential(0.5, (n, ctx + ZONE)).astype(np.float32)
    slope = bg[:, -1:] - bg[:, -2:-1]
    target = (bg[:, -1:] + slope * np.arange(1, ZONE + 1)
              + 1.5 * carb[:, ctx:ctx + 1] * np.linspace(0, 1, ZONE) + rng.normal(0, 2, (n, ZONE)))
    target[rng.random(target.shape) < 0.05] = np.nan
    return _windows(bg, target, ins=ins, carb=carb)


def _graph_out(m: XGBoost, w: Windows) -> np.ndarray:
    with torch.no_grad():
        return m.graph()(*(torch.from_numpy(x) for x in (w.bg, w.insulin, w.carb))).numpy()


@BOOSTERS
def test_trees_float32_graph_survives_pump_increment_drift(monkeypatch, cls):
    monkeypatch.setattr(zoo.boost, 'XGB_ROUNDS', 60)
    m = cls()
    m.fit(_random(3000, m.ctx, 0, pump=True))
    w = _random(1000, m.ctx, 1, pump=True)
    pred = m.predict(w)
    drift = np.abs(_graph_out(m, w) - pred).max()
    off = (np.abs(pred - m.predict_native(w)).max(axis=1) > 1e-2).mean()
    print(f'DUMP {m.name} pump doses: float32 graph vs predict {drift:.2e}  native off {off:.1%}')
    assert drift < 1e-2


def test_lgb_table_keeps_less_equal_at_the_float32_threshold():
    t = 0.1
    t32 = np.float32(t)
    below = t32 if t32 <= t else np.nextafter(t32, np.float32(-1))
    root = {'split_feature': 0, 'threshold': t, 'decision_type': '<=', 'default_left': True,
            'missing_type': 'None', 'left_child': {'leaf_value': -1.0},
            'right_child': {'leaf_value': 1.0}}
    tab = _lgb_table(root)
    cut = tab['threshold'][0]
    assert below < cut and np.nextafter(below, np.float32(1)) >= cut
    assert tab['left'].tolist() == [1, -1, -1] and tab['value'].tolist() == [0.0, -1.0, 1.0]
    assert not tab['default_left'][0]


@BOOSTERS
def test_trees_torch_walk_matches_native_predict_and_export(monkeypatch, cls):
    monkeypatch.setattr(zoo.boost, 'XGB_ROUNDS', 60)
    m = cls()
    m.fit(_random(3000, m.ctx, 0, pump=False))
    w = _random(1000, m.ctx, 1, pump=False)
    truth = w.target
    pred = m.predict(w)
    gap = np.abs(pred - m.predict_native(w))
    native, off = gap.max(), (gap.max(axis=1) > 1e-2).mean()
    g = m.graph().eval()
    args = tuple(torch.from_numpy(x) for x in (w.bg, w.insulin, w.carb))
    with torch.no_grad():
        out = g(*args).numpy()
        ep = torch.export.export(g, tuple(a[:1] for a in args), strict=False)
        exp = ep.module()(*(a[:1] for a in args)).numpy()
        one = g(*(a[:1] for a in args)).numpy()
    ok = np.isfinite(truth)
    err = np.sqrt(np.mean((pred - truth)[ok] ** 2))
    base = np.sqrt(np.mean((Persistence().predict(w) - truth)[ok] ** 2))
    print(f'DUMP {m.name} vs native max {native:.2e} windows off {off:.1%}  graph vs predict {np.abs(out - pred).max():.2e}'
          f'  export {np.abs(exp - one).max():.1e}  depth {m.depth}  rmse {err:.2f}'
          f'  persistence {base:.2f}')
    assert off < 0.01 and (cls is not XGBoost or native < 1.0)
    assert np.abs(out - pred).max() < 1e-2
    assert np.array_equal(exp, one)
    assert err < 0.5 * base


def test_xgboost_split_is_strict_less_and_missing_follows_default(monkeypatch):
    m = XGBoost()
    m.feature, m.threshold = np.array([0, 0, 0]), np.array([1.0, 5.0, -3.0], np.float32)
    m.left, m.right = np.array([1, 1, 2]), np.array([2, 1, 2])
    m.value, m.default_left = np.array([0.0, 5.0, -3.0]), np.array([1, 0, 0])
    m.roots, m.step, m.base, m.depth = np.array([0]), np.ones((1, ZONE)), np.zeros(ZONE), 1
    x = torch.tensor([[0.5], [1.0], [float('nan')]], dtype=torch.float64)
    monkeypatch.setattr(zoo.boost, 'ridge_features', lambda *a: x)
    g = m.graph()
    out = g(torch.zeros((3, 1)), torch.zeros((3, 1)), torch.zeros((3, 1)))
    assert out[:, 0].tolist() == [5.0, -3.0, 5.0]
