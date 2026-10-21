import numpy as np

from finetune_data import HORIZON_MINUTES
from zoo.classical import LinearExtrapolation, Persistence, Ridge
from zoo.windows import ZONE, Windows, _cut, _fill_context, horizon_columns


def _windows(bg: np.ndarray, target: np.ndarray, ins=None, carb=None) -> Windows:
    n, ctx = bg.shape
    z = np.zeros((n, ctx + ZONE), np.float32)
    return Windows(bg.astype(np.float32), z if ins is None else ins, z if carb is None else carb,
                   target.astype(np.float32), np.full(n, 'x', object), np.full(n, -1))


def test_cut_zone_opens_one_step_after_the_reading():
    bg = np.arange(100, dtype=np.float32)
    ins = np.zeros(100, np.float32)
    b, _, _, t = _cut(bg, ins, ins.copy(), np.array([40]), 12, True)
    assert b[0, -1] == 39.0
    assert t[0, 0] == 40.0
    pred_30 = horizon_columns(t)[0, 0]
    print(f'DUMP reading 39, pred_30 truth {pred_30}')
    assert pred_30 == 39.0 + HORIZON_MINUTES[0] // 5


def test_cut_past_the_end_is_nan_and_dose_free():
    bg = np.arange(50, dtype=np.float32)
    ins = np.ones(50, np.float32)
    _, i, _, t = _cut(bg, ins, ins.copy(), np.array([45]), 6, True)
    assert np.isnan(t[0, 5:]).all() and np.isfinite(t[0, :5]).all()
    assert (i[0, 6 + 5:] == 0).all() and (i[0, :6 + 5] == 1).all()


def test_fill_context_carries_forward_then_back():
    bg = np.array([[np.nan, 5.0, np.nan, 7.0]])
    assert _fill_context(bg).tolist() == [[5.0, 5.0, 5.0, 7.0]]


def test_persistence_and_linear_trend():
    ramp = np.tile(np.arange(24, dtype=np.float32) * 2 + 100, (3, 1))
    truth = ramp[:, -1:] + 2 * np.arange(1, ZONE + 1)
    w = _windows(ramp, truth)
    assert np.allclose(Persistence().predict(w), ramp[:, -1:])
    assert np.allclose(LinearExtrapolation().predict(w), truth)


def test_graph_matches_predict_and_exports():
    import torch

    from zoo.registry import models
    rng = np.random.default_rng(1)
    reg = models()
    for name in ('persistence', 'linear_extrapolation', 'ridge_ar', 'ridge_arx', 'glunet', 'nhits'):
        m = reg[name]()
        n = 256
        bg = (130 + np.cumsum(rng.normal(0, 3, (n, m.ctx)), axis=1)).astype(np.float32)
        doses = rng.exponential(0.5, (n, m.ctx + ZONE)).astype(np.float32)
        target = bg[:, -1:] + rng.normal(0, 10, (n, ZONE))
        w = _windows(bg, target, ins=doses, carb=doses * 5)
        if hasattr(m, 'epochs'):
            m.epochs = 1
        m.fit(w)
        g = m.graph().eval()
        args = tuple(torch.from_numpy(x) for x in (w.bg, w.insulin, w.carb))
        with torch.no_grad():
            out = g(*args).numpy()
            ep = torch.export.export(g, tuple(a[:1] for a in args), strict=False)
            exp = ep.module()(*(a[:1] for a in args)).numpy()
        err = np.abs(out - m.predict(w)).max()
        print(f'DUMP {name} graph-vs-predict max {err:.2e} mg/dL')
        assert err < 1e-2
        assert np.abs(exp - out[:1]).max() < 1e-3


def test_ridge_arx_learns_announced_carbs():
    rng = np.random.default_rng(0)
    n, ctx = 4000, 72
    bg = 120 + rng.normal(0, 5, (n, ctx)).astype(np.float32)
    carb = np.zeros((n, ctx + ZONE), np.float32)
    carb[:, ctx] = rng.uniform(0, 60, n)
    target = bg[:, -1:] + 1.5 * carb[:, ctx:ctx + 1] * np.linspace(0, 1, ZONE)[None, :]
    w = _windows(bg, target, carb=carb)
    ar, arx = Ridge(exogenous=False), Ridge(exogenous=True)
    ar.fit(w)
    arx.fit(w)
    err_ar = np.sqrt(np.mean((ar.predict(w) - target) ** 2))
    err_arx = np.sqrt(np.mean((arx.predict(w) - target) ** 2))
    print(f'DUMP ridge AR rmse {err_ar:.2f}  ARX rmse {err_arx:.2f}')
    assert err_arx < 0.2 * err_ar
