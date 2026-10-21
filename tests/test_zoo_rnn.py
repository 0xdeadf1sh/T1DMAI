import numpy as np
import pytest
import torch

from zoo.rnn import GluNet, LSTM
from zoo.windows import ZONE, Windows


def _windows(bg: np.ndarray, target: np.ndarray, ins=None, carb=None) -> Windows:
    n, ctx = bg.shape
    z = np.zeros((n, ctx + ZONE), np.float32)
    return Windows(bg.astype(np.float32), z if ins is None else ins, z if carb is None else carb,
                   target.astype(np.float32), np.full(n, 'x', object), np.full(n, -1))


def _trend_and_carbs(n: int, ctx: int, seed: int = 0) -> Windows:
    rng = np.random.default_rng(seed)
    slope = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
    t = np.arange(-ctx + 1, ZONE + 1, dtype=np.float32)[None, :]
    path = 140 + slope * t
    carb = np.zeros((n, ctx + ZONE), np.float32)
    carb[:, ctx] = rng.uniform(0, 60, n) * (rng.random(n) < 0.5)
    path[:, ctx:] += carb[:, ctx:ctx + 1] * np.linspace(0, 1.5, ZONE)[None, :]
    return _windows(path[:, :ctx], path[:, ctx:], carb=carb)


MODELS = [(LSTM, 24), (GluNet, 48)]


@pytest.mark.parametrize('cls,ctx', MODELS)
def test_default_ctx_shape_and_size(cls, ctx):
    m = cls()
    assert m.ctx == ctx
    w = _trend_and_carbs(7, ctx)
    out = m.predict(w)
    n_par = sum(p.numel() for p in m.net.parameters())
    print(f'DUMP {cls.name} ctx {m.ctx} params {n_par} out {out.shape}')
    assert out.shape == (7, ZONE) and np.isfinite(out).all()
    assert n_par < 500_000


@pytest.mark.parametrize('cls,ctx', MODELS)
def test_beats_persistence_on_trend_and_carbs(cls, ctx):
    w = _trend_and_carbs(4000, ctx)
    m = cls(epochs=15, batch=128, lr=3e-3)
    m.fit(w)
    test = _trend_and_carbs(500, ctx, seed=1)
    err = np.sqrt(np.mean((m.predict(test) - test.target) ** 2))
    err_p = np.sqrt(np.mean((test.bg[:, -1:] - test.target) ** 2))
    print(f'DUMP {cls.name} rmse {err:.2f}  persistence {err_p:.2f}')
    assert err < 0.3 * err_p


@pytest.mark.parametrize('cls,ctx', MODELS)
def test_torch_export_traces(cls, ctx):
    m = cls()
    net = m.net.eval()
    bg, ins, carb = m.inputs(_trend_and_carbs(4, ctx))
    ep = torch.export.export(net, (bg, ins, carb))
    with torch.no_grad():
        ref = net(bg, ins, carb)
        got = ep.module()(bg, ins, carb)
    print(f'DUMP {cls.name} export max abs diff {float((ref - got).abs().max()):.2e}')
    assert torch.allclose(ref, got, atol=1e-5)
