import numpy as np
import pytest
import torch

from zoo.patch import MODELS
from zoo.windows import ZONE, Windows

CTX = 72


def _windows(bg: np.ndarray, target: np.ndarray, ins=None, carb=None) -> Windows:
    n, ctx = bg.shape
    z = np.zeros((n, ctx + ZONE), np.float32)
    return Windows(bg.astype(np.float32), z if ins is None else ins, z if carb is None else carb,
                   target.astype(np.float32), np.full(n, 'x', object), np.full(n, -1))


def _synthetic(n: int, seed: int) -> Windows:
    """Ramps of random slope plus an announced meal whose rise scales with its grams."""
    rng = np.random.default_rng(seed)
    t = np.arange(CTX + ZONE, dtype=np.float32)
    slope = rng.uniform(-1.5, 1.5, (n, 1)).astype(np.float32)
    full = 120 + slope * (t - CTX + 1) + rng.normal(0, 1, (n, CTX + ZONE)).astype(np.float32)
    carb = np.zeros((n, CTX + ZONE), np.float32)
    carb[:, CTX] = rng.uniform(0, 60, n) * (rng.random(n) < 0.5)
    full[:, CTX:] += carb[:, CTX:CTX + 1] * np.linspace(0, 1.5, ZONE, dtype=np.float32)[None, :]
    return _windows(full[:, :CTX], full[:, CTX:], carb=carb)


@pytest.mark.parametrize('name', sorted(MODELS))
def test_shape_and_size(name):
    m = MODELS[name]()
    w = _synthetic(5, 0)
    p = m.predict(w)
    n_params = sum(x.numel() for x in m.net.parameters())
    print(f'DUMP {name} params {n_params}')
    assert m.ctx == CTX and p.shape == (5, ZONE) and np.isfinite(p).all()
    assert n_params < 1_000_000


@pytest.mark.parametrize('name', sorted(MODELS))
def test_beats_persistence(name):
    torch.set_num_threads(4)
    m = MODELS[name](epochs=6, batch=256, lr=3e-3)
    m.fit(_synthetic(6000, 1))
    w = _synthetic(1000, 2)
    err = np.sqrt(np.mean((m.predict(w) - w.target) ** 2))
    err_p = np.sqrt(np.mean((w.bg[:, -1:] - w.target) ** 2))
    print(f'DUMP {name} rmse {err:.2f}  persistence {err_p:.2f}')
    assert err < 0.3 * err_p


@pytest.mark.parametrize('name', sorted(MODELS))
def test_export_traces(name):
    m = MODELS[name]()
    m.net.eval()
    args = tuple(t[:2] for t in m.inputs(_synthetic(2, 3)))
    ep = torch.export.export(m.net, args)
    with torch.no_grad():
        out = ep.module()(*args)
        ref = m.net(*args)
    print(f'DUMP {name} export max abs diff {float((out - ref).abs().max()):.2e}')
    assert torch.allclose(out, ref, atol=1e-5)
