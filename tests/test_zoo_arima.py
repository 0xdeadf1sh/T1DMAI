import numpy as np
import pytest

from zoo.windows import ZONE, Windows

pytest.importorskip('statsmodels')


def _ar_windows(n: int, ctx: int, phi: float, seed: int) -> Windows:
    rng = np.random.default_rng(seed)
    L = ctx + ZONE
    d = np.zeros((n, L + 50))
    e = rng.normal(0, 2, d.shape)
    for t in range(1, d.shape[1]):
        d[:, t] = phi * d[:, t - 1] + e[:, t]
    y = 140 + np.cumsum(d[:, 50:], axis=1)
    doses = rng.exponential(0.3, (n, L)).astype(np.float32) * (rng.random((n, L)) < 0.05)
    return Windows(y[:, :ctx].astype(np.float32), doses, doses * 8, y[:, ctx:].astype(np.float32),
                   np.full(n, 'x', object), np.full(n, -1))


@pytest.mark.parametrize('exogenous', [False, True])
def test_linear_map_is_the_statsmodels_forecast_and_exports(exogenous):
    import torch

    from zoo.arima import StateSpace
    ctx = 24
    w = _ar_windows(120, ctx, 0.6, 0)
    m = StateSpace((1, 1, 0), exogenous=exogenous, ctx=ctx, fit_windows=60)
    m.fit(w)
    p = m.predict(w)
    direct = np.stack([m.forecast_one(w.bg[i], w.insulin[i], w.carb[i]) for i in range(8)])
    err = np.abs(p[:8] - direct).max()
    print(f'DUMP exog={exogenous} phi={m.params[-2]:.3f} linear-vs-statsmodels max {err:.2e}')
    assert err < 1e-6
    assert abs(m.params[-2] - 0.6) < 0.1
    g = m.graph().eval()
    args = tuple(torch.from_numpy(x) for x in (w.bg, w.insulin, w.carb))
    with torch.no_grad():
        out = g(*args).numpy()
    assert np.abs(out - p).max() < 1e-2
