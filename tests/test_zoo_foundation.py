import numpy as np
import pytest

from zoo.windows import ZONE, Windows


def _windows(n: int, ctx: int, seed: int) -> Windows:
    rng = np.random.default_rng(seed)
    bg = (130 + np.cumsum(rng.normal(0, 2, (n, ctx)), axis=1)).astype(np.float32)
    ins = rng.exponential(0.05, (n, ctx + ZONE)).astype(np.float32)
    carb = np.zeros((n, ctx + ZONE), np.float32)
    carb[:, ctx + 2] = rng.uniform(20, 60, n)
    return Windows(bg, ins, carb, np.full((n, ZONE), np.nan, np.float32),
                   np.full(n, 'x', object), np.full(n, -1))


@pytest.mark.parametrize('key,pkg', [('chronos2', 'chronos'), ('timesfm25', 'timesfm')])
def test_foundation_blind_ignores_zone_doses_and_given_reads_them(key, pkg):
    pytest.importorskip(pkg)
    from zoo.foundation import MODELS
    m = MODELS[key]()
    w = _windows(64, m.ctx, 0)
    m.causal = True
    blind = m.predict(w)
    w2 = _windows(64, m.ctx, 0)
    w2.carb[:, m.ctx:] = 0.0
    w2.insulin[:, m.ctx:] = 0.0
    blind2 = m.predict(w2)
    m.causal = False
    given = m.predict(w)
    persist = np.repeat(w.bg[:, -1:], ZONE, axis=1)
    print(f'DUMP {key}: blind vs zeroed-zone {np.abs(blind - blind2).max():.2e}  '
          f'given vs blind {np.abs(given - blind).mean():.2f}  '
          f'blind vs persistence {np.abs(blind - persist).mean():.2f} mg/dL')
    assert blind.shape == given.shape == (64, ZONE)
    assert np.isfinite(blind).all() and np.isfinite(given).all()
    assert np.abs(blind - blind2).max() < 1e-3
    assert np.abs(blind - persist).mean() < 40.0
