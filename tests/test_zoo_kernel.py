import numpy as np

from zoo.classical import Persistence
from zoo.kernel import GP, SVR, Kalman
from zoo.windows import ZONE, Windows


def _windows(bg: np.ndarray, target: np.ndarray, ins=None, carb=None) -> Windows:
    n, ctx = bg.shape
    z = np.zeros((n, ctx + ZONE), np.float32)
    return Windows(bg.astype(np.float32), z if ins is None else ins, z if carb is None else carb,
                   target.astype(np.float32), np.full(n, 'x', object), np.full(n, -1))


def _sines(n: int, ctx: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(ctx + ZONE)[None, :]
    omega = rng.uniform(0.02, 0.08, (n, 1))
    s = 140 + rng.uniform(20, 60, (n, 1)) * np.sin(omega * t + rng.uniform(0, 2 * np.pi, (n, 1)))
    s = s + rng.normal(0, 2, s.shape)
    return s[:, :ctx], s[:, ctx:]


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _beats_persistence(model, n: int = 3000, ratio: float = 0.5) -> Windows:
    bg, target = _sines(n, model.ctx, 0)
    model.fit(_windows(bg, target))
    bg, target = _sines(1000, model.ctx, 1)
    w = _windows(bg, target)
    err, base = _rmse(model.predict(w), target), _rmse(Persistence().predict(w), target)
    print(f'DUMP {model.name} rmse {err:.2f}  persistence {base:.2f}')
    assert err < ratio * base
    return w


def test_svr_beats_persistence_on_sines():
    _beats_persistence(SVR())


def test_gp_beats_persistence_on_sines_and_std_is_sane():
    gp = GP()
    w = _beats_persistence(gp)
    sd = gp.predict_std(w)
    print(f'DUMP gp std step1 {sd[:, 0].mean():.2f}  step24 {sd[:, -1].mean():.2f}')
    assert sd.shape == (len(w), ZONE) and np.isfinite(sd).all() and (sd > 0).all()
    assert sd[:, -1].mean() > sd[:, 0].mean()


def test_gp_state_roundtrip():
    gp = GP()
    bg, target = _sines(800, gp.ctx, 2)
    w = _windows(bg, target)
    gp.fit(w)
    other = GP()
    other.load(gp.state())
    assert np.allclose(other.predict(w), gp.predict(w), atol=1e-3)


def test_kalman_recovers_linear_ramp():
    rng = np.random.default_rng(0)
    n, ctx = 2000, Kalman.ctx
    slope = rng.uniform(-3, 3, (n, 1))
    t = np.arange(ctx + ZONE)[None, :]
    s = 120 + slope * t
    k = Kalman()
    k.fit(_windows(s[:, :ctx], s[:, ctx:]))
    err = np.abs(k.predict(_windows(s[:, :ctx], s[:, ctx:])) - s[:, ctx:]).max()
    print(f'DUMP kalman ramp max err {err:.4f}  phi {k.damping} q {k.q_level} {k.q_trend}')
    assert k.damping == 1.0 and err < 0.5


def test_kalman_beats_persistence_on_sines():
    _beats_persistence(Kalman(), ratio=0.8)
