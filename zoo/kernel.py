"""Kernel and state-space forecasters: RFF epsilon-SVR, sparse GP, local-linear-trend Kalman.

SVR and GP read the ridge ARX features and forecast the delta from the last reading per zone step.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from finetune_data import HORIZON_MINUTES, HORIZON_STEPS
from zoo.classical import Forecaster, Ridge, _Fn, _t, ridge_features
from zoo.windows import ZONE, Windows

STEP_30 = HORIZON_STEPS[HORIZON_MINUTES.index(30)]

SVR_WINDOWS = 200_000
SVR_FEATURES = 1024
SVR_STEPS = 2000
SVR_BATCH = 2048
SVR_LR = 1e-3
SVR_EPSILON = 0.05
SVR_L2 = 1e-4
SVR_RIDGE = 1.0

GP_WINDOWS = 200_000
GP_INDUCING = 1000
GP_KMEANS_POOL = 20_000
GP_KMEANS_ITERS = 5
GP_HYPER_WINDOWS = 3000
GP_LENGTH_SCALES = (0.25, 0.5, 1.0, 2.0)
GP_LINEAR_WEIGHTS = (0.0, 0.1, 1.0)
GP_SIGNAL_GRID = np.logspace(-3, 1, 17)
GP_NOISE_GRID = np.logspace(-4, 0.5, 19)
GP_JITTER = 1e-6
GP_CHUNK = 10_000

KALMAN_CTX = 36
KALMAN_WINDOWS = 50_000
KALMAN_Q_LEVEL = np.logspace(-5, 2, 15)
KALMAN_Q_TREND = np.logspace(-5, 2, 15)
KALMAN_DAMPING = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)
KALMAN_P0 = 1e4


def _subsample(w: Windows, n: int, seed: int) -> Windows:
    if len(w) <= n:
        return w
    return w.take(np.sort(np.random.default_rng(seed).choice(len(w), n, replace=False)))


def _eval(fn, w: Windows, **buffers: np.ndarray) -> np.ndarray:
    """Run a graph function in float64 on a ``Windows`` batch; the same code ``graph()`` wraps."""
    m = SimpleNamespace(**{k: _t(v) for k, v in buffers.items()})
    with torch.no_grad():
        return fn(m, _t(w.bg), _t(w.insulin), _t(w.carb)).numpy()


def _sqdist_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return torch.clamp(d, min=0.0)


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _sqdist_t(_t(a), _t(b)).numpy()


def _median_distance(x: np.ndarray, seed: int) -> float:
    s = x[np.random.default_rng(seed).choice(len(x), min(len(x), 2000), replace=False)]
    d = _sqdist(s, s)
    return float(np.sqrt(np.median(d[np.triu_indices(len(s), 1)])))


class _ArxDelta(Forecaster):
    """Standardized ARX features in, per-step standardized delta from the last reading out."""

    def __init__(self) -> None:
        self.arx = Ridge(exogenous=True)
        self.ctx = self.arx.ctx
        self.mu = self.sd = self.ymu = self.ysd = np.empty(0)

    def _x(self, m, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        return (ridge_features(bg, ins, carb, self.arx.lags, True) - m.mu) / m.sd

    def _prepare(self, w: Windows) -> tuple[np.ndarray, np.ndarray]:
        X = self.arx.features(w)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        y = w.target.astype(np.float64) - w.bg[:, -1:]
        self.ymu, self.ysd = np.nanmean(y, axis=0), np.nanstd(y, axis=0) + 1e-6
        return (X - self.mu) / self.sd, (y - self.ymu) / self.ysd

    def _delta(self, m, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _forward(self, m, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        return bg[:, -1:] + m.ymu + self._delta(m, self._x(m, bg, ins, carb)) * m.ysd

    def _graph_state(self) -> dict[str, np.ndarray]:
        return self.state()

    def predict(self, w: Windows) -> np.ndarray:
        return _eval(self._forward, w, **self._graph_state())

    def graph(self) -> nn.Module:
        return _Fn(self._forward, **self._graph_state())


def _rff(x: torch.Tensor, omega: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return (2.0 / omega.shape[1]) ** 0.5 * torch.cos(x @ omega + phase)


class SVR(_ArxDelta):
    """Epsilon-insensitive linear SVR on [1, x, RFF(x)], one output per zone step.

    RBF random Fourier features with a median-distance length scale; ridge warm start, then Adam.
    """
    name = 'svr'

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self.omega = self.phase = self.coef = np.empty(0)

    def _z(self, x: np.ndarray) -> np.ndarray:
        rff = _rff(_t(x), _t(self.omega), _t(self.phase)).numpy()
        return np.concatenate([np.ones((len(x), 1)), x, rff], axis=1)

    def _delta(self, m, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[1]
        return m.coef[0] + x @ m.coef[1:1 + d] + _rff(x, m.omega, m.phase) @ m.coef[1 + d:]

    def fit(self, w: Windows) -> None:
        w = _subsample(w, SVR_WINDOWS, self.seed)
        x, y = self._prepare(w)
        rng = np.random.default_rng(self.seed)
        ell = _median_distance(x, self.seed)
        self.omega = rng.normal(0.0, 1.0 / ell, (x.shape[1], SVR_FEATURES))
        self.phase = rng.uniform(0.0, 2 * np.pi, SVR_FEATURES)
        p = 1 + x.shape[1] + SVR_FEATURES
        full = np.isfinite(y).all(axis=1)
        G, b = np.zeros((p, p)), np.zeros((p, ZONE))
        for i in range(0, len(x), GP_CHUNK):
            sl = slice(i, i + GP_CHUNK)
            z = self._z(x[sl][full[sl]])
            G += z.T @ z
            b += z.T @ y[sl][full[sl]]
        reg = SVR_RIDGE * np.eye(p)
        reg[0, 0] = 0.0
        W = np.linalg.solve(G + reg, b)
        m, v = np.zeros_like(W), np.zeros_like(W)
        ok = np.isfinite(y)
        y0 = np.nan_to_num(y)
        for t in range(1, SVR_STEPS + 1):
            idx = rng.integers(0, len(x), SVR_BATCH)
            z = self._z(x[idx])
            r = z @ W - y0[idx]
            g = np.sign(r) * ((np.abs(r) > SVR_EPSILON) & ok[idx])
            grad = z.T @ g / SVR_BATCH + SVR_L2 * W
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            W -= SVR_LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        self.coef = W

    def state(self) -> dict[str, np.ndarray]:
        return {'mu': self.mu, 'sd': self.sd, 'ymu': self.ymu, 'ysd': self.ysd,
                'omega': self.omega, 'phase': self.phase, 'coef': self.coef}

    def load(self, state: dict[str, np.ndarray]) -> None:
        for k, v in state.items():
            setattr(self, k, v)


def _kmeans_pp(x: np.ndarray, m: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    c = [x[rng.integers(len(x))]]
    d = ((x - c[0]) ** 2).sum(1)
    for _ in range(1, m):
        c.append(x[rng.choice(len(x), p=d / d.sum())])
        d = np.minimum(d, ((x - c[-1]) ** 2).sum(1))
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(_sqdist(x, c), axis=1)
        cnt = np.bincount(lab, minlength=m)
        s = np.zeros_like(c)
        np.add.at(s, lab, x)
        c[cnt > 0] = s[cnt > 0] / cnt[cnt > 0, None]
    return c


def _gp_kernel(a: torch.Tensor, b: torch.Tensor, ell: torch.Tensor, lin: torch.Tensor) -> torch.Tensor:
    return torch.exp(-_sqdist_t(a, b) / (2 * ell * ell)) + lin * (a @ b.T) / a.shape[1]


class GP(_ArxDelta):
    """Subset-of-regressors GP, RBF + linear kernel, inducing points by k-means++.

    Kernel shape (length scale, linear weight) shared; signal and noise variance per zone step,
    all by exact marginal likelihood on a subset via one eigendecomposition per shape.
    """
    name = 'gp'

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self.inducing = self.alpha = self.ainv = self.noise = np.empty(0)
        self.ell = self.lin = 0.0

    def _k0(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _gp_kernel(_t(a), _t(b), _t(self.ell), _t(self.lin)).numpy()

    def _delta(self, m, x: torch.Tensor) -> torch.Tensor:
        return _gp_kernel(x, m.inducing, m.ell, m.lin) @ m.alpha

    def _hyper(self, x: np.ndarray, y: np.ndarray) -> None:
        med = _median_distance(x, self.seed)
        best = (-np.inf, 0.0, 0.0, None)
        for ls in GP_LENGTH_SCALES:
            for lin in GP_LINEAR_WEIGHTS:
                self.ell, self.lin = ls * med, lin
                lam, U = np.linalg.eigh(self._k0(x, x))
                lam = np.maximum(lam, 0.0)
                p2 = (U.T @ y) ** 2
                var = GP_SIGNAL_GRID[:, None, None] * lam + GP_NOISE_GRID[None, :, None]
                ll = -0.5 * (np.einsum('abn,nk->abk', 1.0 / var, p2)
                             + np.log(var).sum(-1)[..., None])
                flat = ll.reshape(-1, ZONE)
                arg = flat.argmax(axis=0)
                total = flat.max(axis=0).sum()
                if total > best[0]:
                    ia, ib = np.unravel_index(arg, ll.shape[:2])
                    best = (total, ls * med, lin, (GP_SIGNAL_GRID[ia], GP_NOISE_GRID[ib]))
        _, self.ell, self.lin, (sig, noise) = best
        self.ratio = noise / sig
        self.noise = noise

    def fit(self, w: Windows) -> None:
        w = _subsample(w, GP_WINDOWS, self.seed)
        w = w.take(np.flatnonzero(np.isfinite(w.target).all(axis=1)))
        x, y = self._prepare(w)
        rng = np.random.default_rng(self.seed)
        h = rng.choice(len(x), min(len(x), GP_HYPER_WINDOWS), replace=False)
        self._hyper(x[h], y[h])
        pool = x[rng.choice(len(x), min(len(x), GP_KMEANS_POOL), replace=False)]
        self.inducing = _kmeans_pp(pool, min(GP_INDUCING, len(pool)), GP_KMEANS_ITERS, rng)
        m = len(self.inducing)
        C, b = np.zeros((m, m)), np.zeros((m, ZONE))
        for i in range(0, len(x), GP_CHUNK):
            k = self._k0(x[i:i + GP_CHUNK], self.inducing)
            C += k.T @ k
            b += k.T @ y[i:i + GP_CHUNK]
        kmm = self._k0(self.inducing, self.inducing)
        jit = GP_JITTER * np.trace(C) / m * np.eye(m)
        self.ainv = np.stack([np.linalg.inv(C + r * kmm + jit) for r in self.ratio])
        self.alpha = np.einsum('kij,jk->ik', self.ainv, b)

    def _graph_state(self) -> dict[str, np.ndarray]:
        return {'mu': self.mu, 'sd': self.sd, 'ymu': self.ymu, 'ysd': self.ysd,
                'inducing': self.inducing, 'alpha': self.alpha,
                'ell': np.array(self.ell), 'lin': np.array(self.lin)}

    def predict_std(self, w: Windows) -> np.ndarray:
        """(N, ZONE) mg/dL predictive standard deviation, observation noise included."""
        x = _eval(self._x, w, mu=self.mu, sd=self.sd)
        k = self._k0(x, self.inducing)
        q = np.einsum('ni,kij,nj->nk', k, self.ainv, k)
        return np.sqrt(self.noise * (1.0 + q)) * self.ysd

    def state(self) -> dict[str, np.ndarray]:
        return {**self._graph_state(), 'ainv': self.ainv.astype(np.float32), 'noise': self.noise}

    def load(self, state: dict[str, np.ndarray]) -> None:
        for k, v in state.items():
            setattr(self, k, v.astype(np.float64))
        self.ell, self.lin = float(self.ell), float(self.lin)


class Kalman(Forecaster):
    """Damped local-linear-trend Kalman filter over the context, trend propagated over the zone.

    Measurement variance fixed at 1; Q/R ratios and damping by grid search on 30-min RMSE.
    Gains are window-independent, so the forecast is a (ctx, ZONE) map: the filter on the identity.
    """
    name = 'kalman'
    ctx = KALMAN_CTX
    reads_doses = False

    def __init__(self) -> None:
        self.q_level, self.q_trend, self.damping = 1.0, 0.01, 1.0

    @staticmethod
    def _gains(q_level: float, q_trend: float, phi: float, n: int) -> np.ndarray:
        F = np.array([[1.0, phi], [0.0, phi]])
        Q = np.diag([q_level, q_trend])
        P = np.diag([KALMAN_P0, KALMAN_P0])
        K = np.empty((n, 2))
        for t in range(n):
            if t:
                P = F @ P @ F.T + Q
            K[t] = P[:, 0] / (P[0, 0] + 1.0)
            P = P - np.outer(K[t], P[0])
        return K

    @staticmethod
    def _filter(bg: np.ndarray, K: np.ndarray, phi: float) -> tuple[np.ndarray, np.ndarray]:
        level = bg[:, 0].astype(np.float64)
        trend = np.zeros_like(level)
        for t in range(1, bg.shape[1]):
            lp, tp = level + phi * trend, phi * trend
            e = bg[:, t] - lp
            level, trend = lp + K[t, 0] * e, tp + K[t, 1] * e
        return level, trend

    @staticmethod
    def _reach(phi: float) -> np.ndarray:
        return np.cumsum(phi ** np.arange(1, ZONE + 1))

    def _map(self) -> np.ndarray:
        """(ctx, ZONE): forecast = bg @ map."""
        K = self._gains(self.q_level, self.q_trend, self.damping, self.ctx)
        level, trend = self._filter(np.eye(self.ctx), K, self.damping)
        return level[:, None] + trend[:, None] * self._reach(self.damping)[None, :]

    @staticmethod
    def _forward(m, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        return bg @ m.map

    def fit(self, w: Windows) -> None:
        w = _subsample(w, KALMAN_WINDOWS, 0)
        y = w.target[:, STEP_30].astype(np.float64)
        ok = np.isfinite(y)
        bg, y = w.bg[ok], y[ok]
        best = np.inf
        for phi in KALMAN_DAMPING:
            r = self._reach(phi)[STEP_30]
            for ql in KALMAN_Q_LEVEL:
                for qt in KALMAN_Q_TREND:
                    level, trend = self._filter(bg, self._gains(ql, qt, phi, bg.shape[1]), phi)
                    err = np.mean((level + r * trend - y) ** 2)
                    if err < best:
                        best, self.q_level, self.q_trend, self.damping = err, ql, qt, phi

    def predict(self, w: Windows) -> np.ndarray:
        return _eval(self._forward, w, map=self._map())

    def graph(self) -> nn.Module:
        return _Fn(self._forward, map=self._map())

    def state(self) -> dict[str, np.ndarray]:
        return {'q_level': np.array(self.q_level), 'q_trend': np.array(self.q_trend),
                'damping': np.array(self.damping)}

    def load(self, state: dict[str, np.ndarray]) -> None:
        self.q_level, self.q_trend = float(state['q_level']), float(state['q_trend'])
        self.damping = float(state['damping'])


MODELS = {
    'svr': SVR,
    'gp': GP,
    'kalman': Kalman,
}
