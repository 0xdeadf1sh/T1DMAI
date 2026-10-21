"""Classical forecasters: persistence, linear extrapolation, and direct multi-horizon ridge AR/ARX.

Every model maps a ``Windows`` batch to an (N, ZONE) mg/dL forecast and fits population-wide.
"""
from __future__ import annotations

import numpy as np

from zoo.windows import ZONE, Windows

# Step bins, newest first: [0, 30 min), [30, 60), [1, 2 h), [2, 4 h), [4, 6 h) before the origin.
PAST_BINS = ((0, 6), (6, 12), (12, 24), (24, 48), (48, 72))
ZONE_BINS = ((0, 6), (6, 12), (12, 18), (18, 24))


class Forecaster:
    name = ''
    ctx = 1

    def fit(self, w: Windows) -> None:
        pass

    def predict(self, w: Windows) -> np.ndarray:
        raise NotImplementedError

    def state(self) -> dict[str, np.ndarray]:
        return {}

    def load(self, state: dict[str, np.ndarray]) -> None:
        pass


class Persistence(Forecaster):
    name = 'persistence'

    def predict(self, w: Windows) -> np.ndarray:
        return np.repeat(w.bg[:, -1:], ZONE, axis=1)


class LinearExtrapolation(Forecaster):
    """Least-squares slope over the last ``lookback`` readings, carried forward."""
    name = 'linear_extrapolation'

    def __init__(self, lookback: int = 6) -> None:
        self.ctx = lookback

    def predict(self, w: Windows) -> np.ndarray:
        t = np.arange(self.ctx, dtype=np.float64) - (self.ctx - 1) / 2.0
        slope = (w.bg[:, -self.ctx:] @ t) / (t @ t)
        return w.bg[:, -1:] + slope[:, None] * np.arange(1, ZONE + 1)[None, :]


def _bin_sums(x: np.ndarray, bins, end: int, reverse: bool) -> np.ndarray:
    if reverse:
        return np.stack([x[:, end - b:end - a].sum(axis=1) for a, b in bins], axis=1)
    return np.stack([x[:, end + a:end + b].sum(axis=1) for a, b in bins], axis=1)


class Ridge(Forecaster):
    """Direct ridge per zone step on the delta from the last reading.

    AR reads ``lags`` readings; ARX adds binned past and announced insulin and carbohydrate.
    """

    def __init__(self, lags: int = 24, exogenous: bool = False, alpha: float = 1.0) -> None:
        self.lags = lags
        self.exogenous = exogenous
        self.alpha = alpha
        self.ctx = max(lags, PAST_BINS[-1][1]) if exogenous else lags
        self.name = 'ridge_arx' if exogenous else 'ridge_ar'
        self.mu = self.sd = self.coef = np.empty(0)

    def features(self, w: Windows) -> np.ndarray:
        last = w.bg[:, -1:].astype(np.float64)
        cols = [last, w.bg[:, -self.lags:-1] - last]
        if self.exogenous:
            c = w.bg.shape[1]
            for x in (w.insulin, w.carb):
                cols += [_bin_sums(x, PAST_BINS, c, True), _bin_sums(x, ZONE_BINS, c, False)]
        return np.concatenate(cols, axis=1)

    def fit(self, w: Windows) -> None:
        X = self.features(w)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        Z = np.concatenate([np.ones((len(X), 1)), (X - self.mu) / self.sd], axis=1)
        y = w.target.astype(np.float64) - w.bg[:, -1:]
        reg = self.alpha * np.eye(Z.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.empty((Z.shape[1], ZONE))
        for k in range(ZONE):
            ok = np.isfinite(y[:, k])
            Zk = Z[ok]
            self.coef[:, k] = np.linalg.solve(Zk.T @ Zk + reg, Zk.T @ y[ok, k])

    def predict(self, w: Windows) -> np.ndarray:
        X = self.features(w)
        Z = np.concatenate([np.ones((len(X), 1)), (X - self.mu) / self.sd], axis=1)
        return w.bg[:, -1:] + Z @ self.coef

    def state(self) -> dict[str, np.ndarray]:
        return {'mu': self.mu, 'sd': self.sd, 'coef': self.coef}

    def load(self, state: dict[str, np.ndarray]) -> None:
        self.mu, self.sd, self.coef = state['mu'], state['sd'], state['coef']


MODELS = {
    'persistence': Persistence,
    'linear_extrapolation': LinearExtrapolation,
    'ridge_ar': lambda: Ridge(exogenous=False),
    'ridge_arx': lambda: Ridge(exogenous=True),
}
