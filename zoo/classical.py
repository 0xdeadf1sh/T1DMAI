"""Classical forecasters: persistence, linear extrapolation, and direct multi-horizon ridge AR/ARX.

Every model maps a ``Windows`` batch to an (N, ZONE) mg/dL forecast and fits population-wide.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from zoo.windows import ZONE, Windows

# Step bins, newest first: [0, 30 min), [30, 60), [1, 2 h), [2, 4 h), [4, 6 h) before the origin.
PAST_BINS = ((0, 6), (6, 12), (12, 24), (24, 48), (48, 72))
ZONE_BINS = ((0, 6), (6, 12), (12, 18), (18, 24))


class Forecaster:
    name = ''
    ctx = 1
    reads_doses = True  # False: the blind and given tracks are one forecast

    def fit(self, w: Windows) -> None:
        pass

    def predict(self, w: Windows) -> np.ndarray:
        raise NotImplementedError

    def state(self) -> dict[str, np.ndarray]:
        return {}

    def load(self, state: dict[str, np.ndarray]) -> None:
        pass

    def graph(self) -> nn.Module:
        """``(bg, insulin, carb)`` raw tensors -> ``(B, ZONE)`` mg/dL, export-traceable."""
        raise NotImplementedError(f'{self.name} has no exportable graph')


class _Fn(nn.Module):
    def __init__(self, fn, **buffers: np.ndarray) -> None:
        super().__init__()
        self.fn = fn
        for k, v in buffers.items():
            self.register_buffer(k, torch.as_tensor(np.asarray(v, np.float32)))

    def forward(self, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        return self.fn(self, bg, ins, carb)


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, np.float64))


class Persistence(Forecaster):
    name = 'persistence'
    reads_doses = False

    def predict(self, w: Windows) -> np.ndarray:
        return np.repeat(w.bg[:, -1:], ZONE, axis=1)

    def graph(self) -> nn.Module:
        return _Fn(lambda m, bg, ins, carb: bg[:, -1:].repeat(1, ZONE))


def _trend(bg: torch.Tensor, lookback: int) -> torch.Tensor:
    t = torch.arange(lookback, dtype=bg.dtype) - (lookback - 1) / 2.0
    slope = (bg[:, -lookback:] @ t) / (t @ t)
    return bg[:, -1:] + slope[:, None] * torch.arange(1, ZONE + 1, dtype=bg.dtype)[None, :]


class LinearExtrapolation(Forecaster):
    """Least-squares slope over the last ``lookback`` readings, carried forward."""
    name = 'linear_extrapolation'
    reads_doses = False

    def __init__(self, lookback: int = 6) -> None:
        self.ctx = lookback

    def predict(self, w: Windows) -> np.ndarray:
        return _trend(_t(w.bg), self.ctx).numpy()

    def graph(self) -> nn.Module:
        return _Fn(lambda m, bg, ins, carb: _trend(bg, self.ctx))


def _bin_sums(x: torch.Tensor, bins, end: int, reverse: bool) -> torch.Tensor:
    if reverse:
        return torch.stack([x[:, end - b:end - a].sum(dim=1) for a, b in bins], dim=1)
    return torch.stack([x[:, end + a:end + b].sum(dim=1) for a, b in bins], dim=1)


def ridge_features(bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor,
                   lags: int, exogenous: bool) -> torch.Tensor:
    last = bg[:, -1:]
    cols = [last, bg[:, -lags:-1] - last]
    if exogenous:
        c = bg.shape[1]
        for x in (ins, carb):
            cols += [_bin_sums(x, PAST_BINS, c, True), _bin_sums(x, ZONE_BINS, c, False)]
    return torch.cat(cols, dim=1)


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
        self.reads_doses = exogenous
        self.mu = self.sd = self.coef = np.empty(0)

    def features(self, w: Windows) -> np.ndarray:
        return ridge_features(_t(w.bg), _t(w.insulin), _t(w.carb), self.lags, self.exogenous).numpy()

    def graph(self) -> nn.Module:
        def fn(m, bg, ins, carb):
            z = (ridge_features(bg, ins, carb, self.lags, self.exogenous) - m.mu) / m.sd
            return bg[:, -1:] + m.coef[0] + z @ m.coef[1:]
        return _Fn(fn, mu=self.mu, sd=self.sd, coef=self.coef)

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
