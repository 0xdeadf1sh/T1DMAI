"""ARIMA and regression-with-ARIMA-errors (SARIMAX), one parameter set fitted on pooled windows.

With parameters fixed and a gap-free context the Kalman forecast is affine in the window, so fit
stores that map and predict is three matmuls. Fit needs statsmodels. No seasonal term.
"""
from __future__ import annotations

import numpy as np
from torch import nn

from zoo.classical import Forecaster, _Fn
from zoo.windows import ZONE, Windows

FIT_WINDOWS = 2000
SEPARATOR = 96  # NaN steps between strung windows; the integrated state's variance absorbs the jump
ABSORB_TAU_MIN = (20.0, 60.0, 180.0)


def absorbed(dose: np.ndarray) -> np.ndarray:
    """(L, n) doses -> (L, n * len(ABSORB_TAU_MIN)): dose taken up by step t from steps < t.

    Uptake is first-order per time constant, so step t's column reads doses before t only.
    """
    cols = []
    for tau in ABSORB_TAU_MIN:
        a = np.exp(-5.0 / tau)
        on_board = np.zeros(dose.shape[1])
        total = np.zeros(dose.shape[1])
        out = np.empty(dose.shape, np.float64)
        for t in range(len(dose)):
            on_board = a * on_board + dose[t]
            total = total + dose[t]
            out[t] = total - on_board
        cols.append(out)
    return np.concatenate(cols, axis=1)


class StateSpace(Forecaster):
    """Training windows strung into one NaN-separated series; maximum likelihood via statsmodels."""

    def __init__(self, order: tuple[int, int, int], exogenous: bool, ctx: int,
                 fit_windows: int = FIT_WINDOWS) -> None:
        self.order = tuple(order)
        self.exogenous = exogenous
        self.reads_doses = exogenous
        self.ctx = ctx
        self.fit_windows = fit_windows
        self.name = 'sarimax' if exogenous else 'arima'
        self.params = self.Wy = self.Wi = self.Wc = self.c = np.empty(0)
        self.aic = np.nan

    def _exog(self, ins: np.ndarray, carb: np.ndarray) -> np.ndarray | None:
        if not self.exogenous:
            return None
        return absorbed(np.stack([ins, carb], axis=1).astype(np.float64))

    def _model(self, endog: np.ndarray, exog: np.ndarray | None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        return SARIMAX(endog, exog=exog, order=self.order, trend='n',
                       enforce_stationarity=False, enforce_invertibility=False)

    def fit(self, w: Windows) -> None:
        rng = np.random.default_rng(0)
        if len(w) > self.fit_windows:
            w = w.take(np.sort(rng.choice(len(w), self.fit_windows, replace=False)))
        c = self.ctx
        gap = np.full(SEPARATOR, np.nan)
        ys, xs = [], []
        for i in range(len(w)):
            # Context only for SARIMAX: a causal run zeroes zone doses that were in fact taken.
            y = w.bg[i] if self.exogenous else np.concatenate([w.bg[i], w.target[i]])
            ys += [y.astype(np.float64) - w.bg[i, -1], gap]
            if self.exogenous:
                x = self._exog(w.insulin[i, :c], w.carb[i, :c])
                xs += [x - x[-1], np.zeros((SEPARATOR, x.shape[1]))]
        exog = np.concatenate(xs) if self.exogenous else None
        res = self._model(np.concatenate(ys), exog).fit(disp=False, maxiter=200)
        self.params = np.asarray(res.params, np.float64)
        self.aic = float(res.aic)
        print(f'  {self.name}{self.order}: aic {self.aic:.0f}  '
              + ' '.join(f'{n}={v:.4g}' for n, v in zip(res.model.param_names, self.params)),
              flush=True)
        self._linearise()

    def forecast_one(self, bg: np.ndarray, ins: np.ndarray, carb: np.ndarray) -> np.ndarray:
        """statsmodels' own ZONE-step forecast for one window at the fitted parameters."""
        c = self.ctx
        x = self._exog(ins, carb)
        res = self._model(bg.astype(np.float64), None if x is None else x[:c]).filter(self.params)
        return np.asarray(res.forecast(ZONE, exog=None if x is None else x[c:]))

    def _linearise(self) -> None:
        c, L = self.ctx, self.ctx + ZONE
        zb, zd = np.zeros(c), np.zeros(L)
        self.c = self.forecast_one(zb, zd, zd)
        eye = np.eye(c)
        self.Wy = np.stack([self.forecast_one(eye[j], zd, zd) - self.c for j in range(c)], axis=1)
        if self.exogenous:
            eL = np.eye(L)
            self.Wi = np.stack([self.forecast_one(zb, eL[j], zd) - self.c for j in range(L)], axis=1)
            self.Wc = np.stack([self.forecast_one(zb, zd, eL[j]) - self.c for j in range(L)], axis=1)
        else:
            self.Wi = self.Wc = np.zeros((ZONE, L))

    def predict(self, w: Windows) -> np.ndarray:
        return (self.c + w.bg.astype(np.float64) @ self.Wy.T
                + w.insulin.astype(np.float64) @ self.Wi.T + w.carb.astype(np.float64) @ self.Wc.T)

    def graph(self) -> nn.Module:
        return _Fn(lambda m, bg, ins, carb: m.c + bg @ m.Wy.T + ins @ m.Wi.T + carb @ m.Wc.T,
                   c=self.c, Wy=self.Wy, Wi=self.Wi, Wc=self.Wc)

    def state(self) -> dict[str, np.ndarray]:
        return {'params': self.params, 'Wy': self.Wy, 'Wi': self.Wi, 'Wc': self.Wc, 'c': self.c,
                'order': np.asarray(self.order), 'aic': np.asarray(self.aic)}

    def load(self, state: dict[str, np.ndarray]) -> None:
        self.params, self.Wy, self.Wi, self.Wc, self.c = (
            state['params'], state['Wy'], state['Wi'], state['Wc'], state['c'])
        self.order = tuple(int(v) for v in state['order'])
        self.aic = float(state['aic'])


# Lowest AIC and held-out RMSE over p<=4, q<=2, d=1 on causal training windows.
ARIMA_ORDER = (2, 1, 2)
SARIMAX_ORDER = (2, 1, 2)

MODELS = {
    'arima': lambda: StateSpace(ARIMA_ORDER, exogenous=False, ctx=48),
    'sarimax': lambda: StateSpace(SARIMAX_ORDER, exogenous=True, ctx=96),
}
