"""Gradient-boosted trees on the ridge ARX features, one booster per zone step.

Fit needs ``xgboost``; predict and the export graph walk the dumped trees in plain torch.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from zoo.classical import Forecaster, Ridge, _Fn, _t, ridge_features
from zoo.windows import ZONE, Windows

XGB_WINDOWS = 500_000
XGB_ROUNDS = 200
XGB_PARAMS = {'max_depth': 6, 'eta': 0.1, 'min_child_weight': 10.0, 'subsample': 0.8,
              'colsample_bytree': 0.8, 'tree_method': 'hist', 'nthread': 8, 'seed': 0}
PREDICT_CHUNK = 1024
SPLIT_MARGIN = 2e-6  # relative; covers float32 summation-order drift in the dose bin sums


def _trees(booster) -> list[dict]:
    return json.loads(booster.save_raw('json'))['learner']['gradient_booster']['model']['trees']


def _base_score(booster) -> float:
    s = json.loads(booster.save_raw('json'))['learner']['learner_model_param']['base_score']
    return float(s.strip('[]'))


def _depth(left: np.ndarray, right: np.ndarray) -> int:
    depth, frontier = 0, [0]
    while True:
        frontier = [c for n in frontier for c in (left[n], right[n]) if c >= 0]
        if not frontier:
            return depth
        depth += 1


class XGBoost(Forecaster):
    """XGBoost regressor per zone step on the delta from the last reading.

    One node table, leaves pointing at themselves: ``depth`` gather steps walk every tree at once.
    Split is ``x < threshold`` in float32, the threshold lowered into the training gap below it.
    """
    name = 'xgboost'

    def __init__(self) -> None:
        self.arx = Ridge(exogenous=True)
        self.ctx = self.arx.ctx
        self.boosters: list = []
        self.feature = self.threshold = self.left = self.right = self.value = np.empty(0)
        self.default_left = self.roots = self.step = self.base = np.empty(0)
        self.depth = 0

    def _x(self, w: Windows) -> np.ndarray:
        return self.arx.features(w).astype(np.float32)

    def fit(self, w: Windows) -> None:
        import xgboost as xgb
        if len(w) > XGB_WINDOWS:
            w = w.take(np.sort(np.random.default_rng(0).choice(len(w), XGB_WINDOWS, replace=False)))
        params = dict(XGB_PARAMS, device='cuda' if torch.cuda.is_available() else 'cpu')
        X = self._x(w)
        y = w.target.astype(np.float64) - w.bg[:, -1:]
        self.boosters = []
        for k in range(ZONE):
            ok = np.isfinite(y[:, k])
            d = xgb.DMatrix(X[ok], label=y[ok, k])
            p = dict(params, base_score=float(y[ok, k].mean()))
            self.boosters.append(xgb.train(p, d, XGB_ROUNDS))
        self._flatten()
        self._widen(X)

    def _widen(self, X: np.ndarray) -> None:
        """Lower each threshold below the training values within ``SPLIT_MARGIN`` under it.

        Those values are float32 rounding siblings of the cut; all go right, with margin to spare.
        """
        split = self.left != np.arange(len(self.left))
        for f in range(X.shape[1]):
            idx = np.flatnonzero(split & (self.feature == f))
            u = np.unique(X[:, f]).astype(np.float64)
            t = self.threshold[idx].astype(np.float64)
            tol = SPLIT_MARGIN * np.maximum(np.abs(t), 1.0)
            i = np.searchsorted(u, t - tol, side='left')
            prev = np.where(i > 0, u[np.maximum(i - 1, 0)], -np.inf)
            self.threshold[idx] = np.maximum(t - tol, (prev + t) / 2).astype(np.float32)

    def _flatten(self) -> None:
        cols = {k: [] for k in ('feature', 'threshold', 'left', 'right', 'value', 'default_left')}
        roots, step, base, depth, off = [], [], [], 0, 0
        for k, b in enumerate(self.boosters):
            base.append(_base_score(b))
            for t in _trees(b):
                left, right = np.asarray(t['left_children']), np.asarray(t['right_children'])
                n = len(left)
                leaf = left < 0
                own = np.arange(n) + off
                cols['feature'].append(np.where(leaf, 0, t['split_indices']))
                cols['threshold'].append(np.asarray(t['split_conditions'], np.float32))
                cols['left'].append(np.where(leaf, own, left + off))
                cols['right'].append(np.where(leaf, own, right + off))
                cols['value'].append(np.where(leaf, t['split_conditions'], 0.0))
                cols['default_left'].append(np.asarray(t['default_left']))
                depth = max(depth, _depth(left, right))
                roots.append(off)
                step.append(k)
                off += n
        for key, parts in cols.items():
            setattr(self, key, np.concatenate(parts))
        self.roots, self.base, self.depth = np.asarray(roots), np.asarray(base), depth
        self.step = np.eye(ZONE)[np.asarray(step)]

    def _forward(self, m, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        x = ridge_features(bg, ins, carb, self.arx.lags, True).float()
        feature, left, right = m.feature.long(), m.left.long(), m.right.long()
        node = m.roots.long()[None, :].expand(x.shape[0], -1)
        for _ in range(self.depth):
            v = torch.gather(x, 1, feature[node])
            go = (v < m.threshold[node]) | (torch.isnan(v) & (m.default_left[node] > 0.5))
            node = torch.where(go, left[node], right[node])
        return bg[:, -1:] + m.base + m.value[node] @ m.step

    def _graph_state(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in ('feature', 'threshold', 'left', 'right', 'value',
                                              'default_left', 'roots', 'step', 'base')}

    def predict(self, w: Windows) -> np.ndarray:
        ns = SimpleNamespace(**{k: _t(v) for k, v in self._graph_state().items()})
        out = []
        with torch.no_grad():
            for i in range(0, len(w), PREDICT_CHUNK):
                c = w.take(np.arange(i, min(i + PREDICT_CHUNK, len(w))))
                out.append(self._forward(ns, _t(c.bg), _t(c.insulin), _t(c.carb)).numpy())
        return np.concatenate(out) if out else np.empty((0, ZONE))

    def predict_native(self, w: Windows) -> np.ndarray:
        """(N, ZONE) mg/dL from the fitted boosters; equals ``predict`` on the training windows."""
        import xgboost as xgb
        d = xgb.DMatrix(self._x(w))
        return w.bg[:, -1:] + np.stack([b.predict(d) for b in self.boosters], axis=1)

    def graph(self) -> nn.Module:
        return _Fn(self._forward, **self._graph_state())

    def state(self) -> dict[str, np.ndarray]:
        return {**self._graph_state(), 'depth': np.array(self.depth)}

    def load(self, state: dict[str, np.ndarray]) -> None:
        for k, v in state.items():
            setattr(self, k, v)
        self.depth = int(self.depth)


MODELS = {
    'xgboost': XGBoost,
}
