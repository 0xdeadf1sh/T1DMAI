"""Gradient-boosted trees on the ridge ARX features, one booster per zone step.

Fit needs ``xgboost`` or ``lightgbm``; predict and the export graph walk one flattened node table
in plain torch.
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
LGB_PARAMS = {'objective': 'regression', 'max_depth': 6, 'num_leaves': 63, 'learning_rate': 0.1,
              'min_sum_hessian_in_leaf': 10.0, 'min_data_in_leaf': 10, 'bagging_fraction': 0.8,
              'bagging_freq': 1, 'feature_fraction': 0.8, 'num_threads': 8, 'seed': 0,
              'deterministic': True, 'verbose': -1}
PREDICT_CHUNK = 1024
SPLIT_MARGIN = 2e-6  # relative; covers float32 summation-order drift in the dose bin sums
TABLE = ('feature', 'threshold', 'left', 'right', 'value', 'default_left')


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


def _xgb_table(t: dict) -> dict[str, np.ndarray]:
    """One XGBoost tree as local node columns; ``left < 0`` marks a leaf."""
    left = np.asarray(t['left_children'])
    leaf = left < 0
    return {'feature': np.where(leaf, 0, t['split_indices']),
            'threshold': np.asarray(t['split_conditions'], np.float32),
            'left': left, 'right': np.asarray(t['right_children']),
            'value': np.where(leaf, t['split_conditions'], 0.0),
            'default_left': np.asarray(t['default_left'])}


def _lgb_table(root: dict) -> dict[str, np.ndarray]:
    """One LightGBM tree, breadth-first; ``x <= t`` becomes ``x < nextafter(f32(t))``."""
    nodes, i = [root], 0
    while i < len(nodes):
        n = nodes[i]
        if 'leaf_value' not in n:
            assert n['decision_type'] == '<=', n['decision_type']
            nodes += [n['left_child'], n['right_child']]
        i += 1
    ids = {id(n): k for k, n in enumerate(nodes)}
    cols: dict[str, list] = {k: [] for k in TABLE}
    for n in nodes:
        leaf = 'leaf_value' in n
        t = 0.0 if leaf else n['threshold']
        t32 = np.float32(t)
        if t32 > t:
            t32 = np.nextafter(t32, np.float32(-np.inf))
        cols['feature'].append(0 if leaf else n['split_feature'])
        cols['threshold'].append(np.nextafter(t32, np.float32(np.inf)))
        cols['left'].append(-1 if leaf else ids[id(n['left_child'])])
        cols['right'].append(-1 if leaf else ids[id(n['right_child'])])
        cols['value'].append(n['leaf_value'] if leaf else 0.0)
        cols['default_left'].append(not leaf and n['default_left'] and n['missing_type'] == 'NaN')
    return {k: np.asarray(v, np.float32 if k == 'threshold' else None) for k, v in cols.items()}


class _Trees(Forecaster):
    """Tree ensemble per zone step on the delta from the last reading.

    One node table, leaves pointing at themselves: ``depth`` gather steps walk every tree at once.
    Split is ``x < threshold`` in float32, nudged off its training value by ``_widen``.
    """
    left_closed = False  # True: the library's split is ``x <= v``, cut placed just above ``v``

    def __init__(self) -> None:
        self.arx = Ridge(exogenous=True)
        self.ctx = self.arx.ctx
        self.boosters: list = []
        self.feature = self.threshold = self.left = self.right = self.value = np.empty(0)
        self.default_left = self.roots = self.step = self.base = np.empty(0)
        self.depth = 0

    def _x(self, w: Windows) -> np.ndarray:
        return self.arx.features(w).astype(np.float32)

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def fit(self, w: Windows) -> None:
        if len(w) > XGB_WINDOWS:
            w = w.take(np.sort(np.random.default_rng(0).choice(len(w), XGB_WINDOWS, replace=False)))
        X = self._x(w)
        self._train(X, w.target.astype(np.float64) - w.bg[:, -1:])
        self._widen(X)

    def _widen(self, X: np.ndarray) -> None:
        """Move each threshold ``SPLIT_MARGIN`` away from the training value it cuts on.

        Values within the margin are float32 rounding siblings; they keep their side, never
        crossing the next training value. ``left_closed`` cuts sit above it, others below.
        """
        split = self.left != np.arange(len(self.left))
        for f in range(X.shape[1]):
            idx = np.flatnonzero(split & (self.feature == f))
            u = np.unique(X[:, f]).astype(np.float64)
            t = self.threshold[idx].astype(np.float64)
            tol = SPLIT_MARGIN * np.maximum(np.abs(t), 1.0)
            if self.left_closed:
                i = np.searchsorted(u, t + tol, side='right')
                nxt = np.where(i < len(u), u[np.minimum(i, len(u) - 1)], np.inf)
                self.threshold[idx] = np.minimum(t + tol, (nxt + t) / 2).astype(np.float32)
            else:
                i = np.searchsorted(u, t - tol, side='left')
                prev = np.where(i > 0, u[np.maximum(i - 1, 0)], -np.inf)
                self.threshold[idx] = np.maximum(t - tol, (prev + t) / 2).astype(np.float32)

    def _flatten(self, trees: list[tuple[int, dict]], base: list[float]) -> None:
        """``(zone step, local table)`` per tree -> the global table, leaves self-looped."""
        cols: dict[str, list] = {k: [] for k in TABLE}
        roots, step, depth, off = [], [], 0, 0
        for k, t in trees:
            left, right = t['left'], t['right']
            n = len(left)
            leaf = left < 0
            own = np.arange(n) + off
            t = dict(t, left=np.where(leaf, own, left + off), right=np.where(leaf, own, right + off))
            for key in TABLE:
                cols[key].append(t[key])
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
        return {k: getattr(self, k) for k in (*TABLE, 'roots', 'step', 'base')}

    def predict(self, w: Windows) -> np.ndarray:
        ns = SimpleNamespace(**{k: _t(v) for k, v in self._graph_state().items()})
        out = []
        with torch.no_grad():
            for i in range(0, len(w), PREDICT_CHUNK):
                c = w.take(np.arange(i, min(i + PREDICT_CHUNK, len(w))))
                out.append(self._forward(ns, _t(c.bg), _t(c.insulin), _t(c.carb)).numpy())
        return np.concatenate(out) if out else np.empty((0, ZONE))

    def _native(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_native(self, w: Windows) -> np.ndarray:
        """(N, ZONE) mg/dL from the fitted boosters; equals ``predict`` on the training windows."""
        return w.bg[:, -1:] + self._native(self._x(w))

    def graph(self) -> nn.Module:
        return _Fn(self._forward, **self._graph_state())

    def state(self) -> dict[str, np.ndarray]:
        return {**self._graph_state(), 'depth': np.array(self.depth)}

    def load(self, state: dict[str, np.ndarray]) -> None:
        for k, v in state.items():
            setattr(self, k, v)
        self.depth = int(self.depth)


class XGBoost(_Trees):
    name = 'xgboost'

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        import xgboost as xgb
        params = dict(XGB_PARAMS, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.boosters = []
        for k in range(ZONE):
            ok = np.isfinite(y[:, k])
            d = xgb.DMatrix(X[ok], label=y[ok, k])
            p = dict(params, base_score=float(y[ok, k].mean()))
            self.boosters.append(xgb.train(p, d, XGB_ROUNDS))
        self._flatten([(k, _xgb_table(t)) for k, b in enumerate(self.boosters) for t in _trees(b)],
                      [_base_score(b) for b in self.boosters])

    def _native(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        d = xgb.DMatrix(X)
        return np.stack([b.predict(d) for b in self.boosters], axis=1)


class LightGBM(_Trees):
    """LightGBM at XGBoost's depth, rate, rounds and sampling; leaf-wise growth within them."""
    name = 'lightgbm'
    left_closed = True

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        import lightgbm as lgb
        self.boosters = []
        for k in range(ZONE):
            ok = np.isfinite(y[:, k])
            self.boosters.append(lgb.train(LGB_PARAMS, lgb.Dataset(X[ok], label=y[ok, k]),
                                           XGB_ROUNDS))
        self._flatten([(k, _lgb_table(t['tree_structure'])) for k, b in enumerate(self.boosters)
                       for t in b.dump_model()['tree_info']], [0.0] * ZONE)

    def _native(self, X: np.ndarray) -> np.ndarray:
        return np.stack([b.predict(X) for b in self.boosters], axis=1)


MODELS = {
    'xgboost': XGBoost,
    'lightgbm': LightGBM,
}
