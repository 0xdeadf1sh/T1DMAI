"""Chronos-2, TimesFM 2.5 and Timestral through upstream packages, zero-shot, no export graph.

Each registers only where its package imports (``.venv-zoo``). Blind: Chronos-2 and Timestral
read past doses only, TimesFM glucose only. Given: zone doses are known-future covariates.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import torch

from zoo.classical import Forecaster
from zoo.windows import ZONE, Windows

CTX = 288
CHRONOS2_REPO = 'amazon/chronos-2'
TIMESFM_REPO = 'google/timesfm-2.5-200m-pytorch'
CHRONOS2_BATCH = 256
TIMESFM_BATCH = 256
XREG_CHUNK = 256  # XReg fits one ridge per call, so a call never spans more than one chunk
TIMESTRAL_REVISIONS = {
    '5M': 'bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f',
    '28M': '49b26409ac87b5935539a5d8ffe7f397b720db47',
    '123M': '5a147287addda85f6acd8da3fb8858fe9af7e8f7',
    '209M': '9874d252ab0aae16f7f7c999f98f98d6ad7d8365',
}
TIMESTRAL_BATCH = 256


class Chronos2(Forecaster):
    """Ansari et al. 2025; median of the 21-quantile head."""
    name = 'chronos2'
    ctx = CTX

    def __init__(self) -> None:
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            from chronos import Chronos2Pipeline
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._pipe = Chronos2Pipeline.from_pretrained(CHRONOS2_REPO, device_map=dev)
        return self._pipe

    def predict(self, w: Windows) -> np.ndarray:
        if not len(w):
            return np.empty((0, ZONE))
        pipe, c = self._pipeline(), self.ctx
        inputs = []
        for bg, ins, carb in zip(w.bg, w.insulin, w.carb):
            d = {'target': bg, 'past_covariates': {'insulin': ins[:c], 'carb': carb[:c]}}
            if not self.causal:
                d['future_covariates'] = {'insulin': ins[c:], 'carb': carb[c:]}
            inputs.append(d)
        out = pipe.predict(inputs, prediction_length=ZONE, batch_size=CHRONOS2_BATCH)
        mid = list(pipe.quantiles).index(0.5)
        return np.stack([o[0, mid].float().cpu().numpy() for o in out]).astype(np.float64)


class TimesFM25(Forecaster):
    """Das et al. 2024, v2.5 200M; given track adds insulin and carbs through in-context XReg."""
    name = 'timesfm25'
    ctx = CTX

    def __init__(self) -> None:
        self._model = None

    def _net(self):
        if self._model is None:
            os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
            import timesfm
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO)
            m.compile(timesfm.ForecastConfig(
                max_context=CTX, max_horizon=32, normalize_inputs=True,
                use_continuous_quantile_head=True, fix_quantile_crossing=True,
                per_core_batch_size=TIMESFM_BATCH, return_backcast=True))
            self._model = m
        return self._model

    def predict(self, w: Windows) -> np.ndarray:
        if not len(w):
            return np.empty((0, ZONE))
        m = self._net()
        if self.causal:
            point, _ = m.forecast(horizon=ZONE, inputs=list(w.bg))
            return np.asarray(point, np.float64)[:, -ZONE:]
        out = []
        for s in range(0, len(w), XREG_CHUNK):
            sl = slice(s, s + XREG_CHUNK)
            o, _ = m.forecast_with_covariates(
                inputs=list(w.bg[sl]),
                dynamic_numerical_covariates={'insulin': list(w.insulin[sl]),
                                              'carb': list(w.carb[sl])},
                xreg_mode='xreg + timesfm', force_on_cpu=True)
            out += [np.asarray(a, np.float64)[-ZONE:] for a in o]
        return np.stack(out)


class Timestral(Forecaster):
    """Mistral AI 2026; median of the 21-quantile head, upstream two-pass symmetric average."""
    ctx = CTX
    size = ''

    def __init__(self) -> None:
        self._fc = None

    def _forecaster(self):
        if self._fc is None:
            from timestral import TimestralForecaster
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._fc = TimestralForecaster.from_pretrained(
                f'mistralai/Timestral-{self.size}', device=dev,
                revision=TIMESTRAL_REVISIONS[self.size])
        return self._fc

    def predict(self, w: Windows) -> np.ndarray:
        if not len(w):
            return np.empty((0, ZONE))
        fc, c = self._forecaster(), self.ctx
        out = []
        for s in range(0, len(w), TIMESTRAL_BATCH):
            sl = slice(s, s + TIMESTRAL_BATCH)
            doses = np.stack([w.insulin[sl], w.carb[sl]], axis=1)
            cov = ({'past_covariates': doses[..., :c]} if self.causal
                   else {'known_covariates': doses})
            f = fc.forecast(w.bg[sl], prediction_length=ZONE, symmetric_averaging=True, **cov)
            out.append(f.median[:, 0].float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def _timestral(size: str) -> type[Timestral]:
    return type(f'Timestral{size}', (Timestral,),
                {'name': f'timestral_{size.lower()}', 'size': size})


MODELS = {k: v for k, v, pkg in (('chronos2', Chronos2, 'chronos'),
                                  ('timesfm25', TimesFM25, 'timesfm'),
                                  *((f'timestral_{z.lower()}', _timestral(z), 'timestral')
                                    for z in TIMESTRAL_REVISIONS))
          if importlib.util.find_spec(pkg) is not None}
