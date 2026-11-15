"""Chronos-2, TimesFM 2.5 and Timestral (also fine-tuned, ``_ft``) via upstream; no export graph.

Each registers only where its package imports (``.venv-zoo``). Blind: Chronos-2 and Timestral
read past doses only, TimesFM glucose only. Given: zone doses are known-future covariates.
"""
from __future__ import annotations

import importlib.util
import io
import os
import time

import numpy as np
import torch

from config import QUANTILE_LEVELS
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
TIMESTRAL_TRAIN_BATCH = 64
TIMESTRAL_MICRO = 4  # windows per backward; 209M holds ~0.15 GB of activations per window


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
    """Mistral AI 2026; 21-quantile head, upstream two-pass symmetric average.

    ``finetune`` trains every weight on that average's pinball loss in mg/dL, as Chronos-Bolt is
    trained; the best epoch on the validation windows is kept, zero-shot if none beats it.
    """
    ctx = CTX
    size = ''
    finetune = False

    def __init__(self, epochs: int = 1, batch: int = TIMESTRAL_TRAIN_BATCH, lr: float = 1e-5,
                 weight_decay: float = 0.0, seed: int = 0) -> None:
        self._fc = None
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.weight_decay, self.seed = weight_decay, seed

    def _forecaster(self):
        if self._fc is None:
            from timestral import TimestralForecaster
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._fc = TimestralForecaster.from_pretrained(
                f'mistralai/Timestral-{self.size}', device=dev,
                revision=TIMESTRAL_REVISIONS[self.size])
        return self._fc

    def _covariates(self, w: Windows, sl) -> dict[str, np.ndarray]:
        doses = np.stack([w.insulin[sl], w.carb[sl]], axis=1)
        return ({'past_covariates': doses[..., :self.ctx]} if self.causal
                else {'known_covariates': doses})

    def predict_quantiles(self, w: Windows) -> np.ndarray:
        """(N, ZONE, len(QUANTILE_LEVELS)) mg/dL, through upstream ``forecast``."""
        if not len(w):
            return np.empty((0, ZONE, len(QUANTILE_LEVELS)))
        fc = self._forecaster()
        pick = [fc.quantile_levels.index(q) for q in QUANTILE_LEVELS]
        out = []
        for s in range(0, len(w), TIMESTRAL_BATCH):
            sl = slice(s, s + TIMESTRAL_BATCH)
            f = fc.forecast(w.bg[sl], prediction_length=ZONE, symmetric_averaging=True,
                            **self._covariates(w, sl))
            out.append(f.quantiles[:, 0][..., pick].float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def predict(self, w: Windows) -> np.ndarray:
        return self.predict_quantiles(w)[..., QUANTILE_LEVELS.index(0.5)]

    def _quantiles(self, bg: np.ndarray, cov: dict[str, np.ndarray]) -> torch.Tensor:
        """(B, ZONE, 21) with grad: ``forecast`` minus its no-op cap and non-negative clip."""
        fc = self._forecaster()
        ps = fc.patch_size
        assert ZONE <= ps * fc.per_shot_horizon_patches, 'one decode block, so no refine pass'
        dev = fc.device
        x = torch.as_tensor(bg, dtype=torch.float32, device=dev)[:, None]
        cv = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in cov.items()}

        def one(sign: float) -> torch.Tensor:
            rows, mask, roles, kf, kfm, n_tgt = fc._assemble_rows(
                sign * x, torch.ones_like(x, dtype=torch.bool), ZONE,
                *(sign * cv[k] if k in cv else None for k in ('past_covariates', 'known_covariates')))
            b, v, t = rows.shape
            q = fc.model.predict_block(rows.reshape(b * v, t), mask.reshape(b * v, t),
                                       torch.arange(b, device=dev).repeat_interleave(v),
                                       -(-ZONE // ps), roles, kf, kfm)
            return q.reshape(b, v, -1, q.shape[-1])[:, :n_tgt][:, 0, :ZONE]

        return 0.5 * (one(1.0) + (-one(-1.0)).flip(-1))

    def _loss(self, w: Windows, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """(pinball, median error at measured steps) over windows ``idx``."""
        from zoo.chronos import pinball
        fc = self._forecaster()
        q = self._quantiles(w.bg[idx], self._covariates(w, idx))
        y = torch.as_tensor(w.target[idx], dtype=torch.float32, device=fc.device)
        levels = torch.tensor(fc.quantile_levels, device=fc.device)
        err = q[..., fc.median_index] - y
        return pinball(q.transpose(1, 2), y, levels), err[torch.isfinite(err)]

    def _val(self, w: Windows) -> tuple[float, float]:
        model = self._forecaster().model
        model.eval()
        tot = sq = n = 0.0
        with torch.no_grad():
            for s in range(0, len(w), TIMESTRAL_BATCH):
                idx = np.arange(s, min(s + TIMESTRAL_BATCH, len(w)))
                loss, err = self._loss(w, idx)
                tot += float(loss) * len(idx)
                sq += float((err ** 2).sum())
                n += err.numel()
        return tot / max(len(w), 1), (sq / max(n, 1.0)) ** 0.5

    def fit(self, w: Windows, val: Windows | None = None) -> None:
        if not self.finetune:
            return
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        model = self._forecaster().model
        tr = np.arange(len(w))
        has_val = val is not None and len(val) > 0
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = max(self.epochs * (-(-len(tr) // self.batch)), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: max(1.0 - i / steps, 0.0))
        best, best_state = float('inf'), None
        if has_val:
            best, rmse = self._val(val)
            best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
            print(f'  zero-shot val pinball {best:.4f}  median rmse {rmse:.2f} mg/dL', flush=True)
        for ep in range(self.epochs):
            model.train()
            t = time.time()
            tr = rng.permutation(tr)
            for s in range(0, len(tr), self.batch):
                idx = np.sort(tr[s:s + self.batch])
                opt.zero_grad(set_to_none=True)
                for m in range(0, len(idx), TIMESTRAL_MICRO):
                    part = idx[m:m + TIMESTRAL_MICRO]
                    loss, _ = self._loss(w, part)
                    (loss * (len(part) / len(idx))).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            if not has_val:
                print(f'  epoch {ep + 1}/{self.epochs}  ({time.time() - t:.0f} s)', flush=True)
                continue
            v, rmse = self._val(val)
            print(f'  epoch {ep + 1}/{self.epochs}  val pinball {v:.4f}  median rmse '
                  f'{rmse:.2f} mg/dL  ({time.time() - t:.0f} s)', flush=True)
            if v < best:
                best = v
                best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

    def state(self) -> dict[str, np.ndarray]:
        if not self.finetune:
            return {}
        buf = io.BytesIO()
        torch.save(self._forecaster().model.state_dict(), buf)
        return {'state_dict': np.frombuffer(buf.getvalue(), dtype=np.uint8)}

    def load(self, state: dict[str, np.ndarray]) -> None:
        if 'state_dict' in state:
            model = self._forecaster().model
            model.load_state_dict(torch.load(io.BytesIO(state['state_dict'].tobytes()),
                                             map_location=self._forecaster().device))
            model.eval()


def _timestral(size: str, finetune: bool = False) -> type[Timestral]:
    suffix = '_ft' if finetune else ''
    return type(f'Timestral{size}{suffix.upper()}', (Timestral,),
                {'name': f'timestral_{size.lower()}{suffix}', 'size': size, 'finetune': finetune})


MODELS = {k: v for k, v, pkg in (('chronos2', Chronos2, 'chronos'),
                                  ('timesfm25', TimesFM25, 'timesfm'),
                                  *((f'timestral_{z.lower()}{"_ft" * ft}', _timestral(z, ft),
                                     'timestral')
                                    for z in TIMESTRAL_REVISIONS for ft in (False, True)))
          if importlib.util.find_spec(pkg) is not None}
