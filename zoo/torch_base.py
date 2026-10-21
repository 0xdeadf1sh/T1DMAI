"""Shared trainer for neural zoo models: one input layout, one masked loss, one schedule.

``build()`` takes bg ``(B, ctx)`` as the scaled delta from the last reading and log1p doses
``(B, ctx + ZONE)``; returns the ``(B, ZONE)`` delta / ``DELTA_SCALE``, export-traceable.
"""
from __future__ import annotations

import io
import time

import numpy as np
import torch
from torch import nn

from zoo.classical import Forecaster
from zoo.windows import ZONE, Windows

DELTA_SCALE = 50.0  # mg/dL per unit of model output; zone deltas sit within a few units
BG_SCALE = 50.0


def prep(bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor):
    """Raw mg/dL, U and g per step -> the network's three inputs."""
    return ((bg - bg[:, -1:]) / BG_SCALE, torch.log1p(ins.clamp(min=0.0)),
            torch.log1p(carb.clamp(min=0.0) / 10.0))


class _Graph(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        return bg[:, -1:] + self.net(*prep(bg, ins, carb)) * DELTA_SCALE


class TorchForecaster(Forecaster):
    def __init__(self, ctx: int, epochs: int = 8, batch: int = 1024, lr: float = 1e-3,
                 weight_decay: float = 1e-4, val_frac: float = 0.05, seed: int = 0) -> None:
        self.ctx = ctx
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.weight_decay, self.val_frac, self.seed = weight_decay, val_frac, seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net: nn.Module = self.build()

    def build(self) -> nn.Module:
        raise NotImplementedError

    @staticmethod
    def inputs(w: Windows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return prep(*(torch.from_numpy(np.asarray(x, np.float32)) for x in (w.bg, w.insulin, w.carb)))

    def graph(self) -> nn.Module:
        return _Graph(self.net.cpu().eval())

    def _batches(self, w: Windows, idx: np.ndarray, shuffle: bool, rng: np.random.Generator):
        if shuffle:
            idx = rng.permutation(idx)
        for s in range(0, len(idx), self.batch):
            sub = w.take(idx[s:s + self.batch])
            bg, ins, carb = (t.to(self.device) for t in self.inputs(sub))
            y = torch.from_numpy(((sub.target - sub.bg[:, -1:]) / DELTA_SCALE).astype(np.float32))
            yield bg, ins, carb, y.to(self.device)

    def _loss(self, w: Windows, idx: np.ndarray, rng) -> float:
        self.net.eval()
        tot = n = 0.0
        with torch.no_grad():
            for bg, ins, carb, y in self._batches(w, idx, False, rng):
                ok = torch.isfinite(y)
                d = (self.net(bg, ins, carb) - torch.nan_to_num(y))[ok]
                tot += float((d ** 2).sum())
                n += float(ok.sum())
        return tot / max(n, 1.0)

    def fit(self, w: Windows) -> None:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(w))
        n_val = max(int(len(w) * self.val_frac), 1)
        val, tr = order[:n_val], order[n_val:]
        self.net.to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = self.epochs * ((len(tr) + self.batch - 1) // self.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=max(steps, 1))
        best, best_state = float('inf'), None
        for ep in range(self.epochs):
            self.net.train()
            t = time.time()
            for bg, ins, carb, y in self._batches(w, tr, True, rng):
                ok = torch.isfinite(y)
                d = (self.net(bg, ins, carb) - torch.nan_to_num(y))[ok]
                loss = (d ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
            v = self._loss(w, val, rng)
            rmse = DELTA_SCALE * v ** 0.5
            print(f'  epoch {ep + 1}/{self.epochs}  val rmse {rmse:.2f} mg/dL  '
                  f'({time.time() - t:.0f} s)', flush=True)
            if v < best:
                best = v
                best_state = {k: x.detach().clone() for k, x in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, w: Windows) -> np.ndarray:
        self.net.to(self.device).eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(w), self.batch * 4):
                sub = w.take(np.arange(s, min(s + self.batch * 4, len(w))))
                bg, ins, carb = (t.to(self.device) for t in self.inputs(sub))
                out.append(self.net(bg, ins, carb).float().cpu().numpy() * DELTA_SCALE)
        return w.bg[:, -1:] + np.concatenate(out) if out else np.empty((0, ZONE))

    def state(self) -> dict[str, np.ndarray]:
        buf = io.BytesIO()
        torch.save(self.net.state_dict(), buf)
        return {'state_dict': np.frombuffer(buf.getvalue(), dtype=np.uint8)}

    def load(self, state: dict[str, np.ndarray]) -> None:
        sd = torch.load(io.BytesIO(state['state_dict'].tobytes()), map_location='cpu')
        self.net.load_state_dict(sd)
