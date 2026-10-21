"""Patch and hierarchical-MLP forecasters: PatchTST (Nie et al. 2023), N-HiTS (Challu et al. 2023).

Both read the ``TorchForecaster`` layout and stay export-traceable: fixed shapes, matmul gathers,
no data-dependent control flow.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from zoo.torch_base import TorchForecaster
from zoo.windows import ZONE


class _EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, P, d) -> (B, P, d), pre-norm self-attention then feed-forward."""
        b, p, d = x.shape
        qkv = self.qkv(self.n1(x)).reshape(b, p, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, p, d)))
        return x + self.drop(self.ff(self.n2(x)))


class PatchTSTNet(nn.Module):
    """Patched transformer over bg, plus an additive covariate MLP over past and announced doses."""

    def __init__(self, ctx: int, patch: int, stride: int, d: int, heads: int, layers: int,
                 ffn: int, cov_hidden: int, dropout: float) -> None:
        super().__init__()
        n_patch = (ctx - patch) // stride + 1
        idx = torch.arange(n_patch)[:, None] * stride + torch.arange(patch)[None, :]
        self.register_buffer('idx', idx.reshape(-1), persistent=False)
        self.ctx, self.n_patch, self.patch = ctx, n_patch, patch
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.randn(n_patch, d) * 0.02)
        self.layers = nn.ModuleList(_EncoderLayer(d, heads, ffn, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_patch * d, ZONE))
        self.cov = nn.Sequential(nn.Linear(2 * (ctx + ZONE), cov_hidden), nn.GELU(),
                                 nn.Linear(cov_hidden, cov_hidden), nn.GELU(),
                                 nn.Linear(cov_hidden, ZONE))

    def forward(self, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        """bg (B, ctx), ins/carb (B, ctx + ZONE) -> (B, ZONE) scaled delta."""
        assert bg.shape[1] == self.ctx and ins.shape[1] == self.ctx + ZONE
        b = bg.shape[0]
        x = self.embed(bg[:, self.idx].reshape(b, self.n_patch, self.patch)) + self.pos
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x).reshape(b, -1)) + self.cov(torch.cat([ins, carb], dim=1))


class PatchTST(TorchForecaster):
    name = 'patchtst'

    def __init__(self, ctx: int = 72, patch: int = 6, stride: int = 3, d_model: int = 64,
                 heads: int = 4, layers: int = 3, ffn: int = 128, cov_hidden: int = 128,
                 dropout: float = 0.1, **kw) -> None:
        self.arch = dict(patch=patch, stride=stride, d=d_model, heads=heads, layers=layers,
                         ffn=ffn, cov_hidden=cov_hidden, dropout=dropout)
        super().__init__(ctx, **kw)

    def build(self) -> nn.Module:
        return PatchTSTNet(self.ctx, **self.arch)


def _interp_matrix(n: int, length: int) -> torch.Tensor:
    """(n, length) linear interpolation from ``n`` evenly spaced knots onto ``length`` steps."""
    if n == 1:
        return torch.ones(1, length)
    pos = np.arange(length) * (n - 1) / (length - 1)
    lo = np.minimum(np.floor(pos).astype(np.int64), n - 2)
    frac = pos - lo
    m = np.zeros((n, length), np.float32)
    m[lo, np.arange(length)] = 1.0 - frac
    m[lo + 1, np.arange(length)] = frac
    return torch.from_numpy(m)


class _NHiTSBlock(nn.Module):
    def __init__(self, ctx: int, pool: int, n_back: int, n_fore: int, hidden: int,
                 layers: int, dropout: float) -> None:
        super().__init__()
        assert ctx % pool == 0
        self.pool = pool
        dims = [ctx // pool + 2 * (ctx + ZONE)] + [hidden] * layers
        mlp: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            mlp += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*mlp)
        self.theta = nn.Linear(hidden, n_back + n_fore)
        self.n_back = n_back
        self.register_buffer('back', _interp_matrix(n_back, ctx), persistent=False)
        self.register_buffer('fore', _interp_matrix(n_fore, ZONE), persistent=False)

    def forward(self, x: torch.Tensor, exo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, ctx), exo (B, 2 (ctx + ZONE)) -> backcast (B, ctx), forecast (B, ZONE)."""
        p = F.max_pool1d(x.unsqueeze(1), self.pool, self.pool).squeeze(1)
        t = self.theta(self.mlp(torch.cat([p, exo], dim=1)))
        return t[:, :self.n_back] @ self.back, t[:, self.n_back:] @ self.fore


class NHiTSNet(nn.Module):
    """Stacked MLP blocks: multi-rate max-pooled input, coarse-to-fine interpolated outputs."""

    def __init__(self, ctx: int, pools: tuple[int, ...], back_rates: tuple[int, ...],
                 fore_rates: tuple[int, ...], hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.ctx = ctx
        self.blocks = nn.ModuleList(
            _NHiTSBlock(ctx, p, max(ctx // rb, 1), max(ZONE // rf, 1), hidden, layers, dropout)
            for p, rb, rf in zip(pools, back_rates, fore_rates))

    def forward(self, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        """bg (B, ctx), ins/carb (B, ctx + ZONE) -> (B, ZONE) scaled delta."""
        assert bg.shape[1] == self.ctx and ins.shape[1] == self.ctx + ZONE
        exo = torch.cat([ins, carb], dim=1)
        x, y = bg, torch.zeros(bg.shape[0], ZONE, dtype=bg.dtype, device=bg.device)
        for block in self.blocks:
            back, fore = block(x, exo)
            x, y = x - back, y + fore
        return y


class NHiTS(TorchForecaster):
    name = 'nhits'

    def __init__(self, ctx: int = 72, pools: tuple[int, ...] = (6, 3, 1),
                 back_rates: tuple[int, ...] = (6, 3, 1), fore_rates: tuple[int, ...] = (12, 4, 1),
                 hidden: int = 256, layers: int = 2, dropout: float = 0.0, **kw) -> None:
        self.arch = dict(pools=pools, back_rates=back_rates, fore_rates=fore_rates,
                         hidden=hidden, layers=layers, dropout=dropout)
        super().__init__(ctx, **kw)

    def build(self) -> nn.Module:
        return NHiTSNet(self.ctx, **self.arch)


MODELS = {
    'patchtst': PatchTST,
    'nhits': NHiTS,
}
