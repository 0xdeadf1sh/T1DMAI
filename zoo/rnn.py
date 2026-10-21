"""Recurrent and dilated-convolution forecasters: Martinsson 2019 LSTM and Li 2020 GluNet.

Both are reimplemented from the papers' descriptions with a direct ``ZONE``-step regression head;
announced zone doses enter through a small MLP joined to the encoder's summary.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from zoo.torch_base import TorchForecaster
from zoo.windows import ZONE


class _ZoneDoses(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * ZONE, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        """(B, ZONE) zone insulin and carbs -> (B, hidden)."""
        return self.net(torch.cat([ins, carb], dim=1))


class _LSTMNet(nn.Module):
    def __init__(self, ctx: int, hidden: int, zone_hidden: int) -> None:
        super().__init__()
        self.ctx = ctx
        self.lstm = nn.LSTM(3, hidden, batch_first=True)
        self.zone = _ZoneDoses(zone_hidden)
        self.head = nn.Sequential(nn.Linear(hidden + zone_hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, ZONE))

    def forward(self, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        """bg (B, ctx), ins/carb (B, ctx + ZONE) -> (B, ZONE) scaled delta."""
        c = self.ctx
        x = torch.stack([bg, ins[:, :c], carb[:, :c]], dim=2)
        _, (h, _) = self.lstm(x)
        z = self.zone(ins[:, c:], carb[:, c:])
        return self.head(torch.cat([h[-1], z], dim=1))


class LSTM(TorchForecaster):
    """Martinsson et al. 2019: one LSTM layer over 2 h of bg and past doses."""
    name = 'lstm'

    def __init__(self, ctx: int = 24, hidden: int = 128, zone_hidden: int = 64, **kw) -> None:
        self.hidden, self.zone_hidden = hidden, zone_hidden
        super().__init__(ctx, **kw)

    def build(self) -> nn.Module:
        return _LSTMNet(self.ctx, self.hidden, self.zone_hidden)


class _GatedBlock(nn.Module):
    def __init__(self, ch: int, skip: int, dilation: int, kernel: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.filt = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.gate = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.res = nn.Conv1d(ch, ch, 1)
        self.skip = nn.Conv1d(ch, skip, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, ch, T) -> residual (B, ch, T), skip (B, skip, T); causal."""
        p = F.pad(x, (self.pad, 0))
        h = torch.tanh(self.filt(p)) * torch.sigmoid(self.gate(p))
        return x + self.res(h), self.skip(h)


class _GluNetNet(nn.Module):
    def __init__(self, ctx: int, ch: int, skip: int, dilations: tuple[int, ...], kernel: int,
                 zone_hidden: int) -> None:
        super().__init__()
        self.ctx = ctx
        self.inp = nn.Conv1d(3, ch, 1)
        self.blocks = nn.ModuleList(_GatedBlock(ch, skip, d, kernel) for d in dilations)
        self.post = nn.Sequential(nn.ReLU(), nn.Conv1d(skip, skip, 1), nn.ReLU())
        self.zone = _ZoneDoses(zone_hidden)
        self.head = nn.Sequential(nn.Linear(skip + zone_hidden, skip), nn.GELU(),
                                  nn.Linear(skip, ZONE))

    def forward(self, bg: torch.Tensor, ins: torch.Tensor, carb: torch.Tensor) -> torch.Tensor:
        """bg (B, ctx), ins/carb (B, ctx + ZONE) -> (B, ZONE) scaled delta."""
        c = self.ctx
        x = self.inp(torch.stack([bg, ins[:, :c], carb[:, :c]], dim=1))
        s = torch.zeros_like(x[:, :1, -1:])
        for blk in self.blocks:
            x, k = blk(x)
            s = s + k[:, :, -1:]
        h = self.post(s)[:, :, 0]
        return self.head(torch.cat([h, self.zone(ins[:, c:], carb[:, c:])], dim=1))


class GluNet(TorchForecaster):
    """Li et al. 2020: gated residual dilated causal convolutions over 4 h of bg and doses."""
    name = 'glunet'

    def __init__(self, ctx: int = 48, ch: int = 32, skip: int = 64,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32), kernel: int = 2,
                 zone_hidden: int = 64, **kw) -> None:
        self.ch, self.skip, self.dilations = ch, skip, dilations
        self.kernel, self.zone_hidden = kernel, zone_hidden
        super().__init__(ctx, **kw)

    def build(self) -> nn.Module:
        return _GluNetNet(self.ctx, self.ch, self.skip, self.dilations, self.kernel,
                          self.zone_hidden)


MODELS = {
    'lstm': LSTM,
    'glunet': GluNet,
}
