"""Chronos-Bolt (amazon/chronos-bolt-{tiny,mini}) in plain torch, zero-shot and fine-tuned.

Ported from chronos-forecasting ``chronos_bolt.py`` and transformers ``modeling_t5.py``
(Amazon.com, Inc.; The HuggingFace Team; Apache-2.0). Weights (Apache-2.0) via ``urllib``.
"""
from __future__ import annotations

import io
import json
import math
import os
import struct
import time
import urllib.request

import numpy as np
import torch
from torch import nn

from zoo.classical import Forecaster
from zoo.windows import ZONE, Windows

CTX = 288
HF_URL = 'https://huggingface.co/amazon/chronos-bolt-{size}/resolve/main/{file}'
CACHE_ROOT = os.path.expanduser('~/.cache/huggingface')
FILES = ('config.json', 'model.safetensors')
_ST_DTYPES = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16}


def weights_dir(size: str) -> str:
    return os.path.join(CACHE_ROOT, f'chronos-bolt-{size}')


def weights_present(size: str) -> bool:
    return all(os.path.isfile(os.path.join(weights_dir(size), f)) for f in FILES)


def download(size: str) -> str:
    """Fetch ``config.json`` and ``model.safetensors`` once; returns the directory."""
    d = weights_dir(size)
    os.makedirs(d, exist_ok=True)
    for f in FILES:
        path = os.path.join(d, f)
        if not os.path.isfile(path):
            tmp = path + '.part'
            urllib.request.urlretrieve(HF_URL.format(size=size, file=f), tmp)
            os.replace(tmp, path)
    return d


def read_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Header length (u64 LE), JSON header, then raw little-endian tensor bytes."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    out = {}
    for k, v in header.items():
        if k == '__metadata__':
            continue
        a, b = v['data_offsets']
        t = torch.frombuffer(bytearray(blob[a:b]), dtype=_ST_DTYPES[v['dtype']])
        out[k] = t.reshape(v['shape']).float()
    return out


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class ResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int, p: float) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(d_in, d_h)
        self.output_layer = nn.Linear(d_h, d_out)
        self.residual_layer = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.hidden_layer(x))
        return self.dropout(self.output_layer(h)) + self.residual_layer(x)


def relative_bucket(rel: torch.Tensor, bidirectional: bool, num_buckets: int,
                    max_distance: int) -> torch.Tensor:
    """T5 log-spaced relative-position bucket; ``rel`` is key minus query position."""
    buckets = torch.zeros_like(rel)
    if bidirectional:
        num_buckets //= 2
        buckets = buckets + (rel > 0).long() * num_buckets
        rel = rel.abs()
    else:
        rel = -torch.clamp(rel, max=0)
    max_exact = num_buckets // 2
    large = max_exact + (torch.log(rel.float().clamp(min=1) / max_exact)
                         / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
    large = torch.clamp(large, max=num_buckets - 1)
    return buckets + torch.where(rel < max_exact, rel, large)


class Attention(nn.Module):
    def __init__(self, cfg: dict, rel_bias: bool, bidirectional: bool) -> None:
        super().__init__()
        d, self.h, self.dk = cfg['d_model'], cfg['num_heads'], cfg['d_kv']
        inner = self.h * self.dk
        self.q = nn.Linear(d, inner, bias=False)
        self.k = nn.Linear(d, inner, bias=False)
        self.v = nn.Linear(d, inner, bias=False)
        self.o = nn.Linear(inner, d, bias=False)
        self.p = cfg['dropout_rate']
        self.bidirectional = bidirectional
        self.nb = cfg['relative_attention_num_buckets']
        self.md = cfg['relative_attention_max_distance']
        if rel_bias:
            self.relative_attention_bias = nn.Embedding(self.nb, self.h)

    def bias(self, lq: int, lk: int, device: torch.device) -> torch.Tensor:
        rel = torch.arange(lk, device=device)[None, :] - torch.arange(lq, device=device)[:, None]
        b = self.relative_attention_bias(relative_bucket(rel, self.bidirectional, self.nb, self.md))
        return b.permute(2, 0, 1).unsqueeze(0)

    def forward(self, x: torch.Tensor, kv: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, lq, _ = x.shape
        lk = kv.shape[1]
        q = self.q(x).view(B, lq, self.h, self.dk).transpose(1, 2)
        k = self.k(kv).view(B, lk, self.h, self.dk).transpose(1, 2)
        v = self.v(kv).view(B, lk, self.h, self.dk).transpose(1, 2)
        w = torch.softmax(q @ k.transpose(2, 3) + bias, dim=-1)
        w = nn.functional.dropout(w, self.p, self.training)
        return self.o((w @ v).transpose(1, 2).reshape(B, lq, self.h * self.dk))


class SelfLayer(nn.Module):
    def __init__(self, cfg: dict, rel_bias: bool, bidirectional: bool) -> None:
        super().__init__()
        self.SelfAttention = Attention(cfg, rel_bias, bidirectional)
        self.layer_norm = RMSNorm(cfg['d_model'], cfg['layer_norm_epsilon'])
        self.dropout = nn.Dropout(cfg['dropout_rate'])

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        h = self.layer_norm(x)
        return x + self.dropout(self.SelfAttention(h, h, bias))


class CrossLayer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.EncDecAttention = Attention(cfg, False, False)
        self.layer_norm = RMSNorm(cfg['d_model'], cfg['layer_norm_epsilon'])
        self.dropout = nn.Dropout(cfg['dropout_rate'])

    def forward(self, x: torch.Tensor, enc: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.EncDecAttention(self.layer_norm(x), enc, bias))


class DenseReluDense(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.wi = nn.Linear(cfg['d_model'], cfg['d_ff'], bias=False)
        self.wo = nn.Linear(cfg['d_ff'], cfg['d_model'], bias=False)
        self.dropout = nn.Dropout(cfg['dropout_rate'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.dropout(torch.relu(self.wi(x))))


class FFLayer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.DenseReluDense = DenseReluDense(cfg)
        self.layer_norm = RMSNorm(cfg['d_model'], cfg['layer_norm_epsilon'])
        self.dropout = nn.Dropout(cfg['dropout_rate'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.DenseReluDense(self.layer_norm(x)))


class Block(nn.Module):
    def __init__(self, cfg: dict, rel_bias: bool, decoder: bool) -> None:
        super().__init__()
        mods: list[nn.Module] = [SelfLayer(cfg, rel_bias, not decoder)]
        if decoder:
            mods.append(CrossLayer(cfg))
        mods.append(FFLayer(cfg))
        self.layer = nn.ModuleList(mods)


class Stack(nn.Module):
    def __init__(self, cfg: dict, n_layers: int, decoder: bool) -> None:
        super().__init__()
        self.decoder = decoder
        self.block = nn.ModuleList([Block(cfg, i == 0, decoder) for i in range(n_layers)])
        self.final_layer_norm = RMSNorm(cfg['d_model'], cfg['layer_norm_epsilon'])
        self.dropout = nn.Dropout(cfg['dropout_rate'])

    def forward(self, x: torch.Tensor, self_mask: torch.Tensor,
                enc: torch.Tensor | None = None, enc_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` (B, L, d); masks are additive, broadcastable to (B, H, L, Lk)."""
        L = x.shape[1]
        bias = self.block[0].layer[0].SelfAttention.bias(L, L, x.device) + self_mask
        x = self.dropout(x)
        for blk in self.block:
            x = blk.layer[0](x, bias)
            if self.decoder:
                x = blk.layer[1](x, enc, enc_mask)
            x = blk.layer[-1](x)
        return self.dropout(self.final_layer_norm(x))


class ChronosBolt(nn.Module):
    """Context (B, L), NaN where missing -> quantiles (B, Q, H) on the input scale.

    ``scaled=True`` also returns the pre-inverse quantiles and the instance ``(loc, scale)``.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        cc = cfg['chronos_config']
        self.patch_size = cc['input_patch_size']
        assert cc['input_patch_stride'] == self.patch_size and cc['use_reg_token']
        self.context_length = cc['context_length']
        self.prediction_length = cc['prediction_length']
        self.register_buffer('quantiles', torch.tensor(cc['quantiles']), persistent=False)
        d, p = cfg['d_model'], cfg['dropout_rate']
        self.shared = nn.Embedding(2, d)
        self.input_patch_embedding = ResidualBlock(2 * self.patch_size, cfg['d_ff'], d, p)
        self.encoder = Stack(cfg, cfg['num_layers'], False)
        self.decoder = Stack(cfg, cfg['num_decoder_layers'], True)
        nq = len(cc['quantiles'])
        self.output_patch_embedding = ResidualBlock(d, cfg['d_ff'], nq * self.prediction_length, p)
        self.reg_id = cfg['reg_token_id']
        self.start_id = cfg['decoder_start_token_id']

    @staticmethod
    def scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, L) -> mask (B, L), loc (B, 1), scale (B, 1); NaN-aware mean and std."""
        m = (~torch.isnan(x)).to(x.dtype)
        x0 = torch.nan_to_num(x, nan=0.0)
        n = m.sum(-1, keepdim=True)
        loc = torch.nan_to_num((x0 * m).sum(-1, keepdim=True) / n, nan=0.0)
        var = (((x0 - loc) * m) ** 2).sum(-1, keepdim=True) / n
        sc = torch.nan_to_num(var.sqrt(), nan=1.0)
        return m, loc, torch.where(sc == 0, torch.full_like(sc, 1e-5), sc)

    def forward(self, context: torch.Tensor, scaled: bool = False):
        x = context.float()[:, -self.context_length:]
        B, L = x.shape
        m, loc, sc = self.scale(x)
        x = torch.nan_to_num((x - loc) / sc, nan=0.0)
        if L % self.patch_size:
            pad = self.patch_size - L % self.patch_size
            x = torch.cat([x.new_zeros(B, pad), x], dim=1)
            m = torch.cat([m.new_zeros(B, pad), m], dim=1)
        xp = x.view(B, -1, self.patch_size)
        mp = m.view(B, -1, self.patch_size)
        tokens = self.input_patch_embedding(torch.cat([xp * mp, mp], dim=-1))
        tokens = torch.cat([tokens, self.shared.weight[self.reg_id].expand(B, 1, -1)], dim=1)
        keep = torch.cat([mp.sum(-1) > 0, torch.ones(B, 1, dtype=torch.bool, device=x.device)], 1)
        key_mask = torch.where(keep, 0.0, torch.finfo(tokens.dtype).min)[:, None, None, :]
        enc = self.encoder(tokens, key_mask)
        start = self.shared.weight[self.start_id].expand(B, 1, -1)
        dec = self.decoder(start, start.new_zeros(1, 1, 1, 1), enc, key_mask)
        q = self.output_patch_embedding(dec).view(B, -1, self.prediction_length)
        out = q * sc[:, :, None] + loc[:, :, None]
        return (out, q, loc, sc) if scaled else out


def load_pretrained(size: str) -> ChronosBolt:
    d = download(size)
    with open(os.path.join(d, 'config.json')) as f:
        cfg = json.load(f)
    net = ChronosBolt(cfg)
    missing, unexpected = net.load_state_dict(
        read_safetensors(os.path.join(d, 'model.safetensors')), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return net.eval()


def pinball(q: torch.Tensor, target: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Chronos-Bolt's loss on scaled q (B, Q, H) and target (B, h <= H), NaN unmeasured."""
    q = q[..., :target.shape[-1]]
    ok = torch.isfinite(target)[:, None, :].float()
    t = torch.nan_to_num(target)[:, None, :]
    loss = 2 * torch.abs((t - q) * ((t <= q).float() - levels.view(1, -1, 1))) * ok
    return loss.mean(dim=-2).sum(dim=-1).mean()


class ChronosForecaster(Forecaster):
    """CGM-only Chronos-Bolt; the point forecast is the 0.5 quantile of the first ZONE steps."""

    def __init__(self, size: str, finetune: bool, epochs: int = 1, batch: int = 256,
                 lr: float = 1e-5, weight_decay: float = 0.0, val_frac: float = 0.05,
                 seed: int = 0) -> None:
        self.size, self.finetune = size, finetune
        self.name = f"chronos_bolt_{size}_{'ft' if finetune else 'zs'}"
        self.ctx = CTX
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.weight_decay, self.val_frac, self.seed = weight_decay, val_frac, seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._net: ChronosBolt | None = None

    @property
    def net(self) -> ChronosBolt:
        if self._net is None:
            self._net = load_pretrained(self.size).to(self.device)
        return self._net

    @property
    def median_idx(self) -> int:
        return int(torch.argmin((self.net.quantiles - 0.5).abs()))

    def _tensors(self, w: Windows, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        bg = torch.from_numpy(w.bg[idx].astype(np.float32)).to(self.device)
        return bg, torch.from_numpy(w.target[idx].astype(np.float32)).to(self.device)

    def _val(self, w: Windows, idx: np.ndarray) -> tuple[float, float]:
        self.net.eval()
        tot = sq = n = 0.0
        with torch.no_grad():
            for s in range(0, len(idx), self.batch * 4):
                bg, y = self._tensors(w, idx[s:s + self.batch * 4])
                out, q, loc, sc = self.net(bg, scaled=True)
                tot += float(pinball(q, (y - loc) / sc, self.net.quantiles)) * len(bg)
                d = (out[:, self.median_idx, :ZONE] - y)[torch.isfinite(y)]
                sq += float((d ** 2).sum())
                n += float(d.numel())
        return tot / max(len(idx), 1), (sq / max(n, 1.0)) ** 0.5

    def fit(self, w: Windows) -> None:
        if not self.finetune:
            return
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(w))
        n_val = max(int(len(w) * self.val_frac), 1)
        val, tr = order[:n_val], order[n_val:]
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = max(self.epochs * ((len(tr) + self.batch - 1) // self.batch), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: max(1.0 - i / steps, 0.0))
        best, rmse = self._val(w, val)
        best_state = None
        print(f'  zero-shot val pinball {best:.4f}  median rmse {rmse:.2f} mg/dL', flush=True)
        for ep in range(self.epochs):
            self.net.train()
            t = time.time()
            tr = rng.permutation(tr)
            for s in range(0, len(tr), self.batch):
                bg, y = self._tensors(w, tr[s:s + self.batch])
                _, q, loc, sc = self.net(bg, scaled=True)
                loss = pinball(q, (y - loc) / sc, self.net.quantiles)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()
            v, rmse = self._val(w, val)
            print(f'  epoch {ep + 1}/{self.epochs}  val pinball {v:.4f}  median rmse '
                  f'{rmse:.2f} mg/dL  ({time.time() - t:.0f} s)', flush=True)
            if v < best:
                best = v
                best_state = {k: x.detach().clone() for k, x in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

    def predict(self, w: Windows) -> np.ndarray:
        """(N, ctx) bg -> (N, ZONE) mg/dL median."""
        assert w.bg.shape[1] == self.ctx
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(w), self.batch * 4):
                bg, _ = self._tensors(w, np.arange(s, min(s + self.batch * 4, len(w))))
                out.append(self.net(bg)[:, self.median_idx, :ZONE].cpu().numpy())
        return np.concatenate(out).astype(np.float64) if out else np.empty((0, ZONE))

    def state(self) -> dict[str, np.ndarray]:
        if not self.finetune:
            return {}
        buf = io.BytesIO()
        torch.save(self.net.state_dict(), buf)
        return {'state_dict': np.frombuffer(buf.getvalue(), dtype=np.uint8)}

    def load(self, state: dict[str, np.ndarray]) -> None:
        if 'state_dict' in state:
            sd = torch.load(io.BytesIO(state['state_dict'].tobytes()), map_location='cpu')
            self.net.load_state_dict(sd)


MODELS = {
    f'chronos_bolt_{size}_{kind}': (lambda size=size, kind=kind:
                                    ChronosForecaster(size, finetune=kind == 'ft'))
    for size in ('tiny', 'mini') for kind in ('zs', 'ft')
}
