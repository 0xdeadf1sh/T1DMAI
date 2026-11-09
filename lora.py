"""Low-rank adapters on every nn.Linear of a model, merged back into plain weights on save."""
import math

import torch
import torch.nn.functional as F
from torch import nn


def _hook(m: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> torch.Tensor:
    a, b, s = getattr(m, 'lora_A'), getattr(m, 'lora_B'), getattr(m, 'lora_scale')
    return out + F.linear(F.linear(inp[0], a), b) * s


def attach(model: nn.Module, rank: int, alpha: float) -> int:
    """Freeze ``model``, add a rank-``rank`` adapter to each Linear; returns the adapter size."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        r = min(rank, m.in_features, m.out_features)
        w = m.weight
        a = nn.Parameter(torch.randn(r, m.in_features, device=w.device, dtype=w.dtype)
                         / math.sqrt(m.in_features))
        b = nn.Parameter(torch.zeros(m.out_features, r, device=w.device, dtype=w.dtype))
        m.register_parameter('lora_A', a)
        m.register_parameter('lora_B', b)
        setattr(m, 'lora_scale', alpha / r)
        m.register_forward_hook(_hook)
        n += a.numel() + b.numel()
    return n


def merged(model: nn.Module, sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``sd`` (a state dict of ``model``) with each adapter folded into its weight and dropped."""
    out = {k: v for k, v in sd.items() if not k.endswith(('.lora_A', '.lora_B'))}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and hasattr(m, 'lora_A'):
            p = f'{name}.' if name else ''
            a, b = sd[p + 'lora_A'].float(), sd[p + 'lora_B'].float()
            w = out[p + 'weight']
            out[p + 'weight'] = (w.float() + b @ a * getattr(m, 'lora_scale')).to(w.dtype)
    return out
