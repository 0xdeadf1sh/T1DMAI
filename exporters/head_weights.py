"""``bg_head`` dumped beside the graph, so a consumer reproduces ``head_raw`` from ``slot_hidden``.

Plain flat fp32, no pickle: the descriptor names every tensor and shape in file order.
A silently mismatched head is a plausible forecast, not an error — check the digest at load.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import torch

import config as cfg


# bg_head = Sequential(Linear, SiLU, Linear, SiLU, Linear): 0/2/4 weights, 1/3 activation.
# Renamed l0/l1/l2 so the file format does not inherit nn.Sequential's indices.
_LINEARS = (("l0", 0), ("l1", 2), ("l2", 4))


def head_tensors(model) -> "list[tuple[str, torch.Tensor]]":
    """FILE ORDER: within-patch basis, then each Linear's weight and bias.

    Weights stay torch ``(out, in)`` row-major — a row per output unit.
    """
    out: list[tuple[str, torch.Tensor]] = [
        ("step_basis", model.step_basis.detach().float().contiguous()),
    ]
    for name, idx in _LINEARS:
        lin = model.bg_head[idx]
        out.append((f"{name}.weight", lin.weight.detach().float().contiguous()))
        out.append((f"{name}.bias", lin.bias.detach().float().contiguous()))
    return out


def write_head_weights(model, path: str) -> dict[str, Any]:
    """Write the flat fp32 head file; returns the descriptor's ``head`` block.

    The sha256 over the written bytes is the only thing identifying the file: skip it and a
    head paired with the wrong graph still decodes.
    """
    tensors = head_tensors(model)
    blob = b"".join(t.numpy().astype("<f4").tobytes(order="C") for _n, t in tensors)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    return {
        "file": os.path.basename(path),
        "dtype": "fp32",
        "byte_order": "little",
        "layout": "flat C-contiguous tensors concatenated in the order listed below",
        "activation": "silu",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "d_model": int(cfg.D_MODEL),
        "hidden": int(cfg.BG_HEAD_HIDDEN),
        "step_basis_dim": int(cfg.BG_HEAD_STEP_BASIS_DIM),
        "out_dim": int(cfg.BG_HEAD_STEP_BASIS_DIM * (1 + 2 * cfg.N_SPREADS)),
        "tensors": [{"name": n, "shape": list(t.shape)} for n, t in tensors],
        "note": "head_raw[b,m,s,c] = sum_k step_basis[s,k] * l2(silu(l1(silu(l0(h)))))"
                "[b,m,k*(1+2*N_SPREADS)+c], h = slot_hidden[b,m]",
    }
