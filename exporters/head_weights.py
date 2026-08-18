"""The BG head, exported beside the graph so a consumer can re-run it itself.

The ``.pte`` bakes its weights in and offers no seam: a consumer can run the
frozen graph but cannot adapt it. Cutting one output at ``slot_hidden`` (the
final-normed hidden state per masked slot) and shipping ``bg_head``'s own weights
alongside gives it one — the consumer reproduces ``head_raw`` from
``slot_hidden``, and can put an adapter in front of the head or a low-rank delta
on the head's own matrices without re-exporting anything.

This is a plain flat fp32 dump, not a checkpoint: no pickle, no framework, and the
descriptor names every tensor and its shape in file order. The consumer checks the
digest, reads the tensors in order, and gets bit-identical ``head_raw`` from the
same ``slot_hidden`` the graph emitted — a property worth checking at load, since
a silently mismatched head is a plausible-looking forecast rather than an error.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import torch

import config as cfg


# ``bg_head`` is nn.Sequential(Linear, SiLU, Linear, SiLU, Linear) — indices 0/2/4
# carry the weights and 1/3 the activation. Named l0/l1/l2 here so the file format
# does not inherit nn.Sequential's index numbering.
_LINEARS = (("l0", 0), ("l1", 2), ("l2", 4))


def head_tensors(model) -> "list[tuple[str, torch.Tensor]]":
    """The head tensors in FILE ORDER: the within-patch basis, then each Linear's
    weight and bias. Weights stay in torch's ``(out, in)`` row-major layout, so the
    consumer's matvec reads a row per output unit."""
    out: list[tuple[str, torch.Tensor]] = [
        ("step_basis", model.step_basis.detach().float().contiguous()),
    ]
    for name, idx in _LINEARS:
        lin = model.bg_head[idx]
        out.append((f"{name}.weight", lin.weight.detach().float().contiguous()))
        out.append((f"{name}.bias", lin.bias.detach().float().contiguous()))
    return out


def write_head_weights(model, path: str) -> dict[str, Any]:
    """Write the flat fp32 head file and return the descriptor's ``head`` block.

    The returned block carries the tensor order, the shapes and a sha256 over the
    exact bytes written; nothing else identifies the file, and a consumer that skips
    the digest can pair a head with the wrong graph and still decode.
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
