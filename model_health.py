#!/usr/bin/env python3
"""Capacity audit of a trained T1DMAI checkpoint, keyed to ``resize_model.py``'s knobs.

Per knob — ``D_MODEL``, ``N_LAYERS``, ``N_HEADS``, ``FFN_DIM``, ``BG_HEAD_HIDDEN``,
``PATCH_SIZE`` and the context window — the report says whether that part of the network
is under-used (shrinkable) or saturated (a candidate to grow), lists the evidence behind
the call, and prints the ``resize_model.py`` command each verdict implies.

Every dimension is read from the checkpoint's tensor shapes, never from ``config.py``.

Weight-only evidence (always):
  * optimizer staleness — Muon ``momentum_buffer`` / AdamW ``exp_avg_sq`` mapped back to
    parameter names by replaying ``train._build_optimizers``' two-group partition;
  * drift from init against ``model._init_weights``'s analytic std;
  * spectral rank utilization per weight matrix;
  * per-(layer, head) value×output pathway strength, per-(layer, unit) FFN strength,
    per-unit BG-head and time-head strength.

Activation evidence (``--data N``) — N cached windows, built under the checkpoint's own
``masked_channel_policy``:
  * dead / hot units in every FFN, BG-head and time-head layer; residual-dim variance;
  * per-(layer, head) attention entropy, self-mass, masked-column mass, lag reach;
  * per-block residual write size, in/out cosine, adjacent-layer CKA;
  * the spline's per-step departure from the own-patch state;
  * one-at-a-time ablation of every head, attention sublayer, FFN sublayer and whole
    block, and a context-truncation ladder, each scored by Δ pinball on the same windows.

Usage::

    python model_health.py                       # audit checkpoints/t1dmai_best.pt
    python model_health.py --checkpoint X.pt     # a specific checkpoint
    python model_health.py --ema                 # audit the EMA shadow weights
    python model_health.py --data 128            # add the activation + ablation pass
    python model_health.py --data 128 --device cuda
    python model_health.py --json out.json       # also dump machine-readable findings
    python model_health.py --top 40              # more rows in the per-param table

Verdicts are heuristic capacity signals. SHRINK means the evidence says the width is
under-used at this checkpoint, not that quality is invariant to the cut.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import N_INPUT_FEATURES, QUANTILE_LEVELS
from data import (
    BG_MASKED_FEAT, MASKED_CHANNEL_POLICY_BLIND, checkpoint_masked_channel_policy,
)
from resize_model import VALID_HEAD_DIMS


# model._init_weights, mirrored: base_std = 0.02 * sqrt(512 / D_MODEL); residual writes
# (attn.w_o, ffn.w2) use base_std / sqrt(2 * N_LAYERS); the BG-head and time-head final
# layers use their INIT_SCALE. The model exposes none of these, so the drift metric
# reconstructs them here.
INIT_BASE_STD_ANCHOR = 0.02
INIT_BASE_STD_ANCHOR_DMODEL = 512.0
RESIDUAL_WRITES_PER_BLOCK = 2
DEFAULT_BG_HEAD_INIT_SCALE = 1e-2
DEFAULT_TIME_PROBE_INIT_SCALE = 1e-2
STEP_MINUTES = 5

# Thresholds. Every one is printed in the report legend.
RANK_UTIL_SHRINK = 0.50      # mean eff_rank / min(in,out) below this ⇒ width over-provisioned
RANK_UTIL_GROW = 0.85        # above this (flat spectrum) ⇒ saturated
ACTIVE_FRAC_SHRINK = 0.65    # live-unit fraction below this ⇒ shrink the width
ACTIVE_FRAC_GROW = 0.97      # almost every unit live ⇒ candidate to grow
DEAD_UNIT_REL = 0.05         # unit below 5% of its layer's median strength ⇒ dead
HOT_UNIT_REL = 10.0          # unit above 10× its layer's median activation ⇒ hot
HEAD_WEAK_REL = 0.25         # head below 25% of its layer's median V·O strength ⇒ weak
HEAD_DOMINANT_REL = 3.0      # head above 3× its layer's median activation RMS ⇒ dominant
HEAD_ENTROPY_UNIFORM = 0.90  # normalized attention entropy above this ⇒ near-uniform
HEAD_SELF_MASS = 0.80        # attention mass on the query's own patch above this ⇒ self
HEAD_ABLATE_WEAK_PCT = 0.5   # Δ pinball below this % when the head is zeroed ⇒ removable
BLOCK_ABLATE_REMOVABLE_PCT = 1.0   # Δ pinball below this % when the block is skipped
BLOCK_COS_IDENTITY = 0.98    # cos(x_in, x_out) above this ⇒ block near identity
CKA_REDUNDANT = 0.98         # adjacent-layer linear CKA above this ⇒ redundant depth
STALE_OPT_REL = 0.01         # optimizer activity below 1% of the median ⇒ stale
STALE_OPT_ABS = 1e-10        # absolute optimizer-activity floor ⇒ dead
DRIFT_NEAR_INIT = 0.08       # |std/init_std − 1| below this ⇒ near-init
TAIL_REL = 0.01              # singular values below 1% of σ_max count as spectral tail
CTX_TRUNC_TOL_PCT = 1.0      # Δ pinball below this % at a truncation ⇒ context beyond it unused
CTX_FAR_DENSITY_REL = 0.01   # far-lag attention per key below 1% of the near-lag density ⇒ unused
RESID_TOP_K = 8              # dims counted in the residual-variance concentration figure
CONTEXT_LADDER_HOURS = (6, 12, 24, 48, 84, 126)
WIDTH_LADDER_FRACS = (0.125, 0.25, 0.5, 0.75)
LADDER_TOL_PCT = 1.0         # Δ pinball below this % at a ladder rung ⇒ the cut is free
LADDER_HURT_PCT = 5.0        # Δ pinball above this % at the 3/4 rung ⇒ width is binding
LAG_BUCKET_HOURS = (1, 3, 6, 12, 24, 48, 84, 168)

LEGEND = (
    ('rank-util', f'eff_rank/min(in,out); SHRINK < {RANK_UTIL_SHRINK}, GROW > {RANK_UTIL_GROW}'),
    ('live unit', f'strength ≥ {DEAD_UNIT_REL:.0%} of the layer median; SHRINK < {ACTIVE_FRAC_SHRINK:.0%} live, GROW > {ACTIVE_FRAC_GROW:.0%}'),
    ('hot unit', f'activation > {HOT_UNIT_REL:.0f}× the layer median'),
    ('head WEAK', f'V·O strength < {HEAD_WEAK_REL:.0%} of the layer median, or ablation Δ < {HEAD_ABLATE_WEAK_PCT}%'),
    ('head DOMINANT', f'activation RMS > {HEAD_DOMINANT_REL:.0f}× the layer median'),
    ('head UNIFORM / SELF', f'normalized entropy > {HEAD_ENTROPY_UNIFORM}; self-mass > {HEAD_SELF_MASS}'),
    ('block removable', f'skip Δ < {BLOCK_ABLATE_REMOVABLE_PCT}%; near-identity = cos(in,out) > {BLOCK_COS_IDENTITY} with CKA(prev) > {CKA_REDUNDANT} (informational)'),
    ('stale', f'optimizer activity < {STALE_OPT_REL:.0%} of the median or < {STALE_OPT_ABS:g}'),
    ('near-init', f'|std/init_std − 1| < {DRIFT_NEAR_INIT}'),
    ('context', f'truncation Δ < {CTX_TRUNC_TOL_PCT}% ⇒ patches beyond it unused'),
    ('width ladder', f'Δ < {LADDER_TOL_PCT}% at half width ⇒ SHRINK; Δ > {LADDER_HURT_PCT}% at 3/4 width ⇒ binding'),
)


@dataclass
class Arch:
    """Architecture dimensions recovered from the model state-dict shapes."""
    d_model: int
    patch_dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    ffn_dim: int
    bg_head_hidden: int
    patch_size: int
    n_spreads: int
    head_out_width: int
    bg_head_linears: list[str]
    time_head_linears: list[str]
    time_hidden: int
    time_bins: int

    @property
    def patch_hours(self) -> float:
        return self.patch_size * STEP_MINUTES / 60.0

    def view(self) -> dict[str, int]:
        return {
            'D_MODEL': self.d_model, 'N_LAYERS': self.n_layers,
            'N_HEADS': self.n_heads, 'HEAD_DIM': self.head_dim,
            'FFN_DIM': self.ffn_dim, 'BG_HEAD_HIDDEN': self.bg_head_hidden,
            'PATCH_SIZE': self.patch_size,
        }


def derive_arch(sd: dict[str, torch.Tensor]) -> Arch:
    """Every architecture dimension recovered from state-dict shapes, cross-checked."""
    d_model, patch_dim = sd['patch_embed.weight'].shape
    block_ids = sorted({int(k.split('.')[1]) for k in sd if k.startswith('blocks.')})
    n_layers = len(block_ids)
    assert block_ids == list(range(n_layers)), f"non-contiguous block ids: {block_ids}"

    head_dim = sd['blocks.0.attn.q_norm.weight'].shape[0]
    n_heads = d_model // head_dim
    assert head_dim * n_heads == d_model, (
        f"HEAD_DIM({head_dim}) * N_HEADS({n_heads}) != D_MODEL({d_model})"
    )
    ffn_dim = sd['blocks.0.ffn.w1.weight'].shape[0]

    def linears(prefix: str) -> list[str]:
        return sorted((k for k in sd if k.startswith(prefix) and k.endswith('.weight')),
                      key=lambda k: int(k.split('.')[1]))

    bg_linears = linears('bg_head.')
    assert len(bg_linears) == 3, f"bg_head should be 3 Linear layers, got {bg_linears}"
    bg_head_hidden = sd[bg_linears[0]].shape[0]
    head_out_width = sd[bg_linears[-1]].shape[0]
    time_linears = linears('time_head.')
    time_hidden = sd[time_linears[0]].shape[0] if time_linears else 0
    time_bins = sd[time_linears[-1]].shape[0] if time_linears else 0

    assert patch_dim % N_INPUT_FEATURES == 0, (
        f"PATCH_DIM={patch_dim} inconsistent with the frozen "
        f"{N_INPUT_FEATURES}-feature layout"
    )
    patch_size = patch_dim // N_INPUT_FEATURES
    assert head_out_width % 2 == 1, (
        f"head out width {head_out_width} should be 1 + 2*N_SPREADS (odd)")
    n_spreads = (head_out_width - 1) // 2

    return Arch(
        d_model=d_model, patch_dim=patch_dim, n_layers=n_layers, n_heads=n_heads,
        head_dim=head_dim, ffn_dim=ffn_dim, bg_head_hidden=bg_head_hidden,
        patch_size=patch_size, n_spreads=n_spreads, head_out_width=head_out_width,
        bg_head_linears=bg_linears, time_head_linears=time_linears,
        time_hidden=time_hidden, time_bins=time_bins,
    )


def find_best_checkpoint(ckpt_dir: Path) -> Path:
    """Prefer t1dmai_best.pt, then the latest step snapshot."""
    p = ckpt_dir / 't1dmai_best.pt'
    if p.exists():
        return p
    steps = sorted(ckpt_dir.glob('t1dmai_step_*.pt'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    if steps:
        return steps[-1]
    raise FileNotFoundError(f"no T1DMAI checkpoint found in {ckpt_dir}")


# ----------------------------------------------------------------------------- weights
def map_optimizer_activity(
    sd: dict[str, torch.Tensor], ckpt: dict[str, Any], arch: Arch,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Parameter name → optimizer-state gradient-activity scalars, plus any mapping notes.

    Replays ``train._build_optimizers``: Muon owns every ``ndim >= 2`` tensor in TWO
    groups — group 0 the normalized matrices, group 1 the output projections
    (``bg_head[-1].weight``, ``time_head[-1].weight``); AdamW group 0 owns the 1-D
    tensors (its group 1 is the Kendall-Gal pair, outside the model). PyTorch numbers
    ``state`` keys across groups in group order, in each group's parameter order, which
    is ``named_parameters()`` order == state-dict order (the model has no buffers).

    ``activity`` is the RMS of the momentum buffer (Muon) or of ``exp_avg_sq`` (AdamW):
    the gradient energy that has reached the parameter. ``snr`` (AdamW only) is
    ``|exp_avg| / sqrt(exp_avg_sq)`` — near 0 when the gradient is pure noise.
    """
    names = list(sd.keys())
    output_names = {arch.bg_head_linears[-1]}
    if arch.time_head_linears:
        output_names.add(arch.time_head_linears[-1])
    muon_groups = [[n for n in names if sd[n].ndim >= 2 and n not in output_names],
                   [n for n in names if sd[n].ndim >= 2 and n in output_names]]
    adam_groups = [[n for n in names if sd[n].ndim < 2]]
    out: dict[str, dict[str, float]] = {}
    notes: list[str] = []

    def replay(opt_sd: dict[str, Any], groups: list[list[str]], label: str) -> dict[str, int]:
        pg = opt_sd['param_groups']
        got = [len(g['params']) for g in pg]
        expect = [len(g) for g in groups]
        if got[:len(expect)] != expect:
            notes.append(f"{label} param_groups sizes {got} != replayed partition {expect}; "
                         "optimizer activity not mapped")
            return {}
        mapping: dict[str, int] = {}
        for g, group_names in zip(pg, groups):
            for idx, nm in zip(g['params'], group_names):
                mapping[nm] = idx
        return mapping

    muon = ckpt.get('muon_optimizer_state_dict')
    if muon is None:
        notes.append('no muon_optimizer_state_dict in checkpoint')
    else:
        for nm, idx in replay(muon, muon_groups, 'Muon').items():
            buf = muon['state'].get(idx, {}).get('momentum_buffer')
            if buf is not None:
                b = buf.float()
                out[nm] = {'opt': 'muon', 'activity': float(b.pow(2).mean().sqrt()),
                           'first_moment': float(b.abs().mean()),
                           'snr': float('nan'), 'step': float('nan')}
    adam = ckpt.get('adam_optimizer_state_dict')
    if adam is None:
        notes.append('no adam_optimizer_state_dict in checkpoint')
    else:
        for nm, idx in replay(adam, adam_groups, 'AdamW').items():
            st = adam['state'].get(idx, {})
            sq = st.get('exp_avg_sq')
            if sq is None:
                continue
            ea = st.get('exp_avg')
            sq = sq.float()
            snr = float((ea.float().abs() / sq.sqrt().clamp_min(1e-30)).mean()) if ea is not None else float('nan')
            out[nm] = {'opt': 'adam', 'activity': float(sq.mean().sqrt()),
                       'first_moment': float(ea.float().abs().mean()) if ea is not None else float('nan'),
                       'snr': snr, 'step': float(st.get('step', float('nan')))}
    return out, notes


def init_std_for(name: str, arch: Arch, bg_head_init_scale: float,
                 time_init_scale: float) -> tuple[str, float]:
    """``(kind, expected_init_std)`` mirroring ``model._init_weights``."""
    base_std = INIT_BASE_STD_ANCHOR * math.sqrt(INIT_BASE_STD_ANCHOR_DMODEL / arch.d_model)
    residual_std = base_std / math.sqrt(RESIDUAL_WRITES_PER_BLOCK * arch.n_layers)
    if name.endswith('.bias'):
        return 'bias', 0.0
    if name.endswith('norm.weight') or name.endswith('norm1.weight') or name.endswith('norm2.weight'):
        return 'norm', 1.0
    if name == arch.bg_head_linears[-1]:
        return 'head_final', bg_head_init_scale
    if arch.time_head_linears and name == arch.time_head_linears[-1]:
        return 'head_final', time_init_scale
    if name.endswith('attn.w_o.weight') or name.endswith('ffn.w2.weight'):
        return 'residual', residual_std
    if name.endswith('.weight'):
        return 'linear', base_std
    return 'other', float('nan')


def spectral(W: np.ndarray) -> dict[str, float]:
    """stable_rank, entropy eff_rank, util = eff_rank/min(out,in), tail_frac, sigma_max."""
    s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return {'stable_rank': 0.0, 'eff_rank': 0.0, 'util': 0.0, 'tail_frac': 1.0, 'sigma_max': 0.0}
    smax = float(s[0])
    stable = float((s ** 2).sum() / smax ** 2)
    p = s / s.sum()
    eff = float(np.exp(-(p * np.log(p)).sum()))
    return {'stable_rank': stable, 'eff_rank': eff, 'util': eff / min(W.shape),
            'tail_frac': float((s < TAIL_REL * smax).mean()), 'sigma_max': smax}


def col_norms(W: np.ndarray) -> np.ndarray:
    return np.linalg.norm(W, axis=0)


def row_norms(W: np.ndarray) -> np.ndarray:
    return np.linalg.norm(W, axis=1)


def live_mask(strength: np.ndarray, rel: float = DEAD_UNIT_REL) -> np.ndarray:
    """Units clearing ``rel`` × the median strength (fraction of max when the median is 0)."""
    med = float(np.median(strength))
    ref = med if med > 0 else (float(strength.max()) or 1.0)
    return strength >= rel * ref


def hot_mask(strength: np.ndarray, rel: float = HOT_UNIT_REL) -> np.ndarray:
    """Units above ``rel`` × the median of the LIVE units (a bimodal layer's raw median
    sits in the gap between its dead and live clusters)."""
    live = strength[live_mask(strength)]
    med = float(np.median(live)) if live.size else 0.0
    return strength > rel * med if med > 0 else np.zeros_like(strength, dtype=bool)


def top_share(values: np.ndarray, k: int) -> float:
    """Share of the total carried by the ``k`` largest entries."""
    v = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    tot = v.sum()
    return float(v[:k].sum() / tot) if tot > 0 else float('nan')


@dataclass
class ParamRow:
    name: str
    shape: tuple[int, ...]
    numel: int
    kind: str
    std: float
    drift: float
    near_init: bool
    opt: str
    opt_activity: float
    opt_rel: float
    snr: float
    stale: bool


def analyze_params(
    sd: dict[str, torch.Tensor], arch: Arch, opt: dict[str, dict[str, float]],
    bg_head_init_scale: float, time_init_scale: float, weights_only_ema: bool,
) -> list[ParamRow]:
    """Per-parameter drift + optimizer staleness rows."""
    rows: list[ParamRow] = []
    med_act = {'muon': [], 'adam': []}
    for info in opt.values():
        med_act[info['opt']].append(info['activity'])
    med = {k: (float(np.median(v)) if v else 0.0) for k, v in med_act.items()}

    for nm, t in sd.items():
        w = t.detach().cpu().float().numpy()
        kind, istd = init_std_for(nm, arch, bg_head_init_scale, time_init_scale)
        std = float(w.std())
        if kind in ('linear', 'residual', 'head_final') and istd > 0:
            drift = std / istd
            near = abs(drift - 1.0) < DRIFT_NEAR_INIT
        elif kind == 'norm':
            drift = float(np.abs(w - 1.0).mean())
            near = drift < DRIFT_NEAR_INIT
        elif kind == 'bias':
            drift = std
            near = std < 1e-6
        else:
            drift, near = float('nan'), False

        info = opt.get(nm)
        if info and not weights_only_ema:
            o, act, snr = info['opt'], info['activity'], info['snr']
            rel = act / med[o] if med[o] > 0 else float('inf')
            stale = (act < STALE_OPT_ABS) or (rel < STALE_OPT_REL)
            # near-init means UNTRAINED only when the optimizer also shows little
            # gradient energy: a high-activity tensor near init rotated in place.
            near = near and rel < 0.5
        else:
            o, act, rel, snr, stale = '-', float('nan'), float('nan'), float('nan'), False
        rows.append(ParamRow(nm, tuple(w.shape), int(w.size), kind, std, drift, near,
                             o, act, rel, snr, stale))
    return rows


# ----------------------------------------------------------------------- per-unit rows
@dataclass
class HeadRow:
    layer: int
    head: int
    w_strength: float          # ‖W_v[h]‖·‖W_o[:,h]‖
    w_rel: float               # vs the layer median
    act_rms: float = float('nan')
    act_rel: float = float('nan')
    entropy: float = float('nan')     # normalized, masked query rows
    self_mass: float = float('nan')
    masked_mass: float = float('nan')  # mass on masked (announced-plan) columns
    reach_p50_h: float = float('nan')
    reach_p90_h: float = float('nan')
    ablate_pct: float = float('nan')
    flags: list[str] = field(default_factory=list)


@dataclass
class LayerRow:
    layer: int
    util_attn: float
    util_ffn: float
    write_attn: float          # w_o std / init std
    write_ffn: float           # w2 std / init std
    ffn_live_w: int
    ffn_hot_w: int
    ffn_top8_share_w: float
    ffn_live_a: int | None = None
    ffn_hot_a: int | None = None
    attn_rel: float = float('nan')     # ‖attn_out‖/‖x_in‖
    ffn_rel: float = float('nan')      # ‖ffn_out‖/‖x_mid‖
    cos_io: float = float('nan')       # cos(x_in, x_out)
    cka_prev: float = float('nan')
    cka_final: float = float('nan')
    tok_outlier: float = float('nan')  # max token norm / median token norm at block out
    ablate_attn_pct: float = float('nan')
    ablate_ffn_pct: float = float('nan')
    ablate_block_pct: float = float('nan')
    flags: list[str] = field(default_factory=list)


def head_rows_from_weights(np_sd: dict[str, np.ndarray], arch: Arch) -> list[HeadRow]:
    rows: list[HeadRow] = []
    hd = arch.head_dim
    for i in range(arch.n_layers):
        wv, wo = np_sd[f'blocks.{i}.attn.w_v.weight'], np_sd[f'blocks.{i}.attn.w_o.weight']
        s = np.array([np.linalg.norm(wv[h * hd:(h + 1) * hd, :]) * np.linalg.norm(wo[:, h * hd:(h + 1) * hd])
                      for h in range(arch.n_heads)])
        med = float(np.median(s)) or 1.0
        for h in range(arch.n_heads):
            rows.append(HeadRow(i, h, float(s[h]), float(s[h] / med)))
    return rows


def ffn_strength(np_sd: dict[str, np.ndarray], i: int) -> np.ndarray:
    w1, w3, w2 = (np_sd[f'blocks.{i}.ffn.w1.weight'], np_sd[f'blocks.{i}.ffn.w3.weight'],
                  np_sd[f'blocks.{i}.ffn.w2.weight'])
    return 0.5 * (row_norms(w1) + row_norms(w3)) * col_norms(w2)


# ---------------------------------------------------------------------------- verdicts
@dataclass
class Verdict:
    knob: str
    current: str
    verdict: str           # SHRINK / KEEP / GROW / NOTE
    suggest: str           # resize_model.py flags, or a prose note, or ''
    evidence: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def _suggest_heads(arch: Arch, grow: bool) -> str:
    d = arch.d_model
    valid = sorted(h for h in range(1, d + 1) if d % h == 0 and (d // h) in VALID_HEAD_DIMS)
    cur = arch.n_heads
    if grow:
        bigger = [h for h in valid if h > cur]
        return f'--heads {bigger[0]}' if bigger else ''
    smaller = [h for h in valid if h < cur]
    return f'--heads {smaller[-1]}' if smaller else ''


def _suggest_dmodel(arch: Arch, new_d: int) -> str:
    """A valid ``--d-model`` flag string for a target width, or '' if none exists."""
    if new_d <= 0:
        return ''
    if new_d % arch.n_heads == 0 and (new_d // arch.n_heads) in VALID_HEAD_DIMS:
        return f'--d-model {new_d}'
    if new_d % arch.head_dim == 0 and new_d // arch.head_dim >= 1:
        return f'--d-model {new_d} --heads {new_d // arch.head_dim}'
    for h in range(1, new_d + 1):
        if new_d % h == 0 and (new_d // h) in VALID_HEAD_DIMS:
            return f'--d-model {new_d} --heads {h}'
    return ''


def _ladder_line(rows: list[dict[str, float]], width: int) -> str:
    return ', '.join(f"{r['k']}/{width} {r['delta_pct']:+.2f}%" for r in rows)


def _ladder_verdict(rows: list[dict[str, float]], weight_verdict: str) -> tuple[str, str]:
    """Width verdict from a keep-top-k ladder, falling back to the weight-only call.

    Returns ``(verdict, reason)``. The half-width rung decides SHRINK; the 3/4 rung
    decides whether the width is binding (needed for GROW to stand).
    """
    if not rows:
        return weight_verdict, ''
    by_frac = {round(r['frac'], 3): r['delta_pct'] for r in rows}
    half = by_frac.get(0.5)
    tq = by_frac.get(0.75)
    if half is not None and half < LADDER_TOL_PCT:
        return 'SHRINK', f'half width costs {half:+.2f}% (< {LADDER_TOL_PCT}%)'
    if tq is not None and tq < LADDER_TOL_PCT:
        return 'KEEP', f'3/4 width costs {tq:+.2f}% (< {LADDER_TOL_PCT}%): not binding, not free at half'
    if weight_verdict == 'GROW' and tq is not None and tq > LADDER_HURT_PCT:
        return 'GROW', f'3/4 width costs {tq:+.2f}% (> {LADDER_HURT_PCT}%): binding'
    if weight_verdict == 'SHRINK':
        return 'KEEP', f'half width costs {half:+.2f}%: the rank/liveness signal is not borne out'
    return 'KEEP', ''


def _pct(x: float) -> str:
    return f'{x:+.2f}%' if not math.isnan(x) else '   -  '


def _f(x: float, fmt: str = '.2f') -> str:
    return format(x, fmt) if x is not None and not math.isnan(x) else '-'


def build_verdicts(
    sd: dict[str, torch.Tensor], arch: Arch, rows: list[ParamRow],
    ckpt: dict[str, Any], data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Per-knob verdicts plus every per-layer / per-head / spectral table."""
    np_sd = {k: v.detach().cpu().float().numpy() for k, v in sd.items()}
    stale_names = {r.name for r in rows if r.stale}
    spectra: dict[str, dict[str, float]] = {}

    def spec(name: str) -> dict[str, float]:
        if name not in spectra:
            spectra[name] = spectral(np_sd[name])
        return spectra[name]

    L, H, hd = arch.n_layers, arch.n_heads, arch.head_dim
    tc = ckpt.get('training_config', {})
    ph = arch.patch_hours

    # Training-progress gate: random init is full rank, so a barely trained checkpoint
    # reads as "saturated → GROW". Suppress resize advice when the main matrices sit
    # at their init std.
    main_drifts = [abs(r.drift - 1.0) for r in rows
                   if r.kind in ('linear', 'residual', 'head_final') and not math.isnan(r.drift)]
    undertrained = bool(main_drifts) and float(np.median(main_drifts)) < DRIFT_NEAR_INIT

    base_std = INIT_BASE_STD_ANCHOR * math.sqrt(INIT_BASE_STD_ANCHOR_DMODEL / arch.d_model)
    residual_std = base_std / math.sqrt(RESIDUAL_WRITES_PER_BLOCK * L)

    # ---- per-layer rows (weights) ------------------------------------------------
    layers: list[LayerRow] = []
    for i in range(L):
        p = f'blocks.{i}.'
        util_attn = float(np.mean([spec(p + f'attn.{m}.weight')['util'] for m in ('w_q', 'w_k', 'w_v', 'w_o')]))
        util_ffn = float(np.mean([spec(p + f'ffn.{m}.weight')['util'] for m in ('w1', 'w3', 'w2')]))
        fs = ffn_strength(np_sd, i)
        layers.append(LayerRow(
            i, util_attn, util_ffn,
            float(np_sd[p + 'attn.w_o.weight'].std() / residual_std),
            float(np_sd[p + 'ffn.w2.weight'].std() / residual_std),
            int(live_mask(fs).sum()), int(hot_mask(fs).sum()), top_share(fs, RESID_TOP_K),
        ))
    heads = head_rows_from_weights(np_sd, arch)

    # ---- fold the data pass in ----------------------------------------------------
    if data:
        for lr in layers:
            i = lr.layer
            blk = data['blocks'][i]
            lr.ffn_live_a = int(blk['ffn_live_a'])
            lr.ffn_hot_a = int(blk['ffn_hot_a'])
            lr.attn_rel, lr.ffn_rel, lr.cos_io = blk['attn_rel'], blk['ffn_rel'], blk['cos_io']
            lr.cka_prev, lr.cka_final = blk['cka_prev'], blk['cka_final']
            lr.tok_outlier = blk['tok_outlier']
            lr.ablate_attn_pct = data['ablation']['attn'][i]
            lr.ablate_ffn_pct = data['ablation']['ffn'][i]
            lr.ablate_block_pct = data['ablation']['block'][i]
        for hr in heads:
            d = data['heads'][hr.layer][hr.head]
            hr.act_rms, hr.act_rel = d['act_rms'], d['act_rel']
            hr.entropy, hr.self_mass, hr.masked_mass = d['entropy'], d['self_mass'], d['masked_mass']
            hr.reach_p50_h, hr.reach_p90_h = d['reach_p50_h'], d['reach_p90_h']
            hr.ablate_pct = data['ablation']['heads'][hr.layer][hr.head]

    # ---- flags -------------------------------------------------------------------
    for hr in heads:
        if hr.w_rel < DEAD_UNIT_REL or (not math.isnan(hr.act_rel) and hr.act_rel < DEAD_UNIT_REL):
            hr.flags.append('DEAD')
        elif hr.w_rel < HEAD_WEAK_REL or (not math.isnan(hr.ablate_pct) and hr.ablate_pct < HEAD_ABLATE_WEAK_PCT):
            hr.flags.append('WEAK')
        if not math.isnan(hr.act_rel) and hr.act_rel > HEAD_DOMINANT_REL:
            hr.flags.append('DOMINANT')
        if not math.isnan(hr.entropy) and hr.entropy > HEAD_ENTROPY_UNIFORM:
            hr.flags.append('UNIFORM')
        if not math.isnan(hr.self_mass) and hr.self_mass > HEAD_SELF_MASS:
            hr.flags.append('SELF')
    for lr in layers:
        if lr.write_attn < 1.2 and lr.write_ffn < 1.2:
            lr.flags.append('near-init')
        removable = (not math.isnan(lr.ablate_block_pct) and lr.ablate_block_pct < BLOCK_ABLATE_REMOVABLE_PCT)
        redundant = (not math.isnan(lr.cos_io) and lr.cos_io > BLOCK_COS_IDENTITY
                     and not math.isnan(lr.cka_prev) and lr.cka_prev > CKA_REDUNDANT)
        if removable:
            lr.flags.append('REMOVABLE')
        elif redundant:
            lr.flags.append('near-identity')
        if not math.isnan(lr.ablate_attn_pct) and lr.ablate_attn_pct < HEAD_ABLATE_WEAK_PCT:
            lr.flags.append('attn-idle')
        if not math.isnan(lr.ablate_ffn_pct) and lr.ablate_ffn_pct < HEAD_ABLATE_WEAK_PCT:
            lr.flags.append('ffn-idle')
        if lr.ffn_hot_w or (lr.ffn_hot_a or 0):
            lr.flags.append('hot-units')

    verdicts: list[Verdict] = []

    # ---- D_MODEL --------------------------------------------------------------------
    dmodel_maps = ['patch_embed.weight']
    for i in range(L):
        dmodel_maps += [f'blocks.{i}.attn.{m}.weight' for m in ('w_q', 'w_k', 'w_v', 'w_o')]
        dmodel_maps += [f'blocks.{i}.ffn.{m}.weight' for m in ('w1', 'w3', 'w2')]
    dmodel_maps.append(arch.bg_head_linears[0])
    utils_dm = [spec(m)['util'] for m in dmodel_maps]
    mean_util_dm = float(np.mean(utils_dm))
    write = row_norms(np_sd['patch_embed.weight']).copy()
    read = col_norms(np_sd[arch.bg_head_linears[0]]).copy()
    for i in range(L):
        write += row_norms(np_sd[f'blocks.{i}.attn.w_o.weight'])
        write += row_norms(np_sd[f'blocks.{i}.ffn.w2.weight'])
        for m in ('attn.w_q', 'attn.w_k', 'attn.w_v', 'ffn.w1', 'ffn.w3'):
            read += col_norms(np_sd[f'blocks.{i}.{m}.weight'])
    live_dim = live_mask(write) & live_mask(read)
    n_live = int(live_dim.sum())
    n_stale_dm = sum(1 for m in dmodel_maps if m in stale_names)
    ev = [f'rank-util {mean_util_dm:.2f}, mean over {len(dmodel_maps)} D-maps '
          f'(min {min(utils_dm):.2f} {dmodel_maps[int(np.argmin(utils_dm))]}, max {max(utils_dm):.2f})',
          f'residual dims live by weight (written AND read) {n_live}/{arch.d_model}',
          f'weight-space top-{RESID_TOP_K} dim share of write norm {top_share(write, RESID_TOP_K):.0%}']
    eff_active = n_live / arch.d_model
    if data:
        r = data['resid']
        ev.append(f"residual dims with variance {r['live']}/{arch.d_model} "
                  f"({r['live_frac']:.0%}); top-{RESID_TOP_K} dims carry {r['top_share']:.0%} of the variance "
                  f"(uniform would be {RESID_TOP_K / arch.d_model:.0%})")
        if r.get('pca_dims'):
            ev.append(f"final-block PCA: {r['pca_dims']['p90']} dims hold 90% of the variance, "
                      f"{r['pca_dims']['p99']} hold 99%")
        eff_active = r['live_frac']
    if n_stale_dm:
        ev.append(f'{n_stale_dm} stale D-map(s)')
    if mean_util_dm < RANK_UTIL_SHRINK or eff_active < ACTIVE_FRAC_SHRINK:
        vd = 'SHRINK'
    elif mean_util_dm > RANK_UTIL_GROW and eff_active > ACTIVE_FRAC_GROW:
        vd = 'GROW'
    else:
        vd = 'KEEP'
    if data:
        lad = data['ladders']['d_model']
        ev.append('rank ladder (stream projected onto its top-k PCs at every block): ' + _ladder_line(lad, arch.d_model))
        vd, why = _ladder_verdict(lad, vd)
        if why:
            ev.append(why)
    sug = (_suggest_dmodel(arch, arch.d_model // 2) if vd == 'SHRINK'
           else _suggest_dmodel(arch, arch.d_model * 2) if vd == 'GROW' else '')
    verdicts.append(Verdict('D_MODEL', str(arch.d_model), vd, sug, ev,
                            {'mean_rank_util': mean_util_dm, 'live_resid_dims': n_live,
                             'per_map_util': dict(zip(dmodel_maps, utils_dm))}))

    # ---- N_LAYERS -------------------------------------------------------------------
    ev = [f"block writes {min(l.write_attn for l in layers):.1f}–{max(l.write_ffn for l in layers):.1f}× init "
          f"(attn {', '.join(f'{l.write_attn:.1f}' for l in layers)}; ffn {', '.join(f'{l.write_ffn:.1f}' for l in layers)})",
          f"rank-util attn {', '.join(f'{l.util_attn:.2f}' for l in layers)}; ffn {', '.join(f'{l.util_ffn:.2f}' for l in layers)}"]
    near_init_blocks = [l.layer for l in layers if 'near-init' in l.flags]
    removable = [l.layer for l in layers if 'REMOVABLE' in l.flags]
    redundant = [l.layer for l in layers if 'near-identity' in l.flags]
    if data:
        ev.append(f"skip-block Δ pinball {', '.join(_pct(l.ablate_block_pct) for l in layers)}")
        ev.append(f"attn-only Δ {', '.join(_pct(l.ablate_attn_pct) for l in layers)}")
        ev.append(f"ffn-only Δ {', '.join(_pct(l.ablate_ffn_pct) for l in layers)}")
        ev.append(f"cos(in,out) {', '.join(_f(l.cos_io) for l in layers)}; CKA(prev) {', '.join(_f(l.cka_prev) for l in layers)}")
    if near_init_blocks:
        ev.append(f'blocks at init write scale: {near_init_blocks}')
    if removable:
        ev.append(f'blocks whose skip costs < {BLOCK_ABLATE_REMOVABLE_PCT}%: {removable}')
    if redundant:
        ev.append(f'blocks near identity with a representation their predecessor already holds: {redundant}'
                  + (' (skip cost above the removable line, so not cut)' if data else ''))
    cut = len(set(near_init_blocks) | set(removable) | (set() if data else set(redundant)))
    if cut:
        vd, sug = 'SHRINK', f'--layers {max(1, L - cut)}'
    else:
        saturated_w = all(l.util_attn > RANK_UTIL_GROW or l.util_ffn > RANK_UTIL_GROW for l in layers) \
            and min(min(l.write_attn, l.write_ffn) for l in layers) > 2.0
        if data:
            min_blk = min(l.ablate_block_pct for l in layers)
            saturated_d = min_blk > 10.0 * BLOCK_ABLATE_REMOVABLE_PCT
            ev.append(f'weakest block skip costs {min_blk:+.2f}%'
                      + (' — every block load-bearing' if saturated_d else ''))
            vd, sug = ('GROW', f'--layers {L + 1}') if (saturated_d and saturated_w) else ('KEEP', '')
        else:
            vd, sug = ('GROW', f'--layers {L + 1}') if saturated_w else ('KEEP', '')
    verdicts.append(Verdict('N_LAYERS', str(L), vd, sug, ev,
                            {'block_write_attn': [l.write_attn for l in layers],
                             'block_write_ffn': [l.write_ffn for l in layers]}))

    # ---- N_HEADS --------------------------------------------------------------------
    n_total = len(heads)
    dead = [(h.layer, h.head) for h in heads if 'DEAD' in h.flags]
    weak = [(h.layer, h.head) for h in heads if 'WEAK' in h.flags]
    dominant = [(h.layer, h.head) for h in heads if 'DOMINANT' in h.flags]
    idle_attn = [(h.layer, h.head) for h in heads if 'UNIFORM' in h.flags or 'SELF' in h.flags]
    ev = [f'{H} head(s) × {L} layer(s), HEAD_DIM {hd}',
          f'{len(dead)} dead, {len(weak)} weak, {len(dominant)} dominant of {n_total} (layer.head): '
          f"dead {['%d.%d' % x for x in dead]}, weak {['%d.%d' % x for x in weak]}, dominant {['%d.%d' % x for x in dominant]}"]
    if data:
        ev.append(f"uniform/self-attending heads {['%d.%d' % x for x in idle_attn]}")
        abl = [h.ablate_pct for h in heads]
        ev.append(f'head ablation Δ pinball: min {min(abl):+.2f}%, median {float(np.median(abl)):+.2f}%, max {max(abl):+.2f}%')
    sug_shrink, sug_grow = _suggest_heads(arch, False), _suggest_heads(arch, True)
    n_cut = len(set(dead) | set(weak))
    if H == 1:
        vd, sug = 'KEEP', ''
        ev.append('one head per layer: a head ablation is the attention-sublayer ablation, so more-vs-fewer '
                  f'heads is undecidable here; a {sug_grow.split()[-1] if sug_grow else 2}-head run at the same '
                  'D_MODEL is the test')
    elif n_cut >= max(1, n_total // 4) and sug_shrink:
        vd, sug = 'SHRINK', sug_shrink
    elif (data and not dead and not weak and hd > VALID_HEAD_DIMS[0] and sug_grow
          and float(np.median([h.ablate_pct for h in heads])) > 4 * HEAD_ABLATE_WEAK_PCT):
        vd, sug = 'GROW', sug_grow
    else:
        vd, sug = 'KEEP', ''
    if vd == 'SHRINK' and not sug_shrink:
        vd = 'KEEP'
    verdicts.append(Verdict('N_HEADS', str(H), vd, sug, ev,
                            {'dead': dead, 'weak': weak, 'dominant': dominant}))

    # ---- FFN_DIM --------------------------------------------------------------------
    mean_util_ffn = float(np.mean([l.util_ffn for l in layers]))
    live_w = float(np.mean([l.ffn_live_w / arch.ffn_dim for l in layers]))
    ev = [f'rank-util {mean_util_ffn:.2f} (per layer {", ".join(f"{l.util_ffn:.2f}" for l in layers)})',
          f'live units by weight {live_w:.0%} (per layer {", ".join(str(l.ffn_live_w) for l in layers)}/{arch.ffn_dim})',
          f'hot units by weight per layer {", ".join(str(l.ffn_hot_w) for l in layers)}; '
          f'top-{RESID_TOP_K} unit share {", ".join(f"{l.ffn_top8_share_w:.0%}" for l in layers)}']
    eff_active = live_w
    if data:
        live_a = float(np.mean([(l.ffn_live_a or 0) / arch.ffn_dim for l in layers]))
        ev.append(f'live units by activation {live_a:.0%} (per layer {", ".join(str(l.ffn_live_a) for l in layers)}/{arch.ffn_dim}); '
                  f'hot {", ".join(str(l.ffn_hot_a) for l in layers)}')
        ev.append(f"ffn-only ablation Δ {', '.join(_pct(l.ablate_ffn_pct) for l in layers)}")
        eff_active = live_a
    n_stale_ffn = sum(1 for i in range(L) for m in ('w1', 'w2', 'w3') if f'blocks.{i}.ffn.{m}.weight' in stale_names)
    if n_stale_ffn:
        ev.append(f'{n_stale_ffn} stale FFN matrices')
    mult = arch.ffn_dim // arch.d_model
    sug_shrink = (f'--ffn-mult {mult // 2}' if mult >= 2
                  else 'already 1×D — FFN width tracks D_MODEL; lower --d-model instead')
    sug_grow = f'--ffn-mult {mult * 2 if mult >= 1 else 2}'
    if mean_util_ffn < RANK_UTIL_SHRINK or eff_active < ACTIVE_FRAC_SHRINK:
        vd = 'SHRINK'
    elif mean_util_ffn > RANK_UTIL_GROW and eff_active > ACTIVE_FRAC_GROW:
        vd = 'GROW'
    else:
        vd = 'KEEP'
    if data:
        lad = data['ladders']['ffn']
        ev.append('unit ladder (top-k units by activation kept in every layer): ' + _ladder_line(lad, arch.ffn_dim))
        vd, why = _ladder_verdict(lad, vd)
        if why:
            ev.append(why)
    sug = sug_shrink if vd == 'SHRINK' else sug_grow if vd == 'GROW' else ''
    verdicts.append(Verdict('FFN_DIM', f'{arch.ffn_dim} ({mult}×D)', vd, sug, ev,
                            {'mean_util': mean_util_ffn, 'live_w': live_w}))

    # ---- BG_HEAD_HIDDEN --------------------------------------------------------------
    bl = arch.bg_head_linears
    s0 = row_norms(np_sd[bl[0]]) * col_norms(np_sd[bl[1]])
    s1 = row_norms(np_sd[bl[1]]) * col_norms(np_sd[bl[2]])
    live0, live1 = int(live_mask(s0).sum()), int(live_mask(s1).sum())
    bg_util = [spec(bl[0])['util'], spec(bl[1])['util']]
    ev = [f'rank-util layer0 {bg_util[0]:.2f}, layer1 {bg_util[1]:.2f}',
          f'live units by weight layer0 {live0}/{arch.bg_head_hidden}, layer1 {live1}/{arch.bg_head_hidden}; '
          f'hot {int(hot_mask(s0).sum())}, {int(hot_mask(s1).sum())}']
    eff_active = max(live0, live1) / arch.bg_head_hidden
    if data:
        b = data['bg_head']
        ev.append(f"live units by activation layer0 {b['live0']}/{arch.bg_head_hidden}, layer1 {b['live1']}/{arch.bg_head_hidden}; "
                  f"hot {b['hot0']}, {b['hot1']}")
        eff_active = min(b['live0'], b['live1']) / arch.bg_head_hidden
        sp = data['spline']
        ev.append('spline step state vs own-patch state, per step: '
                  + ', '.join(f'{x:.0%}' for x in sp['state_dev']))
        ev.append('median-delta change the spline makes, per step: '
                  + ', '.join(f'{x:.0%}' for x in sp['median_dev'])
                  + f"; spread columns {', '.join(f'{x:.0%}' for x in sp['spread_dev'])}")
    n_stale_bg = sum(1 for n in bl if n in stale_names)
    if n_stale_bg:
        ev.append(f'{n_stale_bg} stale BG-head matrices')
    mult_bg = arch.bg_head_hidden // arch.d_model
    sug_shrink = (f'--bg-head-hidden-mult {mult_bg // 2}' if mult_bg >= 2
                  else 'already 1×D — head width tracks D_MODEL; lower --d-model instead')
    sug_grow = f'--bg-head-hidden-mult {mult_bg * 2 if mult_bg >= 1 else 2}'
    mu = float(np.mean(bg_util))
    if mu < RANK_UTIL_SHRINK or eff_active < ACTIVE_FRAC_SHRINK:
        vd = 'SHRINK'
    elif mu > RANK_UTIL_GROW and eff_active > ACTIVE_FRAC_GROW:
        vd = 'GROW'
    else:
        vd = 'KEEP'
    if data:
        lad = data['ladders']['bg_head']
        ev.append('unit ladder (top-k hidden units kept in both layers): ' + _ladder_line(lad, arch.bg_head_hidden))
        vd, why = _ladder_verdict(lad, vd)
        if why:
            ev.append(why)
    sug = sug_shrink if vd == 'SHRINK' else sug_grow if vd == 'GROW' else ''
    verdicts.append(Verdict('BG_HEAD_HIDDEN', f'{arch.bg_head_hidden} ({mult_bg}×D)', vd, sug, ev,
                            {'live_w': [live0, live1], 'util': bg_util}))

    # ---- TIME_HEAD (not a resize knob; diagnostic) ------------------------------------
    if arch.time_head_linears:
        tl = arch.time_head_linears
        st = row_norms(np_sd[tl[0]]) * col_norms(np_sd[tl[1]])
        ev = [f'{arch.time_hidden} hidden → {arch.time_bins} bins; '
              f'live units by weight {int(live_mask(st).sum())}/{arch.time_hidden}, rank-util {spec(tl[0])["util"]:.2f}']
        if data:
            t = data['time_head']
            ev.append(f"live units by activation {t['live']}/{arch.time_hidden}, hot {t['hot']}")
        verdicts.append(Verdict('TIME_PROBE_HIDDEN', str(arch.time_hidden), 'NOTE',
                                'follows D_MODEL; diagnostic only, never feeds the fan', ev))

    # ---- PATCH_SIZE -----------------------------------------------------------------
    pe = np_sd['patch_embed.weight']
    in_use = col_norms(pe)
    mask_bit_cols = np.arange(BG_MASKED_FEAT, arch.patch_dim, N_INPUT_FEATURES)
    signal_cols = np.setdiff1d(np.arange(arch.patch_dim), mask_bit_cols)
    feat_cols = in_use[signal_cols]
    dead_feat = int((~live_mask(feat_cols)).sum())
    per_feat = [float(in_use[np.arange(f, arch.patch_dim, N_INPUT_FEATURES)].mean())
                for f in range(N_INPUT_FEATURES - 1)]
    bit_rel = float(np.linalg.norm(pe[:, mask_bit_cols].sum(axis=1)) / (np.median(feat_cols) + 1e-12))
    ev = [f'{arch.patch_size} steps/patch ({arch.patch_size * STEP_MINUTES} min); '
          f'{dead_feat}/{feat_cols.size} signal feature-columns dead',
          'mean embed column norm per feature [bg, carb, insulin, exercise]: '
          + ', '.join(f'{x:.3f}' for x in per_feat)
          + f'; bg_masked bit {bit_rel:.2f}× the median signal column',
          'per-step column norm (step 0..S-1), bg feature: '
          + ', '.join(f'{in_use[s * N_INPUT_FEATURES]:.3f}' for s in range(arch.patch_size))]
    verdicts.append(Verdict('PATCH_SIZE', str(arch.patch_size), 'NOTE',
                            'structural — retrain required (resize_model.py --patch-size)', ev,
                            {'dead_feature_cols': dead_feat, 'bg_masked_rel_weight': bit_rel,
                             'per_feature_norm': per_feat}))

    # ---- CONTEXT --------------------------------------------------------------------
    min_ctx = tc.get('min_context_patches')
    max_ctx = tc.get('max_context_patches')
    cur = (f'{min_ctx}–{max_ctx} patches ({min_ctx * ph:.0f}–{max_ctx * ph:.0f} h)'
           if min_ctx and max_ctx else '?')
    ev: list[str] = []
    vd, sug, detail = 'NOTE', '', {}
    if data:
        c = data['context']
        ev.append('attention reach on masked query rows, hours to reach 50/90/99% of mass — '
                  f"final layer {c['final_reach'][0]:.1f}/{c['final_reach'][1]:.1f}/{c['final_reach'][2]:.1f}, "
                  f"rollout {c['roll_reach'][0]:.1f}/{c['roll_reach'][1]:.1f}/{c['roll_reach'][2]:.1f}")
        ev.append('per-layer p90 reach (h): ' + ', '.join(f'{x:.1f}' for x in c['layer_reach_p90']))
        ev.append('attention per available key by lag, relative to ≤1 h (rollout): '
                  + ', '.join(f"{lo}–{hi}h {v:.3f}" for (lo, hi), v in zip(c['lag_edges'], c['roll_density_rel'])))
        ladder = c['ladder']
        ev.append('truncation ladder on forecast-zone slots, Δ pinball when only the last k h of context stay: '
                  + ', '.join(f"{r['hours']}h {r['delta_pct']:+.2f}% (n={r['n']})" for r in ladder if r['n'] > 0))
        min_h = (min_ctx or 0) * ph
        max_h = (max_ctx or 0) * ph
        usable = [r for r in ladder if r['n'] >= 8]
        cheap = [r for r in usable if r['delta_pct'] < CTX_TRUNC_TOL_PCT]
        far_rel = c['roll_density_rel'][-1] if c['roll_density_rel'] else float('nan')
        ratio = (max_ctx / min_ctx) if (min_ctx and max_ctx) else 2.0
        if cheap and min_ctx and max_ctx:
            k = cheap[0]
            if k['hours'] < min_h:
                new_min = max(1, k['patches'])
                vd, sug = 'SHRINK', f'--min-context-patches {new_min} --max-context-patches {int(round(new_min * ratio))}'
                ev.append(f"keeping only the last {k['hours']} h costs {k['delta_pct']:+.2f}% "
                          f"(< {CTX_TRUNC_TOL_PCT}%): context beyond it is not used")
            elif k['hours'] < max_h:
                vd, sug = 'SHRINK', f'--max-context-patches {k["patches"]}'
                ev.append(f"windows longer than {k['hours']} h gain {-k['delta_pct']:+.2f}% from the extra context "
                          f"(< {CTX_TRUNC_TOL_PCT}%): MAX_CONTEXT_PATCHES can drop to {k['patches']}")
            else:
                vd = 'KEEP'
        elif (usable and usable[-1]['delta_pct'] > LADDER_HURT_PCT
              and not math.isnan(far_rel) and far_rel > CTX_FAR_DENSITY_REL):
            vd = 'GROW'
            sug = f'--min-context-patches {min_ctx} --max-context-patches {int(max_ctx * 1.5)}' if (min_ctx and max_ctx) else ''
            ev.append(f"truncating to {usable[-1]['hours']} h still costs {usable[-1]['delta_pct']:+.2f}% and far keys "
                      f"draw {far_rel:.3f}× the near-lag attention: the window is still paying at its longest tested rung")
        else:
            vd = 'KEEP'
        detail = c
    else:
        ev.append('needs --data: reach and the truncation ladder are activation evidence')
    verdicts.append(Verdict('CONTEXT', cur, vd, sug, ev, detail))

    if undertrained:
        for v in verdicts:
            if v.verdict in ('SHRINK', 'GROW'):
                v.verdict, v.suggest = 'KEEP', ''
                v.evidence.insert(0, '[undertrained — verdict suppressed]')

    return {'verdicts': verdicts, 'spectra': spectra, 'undertrained': undertrained,
            'layers': layers, 'heads': heads}


# --------------------------------------------------------------------------- data pass
def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two ``(n, D)`` representations over the same n tokens."""
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xy = (y.T @ x).pow(2).sum()
    xx = (x.T @ x).pow(2).sum().sqrt()
    yy = (y.T @ y).pow(2).sum().sqrt()
    return float(xy / (xx * yy).clamp_min(1e-30))


def _reach(hist: np.ndarray, patch_hours: float, ps: tuple[float, ...] = (0.5, 0.9, 0.99)) -> list[float]:
    """Hours at which the cumulative lag-mass histogram first reaches each quantile."""
    tot = hist.sum()
    if tot <= 0:
        return [float('nan')] * len(ps)
    c = np.cumsum(hist) / tot
    return [float(int(np.searchsorted(c, p)) * patch_hours) for p in ps]


def _bucket(hist: np.ndarray, avail: np.ndarray, patch_hours: float) -> tuple[list[tuple[float, float]], list[float]]:
    """Attention mass per available key inside each lag bucket, relative to the first bucket."""
    edges = [0.0] + [float(h) for h in LAG_BUCKET_HOURS]
    lags_h = np.arange(hist.size) * patch_hours
    dens = []
    spans = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lags_h >= lo) & (lags_h < hi)
        a = avail[sel].sum()
        dens.append(float(hist[sel].sum() / a) if a > 0 else float('nan'))
        spans.append((lo, hi))
    ref = dens[0] if dens and dens[0] and dens[0] > 0 else float('nan')
    return spans, [d / ref if not math.isnan(d) else float('nan') for d in dens]


def run_data_pass(
    arch: Arch, ckpt: dict[str, Any], state_dict: dict[str, torch.Tensor],
    n_samples: int, device: torch.device,
) -> dict[str, Any] | None:
    """Stream cached windows through the model; collect activation, attention, block,
    spline and ablation evidence.

    Patches the in-process ``config`` to the checkpoint's dims and re-imports ``model``
    and ``data`` against them (both bind config constants at import).
    """
    try:
        import config
        tc = ckpt.get('training_config', {})
        config.D_MODEL = arch.d_model
        config.N_HEADS = arch.n_heads
        config.HEAD_DIM = arch.head_dim
        config.FFN_DIM = arch.ffn_dim
        config.BG_HEAD_HIDDEN = arch.bg_head_hidden
        config.TIME_PROBE_HIDDEN = arch.time_hidden or config.TIME_PROBE_HIDDEN
        config.TIME_PROBE_ENABLED = bool(arch.time_head_linears)
        config.N_LAYERS = arch.n_layers
        config.PATCH_SIZE = arch.patch_size
        config.PATCH_DIM = arch.patch_dim
        pph = 60 // (arch.patch_size * STEP_MINUTES)
        config._PATCHES_PER_HOUR = pph
        config.PREDICTION_PATCHES = config.PREDICTION_HORIZON_HOURS * pph
        for k in ('min_context_patches', 'max_context_patches', 'max_masked_patches',
                  'mask_right_edge_quota'):
            if tc.get(k) is not None:
                setattr(config, k.upper(), tc[k])
        if tc.get('mask_span_lengths') is not None:
            config.MASK_SPAN_LENGTHS = tuple(tc['mask_span_lengths'])
        config.MAX_SEQ_LEN = config.MAX_CONTEXT_PATCHES + config.PREDICTION_PATCHES
        config.NIGHT_LONG_HORIZON_PATCHES = config.NIGHT_LONG_HORIZON_HOURS * pph
        for m in ('model', 'data', 'risk_loss', 'attribution', 'inference'):
            sys.modules.pop(m, None)
        from model import T1DMAI
        from data import T1DMDataset, collate_fn
        from risk_loss import pinball_loss
        from attribution import capture_attention, rollout
        from utils import kovatchev_f_target, step_states, create_attention_mask_from_visible

        model = T1DMAI().to(device)
        model.load_state_dict(state_dict)
        model.eval()

        cache_path = tc.get('cache_path', 'simulator_cache/')
        if not Path(cache_path, 'meta.json').exists():
            print(f"[data] cache {cache_path!r} unavailable — skipping data pass")
            return None
        policy = checkpoint_masked_channel_policy(ckpt)
        ds = T1DMDataset(
            master_seed=int(ckpt.get('master_seed', 0)),
            total_steps=max(1, n_samples), batch_size=1,
            normalization_stats=ckpt['normalization_stats'],
            cache_path=cache_path, seed_offset=7_000_000,
            cache_partition='val',
            blind=(policy == MASKED_CHANNEL_POLICY_BLIND),
        )
    except Exception as e:  # noqa: BLE001 — the data pass is best-effort
        print(f"[data] could not initialize data pass: {type(e).__name__}: {e}")
        return None

    L, H, hd, D, S = arch.n_layers, arch.n_heads, arch.head_dim, arch.d_model, arch.patch_size
    ph = arch.patch_hours
    PRED = config.PREDICTION_PATCHES
    T_max = config.MAX_SEQ_LEN

    # ---- accumulators ------------------------------------------------------------
    n_done = 0
    head_act = np.zeros((L, H))
    head_ent = np.zeros((L, H)); head_self = np.zeros((L, H)); head_masked = np.zeros((L, H))
    head_rows_n = 0
    head_hist = np.zeros((L, H, T_max))
    avail_hist = np.zeros(T_max)
    roll_hist = np.zeros(T_max); final_hist = np.zeros(T_max)
    ffn_act = [np.zeros(arch.ffn_dim) for _ in range(L)]
    attn_rel = np.zeros(L); ffn_rel = np.zeros(L); cos_io = np.zeros(L); tok_out = np.zeros(L)
    cka = np.zeros((L + 1, L + 1))
    resid_var = np.zeros(D)
    bg0 = np.zeros(arch.bg_head_hidden); bg1 = np.zeros(arch.bg_head_hidden)
    th = np.zeros(arch.time_hidden) if arch.time_hidden else None
    sp_state_num = np.zeros(S); sp_state_den = 0.0
    sp_med_num = np.zeros(S); sp_med_den = np.zeros(S)
    sp_spr_num = np.zeros(S); sp_spr_den = np.zeros(S)
    base_sum = 0.0
    abl_heads = np.zeros((L, H)); abl_attn = np.zeros(L); abl_ffn = np.zeros(L); abl_block = np.zeros(L)
    ladder = [{'hours': h, 'patches': int(round(h / ph)), 'base': 0.0, 'trunc': 0.0, 'n': 0}
              for h in CONTEXT_LADDER_HOURS]
    cov = torch.zeros(L + 1, D, D, device=device)
    mean_sum = torch.zeros(L + 1, D, device=device)
    n_tok = 0
    windows: list[tuple[torch.Tensor, ...]] = []

    # ---- baseline hooks ---------------------------------------------------------
    cap: dict[str, Any] = {}
    collecting = [True]
    hooks = []

    def keep(key):
        def h(_m, inp, out=None):
            if collecting[0]:
                cap[key] = (inp[0] if out is None else out).detach()
        return h

    for i, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_pre_hook(keep(f'x_in{i}')))
        hooks.append(blk.attn.register_forward_hook(keep(f'attn_out{i}')))
        hooks.append(blk.ffn.register_forward_hook(keep(f'ffn_out{i}')))
        hooks.append(blk.register_forward_hook(keep(f'x_out{i}')))
        hooks.append(blk.attn.w_o.register_forward_pre_hook(keep(f'wo_in{i}')))
        hooks.append(blk.ffn.w2.register_forward_pre_hook(keep(f'w2_in{i}')))
    hooks.append(model.final_norm.register_forward_pre_hook(keep('final_in')))
    hooks.append(model.final_norm.register_forward_hook(keep('final_out')))
    bl = arch.bg_head_linears
    hooks.append(model.bg_head[int(bl[0].split('.')[1])].register_forward_pre_hook(keep('bg_in0')))
    hooks.append(model.bg_head[int(bl[1].split('.')[1])].register_forward_pre_hook(keep('bg_in1')))
    hooks.append(model.bg_head[int(bl[2].split('.')[1])].register_forward_pre_hook(keep('bg_in2')))
    hooks.append(model.bg_head.register_forward_hook(keep('head_raw')))
    if arch.time_head_linears and model.time_head is not None:
        idx = int(arch.time_head_linears[1].split('.')[1])
        hooks.append(model.time_head[idx].register_forward_pre_hook(keep('time_in1')))

    def head_zero_hook(h: int):
        def hk(_m, inp):
            x = inp[0].clone()
            x[..., h * hd:(h + 1) * hd] = 0
            return (x,)
        return hk

    def zero_out(_m, _inp, out):
        return torch.zeros_like(out)

    def skip_block(_m, inp, _out):
        return inp[0]

    n = min(n_samples, len(ds))
    n_abl = L * (H + 3) + len(CONTEXT_LADDER_HOURS)
    n_lad = 3 * len(WIDTH_LADDER_FRACS)
    print(f"[data] streaming {n} cached windows ({policy}); {n_abl} ablation + {n_lad} width-ladder "
          "forwards per window…")
    t0 = time.time()
    try:
        with torch.no_grad():
            for si in range(n):
                batch = collate_fn([ds[si]])
                patches = batch['patches'].to(device)
                attn_mask = batch['attn_mask'].to(device)
                bgf = batch['bg_formula_data']
                anchor_bg = bgf['anchor_bg'].float().to(device)
                mask_idx = bgf['mask_idx'].long().to(device)
                valid = bgf['valid'].bool().to(device)
                y_risk = kovatchev_f_target(batch['targets'].float().to(device))
                n_ctx = int(batch['n_context_patches'][0])
                T = patches.shape[1]
                n_pad = T - (n_ctx + PRED)
                real = torch.arange(n_pad, T, device=device)

                collecting[0] = True
                cap.clear()
                with capture_attention(model) as sink:
                    q_tau, median, time_pred = model(patches, attn_mask, anchor_bg, mask_idx,
                                                     return_time=True)
                collecting[0] = False
                base = float(pinball_loss(q_tau, y_risk, QUANTILE_LEVELS, valid))
                base_sum += base
                n_done += 1
                windows.append((patches.cpu(), attn_mask.cpu(), anchor_bg.cpu(), mask_idx.cpu(),
                                valid.cpu(), y_risk.cpu()))

                # -- attention geometry --------------------------------------------
                m_rows = mask_idx[0][valid[0]].unique()
                am = attn_mask[0]                                       # (T, T)
                masked_col = torch.zeros(T, dtype=torch.bool, device=device)
                masked_col[m_rows] = True
                n_keys = am[m_rows].sum(1).clamp_min(2).float()          # (R,)
                lag = (m_rows[:, None] - torch.arange(T, device=device)[None, :]).abs()  # (R, T)
                allowed = am[m_rows]                                     # (R, T) bool
                avail_hist += torch.bincount(lag[allowed], minlength=T_max).cpu().numpy()[:T_max]
                head_mean = []
                for l, layer_maps in enumerate(sink):
                    A = layer_maps[0][0]                                 # (H, T, T)
                    head_mean.append(A.mean(0))
                    Ar = A[:, m_rows, :]                                 # (H, R, T)
                    p = Ar.clamp_min(1e-30)
                    ent = -(Ar * p.log()).sum(-1) / n_keys.log()         # (H, R)
                    head_ent[l] += ent.mean(1).cpu().numpy()
                    head_self[l] += Ar[:, torch.arange(len(m_rows)), m_rows].mean(1).cpu().numpy()
                    head_masked[l] += Ar[..., masked_col].sum(-1).mean(1).cpu().numpy()
                    hist = torch.zeros(H, T_max, device=device)
                    hist.scatter_add_(1, lag.reshape(1, -1).expand(H, -1), Ar.reshape(H, -1))
                    head_hist[l] += hist.cpu().numpy()
                head_rows_n += 1
                fin = head_mean[-1][m_rows]                              # (R, T)
                final_hist += torch.zeros(T_max, device=device).scatter_add_(0, lag.reshape(-1), fin.reshape(-1)).cpu().numpy()
                rl = rollout(head_mean)[m_rows]
                roll_hist += torch.zeros(T_max, device=device).scatter_add_(0, lag.reshape(-1), rl.reshape(-1)).cpu().numpy()

                # -- block influence, CKA, unit activations ----------------------
                reps = [cap['x_in0'][0, real]]
                for i in range(L):
                    x_in = cap[f'x_in{i}'][0, real]
                    a_out = cap[f'attn_out{i}'][0, real]
                    f_out = cap[f'ffn_out{i}'][0, real]
                    x_out = cap[f'x_out{i}'][0, real]
                    mid = x_in + a_out
                    attn_rel[i] += float(a_out.pow(2).mean().sqrt() / x_in.pow(2).mean().sqrt().clamp_min(1e-12))
                    ffn_rel[i] += float(f_out.pow(2).mean().sqrt() / mid.pow(2).mean().sqrt().clamp_min(1e-12))
                    cos_io[i] += float(torch.nn.functional.cosine_similarity(x_in, x_out, dim=-1).mean())
                    tn = x_out.norm(dim=-1)
                    tok_out[i] += float(tn.max() / tn.median().clamp_min(1e-12))
                    reps.append(x_out)
                    wo_in = cap[f'wo_in{i}'][0, real].reshape(-1, H, hd)
                    head_act[i] += wo_in.pow(2).mean((0, 2)).sqrt().cpu().numpy()
                    ffn_act[i] += cap[f'w2_in{i}'][0, real].abs().mean(0).cpu().numpy()
                for a in range(L + 1):
                    cov[a] += reps[a].T @ reps[a]
                    mean_sum[a] += reps[a].sum(0)
                    for b in range(a, L + 1):
                        c = _linear_cka(reps[a], reps[b])
                        cka[a, b] += c
                        if a != b:
                            cka[b, a] += c
                n_tok += int(real.numel())
                resid_var += cap['final_in'][0, real].var(0).cpu().numpy()
                v = valid[0]
                bg0 += cap['bg_in1'][0, v].abs().reshape(-1, arch.bg_head_hidden).mean(0).cpu().numpy()
                bg1 += cap['bg_in2'][0, v].abs().reshape(-1, arch.bg_head_hidden).mean(0).cpu().numpy()
                if th is not None and 'time_in1' in cap:
                    th += cap['time_in1'][0, v].abs().mean(0).cpu().numpy()

                # -- spline: step state vs the own-patch state -------------------
                x_final = cap['final_out']
                h_spline = cap['bg_in0']                                 # (1, M, S, D)
                h_own = x_final.gather(1, mask_idx.unsqueeze(-1).expand(1, -1, D))
                raw_spline = cap['head_raw'][0, v]                       # (Mv, S, 7)
                raw_own = model.bg_head(h_own.unsqueeze(2).expand(-1, -1, S, -1))[0, v]
                dev = (h_spline[0, v] - h_own[0, v].unsqueeze(1))        # (Mv, S, D)
                sp_state_num += dev.pow(2).sum((0, 2)).cpu().numpy()
                sp_state_den += float(h_own[0, v].pow(2).sum())
                sp_med_num += (raw_spline[..., 0] - raw_own[..., 0]).pow(2).sum(0).cpu().numpy()
                sp_med_den += raw_spline[..., 0].pow(2).sum(0).cpu().numpy()
                sp_spr_num += (raw_spline[..., 1:] - raw_own[..., 1:]).pow(2).sum((0, 2)).cpu().numpy()
                sp_spr_den += raw_spline[..., 1:].pow(2).sum((0, 2)).cpu().numpy()

                # -- ablations ----------------------------------------------------
                def pin(hs) -> float:
                    try:
                        q, _ = model(patches, attn_mask, anchor_bg, mask_idx)
                        return float(pinball_loss(q, y_risk, QUANTILE_LEVELS, valid))
                    finally:
                        for hh in hs:
                            hh.remove()

                for i, blk in enumerate(model.blocks):
                    for h in range(H):
                        abl_heads[i, h] += pin([blk.attn.w_o.register_forward_pre_hook(head_zero_hook(h))])
                    abl_attn[i] += pin([blk.attn.register_forward_hook(zero_out)])
                    abl_ffn[i] += pin([blk.ffn.register_forward_hook(zero_out)])
                    abl_block[i] += pin([blk.register_forward_hook(skip_block)])

                # -- context truncation ladder --------------------------------------
                # Scored on the forecast-zone slots only (the trailing PRED patches), so
                # every rung compares the same slot set; a rung skips windows whose
                # context is already shorter than it.
                keep_slot = valid & (mask_idx >= T - PRED)
                for rung in ladder:
                    k = rung['patches']
                    if k >= n_ctx or int(keep_slot.sum()) == 0:
                        continue
                    cut = T - (k + PRED)
                    is_pad = torch.zeros(1, T, dtype=torch.bool, device=device)
                    is_pad[:, :cut] = True
                    visible = ~masked_col.unsqueeze(0)
                    am_k = create_attention_mask_from_visible(visible, is_pad)
                    p_k = patches.clone()
                    p_k[:, :cut] = 0
                    q_k, _ = model(p_k, am_k, anchor_bg, mask_idx)
                    rung['trunc'] += float(pinball_loss(q_k, y_risk, QUANTILE_LEVELS, keep_slot))
                    rung['base'] += float(pinball_loss(q_tau, y_risk, QUANTILE_LEVELS, keep_slot))
                    rung['n'] += 1

                if (si + 1) % 16 == 0 or si + 1 == n:
                    el = time.time() - t0
                    print(f"[data] {si + 1}/{n} windows, {el:.0f}s elapsed, "
                          f"~{el / (si + 1) * (n - si - 1):.0f}s left")
    except Exception as e:  # noqa: BLE001
        print(f"[data] forward failed mid-stream: {type(e).__name__}: {e}")
        if n_done == 0:
            return None
    finally:
        for hk in hooks:
            hk.remove()

    # ---- pass 2: width ladders on the same windows ----------------------------------
    # D_MODEL: every block output (and the embedding) projected onto the top-k principal
    # components of its own activations — a proxy for a k-wide residual stream, fit on
    # the streamed windows. FFN / BG head: only the top-k units by mean |activation| kept
    # in every layer. Each rung is one forward per window; Δ is against the full model.
    mu = (mean_sum / max(n_tok, 1))
    projectors: dict[int, list[torch.Tensor]] = {}
    pca_dims = {}
    ks_d = sorted({max(1, int(round(f * D))) for f in WIDTH_LADDER_FRACS})
    evals_final = None
    for a in range(L + 1):
        c = cov[a] / max(n_tok, 1) - torch.outer(mu[a], mu[a])
        evals, evecs = torch.linalg.eigh(c)
        evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
        if a == L:
            evals_final = evals.cpu().numpy()
        for k in ks_d:
            V = evecs[:, :k]
            projectors.setdefault(k, []).append(V @ V.T)
    if evals_final is not None and evals_final.sum() > 0:
        cum = np.cumsum(evals_final) / evals_final.sum()
        pca_dims = {'p90': int(np.searchsorted(cum, 0.90) + 1), 'p99': int(np.searchsorted(cum, 0.99) + 1)}
    ks_f = sorted({max(1, int(round(f * arch.ffn_dim))) for f in WIDTH_LADDER_FRACS})
    ks_b = sorted({max(1, int(round(f * arch.bg_head_hidden))) for f in WIDTH_LADDER_FRACS})
    ffn_rank = [np.argsort(-ffn_act[i]) for i in range(L)]
    bg_rank = [np.argsort(-bg0), np.argsort(-bg1)]

    def project_hook(a: int, k: int):
        P = projectors[k][a]
        m = mu[a]

        def hk(_m, _inp, out):
            return m + (out - m) @ P
        return hk

    def unit_keep_hook(order: np.ndarray, k: int, width: int):
        keep = torch.zeros(width, device=device)
        keep[torch.as_tensor(order[:k].copy(), device=device)] = 1.0

        def hk(_m, inp):
            return (inp[0] * keep,)
        return hk

    lad_d = {k: 0.0 for k in ks_d}
    lad_f = {k: 0.0 for k in ks_f}
    lad_b = {k: 0.0 for k in ks_b}
    print(f"[data] width ladders: D_MODEL {ks_d}, FFN_DIM {ks_f}, BG_HEAD_HIDDEN {ks_b}…")
    try:
        with torch.no_grad():
            for (patches, attn_mask, anchor_bg, mask_idx, valid, y_risk) in windows:
                patches, attn_mask = patches.to(device), attn_mask.to(device)
                anchor_bg, mask_idx = anchor_bg.to(device), mask_idx.to(device)
                valid, y_risk = valid.to(device), y_risk.to(device)

                def pin2(hs) -> float:
                    try:
                        q, _ = model(patches, attn_mask, anchor_bg, mask_idx)
                        return float(pinball_loss(q, y_risk, QUANTILE_LEVELS, valid))
                    finally:
                        for hh in hs:
                            hh.remove()

                for k in ks_d:
                    hs = [model.patch_embed.register_forward_hook(project_hook(0, k))]
                    hs += [blk.register_forward_hook(project_hook(i + 1, k)) for i, blk in enumerate(model.blocks)]
                    lad_d[k] += pin2(hs)
                for k in ks_f:
                    hs = [blk.ffn.w2.register_forward_pre_hook(unit_keep_hook(ffn_rank[i], k, arch.ffn_dim))
                          for i, blk in enumerate(model.blocks)]
                    lad_f[k] += pin2(hs)
                for k in ks_b:
                    hs = [model.bg_head[int(bl[1].split('.')[1])].register_forward_pre_hook(
                              unit_keep_hook(bg_rank[0], k, arch.bg_head_hidden)),
                          model.bg_head[int(bl[2].split('.')[1])].register_forward_pre_hook(
                              unit_keep_hook(bg_rank[1], k, arch.bg_head_hidden))]
                    lad_b[k] += pin2(hs)
    except Exception as e:  # noqa: BLE001
        print(f"[data] width ladder failed: {type(e).__name__}: {e}")
        lad_d, lad_f, lad_b = {}, {}, {}

    # ---- reduce -------------------------------------------------------------------
    def frac_live(a: np.ndarray) -> tuple[int, int]:
        return int(live_mask(a).sum()), int(hot_mask(a).sum())

    heads_out: list[list[dict[str, float]]] = []
    for l in range(L):
        act = head_act[l] / n_done
        med = float(np.median(act)) or 1.0
        row = []
        for h in range(H):
            r50, r90, _ = _reach(head_hist[l, h], ph)
            row.append({'act_rms': float(act[h]), 'act_rel': float(act[h] / med),
                        'entropy': float(head_ent[l, h] / head_rows_n),
                        'self_mass': float(head_self[l, h] / head_rows_n),
                        'masked_mass': float(head_masked[l, h] / head_rows_n),
                        'reach_p50_h': r50, 'reach_p90_h': r90})
        heads_out.append(row)

    blocks_out = []
    for i in range(L):
        la, ha = frac_live(ffn_act[i] / n_done)
        blocks_out.append({'ffn_live_a': la, 'ffn_hot_a': ha,
                           'attn_rel': float(attn_rel[i] / n_done), 'ffn_rel': float(ffn_rel[i] / n_done),
                           'cos_io': float(cos_io[i] / n_done), 'tok_outlier': float(tok_out[i] / n_done),
                           'cka_prev': float(cka[i, i + 1] / n_done), 'cka_final': float(cka[i + 1, L] / n_done)})
    rv = resid_var / n_done
    live_r = int(live_mask(rv).sum())
    b0l, b0h = frac_live(bg0 / n_done)
    b1l, b1h = frac_live(bg1 / n_done)
    time_out = None
    if th is not None:
        tl_, th_ = frac_live(th / n_done)
        time_out = {'live': tl_, 'hot': th_}

    def dpct(a: np.ndarray) -> np.ndarray:
        return 100.0 * (a - base_sum) / max(base_sum, 1e-12)

    def ladder_rows(lad: dict[int, float], width: int) -> list[dict[str, float]]:
        return [{'k': k, 'frac': k / width, 'delta_pct': float(dpct(np.array(v)))}
                for k, v in sorted(lad.items())]

    layer_hist = head_hist.sum(1)
    spans, roll_rel = _bucket(roll_hist, avail_hist, ph)
    for rung in ladder:
        rung['delta_pct'] = (100.0 * (rung['trunc'] - rung['base']) / max(rung['base'], 1e-12)
                             if rung['n'] else float('nan'))
    context = {
        'final_reach': _reach(final_hist, ph), 'roll_reach': _reach(roll_hist, ph),
        'layer_reach_p90': [_reach(layer_hist[l], ph)[1] for l in range(L)],
        'lag_edges': spans, 'roll_density_rel': roll_rel,
        'final_density_rel': _bucket(final_hist, avail_hist, ph)[1],
        'ladder': ladder,
    }
    return {
        'n_samples': n_done, 'policy': policy, 'partition': 'val',
        'baseline_pinball': base_sum / n_done,
        'heads': heads_out, 'blocks': blocks_out,
        'cka': (cka / n_done).tolist(),
        'resid': {'live': live_r, 'live_frac': live_r / D, 'top_share': top_share(rv, RESID_TOP_K),
                  'var': rv.tolist(), 'pca_dims': pca_dims},
        'ladders': {'d_model': ladder_rows(lad_d, D), 'ffn': ladder_rows(lad_f, arch.ffn_dim),
                    'bg_head': ladder_rows(lad_b, arch.bg_head_hidden)},
        'bg_head': {'live0': b0l, 'hot0': b0h, 'live1': b1l, 'hot1': b1h},
        'time_head': time_out,
        'spline': {'state_dev': np.sqrt(sp_state_num / max(sp_state_den, 1e-12)).tolist(),
                   'median_dev': np.sqrt(sp_med_num / np.maximum(sp_med_den, 1e-12)).tolist(),
                   'spread_dev': np.sqrt(sp_spr_num / np.maximum(sp_spr_den, 1e-12)).tolist()},
        'ablation': {'heads': dpct(abl_heads).tolist(), 'attn': dpct(abl_attn).tolist(),
                     'ffn': dpct(abl_ffn).tolist(), 'block': dpct(abl_block).tolist()},
        'context': context,
    }


# ------------------------------------------------------------------------------ report
def hr(c: str = '─', n: int = 100) -> str:
    return c * n


def section(title: str, how: str = '') -> None:
    print()
    print(hr())
    print(f"  {title}")
    if how:
        print(f"  {how}")
    print(hr())


def print_report(
    path: Path, ckpt: dict[str, Any], arch: Arch, param_count: int,
    rows: list[ParamRow], res: dict[str, Any], data: dict[str, Any] | None,
    top: int, using_ema: bool, opt_notes: list[str],
) -> None:
    verdicts: list[Verdict] = res['verdicts']
    layers: list[LayerRow] = res['layers']
    heads: list[HeadRow] = res['heads']
    spectra = res['spectra']
    tc = ckpt.get('training_config', {})
    L, H = arch.n_layers, arch.n_heads
    ph = arch.patch_hours

    print(hr('═'))
    print(f"  T1DMAI model health — {path}")
    print(hr('═'))
    step = ckpt.get('step', '?')
    total = tc.get('total_steps')
    warm = tc.get('warmup_steps')
    prog = f" of {total} ({step / total:.0%})" if isinstance(step, int) and total else ''
    phase = ''
    if isinstance(step, int) and warm and total:
        phase = '  phase: warmup' if step < warm else f'  phase: cosine decay ({(step - warm) / max(total - warm, 1):.0%} through)'
    print(f"  step {step}{prog}{phase}")
    print(f"  weights: {'EMA shadow' if using_ema else 'live'}   params {param_count:,} ({param_count / 1e6:.2f}M)   "
          f"arch_version {ckpt.get('arch_version', '?')}   loss_schema {ckpt.get('loss_schema', '?')}")
    bvl, bvs = ckpt.get('best_val_loss'), ckpt.get('best_val_step')
    if bvl is not None:
        print(f"  best_val_loss {bvl:.5f} @ step {bvs}   masked_channel_policy {checkpoint_masked_channel_policy(ckpt)}")
    av = arch.view()
    print("  arch: " + "  ".join(f"{k}={v}" for k, v in av.items())
          + f"  N_SPREADS={arch.n_spreads}  TIME_HEAD={arch.time_hidden}→{arch.time_bins} bins")
    if tc.get('min_context_patches'):
        print(f"  context {tc['min_context_patches']}–{tc['max_context_patches']} patches "
              f"({tc['min_context_patches'] * ph:.0f}–{tc['max_context_patches'] * ph:.0f} h), "
              f"horizon {tc.get('prediction_patches')} patches, M={tc.get('max_masked_patches')} slots, "
              f"spans {tc.get('mask_span_lengths')}, right-edge quota {tc.get('mask_right_edge_quota')}")
    print(f"  cache {tc.get('cache_path', '?')}   batch {tc.get('batch_size', '?')}   "
          f"muon_lr {tc.get('muon_lr', '?')}   adam_lr {tc.get('adam_lr', '?')}")
    for note in opt_notes:
        print(f"  [opt] {note}")
    if res['undertrained']:
        print("  ⚠ checkpoint appears UNDERTRAINED (main matrices at init std) — capacity verdicts suppressed to KEEP")
    elif isinstance(step, int) and total and step < 0.5 * total:
        print(f"  ⚠ checkpoint is {step / total:.0%} through its schedule — verdicts describe this snapshot; "
              "late-schedule weight decay and LR decay move them")

    section("LEGEND")
    for k, v in LEGEND:
        print(f"  {k:22s} {v}")

    # ---- per-param ----------------------------------------------------------------
    section("PARAMETER HEALTH",
            "drift = std/init_std (norms: mean |w−1|; biases: std); opt-act = RMS optimizer buffer; "
            "rel = ×median of its optimizer; snr = |exp_avg|/√exp_avg_sq (AdamW)")
    print(f"  {'parameter':32s} {'shape':>13s} {'params':>9s} {'std':>8s} {'drift':>7s} "
          f"{'opt':>4s} {'opt-act':>9s} {'rel':>6s} {'snr':>5s}  flags")
    rows_sorted = sorted(rows, key=lambda r: r.numel, reverse=True)
    for r in rows_sorted[:top]:
        flags = (['STALE'] if r.stale else []) + (['near-init'] if r.near_init else [])
        print(f"  {r.name:32s} {str(r.shape):>13s} {r.numel:>9,} {r.std:>8.4f} {_f(r.drift):>7s} "
              f"{r.opt:>4s} {_f(r.opt_activity, '.2e'):>9s} {_f(r.opt_rel):>6s} {_f(r.snr):>5s}  {' '.join(flags)}")
    if len(rows) > top:
        print(f"  … {len(rows) - top} more (raise --top to see all)")
    n_stale = sum(1 for r in rows if r.stale)
    n_near = sum(1 for r in rows if r.near_init)
    by_kind: dict[str, list[float]] = {}
    for r in rows:
        if not math.isnan(r.drift):
            by_kind.setdefault(r.kind, []).append(r.drift)
    print(f"  → {n_stale} stale tensor(s), {n_near} near-init of {len(rows)}; median drift by kind: "
          + ', '.join(f'{k} {float(np.median(v)):.2f}' for k, v in by_kind.items()))

    # ---- per-layer ----------------------------------------------------------------
    section("PER-LAYER",
            "util = rank-util (attn: mean q/k/v/o; ffn: mean w1/w3/w2); write = out-projection std ÷ init std; "
            "ffn live/hot by weight (w) and activation (a); Δ = pinball change when that part is ablated")
    hdr = (f"  {'L':>2s} {'util_a':>6s} {'util_f':>6s} {'wr_a':>5s} {'wr_f':>5s} "
           f"{'live_w':>7s} {'hot_w':>5s} {'top8':>5s}")
    if data:
        hdr += (f" {'live_a':>7s} {'hot_a':>5s} {'‖attn‖':>6s} {'‖ffn‖':>6s} {'cos':>5s} {'cka_p':>5s} {'cka_F':>5s} "
                f"{'tok×':>5s} {'Δattn':>7s} {'Δffn':>7s} {'Δblock':>7s}")
    print(hdr + "  flags")
    for l in layers:
        line = (f"  {l.layer:>2d} {l.util_attn:>6.2f} {l.util_ffn:>6.2f} {l.write_attn:>5.1f} {l.write_ffn:>5.1f} "
                f"{l.ffn_live_w:>3d}/{arch.ffn_dim:<3d} {l.ffn_hot_w:>5d} {l.ffn_top8_share_w:>5.0%}")
        if data:
            line += (f" {l.ffn_live_a:>3d}/{arch.ffn_dim:<3d} {l.ffn_hot_a:>5d} {l.attn_rel:>6.2f} {l.ffn_rel:>6.2f} "
                     f"{l.cos_io:>5.2f} {l.cka_prev:>5.2f} {l.cka_final:>5.2f} {l.tok_outlier:>5.1f} "
                     f"{_pct(l.ablate_attn_pct):>7s} {_pct(l.ablate_ffn_pct):>7s} {_pct(l.ablate_block_pct):>7s}")
        print(line + "  " + ' '.join(l.flags))
    if data:
        print("  ‖attn‖ = ‖attn_out‖/‖x_in‖, ‖ffn‖ = ‖ffn_out‖/‖x_mid‖; cos = cos(x_in, x_out); "
              "cka_p = CKA with the previous layer's output, cka_F with the final; tok× = max/median token norm")
        ck = np.array(data['cka'])
        print("  CKA matrix (rows/cols: embed, block 0..N−1):")
        for a in range(L + 1):
            print("    " + ' '.join(f'{ck[a, b]:.2f}' for b in range(L + 1)))

    # ---- per-head -----------------------------------------------------------------
    section("PER-HEAD",
            "w = ‖W_v[h]‖·‖W_o[:,h]‖ (rel = ×layer median); act = RMS head output (rel = ×layer median); "
            "ent = normalized attention entropy on masked query rows; self = mass on own patch; "
            "mskd = mass on masked columns; reach p50/p90 = hours holding that share of mass; Δ = pinball change when zeroed")
    hdr = f"  {'L.h':>5s} {'w':>7s} {'w_rel':>5s}"
    if data:
        hdr += f" {'act':>6s} {'a_rel':>5s} {'ent':>5s} {'self':>5s} {'mskd':>5s} {'p50h':>5s} {'p90h':>5s} {'Δ':>7s}"
    print(hdr + "  flags")
    for h in heads:
        line = f"  {h.layer:>2d}.{h.head:<2d} {h.w_strength:>7.3f} {h.w_rel:>5.2f}"
        if data:
            line += (f" {h.act_rms:>6.3f} {h.act_rel:>5.2f} {h.entropy:>5.2f} {h.self_mass:>5.2f} {h.masked_mass:>5.2f} "
                     f"{h.reach_p50_h:>5.1f} {h.reach_p90_h:>5.1f} {_pct(h.ablate_pct):>7s}")
        print(line + "  " + ' '.join(h.flags))

    # ---- spectral -----------------------------------------------------------------
    section("SPECTRAL CAPACITY", "util = eff_rank / min(in,out); tail% = singular values below 1% of σ_max")
    print(f"  {'matrix':32s} {'shape':>13s} {'stable_rk':>10s} {'eff_rank':>9s} {'util':>6s} {'tail%':>6s}")
    for name in sorted(spectra):
        s = spectra[name]
        sh = tuple(ckpt['_sd_shapes'].get(name, ()))
        print(f"  {name:32s} {str(sh):>13s} {s['stable_rank']:>10.1f} {s['eff_rank']:>9.1f} "
              f"{s['util']:>6.2f} {s['tail_frac'] * 100:>5.0f}%")

    # ---- data summary -------------------------------------------------------------
    if data:
        section(f"ACTIVATION PASS  ({data['n_samples']} cached windows, {data['partition']} partition, "
                f"policy {data['policy']}, baseline pinball {data['baseline_pinball']:.5f})")
        if data['n_samples'] < 64:
            print(f"  ⚠ {data['n_samples']} windows: ablation, ladder and reach figures are noisy below ~64; "
                  "the context ladder scores only the ~half of windows with a right-edge span")
        r = data['resid']
        print(f"  residual dims with variance {r['live']}/{arch.d_model}; top-{RESID_TOP_K} dims carry {r['top_share']:.0%} of the variance")
        b = data['bg_head']
        print(f"  BG head hidden live/hot: layer0 {b['live0']}/{arch.bg_head_hidden} ({b['hot0']} hot), "
              f"layer1 {b['live1']}/{arch.bg_head_hidden} ({b['hot1']} hot)")
        if data['time_head']:
            t = data['time_head']
            print(f"  time head hidden live/hot: {t['live']}/{arch.time_hidden} ({t['hot']} hot)")
        sp = data['spline']
        print("  spline (per step 0..S−1): state departure from the own-patch state "
              + ', '.join(f'{x:.0%}' for x in sp['state_dev']))
        print("    median-delta change vs a flat own-patch decode "
              + ', '.join(f'{x:.0%}' for x in sp['median_dev'])
              + "; spread columns " + ', '.join(f'{x:.0%}' for x in sp['spread_dev']))
        c = data['context']
        print(f"  attention reach on masked rows (h to 50/90/99% of mass): final layer "
              f"{c['final_reach'][0]:.1f}/{c['final_reach'][1]:.1f}/{c['final_reach'][2]:.1f}, "
              f"rollout {c['roll_reach'][0]:.1f}/{c['roll_reach'][1]:.1f}/{c['roll_reach'][2]:.1f}")
        print(f"  {'lag bucket':14s} " + ' '.join(f'{lo:>3.0f}–{hi:<4.0f}' for lo, hi in c['lag_edges']))
        print(f"  {'rollout ×near':14s} " + ' '.join(f'{_f(v, ".3f"):>8s}' for v in c['roll_density_rel']))
        print(f"  {'final   ×near':14s} " + ' '.join(f'{_f(v, ".3f"):>8s}' for v in c['final_density_rel']))
        print("  (attention mass per available key in the bucket, relative to the ≤1 h bucket)")
        print("  context truncation ladder — keep only the last k hours of context, Δ pinball on the forecast-zone slots:")
        for rung in c['ladder']:
            if rung['n']:
                print(f"    {rung['hours']:>4d} h ({rung['patches']:>3d} patches)  Δ {rung['delta_pct']:+7.2f}%   n={rung['n']}")
        ld = data['ladders']
        print("  width ladders — keep only k of the width everywhere, Δ pinball vs the full model:")
        print(f"    D_MODEL (top-k PCs of the stream at every block): {_ladder_line(ld['d_model'], arch.d_model)}")
        print(f"    FFN_DIM (top-k units per layer):                  {_ladder_line(ld['ffn'], arch.ffn_dim)}")
        print(f"    BG_HEAD_HIDDEN (top-k units per layer):           {_ladder_line(ld['bg_head'], arch.bg_head_hidden)}")
        if data['resid'].get('pca_dims'):
            pd_ = data['resid']['pca_dims']
            print(f"    final-block PCA: {pd_['p90']}/{arch.d_model} dims hold 90% of the variance, {pd_['p99']} hold 99%")
        print("  ablation Δ pinball, one part zeroed at a time (positive = the part helps):")
        ab = data['ablation']
        print(f"    {'block':>5s} " + ' '.join(f'{("h%d" % h):>7s}' for h in range(H)) + f" {'attn':>8s} {'ffn':>8s} {'skip':>8s}")
        for i in range(L):
            print(f"    {i:>5d} " + ' '.join(f'{ab["heads"][i][h]:>+7.2f}' for h in range(H))
                  + f" {ab['attn'][i]:>+8.2f} {ab['ffn'][i]:>+8.2f} {ab['block'][i]:>+8.2f}")

    # ---- verdicts -----------------------------------------------------------------
    print()
    print(hr('═'))
    print("  CAPACITY VERDICTS  (resize_model.py knobs)")
    print(hr('═'))
    for v in verdicts:
        print(f"  {v.knob:16s} {v.current:32s} {v.verdict}")
        for e in v.evidence:
            print(f"      · {e}")
        if v.suggest:
            if v.suggest.lstrip().startswith('--'):
                print(f"      → python resize_model.py {v.suggest}")
            else:
                print(f"      → {v.suggest}")
    print(hr('═'))
    shrink = [v.knob for v in verdicts if v.verdict == 'SHRINK']
    grow = [v.knob for v in verdicts if v.verdict == 'GROW']
    print(f"  SHRINK: {', '.join(shrink) if shrink else '(none)'}")
    print(f"  GROW:   {', '.join(grow) if grow else '(none)'}")
    if not data:
        print("  (weight-only audit — pass --data N for activation, attention-reach, block and ablation evidence)")
    print(hr('═'))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit a T1DMAI checkpoint for under-/over-provisioned parts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--checkpoint', type=str, default=None,
                    help='checkpoint .pt (default: best in --ckpt-dir).')
    ap.add_argument('--ckpt-dir', type=str, default='checkpoints',
                    help='directory to search for the best checkpoint.')
    ap.add_argument('--ema', action='store_true',
                    help='audit the EMA shadow weights instead of the live weights.')
    ap.add_argument('--data', type=int, default=0, metavar='N',
                    help='run N cached windows through the model: activations, attention reach, '
                         'block influence, spline, ablations, context ladder (0 = off).')
    ap.add_argument('--top', type=int, default=24,
                    help='rows to show in the per-parameter table.')
    ap.add_argument('--json', type=str, default=None,
                    help='also write the full findings as JSON to this path.')
    ap.add_argument('--device', type=str, default='cpu',
                    help="device for the --data pass ('cuda' or 'cpu').")
    args = ap.parse_args()

    path = Path(args.checkpoint) if args.checkpoint else find_best_checkpoint(Path(args.ckpt_dir))
    print(f"[load] {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    sd_key = 'model_ema_state_dict' if args.ema else 'model_state_dict'
    if sd_key not in ckpt:
        raise KeyError(f"{sd_key} not in checkpoint; keys: {list(ckpt.keys())}")
    sd = ckpt[sd_key]
    ckpt['_sd_shapes'] = {k: tuple(v.shape) for k, v in ckpt['model_state_dict'].items()}

    arch = derive_arch(ckpt['model_state_dict'])
    param_count = sum(v.numel() for v in sd.values())

    import config as _cfg
    live = (_cfg.D_MODEL, _cfg.N_LAYERS, _cfg.N_HEADS, _cfg.FFN_DIM, _cfg.BG_HEAD_HIDDEN, _cfg.PATCH_SIZE)
    derived = (arch.d_model, arch.n_layers, arch.n_heads, arch.ffn_dim, arch.bg_head_hidden, arch.patch_size)
    if live != derived:
        print(f"[warn] config.py {live} != checkpoint-derived {derived} (auditing the checkpoint's architecture).")
    if ckpt.get('arch_version') != _cfg.ARCH_VERSION:
        print(f"[warn] checkpoint arch_version {ckpt.get('arch_version')!r} != config {_cfg.ARCH_VERSION!r}")
    bg_init = float(getattr(_cfg, 'BG_HEAD_INIT_SCALE', DEFAULT_BG_HEAD_INIT_SCALE))
    time_init = float(getattr(_cfg, 'TIME_PROBE_INIT_SCALE', DEFAULT_TIME_PROBE_INIT_SCALE))

    opt, opt_notes = map_optimizer_activity(ckpt['model_state_dict'], ckpt, arch)
    rows = analyze_params(sd, arch, opt, bg_init, time_init, weights_only_ema=args.ema)

    data = None
    if args.data > 0:
        data = run_data_pass(arch, ckpt, sd, args.data, torch.device(args.device))

    res = build_verdicts(sd, arch, rows, ckpt, data)
    print_report(path, ckpt, arch, param_count, rows, res, data, args.top, args.ema, opt_notes)

    if args.json:
        payload = {
            'checkpoint': str(path), 'step': ckpt.get('step'),
            'arch_version': ckpt.get('arch_version'),
            'masked_channel_policy': checkpoint_masked_channel_policy(ckpt),
            'using_ema': args.ema, 'param_count': param_count,
            'undertrained': res['undertrained'],
            'optimizer_notes': opt_notes,
            'arch': asdict(arch), 'spectra': res['spectra'],
            'layers': [asdict(l) for l in res['layers']],
            'heads': [asdict(h) for h in res['heads']],
            'data': data,
            'params': [asdict(r) for r in rows],
            'verdicts': [asdict(v) for v in res['verdicts']],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=float))
        print(f"[json] wrote {args.json}")


if __name__ == '__main__':
    main()
