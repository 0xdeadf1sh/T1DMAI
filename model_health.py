#!/usr/bin/env python3
"""model_health.py — capacity / staleness audit of a trained T1DMAI checkpoint.

Purpose
-------
Inspect the best checkpoint in ``checkpoints/`` and decide, for every
architecture knob that ``resize_model.py`` exposes, whether that part of the
network is

  * **over-provisioned** — low-rank weight maps, dead/ignored units, or
    parameters whose optimizer state shows they never received gradient — and
    can be **shrunk** with little expected loss of quality, or
  * **saturated** — near-full-rank maps, no dead units, evenly used heads — and
    is a candidate to **grow** to lift quality.

The knobs mirror ``resize_model.py`` exactly: ``D_MODEL``, ``N_LAYERS``,
``N_HEADS``, ``FFN_DIM`` (``--ffn-mult``), ``BG_HEAD_HIDDEN``
(``--bg-head-hidden-mult``), ``PATCH_SIZE``, ``MIN/MAX_CONTEXT_PATCHES``.  The
report ends with a per-knob verdict table and the concrete ``resize_model.py``
command that would enact each suggestion.

How it reads the model
----------------------
Every architecture dimension is **derived from the checkpoint's tensor shapes**
(not from the live ``config.py``), so the script audits whatever architecture
produced the weights — any configuration ``resize_model.py`` can emit.  The
frozen input layout (``N_INPUT_FEATURES = 3``, no mask bits) is the only
structural assumption; ``PATCH_SIZE`` is recovered from ``PATCH_DIM``.

Signals (all data-free unless ``--data`` is passed)
---------------------------------------------------
  * **Optimizer staleness.**  AdamW ``exp_avg_sq`` (1-D params) and Muon
    ``momentum_buffer`` (2-D weights) are EMAs of (squared) gradients.  A
    parameter whose buffer is ~0 has effectively stopped receiving gradient —
    it is stale/dead.  Optimizer state is mapped back to parameter names by
    replaying ``train._build_optimizers``' ndim partition over
    ``model.named_parameters()`` order.
  * **Drift from init.**  Each weight's std is compared to the analytic init std
    of ``model._init_weights`` (width-aware ``base_std``, the residual rescale,
    the tiny BG-head-final scale); a ratio near 1.0 means the tensor barely
    moved.  Norm weights are compared to their 1.0 init, biases to 0.
  * **Spectral capacity.**  Per weight matrix: stable rank
    ``‖W‖_F²/σ_max²`` and entropy effective rank ``exp(H(σ/Σσ))``; their ratio
    to ``min(in, out)`` is the map's rank utilization.  Low ⇒ shrinkable.
  * **Per-unit usage.**  FFN hidden units, BG-head hidden units, and attention
    heads are scored by their in/out weight-norm product; units far below the
    median are "ignored" and inflate the corresponding width.
  * **Attention reach.**  ALiBi slopes give each head's effective temporal
    horizon, compared against ``MIN/MAX_CONTEXT_PATCHES``.

With ``--data N`` the script additionally streams ``N`` cached simulator
samples through the model with forward hooks and measures *activation* dead
fractions (FFN / BG-head units that fire ~never), residual-stream per-dimension
variance, and per-head output-activation norm — corroborating the weight-only
verdicts with the model's actual behaviour.

Usage::

    python model_health.py                       # audit checkpoints/t1dmai_best.pt
    python model_health.py --checkpoint X.pt     # a specific checkpoint
    python model_health.py --ema                 # audit the EMA shadow weights
    python model_health.py --data 128            # add the activation pass (needs the cache)
    python model_health.py --json out.json       # also dump machine-readable findings
    python model_health.py --top 40              # list more rows in the per-param table

The verdicts are heuristic capacity signals, not guarantees: a SHRINK means the
evidence says the width is underused at this checkpoint, not that quality is
provably invariant to the cut.  Re-run after retraining; treat them as a guide
for the next ``resize_model.py`` sweep.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Frozen structural assumptions (see CLAUDE.md "FROZEN index map") and the init
# scheme of model._init_weights.  These are NOT resize knobs; PATCH_SIZE is the
# only patch-layout quantity resize_model.py changes, and it multiplies these.
# ---------------------------------------------------------------------------
N_INPUT_FEATURES = 3              # [bg (risk-space z), carb, insulin]
# model._init_weights: base_std = 0.02 * sqrt(512 / D_MODEL); residual-branch
# output projections (attn.w_o, ffn.w2) use base_std / sqrt(2 * N_LAYERS); the
# BG-head final layer uses BG_HEAD_INIT_SCALE.  Mirror them so the drift metric
# can reconstruct the init std without re-seeding.
INIT_BASE_STD_ANCHOR = 0.02
INIT_BASE_STD_ANCHOR_DMODEL = 512.0
RESIDUAL_WRITES_PER_BLOCK = 2     # attn out + ffn out
DEFAULT_BG_HEAD_INIT_SCALE = 1e-2
DEFAULT_N_SPREADS = 3
VALID_HEAD_DIMS = (16, 32, 64, 128)

# ---------------------------------------------------------------------------
# Verdict thresholds (heuristic).  Tunable; printed in the report header.
# ---------------------------------------------------------------------------
RANK_UTIL_SHRINK = 0.50          # mean rank-utilization below this ⇒ width over-provisioned
RANK_UTIL_GROW = 0.85            # rank-utilization above this (flat spectrum) ⇒ saturated
ACTIVE_FRAC_SHRINK = 0.65        # fraction of live units below this ⇒ shrink the width
ACTIVE_FRAC_GROW = 0.97          # almost every unit live ⇒ candidate to grow
DEAD_UNIT_REL = 0.05             # a unit below 5% of the median strength is "ignored"
HEAD_UNDERUSE_REL = 0.25         # a head below 25% of the median strength is "underused"
STALE_OPT_REL = 0.01             # optimizer activity below 1% of the median ⇒ stale
STALE_OPT_ABS = 1e-10            # absolute optimizer-activity floor ⇒ dead
DRIFT_NEAR_INIT = 0.08           # |std/init_std - 1| below this ⇒ "near-init" (barely trained)
TAIL_REL = 0.01                  # singular values below 1% of σ_max counted as spectral tail
ALIBI_REACH_NATS = 5.0           # attention bias of -5 nats ≈ horizon cutoff for a head


# ===========================================================================
# Architecture, derived purely from the checkpoint
# ===========================================================================
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
    bg_head_linears: list[str]   # ordered bg_head.{k}.weight names

    def view(self) -> dict[str, int]:
        return {
            'D_MODEL': self.d_model, 'N_LAYERS': self.n_layers,
            'N_HEADS': self.n_heads, 'HEAD_DIM': self.head_dim,
            'FFN_DIM': self.ffn_dim, 'BG_HEAD_HIDDEN': self.bg_head_hidden,
            'PATCH_SIZE': self.patch_size,
        }


def derive_arch(sd: dict[str, torch.Tensor]) -> Arch:
    """Recover every architecture dimension from state-dict tensor shapes.

    Args:
        sd: model state dict (name -> tensor).

    Returns:
        Arch with all dimensions; cross-checked for internal consistency.
    """
    d_model, patch_dim = sd['patch_embed.weight'].shape
    block_ids = sorted({int(k.split('.')[1]) for k in sd if k.startswith('blocks.')})
    n_layers = len(block_ids)
    assert block_ids == list(range(n_layers)), f"non-contiguous block ids: {block_ids}"

    n_heads = sd['blocks.0.attn.alibi_slopes'].shape[0]
    head_dim = sd['blocks.0.attn.q_norm.weight'].shape[0]
    assert head_dim * n_heads == d_model, (
        f"HEAD_DIM({head_dim}) * N_HEADS({n_heads}) != D_MODEL({d_model})"
    )
    ffn_dim = sd['blocks.0.ffn.w1.weight'].shape[0]

    bg_linears = sorted(
        (k for k in sd if k.startswith('bg_head.') and k.endswith('.weight')),
        key=lambda k: int(k.split('.')[1]),
    )
    bg_head_hidden = sd[bg_linears[0]].shape[0]
    head_out_width = sd[bg_linears[-1]].shape[0]

    # PATCH_DIM = PATCH_SIZE * N_INPUT_FEATURES (frozen layout, no mask bits).
    assert patch_dim % N_INPUT_FEATURES == 0, (
        f"PATCH_DIM={patch_dim} inconsistent with the frozen "
        f"{N_INPUT_FEATURES}-feature layout"
    )
    patch_size = patch_dim // N_INPUT_FEATURES
    # The smooth-basis BG head's final Linear emits BG_HEAD_STEP_BASIS_DIM (=K)
    # coefficients per output channel, expanded across the within-patch steps by a
    # fixed basis: head_out_width = K * (1 + 2*N_SPREADS) — NOT PATCH_SIZE * (...).
    import config as _cfg
    k_basis = _cfg.BG_HEAD_STEP_BASIS_DIM
    assert head_out_width % k_basis == 0, (
        f"head out width {head_out_width} not divisible by "
        f"BG_HEAD_STEP_BASIS_DIM {k_basis}"
    )
    per_channel = head_out_width // k_basis                 # == 1 + 2*N_SPREADS
    assert per_channel % 2 == 1, (
        f"head per-channel width {per_channel} should be 1 + 2*N_SPREADS (odd)")
    n_spreads = (per_channel - 1) // 2

    return Arch(
        d_model=d_model, patch_dim=patch_dim, n_layers=n_layers, n_heads=n_heads,
        head_dim=head_dim, ffn_dim=ffn_dim, bg_head_hidden=bg_head_hidden,
        patch_size=patch_size, n_spreads=n_spreads, head_out_width=head_out_width,
        bg_head_linears=bg_linears,
    )


# ===========================================================================
# Checkpoint loading & optimizer-state mapping
# ===========================================================================
def find_best_checkpoint(ckpt_dir: Path) -> Path:
    """Pick the 'best' checkpoint: prefer t1dmai_best.pt, then latest step."""
    p = ckpt_dir / 't1dmai_best.pt'
    if p.exists():
        return p
    steps = sorted(ckpt_dir.glob('t1dmai_step_*.pt'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    if steps:
        return steps[-1]
    raise FileNotFoundError(f"no T1DMAI checkpoint found in {ckpt_dir}")


def map_optimizer_activity(
    sd: dict[str, torch.Tensor], ckpt: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Map each parameter name to its optimizer-state gradient-activity scalars.

    Replays ``train._build_optimizers``' partition (ndim>=2 -> Muon, else AdamW)
    over ``named_parameters()`` order, which equals the construction order, then
    zips it against ``param_groups[*]['params']`` (the same order PyTorch used to
    key the saved ``state`` dict).

    Args:
        sd: model state dict (gives names + ndim, in order).
        ckpt: full checkpoint dict (gives the two optimizer state dicts).

    Returns:
        name -> {'opt', 'activity', 'first_moment', 'step'}.  ``activity`` is the
        RMS of exp_avg_sq (AdamW) or of the momentum buffer (Muon): the historical
        gradient energy reaching that parameter.  Empty if no optimizer state.
    """
    out: dict[str, dict[str, float]] = {}
    names = list(sd.keys())
    muon_names = [n for n in names if sd[n].ndim >= 2]
    adam_names = [n for n in names if sd[n].ndim < 2]

    muon = ckpt.get('muon_optimizer_state_dict')
    if muon is not None:
        idxs = muon['param_groups'][0]['params']
        if len(idxs) == len(muon_names):
            for idx, nm in zip(idxs, muon_names):
                buf = muon['state'].get(idx, {}).get('momentum_buffer')
                if buf is not None:
                    out[nm] = {
                        'opt': 'muon',
                        'activity': float(buf.float().pow(2).mean().sqrt()),
                        'first_moment': float(buf.float().abs().mean()),
                        'step': float('nan'),
                    }

    adam = ckpt.get('adam_optimizer_state_dict')
    if adam is not None:
        idxs = adam['param_groups'][0]['params']
        if len(idxs) == len(adam_names):
            for idx, nm in zip(idxs, adam_names):
                st = adam['state'].get(idx, {})
                sq = st.get('exp_avg_sq')
                if sq is not None:
                    ea = st.get('exp_avg')
                    out[nm] = {
                        'opt': 'adam',
                        'activity': float(sq.float().mean().sqrt()),
                        'first_moment': float(ea.float().abs().mean()) if ea is not None else float('nan'),
                        'step': float(st.get('step', float('nan'))),
                    }
    return out


# ===========================================================================
# Init-std reconstruction (for the drift-from-init signal)
# ===========================================================================
def init_std_for(name: str, arch: Arch, bg_head_init_scale: float) -> tuple[str, float]:
    """Return (kind, expected_init_std) for a parameter, mirroring model._init_weights.

    Args:
        name: parameter name.
        arch: derived architecture.
        bg_head_init_scale: BG_HEAD_INIT_SCALE (final-layer std).

    Returns:
        (kind, init_std).  kind in {'linear', 'residual', 'head_final', 'norm',
        'bias', 'alibi'}; init_std is the std of the init distribution (NaN for
        kinds with a non-Gaussian reference handled separately).
    """
    base_std = INIT_BASE_STD_ANCHOR * math.sqrt(INIT_BASE_STD_ANCHOR_DMODEL / arch.d_model)
    residual_std = base_std / math.sqrt(RESIDUAL_WRITES_PER_BLOCK * arch.n_layers)

    if name.endswith('alibi_slopes'):
        return 'alibi', float('nan')
    if name.endswith('.bias'):
        return 'bias', 0.0
    if name.endswith('norm.weight') or name.endswith('norm1.weight') or name.endswith('norm2.weight'):
        return 'norm', 1.0
    if name == arch.bg_head_linears[-1]:
        return 'head_final', bg_head_init_scale
    if name.endswith('attn.w_o.weight') or name.endswith('ffn.w2.weight'):
        return 'residual', residual_std
    if name.endswith('.weight'):
        return 'linear', base_std
    return 'other', float('nan')


# ===========================================================================
# Spectral analysis
# ===========================================================================
def spectral(W: np.ndarray) -> dict[str, float]:
    """Singular-value capacity metrics for a 2-D weight matrix.

    Args:
        W: (out, in) weight matrix.

    Returns:
        dict with stable_rank, eff_rank (entropy), util (eff_rank / min(out,in)),
        tail_frac (fraction of σ below TAIL_REL·σ_max), and sigma_max.
    """
    s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return {'stable_rank': 0.0, 'eff_rank': 0.0, 'util': 0.0, 'tail_frac': 1.0, 'sigma_max': 0.0}
    smax = float(s[0])
    stable = float((s ** 2).sum() / smax ** 2)
    p = s / s.sum()
    eff = float(np.exp(-(p * np.log(p)).sum()))
    util = eff / min(W.shape)
    tail = float((s < TAIL_REL * smax).mean())
    return {'stable_rank': stable, 'eff_rank': eff, 'util': util,
            'tail_frac': tail, 'sigma_max': smax}


def col_norms(W: np.ndarray) -> np.ndarray:
    """L2 norm of each column of W (shape (in,))."""
    return np.linalg.norm(W, axis=0)


def row_norms(W: np.ndarray) -> np.ndarray:
    """L2 norm of each row of W (shape (out,))."""
    return np.linalg.norm(W, axis=1)


def active_fraction(strength: np.ndarray) -> tuple[int, float]:
    """Count units whose strength clears DEAD_UNIT_REL × median(strength).

    Args:
        strength: per-unit non-negative strength scores.

    Returns:
        (n_active, active_fraction).
    """
    med = float(np.median(strength))
    if med <= 0:
        # Degenerate: fall back to a fraction-of-max floor.
        ref = float(strength.max()) or 1.0
        live = strength >= DEAD_UNIT_REL * ref
    else:
        live = strength >= DEAD_UNIT_REL * med
    n = int(live.sum())
    return n, n / strength.size


# ===========================================================================
# Per-parameter health table
# ===========================================================================
@dataclass
class ParamRow:
    name: str
    shape: tuple[int, ...]
    numel: int
    kind: str
    std: float
    drift: float           # std / init_std (NaN where not meaningful)
    near_init: bool
    opt: str
    opt_activity: float    # absolute optimizer gradient-energy
    opt_rel: float         # relative to the median within its optimizer
    stale: bool


def analyze_params(
    sd: dict[str, torch.Tensor], arch: Arch, opt: dict[str, dict[str, float]],
    bg_head_init_scale: float, weights_only_ema: bool,
) -> list[ParamRow]:
    """Build the per-parameter health rows (drift + optimizer staleness)."""
    rows: list[ParamRow] = []
    # Median optimizer activity per optimizer, for the relative-staleness test.
    med_act = {'muon': [], 'adam': []}
    for nm, info in opt.items():
        med_act[info['opt']].append(info['activity'])
    med = {k: (float(np.median(v)) if v else 0.0) for k, v in med_act.items()}

    for nm, t in sd.items():
        w = t.detach().cpu().float().numpy()
        kind, istd = init_std_for(nm, arch, bg_head_init_scale)
        std = float(w.std())
        if kind in ('linear', 'residual', 'head_final') and istd > 0:
            drift = std / istd
            near = abs(drift - 1.0) < DRIFT_NEAR_INIT
        elif kind == 'norm':
            drift = float(np.abs(w - 1.0).mean())   # mean drift from 1.0
            near = drift < DRIFT_NEAR_INIT
        elif kind == 'bias':
            drift = std                               # spread away from 0
            near = std < 1e-6
        else:
            drift, near = float('nan'), False

        info = opt.get(nm)
        if info and not weights_only_ema:
            o, act = info['opt'], info['activity']
            rel = act / med[o] if med[o] > 0 else float('inf')
            stale = (act < STALE_OPT_ABS) or (rel < STALE_OPT_REL)
            # "near-init" only signals *untrained* when the optimizer also shows
            # little gradient energy; a high-activity tensor whose std merely sits
            # near init is trained, not stale (e.g. a matrix that rotated in place).
            near = near and rel < 0.5
        else:
            o, act, rel, stale = '-', float('nan'), float('nan'), False

        rows.append(ParamRow(
            name=nm, shape=tuple(w.shape), numel=int(w.size), kind=kind,
            std=std, drift=drift, near_init=near, opt=o,
            opt_activity=act, opt_rel=rel, stale=stale,
        ))
    return rows


# ===========================================================================
# Per-knob capacity verdicts
# ===========================================================================
@dataclass
class Verdict:
    knob: str
    current: str
    verdict: str           # SHRINK / KEEP / GROW / NOTE
    signal: str
    suggest: str           # resize_model.py command or ''
    detail: dict[str, Any] = field(default_factory=dict)


def _suggest_heads(arch: Arch, grow: bool) -> tuple[int, str]:
    """Suggest a valid N_HEADS (keeping HEAD_DIM in VALID_HEAD_DIMS, D_MODEL % heads == 0)."""
    d = arch.d_model
    valid = [h for h in range(1, d + 1) if d % h == 0 and (d // h) in VALID_HEAD_DIMS]
    valid.sort()
    cur = arch.n_heads
    if grow:
        bigger = [h for h in valid if h > cur]
        return (bigger[0], f'--heads {bigger[0]}') if bigger else (cur, '')
    smaller = [h for h in valid if h < cur]
    return (smaller[-1], f'--heads {smaller[-1]}') if smaller else (cur, '')


def _suggest_dmodel(arch: Arch, new_d: int) -> str:
    """Build a VALID ``--d-model`` command for a target width.

    resize_model.py requires ``D_MODEL % N_HEADS == 0`` and
    ``D_MODEL // N_HEADS ∈ VALID_HEAD_DIMS``.  Keep the current N_HEADS if the
    resulting head_dim is valid; otherwise keep the current HEAD_DIM by setting
    N_HEADS = new_d // head_dim.  Returns '' if no valid pairing exists.

    Args:
        arch: derived architecture.
        new_d: target D_MODEL.

    Returns:
        a resize_model.py flag string (e.g. '--d-model 128 --heads 8'), or ''.
    """
    if new_d <= 0:
        return ''
    if new_d % arch.n_heads == 0 and (new_d // arch.n_heads) in VALID_HEAD_DIMS:
        return f'--d-model {new_d}'
    if new_d % arch.head_dim == 0:
        h = new_d // arch.head_dim
        if h >= 1:
            return f'--d-model {new_d} --heads {h}'
    # Last resort: any valid (d, heads) at this width.
    for h in range(1, new_d + 1):
        if new_d % h == 0 and (new_d // h) in VALID_HEAD_DIMS:
            return f'--d-model {new_d} --heads {h}'
    return ''


def verdict_width(
    knob: str, dim: int, util: float, active_n: int, active_frac: float,
    n_stale: int, suggest_shrink: str, suggest_grow: str,
    data_active_frac: float | None = None,
) -> Verdict:
    """Combine rank-utilization + live-unit fraction into a width verdict."""
    da = f', act(data) {data_active_frac:.0%}' if data_active_frac is not None else ''
    sig = f'rank-util {util:.2f}, live units {active_n}/{dim} ({active_frac:.0%}){da}'
    if n_stale:
        sig += f', {n_stale} stale param(s)'
    eff_active = data_active_frac if data_active_frac is not None else active_frac
    if util < RANK_UTIL_SHRINK or eff_active < ACTIVE_FRAC_SHRINK:
        return Verdict(knob, str(dim), 'SHRINK', sig, suggest_shrink)
    if util > RANK_UTIL_GROW and eff_active > ACTIVE_FRAC_GROW:
        return Verdict(knob, str(dim), 'GROW', sig, suggest_grow)
    return Verdict(knob, str(dim), 'KEEP', sig, '')


def build_verdicts(
    sd: dict[str, torch.Tensor], arch: Arch, rows: list[ParamRow],
    data: dict[str, Any] | None,
) -> tuple[list[Verdict], dict[str, Any]]:
    """Produce the per-knob shrink/grow verdicts and a spectral detail dict."""
    np_sd = {k: v.detach().cpu().float().numpy() for k, v in sd.items()}
    stale_names = {r.name for r in rows if r.stale}
    verdicts: list[Verdict] = []
    spectra: dict[str, dict[str, float]] = {}

    # Global training-progress gate.  Random init is *full rank*, so a barely
    # trained checkpoint shows high rank-utilization that would masquerade as
    # "saturated → GROW".  If the main weight matrices have barely moved from
    # init (median |std/init_std - 1| below the near-init band) the checkpoint
    # is undertrained and its capacity verdicts are meaningless — flag it and
    # suppress the resize advice below.
    main_drifts = [abs(r.drift - 1.0) for r in rows
                   if r.kind in ('linear', 'residual', 'head_final') and not math.isnan(r.drift)]
    undertrained = bool(main_drifts) and float(np.median(main_drifts)) < DRIFT_NEAR_INIT

    def spec(name: str) -> dict[str, float]:
        if name not in spectra:
            spectra[name] = spectral(np_sd[name])
        return spectra[name]

    L = arch.n_layers
    hd = arch.head_dim

    # ---- D_MODEL: mean rank-utilization across every D×D map + the embed/head. ----
    dmodel_maps = ['patch_embed.weight']
    for i in range(L):
        dmodel_maps += [f'blocks.{i}.attn.w_q.weight', f'blocks.{i}.attn.w_k.weight',
                        f'blocks.{i}.attn.w_v.weight', f'blocks.{i}.attn.w_o.weight',
                        f'blocks.{i}.ffn.w1.weight', f'blocks.{i}.ffn.w3.weight',
                        f'blocks.{i}.ffn.w2.weight']
    dmodel_maps.append(arch.bg_head_linears[0])
    utils_dm = [spec(m)['util'] for m in dmodel_maps]
    mean_util_dm = float(np.mean(utils_dm))
    # Residual-dimension liveness (basis-dependent; corroborating only): a dim is
    # live if it is both written (rows of out=D_MODEL maps) and read (cols of
    # in=D_MODEL maps) above the dead floor.
    write = row_norms(np_sd['patch_embed.weight']).copy()
    read = np.zeros(arch.d_model)
    for i in range(L):
        write += row_norms(np_sd[f'blocks.{i}.attn.w_o.weight'])
        write += row_norms(np_sd[f'blocks.{i}.ffn.w2.weight'])
        for m in ('attn.w_q', 'attn.w_k', 'attn.w_v', 'ffn.w1', 'ffn.w3'):
            read += col_norms(np_sd[f'blocks.{i}.{m}.weight'])
    read += col_norms(np_sd[arch.bg_head_linears[0]])
    live_dim = (write >= DEAD_UNIT_REL * np.median(write)) & (read >= DEAD_UNIT_REL * np.median(read))
    n_live_dim = int(live_dim.sum())
    data_dm = data.get('resid_active_frac') if data else None
    v = verdict_width('D_MODEL', arch.d_model, mean_util_dm, n_live_dim,
                      n_live_dim / arch.d_model, 0,
                      _suggest_dmodel(arch, arch.d_model // 2),
                      _suggest_dmodel(arch, arch.d_model * 2),
                      data_active_frac=data_dm)
    v.detail = {'mean_rank_util': mean_util_dm, 'live_resid_dims': n_live_dim,
                'per_map_util': dict(zip(dmodel_maps, utils_dm))}
    verdicts.append(v)

    # ---- N_LAYERS: depth. Per-block residual write strength relative to init. ----
    base_std = INIT_BASE_STD_ANCHOR * math.sqrt(INIT_BASE_STD_ANCHOR_DMODEL / arch.d_model)
    residual_std = base_std / math.sqrt(RESIDUAL_WRITES_PER_BLOCK * L)
    block_writes = []
    block_utils = []
    for i in range(L):
        wo, w2 = np_sd[f'blocks.{i}.attn.w_o.weight'], np_sd[f'blocks.{i}.ffn.w2.weight']
        # ratio of write-projection std to its (depth-aware) init std
        r = 0.5 * (wo.std() + w2.std()) / residual_std
        block_writes.append(float(r))
        block_utils.append(0.5 * (spec(f'blocks.{i}.attn.w_o.weight')['util'] +
                                  spec(f'blocks.{i}.ffn.w2.weight')['util']))
    data_blk = data.get('block_contrib') if data else None
    if L == 1:
        grew = block_writes[0] > 2.0 and block_utils[0] > RANK_UTIL_GROW
        verd = 'GROW' if grew else 'KEEP'
        sig = (f'1 block: write {block_writes[0]:.1f}× init, util {block_utils[0]:.2f}'
               + (f', resid contrib {data_blk[0]:.2f}' if data_blk else ''))
        verdicts.append(Verdict('N_LAYERS', '1', verd, sig,
                                '--layers 2' if grew else '',
                                {'block_write_ratio': block_writes, 'block_util': block_utils}))
    else:
        dead_blocks = [i for i, r in enumerate(block_writes) if r < 1.2]
        if dead_blocks:
            verd, sug = 'SHRINK', f'--layers {max(1, L - len(dead_blocks))}'
            sig = f'{len(dead_blocks)}/{L} block(s) near-init write ({dead_blocks})'
        elif all(u > RANK_UTIL_GROW for u in block_utils) and min(block_writes) > 2.0:
            verd, sug = 'GROW', f'--layers {L + 1}'
            sig = f'all {L} blocks saturated (write {min(block_writes):.1f}–{max(block_writes):.1f}× init)'
        else:
            verd, sug = 'KEEP', ''
            sig = f'block writes {min(block_writes):.1f}–{max(block_writes):.1f}× init, util {min(block_utils):.2f}–{max(block_utils):.2f}'
        verdicts.append(Verdict('N_LAYERS', str(L), verd, sig, sug,
                                {'block_write_ratio': block_writes, 'block_util': block_utils}))

    # ---- N_HEADS: per-head value×output pathway strength. ----
    head_strength = np.zeros(arch.n_heads)
    head_detail = []
    for i in range(L):
        wv, wo = np_sd[f'blocks.{i}.attn.w_v.weight'], np_sd[f'blocks.{i}.attn.w_o.weight']
        for h in range(arch.n_heads):
            v_norm = float(np.linalg.norm(wv[h * hd:(h + 1) * hd, :]))
            o_norm = float(np.linalg.norm(wo[:, h * hd:(h + 1) * hd]))
            head_strength[h] += v_norm * o_norm
    med_h = float(np.median(head_strength))
    underused = [h for h in range(arch.n_heads) if head_strength[h] < HEAD_UNDERUSE_REL * med_h]
    data_heads = data.get('head_active_frac') if data else None
    n_target, sug_shrink = _suggest_heads(arch, grow=False)
    _, sug_grow = _suggest_heads(arch, grow=True)
    da = f', data-active {data_heads:.0%}' if data_heads is not None else ''
    sig = f'{len(underused)}/{arch.n_heads} heads underused (<{HEAD_UNDERUSE_REL:.0%} median){da}'
    # Heads are ~param-neutral, so be conservative: SHRINK only when a real
    # fraction is underused.  Never GROW on weight evidence alone (full-rank
    # head subspaces are not evidence that *more* heads would help); only when
    # the data pass shows every head saturated and head_dim has room to split.
    if len(underused) >= max(1, arch.n_heads // 4) and sug_shrink:
        verd, sug = 'SHRINK', sug_shrink
    elif (data_heads is not None and data_heads >= 0.999 and not underused
          and arch.head_dim > VALID_HEAD_DIMS[0] and sug_grow):
        verd, sug = 'GROW', sug_grow
    else:
        verd, sug = 'KEEP', ''
    verdicts.append(Verdict('N_HEADS', str(arch.n_heads), verd, sig, sug,
                            {'head_strength': head_strength.tolist(),
                             'underused_heads': underused}))

    # ---- FFN_DIM: hidden-unit usage (in/out weight-norm product) + rank-util. ----
    ffn_strength = np.zeros(arch.ffn_dim)
    ffn_util = []
    for i in range(L):
        w1, w3, w2 = (np_sd[f'blocks.{i}.ffn.w1.weight'], np_sd[f'blocks.{i}.ffn.w3.weight'],
                      np_sd[f'blocks.{i}.ffn.w2.weight'])
        in_norm = 0.5 * (row_norms(w1) + row_norms(w3))   # gate + value fan-in per hidden unit
        out_norm = col_norms(w2)                          # fan-out per hidden unit
        ffn_strength += in_norm * out_norm
        ffn_util += [spec(f'blocks.{i}.ffn.w1.weight')['util'],
                     spec(f'blocks.{i}.ffn.w2.weight')['util']]
    n_act, frac_act = active_fraction(ffn_strength)
    mult = arch.ffn_dim // arch.d_model
    sug_shrink = (f'--ffn-mult {mult // 2}' if mult >= 2
                  else 'already 1×D — FFN width tracks D_MODEL; lower --d-model instead')
    sug_grow = f'--ffn-mult {mult * 2 if mult >= 1 else 2}'
    data_ffn = data.get('ffn_active_frac') if data else None
    n_stale_ffn = sum(1 for i in range(L)
                      for n in (f'blocks.{i}.ffn.w1.weight', f'blocks.{i}.ffn.w2.weight',
                                f'blocks.{i}.ffn.w3.weight') if n in stale_names)
    v = verdict_width('FFN_DIM', arch.ffn_dim, float(np.mean(ffn_util)), n_act, frac_act,
                      n_stale_ffn, sug_shrink, sug_grow, data_active_frac=data_ffn)
    v.current = f'{arch.ffn_dim} ({mult}×D)'
    v.detail = {'mean_util': float(np.mean(ffn_util)), 'active_units': n_act,
                'active_frac': frac_act}
    verdicts.append(v)

    # ---- BG_HEAD_HIDDEN: both hidden layers (final layer is tiny-init by design). ----
    # layer0 hidden = out of bg_head.0, consumed by bg_head.2 columns.
    # layer1 hidden = out of bg_head.2, consumed by bg_head.4 columns.
    bl = arch.bg_head_linears
    layer_active = []
    bg_util = []
    if len(bl) >= 3:
        l0_in = row_norms(np_sd[bl[0]]); l0_out = col_norms(np_sd[bl[1]])
        layer_active.append(active_fraction(l0_in * l0_out))
        l1_in = row_norms(np_sd[bl[1]]); l1_out = col_norms(np_sd[bl[2]])
        layer_active.append(active_fraction(l1_in * l1_out))
        bg_util = [spec(bl[0])['util'], spec(bl[1])['util']]
    else:
        l0_in = row_norms(np_sd[bl[0]]); l0_out = col_norms(np_sd[bl[-1]])
        layer_active.append(active_fraction(l0_in * l0_out))
        bg_util = [spec(bl[0])['util']]
    # the binding layer (the one needing the most units) governs the width
    n_act_bg = max(a[0] for a in layer_active)
    frac_act_bg = n_act_bg / arch.bg_head_hidden
    mult_bg = arch.bg_head_hidden // arch.d_model
    sug_shrink = (f'--bg-head-hidden-mult {mult_bg // 2}' if mult_bg >= 2
                  else 'already 1×D — head width tracks D_MODEL; lower --d-model instead')
    sug_grow = f'--bg-head-hidden-mult {mult_bg * 2 if mult_bg >= 1 else 2}'
    data_bg = data.get('bghead_active_frac') if data else None
    n_stale_bg = sum(1 for n in bl if n in stale_names)
    v = verdict_width('BG_HEAD_HIDDEN', arch.bg_head_hidden, float(np.mean(bg_util)),
                      n_act_bg, frac_act_bg, n_stale_bg, sug_shrink, sug_grow,
                      data_active_frac=data_bg)
    v.current = f'{arch.bg_head_hidden} ({mult_bg}×D)'
    v.detail = {'per_layer_active': [a[0] for a in layer_active], 'mean_util': float(np.mean(bg_util))}
    verdicts.append(v)

    # ---- PATCH_SIZE: structural; report patch-embed input-column usage only. ----
    pe = np_sd['patch_embed.weight']                       # (D, PATCH_DIM)
    in_use = col_norms(pe)
    feat_cols = in_use[:arch.patch_size * N_INPUT_FEATURES]
    dead_feat = int((feat_cols < DEAD_UNIT_REL * np.median(feat_cols)).sum())
    sig = (f'{arch.patch_size} steps/patch ({arch.patch_size * 5} min); '
           f'{dead_feat}/{feat_cols.size} input feature-columns near-dead')
    verdicts.append(Verdict('PATCH_SIZE', str(arch.patch_size), 'NOTE', sig,
                            'structural — retrain required (see resize_model.py --patch-size)',
                            {'dead_feature_cols': dead_feat}))

    # ---- MIN/MAX_CONTEXT_PATCHES: ALiBi effective reach. ----
    slopes = np.abs(np_sd['blocks.0.attn.alibi_slopes'])
    slopes = slopes[slopes > 0]
    # reach where bias hits -ALIBI_REACH_NATS, in patches; flattest head = longest reach.
    reach_patches = ALIBI_REACH_NATS / slopes
    max_reach = float(reach_patches.max())
    patch_min = arch.patch_size * 5
    max_reach_h = max_reach * patch_min / 60.0
    # Compare the attention reach against the live MAX_CONTEXT_PATCHES (advisory).
    max_ctx = None
    try:
        import config as _cfg
        max_ctx = int(_cfg.MAX_CONTEXT_PATCHES)
    except Exception:  # noqa: BLE001
        pass
    if max_ctx:
        if max_reach > 1.5 * max_ctx:
            tail = (f'reach >> MAX_CONTEXT_PATCHES={max_ctx}: ALiBi does not decay within the '
                    'window (long-range attention used; window not the bottleneck)')
        elif max_reach < 0.5 * max_ctx:
            tail = (f'reach < MAX_CONTEXT_PATCHES={max_ctx}: attention decays well inside the '
                    'window — context may be shrinkable')
        else:
            tail = f'reach comparable to MAX_CONTEXT_PATCHES={max_ctx}'
    else:
        tail = 'context spans up to MAX_CONTEXT_PATCHES'
    sig = f'ALiBi flattest-head reach ~{max_reach:.0f} patches (~{max_reach_h:.1f} h); {tail}'
    verdicts.append(Verdict('CONTEXT_PATCHES', 'MIN/MAX', 'NOTE', sig,
                            'param-neutral — set --min/--max-context-patches per reach',
                            {'alibi_reach_patches': reach_patches.tolist(),
                             'max_reach_hours': max_reach_h}))

    # Apply the undertrained guard: a near-init checkpoint yields no trustworthy
    # capacity advice, so neutralize every SHRINK/GROW to KEEP (the NOTE knobs
    # stay) and let the report banner explain why.
    if undertrained:
        for v in verdicts:
            if v.verdict in ('SHRINK', 'GROW'):
                v.verdict = 'KEEP'
                v.signal = '[undertrained — verdict suppressed] ' + v.signal
                v.suggest = ''

    return verdicts, {'spectra': spectra, 'undertrained': undertrained}


# ===========================================================================
# Optional data-driven activation pass
# ===========================================================================
def run_data_pass(
    arch: Arch, ckpt: dict[str, Any], state_dict: dict[str, torch.Tensor],
    n_samples: int, device: torch.device,
) -> dict[str, Any] | None:
    """Stream cached samples through the model and measure activation liveness.

    Patches the in-process ``config`` to the checkpoint's dims (so ``model``
    binds them), builds ``T1DMAI``, loads ``state_dict``, registers forward
    hooks, and runs ``n_samples`` cached simulator samples.  Returns activation
    dead-fraction / reach summaries, or ``None`` if the cache/model is
    unavailable (the caller continues with the data-free report).

    Args:
        arch: derived architecture.
        ckpt: full checkpoint (for cache_path, master_seed, normalization_stats).
        state_dict: weights to load (live or EMA).
        n_samples: number of cached samples to stream.
        device: torch device.

    Returns:
        dict of data-driven summaries, or None on failure.
    """
    import sys
    try:
        import config
        tc = ckpt.get('training_config', {})
        config.D_MODEL = arch.d_model
        config.N_HEADS = arch.n_heads
        config.HEAD_DIM = arch.head_dim
        config.FFN_DIM = arch.ffn_dim
        config.BG_HEAD_HIDDEN = arch.bg_head_hidden
        config.N_LAYERS = arch.n_layers
        config.PATCH_SIZE = arch.patch_size
        config.PATCH_DIM = arch.patch_dim
        pph = 60 // (arch.patch_size * 5)
        config._PATCHES_PER_HOUR = pph
        config.PREDICTION_PATCHES = config.PREDICTION_HORIZON_HOURS * pph
        config.MAX_SEQ_LEN = config.MAX_CONTEXT_PATCHES + config.PREDICTION_PATCHES
        config.NIGHT_LONG_HORIZON_PATCHES = config.NIGHT_LONG_HORIZON_HOURS * pph
        sys.modules.pop('model', None)
        from model import T1DMAI
        from data import T1DMDataset, collate_fn

        model = T1DMAI().to(device)
        model.load_state_dict(state_dict)
        model.eval()

        cache_path = tc.get('cache_path', 'simulator_cache/')
        if not Path(cache_path, 'meta.json').exists():
            print(f"[data] cache {cache_path!r} unavailable — skipping data pass")
            return None
        ds = T1DMDataset(
            master_seed=int(ckpt.get('master_seed', 0)),
            total_steps=max(1, n_samples), batch_size=1,
            normalization_stats=ckpt['normalization_stats'],
            cache_path=cache_path, seed_offset=7_000_000,  # disjoint from train/norm/conformal bands
        )
    except Exception as e:  # noqa: BLE001 — data pass is best-effort
        print(f"[data] could not initialize data pass: {type(e).__name__}: {e}")
        return None

    # --- accumulators (per-unit mean |activation| over all tokens) ---
    acc: dict[str, np.ndarray] = {}
    cnt: dict[str, int] = {}

    def accum(key: str, t: torch.Tensor) -> None:
        a = t.detach().abs().reshape(-1, t.shape[-1]).float().mean(0).cpu().numpy()
        acc[key] = a if key not in acc else acc[key] + a
        cnt[key] = cnt.get(key, 0) + 1

    resid_var: list[np.ndarray] = []
    head_norm_acc = np.zeros(arch.n_heads)
    attn_reach: list[float] = []
    hooks = []

    def pre_hook(key):
        def h(_m, inp):
            accum(key, inp[0])
        return h

    for i, blk in enumerate(model.blocks):
        hooks.append(blk.ffn.w2.register_forward_pre_hook(pre_hook(f'ffn{i}')))
        # per-head output activation: input to w_o is (B,T,D) = concatenated heads
        def wo_hook(_m, inp, _i=i):
            x = inp[0].detach()                              # (B,T,D) = concatenated heads
            B, T, D = x.shape
            xh = x.reshape(B, T, arch.n_heads, arch.head_dim)
            nonlocal head_norm_acc
            # per-head RMS output activation over (batch, time, head_dim) -> (n_heads,)
            head_norm_acc = head_norm_acc + xh.float().pow(2).mean((0, 1, 3)).sqrt().cpu().numpy()
        hooks.append(blk.attn.w_o.register_forward_pre_hook(wo_hook))

    def final_hook(_m, inp):
        x = inp[0].detach()                                  # (B,T,D)
        resid_var.append(x.float().var(dim=(0, 1)).cpu().numpy())
    hooks.append(model.final_norm.register_forward_pre_hook(final_hook))

    bl = arch.bg_head_linears
    # BG-head hidden activations: inputs to the 2nd and 3rd linear are post-SiLU.
    if len(bl) >= 2:
        idx1 = int(bl[1].split('.')[1])
        hooks.append(model.bg_head[idx1].register_forward_pre_hook(pre_hook('bghead0')))
    if len(bl) >= 3:
        idx2 = int(bl[2].split('.')[1])
        hooks.append(model.bg_head[idx2].register_forward_pre_hook(pre_hook('bghead1')))

    n = min(n_samples, len(ds))
    print(f"[data] streaming {n} cached samples through the model…")
    try:
        with torch.no_grad():
            for i in range(n):
                batch = collate_fn([ds[i]])
                patches = batch['patches'].to(device)
                attn_mask = batch['attn_mask'].to(device)
                last_bg = batch['bg_formula_data']['last_bg'].float().to(device)
                model(patches, attn_mask, last_bg)
    except Exception as e:  # noqa: BLE001
        print(f"[data] forward failed mid-stream: {type(e).__name__}: {e}")
        for h in hooks:
            h.remove()
        if not acc:
            return None
    finally:
        for h in hooks:
            h.remove()

    def dead_frac(key: str) -> float | None:
        if key not in acc:
            return None
        a = acc[key] / cnt[key]
        med = float(np.median(a))
        live = a >= DEAD_UNIT_REL * med if med > 0 else a >= DEAD_UNIT_REL * (a.max() or 1.0)
        return float(live.mean())

    out: dict[str, Any] = {}
    ffn_fracs = [f for k in acc if k.startswith('ffn') for f in [dead_frac(k)] if f is not None]
    if ffn_fracs:
        out['ffn_active_frac'] = float(np.mean(ffn_fracs))
    bg_fracs = [f for k in ('bghead0', 'bghead1') for f in [dead_frac(k)] if f is not None]
    if bg_fracs:
        out['bghead_active_frac'] = float(np.min(bg_fracs))
    if resid_var:
        rv = np.mean(resid_var, axis=0)
        med = float(np.median(rv))
        out['resid_active_frac'] = float((rv >= DEAD_UNIT_REL * med).mean()) if med > 0 else 1.0
    if head_norm_acc.any():
        med = float(np.median(head_norm_acc))
        out['head_active_frac'] = float((head_norm_acc >= HEAD_UNDERUSE_REL * med).mean())
    out['n_samples'] = n
    return out


# ===========================================================================
# Reporting
# ===========================================================================
def hr(c: str = '─', n: int = 88) -> str:
    return c * n


def print_report(
    path: Path, ckpt: dict[str, Any], arch: Arch, param_count: int,
    rows: list[ParamRow], verdicts: list[Verdict], spectra: dict[str, dict[str, float]],
    data: dict[str, Any] | None, top: int, using_ema: bool, undertrained: bool,
) -> None:
    """Render the human-readable report to stdout."""
    print(hr('═'))
    print(f"  T1DMAI model health — {path.name}")
    print(hr('═'))
    step = ckpt.get('step', '?')
    bvl, bvs = ckpt.get('best_val_loss'), ckpt.get('best_val_step')
    print(f"  step {step}   weights: {'EMA shadow' if using_ema else 'live'}   "
          f"params {param_count:,} ({param_count / 1e6:.2f}M)")
    if bvl is not None:
        print(f"  best_val_loss {bvl:.5f} @ step {bvs}   "
              f"arch_version {ckpt.get('arch_version','?')}   loss_schema {ckpt.get('loss_schema','?')}")
    av = arch.view()
    print("  arch: " + "  ".join(f"{k}={v}" for k, v in av.items())
          + f"  N_SPREADS={arch.n_spreads}")

    # ---- per-param health ----
    print()
    print(hr())
    print("  PARAMETER HEALTH  (drift = std/init_std; opt-act = √mean optimizer 2nd-moment; "
          "rel = ×median)")
    print(hr())
    print(f"  {'parameter':32s} {'shape':>13s} {'params':>9s} {'std':>9s} "
          f"{'drift':>7s} {'opt':>4s} {'opt-act':>10s} {'rel':>7s}  flags")
    rows_sorted = sorted(rows, key=lambda r: r.numel, reverse=True)
    shown = 0
    for r in rows_sorted:
        if shown >= top:
            break
        shown += 1
        flags = []
        if r.stale:
            flags.append('STALE')
        if r.near_init:
            flags.append('near-init')
        drift = f'{r.drift:.2f}' if not math.isnan(r.drift) else '  -'
        oa = f'{r.opt_activity:.2e}' if not math.isnan(r.opt_activity) else '   -'
        rel = f'{r.opt_rel:.2f}' if not math.isnan(r.opt_rel) else '  -'
        print(f"  {r.name:32s} {str(r.shape):>13s} {r.numel:>9,} {r.std:>9.4f} "
              f"{drift:>7s} {r.opt:>4s} {oa:>10s} {rel:>7s}  {' '.join(flags)}")
    if len(rows) > top:
        print(f"  … {len(rows) - top} more (raise --top to see all)")
    n_stale = sum(1 for r in rows if r.stale)
    n_near = sum(1 for r in rows if r.near_init)
    print(f"  → {n_stale} stale parameter tensor(s), {n_near} near-init "
          f"(of {len(rows)} total)")

    # ---- spectral table ----
    print()
    print(hr())
    print("  SPECTRAL CAPACITY  (util = eff_rank / min(in,out); low ⇒ over-provisioned)")
    print(hr())
    print(f"  {'matrix':32s} {'shape':>13s} {'stable_rk':>10s} {'eff_rank':>9s} "
          f"{'util':>6s} {'tail%':>6s}")
    for name in sorted(spectra):
        s = spectra[name]
        sh = tuple(ckpt['_sd_shapes'][name]) if '_sd_shapes' in ckpt else ''
        print(f"  {name:32s} {str(sh):>13s} {s['stable_rank']:>10.1f} "
              f"{s['eff_rank']:>9.1f} {s['util']:>6.2f} {s['tail_frac'] * 100:>5.0f}%")

    # ---- data summary ----
    if data:
        print()
        print(hr())
        print(f"  ACTIVATION PASS  ({data.get('n_samples','?')} cached samples)")
        print(hr())
        for k, label in (('ffn_active_frac', 'FFN hidden units firing'),
                         ('bghead_active_frac', 'BG-head hidden units firing'),
                         ('resid_active_frac', 'residual dims with variance'),
                         ('head_active_frac', 'attention heads active')):
            if k in data:
                print(f"  {label:36s} {data[k]:.0%}")

    # ---- verdicts ----
    print()
    print(hr('═'))
    print("  CAPACITY VERDICTS  (resize_model.py knobs)")
    print(hr('═'))
    if undertrained:
        print("  ⚠ checkpoint appears UNDERTRAINED (weights near init) — capacity")
        print("    verdicts are unreliable and have been suppressed to KEEP.")
        print(hr())
    print(f"  {'knob':18s} {'current':12s} {'verdict':8s} signal")
    for v in verdicts:
        print(f"  {v.knob:18s} {v.current:12s} {v.verdict:8s} {v.signal}")
        if v.suggest:
            # An actual resize_model.py flag string starts with '--'; anything
            # else is a prose note (already minimal / structural / param-neutral).
            if v.suggest.lstrip().startswith('--'):
                print(f"  {'':40s} → python resize_model.py {v.suggest}")
            else:
                print(f"  {'':40s} → {v.suggest}")
    print(hr('═'))
    shrink = [v.knob for v in verdicts if v.verdict == 'SHRINK']
    grow = [v.knob for v in verdicts if v.verdict == 'GROW']
    print(f"  SHRINK candidates: {', '.join(shrink) if shrink else '(none)'}")
    print(f"  GROW candidates:   {', '.join(grow) if grow else '(none)'}")
    print(hr('═'))


# ===========================================================================
# Main
# ===========================================================================
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
                    help='also run N cached samples through the model for an activation pass (0 = off).')
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

    # cross-check against the live config.py (advisory only).
    try:
        import config as _cfg
        live = (_cfg.D_MODEL, _cfg.N_LAYERS, _cfg.N_HEADS, _cfg.FFN_DIM,
                _cfg.BG_HEAD_HIDDEN, _cfg.PATCH_SIZE)
        derived = (arch.d_model, arch.n_layers, arch.n_heads, arch.ffn_dim,
                   arch.bg_head_hidden, arch.patch_size)
        if live != derived:
            print(f"[warn] config.py {live} != checkpoint-derived {derived} "
                  "(auditing the checkpoint's architecture).")
        bg_init = float(getattr(_cfg, 'BG_HEAD_INIT_SCALE', DEFAULT_BG_HEAD_INIT_SCALE))
    except Exception:  # noqa: BLE001
        bg_init = DEFAULT_BG_HEAD_INIT_SCALE

    opt = map_optimizer_activity(ckpt['model_state_dict'], ckpt)
    rows = analyze_params(sd, arch, opt, bg_init, weights_only_ema=args.ema)

    data = None
    if args.data > 0:
        data = run_data_pass(arch, ckpt, sd, args.data, torch.device(args.device))

    verdicts, detail = build_verdicts(sd, arch, rows, data)
    print_report(path, ckpt, arch, param_count, rows, verdicts,
                 detail['spectra'], data, args.top, args.ema, detail['undertrained'])

    if args.json:
        payload = {
            'checkpoint': str(path), 'step': ckpt.get('step'),
            'using_ema': args.ema, 'param_count': param_count,
            'undertrained': detail['undertrained'],
            'arch': asdict(arch), 'spectra': detail['spectra'],
            'data': data,
            'params': [asdict(r) for r in rows],
            'verdicts': [asdict(v) for v in verdicts],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=float))
        print(f"[json] wrote {args.json}")


if __name__ == '__main__':
    main()
