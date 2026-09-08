"""
Where a masked patch reads from, under each of the two fixed protocols.

FORECAST is one-sided (right-edge span); INFILL is bracketed. Six figures, plus case figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)                              # day_curves, figstyle
sys.path.insert(0, os.path.join(HERE, 'sim'))         # sim_data

from config import (                                   # noqa: E402
    MAX_CONTEXT_PATCHES, PATCH_SIZE, PREDICTION_PATCHES, N_LAYERS,
)
from attribution import explain                        # noqa: E402
from metrics.core.features import (                    # noqa: E402
    build_feature_stack, context_window, segment_to_channels,
)
import metrics.protocols as PR                         # noqa: E402
from day_curves import load_model                      # noqa: E402
import figstyle as F                                   # noqa: E402
from figstyle import plt                               # noqa: E402
import sim_data                                        # noqa: E402

CTX = MAX_CONTEXT_PATCHES * PATCH_SIZE
PRED = PREDICTION_PATCHES * PATCH_SIZE
# Windows per segment/stride; lower than accuracy probes' since every slot costs a backward.
STRIDE_PATCHES = 24
CAP = 10
# Profile half-width in patches; 48 = 24h, past which mass rounds to zero; kept whole as ``beyond``.
OFFSET_REACH = 48

_OUT = os.path.join(HERE, 'attention.json')


def _protocol_masked_sets(n_ctx: int, rng: np.random.Generator):
    """One masked set per protocol, from ``protocols`` — never built here. Returns
    ``[(protocol_name, MaskedSet), ...]`` in ``PROTOCOLS`` order.
    """
    return [
        (PR.FORECAST.name, PR.forecast_masked_set(n_ctx)),
        (PR.INFILL.name, PR.infill_masked_set(n_ctx, rng)),
    ]


class _Series:
    """Running sums for one attention series of one (protocol, d) cell."""

    def __init__(self) -> None:
        self.n = 0
        self.profile = np.zeros(2 * OFFSET_REACH + 1, dtype=np.float64)
        self.beyond_before = 0.0
        self.beyond_after = 0.0
        self.before = 0.0
        self.after = 0.0
        self.near_before = 0.0
        self.near_after = 0.0
        self.own_span = 0.0
        self.on_masked = 0.0
        self.channels = None                            # (C,) lazily sized

    def add(self, mass: np.ndarray, patch: int, span: tuple[int, int],
            masked: np.ndarray, shares: np.ndarray | None) -> None:
        """Accumulate one scored slot's maps. ``shares=None`` for a series with no gradient read."""
        T = mass.shape[0]
        offsets = np.arange(T) - patch
        near = np.abs(offsets) <= OFFSET_REACH
        np.add.at(self.profile, offsets[near] + OFFSET_REACH, mass[near])
        self.beyond_before += float(mass[(~near) & (offsets < 0)].sum())
        self.beyond_after += float(mass[(~near) & (offsets > 0)].sum())

        start, length = span
        own = (np.arange(T) >= start) & (np.arange(T) < start + length)
        self.own_span += float(mass[own].sum())
        self.before += float(mass[np.arange(T) < start].sum())
        self.after += float(mass[np.arange(T) >= start + length].sum())
        self.on_masked += float(mass[masked].sum())
        # Same split, span removed, over a SYMMETRIC window; confounds placement whole-window.
        reach = near & (~own)
        self.near_before += float(mass[reach & (offsets < 0)].sum())
        self.near_after += float(mass[reach & (offsets > 0)].sum())

        if shares is not None:
            if self.channels is None:
                self.channels = np.zeros(shares.shape[0], dtype=np.float64)
            self.channels += shares
        self.n += 1

    def summary(self, channel_names: list[str]) -> dict:
        if self.n == 0:
            return {'n': 0}
        n = float(self.n)
        # Offset axis is stated once in _meta; repeating it here would copy per layer per d.
        out = {
            'n': self.n,
            'profile': (self.profile / n).tolist(),
            'beyond_before': self.beyond_before / n,
            'beyond_after': self.beyond_after / n,
            'mass_before_span': self.before / n,
            'mass_after_span': self.after / n,
            'near_before': self.near_before / n,
            'near_after': self.near_after / n,
            'mass_own_span': self.own_span / n,
            'mass_on_masked_patches': self.on_masked / n,
        }
        if self.channels is not None:
            out['channel_share'] = {
                name: float(v / n) for name, v in zip(channel_names, self.channels)
            }
        return out


class _Cell:
    """One (protocol, d) cell: the composed series, and every layer's own.

    rollout's 0.5A+0.5I residual puts 0.5**N_LAYERS of mass on the query by construction,
    making mass_own_span incomparable across capacities. final_layer is the depth-free series.
    """

    def __init__(self) -> None:
        self.rollout = _Series()
        self.layers: list[_Series] = []

    def add(self, roll: np.ndarray, per_layer: np.ndarray, patch: int,
            span: tuple[int, int], masked: np.ndarray,
            shares: np.ndarray) -> None:
        self.rollout.add(roll, patch, span, masked, shares)
        while len(self.layers) < per_layer.shape[0]:
            self.layers.append(_Series())
        for series, row in zip(self.layers, per_layer):
            series.add(row, patch, span, masked, None)

    def summary(self, channel_names: list[str]) -> dict:
        out = self.rollout.summary(channel_names)
        layers = [ls.summary(channel_names) for ls in self.layers]
        out['layers'] = layers
        # A name for the block feeding the head, not a second accumulation, which would drift.
        out['final_layer'] = layers[-1] if layers else {'n': 0}
        return out


def run(model, stats: dict, device, segs: list, stride: int, cap: int,
        seed: int) -> dict:
    """Probe every strided window of ``segs`` under both protocols. Returns
    ``{protocol: {d: cell}}`` plus a ``_windows`` count; ``seed`` fixes the infill placement.
    """
    rng = np.random.default_rng(seed)
    bins: dict[str, dict[int, _Cell]] = {}
    channel_names: list[str] = []
    n_windows = 0

    for seg in segs:
        n = (len(seg) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX + PRED:
            continue
        feats = build_feature_stack(seg, stats)
        taken = 0
        for ps in range(CTX, n - PRED + 1, stride):
            if taken >= cap:
                break
            taken += 1
            n_windows += 1
            ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)
            n_ctx = int(ctx.shape[0])
            for name, ms in _protocol_masked_sets(n_ctx, rng):
                masked = np.zeros(ms.seq_len, dtype=bool)
                masked[ms.mask_idx[ms.valid]] = True
                span_of = {}
                for s, L in ms.spans:
                    for p in range(s, s + L):
                        span_of[p] = (s, L)
                cell = bins.setdefault(name, {})
                for patch, d in zip(ms.scored_patches(), ms.scored_d()):
                    # A single-patch span selects one slot from the protocol's own masked set.
                    a = explain(model, ctx, stats, mask_spans=list(ms.spans),
                                span=(int(patch), 1), device=device)
                    if not channel_names:
                        channel_names = list(a.channel_names)
                    cell.setdefault(int(d), _Cell()).add(
                        a.where, a.per_layer, int(patch),
                        span_of[int(patch)], masked,
                        a.channel_share.astype(np.float64),
                    )

    return {
        '_windows': n_windows,
        'protocols': {
            name: {str(d): b.summary(channel_names) for d, b in sorted(cells.items())}
            for name, cells in bins.items()
        },
        'channel_names': channel_names,
    }


def _hours(res: dict) -> np.ndarray:
    """The offset axis in hours — the JSON states it once, in patches."""
    return np.asarray(res['_meta']['offset_patches'], dtype=float) * (
        PATCH_SIZE * 5.0 / 60.0)


def _cells(res: dict, proto: str):
    """``(d, cell)`` for every populated d of one protocol, in d order."""
    return [(int(d), c) for d, c in sorted(res['protocols'].get(proto, {}).items(),
                                           key=lambda kv: int(kv[0]))
            if c.get('n')]


def _series(cell: dict, which: str) -> dict:
    return cell if which == 'rollout' else cell[which]


def _protocols(res: dict) -> list[str]:
    return [p for p in ('forecast', 'infill') if res['protocols'].get(p)]


def _plot_profile(ax, x, cell: dict, which: str, color: str, label: str,
                  ls: str = '-') -> None:
    ser = _series(cell, which)
    if not ser.get('n'):
        return
    ax.plot(x, np.asarray(ser['profile'], dtype=float), color=color, lw=1.5,
            ls=ls, label=label)


def fig_profile(res: dict) -> str:
    """Offset profile, faceted by d, forecast against infill.

    The raw final layer rather than the composed row: the composition's residual
    term is depth-dependent, and the point of this figure is the geometry.
    """
    x = _hours(res)
    ds = sorted({d for p in _protocols(res) for d, _ in _cells(res, p)})[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    for ax, d in zip(axes.ravel(), ds):
        for k, proto in enumerate(_protocols(res)):
            cell = dict(_cells(res, proto)).get(d)
            if cell is None:
                continue
            _plot_profile(ax, x, cell, 'final_layer', F.SERIES[k],
                          f"{proto} (n={cell['final_layer']['n']})")
        ax.axvline(0.0, color=F.AXIS, lw=0.8, zorder=0)
        ax.set_yscale('log')
        ax.set_title(f'd = {d}')
        F.ygrid(ax)
        F.legend(ax, loc='upper left')
    for ax in axes.ravel()[len(ds):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('offset from the masked patch (h)')
    for ax in axes[:, 0]:
        ax.set_ylabel('final-layer attention mass')
    fig.suptitle('Where a masked patch reads from, by distance to visible evidence',
                 y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return F.save(fig, 'attention_profile.png')


def fig_layers(res: dict) -> str:
    """The same profile per layer index — where the locality actually lives."""
    x = _hours(res)
    protos = _protocols(res)
    fig, axes = plt.subplots(1, max(1, len(protos)), figsize=(13.0, 5.0),
                             sharey=True, squeeze=False)
    n_layers = int(res['_meta']['n_layers'])
    for ax, proto in zip(axes[0], protos):
        cells = dict(_cells(res, proto))
        cell = cells.get(1) or next(iter(cells.values()), None)
        if cell is None:
            continue
        for i, ser in enumerate(cell.get('layers', [])):
            if not ser.get('n'):
                continue
            # The ordinal ladder: layer index is ordered, so it rides one hue.
            shade = F.SEQ[min(len(F.SEQ) - 1,
                              int(round(i * (len(F.SEQ) - 1) / max(1, n_layers - 1))))]
            ax.plot(x, np.asarray(ser['profile'], dtype=float), color=shade,
                    lw=1.4, label=f'layer {i}')
        ax.axvline(0.0, color=F.AXIS, lw=0.8, zorder=0)
        ax.set_yscale('log')
        ax.set_xlabel('offset from the masked patch (h)')
        ax.set_title(f'{proto}, d = 1')
        F.ygrid(ax)
    axes[0][0].set_ylabel('attention mass')
    # Below the axes: every panel is dense, so no interior corner fits a legend cleanly.
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=min(8, n_layers),
               frameon=False, fontsize=9, labelcolor=F.INK2)
    fig.suptitle(f'Attention by layer ({n_layers} blocks), before any composition',
                 y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return F.save(fig, 'attention_layers.png')


def fig_rollout(res: dict) -> str:
    """The composed row against the raw final layer, so the artefact is visible."""
    x = _hours(res)
    protos = _protocols(res)
    floor = float(res['_meta']['rollout_identity_floor'])
    fig, axes = plt.subplots(1, max(1, len(protos)), figsize=(13.0, 5.0),
                             sharey=True, squeeze=False)
    for ax, proto in zip(axes[0], protos):
        cell = dict(_cells(res, proto)).get(1)
        if cell is None:
            continue
        _plot_profile(ax, x, cell, 'rollout', F.PAIR[1], 'rollout (0.5A + 0.5I)')
        _plot_profile(ax, x, cell, 'final_layer', F.PAIR[0], 'final layer, raw', '--')
        ax.axvline(0.0, color=F.AXIS, lw=0.8, zorder=0)
        ax.set_yscale('log')
        ax.set_xlabel('offset from the masked patch (h)')
        ax.set_title(f'{proto}, d = 1')
        F.ygrid(ax)
        F.legend(ax, loc='upper left')
    axes[0][0].set_ylabel('attention mass')
    fig.suptitle('The composition puts mass on the query by construction — '
                 f'{floor:.4g} of it at this depth', y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return F.save(fig, 'attention_rollout.png')


def fig_sides(res: dict) -> str:
    """Mass before the span against mass after it, within the symmetric reach."""
    labels, before, after = [], [], []
    for proto in _protocols(res):
        for d, cell in _cells(res, proto):
            ser = cell['final_layer']
            labels.append(f"{proto} d={d} (n={ser['n']})")
            before.append(ser['near_before'])
            after.append(ser['near_after'])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    if labels:
        F.dumbbell_rows(ax, labels, before, after)
    reach_h = OFFSET_REACH * PATCH_SIZE * 5 / 60.0
    ax.set_xlabel('share of final-layer attention mass')
    fig.suptitle(f'Read backward against read forward, within ±{reach_h:.0f} h '
                 'of the span', y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return F.save(fig, 'attention_sides.png')


def fig_channels(res: dict) -> str:
    """Channel share of |grad x input| against d, one line per channel."""
    names = res['channel_names']
    protos = _protocols(res)
    fig, axes = plt.subplots(1, max(1, len(protos)), figsize=(12.0, 5.0),
                             sharey=True, squeeze=False)
    top = 0.0
    for ax, proto in zip(axes[0], protos):
        cells = _cells(res, proto)
        ds = [d for d, _ in cells]
        for name in names:
            ys = [cell['channel_share'][name] for _d, cell in cells]
            top = max(top, max(ys, default=0.0))
            ax.plot(ds, ys, marker='o', ms=4.5, lw=1.6,
                    color=F.channel_color(name), label=name)
        ax.set_xticks(ds)
        ax.set_xlabel('d  (patches to the nearest visible evidence)')
        ax.set_title(proto)
        F.ygrid(ax)
    # Set once after every panel: sharey autoscale otherwise clips a series peaking earlier.
    axes[0][0].set_ylim(0, top * 1.08)
    axes[0][0].set_ylabel('share of |grad x input|')
    # Held outside the axes: a panel this dense has no interior corner free of a series.
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=len(names),
               frameon=False, fontsize=9, labelcolor=F.INK2)
    fig.suptitle('Which channel the gradient reads, against distance from '
                 'observed glucose', y=0.98)
    fig.tight_layout(rect=(0, 0.075, 1, 0.93))
    return F.save(fig, 'attention_channels.png')


def fig_masked(res: dict) -> str:
    """Attention landing on masked patches, against d, both series."""
    protos = _protocols(res)
    floor = float(res['_meta']['rollout_identity_floor'])
    fig, axes = plt.subplots(1, max(1, len(protos)), figsize=(12.0, 5.0),
                             sharey=True, squeeze=False)
    for ax, proto in zip(axes[0], protos):
        cells = _cells(res, proto)
        ds = [d for d, _ in cells]
        ax.plot(ds, [c['mass_on_masked_patches'] for _d, c in cells],
                marker='o', ms=4.5, lw=1.6, color=F.PAIR[1], label='rollout')
        ax.plot(ds, [c['final_layer']['mass_on_masked_patches'] for _d, c in cells],
                marker='s', ms=4.5, lw=1.6, ls='--', color=F.PAIR[0],
                label='final layer, raw')
        F.threshold(ax, floor, f'identity floor {floor:.4g}')
        ax.set_xticks(ds)
        ax.set_xlabel('d  (patches to the nearest visible evidence)')
        ax.set_title(proto)
        F.ygrid(ax)
        F.legend(ax, loc='upper left')
    axes[0][0].set_ylim(0, None)
    axes[0][0].set_ylabel('share of attention on masked patches')
    fig.suptitle("Attention spent on patches whose glucose is withheld", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return F.save(fig, 'attention_masked.png')


# Case studies use attribution's ramps, the same the GUI calls, so colours can't disagree.
N_CASES = 3
CASE_VIEW_PATCHES = 96                     # 48 h of context either side of the span


def _pick_window(seg, stride: int) -> int | None:
    """The eligible window carrying the most announced carbohydrate and insulin.

    A quiet stretch shows nothing; each channel is scored on its own scale before
    summing, since a gram and a unit are not comparable quantities.
    """
    n = (len(seg) // PATCH_SIZE) * PATCH_SIZE
    if n < CTX + PRED:
        return None
    ch = segment_to_channels(seg)
    carb = np.clip(np.asarray(ch['carb'], dtype=np.float64), 0.0, None)
    ins = np.clip(np.asarray(ch['insulin'], dtype=np.float64), 0.0, None)
    carb = carb / (carb.max() or 1.0)
    ins = ins / (ins.max() or 1.0)
    best, best_score = None, -1.0
    for ps in range(CTX, n - PRED + 1, stride):
        lo = max(0, ps - CASE_VIEW_PATCHES * PATCH_SIZE)
        score = float(carb[lo:ps + PRED].sum() + ins[lo:ps + PRED].sum())
        if score > best_score:
            best, best_score = ps, score
    return best


def _case_strip(ax, values: np.ndarray, patches: np.ndarray, color_lo, color_hi,
                withheld: np.ndarray | None, label: str) -> None:
    """One heat row over ``patches``, drawn as a signed or single-hue ramp."""
    from matplotlib.colors import to_rgb
    lo, hi = np.asarray(to_rgb(color_lo)), np.asarray(to_rgb(color_hi))
    rgb = np.ones((1, len(patches), 3))
    for j, v in enumerate(values):
        base = hi if v >= 0 else lo
        rgb[0, j] = 1.0 - (1.0 - base) * min(1.0, abs(float(v)))
    if withheld is not None:
        grey = np.asarray(to_rgb('#8f93a6'))
        for j, w in enumerate(withheld):
            if w:
                rgb[0, j] = 1.0 - (1.0 - grey) * 0.55
    half = 0.5 * (patches[1] - patches[0]) if len(patches) > 1 else 0.25
    ax.imshow(rgb, aspect='auto', interpolation='nearest',
              extent=(patches[0] - half, patches[-1] + half, 0, 1))
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=8,
                  fontsize=9, color=F.MUTED)
    for side in ax.spines.values():
        side.set_visible(False)


def _case_stats(ax, mass: np.ndarray, spec: dict, res: dict, proto: str) -> None:
    """This window's scalars beside the cohort's, as a text block."""
    p = mass[mass > 0]
    entropy = float(-(p * np.log(p)).sum())
    top5 = float(np.sort(mass)[-5:].sum())
    cohort = (dict(_cells(res, proto)).get(1) or {}).get('final_layer', {})
    lines = [
        ('attention entropy', f'{entropy:.2f} nats', f"of {np.log(len(mass)):.2f} max"),
        ('top-5 patches', f'{top5 * 100:.1f}%', 'of all mass'),
        ('read backward', f"{spec['near_before'] * 100:.1f}%",
         f"cohort {cohort.get('near_before', float('nan')) * 100:.1f}%"),
        ('read forward', f"{spec['near_after'] * 100:.1f}%",
         f"cohort {cohort.get('near_after', float('nan')) * 100:.1f}%"),
        ('on withheld patches', f"{spec['mass_on_masked_patches'] * 100:.1f}%",
         f"cohort {cohort.get('mass_on_masked_patches', float('nan')) * 100:.1f}%"),
    ]
    ax.axis('off')
    for i, (name, value, note) in enumerate(lines):
        y = 0.9 - i * 0.19
        ax.text(0.0, y, name, fontsize=9, color=F.MUTED, transform=ax.transAxes)
        ax.text(0.62, y, value, fontsize=10.5, color=F.INK, ha='right',
                transform=ax.transAxes)
        ax.text(0.66, y, note, fontsize=8.5, color=F.MUTED, transform=ax.transAxes)


def fig_case(a, cgm: np.ndarray, span: tuple[int, int], n_ctx: int, proto: str,
             patient: str, res: dict, name: str) -> str:
    """One window: the trace, its strip block, and its own statistics."""
    from attribution import share_ramp, signed_ramp

    T = int(a.where.shape[0])
    lo = max(0, span[0] - CASE_VIEW_PATCHES)
    hi = min(T, span[0] + span[1] + CASE_VIEW_PATCHES)
    patches = np.arange(lo, hi)
    hours = (patches - n_ctx) * (PATCH_SIZE * 5.0 / 60.0)

    # Final layer, not composed row: at depth composition is near-uniform, no contrast to draw.

    # medium's rollout entropy is 5.81 nats of a 5.83 max; the block feeding the head has contrast.
    mass = np.asarray(a.per_layer[-1], dtype=float)

    fig = plt.figure(figsize=(13.5, 9.0))
    # Row 6 is an undrawn spacer, else the bottom strip's tick labels land inside the panel above.
    gs = fig.add_gridspec(
        8, 3, height_ratios=[3.4, 0.36, 0.36, 0.36, 0.36, 0.36, 0.62, 2.4],
        hspace=0.18, wspace=0.3)

    ax_bg = fig.add_subplot(gs[0, :])
    steps = np.arange(lo * PATCH_SIZE, hi * PATCH_SIZE)
    ax_bg.plot((steps / PATCH_SIZE - n_ctx) * 0.5, cgm[lo * PATCH_SIZE:hi * PATCH_SIZE],
               color=F.SERIES[0], lw=1.4)
    ax_bg.axvspan((span[0] - n_ctx) * 0.5, (span[0] + span[1] - n_ctx) * 0.5,
                  color=F.MUTED, alpha=0.18, lw=0)
    F.threshold(ax_bg, 70.0, 'hypo')
    F.threshold(ax_bg, 180.0, 'hyper')
    ax_bg.set_xlim(hours[0], hours[-1])
    ax_bg.set_ylabel('CGM (mg/dL)')
    ax_bg.set_xticks([])
    F.ygrid(ax_bg)

    rows = [('attn', share_ramp(mass, lo, hi)[lo:hi], None,
             (F.CHANNEL['bg_absolute'], F.CHANNEL['bg_absolute']))]
    scale = float(np.abs(a.channels[lo:hi]).max())
    withheld = np.zeros(T, dtype=bool)
    withheld[a.masked_patches] = True
    for c, ch_name in enumerate(a.channel_names):
        rows.append((ch_name.split('_')[0],
                     signed_ramp(a.channels[lo:hi, c], scale),
                     withheld[lo:hi] if c == 0 else None,
                     F.DIVERGING))
    for i, (label, values, wh, color) in enumerate(rows):
        ax = fig.add_subplot(gs[1 + i, :])
        _case_strip(ax, values, hours, color[0], color[1], wh, label)
        ax.set_xlim(hours[0], hours[-1])
        if i == len(rows) - 1:
            ax.set_xlabel('hours from the forecast origin')
        else:
            ax.set_xticks([])

    ax_prof = fig.add_subplot(gs[7, 0])
    x = _hours(res)
    off = np.arange(T) - span[0]
    keep = np.abs(off) <= OFFSET_REACH
    prof = np.zeros_like(x)
    np.add.at(prof, off[keep] + OFFSET_REACH, mass[keep])
    ax_prof.plot(x, prof, color=F.PAIR[1], lw=1.5, label='this window')
    cohort = (dict(_cells(res, proto)).get(1) or {}).get('final_layer')
    if cohort and cohort.get('n'):
        ax_prof.plot(x, np.asarray(cohort['profile'], dtype=float), color=F.MUTED,
                     lw=1.2, ls='--', label='cohort mean')
    ax_prof.set_yscale('log')
    ax_prof.set_xlabel('offset (h)')
    ax_prof.set_ylabel('attention mass')
    F.ygrid(ax_prof)
    F.legend(ax_prof, loc='upper left', fontsize=8)

    ax_ch = fig.add_subplot(gs[7, 1])
    names = list(a.channel_names)
    ax_ch.bar(range(len(names)), [float(v) for v in a.channel_share],
              color=[F.channel_color(n) for n in names], width=0.62)
    ax_ch.set_xticks(range(len(names)))
    ax_ch.set_xticklabels([n.split('_')[0] for n in names], fontsize=9)
    ax_ch.set_ylabel('share of |grad x input|')
    F.ygrid(ax_ch)

    series = _Series()
    series.add(mass, span[0], span, withheld, a.channel_share.astype(np.float64))
    _case_stats(fig.add_subplot(gs[7, 2]), mass, series.summary(names), res, proto)

    fig.suptitle(f'{proto} — patient {patient}, span at '
                 f'{(span[0] - n_ctx) * 0.5:+.1f} h, {span[1]} patches   ·   '
                 f'attention is the final layer, uncomposed', y=0.975)
    return F.save(fig, name)


def cases(model, stats: dict, device, segs: list, stride: int, seed: int,
          res: dict) -> list[str]:
    """One case figure per protocol for each of the first ``N_CASES`` patients.

    The span explained is the whole span, not one slot of it — what a reader means by "this
    forecast" and what the GUI's own strips explain. Returns the repo-relative paths written.
    """
    rng = np.random.default_rng(seed)
    paths: list[str] = []
    for k, seg in enumerate(segs[:N_CASES]):
        ps = _pick_window(seg, stride)
        if ps is None:
            continue
        feats = build_feature_stack(seg, stats)
        ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)
        n_ctx = int(ctx.shape[0])
        cgm = np.asarray(seg.cgm, dtype=np.float64)[ps - CTX:ps + PRED]
        for name, ms in _protocol_masked_sets(n_ctx, rng):
            # Forecast scores its one trailing span; infill's interior spans show the first.
            span = ms.scored[0]
            a = explain(model, ctx, stats, mask_spans=list(ms.spans),
                        span=span, device=device)
            paths.append(fig_case(
                a, cgm, span, n_ctx, name, str(seg.patient), res,
                f'attention_case_{name}_{k}.png'))
    return paths


def figures(res: dict) -> list[str]:
    """Render every aggregate figure; returns their repo-relative paths."""
    F.style()
    return [fig_profile(res), fig_layers(res), fig_rollout(res),
            fig_sides(res), fig_channels(res), fig_masked(res)]


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Checkpoint to probe (default: the live best).')
    p.add_argument('--out', type=str, default=None,
                   help=f'JSON destination (default: {_OUT}).')
    p.add_argument('--stride-patches', type=int, default=STRIDE_PATCHES)
    p.add_argument('--cap', type=int, default=CAP)
    p.add_argument('--sim-seeds', type=int, default=len(sim_data.TEST_SEEDS))
    # One window needs MAX_CONTEXT_PATCHES = 168h of context; 288h default follows the sim report.
    p.add_argument('--sim-hours', type=float, default=288.0)
    p.add_argument('--seed', type=int, default=0,
                   help='Seeds the infill placement.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = (load_model(device, args.checkpoint) if args.checkpoint
                          else load_model(device))
    model = model.to(device).eval()

    seeds = list(sim_data.TEST_SEEDS[:max(1, int(args.sim_seeds))])
    segs = sim_data.make_sim_segments(seeds, hours=float(args.sim_hours))
    print(f"[attention] step={step} device={device} layers={N_LAYERS} "
          f"patients={len(segs)}")

    res = run(model, stats, device, segs,
              stride=max(1, int(args.stride_patches)) * PATCH_SIZE,
              cap=int(args.cap), seed=int(args.seed))
    assert res['_windows'] > 0, (
        f"no windows: a segment needs {CTX + PRED} steps "
        f"({(CTX + PRED) / 12:.0f} h) for one window and --sim-hours is "
        f"{args.sim_hours:g}. Raise it above the {CTX / 12:.0f} h context width."
    )
    res['_meta'] = {
        'step': step,
        'checkpoint': args.checkpoint,
        'n_layers': int(N_LAYERS),
        'n_ctx_patches': int(MAX_CONTEXT_PATCHES),
        'prediction_patches': int(PREDICTION_PATCHES),
        'offset_reach_patches': OFFSET_REACH,
        'offset_patches': list(range(-OFFSET_REACH, OFFSET_REACH + 1)),
        # What the rollout's residual term puts on the query by construction (method, not signal).
        'rollout_identity_floor': float(0.5 ** int(N_LAYERS)),
        'stride_patches': int(args.stride_patches),
        'cap': int(args.cap),
        'sim_seeds': seeds,
        'sim_hours': float(args.sim_hours),
        'infill_seed': int(args.seed),
    }

    out = args.out or _OUT
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)
    paths = figures(res)
    paths += cases(model, stats, device, segs,
                   max(1, int(args.stride_patches)) * PATCH_SIZE,
                   int(args.seed), res)
    print(f"[attention] windows={res['_windows']}  wrote {out}")
    for path in paths:
        print(f"            {path}")
    for name in sorted(res['protocols']):
        for d in sorted(res['protocols'][name], key=int):
            c = res['protocols'][name][d]
            if not c.get('n'):
                continue
            print(f"  {name:<9} d={d}  n={c['n']:<5} "
                  f"near before {c['near_before']:.3f}  near after {c['near_after']:.3f}  "
                  f"| own span {c['mass_own_span']:.3f} "
                  f"(final layer {c['final_layer']['mass_own_span']:.3f})  "
                  f"masked {c['mass_on_masked_patches']:.4f} "
                  f"(final layer {c['final_layer']['mass_on_masked_patches']:.4f})")


if __name__ == '__main__':
    main()
