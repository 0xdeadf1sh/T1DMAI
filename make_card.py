"""Model-card figures for one T1DMAI run, from logs/ + checkpoints/.

Writes figures/card_*.png; the prefix keeps them sorted apart from fig01..fig09.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import config  # loss-schema fallbacks (DILATE_ALPHA/GAMMA, QUANTILE_LEVELS) absent from the checkpoint's serialized config; every structural flag is derived from the weights via _derive_arch
from make_figures import CGEGA_COLUMNS_TRUSTWORTHY  # the single gate on the validation log's cgega_* columns; see the constant's comment for why and what flips it

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


REPO = Path(__file__).resolve().parent
LOG_DIR = REPO / "logs"
CKPT_PATH = REPO / "checkpoints" / "t1dmai_best.pt"
OUT_DIR = REPO / "figures"
METRICS_DIR = REPO / "metrics"

# metrics.core.calibrate.HORIZONS, restated locally so the card has no import-time dependency
# on the evaluation core — it only reads the JSON those scripts emit.
REPORT_HORIZONS = (30, 60, 120)
REPORT_SOURCES = (
    ("sim",       "Simulator (in-domain)", ("sim",)),
)




# muted publication palette: one warm + cool pair, neutrals dominant
INK       = "#1a1f2e"   # primary text (very dark navy, easier than pure black)
SLATE     = "#3d4659"   # secondary text
DIMMED    = "#6b7280"   # tertiary text
MUTED     = "#9aa3b2"   # captions, axis labels
RULE      = "#dcdde2"   # divider lines, borders
SOFT_RULE = "#eceef2"   # subtle backgrounds
CARD      = "#ffffff"   # card surface
PAPER     = "#fbfaf6"   # page background (off-white, warm)

NAVY  = "#2a4a6e"       # primary accent (structural)
CLAY  = "#a85a3a"       # warm accent (metrics, results)
TEAL  = "#3e7d8a"       # cool accent (technical)
GOLD  = "#a8843a"       # warning / calibration
SAGE  = "#5a7d4a"       # secondary positive
PLUM  = "#74416e"       # rare accent (events / sparse)

# background tints: 5% mixes of the accents with white
NAVY_T = "#ecf1f7"
CLAY_T = "#f8efe9"
TEAL_T = "#eef4f6"
GOLD_T = "#f7f1e6"
SAGE_T = "#eef3ea"
PLUM_T = "#f3edf1"


FONT_TITLE   = "DejaVu Serif"
FONT_BODY    = "Ubuntu"
FONT_MONO    = "DejaVu Sans Mono"


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "font.size": 10.5,
            "font.family": FONT_BODY,
            # Ubuntu lacks some math glyphs (→, ², ⁻); DejaVu Sans is the fallback
            "font.sans-serif": ["Ubuntu", "DejaVu Sans", "Liberation Sans"],
            "text.color": INK,
            "axes.edgecolor": RULE,
            "axes.labelcolor": SLATE,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "xtick.bottom": False,
            "ytick.left": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
        }
    )




def _setup_card(figsize: tuple[float, float]) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=PAPER)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(PAPER)
    return fig, ax


def _header(ax, eyebrow: str, title: str, subtitle: str | None = None,
            y_top: float = 0.965) -> float:
    """Shared card header; returns the y content can start at."""
    ax.text(0.025, y_top, eyebrow.upper(), fontsize=8.5, color=CLAY,
            family=FONT_BODY, weight="bold", transform=ax.transAxes)
    ax.text(0.025, y_top - 0.050, title, fontsize=20, color=INK,
            family=FONT_TITLE, va="top", transform=ax.transAxes)
    # room for the serif title's cap-height
    title_block = 0.115
    if subtitle:
        ax.text(0.025, y_top - 0.050 - title_block, subtitle, fontsize=10, color=SLATE,
                family=FONT_BODY, va="top", transform=ax.transAxes)
        title_block += 0.035
    y_rule = y_top - 0.060 - title_block
    ax.plot([0.025, 0.975], [y_rule, y_rule], color=RULE, lw=0.9,
            transform=ax.transAxes)
    return y_rule - 0.038


def _section(ax, y, label, color=NAVY) -> float:
    """Section eyebrow; returns the y of its first content row."""
    ax.text(0.025, y, label.upper(), fontsize=8.5, color=color,
            family=FONT_BODY, weight="bold", transform=ax.transAxes)
    return y - 0.040


def _stat_tile(ax, x, y, w, h, label, big, sub, accent):
    """A clean stat tile with a thin colored top bar."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.005,rounding_size=0.012",
                                 fc=CARD, ec=RULE, lw=0.9, transform=ax.transAxes))
    bar_h = 0.010
    ax.add_patch(Rectangle((x, y + h - bar_h), w, bar_h, fc=accent,
                            ec="none", transform=ax.transAxes))
    ax.text(x + w / 2, y + h - 0.045, label,
            fontsize=8.5, color=DIMMED, family=FONT_BODY, weight="bold", ha="center", transform=ax.transAxes)
    ax.text(x + w / 2, y + h * 0.42, big,
            fontsize=22, color=INK, family=FONT_TITLE, ha="center",
            transform=ax.transAxes)
    ax.text(x + w / 2, y + 0.02, sub,
            fontsize=8.5, color=DIMMED, family=FONT_BODY, ha="center",
            transform=ax.transAxes)




def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:                      # header-only / empty log → no rows to plot
        return {}
    cols: dict[str, list[float]] = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            cols[k].append(float(v) if v not in ("", None) else np.nan)
    return {k: np.asarray(v) for k, v in cols.items()}


def _load_report_stats() -> dict[str, dict]:
    """``{mode: parsed_stats}`` for every report that loaded; empty on a fresh run.

    ``metrics/sim/stats.json`` is written after training, so each file is guarded on its own —
    the card must build with zero reports present.
    """
    out: dict[str, dict] = {}
    for mode, _label, _datasets in REPORT_SOURCES:
        path = METRICS_DIR / mode / "stats.json"
        try:
            out[mode] = json.loads(path.read_text())
        except (FileNotFoundError, ValueError, OSError):
            continue
    return out


def _human_count(n) -> str:
    """Magnitude with the unit picked per value, so a nano model does not collapse to ``0M``:
    ``37195 -> '37K'``, ``2_140_885 -> '2.14M'``, ``3.7e9 -> '3.69B'``, ``950 -> '950'``."""
    n = float(n)
    if n >= 1e9:
        b = n / 1e9
        return f"{b:.2f}B" if b < 10 else f"{b:.1f}B"
    if n >= 1e6:
        m = n / 1e6
        return f"{m:.2f}M" if m < 10 else f"{m:.1f}M" if m < 100 else f"{m:.0f}M"
    if n >= 1e3:
        k = n / 1e3
        return f"{k:.1f}K" if k < 10 else f"{k:.0f}K"
    return f"{n:.0f}"


def _param_breakdown(sd: dict) -> tuple[dict[str, int], int]:
    groups = {
        "Patch embedding":                  r"^patch_embed\.",
        "Temporal attn (Q/K/V/O)":          r"^blocks\.\d+\.attn\.w_[qkvo]\.",
        "Temporal attn (Q/K norm)":         r"^blocks\.\d+\.attn\.(q_norm|k_norm|alibi_slopes)",
        "SwiGLU FFN":                       r"^blocks\.\d+\.ffn\.",
        "Block RMSNorms":                   r"^blocks\.\d+\.norm\d+\.",
        "Final RMSNorm":                    r"^final_norm\.",
        "BG quantile head":                 r"^bg_head\.",
        "Time-of-day probe (aux)":          r"^time_head\.",
    }
    out: dict[str, int] = {}
    total = 0
    ungrouped = 0
    for name, p in sd.items():
        n = p.numel()
        total += n
        for g, pat in groups.items():
            if re.match(pat, name):
                out[g] = out.get(g, 0) + n
                break
        else:
            ungrouped += n
    # Every tensor lands in exactly one group, else the wedges sum to less than the headline
    # total; fail loudly when a new top-level module escapes the regex table above.
    assert ungrouped == 0, (
        f"_param_breakdown: {ungrouped} params matched no group — add a pattern "
        f"(group sums {sum(out.values())} vs total {total})"
    )
    return out, total


def _derive_arch(sd: dict, cfg: dict) -> dict:
    """Architecture flags recovered from the checkpoint weight SHAPES, never from ``config.py``.

    The shapes travel with the checkpoint; ``resize_model.py`` rewrites ``BG_HEAD_HIDDEN`` in
    the live config and leaves no tripwire behind it.  Returns ``bg_head_hidden``,
    ``n_spreads``, ``n_quantiles``, ``time_probe``, ``time_probe_hidden`` and
    ``time_probe_bins`` (0 when the probe was off at train time).
    """
    # bg_head.0: (BG_HEAD_HIDDEN, D_MODEL)
    bg_head_hidden = sd['bg_head.0.weight'].shape[0]
    # the head runs on one step state at a time: out_last = 1 + 2·N_SPREADS
    out_last = sd['bg_head.4.weight'].shape[0]
    assert out_last % 2 == 1, (
        f"head out width {out_last} should be 1 + 2·N_SPREADS (odd)")
    n_spreads = (out_last - 1) // 2
    # time_head.0 → (HIDDEN, D_MODEL), time_head.2 → (N_BINS, HIDDEN); present iff
    # TIME_PROBE_ENABLED at train time, so read presence and dims off the state dict
    time_probe = 'time_head.0.weight' in sd
    return {
        'bg_head_hidden': bg_head_hidden,
        'n_spreads': n_spreads,
        'n_quantiles': 1 + 2 * n_spreads,
        'time_probe': time_probe,
        'time_probe_hidden': int(sd['time_head.0.weight'].shape[0]) if time_probe else 0,
        'time_probe_bins': int(sd['time_head.2.weight'].shape[0]) if time_probe else 0,
    }




def _mse_alpha(cfg: dict) -> float:
    return float(cfg.get('mse_alpha', config.MSE_ALPHA))


def _slot_name(cfg: dict) -> str:
    a = _mse_alpha(cfg)
    return "DILATE" if a == 0.0 else "MSE" if a == 1.0 else "DILATE/MSE"


def card_overview(cfg: dict, summary: dict, total_params: int, arch: dict) -> None:
    fig, ax = _setup_card((13.0, 7.6))
    y = _header(ax, "Model card", f"T1DMAI  ·  {_human_count(total_params)} parameter model",
                "Encoder-only transformer for Type 1 diabetes behavioral-dynamics forecasting")

    # summary['best'][...] is None on an absent or all-NaN column, so format tolerantly
    def _b(key: str, fmt: str, suffix: str = "") -> str:
        v = summary.get('best', {}).get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return fmt.format(v) + suffix

    tiles = [
        ("Parameters",      f"{_human_count(total_params)}",
                            f"{total_params:,} exact",  NAVY),
        ("Best val loss",   _b('val_loss_total', '{:.4f}'),
                            f"pinball + {_slot_name(cfg)}  ·  step {_b('val_loss_step', '{:,}')}",  CLAY),
        ("Best MARD @30m",  _b('mard_30m', '{:.2f}', '%'),
                            "mean absolute relative diff. (@30 min)",  CLAY),
        ("Best Clarke A @30m", _b('clarke_A_30m', '{:.2f}', '%'),
                            "clinically-acceptable zone (@30 min)",  TEAL),
    ]
    n = len(tiles); pad = 0.014
    x0 = 0.025; tile_w = (0.95 - (n - 1) * pad) / n; tile_h = 0.205
    tile_y = y - tile_h - 0.005
    for i, (label, big, sub, color) in enumerate(tiles):
        _stat_tile(ax, x0 + i * (tile_w + pad), tile_y, tile_w, tile_h,
                   label, big, sub, color)

    # Diagnostic probe, best-over-run from summary.json; '—' when it was off, so a
    # TIME_PROBE_ENABLED=False run still lays out.
    tod_tiles = [
        ("TOD MAE",      _b('tod_mae_h', '{:.2f}', ' h'), "hour-of-day error (lower better)",  TEAL),
        ("Clock ±2h",    _b('tod_acc_2h', '{:.0f}', '%'), "origin decoded within 2 h",         SAGE),
        ("Confidence R", _b('tod_conf', '{:.2f}'),        "probe vector magnitude",            GOLD),
    ]
    m = len(tod_tiles); tod_w = (0.95 - (m - 1) * pad) / m; tod_h = 0.15
    tod_y = tile_y - tod_h - 0.018
    for i, (label, big, sub, color) in enumerate(tod_tiles):
        _stat_tile(ax, x0 + i * (tod_w + pad), tod_y, tod_w, tod_h,
                   label, big, sub, color)

    y2 = tod_y - 0.045
    facts = [
        ("Architecture",
         f"D_MODEL = {cfg['d_model']}    layers = {cfg['n_layers']}    "
         f"heads = {cfg['n_heads']}    head_dim = {cfg['d_model']//cfg['n_heads']}    "
         f"FFN = {cfg['ffn_dim']}"),
        ("Context window",
         f"{cfg['min_context_patches']}–{cfg['max_context_patches']} patches    "
         f"({cfg['min_context_patches']*cfg['patch_size']*5//60}–"
         f"{cfg['max_context_patches']*cfg['patch_size']*5//60} hours of CGM history)"),
        ("Prediction horizon",
         f"{cfg['prediction_horizon_hours']} h    "
         f"({cfg['prediction_patches']} patches × {cfg['patch_size']} timesteps)"),
        ("Output",
         f"BG quantile fan    "
         f"({arch['n_quantiles']}-τ ascending, Kovatchev risk space → mg/dL)"),
        ("Excursion recall",
         f"hypo {_b('hypo_recall', '{:.3f}')}    ·    "
         f"hyper {_b('hyper_recall', '{:.3f}')}     (best-over-run)"),
    ]
    y2 = _section(ax, y2, "At a glance")
    for k, v in facts:
        ax.text(0.025, y2, k, fontsize=9.5, color=DIMMED, family=FONT_BODY,
                transform=ax.transAxes)
        ax.text(0.20,  y2, v, fontsize=10, color=INK, family=FONT_MONO,
                transform=ax.transAxes)
        y2 -= 0.044

    ax.plot([0.025, 0.975], [0.055, 0.055], color=RULE, lw=0.6, transform=ax.transAxes)
    ax.text(0.025, 0.025,
            f"Training: {cfg['total_steps']:,} steps · batch {cfg['batch_size']} · "
            f"{_human_count(cfg['total_steps']*cfg['batch_size'])} samples · "
            f"seed {cfg['master_seed']} · on-the-fly simulator, no held-out test set",
            fontsize=8.5, color=DIMMED, transform=ax.transAxes)

    fig.savefig(OUT_DIR / "card_01_overview.png")
    plt.close(fig)




def card_architecture(cfg: dict, total_params: int, arch: dict) -> None:
    fig, ax = _setup_card((11.8, 13.0))
    # inline header: the diagram is tall and needs the room
    ax.text(0.025, 0.975, "ARCHITECTURE", fontsize=8.5, color=CLAY,
            family=FONT_BODY, weight="bold", transform=ax.transAxes)
    ax.text(0.025, 0.945, f"Forward pass · {_human_count(total_params)} arch", fontsize=18, color=INK,
            family=FONT_TITLE, va="top", transform=ax.transAxes)
    ax.text(0.025, 0.885,
            "Pre-norm encoder with two sub-layers per block: temporal self-attention "
            "and a SwiGLU FFN.",
            fontsize=10, color=SLATE, family=FONT_BODY, va="top",
            transform=ax.transAxes)
    ax.plot([0.025, 0.975], [0.855, 0.855], color=RULE, lw=0.9, transform=ax.transAxes)

    # axes-fraction coordinates throughout
    def block(x, y, w, h, title, sub="", fc=CARD, ec=NAVY, title_color=INK,
              title_size=10, sub_size=8.5, bar_color=None):
        if bar_color:
            ax.add_patch(Rectangle((x, y + h - 0.006), w, 0.006, fc=bar_color,
                                    ec="none", transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.005,rounding_size=0.010",
                                     fc=fc, ec=ec, lw=0.9, transform=ax.transAxes))
        if sub:
            ax.text(x + w / 2, y + h * 0.60, title, fontsize=title_size, color=title_color,
                    weight="bold", ha="center", family=FONT_BODY, transform=ax.transAxes)
            ax.text(x + w / 2, y + h * 0.28, sub, fontsize=sub_size, color=DIMMED,
                    ha="center", family=FONT_MONO, transform=ax.transAxes)
        else:
            ax.text(x + w / 2, y + h * 0.50, title, fontsize=title_size, color=title_color,
                    weight="bold", ha="center", family=FONT_BODY, transform=ax.transAxes)

    def arrow(p0, p1, color=MUTED, lw=0.9):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                      color=color, lw=lw, transform=ax.transAxes))

    _pd = config.PATCH_DIM
    _ps_feat = cfg['patch_size']
    block(0.20, 0.795, 0.60, 0.040, f"Input patches  (B, T, {_pd})",
          f"{_ps_feat} timesteps × {config.N_INPUT_FEATURES} features per patch",
          ec=GOLD, bar_color=GOLD, fc=GOLD_T, title_size=10)
    arrow((0.50, 0.795), (0.50, 0.780))

    block(0.30, 0.725, 0.40, 0.045, "Patch embedding",
          f"Linear (bias)   {_pd} → {cfg['d_model']}",
          ec=NAVY, bar_color=NAVY, fc=NAVY_T)
    arrow((0.50, 0.725), (0.50, 0.705))

    bx, by, bw, bh = 0.07, 0.345, 0.86, 0.355
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                 boxstyle="round,pad=0.008,rounding_size=0.018",
                                 fc=SOFT_RULE, ec=RULE, lw=0.9, transform=ax.transAxes))
    ax.text(bx + 0.020, by + bh - 0.035, f"TransformerBlock  × {cfg['n_layers']}",
            fontsize=12, color=INK, family=FONT_TITLE, transform=ax.transAxes)
    ax.text(bx + 0.020, by + bh - 0.058,
            "pre-norm  ·  residual connection on every sub-layer",
            fontsize=8.5, color=DIMMED, family=FONT_BODY, transform=ax.transAxes)

    block(0.12, 0.588, 0.30, 0.032, "RMSNorm", "", ec=RULE, fc=CARD, title_size=9)
    arrow((0.27, 0.588), (0.27, 0.572))
    block(0.12, 0.520, 0.30, 0.050, "Temporal self-attention",
          f"{cfg['n_heads']} heads × dim {cfg['d_model']//cfg['n_heads']}",
          ec=NAVY, bar_color=NAVY, fc=NAVY_T)
    ax.text(0.45, 0.543, "Q/K RMSNorm   ·   RoPE",
            fontsize=9, color=SLATE, family=FONT_BODY, transform=ax.transAxes)
    arrow((0.27, 0.520), (0.27, 0.482))

    block(0.12, 0.448, 0.30, 0.032, "RMSNorm", "", ec=RULE, fc=CARD, title_size=9)
    arrow((0.27, 0.448), (0.27, 0.430))
    block(0.12, 0.378, 0.30, 0.050, f"SwiGLU FFN  ({cfg['ffn_dim']})",
          "x · SiLU(gate(x)) → down",
          ec=SAGE, bar_color=SAGE, fc=SAGE_T)
    ax.text(0.45, 0.403, "gate, up, down linears   ·   no bias",
            fontsize=9, color=SLATE, family=FONT_BODY, transform=ax.transAxes)

    arrow((0.50, 0.340), (0.50, 0.302))

    block(0.32, 0.260, 0.36, 0.040, "Final RMSNorm", "",
          ec=NAVY, bar_color=NAVY, fc=NAVY_T, title_size=10)

    # Both heads read the SAME masked-slot hidden state: the risk-space BG quantile fan, and
    # the time-of-day probe when the checkpoint carries it — a diagnostic, not a forecast.
    # Widths come from _derive_arch, off the weights, never from config.py.
    _hh = arch['bg_head_hidden']
    _d = cfg['d_model']
    _ps = cfg['patch_size']
    _nq = arch['n_quantiles']
    heads = [
        (TEAL, TEAL_T, "BG quantile head",
         f"Linear({_d}→{_hh})→SiLU→Linear→SiLU→Linear({_hh}→{_nq})  ·  {_ps} steps per patch",
         f"{_nq}-τ risk-space fan → kovatchev_f_inv → mg/dL"),
    ]
    if arch.get('time_probe'):
        _th = arch['time_probe_hidden']
        _nb = arch['time_probe_bins']
        heads.append(
            (PLUM, PLUM_T, "Time-of-day probe  ·  diagnostic",
             f"Linear({_d}→{_th})→SiLU→Linear({_th}→{_nb})  ·  per patch",
             f"{_nb} hour bins → clock-face  ·  co-trains trunk, not a forecast"),
        )

    n_heads_drawn = len(heads)
    hgap = 0.022
    w = (0.95 - (n_heads_drawn - 1) * hgap) / n_heads_drawn
    h = 0.082
    y0 = 0.075
    centers = [0.025 + i * (w + hgap) + w / 2 for i in range(n_heads_drawn)]

    arrow((0.50, 0.260), (0.50, 0.172), color=MUTED)
    ax.plot([centers[0], centers[-1]], [0.172, 0.172], color=MUTED, lw=0.9,
            transform=ax.transAxes)
    for cxh in centers:
        arrow((cxh, 0.172), (cxh, 0.159), color=MUTED)

    title_fs = 9.5 if n_heads_drawn == 1 else (9.0 if n_heads_drawn == 2 else 8.5)
    line_fs = 8.0 if n_heads_drawn == 1 else (7.0 if n_heads_drawn == 2 else 6.2)
    for i, (color, color_t, title, line1, line2) in enumerate(heads):
        x0 = 0.025 + i * (w + hgap)
        ax.add_patch(Rectangle((x0, y0 + h - 0.008), w, 0.008,
                                fc=color, ec="none", transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                     boxstyle="round,pad=0.005,rounding_size=0.012",
                                     fc=color_t, ec=color, lw=0.9, transform=ax.transAxes))
        ax.text(x0 + w / 2, y0 + h - 0.026, title,
                fontsize=title_fs, color=INK, weight="bold", ha="center",
                family=FONT_BODY, transform=ax.transAxes)
        ax.text(x0 + w / 2, y0 + h * 0.45, line1,
                fontsize=line_fs, color=SLATE, ha="center",
                family=FONT_MONO, transform=ax.transAxes)
        ax.text(x0 + w / 2, y0 + 0.014, line2,
                fontsize=line_fs, color=DIMMED, ha="center", style="italic",
                family=FONT_BODY, transform=ax.transAxes)

    ax.plot([0.025, 0.975], [0.055, 0.055], color=RULE, lw=0.6, transform=ax.transAxes)
    ax.text(0.025, 0.025,
            f"Total parameters: {total_params:,}    ·    "
            f"FFN_DIM = {cfg['ffn_dim'] // cfg['d_model']} × D_MODEL    ·    "
            f"BG_HEAD_HIDDEN = {arch['bg_head_hidden'] // cfg['d_model']} × D_MODEL    ·    "
            f"{arch['n_quantiles']}-τ quantile fan",
            fontsize=9, color=DIMMED, transform=ax.transAxes)

    fig.savefig(OUT_DIR / "card_02_architecture.png")
    plt.close(fig)




def card_param_breakdown(groups: dict[str, int], total: int) -> None:
    items = sorted(groups.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    vals = np.array([v for _, v in items])
    pct = 100 * vals / total

    # ordered by size, largest on the primary accent
    palette = [SAGE, TEAL, NAVY, CLAY, GOLD, PLUM, MUTED, DIMMED, SLATE]
    colors = [palette[i % len(palette)] for i in range(len(names))]

    fig = plt.figure(figsize=(14.0, 9.4), facecolor=PAPER)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.20,
                          left=0.04, right=0.97, top=0.66, bottom=0.07)

    head_ax = fig.add_axes([0.0, 0.64, 1.0, 0.36]); head_ax.axis("off")
    _header(head_ax, "Parameters",
            "Where the parameters live",
            f"Total: {total:,}  ·  largest contributor: {names[0]} ({pct[0]:.1f}%)",
            y_top=0.92)

    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(PAPER)
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="none", height=0.60)
    ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, vals.max() * 1.18)
    ax.tick_params(axis="x", labelsize=9, colors=SLATE, bottom=True)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(RULE)
    ax.set_xlabel("parameter count", color=SLATE, fontsize=9.5)
    ax.xaxis.grid(True, linestyle=":", alpha=0.6, color=RULE)
    ax.set_axisbelow(True)
    for bar, v, p in zip(bars, vals, pct):
        ax.text(bar.get_width() + vals.max() * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,}    {p:.1f}%", va="center", fontsize=9, color=INK,
                family=FONT_MONO)

    ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(PAPER)
    wedges, _ = ax.pie(
        vals, colors=colors, startangle=90, counterclock=False,
        wedgeprops={"edgecolor": PAPER, "linewidth": 2, "width": 0.32},
    )
    ax.text(0, 0.10, f"{_human_count(total)}", ha="center", va="center",
            fontsize=22, color=INK, family=FONT_TITLE)
    ax.text(0, -0.10, "parameters", ha="center", va="center",
            fontsize=9, color=DIMMED, family=FONT_BODY)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.25, 1.25)

    fig.savefig(OUT_DIR / "card_03_param_breakdown.png")
    plt.close(fig)




def _policy_wording(cfg: dict) -> dict[str, str]:
    """Card wording for what a masked patch carries, off the checkpoint's own stamp.

    The card is published, so a ``blind`` checkpoint must not be described in the announced
    convention's terms: its dose channels are withheld at ``data.zero_dose_fill``, which makes
    every dose-response claim on the card false and the what-if regime unavailable.
    Returns ``policy``, ``dose_note``, ``regime``, ``row``, ``footer`` and ``caveat`` (empty
    under ``announced``).
    """
    from data import MASKED_CHANNEL_POLICY_BLIND, stored_masked_channel_policy
    policy = stored_masked_channel_policy(cfg)
    if policy == MASKED_CHANNEL_POLICY_BLIND:
        return {
            "policy": policy,
            "dose_note": "withheld where masked",
            "regime": "blind (unconditioned)",
            "row": "blind",
            "footer": ("masked-patch carbs/insulin are withheld at the no-dose fill "
                       "(blind regime)"),
            "caveat": ("Masked patches carry no dose information: carb, insulin and\n"
                       "exercise are withheld with BG at the no-dose fill. The model\n"
                       "cannot respond to an announced dose, so what-if response is\n"
                       "not available."),
        }
    return {
        "policy": policy,
        "dose_note": "announced future",
        "regime": "announced (conditioned)",
        "row": "announced",
        "footer": "masked-patch carbs/insulin are announced (what-if regime)",
        "caveat": "",
    }




def card_io_schema(cfg: dict, arch: dict) -> None:
    fig, ax = _setup_card((13.8, 9.8))
    # panels are pinned at absolute fractions, not flowed from the returned y, so lift the
    # header to keep its rule clear of the panel tops at 0.79
    y = _header(ax, "Inputs & Outputs",
                "Tensor shapes and channel schema",
                "What the model consumes (per patch) and what it returns (per timestep).",
                y_top=0.985)

    p_x, p_y, p_w, p_h = 0.025, 0.43, 0.45, 0.36
    ax.add_patch(FancyBboxPatch((p_x, p_y), p_w, p_h,
                                 boxstyle="round,pad=0.008,rounding_size=0.014",
                                 fc=CARD, ec=RULE, lw=0.9, transform=ax.transAxes))
    ax.add_patch(Rectangle((p_x, p_y + p_h - 0.008), p_w, 0.008, fc=NAVY, ec="none",
                            transform=ax.transAxes))
    ax.text(p_x + 0.015, p_y + p_h - 0.035, "PER-PATCH INPUT", fontsize=9, color=NAVY,
            weight="bold", transform=ax.transAxes)
    ax.text(p_x + 0.015, p_y + p_h - 0.060,
            f"{cfg['patch_size']} timesteps × {config.N_INPUT_FEATURES} features  =  {config.PATCH_DIM} numbers",
            fontsize=9.5, color=SLATE, family=FONT_BODY, transform=ax.transAxes)

    # Exactly N_INPUT_FEATURES: bg, three dose plan channels, the mask bit. Exercise is a
    # carb-equivalent glucose-disposal curve on the carb scale, not an intensity.
    # What feats 1-3 carry on a MASKED patch is the checkpoint's policy, not the schema.
    # Feat 4 is the one input that is not a normalized signal: one BIT per patch, 1.0 where
    # feat 0 is withheld. Announced rather than inferred — a span may end at the last patch,
    # start at patch 0 or sit between visible ones, and z = 0 in a masked bg slot is a legal
    # reading, not a sentinel.
    wording = _policy_wording(cfg)
    feats = [
        ("BG absolute",           "mg/dL",     "always · 0 where masked"),
        ("Carbs",                 "g / 5 min", wording["dose_note"]),
        ("Insulin",               "U / 5 min", wording["dose_note"]),
        ("Exercise (carb-equiv)", "g / 5 min", wording["dose_note"]),
        ("BG masked",             "bit/patch", "1 where BG is withheld"),
    ]
    assert len(feats) == config.N_INPUT_FEATURES
    row_y = p_y + p_h - 0.090
    for i, (name, unit, note) in enumerate(feats):
        if i % 2 == 1:
            ax.add_patch(Rectangle((p_x + 0.010, row_y - 0.013), p_w - 0.020, 0.025,
                                    fc=SOFT_RULE, ec="none", transform=ax.transAxes))
        ax.text(p_x + 0.018, row_y, f"{i+1:>2}.", fontsize=9, color=DIMMED,
                family=FONT_MONO, transform=ax.transAxes)
        ax.text(p_x + 0.045, row_y, name, fontsize=10, color=INK,
                family=FONT_BODY, transform=ax.transAxes)
        ax.text(p_x + 0.220, row_y, unit, fontsize=9, color=SLATE,
                family=FONT_MONO, transform=ax.transAxes)
        ax.text(p_x + 0.320, row_y, note, fontsize=9, color=DIMMED,
                family=FONT_BODY, transform=ax.transAxes, style="italic")
        row_y -= 0.029

    # Under ``blind`` the caveat is the load-bearing half: a reader taking the announced
    # framing would expect a dose response the weights cannot produce.
    ax.text(p_x + 0.018, row_y - 0.006,
            f"Masked-patch dose channels: {wording['regime']}",
            fontsize=9, color=NAVY, weight="bold", transform=ax.transAxes)
    if wording["caveat"]:
        ax.text(p_x + 0.018, row_y - 0.032, wording["caveat"],
                fontsize=8.5, color=SLATE, va="top", family=FONT_BODY,
                transform=ax.transAxes)

    o_x, o_y, o_w, o_h = 0.515, 0.58, 0.46, 0.21
    ax.add_patch(FancyBboxPatch((o_x, o_y), o_w, o_h,
                                 boxstyle="round,pad=0.008,rounding_size=0.014",
                                 fc=CARD, ec=RULE, lw=0.9, transform=ax.transAxes))
    ax.add_patch(Rectangle((o_x, o_y + o_h - 0.008), o_w, 0.008, fc=CLAY, ec="none",
                            transform=ax.transAxes))
    ax.text(o_x + 0.015, o_y + o_h - 0.035, "PER-TIMESTEP OUTPUT", fontsize=9,
            color=CLAY, weight="bold", transform=ax.transAxes)
    ax.text(o_x + 0.015, o_y + o_h - 0.060,
            f"Risk-space BG quantile fan  ·  {arch['n_quantiles']} ascending τ  ·  "
            f"{cfg['prediction_patches']*cfg['patch_size']} timesteps",
            fontsize=9.5, color=SLATE, family=FONT_BODY, transform=ax.transAxes)

    # ascending fan of N_QUANTILES risk-space quantiles per timestep; inference inverts each
    # through kovatchev_f_inv to mg/dL, τ=0.5 being the headline forecast
    levels = config.QUANTILE_LEVELS
    mid = len(levels) // 2
    chans = []
    for i, tau in enumerate(levels):
        role = "median (headline)" if i == mid else ("lower band" if i < mid else "upper band")
        color = CLAY if i == mid else TEAL
        chans.append((f"{tau:g}", f"τ = {tau:g}", "risk → mg/dL", role, color))
    row_y = o_y + o_h - 0.095
    dy = 0.105 / max(len(chans), 1)
    for idx, name, unit, note, color in chans:
        circ_r = 0.012
        ax.add_patch(plt.Circle((o_x + 0.030, row_y + 0.004), circ_r, fc=color,
                                 ec="none", transform=ax.transAxes))
        ax.text(o_x + 0.060, row_y, name, fontsize=10, color=INK,
                weight="bold", family=FONT_BODY, transform=ax.transAxes)
        ax.text(o_x + 0.220, row_y, unit, fontsize=9, color=SLATE,
                family=FONT_MONO, transform=ax.transAxes)
        ax.text(o_x + 0.330, row_y, note, fontsize=9, color=DIMMED, style="italic",
                family=FONT_BODY, transform=ax.transAxes)
        row_y -= dy

    n_x, n_y, n_w, n_h = 0.515, 0.43, 0.46, 0.135
    ax.add_patch(FancyBboxPatch((n_x, n_y), n_w, n_h,
                                 boxstyle="round,pad=0.008,rounding_size=0.014",
                                 fc=GOLD_T, ec=GOLD, lw=0.9, transform=ax.transAxes))
    _forecast_line = (
        "A time-of-day probe adds a diagnostic clock output (not a forecast)."
        if arch.get('time_probe') else
        "The τ=0.5 median is the headline forecast (kovatchev_f_inv → mg/dL)."
    )
    ax.text(n_x + 0.015, n_y + n_h - 0.030, "BG IS THE ONLY FORECAST OUTPUT",
            fontsize=9, color=GOLD, weight="bold",
            transform=ax.transAxes)
    ax.text(n_x + 0.015, n_y + n_h - 0.060,
            "The BG head emits an ascending fan of risk-space quantiles;\n"
            "there are no dynamics outputs. Carbs / insulin / exercise are inputs only;\n"
            "IS / HGO are simulator latents — dropped, neither in nor out.\n"
            f"{_forecast_line}",
            fontsize=9, color=SLATE, va="top", family=FONT_BODY,
            transform=ax.transAxes)

    _nq = arch['n_quantiles']
    _time_line = (
        f"\ntime_pred(diag) : (B, {cfg['prediction_patches']}, {arch['time_probe_bins']})"
        f"  ← hour-of-day bin logits"
        if arch.get('time_probe') else ""
    )
    for (x0, title, body, color) in [
        (0.025, "Input tensor shape",
         f"patches         : (B, T, {config.PATCH_DIM})\n"
         f"T               : [{cfg['min_context_patches']}, "
         f"{cfg['max_context_patches']}+{cfg['prediction_patches']}]\n"
         f"attention mask  : (T, T)  bool\n"
         f"last_bg anchor  : (B,)  mg/dL",
         NAVY),
        (0.515, "Output tensor shapes",
         f"q_tau (risk)    : (B, {cfg['prediction_patches']}, "
         f"{cfg['patch_size']}, {_nq})\n"
         f"median (risk)   : (B, {cfg['prediction_patches']}, "
         f"{cfg['patch_size']})\n"
         f"median_bg       : (B·{cfg['prediction_patches']}·{cfg['patch_size']},)  mg/dL\n"
         f"bands           : (P, S, {_nq})  mg/dL  ← f_inv(q_tau)"
         + _time_line,
         CLAY),
    ]:
        w, h = 0.46, 0.32
        ax.add_patch(FancyBboxPatch((x0, 0.085), w, h,
                                     boxstyle="round,pad=0.008,rounding_size=0.014",
                                     fc=CARD, ec=RULE, lw=0.9, transform=ax.transAxes))
        ax.add_patch(Rectangle((x0, 0.085 + h - 0.008), w, 0.008, fc=color, ec="none",
                                transform=ax.transAxes))
        ax.text(x0 + 0.015, 0.085 + h - 0.030, title.upper(),
                fontsize=9, color=color, weight="bold",
                transform=ax.transAxes)
        ax.text(x0 + 0.015, 0.085 + h - 0.085, body,
                fontsize=10, color=INK, family=FONT_MONO, va="top",
                transform=ax.transAxes)

    fig.savefig(OUT_DIR / "card_04_io_schema.png")
    plt.close(fig)




def card_training_recipe(cfg: dict) -> None:
    fig, ax = _setup_card((11.8, 10.2))
    y = _header(ax, "Training recipe",
                "Knobs in effect for this run",
                "Resolved config — config.py defaults with CLI-flag overrides applied.")

    sections = [
        ("Data", NAVY,
         [
             ("Source",            "on-the-fly T1DMSIM simulator (no held-out split)"),
             ("Patient sampling",  f"normal mix  +  {int(100*cfg['patient_uniform_sample_prob'])}% uniform-skill draws (tail oversampling)"),
             ("Simulator warmup",  f"{cfg['simulator_warmup_hours']:.0f} h discarded per run"),
             ("Total samples",     f"{_human_count(cfg['total_steps']*cfg['batch_size'])}    "
                                   f"({cfg['total_steps']:,} steps × batch {cfg['batch_size']})"),
         ]),
        ("Optimizer", TEAL,
         [
             ("Muon (2D weights)",   f"lr = {cfg['muon_lr']:.4f}    "
                                      f"momentum = {cfg['muon_momentum']:.2f}    "
                                      f"NS = {config.MUON_NS_ITERATIONS} iters    "
                                      f"wd = {config.MUON_WEIGHT_DECAY:.2f}"),
             ("AdamW (1D / embeds)", f"lr = {cfg['adam_lr']:.0e}    "
                                      f"β = {config.ADAM_BETAS}    wd = {cfg['adam_weight_decay']:.2f}"),
             ("Schedule",            f"linear warmup {cfg['warmup_steps']:,}    →    "
                                      f"cosine decay to {cfg['lr_min_ratio']*100:.0f}% of peak"),
             ("Grad clip",           f"‖∇‖₂  ≤  {cfg['gradient_clip_norm']:.1f}"),
             ("EMA (eval only)",     f"decay = {cfg['ema_decay']}"),
         ]),
        ("Loss", CLAY,
         [
             ("Quantile pinball",    f"L_Q  ·  {len(config.QUANTILE_LEVELS)}-τ fan in Kovatchev RISK space"),
             ("DILATE (shape/TDI)",  f"L_D  ·  α = {config.DILATE_ALPHA:.2f}·shape + {1-config.DILATE_ALPHA:.2f}·TDI"
                                      f"    soft-DTW γ = {config.DILATE_GAMMA:.1f}"),
             ("MSE (median)",       f"L_M  ·  slot = {1 - _mse_alpha(cfg):.2f}·L_D + {_mse_alpha(cfg):.2f}·L_M"),
             ("Combination",         f"learned Kendall-Gal log-variances  σ_Q, σ_D  (init {config.KENDALL_LOGVAR_INIT:.1f})"),
             ("Clinical thresholds", f"hypo {cfg['bg_hypo_threshold']:.0f} / hyper "
                                      f"{cfg['bg_hyper_threshold']:.0f} mg/dL"),
             ("Quantile spread",     f"softplus gaps, floor {config.BG_QUANTILE_SPREAD_MIN:.0e}  (strict ordering)"),
         ]),
        ("Reproducibility", SAGE,
         [
             ("Master seed",         f"{cfg['master_seed']}"),
             ("DataLoader workers",  f"{cfg.get('num_workers','?')}"),
             ("Validation interval", f"every {cfg['validation_interval']:,} steps    "
                                      f"({config.VALIDATION_N_PATIENTS} patients)"),
             ("Checkpoint interval", f"every {cfg['checkpoint_interval']:,} steps"),
         ]),
    ]

    cur = y
    for title, color, rows in sections:
        cur = _section(ax, cur, title, color=color)
        for k, v in rows:
            ax.text(0.045, cur, k, fontsize=9.5, color=DIMMED, family=FONT_BODY,
                    transform=ax.transAxes)
            ax.text(0.30,  cur, v, fontsize=10, color=INK, family=FONT_MONO,
                    transform=ax.transAxes)
            cur -= 0.035
        cur -= 0.015

    fig.savefig(OUT_DIR / "card_05_training_recipe.png")
    plt.close(fig)




def card_loss_design(cfg: dict, train: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(13.8, 8.8), facecolor=PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.18,
                          left=0.04, right=0.97, top=0.66, bottom=0.08)

    _a = config.DILATE_ALPHA
    _ra = _mse_alpha(cfg)
    terms = [
        ("Quantile pinball  (L_Q)",
         f"Σ_τ ρ_τ(f(BG_true) − q_τ)  over the {len(config.QUANTILE_LEVELS)}-τ fan",
         "Kovatchev RISK space  ·  the calibrated forecast",
         NAVY),
        ("DILATE shape  (soft-DTW)",
         "soft-DTW alignment of pred vs true BG curve",
         f"mix weight: α = {_a:.2f}    soft-DTW γ = {config.DILATE_GAMMA:.1f}",
         TEAL),
        ("DILATE temporal (TDI)",
         "temporal-distortion index — penalizes timing drift",
         f"mix weight: 1 − α = {1 - _a:.2f}",
         SAGE),
        ("Median MSE  (L_M)",
         "mean (median − f(BG_true))²  over valid (slot, step)",
         f"D slot = {1 - _ra:.2f}·L_D + {_ra:.2f}·L_M    (MSE_ALPHA = {_ra:.2f})",
         TEAL),
        ("Kendall-Gal combine",
         "½·e^(−2σ_Q)·L_Q + σ_Q  +  ½·e^(−2σ_D)·L_DR + σ_D",
         f"learned log-variances σ_Q, σ_D  (init {config.KENDALL_LOGVAR_INIT:.1f}, clamped [−7, 7])",
         CLAY),
    ]

    _n_terms = len(terms)
    _count_word = {3: "Three", 4: "Four", 5: "Five", 6: "Six",
                   7: "Seven", 8: "Eight"}.get(_n_terms, str(_n_terms))

    head_ax = fig.add_axes([0.0, 0.64, 1.0, 0.36]); head_ax.axis("off")
    _header(head_ax, "Loss design",
            "Composite training objective",
            f"{_count_word} terms — a quantile pinball, DILATE shape/TDI and a median MSE, "
            "combined under learned Kendall-Gal uncertainty weights.",
            y_top=0.92)

    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(PAPER)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    gap = 0.013
    h = (1.0 - (_n_terms - 1) * gap) / _n_terms
    for i, (name, formula, weight, color) in enumerate(terms):
        y0 = 1.0 - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((0.005, y0), 0.99, h,
                                     boxstyle="round,pad=0.005,rounding_size=0.012",
                                     fc=CARD, ec=RULE, lw=0.9))
        ax.add_patch(Rectangle((0.005, y0), 0.008, h, fc=color, ec="none"))
        ax.text(0.030, y0 + h - 0.045, name, fontsize=11, color=INK,
                weight="bold", family=FONT_BODY)
        ax.text(0.030, y0 + h * 0.45, formula, fontsize=9.5, color=SLATE,
                family=FONT_MONO)
        ax.text(0.030, y0 + 0.025, weight, fontsize=8.5, color=DIMMED,
                family=FONT_MONO)

    # the learned Kendall-Gal log-variances per training step — a trace of the L_Q / L_DR
    # balance as it moves, not a static split
    ax = fig.add_subplot(gs[0, 1]); ax.set_facecolor(PAPER)
    sig_series = [("log_sigma_Q", "σ_Q  (pinball)", NAVY),
                  ("log_sigma_D", "σ_D  (DILATE/MSE)", TEAL)]
    steps = train.get("step")
    drew = False
    if steps is not None:
        for col, lbl, color in sig_series:
            y = train.get(col)
            if y is None or np.all(np.isnan(y)):
                continue
            ax.plot(steps, y, color=color, linewidth=1.8, label=lbl)
            drew = True
    if drew:
        ax.set_xlabel("training step", color=SLATE)
        ax.set_ylabel("log-variance  (log σ)", color=SLATE)
        ax.legend(loc="best", frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "log-σ trace unavailable", ha="center", va="center",
                fontsize=10, color=DIMMED, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Learned loss weighting", loc="left",
                 fontsize=11, color=INK, weight="bold", pad=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5, color=RULE)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(RULE)
    ax.spines["left"].set_visible(True);   ax.spines["left"].set_color(RULE)
    ax.tick_params(bottom=True, left=True, colors=SLATE)

    fig.savefig(OUT_DIR / "card_06_loss_design.png")
    plt.close(fig)




def card_compute_budget(train: dict[str, np.ndarray], tsum: dict, cfg: dict) -> None:
    elapsed_h = tsum["progress"]["elapsed_hours"]
    sps = tsum["progress"]["steps_per_second"]
    samples = cfg["total_steps"] * cfg["batch_size"]
    # Every window is a fresh simulator patient, so patients seen == window draws; hours seen
    # is their CGM-time (avg context + prediction patches × 30 min/patch); patches seen is the
    # PADDED sequence the model forward-passes (context padded to MAX).
    patients = samples
    _avg_patches = (cfg["min_context_patches"] + cfg["max_context_patches"]) / 2 + cfg["prediction_patches"]
    hours_seen = samples * _avg_patches * cfg["patch_size"] * 5 / 60.0
    patches_seen = samples * (cfg["max_context_patches"] + cfg["prediction_patches"])
    gpu_mem_mb = float(np.nanmean(train["gpu_memory_mb"])) if "gpu_memory_mb" in train else float("nan")
    st = train["step_time_seconds"]
    st_median_ms = 1000.0 * float(np.nanmedian(st))
    st_p99_ms = 1000.0 * float(np.nanpercentile(st[np.isfinite(st)], 99))

    fig = plt.figure(figsize=(13.8, 9.2), facecolor=PAPER)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.45, 0.55],
                          left=0.04, right=0.97, top=0.66, bottom=0.08,
                          wspace=0.18, hspace=0.50)

    head_ax = fig.add_axes([0.0, 0.64, 1.0, 0.36]); head_ax.axis("off")
    _header(head_ax, "Compute", "Training budget",
            f"{elapsed_h:.2f} h wallclock  ·  {sps:.2f} steps/sec  ·  "
            f"{_human_count(samples)} samples  ·  {_human_count(hours_seen)} CGM-h seen", y_top=0.92)

    tile_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    tile_data = [
        ("Wallclock", f"{elapsed_h:.2f} h", "of training", NAVY),
        ("Throughput", f"{sps:.2f}", "steps / sec", TEAL),
        ("Samples seen", f"{_human_count(samples)}",
         f"{cfg['total_steps']:,} × batch {cfg['batch_size']}", SAGE),
    ]
    for ax, (label, big, sub, color) in zip(tile_axes, tile_data):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_facecolor(PAPER)
        ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.92,
                                     boxstyle="round,pad=0.005,rounding_size=0.018",
                                     fc=CARD, ec=RULE, lw=0.9))
        ax.add_patch(Rectangle((0.02, 0.94), 0.96, 0.03, fc=color, ec="none"))
        ax.text(0.5, 0.82, label.upper(), fontsize=9, color=DIMMED,
                weight="bold", ha="center")
        ax.text(0.5, 0.46, big, fontsize=26, color=INK, family=FONT_TITLE, ha="center")
        ax.text(0.5, 0.18, sub, fontsize=9, color=DIMMED, ha="center")

    ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(PAPER)
    s = train["step"]
    ema = np.empty_like(s, dtype=float)
    ema[0] = st[0]
    for i in range(1, len(s)):
        ema[i] = 0.97 * ema[i - 1] + 0.03 * st[i]
    ax.fill_between(s, 0, 1000 * ema, color=NAVY, alpha=0.10)
    ax.plot(s, 1000 * ema, color=NAVY, linewidth=1.8, label="step time (EMA)")
    ax.axhline(st_median_ms, color=CLAY, linestyle="--", linewidth=1.0,
               label=f"median  =  {st_median_ms:.0f} ms")

    ax.set_xlabel("training step", color=SLATE)
    ax.set_ylabel("ms / step",     color=SLATE)
    ax.set_title("Per-step latency", loc="left",
                 fontsize=11, color=INK, weight="bold", pad=10)
    ax.set_ylim(0, max(st_median_ms * 2.4, 1000 * float(np.nanpercentile(ema, 99)) * 1.2))
    ax.grid(True, linestyle=":", alpha=0.5, color=RULE)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(RULE)
    ax.spines["left"].set_visible(True);   ax.spines["left"].set_color(RULE)
    ax.tick_params(bottom=True, left=True, colors=SLATE)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    strip_ax = fig.add_axes([0.04, 0.39, 0.93, 0.05]); strip_ax.axis("off")
    strip_ax.set_xlim(0, 1); strip_ax.set_ylim(0, 1)
    extras = [
        ("Patients seen",     _human_count(patients)),
        ("Hours seen ≈",      f"{_human_count(hours_seen)} h"),
        ("Patches seen ≈",    _human_count(patches_seen)),
        ("Median step",       f"{st_median_ms:.0f} ms"),
        ("p99 step",          f"{st_p99_ms:.0f} ms"),
        ("GPU mem (mean)",    f"{gpu_mem_mb:.0f} MB"),
    ]
    xs = [0.00, 0.165, 0.33, 0.50, 0.65, 0.81]
    for x, (k, v) in zip(xs, extras):
        strip_ax.text(x, 0.75, k.upper(), fontsize=8, color=DIMMED, weight="bold")
        strip_ax.text(x, 0.18, v, fontsize=10.5, color=INK, family=FONT_MONO)

    fig.savefig(OUT_DIR / "card_07_compute_budget.png")
    plt.close(fig)




def card_metrics_card(cfg: dict, val: dict[str, np.ndarray]) -> None:
    # ~32 rows + 6 eyebrows at full strength, so the pitch is tightened and the figure
    # stretched to keep the table inside ax [0, 1]
    ROW = 0.0145      # per-row vertical pitch (axes fraction)
    BAND = 0.014      # alternating row-shade height (axes fraction)
    SEC_PAD = 0.003   # trailing gap after each section
    SEC_HEAD = 0.026  # gap between a section eyebrow and its first row
    CARD_H = 17.5     # inches, sized for the full section set below

    sections = [
        ("Loss & overall accuracy", NAVY,
         [("val_loss_total",  False, "total loss (lower better)",  "{:.4f}", "{:.4f}"),
          ("evalfix_mard@30",     False, "MARD @30m %",             "{:.2f}", "{:.2f}"),
          ("evalfix_mard@60",     False, "MARD @60m %",             "{:.2f}", "{:.2f}"),
          ("evalfix_mard@120",    False, "MARD @120m %",            "{:.2f}", "{:.2f}"),
          ("evalfix_clarke_A@30", True,  "Clarke zone A @30m %",    "{:.2f}", "{:.2f}"),
          ("evalfix_clarke_A@60", True,  "Clarke zone A @60m %",    "{:.2f}", "{:.2f}"),
          ("evalfix_clarke_A@120",True,  "Clarke zone A @120m %",   "{:.2f}", "{:.2f}")]),
        ("Multi-horizon BG RMSE  (mg/dL)", TEAL,
         [("bg_rmse_30",  False, "30 min",  "{:.2f}", "{:.2f}"),
          ("bg_rmse_60",  False, "60 min",  "{:.2f}", "{:.2f}"),
          ("bg_rmse_120", False, "120 min", "{:.2f}", "{:.2f}"),
          ("bg_rmse_180", False, "180 min", "{:.2f}", "{:.2f}"),
          ("bg_rmse_360", False, "360 min", "{:.2f}", "{:.2f}"),
          ("bg_rmse_480", False, "480 min", "{:.2f}", "{:.2f}")]),
        ("Excursion detection", CLAY,
         [("hypo_recall",     True, f"hypo recall  (BG < {config.BG_HYPO_THRESHOLD:.0f} mg/dL)",   "{:.3f}", "{:.3f}"),
          ("hypo_precision",  True, "hypo precision",                 "{:.3f}", "{:.3f}"),
          ("hyper_recall",    True, f"hyper recall  (BG > {config.BG_HYPER_THRESHOLD:.0f} mg/dL)", "{:.3f}", "{:.3f}"),
          ("hyper_precision", True, "hyper precision",                "{:.3f}", "{:.3f}")]),
        # Coverage of the 90% band (τ 0.05–0.95), target 0.90. Over- and under-coverage both
        # miss, so these track toward 0.90 as a distance, the closest a best/last-10 table gets.
        # Every coverage row is followed by the mean band WIDTH that bought it — coverage alone
        # is not a calibration claim. A run with no sharpness column renders "—" rather than
        # dropping the row, so the omission stays visible.
        ("Calibration  (90% band coverage, target 0.90)", GOLD,
         [("coverage90@30",  True,  "coverage @30m",              "{:.3f}", "{:.3f}"),
          ("sharp90@30",     False, "  band width @30m  (mg/dL)", "{:.1f}", "{:.1f}"),
          ("coverage90@60",  True,  "coverage @60m",              "{:.3f}", "{:.3f}"),
          ("sharp90@60",     False, "  band width @60m  (mg/dL)", "{:.1f}", "{:.1f}"),
          ("coverage90@120", True,  "coverage @120m",             "{:.3f}", "{:.3f}"),
          ("sharp90@120",    False, "  band width @120m (mg/dL)", "{:.1f}", "{:.1f}")]),
    ]
    # rows added since CARD_H was sized; the canvas grows by exactly their space, so the
    # physical pitch and every font size stay put
    _EXTRA_ROWS = 3

    # CG-EGA (Kovatchev 2004): %AP higher better, %EP lower better, per region. Read straight
    # out of validation_log.csv, so the section stands or falls with CGEGA_COLUMNS_TRUSTWORTHY;
    # withheld, the canvas shortens by exactly their space and the table ends a section early.
    cgega_section = (
        "CG-EGA clinical accuracy", PLUM,
        [("cgega_ap_hypo",  True,  "%AP hypo  (accurate, higher better)",    "{:.3f}", "{:.3f}"),
         ("cgega_ep_hypo",  False, "%EP hypo  (erroneous, lower better)",    "{:.3f}", "{:.3f}"),
         ("cgega_ap_eu",    True,  "%AP euglycemic  (higher better)",        "{:.3f}", "{:.3f}"),
         ("cgega_ep_eu",    False, "%EP euglycemic  (lower better)",         "{:.3f}", "{:.3f}"),
         ("cgega_ap_hyper", True,  "%AP hyper  (higher better)",             "{:.3f}", "{:.3f}"),
         ("cgega_ep_hyper", False, "%EP hyper  (lower better)",              "{:.3f}", "{:.3f}")])
    h_scale = 1.0 + _EXTRA_ROWS * ROW
    if CGEGA_COLUMNS_TRUSTWORTHY:
        sections.insert(3, cgega_section)
    else:
        h_scale -= SEC_HEAD + SEC_PAD + len(cgega_section[2]) * ROW
        print("  · card_08_metrics: CG-EGA section omitted "
              "(CGEGA_COLUMNS_TRUSTWORTHY is False)")
    ROW /= h_scale
    BAND /= h_scale
    SEC_PAD /= h_scale
    SEC_HEAD /= h_scale

    fig, ax = _setup_card((13.0, CARD_H * h_scale))
    y = _header(ax, "Validation metrics",
                "Headline results",
                "Best-over-run (with the step where it was achieved) and mean over the "
                "last 10 validation rows. Masked-patch carb/insulin/exercise are "
                f"{_policy_wording(cfg)['regime']}.")

    def _tight_section(yy: float, lab: str, color=NAVY) -> float:
        """Section eyebrow at a tighter gap than ``_section``'s 0.040 — this card cannot
        afford it across every metric section plus the curve-match block."""
        ax.text(0.025, yy, lab.upper(), fontsize=8.5, color=color,
                family=FONT_BODY, weight="bold", transform=ax.transAxes)
        return yy - SEC_HEAD

    def _col(col: str) -> np.ndarray:
        """Column accessor tolerant of an absent column (older CSV schema)."""
        if col in val:
            return val[col]
        return np.full_like(val["step"], np.nan, dtype=float)

    def stat(col: str, higher_is_better: bool):
        yv = _col(col); sv = val["step"]
        mask = np.isfinite(yv); yy, ss = yv[mask], sv[mask]
        if yy.size == 0:
            return float("nan"), 0, float("nan")
        idx = int(np.argmax(yy) if higher_is_better else np.argmin(yy))
        # last 10 ACTUAL validation rows, not the last 10 finite cells: a sparse column would
        # otherwise reach back arbitrarily far and mislabel the window
        last10 = float(np.nanmean(yv[-10:])) if np.isfinite(yv[-10:]).any() else float("nan")
        return yy[idx], int(ss[idx]), last10

    def _fmt(v: float, fmt: str) -> str:
        return fmt.format(v) if np.isfinite(v) else "—"

    ax.text(0.045, y - 0.005, "METRIC", fontsize=8.5, color=DIMMED, weight="bold", transform=ax.transAxes)
    ax.text(0.585, y - 0.005, "BEST  (STEP)", fontsize=8.5, color=DIMMED, weight="bold", transform=ax.transAxes)
    ax.text(0.835, y - 0.005, "LAST-10 MEAN", fontsize=8.5, color=DIMMED, weight="bold", transform=ax.transAxes)
    ax.plot([0.025, 0.975], [y - 0.020, y - 0.020], color=RULE, lw=0.7,
            transform=ax.transAxes)

    cur = y - 0.024
    for title, color, rows in sections:
        cur = _tight_section(cur, title, color=color)
        for i, (col, hib, label, bf, lf) in enumerate(rows):
            if i % 2 == 1:
                ax.add_patch(Rectangle((0.025, cur - BAND / 2), 0.95, BAND,
                                        fc=SOFT_RULE, ec="none", transform=ax.transAxes))
            best, step, l10 = stat(col, hib)
            ax.text(0.045, cur, label, fontsize=9.0, color=INK, family=FONT_BODY,
                    transform=ax.transAxes)
            best_txt = f"{_fmt(best, bf)}   @ {step:,}" if np.isfinite(best) else "—"
            ax.text(0.585, cur, best_txt,
                    fontsize=9.0, color=INK, family=FONT_MONO, transform=ax.transAxes)
            ax.text(0.835, cur, _fmt(l10, lf),
                    fontsize=9.0, color=INK, family=FONT_MONO, transform=ax.transAxes)
            cur -= ROW
        cur -= SEC_PAD

    # per-patch 30-min ΔBG, all off the headline median_bg forecast
    cur = _tight_section(cur, "Forecast curve / trend quality", color=CLAY)
    match_rows = [
        ("bg_curve_corr",   "BG curve corr  (anchor-relative)", "{:.3f}"),
        ("roc_corr",        "ΔBG direction corr  (per-patch)",  "{:.3f}"),
        ("trend_amp_ratio", "ΔBG amplitude ratio  (target ~1)", "{:.3f}"),
    ]
    for i, (col, label, fmt) in enumerate(match_rows):
        if i % 2 == 1:
            ax.add_patch(Rectangle((0.025, cur - BAND / 2), 0.95, BAND,
                                    fc=SOFT_RULE, ec="none", transform=ax.transAxes))
        best, step, l10 = stat(col, True)
        ax.text(0.045, cur, label, fontsize=9.0, color=INK, family=FONT_BODY, transform=ax.transAxes)
        best_txt = f"{_fmt(best, fmt)}   @ {step:,}" if np.isfinite(best) else "—"
        ax.text(0.585, cur, best_txt, fontsize=9.0, color=INK, family=FONT_MONO, transform=ax.transAxes)
        ax.text(0.835, cur, _fmt(l10, fmt), fontsize=9.0, color=INK, family=FONT_MONO, transform=ax.transAxes)
        cur -= ROW

    fig.savefig(OUT_DIR / "card_08_metrics.png")
    plt.close(fig)




def card_evaluation(cfg: dict, reports: dict[str, dict]) -> None:
    """In-domain simulator evaluation, from the post-training ``metrics/sim/stats.json``.

    Surfaces the precision-floored per-horizon hypo decision offsets (``selected_offsets``) and
    the per-horizon hypo/hyper event recall/precision (``event_metrics``).  Skipped entirely
    when no report exists.
    """
    if not reports:
        return  # No reports yet — omit the section gracefully.

    H = REPORT_HORIZONS
    wording = _policy_wording(cfg)
    fig, ax = _setup_card((13.6, 13.2))
    step = None
    for s in reports.values():
        step = (s.get("_meta") or {}).get("step")
        if step is not None:
            break
    sub = ("Post-training evaluation on held-out simulator patients. "
           "Precision-floored hypo decision offsets and per-horizon excursion "
           "recall / precision.")
    if step is not None:
        sub += f"  (checkpoint step {step:,})"
    y = _header(ax, "Evaluation", "In-domain simulator performance", sub)

    def _fmt(v, fmt: str = "{:.2f}") -> str:
        try:
            return fmt.format(float(v)) if v is not None and np.isfinite(float(v)) else "—"
        except (TypeError, ValueError):
            return "—"

    cur = y
    color_cycle = {"sim": TEAL}

    for mode, label, datasets in REPORT_SOURCES:
        stats = reports.get(mode)
        if not stats:
            continue
        present = [d for d in datasets if isinstance(stats.get(d), dict)]
        if not present:
            continue
        mcolor = color_cycle.get(mode, NAVY)
        cur = _section(ax, cur, f"{label}", color=mcolor)

        for ds in present:
            res = stats[ds]
            n_test = res.get("n_test_windows")
            ds_head = ds if mode == "sim" else ds.upper()
            sub_bits = []
            if n_test is not None:
                sub_bits.append(f"{n_test} test windows")
            ax.text(0.045, cur, ds_head, fontsize=10, color=INK, weight="bold",
                    family=FONT_BODY, transform=ax.transAxes)
            if sub_bits:
                ax.text(0.300, cur, "   ·   ".join(sub_bits), fontsize=8.5,
                        color=DIMMED, family=FONT_MONO, transform=ax.transAxes)
            cur -= 0.028

            so = res.get("selected_offsets")
            if isinstance(so, dict) and so.get("hypo"):
                floor = so.get("min_precision")
                floor_txt = f"  (precision floor {_fmt(floor, '{:.2f}')})" if floor is not None else ""
                ax.text(0.060, cur, f"Hypo decision offset{floor_txt}",
                        fontsize=8.5, color=mcolor, weight="bold", style="italic",
                        transform=ax.transAxes)
                cur -= 0.024
                ax.text(0.075, cur, "horizon", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                ax.text(0.280, cur, "offset", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                ax.text(0.430, cur, "cal R / P", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                ax.text(0.680, cur, "test R / P", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                cur -= 0.022
                for j, h in enumerate(H):
                    d = so["hypo"].get(str(h))
                    if not isinstance(d, dict):
                        continue
                    if j % 2 == 1:
                        ax.add_patch(Rectangle((0.060, cur - 0.011), 0.915, 0.022,
                                                fc=SOFT_RULE, ec="none", transform=ax.transAxes))
                    ax.text(0.075, cur, f"{h} min", fontsize=9, color=INK, family=FONT_BODY, transform=ax.transAxes)
                    ax.text(0.280, cur, f"{_fmt(d.get('offset'), '{:.1f}')} mg/dL",
                            fontsize=9, color=INK, family=FONT_MONO, transform=ax.transAxes)
                    ax.text(0.430, cur, f"{_fmt(d.get('cal_recall'))} / {_fmt(d.get('cal_precision'))}",
                            fontsize=9, color=SLATE, family=FONT_MONO, transform=ax.transAxes)
                    ax.text(0.680, cur, f"{_fmt(d.get('test_recall'))} / {_fmt(d.get('test_precision'))}",
                            fontsize=9, color=INK, family=FONT_MONO, transform=ax.transAxes)
                    cur -= 0.025
                cur -= 0.006

            em = res.get("event_metrics")
            if isinstance(em, dict):
                ax.text(0.060, cur, "Event recall / precision  (gain = 1.0)",
                        fontsize=8.5, color=mcolor, weight="bold", style="italic",
                        transform=ax.transAxes)
                cur -= 0.024
                ax.text(0.075, cur, "horizon", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                ax.text(0.280, cur, "hypo R / P", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                ax.text(0.530, cur, "hyper R / P", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                ax.text(0.790, cur, "RMSE pt", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                cur -= 0.022
                for j, h in enumerate(H):
                    d = em.get(str(h))
                    if not isinstance(d, dict):
                        continue
                    hy = d.get("hypo") or {}
                    yp = d.get("hyper") or {}
                    if j % 2 == 1:
                        ax.add_patch(Rectangle((0.060, cur - 0.011), 0.915, 0.022,
                                                fc=SOFT_RULE, ec="none", transform=ax.transAxes))
                    ax.text(0.075, cur, f"{h} min", fontsize=9, color=INK, family=FONT_BODY, transform=ax.transAxes)
                    ax.text(0.280, cur, f"{_fmt(hy.get('recall'))} / {_fmt(hy.get('precision'))}",
                            fontsize=9, color=NAVY, family=FONT_MONO, transform=ax.transAxes)
                    ax.text(0.530, cur, f"{_fmt(yp.get('recall'))} / {_fmt(yp.get('precision'))}",
                            fontsize=9, color=CLAY, family=FONT_MONO, transform=ax.transAxes)
                    ax.text(0.790, cur, _fmt(d.get("rmse_point"), "{:.1f}"),
                            fontsize=9, color=SLATE, family=FONT_MONO, transform=ax.transAxes)
                    cur -= 0.025
                cur -= 0.010

            # bedtime → morning, under the checkpoint's own masked-channel policy
            no = res.get("night_onset")
            if isinstance(no, dict) and no.get("n_nights"):
                ax.text(0.060, cur, f"Night-onset excursion  ({no['n_nights']} nights, "
                        f"{int(config.NOCTURNAL_START_HOUR):02d}:00→"
                        f"{int(config.NOCTURNAL_END_HOUR):02d}:00, {wording['row']})",
                        fontsize=8.5, color=mcolor, weight="bold", style="italic",
                        transform=ax.transAxes)
                cur -= 0.024
                ax.text(0.075, cur, "regime", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                ax.text(0.300, cur, "hypo R / P", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                ax.text(0.600, cur, "hyper R / P", fontsize=8, color=DIMMED, weight="bold", transform=ax.transAxes)
                cur -= 0.022
                mm = no.get("cond") or {}
                hy = mm.get("hypo") or {}
                yp = mm.get("hyper") or {}
                ax.text(0.075, cur, wording["row"], fontsize=9, color=INK, family=FONT_BODY, transform=ax.transAxes)
                ax.text(0.300, cur, f"{_fmt(hy.get('recall'))} / {_fmt(hy.get('precision'))}",
                        fontsize=9, color=NAVY, family=FONT_MONO, transform=ax.transAxes)
                ax.text(0.600, cur, f"{_fmt(yp.get('recall'))} / {_fmt(yp.get('precision'))}",
                        fontsize=9, color=CLAY, family=FONT_MONO, transform=ax.transAxes)
                cur -= 0.025
                cur -= 0.010

        cur -= 0.008

    ax.plot([0.025, 0.975], [0.040, 0.040], color=RULE, lw=0.6, transform=ax.transAxes)
    ax.text(0.025, 0.020,
            "Offsets and recall/precision are read from metrics/sim/stats.json "
            f"(written post-training by the report scripts). {wording['footer']}; "
            "the offset is fit on a disjoint calibration split.",
            fontsize=8, color=DIMMED, transform=ax.transAxes)

    fig.savefig(OUT_DIR / "card_09_evaluation.png")
    plt.close(fig)




def card_showcase(cfg: dict, summary: dict,
                  val: dict[str, np.ndarray],
                  train: dict[str, np.ndarray],
                  tsum: dict, total_params: int, arch: dict,
                  mode: str = "best") -> None:
    """Architecture beside clinical results; ``mode`` 'best' is best-over-run, 'final' the
    last validation checkpoint."""

    assert mode in ("best", "final")
    final_step = int(val["step"][-1])

    if mode == "best":
        out_name = "card_00_showcase.png"
        header_eyebrow = "MODEL CARD"
        header_sub = ("Encoder-only transformer for Type 1 diabetes "
                      "behavioral-dynamics forecasting  —  "
                      "architecture and clinically-relevant performance.")
        chart_subtitle = "Best per horizon"
        excursion_subtitle = "Recall and precision  (best-over-run)"
    else:  # "final"
        out_name = "card_00b_showcase_final.png"
        header_eyebrow = f"MODEL CARD  ·  FINAL STEP  ·  {final_step:,}"
        header_sub = ("Encoder-only transformer for Type 1 diabetes "
                      "behavioral-dynamics forecasting  —  "
                      f"architecture and end-of-training performance (step {final_step:,}).")
        chart_subtitle = f"Final values  (step {final_step:,})"
        excursion_subtitle = f"Recall and precision  (final, step {final_step:,})"

    # the headline card names its regime: every clinical figure below reads differently when
    # masked-patch doses are withheld
    header_sub += f"  Masked-patch doses: {_policy_wording(cfg)['regime']}."

    fig, ax = _setup_card((18.7, 13.2))

    ax.text(0.025, 0.975, header_eyebrow, fontsize=10, color=CLAY,
            family=FONT_BODY, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.025, 0.955, f"T1DMAI  ·  {_human_count(total_params)} parameter model",
            fontsize=30, color=INK, family=FONT_TITLE, va="top",
            transform=ax.transAxes)
    ax.text(0.025, 0.890, header_sub,
            fontsize=12, color=SLATE, family=FONT_BODY, va="top",
            transform=ax.transAxes)
    ax.plot([0.025, 0.975], [0.860, 0.860], color=RULE, lw=1.0,
            transform=ax.transAxes)

    def _col_sc(col: str) -> np.ndarray:
        """Column accessor tolerant of an absent column (older CSV schema)."""
        return val[col] if col in val else np.full_like(val["step"], np.nan, dtype=float)

    def _stat_str(v: float, fmt: str, suffix: str = "") -> str:
        """Hero stat; a non-finite value renders '—' rather than 'nan'."""
        return "—" if not np.isfinite(v) else fmt.format(v) + suffix

    def stat_best(col: str, higher_is_better: bool):
        yv = _col_sc(col); sv = val["step"]
        mask = np.isfinite(yv); yy, ss = yv[mask], sv[mask]
        if yy.size == 0:                       # all-NaN / absent column → sentinel
            return float("nan"), 0
        idx = int(np.argmax(yy) if higher_is_better else np.argmin(yy))
        return float(yy[idx]), int(ss[idx])

    def stat_final(col: str):
        yv = _col_sc(col); sv = val["step"]
        if not np.isfinite(yv).any():
            return float("nan"), 0
        return float(yv[-1]), int(sv[-1])

    def stat_for(col: str, higher_is_better: bool):
        return stat_best(col, higher_is_better) if mode == "best" else stat_final(col)

    def tile_footer(col: str, higher_is_better: bool, fmt: str) -> str:
        v_best, step_best = stat_best(col, higher_is_better)
        if not np.isfinite(v_best):
            return "n/a"
        if mode == "best":
            return f"best @ step {step_best:,}"
        v_final, _ = stat_final(col)
        delta = v_final - v_best
        sign = "+" if delta >= 0 else ""
        return f"best {fmt.format(v_best)} @ {step_best:,}  ·  Δ {sign}{fmt.format(delta).replace('+','')}"

    mard_v,  _ = stat_for("evalfix_mard@30",      False)
    clarke_v, _ = stat_for("evalfix_clarke_A@30", True)
    hypo_v,  _ = stat_for("hypo_recall",   True)
    hyper_v, _ = stat_for("hyper_recall",  True)

    hero_tiles = [
        ("MARD @30m",     _stat_str(mard_v, "{:.2f}", "%"),   "mean absolute relative difference (@30 min)",
         tile_footer("evalfix_mard@30",     False, "{:.2f}"),  CLAY),
        ("Clarke zone A @30m", _stat_str(clarke_v, "{:.2f}", "%"), "error-grid clinically-acceptable zone (@30 min)",
         tile_footer("evalfix_clarke_A@30", True,  "{:.2f}"),  SAGE),
        ("Hypo recall",   _stat_str(hypo_v, "{:.3f}"),    f"BG < {cfg['bg_hypo_threshold']:.0f} mg/dL detection",
         tile_footer("hypo_recall",  True,  "{:.3f}"),  NAVY),
        ("Hyper recall",  _stat_str(hyper_v, "{:.3f}"),   f"BG > {cfg['bg_hyper_threshold']:.0f} mg/dL detection",
         tile_footer("hyper_recall", True,  "{:.3f}"),  PLUM),
    ]
    x0 = 0.025
    span = 0.975 - 0.025
    n = len(hero_tiles); pad = 0.013
    tile_w = (span - (n - 1) * pad) / n
    tile_y = 0.685
    tile_h = 0.155
    for i, (label, big, sub, foot, color) in enumerate(hero_tiles):
        tx = x0 + i * (tile_w + pad)
        ax.add_patch(FancyBboxPatch((tx, tile_y), tile_w, tile_h,
                                     boxstyle="round,pad=0.005,rounding_size=0.014",
                                     fc=CARD, ec=RULE, lw=0.9, transform=ax.transAxes))
        ax.add_patch(Rectangle((tx, tile_y + tile_h - 0.010), tile_w, 0.010,
                                fc=color, ec="none", transform=ax.transAxes))
        ax.text(tx + tile_w / 2, tile_y + tile_h - 0.028, label.upper(),
                fontsize=10, color=DIMMED, weight="bold", ha="center", va="top",
                transform=ax.transAxes)
        ax.text(tx + tile_w / 2, tile_y + tile_h * 0.42, big,
                fontsize=32, color=INK, family=FONT_TITLE, ha="center",
                va="center", transform=ax.transAxes)
        ax.text(tx + tile_w / 2, tile_y + 0.030, sub,
                fontsize=9.5, color=SLATE, ha="center", va="center",
                transform=ax.transAxes)
        ax.text(tx + tile_w / 2, tile_y + 0.012, foot,
                fontsize=8.5, color=DIMMED, ha="center", va="center",
                family=FONT_MONO, transform=ax.transAxes)

    ax.plot([0.025, 0.975], [0.655, 0.655], color=RULE, lw=0.6,
            transform=ax.transAxes)

    ax.text(0.030, 0.630, "ARCHITECTURE", fontsize=10, color=CLAY,
            weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.030, 0.610, "Forward pass",
            fontsize=14, color=INK, family=FONT_TITLE, va="top",
            transform=ax.transAxes)

    def block(x, y, w, h, title, sub="", fc=CARD, ec=NAVY, title_size=8.5,
              sub_size=7.5, bar_color=None):
        if bar_color:
            ax.add_patch(Rectangle((x, y + h - 0.005), w, 0.005,
                                    fc=bar_color, ec="none", transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.004,rounding_size=0.008",
                                     fc=fc, ec=ec, lw=0.7, transform=ax.transAxes))
        if sub:
            ax.text(x + w / 2, y + h * 0.62, title, fontsize=title_size, color=INK,
                    weight="bold", ha="center", family=FONT_BODY,
                    transform=ax.transAxes)
            ax.text(x + w / 2, y + h * 0.25, sub, fontsize=sub_size, color=DIMMED,
                    ha="center", family=FONT_MONO, transform=ax.transAxes)
        else:
            ax.text(x + w / 2, y + h * 0.50, title, fontsize=title_size, color=INK,
                    weight="bold", ha="center", family=FONT_BODY,
                    transform=ax.transAxes)

    def arrow(p0, p1, color=MUTED, lw=0.8):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                                      color=color, lw=lw, transform=ax.transAxes))

    # Architecture diagram laid out in x ∈ [0.045, 0.425], y ∈ [0.07, 0.575]
    cx = 0.235  # center x of the column

    block(0.135, 0.555, 0.20, 0.022, f"Input  (B, T, {config.PATCH_DIM})", "",
          ec=GOLD, bar_color=GOLD, fc=GOLD_T, title_size=8.5)
    arrow((cx, 0.555), (cx, 0.547))

    block(0.150, 0.510, 0.17, 0.030, "Patch embedding",
          f"Linear  {config.PATCH_DIM} → {cfg['d_model']}",
          ec=NAVY, bar_color=NAVY, fc=NAVY_T)
    arrow((cx, 0.510), (cx, 0.500))

    bx, by, bw, bh = 0.055, 0.215, 0.360, 0.280
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                 boxstyle="round,pad=0.006,rounding_size=0.012",
                                 fc=SOFT_RULE, ec=RULE, lw=0.8, transform=ax.transAxes))
    ax.text(bx + 0.012, by + bh - 0.022,
            f"TransformerBlock  × {cfg['n_layers']}",
            fontsize=10, color=INK, family=FONT_TITLE, transform=ax.transAxes)
    ax.text(bx + 0.012, by + bh - 0.040,
            "pre-norm  ·  residual on every sub-layer",
            fontsize=7.5, color=DIMMED, transform=ax.transAxes)

    block(0.090, 0.418, 0.290, 0.018, "RMSNorm", "", ec=RULE, fc=CARD, title_size=8)
    arrow((cx, 0.418), (cx, 0.412))
    block(0.090, 0.378, 0.290, 0.028, "Temporal self-attention",
          f"{cfg['n_heads']} heads × dim {cfg['d_model']//cfg['n_heads']}  ·  RoPE",
          ec=NAVY, bar_color=NAVY, fc=NAVY_T)
    arrow((cx, 0.378), (cx, 0.300))

    block(0.090, 0.282, 0.290, 0.018, "RMSNorm", "", ec=RULE, fc=CARD, title_size=8)
    arrow((cx, 0.282), (cx, 0.276))
    block(0.090, 0.246, 0.290, 0.024, f"SwiGLU FFN  ({cfg['ffn_dim']})",
          "x · SiLU(gate(x)) → down",
          ec=SAGE, bar_color=SAGE, fc=SAGE_T)

    arrow((cx, 0.244), (cx, 0.200))

    block(0.155, 0.170, 0.160, 0.024, "Final RMSNorm", "",
          ec=NAVY, bar_color=NAVY, fc=NAVY_T)

    # to the output heads: the risk-space BG quantile fan, plus the diagnostic time-of-day
    # probe when the checkpoint carries it
    _hh = arch['bg_head_hidden']
    _d = cfg['d_model']
    _ps = cfg['patch_size']
    _nq = arch['n_quantiles']
    oh_heads = [
        (TEAL, TEAL_T, "BG quantile head",
         f"Linear({_d}→{_hh})→SiLU→Linear→SiLU→Linear",
         f"{_nq}-τ risk fan → mg/dL"),
    ]
    if arch.get('time_probe'):
        oh_heads.append(
            (PLUM, PLUM_T, "Time-of-day probe",
             f"Linear({_d}→{arch['time_probe_hidden']})→SiLU→Linear",
             f"{arch['time_probe_bins']} hour bins · diagnostic"),
        )

    n_oh = len(oh_heads)
    oh_gap = 0.012
    oh_x0, oh_span = 0.050, 0.380
    oh_w = (oh_span - (n_oh - 1) * oh_gap) / n_oh
    oh_h = 0.090
    oh_y = 0.050
    oh_centers = [oh_x0 + i * (oh_w + oh_gap) + oh_w / 2 for i in range(n_oh)]

    arrow((cx, 0.170), (cx, 0.151), color=MUTED)
    ax.plot([oh_centers[0], oh_centers[-1]], [0.151, 0.151], color=MUTED, lw=0.8,
            transform=ax.transAxes)
    for ocx in oh_centers:
        arrow((ocx, 0.151), (ocx, 0.140), color=MUTED)

    oh_title_fs = 8.5 if n_oh <= 2 else 7.3
    oh_line_fs = 7.0 if n_oh <= 2 else 6.2
    for i, (color, color_t, title, line1, line2) in enumerate(oh_heads):
        oh_x = oh_x0 + i * (oh_w + oh_gap)
        ax.add_patch(Rectangle((oh_x, oh_y + oh_h - 0.005), oh_w, 0.005,
                                fc=color, ec="none", transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((oh_x, oh_y), oh_w, oh_h,
                                     boxstyle="round,pad=0.004,rounding_size=0.010",
                                     fc=color_t, ec=color, lw=0.8, transform=ax.transAxes))
        ax.text(oh_x + oh_w / 2, oh_y + oh_h - 0.016, title,
                fontsize=oh_title_fs, color=INK, weight="bold", ha="center",
                transform=ax.transAxes)
        ax.text(oh_x + oh_w / 2, oh_y + oh_h * 0.40, line1,
                fontsize=oh_line_fs, color=SLATE, ha="center", family=FONT_MONO,
                transform=ax.transAxes)
        ax.text(oh_x + oh_w / 2, oh_y + 0.012, line2,
                fontsize=oh_line_fs, color=DIMMED, ha="center", style="italic",
                transform=ax.transAxes)

    ax.plot([0.443, 0.443], [0.060, 0.635], color=RULE, lw=0.6,
            transform=ax.transAxes)


    ax.text(0.465, 0.630, "MULTI-HORIZON BG RMSE", fontsize=10, color=CLAY,
            weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.465, 0.610, chart_subtitle,
            fontsize=14, color=INK, family=FONT_TITLE, va="top",
            transform=ax.transAxes)

    chart_ax = fig.add_axes([0.480, 0.380, 0.495, 0.190])
    chart_ax.set_facecolor(PAPER)
    horizons = [("30 min",  "bg_rmse_30"),
                ("60 min",  "bg_rmse_60"),
                ("120 min", "bg_rmse_120"),
                ("180 min", "bg_rmse_180"),
                ("360 min", "bg_rmse_360"),
                ("480 min", "bg_rmse_480")]
    labels = [h for h, _ in horizons]
    values = [stat_for(col, False)[0] for _, col in horizons]
    cmap = plt.colormaps["viridis"]
    bar_colors = [cmap(0.10 + 0.75 * i / (len(horizons) - 1))
                  for i in range(len(horizons))]
    x_pos = np.arange(len(values))
    bars = chart_ax.bar(x_pos, values, color=bar_colors, edgecolor="none", width=0.66)
    chart_ax.set_xticks(x_pos)
    chart_ax.set_xticklabels([])  # hide auto labels; horizon labels added as annotations below
    chart_ax.set_ylim(0, max(values) * 1.30)
    chart_ax.set_ylabel("mg/dL", color=SLATE, fontsize=10)
    chart_ax.tick_params(axis="x", bottom=False)
    chart_ax.tick_params(axis="y", labelsize=9, colors=SLATE, left=True)
    chart_ax.spines["left"].set_visible(True);   chart_ax.spines["left"].set_color(RULE)
    chart_ax.spines["bottom"].set_visible(True); chart_ax.spines["bottom"].set_color(RULE)
    chart_ax.spines["top"].set_visible(False);   chart_ax.spines["right"].set_visible(False)
    chart_ax.yaxis.grid(True, linestyle=":", alpha=0.6, color=RULE)
    chart_ax.set_axisbelow(True)
    for bar, v, lab in zip(bars, values, labels):
        chart_ax.text(bar.get_x() + bar.get_width() / 2,
                       v + max(values) * 0.025,
                       f"{v:.1f}",
                       ha="center", va="bottom", fontsize=10, color=INK,
                       family=FONT_MONO)
        chart_ax.text(bar.get_x() + bar.get_width() / 2,
                       max(values) * 0.035,
                       lab,
                       ha="center", va="bottom", fontsize=9, color="white",
                       weight="bold")

    ax.text(0.465, 0.298, "EXCURSION DETECTION", fontsize=10, color=CLAY,
            weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.465, 0.278, excursion_subtitle,
            fontsize=14, color=INK, family=FONT_TITLE, va="top",
            transform=ax.transAxes)

    cell_x0 = 0.465
    cell_x1 = 0.975
    cell_gap = 0.014
    cell_w = (cell_x1 - cell_x0 - cell_gap) / 2
    cell_h = 0.095
    top_row_y = 0.155

    cells = [
        (0, 0, "hypo_recall",    "Hypo recall",    f"BG < {cfg['bg_hypo_threshold']:.0f} mg/dL",   NAVY),
        (0, 1, "hypo_precision", "Hypo precision", "no false alarms", NAVY),
        (1, 0, "hyper_recall",   "Hyper recall",   f"BG > {cfg['bg_hyper_threshold']:.0f} mg/dL",  CLAY),
        (1, 1, "hyper_precision","Hyper precision","no false alarms", CLAY),
    ]
    for row, col, col_name, label, sub, color in cells:
        cx_cell = cell_x0 + col * (cell_w + cell_gap)
        cy_cell = top_row_y - row * (cell_h + cell_gap)
        val_num, val_step = stat_for(col_name, True)
        if mode == "best":
            footer_text = f"step {val_step:,}"
        else:
            v_best, step_best = stat_best(col_name, True)
            delta = val_num - v_best
            sign = "+" if delta >= 0 else "−"
            footer_text = f"best {v_best:.3f} @ {step_best:,}   Δ {sign}{abs(delta):.3f}"
        ax.add_patch(FancyBboxPatch((cx_cell, cy_cell), cell_w, cell_h,
                                     boxstyle="round,pad=0.004,rounding_size=0.010",
                                     fc=CARD, ec=RULE, lw=0.8, transform=ax.transAxes))
        ax.add_patch(Rectangle((cx_cell, cy_cell), 0.006, cell_h,
                                fc=color, ec="none", transform=ax.transAxes))
        ax.text(cx_cell + 0.022, cy_cell + cell_h - 0.018, label.upper(),
                fontsize=10, color=color, weight="bold", va="top",
                transform=ax.transAxes)
        ax.text(cx_cell + 0.022, cy_cell + cell_h - 0.040, sub,
                fontsize=8.5, color=DIMMED, va="top", transform=ax.transAxes)
        ax.text(cx_cell + 0.022, cy_cell + 0.013,
                footer_text,
                fontsize=8, color=MUTED, family=FONT_MONO, va="bottom",
                transform=ax.transAxes)
        ax.text(cx_cell + cell_w - 0.020, cy_cell + cell_h * 0.45,
                f"{val_num:.3f}",
                fontsize=30, color=INK, family=FONT_TITLE, ha="right",
                va="center", transform=ax.transAxes)

    ax.plot([0.025, 0.975], [0.038, 0.038], color=RULE, lw=0.6,
            transform=ax.transAxes)
    elapsed_h = tsum["progress"]["elapsed_hours"]
    sps = tsum["progress"]["steps_per_second"]
    footer_parts = [
        f"{total_params:,} parameters ({_human_count(total_params)})",
        f"{cfg['total_steps']:,} steps × batch {cfg['batch_size']} "
        f"= {_human_count(cfg['total_steps']*cfg['batch_size'])} samples",
        f"{elapsed_h:.2f} h wallclock · {sps:.2f} steps/sec",
        f"seed {cfg['master_seed']}",
        "on-the-fly T1DMSIM simulator",
    ]
    ax.text(0.025, 0.018, "    ·    ".join(footer_parts),
            fontsize=9, color=DIMMED, transform=ax.transAxes)

    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)




def main() -> None:
    import argparse
    import torch
    ap = argparse.ArgumentParser(description="Render the T1DMAI model-card figures.")
    ap.add_argument("--checkpoint", default=str(CKPT_PATH),
                    help="checkpoint .pt (default: checkpoints/t1dmai_best.pt)")
    ap.add_argument("--logs", default=str(LOG_DIR),
                    help="run log dir holding training_summary.json / training_log.csv / "
                         "validation_log.csv (default: logs/). Point this at a model's OWN "
                         "log dir — e.g. models/logs_<name> — when its checkpoint lives "
                         "outside checkpoints/ and the repo-root logs/ belongs to another run.")
    args = ap.parse_args()
    ckpt_path = Path(args.checkpoint)
    log_dir = Path(args.logs)
    _set_style()
    OUT_DIR.mkdir(exist_ok=True)

    # The checkpoint's embedded ``training_config`` is the resolved CLI > config.py snapshot
    # that produced these weights — authoritative, unlike the mutable config.py or a
    # logs/resolved_config.json that may belong to a newer run. Flags it does not carry come
    # from the weight shapes, via _derive_arch.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    cfg = ckpt["training_config"]

    tsum = json.loads((log_dir / "training_summary.json").read_text())
    train = _read_csv(log_dir / "training_log.csv")
    val = _read_csv(log_dir / "validation_log.csv")
    if "step" not in val or len(val["step"]) == 0:
        raise SystemExit(
            f"{log_dir / 'validation_log.csv'} has no validation rows (header only) — "
            f"the model card needs a run that reached at least one validation "
            f"(VALIDATION_INTERVAL steps). Point --logs at that run's log dir, e.g. "
            f"--logs models/logs_<name> --checkpoint models/<name>.pt."
        )
    summary = json.loads((OUT_DIR / "summary.json").read_text())
    reports = _load_report_stats()  # empty until the metrics/ reports exist

    groups, total_params = _param_breakdown(sd)
    arch = _derive_arch(sd, cfg)

    print(f"Rendering model card for the {_human_count(total_params)} run...")
    card_showcase(cfg, summary, val, train, tsum, total_params, arch, mode="best")
    card_showcase(cfg, summary, val, train, tsum, total_params, arch, mode="final")
    card_overview(cfg, summary, total_params, arch)
    card_architecture(cfg, total_params, arch)
    card_param_breakdown(groups, total_params)
    card_io_schema(cfg, arch)
    card_training_recipe(cfg)
    card_loss_design(cfg, train)
    card_compute_budget(train, tsum, cfg)
    card_metrics_card(cfg, val)
    n_cards = 10
    if reports:
        card_evaluation(cfg, reports)
        n_cards += 1
    print(f"  → wrote {n_cards} card_*.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
