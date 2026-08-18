"""Render the figures the README embeds, into screenshots/.

Three figures, each in a light and a dark variant so the README can serve the
matching one through a <picture> element:

    architecture{,-dark}.png   end-to-end pipeline: signals -> encoder -> quantile
                               fan -> mg/dL, and the three things that consume it
    risk-space{,-dark}.png     the Kovatchev warp and the loss asymmetry it creates
    forecast{,-dark}.png       three forward passes of a trained checkpoint over one
                               context window — backcast, infill and forecast — each
                               cropped to 24 h around its masked span

Usage:
    python make_readme_figures.py --skip-forecast       # diagrams only, no checkpoint
    python make_readme_figures.py --checkpoint PATH     # all three
    python make_readme_figures.py --checkpoint PATH --seed N

The masked-BG figure has no default checkpoint: the capacity ladder's weights are
per-run artifacts under the gitignored ``models/<capacity>/checkpoints/``.
Name one or pass ``--skip-forecast``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import config
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
from utils import kovatchev_f_np as _f

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "screenshots"

FONT_TITLE = "DejaVu Serif"
FONT_BODY = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"


# --------------------------------------------------------------------- palette

class Theme:
    """One coherent set of surface / ink / accent colours."""

    def __init__(self, name: str, paper: str, card: str, ink: str, slate: str,
                 muted: str, rule: str, tint: float) -> None:
        self.name = name
        self.paper, self.card = paper, card
        self.ink, self.slate, self.muted, self.rule = ink, slate, muted, rule
        self._tint = tint          # blend weight toward the surface for fills

    NAVY = "#2a4a6e"
    CLAY = "#a85a3a"
    TEAL = "#3e7d8a"
    GOLD = "#a8843a"
    SAGE = "#5a7d4a"
    PLUM = "#74416e"

    def fill(self, accent: str) -> str:
        """A soft fill for ``accent`` on this theme's surface."""
        return _blend(accent, self.paper, self._tint)

    def edge(self, accent: str) -> str:
        return _blend(accent, self.paper, 0.30 if self.name == "light" else 0.20)


def _blend(a: str, b: str, w: float) -> str:
    """Mix colour ``a`` toward ``b`` by weight ``w`` (0 = a, 1 = b)."""
    ca, cb = mcolors.to_rgb(a), mcolors.to_rgb(b)
    r, g, bl = (x * (1 - w) + y * w for x, y in zip(ca, cb))
    return mcolors.to_hex((r, g, bl))


LIGHT = Theme("light", paper="#fbfaf6", card="#ffffff", ink="#1a1f2e",
              slate="#3d4659", muted="#8b93a3", rule="#d8dae1", tint=0.88)
DARK = Theme("dark", paper="#14161c", card="#1b1e26", ink="#e9ebf2",
             slate="#b3bacb", muted="#7d859a", rule="#333a49", tint=0.80)


def _style(t: Theme) -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.facecolor": t.paper,
        "figure.facecolor": t.paper,
        "axes.facecolor": t.paper,
        "font.size": 10.5,
        "font.family": FONT_BODY,
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
        "text.color": t.ink,
        "axes.edgecolor": t.rule,
        "axes.labelcolor": t.slate,
        "axes.titlecolor": t.ink,
        "xtick.color": t.slate,
        "ytick.color": t.slate,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.7,
    })


# ------------------------------------------------------------------ primitives

def _canvas(figsize: tuple[float, float], t: Theme):
    fig, ax = plt.subplots(figsize=figsize, facecolor=t.paper)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(t.paper)
    return fig, ax


def _box(ax, t: Theme, x, y, w, h, title, sub=None, accent=None, mono_sub=True,
         title_size=10.5, sub_size=8.4, lw=1.1, zorder=2):
    """A rounded box with a bold title and an optional second line."""
    accent = accent or t.slate
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=t.fill(accent), edgecolor=t.edge(accent), linewidth=lw,
        zorder=zorder, mutation_aspect=1.0))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + h * 0.17, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=t.ink, zorder=zorder + 1)
        ax.text(x + w / 2, cy - h * 0.20, sub, ha="center", va="center",
                fontsize=sub_size, color=t.slate, zorder=zorder + 1,
                family=FONT_MONO if mono_sub else FONT_BODY)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                fontsize=title_size, weight="bold", color=t.ink, zorder=zorder + 1)


def _arrow(ax, t: Theme, xy_from, xy_to, style="-|>", lw=1.2, ls="-", color=None):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=11,
        linewidth=lw, linestyle=ls, color=color or t.muted,
        shrinkA=0, shrinkB=0, zorder=1))


def _down(ax, t, x, y0, y1, **kw):
    _arrow(ax, t, (x, y0), (x, y1), **kw)


# ---------------------------------------------------------------- architecture

def draw_architecture(t: Theme, path: Path) -> None:
    _style(t)
    fig, ax = _canvas((11.0, 12.4), t)

    L, R = 0.045, 0.955
    W = R - L

    # --- source ------------------------------------------------------------
    _box(ax, t, L, 0.940, W, 0.052, "T1DMSIM behaviour simulator",
         "seed → patient → events → 5-min stream", accent=t.SAGE, mono_sub=False)
    ax.text(0.5, 0.930, "training corpus", ha="center", va="top",
            fontsize=8.0, color=t.muted, style="italic")
    _down(ax, t, 0.5, 0.922, 0.882)

    # --- observed signal ---------------------------------------------------
    _box(ax, t, L, 0.828, W, 0.052, "Four input channels, one 5-minute grid",
         "CGM  mg/dL · carbohydrate  g / step · insulin  U / step"
         " · exercise  g / step (carbohydrate-equivalent)",
         accent=t.NAVY)
    ax.text(0.5, 0.820, "insulin sensitivity and hepatic output are simulator latents — deliberately not inputs",
            ha="center", va="top", fontsize=8.0, color=t.muted, style="italic")
    _down(ax, t, 0.5, 0.812, 0.792)

    # --- transforms --------------------------------------------------------
    tw = 0.440
    _box(ax, t, L, 0.736, tw, 0.052, "CGM → Kovatchev f → z-score",
         "the model reads and writes glucose in risk space", accent=t.CLAY,
         mono_sub=False, title_size=10.2)
    _box(ax, t, R - tw, 0.736, tw, 0.052, "carb, insulin, exercise → log1p → z-score",
         "keeps rare event spikes off the scale", accent=t.CLAY,
         mono_sub=False, title_size=10.2)
    _arrow(ax, t, (L + tw / 2, 0.732), (0.5, 0.712))
    _arrow(ax, t, (R - tw / 2, 0.732), (0.5, 0.712))
    _down(ax, t, 0.5, 0.712, 0.700)

    # --- patching ----------------------------------------------------------
    _box(ax, t, L, 0.644, W, 0.052, "Patch embedding",
         f"{config.PATCH_SIZE} steps × {config.N_INPUT_FEATURES} features "
         f"({config.PATCH_SIZE * 5} min)  →  one token  →  Linear → D_MODEL",
         accent=t.NAVY)
    ax.text(0.5, 0.636,
            f"window {config.MIN_CONTEXT_PATCHES + config.PREDICTION_PATCHES}–"
            f"{config.MAX_SEQ_LEN} patches   │   the fifth feature is bg_masked, "
            "a per-patch bit: 1 where glucose is withheld, and the head predicts it",
            ha="center", va="top", fontsize=8.0, color=t.muted, style="italic")
    _down(ax, t, 0.5, 0.626, 0.608)

    # --- encoder stack -----------------------------------------------------
    gy, gh = 0.398, 0.210
    ax.add_patch(FancyBboxPatch(
        (L, gy), W, gh, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=t.fill(t.slate), edgecolor=t.edge(t.slate), linewidth=1.1, zorder=1))
    ax.text(L + 0.022, gy + gh - 0.020, "Transformer block  × N_LAYERS",
            fontsize=12, family=FONT_TITLE, color=t.ink, va="center")
    ax.text(L + 0.022, gy + gh - 0.044, "pre-norm · residual on both sub-layers · fp32 throughout",
            fontsize=8.2, color=t.muted, va="center", style="italic")

    bx, bw = L + 0.028, 0.470
    _box(ax, t, bx, gy + 0.146, bw, 0.026, "RMSNorm", accent=t.slate,
         title_size=9.2, lw=0.9, zorder=3)
    _down(ax, t, bx + bw / 2, gy + 0.144, gy + 0.136)
    _box(ax, t, bx, gy + 0.094, bw, 0.042, "Temporal self-attention",
         "RoPE · QK-norm", accent=t.NAVY,
         title_size=10, sub_size=8.0, mono_sub=False, zorder=3)
    _down(ax, t, bx + bw / 2, gy + 0.092, gy + 0.080)
    _box(ax, t, bx, gy + 0.054, bw, 0.026, "RMSNorm", accent=t.slate,
         title_size=9.2, lw=0.9, zorder=3)
    _down(ax, t, bx + bw / 2, gy + 0.052, gy + 0.044)
    _box(ax, t, bx, gy + 0.018, bw, 0.026, "SwiGLU FFN", accent=t.SAGE,
         title_size=9.2, lw=0.9, zorder=3)

    # attention-mask inset
    mx, my, ms = 0.712, gy + 0.062, 0.118
    _mask_inset(ax, t, mx, my, ms)

    _down(ax, t, 0.5, gy, 0.372)

    # --- final norm --------------------------------------------------------
    _box(ax, t, 0.290, 0.322, 0.420, 0.042, "Final RMSNorm",
         "gathered by mask_idx — masked patches only, from here on",
         accent=t.NAVY, sub_size=8.0, mono_sub=False)
    _arrow(ax, t, (0.5, 0.320), (0.5, 0.310))
    ax.plot([0.235, 0.765], [0.310, 0.310], color=t.muted, lw=1.2, zorder=1)
    _down(ax, t, 0.235, 0.310, 0.288)
    _down(ax, t, 0.765, 0.310, 0.288)

    # --- heads -------------------------------------------------------------
    _box(ax, t, L, 0.230, 0.435, 0.056, "Blood-glucose quantile head",
         "3-layer MLP → K low-frequency coefficients per masked patch",
         accent=t.TEAL, sub_size=8.2, mono_sub=False)
    _box(ax, t, R - 0.435, 0.230, 0.435, 0.056, "Time-of-day probe",
         "12 circular hour bins per masked patch · co-trains the trunk\n"
         "reads the clock off the trajectory, with no clock input",
         accent=t.PLUM, sub_size=8.2, mono_sub=False)
    _down(ax, t, 0.2325, 0.228, 0.196)

    # --- assembly + decode --------------------------------------------------
    _box(ax, t, L, 0.130, 0.640, 0.064, "Quantile assembly",
         "per-slot anchor f(anchor_bg), one-sided and left-preferring "
         " +  per-span DCT median\n"
         " +  softplus cumsum  →  7 ascending quantiles  "
         "τ = .05 .10 .25 .50 .75 .90 .95",
         accent=t.TEAL, sub_size=8.2, mono_sub=False)
    _arrow(ax, t, (L + 0.640, 0.162), (0.735, 0.162))
    _box(ax, t, 0.740, 0.130, R - 0.740, 0.064, "f⁻¹  →  mg/dL",
         "median line\n+ six band edges", accent=t.CLAY, sub_size=8.2, mono_sub=False)

    # graph-cut annotation
    ax.plot([L - 0.010, R + 0.010], [0.219, 0.219], color=t.GOLD, lw=1.0,
            ls=(0, (5, 4)), zorder=4)
    ax.text(R + 0.006, 0.213,
            "the exported graph stops here — the assembly below it runs off the descriptor",
            ha="right", va="top", fontsize=8.2, color=t.GOLD, style="italic")

    _down(ax, t, 0.5, 0.128, 0.112)
    ax.plot([0.165, 0.835], [0.112, 0.112], color=t.muted, lw=1.2, zorder=1)
    for cx in (0.165, 0.5, 0.835):
        _down(ax, t, cx, 0.112, 0.094)

    # --- consumers ---------------------------------------------------------
    cw = 0.290
    _box(ax, t, L, 0.036, cw, 0.056, "Validation · metrics",
         "fresh simulator patients", accent=t.slate, sub_size=8.2, mono_sub=False)
    _box(ax, t, 0.5 - cw / 2, 0.036, cw, 0.056, "Interactive GUI",
         "what-if plans · free-form masking", accent=t.slate, sub_size=8.2, mono_sub=False)
    _box(ax, t, R - cw, 0.036, cw, 0.056, "Exporters",
         ".pte / .tflite + descriptor", accent=t.slate, sub_size=8.2, mono_sub=False)
    ax.text(R - cw / 2, 0.028, "loaded on device by T1DMDROID", ha="center", va="top",
            fontsize=8.0, color=t.muted, style="italic")

    fig.savefig(path, bbox_inches="tight", pad_inches=0.18, facecolor=t.paper)
    plt.close(fig)
    print(f"wrote {path}")


def _mask_inset(ax, t: Theme, x: float, y: float, s: float) -> None:
    """A 2x2 schematic of the attention mask.

    The axes are VISIBLE and MASKED, not context and prediction: a masked span
    may end at the last patch (forecast), start at patch 0 (backcast) or sit
    between visible patches (infill), so neither axis is a position.
    """
    half = s / 2
    allow, block = t.fill(t.NAVY), t.paper
    cells = [((0, 1), allow), ((1, 1), block), ((0, 0), allow), ((1, 0), allow)]
    for (cx, cy), fc in cells:
        ax.add_patch(Rectangle((x + cx * half, y + cy * half), half, half,
                               facecolor=fc, edgecolor=t.edge(t.NAVY),
                               linewidth=0.9, zorder=3))
    ax.text(x + half / 2, y + half * 1.5, "vis\n↔vis", ha="center", va="center",
            fontsize=7.2, color=t.slate, zorder=4, linespacing=1.15)
    ax.text(x + half * 1.5, y + half * 1.5, "blocked", ha="center", va="center",
            fontsize=7.2, color=t.muted, zorder=4)
    ax.text(x + half / 2, y + half / 2, "masked\n→vis", ha="center", va="center",
            fontsize=7.2, color=t.slate, zorder=4, linespacing=1.15)
    ax.text(x + half * 1.5, y + half / 2, "masked\n↔masked", ha="center", va="center",
            fontsize=7.2, color=t.slate, zorder=4, linespacing=1.15)
    ax.text(x + half, y + s + 0.012, "attention mask", ha="center", va="bottom",
            fontsize=8.6, weight="bold", color=t.ink, zorder=4)
    ax.text(x + half, y - 0.012,
            "evidence never reads a prediction\n"
            f"up to {config.MASK_MAX_SPANS} masked spans of "
            f"{config.MASK_SPAN_LENGTHS[0]}–{config.MASK_SPAN_LENGTHS[-1]} patches, "
            "never abutting,\nplaced anywhere in the window: forecast at the\n"
            "last patch, backcast at patch 0, infill between",
            ha="center", va="top", fontsize=7.2, color=t.muted, style="italic",
            zorder=4, linespacing=1.35)


# ------------------------------------------------------------------ risk space

# The plotted span is the physical clamp, so the panels cover every BG the
# simulator and the model can produce. The transform's anchors sit inside it:
# f is solved so f(40) = -sqrt(10) and f(400) = +sqrt(10).
BG_LO, BG_HI = BG_CLAMP_MIN, BG_CLAMP_MAX
BG_ANCHOR_LO = 40.0


def _num(x: float, nd: int = 2) -> str:
    """``x`` with the typographic minus the rest of the labels use."""
    return f"{x:.{nd}f}".replace("-", "−")


def draw_risk_space(t: Theme, path: Path) -> None:
    _style(t)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), facecolor=t.paper,
                             gridspec_kw={"wspace": 0.24})

    g = np.linspace(BG_LO, BG_HI, 800)
    r = _f(g)

    # --- panel 1: the warp --------------------------------------------------
    a = axes[0]
    a.set_facecolor(t.paper)
    a.axvspan(70.0, 180.0, color=t.fill(t.SAGE), zorder=0)
    a.plot(g, r, color=t.CLAY, lw=2.2, zorder=3)
    a.axhline(0.0, color=t.rule, lw=0.9, zorder=1)

    for bg in (70.0, 180.0):
        a.plot([BG_LO, bg], [_f(bg), _f(bg)], color=t.muted, lw=0.8,
               ls=(0, (3, 3)), zorder=2)
    a.plot([128.0], [0.0], "o", color=t.CLAY, ms=5, zorder=4)
    a.annotate("zero risk ≈ 128 mg/dL", (128.0, 0.0), textcoords="offset points",
               xytext=(12, -18), fontsize=8.6, color=t.slate)
    a.plot([BG_ANCHOR_LO], [_f(BG_ANCHOR_LO)], "o", color=t.CLAY, ms=5, zorder=4)
    a.annotate("f(40) = −√10", (BG_ANCHOR_LO, _f(BG_ANCHOR_LO)),
               textcoords="offset points", xytext=(10, -16), fontsize=8.6,
               color=t.slate)
    a.annotate(f"f({BG_LO:.0f}) = {_num(float(_f(BG_LO)))}", (BG_LO, _f(BG_LO)),
               textcoords="offset points", xytext=(10, 3), fontsize=8.6,
               color=t.slate)
    a.annotate("f(400) = +√10", (BG_HI, _f(BG_HI)), textcoords="offset points",
               xytext=(-8, 6), ha="right", fontsize=8.6, color=t.slate)
    a.text(125.0, -2.55, "target\nrange", fontsize=8.4, color=t.slate,
           ha="center", va="center", style="italic", linespacing=1.3)

    a.set_xlabel("blood glucose (mg/dL)")
    a.set_ylabel("risk space  f(g)")
    a.set_xlim(BG_LO, BG_HI)
    a.set_ylim(float(_f(BG_LO)) - 0.4, float(_f(BG_HI)) + 0.4)
    a.set_title("The transform the model forecasts in", fontsize=11.5, loc="left",
                color=t.ink, pad=10)
    a.spines["left"].set_color(t.rule)
    a.spines["bottom"].set_color(t.rule)

    # --- panel 2: what it costs --------------------------------------------
    b = axes[1]
    b.set_facecolor(t.paper)
    err = 20.0
    gg = np.linspace(BG_LO, BG_HI - err, 600)
    cost = np.abs(_f(gg + err) - _f(gg))
    b.fill_between(gg, 0, cost, color=t.fill(t.CLAY), zorder=1)
    b.plot(gg, cost, color=t.CLAY, lw=2.2, zorder=3)

    for bg in (60.0, 120.0, 300.0):
        c = float(np.abs(_f(bg + err) - _f(bg)))
        b.plot([bg, bg], [0, c], color=t.muted, lw=0.8, ls=(0, (3, 3)), zorder=2)
        b.plot([bg], [c], "o", color=t.CLAY, ms=4.5, zorder=4)
        b.annotate(f"{c:.2f} at {bg:.0f}", (bg, c), textcoords="offset points",
                   xytext=(8, 6), fontsize=8.6, color=t.slate)

    b.set_xlabel("blood glucose (mg/dL)")
    b.set_ylabel("| f(g + 20) − f(g) |")
    b.set_xlim(BG_LO, BG_HI)
    b.set_ylim(0, None)
    b.set_title("What the same 20 mg/dL error costs", fontsize=11.5, loc="left",
                color=t.ink, pad=10)
    b.spines["left"].set_color(t.rule)
    b.spines["bottom"].set_color(t.rule)

    fig.text(0.5, -0.030,
             "A 20 mg/dL miss at 60 mg/dL costs roughly four times the risk-space error of "
             "the same miss at 300. Optimising here, rather than in mg/dL, is what makes "
             "the model treat lows as the expensive mistake.",
             ha="center", va="top", fontsize=9.2, color=t.slate)

    fig.savefig(path, bbox_inches="tight", pad_inches=0.22, facecolor=t.paper)
    plt.close(fig)
    print(f"wrote {path}")


# ------------------------------------------------------------------- masked BG

def _load_model(checkpoint: str, device):
    import torch
    from model import T1DMAI
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    ema = ck.get("model_ema_state_dict")
    merged = {k: ema.get(k, v) for k, v in sd.items()} if ema else dict(sd)
    m = T1DMAI().to(device)
    m.load_state_dict(merged, strict=True)
    m.eval()
    return m, ck


def _sim_window(seed: int, stats, hours: float):
    """One simulator patient: normalized features and the raw channels behind them.

    ``exercise`` is T1DMSIM's carbohydrate-equivalent glucose-disposal curve in
    g/step, on the same scale as ``carb`` — never an intensity.
    """
    import numpy as np
    from T1DMSIM.simulator import T1DMSimulator, BG_CLAMP_MIN, BG_CLAMP_MAX
    from data import simulate_discard_warmup
    from normalization import normalize

    raw = simulate_discard_warmup(T1DMSimulator(seed=seed), hours)
    bg = np.clip(raw["bg_observed"], BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.maximum(raw["total_carb"], 0.0).astype(np.float32)
    ins = np.maximum(raw["total_insulin"], 0.0).astype(np.float32)
    exr = np.maximum(raw["total_exercise"], 0.0).astype(np.float32)
    from data import BG_MASKED_FEAT

    # Four normalized signal columns, then the bg_masked announcement bit above
    # them. The bit is not a signal — no statistics, never through normalize — and
    # every step here is an observed reading, so its column stays 0.0; the masked
    # set is written into the patches downstream, by the builder that knows it.
    cols = [bg, carb, ins, exr]
    assert len(cols) == BG_MASKED_FEAT < config.N_INPUT_FEATURES, (
        f"{len(cols)} raw signal columns against BG_MASKED_FEAT={BG_MASKED_FEAT}, "
        f"N_INPUT_FEATURES={config.N_INPUT_FEATURES}")
    feats = np.zeros((len(bg), config.N_INPUT_FEATURES), dtype=np.float32)
    feats[:, :BG_MASKED_FEAT] = normalize(np.stack(cols, axis=-1), stats)
    return feats, bg, carb, ins, exr, raw["hour_of_day"].astype(np.float32)


CROP_HOURS = 24.0          # the slice each panel shows out of the whole window


def _anchor_step(span_start: int, span_len: int, patch_size: int) -> tuple[int, bool]:
    """The window step index this span's fan is anchored on, and which end it joins.

    The anchor rule is one-sided and LEFT-PREFERRING: every slot of a span takes
    the last step of its left neighbour, and the first step of its right
    neighbour only when the span opens the window. So a fan drawn for a span at
    patch 0 leaves the observed line at its RIGHT edge, and every other fan at
    its left. The head decodes a delta from that step, so the fan has zero width
    there and the join is structural, not cosmetic.
    """
    if span_start == 0:
        return (span_start + span_len) * patch_size, False      # join on the right
    return span_start * patch_size - 1, True                    # join on the left


def draw_masked_bg(t: Theme, path: Path, checkpoint: str, seed: int) -> None:
    import torch
    import config as cfg
    from inference import predict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = _load_model(checkpoint, device)
    stats = ck["normalization_stats"]

    S, P = cfg.PATCH_SIZE, cfg.PREDICTION_PATCHES
    n_ctx = cfg.MAX_CONTEXT_PATCHES
    pph = 60 // (5 * S)                       # patches per hour
    ctx_steps, pred_steps = n_ctx * S, P * S
    seq_len = n_ctx + P
    crop_patches = int(CROP_HOURS * pph)

    # The trailing forecast is masked in every case, so what is left of the
    # head's slots is what a backcast or an infill span may take.
    span_len = cfg.MAX_MASKED_PATCHES - P

    # 10 h of slack past the window, so the patch-aligned start below never runs
    # the window off the end of a short trajectory.
    feats, bg, carb, ins, exr, _hour = _sim_window(
        seed, stats, hours=(ctx_steps + pred_steps) * 5.0 / 60.0 + 10.0)
    start = ((len(bg) - ctx_steps - pred_steps) // S) * S
    ctx_np = feats[start:start + ctx_steps].reshape(n_ctx, S, cfg.N_INPUT_FEATURES)
    context = torch.from_numpy(ctx_np.copy())
    origin = start + ctx_steps

    # The whole announceable set, so every pass conditions on exactly what
    # training conditioned on. Anything left out is silently read as "none".
    # It is announced in all three passes, not just the forecast: the trailing
    # zone is masked under every masked set, so leaving it out would condition
    # the backcast and infill passes differently from the forecast one.
    ANNOUNCE = tuple(cfg.CHANNEL_TO_FEAT)                  # (0, 1, 2)
    ov = {ch: torch.from_numpy(
              feats[origin:origin + pred_steps, cfg.CHANNEL_TO_FEAT[ch]]
              .reshape(-1, S).copy())
          for ch in ANNOUNCE}

    infill_start = ((n_ctx // 2) // pph) * pph              # on an hour boundary
    CASES = (
        ("backcast", "the span opens the window",
         [(0, span_len), (n_ctx, P)], 0, span_len),
        ("infill", "the span sits between visible patches",
         [(infill_start, span_len), (n_ctx, P)], infill_start, span_len),
        ("forecast", "the span ends at the last patch",
         None, n_ctx, P),
    )

    def run(spans):
        return predict(model, context, normalization_stats=stats, device=device,
                       overrides=ov, mask_spans=spans)

    def span_fan(res, span_start: int, length: int):
        """The drawn span's own rows, in patch order: (bands, median) in mg/dL."""
        idx = res["mask_idx"].cpu().numpy()
        sel = np.flatnonzero((idx >= span_start) & (idx < span_start + length))
        sel = sel[np.argsort(idx[sel])]
        assert len(sel) == length, f"{len(sel)} rows for a {length}-patch span"
        bands = res["bands"].cpu().numpy()[sel].reshape(-1, cfg.N_QUANTILES)
        med = res["median_bg"].cpu().numpy().reshape(len(idx), S)[sel].reshape(-1)
        return bands, med

    _style(t)
    fig = plt.figure(figsize=(11.5, 13.0), facecolor=t.paper)
    gs = fig.add_gridspec(8, 1,
                          height_ratios=[0.95, 0.72, 2.05, 0.72, 2.05, 0.72, 2.05, 0.50],
                          hspace=0.0, left=0.085, right=0.985, top=0.930, bottom=0.098)
    loc_ax = fig.add_subplot(gs[0])
    for r in (1, 3, 5):
        fig.add_subplot(gs[r]).axis("off")
    crop_axes = [fig.add_subplot(gs[r]) for r in (2, 4, 6)]
    plan_ax = fig.add_subplot(gs[7])

    hours = lambda w: w * 5.0 / 60.0            # window step index -> hours
    window_bg = bg[start:start + ctx_steps + pred_steps]

    # ----------------------------------------------------------- locator strip
    loc_ax.set_facecolor(t.paper)
    loc_ax.axhspan(70, 180, color=t.fill(t.SAGE), zorder=0)
    loc_ax.plot(hours(np.arange(len(window_bg))), window_bg, color=t.ink, lw=0.7,
                zorder=4)
    for name, _sub, _spans, s0, length in CASES:
        loc_ax.axvspan(hours(s0 * S), hours((s0 + length) * S),
                       color=t.CLAY, alpha=0.55, lw=0, zorder=5)
        c0 = min(max(s0 + length // 2 - crop_patches // 2, 0), seq_len - crop_patches)
        loc_ax.axvspan(hours(c0 * S), hours((c0 + crop_patches) * S),
                       color=t.slate, alpha=0.10, lw=0, zorder=2)
        loc_ax.text(hours((c0 + crop_patches // 2) * S), 1.06, name,
                    transform=loc_ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=9.2, color=t.slate)
    loc_ax.set_xlim(0, hours(len(window_bg) - 1))
    loc_ax.set_ylim(40, 400)
    loc_ax.set_yticks([])
    loc_ax.set_xticks(np.arange(0, hours(len(window_bg)) + 1, 24.0))
    loc_ax.set_ylabel("window", fontsize=8.6, color=t.muted, style="italic")
    loc_ax.tick_params(axis="x", labelsize=8.6)
    loc_ax.spines["left"].set_visible(False)
    loc_ax.spines["bottom"].set_color(t.rule)

    # ------------------------------------------------------------------ crops
    for ax, (name, sub, spans, s0, length) in zip(crop_axes, CASES):
        bands, med = span_fan(run(spans), s0, length)
        a_step, join_left = _anchor_step(s0, length, S)
        anchor = float(window_bg[a_step])

        m0, m1 = s0 * S, (s0 + length) * S              # masked steps, window coords
        c0 = min(max(s0 + length // 2 - crop_patches // 2, 0), seq_len - crop_patches)
        v0, v1 = c0 * S, (c0 + crop_patches) * S        # crop steps, window coords

        # The fan and the median leave the observed line at the anchor, so both
        # are joined to it on the side the anchor rule chose.
        if join_left:
            tm = hours(np.arange(a_step, m1))
            med_j = np.concatenate([[anchor], med])
            bands_j = np.vstack([np.full((1, bands.shape[1]), anchor), bands])
        else:
            tm = hours(np.arange(m0, a_step + 1))
            med_j = np.concatenate([med, [anchor]])
            bands_j = np.vstack([bands, np.full((1, bands.shape[1]), anchor)])

        ax.set_facecolor(t.paper)
        ax.axhspan(70, 180, color=t.fill(t.SAGE), zorder=0)
        ax.axhline(70, color=t.muted, lw=0.7, ls=(0, (4, 4)), zorder=1)
        ax.axhline(180, color=t.muted, lw=0.7, ls=(0, (4, 4)), zorder=1)
        ax.axvspan(hours(m0), hours(m1 - 1), color=t.CLAY, alpha=0.07, lw=0, zorder=1)

        for lo, hi, alpha in ((0, 6, 0.16), (1, 5, 0.24), (2, 4, 0.34)):
            ax.fill_between(tm, bands_j[:, lo], bands_j[:, hi], color=t.TEAL,
                            alpha=alpha, lw=0, zorder=2)
        # Observed either side of the span, withheld inside it: the model is
        # shown the solid stretches and asked for the dashed one.
        for a, b in ((v0, m0 + 1), (m1 - 1, v1)):
            if b - a > 1:
                w = np.arange(max(a, v0), min(b, v1))
                ax.plot(hours(w), window_bg[w], color=t.ink, lw=1.6, zorder=5)
        w = np.arange(m0, m1)
        ax.plot(hours(w), window_bg[w], color=t.ink, lw=1.6, ls=(0, (4, 3)), zorder=5)
        ax.plot(tm, med_j, color=t.CLAY, lw=2.1, zorder=6)
        ax.plot([hours(a_step)], [anchor], marker="o", ms=4.0, color=t.CLAY,
                mec=t.paper, mew=0.9, zorder=7)

        seen = np.concatenate([window_bg[v0:v1], bands_j[:, 5], bands_j[:, 1], med_j])
        # A little air either side, so a span flush against the crop edge — the
        # backcast one always is, and the forecast one ends there — is not read
        # as a line running off the panel.
        ax.set_xlim(hours(v0) - 0.4, hours(v1 - 1) + 0.4)
        ax.set_ylim(max(40.0, min(85.0, float(np.nanmin(seen)) * 0.90)),
                    min(400.0, max(220.0, float(np.nanmax(seen)) * 1.07)))
        ax.set_ylabel("blood glucose (mg/dL)")
        ax.set_title(f"{name} — {sub}", fontsize=11.5, loc="left", color=t.ink, pad=8)
        ax.set_xticks(np.arange(np.ceil(hours(v0) / 6.0) * 6.0, hours(v1), 6.0))
        ax.tick_params(axis="x", labelsize=8.6, length=3)
        ax.set_xlabel("hours into the context window", fontsize=8.8, color=t.muted,
                      labelpad=2)
        ax.spines["left"].set_color(t.rule)
        ax.spines["bottom"].set_color(t.rule)

    crop_axes[-1].set_xticklabels([])
    crop_axes[-1].set_xlabel("")
    crop_axes[-1].tick_params(axis="x", length=0)
    crop_axes[-1].spines["bottom"].set_visible(False)

    # The plan strip belongs to the bottom crop's axis, and carries the channels
    # the model reads at every patch, masked or not.
    _n, _s, _sp, s0, length = CASES[-1]
    c0 = min(max(s0 + length // 2 - crop_patches // 2, 0), seq_len - crop_patches)
    sl = slice(start + c0 * S, start + (c0 + crop_patches) * S)
    td = hours(np.arange(c0 * S, (c0 + crop_patches) * S))
    plan_ax.set_facecolor(t.paper)
    plan_ax.fill_between(td, 0, carb[sl] / max(float(carb[sl].max()), 1e-6),
                         color=t.SAGE, alpha=0.55, lw=0, zorder=2)
    plan_ax.plot(td, ins[sl] / max(float(ins[sl].max()), 1e-6),
                 color=t.PLUM, lw=1.3, zorder=3)
    plan_ax.plot(td, exr[sl] / max(float(exr[sl].max()), 1e-6),
                 color=t.GOLD, lw=1.3, ls=(0, (3, 2)), zorder=3)
    plan_ax.axvspan(hours(s0 * S), hours((s0 + length) * S - 1),
                    color=t.CLAY, alpha=0.07, lw=0, zorder=1)
    plan_ax.set_xlim(crop_axes[-1].get_xlim())
    plan_ax.set_ylim(0, 1.45)
    plan_ax.set_yticks([])
    plan_ax.set_ylabel("plan", fontsize=8.6, color=t.muted, style="italic")
    plan_ax.set_xticks(crop_axes[-1].get_xticks())
    plan_ax.tick_params(axis="x", labelsize=8.6, length=3)
    plan_ax.spines["left"].set_visible(False)
    plan_ax.spines["bottom"].set_color(t.rule)

    handles = [
        Line2D([], [], color=t.ink, lw=1.6, label="observed CGM"),
        Line2D([], [], color=t.ink, lw=1.6, ls=(0, (4, 3)), label="withheld CGM"),
        Line2D([], [], color=t.CLAY, lw=2.1, label="predicted median"),
    ]
    loc_ax.legend(handles=handles, loc="lower left", fontsize=9, ncols=3,
                  labelcolor=t.slate, bbox_to_anchor=(0.0, 1.22))
    loc_ax.text(1.0, 1.22, "bands: τ .05–.95 · .10–.90 · .25–.75",
                transform=loc_ax.transAxes, ha="right", va="bottom",
                fontsize=8.8, color=t.muted)
    fig.text(0.5, 0.050,
             "One objective, three placements of the masked span. Each panel is a "
             f"separate forward pass over the same {n_ctx}-patch window above,\n"
             f"cropped to the {CROP_HOURS:.0f} h around its span. The plan strip carries "
             "carbohydrate appearance (filled), insulin action (solid)\n"
             "and exercise disposal (dashed), each normalised to its own peak; the model "
             f"reads them at masked patches too.  ·  T1DMSIM patient, seed {seed}.",
             ha="center", va="top", fontsize=9.0, color=t.slate, linespacing=1.7)

    fig.savefig(path, facecolor=t.paper)
    plt.close(fig)
    print(f"wrote {path}")


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    # No default: the ladder's checkpoints are per-run artifacts under the
    # gitignored models/<capacity>/, so any baked-in path is a promise the
    # tree cannot keep. The figure names its checkpoint or is skipped.
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint .pt for the masked-BG figure; required unless "
                         "--skip-forecast")
    # The seed the committed screenshots/ figures were drawn from; a rerun that
    # changes it silently replaces the README's patient with a different one.
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--skip-forecast", action="store_true",
                    help="draw only the two diagrams (no checkpoint)")
    args = ap.parse_args()

    if not args.skip_forecast:
        if args.checkpoint is None:
            raise SystemExit(
                "the masked-BG figure needs a checkpoint: pass --checkpoint PATH, "
                "or --skip-forecast to draw the two diagrams alone.")
        if not Path(args.checkpoint).is_file():
            raise SystemExit(f"no such checkpoint: {args.checkpoint}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for theme in (LIGHT, DARK):
        suffix = "" if theme.name == "light" else "-dark"
        draw_architecture(theme, out / f"architecture{suffix}.png")
        draw_risk_space(theme, out / f"risk-space{suffix}.png")
        if not args.skip_forecast:
            draw_masked_bg(theme, out / f"forecast{suffix}.png",
                           args.checkpoint, args.seed)


if __name__ == "__main__":
    os.chdir(REPO)
    main()
