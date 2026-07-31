"""Render the figures the README embeds, into screenshots/.

Three figures, each in a light and a dark variant so the README can serve the
matching one through a <picture> element:

    architecture{,-dark}.png   end-to-end pipeline: signals -> encoder -> quantile
                               fan -> mg/dL, and the three things that consume it
    risk-space{,-dark}.png     the Kovatchev warp and the loss asymmetry it creates
    forecast{,-dark}.png       an actual forecast from a trained checkpoint: the
                               2 h single-pass fan and the 8 h autoregressive roll

Usage:
    python make_readme_figures.py                       # all three
    python make_readme_figures.py --skip-forecast       # diagrams only (no torch)
    python make_readme_figures.py --checkpoint PATH --seed N
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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

    # --- sources -----------------------------------------------------------
    _box(ax, t, L, 0.940, 0.435, 0.052, "T1DMSIM behaviour simulator",
         "seed → patient → events → 5-min stream", accent=t.SAGE, mono_sub=False)
    _box(ax, t, R - 0.435, 0.940, 0.435, 0.052, "Real CGM cohorts",
         "OhioT1DM · AZT1D · ShanghaiT1DM · UVA/Padova",
         accent=t.PLUM, mono_sub=False)
    ax.text(R - 0.2175, 0.930, "logged events → absorption / action kernels",
            ha="center", va="top", fontsize=8.0, color=t.muted, style="italic")
    ax.text(L + 0.2175, 0.930, "pretraining corpus", ha="center", va="top",
            fontsize=8.0, color=t.muted, style="italic")

    _arrow(ax, t, (L + 0.2175, 0.930), (L + 0.2175, 0.900))
    _arrow(ax, t, (R - 0.2175, 0.930), (R - 0.2175, 0.900))
    ax.plot([L + 0.2175, R - 0.2175], [0.900, 0.900], color=t.muted, lw=1.2, zorder=1)
    _down(ax, t, 0.5, 0.900, 0.882)

    # --- observed signal ---------------------------------------------------
    _box(ax, t, L, 0.828, W, 0.052, "Three observed channels, one 5-minute grid",
         "CGM  mg/dL      ·      carbohydrate  g / step      ·      insulin  U / step",
         accent=t.NAVY)
    ax.text(0.5, 0.820, "insulin sensitivity and hepatic output are simulator latents — deliberately not inputs",
            ha="center", va="top", fontsize=8.0, color=t.muted, style="italic")
    _down(ax, t, 0.5, 0.812, 0.792)

    # --- transforms --------------------------------------------------------
    tw = 0.440
    _box(ax, t, L, 0.736, tw, 0.052, "CGM  →  Kovatchev f  →  z-score",
         "the model reads and writes glucose in risk space", accent=t.CLAY, mono_sub=False)
    _box(ax, t, R - tw, 0.736, tw, 0.052, "carb, insulin  →  log1p  →  z-score",
         "keeps rare event spikes off the scale", accent=t.CLAY, mono_sub=False)
    _arrow(ax, t, (L + tw / 2, 0.732), (0.5, 0.712))
    _arrow(ax, t, (R - tw / 2, 0.732), (0.5, 0.712))
    _down(ax, t, 0.5, 0.712, 0.700)

    # --- patching ----------------------------------------------------------
    _box(ax, t, L, 0.644, W, 0.052, "Patch embedding",
         "6 steps × 3 features (30 min)  →  one token  →  Linear → D_MODEL",
         accent=t.NAVY)
    ax.text(0.5, 0.636,
            "context 16–48 patches (8–24 h)   │   horizon 4 patches (2 h): glucose blanked, "
            "carb and insulin carry the announced plan",
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
         "RoPE · QK-norm · learned per-head ALiBi", accent=t.NAVY,
         title_size=10, sub_size=8.0, mono_sub=False, zorder=3)
    _down(ax, t, bx + bw / 2, gy + 0.092, gy + 0.080)
    _box(ax, t, bx, gy + 0.054, bw, 0.026, "RMSNorm", accent=t.slate,
         title_size=9.2, lw=0.9, zorder=3)
    _down(ax, t, bx + bw / 2, gy + 0.052, gy + 0.044)
    _box(ax, t, bx, gy + 0.018, bw, 0.026, "SwiGLU FFN", accent=t.SAGE,
         title_size=9.2, lw=0.9, zorder=3)

    # attention-mask inset
    mx, my, ms = 0.700, gy + 0.036, 0.146
    _mask_inset(ax, t, mx, my, ms)

    _down(ax, t, 0.5, gy, 0.372)

    # --- final norm --------------------------------------------------------
    _box(ax, t, 0.315, 0.322, 0.370, 0.042, "Final RMSNorm",
         "prediction-zone tokens only, from here on", accent=t.NAVY,
         sub_size=8.0, mono_sub=False)
    _arrow(ax, t, (0.5, 0.320), (0.5, 0.310))
    ax.plot([0.235, 0.765], [0.310, 0.310], color=t.muted, lw=1.2, zorder=1)
    _down(ax, t, 0.235, 0.310, 0.288)
    _down(ax, t, 0.765, 0.310, 0.288)

    # --- heads -------------------------------------------------------------
    _box(ax, t, L, 0.230, 0.435, 0.056, "Blood-glucose quantile head",
         "3-layer MLP → K low-frequency coefficients per patch", accent=t.TEAL,
         sub_size=8.2, mono_sub=False)
    _box(ax, t, R - 0.435, 0.230, 0.435, 0.056, "Time-of-day probe",
         "12 circular hour bins per patch · co-trains the trunk\n"
         "reads the clock off the trajectory, with no clock input",
         accent=t.PLUM, sub_size=8.2, mono_sub=False)
    _down(ax, t, 0.2325, 0.228, 0.196)

    # --- assembly + decode --------------------------------------------------
    _box(ax, t, L, 0.130, 0.640, 0.064, "Quantile assembly",
         "anchor f(last_bg)  +  global DCT median  +  softplus cumsum\n"
         "→  7 ascending quantiles  τ = .05 .10 .25 .50 .75 .90 .95",
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
         "simulator and real cohorts", accent=t.slate, sub_size=8.2, mono_sub=False)
    _box(ax, t, 0.5 - cw / 2, 0.036, cw, 0.056, "Interactive GUI",
         "what-if plans · rolling forecast", accent=t.slate, sub_size=8.2, mono_sub=False)
    _box(ax, t, R - cw, 0.036, cw, 0.056, "Exporters",
         ".pte / .tflite + descriptor", accent=t.slate, sub_size=8.2, mono_sub=False)
    ax.text(R - cw / 2, 0.028, "loaded on device by T1DMDROID", ha="center", va="top",
            fontsize=8.0, color=t.muted, style="italic")

    fig.savefig(path, bbox_inches="tight", pad_inches=0.18, facecolor=t.paper)
    plt.close(fig)
    print(f"wrote {path}")


def _mask_inset(ax, t: Theme, x: float, y: float, s: float) -> None:
    """A 2x2 schematic of the hybrid attention mask."""
    half = s / 2
    allow, block = t.fill(t.NAVY), t.paper
    cells = [((0, 1), allow), ((1, 1), block), ((0, 0), allow), ((1, 0), allow)]
    for (cx, cy), fc in cells:
        ax.add_patch(Rectangle((x + cx * half, y + cy * half), half, half,
                               facecolor=fc, edgecolor=t.edge(t.NAVY),
                               linewidth=0.9, zorder=3))
    ax.text(x + half / 2, y + half * 1.5, "ctx\n↔ctx", ha="center", va="center",
            fontsize=7.2, color=t.slate, zorder=4, linespacing=1.15)
    ax.text(x + half * 1.5, y + half * 1.5, "blocked", ha="center", va="center",
            fontsize=7.2, color=t.muted, zorder=4)
    ax.text(x + half / 2, y + half / 2, "pred\n→ctx", ha="center", va="center",
            fontsize=7.2, color=t.slate, zorder=4, linespacing=1.15)
    ax.text(x + half * 1.5, y + half / 2, "pred\n↔pred", ha="center", va="center",
            fontsize=7.2, color=t.slate, zorder=4, linespacing=1.15)
    ax.text(x + half, y + s + 0.012, "attention mask", ha="center", va="bottom",
            fontsize=8.6, weight="bold", color=t.ink, zorder=4)
    ax.text(x + half, y - 0.012, "context never sees the horizon",
            ha="center", va="top", fontsize=7.6, color=t.muted, style="italic", zorder=4)


# ------------------------------------------------------------------ risk space

_K_SCALE = 2.2211457449985317
_K_POWER = 1.084
_K_OFFSET = 5.540076976170212
BG_LO, BG_HI = 40.0, 400.0


def _f(g):
    return _K_SCALE * (np.log(g) ** _K_POWER - _K_OFFSET)


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
    a.annotate("f(40) = −√10", (BG_LO, _f(BG_LO)), textcoords="offset points",
               xytext=(10, 2), fontsize=8.6, color=t.slate)
    a.annotate("f(400) = +√10", (BG_HI, _f(BG_HI)), textcoords="offset points",
               xytext=(-8, 6), ha="right", fontsize=8.6, color=t.slate)
    a.text(125.0, -2.55, "target\nrange", fontsize=8.4, color=t.slate,
           ha="center", va="center", style="italic", linespacing=1.3)

    a.set_xlabel("blood glucose (mg/dL)")
    a.set_ylabel("risk space  f(g)")
    a.set_xlim(BG_LO, BG_HI)
    a.set_ylim(-3.5, 3.5)
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


# -------------------------------------------------------------------- forecast

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


def _sim_window(seed: int, stats):
    """One simulator patient: normalized features, raw mg/dL, carbs and insulin."""
    import numpy as np
    from T1DMSIM.simulator import T1DMSimulator, BG_CLAMP_MIN, BG_CLAMP_MAX
    from data import simulate_discard_warmup
    from normalization import normalize

    raw = simulate_discard_warmup(T1DMSimulator(seed=seed), 60.0)
    bg = np.clip(raw["bg_observed"], BG_CLAMP_MIN, BG_CLAMP_MAX).astype(np.float32)
    carb = np.maximum(raw["total_carb"], 0.0).astype(np.float32)
    ins = np.maximum(raw["total_insulin"], 0.0).astype(np.float32)
    feats = normalize(np.stack([bg, carb, ins], axis=-1), stats)
    return feats, bg, carb, ins, raw["hour_of_day"].astype(np.float32)


def draw_forecast(t: Theme, path: Path, checkpoint: str, seed: int) -> None:
    import torch
    import config as cfg
    from inference import predict, predict_rolling

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = _load_model(checkpoint, device)
    stats = ck["normalization_stats"]

    S, P = cfg.PATCH_SIZE, cfg.PREDICTION_PATCHES
    n_ctx = cfg.MAX_CONTEXT_PATCHES
    ctx_steps, pred_steps = n_ctx * S, P * S
    n_rolls = cfg.NIGHT_LONG_HORIZON_HOURS // cfg.PREDICTION_HORIZON_HOURS
    long_steps = n_rolls * pred_steps

    feats, bg, carb, ins, hour = _sim_window(seed, stats)
    start = ((len(bg) - ctx_steps - long_steps - 12) // S) * S
    ctx_np = feats[start:start + ctx_steps].reshape(n_ctx, S, cfg.N_INPUT_FEATURES)
    context = torch.from_numpy(ctx_np.copy())
    origin = start + ctx_steps

    def slot(ch, a, b):
        col = feats[a:b, cfg.CHANNEL_TO_FEAT[ch]].reshape(-1, S)
        return col

    ov = {ch: torch.from_numpy(slot(ch, origin, origin + pred_steps).copy())
          for ch in (0, 1)}
    short = predict(model, context, normalization_stats=stats, device=device, overrides=ov)

    def roll_ov(roll_idx, _mu, _abs_n_ctx):
        a = origin + roll_idx * pred_steps
        b = a + pred_steps
        if b > len(feats):
            return None
        norm = {ch: slot(ch, a, b).copy() for ch in (0, 1)}
        raw_ = {0: carb[a:b].reshape(-1, S).copy(), 1: ins[a:b].reshape(-1, S).copy()}
        return norm, raw_

    long = predict_rolling(model, context, n_rolls=int(n_rolls),
                           normalization_stats=stats, device=device,
                           overrides_fn=roll_ov)

    _style(t)
    fig = plt.figure(figsize=(11.5, 8.4), facecolor=t.paper)
    gs = fig.add_gridspec(5, 1, height_ratios=[3.0, 0.60, 1.30, 3.0, 0.60],
                          hspace=0.0, left=0.085, right=0.985, top=0.90, bottom=0.13)
    axes = [fig.add_subplot(gs[i]) for i in (0, 1, 3, 4)]
    fig.add_subplot(gs[2]).axis("off")

    panels = (
        (axes[0], axes[1], short["bands"].reshape(-1, cfg.N_QUANTILES).cpu().numpy(),
         short["median_bg"].cpu().numpy(), 4 * 12,
         f"{cfg.PREDICTION_HORIZON_HOURS} h forecast — a single forward pass"),
        (axes[2], axes[3], long["bands"].reshape(-1, cfg.N_QUANTILES).cpu().numpy(),
         long["pred_bg"].cpu().numpy(), 12 * 12,
         f"{int(n_rolls * cfg.PREDICTION_HORIZON_HOURS)} h forecast — "
         f"{int(n_rolls)} autoregressive rolls"),
    )

    for ax, dx, bands, med, ctx_show, title in panels:
        h = len(med)
        # t=0 is the forecast origin: the LAST CONTEXT step, index origin-1, whose BG is
        # the model's own last_bg anchor (data._build_sample takes bg[pred_start-1]).
        # Prediction step k is index origin+k, at (k+1)*5 min. Each future series is
        # therefore prepended with the anchor, so it leaves the observed line instead of
        # starting one step clear of it. For the bands that join is not cosmetic: the
        # head decodes a delta from last_bg, so the fan has zero width at the origin.
        tc = np.arange(-ctx_show, 1) * 5.0 / 60.0
        tf = np.arange(h + 1) * 5.0 / 60.0
        x0, x1 = tc[0], tf[-1]
        anchor = bg[origin - 1]
        ctx_bg = bg[origin - 1 - ctx_show:origin]
        fut_bg = np.concatenate([[anchor], bg[origin:origin + h]])
        fut_med = np.concatenate([[anchor], med])
        fut_bands = np.vstack([np.full((1, bands.shape[1]), anchor), bands])

        ax.set_facecolor(t.paper)
        ax.axhspan(70, 180, color=t.fill(t.SAGE), zorder=0)
        ax.axhline(70, color=t.muted, lw=0.7, ls=(0, (4, 4)), zorder=1)
        ax.axhline(180, color=t.muted, lw=0.7, ls=(0, (4, 4)), zorder=1)
        for lo, hi, alpha in ((0, 6, 0.16), (1, 5, 0.24), (2, 4, 0.34)):
            ax.fill_between(tf, fut_bands[:, lo], fut_bands[:, hi], color=t.TEAL,
                            alpha=alpha, lw=0, zorder=2)
        ax.plot(tc, ctx_bg, color=t.ink, lw=1.6,
                zorder=5, label="observed CGM")
        ax.plot(tf, fut_bg, color=t.ink, lw=1.6, ls=(0, (4, 3)),
                zorder=5, label="true future CGM")
        ax.plot(tf, fut_med, color=t.CLAY, lw=2.1, zorder=6, label="forecast median")
        ax.axvline(0.0, color=t.muted, lw=1.0, zorder=3)

        seen = np.concatenate([ctx_bg, fut_bg, fut_bands[:, 5], fut_med])
        top = min(400.0, max(220.0, float(np.nanmax(seen)) * 1.07))
        bot = max(40.0, min(85.0, float(np.nanmin(seen)) * 0.90))
        ax.set_xlim(x0, x1)
        ax.set_ylim(bot, top)
        ax.set_ylabel("blood glucose (mg/dL)")
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, fontsize=11.5, loc="left", color=t.ink, pad=9)
        ax.spines["left"].set_color(t.rule)
        ax.spines["bottom"].set_visible(False)

        # announced plan, on its own strip
        sl = slice(origin - 1 - ctx_show, origin + h)
        td = np.arange(-ctx_show, h + 1) * 5.0 / 60.0
        c_max = max(float(carb[sl].max()), 1e-6)
        i_max = max(float(ins[sl].max()), 1e-6)
        dx.set_facecolor(t.paper)
        dx.fill_between(td, 0, carb[sl] / c_max, color=t.SAGE, alpha=0.55, lw=0,
                        zorder=2)
        dx.plot(td, ins[sl] / i_max, color=t.PLUM, lw=1.3, zorder=3)
        dx.axvline(0.0, color=t.muted, lw=1.0, zorder=1)
        dx.set_xlim(x0, x1)
        dx.set_ylim(0, 1.45)
        dx.set_yticks([])
        dx.set_ylabel("plan", fontsize=8.6, color=t.muted, style="italic")
        dx.set_xlabel("hours from the forecast origin")
        dx.spines["left"].set_visible(False)
        dx.spines["bottom"].set_color(t.rule)

    axes[0].legend(loc="upper left", fontsize=9, ncols=3, labelcolor=t.slate,
                   bbox_to_anchor=(0.0, 1.30))
    axes[0].text(1.0, 1.30, "bands: τ .05–.95 · .10–.90 · .25–.75",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=8.8, color=t.muted)
    fig.text(0.5, 0.062,
             "The plan strip carries the announced carbohydrate appearance (filled) and "
             "insulin action (line), each normalised to its own peak.\n"
             f"T1DMSIM patient, seed {seed}; the window's carbohydrate and insulin are "
             "announced to the model, as a declared plan would be in deployment.",
             ha="center", va="top", fontsize=9.0, color=t.slate, linespacing=1.6)

    fig.savefig(path, facecolor=t.paper)
    plt.close(fig)
    print(f"wrote {path}")


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--checkpoint", default="models/large/weights_multi.pt")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip-forecast", action="store_true",
                    help="draw only the two diagrams (no torch, no checkpoint)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for theme in (LIGHT, DARK):
        suffix = "" if theme.name == "light" else "-dark"
        draw_architecture(theme, out / f"architecture{suffix}.png")
        draw_risk_space(theme, out / f"risk-space{suffix}.png")
        if not args.skip_forecast:
            draw_forecast(theme, out / f"forecast{suffix}.png",
                          args.checkpoint, args.seed)


if __name__ == "__main__":
    os.chdir(REPO)
    main()
