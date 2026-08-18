"""Render paper-style training/validation figures for a T1DMAI run.

Reads logs/{training_log.csv, validation_log.csv, training_summary.json,
resolved_config.json} and writes a fixed set of PNG figures into figures/.

Usage:
    python make_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent
LOG_DIR = REPO / "logs"
OUT_DIR = REPO / "figures"

# Every cgega_* column in logs/validation_log.csv (and in each checkpoint's
# val_history) was written by train.py with the CG-EGA arguments transposed, so
# each stored value scores (true, pred) instead of (pred, true). They cannot be
# recomputed: they record a training run that no longer exists to re-score. This
# flag is the single gate on every consumer of those columns — here and in
# make_card.py, which imports it — so the panels are suppressed rather than
# deleted. Flip it to True once a retrain has regenerated the columns under the
# fixed argument order and every suppressed panel returns unchanged. It governs
# ONLY the validation-log columns; metrics/ and metrics/core/ recompute CG-EGA from
# stored forecasts and are unaffected.
CGEGA_COLUMNS_TRUSTWORTHY = False

# Architecture label shown in figure suptitles, derived from the resolved
# config at run time (set by run()).
_ARCH_LABEL = ""


def _arch_label(cfg: dict) -> str:
    return (f"D={cfg['d_model']}, {cfg['n_layers']}L, {cfg['n_heads']}H, "
            f"FFN={cfg['ffn_dim']}")


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10.5,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    if not rows:
        return {}
    cols: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            if v == "" or v is None:
                cols[k].append(np.nan)
            else:
                try:
                    cols[k].append(float(v))
                except ValueError:
                    cols[k].append(np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _ema(x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * x[i]
    return out


def _annotate_best(ax, x, y, label_fmt: str, higher_is_better: bool = False, color="k"):
    mask = np.isfinite(y)
    if not mask.any():
        return
    xx, yy = x[mask], y[mask]
    idx = int(np.argmax(yy) if higher_is_better else np.argmin(yy))
    ax.scatter([xx[idx]], [yy[idx]], color=color, s=22, zorder=5, edgecolor="white", linewidth=0.8)
    ax.annotate(
        label_fmt.format(val=yy[idx], step=int(xx[idx])),
        xy=(xx[idx], yy[idx]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
    )


def _suptitle(fig, title: str) -> None:
    fig.suptitle(f"{title}  —  {_ARCH_LABEL}", fontsize=12.5, y=1.005)


# ---------------------------------------------------------------- figures


def fig_loss(train, val, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    s_train = train["step"]
    l_train = train["loss_total"]
    l_ema = _ema(l_train, alpha=0.04)
    ax.plot(s_train, l_train, color="#9bb6d6", linewidth=0.7, alpha=0.65, label="train loss (per-100-step)")
    ax.plot(s_train, l_ema, color="#1f4e8c", linewidth=1.4, label="train loss (EMA-smoothed)")
    s_val = val["step"]
    v_total = val["val_loss_total"]
    ax.plot(s_val, v_total, color="#c5343c", linewidth=1.7, label="validation loss")
    _annotate_best(ax, s_val, v_total, "best val {val:.4f} @ step {step}", color="#c5343c")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (total)")
    ax.set_title("Training and validation loss")
    ax.legend(loc="upper right")
    _suptitle(fig, "Loss curves")
    fig.savefig(outdir / "fig01_loss.png")
    plt.close(fig)


def fig_loss_components(train, outdir: Path) -> None:
    """Risk-space loss components: pinball L_Q, DILATE L_D (+ shape / TDI split) and the learned Kendall-Gal log-σ weights."""
    comps = [("L_Q  (quantile pinball)", "loss_Q"),
             ("L_D  (DILATE total)", "loss_D"),
             ("L_D shape  (soft-DTW)", "loss_D_shape"),
             ("L_D TDI  (temporal)", "loss_D_tdi"),
             ("log σ_Q  (pinball uncertainty)", "log_sigma_Q"),
             ("log σ_D  (DILATE uncertainty)", "log_sigma_D")]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 5.6), sharex=True)
    s = train["step"]
    flat = list(axes.flat)
    for ax, (name, col) in zip(flat, comps):
        if col not in train:
            ax.set_visible(False)
            continue
        y = train[col]
        ax.plot(s, y, color="#c9d6e8", linewidth=0.6, alpha=0.7)
        ax.plot(s, _ema(y, 0.04), color="#1f4e8c", linewidth=1.3)
        ax.set_title(name)
        ax.set_ylabel("loss")
    for ax in flat[len(comps):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("training step")
    _suptitle(fig, "Loss components")
    fig.tight_layout()
    fig.savefig(outdir / "fig02_loss_components.png")
    plt.close(fig)


def fig_bg_rmse_horizons(val, outdir: Path) -> None:
    horizons = [("30 min", "bg_rmse_30"), ("60 min", "bg_rmse_60"),
                ("120 min", "bg_rmse_120"), ("180 min", "bg_rmse_180"),
                ("360 min", "bg_rmse_360"), ("480 min", "bg_rmse_480")]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    s = val["step"]
    cmap = plt.colormaps["viridis"]
    n = len(horizons)
    for i, (name, col) in enumerate(horizons):
        if col not in val:
            continue
        y = val[col]
        ax.plot(s, y, label=name, color=cmap(i / max(n - 1, 1)), linewidth=1.3)
    ax.set_xlabel("training step")
    ax.set_ylabel("BG RMSE  [mg/dL]")
    ax.set_title("Validation BG RMSE by prediction horizon")
    ax.legend(loc="upper right", ncols=2, title="horizon")
    _suptitle(fig, "Multi-horizon BG RMSE")
    fig.savefig(outdir / "fig04_bg_rmse_horizons.png")
    plt.close(fig)


def fig_clinical(val, outdir: Path) -> None:
    # (column, colour, higher_is_better, annotation, y-label, title). The CG-EGA
    # panel is appended only when its source column is trustworthy; the figure is
    # then laid out over however many panels survive, so a suppressed one leaves
    # a narrower figure rather than an empty axis.
    panels = [
        ("evalfix_mard@30", "#1f4e8c", False, "best {val:.2f}% @ {step}",
         "MARD @30m  [%]", "Mean Absolute Relative Difference (@30 min)"),
        ("evalfix_clarke_A@30", "#2a8a3e", True, "best {val:.2f}% @ {step}",
         "Clarke A @30m  [%]", "Clarke Error-Grid zone A (@30 min)"),
    ]
    if CGEGA_COLUMNS_TRUSTWORTHY:
        panels.append(
            ("cgega_ap_eu", "#8e44ad", True, "best {val:.3f} @ {step}",
             "CG-EGA %AP (eu)", "CG-EGA clinical accuracy (euglycemic %AP)"))
    else:
        print("  · fig05_clinical: CG-EGA panel omitted "
              "(CGEGA_COLUMNS_TRUSTWORTHY is False)")

    fig, axes = plt.subplots(1, len(panels), figsize=(4.53 * len(panels), 4.6),
                             squeeze=False)
    s = val["step"]
    for ax, (col, color, higher, fmt, ylabel, title) in zip(axes[0], panels):
        y = val[col]
        ax.plot(s, y, color=color, linewidth=1.4)
        _annotate_best(ax, s, y, fmt, higher_is_better=higher, color=color)
        ax.set_xlabel("training step"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.locator_params(axis="x", nbins=5)

    _suptitle(fig, "Clinical-style metrics")
    fig.tight_layout()
    fig.savefig(outdir / "fig05_clinical.png")
    plt.close(fig)


def fig_excursion(val, outdir: Path) -> None:
    panels = [
        ("hypo recall", "hypo_recall", "#1f4e8c"),
        ("hypo precision", "hypo_precision", "#1f4e8c"),
        ("hyper recall", "hyper_recall", "#c5343c"),
        ("hyper precision", "hyper_precision", "#c5343c"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0), sharex=True)
    s = val["step"]
    for ax, (name, col, c) in zip(axes.flat, panels):
        if col not in val:
            ax.set_visible(False)
            continue
        y = val[col]
        ax.plot(s, y, color=c, linewidth=1.4)
        _annotate_best(ax, s, y, "best {val:.3f} @ {step}", higher_is_better=True, color=c)
        ax.set_title(name)
        ax.set_ylim(min(0.5, np.nanmin(y) - 0.02), 1.0)
        ax.set_ylabel(name.split()[1])
    for ax in axes[-1, :]:
        ax.set_xlabel("training step")
    _suptitle(fig, "Hypo / hyper excursion detection")
    fig.tight_layout()
    fig.savefig(outdir / "fig06_excursion.png")
    plt.close(fig)


def fig_calibration(val, outdir: Path) -> None:
    """Marginal coverage of the central-90% quantile band (the outer
    ``QUANTILE_LEVELS`` edges, τ 0.05 / 0.95), per horizon — beside the width
    that bought it.

    Coverage alone is not a calibration statement: a band wide enough covers
    everything. The sharpness axis is always drawn, so a run that reaches 0.90
    cannot be read without the mg/dL width it took. When the validation pass
    emitted no ``sharp90@`` column the axis says so rather than disappearing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    s = val["step"]
    series = [("@30m", 30, "#1f4e8c"),
              ("@60m", 60, "#2a8a3e"),
              ("@120m", 120, "#c5343c")]

    ax = axes[0]
    for name, h, c in series:
        col = f"coverage90@{h}"
        if col not in val:
            continue
        y = val[col]
        if not np.isfinite(y).any():
            continue
        ax.plot(s, y, color=c, linewidth=1.4, label=name)
    ax.axhline(0.90, color="k", linestyle=":", linewidth=1.0, label="target = 0.90")
    ax.fill_between(s, 0.88, 0.92, color="gray", alpha=0.10, label="±2pp band")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("training step"); ax.set_ylabel("coverage")
    ax.set_title("90% quantile-band coverage (target 0.90)")
    ax.legend(loc="best", title="horizon")

    ax = axes[1]
    drawn = False
    for name, h, c in series:
        col = f"sharp90@{h}"
        if col not in val:
            continue
        y = val[col]
        if not np.isfinite(y).any():
            continue
        ax.plot(s, y, color=c, linewidth=1.4, label=name)
        drawn = True
    ax.set_xlabel("training step"); ax.set_ylabel("mean band width  [mg/dL]")
    ax.set_title("90% band sharpness (narrower = better at equal coverage)")
    if drawn:
        ax.legend(loc="best", title="horizon")
    else:
        ax.text(0.5, 0.5, "sharp90@ unpopulated in this run's validation log",
                ha="center", va="center", fontsize=9.5, color="#888888",
                transform=ax.transAxes)
        print("  · fig07_calibration: sharpness axis empty "
              "(validation_log.csv carries no sharp90@ value)")

    _suptitle(fig, "Uncertainty calibration")
    fig.tight_layout()
    fig.savefig(outdir / "fig07_calibration.png")
    plt.close(fig)


def fig_optim(train, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    s = train["step"]

    ax = axes[0]
    g = train["grad_norm"]
    ax.plot(s, g, color="#cccccc", linewidth=0.6, alpha=0.8)
    ax.plot(s, _ema(g, 0.04), color="#1f4e8c", linewidth=1.4, label="EMA")
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("‖∇‖₂  (log scale)")
    ax.set_title("Gradient norm")
    ax.legend(loc="best")

    ax = axes[1]
    if "lr_muon" in train: ax.plot(s, train["lr_muon"], label="Muon LR", color="#1f4e8c", linewidth=1.4)
    if "lr_adam" in train: ax.plot(s, train["lr_adam"], label="AdamW LR", color="#c5343c", linewidth=1.4)
    ax.set_xlabel("training step"); ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedule")
    ax.legend(loc="best")

    _suptitle(fig, "Optimizer health")
    fig.tight_layout()
    fig.savefig(outdir / "fig08_optim.png")
    plt.close(fig)


def fig_summary(train, val, outdir: Path) -> None:
    fig = plt.figure(figsize=(14.8, 9.6))
    gs = fig.add_gridspec(3, 3, hspace=0.58, wspace=0.42)

    s_t, s_v = train["step"], val["step"]
    l_t = train["loss_total"]
    ax = fig.add_subplot(gs[0, :])
    ax.plot(s_t, l_t, color="#9bb6d6", linewidth=0.5, alpha=0.6)
    ax.plot(s_t, _ema(l_t, 0.04), color="#1f4e8c", linewidth=1.4, label="train (EMA)")
    ax.plot(s_v, val["val_loss_total"], color="#c5343c", linewidth=1.6, label="validation")
    ax.set_title("Loss"); ax.set_xlabel("step"); ax.set_ylabel("total loss")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s_v, val["evalfix_mard@30"], color="#1f4e8c", linewidth=1.3)
    ax.set_title("MARD% @30m"); ax.set_xlabel("step"); ax.set_ylabel("%")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(s_v, val["evalfix_clarke_A@30"], color="#2a8a3e", linewidth=1.3)
    ax.set_title("Clarke A % @30m"); ax.set_xlabel("step"); ax.set_ylabel("%")

    ax = fig.add_subplot(gs[1, 2])
    for col, c in [("bg_rmse_30", "#1f77b4"), ("bg_rmse_120", "#2ca02c"),
                   ("bg_rmse_480", "#d62728")]:
        ax.plot(s_v, val[col], label=col.replace("bg_rmse_", "")+"m", color=c, linewidth=1.2)
    ax.set_title("BG RMSE (short/mid/long)"); ax.set_xlabel("step"); ax.set_ylabel("mg/dL")
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(s_v, val["hypo_recall"], color="#1f4e8c", linewidth=1.3, label="hypo R")
    ax.plot(s_v, val["hyper_recall"], color="#c5343c", linewidth=1.3, label="hyper R")
    ax.set_ylim(0.5, 1.0); ax.set_title("Excursion recall"); ax.set_xlabel("step"); ax.legend(loc="best", fontsize=8)

    # Coverage never appears without the width that bought it: the band's mean
    # mg/dL sharpness rides the twin axis, dashed.
    ax = fig.add_subplot(gs[2, 1])
    if "coverage90@60" in val:
        ax.plot(s_v, val["coverage90@60"], color="#1f4e8c", linewidth=1.3)
    ax.axhline(0.90, color="k", linestyle=":", linewidth=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("90% coverage @60m (target = 0.90)"); ax.set_xlabel("step"); ax.set_ylabel("coverage")
    ax_w = ax.twinx()
    ax_w.grid(False)
    if "sharp90@60" in val and np.isfinite(val["sharp90@60"]).any():
        ax_w.plot(s_v, val["sharp90@60"], color="#8a8a8a", linewidth=1.1, linestyle="--")
        ax_w.set_ylabel("width [mg/dL]", fontsize=8.5, color="#8a8a8a")
    else:
        ax_w.set_ylabel("width n/a", fontsize=8.5, color="#bbbbbb")
        ax_w.set_yticks([])

    ax = fig.add_subplot(gs[2, 2])
    g = train["grad_norm"]
    ax.plot(s_t, _ema(g, 0.04), color="#1f4e8c", linewidth=1.3)
    ax.set_yscale("log")
    ax.set_title("‖∇‖₂  (EMA)"); ax.set_xlabel("step")

    for a in fig.axes:
        a.locator_params(axis="x", nbins=5)
    _suptitle(fig, "Run overview")
    fig.savefig(outdir / "fig09_summary.png")
    plt.close(fig)


def fig_curve_match(val, outdir: Path) -> None:
    """Forecast curve / trend correlation over training (higher is better, ~[-1, 1]).

    bg_curve_corr is the anchor-relative forecast-shape correlation; roc_corr is
    the per-patch (30-min) ΔBG direction correlation. Both score the headline
    median_bg forecast.
    """
    series = [
        ("BG curve corr (anchor-relative)", "bg_curve_corr", "#1f4e8c"),
        ("ΔBG direction corr (per-patch)", "roc_corr", "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    s = val["step"]
    for name, col, c in series:
        if col not in val:
            continue
        y = val[col]
        if not np.isfinite(y).any():
            continue
        ax.plot(s, y, color=c, linewidth=1.4, label=name)
        _annotate_best(ax, s, y, "best {val:.3f} @ {step}", higher_is_better=True, color=c)
    ax.axhline(0.0, color="k", linestyle=":", linewidth=0.9)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("correlation")
    ax.set_title("Forecast curve / trend correlation (higher = better)")
    ax.legend(loc="best")
    _suptitle(fig, "Curve & trend match")
    fig.savefig(outdir / "fig11_curve_match.png")
    plt.close(fig)


def fig_trend_quality(val, outdir: Path) -> None:
    """Trend-head quality on per-PATCH (30-min) ΔBG.

    roc_corr (corr of pred vs true 30-min ΔBG), trend_gain_beta (OLS slope,
    target β≈1), and roc_rmse on its own panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    s = val["step"]

    ax = axes[0]
    rc = val.get("roc_corr")
    if rc is not None and np.isfinite(rc).any():
        ax.plot(s, rc, color="#1f4e8c", linewidth=1.4, label="roc_corr")
        _annotate_best(ax, s, rc, "best {val:.3f} @ {step}", higher_is_better=True, color="#1f4e8c")
    beta = val.get("trend_gain_beta")
    if beta is not None and np.isfinite(beta).any():
        ax.plot(s, beta, color="#8e44ad", linewidth=1.4, label="trend_gain_beta")
    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.9, label="β target = 1.0")
    ax.set_xlabel("training step")
    ax.set_ylabel("corr / slope")
    ax.set_title("Trend corr & gain β (per-patch 30-min ΔBG)")
    ax.legend(loc="best")

    ax = axes[1]
    rr = val.get("roc_rmse")
    if rr is not None and np.isfinite(rr).any():
        ax.plot(s, rr, color="#c5343c", linewidth=1.4)
        _annotate_best(ax, s, rr, "best {val:.2f} @ {step}", color="#c5343c")
    ax.set_xlabel("training step")
    ax.set_ylabel("ΔBG RMSE  [mg/dL]")
    ax.set_title("Trend ΔBG RMSE (per-patch 30-min)")

    _suptitle(fig, "Trend quality")
    fig.tight_layout()
    fig.savefig(outdir / "fig12_trend_quality.png")
    plt.close(fig)


def fig_cgega_regions(val, outdir: Path) -> None:
    """CG-EGA per region: %AP (accurate, higher better), %EP (erroneous, lower better).

    Every series on this figure comes from a cgega_* validation column, so when
    those columns are untrustworthy nothing is left to draw: the PNG is skipped
    outright rather than written empty.
    """
    if not CGEGA_COLUMNS_TRUSTWORTHY:
        print("  · fig13_cgega_regions: skipped, no PNG written "
              "(CGEGA_COLUMNS_TRUSTWORTHY is False)")
        return
    regions = [("hypo", "#1f4e8c"), ("eu", "#2a8a3e"), ("hyper", "#c5343c")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    s = val["step"]

    ax = axes[0]
    for region, c in regions:
        col = f"cgega_ap_{region}"
        if col not in val:
            continue
        y = val[col]
        if not np.isfinite(y).any():
            continue
        ax.plot(s, y, color=c, linewidth=1.4, label=region)
        _annotate_best(ax, s, y, "{val:.3f}", higher_is_better=True, color=c)
    ax.set_xlabel("training step")
    ax.set_ylabel("%AP")
    ax.set_title("CG-EGA %Accurate (higher = better)")
    ax.legend(loc="best", title="region")

    ax = axes[1]
    for region, c in regions:
        col = f"cgega_ep_{region}"
        if col not in val:
            continue
        y = val[col]
        if not np.isfinite(y).any():
            continue
        ax.plot(s, y, color=c, linewidth=1.4, label=region)
        _annotate_best(ax, s, y, "{val:.3f}", color=c)
    ax.set_xlabel("training step")
    ax.set_ylabel("%EP")
    ax.set_title("CG-EGA %Erroneous (lower = better)")
    ax.legend(loc="best", title="region")

    _suptitle(fig, "CG-EGA all regions")
    fig.tight_layout()
    fig.savefig(outdir / "fig13_cgega_regions.png")
    plt.close(fig)


def fig_clarke_zones(val, outdir: Path) -> None:
    """Clarke error-grid zone split: A+B (good), D, E (dangerous)."""
    zones = [
        ("A+B %", "clarke_AB_pct", "#2a8a3e", True),
        ("D %", "clarke_D_pct", "#e8a33d", False),
        ("E %", "clarke_E_pct", "#c5343c", False),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    s = val["step"]
    for name, col, c, higher in zones:
        if col not in val:
            continue
        y = val[col]
        if not np.isfinite(y).any():
            continue
        ax.plot(s, y, color=c, linewidth=1.4, label=name)
        _annotate_best(ax, s, y, "{val:.2f}% @ {step}", higher_is_better=higher, color=c)
    ax.set_xlabel("training step")
    ax.set_ylabel("zone fraction  [%]")
    ax.set_title("Clarke error-grid zone split")
    ax.legend(loc="best", title="zone")
    _suptitle(fig, "Clarke zones")
    fig.savefig(outdir / "fig14_clarke_zones.png")
    plt.close(fig)


# tod_* validation columns that fig_time_of_day consumes; the figure is skipped
# entirely when none is present or every one is all-NaN (a probe-off run, i.e.
# config.TIME_PROBE_ENABLED False, writes them but leaves them empty).
_TOD_COLUMNS = (
    "tod_mae_h", "tod_p90_h", "tod_mae_hiconf",
    "tod_acc_1h", "tod_acc_2h", "tod_acc_bin",
    "tod_bias_h", "tod_std_h", "tod_gross_rate", "tod_conf",
)


def _tod_present(val: dict[str, np.ndarray]) -> bool:
    """True iff at least one tod_* column exists and carries a finite value."""
    return any(c in val and np.isfinite(val[c]).any() for c in _TOD_COLUMNS)


def fig_time_of_day(val, outdir: Path) -> None:
    """Time-of-day probe reliability over training, read straight from the
    tod_* validation columns (no model forward needed).

    Panel A — circular error in hours (tod_mae_h, tod_p90_h, tod_mae_hiconf),
    lower is better, against the uniform-guess chance floor. For a target drawn
    uniformly over the 24 h clock the expected absolute circular error is 6 h, so
    that line is a fixed illustrative reference (a distributional math fact, not
    a config constant). Panel B — accuracy percentages (±1 h, ±2 h, 4-bin) with
    their uniform-guess chance lines (a ±1 h window spans 2 of 24 h ⇒ ≈8.3%, ±2 h
    ⇒ ≈16.7%, one of four bins ⇒ 25%; all illustrative). Panel C — reliability:
    signed circular bias (target 0), circular-sd precision, gross-error rate
    (>3 h), and the confidence magnitude R.
    """
    if not _tod_present(val):
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    s = val["step"]

    # Panel A — error in hours (lower is better).
    ax = axes[0]
    err_series = [
        ("MAE", "tod_mae_h", "#1f4e8c", "-"),
        ("p90 err", "tod_p90_h", "#c5343c", "--"),
        ("MAE hi-conf", "tod_mae_hiconf", "#2a8a3e", ":"),
    ]
    for name, col, c, ls in err_series:
        y = val.get(col)
        if y is None or not np.isfinite(y).any():
            continue
        ax.plot(s, y, color=c, linewidth=1.4, linestyle=ls, label=name)
    _annotate_best(ax, s, val.get("tod_mae_h", np.full_like(s, np.nan)),
                   "best MAE {val:.2f} h @ {step}", color="#1f4e8c")
    ax.axhline(6.0, color="k", linestyle=":", linewidth=1.0, label="chance ≈ 6 h")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("circular error  [h]")
    ax.set_title("Clock error (lower = better)")
    ax.legend(loc="best")

    # Panel B — accuracy % (higher is better); chance lines are uniform-guess.
    ax = axes[1]
    acc_series = [
        ("±1 h", "tod_acc_1h", "#1f4e8c", 100.0 * 2.0 / 24.0),
        ("±2 h", "tod_acc_2h", "#2a8a3e", 100.0 * 4.0 / 24.0),
        ("4-bin", "tod_acc_bin", "#8e44ad", 100.0 / 4.0),
    ]
    for name, col, c, chance in acc_series:
        y = val.get(col)
        if y is None or not np.isfinite(y).any():
            continue
        ax.plot(s, y, color=c, linewidth=1.4, label=name)
        ax.axhline(chance, color=c, linestyle=":", linewidth=0.9, alpha=0.6)
    _annotate_best(ax, s, val.get("tod_acc_2h", np.full_like(s, np.nan)),
                   "best ±2h {val:.0f}% @ {step}", higher_is_better=True, color="#2a8a3e")
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy  [%]")
    ax.set_title("Clock accuracy (chance = dotted)")
    ax.legend(loc="best")

    # Panel C — reliability. Left axis carries the O(1) hour/R series, a twin
    # right axis the gross-error percentage.
    ax = axes[2]
    handles = []
    bias = val.get("tod_bias_h")
    if bias is not None and np.isfinite(bias).any():
        handles += ax.plot(s, bias, color="#8e44ad", linewidth=1.4, label="bias [h]")
    std = val.get("tod_std_h")
    if std is not None and np.isfinite(std).any():
        handles += ax.plot(s, std, color="#1f4e8c", linewidth=1.4, label="sd [h]")
    conf = val.get("tod_conf")
    if conf is not None and np.isfinite(conf).any():
        handles += ax.plot(s, conf, color="#2a8a3e", linewidth=1.4, label="conf R")
    ax.axhline(0.0, color="k", linestyle=":", linewidth=0.9)
    ax.set_xlabel("training step")
    ax.set_ylabel("hours  /  R")
    ax.set_title("Clock reliability")

    gross = val.get("tod_gross_rate")
    if gross is not None and np.isfinite(gross).any():
        ax2 = ax.twinx()
        ax2.grid(False)
        handles += ax2.plot(s, gross, color="#c5343c", linewidth=1.4,
                            linestyle="--", label="gross >3h [%]")
        ax2.set_ylabel("gross-error rate  [%]", color="#c5343c")
        ax2.set_ylim(bottom=0.0)
    ax.legend(handles=handles, loc="best")

    _suptitle(fig, "Time-of-day probe")
    fig.tight_layout()
    fig.savefig(outdir / "fig16_time_of_day.png")
    plt.close(fig)


def fig_tir(val, outdir: Path) -> None:
    """Time-in-range: predicted vs true TIR and the mean absolute TIR error."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    s = val["step"]

    # pred_tir / true_tir / tir_err are FRACTIONS in the log — the validation
    # table renders them as `* 100.0`. Scaling here is what makes this panel and
    # that table the same statistic instead of one reading 100x low against its
    # own percentage axis.
    ax = axes[0]
    pred = val.get("pred_tir")
    true = val.get("true_tir")
    if pred is not None and np.isfinite(pred).any():
        ax.plot(s, 100.0 * pred, color="#1f4e8c", linewidth=1.4, label="predicted TIR")
    if true is not None and np.isfinite(true).any():
        ax.plot(s, 100.0 * true, color="#2a8a3e", linewidth=1.4, label="true TIR")
    ax.set_xlabel("training step")
    ax.set_ylabel("time in range  [%]")
    ax.set_title("Time-in-range: predicted vs true")
    ax.legend(loc="best")

    # ABSOLUTE, not signed. The producer is `(pred_frac - true_frac).abs()`
    # accumulated per window, so the two directions cancelled inside it and the
    # series cannot be negative: a "pred - true" title invites reading a sign
    # that is not carried, and a zero line the series can never reach reads as a
    # target rather than as an unattainable bound.
    ax = axes[1]
    err = val.get("tir_err")
    if err is not None and np.isfinite(err).any():
        ax.plot(s, 100.0 * err, color="#c5343c", linewidth=1.4)
        _annotate_best(ax, s, 100.0 * np.abs(err), "best {val:.2f} pp @ {step}",
                       color="#c5343c")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean |TIR error|  [pp]")
    ax.set_title("TIR error (mean absolute, per window)")

    _suptitle(fig, "Time-in-range error")
    fig.tight_layout()
    fig.savefig(outdir / "fig15_tir.png")
    plt.close(fig)


# ---------------------------------------------------------------- driver


def run() -> None:
    global _ARCH_LABEL
    if not LOG_DIR.exists():
        raise FileNotFoundError(f"missing logs dir: {LOG_DIR}")
    OUT_DIR.mkdir(exist_ok=True)

    cfg = json.loads((LOG_DIR / "resolved_config.json").read_text())
    _ARCH_LABEL = _arch_label(cfg)

    train = _read_csv(LOG_DIR / "training_log.csv")
    val = _read_csv(LOG_DIR / "validation_log.csv")

    fig_loss(train, val, OUT_DIR)
    fig_loss_components(train, OUT_DIR)
    fig_bg_rmse_horizons(val, OUT_DIR)
    fig_clinical(val, OUT_DIR)
    fig_excursion(val, OUT_DIR)
    fig_calibration(val, OUT_DIR)
    fig_optim(train, OUT_DIR)
    fig_summary(train, val, OUT_DIR)
    fig_curve_match(val, OUT_DIR)
    fig_trend_quality(val, OUT_DIR)
    fig_cgega_regions(val, OUT_DIR)
    fig_clarke_zones(val, OUT_DIR)
    fig_tir(val, OUT_DIR)
    fig_time_of_day(val, OUT_DIR)

    def _best(col: str, higher: bool) -> float | None:
        y = val.get(col)
        if y is None or not np.isfinite(y).any():
            return None
        return float(np.nanmax(y) if higher else np.nanmin(y))

    def _best_step(col: str, higher: bool) -> int | None:
        """Training step at the best finite value of ``col`` (None if all-NaN/absent)."""
        y = val.get(col)
        if y is None or not np.isfinite(y).any():
            return None
        idx = int(np.nanargmax(y) if higher else np.nanargmin(y))
        return int(val["step"][idx])

    figures = [
        "fig01_loss.png", "fig02_loss_components.png",
        "fig04_bg_rmse_horizons.png", "fig05_clinical.png", "fig06_excursion.png",
        "fig07_calibration.png", "fig08_optim.png", "fig09_summary.png",
        "fig11_curve_match.png", "fig12_trend_quality.png",
        "fig14_clarke_zones.png", "fig15_tir.png",
    ]
    if CGEGA_COLUMNS_TRUSTWORTHY:
        figures.insert(10, "fig13_cgega_regions.png")
    if _tod_present(val):
        figures.append("fig16_time_of_day.png")

    # Best-over-training CG-EGA. Withheld with the rest of the cgega_* readers —
    # a best-over-run taken from a transposed column would sit in summary.json
    # beside corrected metrics/ numbers and contradict them.
    cgega_best = {}
    if CGEGA_COLUMNS_TRUSTWORTHY:
        cgega_best = {
            "cgega_ap_eu": _best("cgega_ap_eu", higher=True),
            "cgega_ap_hypo": _best("cgega_ap_hypo", higher=True),
            "cgega_ap_hyper": _best("cgega_ap_hyper", higher=True),
        }
    else:
        print("  · summary.json: cgega_ap_{eu,hypo,hyper} omitted "
              "(CGEGA_COLUMNS_TRUSTWORTHY is False)")

    summary = {
        "arch_label": _ARCH_LABEL,
        "n_train_rows": int(len(train["step"])),
        "n_val_rows": int(len(val["step"])),
        "final_step": int(train["step"][-1]),
        "figures": figures,
        "best": {
            "val_loss_total": _best("val_loss_total", higher=False),
            "val_loss_step": _best_step("val_loss_total", higher=False),
            "mard_30m": _best("evalfix_mard@30", higher=False),
            "clarke_A_30m": _best("evalfix_clarke_A@30", higher=True),
            "hypo_recall": _best("hypo_recall", higher=True),
            "hyper_recall": _best("hyper_recall", higher=True),
            "bg_curve_corr": _best("bg_curve_corr", higher=True),
            "roc_corr": _best("roc_corr", higher=True),
            "roc_rmse": _best("roc_rmse", higher=False),
            "trend_amp_ratio": _best("trend_amp_ratio", higher=True),
            **cgega_best,
            "clarke_AB_pct": _best("clarke_AB_pct", higher=True),
            "tir_err_abs": (None if val.get("tir_err") is None or not np.isfinite(val["tir_err"]).any()
                            else float(np.nanmin(np.abs(val["tir_err"])))),
            "coverage90@60": _best("coverage90@60", higher=True),
            "tod_mae_h": _best("tod_mae_h", higher=False),
            "tod_acc_2h": _best("tod_acc_2h", higher=True),
            "tod_conf": _best("tod_conf", higher=True),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  → wrote {len(figures)} PNGs + summary.json to {OUT_DIR}")


def main() -> None:
    _set_style()
    print(f"Rendering figures from {LOG_DIR} ...")
    run()


if __name__ == "__main__":
    main()
