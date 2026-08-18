"""
Shared figure chrome for the top-level ``metrics/`` probes.

One copy of the palette and the chrome, so a cohort carries the same hue in every
figure under ``metrics/figures/`` and a reader learns the mapping once. Each probe
owns its own panels; only the look lives here.

Colour jobs, kept deliberately apart — a mark encodes exactly one of them:

``COHORT``     identity. A source's hue never changes and is never reassigned by
               rank, so it survives a figure that drops one.
``DOSE_CARB``  ordinal. A dose ladder is an ordered magnitude, so it rides one hue
``DOSE_INS``   light→dark rather than three categorical hues.
``PAIR``       before→after (base→shifted, logged→reconstructed):
               one hue in two shades. Identity there is carried by the row label,
               so the two jobs never collide on one mark.

Palette provenance: the categorical trio and the blue ramp are the data-viz
reference palette; the orange ramp was stepped against it. All three were run
through the palette validator on this surface. The trio passes every all-pairs
gate (worst CVD ΔE 9.2, worst normal-vision ΔE 24.0); both ordinal ramps pass
lightness-monotonicity, the adjacent-step gap and the light-end contrast floor.
Aqua measures 2.74:1 against the surface, under the 3:1 gate, so every figure
using it carries direct value labels — the documented relief for that warning.

Thresholds are drawn as labelled dashed rules in muted ink rather than in status
red: they mark a clinical boundary on the axis, not the state of any series, and
a status colour that appears without an icon-and-label pairing reads as one.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figures')

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# Categorical slots 1-3, in fixed order. Never cycled, never rank-assigned.
SERIES = ('#2a78d6', '#eb6834', '#1baf7a')
COHORT = {'sim': SERIES[0]}
COHORT_LABEL = {'sim': 'T1DMSIM', 'T1DMSIM': 'T1DMSIM'}

DOSE_CARB = ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281')
DOSE_INS = ('#f2905f', '#eb6834', '#cc5321', '#a03f18', '#752d11')
# Continuous magnitude (density) may recede into the surface at its light end, as
# an ordinal ladder may not — hence the extra pale step this ramp opens on.
SEQ = ('#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#104281')
PAIR = ('#5598e7', '#104281')                  # before → after, one hue two shades
REFERENCE = MUTED                              # the simulator, drawn as a reference rule


def style() -> None:
    """Install the shared rcParams. Idempotent; call once per script."""
    plt.rcParams.update({
        'figure.dpi': 110,
        'figure.facecolor': SURFACE,
        'savefig.dpi': 200,
        'savefig.facecolor': SURFACE,
        'savefig.bbox': 'tight',
        'font.family': 'DejaVu Sans',
        'font.size': 9.5,
        'axes.facecolor': SURFACE,
        'axes.titlesize': 10.5,
        'axes.titlelocation': 'left',
        'axes.titlecolor': INK,
        'axes.labelsize': 9.5,
        'axes.labelcolor': INK2,
        'axes.edgecolor': AXIS,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': GRID,
        'grid.linestyle': '-',                 # solid hairline; dashes are reserved for thresholds
        'grid.linewidth': 0.8,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK2,
        'ytick.labelcolor': INK2,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'lines.linewidth': 2.0,
        'lines.solid_capstyle': 'round',
        'legend.frameon': False,
        'legend.fontsize': 8.5,
        'legend.labelcolor': INK2,
    })


def cohort_color(ds: str) -> str:
    """The hue that identifies ``ds`` everywhere in this directory."""
    return COHORT.get(ds, MUTED)


def label(ds: str) -> str:
    return COHORT_LABEL.get(ds, ds)


def threshold(ax, y: float, text: str, side: str = 'right') -> None:
    """A labelled clinical boundary: dashed muted rule, never a status colour."""
    ax.axhline(y, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    # The rule can land anywhere, including across a bar, so the label carries a
    # surface-coloured backing rather than relying on whatever is behind it.
    ax.annotate(text, xy=(1.0 if side == 'right' else 0.0, y),
                xycoords=('axes fraction', 'data'), xytext=(-2 if side == 'right' else 2, 3),
                textcoords='offset points', ha=side, va='bottom', fontsize=7.5, color=MUTED,
                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.8), zorder=4)


def zeroline(ax) -> None:
    """The baseline a signed response is read against."""
    ax.axhline(0.0, color=AXIS, lw=1.0, zorder=1)


def bar_labels(ax, bars, fmt: str = '{:.2f}', dy: float = 2.0, tops=None) -> None:
    """Value at the cap of every column — the relief the aqua contrast warning needs.

    ``tops`` overrides where the label anchors, so a column carrying an error bar
    labels above the whisker rather than under it.
    """
    for k, b in enumerate(bars):
        h = b.get_height()
        if h is None or h != h:
            continue
        y = h if tops is None else tops[k]
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, y),
                    xytext=(0, dy if h >= 0 else -dy - 7), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7.5, color=INK2)


def ygrid(ax) -> None:
    """Horizontal rules only — a vertical grid through categorical bars is noise."""
    ax.grid(axis='y'); ax.grid(axis='x', visible=False)


def legend(ax, **kw):
    """``ax.legend`` with the text held in ink, never in the series colour."""
    kw.setdefault('labelcolor', INK2)
    return ax.legend(**kw)


def pair_legend(host, before: str, after: str, **kw):
    """Key for a before→after panel: the two shades, named. Host is an Axes or a Figure."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', linestyle='none', markersize=7, markerfacecolor=c,
                      markeredgecolor=SURFACE, markeredgewidth=2, label=t)
               for c, t in zip(PAIR, (before, after))]
    kw.setdefault('labelcolor', INK2)
    kw.setdefault('frameon', False)
    kw.setdefault('fontsize', 8.5)
    return host.legend(handles=handles, **kw)


def rows(ax, labels: list[str]) -> None:
    """Chrome for a horizontal row chart: label each row, rule along x only."""
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.invert_yaxis()
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.margins(x=0.24)                         # room for the widest end label


def density_cmap():
    """One-hue light→dark ramp for a count density (hexbin). Never a rainbow."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('t1dm_seq', SEQ)


def cohort_legend(fig, dss, y: float = 0.975):
    """One figure-level cohort key, so no panel has to give up space to a legend box."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cohort_color(d), edgecolor=SURFACE, label=label(d)) for d in dss]
    return fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.995, y),
                      ncol=len(dss), frameon=False, fontsize=8.5, labelcolor=INK2)


def dumbbell_rows(ax, labels: list[str], before: list[float], after: list[float],
                  fmt: str = '{:.2f}') -> None:
    """Before→after rows: one hue in two shades, joined, values direct-labelled.

    A row whose two ends nearly coincide gets ONE combined ``a → b`` label past the
    outer end instead of two labels that would overprint each other — the span the
    decision reads is the whole panel's, so it is taken once, here, rather than
    guessed per row.
    """
    vals = [v for v in list(before) + list(after) if v is not None]
    span = (max(vals) - min(vals)) or 1.0
    for y, (a, b) in enumerate(zip(before, after)):
        ax.plot([a, b], [y, y], color=PAIR[0], lw=1.5, zorder=2, solid_capstyle='butt')
        ax.scatter([a], [y], s=52, color=PAIR[0], zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.scatter([b], [y], s=52, color=PAIR[1], zorder=3, edgecolor=SURFACE, linewidth=2)
        if abs(b - a) < 0.12 * span:
            ax.annotate(f'{fmt.format(a)} → {fmt.format(b)}', xy=(max(a, b), y), xytext=(8, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=7.5, color=INK2)
            continue
        for v, out in ((a, -8 if a <= b else 8), (b, 8 if a <= b else -8)):
            ax.annotate(fmt.format(v), xy=(v, y), xytext=(out, 0), textcoords='offset points',
                        ha='left' if out > 0 else 'right', va='center',
                        fontsize=7.5, color=INK2)
    rows(ax, labels)


def save(fig, name: str) -> str:
    """Write ``name`` into metrics/figures/ and return the repo-relative path."""
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    plt.close(fig)
    return os.path.relpath(path, os.path.dirname(HERE))
