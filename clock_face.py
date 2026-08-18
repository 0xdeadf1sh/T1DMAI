"""Matplotlib adapter for the time-of-day probe clock-face histogram.

A thin, math-free host adapter over ``utils.clock_wedge_geometry`` (the single
place the wedge angles / magnitudes / resultant hand and the rotation-invariant
``R`` are derived).  All trigonometry lives in the geometry core; this module
plots its y-up unit-disk coordinates as-is (matplotlib is already y-up).
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: F401 (kept for parity with metrics.core.figures)
    _OK = True
except Exception:
    _OK = False

from utils import clock_wedge_geometry


def draw_clock_axis(ax, probs: "np.ndarray | None", *, rotation_hours: float = 0.0,
                    wedge_color='C0', hand_color='0.15', face_color='none',
                    show_hand: bool = True, title: "str | None" = None) -> None:
    """Render one hour-of-day clock-face histogram into a matplotlib Axes.

    Args:
        ax: target matplotlib Axes (typically an ``inset_axes``); no-op if ``ax``
            is None or matplotlib is unavailable.
        probs: ``(n_bins,)`` non-negative softmax belief on the hour-of-day dial,
            or None (a caller may pass None safely => no-op).
        rotation_hours: rigid dial rotation applied to every hour, in hours
            (0 => the literal native clock).
        wedge_color: fill color for the bin wedges.
        hand_color: color of the resultant hand.
        face_color: fill color of the unit face circle ('none' => no face).
        show_hand: draw the resultant hand when True.
        title: optional tiny axis title.

    Returns:
        None. Draws wedges (via ``utils.clock_wedge_geometry``), an optional face
        circle, and the resultant hand into ``ax`` in y-up unit-disk coordinates.
    """
    if not _OK or ax is None or probs is None:
        return
    probs = np.asarray(probs, dtype=np.float64)
    geom = clock_wedge_geometry(probs, rotation_hours=rotation_hours)

    if face_color != 'none':
        theta = np.linspace(0.0, 2.0 * np.pi, 128)
        ax.fill(np.cos(theta), np.sin(theta), color=face_color, zorder=0)

    for k in range(geom.wedges.shape[0]):
        if geom.magnitudes[k] <= 0.0:
            continue
        w = geom.wedges[k]
        ax.fill(w[:, 0], w[:, 1], color=wedge_color, zorder=1)

    if show_hand:
        hx, hy = float(geom.hand[0]), float(geom.hand[1])
        ax.plot([0.0, hx], [0.0, hy], color=hand_color, lw=1.0, zorder=2)

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect('equal')
    ax.axis('off')
    if title is not None:
        ax.set_title(title, fontsize=5, pad=1)
