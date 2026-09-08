"""Matplotlib adapter for the clock-face histogram; trig lives in ``utils.clock_wedge_geometry``."""

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
    """Hour-of-day histogram into ``ax``, y-up unit-disk coordinates.

    ``probs`` ``(n_bins,)`` non-negative softmax belief; ``rotation_hours`` in hours.
    No-op when ``ax`` or ``probs`` is None, or matplotlib is unavailable.
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
