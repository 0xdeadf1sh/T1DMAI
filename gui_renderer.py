"""
T1DMAI GUI Renderer — chart drawing primitives.
================================================

Pure rendering helpers used by ``gui.py``.  Nothing here owns mutable state
beyond the per-call ``ChartTransform`` — every drawing routine takes the
target surface, the transform, and the data, and just blits.

Provides:
  * ``ChartTransform`` — conversion between screen pixels and chart
    (patch, value) coordinates.  All chart code goes through here so
    panning / zooming is a single transform update.
  * ``draw_grid`` / ``draw_axes`` / ``draw_curve`` / ``draw_now_line`` — the
    drawing primitives. Patch-aligned major gridlines fall every 6 hours
    (= 6 × ``_PATCHES_PER_HOUR`` patches).

pygame is imported lazily inside a try/except so this module loads cleanly
in environments where pygame isn't available (e.g. headless test runs).
``PYGAME_AVAILABLE`` is the gate every drawing routine checks at entry.
"""

import math
import numpy as np
from typing import Any

from utils import clock_reference_ticks

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# Global UI scale factor. Multiply font sizes, widget rects, and layout
# spacings by this to produce final pixel values, so the whole interface
# scales as one piece for hi-DPI / large displays. Both ``gui.py`` and
# ``gui_controls.py`` import ``UI_SCALE`` and ``ui_px`` from here so they
# stay in lockstep.
UI_SCALE: float = 1.5


def ui_px(v: float) -> int:
    """Scale a design-time pixel value by ``UI_SCALE`` and round to int."""
    return int(round(v * UI_SCALE))


# ============================================================================
# Coordinate Transform
# ============================================================================

class ChartTransform:
    """
    Bidirectional coordinate transform between screen pixels and chart data space.

    Screen coordinates: (0, 0) at top-left, x→right, y→down.
    Chart coordinates: (x=time in patches, y=data value) with y-axis inverted
    (lower data values appear lower on screen).

    Args:
        screen_x: Left edge of chart area in screen pixels.
        screen_y: Top edge of chart area in screen pixels.
        screen_w: Width of chart area in pixels.
        screen_h: Height of chart area in pixels.
        chart_x_min: Minimum time value (patch index or minutes).
        chart_x_max: Maximum time value.
        chart_y_min: Minimum data value (bottom of chart).
        chart_y_max: Maximum data value (top of chart).
    """

    def __init__(
        self,
        screen_x: float,
        screen_y: float,
        screen_w: float,
        screen_h: float,
        chart_x_min: float,
        chart_x_max: float,
        chart_y_min: float,
        chart_y_max: float,
    ) -> None:
        self.sx = screen_x
        self.sy = screen_y
        self.sw = screen_w
        self.sh = screen_h
        self.cx_min = chart_x_min
        self.cx_max = chart_x_max
        self.cy_min = chart_y_min
        self.cy_max = chart_y_max

    def chart_to_screen(self, cx: float, cy: float) -> tuple[float, float]:
        """
        Convert chart coordinates to screen coordinates.

        Args:
            cx: Chart x value (time).
            cy: Chart y value (data).

        Returns:
            (sx, sy): Screen pixel coordinates.
        """
        x_range = max(self.cx_max - self.cx_min, 1e-9)
        y_range = max(self.cy_max - self.cy_min, 1e-9)

        sx = self.sx + (cx - self.cx_min) / x_range * self.sw
        # Y is inverted: higher data value → lower screen y (higher on screen)
        sy = self.sy + (1.0 - (cy - self.cy_min) / y_range) * self.sh
        return sx, sy

    def screen_to_chart(self, sx: float, sy: float) -> tuple[float, float]:
        """
        Convert screen coordinates to chart coordinates.

        Args:
            sx: Screen x in pixels.
            sy: Screen y in pixels.

        Returns:
            (cx, cy): Chart coordinates.
        """
        x_range = max(self.cx_max - self.cx_min, 1e-9)
        y_range = max(self.cy_max - self.cy_min, 1e-9)

        cx = self.cx_min + (sx - self.sx) / self.sw * x_range
        cy = self.cy_min + (1.0 - (sy - self.sy) / self.sh) * y_range
        return cx, cy

    def x_to_screen(self, cx: float) -> float:
        """Convert chart x to screen x only."""
        x_range = max(self.cx_max - self.cx_min, 1e-9)
        return self.sx + (cx - self.cx_min) / x_range * self.sw

    def y_to_screen(self, cy: float) -> float:
        """Convert chart y to screen y only."""
        y_range = max(self.cy_max - self.cy_min, 1e-9)
        return self.sy + (1.0 - (cy - self.cy_min) / y_range) * self.sh

    def update(
        self,
        chart_x_min: float | None = None,
        chart_x_max: float | None = None,
        chart_y_min: float | None = None,
        chart_y_max: float | None = None,
    ) -> None:
        """Update chart bounds in place."""
        if chart_x_min is not None:
            self.cx_min = chart_x_min
        if chart_x_max is not None:
            self.cx_max = chart_x_max
        if chart_y_min is not None:
            self.cy_min = chart_y_min
        if chart_y_max is not None:
            self.cy_max = chart_y_max


# ============================================================================
# Pygame Drawing Utilities (only available when pygame is installed)
# ============================================================================

def draw_grid(
    surface: Any,
    transform: ChartTransform,
    n_context_patches: int,
    major_interval_patches: float = 12.0,  # default: 6 h grid (= 6 × _PATCHES_PER_HOUR patches)
    minor_interval_patches: float = 2.0,   # default: hourly (= _PATCHES_PER_HOUR patches)
    grid_color: tuple[int, int, int] = (48, 48, 56),
    minor_color: tuple[int, int, int] = (36, 36, 44),
    font: Any = None,
    text_color: tuple[int, int, int] = (140, 140, 155),
) -> None:
    """
    Draw time gridlines (and optionally labels) on the chart.

    Intervals are in patches and may be fractional; the loop walks
    ``np.arange`` so sub-patch divisions render when the user is
    zoomed in. The intervals are usually picked adaptively by the caller
    (see ``_adaptive_time_intervals`` in ``gui.py``).
    """
    if not PYGAME_AVAILABLE:
        return

    major = max(float(major_interval_patches), 1e-3)
    minor = max(float(minor_interval_patches), 1e-3)

    chart_bottom = int(transform.sy + transform.sh)
    chart_top = int(transform.sy)
    sx_left = transform.sx
    sx_right = transform.sx + transform.sw

    # Minor lines first so major lines paint on top.
    minor_start = math.floor(transform.cx_min / minor) * minor
    minor_end = transform.cx_max + minor
    if minor < major and minor > 0:
        for p in np.arange(minor_start, minor_end, minor):
            sx = int(transform.x_to_screen(p))
            if sx < sx_left or sx > sx_right:
                continue
            # Skip those that coincide with major lines.
            if abs((p / major) - round(p / major)) < 1e-6:
                continue
            pygame.draw.line(surface, minor_color, (sx, chart_top), (sx, chart_bottom))

    major_start = math.floor(transform.cx_min / major) * major
    major_end = transform.cx_max + major
    for p in np.arange(major_start, major_end, major):
        sx = int(transform.x_to_screen(p))
        if sx < sx_left or sx > sx_right:
            continue
        pygame.draw.line(surface, grid_color, (sx, chart_top), (sx, chart_bottom))
        if font is not None:
            from config import PATCH_SIZE as _PS
            # patches × (steps/patch) × STEP_MINUTES (== 5 = 30 / PATCH_SIZE)
            total_minutes = int(round(p * _PS * 5))
            hours = (total_minutes // 60) % 24
            minutes = total_minutes % 60
            label = f"{hours:02d}:{minutes:02d}"
            img = font.render(label, True, text_color)
            surface.blit(img, (sx - img.get_width() // 2, chart_bottom + 4))


def draw_y_band(
    surface: Any,
    transform: ChartTransform,
    chart_y_low: float,
    chart_y_high: float,
    color: tuple[int, int, int],
    alpha: int = 28,
) -> None:
    """Fill a horizontal band between two chart-y values.

    ``chart_y_low``/``chart_y_high`` are in chart data space (numerically
    low/high — the function handles the screen-y inversion). The band is
    clipped to the chart rect, so callers can pass values that fall
    outside the current vertical view without manual clamping.
    """
    if not PYGAME_AVAILABLE:
        return
    sy_top = transform.y_to_screen(chart_y_high)
    sy_bot = transform.y_to_screen(chart_y_low)
    top = max(transform.sy, sy_top)
    bot = min(transform.sy + transform.sh, sy_bot)
    if bot - top < 1:
        return
    band = pygame.Surface((int(transform.sw), int(bot - top)), pygame.SRCALPHA)
    band.fill((*color, alpha))
    surface.blit(band, (int(transform.sx), int(top)))


def draw_now_line(
    surface: Any,
    transform: ChartTransform,
    n_context_patches: int,
    color: tuple[int, int, int] = (255, 255, 255),
    alpha: int = 120,
    font: Any = None,
    text_color: tuple[int, int, int] = (220, 220, 230),
) -> None:
    """
    Draw the vertical NOW divider line separating context from prediction.

    Args:
        surface: Pygame surface.
        transform: ChartTransform.
        n_context_patches: Position of the NOW line.
        color: Line color.
        alpha: Line alpha (0-255).
        font: Pygame font for "NOW" label.
        text_color: Label color.
    """
    if not PYGAME_AVAILABLE:
        return

    sx = int(transform.x_to_screen(n_context_patches))
    chart_top = int(transform.sy)
    chart_bottom = int(transform.sy + transform.sh)

    now_surf = pygame.Surface((2, chart_bottom - chart_top), pygame.SRCALPHA)
    now_surf.fill((*color, alpha))
    surface.blit(now_surf, (sx - 1, chart_top))

    if font is not None:
        label = font.render("NOW", True, text_color)
        surface.blit(label, (sx - label.get_width() // 2, chart_top - 20))


def draw_curve(
    surface: Any,
    transform: ChartTransform,
    times: np.ndarray,
    values: np.ndarray,
    color: tuple[int, int, int],
    width: int = 2,
    alpha: int = 255,
) -> None:
    """
    Draw a line curve on the chart.

    Args:
        surface: Pygame surface.
        transform: ChartTransform.
        times: (N,) array of x values (patch indices).
        values: (N,) array of y values.
        color: Line color.
        width: Line width in pixels.
        alpha: Line alpha (0-255).
    """
    if not PYGAME_AVAILABLE or len(times) < 2:
        return

    points = []
    for t, v in zip(times, values):
        sx, sy = transform.chart_to_screen(float(t), float(v))
        sx = max(transform.sx, min(transform.sx + transform.sw, sx))
        sy = max(transform.sy, min(transform.sy + transform.sh, sy))
        points.append((int(sx), int(sy)))

    if len(points) >= 2:
        if alpha < 255:
            line_surf = pygame.Surface(
                (int(transform.sw), int(transform.sh)), pygame.SRCALPHA
            )
            draw_color = (*color, alpha)
            pygame.draw.lines(line_surf, draw_color, False, [
                (int(sx - transform.sx), int(sy - transform.sy))
                for sx, sy in points
            ], width)
            surface.blit(line_surf, (int(transform.sx), int(transform.sy)))
        else:
            pygame.draw.lines(surface, color, False, points, width)


def draw_text(
    surface: Any,
    font: Any,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (220, 220, 230),
) -> None:
    """Draw text at screen position (x, y)."""
    if not PYGAME_AVAILABLE or font is None:
        return
    img = font.render(text, True, color)
    surface.blit(img, (x, y))


def draw_color_swatch(
    surface: Any,
    x: int,
    y: int,
    color: tuple[int, int, int],
    size: int = 12,
) -> None:
    """Draw a small colored square swatch."""
    if not PYGAME_AVAILABLE:
        return
    pygame.draw.rect(surface, color, (x, y, size, size))


def draw_clock_face(
    surface: Any,
    *,
    cx: int,
    cy: int,
    radius: int,
    geom: Any,
    face_color: tuple[int, int, int],
    wedge_color: tuple[int, int, int],
    hand_color: tuple[int, int, int],
    tick_color: tuple[int, int, int] | None = None,
    R: float | None = None,
) -> None:
    """Blit a ``utils.ClockGeometry`` clock-face histogram onto a pygame surface.

    Thin adapter — all trigonometry lives in ``utils.clock_wedge_geometry``; the
    only host-specific step is the y-DOWN screen flip
    ``(x, y) -> (cx + radius*x, cy - radius*y)`` (the geometry is y-up unit-disk).

    Args:
        surface: pygame surface to draw onto.
        cx, cy: clock-face center in screen pixels.
        radius: clock-face radius in screen pixels.
        geom: a ``utils.ClockGeometry`` (``wedges (n_bins, arc_segments+2, 2)``,
            ``magnitudes (n_bins,)``, ``hand (2,)``, ``R``) — y-up unit coords.
        face_color: filled backing disk color.
        wedge_color: per-bin wedge fill color.
        hand_color: resultant-hand line color.
        tick_color: optional 12/3/6/9 tick color; ``None`` skips the ticks.
        R: unused for geometry (invariant, already baked into ``geom``); accepted
            so callers may pass ``geom.R`` explicitly without effect.
    """
    if not PYGAME_AVAILABLE:
        return

    def _to_screen(x: float, y: float) -> tuple[int, int]:
        return int(round(cx + radius * x)), int(round(cy - radius * y))

    pygame.draw.circle(surface, face_color, (cx, cy), radius)

    for k in range(geom.wedges.shape[0]):
        if geom.magnitudes[k] <= 1e-6:
            continue
        pts = [_to_screen(float(vx), float(vy)) for vx, vy in geom.wedges[k]]
        pygame.draw.polygon(surface, wedge_color, pts)

    if tick_color is not None:
        for tick in clock_reference_ticks():
            ux, uy = float(tick[0]), float(tick[1])
            p0 = _to_screen(0.88 * ux, 0.88 * uy)
            p1 = _to_screen(ux, uy)
            pygame.draw.line(surface, tick_color, p0, p1, 1)

    hx, hy = float(geom.hand[0]), float(geom.hand[1])
    pygame.draw.line(surface, hand_color, (cx, cy), _to_screen(hx, hy), 2)
