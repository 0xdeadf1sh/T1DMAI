"""Chart drawing primitives for ``gui.py``: no mutable state beyond the per-call ``ChartTransform``.

pygame is imported in a try/except so the module loads headless; ``PYGAME_AVAILABLE`` is the
gate every drawing routine checks at entry.
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


# Multiplies every font size, widget rect and layout spacing; gui.py imports it from here.
UI_SCALE: float = 1.5


def ui_px(v: float) -> int:
    return int(round(v * UI_SCALE))


class ChartTransform:
    """Screen pixels ((0, 0) top-left, y down) ↔ chart space (x in patches, y the value, y up)."""

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
        x_range = max(self.cx_max - self.cx_min, 1e-9)
        y_range = max(self.cy_max - self.cy_min, 1e-9)

        sx = self.sx + (cx - self.cx_min) / x_range * self.sw
        sy = self.sy + (1.0 - (cy - self.cy_min) / y_range) * self.sh
        return sx, sy

    def screen_to_chart(self, sx: float, sy: float) -> tuple[float, float]:
        x_range = max(self.cx_max - self.cx_min, 1e-9)
        y_range = max(self.cy_max - self.cy_min, 1e-9)

        cx = self.cx_min + (sx - self.sx) / self.sw * x_range
        cy = self.cy_min + (1.0 - (sy - self.sy) / self.sh) * y_range
        return cx, cy

    def x_to_screen(self, cx: float) -> float:
        x_range = max(self.cx_max - self.cx_min, 1e-9)
        return self.sx + (cx - self.cx_min) / x_range * self.sw

    def y_to_screen(self, cy: float) -> float:
        y_range = max(self.cy_max - self.cy_min, 1e-9)
        return self.sy + (1.0 - (cy - self.cy_min) / y_range) * self.sh

    def update(
        self,
        chart_x_min: float | None = None,
        chart_x_max: float | None = None,
        chart_y_min: float | None = None,
        chart_y_max: float | None = None,
    ) -> None:
        if chart_x_min is not None:
            self.cx_min = chart_x_min
        if chart_x_max is not None:
            self.cx_max = chart_x_max
        if chart_y_min is not None:
            self.cy_min = chart_y_min
        if chart_y_max is not None:
            self.cy_max = chart_y_max


def draw_grid(
    surface: Any,
    transform: ChartTransform,
    n_context_patches: int,
    major_interval_patches: float = 12.0,  # 6 h, at _PATCHES_PER_HOUR = 2
    minor_interval_patches: float = 2.0,   # 1 h
    grid_color: tuple[int, int, int] = (48, 48, 56),
    minor_color: tuple[int, int, int] = (36, 36, 44),
    font: Any = None,
    text_color: tuple[int, int, int] = (140, 140, 155),
) -> None:
    """Time gridlines, and labels when ``font`` is given.

    Intervals are in patches and may be fractional, so sub-patch divisions render when zoomed
    in; the caller usually picks them adaptively (``gui._adaptive_time_intervals``).
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
            # a minor line landing on a major one
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
            # patches × steps/patch × 5 min/step
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
    """Fill a horizontal band between two chart-space y values, numerically low/high.

    Clipped to the chart rect, so an out-of-view value needs no clamping first.
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
    """The vertical NOW divider at patch ``n_context_patches``, context from prediction."""
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
    """Line curve: ``times`` ``(N,)`` patch indices, ``values`` ``(N,)`` chart-space."""
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
    """Blit a ``utils.ClockGeometry`` clock face; ``cx``/``cy``/``radius`` in screen pixels.

    Trig lives in ``utils.clock_wedge_geometry`` (y-up unit-disk); the only host step is the
    y-DOWN flip ``(x, y) -> (cx + radius*x, cy - radius*y)``. ``tick_color=None`` skips ticks.
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
