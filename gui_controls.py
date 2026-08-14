# pyright: reportPossiblyUnboundVariable=false
# All ``pygame`` references are runtime-guarded by ``PYGAME_AVAILABLE`` (see
# the try/except import below); pyright can't follow that flag.
"""
T1DMAI GUI Controls — input widgets.
=====================================

Sliders, buttons, toggles, and the meal / bolus event builders
used by the main GUI.  Every widget is a self-contained class with two
public-ish methods:

* ``draw(surface)``   — blits the widget at its current state.
* ``handle_event(e)`` — consumes a pygame event and updates state.
                         Returns True if the event was consumed.

The base ``Widget`` class wraps the bounding rectangle and a visible /
enabled pair of flags.  Concrete widgets subclass it and add their own
event hooks.

pygame is imported under a try / except so the module loads cleanly when
pygame is unavailable (e.g. on headless CI).  ``PYGAME_AVAILABLE`` is the
gate widgets check before touching pygame APIs.
"""

import math
import numpy as np
from typing import Any, Callable

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from gui_renderer import draw_text, draw_color_swatch, ChartTransform, ui_px


# ============================================================================
# Base widget
# ============================================================================

class Widget:
    """Base class for all GUI widgets."""

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.rect = (x, y, w, h)
        self.visible = True
        self.enabled = True

    def contains(self, mx: int, my: int) -> bool:
        """Return True if (mx, my) is inside this widget."""
        x, y, w, h = self.rect
        return x <= mx < x + w and y <= my < y + h

    def draw(self, surface: Any, font: Any) -> None:
        """Draw the widget. Override in subclasses."""
        pass

    def handle_event(self, event: Any) -> bool:
        """Handle a pygame event. Returns True if consumed."""
        return False


# ============================================================================
# Button
# ============================================================================

class Button(Widget):
    """
    A clickable button with a text label.

    Args:
        x, y, w, h: Position and size.
        label: Button text.
        callback: Function to call on click.
        color: Button background color.
        hover_color: Color when hovered.
        text_color: Label color.
    """

    def __init__(
        self,
        x: int, y: int, w: int, h: int,
        label: str,
        callback: Callable | None = None,
        color: tuple[int, int, int] = (60, 70, 90),
        hover_color: tuple[int, int, int] = (80, 100, 130),
        text_color: tuple[int, int, int] = (220, 220, 230),
    ) -> None:
        super().__init__(x, y, w, h)
        self.label = label
        self.callback = callback
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self._hovered = False

    def draw(self, surface: Any, font: Any) -> None:
        if not self.visible or not PYGAME_AVAILABLE:
            return
        x, y, w, h = self.rect
        bg = self.hover_color if self._hovered else self.color
        if not self.enabled:
            bg = (40, 40, 50)
        pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=4)
        pygame.draw.rect(surface, (80, 90, 110), (x, y, w, h), 1, border_radius=4)
        if font is not None:
            img = font.render(self.label, True, self.text_color if self.enabled else (100, 100, 120))
            surface.blit(img, (x + w // 2 - img.get_width() // 2,
                                y + h // 2 - img.get_height() // 2))

    def handle_event(self, event: Any) -> bool:
        if not self.visible or not self.enabled or not PYGAME_AVAILABLE:
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.contains(*event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.contains(*event.pos):
                if self.callback:
                    self.callback()
                return True
        return False


# ============================================================================
# Toggle (checkbox-style)
# ============================================================================

class Toggle(Widget):
    """
    A toggle button showing a color swatch, label, and current value.

    Args:
        x, y, w, h: Position and size.
        label: Channel name.
        color: Channel color swatch.
        initial_state: Whether toggle is on (True) or off (False).
        callback: Called with new state on change.
    """

    def __init__(
        self,
        x: int, y: int, w: int, h: int,
        label: str,
        color: tuple[int, int, int] = (100, 180, 100),
        initial_state: bool = True,
        callback: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(x, y, w, h)
        self.label = label
        self.color = color
        self.state = initial_state
        self.callback = callback
        self.current_value: str = ""  # value at cursor, updated by GUI

    def draw(self, surface: Any, font: Any) -> None:
        if not self.visible or not PYGAME_AVAILABLE:
            return
        x, y, w, h = self.rect
        # Checkbox
        checkbox_size = ui_px(14)
        gap = ui_px(6)
        cb_x = x + ui_px(4)
        cb_y = y + h // 2 - checkbox_size // 2
        pygame.draw.rect(surface, (50, 60, 80), (cb_x, cb_y, checkbox_size, checkbox_size))
        pygame.draw.rect(surface, (90, 100, 120), (cb_x, cb_y, checkbox_size, checkbox_size), 1)
        if self.state:
            # Checkmark
            inset = ui_px(2)
            pygame.draw.rect(surface, self.color,
                             (cb_x + inset, cb_y + inset,
                              checkbox_size - 2 * inset, checkbox_size - 2 * inset))
        # Color swatch
        swatch_x = cb_x + checkbox_size + gap
        draw_color_swatch(surface, swatch_x, cb_y, self.color, checkbox_size)
        # Label
        if font is not None:
            label_color = (220, 220, 230) if self.state else (100, 100, 120)
            img = font.render(self.label, True, label_color)
            surface.blit(img, (swatch_x + checkbox_size + gap, y + h // 2 - img.get_height() // 2))
            if self.current_value:
                val_img = font.render(self.current_value, True, (180, 180, 190))
                surface.blit(val_img, (x + w - val_img.get_width() - gap,
                                       y + h // 2 - val_img.get_height() // 2))

    def handle_event(self, event: Any) -> bool:
        if not self.visible or not PYGAME_AVAILABLE:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.contains(*event.pos):
                self.state = not self.state
                if self.callback:
                    self.callback(self.state)
                return True
        return False


# ============================================================================
# Slider
# ============================================================================

class Slider(Widget):
    """
    A horizontal slider with a label and numeric display.

    Args:
        x, y, w, h: Position and size.
        label: Slider label.
        min_val, max_val: Range.
        initial: Initial value.
        step: Value snap step (0 = continuous).
        fmt: Format string for display.
        callback: Called with new value on change.
    """

    def __init__(
        self,
        x: int, y: int, w: int, h: int,
        label: str,
        min_val: float,
        max_val: float,
        initial: float,
        step: float = 0.0,
        fmt: str = "{:.1f}",
        callback: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(x, y, w, h)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.step = step
        self.fmt = fmt
        self.callback = callback
        self._dragging = False
        # Reserve a label area on the left and a value-readout area on the
        # right, both scaled so the slider stays readable at any UI_SCALE.
        self._label_w = ui_px(80)
        self._value_w = ui_px(40)
        self._track_x = x + self._label_w
        self._track_w = w - self._label_w - self._value_w

    def set_position(self, x: int, y: int, w: int) -> None:
        """Reposition the slider in place. Keeps the existing height and
        recomputes the internal track geometry so the handle and value
        readout stay in their reserved gutters."""
        _, _, _, h = self.rect
        self.rect = (x, y, w, h)
        self._track_x = x + self._label_w
        self._track_w = w - self._label_w - self._value_w

    def _val_to_pos(self) -> int:
        frac = (self.value - self.min_val) / max(self.max_val - self.min_val, 1e-9)
        return int(self._track_x + frac * self._track_w)

    def _pos_to_val(self, px: int) -> float:
        frac = (px - self._track_x) / max(self._track_w, 1e-9)
        frac = max(0.0, min(1.0, frac))
        val = self.min_val + frac * (self.max_val - self.min_val)
        if self.step > 0:
            val = round(val / self.step) * self.step
        return max(self.min_val, min(self.max_val, val))

    def draw(self, surface: Any, font: Any) -> None:
        if not self.visible or not PYGAME_AVAILABLE:
            return
        x, y, w, h = self.rect
        cy = y + h // 2

        # Label
        if font is not None:
            img = font.render(self.label, True, (180, 180, 190))
            surface.blit(img, (x + 4, cy - img.get_height() // 2))

        # Track
        track_h = ui_px(6)
        track_radius = ui_px(3)
        handle_r = ui_px(7)
        track_y = cy - track_h // 2
        pygame.draw.rect(surface, (50, 55, 70),
                         (self._track_x, track_y, self._track_w, track_h),
                         border_radius=track_radius)

        # Fill
        handle_pos = self._val_to_pos()
        fill_w = handle_pos - self._track_x
        if fill_w > 0:
            pygame.draw.rect(surface, (70, 120, 200),
                             (self._track_x, track_y, fill_w, track_h),
                             border_radius=track_radius)

        # Handle
        pygame.draw.circle(surface, (100, 150, 230), (handle_pos, cy), handle_r)

        # Value
        if font is not None:
            val_str = self.fmt.format(self.value)
            val_img = font.render(val_str, True, (220, 220, 230))
            surface.blit(val_img, (x + w - val_img.get_width() - 4, cy - val_img.get_height() // 2))

    def handle_event(self, event: Any) -> bool:
        if not self.visible or not self.enabled or not PYGAME_AVAILABLE:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.contains(*event.pos):
                self._dragging = True
                self.value = self._pos_to_val(event.pos[0])
                if self.callback:
                    self.callback(self.value)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False
        elif event.type == pygame.MOUSEMOTION and self._dragging:
            self.value = self._pos_to_val(event.pos[0])
            if self.callback:
                self.callback(self.value)
            return True
        return False


# ============================================================================
# Modal sub-windows
# ============================================================================
#
# A ``ModalWindow`` is a centered floating panel with a title bar, a close
# button, a dimmed backdrop, and exclusive event capture while visible.
# The Meal Builder, Bolus Builder, and Help panel are all ModalWindows;
# the host calls ``draw(surface, font, font_large)`` once per frame after
# all other UI, and routes events through ``handle_event(event)`` before
# any background widgets so the modal can swallow them.

# Visual constants (kept here rather than in gui.py so this module
# doesn't reach back into the host for styling).
MODAL_BG = (34, 36, 50)
MODAL_BORDER = (70, 76, 100)
MODAL_TITLE_BG = (48, 52, 72)
MODAL_TITLE_FG = (235, 238, 250)
MODAL_BODY_FG = (210, 214, 226)
MODAL_BODY_DIM = (150, 156, 174)
MODAL_BACKDROP = (0, 0, 0, 150)
MODAL_CLOSE_HOVER = (180, 80, 90)


class ModalWindow:
    """Centered floating panel with a title bar and close button.

    Subclasses override ``_draw_body`` and ``_handle_body`` to render and
    handle events inside the body rect. The body rect is in *screen*
    coordinates (i.e. shifted by the modal's current top-left), so
    widgets inside should be repositioned in ``_layout_body`` each frame
    — the host can resize the window between frames.

    While ``visible`` is True, ``handle_event`` returns True for any
    mouse/keyboard event the host passes in, so background widgets stop
    receiving input.
    """

    TITLE_BAR_H = ui_px(34)
    PAD = ui_px(14)

    def __init__(self, title: str, w: int, h: int) -> None:
        self.title = title
        self.w = w
        self.h = h
        self.x = 0
        self.y = 0
        self.visible = False
        self._close_rect = (0, 0, 0, 0)
        self._close_hovered = False

    def toggle(self) -> None:
        self.visible = not self.visible

    def _layout(self, screen_w: int, screen_h: int) -> None:
        """Re-center on the current screen and reflow child widgets."""
        self.x = max(0, (screen_w - self.w) // 2)
        self.y = max(0, (screen_h - self.h) // 2)
        self._close_rect = (
            self.x + self.w - ui_px(28), self.y + ui_px(7),
            ui_px(20), ui_px(20),
        )
        self._layout_body()

    def _layout_body(self) -> None:
        """Hook for subclasses to reposition body widgets after a
        re-layout. Default is no-op."""

    def body_rect(self) -> tuple[int, int, int, int]:
        return (
            self.x + self.PAD,
            self.y + self.TITLE_BAR_H + self.PAD,
            self.w - 2 * self.PAD,
            self.h - self.TITLE_BAR_H - 2 * self.PAD,
        )

    def draw(self, surface: Any, font: Any, font_large: Any) -> None:
        if not self.visible or not PYGAME_AVAILABLE:
            return
        sw, sh = surface.get_size()
        self._layout(sw, sh)

        backdrop = pygame.Surface((sw, sh), pygame.SRCALPHA)
        backdrop.fill(MODAL_BACKDROP)
        surface.blit(backdrop, (0, 0))

        radius = ui_px(8)
        pygame.draw.rect(surface, MODAL_BG,
                         (self.x, self.y, self.w, self.h),
                         border_radius=radius)
        pygame.draw.rect(surface, MODAL_TITLE_BG,
                         (self.x, self.y, self.w, self.TITLE_BAR_H),
                         border_top_left_radius=radius,
                         border_top_right_radius=radius)
        pygame.draw.rect(surface, MODAL_BORDER,
                         (self.x, self.y, self.w, self.h), 1,
                         border_radius=radius)

        if font_large is not None:
            img = font_large.render(self.title, True, MODAL_TITLE_FG)
            surface.blit(
                img,
                (self.x + ui_px(14),
                 self.y + (self.TITLE_BAR_H - img.get_height()) // 2),
            )

        cx, cy, cw, ch = self._close_rect
        bg = MODAL_CLOSE_HOVER if self._close_hovered else MODAL_TITLE_BG
        pygame.draw.rect(surface, bg, (cx, cy, cw, ch),
                         border_radius=ui_px(3))
        pad = ui_px(5)
        pygame.draw.line(surface, MODAL_TITLE_FG,
                         (cx + pad, cy + pad),
                         (cx + cw - pad, cy + ch - pad), 2)
        pygame.draw.line(surface, MODAL_TITLE_FG,
                         (cx + cw - pad, cy + pad),
                         (cx + pad, cy + ch - pad), 2)

        self._draw_body(surface, font, font_large)

    def _draw_body(self, surface: Any, font: Any, font_large: Any) -> None:
        pass

    def handle_event(self, event: Any) -> bool:
        """Modal-aware event filter.

        While the modal is visible we *swallow* mouse buttons + wheel
        (so the chart, sidebar and background widgets don't react to
        clicks/scrolls landing on the dimmed backdrop) and Esc (close).
        Mouse motion and key presses pass through — motion is needed so
        the host can keep tracking the cursor, and pass-through keys
        let M / B / H still toggle the modal off and other shortcuts
        keep working."""
        if not self.visible or not PYGAME_AVAILABLE:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.visible = False
                return True
            return False

        if event.type == pygame.MOUSEMOTION:
            cx, cy, cw, ch = self._close_rect
            self._close_hovered = (
                cx <= event.pos[0] < cx + cw and cy <= event.pos[1] < cy + ch
            )
            self._handle_body(event)
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            cx, cy, cw, ch = self._close_rect
            if cx <= event.pos[0] < cx + cw and cy <= event.pos[1] < cy + ch:
                self.visible = False
                self._close_hovered = False
                return True

        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                          pygame.MOUSEWHEEL):
            self._handle_body(event)
            return True

        return False

    def _handle_body(self, event: Any) -> None:
        pass


# ============================================================================
# Help window
# ============================================================================

# Section title + list of body lines. Bodies are rendered with the small
# font; titles with the body font. Edit here to update the in-app guide.
HELP_SECTIONS: list[tuple[str, list[str]]] = [
    ("Overview", [
        "T1DMAI predicts blood-glucose trajectories from recent context.",
        "The chart shows the model's input window (highlighted blue) and",
        "the prediction zone (darker, right of NOW). The model only sees",
        "the most recent 24 h before NOW — older history is for reference.",
    ]),
    ("Prediction", [
        "SPACE / Enter — single-pass 2 h prediction from the current context.",
        "L / Shift+SPACE — long prediction: rolls out to cover the doses you",
        "                  painted (floor 8 h, cap 12 h), conditioned on them.",
        "W              — toggle What-If mode.",
        "F              — roll the prediction forward one window.",
        "G              — advance the simulator forward (truth only).",
        "V              — evaluate model vs. simulator.",
    ]),
    ("Editing inputs", [
        "E — toggle the Curve Editor (raised-cosine bells, draggable points).",
        "P — toggle the Pencil: drag in the prediction zone to draw a carb,",
        "    insulin or exercise curve freehand; the stroke is smoothed.",
        "    Draw past the 2 h mark to plan doses far ahead, then press L.",
        "Tab / Shift+Tab — cycle the channel being edited (carbs / insulin /",
        "                  exercise, the last in g/step carb-equivalent).",
        "C — clear all manual curves and pencil strokes.",
        "Del — remove the selected curve point.",
        "Ctrl+Z — undo the last curve / pencil edit.",
    ]),
    ("Channels", [
        "1–4     — toggle channel visibility (BG/Carbs/Insulin/Exercise).",
        "A       — toggle all channels at once.",
    ]),
    ("View", [
        "Mouse wheel  — zoom around the cursor (or scroll the sidebar).",
        "+ / -        — zoom in / out.",
        "Middle / Right drag — pan the time axis.",
        "R            — reset the view.",
        "S            — screenshot the chart.",
    ]),
    ("Session", [
        "N — generate a new patient.",
        "H — toggle this help window.",
        "Esc — close any open modal.",
        "Q — quit.",
    ]),
]


class HelpWindow(ModalWindow):
    """Scrollable in-app usage guide."""

    LINE_PAD = ui_px(4)
    SECTION_GAP = ui_px(12)

    def __init__(self) -> None:
        super().__init__("Help", ui_px(620), ui_px(560))
        self.scroll = 0
        self._content_h = 0

    def _draw_body(self, surface: Any, font: Any, font_large: Any) -> None:
        bx, by, bw, bh = self.body_rect()
        prev_clip = surface.get_clip()
        surface.set_clip(pygame.Rect(bx, by, bw, bh))

        line_h = (font.get_height() if font else ui_px(16)) + self.LINE_PAD
        head_h = (font_large.get_height() if font_large else line_h) + self.LINE_PAD

        y = by - self.scroll
        for i, (title, lines) in enumerate(HELP_SECTIONS):
            if i > 0:
                y += self.SECTION_GAP
            if font_large is not None:
                img = font_large.render(title, True, MODAL_TITLE_FG)
                surface.blit(img, (bx, y))
            y += head_h
            if font is not None:
                for line in lines:
                    img = font.render(line, True, MODAL_BODY_FG)
                    surface.blit(img, (bx + ui_px(8), y))
                    y += line_h

        content_end = y + self.scroll
        self._content_h = content_end - by

        surface.set_clip(prev_clip)

        # Scrollbar indicator on the right edge of the body.
        if self._content_h > bh:
            track_w = ui_px(5)
            track_x = bx + bw - track_w
            pygame.draw.rect(surface, (50, 54, 72),
                             (track_x, by, track_w, bh),
                             border_radius=track_w // 2)
            thumb_h = max(ui_px(24), int(bh * bh / self._content_h))
            max_scroll = self._content_h - bh
            frac = self.scroll / max_scroll if max_scroll > 0 else 0
            thumb_y = by + int(frac * (bh - thumb_h))
            pygame.draw.rect(surface, (120, 126, 156),
                             (track_x, thumb_y, track_w, thumb_h),
                             border_radius=track_w // 2)

    def _handle_body(self, event: Any) -> None:
        if event.type == pygame.MOUSEWHEEL:
            _, _, _, bh = self.body_rect()
            max_scroll = max(0, self._content_h - bh)
            self.scroll = max(
                0, min(self.scroll - event.y * ui_px(40), max_scroll)
            )


# ============================================================================
# Event editor modal
# ============================================================================

# Slider spec for each high-level event kind. The first slider is always
# the time offset; the remaining sliders are kind-specific. Each tuple
# has the form (attr_name, label, lo, hi, default, step, fmt) — attr_name
# is the field on gui_state.Event to write back on Save.
EVENT_SLIDER_SPECS: dict[str, list[tuple]] = {
    'juice': [
        ('time_offset_min', "Time (min)", -30, 360,  0, 5, "{:+.0f}m"),
        ('magnitude',       "Carbs (g)",    1, 100, 20, 1, "{:.0f}g"),
    ],
    'fast_insulin': [
        ('time_offset_min', "Time (min)", -30, 360, 0, 5, "{:+.0f}m"),
        ('magnitude',       "Units",        0.5, 20, 4, 0.5, "{:.1f}U"),
    ],
    'basal_insulin': [
        ('time_offset_min', "Time (min)", -60, 360,  0, 15, "{:+.0f}m"),
        ('magnitude',       "Units",         1, 60, 20,  1, "{:.0f}U"),
    ],
    'meal': [
        ('time_offset_min', "Time (min)", -30, 360,  0,  5, "{:+.0f}m"),
        ('magnitude',       "Carbs (g)",    5, 250, 60,  5, "{:.0f}g"),
        ('fast_frac',       "Fast frac",    0,   1, 0.5, 0, "{:.2f}"),
    ],
}

EVENT_KIND_TITLES = {
    'juice':         "Juice",
    'fast_insulin':  "Fast Insulin",
    'basal_insulin': "Basal Insulin",
    'meal':          "Meal",
}


class EventEditorModal(ModalWindow):
    """Modal for creating or editing a high-level Event (juice, insulin,
    meal). Sliders match the kind; Save fires ``save_callback(event)``;
    Delete (only for existing events) fires ``delete_callback()``."""

    BUTTON_H = ui_px(30)
    BUTTON_GAP = ui_px(10)

    def __init__(self) -> None:
        super().__init__("Event", ui_px(420), ui_px(260))
        self.sliders: list[Slider] = []
        self.kind: str = 'juice'
        self.meal_name: str = ''
        self.editing_idx: int = -1
        self.save_callback: Callable[[dict], None] | None = None
        self.delete_callback: Callable[[], None] | None = None
        self._save_rect = (0, 0, 0, 0)
        self._cancel_rect = (0, 0, 0, 0)
        self._delete_rect = (0, 0, 0, 0)
        self._save_hover = False
        self._cancel_hover = False
        self._delete_hover = False

    def open_for(
        self,
        kind: str,
        save_callback: Callable[[dict], None],
        initial: dict | None = None,
        editing_idx: int = -1,
        delete_callback: Callable[[], None] | None = None,
        meal_name: str = '',
    ) -> None:
        specs = EVENT_SLIDER_SPECS.get(kind)
        if specs is None:
            return
        self.kind = kind
        self.meal_name = meal_name
        self.editing_idx = editing_idx
        self.save_callback = save_callback
        self.delete_callback = delete_callback

        n = len(specs)
        slider_h = ui_px(32)
        slider_pitch = ui_px(42)
        body_h = n * slider_pitch + self.BUTTON_H + ui_px(24)
        self.h = self.TITLE_BAR_H + 2 * self.PAD + body_h

        title = EVENT_KIND_TITLES.get(kind, "Event")
        if kind == 'meal' and meal_name:
            title = meal_name.capitalize()
        self.title = title

        self.sliders = []
        for spec in specs:
            attr, label, lo, hi, default, step, fmt = spec
            val = float(initial[attr]) if (initial and attr in initial) else float(default)
            val = max(lo, min(hi, val))
            self.sliders.append(
                Slider(0, 0, self.w - 2 * self.PAD, slider_h,
                       label, lo, hi, val, step=step, fmt=fmt)
            )

        self.visible = True

    def _layout_body(self) -> None:
        x, y, w, h = self.body_rect()
        slider_pitch = ui_px(42)
        for i, slider in enumerate(self.sliders):
            slider.set_position(x, y + i * slider_pitch, w)

        btn_y = y + h - self.BUTTON_H
        btn_w = (w - 2 * self.BUTTON_GAP) // 3 if self.editing_idx >= 0 \
            else (w - self.BUTTON_GAP) // 2

        if self.editing_idx >= 0:
            self._delete_rect = (x, btn_y, btn_w, self.BUTTON_H)
            self._cancel_rect = (x + btn_w + self.BUTTON_GAP, btn_y,
                                 btn_w, self.BUTTON_H)
            self._save_rect = (x + 2 * (btn_w + self.BUTTON_GAP), btn_y,
                               w - 2 * (btn_w + self.BUTTON_GAP), self.BUTTON_H)
        else:
            self._delete_rect = (0, 0, 0, 0)
            self._cancel_rect = (x, btn_y, btn_w, self.BUTTON_H)
            self._save_rect = (x + btn_w + self.BUTTON_GAP, btn_y,
                               w - btn_w - self.BUTTON_GAP, self.BUTTON_H)

    def _draw_button(self, surface: Any, font: Any, rect, label: str,
                     hovered: bool, color: tuple, hover_color: tuple) -> None:
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return
        bg = hover_color if hovered else color
        pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=ui_px(4))
        if font is not None:
            img = font.render(label, True, MODAL_TITLE_FG)
            surface.blit(
                img,
                (x + (w - img.get_width()) // 2,
                 y + (h - img.get_height()) // 2),
            )

    def _draw_body(self, surface: Any, font: Any, font_large: Any) -> None:
        for slider in self.sliders:
            slider.draw(surface, font)

        self._draw_button(surface, font, self._save_rect, "Save",
                          self._save_hover, (70, 130, 70), (90, 160, 90))
        self._draw_button(surface, font, self._cancel_rect, "Cancel",
                          self._cancel_hover, (70, 76, 100), (100, 108, 140))
        if self.editing_idx >= 0:
            self._draw_button(surface, font, self._delete_rect, "Delete",
                              self._delete_hover, (170, 70, 80), (210, 90, 100))

    def _values_dict(self) -> dict:
        specs = EVENT_SLIDER_SPECS[self.kind]
        out = {'kind': self.kind}
        for slider, spec in zip(self.sliders, specs):
            out[spec[0]] = float(slider.value)
        if self.kind == 'meal':
            out['name'] = self.meal_name
        return out

    @staticmethod
    def _hit(rect, mx: int, my: int) -> bool:
        x, y, w, h = rect
        return x <= mx < x + w and y <= my < y + h

    def _handle_body(self, event: Any) -> None:
        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            self._save_hover = self._hit(self._save_rect, mx, my)
            self._cancel_hover = self._hit(self._cancel_rect, mx, my)
            self._delete_hover = self._hit(self._delete_rect, mx, my)
            for slider in self.sliders:
                slider.handle_event(event)
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if self._hit(self._save_rect, mx, my):
                if self.save_callback is not None:
                    self.save_callback(self._values_dict())
                self.visible = False
                return
            if self._hit(self._cancel_rect, mx, my):
                self.visible = False
                return
            if self.editing_idx >= 0 and self._hit(self._delete_rect, mx, my):
                if self.delete_callback is not None:
                    self.delete_callback()
                self.visible = False
                return

        for slider in self.sliders:
            if slider.handle_event(event):
                return
