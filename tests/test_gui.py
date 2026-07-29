"""Tests for GUI modules — coordinate transforms, state management, curve generation.

The GUI has been re-keyed onto the risk-space ``predict`` API
({q_tau, median_bg, bands}); gui.py / gui_state.py no longer reference the
deleted dynamics/trend/physics symbols, so the module imports cleanly.
"""

import numpy as np
import pytest


def test_gui_imports():
    """GUI modules import without error."""
    import gui  # noqa: F401
    import gui_renderer  # noqa: F401
    import gui_controls  # noqa: F401
    import gui_state  # noqa: F401


def test_chart_transform():
    """Screen-to-chart and chart-to-screen coordinate transforms are inverses."""
    from gui_renderer import ChartTransform

    ct = ChartTransform(
        screen_x=100, screen_y=50, screen_w=800, screen_h=400,
        chart_x_min=0, chart_x_max=720, chart_y_min=40, chart_y_max=400,
    )

    # Round-trip: chart → screen → chart
    cx, cy = 360.0, 120.0
    sx, sy = ct.chart_to_screen(cx, cy)
    cx2, cy2 = ct.screen_to_chart(sx, sy)

    print(f"\n[DUMP] chart_transform | chart ({cx:.1f},{cy:.1f}) → screen ({sx:.1f},{sy:.1f}) → chart ({cx2:.4f},{cy2:.4f})")

    assert abs(cx - cx2) < 0.01, f"X round-trip failed: {cx} != {cx2}"
    assert abs(cy - cy2) < 0.01, f"Y round-trip failed: {cy} != {cy2}"


def test_chart_transform_corners():
    """Corner points map correctly."""
    from gui_renderer import ChartTransform

    ct = ChartTransform(
        screen_x=0, screen_y=0, screen_w=100, screen_h=100,
        chart_x_min=0, chart_x_max=100, chart_y_min=0, chart_y_max=100,
    )

    # Chart origin (0, 0) → screen bottom-left
    sx, sy = ct.chart_to_screen(0.0, 0.0)
    assert abs(sx - 0.0) < 0.01
    assert abs(sy - 100.0) < 0.01  # y=0 data → bottom of screen

    # Chart (100, 100) → screen top-right
    sx, sy = ct.chart_to_screen(100.0, 100.0)
    assert abs(sx - 100.0) < 0.01
    assert abs(sy - 0.0) < 0.01


def test_override_state():
    """GUIState correctly tracks channel overrides."""
    from gui_state import GUIState

    state = GUIState()
    assert len(state.overrides) == 0

    # Add carb override
    from config import PREDICTION_PATCHES, PATCH_SIZE
    carb_override = np.ones((PREDICTION_PATCHES, PATCH_SIZE)) * 5.0
    state.set_override(0, carb_override)
    assert 0 in state.overrides
    assert state.overrides[0].shape == (PREDICTION_PATCHES, PATCH_SIZE)

    # Values are correct
    np.testing.assert_allclose(state.overrides[0], 5.0)

    # Clear
    state.clear_overrides()
    assert len(state.overrides) == 0

    print("\n[DUMP] override_state | set and clear verified")


def test_channel_toggles():
    """GUIState toggle methods work correctly."""
    from gui_state import GUIState

    state = GUIState()

    # Initial state: first 3 visible
    assert state.channel_visible[0] is True

    # Toggle individual
    state.toggle_channel(0)
    assert state.channel_visible[0] is False

    state.toggle_channel(0)
    assert state.channel_visible[0] is True

    # Toggle all (all visible → all hidden)
    state.toggle_all_channels()
    assert not any(state.channel_visible)

    # Toggle all again (all hidden → all visible)
    state.toggle_all_channels()
    assert all(state.channel_visible)

    print("\n[DUMP] channel_toggles | all toggle operations verified")
