import numpy as np
import pytest


def test_gui_imports():
    import gui  # noqa: F401
    import gui_renderer  # noqa: F401
    import gui_controls  # noqa: F401
    import gui_state  # noqa: F401


def test_chart_transform():
    from gui_renderer import ChartTransform

    ct = ChartTransform(
        screen_x=100, screen_y=50, screen_w=800, screen_h=400,
        chart_x_min=0, chart_x_max=720, chart_y_min=40, chart_y_max=400,
    )

    cx, cy = 360.0, 120.0
    sx, sy = ct.chart_to_screen(cx, cy)
    cx2, cy2 = ct.screen_to_chart(sx, sy)

    print(f"\n[DUMP] chart_transform | chart ({cx:.1f},{cy:.1f}) → screen ({sx:.1f},{sy:.1f}) → chart ({cx2:.4f},{cy2:.4f})")

    assert abs(cx - cx2) < 0.01, f"X round-trip failed: {cx} != {cx2}"
    assert abs(cy - cy2) < 0.01, f"Y round-trip failed: {cy} != {cy2}"


def test_chart_transform_corners():
    from gui_renderer import ChartTransform

    ct = ChartTransform(
        screen_x=0, screen_y=0, screen_w=100, screen_h=100,
        chart_x_min=0, chart_x_max=100, chart_y_min=0, chart_y_max=100,
    )

    sx, sy = ct.chart_to_screen(0.0, 0.0)
    assert abs(sx - 0.0) < 0.01
    assert abs(sy - 100.0) < 0.01  # y axis inverted: chart 0 is screen bottom
    sx, sy = ct.chart_to_screen(100.0, 100.0)
    assert abs(sx - 100.0) < 0.01
    assert abs(sy - 0.0) < 0.01


def test_override_state():
    from gui_state import GUIState

    state = GUIState()
    assert len(state.overrides) == 0

    from config import PREDICTION_PATCHES, PATCH_SIZE
    carb_override = np.ones((PREDICTION_PATCHES, PATCH_SIZE)) * 5.0
    state.set_override(0, carb_override)
    assert 0 in state.overrides
    assert state.overrides[0].shape == (PREDICTION_PATCHES, PATCH_SIZE)

    np.testing.assert_allclose(state.overrides[0], 5.0)

    state.clear_overrides()
    assert len(state.overrides) == 0

    print("\n[DUMP] override_state | set and clear verified")


def test_channel_toggles():
    from gui_state import GUIState

    state = GUIState()

    assert state.channel_visible[0] is True

    state.toggle_channel(0)
    assert state.channel_visible[0] is False

    state.toggle_channel(0)
    assert state.channel_visible[0] is True

    state.toggle_all_channels()
    assert not any(state.channel_visible)

    state.toggle_all_channels()
    assert all(state.channel_visible)

    print("\n[DUMP] channel_toggles | all toggle operations verified")
