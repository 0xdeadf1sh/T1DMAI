"""Horizon → step-index map, single copy.

5-min CGM grid: ``h_min`` lands on the 0-based patch-end step ``h_min // 5 - 1``.
Derived from ``config`` at runtime, so a non-default ``PREDICTION_HORIZON_HOURS`` cannot mis-index.
"""
from __future__ import annotations

import config

# minutes; the forecast grid regardless of PATCH_SIZE, one patch = PATCH_SIZE steps
GRID_MIN = 5

# steps, single forward pass
PRED_STEPS = config.PREDICTION_PATCHES * config.PATCH_SIZE

# minutes
HORIZONS = (30, 60, 120)


def horizon_step_index(h_min: int) -> int:
    """0-based patch-end step index; 30→5, 60→11, 120→23 at the default geometry."""
    return h_min // GRID_MIN - 1


HORIZON_IDX = {h: horizon_step_index(h) for h in HORIZONS}

# loud at import: under PREDICTION_HORIZON_HOURS < 2 the 120-min slot reads past PRED_STEPS
_max_idx = max(HORIZON_IDX.values())
assert _max_idx < PRED_STEPS, (
    f"prediction length PRED_STEPS={PRED_STEPS} does not cover the largest "
    f"requested horizon {max(HORIZONS)} min (step index {_max_idx}); "
    f"PREDICTION_HORIZON_HOURS is too small for HORIZONS={HORIZONS}"
)

# Plotting axis only (rolled forecast): 30 min anchor, then hourly to NIGHT_LONG_HORIZON_HOURS.
_NIGHT_LONG_MIN = int(round(config.NIGHT_LONG_HORIZON_HOURS * 60))
FIGURE_HORIZONS = (30,) + tuple(range(60, _NIGHT_LONG_MIN + 1, 60))
FIGURE_HORIZON_IDX = {h: horizon_step_index(h) for h in FIGURE_HORIZONS}

# steps of rolled forecast, largest figure horizon
FIGURE_PRED_STEPS = max(FIGURE_HORIZON_IDX.values()) + 1
