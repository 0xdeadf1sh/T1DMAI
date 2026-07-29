"""
Single source of truth for the real-data horizon → step-index map.

The forecast is laid out on the canonical 5-minute CGM grid; a horizon of
``h_min`` minutes lands on the 0-based patch-end step ``h_min // 5 - 1`` (e.g.
the 30-min value is step 5, 60-min is step 11, 120-min is step 23). This was
duplicated as a literal ``{30: 5, 60: 11, 120: 23}`` across the real-data
modules and hardcoded to the 2 h / 5-min geometry; centralizing it here makes
the index derive from ``config`` at runtime so a non-default
``PREDICTION_HORIZON_HOURS`` no longer reads past the array (short horizon) or
silently mis-indexes the last step (long horizon).

The reported SET of horizons is unchanged (``HORIZONS``); only the index math
and a coverage assertion live here.
"""
from __future__ import annotations

import config

# Canonical CGM cadence in minutes. The forecast array is sampled on this grid
# regardless of PATCH_SIZE; one patch spans ``PATCH_SIZE`` of these steps.
GRID_MIN = 5

# Total predicted steps available in a single forward pass.
PRED_STEPS = config.PREDICTION_PATCHES * config.PATCH_SIZE

# The horizons (minutes) actually reported by the real-data suite. Unchanged.
HORIZONS = (30, 60, 120)


def horizon_step_index(h_min: int) -> int:
    """0-based patch-end step index for a horizon of ``h_min`` minutes.

    ``h_min // GRID_MIN - 1`` — the last step inside the ``h_min``-minute prefix
    of the 5-min forecast grid (30→5, 60→11, 120→23 at the default geometry).
    """
    return h_min // GRID_MIN - 1


# Precomputed map for the reported horizons, derived (not literal). At the
# default 2 h horizon this equals the legacy ``{30: 5, 60: 11, 120: 23}``.
HORIZON_IDX = {h: horizon_step_index(h) for h in HORIZONS}

# Coverage: every reported horizon must fall inside the single-pass prediction
# array. At PREDICTION_HORIZON_HOURS < 2 the 120-min slot (step 23) would read
# past PRED_STEPS; this turns that silent OOB into a loud failure at import.
_max_idx = max(HORIZON_IDX.values())
assert _max_idx < PRED_STEPS, (
    f"prediction length PRED_STEPS={PRED_STEPS} does not cover the largest "
    f"requested horizon {max(HORIZONS)} min (step index {_max_idx}); "
    f"PREDICTION_HORIZON_HOURS is too small for HORIZONS={HORIZONS}"
)

# --------------------------------------------------------------------------- #
# Figure-only hourly horizon grid.
# --------------------------------------------------------------------------- #
# The comparison FIGURES (rmse_vs_horizon / parity / clarke) report hour-by-hour
# out to the night long horizon, sourced from a ROLLED forecast. The reported
# metric SUITE — and the README tables, conformal intervals and selected offsets
# that derive from it — stays on the canonical ``HORIZONS`` above; this grid is
# purely a plotting axis. 30 min is kept as the near-term clinical point (and the
# published-peer anchor), then one point per hour up to NIGHT_LONG_HORIZON_HOURS.
# When no rolling is configured (NIGHT_LONG_HORIZON_HOURS == PREDICTION_HORIZON_HOURS)
# this collapses to the horizons a single forward pass already covers.
_NIGHT_LONG_MIN = int(round(config.NIGHT_LONG_HORIZON_HOURS * 60))
FIGURE_HORIZONS = (30,) + tuple(range(60, _NIGHT_LONG_MIN + 1, 60))
FIGURE_HORIZON_IDX = {h: horizon_step_index(h) for h in FIGURE_HORIZONS}

# Rolled-forecast length (steps) needed to cover the largest figure horizon.
FIGURE_PRED_STEPS = max(FIGURE_HORIZON_IDX.values()) + 1
