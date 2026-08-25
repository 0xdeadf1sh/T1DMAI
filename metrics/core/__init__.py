"""Source-agnostic evaluation core: Segment, feature bridge, window collector, horizons, suite.

Training, calibration and the offline report score the same quantities through these definitions.
"""
from __future__ import annotations

from .schema import Segment, GRID_MIN, MGDL_PER_MMOL

__all__ = ['Segment', 'GRID_MIN', 'MGDL_PER_MMOL']
