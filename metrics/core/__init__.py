"""Shared evaluation core: the canonical Segment, the model-input bridge, the
window collector, the horizon map and the comparison metric suite.

Everything here is source-agnostic. ``metrics/sim/`` builds Segments and Windows
from fresh T1DMSIM patients; ``train.py`` and ``calibrate_conformal.py`` reach in
for the window and metric definitions so training, calibration and the offline
report score the same quantities the same way.
"""
from __future__ import annotations

from .schema import Segment, GRID_MIN, MGDL_PER_MMOL

__all__ = ['Segment', 'GRID_MIN', 'MGDL_PER_MMOL']
