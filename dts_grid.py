"""DTS Error Grid (Klonoff et al. 2024, J Diabetes Sci Technol 18(6):1346-1361; public domain).
Risk function (not the vertex table) is authoritative; tests/test_dts_grid.py pins it to all 16
Table A1 vertices within 1.11 mg/dL. The 50 mg/dL per-argument clamp is a RECONSTRUCTION from
prose. Fed true-vs-predicted BG at one horizon, not simultaneous monitor readings."""
from __future__ import annotations

import numpy as np

# the suite's physical BG floor, imported rather than restated; feeds the units tripwire below
from T1DMSIM.simulator import BG_CLAMP_MIN

__all__ = [
    "DTS_LOW_CLAMP_MGDL",
    "DTS_UNITS_FLOOR_MGDL",
    "DTS_OVERESTIMATE_COEFF",
    "DTS_UNDERESTIMATE_COEFF",
    "DTS_ZONE_EDGES",
    "ZONE_NAMES",
    "dts_risk",
    "dts_zones",
    "dts_zone_counts",
    "dts_zone_fractions",
]

# below this the panel treated every glucose as one value; a RECONSTRUCTION, see module docstring
DTS_LOW_CLAMP_MGDL: float = 50.0

# The two limbs of the risk function. Overestimates are penalised harder.
DTS_OVERESTIMATE_COEFF: float = 2.75
DTS_UNDERESTIMATE_COEFF: float = 2.25

# |Risk| upper edges of zones A-D, ascending; above the last is E; each edge is CLOSED (lower zone)
DTS_ZONE_EDGES: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)

ZONE_NAMES: tuple[str, ...] = ("a", "b", "c", "d", "e")
assert len(ZONE_NAMES) == len(DTS_ZONE_EDGES) + 1

# published domain 1-600 mg/dL; both ends WARN only, extrapolation is undefined past it
DTS_DOMAIN_MIN_MGDL: float = 1.0
DTS_DOMAIN_MAX_MGDL: float = 600.0

# UNITS TRIPWIRE, not the grid's domain floor: without it risk/z arrays clamp to 50, faking zone A
DTS_UNITS_FLOOR_MGDL: float = BG_CLAMP_MIN


def _as_mgdl(x: np.ndarray, name: str) -> np.ndarray:
    """Validate one side of the pair — glucose in mg/dL — and return it as float64."""
    a = np.asarray(x, dtype=np.float64)
    assert np.isfinite(a).all(), f"dts_grid: {name} carries non-finite values"
    assert a.min() >= DTS_UNITS_FLOOR_MGDL, (
        f"dts_grid: {name} has a minimum of {a.min():.4g}, below the physical BG "
        f"floor of {DTS_UNITS_FLOOR_MGDL} mg/dL — this is the units tripwire, and "
        "a risk-space or normalized array is what usually trips it"
    )
    import warnings
    if a.min() < DTS_DOMAIN_MIN_MGDL or a.max() > DTS_DOMAIN_MAX_MGDL:
        warnings.warn(
            f"dts_grid: {name} spans [{a.min():.4g}, {a.max():.4g}] mg/dL, outside "
            f"the grid's published [{DTS_DOMAIN_MIN_MGDL}, {DTS_DOMAIN_MAX_MGDL}] "
            "domain; the risk function extrapolates but the paper does not define "
            "zones there",
            RuntimeWarning, stacklevel=3,
        )
    return a


def dts_risk(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """SIGNED DTS risk of each (reference, monitor) pair — float64, input shape.

    y_true is the reference (true BG here), y_pred the monitor (the prediction), mg/dL, same
    shape. Positive where prediction overestimates; zones take |value|, sign kept for direction.
    """
    ref = _as_mgdl(y_true, "y_true")
    mon = _as_mgdl(y_pred, "y_pred")
    assert ref.shape == mon.shape, (
        f"dts_grid: shape mismatch, y_true {ref.shape} vs y_pred {mon.shape}")
    r = np.maximum(ref, DTS_LOW_CLAMP_MGDL)
    m = np.maximum(mon, DTS_LOW_CLAMP_MGDL)
    coeff = np.where(m > r, DTS_OVERESTIMATE_COEFF, DTS_UNDERESTIMATE_COEFF)
    return coeff * np.log(m / r)


def dts_zones(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Zone INDEX of each pair: 0 = A .. 4 = E — the risk ordering, indexing ``ZONE_NAMES``.

    ``y_true`` reference mg/dL, ``y_pred`` monitor mg/dL; int8, input shape.
    """
    risk = np.abs(dts_risk(y_true, y_pred))
    # side='left' makes each edge CLOSED: |Risk| of exactly 0.5 stays in A, not bumped up a zone
    return np.searchsorted(np.asarray(DTS_ZONE_EDGES), risk, side="left").astype(np.int8)


def dts_zone_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """``{'a'..'e': count, 'total': count}`` as floats, both arguments mg/dL.

    Counts rather than shares: a validation pass sums these over batches of
    unequal size and divides once at the end.
    """
    zones = dts_zones(y_true, y_pred)
    out = {name: float((zones == i).sum()) for i, name in enumerate(ZONE_NAMES)}
    out["total"] = float(zones.size)
    return out


def dts_zone_fractions(counts: dict[str, float]) -> dict[str, float | None]:
    """``{'a'..'e': fraction}`` in [0, 1] from :func:`dts_zone_counts`, summed or not.

    Every value is None at zero total — a zone share over no points is unmeasured,
    not 0%.
    """
    total = float(counts.get("total", 0.0))
    if total <= 0.0:
        return {name: None for name in ZONE_NAMES}
    return {name: float(counts.get(name, 0.0)) / total for name in ZONE_NAMES}
