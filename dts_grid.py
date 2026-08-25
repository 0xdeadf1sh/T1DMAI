"""
Diabetes Technology Society (DTS) Error Grid.

Klonoff DC, Freckmann G, Pleus S, Kovatchev BP, Kerr D, Tse C, Li C, et al.
"The Diabetes Technology Society Error Grid and Trend Accuracy Matrix for
Glucose Monitors." J Diabetes Sci Technol. 2024 Nov;18(6):1346-1361.
doi:10.1177/19322968241275701. PMID 39369312. PMCID PMC11531029.
The grid itself is public domain, so the geometry below carries no attribution
obligation.

Not the Surveillance Error Grid (Klonoff et al. 2014 — the continuous risk
surface this one smooths and zones), not Parkes/Consensus, not Clarke (scored
separately by ``train.py``; the two grids disagree by construction and are both
reported), and not ``cg_ega.py``, which is point AND rate binned by glycemic
region.

THE DEFINITION IS THE RISK FUNCTION, NOT THE VERTEX TABLE. This implements the
closed-form risk function of Supplemental Appendix 2; ``tests/test_dts_grid.py``
pins it against all 16 published Table A1 edge vertices, which makes the table an
ORACLE rather than a second implementation. The two disagree by up to 1.11 mg/dL,
largest on the vertical below-clamp segments where no chord exists at all
(D-lower and E-lower, 1.11 mg/dL each). That is consistent with the paper's own
"after rounding and simplification" of the 2.75 and 2.25 coefficients, Table A1
having been drawn from the unrounded fit — an inference, not something the paper
states. Either way the tolerance belongs to the table.

2.75 for an OVERESTIMATE against 2.25 for an underestimate is the grid's whole
clinical content: the panel judged a falsely high reading, which prompts insulin,
riskier than a falsely low one. It is visible only in LOG space or at an outer
edge. Exponentiating very nearly cancels it at the A edge — zone A runs 0.80074x
to 1.19940x of reference, -19.93% against +19.94%, a symmetric +/-20% band by
coincidence — while the E edge sits at +257% against -79%.

THE 50 mg/dL CLAMP IS A RECONSTRUCTION. The printed formula's third branch
covers only the corner where BOTH values are under 50. The main text says
something stronger: the developing clinicians "did not differentiate between
values <= 50 mg/dL", which is a clamp on EACH argument, and that is what this
module applies before the ratio. It reproduces every one of the 16 published
vertices to <= 1.11 mg/dL, which the unclamped formula cannot do at all — the
flat and vertical border segments of Figure A1 exist only because of it — and it
makes the function total on [0, inf), where the unclamped log is singular at
0 mg/dL against this repository's 10 mg/dL physical floor. It is printed nowhere
in that form, so every zone assignment with either value under 50 mg/dL rests on
a reading of one English sentence, corroborated numerically. A figure from this
module quoted against a published DTS number carries that caveat; the only
available oracle is the Society's own R Shiny tool, whose source is not
published.

APPLYING A MONITOR GRID TO A FORECAST. The grid grades a glucose MONITOR:
reference and monitor are simultaneous, and the zones encode the risk of acting
NOW on a wrong reading of NOW. This module is fed true BG at horizon h against
the prediction for horizon h — a substitution the source neither makes nor
mentions, and the same one ``train.py`` already makes for Clarke. It is reported
as a risk-weighted summary of forecast error, NOT as a clinical accuracy claim
about a device: a forecast error at 30-120 min has different consequences from a
measurement error now, and the 2.75/2.25 asymmetry was elicited for readings, not
forecasts, so whether it points the same way for a prediction is untested and it
is not re-tuned here. PRED-EGA (Sivananthan et al., Diabetes Technol Ther
2011;13(8):787-796) is the prediction-specific grid; nothing here replaces it.

REPORTING. The headline figure is pZA, the percentage in zone A alone; the paper
is explicit that reporting "zone A + zone B" as if both were acceptable is
inappropriate, so this module exposes per-zone shares and no A+B convenience. No
ISO or FDA criterion references this grid, so nothing here carries a pass mark.
The paper's only calibration anchor is an empirical fit across 31 studies,
``MARD = 8 + 0.33 * (96 - pZA)``, a rough conversion and not a target.
"""
from __future__ import annotations

import numpy as np

# The suite's physical BG floor, imported rather than restated; the units
# tripwire below.
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

# Below this the panel treated every glucose as the same value. A clamp on EACH
# argument, reconstructed from the paper's prose — see the module docstring.
DTS_LOW_CLAMP_MGDL: float = 50.0

# The two limbs of the risk function. Overestimates are penalised harder.
DTS_OVERESTIMATE_COEFF: float = 2.75
DTS_UNDERESTIMATE_COEFF: float = 2.25

# |Risk| upper edges of zones A, B, C, D; above the last is E. Ascending, and
# each edge is CLOSED — a point exactly on one takes the lower-risk zone.
DTS_ZONE_EDGES: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)

ZONE_NAMES: tuple[str, ...] = ("a", "b", "c", "d", "e")
assert len(ZONE_NAMES) == len(DTS_ZONE_EDGES) + 1

# Published domain of the grid and of the SEG it derives from: 1-600 mg/dL. Both
# ends only WARN — above 600 the risk function extrapolates cleanly and the paper
# is silent, and 1 mg/dL is far too low for a units check, since Kovatchev risk
# over the legal BG range reaches +3.16 and a whole risk-space array clears it.
DTS_DOMAIN_MIN_MGDL: float = 1.0
DTS_DOMAIN_MAX_MGDL: float = 600.0

# The UNITS TRIPWIRE is the repository's physical BG floor, NOT the grid's domain
# floor: without it the 50 mg/dL clamp takes an array of z-scores or risk values
# to 50 in every cell, scores zero risk and reports a flawless 100% zone A with
# nothing downstream able to tell. Every legal mg/dL glucose clears BG_CLAMP_MIN,
# and the whole realised risk range [-6.82, +3.16] sits below it.
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

    ``y_true`` is the reference (here: true BG at the horizon), ``y_pred`` the
    monitor (here: the prediction), both mg/dL, same shape. Positive where the
    prediction overestimates. Zones take the absolute value; the sign is kept
    because the direction is the clinically interesting half.
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
    # ``side='left'`` is what makes each edge CLOSED: |Risk| of exactly 0.5 stays
    # in A. ``side='right'`` would push every on-border point up a zone.
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
