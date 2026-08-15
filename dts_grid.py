"""
Diabetes Technology Society (DTS) Error Grid.

Klonoff DC, Freckmann G, Pleus S, Kovatchev BP, Kerr D, Tse C, Li C, et al.
"The Diabetes Technology Society Error Grid and Trend Accuracy Matrix for
Glucose Monitors." J Diabetes Sci Technol. 2024 Nov;18(6):1346-1361.
doi:10.1177/19322968241275701. PMID 39369312. PMCID PMC11531029.

The grid itself is in the PUBLIC DOMAIN (the paper says so in as many words), so
the geometry below carries no attribution obligation; the citation is here
because a clinical figure with no source is not checkable.

WHICH GRID THIS IS
------------------
Three grids get conflated under nearby names, and they are not the same object:

* the DTS Error Grid — this one. Five discrete zones A-E on straight-line
  borders, derived by fitting a smooth risk function to the SEG surface.
* the Surveillance Error Grid (SEG), Klonoff et al. 2014 — the CONTINUOUS risk
  surface this grid smooths and zones. Same thresholds, boxy borders.
* Parkes/Consensus and Clarke — neither is DTS. ``train.py`` scores Clarke
  separately; the two grids disagree by construction and are both reported.

``cg_ega.py`` is a fourth thing again: point AND rate, binned by glycemic region.
The DTS paper positions its (unimplemented here) Trend Accuracy Matrix as the
simpler replacement for CG-EGA's rate axis.

THE DEFINITION IS THE RISK FUNCTION, NOT THE VERTEX TABLE
---------------------------------------------------------
The paper publishes both a closed-form risk function (Supplemental Appendix 2)
and a table of border vertices (Table A1) for drawing. This module implements the
FUNCTION. The table's sloped borders are chords whose endpoints are rounded
inconsistently — low end up, high end down — so a polygon built from it has a
slightly different slope than the contour it is meant to trace, and the two
disagree by up to ~1.1 mg/dL. ``tests/test_dts_grid.py`` pins the function
against all 16 published edge vertices at that tolerance, which makes the table
an ORACLE rather than a second implementation.

    Risk = 2.75 * ln(monitor / reference)   when monitor > reference
           2.25 * ln(monitor / reference)   when monitor <= reference

Zones are taken on the ABSOLUTE value, closed above, so a point exactly on a
border falls in the LOWER-risk zone::

    A  |Risk| <= 0.5     no risk
    B  |Risk| <= 1.5     mild
    C  |Risk| <= 2.5     moderate
    D  |Risk| <= 3.5     high
    E  |Risk|  > 3.5     extreme

The asymmetry is deliberate and is the grid's whole clinical content: 2.75 for an
OVERESTIMATE against 2.25 for an underestimate, because the panel judged a
falsely high reading (which prompts insulin) riskier than a falsely low one. It
lives in LOG-ratio space, where the overestimate side of every zone is the
tighter one — at the A edge, ``+0.1818`` against ``-0.2222``.

Do not read that off the ratios, though. Exponentiating very nearly cancels it at
the A edge: zone A runs ``0.80074x`` to ``1.19940x`` of reference, i.e. -19.93%
against +19.94%, so the innermost zone reads as a symmetric +/-20% band by
coincidence rather than by design. The asymmetry becomes visible further out —
the E edge sits at ``+257%`` against ``-79%`` — so a check that the coefficients
are still doing their job has to be made in log space or at an outer edge.

THE 50 mg/dL CLAMP IS A RECONSTRUCTION — read this before trusting a low-BG zone
-------------------------------------------------------------------------------
The printed formula carries a third branch, "0 if monitor < 50 and reference <
50", which covers only the corner where BOTH are low. The main text says
something stronger and more useful: the developing clinicians "did not
differentiate between values <= 50 mg/dL", which "was tantamount to treating all
values <= 50 mg/dL as the same" — that is a clamp on EACH argument, not a special
case for the corner.

This module clamps each argument at ``DTS_LOW_CLAMP_MGDL`` before the ratio. Two
things justify it and one thing does not:

* it reproduces every one of the 16 published edge vertices to <= 1.11 mg/dL,
  which the unclamped formula cannot do at all — the flat and vertical border
  segments of Figure A1 exist only because of it;
* it makes the function total on [0, inf): without it a reference or monitor of
  0 mg/dL is a log singularity, and this repository's physical floor is 10 mg/dL;
* it is NOT printed anywhere in that form. An implementer reading only the
  appendix would get the low-glucose corner wrong in the other direction.

So: zone assignments where either value is under 50 mg/dL rest on a reading of
one English sentence, corroborated numerically. If a figure from this module is
ever quoted against a published DTS number, say so. The only available oracle is
the Society's own R Shiny tool, whose source is not published.

APPLYING A MONITOR GRID TO A FORECAST
-------------------------------------
The DTS grid grades a glucose MONITOR: reference and monitor are simultaneous
measurements, and the zones encode the risk of acting NOW on a wrong reading of
NOW. This module is fed the true BG at horizon h against the PREDICTION for
horizon h, which is a substitution the source neither makes nor sanctions — it
never mentions prediction. It is the same substitution ``train.py`` already makes
for Clarke, and it is reported here for the same reason: as a risk-weighted
summary of forecast error, NOT as a clinical accuracy claim about a device.

Two specific caveats worth carrying:

* a forecast error at 30-120 min has different consequences from a measurement
  error now — there is time to re-measure, and the action taken differs. The
  panel never scored that scenario;
* the 2.75/2.25 asymmetry was elicited for readings, not forecasts, and whether
  it points the same way for a prediction is untested. Do not re-tune it.

A prediction-specific grid exists — PRED-EGA (Sivananthan et al., Diabetes
Technol Ther 2011;13(8):787-796), built precisely because CG-EGA had to be
modified before it could assess predictors. Nothing here replaces it.

REPORTING
---------
The paper is unusually explicit that the headline figure is pZA, the percentage
in zone A alone, and that reporting "zone A + zone B" as if both were acceptable
is inappropriate. This module therefore exposes per-zone shares and no A+B
convenience, and the trainers' tables render all five. No acceptance threshold
exists for this grid — no ISO or FDA criterion references it — so nothing here
carries a pass mark. The paper's only calibration anchor is an empirical fit
across 31 studies, ``MARD = 8 + 0.33 * (96 - pZA)``, which is a rough conversion
and not a target.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "DTS_LOW_CLAMP_MGDL",
    "DTS_OVERESTIMATE_COEFF",
    "DTS_UNDERESTIMATE_COEFF",
    "DTS_ZONE_EDGES",
    "ZONE_NAMES",
    "dts_risk",
    "dts_zones",
    "dts_zone_counts",
    "dts_zone_fractions",
]

# Below this, the panel treated every glucose as the same value; see the module
# docstring for why this is a clamp on each argument and what that rests on.
DTS_LOW_CLAMP_MGDL: float = 50.0

# The two limbs of the risk function. Overestimates are penalised harder.
DTS_OVERESTIMATE_COEFF: float = 2.75
DTS_UNDERESTIMATE_COEFF: float = 2.25

# |Risk| upper edges of zones A, B, C, D; anything above the last is E. Ascending,
# and each is CLOSED (a point exactly on an edge takes the lower-risk zone).
DTS_ZONE_EDGES: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5)

ZONE_NAMES: tuple[str, ...] = ("a", "b", "c", "d", "e")
assert len(ZONE_NAMES) == len(DTS_ZONE_EDGES) + 1

# The published domain of the grid and of the SEG it derives from: 1-600 mg/dL.
# The floor is the UNITS TRIPWIRE — every legal mg/dL glucose clears it, and a
# risk-space or z-space array does not, which matters here more than usual: the
# clamp above would take an array of z-scores to 50 mg/dL in every cell and
# report a flawless 100% zone A. The ceiling only warns, since the function
# extrapolates cleanly above it and the paper simply does not say.
DTS_DOMAIN_MIN_MGDL: float = 1.0
DTS_DOMAIN_MAX_MGDL: float = 600.0


def _as_mgdl(x: np.ndarray, name: str) -> np.ndarray:
    """Validate one side of the pair and return it as float64.

    Args:
        x: array of glucose values, mg/dL.
        name: which side, for the assertion message.

    Returns:
        ``np.float64`` view/copy of ``x``.
    """
    a = np.asarray(x, dtype=np.float64)
    assert np.isfinite(a).all(), f"dts_grid: {name} carries non-finite values"
    assert a.min() >= DTS_DOMAIN_MIN_MGDL, (
        f"dts_grid: {name} has a minimum of {a.min():.4g}, below the grid's "
        f"{DTS_DOMAIN_MIN_MGDL} mg/dL domain floor — this is the units tripwire, "
        "and a risk-space or normalized array is what usually trips it"
    )
    if a.max() > DTS_DOMAIN_MAX_MGDL:
        import warnings
        warnings.warn(
            f"dts_grid: {name} reaches {a.max():.4g} mg/dL, above the grid's "
            f"published {DTS_DOMAIN_MAX_MGDL} mg/dL domain; the risk function "
            "extrapolates but the paper does not define zones there",
            RuntimeWarning, stacklevel=3,
        )
    return a


def dts_risk(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """SIGNED DTS risk of each (reference, monitor) pair.

    Positive where the prediction overestimates, negative where it underestimates.
    Zones are taken on the absolute value; the sign is kept because the direction
    is the clinically interesting half and a caller may want it.

    Args:
        y_true: reference glucose (mg/dL), any shape. Here: true BG at the horizon.
        y_pred: monitor glucose (mg/dL), same shape. Here: the prediction.

    Returns:
        float64 array of the same shape.
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
    """Zone INDEX of each pair: 0 = A, 1 = B, 2 = C, 3 = D, 4 = E.

    Indices rather than letters so a caller can slice a horizon out and count,
    and so the ordering is the risk ordering.

    Args:
        y_true: reference glucose (mg/dL). y_pred: monitor glucose (mg/dL).

    Returns:
        int8 array of the same shape, indexing ``ZONE_NAMES``.
    """
    risk = np.abs(dts_risk(y_true, y_pred))
    # ``side='left'`` is what makes each edge CLOSED: |Risk| exactly 0.5 sorts
    # before the 0.5 edge and stays in A. ``side='right'`` would push every
    # on-border point up a zone.
    return np.searchsorted(np.asarray(DTS_ZONE_EDGES), risk, side="left").astype(np.int8)


def dts_zone_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Per-zone point counts plus the total, for accumulation across batches.

    Counts rather than shares because a validation pass sums these over batches
    of unequal size and divides once at the end; averaging shares would weight
    a small batch equally with a large one.

    Args:
        y_true: reference glucose (mg/dL). y_pred: monitor glucose (mg/dL).

    Returns:
        ``{'a'..'e': count, 'total': count}`` as floats.
    """
    zones = dts_zones(y_true, y_pred)
    out = {name: float((zones == i).sum()) for i, name in enumerate(ZONE_NAMES)}
    out["total"] = float(zones.size)
    return out


def dts_zone_fractions(counts: dict[str, float]) -> dict[str, float | None]:
    """Per-zone shares in [0, 1] from accumulated counts.

    Args:
        counts: from :func:`dts_zone_counts`, or the same keys summed over batches.

    Returns:
        ``{'a'..'e': fraction}``; every value is None when the total is zero — a
        zone share over no points is not 0%, it is unmeasured.
    """
    total = float(counts.get("total", 0.0))
    if total <= 0.0:
        return {name: None for name in ZONE_NAMES}
    return {name: float(counts.get(name, 0.0)) / total for name in ZONE_NAMES}
