"""The DTS Error Grid, pinned against the coordinates its paper publishes.

``dts_grid`` implements the closed-form risk function of Klonoff et al. 2024
(J Diabetes Sci Technol 18(6):1346-1361). The paper ALSO publishes a table of
border vertices for drawing the grid, and that table is an independent statement
of the same geometry — so it is the oracle here. Every one of the 16 edge
vertices of Table A1 is checked against the function.

They agree to 1.11 mg/dL, not exactly. The gap is largest on the VERTICAL
below-clamp segments, where the border is not a chord of anything, so it is not a
drawing artefact; it is consistent with the paper's coefficients being rounded
("after rounding and simplification", Appendix 2) while Table A1 was drawn from
the unrounded fit. Either way it is a property of the table rather than of the
implementation. ``VERTEX_TOL_MGDL`` is set just above the worst observed gap and
is deliberately tight enough that a wrong coefficient, a dropped clamp or a
transposed axis fails it — each of those moves a vertex by tens of mg/dL.

The 50 mg/dL clamp is the one part of ``dts_grid`` that is a reading of the
paper's prose rather than of its formula (see that module's docstring). It is
also the part these vertices constrain hardest: eight of the sixteen are the flat
and vertical border segments that exist ONLY under the clamp.
"""

import numpy as np
import pytest

from dts_grid import (
    DTS_LOW_CLAMP_MGDL, DTS_OVERESTIMATE_COEFF, DTS_UNDERESTIMATE_COEFF,
    DTS_ZONE_EDGES, ZONE_NAMES, dts_risk, dts_zone_counts, dts_zone_fractions,
    dts_zones,
)

# Table A1, "Coordinates (Reference/Monitor) of the Lines Defining the Zones of
# the DTS Error Grid", Supplemental Appendix 2. Each border is a 3-point
# polyline; the two ENDPOINTS of each are what pin the geometry, so the middle
# point (the corner, at the clamp) is implied by the pair.
#
# LOWER borders run below the line of identity (monitor under-reads), UPPER above
# it. The zone each border bounds is the one INSIDE it: the B lines are the A|B
# border, so zone A is what they enclose.
#
# NOTE the published supplement labels the E rows "D Lower"/"D Upper" — a
# copy-paste slip in the table's sub-headers, not in the coordinates: these sit
# outside the D lines in Figure A1 and are unambiguously E.
TABLE_A1 = {
    # border: (corner at the clamp, far endpoint), in (reference, monitor) mg/dL
    ('b', 'lower'): ((62.5, 0.0), (62.5, 50.0), (600.0, 480.0)),
    ('b', 'upper'): ((0.0, 60.0), (50.0, 60.0), (500.0, 600.0)),
    ('c', 'lower'): ((97.5, 0.0), (97.5, 50.0), (600.0, 307.0)),
    ('c', 'upper'): ((0.0, 86.5), (50.0, 86.5), (347.0, 600.0)),
    ('d', 'lower'): ((153.0, 0.0), (153.0, 50.0), (600.0, 197.0)),
    ('d', 'upper'): ((0.0, 124.0), (50.0, 124.0), (241.0, 600.0)),
    ('e', 'lower'): ((238.0, 0.0), (238.0, 50.0), (600.0, 126.0)),
    ('e', 'upper'): ((0.0, 179.0), (50.0, 179.0), (167.0, 600.0)),
}

# The |Risk| contour each border traces: the upper edge of the zone INSIDE it.
BORDER_RISK = {'b': 0.5, 'c': 1.5, 'd': 2.5, 'e': 3.5}

# Worst observed table-vs-function gap is 1.11 mg/dL (both D-lower and E-lower at
# the clamp corner). Anything materially larger is a defect, not rounding.
VERTEX_TOL_MGDL = 1.2


def _solve_monitor(reference: float, risk: float, upper: bool) -> float:
    """The monitor value on a given |Risk| contour at a given reference.

    Inverts the risk function by hand — deliberately NOT by calling ``dts_risk``,
    so the check is against an independent statement of the algebra rather than
    against the implementation restated.
    """
    r = max(reference, DTS_LOW_CLAMP_MGDL)
    if upper:
        return r * np.exp(risk / DTS_OVERESTIMATE_COEFF)
    return r * np.exp(-risk / DTS_UNDERESTIMATE_COEFF)


def _solve_reference(monitor: float, risk: float, upper: bool) -> float:
    """The reference value on a given |Risk| contour at a given monitor."""
    m = max(monitor, DTS_LOW_CLAMP_MGDL)
    if upper:
        return m * np.exp(-risk / DTS_OVERESTIMATE_COEFF)
    return m * np.exp(risk / DTS_UNDERESTIMATE_COEFF)


@pytest.mark.parametrize('zone', ['b', 'c', 'd', 'e'])
def test_published_vertices_lie_on_the_risk_contour(zone):
    """Both endpoints of both published borders of each zone, solved from the
    function, land on the tabulated coordinate.

    This is the whole geometry in one check: the two coefficients, the clamp, the
    four |Risk| edges and the axis convention all have to be right for any of the
    eight numbers to come out.
    """
    risk = BORDER_RISK[zone]
    worst = 0.0
    for side in ('lower', 'upper'):
        corner, _, far = TABLE_A1[(zone, side)]
        upper = side == 'upper'
        if upper:
            # corner is (reference <= clamp, monitor); far is (reference, monitor = 600)
            got_corner = _solve_monitor(corner[0], risk, upper)
            d_corner = abs(got_corner - corner[1])
            got_far = _solve_reference(far[1], risk, upper)
            d_far = abs(got_far - far[0])
            print(f"[DUMP] {zone.upper()}-{side}: monitor at ref<=clamp "
                  f"{got_corner:7.2f} vs {corner[1]:6.1f} (Δ{d_corner:.2f}); "
                  f"reference at monitor 600 {got_far:7.2f} vs {far[0]:6.1f} "
                  f"(Δ{d_far:.2f})")
        else:
            got_corner = _solve_reference(corner[1] or DTS_LOW_CLAMP_MGDL, risk, upper)
            d_corner = abs(got_corner - corner[0])
            got_far = _solve_monitor(far[0], risk, upper)
            d_far = abs(got_far - far[1])
            print(f"[DUMP] {zone.upper()}-{side}: reference at monitor<=clamp "
                  f"{got_corner:7.2f} vs {corner[0]:6.1f} (Δ{d_corner:.2f}); "
                  f"monitor at ref 600 {got_far:7.2f} vs {far[1]:6.1f} "
                  f"(Δ{d_far:.2f})")
        worst = max(worst, d_corner, d_far)
    assert worst <= VERTEX_TOL_MGDL, (
        f"zone {zone.upper()} borders miss Table A1 by {worst:.3f} mg/dL, over "
        f"the {VERTEX_TOL_MGDL} mg/dL rounding tolerance")


def test_the_asymmetry_is_in_log_space_not_in_the_ratio():
    """Overestimates are penalised harder — and that is NOT visible at zone A.

    This is the grid's whole clinical content (2.75 against 2.25, because a
    falsely high reading prompts insulin), and it is easy to check in a way that
    passes on a symmetric implementation. Exponentiating nearly cancels the
    coefficients at the innermost edge: A runs -19.93% / +19.94%, so zone A reads
    as a symmetric +/-20% band by coincidence.

    The claim with content is therefore made in LOG space, where the overestimate
    side of every edge is the tighter one, and again at the OUTER edge, where the
    cancellation has long since stopped.
    """
    ref = np.full(3, 200.0)
    pred = np.array([200.0 * 0.8, 200.0, 200.0 * 1.2])
    risk = dts_risk(ref, pred)
    print(f"\n[DUMP] risk at 0.8x / 1.0x / 1.2x of 200 mg/dL: {np.round(risk, 4)}")
    assert risk[1] == 0.0
    assert abs(risk[0]) == pytest.approx(0.5021, abs=1e-3)
    assert abs(risk[2]) == pytest.approx(0.5014, abs=1e-3)
    assert list(dts_zones(ref, pred)) == [1, 0, 1]

    for edge in DTS_ZONE_EDGES:
        over = _solve_monitor(200.0, edge, upper=True) / 200.0
        under = _solve_monitor(200.0, edge, upper=False) / 200.0
        print(f"[DUMP] |Risk|={edge}: {under:7.5f}x .. {over:7.5f}x  "
              f"(log {np.log(under):+.5f} .. {np.log(over):+.5f})  "
              f"linear +{over - 1:.5f} / -{1 - under:.5f}")
        assert abs(np.log(over)) < abs(np.log(under)), (
            f"at |Risk|={edge} the overestimate side is not the tighter one — "
            "the 2.75/2.25 asymmetry has been lost or inverted")

    # At the A edge the two linear deviations agree to 1.5e-4, which is why the
    # log-space form above is the one that can fail.
    a_over = _solve_monitor(200.0, DTS_ZONE_EDGES[0], upper=True) / 200.0 - 1.0
    a_under = 1.0 - _solve_monitor(200.0, DTS_ZONE_EDGES[0], upper=False) / 200.0
    assert abs(a_over - a_under) < 2e-4

    # At the outermost edge it is unmistakable in ratio terms too.
    e_over = _solve_monitor(200.0, DTS_ZONE_EDGES[-1], upper=True) / 200.0
    e_under = _solve_monitor(200.0, DTS_ZONE_EDGES[-1], upper=False) / 200.0
    assert e_over - 1.0 > 2.0 and 1.0 - e_under < 0.8


def test_the_zone_transition_happens_at_the_edge_and_the_edge_is_closed():
    """Each edge is CLOSED: ``0-0.5`` is A, ``>0.5-1.5`` is B, in the paper's own
    notation, so a point ON a border takes the LOWER-risk zone.

    A pair sitting EXACTLY on an edge cannot be constructed: the contour is at
    ``monitor/reference = exp(edge/coeff)``, and the exp/log round trip is not
    exact in float64 — it lands a few ULP off, in a direction that varies by edge,
    which is a property of the arithmetic and not of the rule. So the transition
    is BRACKETED at a relative 1e-12, which is three orders of magnitude above
    that noise and ten below any glucose difference that could mean anything; and
    the closed half is asserted where it IS exactly representable, at |Risk| = 0.
    """
    nudge = 1e-12
    ref = np.full(len(DTS_ZONE_EDGES), 100.0)
    on = np.array([_solve_monitor(100.0, e, upper=True) for e in DTS_ZONE_EDGES])
    below, above = on * (1.0 - nudge), on * (1.0 + nudge)
    z_below, z_above = dts_zones(ref, below), dts_zones(ref, above)
    print(f"\n[DUMP] {nudge:g} relative below each edge -> "
          f"{[ZONE_NAMES[z].upper() for z in z_below]}; above -> "
          f"{[ZONE_NAMES[z].upper() for z in z_above]}")
    assert list(z_below) == list(range(len(DTS_ZONE_EDGES))), (
        "just inside an edge already scored as the higher-risk zone")
    assert list(z_above) == [i + 1 for i in range(len(DTS_ZONE_EDGES))], (
        "just outside an edge did not cross into the higher-risk zone")

    # The exactly-representable case of the closed rule: zero risk is zone A.
    assert dts_risk(np.array([137.0]), np.array([137.0]))[0] == 0.0
    assert ZONE_NAMES[int(dts_zones(np.array([137.0]), np.array([137.0]))[0])] == 'a'


def test_the_clamp_governs_the_low_glucose_corner():
    """Under 50 mg/dL every glucose is treated as 50, on BOTH axes.

    Without the clamp the low corner is a log singularity and the flat/vertical
    border segments of Figure A1 cannot exist. This is the reconstructed part of
    the geometry — see ``dts_grid``'s docstring — so it gets its own check rather
    than riding on the vertex test.
    """
    # Two pairs that differ only below the clamp must score identically.
    a = dts_risk(np.array([30.0, 50.0]), np.array([200.0, 200.0]))
    print(f"\n[DUMP] risk(30 -> 200) {a[0]:.6f} vs risk(50 -> 200) {a[1]:.6f}")
    assert a[0] == pytest.approx(a[1], abs=1e-12), (
        "references of 30 and 50 mg/dL scored differently — the clamp is not "
        "applied to the reference axis")
    b = dts_risk(np.array([200.0, 200.0]), np.array([10.0, 50.0]))
    assert b[0] == pytest.approx(b[1], abs=1e-12), (
        "monitors of 10 and 50 mg/dL scored differently — the clamp is not "
        "applied to the monitor axis")
    # The paper's printed third branch: both under the clamp is zero risk.
    assert dts_risk(np.array([20.0]), np.array([45.0]))[0] == 0.0
    # And a badly-missed hypo is D, not E: 200 -> 10 clamps to 200 -> 50.
    assert ZONE_NAMES[int(dts_zones(np.array([200.0]), np.array([10.0]))[0])] == 'd'


def test_a_risk_space_array_trips_the_units_tripwire():
    """Feeding normalized or risk-space values raises instead of scoring.

    This is the failure mode the clamp creates: z-scores are all under 50, so
    every pair would clamp to 50/50, score 0 risk, and report a flawless 100%
    zone A. Nothing downstream could tell.
    """
    z = np.array([-1.2, 0.3, 2.1])
    with pytest.raises(AssertionError, match='units tripwire'):
        dts_zones(z, z)
    with pytest.raises(AssertionError, match='units tripwire'):
        dts_zones(np.array([120.0, 90.0, 200.0]), z)


def test_counts_and_fractions_round_trip():
    """Shares come from summed counts, and an empty grid is unmeasured, not 0%."""
    rng = np.random.default_rng(0)
    true = rng.uniform(40.0, 350.0, size=(20, 24))
    pred = true * rng.uniform(0.5, 1.6, size=true.shape)
    counts = dts_zone_counts(true, pred)
    assert counts['total'] == float(true.size)
    assert sum(counts[n] for n in ZONE_NAMES) == counts['total'], \
        "a point fell in no zone, or in two"
    fr = dts_zone_fractions(counts)
    assert sum(fr[n] for n in ZONE_NAMES) == pytest.approx(1.0)
    print(f"\n[DUMP] zone shares "
          f"{ {n: round(fr[n], 4) for n in ZONE_NAMES} }")
    empty = dts_zone_fractions({'total': 0.0})
    assert all(empty[n] is None for n in ZONE_NAMES), \
        "a zone share over no points must be None, not 0.0"


def test_both_trainers_render_every_zone_and_no_a_plus_b():
    """The five zones reach the page, in ``train.py`` and ``train_blind.py`` alike,
    and no A+B convenience row appears anywhere.

    The paper states in terms that presenting zone A+B as if both were clinically
    acceptable is inappropriate and that pZA alone is the measure — so an A+B row
    would be a claim the source refuses to make. Clarke's own A+B row is a
    different grid and stays.

    Both trainers are checked because the blind fork is a COPY: the grid is common
    ground between them, not one of the seven things they differ in, and nothing
    but a test keeps it that way.
    """
    import train
    import train_blind

    fed = {f'dts_{z}_pct': v for z, v in
           zip(ZONE_NAMES, (91.2, 7.4, 1.0, 0.3, 0.1))}
    for mod in (train, train_blind):
        page = mod._render_validation_table(1, dict(fed))
        assert 'Clinical Error Grid Analysis (DTS)' in page, (
            f"{mod.__name__}: the DTS section is absent from the table")
        for label in ('pZA', 'zone B', 'zone C', 'zone D', 'zone E'):
            assert label in page, f"{mod.__name__}: {label!r} is not rendered"
        assert '91.20' in page and '0.10' in page, (
            f"{mod.__name__}: zone values are not rendered")
        assert 'dts_A+B' not in page and 'DTS A+B' not in page, (
            f"{mod.__name__}: an A+B row appeared, which the paper calls "
            "inappropriate")
        cols = [c for c, _ in mod._val_log_columns()]
        assert [c for c in cols if c.startswith('dts_')], \
            f"{mod.__name__}: no dts_ column in the CSV header"
        assert not [c for c in cols if c.lower() in ('dts_ab_pct', 'dts_ab')], \
            f"{mod.__name__}: an A+B column exists in the CSV"
    print(f"\n[DUMP] both trainers render all five zones; "
          f"{len([c for c, _ in train._val_log_columns() if c.startswith('dts_')])} "
          "dts_ columns in the header")


def test_transposing_the_axes_changes_the_answer():
    """Argument order is load-bearing: (true, pred), never (pred, true).

    The grid is asymmetric, so a transposed call returns a well-formed table of a
    different statistic — the same trap ``cg_ega`` documents for its own inputs.
    """
    true = np.array([180.0, 90.0, 250.0])
    pred = np.array([140.0, 130.0, 200.0])
    forward = dts_risk(true, pred)
    reversed_ = dts_risk(pred, true)
    print(f"\n[DUMP] forward {np.round(forward, 4)}  transposed "
          f"{np.round(reversed_, 4)}")
    assert not np.allclose(np.abs(forward), np.abs(reversed_)), (
        "transposing reference and monitor left every |risk| unchanged — the "
        "2.75/2.25 asymmetry is gone")
