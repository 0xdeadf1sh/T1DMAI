import math

import numpy as np
import pytest
import torch

import utils
from config import (
    PREDICTION_HORIZON_HOURS,
    PREDICTION_PATCHES,
    TIME_PROBE_BIN_HOURS,
    TIME_PROBE_N_BINS,
)

ADV_HOURS = PREDICTION_HORIZON_HOURS / PREDICTION_PATCHES
N_BINS = TIME_PROBE_N_BINS
BIN_HOURS = TIME_PROBE_BIN_HOURS


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _spike_belief(peak_bin: int, sharpness: float = 3.0) -> np.ndarray:
    k = np.arange(N_BINS, dtype=np.float64)
    d = np.minimum(np.abs(k - peak_bin), N_BINS - np.abs(k - peak_bin))
    return _softmax(sharpness * np.cos(2.0 * math.pi * d / N_BINS))


def _circular_mean_hour(probs: np.ndarray) -> float:
    cos, sin = utils._resultant_np(probs, probs.shape[-1])
    return (math.atan2(sin, cos) % (2.0 * math.pi)) * (24.0 / (2.0 * math.pi))


def test_aggregate_recovers_origin():
    b0 = _spike_belief(peak_bin=N_BINS // 3)
    probs = np.stack(
        [utils._fractional_roll(b0, p * ADV_HOURS / BIN_HOURS) for p in range(PREDICTION_PATCHES)]
    )
    fused = utils.aggregate_origin_belief(probs, ADV_HOURS, BIN_HOURS)
    print("[DUMP] fused sum", fused.sum(), "argmax", fused.argmax(), "b0 argmax", b0.argmax())
    assert abs(fused.sum() - 1.0) < 1e-9
    err = abs((_circular_mean_hour(fused) - _circular_mean_hour(b0) + 12.0) % 24.0 - 12.0)
    print("[DUMP] circular mean hour err", err)
    assert err <= 0.3
    assert fused.argmax() == b0.argmax()


def test_aggregate_p1_identity():
    b0 = _spike_belief(peak_bin=1)[None, :]
    fused = utils.aggregate_origin_belief(b0, ADV_HOURS, BIN_HOURS)
    assert np.allclose(fused, b0[0])


def test_clock_geometry_shapes():
    probs = _spike_belief(peak_bin=2)
    arc = 6
    geom = utils.clock_wedge_geometry(probs, arc_segments=arc)
    assert geom.wedges.shape == (N_BINS, arc + 2, 2)
    assert geom.magnitudes.shape == (N_BINS,)
    assert geom.hand.shape == (2,)
    assert np.isscalar(geom.R) or np.ndim(geom.R) == 0


def test_R_rotation_invariant():
    probs = _spike_belief(peak_bin=5)
    R0 = utils.clock_wedge_geometry(probs, rotation_hours=0.0).R
    for rot in (3.0, 6.0, 12.0, -5.0):
        R = utils.clock_wedge_geometry(probs, rotation_hours=rot).R
        assert abs(R - R0) <= 1e-6, (rot, R, R0)


def test_hand_rotates_by_2pi_t_over_24():
    probs = _spike_belief(peak_bin=4)
    h0 = utils.clock_wedge_geometry(probs, rotation_hours=0.0).hand
    for t in (1.0, 3.5, 7.0, 23.0):
        ht = utils.clock_wedge_geometry(probs, rotation_hours=t).hand
        a0 = math.atan2(h0[0], h0[1])
        at = math.atan2(ht[0], ht[1])
        expected = 2.0 * math.pi * t / 24.0
        d = (at - a0 - expected + math.pi) % (2.0 * math.pi) - math.pi
        print("[DUMP] t", t, "angle delta residual", d)
        assert abs(d) <= 1e-6


def test_wrap_adjacency():
    probs = np.zeros(N_BINS)
    probs[0] = 0.5
    probs[N_BINS - 1] = 0.5
    hour = _circular_mean_hour(probs)
    err = min(hour, 24.0 - hour)
    print("[DUMP] near-midnight resultant hour", hour, "err", err)
    assert err <= BIN_HOURS
    geom = utils.clock_wedge_geometry(probs)
    last_end = geom.wedges[N_BINS - 1, -1, :]
    first_start = geom.wedges[0, 1, :]
    assert np.allclose(last_end / (np.linalg.norm(last_end) + 1e-12),
                       first_start / (np.linalg.norm(first_start) + 1e-12), atol=1e-9)


def test_all_zero_probs():
    probs = np.zeros(N_BINS)
    geom = utils.clock_wedge_geometry(probs)
    assert np.all(geom.magnitudes == 0.0)
    assert geom.R == 0.0
    assert np.isfinite(geom.wedges).all()
    assert np.isfinite(geom.hand).all()


def test_numpy_torch_resultant_agreement():
    rng = np.random.default_rng(0)
    probs = _softmax(rng.standard_normal(N_BINS))
    cos, sin = utils._resultant_np(probs, N_BINS)
    res_t = utils.time_of_day_resultant(torch.tensor(probs), N_BINS)
    print("[DUMP] np", (cos, sin), "torch", res_t.tolist())
    assert abs(cos - float(res_t[0])) <= 1e-5
    assert abs(sin - float(res_t[1])) <= 1e-5


def test_render_rotation_matches_trained_rotation():
    probs = _spike_belief(peak_bin=6)
    cos, sin = utils._resultant_np(probs, N_BINS)
    dtheta = 2.0 * math.pi * ADV_HOURS / 24.0
    rc = cos * math.cos(dtheta) - sin * math.sin(dtheta)
    rs = cos * math.sin(dtheta) + sin * math.cos(dtheta)
    hand = utils.clock_wedge_geometry(probs, rotation_hours=ADV_HOURS).hand
    print("[DUMP] hand", hand.tolist(), "expected", [rs, rc])
    assert np.allclose(hand, np.array([rs, rc]), atol=1e-5)


def test_matplotlib_smoke():
    clock_face = pytest.importorskip("clock_face")
    if not getattr(clock_face, "_OK", False):
        pytest.skip("matplotlib unavailable")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot()
    probs = _spike_belief(peak_bin=3)
    clock_face.draw_clock_axis(ax, probs)
    assert len(ax.patches) + len(ax.lines) >= 1
    ax2 = fig.add_subplot()
    clock_face.draw_clock_axis(ax2, None)
    plt.close(fig)
