"""Tests for metrics/attention.py — the accounting, not the model.

The probe reduces a (T,) attention row into offset bins and side splits. Those
sums have to conserve mass and partition the window, or a profile that looks
plausible is quietly dropping or double-counting attention.
"""

import numpy as np


def _series(reach: int = 4):
    import metrics.attention as A
    A.OFFSET_REACH = reach          # a small reach so 'beyond' is exercised
    return A


def test_offset_bins_and_beyond_conserve_the_whole_row():
    """profile + beyond_before + beyond_after == the row's total mass."""
    A = _series(reach=4)
    T, patch = 20, 12
    mass = np.linspace(1.0, 3.0, T)
    mass /= mass.sum()
    masked = np.zeros(T, dtype=bool)
    masked[patch] = True

    s = A._Series()
    s.add(mass, patch, (patch, 1), masked, np.array([0.25] * 4))

    total = s.profile.sum() + s.beyond_before + s.beyond_after
    np.testing.assert_allclose(total, mass.sum(), rtol=0, atol=1e-12)
    print(f"[DUMP] profile {s.profile.sum():.6f} + beyond "
          f"{s.beyond_before + s.beyond_after:.6f} = {total:.6f}")


def test_the_window_split_partitions_the_row():
    """before + own span + after == the whole row, with no overlap."""
    A = _series(reach=4)
    T, span = 20, (8, 3)
    mass = np.full(T, 1.0 / T)
    masked = np.zeros(T, dtype=bool)
    masked[span[0]:span[0] + span[1]] = True

    s = A._Series()
    s.add(mass, span[0], span, masked, np.array([0.25] * 4))

    np.testing.assert_allclose(s.before + s.own_span + s.after, 1.0, atol=1e-12)
    np.testing.assert_allclose(s.own_span, span[1] / T, atol=1e-12)
    np.testing.assert_allclose(s.on_masked, span[1] / T, atol=1e-12)


def test_the_near_split_excludes_the_span_and_is_symmetric_in_reach():
    """The placement-free split must not count the span's own patches."""
    A = _series(reach=4)
    T, span, patch = 40, (18, 3), 18
    mass = np.full(T, 1.0 / T)
    masked = np.zeros(T, dtype=bool)
    masked[span[0]:span[0] + span[1]] = True

    s = A._Series()
    s.add(mass, patch, span, masked, np.array([0.25] * 4))

    # Reach 4 either side of patch 18, minus the span's patches 18-20:
    # before = 14..17 (4 patches), after = 21..22 (2 patches).
    np.testing.assert_allclose(s.near_before, 4 / T, atol=1e-12)
    np.testing.assert_allclose(s.near_after, 2 / T, atol=1e-12)
    assert s.near_before + s.near_after < s.before + s.after


def test_a_right_edge_span_has_nothing_after_it():
    """The forecast geometry, which is the probe's own sanity check."""
    A = _series(reach=4)
    T, span = 20, (17, 3)          # flush against the right edge
    mass = np.full(T, 1.0 / T)
    masked = np.zeros(T, dtype=bool)
    masked[span[0]:] = True

    s = A._Series()
    s.add(mass, span[0], span, masked, np.array([0.25] * 4))

    assert s.after == 0.0 and s.near_after == 0.0


def test_the_cell_reports_every_layer_and_names_the_last():
    """The rollout, every layer, and ``final_layer`` as a name for the last one.

    ``final_layer`` must not be a second accumulation of the same rows: two
    accumulators for one series drift the moment one of them changes.
    """
    A = _series(reach=4)
    T, patch, n_layers = 20, 10, 3
    roll = np.full(T, 1.0 / T)
    per_layer = np.zeros((n_layers, T))
    for i in range(n_layers):
        per_layer[i, patch - 1 - i] = 1.0        # each layer reads one patch back
    masked = np.zeros(T, dtype=bool); masked[patch] = True

    cell = A._Cell()
    cell.add(roll, per_layer, patch, (patch, 1), masked, np.array([0.25] * 4))
    out = cell.summary(['a', 'b', 'c', 'd'])

    assert out['n'] == 1
    assert len(out['layers']) == n_layers
    assert all(ls['n'] == 1 for ls in out['layers'])
    assert out['final_layer'] == out['layers'][-1]
    # The gradient read belongs to the slot, not to one attention series.
    assert 'channel_share' in out
    assert all('channel_share' not in ls for ls in out['layers'])
    # Only the rollout put mass on the masked patch here; no layer did.
    np.testing.assert_allclose(out['mass_on_masked_patches'], 1.0 / T, atol=1e-12)
    for ls in out['layers']:
        np.testing.assert_allclose(ls['mass_on_masked_patches'], 0.0, atol=1e-12)


def test_the_layer_stack_grows_to_whatever_depth_arrives():
    """A cell is built before the depth is known, so the stack sizes itself."""
    A = _series(reach=4)
    T, patch = 20, 10
    masked = np.zeros(T, dtype=bool)
    cell = A._Cell()
    for _ in range(2):
        cell.add(np.full(T, 1.0 / T), np.full((5, T), 1.0 / T), patch,
                 (patch, 1), masked, np.array([0.25] * 4))
    out = cell.summary(['a', 'b', 'c', 'd'])
    assert len(out['layers']) == 5
    assert all(ls['n'] == 2 for ls in out['layers'])
