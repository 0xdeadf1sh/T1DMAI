"""The categorical per-patch time-of-day bin probe.

The jump and cross-window tests pick ``n_bins = 12`` with ``advance = 2 h`` so one
bin is exactly one horizon of advance.
"""

import math

import pytest
import torch


def _onehot_logits(bins: torch.Tensor, n_bins: int, hot: float = 20.0) -> torch.Tensor:
    """``(..., n_bins)`` fp32 logits; ``exp(hot)`` must dominate so softmax ≈ one-hot."""
    return torch.nn.functional.one_hot(bins, n_bins).to(torch.float32) * hot


def _forward_inputs(B: int = 2):
    """``(patches, attn_mask, anchor_bg, mask_idx)`` right-edge, ``M = PREDICTION_PATCHES``
    so the probe emits one row per horizon patch."""
    from tests.forward_inputs import right_edge_inputs

    return right_edge_inputs(B, all_true_mask=True)


def test_bin_target_rows_sum_to_one():
    from utils import time_of_day_bin_target

    n_bins = 12
    hours = torch.arange(0.0, 24.0, 0.5)          # (48,)
    tgt = time_of_day_bin_target(hours, n_bins, 0.75)
    assert tgt.shape == (hours.numel(), n_bins), f"shape {tuple(tgt.shape)}"
    sums = tgt.sum(dim=-1)
    print(f"\n[DUMP] bin_target | rows sum in "
          f"[{float(sums.min()):.6f}, {float(sums.max()):.6f}] (want 1.0)")
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "rows must sum to 1"


def test_bin_target_onehot_when_no_smooth():
    from utils import time_of_day_bin_target, time_of_day_bin_centers

    n_bins = 12
    centers = time_of_day_bin_centers(n_bins)
    for k in range(n_bins):
        row = time_of_day_bin_target(centers[k], n_bins, 0.0)      # (n_bins,)
        assert row.shape == (n_bins,)
        assert float(row.sum()) == pytest.approx(1.0)
        nz = int((row > 0).sum())
        assert nz == 1, f"one-hot must have a single nonzero, got {nz}"
        assert int(row.argmax()) == k, f"one-hot at {int(row.argmax())} != bin {k}"
    print(f"\n[DUMP] bin_target | smooth<=0 -> exact one-hot on the containing bin "
          f"(all {n_bins} bins ✓)")


def test_bin_target_peak_at_correct_bin():
    from utils import time_of_day_bin_target, time_of_day_bin_centers

    n_bins = 12
    centers = time_of_day_bin_centers(n_bins)
    peaks = []
    for k in range(n_bins):
        row = time_of_day_bin_target(centers[k], n_bins, 0.75)
        peaks.append(int(row.argmax()))
        assert int(row.argmax()) == k, f"peak {int(row.argmax())} != containing bin {k}"
    print(f"\n[DUMP] bin_target | soft-label argmax follows the containing bin: {peaks}")


def test_bin_target_circular_wrap():
    from utils import time_of_day_bin_target

    n_bins = 12                                    # 2 h bins
    row = time_of_day_bin_target(torch.tensor(23.9), n_bins, 0.75)   # (n_bins,)
    top2 = set(int(i) for i in torch.topk(row, 2).indices.tolist())
    print(f"\n[DUMP] bin_target | hour=23.9 top-2 bins={sorted(top2)} "
          f"p[0]={float(row[0]):.4f} p[11]={float(row[11]):.4f} p[6]={float(row[6]):.4f}")
    assert top2 == {0, 11}, f"mass must wrap the seam onto bins {{0, 11}}, got {sorted(top2)}"
    # wraps the SHORT way: bin 0 outweighs a bin an equal step the other way
    assert float(row[0]) > float(row[2]), "wrap must put more mass on bin 0 than bin 2"
    assert float(row[0]) > float(row[6]), "antipodal bin must stay near-empty"


def test_decode_bins_roundtrip_and_confidence():
    from utils import (time_of_day_decode_bins, time_of_day_bin_centers,
                       circular_hour_error)

    n_bins = 12
    centers = time_of_day_bin_centers(n_bins)
    logits = _onehot_logits(torch.arange(n_bins), n_bins)          # (n_bins, n_bins)
    hours, R = time_of_day_decode_bins(logits, n_bins)
    assert hours.shape == (n_bins,) and R.shape == (n_bins,)
    max_err = float(circular_hour_error(hours, centers).max())
    min_R = float(R.min())
    print(f"\n[DUMP] decode | one-hot roundtrip max circular err={max_err:.2e} h  "
          f"min R={min_R:.6f} (want ~1)")
    assert max_err < 1e-3, f"one-hot decode must return the bin center, err={max_err}"
    assert min_R > 0.999, f"a one-hot must decode to confidence ~1, got {min_R}"

    uni_hour, uni_R = time_of_day_decode_bins(torch.zeros(1, n_bins), n_bins)
    print(f"[DUMP] decode | uniform logit -> R={float(uni_R):.2e} (want ~0)")
    assert float(uni_R) < 1e-4, f"uniform bins have no preferred hour, R={float(uni_R)}"


def _ramp_and_jump(n_bins: int, P: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(ramp_logits, jump_logits)``, each ``(B, P, n_bins)``."""
    base = torch.tensor([0, 4])                                    # (B,)
    steps = torch.arange(P)
    ramp_bins = (base[:, None] + steps[None, :]) % n_bins          # +1 bin/patch
    scramble = torch.tensor([0, 5, 2, 9, 3, 11])[:P]
    jump_bins = (base[:, None] + scramble[None, :]) % n_bins
    return _onehot_logits(ramp_bins, n_bins), _onehot_logits(jump_bins, n_bins)


def test_inter_patch_jump_ramp_vs_jump():
    from utils import time_inter_patch_jump_hours

    n_bins, P = 12, 6
    adv = 24.0 / n_bins
    ramp, jump = _ramp_and_jump(n_bins, P)

    jr = time_inter_patch_jump_hours(ramp, n_bins, adv)            # (B,)
    jj = time_inter_patch_jump_hours(jump, n_bins, adv)
    assert jr.shape == (ramp.shape[0],) and jj.shape == (jump.shape[0],)
    print(f"\n[DUMP] jump | ramp mean|dev|={float(jr.mean()):.3e} h  "
          f"jump mean|dev|={float(jj.mean()):.3f} h")
    assert float(jr.mean()) < 1e-3, f"consistent ramp must not jump, got {float(jr.mean())} h"
    assert float(jj.mean()) > 1.0, f"scrambled set must jump, got {float(jj.mean())} h"

    zero = time_inter_patch_jump_hours(ramp[:, :1, :], n_bins, adv)
    assert zero.shape == (ramp.shape[0],) and float(zero.abs().max()) == 0.0


def _origin_onehot(bins, n_bins, P=2):
    """(B, P, n_bins) near-one-hot logits with the ORIGIN patch (0) at each given bin."""
    logits = torch.full((len(bins), P, n_bins), -10.0)
    for i, b in enumerate(bins):
        logits[i, :, int(b) % n_bins] = 10.0
    return logits


def test_cross_window_consistency_loss():
    """An all-False valid mask must give a finite 0, not NaN."""
    from utils import time_cross_window_consistency_loss
    n_bins, H = 12, 2.0                       # one 2 h bin == one horizon
    k = _origin_onehot([3, 7], n_bins)
    nxt_good = _origin_onehot([4, 8], n_bins)  # +1 bin == +H
    nxt_bad = _origin_onehot([9, 1], n_bins)
    good = float(time_cross_window_consistency_loss(k, nxt_good, n_bins, H))
    bad = float(time_cross_window_consistency_loss(k, nxt_bad, n_bins, H))
    print(f"\n[DUMP] xwin cons | good={good:.3e}  bad={bad:.3e}")
    assert good < 1e-3 and bad > 1e-2 and bad > good
    z = time_cross_window_consistency_loss(k, nxt_bad, n_bins, H,
                                           valid=torch.zeros(2, dtype=torch.bool))
    assert z.shape == () and float(z) == 0.0
    only0 = time_cross_window_consistency_loss(
        k, torch.stack([nxt_good[0], nxt_bad[1]]), n_bins, H,
        valid=torch.tensor([True, False]))
    assert float(only0) < 1e-3


def test_cross_window_jump_hours():
    from utils import time_cross_window_jump_hours
    n_bins, H = 12, 2.0
    k = _origin_onehot([3, 7], n_bins)
    jg = time_cross_window_jump_hours(k, _origin_onehot([4, 8], n_bins), n_bins, H)
    jb = time_cross_window_jump_hours(k, _origin_onehot([7, 7], n_bins), n_bins, H)
    assert jg.shape == (2,) and jb.shape == (2,)
    print(f"\n[DUMP] xwin jump | good={float(jg.mean()):.3e} h  bad={float(jb.mean()):.3f} h")
    assert float(jg.mean()) < 1e-3 and float(jb.mean()) > 1.0


def test_model_forward_bin_logits_shape_and_disable(monkeypatch):
    """Probe presence leaves ``(q_tau, median)`` bit-identical: its init draws under a
    saved RNG state, so the forecast weights are probe-neutral."""
    import config
    import model as model_mod
    from config import PREDICTION_PATCHES, TIME_PROBE_N_BINS

    if not model_mod.TIME_PROBE_ENABLED:
        pytest.skip("probe disabled; the enabled-forward contract is vacuous")

    B = 2
    torch.manual_seed(0)
    m_with = model_mod.T1DMAI().eval()
    patches, attn_mask, anchor_bg, mask_idx = _forward_inputs(B)
    with torch.no_grad():
        q1, med1, time_pred = m_with(patches, attn_mask, anchor_bg, mask_idx, return_time=True)

    assert time_pred is not None, "enabled probe must return per-patch logits"
    expected = (B, PREDICTION_PATCHES, TIME_PROBE_N_BINS)
    assert time_pred.shape == expected, f"time_pred shape {tuple(time_pred.shape)} != {expected}"

    monkeypatch.setattr(model_mod, 'TIME_PROBE_ENABLED', False)
    torch.manual_seed(0)
    m_without = model_mod.T1DMAI().eval()
    assert m_without.time_head is None, "disabled probe must not build time_head"
    with torch.no_grad():
        q0, med0, tp0 = m_without(patches, attn_mask, anchor_bg, mask_idx, return_time=True)

    assert tp0 is None, "disabled probe must yield time_pred=None"
    max_q = float((q0 - q1).abs().max())
    max_m = float((med0 - med1).abs().max())
    print(f"\n[DUMP] forward | logits shape {tuple(time_pred.shape)}  "
          f"disable: max|dq|={max_q:.2e} max|dm|={max_m:.2e} (want 0)")
    assert torch.equal(q0, q1), "probe presence perturbed q_tau (I2 violated)"
    assert torch.equal(med0, med1), "probe presence perturbed median (I2 violated)"


def test_within_window_penalty_retired():
    """The cross-window knobs carry the penalty; the within-window witness stays
    unpenalized."""
    import config
    import utils

    assert not hasattr(config, 'TIME_PROBE_CONSISTENCY_WEIGHT'), \
        "the within-window consistency weight must be removed from config"
    assert not hasattr(utils, 'time_advance_consistency_loss'), \
        "utils.time_advance_consistency_loss must be retired"

    assert hasattr(config, 'TIME_PROBE_CROSS_WINDOW_WEIGHT'), \
        "config must expose TIME_PROBE_CROSS_WINDOW_WEIGHT"
    assert hasattr(config, 'TIME_PROBE_CROSS_WINDOW_FRACTION'), \
        "config must expose TIME_PROBE_CROSS_WINDOW_FRACTION"
    for name in ('time_cross_window_consistency_loss',
                 'time_cross_window_jump_hours',
                 'time_inter_patch_jump_hours'):
        assert hasattr(utils, name), f"utils must expose {name}"
    print(f"\n[DUMP] retire | consistency weight gone; xwin weight="
          f"{config.TIME_PROBE_CROSS_WINDOW_WEIGHT} frac="
          f"{config.TIME_PROBE_CROSS_WINDOW_FRACTION}; helpers present ✓")
