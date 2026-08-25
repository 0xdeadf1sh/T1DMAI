"""The time-of-day probe: per-slot hour-of-day logits over ``TIME_PROBE_N_BINS``
circular bins, no mean-pool.

At ``TIME_PROBE_DETACH=False`` its gradient co-trains the shared trunk; the forward
VALUE of the forecast is unaffected either way. Logits are ``(B, M, N_BINS)`` and
slot ``j`` is patch ``mask_idx[:, j]``, so only a right-edge span makes slot ``j``
the patch at ``pred_start + j``.

The bin helpers themselves are covered in ``tests/test_time_probe_bins.py``.
"""

import math

import pytest
import torch

import config
from config import PREDICTION_PATCHES
from tests.forward_inputs import right_edge_inputs


def _forward_inputs(B: int = 2):
    """``(patches, attn_mask, anchor_bg, mask_idx)`` right-edge, every slot valid."""
    return right_edge_inputs(B, all_true_mask=True)


def test_circular_hour_error_values():
    """The shorter of the two arcs, in [0, 12]."""
    from utils import circular_hour_error

    cases = [
        ((23.0, 1.0), 2.0),    # wraps across midnight the short way
        ((1.0, 23.0), 2.0),    # symmetric
        ((0.0, 12.0), 12.0),   # antipodal — the maximum
        ((6.0, 6.0), 0.0),     # identical
        ((0.0, 23.5), 0.5),    # a half-hour short of a full wrap
    ]
    for (pred, true), expected in cases:
        got = circular_hour_error(torch.tensor(pred), torch.tensor(true))
        assert torch.allclose(got, torch.tensor(expected), atol=1e-4), (
            f"circular_hour_error({pred}, {true}) = {float(got):.4f}, expected {expected}"
        )
    print("\n[DUMP] tod | circular_hour_error values "
          "(23,1)=2 (1,23)=2 (0,12)=12 (6,6)=0 (0,23.5)=0.5 ✓")


def test_circular_hour_residual_signed():
    """Signed keeps direction; |residual| == the unsigned circular error."""
    from utils import circular_hour_residual, circular_hour_error

    assert torch.allclose(circular_hour_residual(torch.tensor(1.0), torch.tensor(23.0)),
                          torch.tensor(2.0), atol=1e-4)    # 1am reads 2h AHEAD of 11pm
    assert torch.allclose(circular_hour_residual(torch.tensor(23.0), torch.tensor(1.0)),
                          torch.tensor(-2.0), atol=1e-4)   # 11pm reads 2h BEHIND 1am
    assert torch.allclose(circular_hour_residual(torch.tensor(6.0), torch.tensor(6.0)),
                          torch.tensor(0.0), atol=1e-4)
    # holds for every non-antipodal pair
    p = torch.arange(0.0, 24.0, 0.25)
    t = (p + torch.linspace(-11.0, 11.0, p.numel())) % 24.0
    assert torch.allclose(circular_hour_residual(p, t).abs(),
                          circular_hour_error(p, t), atol=1e-4)
    print("\n[DUMP] tod | signed residual (1,23)=+2 (23,1)=-2; |residual|==error ✓")


def test_clock_bias_and_precision():
    from utils import circular_bias_hours, circular_std_hours

    true = torch.arange(0.0, 24.0)

    # a constant +2 h offset: all bias, ~zero spread
    pred_off = (true + 2.0) % 24.0
    assert torch.allclose(circular_bias_hours(pred_off, true), torch.tensor(2.0), atol=1e-3)
    assert float(circular_std_hours(pred_off, true)) < 1e-2

    # a perfect clock: zero bias, zero spread
    assert torch.allclose(circular_bias_hours(true, true), torch.tensor(0.0), atol=1e-4)
    assert float(circular_std_hours(true, true)) < 1e-4

    # symmetric ±1 h jitter: ~zero bias, ~1 h precision
    jitter = torch.ones(24)
    jitter[1::2] = -1.0
    pred_jit = (true + jitter) % 24.0
    bias = float(circular_bias_hours(pred_jit, true))
    sd = float(circular_std_hours(pred_jit, true))
    assert abs(bias) < 1e-3, f"symmetric jitter should be unbiased, got {bias:.3f} h"
    assert 0.5 < sd < 1.5, f"±1h jitter precision out of band: {sd:.3f} h"
    print(f"\n[DUMP] tod | bias/precision: +2h offset -> bias~2 sd~0; "
          f"pm1h jitter -> bias~{bias:.1e} sd~{sd:.2f}h checks out")


def test_forward_return_time_arity():
    """The 3rd element is the per-patch bin logits, or ``None`` when the probe is off."""
    from model import T1DMAI

    torch.manual_seed(0)
    B = 2
    model = T1DMAI().eval()
    patches, attn_mask, anchor_bg, mask_idx = _forward_inputs(B)

    with torch.no_grad():
        out2 = model(patches, attn_mask, anchor_bg, mask_idx)
        out3 = model(patches, attn_mask, anchor_bg, mask_idx, return_time=True)

    assert len(out2) == 2, f"return_time=False must give a 2-tuple, got len {len(out2)}"
    assert len(out3) == 3, f"return_time=True must give a 3-tuple, got len {len(out3)}"

    time_pred = out3[2]
    if config.TIME_PROBE_ENABLED:
        assert time_pred is not None, "time_pred must be present when TIME_PROBE_ENABLED"
        expected = (B, PREDICTION_PATCHES, config.TIME_PROBE_N_BINS)
        assert time_pred.shape == expected, (
            f"time_pred shape {tuple(time_pred.shape)} != {expected}")
        print(f"\n[DUMP] tod | return_time arity 2/3 ✓ time_pred shape {tuple(time_pred.shape)}")
    else:
        assert time_pred is None, "time_pred must be None when TIME_PROBE_ENABLED is False"
        print("\n[DUMP] tod | probe disabled: return_time=True yields time_pred=None ✓")


def test_probe_does_not_perturb_forecast():
    """``q_tau`` / ``median`` are byte-identical with and without ``return_time``: the
    probe head never feeds them."""
    from model import T1DMAI

    torch.manual_seed(0)
    model = T1DMAI().eval()
    # one set of inputs for both calls; the forward draws no RNG, so any difference
    # comes solely from the probe
    patches, attn_mask, anchor_bg, mask_idx = _forward_inputs(B=2)

    with torch.no_grad():
        q0, m0 = model(patches, attn_mask, anchor_bg, mask_idx)
        q1, m1, _ = model(patches, attn_mask, anchor_bg, mask_idx, return_time=True)

    max_q = float((q0 - q1).abs().max())
    max_m = float((m0 - m1).abs().max())
    print(f"\n[DUMP] tod | I1 forecast invariance: max|dq|={max_q:.2e} max|dm|={max_m:.2e}")
    assert torch.equal(q0, q1), "q_tau changed when return_time=True (I1 violated)"
    assert torch.equal(m0, m1), "median changed when return_time=True (I1 violated)"


def test_detach_isolates_trunk():
    """A probe-only loss reaches ``time_head`` and leaves the trunk grad-free."""
    from model import T1DMAI

    if not (config.TIME_PROBE_ENABLED and config.TIME_PROBE_DETACH):
        pytest.skip("time probe disabled or not detached")

    torch.manual_seed(0)
    model = T1DMAI()
    patches, attn_mask, anchor_bg, mask_idx = _forward_inputs(B=2)

    _, _, time_pred = model(patches, attn_mask, anchor_bg, mask_idx, return_time=True)
    assert time_pred is not None, "enabled probe must return a time_pred"
    # probe-only objective: MSE to a fixed non-trivial target
    target = torch.ones_like(time_pred)
    loss = (time_pred - target).pow(2).mean()

    model.zero_grad()
    loss.backward()

    # the detach severs the graph, so patch_embed gets no grad, or a zero one
    pe_grad = model.patch_embed.weight.grad
    trunk_touched = pe_grad is not None and int(torch.count_nonzero(pe_grad)) > 0
    # the first Linear of the 2-layer SiLU MLP must get a real gradient
    th_grad = model.time_head[0].weight.grad
    probe_nonzero = th_grad is not None and int(torch.count_nonzero(th_grad)) > 0

    print(f"\n[DUMP] tod | detach: trunk(patch_embed) touched={trunk_touched} "
          f"probe(time_head[0]) grad-nonzero={probe_nonzero}")
    assert not trunk_touched, "detached probe leaked gradient into patch_embed (trunk)"
    assert probe_nonzero, "probe loss produced no gradient on time_head[0]"


def test_undetached_probe_shapes_trunk():
    """At ``TIME_PROBE_DETACH=False`` a probe-only loss reaches both ``time_head`` and
    the trunk: the representation is co-trained on circadian phase."""
    from model import T1DMAI

    if not (config.TIME_PROBE_ENABLED and not config.TIME_PROBE_DETACH):
        pytest.skip("time probe disabled or detached (read-only)")

    torch.manual_seed(0)
    model = T1DMAI()
    patches, attn_mask, anchor_bg, mask_idx = _forward_inputs(B=2)

    _, _, time_pred = model(patches, attn_mask, anchor_bg, mask_idx, return_time=True)
    assert time_pred is not None, "enabled probe must return a time_pred"
    target = torch.ones_like(time_pred)
    loss = (time_pred - target).pow(2).mean()

    model.zero_grad()
    loss.backward()

    pe_grad = model.patch_embed.weight.grad
    trunk_touched = pe_grad is not None and int(torch.count_nonzero(pe_grad)) > 0
    th_grad = model.time_head[0].weight.grad
    probe_nonzero = th_grad is not None and int(torch.count_nonzero(th_grad)) > 0

    print(f"\n[DUMP] tod | un-detached: trunk(patch_embed) touched={trunk_touched} "
          f"probe(time_head[0]) grad-nonzero={probe_nonzero}")
    assert trunk_touched, "un-detached probe must co-train the trunk (patch_embed grad expected)"
    assert probe_nonzero, "probe loss produced no gradient on time_head[0]"


def test_time_head_presence():
    from model import T1DMAI

    model = T1DMAI()
    present = model.time_head is not None
    print(f"\n[DUMP] tod | time_head present={present} TIME_PROBE_ENABLED={config.TIME_PROBE_ENABLED}")
    assert present == config.TIME_PROBE_ENABLED, (
        "time_head presence must track config.TIME_PROBE_ENABLED")


def test_probe_construction_preserves_forecast_init_rng(monkeypatch):
    """Building with vs without the probe must give byte-identical forecast weights.

    The probe's ``nn.Linear`` consumes RNG; in the main stream that draw shifts every
    forecast weight inited after it, so the head is built under a saved/restored RNG
    state and re-inited last.
    """
    import model as model_mod

    if not model_mod.TIME_PROBE_ENABLED:
        pytest.skip("probe disabled; I2 is trivially satisfied")

    torch.manual_seed(0)
    m_with = model_mod.T1DMAI()
    forecast_with = {n: p.detach().clone()
                     for n, p in m_with.named_parameters()
                     if not n.startswith('time_head.')}

    monkeypatch.setattr(model_mod, 'TIME_PROBE_ENABLED', False)
    torch.manual_seed(0)
    m_without = model_mod.T1DMAI()

    assert m_with.time_head is not None and m_without.time_head is None
    mismatched = [n for n, p in m_without.named_parameters()
                  if not torch.equal(p, forecast_with[n])]
    print(f"\n[DUMP] I2 | {len(forecast_with)} forecast tensors, "
          f"{len(mismatched)} shifted by the probe (want 0)")
    assert not mismatched, f"probe construction shifted forecast init for: {mismatched[:6]}"


def _stats():
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def test_cross_window_training_step_smoke():
    """data.py -> two forwards -> L_cross -> backward, on a tiny ``num_workers=0`` batch.

    The second forward consumes the SHIPPED ``next_window`` tensors, the same fields
    train.py hands the model, so a collate that builds window k+1's mask from the wrong
    masked set fails here. The penalty is ``utils``'s at a fixed one-horizon advance,
    not train.py's per-sample variant: this covers the data.py -> model plumbing only.
    """
    from data import T1DMDataset, collate_fn
    from model import T1DMAI
    from config import (TIME_PROBE_ENABLED, TIME_PROBE_CROSS_WINDOW_WEIGHT,
                        TIME_PROBE_N_BINS, PREDICTION_HORIZON_HOURS,
                        PREDICTION_PATCHES)
    from utils import (create_attention_mask_from_visible,
                       time_cross_window_consistency_loss,
                       time_cross_window_jump_hours)

    if not (TIME_PROBE_ENABLED and TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0):
        pytest.skip("cross-window probe off — nothing to co-train")

    stats = _stats()
    B = 2
    dataset = T1DMDataset(master_seed=7, total_steps=10, batch_size=B,
                          normalization_stats=stats)
    batch = collate_fn([dataset[i] for i in range(B)])
    assert 'next_window' in batch, "probe on => data.py must ship next_window"
    nw = batch['next_window']
    assert bool(nw['valid'].any()), "need at least one valid next window for the smoke"

    torch.manual_seed(0)
    model = T1DMAI()
    patches = batch['patches']
    attn_mask = batch['attn_mask']
    bfd = batch['bg_formula_data']

    # the two windows share n_ctx and so the left-pad, but NOT the masked set: window
    # k+1 has its own right-edge span, so it needs its own anchor_bg, mask_idx and
    # attention mask. The rebuild is an ORACLE, not the input — the forward below runs
    # on the shipped field, which is pinned to it.
    max_T = patches.shape[1]
    is_pad = torch.zeros(B, max_T, dtype=torch.bool)
    nw_masked = torch.zeros(B, max_T, dtype=torch.bool)
    for i in range(B):
        n_pad = max_T - (int(batch['n_context_patches'][i]) + PREDICTION_PATCHES)
        is_pad[i, :n_pad] = True
        nw_masked[i, nw['mask_idx'][i][nw['valid_slots'][i]]] = True
    rebuilt_attn_mask = create_attention_mask_from_visible(~nw_masked, is_pad)
    assert torch.equal(nw['attn_mask'], rebuilt_attn_mask), (
        "next_window['attn_mask'] is not the mask window k+1's own masked set builds")

    _, _, time_pred_k = model(patches, attn_mask, bfd['anchor_bg'].float(),
                              bfd['mask_idx'], return_time=True)
    _, _, time_pred_next = model(nw['patches'], nw['attn_mask'],
                                 nw['anchor_bg'].float(), nw['mask_idx'],
                                 return_time=True)
    assert time_pred_k is not None and time_pred_next is not None
    assert time_pred_k.shape == time_pred_next.shape

    l_cross = time_cross_window_consistency_loss(
        time_pred_k, time_pred_next, TIME_PROBE_N_BINS,
        PREDICTION_HORIZON_HOURS, valid=nw['valid'],
    )
    assert l_cross.shape == () and torch.isfinite(l_cross), \
        f"L_cross must be a finite scalar, got {l_cross}"

    model.zero_grad()
    l_cross.backward()

    # the cross-window gradient must reach both the probe head and the trunk
    th_grad = model.time_head[0].weight.grad
    pe_grad = model.patch_embed.weight.grad
    probe_ok = th_grad is not None and torch.isfinite(th_grad).all() \
        and int(torch.count_nonzero(th_grad)) > 0
    trunk_ok = pe_grad is not None and torch.isfinite(pe_grad).all() \
        and int(torch.count_nonzero(pe_grad)) > 0
    assert probe_ok, "L_cross produced no finite gradient on the probe head"
    assert trunk_ok, "L_cross must co-train the trunk (finite patch_embed grad expected)"

    # the witness is a finite per-sample hour deviation
    xwin = time_cross_window_jump_hours(
        time_pred_k, time_pred_next, TIME_PROBE_N_BINS, PREDICTION_HORIZON_HOURS)
    assert xwin.shape == (B,) and torch.isfinite(xwin).all()
    masked_mean = float(xwin.detach()[nw['valid']].mean())

    print(f"\n[DUMP] xwin smoke | L_cross={float(l_cross.detach()):.4e} finite; "
          f"probe+trunk grads finite/nonzero; tod_xwin_jump_h~{masked_mean:.3f} h")
