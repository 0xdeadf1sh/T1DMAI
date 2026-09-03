"""Two silent failures of the masked-BG objective.

Padded head slots are 41.8% of the head's output on the average sample; drop
``valid`` on the loss path and they train against patch 0's BG behind a plausible
anchor, every shape and every loss curve looking ordinary.

``dilate_loss`` reduces with ``.mean()``, so an empty ``(0, H)`` bucket returns NaN
with no exception, and NaN flows past ``val_total < best_val_loss``, ending the run
with no best checkpoint.
"""

import numpy as np
import pytest
import torch

from config import (MAX_MASKED_PATCHES, MASK_SPAN_LENGTHS, PATCH_SIZE,
                    PREDICTION_PATCHES, QUANTILE_LEVELS)
from data import sample_mask_spans
from model import T1DMAI
from risk_loss import KendallGalWeighting, pinball_loss, risk_total_loss
from tests.forward_inputs import masked_set_inputs, right_edge_inputs


def _targets(B: int, M: int = MAX_MASKED_PATCHES, seed: int = 0) -> torch.Tensor:
    """``(B, M, PATCH_SIZE)`` raw mg/dL targets, one row per head slot."""
    g = torch.Generator().manual_seed(seed)
    return 70.0 + 130.0 * torch.rand(B, M, PATCH_SIZE, generator=g)


def _fan(B: int, M: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A differentiable ascending ``(q_tau, median)`` pair, ``q_tau`` the leaf."""
    g = torch.Generator().manual_seed(seed)
    centre = torch.randn(B, M, PATCH_SIZE, 1, generator=g)
    offs = torch.tensor(QUANTILE_LEVELS) - 0.5
    q_tau = (centre + offs).requires_grad_(True)
    median = q_tau[..., QUANTILE_LEVELS.index(0.5)]
    return q_tau, median


def test_padded_slots_get_exactly_zero_gradient():
    """Padded ``q_tau.grad`` is exactly 0.0, not small; dropping ``valid`` must make
    it non-zero or the assertion has no teeth."""
    B, M = 6, MAX_MASKED_PATCHES
    valid = torch.zeros(B, M, dtype=torch.bool)
    n_valid = [4, 1, 8, 3, 2, 5]
    for b, k in enumerate(n_valid):
        valid[b, :k] = True
    mask_idx = torch.zeros(B, M, dtype=torch.int64)
    for b, k in enumerate(n_valid):
        mask_idx[b, :k] = torch.arange(2, 2 + k)      # one contiguous span per row
    true_bg = _targets(B, M, seed=1)

    q_tau, median = _fan(B, M, seed=2)
    total, comps = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting(),
                                   valid=valid, mask_idx=mask_idx)
    total.backward()
    grad = q_tau.grad
    assert grad is not None
    pad_grad = grad[~valid]
    assert float(pad_grad.abs().max()) == 0.0, (
        f"padded slots carry gradient {float(pad_grad.abs().max()):.3e} — ``valid`` "
        "was dropped somewhere on the loss path")
    assert float(grad[valid].abs().max()) > 0.0, "valid slots must carry gradient"

    # ``mask_idx`` goes with ``valid``: a padded row's index column is not ascending
    # and the loss asserts on that
    q2, m2 = _fan(B, M, seed=2)
    total2, _ = risk_total_loss(q2, m2, true_bg, KendallGalWeighting())
    total2.backward()
    assert float(q2.grad[~valid].abs().max()) > 0.0, (
        "without ``valid`` the padded slots must be supervised — otherwise this "
        "test proves nothing")

    # pinball's denominator is the valid slot count, so a zeroed numerator is not enough
    from utils import kovatchev_f_target
    y_risk = kovatchev_f_target(true_bg)
    q3, _ = _fan(B, M, seed=2)
    lq = pinball_loss(q3, y_risk, QUANTILE_LEVELS, valid=valid)
    lq.backward()
    assert float(q3.grad[~valid].abs().max()) == 0.0
    expected_denom = float(valid.sum()) * PATCH_SIZE * len(QUANTILE_LEVELS)
    per_slot = pinball_loss(q3.detach(), y_risk, QUANTILE_LEVELS, valid=valid)
    dense = pinball_loss(q3.detach()[valid].unsqueeze(0),
                         y_risk[valid].unsqueeze(0), QUANTILE_LEVELS)
    assert torch.allclose(per_slot, dense, atol=1e-6), (
        f"the reduction must divide by the VALID slot count ({expected_denom:.0f}) — "
        f"masked mean {float(per_slot):.6f} != dense mean over the same slots "
        f"{float(dense):.6f}")
    print(f"\n[DUMP] padded gradient | valid counts {n_valid} of M={M}; "
          f"pad grad exactly {float(pad_grad.abs().max()):.1f}; without valid "
          f"{float(q2.grad[~valid].abs().max()):.3e}; loss_Q reduction over valid "
          f"slots only ✓")


def test_mse_ignores_padded_slots(monkeypatch):
    """At ``MSE_ALPHA == 1`` a padded slot's ``q_tau.grad`` is exactly 0.0, and ``loss_M``
    equals the dense MSE over the valid slots alone — the denominator is the valid mass."""
    import config
    from utils import kovatchev_f_target
    monkeypatch.setattr(config, 'MSE_ALPHA', 1.0)
    B, M = 6, MAX_MASKED_PATCHES
    valid = torch.zeros(B, M, dtype=torch.bool)
    n_valid = [4, 1, 8, 3, 2, 5]
    for b, k in enumerate(n_valid):
        valid[b, :k] = True
    mask_idx = torch.zeros(B, M, dtype=torch.int64)
    for b, k in enumerate(n_valid):
        mask_idx[b, :k] = torch.arange(2, 2 + k)
    true_bg = _targets(B, M, seed=1)

    q_tau, median = _fan(B, M, seed=2)
    total, comps = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting(),
                                   valid=valid, mask_idx=mask_idx)
    total.backward()
    assert float(q_tau.grad[~valid].abs().max()) == 0.0
    assert float(q_tau.grad[valid].abs().max()) > 0.0

    y_risk = kovatchev_f_target(true_bg)
    dense = ((median.detach()[valid] - y_risk[valid]) ** 2).mean()
    assert torch.allclose(comps['loss_M'], dense, atol=1e-6), (
        f"loss_M {float(comps['loss_M']):.6f} != dense MSE over valid slots {float(dense):.6f}")
    assert float(comps['loss_D']) == 0.0
    print(f"\n[DUMP] mse padded | valid counts {n_valid}; pad grad exactly 0; "
          f"loss_M={float(comps['loss_M']):.4f} == dense over valid ✓")


def test_mse_all_padded_and_exact_fit_are_exact_zero(monkeypatch):
    """At ``MSE_ALPHA == 1`` an all-padded batch and an exact fit both give ``loss_M``
    exactly 0.0 with a finite gradient — the clamped denominator, not a 0/0."""
    import config
    from utils import kovatchev_f_target
    monkeypatch.setattr(config, 'MSE_ALPHA', 1.0)
    B, M = 3, MAX_MASKED_PATCHES
    true_bg = _targets(B, M, seed=3)

    q_tau, median = _fan(B, M, seed=4)
    total, comps = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting(),
                                   valid=torch.zeros(B, M, dtype=torch.bool))
    assert float(comps['loss_M']) == 0.0
    assert torch.isfinite(total)

    offs = torch.tensor(QUANTILE_LEVELS) - 0.5
    y_risk = kovatchev_f_target(true_bg)
    q_fit = (y_risk.unsqueeze(-1) + offs).requires_grad_(True)
    m_fit = q_fit[..., QUANTILE_LEVELS.index(0.5)]
    assert torch.equal(m_fit, y_risk)
    total_fit, comps_fit = risk_total_loss(q_fit, m_fit, true_bg, KendallGalWeighting())
    assert float(comps_fit['loss_M']) == 0.0
    total_fit.backward()
    assert torch.isfinite(q_fit.grad).all(), "exact fit leaked a non-finite gradient"
    print(f"\n[DUMP] mse exact zero | all-padded loss_M={float(comps['loss_M'])}, "
          f"exact-fit loss_M={float(comps_fit['loss_M'])}, grad finite ✓")


def test_dense_defaults_reproduce_the_right_edge_case():
    """``valid=None`` and ``mask_idx=None`` are the dense right-edge case.

    They must equal an all-True ``valid`` with the matching ascending ``mask_idx``,
    or every default call site changes meaning instead of breaking.
    """
    B, M = 4, PREDICTION_PATCHES
    true_bg = _targets(B, M, seed=5)
    valid = torch.ones(B, M, dtype=torch.bool)
    mask_idx = torch.arange(M, dtype=torch.int64).expand(B, M).contiguous()

    q_a, m_a = _fan(B, M, seed=6)
    total_a, comps_a = risk_total_loss(q_a, m_a, true_bg, KendallGalWeighting())
    q_b, m_b = _fan(B, M, seed=6)
    total_b, comps_b = risk_total_loss(q_b, m_b, true_bg, KendallGalWeighting(),
                                       valid=valid, mask_idx=mask_idx)
    assert torch.allclose(total_a, total_b, atol=1e-6), \
        f"dense defaults gave {float(total_a.detach())}, explicit gave {float(total_b.detach())}"
    for key in ('loss_Q', 'loss_D', 'loss_D_shape', 'loss_D_tdi'):
        assert torch.allclose(comps_a[key], comps_b[key], atol=1e-6), \
            f"{key} differs between the dense default and the explicit form"
    assert float(comps_a[f'n_spans_L{PREDICTION_PATCHES}']) == B, \
        "the dense default must bucket as one span of PREDICTION_PATCHES per row"

    from utils import kovatchev_f_target
    y_risk = kovatchev_f_target(true_bg)
    assert torch.allclose(
        pinball_loss(q_a.detach(), y_risk, QUANTILE_LEVELS),
        pinball_loss(q_a.detach(), y_risk, QUANTILE_LEVELS, valid=valid),
        atol=1e-7), "pinball's valid=None must equal an all-True valid"
    print(f"\n[DUMP] dense default | valid/mask_idx None == all-True at "
          f"M={PREDICTION_PATCHES}: total {float(total_a.detach()):.6f} ✓")


def test_padded_slot_contents_do_not_move_the_loss():
    """Moving a padded slot's anchor and target leaves the loss and every parameter
    gradient bit-identical.

    The forward's units tripwire reads all ``M``, so a padded anchor is legal mg/dL —
    which is why nothing downstream would flag it being supervised.
    """
    torch.manual_seed(0)
    model = T1DMAI().train()
    n_ctx = 12
    spans_per_row = [[(0, 2), (6, 3)], [(4, 1)], [(n_ctx, PREDICTION_PATCHES)],
                     [(2, 4), (9, 2)]]
    patches, attn, anchor_bg, mask_idx, valid = masked_set_inputs(
        spans_per_row, n_ctx, seed=11)
    true_bg = _targets(len(spans_per_row), seed=3)

    def run(anchor, target):
        model.zero_grad(set_to_none=True)
        q_tau, median = model(patches, attn, anchor, mask_idx)
        total, _ = risk_total_loss(q_tau, median, target, KendallGalWeighting(),
                                   valid=valid, mask_idx=mask_idx)
        total.backward()
        return float(total.detach()), {n: p.grad.clone()
                                       for n, p in model.named_parameters()
                                       if p.grad is not None}

    base_loss, base_grads = run(anchor_bg, true_bg)

    moved_anchor = anchor_bg.clone()
    moved_anchor[~valid] = 260.0                    # legal mg/dL, different value
    moved_target = true_bg.clone()
    moved_target[~valid] = 300.0
    alt_loss, alt_grads = run(moved_anchor, moved_target)

    assert base_loss == alt_loss, (
        f"the loss moved with the padded slots' contents: {base_loss!r} vs {alt_loss!r}")
    for name, g in base_grads.items():
        assert torch.equal(g, alt_grads[name]), \
            f"parameter {name} received gradient from a padded slot"
    print(f"\n[DUMP] padded contents | loss {base_loss:.6f} unchanged and "
          f"{len(base_grads)} parameter grads bit-identical after moving every "
          f"padded anchor/target ✓")


def test_forecast_protocol_is_finite_with_three_empty_buckets():
    """One right-edge span of ``PREDICTION_PATCHES`` leaves buckets L = 1, 2, 3 empty
    in every batch, permanently.

    ``loss_D`` must equal the L = 4 bucket and the empty ones must report zero spans
    rather than dropping out of the log.
    """
    torch.manual_seed(0)
    model = T1DMAI().eval()
    B, n_ctx = 8, 16
    patches, attn, anchor_bg, mask_idx = right_edge_inputs(
        B, n_ctx=n_ctx, M=MAX_MASKED_PATCHES, seed=21)
    valid = torch.zeros(B, MAX_MASKED_PATCHES, dtype=torch.bool)
    valid[:, :PREDICTION_PATCHES] = True
    true_bg = _targets(B, seed=4)

    q_tau, median = model(patches, attn, anchor_bg, mask_idx)
    total, comps = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting(),
                                   valid=valid, mask_idx=mask_idx)

    assert torch.isfinite(total), f"forecast protocol produced {float(total)}"
    for key in ('loss_Q', 'loss_D', 'loss_D_shape', 'loss_D_tdi'):
        assert torch.isfinite(comps[key]), f"{key} is {float(comps[key])}"
    for L in MASK_SPAN_LENGTHS:
        n = float(comps[f'n_spans_L{L}'])
        assert n == (B if L == PREDICTION_PATCHES else 0.0), \
            f"bucket L={L} holds {n} spans under the forecast protocol"
        assert torch.isfinite(comps[f'loss_D_L{L}']), (
            f"empty bucket L={L} reported {float(comps[f'loss_D_L{L}'])} — a (0, H) "
            "dilate_loss call returns NaN with no exception")
    assert torch.allclose(comps['loss_D'], comps[f'loss_D_L{PREDICTION_PATCHES}']), \
        "with one populated bucket loss_D must be that bucket's value"
    assert float(comps['n_masked_mean']) == float(PREDICTION_PATCHES)
    assert float(comps['n_spans_mean']) == 1.0
    print(f"\n[DUMP] forecast protocol | total={float(total.detach()):.4f} finite; buckets "
          f"{{{', '.join(f'L{L}: {float(comps[f'n_spans_L{L}']):.0f}' for L in MASK_SPAN_LENGTHS)}}}; "
          f"loss_D={float(comps['loss_D']):.4f} == loss_D_L{PREDICTION_PATCHES} ✓")


def test_infill_protocol_is_finite_including_empty_buckets():
    """The sampler's own masked sets at B = 8: with ~2 spans per sample a length is
    missing from all 8 rows in a few percent of batches.

    A span-count-weighted mean cannot rescue that — ``0.0 * nan == nan``.
    """
    torch.manual_seed(0)
    model = T1DMAI().eval()
    rng = np.random.default_rng(7)
    B, n_ctx = 8, 16
    T = n_ctx + PREDICTION_PATCHES

    n_batches = 24
    empty_bucket_batches = 0
    totals = []
    for step in range(n_batches):
        spans_per_row = [[(int(s), int(L)) for s, L in sample_mask_spans(T, rng)]
                         for _ in range(B)]
        patches, attn, anchor_bg, mask_idx, valid = masked_set_inputs(
            spans_per_row, n_ctx, seed=100 + step)
        true_bg = _targets(B, seed=200 + step)

        q_tau, median = model(patches, attn, anchor_bg, mask_idx)
        total, comps = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting(),
                                       valid=valid, mask_idx=mask_idx)
        assert torch.isfinite(total), (
            f"infill batch {step} produced {float(total)}; span lengths "
            f"{[[L for _s, L in row] for row in spans_per_row]}")
        assert torch.isfinite(comps['loss_D']) and torch.isfinite(comps['loss_Q'])
        counts = {L: float(comps[f'n_spans_L{L}']) for L in MASK_SPAN_LENGTHS}
        if any(c == 0.0 for c in counts.values()):
            empty_bucket_batches += 1
        for L, c in counts.items():
            assert torch.isfinite(comps[f'loss_D_L{L}']), \
                f"bucket L={L} ({c:.0f} spans) reported a non-finite loss"
        totals.append(float(total.detach()))
        # every masked patch of every row is supervised exactly once
        assert float(comps['n_masked_mean']) == pytest.approx(
            sum(sum(L for _s, L in row) for row in spans_per_row) / B)
    print(f"\n[DUMP] infill protocol | {n_batches} batches at B={B}: all totals "
          f"finite (range [{min(totals):.4f}, {max(totals):.4f}]); "
          f"{empty_bucket_batches} batch(es) had at least one empty DILATE bucket ✓")
