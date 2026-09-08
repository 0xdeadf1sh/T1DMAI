"""Gradient correctness for the hand-rolled soft-DTW / DILATE autograd (dilate.py).

A wrong backward stays invisible: forward can be finite while gradient is wrong. DILATE
preserves fp64, so fp64 gradcheck is meaningful; Triton is checked against it separately.
"""
import pytest
import torch

import dilate
from dilate import SoftDTWBatch, dilate_loss

import risk_loss
from risk_loss import risk_total_loss, KendallGalWeighting


def test_softdtw_gradcheck():
    torch.manual_seed(0)
    cost = torch.rand(2, 4, 4, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda c: SoftDTWBatch.apply(c, 1.0), (cost,),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )
    print("[DUMP] softdtw gradcheck:", ok)
    assert ok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton sweep is CUDA-only")
def test_softdtw_triton_matches_reference():
    """H sweeps the dispatched buckets (L * PATCH_SIZE) plus both diagonal-walk
    boundaries: H = 1, and an H whose next power of two overshoots the diagonal."""
    from config import DILATE_GAMMA, MASK_SPAN_LENGTHS, PATCH_SIZE

    assert dilate._HAVE_TRITON, "triton missing: the CUDA fp32 path is untested"
    torch.manual_seed(0)
    # dilate_loss rejects H=0/B=0, but a direct apply() reaches them; both sweeps must agree.
    assert not dilate._use_triton(torch.rand(3, 0, 0, device="cuda"))
    empty = SoftDTWBatch.apply(torch.rand(0, 6, 6, device="cuda"), DILATE_GAMMA)
    assert empty.shape == (0,)

    horizons = sorted({1, 2, 5} | {L * PATCH_SIZE for L in MASK_SPAN_LENGTHS})
    worst_val = worst_grad = 0.0
    for H in horizons:
        for B in (1, 7, 64):
            base = torch.rand(B, H, H, device="cuda") * 8.0
            ref_c = base.clone().requires_grad_(True)
            tri_c = base.clone().requires_grad_(True)
            assert not dilate._use_triton(ref_c.double())
            assert dilate._use_triton(tri_c)

            # fp64 selects the eager sweep, so the arms differ only in which sweep filled R and E.
            ref_v = SoftDTWBatch.apply(ref_c.double(), DILATE_GAMMA).float()
            tri_v = SoftDTWBatch.apply(tri_c, DILATE_GAMMA)
            g = torch.randn(B, device="cuda")
            ref_v.backward(g)
            tri_v.backward(g)

            worst_val = max(worst_val, float((ref_v - tri_v).detach().abs().max()))
            worst_grad = max(worst_grad, float((ref_c.grad - tri_c.grad).abs().max()))
    print(f"[DUMP] triton vs reference: max |Δvalue|={worst_val:.3e} "
          f"max |Δgrad|={worst_grad:.3e}")
    assert worst_val < 1e-4
    assert worst_grad < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton sweep is CUDA-only")
def test_softdtw_triton_propagates_nan():
    """A non-finite cost reaches the value; train.py's isfinite guard handles it."""
    from config import DILATE_GAMMA

    cost = torch.rand(2, 6, 6, device="cuda")
    cost[0, 3, 3] = float("nan")
    value = SoftDTWBatch.apply(cost, DILATE_GAMMA)
    print("[DUMP] triton nan propagation:", value.tolist())
    assert bool(torch.isnan(value[0]))
    assert bool(torch.isfinite(value[1]))


def test_tdi_finite_difference_matches_exact_directional_derivative():
    """TDI is a one-sided FD of <A, Ω>, A = ∂sDTW/∂C; it must converge as eps shrinks."""
    import dilate
    from dilate import _pairwise_sq_cost, _omega_distance
    torch.manual_seed(0)
    B, H, gamma = 3, 6, 1.0
    m = torch.randn(B, H, dtype=torch.float64)
    y = torch.randn(B, H, dtype=torch.float64)

    cost = _pairwise_sq_cost(m, y).requires_grad_(True)
    (A,) = torch.autograd.grad(SoftDTWBatch.apply(cost, gamma).sum(), cost)
    omega = _omega_distance(H, m.device, m.dtype)
    tdi_exact = float((A * omega.unsqueeze(0)).sum(dim=(1, 2)).mean())

    errs = [abs(float(dilate.dilate_loss(m, y, gamma=gamma, tdi_fd_eps=eps)[2]) - tdi_exact)
            for eps in (0.1, 0.01, 0.001)]
    print(f"[DUMP] TDI FD vs exact <A,Ω>={tdi_exact:.6f} errs(eps=.1,.01,.001)={errs}")
    assert errs[2] < errs[0], "FD bias must shrink as eps shrinks"
    assert errs[2] < 1e-2, "FD TDI must approach the exact directional derivative"


def test_dilate_loss_gradcheck_median():
    torch.manual_seed(0)
    m = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 5, dtype=torch.float64)
    ok = torch.autograd.gradcheck(
        lambda mm: dilate_loss(mm, y, gamma=1.0)[0], (m,),
        eps=1e-6, atol=1e-3, rtol=1e-2,
    )
    print("[DUMP] dilate_loss gradcheck:", ok)
    assert ok


def test_dilate_divergence_zero_at_match():
    y = torch.randn(2, 6, dtype=torch.float64)
    m = y.clone().requires_grad_(True)
    _loss, shape, _tdi = dilate_loss(m, y, gamma=1.0)
    shape_val = float(shape.detach())
    shape.backward()
    gnorm = float(m.grad.norm())
    print(f"[DUMP] dilate divergence at m==y: shape={shape_val:.3e} grad_norm={gnorm:.3e}")
    assert abs(shape_val) < 1e-6
    assert gnorm < 1e-5


# risk_loss.py centres median/y_risk over (P*S) before DILATE; DC gradient ~0, L_Q owns level.

def test_dilate_dc_invariant_after_centring():
    torch.manual_seed(0)
    m = torch.randn(3, 7, dtype=torch.float64)
    y = torch.randn(3, 7, dtype=torch.float64)

    def centred_dilate(mm: torch.Tensor) -> torch.Tensor:
        mc = mm - mm.mean(dim=1, keepdim=True)
        yc = y - y.mean(dim=1, keepdim=True)
        return dilate_loss(mc, yc, gamma=1.0)[0]

    base = float(centred_dilate(m))
    shifted = float(centred_dilate(m + 4.2))  # bulk DC shift, removed by centring
    assert abs(base - shifted) < 1e-9, (
        f"centred DILATE must be DC-invariant: {base} vs {shifted}")

    m_req = m.clone().requires_grad_(True)
    mc = m_req - m_req.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    dilate_loss(mc, yc, gamma=1.0)[0].backward()
    dc_grad = m_req.grad.sum(dim=1).abs().max()  # projection onto the all-ones dir
    print(f"[DUMP] dilate DC-invariance | Δloss={abs(base - shifted):.2e} "
          f"|grad·1|max={float(dc_grad):.2e}")
    assert float(dc_grad) < 1e-9, "centred DILATE DC/level gradient must vanish"


def _risk_inputs(B: int = 2):
    """``(head_raw, q_tau, median, true_bg_mgdl)`` in the head's risk-space contract."""
    import config
    from utils import assemble_quantiles
    P, S = config.PREDICTION_PATCHES, config.PATCH_SIZE
    torch.manual_seed(0)
    head_raw = torch.randn(B, P, S, 1 + 2 * config.N_SPREADS, requires_grad=True)
    last_bg = torch.full((B,), 120.0)
    q_tau, median = assemble_quantiles(head_raw, last_bg)
    true_bg = torch.full((B, P, S), 110.0) + 20.0 * torch.randn(B, P, S)
    true_bg = true_bg.clamp(40.0, 300.0)
    return head_raw, q_tau, median, true_bg


def test_risk_loss_dilate_centring_drops_median_dc_gradient():
    """L_D gradients nothing along the median's DC direction; the uncentred pinball
    term is what pins the level."""
    import config
    from utils import assemble_quantiles
    P, S = config.PREDICTION_PATCHES, config.PATCH_SIZE

    torch.manual_seed(1)
    head_raw = torch.randn(2, P, S, 1 + 2 * config.N_SPREADS)
    last_bg = torch.full((2,), 120.0)
    q_tau, median = assemble_quantiles(head_raw, last_bg)
    true_bg = torch.full((2, P, S), 130.0)

    dc = torch.zeros((), requires_grad=True)
    median_shifted = median + dc  # pure DC perturbation of the median trajectory

    # reconstruct just the centred-DILATE path risk_loss runs
    b, p, s = median_shifted.shape
    m_flat = median_shifted.reshape(b, p * s)
    y_flat = risk_loss.kovatchev_f_target(true_bg).reshape(b, p * s)
    m_flat = m_flat - m_flat.mean(dim=1, keepdim=True)
    y_flat = y_flat - y_flat.mean(dim=1, keepdim=True)
    loss_D = dilate_loss(m_flat, y_flat, alpha=config.DILATE_ALPHA,
                         gamma=config.DILATE_GAMMA)[0]
    loss_D.backward()
    print(f"[DUMP] risk_loss DILATE | dL_D/dDC={float(dc.grad):.2e} (centring -> ~0)")
    assert abs(float(dc.grad)) < 1e-6, "centred DILATE must not gradient the median DC"


def test_dilate_loss_propagates_nan_without_raising():
    """A NaN median must surface as a NaN loss for train.py's isfinite / EMA-restore
    guard to catch, never as an AssertionError inside the DP."""
    m = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 6, dtype=torch.float64)
    with torch.no_grad():
        m[0, 2] = float('nan')
    loss, shape, tdi = dilate_loss(m, y, gamma=1.0)  # must NOT raise
    assert not torch.isfinite(loss), (
        "a NaN median must propagate to a non-finite loss, not be asserted away")
    # backward through the NaN must not raise either
    loss_val = float(loss.detach())
    loss.backward()
    print(f"[DUMP] dilate NaN propagation | loss={loss_val} "
          f"finite={bool(torch.isfinite(loss.detach()))} (no raise) ✓")


def test_risk_total_loss_propagates_nan_to_guard():
    head_raw, q_tau, median, true_bg = _risk_inputs()
    median = median.clone()
    with torch.no_grad():
        median[0, 0, 0] = float('nan')
    total, comps = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting())
    assert not torch.isfinite(total), "NaN median must reach a non-finite total, not raise"
    print(f"[DUMP] risk_total_loss NaN | total={float(total.detach())} (propagated, no raise) ✓")
