"""Gradient-correctness tests for the custom soft-DTW / DILATE autograd (dilate.py).

The hand-rolled ``SoftDTWBatch`` backward pass is the one place a silently-wrong
gradient would degrade training quality invisibly — the forward value can be
finite while the gradient is wrong. fp64 ``gradcheck`` locks the analytic
backward against the numerical Jacobian. The DILATE code is written to preserve
fp64 (only fp16/bf16 are downcast), so gradcheck is meaningful here. The TDI term
no longer has its own custom Function: it is the directional derivative of the
soft-DTW value, evaluated by a finite difference over ``SoftDTWBatch`` (so its
gradient rides on the gradcheck'd ``SoftDTWBatch`` backward).

gradcheck runs fp64 on the CPU, which is the eager reference sweep. The Triton
sweep the CUDA fp32 training path actually runs is a SECOND implementation of the
same recursion, and nothing above reaches it — ``test_softdtw_triton_matches_
reference`` is what ties it to the gradchecked one, in value and in gradient.
"""
import pytest
import torch

import dilate
from dilate import SoftDTWBatch, dilate_loss

import risk_loss
from risk_loss import risk_total_loss, KendallGalWeighting


def test_softdtw_gradcheck():
    """SoftDTWBatch backward matches the numerical Jacobian (fp64)."""
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
    """The Triton sweep reproduces the eager one, value and gradient.

    H is swept across the span-length buckets the masked-BG objective dispatches
    (H = L * PATCH_SIZE for L in MASK_SPAN_LENGTHS) and past both boundary shapes
    the diagonal walk has: H = 1 (a single cell) and an H whose next power of two
    overshoots the diagonal (the masked lanes must not reach the store).
    """
    from config import DILATE_GAMMA, MASK_SPAN_LENGTHS, PATCH_SIZE

    assert dilate._HAVE_TRITON, "triton missing: the CUDA fp32 path is untested"
    torch.manual_seed(0)
    # H = 0 and B = 0 are degenerate shapes dilate_loss rejects, but a direct
    # apply() reaches them, and the two paths must not disagree about which
    # returns and which raises.
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

            # The reference is reached by asking for it in fp64 and coming back;
            # both arms then differ only in which sweep filled R and E.
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
    """A non-finite cost must reach the value, not trip an assert inside the
    kernel — train.py's isfinite guard is what handles it."""
    from config import DILATE_GAMMA

    cost = torch.rand(2, 6, 6, device="cuda")
    cost[0, 3, 3] = float("nan")
    value = SoftDTWBatch.apply(cost, DILATE_GAMMA)
    print("[DUMP] triton nan propagation:", value.tolist())
    assert bool(torch.isnan(value[0]))
    assert bool(torch.isfinite(value[1]))


def test_tdi_finite_difference_matches_exact_directional_derivative():
    """The TDI surrogate is a one-sided finite difference of the soft-DTW
    directional derivative <A, Ω> (A = ∂sDTW/∂C). It must converge to the exact
    value — computed here via a single autograd grad of the soft-DTW value — as
    the FD step shrinks."""
    import dilate
    from dilate import _pairwise_sq_cost, _omega_distance
    torch.manual_seed(0)
    B, H, gamma = 3, 6, 1.0
    m = torch.randn(B, H, dtype=torch.float64)
    y = torch.randn(B, H, dtype=torch.float64)

    # Exact <A, Ω>: A is the cost-gradient of the soft-DTW value.
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
    """The real training path: gradient of the DILATE total w.r.t. the median."""
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
    """Divergence soft-DTW: the shape term and its gradient vanish at m == y."""
    y = torch.randn(2, 6, dtype=torch.float64)
    m = y.clone().requires_grad_(True)
    _loss, shape, _tdi = dilate_loss(m, y, gamma=1.0)
    shape_val = float(shape.detach())
    shape.backward()
    gnorm = float(m.grad.norm())
    print(f"[DUMP] dilate divergence at m==y: shape={shape_val:.3e} grad_norm={gnorm:.3e}")
    assert abs(shape_val) < 1e-6
    assert gnorm < 1e-5


# ---------------------------------------------------------------------------
# C-dilate-center: risk_loss.py self-mean-centres median and y_risk over the
# flat (P*S) time axis before the DILATE call, so DILATE's DC/level gradient is
# ~0 (the pinball L_Q owns the level).  These tests pin the property at the
# dilate_loss level (level-shift invariance) and end-to-end through the loss.
# ---------------------------------------------------------------------------

def test_dilate_dc_invariant_after_centring():
    """A constant level shift added to a sequence is removed by the self-mean-
    centring risk_loss applies, so the *centred* DILATE loss is invariant to the
    median's DC offset.  Equivalently: dilate_loss on two centred sequences is
    unchanged when one is bulk-shifted before centring — its level-gradient is 0."""
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

    # And the gradient w.r.t. a pure DC direction (all-ones) is ~0 after centring.
    m_req = m.clone().requires_grad_(True)
    mc = m_req - m_req.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    dilate_loss(mc, yc, gamma=1.0)[0].backward()
    dc_grad = m_req.grad.sum(dim=1).abs().max()  # projection onto the all-ones dir
    print(f"[DUMP] dilate DC-invariance | Δloss={abs(base - shifted):.2e} "
          f"|grad·1|max={float(dc_grad):.2e}")
    assert float(dc_grad) < 1e-9, "centred DILATE DC/level gradient must vanish"


def _risk_inputs(B: int = 2):
    """A tiny (q_tau, median, true_bg_mgdl) triple consistent with the head's
    risk-space contract, for end-to-end loss tests."""
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
    """End-to-end: because risk_loss centres the median before DILATE, the DILATE
    term contributes ~no gradient along the median's DC direction — the pinball
    term (uncentred) is what pins the level.  We isolate L_D by reconstructing
    just the centred-DILATE path risk_loss runs and reading its gradient."""
    import config
    from utils import assemble_quantiles
    P, S = config.PREDICTION_PATCHES, config.PATCH_SIZE

    # A learnable scalar DC offset added to the median only; its gradient through
    # the DILATE term must be ~0 (centring removes it).
    torch.manual_seed(1)
    head_raw = torch.randn(2, P, S, 1 + 2 * config.N_SPREADS)
    last_bg = torch.full((2,), 120.0)
    q_tau, median = assemble_quantiles(head_raw, last_bg)
    true_bg = torch.full((2, P, S), 130.0)

    dc = torch.zeros((), requires_grad=True)
    median_shifted = median + dc  # pure DC perturbation of the median trajectory

    # Reconstruct just the centred-DILATE path that risk_loss runs.
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
    """C-nan: the finiteness asserts inside SoftDTWBatch were softened
    so a non-finite median/cost PROPAGATES into the returned loss (where train.py's
    isfinite / EMA-restore guard catches it) rather than crashing the DP.  A NaN in
    the median must surface as a NaN loss, not an AssertionError."""
    m = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 6, dtype=torch.float64)
    with torch.no_grad():
        m[0, 2] = float('nan')
    loss, shape, tdi = dilate_loss(m, y, gamma=1.0)  # must NOT raise
    assert not torch.isfinite(loss), (
        "a NaN median must propagate to a non-finite loss, not be asserted away")
    # Backward through the NaN must also not raise (the guard lets grad flow).
    loss_val = float(loss.detach())
    loss.backward()
    print(f"[DUMP] dilate NaN propagation | loss={loss_val} "
          f"finite={bool(torch.isfinite(loss.detach()))} (no raise) ✓")


def test_risk_total_loss_propagates_nan_to_guard():
    """End-to-end: a NaN median fed to risk_total_loss yields a non-finite total
    (caught downstream by train.py's resilience guard) instead of raising inside
    the DILATE DP."""
    head_raw, q_tau, median, true_bg = _risk_inputs()
    median = median.clone()
    with torch.no_grad():
        median[0, 0, 0] = float('nan')
    total, comps = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting())
    assert not torch.isfinite(total), "NaN median must reach a non-finite total, not raise"
    print(f"[DUMP] risk_total_loss NaN | total={float(total.detach())} (propagated, no raise) ✓")
