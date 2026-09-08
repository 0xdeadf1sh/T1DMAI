"""The risk-space training stack: pinball, soft-DTW/DILATE, Kendall-Gal weighting,
risk_total_loss, loop plumbing; BG forecast is one quantile head trained by pinball+DILATE+MSE.
"""

import math

import pytest
import torch


def _get_stats():
    import os
    from normalization import compute_normalization_stats, load_normalization_stats, NORM_STATS_FILE
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def test_pinball_rho_tau_values():
    """ρ_τ(a,b) = (a-b)·(τ - 1[a<b]), over- and under-prediction, asymmetry tracking τ."""
    from risk_loss import pinball_loss

    # one patch, one step, one τ level, so the mean IS the per-element ρ
    a = torch.tensor([[[2.0]]])                  # (B,P,S) target
    b_over = torch.tensor([[[[3.0]]]])           # (B,P,S,1)
    b_under = torch.tensor([[[[1.0]]]])

    for tau in (0.1, 0.5, 0.9):
        lo = float(pinball_loss(b_over, a, (tau,)))
        # a-b = -1, a<b True => ρ = (-1)*(τ-1) = 1-τ
        assert abs(lo - (1.0 - tau)) < 1e-6, f"over-pred ρ wrong at τ={tau}: {lo}"
        hi = float(pinball_loss(b_under, a, (tau,)))
        # a-b = +1, a<b False => ρ = (1)*(τ-0) = τ
        assert abs(hi - tau) < 1e-6, f"under-pred ρ wrong at τ={tau}: {hi}"
    # the check loss is non-negative everywhere
    assert float(pinball_loss(b_over, a, (0.5,))) >= 0.0
    print("\n[DUMP] pinball | ρ_τ matches (1-τ) over-pred, τ under-pred ✓")


def test_pinball_gradients():
    """Over-prediction at high τ pulls down, under-prediction at low τ pulls up; the
    gradient stays inside the check-loss subgradient band [-1, 1]."""
    from risk_loss import pinball_loss

    a = torch.tensor([[[0.0]]])
    q = torch.tensor([[[[1.0]]]], requires_grad=True)   # over-prediction
    loss = pinball_loss(q, a, (0.9,))
    loss.backward()
    g = float(q.grad)
    assert math.isfinite(g)
    assert -1.0 - 1e-6 <= g <= 1.0 + 1e-6, f"pinball grad out of subgradient band: {g}"
    # At tau=0.9: d/dq (a-q)(tau-1) = 1-tau = 0.1 > 0, so positive gradient pushes it DOWN.
    assert g > 0.0, f"over-prediction must have positive grad (push down): {g}"
    print(f"\n[DUMP] pinball grad | over-pred @τ=0.9 grad={g:.3f} (push down) ✓")


def test_soft_dtw_grads_and_gamma_sweep():
    """Differentiable and finite across a gamma sweep, even at the worst-case risk pair
    f(BG_CLAMP_MIN) vs f(BG_CLAMP_MAX) over a 24-step horizon.

    That pair is derived through the transform, never a numeral, so it follows the clamp.
    """
    from dilate import SoftDTWBatch, _pairwise_sq_cost
    from config import DILATE_GAMMA

    B, H = 2, 24
    torch.manual_seed(0)
    x = torch.randn(B, H, requires_grad=True)
    y = torch.randn(B, H)

    for gamma in (0.1, DILATE_GAMMA, 1.0, 5.0):
        cost = _pairwise_sq_cost(x, y)
        val = SoftDTWBatch.apply(cost, float(gamma))     # (B,)
        assert val.shape == (B,)
        assert torch.isfinite(val).all(), f"sDTW value non-finite at γ={gamma}"
        g, = torch.autograd.grad(val.sum(), x, retain_graph=False)
        assert torch.isfinite(g).all(), f"sDTW grad non-finite at γ={gamma}"
        x.grad = None

    # the widest cell cost is (f(BG_CLAMP_MAX) - f(BG_CLAMP_MIN))^2
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    from utils import kovatchev_f
    r_lo, r_hi = kovatchev_f(
        torch.tensor([float(BG_CLAMP_MIN), float(BG_CLAMP_MAX)])).tolist()
    xa = torch.full((1, H), r_lo, requires_grad=True)
    ya = torch.full((1, H), r_hi)
    cost = _pairwise_sq_cost(xa, ya)
    assert torch.isfinite(cost).all(), "cell cost must stay finite at the risk extremes"
    val = SoftDTWBatch.apply(cost, float(DILATE_GAMMA))
    assert torch.isfinite(val).all(), "sDTW must stay finite at the risk extremes"
    (gx,) = torch.autograd.grad(val.sum(), xa)
    assert torch.isfinite(gx).all()
    print(f"\n[DUMP] soft_dtw | finite over γ-sweep + worst-case risk pair "
          f"f({BG_CLAMP_MIN:g})={r_lo:.4f} f({BG_CLAMP_MAX:g})={r_hi:.4f} "
          f"cell={float(cost.detach().max()):.4f} sDTW={float(val.detach().max()):.2f} "
          f"max|g|={float(gx.abs().max()):.2f} ✓")


def test_dilate_divergence_zero_at_match():
    """The shape term is the DIVERGENCE ``sDTW(m,y) - ½sDTW(m,m) - ½sDTW(y,y)``, so it
    and its gradient vanish at m == y; a plain ``sDTW(x,x)`` would not."""
    from dilate import dilate_loss
    from config import DILATE_ALPHA, DILATE_GAMMA

    B, H = 3, 12
    torch.manual_seed(1)
    y = torch.randn(B, H)
    m = y.clone().requires_grad_(True)

    loss, shape, tdi = dilate_loss(m, y, alpha=DILATE_ALPHA, gamma=float(DILATE_GAMMA))
    shape_val = float(shape.detach())
    assert abs(shape_val) < 1e-3, f"divergence shape must be ≈0 at m==y, got {shape_val}"
    # at m == y the TDI aligns on the diagonal, so ~0 off-diagonal mass
    g, = torch.autograd.grad(shape, m, retain_graph=True)
    assert g.abs().max() < 1e-2, f"shape grad must vanish at m==y, got {g.abs().max():.3e}"
    print(f"\n[DUMP] dilate | divergence shape={shape_val:.3e}, grad≈0 at m==y ✓")


def test_dilate_flatten_order_sensitive():
    """A P/S transpose must change the loss, which is what pins the
    patch-major/step-minor flatten order."""
    from dilate import dilate_loss
    from risk_loss import _to_patch_major
    from config import DILATE_ALPHA, DILATE_GAMMA

    B, P, S = 1, 4, 6
    # a time-monotone ramp in risk space, distinct per (p, s)
    base = torch.arange(P * S, dtype=torch.float32).reshape(1, P, S) * 0.1
    target = base.clone()
    # the forecast is the target shifted one step in TRUE time order
    pred = base.clone()
    pred[:, :, :] = base[:, :, :] + 0.3

    # the correct patch-major flatten is time-monotone
    m_pm = _to_patch_major(pred)
    y_pm = _to_patch_major(target)
    loss_pm, _, _ = dilate_loss(m_pm, y_pm, alpha=DILATE_ALPHA, gamma=float(DILATE_GAMMA))

    # WRONG order: transposing P/S before flattening scrambles the time axis
    m_wrong = pred.transpose(1, 2).reshape(B, P * S)
    y_wrong = target.transpose(1, 2).reshape(B, P * S)
    loss_wrong, _, _ = dilate_loss(m_wrong, y_wrong, alpha=DILATE_ALPHA, gamma=float(DILATE_GAMMA))

    assert not math.isclose(float(loss_pm), float(loss_wrong), rel_tol=1e-4), (
        f"P/S transpose must change DILATE: {float(loss_pm)} vs {float(loss_wrong)}")
    print(f"\n[DUMP] dilate flatten | patch-major={float(loss_pm):.4f} != "
          f"transposed={float(loss_wrong):.4f} ✓")


def test_to_patch_major_rejects_transpose():
    """The flat output runs time-monotonically: patch p step s at index p*S + s."""
    from risk_loss import _to_patch_major
    B, P, S = 2, 3, 4
    x = torch.arange(B * P * S, dtype=torch.float32).reshape(B, P, S)
    flat = _to_patch_major(x)
    assert flat.shape == (B, P * S)
    for p in range(P):
        for s in range(S):
            assert torch.equal(flat[:, p * S + s], x[:, p, s])
    print("\n[DUMP] _to_patch_major | C-contiguous patch-major flatten verified ✓")


def test_kendall_gal_weighting_combine():
    """risk_total_loss combines 0.5*exp(-2*sQ)*L_Q + sQ + 0.5*exp(-2*sD)*L_DR + sD, where
    L_DR = (1-MSE_ALPHA)*L_D + MSE_ALPHA*L_M.

    At KENDALL_LOGVAR_INIT==0 that reduces to 0.5*L_Q + 0.5*L_DR; log_sigma_* echo the params.
    """
    from risk_loss import risk_total_loss, KendallGalWeighting
    from config import PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES, MSE_ALPHA
    from utils import assemble_quantiles

    B, P, S = 2, PREDICTION_PATCHES, PATCH_SIZE
    torch.manual_seed(0)
    head_raw = torch.randn(B, P, S, 1 + 2 * ((N_QUANTILES - 1) // 2))
    last_bg = torch.full((B,), 120.0)
    q_tau, median = assemble_quantiles(head_raw, last_bg)
    true_bg = torch.full((B, P, S), 130.0)
    true_bg[:, : P // 2] = 70.0

    weighting = KendallGalWeighting()  # inits at KENDALL_LOGVAR_INIT == 0.0
    total, comp = risk_total_loss(q_tau, median, true_bg, weighting)

    # at log-σ == 0: 0.5·exp(0)·L_Q + 0 + 0.5·exp(0)·L_DR + 0 == 0.5(L_Q + L_DR)
    slot = (1.0 - MSE_ALPHA) * comp['loss_D'] + MSE_ALPHA * comp['loss_M']
    expected = 0.5 * comp['loss_Q'] + 0.5 * slot
    assert torch.allclose(total, expected, atol=1e-6), (
        f"Kendall-Gal combine broken at σ=0: {float(total)} != {float(expected)}")
    assert 'loss_smooth' not in comp, "L_smooth penalty must be gone from components"
    assert 'log_sigma_Q' in comp and 'log_sigma_D' in comp, \
        "log-σ components must be present under the learned weighting"
    assert abs(float(comp['log_sigma_Q'])) < 1e-6 and abs(float(comp['log_sigma_D'])) < 1e-6

    # Bumping sigma_Q down-weights L_Q (0.5*exp(-2*sigma) shrinks) and adds the +sigma_Q barrier.
    with torch.no_grad():
        weighting.log_sigma_Q.add_(1.0)
    total2, comp2 = risk_total_loss(q_tau, median, true_bg, weighting)
    slot2 = (1.0 - MSE_ALPHA) * comp2['loss_D'] + MSE_ALPHA * comp2['loss_M']
    exp2 = (0.5 * math.exp(-2.0) * comp2['loss_Q'] + 1.0
            + 0.5 * slot2)
    assert torch.allclose(total2, exp2, atol=1e-6), (
        f"Kendall-Gal combine broken at σ_Q=1: {float(total2)} != {float(exp2)}")
    print(f"\n[DUMP] kendall-gal | total@σ0={float(total):.4f} == 0.5(L_Q+L_DR), "
          f"σ_Q bump tracked ✓")


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_mse_alpha_mixes_dilate_and_mse(monkeypatch, alpha):
    """``MSE_ALPHA`` weights the Kendall-Gal D slot ``(1−α)·L_D + α·L_M``: 0 skips the
    MSE (``loss_M == 0``), 1 skips the soft-DTW (``loss_D`` and every ``loss_D_L{L}`` == 0),
    and ``loss_M`` is the hand-computed risk-space MSE of the median."""
    import config
    from risk_loss import risk_total_loss, KendallGalWeighting
    from utils import assemble_quantiles, kovatchev_f_target

    monkeypatch.setattr(config, 'MSE_ALPHA', alpha)
    B, P, S = 2, config.PREDICTION_PATCHES, config.PATCH_SIZE
    torch.manual_seed(1)
    head_raw = torch.randn(B, P, S, 1 + 2 * ((config.N_QUANTILES - 1) // 2))
    q_tau, median = assemble_quantiles(head_raw, torch.full((B,), 120.0))
    true_bg = 90.0 + 80.0 * torch.rand(B, P, S)

    total, comp = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting())
    y_risk = kovatchev_f_target(true_bg)
    mse_hand = ((median - y_risk) ** 2).mean()

    if alpha == 0.0:
        assert float(comp['loss_M']) == 0.0
        assert float(comp['loss_D']) > 0.0
    elif alpha == 1.0:
        assert float(comp['loss_D']) == 0.0
        assert all(float(comp[f'loss_D_L{L}']) == 0.0 for L in config.MASK_SPAN_LENGTHS)
        assert torch.allclose(comp['loss_M'], mse_hand, atol=1e-6)
    else:
        assert float(comp['loss_D']) > 0.0
        assert torch.allclose(comp['loss_M'], mse_hand, atol=1e-6)
    slot = (1.0 - alpha) * comp['loss_D'] + alpha * comp['loss_M']
    expected = 0.5 * comp['loss_Q'] + 0.5 * slot
    assert torch.allclose(total, expected, atol=1e-6), (
        f"α={alpha}: total {float(total)} != 0.5·L_Q + 0.5·((1−α)·L_D + α·L_M) {float(expected)}")
    assert torch.isfinite(total)
    # the span counters are logged from the bucket table, not from the soft-DTW loop
    assert float(comp['n_spans_mean']) == 1.0 and float(comp['n_masked_mean']) == float(P), (
        f"α={alpha}: n_spans_mean {float(comp['n_spans_mean'])} n_masked_mean "
        f"{float(comp['n_masked_mean'])} — the α=1 skip dropped the counters")
    print(f"\n[DUMP] mse_alpha={alpha} | L_Q={float(comp['loss_Q']):.4f} "
          f"L_D={float(comp['loss_D']):.4f} L_M={float(comp['loss_M']):.4f} "
          f"total={float(total):.4f} ✓")


def test_risk_total_loss_f_applied_once_and_finite():
    """The target arrives as mg/dL, NOT f-transformed; ``kovatchev_f_target`` is applied
    exactly once inside."""
    from risk_loss import risk_total_loss, KendallGalWeighting
    from config import (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES, QUANTILE_LEVELS)
    from utils import assemble_quantiles
    import T1DMSIM.simulator as sim

    B, P, S = 2, PREDICTION_PATCHES, PATCH_SIZE
    torch.manual_seed(0)
    # Genuine ascending quantiles off a head_raw + anchor, so q_tau/median are self-consistent.
    head_raw = torch.randn(B, P, S, 1 + 2 * ((N_QUANTILES - 1) // 2), requires_grad=True)
    last_bg = torch.full((B,), 120.0)
    q_tau, median = assemble_quantiles(head_raw, last_bg)

    # mg/dL target inside the physical band, not f'd
    true_bg = torch.full((B, P, S), 130.0)
    true_bg[:, : P // 2] = 70.0
    assert (true_bg >= sim.BG_CLAMP_MIN).all(), "target must be mg/dL"

    total, comp = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting())
    assert total.ndim == 0 and torch.isfinite(total), "total must be a finite scalar"
    for k in ('loss_Q', 'loss_D', 'loss_D_shape', 'loss_D_tdi',
              'log_sigma_Q', 'log_sigma_D'):
        assert k in comp, f"missing logging component {k}"
        assert torch.isfinite(comp[k]).all()
    assert 'loss_smooth' not in comp, "L_smooth penalty must be gone from components"

    total.backward()
    assert head_raw.grad is not None and torch.isfinite(head_raw.grad).all(), \
        "gradient must flow to the quantile head"
    print(f"\n[DUMP] risk_total_loss | total={float(total.detach()):.4f} "
          f"L_Q={float(comp['loss_Q']):.4f} L_D={float(comp['loss_D']):.4f} ✓")


def test_risk_total_loss_lower_when_matched():
    """Lower when the median tracks the f'd target than when it is anti-signed."""
    from risk_loss import risk_total_loss, KendallGalWeighting
    from config import PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES
    from utils import assemble_quantiles, kovatchev_f

    B, P, S = 2, PREDICTION_PATCHES, PATCH_SIZE
    weighting = KendallGalWeighting()
    last_bg = torch.full((B,), 120.0)
    # Ramp the truth: a flat truth collapses both signs onto the same anchor at zero delta.
    true_bg = torch.full((B, P, S), 120.0)
    for p in range(P):
        for s in range(S):
            true_bg[:, p, s] = 120.0 + 1.0 * (p * S + s)

    n_spread = (N_QUANTILES - 1) // 2
    anchor = kovatchev_f(last_bg)  # (B,)

    def _loss_for(delta_sign: float) -> float:
        head = torch.zeros(B, P, S, 1 + 2 * n_spread)
        # Per-step median delta = sign * (f(true_step) - f(last)): +1 tracks truth, -1 anti-tracks.
        for p in range(P):
            for s in range(S):
                tgt_risk = kovatchev_f(true_bg[:, p, s])
                head[:, p, s, 0] = delta_sign * (tgt_risk - anchor)
        q, m = assemble_quantiles(head, last_bg)
        tot, _ = risk_total_loss(q, m, true_bg, weighting)
        return float(tot.detach())

    matched = _loss_for(+1.0)
    anti = _loss_for(-1.0)
    assert matched < anti, f"matched median must score below anti: {matched} >= {anti}"
    print(f"\n[DUMP] risk_total_loss | matched={matched:.4f} < anti={anti:.4f} ✓")


def test_excursion_target_declines_with_horizon():
    """``max(floor, base - slope*(h/30-1))``, the display-only colouring rule for the
    Excursions-by-Horizon rows."""
    from train import _excursion_target, _excursion_bucket_horizons, PREDICTION_PATCHES

    assert _excursion_target((90.0, 13.0, 50.0), 30) == 90.0    # base @30
    assert _excursion_target((90.0, 13.0, 50.0), 60) == 77.0    # one 30-min step
    assert _excursion_target((90.0, 13.0, 50.0), 120) == 51.0   # three steps
    assert _excursion_target((90.0, 13.0, 50.0), 1800) == 50.0  # floor clamps far horizon

    hs = _excursion_bucket_horizons(PREDICTION_PATCHES)
    assert len(hs) >= 2 and hs[0] == 30
    spec = (90.0, 13.0, 50.0)
    vals = [_excursion_target(spec, h) for h in hs]
    assert all(a >= b for a, b in zip(vals, vals[1:])), f"must be non-increasing: {vals}"
    assert vals[0] > vals[-1], f"must net-decline across the window: {vals}"
    print(f"\n[DUMP] excursion_target | {vals} declines ✓")


def test_offset_sampler_basic():
    from train import _OffsetSampler

    sampler = _OffsetSampler(total_len=100, offset=40)
    indices = list(sampler)

    assert len(sampler) == 60, f"Expected length 60, got {len(sampler)}"
    assert indices[0] == 40, f"First index should be 40, got {indices[0]}"
    assert indices[-1] == 99, f"Last index should be 99, got {indices[-1]}"
    assert indices == list(range(40, 100))
    print(f"\n[DUMP] offset_sampler | first 5 indices: {indices[:5]}, last 5: {indices[-5:]}")


def test_offset_sampler_zero():
    from train import _OffsetSampler

    sampler = _OffsetSampler(total_len=50, offset=0)
    indices = list(sampler)

    assert len(sampler) == 50
    assert indices == list(range(50))


def test_resume_data_alignment():
    """After a resume the dataset is indexed from the resumed step, not step 0."""
    from data import T1DMDataset
    from train import _OffsetSampler
    from utils import compute_patient_seed

    master_seed = 99
    total_steps = 30
    batch_size = 4
    resume_step = 15

    stats = _get_stats()
    dataset = T1DMDataset(
        master_seed=master_seed,
        total_steps=total_steps,
        batch_size=batch_size,
        normalization_stats=stats,
    )

    sampler = _OffsetSampler(len(dataset), offset=resume_step * batch_size)
    indices = list(sampler)
    assert indices[0] == resume_step * batch_size
    assert indices[-1] == len(dataset) - 1

    expected_resume = compute_patient_seed(master_seed, resume_step, 0)
    expected_step0 = compute_patient_seed(master_seed, 0, 0)
    assert expected_resume != expected_step0
    print(f"\n[DUMP] resume_align | step {resume_step} seed={expected_resume} != step 0 seed={expected_step0}")


def test_checkpoint_save_load():
    from model import T1DMAI
    from tests.forward_inputs import right_edge_inputs
    import os

    model = T1DMAI()
    model.eval()

    B = 1
    patches, mask, anchor_bg, mask_idx = right_edge_inputs(B, seed=0)

    with torch.no_grad():
        q1, m1 = model(patches, mask, anchor_bg, mask_idx)

    path = '/tmp/test_checkpoint_t1dmai.pt'
    torch.save({'model_state_dict': model.state_dict()}, path)

    model2 = T1DMAI()
    checkpoint = torch.load(path, weights_only=True)
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.eval()

    with torch.no_grad():
        q2, m2 = model2(patches, mask, anchor_bg, mask_idx)

    assert torch.allclose(q1, q2, atol=1e-6), "Loaded model produces different q_tau"
    assert torch.allclose(m1, m2, atol=1e-6), "Loaded model produces different median"

    os.remove(path)
    print("\n[DUMP] checkpoint | save/load verified identical")


def test_muon_optimizer_step():
    from model import T1DMAI
    from muon import Muon
    from tests.forward_inputs import right_edge_inputs

    model = T1DMAI()

    initial_params = {name: p.clone() for name, p in model.named_parameters()}

    B = 1
    patches, mask, anchor_bg, mask_idx = right_edge_inputs(B, seed=1)

    q_tau, median = model(patches, mask, anchor_bg, mask_idx)
    loss = q_tau.sum() + median.sum()
    loss.backward()

    muon_params = []
    adam_params = []
    for _name, p in model.named_parameters():
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    muon_opt = Muon(muon_params, lr=0.02, momentum=0.95)
    adam_opt = torch.optim.AdamW(adam_params, lr=3e-4)

    muon_opt.step()
    adam_opt.step()

    n_changed = 0
    for name, p in model.named_parameters():
        if not torch.allclose(p, initial_params[name], atol=1e-8):
            n_changed += 1

    print(f"\n[DUMP] optimizer | {n_changed} / {len(initial_params)} parameters changed")
    assert n_changed > 0, "No parameters were updated"


def test_training_10_steps():
    from train import train

    losses = train(
        total_steps=10,
        batch_size=4,
        master_seed=42,
        num_workers=0,
        log_interval=1,
        checkpoint_interval=999999,
        validation_interval=999999,
    )

    assert len(losses) == int(10), f"expected 10 step losses, got {len(losses)}"
    assert all(not math.isnan(l) for l in losses), "NaN loss during training"
    assert all(not math.isinf(l) for l in losses), "Inf loss during training"

    print(f"\n[DUMP] training_10step | losses: {[f'{l:.4f}' for l in losses]}")
    print(f"[DUMP] training_10step | first loss: {losses[0]:.4f}, last loss: {losses[-1]:.4f}")


def test_training_100_steps_loss_trend():
    """Stability, not convergence: on-the-fly data trains a fresh patient every step."""
    from train import train

    losses = train(
        total_steps=100,
        batch_size=4,
        master_seed=42,
        num_workers=0,
        log_interval=10,
        checkpoint_interval=999999,
        validation_interval=999999,
    )

    assert len(losses) == 100
    assert all(math.isfinite(l) for l in losses), "non-finite loss during training"

    def _win(a: int, b: int) -> float:
        seg = losses[a:b]
        return sum(seg) / len(seg)

    first = _win(0, 25)
    last = _win(75, 100)
    worst = max(_win(i, i + 25) for i in range(0, 76, 5))
    print(f"\n[DUMP] training_100step | first25={first:.4f} last25={last:.4f} "
          f"worst25={worst:.4f}")

    # Weighted loss can go negative through the +log_sigma barriers, so compare drift in magnitude.
    assert math.isfinite(worst) and math.isfinite(last)
    assert worst < abs(first) + max(2.0, 1.5 * abs(first)), (
        f"loss spiked mid-run: worst25={worst:.4f} vs first25={first:.4f}")


def test_weight_decay_schedule_correction():
    """AdamC (arXiv 2506.02285): the normalized Muon matrices have their weight decay
    rescaled by gamma_t/gamma_max each step, so the effective decay is
    gamma_t^2/gamma_max * lambda. The output-head, AdamW and Kendall groups are untouched."""
    from model import T1DMAI
    from risk_loss import KendallGalWeighting
    from train import _build_optimizers, _update_lr
    from config import MUON_WEIGHT_DECAY, ADAM_WEIGHT_DECAY

    model = T1DMAI()
    weighting = KendallGalWeighting()
    muon_opt, adam_opt = _build_optimizers(
        model, weighting, muon_lr=0.02, adam_lr=0.003, muon_momentum=0.95)

    # locate the corrected (normalized) and output Muon groups by their flag
    corrected_group = next(g for g in muon_opt.param_groups if g.get('wd_corrected', False))
    output_group = next(g for g in muon_opt.param_groups if not g.get('wd_corrected', False))

    def _in(group: dict, param: torch.Tensor) -> bool:
        return any(p is param for p in group['params'])

    # Output projections excluded from the correction; a matrix like patch_embed is included.
    assert _in(output_group, model.bg_head[-1].weight)
    assert not _in(corrected_group, model.bg_head[-1].weight)
    if model.time_head is not None:
        assert _in(output_group, model.time_head[-1].weight)
        assert not _in(corrected_group, model.time_head[-1].weight)
    assert _in(corrected_group, model.patch_embed.weight)
    assert not _in(output_group, model.patch_embed.weight)

    # the AdamW groups: the decayed embedding/1D one and the wd=0 Kendall one
    adam_decayed = next(g for g in adam_opt.param_groups if g['weight_decay'] != 0.0)
    kendall_group = next(g for g in adam_opt.param_groups if g['weight_decay'] == 0.0)

    warmup_steps, total_steps, lr_min_ratio = 2000, 100000, 0.001

    def ratio(step: int) -> float:
        # byte-identical to _update_lr's warmup + cosine multiplier
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))

    for step in (0, 1000, 2000, 20000, 99000):
        _update_lr(muon_opt, adam_opt, step, 0.02, 0.003, warmup_steps, total_steps,
                   lr_min_ratio, wd_correction=True)
        r = ratio(step)
        wd = corrected_group['weight_decay']
        eff = corrected_group['lr'] * corrected_group['weight_decay']
        print(f"[DUMP] wd_correction | step={step:6d} ratio={r:.6f} "
              f"corrected_wd={wd:.6e} eff_decay={eff:.6e}")
        # weight_decay scaled to base*ratio.
        assert math.isclose(wd, MUON_WEIGHT_DECAY * r, rel_tol=1e-9, abs_tol=1e-12)
        # Effective decay lr*wd = peak*lambda*ratio^2 = gamma_t^2/gamma_max*lambda (Alg 1 line 12).
        assert eff == pytest.approx(0.02 * MUON_WEIGHT_DECAY * r * r, rel=1e-9, abs=1e-15)
        # Excluded groups keep their constant decay.
        assert output_group['weight_decay'] == MUON_WEIGHT_DECAY
        assert adam_decayed['weight_decay'] == ADAM_WEIGHT_DECAY
        assert kendall_group['weight_decay'] == 0.0

    # At peak LR (ratio==1) the corrected decay is bit-identical to plain decay, peak*lambda.
    _update_lr(muon_opt, adam_opt, warmup_steps, 0.02, 0.003, warmup_steps, total_steps,
               lr_min_ratio, wd_correction=True)
    assert ratio(warmup_steps) == 1.0
    assert corrected_group['lr'] * corrected_group['weight_decay'] == 0.02 * MUON_WEIGHT_DECAY

    # Drive the group to a drifted tail value with no intervening ratio==1 reset first.
    _update_lr(muon_opt, adam_opt, 99000, 0.02, 0.003, warmup_steps, total_steps,
               lr_min_ratio, wd_correction=True)
    assert corrected_group['weight_decay'] < MUON_WEIGHT_DECAY   # drifted below base
    _update_lr(muon_opt, adam_opt, 99000, 0.02, 0.003, warmup_steps, total_steps,
               lr_min_ratio, wd_correction=False)
    assert corrected_group['weight_decay'] == MUON_WEIGHT_DECAY
    print(f"[DUMP] wd_correction | wd_correction=False restores corrected_wd="
          f"{corrected_group['weight_decay']:.6e} == baseline {MUON_WEIGHT_DECAY:.6e} ✓")


def test_csv_header_and_row_same_length():
    """Header and row writer stay element-for-element aligned.

    Both come from one (name, decimals) spec. The checkpoint's val_record is a THIRD
    surface, deliberately not compared: it carries keys no CSV column has and misses others.
    """
    import csv
    import io

    from train import _train_log_columns, _val_log_columns, _csv_row

    for name, columns in (('train', _train_log_columns()),
                          ('val', _val_log_columns())):
        header = [c for c, _ in columns]
        row = _csv_row(columns, {})
        assert len(header) == len(row), (
            f"{name} log: header has {len(header)} columns, the row writer "
            f"emits {len(row)}"
        )
        assert len(set(header)) == len(header), (
            f"{name} log: duplicate column names "
            f"{sorted({c for c in header if header.count(c) > 1})}"
        )

        # Through the writer the loop uses, so a widening only visible when serialised is caught.
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(header)
        w.writerow(row)
        parsed = list(csv.reader(io.StringIO(buf.getvalue())))
        assert len(parsed[0]) == len(parsed[1]), (
            f"{name} log: written header has {len(parsed[0])} cells, the "
            f"written row has {len(parsed[1])}"
        )
        print(f"[DUMP] csv schema | {name}: {len(header)} columns, header == row ✓")


def test_val_log_has_no_dead_alias():
    """The pinball term IS ``loss_Q``, and two names for it read as two measurements."""
    from train import _val_log_columns

    names = [c for c, _ in _val_log_columns()]
    assert 'val_loss_Q' in names
    assert 'val_pinball' not in names, "val_pinball is a dead alias of val_loss_Q"
    print("[DUMP] csv schema | val_pinball absent, val_loss_Q present ✓")


def test_val_log_bins_masked_bg_on_d():
    """Every masked-BG family is reported per ``d``, and none is pooled.

    A pooled masked-BG scalar falls without the model improving (sampler concentrates at
    small d) and must not exist to be selected on. Axes come from metrics.protocols.
    """
    from config import PREDICTION_PATCHES
    from metrics.protocols import FORECAST, INFILL, column, reachable_d
    from train import _val_log_columns, _excursion_bucket_horizons

    names = [c for c, _ in _val_log_columns()]
    eh = _excursion_bucket_horizons(PREDICTION_PATCHES)
    fc_d = reachable_d(FORECAST)
    inf_d = reachable_d(INFILL)

    # the forecast horizons and its d bins are one axis, one for one
    assert len(eh) == len(fc_d) == PREDICTION_PATCHES

    for h in eh:
        for fam in ('crps', 'winkler90', 'sharp90', 'sharp50', 'joint_cov90'):
            assert f'{fam}@{h}' in names, f"missing {fam}@{h}"
    for d in inf_d:
        for fam in ('crps_n', 'rmse', 'rmse_interp', 'crps', 'winkler90',
                    'marginal90_cov', 'marginal90_width_mean'):
            col = column(INFILL, fam, d)
            assert col in names, f"missing {col}"

    # sharpness sits beside coverage everywhere coverage is reported
    for col in [c for c in names if c.startswith('coverage90@')]:
        assert col.replace('coverage90@', 'sharp90@') in names, (
            f"{col} has no sharpness companion")

    # No axis suffix IS a pooled masked-BG scalar; catches a route around column()'s refusal.
    for fam in ('crps', 'winkler90'):
        assert fam not in names, f"{fam} is pooled over d"
    for name in names:
        if name.startswith(INFILL.prefix):
            assert '@d' in name, f"{name} is an infill column with no d"

    print(f"[DUMP] csv schema | forecast d axis {eh} (one-sided d=1..{len(fc_d)}), "
          f"infill d axis {list(inf_d)}, no pooled masked-BG column ✓")
