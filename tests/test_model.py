"""Tests for model.py — shape checks, parameter count, forward pass correctness.

Risk-space redesign: the model emits a single quantile BG head; the forward
signature is ``forward(patches, attn_mask, last_bg) -> (q_tau, median)`` in
RISK space. There are no per-channel dynamics heads, MDN, trend, event, or
alarm heads, and the forward takes no extra output-toggle kwargs.
"""

import math

import pytest
import torch


def _last_bg(B: int) -> torch.Tensor:
    """A valid (B,) mg/dL anchor inside the physical band."""
    return torch.full((B,), 120.0)


def test_model_instantiation():
    """Model creates without error."""
    from model import T1DMAI
    model = T1DMAI()
    assert model is not None


def test_parameter_count():
    """Parameter count lands within a sane range (resize_model can move it)."""
    from model import T1DMAI
    model = T1DMAI()
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[DUMP] model | total parameters: {total:,}")
    print(f"[DUMP] model | total parameters (M): {total / 1e6:.2f}M")
    assert 10_000 < total < 1_000_000_000, f"Parameter count {total} outside sane range"


def test_parameter_breakdown():
    """Print parameter count per component."""
    from model import T1DMAI
    model = T1DMAI()
    print("\n[DUMP] model | parameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} = {param.numel():,}")


def test_old_heads_removed():
    """The dynamics / MDN / trend / event / alarm heads of the old architecture
    must be gone — the quantile BG head is the only output head."""
    from model import T1DMAI
    m = T1DMAI()
    for attr in ('channel_heads', 'trend_head', 'event_classifier',
                 'excursion_alarm_head', 'last_mdn', 'bg_correction_head',
                 'bg_roc_head'):
        assert not hasattr(m, attr), f"stale head/attr '{attr}' must be removed"
    assert hasattr(m, 'bg_head'), "the quantile bg_head must exist"


def test_forward_shape():
    """Forward pass produces the (q_tau, median) 2-tuple with correct shapes,
    median == q_tau[...,3], and ascending quantiles."""
    from model import T1DMAI
    from config import (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES,
                        PATCH_DIM, MIN_CONTEXT_PATCHES)
    model = T1DMAI()
    model.eval()

    B = 2
    n_ctx = MIN_CONTEXT_PATCHES    # minimum context (8 hours)
    n_pred = PREDICTION_PATCHES
    T = n_ctx + n_pred

    patches = torch.randn(B, T, PATCH_DIM)
    attn_mask = torch.ones(T, T, dtype=torch.bool)

    with torch.no_grad():
        out = model(patches, attn_mask, _last_bg(B))
    assert len(out) == 2, "forward must return the 2-tuple (q_tau, median)"
    q_tau, median = out

    assert q_tau.shape == (B, PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES), \
        f"q_tau shape {q_tau.shape}"
    assert median.shape == (B, PREDICTION_PATCHES, PATCH_SIZE), \
        f"median shape {median.shape}"
    assert torch.allclose(median, q_tau[..., 3], atol=1e-6), "median must == q_tau[...,3]"
    diffs = q_tau[..., 1:] - q_tau[..., :-1]
    assert (diffs >= -1e-6).all(), "quantiles must be ascending in τ"

    print(f"\n[DUMP] forward | q_tau shape: {q_tau.shape}")
    print(f"[DUMP] forward | median mean: {median.mean():.4f}, std: {median.std():.4f}")


def test_forward_no_nan():
    """No NaN or Inf in output for random input."""
    from model import T1DMAI
    from config import PATCH_DIM, MIN_CONTEXT_PATCHES, PREDICTION_PATCHES
    model = T1DMAI()
    model.eval()

    B = 2
    T = MIN_CONTEXT_PATCHES + PREDICTION_PATCHES
    patches = torch.randn(B, T, PATCH_DIM)
    attn_mask = torch.ones(T, T, dtype=torch.bool)

    with torch.no_grad():
        q_tau, median = model(patches, attn_mask, _last_bg(B))

    assert not torch.isnan(q_tau).any(), "NaN detected in q_tau"
    assert not torch.isnan(median).any(), "NaN detected in median"
    assert not torch.isinf(q_tau).any(), "Inf detected in q_tau"
    assert not torch.isinf(median).any(), "Inf detected in median"


def test_forward_last_bg_required_and_validated():
    """last_bg is a required (B,) mg/dL tensor; a z-scored anchor (~[-3,3]) must
    trip the forward-top units assert."""
    from model import T1DMAI
    from config import PATCH_DIM, MIN_CONTEXT_PATCHES, PREDICTION_PATCHES
    model = T1DMAI().eval()

    B = 2
    T = MIN_CONTEXT_PATCHES + PREDICTION_PATCHES
    patches = torch.randn(B, T, PATCH_DIM)
    attn_mask = torch.ones(T, T, dtype=torch.bool)

    # A z-space anchor (the bug the assert exists to catch) must raise.
    with torch.no_grad():
        with pytest.raises(AssertionError):
            model(patches, attn_mask, torch.tensor([-1.5, 0.7]))
    print("\n[DUMP] forward | z-space last_bg trips the units assert ✓")


def test_forward_variable_context():
    """Model handles different context lengths."""
    from model import T1DMAI
    from utils import create_attention_mask
    from config import PREDICTION_PATCHES, PATCH_DIM, MIN_CONTEXT_PATCHES
    model = T1DMAI()
    model.eval()

    # Short context (minimum allowed)
    T_short = MIN_CONTEXT_PATCHES + PREDICTION_PATCHES
    patches_short = torch.randn(1, T_short, PATCH_DIM)
    mask_short = create_attention_mask(MIN_CONTEXT_PATCHES, PREDICTION_PATCHES)

    # Long context (~2× minimum)
    long_ctx = MIN_CONTEXT_PATCHES * 2
    T_long = long_ctx + PREDICTION_PATCHES
    patches_long = torch.randn(1, T_long, PATCH_DIM)
    mask_long = create_attention_mask(long_ctx, PREDICTION_PATCHES)

    with torch.no_grad():
        q_short, _ = model(patches_short, mask_short, _last_bg(1))
        q_long, _ = model(patches_long, mask_long, _last_bg(1))

    assert q_short.shape[1] == PREDICTION_PATCHES
    assert q_long.shape[1] == PREDICTION_PATCHES


def test_attention_mask_pred_bidirectional():
    """The prediction zone attends to itself BIDIRECTIONALLY (the pred->pred block
    is all-True), while context->prediction stays BLOCKED (all-False) so no future
    leaks into the context. Context<->context and prediction->context are full."""
    from utils import create_attention_mask
    from config import PREDICTION_PATCHES

    n_ctx = 5
    n_pred = PREDICTION_PATCHES
    mask = create_attention_mask(n_ctx, n_pred)

    pred_block = mask[n_ctx:, n_ctx:]
    assert pred_block.all(), "prediction->prediction must be bidirectional (all True)"
    assert not mask[:n_ctx, n_ctx:].any(), "context->prediction must be blocked (all False)"
    assert mask[:n_ctx, :n_ctx].all(), "context<->context must be full"
    assert mask[n_ctx:, :n_ctx].all(), "prediction->context must be full"
    print(f"\n[DUMP] mask | pred->pred bidirectional, context->pred blocked "
          f"(n_ctx={n_ctx}, n_pred={n_pred}) ✓")


def test_rope_positions():
    """RoPE does not produce NaN for various sequence lengths."""
    from model import T1DMAI
    from config import PATCH_DIM, MAX_SEQ_LEN, MIN_CONTEXT_PATCHES, PREDICTION_PATCHES
    model = T1DMAI()
    seq_lens = [
        MIN_CONTEXT_PATCHES + PREDICTION_PATCHES,
        2 * MIN_CONTEXT_PATCHES,
        MAX_SEQ_LEN // 2,
        MAX_SEQ_LEN,
    ]
    for T in seq_lens:
        patches = torch.randn(1, T, PATCH_DIM)
        mask = torch.ones(T, T, dtype=torch.bool)
        with torch.no_grad():
            q_tau, _ = model(patches, mask, _last_bg(1))
        assert not torch.isnan(q_tau).any(), f"NaN at T={T}"
    print(f"\n[DUMP] rope | passed all sequence lengths: {seq_lens}")


def test_rope_orthogonal_and_identity_at_pos0():
    """RoPE's rotation is orthogonal, so ``apply_rope`` preserves the per-position
    L2 norm of q exactly, and it is the IDENTITY at position 0 (freqs=0 → cos=1,
    sin=0).  This pins the half-rotation convention against the cos/sin table:
    a mismatched split ([even,odd] pairing vs the [first-half,second-half] split
    the table was built for) would BREAK norm preservation."""
    from model import build_rope_cache, apply_rope
    from config import HEAD_DIM, N_HEADS

    torch.manual_seed(0)
    B, T = 2, 20
    q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    cos, sin = build_rope_cache(T, HEAD_DIM)
    assert cos.shape == (T, HEAD_DIM) and sin.shape == (T, HEAD_DIM)

    # Position 0 has zero frequency → cos=1, sin=0, so the table is the identity.
    assert torch.allclose(cos[0], torch.ones(HEAD_DIM), atol=1e-6)
    assert torch.allclose(sin[0], torch.zeros(HEAD_DIM), atol=1e-6)

    q_rot = apply_rope(q, cos, sin)
    assert q_rot.shape == q.shape

    # (a) orthogonality — per-position L2 norm is invariant under the rotation.
    n_in = torch.linalg.vector_norm(q, dim=-1)
    n_out = torch.linalg.vector_norm(q_rot, dim=-1)
    max_norm_dev = float((n_out - n_in).abs().max())
    assert max_norm_dev < 1e-4, f"RoPE changed the q norm (not orthogonal): {max_norm_dev}"

    # (b) identity at position 0 — the rotation by zero angle is a no-op.
    max_pos0_dev = float((q_rot[:, :, 0, :] - q[:, :, 0, :]).abs().max())
    assert max_pos0_dev < 1e-6, f"RoPE not identity at position 0: {max_pos0_dev}"
    print(f"\n[DUMP] rope | norm-preserving (max dev {max_norm_dev:.2e}), "
          f"identity@pos0 (max dev {max_pos0_dev:.2e}) ✓")


def test_bidirectional_pred_forward_finite():
    """With the now-bidirectional prediction block, a fixed small input runs a
    finite forward — the RoPE tables built inside ``model.forward`` (base
    ROPE_BASE) compose with the bidirectional pred attention without NaN/Inf."""
    from model import T1DMAI
    from config import PATCH_DIM, MIN_CONTEXT_PATCHES, PREDICTION_PATCHES
    from utils import create_attention_mask

    torch.manual_seed(0)
    model = T1DMAI().eval()
    B = 3
    n_ctx = MIN_CONTEXT_PATCHES
    T = n_ctx + PREDICTION_PATCHES
    patches = torch.randn(B, T, PATCH_DIM)
    mask = create_attention_mask(n_ctx, PREDICTION_PATCHES)
    with torch.no_grad():
        q_tau, median = model(patches, mask, _last_bg(B))
    assert torch.isfinite(q_tau).all() and torch.isfinite(median).all(), \
        "bidirectional-pred forward produced a non-finite output"
    print(f"\n[DUMP] rope | bidirectional-pred forward finite "
          f"(q_tau {tuple(q_tau.shape)}, median {tuple(median.shape)}) ✓")


def test_init_median_is_persistence():
    """At init (BG_HEAD_INIT_SCALE small, bias 0) the median delta ≈ 0, so the
    median risk ≈ f(last_bg) — i.e. the initial forecast is persistence."""
    from model import T1DMAI
    from config import PATCH_DIM, MIN_CONTEXT_PATCHES, PREDICTION_PATCHES
    from utils import kovatchev_f

    torch.manual_seed(0)
    m = T1DMAI().eval()
    B = 4
    T = MIN_CONTEXT_PATCHES + PREDICTION_PATCHES
    patches = torch.randn(B, T, PATCH_DIM)
    mask = torch.ones(T, T, dtype=torch.bool)
    last_bg = torch.tensor([90.0, 120.0, 150.0, 200.0])
    with torch.no_grad():
        _, median = m(patches, mask, last_bg)
    anchor = kovatchev_f(last_bg).view(B, 1, 1)
    max_dev = float((median - anchor).abs().max())
    print(f"\n[DUMP] init_persistence | max|median - f(last_bg)| = {max_dev:.4f}")
    assert max_dev < 0.5, (
        f"initial median should sit near the persistence anchor f(last_bg), "
        f"max deviation {max_dev}")


def test_gradient_flow():
    """Gradients flow through all forecast-path parameters when the quantile head
    is exercised. The detached time-of-day probe (``time_head.*``) is deliberately
    off the forecast path (I1/I3) and is excluded here; its own gradient isolation
    is covered by ``test_time_probe.test_detach_isolates_trunk``."""
    from model import T1DMAI
    from config import PATCH_DIM, MIN_CONTEXT_PATCHES, PREDICTION_PATCHES
    model = T1DMAI()

    B = 2
    T = MIN_CONTEXT_PATCHES + PREDICTION_PATCHES
    patches = torch.randn(B, T, PATCH_DIM)
    mask = torch.ones(T, T, dtype=torch.bool)

    q_tau, median = model(patches, mask, _last_bg(B))
    # Sum over every quantile band so each head slot gets a grad.
    loss = q_tau.sum() + median.sum()
    loss.backward()

    no_grad_params = []
    for name, param in model.named_parameters():
        # The time-of-day probe is a DETACHED diagnostic head off the forecast
        # path by design — a forecast-only loss must not reach it.
        if name.startswith('time_head.'):
            continue
        if param.grad is None:
            no_grad_params.append(name)

    if no_grad_params:
        print(f"\n[DUMP] gradient | params with no gradient: {no_grad_params}")
    assert len(no_grad_params) == 0, f"No gradient for: {no_grad_params}"
