"""model.py — ``forward(patches, attn_mask, anchor_bg, mask_idx) -> (q_tau, median)`` in RISK space.

The head GATHERS ``M`` masked patches by index, one anchor per slot, never a
trailing slice. ``T`` is the batch's longest sample, ``<= MAX_SEQ_LEN``."""

import math

import pytest
import torch

from tests.forward_inputs import masked_set_inputs, right_edge_inputs


def test_model_instantiation():
    from model import T1DMAI
    model = T1DMAI()
    assert model is not None


def test_parameter_count():
    """A range, not a value: resize_model.py moves the count."""
    from model import T1DMAI
    model = T1DMAI()
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[DUMP] model | total parameters: {total:,}")
    print(f"[DUMP] model | total parameters (M): {total / 1e6:.2f}M")
    assert 10_000 < total < 1_000_000_000, f"Parameter count {total} outside sane range"


def test_parameter_breakdown():
    from model import T1DMAI
    model = T1DMAI()
    print("\n[DUMP] model | parameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} = {param.numel():,}")


def test_old_heads_removed():
    """The quantile BG head is the only output head."""
    from model import T1DMAI
    m = T1DMAI()
    for attr in ('channel_heads', 'trend_head', 'event_classifier',
                 'excursion_alarm_head', 'last_mdn', 'bg_correction_head',
                 'bg_roc_head'):
        assert not hasattr(m, attr), f"stale head/attr '{attr}' must be removed"
    assert hasattr(m, 'bg_head'), "the quantile bg_head must exist"


def test_forward_shape():
    """The slot axis is ``M``, the width of ``mask_idx``; a right-edge forecast sets it
    to ``PREDICTION_PATCHES``."""
    from model import T1DMAI
    from config import PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES
    model = T1DMAI()
    model.eval()

    B = 2
    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(B, seed=0)

    with torch.no_grad():
        out = model(patches, attn_mask, anchor_bg, mask_idx)
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


def test_forward_gathers_an_arbitrary_masked_set():
    """Backcast, infill and forecast through one forward at ``M = MAX_MASKED_PATCHES``.

    A trailing slice would emit these same shapes and the wrong patches, so what is
    asserted is the gather: slot ``j`` is the hidden state at ``mask_idx[:, j]``.
    """
    from model import T1DMAI
    from config import MAX_MASKED_PATCHES, PATCH_SIZE, N_QUANTILES, PREDICTION_PATCHES
    model = T1DMAI().eval()

    n_ctx = 16
    spans_per_row = [
        [(0, 3)],                       # backcast: span starts at patch 0
        [(4, 2), (9, 1)],               # infill: interior spans, separated
        [(n_ctx, PREDICTION_PATCHES)],  # forecast: span ends at T-1
    ]
    patches, attn_mask, anchor_bg, mask_idx, valid = masked_set_inputs(
        spans_per_row, n_ctx, seed=3)
    B = len(spans_per_row)

    with torch.no_grad():
        q_tau, median = model(patches, attn_mask, anchor_bg, mask_idx)
    assert q_tau.shape == (B, MAX_MASKED_PATCHES, PATCH_SIZE, N_QUANTILES)
    assert median.shape == (B, MAX_MASKED_PATCHES, PATCH_SIZE)
    assert torch.isfinite(q_tau).all() and torch.isfinite(median).all()

    # Only gather index moves; patches/mask/span length fixed, positional slice would match.
    moved = mask_idx.clone()
    moved[1, :2] = torch.tensor([6, 7])   # still ascending, still separated from 9
    with torch.no_grad():
        q_moved, _ = model(patches, attn_mask, anchor_bg, moved)
    assert not torch.allclose(q_moved[1, :2], q_tau[1, :2], atol=1e-6), \
        "moving a span's mask_idx did not move the head slots — positional slice?"
    assert torch.allclose(q_moved[0], q_tau[0], atol=1e-6), \
        "an untouched row's slots must be unaffected"
    print(f"\n[DUMP] gather | backcast/infill/forecast in one batch, "
          f"q_tau {tuple(q_tau.shape)}, valid counts "
          f"{valid.sum(dim=1).tolist()} ✓")


def test_forward_no_nan():
    from model import T1DMAI
    model = T1DMAI()
    model.eval()

    B = 2
    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(
        B, seed=1, all_true_mask=True)

    with torch.no_grad():
        q_tau, median = model(patches, attn_mask, anchor_bg, mask_idx)

    assert not torch.isnan(q_tau).any(), "NaN detected in q_tau"
    assert not torch.isnan(median).any(), "NaN detected in median"
    assert not torch.isinf(q_tau).any(), "Inf detected in q_tau"
    assert not torch.isinf(median).any(), "Inf detected in median"


def test_forward_anchor_required_and_validated():
    """``anchor_bg`` is a required ``(B, M)`` mg/dL tensor.

    The units guarantee is pool-independent: every legal z satisfies
    z_max < BG_CLAMP_MIN - 1e-3, the floor the assert reads, over ALL M slots.
    """
    from model import T1DMAI
    model = T1DMAI().eval()

    B = 2
    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(B, seed=2)

    with torch.no_grad():
        # a z-space anchor is the bug the assert exists to catch
        with pytest.raises(AssertionError):
            model(patches, attn_mask, torch.full_like(anchor_bg, -1.5), mask_idx)
        # one bad slot is enough: the tripwire reads every slot, not slot 0
        one_bad = anchor_bg.clone()
        one_bad[1, -1] = 0.7
        with pytest.raises(AssertionError):
            model(patches, attn_mask, one_bad, mask_idx)
        # a (B,) broadcast anchor is refused outright
        with pytest.raises(AssertionError):
            model(patches, attn_mask, anchor_bg[:, 0], mask_idx)
    print("\n[DUMP] forward | z-space anchor (any slot) and (B,) anchor both trip ✓")


def test_forward_variable_context():
    """``T`` is per batch, never ``MAX_SEQ_LEN``."""
    from model import T1DMAI
    from config import PREDICTION_PATCHES, MIN_CONTEXT_PATCHES
    model = T1DMAI()
    model.eval()

    short = right_edge_inputs(1, n_ctx=MIN_CONTEXT_PATCHES, seed=4)
    long = right_edge_inputs(1, n_ctx=MIN_CONTEXT_PATCHES * 2, seed=5)

    with torch.no_grad():
        q_short, _ = model(*short)
        q_long, _ = model(*long)

    assert q_short.shape[1] == PREDICTION_PATCHES
    assert q_long.shape[1] == PREDICTION_PATCHES
    assert short[0].shape[1] != long[0].shape[1], "the two T must differ"


def test_attention_mask_pred_bidirectional():
    """Context->prediction stays blocked so no future leaks into the context."""
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
        inputs = right_edge_inputs(1, n_ctx=T - PREDICTION_PATCHES, seed=T,
                                   all_true_mask=True)
        with torch.no_grad():
            q_tau, _ = model(*inputs)
        assert not torch.isnan(q_tau).any(), f"NaN at T={T}"
    print(f"\n[DUMP] rope | passed all sequence lengths: {seq_lens}")


def test_rope_orthogonal_and_identity_at_pos0():
    """Pins the half-rotation convention against the cos/sin table: an [even,odd]
    pairing against a [first-half,second-half] table breaks norm preservation."""
    from model import build_rope_cache, apply_rope
    from config import HEAD_DIM, N_HEADS

    torch.manual_seed(0)
    B, T = 2, 20
    q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    cos, sin = build_rope_cache(T, HEAD_DIM)
    assert cos.shape == (T, HEAD_DIM) and sin.shape == (T, HEAD_DIM)

    # position 0 has zero frequency: cos=1, sin=0, so the table is the identity
    assert torch.allclose(cos[0], torch.ones(HEAD_DIM), atol=1e-6)
    assert torch.allclose(sin[0], torch.zeros(HEAD_DIM), atol=1e-6)

    q_rot = apply_rope(q, cos, sin)
    assert q_rot.shape == q.shape

    n_in = torch.linalg.vector_norm(q, dim=-1)
    n_out = torch.linalg.vector_norm(q_rot, dim=-1)
    max_norm_dev = float((n_out - n_in).abs().max())
    assert max_norm_dev < 1e-4, f"RoPE changed the q norm (not orthogonal): {max_norm_dev}"

    max_pos0_dev = float((q_rot[:, :, 0, :] - q[:, :, 0, :]).abs().max())
    assert max_pos0_dev < 1e-6, f"RoPE not identity at position 0: {max_pos0_dev}"
    print(f"\n[DUMP] rope | norm-preserving (max dev {max_norm_dev:.2e}), "
          f"identity@pos0 (max dev {max_pos0_dev:.2e}) ✓")


def test_bidirectional_pred_forward_finite():
    from model import T1DMAI

    torch.manual_seed(0)
    model = T1DMAI().eval()
    B = 3
    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(B, seed=6)
    with torch.no_grad():
        q_tau, median = model(patches, attn_mask, anchor_bg, mask_idx)
    assert torch.isfinite(q_tau).all() and torch.isfinite(median).all(), \
        "bidirectional-pred forward produced a non-finite output"
    print(f"\n[DUMP] rope | bidirectional-pred forward finite "
          f"(q_tau {tuple(q_tau.shape)}, median {tuple(median.shape)}) ✓")


def test_init_median_is_persistence():
    """BG_HEAD_INIT_SCALE small with zero bias ⇒ median delta ≈ 0, so the initial
    forecast persists from each slot's OWN anchor."""
    from model import T1DMAI
    from utils import kovatchev_f

    torch.manual_seed(0)
    m = T1DMAI().eval()
    B = 4
    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(B, seed=7)
    # a different anchor per row: the median must follow each row's own value
    anchor_bg = torch.tensor([90.0, 120.0, 150.0, 200.0]).unsqueeze(1).expand_as(anchor_bg)
    with torch.no_grad():
        _, median = m(patches, attn_mask, anchor_bg.contiguous(), mask_idx)
    anchor = kovatchev_f(anchor_bg[:, 0]).view(B, 1, 1)
    max_dev = float((median - anchor).abs().max())
    print(f"\n[DUMP] init_persistence | max|median - f(anchor_bg)| = {max_dev:.4f}")
    assert max_dev < 0.5, (
        f"initial median should sit near the persistence anchor f(anchor_bg), "
        f"max deviation {max_dev}")


def test_gradient_flow():
    """``time_head.*`` sits off the forecast path and is excluded; its isolation is
    ``test_time_probe.test_detach_isolates_trunk``'s subject."""
    from model import T1DMAI
    model = T1DMAI()

    B = 2
    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(
        B, seed=8, all_true_mask=True)

    q_tau, median = model(patches, attn_mask, anchor_bg, mask_idx)
    # sum every quantile band so each head slot gets a grad
    loss = q_tau.sum() + median.sum()
    loss.backward()

    no_grad_params = []
    for name, param in model.named_parameters():
        if name.startswith('time_head.'):
            continue
        if param.grad is None:
            no_grad_params.append(name)

    if no_grad_params:
        print(f"\n[DUMP] gradient | params with no gradient: {no_grad_params}")
    assert len(no_grad_params) == 0, f"No gradient for: {no_grad_params}"
