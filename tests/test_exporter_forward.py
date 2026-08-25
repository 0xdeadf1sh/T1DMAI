"""The exported ``HeadRawForward`` wrapper, RUN against the real model.

A stale block call raises only when the wrapper is CALLED and a stale reading
position never raises at all, so every test here executes it. Lowering,
serialization and on-device numerics are NOT covered.
"""

import pytest
import torch

import config as cfg
from exporters.executorch_xnnpack import (
    eager_time_logits, right_edge_slots, stock_head_raw,
)
from exporters.modified_forward import NEG_FILL, HeadRawForward, build_struct_mask
from model import T1DMAI
from tests.forward_inputs import right_edge_inputs
from utils import create_attention_mask

# the export's one fixed shape: context left-padded into MAX_CONTEXT_PATCHES slots,
# prediction patches at the right edge
T = cfg.MAX_SEQ_LEN
C = cfg.MAX_CONTEXT_PATCHES
P = cfg.PREDICTION_PATCHES
HEAD_RAW_SHAPE = (1, P, cfg.PATCH_SIZE, 1 + 2 * cfg.N_SPREADS)
TIME_LOGITS_SHAPE = (1, P, cfg.TIME_PROBE_N_BINS)

# head_raw is cut before assemble_quantiles, so this only has to clear the stock
# forward's mg/dL units tripwire
ANCHOR_MGDL = 120.0


@pytest.fixture(scope="module")
def model() -> T1DMAI:
    """Module-scoped: 8 blocks is not free."""
    torch.manual_seed(0)
    m = T1DMAI().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def full_context():
    """``(patches, struct, bool_mask)`` at ``n_ctx = MAX_CONTEXT_PATCHES``.

    No padding, so struct and bool mask describe the same attention pattern.
    """
    patches, _attn, _anchor, _mask_idx = right_edge_inputs(B=1, n_ctx=C, seed=7)
    struct = build_struct_mask(C, dtype=torch.float32)
    return patches, struct, (struct == 0.0)


def test_wrapper_runs_and_emits_the_two_declared_outputs(model, full_context):
    patches, struct, _bool_mask = full_context
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        head_raw, time_logits = wrapper(patches, struct)
    print(f"[DUMP] head_raw={tuple(head_raw.shape)} time_logits={tuple(time_logits.shape)}")

    assert head_raw.shape == HEAD_RAW_SHAPE
    assert time_logits.shape == TIME_LOGITS_SHAPE
    assert torch.isfinite(head_raw).all() and torch.isfinite(time_logits).all()


def test_head_raw_matches_the_stock_forward_on_the_same_masked_set(model, full_context):
    """``exp(NEG_FILL)`` underflows to 0.0 in fp32, matching a ``-inf`` bool mask."""
    patches, struct, bool_mask = full_context
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        head_raw, _time_logits = wrapper(patches, struct)
    stock = stock_head_raw(model, patches, bool_mask, ANCHOR_MGDL)
    delta = float((head_raw - stock).abs().max())
    print(f"[DUMP] modified(struct) vs stock(bool) head_raw max|d| = {delta:.3e}")

    assert stock.shape == HEAD_RAW_SHAPE
    assert delta < 1e-6, "the additive mask diverged from the bool mask"


def test_time_logits_match_the_stock_probe(model, full_context):
    """The probe reads the same hidden states, not a second head."""
    patches, struct, bool_mask = full_context
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        _head_raw, time_logits = wrapper(patches, struct)
    stock = eager_time_logits(model, patches, bool_mask, ANCHOR_MGDL)
    delta = float((time_logits - stock).abs().max())
    print(f"[DUMP] modified vs stock time_logits max|d| = {delta:.3e}")

    assert delta < 1e-6


def test_head_reads_the_trailing_prediction_patches(model, full_context):
    """The right-edge specialisation, SPEC/inference.md §3.1.

    The wrapper slices where the general forward gathers, so a window shifted one
    patch left must NOT reproduce the output, or the match proves nothing.
    """
    patches, struct, bool_mask = full_context
    wrapper = HeadRawForward(model).eval()
    with torch.no_grad():
        head_raw, _tl = wrapper(patches, struct)

    _anchor, mask_idx = right_edge_slots(1, T, ANCHOR_MGDL)
    assert mask_idx[0].tolist() == list(range(T - P, T)), "fixture is not the right edge"

    with torch.no_grad():
        x = model.patch_embed(patches)
        from model import build_rope_cache
        rope_cos, rope_sin = build_rope_cache(T, cfg.HEAD_DIM, device=x.device, dtype=x.dtype)
        for block in model.blocks:
            x = block(x, rope_cos, rope_sin, bool_mask)
        x = model.final_norm(x)
        coeff = model.bg_head(x).view(
            1, T, cfg.BG_HEAD_STEP_BASIS_DIM, 1 + 2 * cfg.N_SPREADS
        )
        per_patch = torch.einsum('sk,btkc->btsc', model.step_basis, coeff)

    at_edge = float((head_raw - per_patch[:, T - P:, ...]).abs().max())
    shifted = float((head_raw - per_patch[:, T - P - 1:T - 1, ...]).abs().max())
    print(f"[DUMP] slice at right edge max|d| = {at_edge:.3e}; shifted one patch left = {shifted:.3e}")

    assert at_edge < 1e-6, "the head does not read the trailing PREDICTION_PATCHES"
    assert shifted > 1e-3, "a shifted window reproduces the output; the test proves nothing"


def test_struct_mask_is_the_additive_form_of_the_stock_bool_mask():
    """``build_struct_mask`` is a second construction of ``create_attention_mask``'s
    pattern; unpadded they must agree entry for entry or the graph ships a mask the
    model was never trained under."""
    struct = build_struct_mask(C, dtype=torch.float32)
    stock_bool = create_attention_mask(C, P)
    expected = torch.where(
        stock_bool, torch.zeros_like(struct), torch.full_like(struct, NEG_FILL)
    )
    print(f"[DUMP] struct attend fraction = {float((struct == 0.0).float().mean()):.4f}")

    assert struct.shape == (T, T)
    assert torch.equal(struct, expected)
    assert set(struct.unique().tolist()) == {0.0, NEG_FILL}


def test_padding_never_reaches_a_prediction_token(model):
    """Pad columns are blocked, so whatever the caller left in a pad patch cannot
    move ``head_raw``."""
    n_ctx = cfg.MIN_CONTEXT_PATCHES
    pad0 = C - n_ctx
    patches, _attn, _anchor, _mask_idx = right_edge_inputs(B=1, n_ctx=C, seed=11)
    struct = build_struct_mask(n_ctx, dtype=torch.float32)
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        head_raw, time_logits = wrapper(patches, struct)
        perturbed = patches.clone()
        perturbed[:, :pad0, :] += 37.0
        head_raw_p, time_logits_p = wrapper(perturbed, struct)
    d_head = float((head_raw - head_raw_p).abs().max())
    d_time = float((time_logits - time_logits_p).abs().max())
    print(f"[DUMP] pad perturbation moved head_raw by {d_head:.3e}, time_logits by {d_time:.3e}")

    assert torch.isfinite(head_raw).all(), "a fully-masked row NaN'd the softmax"
    assert d_head < 1e-6 and d_time < 1e-6


def test_wrapper_is_torch_exportable(model, full_context):
    """The only export step reachable without the backends."""
    patches, struct, _bool_mask = full_context
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        eager = wrapper(patches, struct)
        exported = torch.export.export(wrapper, (patches, struct), strict=False)
    traced = exported.module()(patches, struct)
    deltas = [float((t - e).abs().max()) for t, e in zip(traced, eager)]
    print(f"[DUMP] traced vs eager max|d| = {deltas}")

    assert len(traced) == 2, "the graph must emit (head_raw, time_logits), in that order"
    assert traced[0].shape == HEAD_RAW_SHAPE
    assert traced[1].shape == TIME_LOGITS_SHAPE
    assert all(d < 1e-6 for d in deltas)
