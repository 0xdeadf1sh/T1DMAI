"""The experimental ``--spline-edge extrapolate`` rule in ``utils.step_states``.

Default ``repeat`` stays bit-identical to the clamped rule; ``extrapolate`` carries a straight
line in hidden space through the last patch at CONSTANT spacing, which ``repeat`` cannot —
asserted both ways, so the test states the defect it fixes."""

import contextlib

import pytest
import torch

import config as cfg
from utils import bspline_step_weights, create_attention_mask_from_visible, step_states

S = cfg.PATCH_SIZE
D = cfg.D_MODEL
L = cfg.PREDICTION_PATCHES


@contextlib.contextmanager
def spline_edge(rule: str):
    """Run the body under ``rule``, then restore whatever the process was using."""
    previous = cfg.SPLINE_EDGE
    cfg.set_spline_edge(rule)
    try:
        yield rule
    finally:
        cfg.set_spline_edge(previous)


def _forecast_window(T: int = 16, seed: int = 7):
    """A right-edge span of L patches: left neighbour readable, no right neighbour."""
    first = T - L
    visible = torch.ones(1, T, dtype=torch.bool)
    visible[0, first:] = False
    attn = create_attention_mask_from_visible(visible, torch.zeros(1, T, dtype=torch.bool))
    torch.manual_seed(seed)
    return torch.randn(1, T, D), torch.arange(first, T).view(1, L), attn


def _interior_window(T: int = 16, first: int = 5, seed: int = 11):
    """A span with a readable visible patch on BOTH sides."""
    visible = torch.ones(1, T, dtype=torch.bool)
    visible[0, first:first + L] = False
    attn = create_attention_mask_from_visible(visible, torch.zeros(1, T, dtype=torch.bool))
    torch.manual_seed(seed)
    return torch.randn(1, T, D), torch.arange(first, first + L).view(1, L), attn


def _clamped_reference(x, mask_idx, attn):
    """The pre-flag step states: the spline with the end node REPEATED, straight from W."""
    first, last = int(mask_idx[0, 0]), int(mask_idx[0, -1])
    has_l = first > 0 and bool(attn[0, first, first - 1])
    has_r = last + 1 < x.shape[1] and bool(attn[0, last, last + 1])
    nodes = x[0, first - int(has_l):last + 1 + int(has_r)]
    with spline_edge('repeat'):
        W = bspline_step_weights(L, has_l, has_r)
    return (W @ nodes).reshape(L, S, D)


def test_default_is_the_clamped_rule_bit_for_bit():
    """Unset, the flag changes nothing: step_states equals the repeated-end-node matrix."""
    x, mask_idx, attn = _forecast_window()
    assert cfg.SPLINE_EDGE == cfg.SPLINE_EDGE_DEFAULT == 'repeat'
    h = step_states(x, mask_idx, attn)[0]
    ref = _clamped_reference(x, mask_idx, attn)
    print(f"\n[DUMP] repeat vs clamped matrix | max|d| = {float((h - ref).abs().max()):.3e}")
    assert h.shape == (L, S, D)
    assert torch.allclose(h, ref, atol=1e-6)


def test_interior_span_is_identical_under_both_rules():
    """Both neighbours readable, so no virtual node, so the rule cannot reach the states."""
    x, mask_idx, attn = _interior_window()
    with spline_edge('repeat'):
        a = step_states(x, mask_idx, attn)
        wa = bspline_step_weights(L, True, True)
    with spline_edge('extrapolate'):
        b = step_states(x, mask_idx, attn)
        wb = bspline_step_weights(L, True, True)
    print(f"\n[DUMP] interior span | max|d| states {float((a - b).abs().max()):.3e} "
          f"weights {float((wa - wb).abs().max()):.3e}")
    assert torch.equal(a, b)
    assert torch.equal(wa, wb)


def _line_window():
    """A right-edge span whose node states lie on a straight line at equal spacing."""
    T = 16
    first = T - L
    x = torch.zeros(1, T, D)
    torch.manual_seed(3)
    origin, direction = torch.randn(D), torch.randn(D)
    for p in range(first - 1, T):                      # node 0 is the visible left neighbour
        x[0, p] = origin + float(p) * direction
    visible = torch.ones(1, T, dtype=torch.bool)
    visible[0, first:] = False
    attn = create_attention_mask_from_visible(visible, torch.zeros(1, T, dtype=torch.bool))
    return x, torch.arange(first, T).view(1, L), attn, origin, direction


def _line_coordinate(h, origin, direction):
    """Per-step position along ``direction``, plus the off-line residual."""
    flat = h.reshape(L * S, D) - origin
    unit = direction / direction.norm()
    t = flat @ unit
    return t, (flat - t.unsqueeze(-1) * unit).norm(dim=-1)


def test_extrapolate_carries_a_straight_line_at_constant_spacing():
    """The property repeat lacks: on a line, every step spacing equals 1/S of a node."""
    x, mask_idx, attn, origin, direction = _line_window()
    with spline_edge('extrapolate'):
        h = step_states(x, mask_idx, attn)[0]
    t, resid = _line_coordinate(h, origin, direction)
    d1 = torch.diff(t) / direction.norm()
    far = d1[(L - 1) * S - 1:]                         # spacings inside the last patch
    print(f"\n[DUMP] extrapolate | off-line resid max {float(resid.max()):.3e}  "
          f"last-patch spacing {[round(v, 5) for v in far.tolist()]}")
    assert float(resid.max()) < 1e-4, "the states left the line"
    assert torch.allclose(far, torch.full_like(far, 1.0 / S), atol=1e-4)


def test_repeat_fails_the_straight_line_property():
    """Same line under repeat: the last patch decelerates into the repeated end node."""
    x, mask_idx, attn, origin, direction = _line_window()
    with spline_edge('repeat'):
        h = step_states(x, mask_idx, attn)[0]
    t, _resid = _line_coordinate(h, origin, direction)
    d1 = torch.diff(t) / direction.norm()
    far = d1[(L - 1) * S - 1:]
    print(f"\n[DUMP] repeat | last-patch spacing {[round(v, 5) for v in far.tolist()]}  "
          f"terminal/nominal = {float(far[-1]) * S:.4f}")
    assert not torch.allclose(far, torch.full_like(far, 1.0 / S), atol=1e-4)
    assert float(far[-1]) * S < 0.3, "the terminal step should collapse toward the end node"


def test_weight_rows_still_sum_to_one_under_extrapolate():
    """Virtual nodes are AFFINE in the real ones, so the rows stay a partition of unity."""
    with spline_edge('extrapolate'):
        for span in (1, 2, L, 8):
            for has_left in (False, True):
                for has_right in (False, True):
                    W = bspline_step_weights(span, has_left, has_right)
                    assert W.shape == (span * S, span + has_left + has_right)
                    assert torch.allclose(W.sum(dim=1), torch.ones(span * S), atol=1e-6)
    print(f"\n[DUMP] extrapolate weight rows sum to 1, L in (1, 2, {L}, 8) x both edges")


def test_one_patch_span_with_no_neighbour_falls_back_to_repeat():
    """No second node to extrapolate from on either side, so the edge node repeats."""
    T = 16
    visible = torch.ones(1, T, dtype=torch.bool)
    visible[0, T - 1] = False
    attn = create_attention_mask_from_visible(visible, torch.zeros(1, T, dtype=torch.bool))
    attn[0, T - 1, T - 2] = False                      # cut the left neighbour off too
    torch.manual_seed(5)
    x = torch.randn(1, T, D)
    mask_idx = torch.tensor([[T - 1]])
    with spline_edge('repeat'):
        a = step_states(x, mask_idx, attn)
    with spline_edge('extrapolate'):
        b = step_states(x, mask_idx, attn)
    print(f"\n[DUMP] L=1 no neighbour | max|d| = {float((a - b).abs().max()):.3e}")
    assert torch.equal(a, b)


@pytest.mark.parametrize('rule', cfg.SPLINE_EDGE_RULES)
def test_modified_forward_still_torch_exports_under_both_rules(rule):
    """The rule sits inside the exported graph; both branches must trace at a fixed shape."""
    from exporters.modified_forward import (
        HeadRawForward, build_slot_selection, build_struct_mask_from_visible, window_labels)
    from model import T1DMAI

    T = 24
    visible, is_pad, idx = window_labels(T - L, None, T, L)
    struct = build_struct_mask_from_visible(visible, is_pad)
    slot_sel = build_slot_selection(idx, T=T, m_slots=len(idx))
    with spline_edge(rule):
        torch.manual_seed(0)
        wrapper = HeadRawForward(T1DMAI().eval()).eval()
        for p in wrapper.parameters():
            p.requires_grad_(False)
        patches = torch.randn(1, T, cfg.PATCH_DIM)
        with torch.no_grad():
            eager = wrapper(patches, struct, slot_sel)
            ep = torch.export.export(wrapper, (patches, struct, slot_sel), strict=False)
            traced = ep.module()(patches, struct, slot_sel)
    d = [float((t - e).abs().max()) for t, e in zip(traced, eager)]
    print(f"\n[DUMP] export under {rule} | shapes {[tuple(t.shape) for t in traced]} max|d| {d}")
    assert traced[0].shape == (1, L, S, 1 + 2 * cfg.N_SPREADS)
    assert traced[2].shape == (1, T, D)
    assert all(v < 1e-6 for v in d)


def test_an_extrapolate_checkpoint_refuses_to_export(tmp_path):
    """The phone's head implements repeat only, so the export loader rejects the other rule."""
    from exporters.modified_forward import load_model
    from model import T1DMAI
    from utils import checkpoint_spline_edge

    assert checkpoint_spline_edge(None) == checkpoint_spline_edge({}) == cfg.SPLINE_EDGE_DEFAULT
    torch.manual_seed(0)
    ck = {'arch_version': cfg.ARCH_VERSION, 'input_layout': 'curves',
          'model_state_dict': T1DMAI().state_dict(),
          'training_config': {'spline_edge': 'extrapolate'}}
    path = tmp_path / 'extrap.pt'
    torch.save(ck, path)
    print(f"\n[DUMP] export refusal | checkpoint rule {checkpoint_spline_edge(ck)}")
    with pytest.raises(AssertionError, match="implements 'repeat' only"):
        load_model(str(path))
    ck['training_config']['spline_edge'] = 'repeat'
    torch.save(ck, path)
    load_model(str(path))
