"""The B-spline step decoder: the weight matrix, the node rule, and the seam statistics.

Every masked patch's PATCH_SIZE steps are read off one uniform cubic B-spline over the
span's node states, so a patch edge is an ordinary interior point of the curve. These
tests pin the matrix numerically and then measure the edges against the interior.
"""

import pytest
import torch

from config import (D_MODEL, MAX_MASKED_PATCHES, PATCH_SIZE, PREDICTION_PATCHES,
                    QUANTILE_LEVELS)
from tests.forward_inputs import masked_set_inputs

S = PATCH_SIZE
L = 4              # three interior patch edges, the seam statistics' subject
N_CTX = 40
ROWS = 24          # rows per span kind: the seam ratios are batch means, not per-row

# Row (i-1)*S + j is step j of masked patch i. For L=4 with a left node only, step j=0
# of patch i=1 sits at c = 1 - 2.5/6, so k=0, u=c, and the o=-1 node clamps onto node 0:
# its weight (1-u)^3/6 adds to node 0's (3u^3-6u^2+4)/6.
PIN_LEFT_END = (0.4376929012345679, 0.5292245370370370, 0.0330825617283950, 0.0, 0.0)
# the mirror: no left node, so lo=1 and hi=L+1; step j=S-1 of patch i=L sits at u=2.5/6
# and the o=+2 node clamps onto the right node L+1
PIN_RIGHT_END = (0.0, 0.0, 0.0330825617283950, 0.5292245370370370, 0.4376929012345679)

_SEAM_STEP = [p * S + S - 1 for p in range(L - 1)]


def _span_batch(seed: int):
    """One batch of ROWS forecast rows, ROWS backcast rows and ROWS infill rows, each a
    single span of L patches."""
    T = N_CTX + PREDICTION_PATCHES
    spans = ([[(T - L, L)]] * ROWS + [[(0, L)]] * ROWS + [[(N_CTX // 2, L)]] * ROWS)
    return masked_set_inputs(spans, N_CTX, M=MAX_MASKED_PATCHES, seed=seed)


def _seam_ratios(curve: torch.Tensor) -> "tuple[float, float]":
    """``(mean |step| at the patch edges / interior, same for |slope change|)``.

    ``curve`` is ``(rows, L*S)``. A slope change spans the edge from either side, so both
    second differences touching an edge step count as seam.
    """
    n = curve.shape[1]
    d = curve[:, 1:] - curve[:, :-1]
    dd = d[:, 1:] - d[:, :-1]
    seam = torch.zeros(n - 1, dtype=torch.bool)
    seam[_SEAM_STEP] = True
    curv = torch.zeros(n - 2, dtype=torch.bool)
    for j in _SEAM_STEP:
        curv[j - 1] = True
        if j < n - 2:
            curv[j] = True
    return (
        float(d[:, seam].abs().mean() / d[:, ~seam].abs().mean().clamp_min(1e-12)),
        float(dd[:, curv].abs().mean() / dd[:, ~curv].abs().mean().clamp_min(1e-12)),
    )


@pytest.fixture(scope="module")
def scaled_model():
    """A random model with the head's output layer scaled x100: at BG_HEAD_INIT_SCALE the
    median is flat to 1e-2 risk and the seam ratios would be measuring rounding."""
    from model import T1DMAI
    torch.manual_seed(0)
    m = T1DMAI().eval()
    with torch.no_grad():
        m.bg_head[-1].weight.mul_(100.0)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def spans_batch():
    return _span_batch(0)


def test_bspline_step_weights_shape_and_rows_sum_to_one():
    from utils import bspline_step_weights

    for span in (1, 2, L, 8):
        for has_left in (False, True):
            for has_right in (False, True):
                W = bspline_step_weights(span, has_left, has_right)
                assert W.shape == (span * S, span + has_left + has_right), (
                    f"L={span} left={has_left} right={has_right} shape {tuple(W.shape)}")
                assert torch.allclose(W.sum(1), torch.ones(span * S), atol=1e-6), (
                    f"L={span} left={has_left} right={has_right} rows do not sum to 1")
    print(f"\n[DUMP] bspline_step_weights | shapes and unit row sums over L in "
          f"(1, 2, {L}, 8) x both ends ✓")


def test_bspline_step_weights_end_nodes_absorb_the_clamped_weight():
    """The pinned rows: an end node carries its own weight plus the clamped neighbour's."""
    from utils import bspline_step_weights

    left = bspline_step_weights(L, True, False)[0]
    right = bspline_step_weights(L, False, True)[(L - 1) * S + S - 1]
    print(f"\n[DUMP] bspline_step_weights | left-end row {[round(v, 6) for v in left.tolist()]}"
          f" right-end row {[round(v, 6) for v in right.tolist()]}")

    assert torch.allclose(left, torch.tensor(PIN_LEFT_END), atol=1e-6)
    assert torch.allclose(right, torch.tensor(PIN_RIGHT_END), atol=1e-6)


def test_a_pad_row_is_never_a_node():
    """A span opening at the first real patch has no left node, so the pad state beside it
    cannot enter — poison it and the step states are unchanged, bit for bit."""
    from utils import (bspline_step_weights, create_attention_mask_from_visible,
                       step_states)

    T, pad0 = 16, 3
    first = pad0
    is_pad = torch.zeros(1, T, dtype=torch.bool)
    is_pad[0, :pad0] = True
    visible = torch.ones(1, T, dtype=torch.bool)
    visible[0, first:first + L] = False
    attn = create_attention_mask_from_visible(visible, is_pad)
    torch.manual_seed(4)
    x = torch.randn(1, T, D_MODEL)
    mask_idx = torch.arange(first, first + L).view(1, L)

    h = step_states(x, mask_idx, attn)
    poisoned = x.clone()
    poisoned[0, :pad0] = 1e6
    h_pois = step_states(poisoned, mask_idx, attn)
    print(f"\n[DUMP] pad node | max|h| = {float(h.abs().max()):.3f}, "
          f"poisoned max|h| = {float(h_pois.abs().max()):.3f}")

    assert torch.isfinite(h_pois).all(), "a pad state reached the spline"
    assert torch.equal(h, h_pois)

    # the nodes are the L masked patches plus the visible right neighbour, and nothing left
    W = bspline_step_weights(L, False, True)
    assert torch.allclose(
        h[0].reshape(L * S, D_MODEL), W @ x[0, first:first + L + 1], atol=1e-5)


def test_seam_step_and_slope_change_track_the_interior(scaled_model, spans_batch):
    """Across the three patch edges the mean |median step| and mean |slope change| sit
    within [0.5, 2.0]x their interior means, on all three span kinds."""
    patches, attn, anchor_bg, mask_idx, _valid = spans_batch
    with torch.no_grad():
        _q, median = scaled_model(patches, attn, anchor_bg, mask_idx)

    for k, kind in enumerate(('forecast', 'backcast', 'infill')):
        block = median[k * ROWS:(k + 1) * ROWS, :L].reshape(ROWS, L * S)
        r_step, r_curv = _seam_ratios(block)
        print(f"[DUMP] seam/interior ({kind:8s}) | step {r_step:.3f} slope change {r_curv:.3f}")
        assert 0.5 <= r_step <= 2.0, f"{kind}: seam/interior step ratio {r_step:.3f}"
        assert 0.5 <= r_curv <= 2.0, f"{kind}: seam/interior slope-change ratio {r_curv:.3f}"


def test_a_per_patch_constant_decode_fails_that_band(scaled_model, spans_batch):
    """The band is not vacuous: hold one state flat across each patch's steps — the
    superseded per-patch decode — and every step change lands on the patch edges."""
    from utils import assemble_quantiles

    patches, attn, anchor_bg, mask_idx, _valid = spans_batch
    cap: dict[str, torch.Tensor] = {}
    handle = scaled_model.final_norm.register_forward_hook(
        lambda _m, _i, out: cap.__setitem__('x', out))
    try:
        with torch.no_grad():
            scaled_model(patches, attn, anchor_bg, mask_idx)
    finally:
        handle.remove()

    B, M = mask_idx.shape
    node = cap['x'].gather(1, mask_idx.unsqueeze(-1).expand(B, M, D_MODEL))
    with torch.no_grad():
        head_raw = scaled_model.bg_head(node.unsqueeze(2).expand(B, M, S, D_MODEL))
        _q, median = assemble_quantiles(head_raw, anchor_bg, mask_idx)
    r_step, r_curv = _seam_ratios(median[:ROWS, :L].reshape(ROWS, L * S))
    print(f"\n[DUMP] per-patch-constant control | step {r_step:.3g} slope change {r_curv:.3g}")

    assert r_step > 2.0 and r_curv > 2.0


def test_init_median_is_the_anchor_at_every_slot():
    """At BG_HEAD_INIT_SCALE the head's output is near zero, so the median sits on
    f(anchor_bg) — the forecast starts at persistence."""
    from model import T1DMAI
    from utils import kovatchev_f

    torch.manual_seed(0)
    m = T1DMAI().eval()
    patches, attn, anchor_bg, mask_idx, _valid = _span_batch(1)
    with torch.no_grad():
        _q, median = m(patches, attn, anchor_bg, mask_idx)
    dev = float((median - kovatchev_f(anchor_bg).unsqueeze(-1)).abs().max())
    print(f"\n[DUMP] init | max|median - f(anchor_bg)| over every slot = {dev:.4f}")

    assert dev < 0.05


def test_median_is_the_middle_quantile_and_the_fan_ascends(scaled_model, spans_batch):
    patches, attn, anchor_bg, mask_idx, _valid = spans_batch
    with torch.no_grad():
        q_tau, median = scaled_model(patches, attn, anchor_bg, mask_idx)
    k = QUANTILE_LEVELS.index(0.5)
    diffs = q_tau[..., 1:] - q_tau[..., :-1]
    print(f"\n[DUMP] fan | median == q_tau[..., {k}], min adjacent gap "
          f"{float(diffs.min()):.3e}")

    assert torch.equal(median, q_tau[..., k])
    assert (diffs > 0).all()
