"""Tests for attribution.py — the attention tap, the rollout, the saliency fold.

The whole module is read-only diagnostics, so the load-bearing property is that
it changes nothing: arming the tap must leave the forward bit-identical, and the
grad-enabled build must produce the same numbers as the ``no_grad`` one.  The
rest pins the maths — the tap reproduces what SDPA computes internally, the
attention mask's blocks survive into the captured weights, and the step-major
stride fold matches an explicit per-column loop.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def _make_context(n_ctx: int | None = None, flat_bg: bool = False) -> torch.Tensor:
    """Random normalized context with a slow, in-band bg trajectory.

    bg (feat 0) is NOT held at z=0. A channel pinned at its normalized mean has a
    saliency column of exactly zero whatever the gradient is — ``grad ⊙ input``
    with a zero input — so a z=0 bg fixture would run every attribution test with
    the channel that matters most silently dead. The sinusoid stays well inside
    the risk band, so the anchor's ``f_inv`` round trip never clamps.
    ``flat_bg=True`` asks for the degenerate case on purpose.
    """
    from config import PATCH_SIZE, N_INPUT_FEATURES, MIN_CONTEXT_PATCHES
    if n_ctx is None:
        n_ctx = MIN_CONTEXT_PATCHES
    ctx = torch.randn(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)
    if flat_bg:
        ctx[:, :, 0] = 0.0
    else:
        steps = torch.arange(n_ctx * PATCH_SIZE, dtype=torch.float32)
        ctx[:, :, 0] = (0.5 * torch.sin(steps / 37.0)).reshape(n_ctx, PATCH_SIZE)
    return ctx


def _get_stats():
    import os
    from normalization import (
        compute_normalization_stats, load_normalization_stats, NORM_STATS_FILE,
    )
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def _model():
    from model import T1DMAI
    torch.manual_seed(0)
    model = T1DMAI()
    model.eval()
    return model


def test_attention_weights_reproduce_sdpa():
    """The tap computes the distribution the fused kernel uses internally."""
    from model import attention_weights

    torch.manual_seed(7)
    B, H, T, D = 2, 3, 11, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    mask = torch.rand(T, T) > 0.3
    mask.fill_diagonal_(True)  # no all-blocked row, as the mask builder guarantees

    attn = attention_weights(q, k, mask)
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    assert attn.shape == (B, H, T, T)
    torch.testing.assert_close(attn @ v, reference, rtol=1e-5, atol=1e-6)
    print(f"[DUMP] max |A@v - SDPA| = {(attn @ v - reference).abs().max():.3e}")


def test_captured_weights_are_row_stochastic_and_respect_the_mask():
    """Every captured row is a distribution, and a blocked column gets no mass."""
    from attribution import capture_attention
    from inference import _resolve_mask_spans, _run_forward, PREDICTION_PATCHES

    model = _model()
    stats = _get_stats()
    context = _make_context()
    n_ctx = int(context.shape[0])
    spans = _resolve_mask_spans(None, n_ctx)

    with capture_attention(model) as captured:
        _run_forward(model, context, stats, mask_spans=spans)
        layers = [calls[0].detach().clone() for calls in captured]

    assert len(layers) == len(model.blocks)
    # Rebuild the same mask the forward used: visible everywhere but the masked
    # set, no padding on the inference path.
    visible = torch.ones(1, n_ctx + PREDICTION_PATCHES, dtype=torch.bool)
    for start, length in spans:
        visible[0, start:start + length] = False
    from utils import create_attention_mask_from_visible
    mask = create_attention_mask_from_visible(visible)[0]

    for i, attn in enumerate(layers):
        sums = attn.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0, atol=1e-5)
        blocked = attn[..., ~mask]
        assert blocked.numel() > 0, "the test mask blocks nothing — it proves nothing"
        assert float(blocked.abs().max()) == 0.0, (
            f"layer {i} put mass on a blocked column"
        )
        print(f"[DUMP] layer {i} row-sum dev "
              f"{float((sums - 1).abs().max()):.3e}, blocked max "
              f"{float(blocked.abs().max()):.3e}")


def test_arming_the_tap_leaves_the_forward_bit_identical():
    """The diagnostic must not perturb the forecast it is diagnosing."""
    from attribution import capture_attention
    from inference import predict

    model = _model()
    stats = _get_stats()
    context = _make_context()

    plain = predict(model, context, normalization_stats=stats)
    with capture_attention(model):
        tapped = predict(model, context, normalization_stats=stats)

    for key in ('q_tau', 'median', 'median_bg', 'bands'):
        assert torch.equal(plain[key], tapped[key]), f"{key} moved under the tap"


def test_capture_disarms_after_an_exception():
    """A raising forward must not leave the taps armed and accumulating."""
    from attribution import capture_attention

    model = _model()
    with pytest.raises(RuntimeError):
        with capture_attention(model):
            raise RuntimeError("boom")
    assert all(block.attn.attn_sink is None for block in model.blocks)


def test_grad_forward_matches_no_grad_forward():
    """``grad=True`` changes the graph, never the values."""
    from inference import _run_forward

    model = _model()
    stats = _get_stats()
    context = _make_context()

    plain = _run_forward(model, context, stats)
    live = _run_forward(model, context, stats, grad=True)

    assert live['patches'].requires_grad
    assert not plain['patches'].requires_grad
    for key in ('q_tau', 'median'):
        assert torch.equal(plain[key], live[key].detach()), f"{key} moved under grad"


def test_rollout_is_row_stochastic():
    """Composition keeps every row a distribution over the patch axis."""
    from attribution import rollout

    torch.manual_seed(3)
    T = 9
    layers = []
    for _ in range(4):
        raw = torch.rand(T, T)
        layers.append(raw / raw.sum(dim=-1, keepdim=True))

    composed = rollout(layers)
    assert composed.shape == (T, T)
    assert float(composed.min()) >= 0.0
    sums = composed.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0, atol=1e-5)


def test_rollout_of_one_identity_layer_is_the_identity():
    """``0.5·I + 0.5·I`` renormalized is ``I`` — the composition adds nothing."""
    from attribution import rollout

    eye = torch.eye(6)
    torch.testing.assert_close(rollout([eye, eye]), eye, rtol=0, atol=1e-6)


def test_channel_saliency_folds_the_step_major_stride():
    """The fold matches an explicit per-column sum, and excludes the mask bit."""
    from attribution import channel_saliency
    from config import PATCH_SIZE, N_INPUT_FEATURES
    from normalization import CHANNEL_NAMES

    torch.manual_seed(11)
    T = 5
    grad = torch.randn(T, PATCH_SIZE * N_INPUT_FEATURES)
    patches = torch.randn(T, PATCH_SIZE * N_INPUT_FEATURES)

    saliency = channel_saliency(grad, patches)
    assert saliency.shape == (T, len(CHANNEL_NAMES))

    expected = torch.zeros(T, len(CHANNEL_NAMES))
    for feat in range(len(CHANNEL_NAMES)):
        for step in range(PATCH_SIZE):
            col = step * N_INPUT_FEATURES + feat
            expected[:, feat] += grad[:, col] * patches[:, col]
    torch.testing.assert_close(saliency, expected, rtol=1e-6, atol=1e-6)

    # Feat 4 is an announcement, not a channel: moving it alone must not move
    # any saliency column.
    from data import BG_MASKED_FEAT
    bumped = grad.clone()
    bumped[:, BG_MASKED_FEAT::N_INPUT_FEATURES] += 100.0
    torch.testing.assert_close(
        channel_saliency(bumped, patches), saliency, rtol=1e-6, atol=1e-6,
    )


def test_explain_shapes_and_default_span():
    """The default span is the trailing forecast, and every map spans the window."""
    from attribution import explain
    from inference import PREDICTION_PATCHES
    from normalization import CHANNEL_NAMES

    model = _model()
    stats = _get_stats()
    context = _make_context()
    n_ctx = int(context.shape[0])
    T = n_ctx + PREDICTION_PATCHES

    result = explain(model, context, stats)

    assert result.span == (n_ctx, PREDICTION_PATCHES)
    assert result.n_ctx == n_ctx
    assert result.where.shape == (T,)
    assert result.per_layer.shape == (len(model.blocks), T)
    assert result.channels.shape == (T, len(CHANNEL_NAMES))
    assert result.channel_names == list(CHANNEL_NAMES)
    np.testing.assert_allclose(result.where.sum(), 1.0, atol=1e-5)
    np.testing.assert_allclose(result.per_layer.sum(axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(result.channel_share.sum(), 1.0, atol=1e-6)
    assert np.all(np.isfinite(result.channels))
    assert list(result.slot_patches) == list(range(n_ctx, n_ctx + PREDICTION_PATCHES))
    print(f"[DUMP] channel share {dict(zip(CHANNEL_NAMES, result.channel_share))}")


def test_explain_reads_an_interior_span():
    """An infill span explains its own slots, not the trailing forecast's."""
    from attribution import explain
    from inference import PREDICTION_PATCHES

    model = _model()
    stats = _get_stats()
    context = _make_context()
    n_ctx = int(context.shape[0])
    infill = (40, 3)
    spans = [infill, (n_ctx, PREDICTION_PATCHES)]

    result = explain(model, context, stats, mask_spans=spans, span=infill)

    assert result.span == infill
    assert list(result.slot_patches) == [40, 41, 42]


def test_bg_saliency_matches_the_true_response_of_the_forecast():
    """The BG column is the forecast's real sensitivity, anchor included.

    A masked span's median is ``f(anchor) + delta`` and the anchor is most of it,
    but ``model.forward`` detaches it and ``inference`` builds it from
    ``context`` rather than from the ``patches`` leaf. Without the anchor term
    restored this column reports the head's delta alone — and its SIGN disagrees
    with the forecast's actual response on most patients. A finite difference is
    the only check that catches that, so this is the gate on it.
    """
    from attribution import explain
    from inference import _resolve_mask_spans, _run_forward, PREDICTION_PATCHES

    model = _model()
    stats = _get_stats()
    context = _make_context()
    n_ctx = int(context.shape[0])
    span = (n_ctx, PREDICTION_PATCHES)
    spans = _resolve_mask_spans(None, n_ctx)

    def target_at(ctx: torch.Tensor) -> float:
        out = _run_forward(model, ctx, stats, mask_spans=spans)
        sel = (out['valid'] & (out['mask_idx'] >= span[0])
               & (out['mask_idx'] < span[0] + span[1]))
        return float(out['median'][sel].mean())

    result = explain(model, context, stats, span=span)
    assert result.anchor_in_graph, "the anchor term was dropped — clamped fixture?"
    reported = float(result.channels[:, 0].sum())

    eps = 1e-3
    lo, hi = context.clone(), context.clone()
    lo[:, :, 0] *= (1 - eps)
    hi[:, :, 0] *= (1 + eps)
    true = (target_at(hi) - target_at(lo)) / (2 * eps)

    print(f"[DUMP] bg saliency sum {reported:+.5f} vs finite difference {true:+.5f}")
    assert abs(true) > 1e-3, "degenerate fixture — the forecast does not read bg"
    assert np.sign(reported) == np.sign(true), (
        f"bg saliency sign {reported:+.5f} contradicts the true response {true:+.5f}"
    )
    np.testing.assert_allclose(reported, true, rtol=0.05)


def test_every_channel_carries_saliency_on_a_live_context():
    """No channel is silently dead — the fixture exercises all four."""
    from attribution import explain
    from normalization import CHANNEL_NAMES

    model = _model()
    stats = _get_stats()
    result = explain(model, _make_context(), stats)

    for ch, name in enumerate(CHANNEL_NAMES):
        assert float(np.abs(result.channels[:, ch]).sum()) > 0.0, (
            f"channel {name} has an all-zero saliency column"
        )
        assert result.channel_share[ch] > 0.0, f"channel {name} has a zero share"


def test_a_channel_pinned_at_its_mean_reads_as_zero():
    """The documented blind spot of ``grad ⊙ input``, pinned so it stays known."""
    from attribution import explain
    from normalization import CHANNEL_NAMES

    model = _model()
    stats = _get_stats()
    result = explain(model, _make_context(flat_bg=True), stats)

    bg = CHANNEL_NAMES.index('bg_absolute')
    assert float(np.abs(result.channels[:, bg]).sum()) == 0.0
    assert float(result.channel_share[bg]) == 0.0


def test_the_anchor_cell_is_a_visible_context_patch():
    """The anchor term is read off the span's left neighbour, never a masked patch."""
    from attribution import explain
    from inference import PREDICTION_PATCHES

    model = _model()
    stats = _get_stats()
    context = _make_context()
    n_ctx = int(context.shape[0])
    infill = (40, 3)

    result = explain(model, context, stats,
                     mask_spans=[infill, (n_ctx, PREDICTION_PATCHES)], span=infill)

    assert result.anchor_patch == infill[0] - 1
    assert result.anchor_in_graph


def test_explain_refuses_a_span_outside_the_masked_set():
    """Nothing was predicted there, so there is nothing to explain."""
    from attribution import explain

    model = _model()
    stats = _get_stats()
    context = _make_context()

    with pytest.raises(ValueError, match="no head slot"):
        explain(model, context, stats, span=(10, 2))


def test_return_rolls_records_every_forward_without_changing_one():
    """The roll record is an observation: same rolls, same numbers, one new key."""
    from inference import predict_rolling

    model = _model()
    stats = _get_stats()
    context = _make_context()

    plain = predict_rolling(model, context, n_rolls=3, normalization_stats=stats)
    recorded = predict_rolling(model, context, n_rolls=3, normalization_stats=stats,
                               return_rolls=True)

    assert set(recorded) - set(plain) == {'roll_inputs'}
    for key in ('pred_bg', 'q_tau', 'bands'):
        assert torch.equal(plain[key], recorded[key]), f"{key} moved under return_rolls"
    assert len(recorded['roll_inputs']) == 3
    for entry in recorded['roll_inputs']:
        assert set(entry) == {'context', 'overrides', 'offset'}


def test_roll_windows_slide_with_the_saturated_context():
    """Once the context is at its cap every roll's window slides, and says so.

    A roll past saturation no longer starts at the caller's patch 0, so its
    ``offset`` is the only thing that puts its maps under the right stretch of
    trace. Below the cap the offset stays 0 and this never fires, which is why
    the fixture starts saturated.
    """
    from inference import predict_rolling, PREDICTION_PATCHES
    from config import MAX_CONTEXT_PATCHES

    model = _model()
    stats = _get_stats()
    context = _make_context(n_ctx=MAX_CONTEXT_PATCHES)

    rolls = predict_rolling(model, context, n_rolls=3, normalization_stats=stats,
                            return_rolls=True)['roll_inputs']

    assert [e['offset'] for e in rolls] == [0, PREDICTION_PATCHES, 2 * PREDICTION_PATCHES]
    for k, entry in enumerate(rolls):
        n_ctx = int(entry['context'].shape[0])
        assert n_ctx == MAX_CONTEXT_PATCHES
        assert entry['offset'] + n_ctx == MAX_CONTEXT_PATCHES + k * PREDICTION_PATCHES, (
            f"roll {k}'s prediction zone does not land where the trajectory puts it"
        )


def test_explain_reads_one_roll_of_a_rolling_forecast():
    """Each recorded roll explains itself, at its own place on the caller's axis."""
    from attribution import explain
    from inference import predict_rolling, PREDICTION_PATCHES

    model = _model()
    stats = _get_stats()
    context = _make_context()

    rolls = predict_rolling(model, context, n_rolls=3, normalization_stats=stats,
                            return_rolls=True)['roll_inputs']

    for k, entry in enumerate(rolls):
        result = explain(model, entry['context'], stats,
                         overrides=entry['overrides'],
                         window_offset=entry['offset'])
        n_ctx = int(entry['context'].shape[0])
        assert result.span == (n_ctx, PREDICTION_PATCHES)
        assert result.window_offset == entry['offset']
        assert result.where.shape == (n_ctx + PREDICTION_PATCHES,)
        # Where the maps land once the offset is applied: this roll's own
        # prediction zone, in the caller's patch numbering.
        absolute_start = result.span[0] + result.window_offset
        assert absolute_start == int(context.shape[0]) + k * PREDICTION_PATCHES


def test_window_offset_is_bookkeeping_only():
    """It moves where the maps are drawn, never what they say."""
    from attribution import explain

    model = _model()
    stats = _get_stats()
    context = _make_context()

    plain = explain(model, context, stats)
    shifted = explain(model, context, stats, window_offset=17)

    assert plain.window_offset == 0 and shifted.window_offset == 17
    np.testing.assert_array_equal(plain.where, shifted.where)
    np.testing.assert_array_equal(plain.channels, shifted.channels)
    assert plain.span == shifted.span


def test_masked_patches_are_reported_as_withheld_not_as_zero():
    """A masked patch carries no BG to attribute, and the result must say so.

    The builder writes a literal 0.0 into feat 0 there, so ``grad ⊙ input`` is
    zero whatever the gradient is. Drawn as a zero contribution that reads as
    "the model ignores this patch", which is the opposite of true — the mask bit
    is set and the model plainly conditions on it. ``masked_patches`` is what
    lets a reader tell the two apart.
    """
    from attribution import explain
    from inference import PREDICTION_PATCHES

    model = _model()
    stats = _get_stats()
    context = _make_context()
    n_ctx = int(context.shape[0])
    infill = (40, 3)
    spans = [infill, (n_ctx, PREDICTION_PATCHES)]

    result = explain(model, context, stats, mask_spans=spans, span=infill)

    expected = list(range(40, 43)) + list(range(n_ctx, n_ctx + PREDICTION_PATCHES))
    assert sorted(result.masked_patches.tolist()) == expected

    bg = result.channel_names.index('bg_absolute')
    assert np.all(result.channels[result.masked_patches, bg] == 0.0), (
        "a masked patch's bg saliency should be exactly zero — if it is not, the "
        "builder stopped zeroing feat 0 and the withheld marking is now a lie"
    )
    # A visible patch is not withheld, and does carry a reading.
    visible = [p for p in range(n_ctx) if p not in set(expected)]
    assert np.abs(result.channels[visible, bg]).sum() > 0.0


def test_attention_to_a_masked_patch_survives_the_ink_floor():
    """The model discounts a masked patch heavily but not to nothing.

    A floor of a hundredth of an even share rendered exactly that residue as
    zero ink, which is why the ramp spans three decades rather than two.
    """
    from attribution import share_ramp

    T = 200
    mass = np.full(T, 1.0 / T)
    mass[100] = 2.5e-3 / T * 1.0    # the measured residue: ~1/400th of a share
    mass /= mass.sum()

    ramped = share_ramp(mass, 0, T)
    assert ramped[100] > 0.0, "a discounted-but-read patch renders as nothing"
    assert ramped[100] < ramped[0], "and still reads as far below its share"


def test_share_ramp_shows_the_window_below_an_even_share():
    """The ramp must not collapse everything under an even share to one value.

    That is what a linear ramp on the excess does, and on a composed attention
    row roughly nine patches in ten sit at or below their share — so the whole
    of that would render identically black and the strip would show only the
    handful of peaks.
    """
    from attribution import share_ramp

    T = 200
    mass = np.full(T, 0.2 / (T - 4))       # 20% spread evenly over the window
    mass[-4:] = 0.8 / 4                    # 80% on four patches, as a forecast row is
    mass /= mass.sum()

    ramped = share_ramp(mass, 0, T)

    assert ramped.min() >= 0.0 and ramped.max() <= 1.0
    below = ramped[:-4]
    assert below.min() > 0.0, "patches under an even share were clipped to nothing"
    assert below.max() < ramped[-4:].min(), "the peaks must still dominate"
    # Monotone in mass, so brighter always means more attention.
    order = np.argsort(mass)
    assert np.all(np.diff(ramped[order]) >= -1e-9)


def test_share_ramp_keeps_an_even_share_on_scale():
    """A view holding nothing but faint patches must not be amplified."""
    from attribution import share_ramp

    T = 100
    mass = np.full(T, 1.0 / T)
    mass[0] = 10.0 / T
    mass /= mass.sum()

    # A view over the faint tail alone: the top of the scale stays at an even
    # share rather than dropping to the tail's own peak.
    tail = share_ramp(mass, 50, T)
    full = share_ramp(mass, 0, T)
    assert tail[60] < 1.0
    assert full[0] == 1.0
    assert tail[60] >= full[60]


def test_signed_ramp_keeps_sign_and_order():
    """The display ramp compresses magnitude without reordering or flipping."""
    from attribution import signed_ramp

    values = np.array([-4.0, -1.0, 0.0, 0.25, 1.0, 4.0])
    ramped = signed_ramp(values, scale=4.0)

    assert np.all(np.sign(ramped) == np.sign(values))
    assert np.all(np.abs(ramped) <= 1.0 + 1e-9)
    assert np.all(np.diff(ramped) > 0)
    np.testing.assert_allclose(signed_ramp(values, scale=0.0), np.zeros_like(values))
