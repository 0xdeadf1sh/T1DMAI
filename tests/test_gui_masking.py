"""Tests for the GUI's free-form masking model (``gui_state``, no display).

The masked set the GUI emits is the same object ``data.sample_mask_spans`` draws
and ``inference.predict`` takes, so it must satisfy the same four rules
``inference._resolve_mask_spans`` asserts: sorted, never abutting, inside the
window, and with the whole future zone masked.  Everything here runs on the pure
functions, so no pygame surface and no model is needed.
"""

import numpy as np
import pytest

from config import MASK_MAX_SPANS, MASK_SPAN_LENGTHS, MAX_MASKED_PATCHES, PATCH_SIZE
from gui_state import (
    MASK_PRESET_BEGIN_FILL, MASK_PRESET_FORECAST, MASK_PRESET_INFILL, MASK_PRESETS,
    GUIState, MaskSpan, add_user_span, dose_painting_enabled, emit_mask_spans,
    mask_dose_fill, mask_span_ood, merge_mask_spans, preset_user_spans,
    span_anchor_cell, user_mask_capacity, validate_user_spans,
)

N_CTX = 48
N_PRED = 4


# ---------------------------------------------------------------- the budget


def test_capacity_is_the_head_minus_the_forecast():
    """The trailing forecast span is mandatory, so it is charged up front."""
    cap = user_mask_capacity(N_PRED)
    print(f"\n[DUMP] capacity | M={MAX_MASKED_PATCHES} n_pred={N_PRED} → {cap}")
    assert cap == MAX_MASKED_PATCHES - N_PRED
    assert user_mask_capacity(MAX_MASKED_PATCHES + 5) == 0, (
        "capacity must floor at 0, not go negative"
    )


def test_cap_refuses_the_span_that_would_breach_it():
    """At the cap the set is left EXACTLY as it was — no half-applied drag."""
    cap = user_mask_capacity(N_PRED)
    spans = [MaskSpan(0, cap)]
    kept, reason = add_user_span(spans, MaskSpan(10 + cap, 1), N_CTX, N_PRED)
    print(f"[DUMP] cap refusal | {reason}")
    assert reason, "adding past the cap must be refused"
    assert [s.as_tuple() for s in kept] == [s.as_tuple() for s in spans]


def test_cap_counts_the_forecast_span():
    """A user set exactly at the cap emits MAX_MASKED_PATCHES total."""
    st = GUIState()
    st.context = np.zeros((N_CTX, PATCH_SIZE, 5))
    assert st.add_mask_span(0, user_mask_capacity(N_PRED), N_PRED) == ''
    total = st.masked_patch_count(N_PRED)
    print(f"[DUMP] total masked | {total}")
    assert total == MAX_MASKED_PATCHES
    assert st.mask_budget_left(N_PRED) == 0


# ---------------------------------------------------------------- the separator


def test_abutting_spans_merge_into_one():
    """An abutting pair IS one longer span — the sampler never emits the pair."""
    merged = merge_mask_spans([MaskSpan(4, 2), MaskSpan(6, 3)])
    print(f"[DUMP] merge | {[s.as_tuple() for s in merged]}")
    assert [s.as_tuple() for s in merged] == [(4, 5)]


def test_overlapping_spans_merge():
    merged = merge_mask_spans([MaskSpan(10, 4), MaskSpan(12, 3)])
    assert [s.as_tuple() for s in merged] == [(10, 5)]


def test_separated_spans_survive_unmerged():
    """One visible patch between them is enough."""
    merged = merge_mask_spans([MaskSpan(10, 2), MaskSpan(13, 2)])
    assert [s.as_tuple() for s in merged] == [(10, 2), (13, 2)]


def test_merge_sorts():
    merged = merge_mask_spans([MaskSpan(20, 1), MaskSpan(4, 1)])
    assert [s.as_tuple() for s in merged] == [(4, 1), (20, 1)]


def test_span_touching_the_forecast_is_refused():
    """Patch n_ctx-1 must stay visible: it separates the user's last span from
    the trailing forecast span, which starts at n_ctx."""
    ok = validate_user_spans([MaskSpan(N_CTX - 3, 2)], N_CTX, N_PRED)
    bad = validate_user_spans([MaskSpan(N_CTX - 2, 2)], N_CTX, N_PRED)
    print(f"[DUMP] separator | last={N_CTX - 2} ok={ok!r} last={N_CTX - 1} bad={bad!r}")
    assert ok == '', "a span ending at n_ctx-2 leaves the separator and is legal"
    assert bad, "a span ending at n_ctx-1 abuts the forecast and must be refused"


def test_span_outside_the_window_is_refused():
    assert validate_user_spans([MaskSpan(-1, 2)], N_CTX, N_PRED)
    assert validate_user_spans([MaskSpan(N_CTX + 2, 1)], N_CTX, N_PRED)


# ---------------------------------------------------------------- what is emitted


@pytest.mark.parametrize("preset", MASK_PRESETS)
def test_emitted_set_satisfies_the_inference_contract(preset):
    """Sorted, non-abutting, in bounds, whole future zone masked — the four rules
    ``inference._resolve_mask_spans`` asserts, checked here rather than crashing
    a prediction thread."""
    spans = preset_user_spans(preset, N_CTX, N_PRED)
    emitted = emit_mask_spans(spans, N_CTX, N_PRED)
    seq_len = N_CTX + N_PRED
    print(f"[DUMP] emit {preset} | {emitted}")

    assert emitted == sorted(emitted), "spans must be ascending"
    prev_end = -1
    for start, length in emitted:
        assert length >= 1
        assert 0 <= start and start + length <= seq_len
        assert start > prev_end, f"{emitted} abuts or overlaps"
        prev_end = start + length
    assert sum(L for _s, L in emitted) <= MAX_MASKED_PATCHES

    masked = {p for s, L in emitted for p in range(s, s + L)}
    assert set(range(N_CTX, seq_len)) <= masked, "the future zone must be masked"
    assert len(masked) < seq_len, "some evidence must stay visible"


def test_forecast_preset_emits_the_trailing_span_alone():
    """The trailing forecast is the DEFAULT and still one preset of the same
    objective, not a separate mode."""
    emitted = emit_mask_spans(
        preset_user_spans(MASK_PRESET_FORECAST, N_CTX, N_PRED), N_CTX, N_PRED,
    )
    assert emitted == [(N_CTX, N_PRED)]


def test_begin_fill_starts_at_patch_zero():
    spans = preset_user_spans(MASK_PRESET_BEGIN_FILL, N_CTX, N_PRED)
    assert spans and spans[0].start == 0, "begin-fill is the span at patch 0"


def test_infill_is_interior():
    """Never at patch 0 — that case is begin-fill, and its anchor comes from the
    right neighbour instead of the left."""
    spans = preset_user_spans(MASK_PRESET_INFILL, N_CTX, N_PRED)
    assert spans and spans[0].start >= 1
    assert spans[0].last <= N_CTX - 2


def test_emit_refuses_an_illegal_set():
    with pytest.raises(AssertionError):
        emit_mask_spans([MaskSpan(N_CTX - 1, 1)], N_CTX, N_PRED)


def test_emitted_set_is_accepted_by_inference():
    """The real contract, not a restatement of it: hand every preset's emitted
    set to the resolver ``inference.predict`` runs."""
    from inference import _resolve_mask_spans
    for preset in MASK_PRESETS:
        spans = preset_user_spans(preset, N_CTX, N_PRED)
        emitted = emit_mask_spans(spans, N_CTX, N_PRED)
        resolved = _resolve_mask_spans(emitted, N_CTX)
        print(f"[DUMP] resolved {preset} | {resolved}")
        assert resolved == emitted


# ---------------------------------------------------------------- OOD hints


def test_in_distribution_presets_raise_no_ood_marker():
    for preset in MASK_PRESETS:
        emitted = emit_mask_spans(
            preset_user_spans(preset, N_CTX, N_PRED), N_CTX, N_PRED,
        )
        assert mask_span_ood(emitted, N_CTX, N_PRED) == {}, preset


def test_too_many_spans_is_flagged():
    """The sampler draws at most MASK_MAX_SPANS; more is unsupervised."""
    spans = [MaskSpan(2 * i, 1) for i in range(MASK_MAX_SPANS)]
    emitted = emit_mask_spans(spans, N_CTX, N_PRED)
    assert len(emitted) == MASK_MAX_SPANS + 1
    flags = mask_span_ood(emitted, N_CTX, N_PRED)
    print(f"[DUMP] span-count OOD | {len(emitted)} spans → {len(flags)} flagged")
    assert len(flags) == len(emitted), "every span of an oversized set is flagged"


def test_length_and_distance_hints_are_unreachable_today():
    """The user's budget is MAX_MASKED_PATCHES - n_pred = max(MASK_SPAN_LENGTHS),
    so no span the cap admits can be longer than the length law, and none can sit
    farther than that from evidence.  Both hints are dead at these constants —
    pinned here so the day the constants move, this test says which one changed.
    """
    budget = user_mask_capacity(N_PRED)
    max_len = max(MASK_SPAN_LENGTHS)
    print(f"[DUMP] reachability | budget={budget} max_span_len={max_len}")
    assert budget <= max_len, (
        "the length / distance OOD hints are now reachable — drop this test"
    )


def test_long_span_hint_reads_the_length_law(monkeypatch):
    """The threshold is MASK_SPAN_LENGTHS', not a literal 8: shorten the law and
    a span the old law allowed becomes out of distribution."""
    import gui_state
    monkeypatch.setattr(gui_state.config, 'MASK_SPAN_LENGTHS', (1, 2, 3))
    emitted = [(10, 4), (N_CTX, 2)]
    flags = mask_span_ood(emitted, N_CTX, 2)
    print(f"[DUMP] long-span OOD | {flags}")
    assert 0 in flags and '3' in flags[0]
    assert 1 not in flags, "the 2-patch trailing span is still within the law"


def test_distance_never_exceeds_span_length():
    """Why the distance hint can never fire on its own.

    ``d`` is the distance to the nearest visible evidence on EITHER side, so a
    two-sided span caps it at ``ceil(L/2)`` and a one-sided span at ``L``: ``d <=
    L`` for every span, always.  ``d > max(MASK_SPAN_LENGTHS)`` therefore implies
    ``L > max(MASK_SPAN_LENGTHS)``, and the length hint has already fired.  The
    distance check is kept because it is the condition the objective is actually
    about — this test is what says it is currently subsumed.
    """
    from data import _mask_slots
    for emitted, n_ctx, n_pred in [
        ([(0, 5), (20, 3), (N_CTX, N_PRED)], N_CTX, N_PRED),
        ([(1, 4), (N_CTX, N_PRED)], N_CTX, N_PRED),
        ([(N_CTX - 3, 1), (N_CTX, N_PRED)], N_CTX, N_PRED),
    ]:
        _mi, _v, d, _a = _mask_slots(emitted, n_ctx + n_pred)
        slot = 0
        for start, length in emitted:
            span_d = int(d[slot:slot + length].max())
            slot += length
            print(f"[DUMP] d<=L | span ({start},{length}) max d {span_d}")
            assert span_d <= length


def test_oversized_set_is_reported_not_crashed():
    """``_mask_slots`` has only MAX_MASKED_PATCHES slots. An over-budget set is
    refused upstream, so this is a guard, not a path — it must say so rather than
    index past the head."""
    emitted = [(0, MAX_MASKED_PATCHES), (N_CTX, N_PRED)]
    flags = mask_span_ood(emitted, N_CTX, N_PRED)
    print(f"[DUMP] oversized | {flags}")
    assert len(flags) == 2 and 'slots' in flags[0]


# ---------------------------------------------------------------- the anchor


def test_anchor_is_the_left_neighbour_last_step():
    patch, step = span_anchor_cell(20, 3)
    print(f"[DUMP] anchor interior | span (20,3) → patch {patch} step {step}")
    assert (patch, step) == (19, PATCH_SIZE - 1)


def test_anchor_of_a_span_at_patch_zero_is_the_right_neighbour():
    """The only no-left-neighbour case there is — the begin-fill one."""
    patch, step = span_anchor_cell(0, 3)
    print(f"[DUMP] anchor begin-fill | span (0,3) → patch {patch} step {step}")
    assert (patch, step) == (3, 0)


def test_anchor_matches_datas_own_rule():
    from data import _anchor_step_for_span
    for start, length in [(0, 1), (0, 5), (7, 2), (N_CTX, N_PRED)]:
        patch, step = span_anchor_cell(start, length)
        assert patch * PATCH_SIZE + step == _anchor_step_for_span(start, length)


def test_selected_anchor_falls_back_to_the_forecast():
    """With nothing selected the readout is the context edge, exactly what it
    showed before masking existed."""
    st = GUIState()
    st.context = np.zeros((N_CTX, PATCH_SIZE, 5))
    assert st.selected_anchor_cell(N_PRED) == (N_CTX - 1, PATCH_SIZE - 1)
    assert st.add_mask_span(10, 2, N_PRED) == ''
    st.selected_mask_idx = 0
    assert st.selected_anchor_cell(N_PRED) == (9, PATCH_SIZE - 1)


# ---------------------------------------------------------------- per-policy doses


def test_announced_policy_keeps_the_recorded_doses():
    from data import masked_channel_policy
    stats = _fake_stats()
    assert mask_dose_fill(masked_channel_policy(blind=False), stats) is None
    assert dose_painting_enabled(masked_channel_policy(blind=False))


def test_blind_policy_takes_the_zero_dose_fill():
    from data import MASKABLE_FEATS, masked_channel_policy, zero_dose_fill
    stats = _fake_stats()
    fill = mask_dose_fill(masked_channel_policy(blind=True), stats)
    print(f"[DUMP] blind fill | {fill}")
    assert fill == zero_dose_fill(stats), "the fill is data's, not a second copy"
    assert set(fill) == set(MASKABLE_FEATS)
    assert not dose_painting_enabled(masked_channel_policy(blind=True))


def test_blind_fill_needs_stats():
    """Without stats there is no z-space to place normalize(0) in, so the fill is
    withheld rather than guessed."""
    from data import masked_channel_policy
    assert mask_dose_fill(masked_channel_policy(blind=True), None) is None


def test_absent_stamp_reads_as_announced():
    from data import stored_masked_channel_policy
    assert mask_dose_fill(stored_masked_channel_policy({}), _fake_stats()) is None


def _fake_stats() -> dict[str, dict[str, float]]:
    from normalization import CHANNEL_NAMES
    return {c: {'mean': 0.4, 'std': 1.3} for c in CHANNEL_NAMES}


# ---------------------------------------------------------------- state lifetime


def test_context_change_clears_the_spans():
    """Spans hold absolute patch positions, so a context that grows or is
    replaced would leave them masking different data than the user drew."""
    st = GUIState()
    st.context = np.zeros((N_CTX, PATCH_SIZE, 5))
    assert st.add_mask_span(10, 2, N_PRED) == ''
    st.clear_mask_spans()
    assert st.mask_spans == []
    assert st.mask_preset == MASK_PRESET_FORECAST
    assert st.emitted_mask_spans(N_PRED) == [(N_CTX, N_PRED)]


def test_remove_one_span():
    st = GUIState()
    st.context = np.zeros((N_CTX, PATCH_SIZE, 5))
    assert st.add_mask_span(4, 2, N_PRED) == ''
    assert st.add_mask_span(20, 2, N_PRED) == ''
    st.remove_mask_span(0)
    assert [s.as_tuple() for s in st.mask_spans] == [(20, 2)]
    st.remove_mask_span(7)          # out of range: a no-op, not a crash
    assert len(st.mask_spans) == 1


# ---------------------------------------------------------------- slot → patch


def test_contiguous_runs_split_at_the_separator():
    """``mask_idx`` is one slot per masked patch in span order, so a break in the
    sequence IS a span boundary — the adjacency rule ``utils._span_layout`` uses.
    """
    from gui import _contiguous_runs
    runs = _contiguous_runs(np.array([4, 5, 6, 20, 21, 48, 49, 50, 51]))
    print(f"\n[DUMP] runs | {runs}")
    assert runs == [(0, 3), (3, 5), (5, 9)]
    assert _contiguous_runs(np.array([], dtype=np.int64)) == []
    assert _contiguous_runs(np.array([7])) == [(0, 1)]


def test_context_curve_breaks_over_a_masked_span():
    """The true BG under a masked span is the answer; drawing it there turns the
    fan into a comparison the user did not ask for."""
    from gui import _split_at_masked
    times = np.arange(0, 10, 0.5, dtype=np.float32)
    values = np.zeros_like(times)
    segs = _split_at_masked(times, values, [MaskSpan(3, 2)])
    print(f"[DUMP] split | {[(float(t[0]), float(t[-1])) for t, _v in segs]}")
    assert len(segs) == 2
    assert segs[0][-1].size and float(segs[0][0].max()) < 3.0
    assert float(segs[1][0].min()) >= 5.0


# ---------------------------------------------------------------- end to end


def _fake_context(n_ctx: int):
    """A normalized context of the shape ``inference.predict`` takes."""
    import torch
    from config import N_INPUT_FEATURES
    rng = np.random.default_rng(0)
    ctx = rng.normal(0.0, 0.5, (n_ctx, PATCH_SIZE, N_INPUT_FEATURES)).astype(np.float32)
    ctx[..., -1] = 0.0                       # feat 4 is a bit; predict rewrites it
    return torch.from_numpy(ctx)


def _gui_state_with_context(policy: str):
    from config import MIN_CONTEXT_PATCHES
    from gui_state import GUIState
    n_ctx = max(MIN_CONTEXT_PATCHES + 2, 26)
    st = GUIState()
    st.context = _fake_context(n_ctx)
    st.norm_stats = _fake_stats()
    st.masked_channel_policy = policy
    st.bg_raw = np.full(n_ctx * PATCH_SIZE, 120.0, dtype=np.float32)
    st.context_raw = np.zeros((n_ctx * PATCH_SIZE, 4), dtype=np.float32)
    return st, n_ctx


@pytest.mark.parametrize("policy", ["announced", "blind"])
def test_prediction_rows_land_on_the_masked_patches(policy):
    """The whole path: user spans → emitted set → forward → per-row patch index.

    ``span_patches`` is what the chart places each fan by, so a slot→patch
    mismatch here is a fan drawn over the wrong stretch of the day.
    """
    import torch
    from gui import _run_prediction
    from model import T1DMAI

    st, n_ctx = _gui_state_with_context(policy)
    n_pred = _live_n_pred()
    assert st.add_mask_span(4, 3, n_pred) == ''
    assert st.add_mask_span(12, 2, n_pred) == ''
    emitted = st.emitted_mask_spans(n_pred)
    print(f"\n[DUMP] {policy} emitted | {emitted}")

    model = T1DMAI()
    model.eval()
    _run_prediction(st, model, torch.device('cpu'), 0.5, 0.5, 4.0)
    assert not st.status_message.startswith("Error"), st.status_message

    want = [p for s, L in emitted for p in range(s, s + L)]
    got = st.prediction.span_patches.tolist()
    print(f"[DUMP] {policy} span_patches | {got}")
    assert got == want, "each row must sit on the patch the head gathered"
    assert st.prediction.bands.shape[0] == len(want)
    assert len(st.prediction.median_bg) == len(want) * PATCH_SIZE


def test_blind_withholds_doses_on_masked_patches_only():
    """Under 'blind' the masked spans carry data.zero_dose_fill; every visible
    patch keeps the record, and the trailing zone is inference's own
    normalize(0) — which IS the fill."""
    from config import MASKABLE_FEATS, N_INPUT_FEATURES
    from data import zero_dose_fill
    from gui import _masked_context

    st, n_ctx = _gui_state_with_context('blind')
    n_pred = _live_n_pred()
    assert st.add_mask_span(4, 3, n_pred) == ''
    emitted = st.emitted_mask_spans(n_pred)
    out = _masked_context(st, emitted)
    fill = zero_dose_fill(st.norm_stats)

    assert out is not st.context, "the caller's context must not be mutated"
    for feat in MASKABLE_FEATS:
        masked_block = out[4:7, :, feat].numpy()
        print(f"[DUMP] blind feat {feat} | masked {masked_block.min():.4f}"
              f"..{masked_block.max():.4f} want {fill[feat]:.4f}")
        assert np.allclose(masked_block, fill[feat])
        assert np.allclose(out[8:20, :, feat].numpy(),
                           st.context[8:20, :, feat].numpy()), (
            "a visible patch keeps its recorded dose")
    assert np.allclose(out[..., 0].numpy(), st.context[..., 0].numpy()), (
        "feat 0 is withheld by inference, not here")
    assert out.shape[-1] == N_INPUT_FEATURES


def test_announced_context_is_handed_through_unchanged():
    from gui import _masked_context
    st, _n_ctx = _gui_state_with_context('announced')
    n_pred = _live_n_pred()
    assert st.add_mask_span(4, 3, n_pred) == ''
    assert _masked_context(st, st.emitted_mask_spans(n_pred)) is st.context


def _live_n_pred() -> int:
    """The trailing span length ``inference.predict`` actually builds."""
    import inference
    return int(inference.PREDICTION_PATCHES)
