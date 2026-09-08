"""The exported ``HeadRawForward`` wrapper and the head side file, RUN against the real model.

A stale reading position never raises, so every test here executes the wrapper. Lowering,
serialization and on-device numerics are NOT covered.
"""

import pytest
import torch

import config as cfg
from exporters.executorch_xnnpack import (
    build_representative_input, eager_time_logits, head_from_hidden, stock_head_raw,
)
from exporters.head_weights import write_head_weights
from exporters.modified_forward import NEG_FILL, HeadRawForward, build_struct_mask
from model import T1DMAI
from utils import create_attention_mask, step_states

# The export's fixed shape: context left-padded into MAX_CONTEXT_PATCHES, prediction at the edge.
T = cfg.MAX_SEQ_LEN
C = cfg.MAX_CONTEXT_PATCHES
P = cfg.PREDICTION_PATCHES
M = cfg.MAX_MASKED_PATCHES
HEAD_RAW_SHAPE = (1, M, cfg.PATCH_SIZE, 1 + 2 * cfg.N_SPREADS)
TIME_LOGITS_SHAPE = (1, M, cfg.TIME_PROBE_N_BINS)
HIDDEN_SHAPE = (1, T, cfg.D_MODEL)


@pytest.fixture(scope="module")
def model() -> T1DMAI:
    """Module-scoped: the trunk is not free."""
    torch.manual_seed(0)
    m = T1DMAI().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def stats():
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


@pytest.fixture(scope="module")
def forecast(stats):
    """The trailing-forecast window, built by the exporter's own builder so the modified
    and stock paths cannot describe different masked sets."""
    return build_representative_input(stats, seq_len=T)


@pytest.fixture(scope="module")
def infill(stats):
    """One infill span mid-context plus the trailing forecast: two spans, so a decoder
    that treats the whole slot axis as one span diverges here and not on the forecast."""
    return build_representative_input(
        stats, seq_len=T, mask_spans=[(C // 2, 4), (C, P)])


@pytest.fixture(scope="module")
def head_file(model, tmp_path_factory):
    """``(path, block)`` — the flat head file beside the graph."""
    path = str(tmp_path_factory.mktemp("head") / "m.head.bin")
    return path, write_head_weights(model, path)


def test_wrapper_runs_and_emits_the_three_declared_outputs(model, forecast):
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        head_raw, time_logits, hidden = wrapper(
            forecast.patches, forecast.struct, forecast.slot_sel)
    print(f"[DUMP] head_raw={tuple(head_raw.shape)} time_logits={tuple(time_logits.shape)} "
          f"hidden={tuple(hidden.shape)}")

    assert head_raw.shape == HEAD_RAW_SHAPE
    assert time_logits.shape == TIME_LOGITS_SHAPE
    assert hidden.shape == HIDDEN_SHAPE
    assert all(torch.isfinite(t).all() for t in (head_raw, time_logits, hidden))


@pytest.mark.parametrize("name", ["forecast", "infill"])
def test_head_raw_matches_the_stock_forward_on_the_same_masked_set(
        model, name, request):
    """``exp(NEG_FILL)`` underflows to 0.0 in fp32, matching a ``-inf`` bool mask."""
    w = request.getfixturevalue(name)
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        head_raw, _tl, _h = wrapper(w.patches, w.struct, w.slot_sel)
    stock = stock_head_raw(model, w)
    n = w.n_masked
    delta = float((head_raw[:, :n] - stock[:, :n]).abs().max())
    print(f"[DUMP] modified(struct) vs stock(bool) head_raw ({name}) max|d| = {delta:.3e}")

    assert stock.shape == HEAD_RAW_SHAPE
    assert delta < 1e-6, "the additive mask diverged from the bool mask"


def test_time_logits_match_the_stock_probe(model, forecast):
    """The probe reads the same slot hidden states, not a second head."""
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        _hr, time_logits, _h = wrapper(
            forecast.patches, forecast.struct, forecast.slot_sel)
    stock = eager_time_logits(model, forecast)
    n = forecast.n_masked
    delta = float((time_logits[:, :n] - stock[:, :n]).abs().max())
    print(f"[DUMP] modified vs stock time_logits max|d| = {delta:.3e}")

    assert delta < 1e-6


@pytest.mark.parametrize("name", ["forecast", "infill"])
def test_head_file_reproduces_head_raw_from_the_graph_hidden(
        model, head_file, name, request):
    """The adapter seam: the side file's MLP over the nodes gathered from ``hidden``, with
    the same B-spline weights, must reproduce the graph's own ``head_raw``. Disagreement
    means the head file and the graph are different heads."""
    w = request.getfixturevalue(name)
    path, block = head_file
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        head_raw, _tl, hidden = wrapper(w.patches, w.struct, w.slot_sel)
    rebuilt = head_from_hidden(path, block, hidden, w)
    n = w.n_masked
    delta = float((rebuilt[:, :n] - head_raw[:, :n]).abs().max())
    print(f"[DUMP] head file vs graph head_raw ({name}) max|d| = {delta:.3e}")

    assert rebuilt.shape == HEAD_RAW_SHAPE
    assert delta < 1e-4


def test_head_raw_reads_the_patches_slot_sel_names(model, forecast):
    """``slot_sel`` is the whole masked-set contract. The head must be the step states of
    exactly those patches, and a set shifted one patch left must NOT reproduce it, or the
    parity above proves nothing."""
    w = forecast
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        head_raw, _tl, hidden = wrapper(w.patches, w.struct, w.slot_sel)
        named = model.bg_head(step_states(hidden, w.mask_idx, w.bool_mask))
        shifted = model.bg_head(
            step_states(hidden, (w.mask_idx - 1).clamp(min=0), w.bool_mask))
    n = w.n_masked
    at = float((head_raw[:, :n] - named[:, :n]).abs().max())
    off = float((head_raw[:, :n] - shifted[:, :n]).abs().max())
    print(f"[DUMP] named slots max|d| = {at:.3e}; shifted one patch left = {off:.3e}")

    assert at < 1e-6, "head_raw is not the step states of the slots slot_sel names"
    assert off > 1e-3, "a shifted masked set reproduces the output; the test proves nothing"


def test_struct_mask_is_the_additive_form_of_the_stock_bool_mask():
    """``build_struct_mask`` is a second construction of ``create_attention_mask``'s
    pattern; unpadded they must agree entry for entry or the graph ships a mask the model
    was never trained under."""
    struct = build_struct_mask(C, dtype=torch.float32)
    stock_bool = create_attention_mask(C, P)
    expected = torch.where(
        stock_bool, torch.zeros_like(struct), torch.full_like(struct, NEG_FILL)
    )
    print(f"[DUMP] struct attend fraction = {float((struct == 0.0).float().mean()):.4f}")

    assert struct.shape == (T, T)
    assert torch.equal(struct, expected)
    assert set(struct.unique().tolist()) == {0.0, NEG_FILL}


def test_padding_never_reaches_a_prediction_token(model, stats):
    """Pad columns are blocked and a pad patch is never a decoder node, so whatever the
    caller left in one cannot move ``head_raw``."""
    n_ctx = cfg.MIN_CONTEXT_PATCHES
    pad0 = C - n_ctx
    w = build_representative_input(stats, n_ctx=n_ctx, seq_len=T)
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        head_raw, time_logits, _h = wrapper(w.patches, w.struct, w.slot_sel)
        perturbed = w.patches.clone()
        perturbed[:, :pad0, :] += 37.0
        head_raw_p, time_logits_p, _hp = wrapper(perturbed, w.struct, w.slot_sel)
    n = w.n_masked
    d_head = float((head_raw[:, :n] - head_raw_p[:, :n]).abs().max())
    d_time = float((time_logits[:, :n] - time_logits_p[:, :n]).abs().max())
    print(f"[DUMP] pad perturbation moved head_raw by {d_head:.3e}, time_logits by {d_time:.3e}")

    assert torch.isfinite(head_raw).all(), "a fully-masked row NaN'd the softmax"
    assert d_head < 1e-6 and d_time < 1e-6


def test_wrapper_is_torch_exportable(model, forecast):
    """The only export step reachable without the backends."""
    w = forecast
    wrapper = HeadRawForward(model).eval()

    with torch.no_grad():
        eager = wrapper(w.patches, w.struct, w.slot_sel)
        exported = torch.export.export(
            wrapper, (w.patches, w.struct, w.slot_sel), strict=False)
    traced = exported.module()(w.patches, w.struct, w.slot_sel)
    deltas = [float((t - e).abs().max()) for t, e in zip(traced, eager)]
    print(f"[DUMP] traced vs eager max|d| = {deltas}")

    assert len(traced) == 3, (
        "the graph must emit (head_raw, time_logits, hidden), in that order")
    assert traced[0].shape == HEAD_RAW_SHAPE
    assert traced[1].shape == TIME_LOGITS_SHAPE
    assert traced[2].shape == HIDDEN_SHAPE
    assert all(d < 1e-6 for d in deltas)
