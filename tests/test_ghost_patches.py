"""The experimental ``--ghost-patches`` flag: a masked patch past the horizon, discarded.

At 0 the pipeline is bit-identical to the pre-flag one, pinned against frozen values.
At 1 the pinned span grows by a patch that gives the last SCORED patch a right neighbour,
and that patch enters no loss and no metric — the decoded fan stays 24 steps."""

import contextlib

import numpy as np
import pytest
import torch

import config as cfg
from utils import create_attention_mask_from_visible, step_states

S = cfg.PATCH_SIZE
P = cfg.PREDICTION_PATCHES

# The synthetic row's geometry; the frozen values below describe a sample drawn at these.
GEOMETRY = {
    'PATCH_SIZE': 6, 'N_INPUT_FEATURES': 4, 'MIN_CONTEXT_PATCHES': 168,
    'MAX_CONTEXT_PATCHES': 336, 'PREDICTION_PATCHES': 4, '_MAX_MASKED_PATCHES_BASE': 12,
    'MASK_MAX_SPANS': 3, 'MASK_SPAN_LENGTHS': (1, 2, 3, 4, 5, 6, 7, 8),
    'MASK_RIGHT_EDGE_QUOTA': 0.50,
}
ARCH = {'D_MODEL': 16, 'N_LAYERS': 16, 'N_HEADS': 1, 'FFN_DIM': 16, 'BG_HEAD_HIDDEN': 16}

STATS = {
    'bg_absolute': {'mean': 0.3, 'std': 1.0},
    'carb_intake': {'mean': 0.02, 'std': 0.2},
    'insulin_combined': {'mean': 0.01, 'std': 0.1},
}


def _geometry_matches(spec: dict) -> bool:
    return all(getattr(cfg, k) == v for k, v in spec.items())


@contextlib.contextmanager
def ghost_patches(n: int):
    """Run the body under ``n`` ghost patches, then restore the process's own count."""
    previous = cfg.GHOST_PATCHES
    cfg.set_ghost_patches(n)
    try:
        yield n
    finally:
        cfg.set_ghost_patches(previous)


def _synthetic_row(n_extra_patches: int = 6) -> dict:
    """A deterministic simulator-shaped row: bg mg/dL, two sparse dose channels, the clock."""
    n = (cfg.MAX_CONTEXT_PATCHES + cfg.PREDICTION_PATCHES + n_extra_patches) * cfg.PATCH_SIZE
    t = np.arange(n, dtype=np.float64)
    carb = np.zeros(n)
    carb[::53] = 0.8
    ins = np.zeros(n)
    ins[::17] = 0.05
    return {
        'bg_observed': (130.0 + 45.0 * np.sin(t / 37.0)
                        + 12.0 * np.cos(t / 11.0)).astype(np.float32),
        'total_carb': carb.astype(np.float32),
        'total_insulin': ins.astype(np.float32),
        'hour_of_day': ((t * 5.0 / 60.0) % 24.0).astype(np.float32),
    }


def _samples(n: int = 4) -> list:
    from data import _build_sample
    row = _synthetic_row()
    return [
        _build_sample(data=row, icr=10.0, stats=STATS,
                      rng=np.random.default_rng(1000 + i), boundary=True,
                      skills=np.zeros(cfg.N_SKILLS, dtype=np.float32), arm=0)
        for i in range(n)
    ]


def _loss(batch, seed: int = 0):
    from model import T1DMAI
    from risk_loss import KendallGalWeighting, risk_total_loss
    torch.manual_seed(seed)
    model = T1DMAI().eval()
    bf = batch['bg_formula_data']
    with torch.no_grad():
        q, m = model(batch['patches'], batch['attn_mask'],
                     bf['anchor_bg'].float(), bf['mask_idx'].long())
        total, parts = risk_total_loss(q.float(), m.float(), batch['targets'].float(),
                                       KendallGalWeighting(), valid=bf['valid'],
                                       mask_idx=bf['mask_idx'].long())
    return model, total, parts, m


def test_default_is_zero_and_the_geometry_is_the_pre_flag_one():
    """Unset, the flag adds no slot and no patch: M and MAX_SEQ_LEN are what they were."""
    assert cfg.GHOST_PATCHES == cfg.GHOST_PATCHES_DEFAULT == 0
    assert cfg.MAX_MASKED_PATCHES == cfg._MAX_MASKED_PATCHES_BASE
    assert cfg.MAX_SEQ_LEN == cfg.MAX_CONTEXT_PATCHES + cfg.PREDICTION_PATCHES
    assert cfg.FORECAST_SPAN_PATCHES == cfg.PREDICTION_PATCHES
    with ghost_patches(1):
        assert cfg.MAX_MASKED_PATCHES == cfg._MAX_MASKED_PATCHES_BASE + 1
        assert cfg.MAX_SEQ_LEN == cfg.MAX_CONTEXT_PATCHES + cfg.PREDICTION_PATCHES + 1
        assert cfg.FORECAST_SPAN_PATCHES == cfg.PREDICTION_PATCHES + 1
    print(f"\n[DUMP] M {cfg.MAX_MASKED_PATCHES} -> {cfg._MAX_MASKED_PATCHES_BASE + 1}, "
          f"MAX_SEQ_LEN {cfg.MAX_SEQ_LEN} -> {cfg.MAX_SEQ_LEN + 1} under the flag")


@pytest.mark.skipif(not _geometry_matches(GEOMETRY), reason='sampler geometry moved')
def test_sample_and_batch_are_bit_identical_at_ghost_zero():
    """Frozen pre-flag values: one sample's tensors, and the collated batch built from four."""
    from data import collate_fn
    samples = _samples()
    s0, bf = samples[0], samples[0]['bg_formula_data']
    assert [int(s['n_context_patches']) for s in samples] == [202, 320, 270, 218]
    assert tuple(s0['patches'].shape) == (206, 24)
    assert float(s0['patches'].double().sum()) == -498.07726139575243
    assert tuple(s0['targets'].shape) == (12, 6)
    assert float(s0['targets'].double().sum()) == 6906.987548828125
    assert bf['mask_idx'].tolist() == [159, 160, 161, 162, 163, 164, 165, 202, 203, 204, 205, 0]
    assert bf['valid'].tolist() == [True] * 11 + [False]
    assert bf['d'].tolist() == [1, 2, 3, 4, 3, 2, 1, 1, 2, 3, 4, 0]
    assert float(np.asarray(bf['anchor_bg'], dtype=np.float64).sum()) == 1102.8693389892578
    assert float(np.asarray(bf['true_bg_trajectory'], dtype=np.float64).sum()) == 2519.6552734375

    batch = collate_fn(samples)
    assert int(batch['patches'].shape[1]) == 324
    assert float(batch['patches'].double().sum()) == -2610.7965172082186
    assert int(batch['attn_mask'].sum()) == 262689
    assert int(batch['pool_mask'].sum()) == 989
    assert int(batch['bg_formula_data']['mask_idx'].sum()) == 10247
    print(f"\n[DUMP] ghost 0 frozen sample | T={batch['patches'].shape[1]} "
          f"attn={int(batch['attn_mask'].sum())} masked={bf['valid'].sum()}")


@pytest.mark.skipif(not (_geometry_matches(GEOMETRY) and _geometry_matches(ARCH)),
                    reason='sampler geometry or architecture moved')
def test_loss_and_validation_metrics_are_bit_identical_at_ghost_zero():
    """Frozen pre-flag values: the objective loss, and the forecast forward's roughness."""
    import train
    from data import collate_fn
    batch = collate_fn(_samples())
    model, total, parts, _median = _loss(batch)
    assert float(total) == 4.7951154708862305
    assert float(parts['loss_Q']) == 0.16710986196994781
    assert float(parts['loss_D']) == 9.423121452331543

    bf = batch['bg_formula_data']
    fc = train._forecast_protocol(batch['patches'], bf['mask_idx'].long(),
                                  bf['valid'], batch['n_context_patches'])
    assert fc['rows'].tolist() == [0, 1, 2, 3]
    assert fc['mask_idx'][0].tolist() == [320, 321, 322, 323]
    with torch.no_grad():
        _q, median = model(fc['patches'], fc['attn_mask'],
                           bf['last_bg'].float()[fc['rows']]
                           .unsqueeze(1).expand(-1, fc['mask_idx'].shape[1]),
                           fc['mask_idx'])
    median = median[:, :fc['n_scored']].float()
    assert tuple(median.shape) == (4, 4, 6)
    flat = median.reshape(median.shape[0], -1)
    d2 = (flat[:, 2:] - 2.0 * flat[:, 1:-1] + flat[:, :-2]).abs()
    far0 = (P - 1) * S - 1
    print(f"\n[DUMP] ghost 0 frozen loss {float(total):.10f} "
          f"rough {float(d2.mean()):.6e} far {float(d2[:, far0:].mean()):.6e}")
    assert float(d2.sum() / d2.numel()) == 7.802654727129266e-05
    assert float(d2[:, far0:].sum() / d2[:, far0:].numel()) == 6.762742850696668e-05


@pytest.mark.skipif(not _geometry_matches(GEOMETRY), reason='sampler geometry moved')
def test_the_flag_extends_the_pinned_span_and_leaves_the_other_spans_alone():
    """Same rng draws: the unpinned spans do not move, and only the pinned one grows."""
    from data import collate_fn
    base = _samples(1)[0]['bg_formula_data']
    with ghost_patches(1):
        s = _samples(1)[0]
        g = s['bg_formula_data']
        batch = collate_fn([s])
    present, valid = g['present'], g['valid']
    ghost = present & ~valid
    assert g['mask_idx'][:11].tolist() == base['mask_idx'][:11].tolist()
    assert int(ghost.sum()) == 1 and bool(ghost[11])
    assert g['mask_idx'][11] == base['mask_idx'][10] + 1
    assert g['d'][:11].tolist() == base['d'][:11].tolist()
    assert tuple(s['patches'].shape) == (207, 24)
    assert tuple(s['targets'].shape) == (13, 6)
    assert g['true_bg_trajectory'].shape == (P * S,)
    # The ghost patch is masked, announced and attended — only scoring drops it.
    assert bool(batch['bg_formula_data']['present'][0, 11])
    assert not bool(batch['bg_formula_data']['valid'][0, 11])
    print(f"\n[DUMP] ghost 1 | pinned span {base['mask_idx'][7:11].tolist()} -> "
          f"{g['mask_idx'][7:12].tolist()}, ghost slot 11")


def test_the_ghost_steps_enter_no_loss_term():
    """Garbage in the ghost slot's target cannot move the loss, its parts, or any bucket."""
    from data import collate_fn
    with ghost_patches(1):
        batch = collate_fn(_samples())
        _m, total, parts, _med = _loss(batch)
        ghost = batch['bg_formula_data']['present'] & ~batch['bg_formula_data']['valid']
        assert int(ghost.sum()) == batch['patches'].shape[0]
        wrecked = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        wrecked['targets'] = batch['targets'].clone()
        wrecked['targets'][ghost] = torch.tensor([39.0, 399.0, 39.0, 399.0, 39.0, 399.0])
        _m2, total2, parts2, _med2 = _loss(wrecked)
    print(f"\n[DUMP] ghost target garbage | loss {float(total):.10f} vs {float(total2):.10f}")
    assert float(total) == float(total2)
    for k, v in parts.items():
        assert float(v) == float(parts2[k]), f"{k} moved with the ghost target"


def test_last_scored_patch_blends_like_an_interior_one_under_the_flag():
    """The property the flag buys: a real right-neighbour node for the last SCORED patch."""
    T = 24
    torch.manual_seed(5)
    x = torch.randn(1, T, cfg.D_MODEL)
    first = T - P - 1

    def _states(width: int, window: int):
        vis = torch.ones(1, window, dtype=torch.bool)
        vis[0, first:first + width] = False
        attn = create_attention_mask_from_visible(
            vis, torch.zeros(1, window, dtype=torch.bool))
        return step_states(x[:, :window], torch.arange(
            first, first + width).view(1, width), attn)[0]

    # Ghost: P+1 patches flush right at T. Reference: P patches, patch T-1 VISIBLE.
    h_ghost = _states(P + 1, T)
    h_interior = _states(P, T)
    # Pre-flag: the same P patches flush right, no right neighbour at all.
    h_flat = _states(P, T - 1)
    d_fix = float((h_ghost[P - 1] - h_interior[P - 1]).abs().max())
    d_gap = float((h_flat[P - 1] - h_interior[P - 1]).abs().max())
    print(f"\n[DUMP] last scored patch | ghost-vs-interior {d_fix:.3e}  "
          f"no-ghost-vs-interior {d_gap:.3e}")
    assert torch.equal(h_ghost[P - 1], h_interior[P - 1])
    assert d_gap > 1e-3, 'without the ghost the last patch already blends like an interior one'


def test_modified_forward_still_exports_under_the_flag():
    """The ghost is one more masked patch in the graph's fixed masked set, nothing more."""
    from exporters.modified_forward import (
        HeadRawForward, build_slot_selection, build_struct_mask_from_visible, window_labels)
    from model import T1DMAI

    T = 24
    with ghost_patches(1):
        W = cfg.FORECAST_SPAN_PATCHES
        visible, is_pad, idx = window_labels(T - W, None, T, W)
        struct = build_struct_mask_from_visible(visible, is_pad)
        slot_sel = build_slot_selection(idx, T=T, m_slots=len(idx))
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
    print(f"\n[DUMP] export under ghost 1 | shapes {[tuple(t.shape) for t in traced]} max|d| {d}")
    assert traced[0].shape == (1, P + 1, S, 1 + 2 * cfg.N_SPREADS)
    assert all(v < 1e-6 for v in d)


def test_a_ghost_checkpoint_refuses_to_export(tmp_path):
    """The phone's head implements 0 ghost patches, so the export loader rejects the flag."""
    from exporters.modified_forward import load_model
    from model import T1DMAI
    from utils import checkpoint_ghost_patches

    assert checkpoint_ghost_patches(None) == checkpoint_ghost_patches({}) == 0
    torch.manual_seed(0)
    ck = {'arch_version': cfg.ARCH_VERSION, 'input_layout': 'curves',
          'model_state_dict': T1DMAI().state_dict(),
          'training_config': {'ghost_patches': 1}}
    path = tmp_path / 'ghost.pt'
    torch.save(ck, path)
    print(f"\n[DUMP] export refusal | checkpoint ghost {checkpoint_ghost_patches(ck)}")
    with pytest.raises(AssertionError, match='implements 0 ghost patches only'):
        load_model(str(path))
    ck['training_config']['ghost_patches'] = 0
    torch.save(ck, path)
    load_model(str(path))


def test_predict_decodes_24_steps_under_the_flag():
    """The fan handed to a caller never carries the ghost: same shapes at 0 and at 1."""
    from inference import predict
    from model import T1DMAI

    torch.manual_seed(3)
    model = T1DMAI().eval()
    ctx = torch.randn(cfg.MIN_CONTEXT_PATCHES, S, cfg.N_INPUT_FEATURES)
    ctx[:, :, 0] = 0.0
    out0 = predict(model, ctx, normalization_stats=STATS)
    with ghost_patches(1):
        out1 = predict(model, ctx, normalization_stats=STATS)
    print(f"\n[DUMP] predict | median_bg {tuple(out0['median_bg'].shape)} -> "
          f"{tuple(out1['median_bg'].shape)}")
    for key in ('q_tau', 'median', 'median_bg', 'bands', 'mask_idx'):
        assert out0[key].shape == out1[key].shape, key
    assert out1['median_bg'].shape == (P * S,)
    assert out1['mask_idx'].tolist() == out0['mask_idx'].tolist()
    assert torch.isfinite(out1['bands']).all()


def test_both_spline_edge_rules_compose_with_the_flag():
    """The rule governs the ghost patch, and its reach into the scored ones all but vanishes.

    Node i+2 is virtual for the last scored patch either way, but it carries u**3/6 <= 0.013
    once a ghost node fills i+1, against the whole repeated-edge blend without one."""
    from tests.test_spline_edge import spline_edge
    T = 24
    torch.manual_seed(9)
    x = torch.randn(1, T, cfg.D_MODEL)
    first = T - P - 1

    def _both(width: int, window: int):
        vis = torch.ones(1, window, dtype=torch.bool)
        vis[0, first:first + width] = False
        attn = create_attention_mask_from_visible(
            vis, torch.zeros(1, window, dtype=torch.bool))
        idx = torch.arange(first, first + width).view(1, width)
        with spline_edge('repeat'):
            a = step_states(x[:, :window], idx, attn)[0]
        with spline_edge('extrapolate'):
            b = step_states(x[:, :window], idx, attn)[0]
        return a, b

    a_g, b_g = _both(P + 1, T)
    a_n, b_n = _both(P, T - 1)
    reach_ghost = float((a_g[P - 1] - b_g[P - 1]).abs().max())
    reach_none = float((a_n[P - 1] - b_n[P - 1]).abs().max())
    print(f"\n[DUMP] edge rules on the last SCORED patch | ghost 1 {reach_ghost:.3e}  "
          f"ghost 0 {reach_none:.3e}  on the ghost patch "
          f"{float((a_g[P] - b_g[P]).abs().max()):.3e}")
    assert not torch.equal(a_g[P], b_g[P]), 'the rule must still govern the ghost patch'
    assert reach_ghost < 0.1 * reach_none, 'the ghost node did not take the edge blend'
    assert torch.equal(a_g[:P - 1], b_g[:P - 1]), 'the rule reached an interior scored patch'


def test_the_random_window_path_refuses_the_flag():
    """A drawn masked set has no span pinned flush right, so the flag says so and stops."""
    from data import _build_sample
    row = _synthetic_row()
    with ghost_patches(1):
        with pytest.raises(RuntimeError, match='T1DMAI_GHOST_PATCHES=0'):
            _build_sample(data=row, icr=10.0, stats=STATS,
                          rng=np.random.default_rng(0), boundary=False)
    print('\n[DUMP] random-window path refuses --ghost-patches 1')
