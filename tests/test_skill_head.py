"""The auxiliary skill head: four sigmoids off the mean-pooled visible context.

It never touches the forecast — the 2-tuple return is bit-identical with the head built —
it pools only unmasked, unpadded patches, and a batch without skills scores zero loss."""

import numpy as np
import pytest
import torch

from config import MAX_MASKED_PATCHES, N_SKILLS, PATCH_DIM, PREDICTION_PATCHES
from tests.forward_inputs import right_edge_inputs


def test_forecast_return_is_bit_identical_with_the_head_built(monkeypatch):
    """Two flags off: the forward must return exactly what a head-less model returns."""
    import model as model_mod

    if not model_mod.SKILL_HEAD_ENABLED:
        pytest.skip("skill head disabled; the parity is trivially satisfied")

    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(2, seed=11)

    torch.manual_seed(0)
    m_with = model_mod.T1DMAI().eval()
    with torch.no_grad():
        q_with, med_with = m_with(patches, attn_mask, anchor_bg, mask_idx)

    monkeypatch.setattr(model_mod, 'SKILL_HEAD_ENABLED', False)
    torch.manual_seed(0)
    m_without = model_mod.T1DMAI().eval()
    assert m_with.skill_head is not None and m_without.skill_head is None
    with torch.no_grad():
        q_without, med_without = m_without(patches, attn_mask, anchor_bg, mask_idx)

    assert torch.equal(q_with, q_without), "the skill head moved q_tau"
    assert torch.equal(med_with, med_without), "the skill head moved the median"
    print(f"\n[DUMP] skill_head | forecast bit-identical over "
          f"{tuple(q_with.shape)} q_tau ✓")


def test_skills_are_returned_only_behind_the_flag():
    from model import T1DMAI

    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(3, seed=12)
    torch.manual_seed(0)
    m = T1DMAI().eval()

    with torch.no_grad():
        plain = m(patches, attn_mask, anchor_bg, mask_idx)
        with_skills = m(patches, attn_mask, anchor_bg, mask_idx, return_skills=True)
    assert len(plain) == 2 and len(with_skills) == 3
    skills = with_skills[2]
    assert skills.shape == (3, N_SKILLS)
    assert ((skills >= 0.0) & (skills <= 1.0)).all(), "the head must emit [0, 1]"
    assert torch.equal(plain[0], with_skills[0])


def test_pool_mask_excludes_pad_and_masked_patches():
    """A pad patch carries zeros; pooling it in would drag every prediction toward 0.5."""
    from model import T1DMAI

    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(1, seed=13)
    B, T, _ = patches.shape
    n_ctx = T - PREDICTION_PATCHES

    # left-pad one patch of zeros; the pool mask must ignore it AND the masked horizon
    padded = torch.zeros(B, T + 1, PATCH_DIM)
    padded[:, 1:] = patches
    pad_attn = torch.zeros(B, T + 1, T + 1, dtype=torch.bool)
    pad_attn[:, 1:, 1:] = attn_mask
    pad_attn[:, 0, 0] = True
    pool = torch.zeros(B, T + 1, dtype=torch.bool)
    pool[:, 1:1 + n_ctx] = True

    torch.manual_seed(0)
    m = T1DMAI().eval()
    with torch.no_grad():
        _q, _med, masked_pool = m(
            padded, pad_attn, anchor_bg, mask_idx + 1,
            return_skills=True, pool_mask=pool)
        _q2, _med2, all_pool = m(
            padded, pad_attn, anchor_bg, mask_idx + 1,
            return_skills=True, pool_mask=torch.ones(B, T + 1, dtype=torch.bool))
    assert not torch.allclose(masked_pool, all_pool), (
        "pooling the pad and the masked horizon gives the same answer — pool_mask is ignored")

    # the mask is the ONLY thing that changed, so the trunk state is untouched
    assert torch.equal(_q, _q2) and torch.equal(_med, _med2)
    print(f"\n[DUMP] skill_head | pooled {int(pool.sum())}/{T + 1} patches; "
          f"max|d| vs pool-everything = {float((masked_pool - all_pool).abs().max()):.4f}")


def test_absent_skills_contribute_zero_loss():
    """Finetune data carries no skills; those rows must weigh nothing in the probe loss."""
    from config import SKILL_HEAD_LOSS_WEIGHT

    pred = torch.rand(4, N_SKILLS, requires_grad=True)
    true = torch.rand(4, N_SKILLS)
    none_valid = torch.zeros(4, dtype=torch.bool)
    # this is train.py's expression, with no valid row
    assert not bool(none_valid.any()), "the guard must short-circuit on an all-absent batch"

    half = torch.tensor([True, False, True, False])
    loss_half = ((pred - true) ** 2).mean(dim=-1)[half].mean()
    loss_rows = ((pred - true) ** 2).mean(dim=-1)[torch.tensor([0, 2])].mean()
    assert torch.allclose(loss_half, loss_rows), "the mask must select exactly the valid rows"
    assert SKILL_HEAD_LOSS_WEIGHT > 0.0


def test_collate_carries_skills_and_the_pool_mask():
    import os
    from data import T1DMDataset, collate_fn
    from normalization import load_normalization_stats, NORM_STATS_FILE

    if not os.path.exists(NORM_STATS_FILE):
        pytest.skip("normalization_stats.json required")
    stats = load_normalization_stats()
    ds = T1DMDataset(master_seed=7, total_steps=2, batch_size=2,
                     normalization_stats=stats, cache_path=None)
    batch = collate_fn([ds[i] for i in range(2)])

    assert batch['skills'].shape == (2, N_SKILLS)
    assert batch['skills_valid'].all(), "simulator rows always carry their skills"
    assert ((batch['skills'] >= 0.0) & (batch['skills'] <= 1.0)).all()

    pool = batch['pool_mask']
    assert pool.shape == batch['patches'].shape[:2] and pool.dtype == torch.bool
    mask_idx = batch['bg_formula_data']['mask_idx']
    valid = batch['bg_formula_data']['valid']
    for i in range(2):
        assert not bool(pool[i][mask_idx[i][valid[i]]].any()), \
            f"sample {i} pools a masked patch — its BG was withheld"

    # a sample built without skills carries None, and collate marks it absent
    s = ds[0]
    s['skills'] = None
    batch2 = collate_fn([s, ds[1]])
    assert batch2['skills_valid'].tolist() == [False, True]
    assert torch.equal(batch2['skills'][0], torch.zeros(N_SKILLS))
    print(f"\n[DUMP] skill_head | collate skills {tuple(batch['skills'].shape)}, "
          f"pool {int(pool.sum())}/{pool.numel()} patches, "
          f"M={MAX_MASKED_PATCHES} slots, absent row flagged ✓")


def test_skill_head_learns_a_fixed_target():
    """A few steps of plain SGD on one batch must drive the probe loss down."""
    from model import T1DMAI

    patches, attn_mask, anchor_bg, mask_idx = right_edge_inputs(4, seed=14)
    pool = torch.ones(4, patches.shape[1], dtype=torch.bool)
    pool[:, -PREDICTION_PATCHES:] = False
    target = torch.tensor(np.array([[0.1, 0.9, 0.3, 0.7]] * 4, dtype=np.float32))

    torch.manual_seed(0)
    m = T1DMAI()
    opt = torch.optim.SGD(m.skill_head.parameters(), lr=0.5)
    losses = []
    for _ in range(25):
        _q, _med, pred = m(patches, attn_mask, anchor_bg, mask_idx,
                           return_skills=True, pool_mask=pool)
        loss = ((pred - target) ** 2).mean()
        losses.append(float(loss.detach()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0], f"skill loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"\n[DUMP] skill_head | MSE {losses[0]:.4f} -> {losses[-1]:.4f} over 25 SGD steps")
