"""K < PATCH_SIZE makes the within-patch period-2 median zigzag unrepresentable."""
import math

import torch

import config
from config import (PREDICTION_PATCHES, PATCH_SIZE, MAX_CONTEXT_PATCHES,
                    N_QUANTILES, BG_HEAD_STEP_BASIS_DIM)
from model import T1DMAI, make_step_basis
from tests.forward_inputs import right_edge_inputs


def test_make_step_basis_orthonormal_dct_and_poly():
    for kind in ('dct', 'poly'):
        B = make_step_basis(PATCH_SIZE, BG_HEAD_STEP_BASIS_DIM, kind)
        assert B.shape == (PATCH_SIZE, BG_HEAD_STEP_BASIS_DIM)
        gram = B.T @ B
        assert torch.allclose(gram, torch.eye(BG_HEAD_STEP_BASIS_DIM), atol=1e-5), (
            f"{kind} basis columns not orthonormal:\n{gram}")
    print(f"\n[DUMP] make_step_basis | dct & poly (PATCH_SIZE={PATCH_SIZE}, "
          f"K={BG_HEAD_STEP_BASIS_DIM}) orthonormal ✓")


def _within_patch_hf_energy(med):
    """(in-basis, excluded) DCT energy; excluded = k >= BG_HEAD_STEP_BASIS_DIM, DC dropped."""
    s = torch.arange(PATCH_SIZE, dtype=torch.float64)
    full = torch.stack(
        [torch.cos(math.pi * (s + 0.5) * k / PATCH_SIZE) for k in range(PATCH_SIZE)], dim=1)
    full = (full / full.norm(dim=0, keepdim=True)).float()        # (S, S)
    mp = med - med.mean(dim=-1, keepdim=True)
    coef = torch.einsum('bps,sk->bpk', mp, full)                  # (B,P,S) DCT coeffs
    incl = coef[..., :BG_HEAD_STEP_BASIS_DIM].pow(2).sum().item()
    excl = coef[..., BG_HEAD_STEP_BASIS_DIM:].pow(2).sum().item()
    return incl, excl


def test_within_patch_median_has_no_high_frequency_energy(monkeypatch):
    """'independent': excluded modes pinned to ~0. 'global': low-passed, suppressed not pinned."""
    torch.manual_seed(0)
    m = T1DMAI().eval()
    patches, attn, anchor_bg, mask_idx = right_edge_inputs(
        3, n_ctx=MAX_CONTEXT_PATCHES, anchor_mgdl=140.0, seed=0)

    monkeypatch.setattr(config, "BG_HEAD_MEDIAN_MODE", 'independent', raising=False)
    with torch.no_grad():
        q, med = m(patches, attn, anchor_bg, mask_idx)
    assert q.shape == (3, PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)
    assert torch.allclose(med, q[..., 3], atol=1e-6), "median must equal q_tau[...,3]"
    incl_i, excl_i = _within_patch_hf_energy(med)
    assert excl_i < 1e-8, f"'independent' excluded high-freq energy {excl_i:.3e} must be ~0"

    monkeypatch.setattr(config, "BG_HEAD_MEDIAN_MODE", 'global', raising=False)
    with torch.no_grad():
        _, med_g = m(patches, attn, anchor_bg, mask_idx)
    incl_g, excl_g = _within_patch_hf_energy(med_g)
    assert excl_g < 1e-3, f"'global' excluded high-freq energy {excl_g:.3e} not suppressed"
    assert excl_g < 1e-2 * max(incl_g, 1e-12), (
        f"'global' high-freq energy {excl_g:.3e} not << in-basis {incl_g:.3e}")
    print(f"\n[DUMP] within-patch median spectrum | independent EXCLUDED={excl_i:.3e} (~0) ; "
          f"global in-basis={incl_g:.4f} EXCLUDED={excl_g:.3e} (suppressed) ✓")
