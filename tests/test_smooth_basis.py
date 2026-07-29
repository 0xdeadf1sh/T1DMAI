"""Tests for the smooth-basis BG head (structural anti-oscillation).

Pins the two structural fixes for the far-horizon median heartbeat:

* The per-patch head emits K = BG_HEAD_STEP_BASIS_DIM coefficients expanded across
  the PATCH_SIZE within-patch steps by a fixed orthonormal basis, so a within-patch
  period-2 zigzag of the median is UNREPRESENTABLE when K < PATCH_SIZE.
* The R3 'global' median mode is a low-frequency projection over the full P*S
  horizon, so the within-patch high-frequency modes are suppressed structurally.

(Both the permissive median-SEAM penalty L_seam and the median-curvature penalty
L_smooth are gone; the smooth-basis head and the global-median low-pass carry the
anti-oscillation structurally, and DILATE supplies the shape supervision.)
"""
import math

import torch

import config
from config import (PATCH_DIM, PREDICTION_PATCHES, PATCH_SIZE, MAX_CONTEXT_PATCHES,
                    N_QUANTILES, BG_HEAD_STEP_BASIS_DIM)
from model import T1DMAI, make_step_basis
from utils import create_attention_mask


def test_make_step_basis_orthonormal_dct_and_poly():
    """Both basis kinds return (PATCH_SIZE, K) with L2-orthonormal columns."""
    for kind in ('dct', 'poly'):
        B = make_step_basis(PATCH_SIZE, BG_HEAD_STEP_BASIS_DIM, kind)
        assert B.shape == (PATCH_SIZE, BG_HEAD_STEP_BASIS_DIM)
        gram = B.T @ B
        assert torch.allclose(gram, torch.eye(BG_HEAD_STEP_BASIS_DIM), atol=1e-5), (
            f"{kind} basis columns not orthonormal:\n{gram}")
    print(f"\n[DUMP] make_step_basis | dct & poly (PATCH_SIZE={PATCH_SIZE}, "
          f"K={BG_HEAD_STEP_BASIS_DIM}) orthonormal ✓")


def _within_patch_hf_energy(med):
    """Energy of the median's within-patch DCT modes k >= BG_HEAD_STEP_BASIS_DIM (the
    period-2 zigzag band), summed over the batch/patches after dropping each patch's
    DC level."""
    s = torch.arange(PATCH_SIZE, dtype=torch.float64)
    full = torch.stack(
        [torch.cos(math.pi * (s + 0.5) * k / PATCH_SIZE) for k in range(PATCH_SIZE)], dim=1)
    full = (full / full.norm(dim=0, keepdim=True)).float()        # (S, S)
    mp = med - med.mean(dim=-1, keepdim=True)                     # drop DC
    coef = torch.einsum('bps,sk->bpk', mp, full)                  # (B,P,S) DCT coeffs
    incl = coef[..., :BG_HEAD_STEP_BASIS_DIM].pow(2).sum().item()
    excl = coef[..., BG_HEAD_STEP_BASIS_DIM:].pow(2).sum().item()
    return incl, excl


def test_within_patch_median_has_no_high_frequency_energy(monkeypatch):
    """Two anti-oscillation guarantees on the within-patch median spectrum:

    * Under the legacy 'independent' / 'cumulative' median modes the median is exactly
      the per-patch ``step_basis`` expansion (``K < PATCH_SIZE``), so its energy in the
      EXCLUDED high-frequency DCT modes (k >= K) is ~0 EXACTLY — the period-2 zigzag is
      structurally unrepresentable by the head.
    * Under the R3 default 'global' mode the median is a GLOBAL low-frequency projection
      over the full P*S horizon, so the within-patch high modes are no longer pinned to
      exactly 0 by the head construction — but the global low-pass (G small) still
      SUPPRESSES them to a negligible level (orders of magnitude below the in-basis
      energy). The period-2 zigzag is removed structurally either way."""
    torch.manual_seed(0)
    m = T1DMAI().eval()
    n_ctx = MAX_CONTEXT_PATCHES
    patches = torch.randn(3, n_ctx + PREDICTION_PATCHES, PATCH_DIM)
    attn = create_attention_mask(n_ctx, PREDICTION_PATCHES)
    last_bg = torch.full((3,), 140.0)

    # Legacy 'independent': median == anchor + per-patch step_basis curve ⇒ exact ~0.
    monkeypatch.setattr(config, "BG_HEAD_MEDIAN_MODE", 'independent', raising=False)
    with torch.no_grad():
        q, med = m(patches, attn, last_bg)
    assert q.shape == (3, PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)
    assert torch.allclose(med, q[..., 3], atol=1e-6), "median must equal q_tau[...,3]"
    incl_i, excl_i = _within_patch_hf_energy(med)
    assert excl_i < 1e-8, f"'independent' excluded high-freq energy {excl_i:.3e} must be ~0"

    # R3 'global': within-patch high modes merely SUPPRESSED (not pinned), but tiny.
    monkeypatch.setattr(config, "BG_HEAD_MEDIAN_MODE", 'global', raising=False)
    with torch.no_grad():
        _, med_g = m(patches, attn, last_bg)
    incl_g, excl_g = _within_patch_hf_energy(med_g)
    assert excl_g < 1e-3, f"'global' excluded high-freq energy {excl_g:.3e} not suppressed"
    assert excl_g < 1e-2 * max(incl_g, 1e-12), (
        f"'global' high-freq energy {excl_g:.3e} not << in-basis {incl_g:.3e}")
    print(f"\n[DUMP] within-patch median spectrum | independent EXCLUDED={excl_i:.3e} (~0) ; "
          f"global in-basis={incl_g:.4f} EXCLUDED={excl_g:.3e} (suppressed) ✓")
