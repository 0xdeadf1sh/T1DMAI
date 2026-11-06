"""config.SPREAD_TIE_TO_MEDIAN: band offsets = span-mean spread × (1 + β·|Δm|), off-SPEC."""

import pytest
import torch
import torch.nn.functional as F


def _offsets(q, median_idx):
    m = q[..., median_idx].unsqueeze(-1)
    return q[..., median_idx + 1:] - m, m - q[..., :median_idx].flip(-1)


def test_tied_offsets_are_span_mean_times_median_scale():
    from utils import assemble_quantiles
    from config import QUANTILE_LEVELS, N_SPREADS, BG_QUANTILE_SPREAD_MIN

    torch.manual_seed(3)
    B, M, S, beta = 2, 4, 6, 0.7
    head_raw = torch.randn(B, M, S, 1 + 2 * N_SPREADS)
    mask_idx = torch.tensor([[10, 11, 20, 21], [5, 6, 7, 30]])
    anchor = torch.tensor([[120.0, 120.0, 90.0, 90.0], [150.0, 150.0, 150.0, 70.0]])
    q, m = assemble_quantiles(head_raw, anchor, mask_idx, tie_beta=torch.tensor(beta))
    up, dn = _offsets(q, QUANTILE_LEVELS.index(0.5))

    spread = F.softplus(head_raw[..., 1:]) + BG_QUANTILE_SPREAD_MIN
    scale = 1.0 + beta * head_raw[..., 0].abs()
    spans = {0: [[0, 1], [2, 3]], 1: [[0, 1, 2], [3]]}
    for b, groups in spans.items():
        for g in groups:
            mean = spread[b, g].mean(dim=(0, 1))
            want_up = mean[:N_SPREADS].cumsum(-1) * scale[b, g].unsqueeze(-1)
            want_dn = mean[N_SPREADS:].cumsum(-1) * scale[b, g].unsqueeze(-1)
            assert torch.allclose(up[b, g], want_up, atol=1e-5)
            assert torch.allclose(dn[b, g], want_dn, atol=1e-5)
    assert (q.diff(dim=-1) > 0).all(), "the tied fan stays strictly ascending"
    print("\n[DUMP] spread tie | offsets = span-mean cumsum × (1+β|Δm|), per span ✓")


def test_tied_edges_follow_a_diving_median():
    from utils import assemble_quantiles, kovatchev_f
    from config import QUANTILE_LEVELS, N_SPREADS

    S = 6
    head_raw = torch.zeros(1, 4, S, 1 + 2 * N_SPREADS)
    dive = -torch.linspace(0.0, 3.0, 4 * S).view(4, S)
    head_raw[0, :, :, 0] = dive
    head_raw[0, :, :, 1:] = torch.randn(4, S, 2 * N_SPREADS) * 3.0
    mask_idx = torch.tensor([[40, 41, 42, 43]])
    anchor = torch.full((1, 4), 110.0)
    q, m = assemble_quantiles(head_raw, anchor, mask_idx, tie_beta=torch.tensor(0.0))
    lo = q[0, ..., 0].reshape(-1)
    want = kovatchev_f(torch.tensor(110.0)) + dive.reshape(-1)
    off = want - lo
    assert torch.allclose(off, off[0].expand_as(off), atol=1e-5), "β=0: edge parallel to median"
    i = QUANTILE_LEVELS.index(0.5)
    assert torch.allclose(q[0, ..., i].reshape(-1), want, atol=1e-5)
    print(f"\n[DUMP] spread tie | β=0 lower edge parallel to a {float(-dive.min()):.1f}-risk dive ✓")


def test_tie_buffer_fails_a_strict_load_across_settings(monkeypatch):
    import model as model_mod

    monkeypatch.setattr(model_mod, "SPREAD_TIE_TO_MEDIAN", False)
    plain = model_mod.T1DMAI()
    assert "spread_tie_beta" not in plain.state_dict()
    monkeypatch.setattr(model_mod, "SPREAD_TIE_TO_MEDIAN", True)
    monkeypatch.setattr(model_mod, "SPREAD_TIE_BETA", 0.5)
    tied = model_mod.T1DMAI()
    assert float(tied.state_dict()["spread_tie_beta"]) == 0.5
    assert sum(p.numel() for p in tied.parameters()) == sum(p.numel() for p in plain.parameters())
    with pytest.raises(RuntimeError, match="spread_tie_beta"):
        plain.load_state_dict(tied.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="spread_tie_beta"):
        tied.load_state_dict(plain.state_dict(), strict=True)


def test_exporter_refuses_a_tied_checkpoint(tmp_path, monkeypatch):
    import model as model_mod
    from exporters.modified_forward import load_model

    monkeypatch.setattr(model_mod, "SPREAD_TIE_TO_MEDIAN", True)
    path = tmp_path / "tied.pt"
    torch.save({"model_state_dict": model_mod.T1DMAI().state_dict()}, path)
    with pytest.raises(AssertionError, match="SPREAD_TIE_TO_MEDIAN"):
        load_model(str(path))
