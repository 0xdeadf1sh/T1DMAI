"""SPEC/inference.md §8.1: band offsets = span-mean spread × g(step) × (1 + β·|Δm|)."""

import math

import pytest
import torch
import torch.nn.functional as F


def _offsets(q, median_idx):
    m = q[..., median_idx].unsqueeze(-1)
    return q[..., median_idx + 1:] - m, m - q[..., :median_idx].flip(-1)


def _g(n, has_l, has_r):
    d = [min(j + 1, n - j) if has_l and has_r else j + 1 if has_l else n - j if has_r else 1
         for j in range(n)]
    root = [math.sqrt(x) for x in d]
    mean = sum(root) / n
    return torch.tensor([r / mean for r in root], dtype=torch.float64)


def test_tied_offsets_are_span_mean_times_growth_times_median_scale():
    from utils import assemble_quantiles
    from config import QUANTILE_LEVELS, N_SPREADS, BG_QUANTILE_SPREAD_MIN

    torch.manual_seed(3)
    B, M, S, beta = 2, 4, 6, 0.7
    head_raw = torch.randn(B, M, S, 1 + 2 * N_SPREADS)
    mask_idx = torch.tensor([[10, 11, 20, 21], [5, 6, 7, 30]])
    anchor = torch.tensor([[120.0, 120.0, 90.0, 90.0], [150.0, 150.0, 150.0, 70.0]])
    has_l = torch.tensor([[True, True, True, True], [False, False, False, True]])
    has_r = torch.tensor([[True, True, False, False], [True, True, True, False]])
    q, _ = assemble_quantiles(head_raw, anchor, mask_idx, tie_beta=torch.tensor(beta),
                              sides=(has_l, has_r))
    up, dn = _offsets(q, QUANTILE_LEVELS.index(0.5))

    spread = F.softplus(head_raw[..., 1:]) + BG_QUANTILE_SPREAD_MIN
    scale = 1.0 + beta * head_raw[..., 0].abs()
    spans = {0: [[0, 1], [2, 3]], 1: [[0, 1, 2], [3]]}
    for b, groups in spans.items():
        for g in groups:
            mean = spread[b, g].mean(dim=(0, 1))
            grow = _g(len(g) * S, bool(has_l[b, g[0]]), bool(has_r[b, g[0]])).view(len(g), S)
            k = (grow.float() * scale[b, g]).unsqueeze(-1)
            assert torch.allclose(up[b, g], mean[:N_SPREADS].cumsum(-1) * k, atol=1e-5)
            assert torch.allclose(dn[b, g], mean[N_SPREADS:].cumsum(-1) * k, atol=1e-5)
    assert (q.diff(dim=-1) > 0).all(), "the tied fan stays strictly ascending"
    print("\n[DUMP] spread tie | offsets = span-mean cumsum × g × (1+β|Δm|), per span ✓")


def test_growth_profile_per_geometry():
    n = 24
    fc, bc, inf = _g(n, True, False), _g(n, False, True), _g(n, True, True)
    assert (fc.diff() > 0).all() and (bc.diff() < 0).all(), "grows away from the read side"
    assert torch.allclose(inf, inf.flip(0)) and inf[0] < inf[n // 2 - 1]
    for g in (fc, bc, inf, _g(n, False, False)):
        assert abs(float(g.mean()) - 1.0) < 1e-12, "g averages 1 over the span"
    assert float(fc[-1] / fc[0]) == pytest.approx(math.sqrt(n))
    print(f"\n[DUMP] growth | forecast g {float(fc[0]):.3f} → {float(fc[-1]):.3f}")


def test_forward_grows_the_forecast_band_from_its_origin():
    from utils import assemble_quantiles
    from config import QUANTILE_LEVELS, N_SPREADS

    head_raw = torch.zeros(1, 4, 6, 1 + 2 * N_SPREADS)
    q, _ = assemble_quantiles(head_raw, torch.tensor([120.0]), tie_beta=1.0)
    up, _ = _offsets(q, QUANTILE_LEVELS.index(0.5))
    edge = up[0, ..., -1].reshape(-1)
    assert (edge.diff() > 0).all(), "a flat median's forecast band still widens with the horizon"


def test_growth_and_beta_travel_in_the_state_dict(monkeypatch):
    import model as model_mod

    monkeypatch.setattr(model_mod, "SPREAD_TIE_BETA", 0.5)
    tied = model_mod.T1DMAI()
    assert float(tied.state_dict()["spread_tie_beta"]) == 0.5
    for key in ("spread_tie_beta", "spread_growth"):
        old = {k: v for k, v in tied.state_dict().items() if k != key}
        with pytest.raises(RuntimeError, match=key):
            model_mod.T1DMAI().load_state_dict(old, strict=True)


@pytest.mark.parametrize("key", ["spread_tie_beta", "spread_growth"])
def test_exporter_refuses_a_checkpoint_without_the_marker(tmp_path, key):
    import model as model_mod
    from exporters.modified_forward import load_model

    sd = {k: v for k, v in model_mod.T1DMAI().state_dict().items() if k != key}
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": sd}, path)
    with pytest.raises(AssertionError, match=key):
        load_model(str(path))
