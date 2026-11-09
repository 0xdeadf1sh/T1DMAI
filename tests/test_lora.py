import torch
from torch import nn

import lora


def _net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.SiLU(), nn.Linear(8, 3, bias=False))


def test_attach_is_identity_and_freezes_base():
    net, ref = _net(), _net()
    x = torch.randn(5, 6)
    n = lora.attach(net, rank=2, alpha=4.0)
    assert n == 2 * 6 + 8 * 2 + 2 * 8 + 3 * 2
    assert torch.allclose(net(x), ref(x))
    trainable = {k for k, p in net.named_parameters() if p.requires_grad}
    assert trainable == {'0.lora_A', '0.lora_B', '2.lora_A', '2.lora_B'}


def test_merged_state_dict_reproduces_adapted_output():
    net = _net()
    lora.attach(net, rank=2, alpha=4.0)
    with torch.no_grad():
        for m in net.modules():
            if hasattr(m, 'lora_B'):
                m.lora_B.normal_()
    x = torch.randn(5, 6)
    plain = _net()
    plain.load_state_dict(lora.merged(net, net.state_dict()), strict=True)
    assert torch.allclose(net(x), plain(x), atol=1e-5)
