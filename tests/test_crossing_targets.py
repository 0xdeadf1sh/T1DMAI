"""``utils.crossing_targets``: cumulative within a span, restarting at span starts, zero on pads."""
import torch

from utils import crossing_targets


def test_cumulative_restarts_at_span_start_and_zeroes_padded_slots():
    # Two spans: patches [3,4] and [7]; slot 3 padded (gathers patch 0). S = 3.
    true_bg = torch.tensor([[
        [120.0, 65.0, 80.0],     # slot 0, span A: crosses hypo at step 1
        [90.0, 200.0, 100.0],    # slot 1, span A: hyper at step 1; hypo carried from slot 0
        [150.0, 60.0, 190.0],    # slot 2, span B: restarts; hypo at step 1, hyper at step 2
        [50.0, 50.0, 50.0],      # slot 3, padded
    ]])
    mask_idx = torch.tensor([[3, 4, 7, 0]])
    valid = torch.tensor([[True, True, True, False]])
    out = crossing_targets(true_bg, mask_idx, valid, 70.0, 180.0)
    assert out.shape == (1, 4, 3, 2)
    hypo, hyper = out[0, ..., 0], out[0, ..., 1]
    assert hypo.tolist() == [[0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 0]]
    assert hyper.tolist() == [[0, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]]
    print(f"\n[DUMP] crossing_targets | hypo {hypo.tolist()} hyper {hyper.tolist()}")


def test_single_span_is_a_running_any():
    true_bg = torch.tensor([[[100.0, 100.0], [69.0, 100.0], [100.0, 100.0]]])
    mask_idx = torch.tensor([[5, 6, 7]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    out = crossing_targets(true_bg, mask_idx, valid, 70.0, 180.0)
    assert out[0, :, :, 0].tolist() == [[0, 0], [1, 1], [1, 1]]
    assert out[0, :, :, 1].sum() == 0
