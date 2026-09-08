"""inference.py — standard, what-if and rolling prediction.
``predict`` returns risk-space ``q_tau`` / ``median`` plus mg/dL
``median_bg = f_inv(median)`` and ``bands = f_inv(q_tau)``. ``predict_rolling``
re-feeds BG autoregressively only."""

import numpy as np
import torch


def _make_context(n_ctx: int | None = None) -> torch.Tensor:
    """Random normalized ``(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)`` context.

    feat 0 is held at z = 0 so its anchor decodes well inside the physical BG band;
    a random N(0,1) bg slot can decode to a band edge.
    """
    from config import PATCH_SIZE, N_INPUT_FEATURES, MIN_CONTEXT_PATCHES
    if n_ctx is None:
        n_ctx = MIN_CONTEXT_PATCHES
    ctx = torch.randn(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)
    ctx[:, :, 0] = 0.0  # bg z=0 decodes to the channel mean, in band
    return ctx


def _get_stats():
    import os
    from normalization import compute_normalization_stats, load_normalization_stats, NORM_STATS_FILE
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def _sim_clamps() -> tuple[float, float]:
    import T1DMSIM.simulator as sim
    return float(sim.BG_CLAMP_MIN), float(sim.BG_CLAMP_MAX)


def test_standard_prediction():
    from inference import predict
    from model import T1DMAI
    from config import PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES

    model = T1DMAI()
    model.eval()
    stats = _get_stats()
    bg_min, bg_max = _sim_clamps()

    context = _make_context()
    result = predict(model, context, patient_seed=42, normalization_stats=stats)

    assert result['q_tau'].shape == (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)
    assert result['median'].shape == (PREDICTION_PATCHES, PATCH_SIZE)
    assert result['median_bg'].shape == (PREDICTION_PATCHES * PATCH_SIZE,)
    assert result['bands'].shape == (PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)

    assert not torch.isnan(result['q_tau']).any()
    # the headline forecast is f_inv(median), so it must sit inside the physical band
    mb = result['median_bg']
    assert torch.isfinite(mb).all()
    assert (mb >= bg_min - 1e-3).all() and (mb <= bg_max + 1e-3).all(), (
        f"median_bg out of band: [{mb.min():.2f}, {mb.max():.2f}]")
    bands = result['bands']
    assert (bands[..., 1:] - bands[..., :-1] >= -1e-3).all(), "bands must be ascending"
    assert (bands >= bg_min - 1e-3).all() and (bands <= bg_max + 1e-3).all()

    print(f"\n[DUMP] inference | median_bg range: [{mb.min():.2f}, {mb.max():.2f}]")
    print(f"[DUMP] inference | band spread @step0: "
          f"{(bands[0, 0, -1] - bands[0, 0, 0]).item():.2f} mg/dL")


def test_what_if_prediction():
    """The carb override routes to feat 1 through CHANNEL_TO_FEAT; the model is always
    conditioned, so there are no mask bits."""
    from inference import predict, predict_what_if
    from model import T1DMAI
    from config import PREDICTION_PATCHES, PATCH_SIZE

    model = T1DMAI()
    model.eval()
    stats = _get_stats()

    context = _make_context()

    # carb/insulin at their zero-RAW normalized baseline
    result_standard = predict(model, context, patient_seed=42, normalization_stats=stats)

    # output-channel index 0 is carb
    carb_override = torch.zeros(PREDICTION_PATCHES, PATCH_SIZE)
    carb_override[2:5, :] = 5.0  # normalized big meal in patches 2-4

    result_whatif = predict_what_if(
        model, context, patient_seed=42,
        overrides={0: carb_override}, normalization_stats=stats,
    )

    diff = (result_whatif['median'] - result_standard['median']).abs().mean()
    print(f"\n[DUMP] what_if | mean median (risk) difference: {diff:.4f}")
    assert not torch.isnan(result_whatif['median']).any()
    # the override writes real features into the carb slot, so even an untrained model must move
    assert diff > 1e-4, (
        f"what-if override had no effect on the forecast (diff={diff:.2e}) — "
        f"overrides likely ignored")


def test_rolling_prediction():
    """The re-feed is BG-autoregressive only, and stats are required."""
    import pytest
    from inference import predict_rolling
    from model import T1DMAI
    from config import PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES

    model = T1DMAI()
    model.eval()
    stats = _get_stats()
    bg_min, bg_max = _sim_clamps()

    context = _make_context()

    # no stats ⇒ hard fail: the autoregressive BG renormalization needs them
    with pytest.raises(ValueError, match="normalization_stats"):
        predict_rolling(model, context, patient_seed=42, n_rolls=3)

    n_rolls = 3
    result = predict_rolling(
        model, context, patient_seed=42, n_rolls=n_rolls,
        normalization_stats=stats,
    )

    expected_patches = n_rolls * PREDICTION_PATCHES
    assert result['q_tau'].shape[0] == expected_patches, (
        f"Expected {expected_patches} patches, got {result['q_tau'].shape[0]}")
    assert result['q_tau'].shape[-1] == N_QUANTILES
    assert result['bands'].shape[0] == expected_patches
    assert 'pred_bg' in result, "pred_bg (mg/dL trajectory) must be present"
    assert result['pred_bg'].shape[0] == expected_patches * PATCH_SIZE

    pb = result['pred_bg']
    assert torch.isfinite(pb).all()
    assert (pb >= bg_min - 1e-3).all() and (pb <= bg_max + 1e-3).all(), (
        f"rolled pred_bg out of band: [{pb.min():.2f}, {pb.max():.2f}]")
    assert not torch.isnan(result['q_tau']).any()

    # risk-space fan carries across rolls: later roll's terminal-step half-width >= first's.
    bands = result['bands'].reshape(
        n_rolls, PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)
    roll_end_halfwidth = [
        float((bands[r, -1, -1, -1] - bands[r, -1, -1, 0]) * 0.5)
        for r in range(n_rolls)
    ]
    assert all(b >= a - 1e-3 for a, b in
               zip(roll_end_halfwidth, roll_end_halfwidth[1:])), (
        f"band fan must not reset across rolls (non-decreasing carry): "
        f"{roll_end_halfwidth}")
    assert roll_end_halfwidth[-1] >= roll_end_halfwidth[0] - 1e-3, (
        f"final-roll fan narrower than first roll — carry_spread lost: "
        f"{roll_end_halfwidth}")

    # No-override roll seeds feats from normalize(0), not z=0: z=0 -> phantom ~0.39g/0.14U/0.025g.
    from normalization import normalize, denormalize
    import numpy as np
    from config import N_INPUT_FEATURES, CHANNEL_TO_FEAT
    zero_raw_z = normalize(
        np.zeros((1, N_INPUT_FEATURES), dtype=np.float32), stats)
    decoded = denormalize(zero_raw_z, stats)[0]
    assert abs(float(decoded[1])) < 1e-3, (
        f"re-fed carb baseline must decode to ~0 g, got {float(decoded[1]):.4f}")
    assert abs(float(decoded[2])) < 1e-3, (
        f"re-fed insulin baseline must decode to ~0 U, got {float(decoded[2]):.4f}")
    exercise_feat = CHANNEL_TO_FEAT[2]
    assert abs(float(decoded[exercise_feat])) < 1e-3, (
        f"re-fed exercise baseline must decode to ~0 g/step, got "
        f"{float(decoded[exercise_feat]):.4f}")
    # z=0 decodes to a nonzero phantom dose
    phantom = denormalize(
        np.zeros((1, N_INPUT_FEATURES), dtype=np.float32), stats)[0]
    assert float(phantom[1]) > 1e-3 or float(phantom[2]) > 1e-3, (
        "z=0 should decode to a phantom dose — the baseline fix is meaningful")
    assert float(phantom[exercise_feat]) > 1e-3, (
        "z=0 in the exercise slot should decode to a phantom session — feat 3 "
        "left at literal 0.0 on an unconditioned roll is the trap this pins")

    print(f"\n[DUMP] rolling | {n_rolls} rolls, pred_bg range "
          f"[{pb.min():.2f}, {pb.max():.2f}]")
    print(f"[DUMP] rolling | roll-end band half-widths (risk-fan carry): "
          f"{[round(h, 3) for h in roll_end_halfwidth]}")
    print(f"[DUMP] rolling | zero-RAW baseline decodes carb={float(decoded[1]):.4f}g "
          f"insulin={float(decoded[2]):.4f}U exercise="
          f"{float(decoded[exercise_feat]):.4f}g/step vs z=0 phantom "
          f"carb={float(phantom[1]):.4f}g insulin={float(phantom[2]):.4f}U "
          f"exercise={float(phantom[exercise_feat]):.4f}g/step ✓")


def test_bands_bracket_median():
    from inference import predict
    from model import T1DMAI
    from config import PREDICTION_PATCHES, PATCH_SIZE

    model = T1DMAI()
    model.eval()
    stats = _get_stats()

    context = _make_context()
    result = predict(model, context, patient_seed=42, normalization_stats=stats)

    bands = result['bands']                                   # (P, S, 7) mg/dL
    median_bg = result['median_bg'].reshape(PREDICTION_PATCHES, PATCH_SIZE)
    lo = bands[..., 0]
    hi = bands[..., -1]
    mid = bands[..., 3]
    assert (lo <= mid + 1e-3).all() and (hi >= mid - 1e-3).all()
    # the mg/dL band median, f_inv of the risk median, equals median_bg
    assert torch.allclose(mid, median_bg, atol=1e-2)
    print(f"\n[DUMP] bands | central-90% width @step0 = "
          f"{(hi[0, 0] - lo[0, 0]).item():.2f} mg/dL; median == band[...,3] ✓")


def test_predict_return_time_contract():
    """``time_pred`` is raw logits, or ``None`` when the probe is off; the default call
    omits the key entirely and leaves ``q_tau`` / ``median`` bit-identical."""
    from inference import predict
    from model import T1DMAI
    from config import PREDICTION_PATCHES, TIME_PROBE_N_BINS, TIME_PROBE_ENABLED

    model = T1DMAI()
    model.eval()
    stats = _get_stats()

    context = _make_context()
    base = predict(model, context, patient_seed=42, normalization_stats=stats)
    timed = predict(model, context, patient_seed=42, normalization_stats=stats,
                    return_time=True)

    assert 'time_pred' not in base, "default predict() must not add 'time_pred'"
    assert 'time_pred' in timed, "return_time=True must add 'time_pred'"

    # the forecast VALUE is untouched by the opt-in
    assert torch.equal(base['q_tau'], timed['q_tau']), "q_tau drifted with return_time"
    assert torch.equal(base['median'], timed['median']), "median drifted with return_time"

    tp = timed['time_pred']
    if TIME_PROBE_ENABLED:
        assert tp is not None, "probe enabled ⇒ time_pred must be a tensor"
        assert tp.shape == (PREDICTION_PATCHES, TIME_PROBE_N_BINS), (
            f"expected ({PREDICTION_PATCHES}, {TIME_PROBE_N_BINS}), got {tuple(tp.shape)}")
        assert torch.isfinite(tp).all(), "logits must be finite"
        print(f"\n[DUMP] predict return_time | time_pred shape {tuple(tp.shape)} "
              f"logit range [{tp.min():.3f}, {tp.max():.3f}]")
    else:
        assert tp is None, "probe disabled ⇒ time_pred must be None"
        print("\n[DUMP] predict return_time | probe disabled, time_pred is None")


def test_predict_what_if_return_time_forwarded():
    from inference import predict_what_if
    from model import T1DMAI
    from config import PREDICTION_PATCHES, PATCH_SIZE, TIME_PROBE_N_BINS, TIME_PROBE_ENABLED

    model = T1DMAI()
    model.eval()
    stats = _get_stats()

    context = _make_context()
    carb_override = torch.zeros(PREDICTION_PATCHES, PATCH_SIZE)
    carb_override[2:5, :] = 5.0

    base = predict_what_if(model, context, patient_seed=42,
                           overrides={0: carb_override}, normalization_stats=stats)
    timed = predict_what_if(model, context, patient_seed=42,
                            overrides={0: carb_override}, normalization_stats=stats,
                            return_time=True)

    assert 'time_pred' not in base, "default predict_what_if() must not add 'time_pred'"
    assert 'time_pred' in timed, "return_time=True must forward to predict()"
    tp = timed['time_pred']
    if TIME_PROBE_ENABLED:
        assert tp is not None and tp.shape == (PREDICTION_PATCHES, TIME_PROBE_N_BINS)
    else:
        assert tp is None
    print(f"\n[DUMP] what_if return_time | time_pred "
          f"{None if tp is None else tuple(tp.shape)}")


def test_predict_rolling_return_time_contract():
    """``time_pred`` here is ROLL 0's, ``(PREDICTION_PATCHES, TIME_PROBE_N_BINS)``."""
    from inference import predict_rolling
    from model import T1DMAI
    from config import PREDICTION_PATCHES, TIME_PROBE_N_BINS, TIME_PROBE_ENABLED

    model = T1DMAI()
    model.eval()
    stats = _get_stats()

    context = _make_context()
    n_rolls = 3
    base = predict_rolling(model, context, patient_seed=42, n_rolls=n_rolls,
                           normalization_stats=stats)
    timed = predict_rolling(model, context, patient_seed=42, n_rolls=n_rolls,
                            normalization_stats=stats, return_time=True)

    assert 'time_pred' not in base, "default predict_rolling() must not add 'time_pred'"
    assert 'time_pred' in timed, "return_time=True must add roll-0 'time_pred'"

    # the rolling forecast VALUE is untouched by the opt-in
    for key in ('pred_bg', 'q_tau', 'bands'):
        assert torch.equal(base[key], timed[key]), f"{key} drifted with return_time"

    tp = timed['time_pred']
    if TIME_PROBE_ENABLED:
        assert tp is not None, "probe enabled ⇒ roll-0 time_pred must be a tensor"
        assert tp.shape == (PREDICTION_PATCHES, TIME_PROBE_N_BINS), (
            f"roll-0 time_pred: expected ({PREDICTION_PATCHES}, {TIME_PROBE_N_BINS}), "
            f"got {tuple(tp.shape)}")
        assert torch.isfinite(tp).all()
        print(f"\n[DUMP] rolling return_time | roll-0 time_pred shape {tuple(tp.shape)}")
    else:
        assert tp is None
        print("\n[DUMP] rolling return_time | probe disabled, time_pred is None")
