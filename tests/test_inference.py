"""Tests for inference.py — standard, what-if, rolling prediction modes.

Risk-space redesign: ``predict`` returns risk-space ``q_tau`` / ``median`` plus
the mg/dL ``median_bg`` (the headline forecast = ``f_inv(median)``) and ``bands``
(= ``f_inv(q_tau)``). There are no ``mu`` / ``sigma_sq`` / ``mu_raw`` dynamics
fields. ``predict_rolling`` re-feeds BG autoregressively only.
"""

import numpy as np
import torch


def _make_context(n_ctx: int | None = None) -> torch.Tensor:
    """Build a random normalized context tensor (n_ctx, PATCH_SIZE, N_INPUT_FEATURES).

    The bg_absolute feature (index 0) is now RISK-space (Kovatchev f applied
    before the z-score) and is held at the normalized mean (z=0) so its
    denormalized ``last_bg`` anchor (z -> risk -> f_inv -> mg/dL) sits safely
    inside the physical BG band — a random N(0,1) bg slot can otherwise decode to
    a band edge. Every other feature stays random.
    """
    from config import PATCH_SIZE, N_INPUT_FEATURES, MIN_CONTEXT_PATCHES
    if n_ctx is None:
        n_ctx = MIN_CONTEXT_PATCHES
    ctx = torch.randn(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)
    ctx[:, :, 0] = 0.0  # bg z=0 -> channel mean (in-band) after denormalization
    return ctx


def _get_stats():
    """Load or compute normalization stats."""
    import os
    from normalization import compute_normalization_stats, load_normalization_stats, NORM_STATS_FILE
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def _sim_clamps() -> tuple[float, float]:
    import T1DMSIM.simulator as sim
    return float(sim.BG_CLAMP_MIN), float(sim.BG_CLAMP_MAX)


def test_standard_prediction():
    """Standard prediction produces correctly shaped risk-space quantiles plus
    an in-band mg/dL headline forecast (median_bg) and mg/dL bands."""
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
    # The headline BG forecast is f_inv(median): finite and inside the physical band.
    mb = result['median_bg']
    assert torch.isfinite(mb).all()
    assert (mb >= bg_min - 1e-3).all() and (mb <= bg_max + 1e-3).all(), (
        f"median_bg out of band: [{mb.min():.2f}, {mb.max():.2f}]")
    # Bands are ascending in τ and inside the band.
    bands = result['bands']
    assert (bands[..., 1:] - bands[..., :-1] >= -1e-3).all(), "bands must be ascending"
    assert (bands >= bg_min - 1e-3).all() and (bands <= bg_max + 1e-3).all()

    print(f"\n[DUMP] inference | median_bg range: [{mb.min():.2f}, {mb.max():.2f}]")
    print(f"[DUMP] inference | band spread @step0: "
          f"{(bands[0, 0, -1] - bands[0, 0, 0]).item():.2f} mg/dL")


def test_what_if_prediction():
    """What-if mode with an overridden carb channel runs without NaN and the
    override (routed to feat 1 via CHANNEL_TO_FEAT) actually changes the
    forecast. There are no mask bits — the model is always conditioned."""
    from inference import predict, predict_what_if
    from model import T1DMAI
    from config import PREDICTION_PATCHES, PATCH_SIZE

    model = T1DMAI()
    model.eval()
    stats = _get_stats()

    context = _make_context()

    # Standard prediction (carb/insulin at their zero-RAW normalized baseline).
    result_standard = predict(model, context, patient_seed=42, normalization_stats=stats)

    # What-if: override carb channel (output-channel index 0) with a big meal.
    carb_override = torch.zeros(PREDICTION_PATCHES, PATCH_SIZE)
    carb_override[2:5, :] = 5.0  # normalized big meal in patches 2-4

    result_whatif = predict_what_if(
        model, context, patient_seed=42,
        overrides={0: carb_override}, normalization_stats=stats,
    )

    diff = (result_whatif['median'] - result_standard['median']).abs().mean()
    print(f"\n[DUMP] what_if | mean median (risk) difference: {diff:.4f}")
    assert not torch.isnan(result_whatif['median']).any()
    # The override writes real input features into the pred-zone carb slot (feat 1),
    # so even an untrained model must produce a measurably different output.
    assert diff > 1e-4, (
        f"what-if override had no effect on the forecast (diff={diff:.2e}) — "
        f"overrides likely ignored")


def test_rolling_prediction():
    """Rolling prediction extends beyond a single window; requires stats; the
    re-feed is BG-autoregressive only."""
    import pytest
    from inference import predict_rolling
    from model import T1DMAI
    from config import PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES

    model = T1DMAI()
    model.eval()
    stats = _get_stats()
    bg_min, bg_max = _sim_clamps()

    context = _make_context()

    # No stats ⇒ hard fail (the BG renormalization for autoregressive context
    # patches can't be computed without it).
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

    # carry_spread invariant: the RISK-space fan is carried across roll
    # boundaries (it does not sawtooth-reset to a narrow band at each new roll),
    # so the central-90% half-width at the LAST step of a later roll is >= the
    # half-width at the last step of the first roll.  Measured on the mg/dL
    # bands at the terminal step (P-1, S-1) of each roll.
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

    # Re-fed carb/insulin phantom-dose fix (C-rolling-phantom): a roll with no
    # overrides seeds the appended carb/insulin feats from the zero-RAW normalized
    # baseline (``normalize(0)`` per sparse channel), NOT z=0.  Verified on the
    # only public bridge: that baseline z decodes back to ~0 g / ~0 U (whereas a
    # naive z=0 would denormalize to a phantom ~0.39 g / ~0.14 U dose).
    from normalization import normalize, denormalize
    import numpy as np
    from config import N_INPUT_FEATURES
    zero_raw_z = normalize(np.zeros((1, 3), dtype=np.float32), stats)
    decoded = denormalize(zero_raw_z, stats)[0]
    assert abs(float(decoded[1])) < 1e-3, (
        f"re-fed carb baseline must decode to ~0 g, got {float(decoded[1]):.4f}")
    assert abs(float(decoded[2])) < 1e-3, (
        f"re-fed insulin baseline must decode to ~0 U, got {float(decoded[2]):.4f}")
    # A z=0 carb/insulin slot would instead decode to a nonzero phantom dose.
    phantom = denormalize(
        np.zeros((1, 3), dtype=np.float32), stats)[0]
    assert float(phantom[1]) > 1e-3 or float(phantom[2]) > 1e-3, (
        "z=0 should decode to a phantom dose — the baseline fix is meaningful")

    print(f"\n[DUMP] rolling | {n_rolls} rolls, pred_bg range "
          f"[{pb.min():.2f}, {pb.max():.2f}]")
    print(f"[DUMP] rolling | roll-end band half-widths (risk-fan carry): "
          f"{[round(h, 3) for h in roll_end_halfwidth]}")
    print(f"[DUMP] rolling | zero-RAW baseline decodes carb={float(decoded[1]):.4f}g "
          f"insulin={float(decoded[2]):.4f}U vs z=0 phantom "
          f"carb={float(phantom[1]):.4f}g insulin={float(phantom[2]):.4f}U ✓")


def test_bands_bracket_median():
    """The mg/dL band edges bracket the headline median forecast: the lowest
    band (τ=.05) sits at/below and the highest (τ=.95) at/above median_bg at
    every horizon step."""
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
    # The mg/dL band-median (f_inv of risk median) equals median_bg.
    assert torch.allclose(mid, median_bg, atol=1e-2)
    print(f"\n[DUMP] bands | central-90% width @step0 = "
          f"{(hi[0, 0] - lo[0, 0]).item():.2f} mg/dL; median == band[...,3] ✓")


def test_predict_return_time_contract():
    """``predict(..., return_time=True)`` adds a ``time_pred`` key of shape
    ``(PREDICTION_PATCHES, TIME_PROBE_N_BINS)`` (raw logits) — or ``None`` when the
    probe is disabled — while the default call omits the key entirely and yields
    bit-identical ``q_tau`` / ``median`` (Slice A opt-in, backward-compat)."""
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

    # Default path never leaks the diagnostic key.
    assert 'time_pred' not in base, "default predict() must not add 'time_pred'"
    assert 'time_pred' in timed, "return_time=True must add 'time_pred'"

    # The forecast VALUE is untouched by the opt-in (diagnostic-only invariant).
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
    """``predict_what_if(..., return_time=True)`` forwards the flag so the dict
    gains ``time_pred``; the default call omits it."""
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
    """``predict_rolling(..., return_time=True)`` adds roll-0's per-patch
    ``time_pred`` ``(PREDICTION_PATCHES, TIME_PROBE_N_BINS)`` / ``None``; the default
    call omits it and stays bit-identical on ``pred_bg`` / ``q_tau`` / ``bands``."""
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

    # Rolling forecast VALUE is untouched (roll-0 time gating must not perturb it).
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
