"""Tests for utils.py — seed hashing, attention mask, the Kovatchev risk
transform (the sole mg/dL<->risk bridge), quantile assembly, and the
three-hop normalize/f/f_inv round-trip."""

import math

import pytest
import torch


def test_seed_hash_deterministic():
    from utils import compute_patient_seed
    seed1 = compute_patient_seed(42, 100, 5)
    seed2 = compute_patient_seed(42, 100, 5)
    assert seed1 == seed2


def test_seed_hash_unique():
    from utils import compute_patient_seed
    seeds = set()
    for i in range(1000):
        for j in range(32):
            seeds.add(compute_patient_seed(42, i, j))
    assert len(seeds) == 1000 * 32, "Seed collision detected"


def test_seed_hash_range():
    from utils import compute_patient_seed
    for i in range(100):
        seed = compute_patient_seed(42, i, 0)
        assert 0 <= seed < 2**63


def test_attention_mask_shape():
    from utils import create_attention_mask
    n_ctx, n_pred = 4, 3
    mask = create_attention_mask(n_ctx, n_pred)
    T = n_ctx + n_pred
    assert mask.shape == (T, T)

    assert mask[:n_ctx, :n_ctx].all(), "Context-to-context should be all True"

    assert not mask[:n_ctx, n_ctx:].any(), "Context-to-prediction should be all False"

    assert mask[n_ctx:, :n_ctx].all(), "Prediction-to-context should be all True"

    # prediction zone attends itself in BOTH directions; no leak (context->pred already blocked).
    pred_block = mask[n_ctx:, n_ctx:]
    assert pred_block.all(), "Prediction-to-prediction should be all True (bidirectional)"

    print("\n[DUMP] attention_mask | visual (4 ctx + 3 pred):")
    for i in range(T):
        row = ""
        for j in range(T):
            row += "1 " if mask[i, j] else "0 "
        label = "ctx " if i < n_ctx else "pred"
        print(f"  {label} {i}: {row}")


def test_attention_mask_is_not_memoized():
    """No cheap key identifies an arbitrary masked set, so a memo on
    ``(n_context, n_prediction)`` hands one sample's mask to another with no shape error.
    Witness: two DIFFERENT masked sets at the SAME ``n_ctx`` must produce different masks.
    The bit-identity gate can't catch a returning memo — it runs the one key it was built for."""
    import utils
    from utils import create_attention_mask, create_attention_mask_from_visible

    n_ctx, T = 8, 12

    def visible(masked_patches):
        v = torch.ones(1, T, dtype=torch.bool)
        v[0, list(masked_patches)] = False
        return v

    # same n_ctx, two different masked sets of the same size
    a = create_attention_mask_from_visible(visible([2, 3, 8, 9]))
    b = create_attention_mask_from_visible(visible([4, 5, 10, 11]))
    assert a.shape == b.shape == (1, T, T)
    assert not torch.equal(a, b), (
        "two different masked sets at the same n_ctx produced the SAME mask — "
        "a memo keyed on (n_context, n_prediction) is back")

    # repeated identical calls hand back INDEPENDENT tensors: editing one must not poison the next.
    assert not hasattr(utils, '_ATTENTION_MASK_CACHE'), \
        "utils._ATTENTION_MASK_CACHE must be deleted — no memo can be correct here"
    m1 = create_attention_mask(n_ctx, T - n_ctx)
    m2 = create_attention_mask(n_ctx, T - n_ctx)
    assert torch.equal(m1, m2), "the same masked set must give the same mask"
    assert m1.data_ptr() != m2.data_ptr(), \
        "each call must return a FRESH tensor, not a shared cached one"
    m1[0, 0] = False
    assert bool(m2[0, 0]), "editing one returned mask perturbed another"
    n_diff = int((a != b).sum())
    print(f"\n[DUMP] mask memo | two masked sets at n_ctx={n_ctx} differ in "
          f"{n_diff} of {T * T} entries; no cache attribute; fresh tensor per call ✓")


# Kovatchev transform is the ONLY mg/dL<->risk bridge; f_inv clamps risk in, exp, clamps output.

def _sim_clamps() -> tuple[float, float]:
    import T1DMSIM.simulator as sim
    return float(sim.BG_CLAMP_MIN), float(sim.BG_CLAMP_MAX)


def test_kovatchev_constants_against_reference():
    """SCALE 2.2211457449985317 / POWER 1.084 / OFFSET 5.540076976170212, against an
    independent reference and against the defining endpoints. f(40) = -sqrt(10) and
    f(400) = +sqrt(10), so the risk 10*f^2 saturates at 100 at both CGM device rails."""
    from utils import kovatchev_f

    SCALE, POWER, OFFSET = 2.2211457449985317, 1.084, 5.540076976170212

    def f_ref(g: float) -> float:
        return SCALE * (math.log(g) ** POWER - OFFSET)

    for g in (70.0, 120.0, 180.0, 250.0):
        got = float(kovatchev_f(torch.tensor(g)))
        assert abs(got - f_ref(g)) < 1e-4, (
            f"kovatchev_f({g})={got} != reference {f_ref(g)} — constants drifted")
    # the defining endpoints: f = -/+ sqrt(10) at the [40, 400] rails
    s10 = math.sqrt(10.0)
    assert abs(float(kovatchev_f(torch.tensor(40.0))) + s10) < 1e-4, "f(40) != -sqrt(10)"
    assert abs(float(kovatchev_f(torch.tensor(400.0))) - s10) < 1e-4, "f(400) != +sqrt(10)"
    # the zero-risk euglycemic centre is ~128 mg/dL, the log^POWER centre of [40, 400]
    assert abs(float(kovatchev_f(torch.tensor(127.97)))) < 0.02
    print(f"\n[DUMP] kovatchev[40,400] | f(40)={f_ref(40.0):.4f} f(400)={f_ref(400.0):.4f} "
          f"f(128)={f_ref(127.97):.4f} (endpoints = -/+ sqrt10)")


def test_kovatchev_f_units_tripwire():
    """``kovatchev_f`` guards controlled callers: a z-scored value trips the hard
    ``g >= BG_CLAMP_MIN`` assert. Pool-independent, since every legal z satisfies
    z_max < BG_CLAMP_MIN - 1e-3. Re-f of an f'd value lands below that floor, trips too."""
    from utils import kovatchev_f

    bg_min, _ = _sim_clamps()
    # a legitimate mg/dL anchor passes
    _ = kovatchev_f(torch.tensor([bg_min, 100.0, 250.0]))
    # a z-space vector is what the tripwire exists to catch
    with pytest.raises(AssertionError):
        kovatchev_f(torch.tensor([-2.5, 0.3, 1.7]))
    # re-f of an already-risk value must raise too
    risk_vals = kovatchev_f(torch.tensor([100.0, 250.0]))
    with pytest.raises(AssertionError):
        kovatchev_f(risk_vals)
    print("\n[DUMP] kovatchev_f tripwire | z-space + re-f both raise ✓")


def test_kovatchev_f_inv_range_and_inverse():
    """f_inv clamps its risk input, then its mg/dL output, to the physical band,
    and inverts f on the in-band interior. Pathological risk inputs (huge +/-,
    NaN-inducing negative bases) stay finite and in-bounds."""
    from utils import kovatchev_f, kovatchev_f_inv

    bg_min, bg_max = _sim_clamps()
    grid = torch.tensor([bg_min, 40.0, 70.0, 120.0, 180.0, 300.0, bg_max])
    risk = kovatchev_f(grid)
    back = kovatchev_f_inv(risk)
    assert torch.allclose(back, grid, atol=1e-2, rtol=1e-3), (
        f"f_inv(f(g)) must recover g: {back.tolist()} vs {grid.tolist()}")

    # risk extremes that would overflow exp/log a negative base must clamp to [BG_CLAMP_MIN, MAX].
    extreme = torch.tensor([-1e3, -8.2, -3.5, 3.0, 40.0, 1e3])
    out = kovatchev_f_inv(extreme)
    assert torch.isfinite(out).all(), "f_inv must stay finite on extreme risk inputs"
    assert (out >= bg_min - 1e-3).all() and (out <= bg_max + 1e-3).all(), (
        f"f_inv output must be clamped to [{bg_min}, {bg_max}], got "
        f"[{out.min():.3f}, {out.max():.3f}]")
    print(f"\n[DUMP] kovatchev_f_inv | inverts on band, extremes -> "
          f"[{out.min():.2f}, {out.max():.2f}]")


def test_kovatchev_f_inv_nan_guard():
    """A non-finite risk input is scrubbed to the band edges BEFORE the clamp, so
    ``f_inv`` can never emit a NaN mg/dL. ``clamp`` alone lets NaN through — it compares
    false against both bounds — so the explicit ``nan_to_num`` is the load-bearing guard."""
    from utils import kovatchev_f_inv

    bg_min, bg_max = _sim_clamps()
    nasty = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0, -100.0])
    out = kovatchev_f_inv(nasty)
    assert torch.isfinite(out).all(), (
        f"f_inv must scrub non-finite risk inputs, got {out.tolist()}")
    assert (out >= bg_min - 1e-3).all() and (out <= bg_max + 1e-3).all()
    # the scrub maps +inf to r_hi and -inf / NaN to r_lo
    assert float(out[1]) == pytest.approx(bg_max, abs=1e-2), "posinf -> BG ceiling"
    assert float(out[2]) == pytest.approx(bg_min, abs=1e-2), "neginf -> BG floor"
    assert float(out[0]) == pytest.approx(bg_min, abs=1e-2), "NaN -> BG floor"
    print(f"\n[DUMP] kovatchev_f_inv NaN guard | nan/inf -> "
          f"[{float(out.min()):.2f}, {float(out.max()):.2f}], all finite ✓")


def test_kovatchev_f_target_clamps_not_tripwire():
    """``kovatchev_f_target`` clamps mg/dL into the band before f rather than asserting,
    so an out-of-band physical value is clamped, never raised on."""
    from utils import kovatchev_f, kovatchev_f_target

    bg_min, bg_max = _sim_clamps()
    # slightly out-of-band physical values: clamped, not raised
    g = torch.tensor([bg_min - 5.0, 100.0, bg_max + 50.0])
    out = kovatchev_f_target(g)
    assert torch.isfinite(out).all()
    g_in = torch.tensor([70.0, 120.0, 180.0])
    assert torch.allclose(kovatchev_f_target(g_in), kovatchev_f(g_in), atol=1e-5)
    # the clamped extremes equal f at the respective bounds
    assert abs(float(out[0]) - float(kovatchev_f(torch.tensor(bg_min)))) < 1e-4
    assert abs(float(out[2]) - float(kovatchev_f(torch.tensor(bg_max)))) < 1e-4
    print("\n[DUMP] kovatchev_f_target | clamps out-of-band, matches f in-band ✓")


def test_f_once_per_target_batch_is_mgdl():
    """A batch BG target is mg/dL, not yet f-transformed: f is applied exactly once at
    the top of the loss, and a double-f'd target sits near zero and trips this floor."""
    bg_min, bg_max = _sim_clamps()
    # stands in for data.py's target, in the mg/dL physical band
    true_bg = torch.tensor([[55.0, 70.0, 120.0, 180.0, 250.0],
                            [40.0, 90.0, 140.0, 200.0, 300.0]])
    assert (true_bg >= bg_min - 1e-3).all(), "target must be mg/dL, not f-transformed"
    assert (true_bg <= bg_max + 1e-3).all()
    # a doubly-f'd target lands wholly below the mg/dL floor
    from utils import kovatchev_f
    risk = kovatchev_f(true_bg)
    assert (risk < bg_min).all(), "f-transformed values are NOT in the mg/dL band"
    print("\n[DUMP] f-once | batch target in mg/dL band; f'd values fail the floor ✓")


def test_three_hop_round_trip():
    """normalize(f_inv(f(denorm(x)))) ≈ x over a mg/dL grid — the full
    z-space -> mg/dL -> risk -> mg/dL -> z-space cycle used by predict_rolling's
    autoregressive BG re-feed (slot 0). Every bridge is exercised exactly once."""
    import numpy as np
    from normalization import normalize, denormalize, CHANNEL_NAMES
    from utils import kovatchev_f, kovatchev_f_inv

    stats = _get_stats()
    bg_name = CHANNEL_NAMES[0]
    assert bg_name == 'bg_absolute', f"channel 0 must be bg_absolute, got {bg_name}"

    bg_min, bg_max = _sim_clamps()
    mgdl = np.linspace(bg_min + 5.0, bg_max - 5.0, 40).astype(np.float32)
    # z-space first: the representation predict_rolling carries in slot 0
    z = normalize(mgdl[:, None], stats, channel_names=[bg_name])[:, 0]

    # z -> mg/dL -> risk -> mg/dL -> z
    back_mgdl = denormalize(z[:, None], stats, channel_names=[bg_name])[:, 0]
    risk = kovatchev_f(torch.tensor(back_mgdl))
    inv_mgdl = kovatchev_f_inv(risk).numpy()
    z2 = normalize(inv_mgdl[:, None], stats, channel_names=[bg_name])[:, 0]

    max_err = float(np.abs(z2 - z).max())
    np.testing.assert_allclose(z2, z, atol=1e-3, rtol=1e-3)
    print(f"\n[DUMP] three-hop | normalize(f_inv(f(denorm(x)))) max|Δz|={max_err:.2e}")


def _get_stats():
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


# assemble_quantiles: raw (B,P,S,1+2*N_SPREADS) -> ascending risk quantiles, anchor f(last_bg).

def test_assemble_quantiles_index_for_index_and_gap():
    """assemble_quantiles emits (B,P,S,7) ascending quantiles whose index-for-
    index ordering matches QUANTILE_LEVELS, the median == q[...,3] == anchor+delta,
    and consecutive bands are separated by at least BG_QUANTILE_SPREAD_MIN
    (the strict-gap / no-σ-collapse guarantee)."""
    from utils import assemble_quantiles, kovatchev_f
    from config import (N_QUANTILES, N_SPREADS, QUANTILE_LEVELS,
                        BG_QUANTILE_SPREAD_MIN)

    B, P, S = 2, 3, 4
    torch.manual_seed(0)
    head_raw = torch.randn(B, P, S, 1 + 2 * N_SPREADS)
    last_bg = torch.tensor([90.0, 180.0])

    q_tau, median = assemble_quantiles(head_raw, last_bg)
    assert q_tau.shape == (B, P, S, N_QUANTILES), f"bad q_tau shape {q_tau.shape}"
    assert median.shape == (B, P, S), f"bad median shape {median.shape}"

    assert torch.allclose(median, q_tau[..., 3], atol=1e-6), "median must == q[...,3]"

    # index for index, not merely monotone in value: level ORDER must line up with QUANTILE_LEVELS.
    assert list(QUANTILE_LEVELS) == sorted(QUANTILE_LEVELS)
    diffs = q_tau[..., 1:] - q_tau[..., :-1]
    assert (diffs >= BG_QUANTILE_SPREAD_MIN - 1e-6).all(), (
        f"strict gap violated: min adjacent gap {float(diffs.min()):.3e} "
        f"< {BG_QUANTILE_SPREAD_MIN}")

    expected_m = kovatchev_f(last_bg).view(B, 1, 1) + head_raw[..., 0]
    assert torch.allclose(median, expected_m, atol=1e-5), (
        "median must equal anchor + delta")
    print(f"\n[DUMP] assemble_quantiles | shape {tuple(q_tau.shape)}, "
          f"min gap {float(diffs.min()):.3e} >= {BG_QUANTILE_SPREAD_MIN}, median==q3 ✓")


def test_assemble_quantiles_init_is_persistence():
    """With the head's raw output ≈ 0 (the BG_HEAD_INIT_SCALE=1e-2 init), the
    median delta ≈ 0 so the median risk ≈ f(last_bg) — i.e. the initial forecast
    is persistence (flat from last_bg) once f_inv'd."""
    from utils import assemble_quantiles, kovatchev_f, kovatchev_f_inv

    B, P, S = 2, 2, 3
    head_raw = torch.zeros(B, P, S, 7)   # 1 + 2*N_SPREADS with N_SPREADS=3
    last_bg = torch.tensor([95.0, 160.0])
    _, median = assemble_quantiles(head_raw, last_bg)
    anchor = kovatchev_f(last_bg).view(B, 1, 1).expand(B, P, S)
    assert torch.allclose(median, anchor, atol=1e-6), "zero-delta median must == anchor"
    # f_inv of the median recovers last_bg at every horizon step (persistence).
    recovered = kovatchev_f_inv(median)
    assert torch.allclose(recovered, last_bg.view(B, 1, 1).expand(B, P, S), atol=1e-1)
    print("\n[DUMP] assemble_quantiles | zero-raw -> median == f(last_bg) (persistence) ✓")


def test_assemble_quantiles_carry_spread_default_is_identity():
    """The new ``carry_spread`` param defaults to 0.0 and is bit-identical to the
    bare fan (the backward-compatibility invariant — model.py and every TRAINING
    caller keep the default and must observe no change)."""
    from utils import assemble_quantiles

    B, P, S = 2, 3, 4
    torch.manual_seed(7)
    head_raw = torch.randn(B, P, S, 7)
    last_bg = torch.tensor([85.0, 200.0])

    q_default, m_default = assemble_quantiles(head_raw, last_bg)
    q_zero, m_zero = assemble_quantiles(head_raw, last_bg, carry_spread=0.0)
    assert torch.equal(q_default, q_zero), "carry_spread=0.0 must be bit-identical to default"
    assert torch.equal(m_default, m_zero)
    print("\n[DUMP] assemble_quantiles | carry_spread default == 0.0 (bit-identical) ✓")


def test_assemble_quantiles_carry_spread_widens_band():
    """A positive ``carry_spread`` inflates the band symmetrically about the median and
    leaves the median untouched, which is what keeps a roll's fan monotone."""
    from utils import assemble_quantiles
    from config import QUANTILE_LEVELS

    B, P, S = 2, 2, 3
    torch.manual_seed(11)
    head_raw = torch.randn(B, P, S, 7)
    last_bg = torch.tensor([110.0, 150.0])
    median_idx = QUANTILE_LEVELS.index(0.5)

    carry = 0.4
    q0, m0 = assemble_quantiles(head_raw, last_bg, carry_spread=0.0)
    qc, mc = assemble_quantiles(head_raw, last_bg, carry_spread=carry)

    assert torch.allclose(m0, mc, atol=1e-6), "carry must not move the median"
    assert torch.allclose(qc[..., median_idx], q0[..., median_idx], atol=1e-6)

    # edge dist from median = hypot(carry, native): QUADRATURE not +carry (SPEC/inference.md §8.1).
    m0 = q0[..., median_idx].unsqueeze(-1)
    up_off0 = q0[..., median_idx + 1:] - m0
    dn_off0 = m0 - q0[..., :median_idx]
    want_up = m0 + torch.hypot(torch.full_like(up_off0, carry), up_off0)
    want_dn = m0 - torch.hypot(torch.full_like(dn_off0, carry), dn_off0)
    assert torch.allclose(qc[..., median_idx + 1:], want_up, atol=1e-6)
    assert torch.allclose(qc[..., :median_idx], want_dn, atol=1e-6)

    # strictly wider everywhere off median, by LESS than the 2*carry an additive carry would add.
    width0 = q0[..., -1] - q0[..., 0]
    widthc = qc[..., -1] - qc[..., 0]
    assert (widthc > width0).all(), "carry must strictly widen the 5-95 band"
    assert (widthc - width0 < 2 * carry).all(), (
        "quadrature must widen by less than the perfectly-correlated 2*carry")
    print(f"\n[DUMP] assemble_quantiles | carry={carry} widens 5-95 band in quadrature, median fixed ✓")


def test_assemble_quantiles_carry_spread_is_per_level():
    """ONE offset PER LEVEL, laid out like the head's spread columns,
    ``[.75 .9 .95 | .25 .1 .05]``; a scalar is that scalar in all six slots. One carry
    shared by the levels is the roll-seam defect: it seeds .75 from .95's accumulation."""
    from utils import assemble_quantiles
    from config import QUANTILE_LEVELS, N_SPREADS

    B, P, S = 2, 2, 3
    torch.manual_seed(13)
    head_raw = torch.randn(B, P, S, 7)
    last_bg = torch.tensor([95.0, 180.0])
    median_idx = QUANTILE_LEVELS.index(0.5)

    carry = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])   # up .75/.9/.95 | dn .25/.1/.05
    q0, m0 = assemble_quantiles(head_raw, last_bg, carry_spread=0.0)
    qc, mc = assemble_quantiles(head_raw, last_bg, carry_spread=carry)

    assert torch.allclose(m0, mc, atol=1e-6), "carry must not move the median"
    assert torch.allclose(qc[..., median_idx], q0[..., median_idx], atol=1e-6)

    # upper edges take carry[:3]; lower take carry[3:] flipped ascending, each in quadrature.
    m0 = q0[..., median_idx].unsqueeze(-1)
    up_off0 = q0[..., median_idx + 1:] - m0
    dn_off0 = m0 - q0[..., :median_idx]
    want_up = m0 + torch.hypot(carry[:N_SPREADS].expand_as(up_off0), up_off0)
    want_dn = m0 - torch.hypot(carry[N_SPREADS:].flip(0).expand_as(dn_off0), dn_off0)
    assert torch.allclose(qc[..., median_idx + 1:], want_up, atol=1e-6), (
        "upper edges must compose with their OWN carry")
    assert torch.allclose(qc[..., :median_idx], want_dn, atol=1e-6), (
        "lower edges must compose with their OWN carry")

    # a scalar is the six-slot vector of that scalar
    q_scalar, _ = assemble_quantiles(head_raw, last_bg, carry_spread=0.37)
    q_vector, _ = assemble_quantiles(
        head_raw, last_bg, carry_spread=torch.full((2 * N_SPREADS,), 0.37))
    assert torch.allclose(q_scalar, q_vector, atol=1e-6), (
        "a scalar carry must equal the same value in every level slot")
    print("\n[DUMP] assemble_quantiles | per-level carry [.75 .9 .95 | .25 .1 .05] ✓")


def test_reshape_consistency_with_to_patch_major():
    """``risk_loss._to_patch_major`` is the plain C-contiguous reshape, flat = p*S+s, so
    DILATE's median term runs monotonically along time; a P/S transpose scrambles it with
    no shape error."""
    from risk_loss import _to_patch_major
    from config import PREDICTION_PATCHES, PATCH_SIZE

    B, P, S = 5, PREDICTION_PATCHES, PATCH_SIZE
    torch.manual_seed(2)
    delta = torch.randn(B, P, S)
    assert torch.equal(delta.reshape(B, P * S), _to_patch_major(delta)), (
        "assemble_quantiles reshape must match _to_patch_major (patch-major C-contiguous)")
    print("\n[DUMP] R3 reshape | delta.reshape(B,P*S) == _to_patch_major(delta) ✓")


# ModelEMA finiteness guard: one NaN in live weights mustn't poison shadow (decay*NaN+.. stays NaN).

class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(3, 2)


def test_model_ema_skips_non_finite_update():
    """ModelEMA.update skips a tensor whose incoming live weights are non-finite,
    so a single transient NaN cannot fold into the shadow and persist forever
    (decay·NaN + (1-decay)·x == NaN on every subsequent step)."""
    from utils import ModelEMA

    torch.manual_seed(0)
    model = _TinyModel()
    ema = ModelEMA(model, decay=0.9)
    shadow_before = {k: v.clone() for k, v in ema.shadow.items()}

    # Poison the live weights with a NaN, then update: the shadow must NOT absorb it.
    with torch.no_grad():
        model.lin.weight[0, 0] = float('nan')
    ema.update(model)
    for k, v in ema.shadow.items():
        assert torch.isfinite(v).all(), f"shadow tensor {k} absorbed a non-finite value"
    # The poisoned tensor's shadow is unchanged (blend skipped entirely).
    assert torch.equal(ema.shadow['lin.weight'], shadow_before['lin.weight']), (
        "the NaN-carrying tensor's shadow must be left at its last good value")

    # A subsequent CLEAN update resumes blending normally (the shadow is still alive).
    with torch.no_grad():
        model.lin.weight[0, 0] = 5.0
    ema.update(model)
    assert torch.isfinite(ema.shadow['lin.weight']).all()
    assert not torch.equal(ema.shadow['lin.weight'], shadow_before['lin.weight']), (
        "a clean update after the skip must move the shadow again")
    print("\n[DUMP] ModelEMA | NaN update skipped, shadow stays finite, clean update resumes ✓")
