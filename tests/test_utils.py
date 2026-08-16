"""Tests for utils.py — seed hashing, attention mask, the Kovatchev risk
transform (the sole mg/dL<->risk bridge), quantile assembly, and the
three-hop normalize/f/f_inv round-trip."""

import math

import pytest
import torch


def test_seed_hash_deterministic():
    """Same (master_seed, step, position) always produces the same patient seed."""
    from utils import compute_patient_seed
    seed1 = compute_patient_seed(42, 100, 5)
    seed2 = compute_patient_seed(42, 100, 5)
    assert seed1 == seed2


def test_seed_hash_unique():
    """Different (step, position) pairs produce different seeds (no collisions in 32K pairs)."""
    from utils import compute_patient_seed
    seeds = set()
    for i in range(1000):
        for j in range(32):
            seeds.add(compute_patient_seed(42, i, j))
    assert len(seeds) == 1000 * 32, "Seed collision detected"


def test_seed_hash_range():
    """Patient seeds are in valid range [0, 2^63 - 1]."""
    from utils import compute_patient_seed
    for i in range(100):
        seed = compute_patient_seed(42, i, 0)
        assert 0 <= seed < 2**63


def test_attention_mask_shape():
    """Attention mask has correct shape and four-region pattern."""
    from utils import create_attention_mask
    n_ctx, n_pred = 4, 3
    mask = create_attention_mask(n_ctx, n_pred)
    T = n_ctx + n_pred
    assert mask.shape == (T, T)

    # Context-to-context: all True (bidirectional)
    assert mask[:n_ctx, :n_ctx].all(), "Context-to-context should be all True"

    # Context-to-prediction: all False (context doesn't see future)
    assert not mask[:n_ctx, n_ctx:].any(), "Context-to-prediction should be all False"

    # Prediction-to-context: all True
    assert mask[n_ctx:, :n_ctx].all(), "Prediction-to-context should be all True"

    # Prediction-to-prediction: all True (BIDIRECTIONAL within the horizon).
    # The prediction zone attends to itself in both directions — no future leak
    # arises because context-to-prediction is already blocked above.
    pred_block = mask[n_ctx:, n_ctx:]
    assert pred_block.all(), "Prediction-to-prediction should be all True (bidirectional)"

    # DUMP: print the mask as a visual grid
    print("\n[DUMP] attention_mask | visual (4 ctx + 3 pred):")
    for i in range(T):
        row = ""
        for j in range(T):
            row += "1 " if mask[i, j] else "0 "
        label = "ctx " if i < n_ctx else "pred"
        print(f"  {label} {i}: {row}")


def test_attention_mask_is_not_memoized():
    """THE MEMO IS GONE.  The old cache keyed on ``(n_context, n_prediction)`` and
    handed the identical tensor to every caller at that key — so a mask built for
    one masked set was returned for another, with no shape error and nothing to
    notice.  The masked set is now arbitrary and no cheap key identifies it.

    Two witnesses, and the first is the one that matters: two DIFFERENT masked
    sets at the SAME ``n_ctx`` must produce DIFFERENT masks.  The bit-identity
    gate cannot catch a returning memo — it runs the one key the memo was built
    for."""
    import utils
    from utils import create_attention_mask, create_attention_mask_from_visible

    n_ctx, T = 8, 12

    def visible(masked_patches):
        v = torch.ones(1, T, dtype=torch.bool)
        v[0, list(masked_patches)] = False
        return v

    # Same n_ctx (8 visible patches), two different masked sets of the same size.
    a = create_attention_mask_from_visible(visible([2, 3, 8, 9]))
    b = create_attention_mask_from_visible(visible([4, 5, 10, 11]))
    assert a.shape == b.shape == (1, T, T)
    assert not torch.equal(a, b), (
        "two different masked sets at the same n_ctx produced the SAME mask — "
        "a memo keyed on (n_context, n_prediction) is back")

    # No module-level cache, and repeated identical calls hand back independent
    # tensors: a caller that edits one must not poison the next.
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


# ---------------------------------------------------------------------------
# Kovatchev risk transform — the ONLY (b)mg/dL <-> (c)risk-space bridge.
# kovatchev_f imports BG_CLAMP_MIN/MAX from the simulator (the units tripwire);
# kovatchev_f_inv clamps the risk input, exp's, then clamps the mg/dL output.
# ---------------------------------------------------------------------------

def _sim_clamps() -> tuple[float, float]:
    import T1DMSIM.simulator as sim
    return float(sim.BG_CLAMP_MIN), float(sim.BG_CLAMP_MAX)


def test_kovatchev_constants_against_reference():
    """The [40,400]-anchored Kovatchev constants (SCALE 2.2211457449985317 /
    POWER 1.084 / OFFSET 5.540076976170212) must back the implemented f —
    verified by (a) an independent reference evaluation at three BG levels and
    (b) the DEFINING endpoint property f(40) = -sqrt(10), f(400) = +sqrt(10), so
    the risk 10*f^2 saturates at 100 exactly at both CGM device rails."""
    from utils import kovatchev_f

    SCALE, POWER, OFFSET = 2.2211457449985317, 1.084, 5.540076976170212

    def f_ref(g: float) -> float:
        return SCALE * (math.log(g) ** POWER - OFFSET)

    for g in (70.0, 120.0, 180.0, 250.0):
        got = float(kovatchev_f(torch.tensor(g)))
        assert abs(got - f_ref(g)) < 1e-4, (
            f"kovatchev_f({g})={got} != reference {f_ref(g)} — constants drifted")
    # Defining endpoints: re-anchored so f = -/+ sqrt(10) at the [40, 400] rails.
    s10 = math.sqrt(10.0)
    assert abs(float(kovatchev_f(torch.tensor(40.0))) + s10) < 1e-4, "f(40) != -sqrt(10)"
    assert abs(float(kovatchev_f(torch.tensor(400.0))) - s10) < 1e-4, "f(400) != +sqrt(10)"
    # The zero-risk euglycemic center sits at ~128 mg/dL (the log^POWER center of
    # [40, 400]); f is near zero there.
    assert abs(float(kovatchev_f(torch.tensor(127.97)))) < 0.02
    print(f"\n[DUMP] kovatchev[40,400] | f(40)={f_ref(40.0):.4f} f(400)={f_ref(400.0):.4f} "
          f"f(128)={f_ref(127.97):.4f} (endpoints = -/+ sqrt10)")


def test_kovatchev_f_units_tripwire():
    """kovatchev_f is the CONTROLLED-caller guard: a z-scored value trips the hard
    ``g >= BG_CLAMP_MIN`` assert loudly (the units tripwire).  The guarantee is
    pool-independent — every legal z satisfies z_max < BG_CLAMP_MIN - 1e-3.
    Re-f of an f'd value (output in [f(BG_CLAMP_MIN), f(BG_CLAMP_MAX)], which sits
    wholly below that floor) must likewise trip."""
    from utils import kovatchev_f

    bg_min, _ = _sim_clamps()
    # A legitimate mg/dL anchor passes.
    _ = kovatchev_f(torch.tensor([bg_min, 100.0, 250.0]))
    # A z-space vector (what the tripwire exists to catch) must raise.
    with pytest.raises(AssertionError):
        kovatchev_f(torch.tensor([-2.5, 0.3, 1.7]))
    # Re-f of an already-risk value (in [f(BG_CLAMP_MIN), f(BG_CLAMP_MAX)]) must
    # also raise.
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

    # Worst-case risk extremes that would otherwise overflow exp or take the log
    # of a negative base — must clamp into [BG_CLAMP_MIN, BG_CLAMP_MAX], finite.
    extreme = torch.tensor([-1e3, -8.2, -3.5, 3.0, 40.0, 1e3])
    out = kovatchev_f_inv(extreme)
    assert torch.isfinite(out).all(), "f_inv must stay finite on extreme risk inputs"
    assert (out >= bg_min - 1e-3).all() and (out <= bg_max + 1e-3).all(), (
        f"f_inv output must be clamped to [{bg_min}, {bg_max}], got "
        f"[{out.min():.3f}, {out.max():.3f}]")
    print(f"\n[DUMP] kovatchev_f_inv | inverts on band, extremes -> "
          f"[{out.min():.2f}, {out.max():.2f}]")


def test_kovatchev_f_inv_nan_guard():
    """f_inv scrubs a non-finite risk input (NaN / ±inf) to the band edges
    BEFORE clamp, so it can never silently emit a NaN mg/dL.  ``clamp`` alone
    would let NaN flow straight through (NaN compares false against both bounds),
    so the explicit ``nan_to_num`` is the load-bearing guard."""
    from utils import kovatchev_f_inv

    bg_min, bg_max = _sim_clamps()
    nasty = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0, -100.0])
    out = kovatchev_f_inv(nasty)
    assert torch.isfinite(out).all(), (
        f"f_inv must scrub non-finite risk inputs, got {out.tolist()}")
    assert (out >= bg_min - 1e-3).all() and (out <= bg_max + 1e-3).all()
    # +inf risk -> ceiling, -inf / NaN -> floor (the scrub maps them to r_hi/r_lo).
    assert float(out[1]) == pytest.approx(bg_max, abs=1e-2), "posinf -> BG ceiling"
    assert float(out[2]) == pytest.approx(bg_min, abs=1e-2), "neginf -> BG floor"
    assert float(out[0]) == pytest.approx(bg_min, abs=1e-2), "NaN -> BG floor"
    print(f"\n[DUMP] kovatchev_f_inv NaN guard | nan/inf -> "
          f"[{float(out.min()):.2f}, {float(out.max()):.2f}], all finite ✓")


def test_kovatchev_f_target_clamps_not_tripwire():
    """The target-path f (kovatchev_f_target) PHYSICALLY clamps mg/dL into the
    band before f rather than asserting — it must NOT raise on a sub-floor or
    super-ceiling physical value (a rare smoother backstop), only clamp it."""
    from utils import kovatchev_f, kovatchev_f_target

    bg_min, bg_max = _sim_clamps()
    # Slightly out-of-band physical values: clamped, not raised.
    g = torch.tensor([bg_min - 5.0, 100.0, bg_max + 50.0])
    out = kovatchev_f_target(g)
    assert torch.isfinite(out).all()
    # In-band values match the plain f exactly.
    g_in = torch.tensor([70.0, 120.0, 180.0])
    assert torch.allclose(kovatchev_f_target(g_in), kovatchev_f(g_in), atol=1e-5)
    # The clamped extremes equal f at the respective bounds.
    assert abs(float(out[0]) - float(kovatchev_f(torch.tensor(bg_min)))) < 1e-4
    assert abs(float(out[2]) - float(kovatchev_f(torch.tensor(bg_max)))) < 1e-4
    print("\n[DUMP] kovatchev_f_target | clamps out-of-band, matches f in-band ✓")


def test_f_once_per_target_batch_is_mgdl():
    """A realistic batch BG target tensor lives in mg/dL (>= BG_CLAMP_MIN), i.e.
    it is NOT yet f-transformed — f is applied exactly once at the top of the
    loss, never baked into the batch. (If a target had been double-f'd it would
    sit near zero and trip this floor.)"""
    bg_min, bg_max = _sim_clamps()
    # Stand-in for data.py's smoothed pred-zone target (mg/dL physical band).
    true_bg = torch.tensor([[55.0, 70.0, 120.0, 180.0, 250.0],
                            [40.0, 90.0, 140.0, 200.0, 300.0]])
    assert (true_bg >= bg_min - 1e-3).all(), "target must be mg/dL, not f-transformed"
    assert (true_bg <= bg_max + 1e-3).all()
    # A doubly-f'd target lands in [f(BG_CLAMP_MIN), f(BG_CLAMP_MAX)], wholly below
    # the mg/dL floor, and fails it — assert the trap.
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
    # Hop into z-space first (the representation predict_rolling carries in slot 0).
    z = normalize(mgdl[:, None], stats, channel_names=[bg_name])[:, 0]

    # z -> mg/dL -> risk -> mg/dL -> z.
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


# ---------------------------------------------------------------------------
# assemble_quantiles — turns the BG head's raw (B,P,S, 1 + 2*N_SPREADS) output
# into ascending risk-space quantiles anchored at f(last_bg).
# ---------------------------------------------------------------------------

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

    # Median is the index-3 quantile.
    assert torch.allclose(median, q_tau[..., 3], atol=1e-6), "median must == q[...,3]"

    # Index-for-index ascending (NOT merely monotone in value — the level ORDER
    # must line up with QUANTILE_LEVELS).
    assert list(QUANTILE_LEVELS) == sorted(QUANTILE_LEVELS)
    diffs = q_tau[..., 1:] - q_tau[..., :-1]
    assert (diffs >= BG_QUANTILE_SPREAD_MIN - 1e-6).all(), (
        f"strict gap violated: min adjacent gap {float(diffs.min()):.3e} "
        f"< {BG_QUANTILE_SPREAD_MIN}")

    # Median assembly depends on the BG_HEAD_MEDIAN_MODE gate.  Under 'global' the
    # subspace dimension is per-SPAN — ``global_median_dim(L)``, not the configured
    # G, which is only the value at L == PREDICTION_PATCHES.  The legacy (B,)
    # anchor form makes all P slots one span, so L == P here.
    from config import (BG_HEAD_MEDIAN_MODE, BG_HEAD_STEP_BASIS_TYPE)
    from utils import get_global_median_basis, global_median_dim
    anchor = kovatchev_f(last_bg).view(B, 1, 1)
    delta = head_raw[..., 0]
    if BG_HEAD_MEDIAN_MODE == 'global':
        n = P * S
        g = min(global_median_dim(P), n)
        Bg = get_global_median_basis(n, g, BG_HEAD_STEP_BASIS_TYPE,
                                     device=delta.device, dtype=delta.dtype)
        dflat = delta.reshape(B, n)
        expected_m = anchor + ((dflat @ Bg) @ Bg.T).reshape(B, P, S)
    elif BG_HEAD_MEDIAN_MODE == 'cumulative':
        d_rel = delta - delta[..., :1]
        rise = delta[..., -1] - delta[..., 0]
        o = torch.cumsum(rise, dim=1) - rise
        expected_m = anchor + o.unsqueeze(-1) + d_rel
    else:
        expected_m = anchor + delta
    assert torch.allclose(median, expected_m, atol=1e-5), (
        "median must equal the active (global/cumulative/flat) assembly")
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
    """A positive ``carry_spread`` inflates the risk-space band SYMMETRICALLY about
    the median (τ>.5 edges shift up by carry, τ<.5 down by carry) while leaving the
    median untouched — the exact effect predict_rolling relies on to keep the fan
    monotone across roll boundaries."""
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

    # Median is untouched.
    assert torch.allclose(m0, mc, atol=1e-6), "carry must not move the median"
    assert torch.allclose(qc[..., median_idx], q0[..., median_idx], atol=1e-6)

    # Upper edges shifted up by exactly +carry, lower edges down by exactly -carry.
    up_delta = qc[..., median_idx + 1:] - q0[..., median_idx + 1:]
    dn_delta = qc[..., :median_idx] - q0[..., :median_idx]
    assert torch.allclose(up_delta, torch.full_like(up_delta, carry), atol=1e-6)
    assert torch.allclose(dn_delta, torch.full_like(dn_delta, -carry), atol=1e-6)

    # Net: the band is strictly wider everywhere off the median.
    width0 = q0[..., -1] - q0[..., 0]
    widthc = qc[..., -1] - qc[..., 0]
    assert (widthc > width0).all(), "carry must strictly widen the 5-95 band"
    assert torch.allclose(widthc - width0, torch.full_like(width0, 2 * carry), atol=1e-6)
    print(f"\n[DUMP] assemble_quantiles | carry={carry} widens 5-95 band by 2*carry, median fixed ✓")


# ---------------------------------------------------------------------------
# R1 — cumulative cross-patch-continuity median head (BG_HEAD_MEDIAN_MODE='cumulative').
# ---------------------------------------------------------------------------

def _force_mode(monkeypatch, mode: str):
    """Set BG_HEAD_MEDIAN_MODE in config (assemble_quantiles re-imports it from config
    at call time, so patching config alone suffices)."""
    import config
    monkeypatch.setattr(config, "BG_HEAD_MEDIAN_MODE", mode, raising=False)


def _force_cumulative(monkeypatch, value: bool):
    """Back-compat shim for the R1 tests: True → 'cumulative', False → 'independent'
    (the two legacy bit-identical paths the old boolean selected)."""
    _force_mode(monkeypatch, 'cumulative' if value else 'independent')


def test_assemble_quantiles_cumulative_c0_continuity(monkeypatch):
    """R1 ON: for every interior patch seam, the median end of patch p equals the
    median start of patch p+1 exactly (C0 continuity), for random nonzero head_raw."""
    from utils import assemble_quantiles
    _force_cumulative(monkeypatch, True)

    B, P, S = 3, 4, 6
    torch.manual_seed(123)
    head_raw = torch.randn(B, P, S, 7) * 1.7      # nonzero, sizeable deltas
    last_bg = torch.tensor([88.0, 140.0, 205.0])
    _, median = assemble_quantiles(head_raw, last_bg)

    end = median[:, :-1, S - 1]        # (B, P-1) end of each non-final patch
    start = median[:, 1:, 0]           # (B, P-1) start of each subsequent patch
    gap = (end - start).abs()
    assert torch.allclose(end, start, atol=1e-6), (
        f"C0 seam continuity violated: max |gap| = {float(gap.max()):.3e}")
    print(f"\n[DUMP] R1 C0 continuity | max seam |Δ| = {float(gap.max()):.3e} ✓")


def test_assemble_quantiles_cumulative_first_step_pins_anchor(monkeypatch):
    """R1 ON: median[...,0,0] == f(clamp(last_bg)) EXACTLY for ANY head_raw — the
    deliberate first-step pinning that removes the context->forecast jump."""
    from utils import assemble_quantiles, kovatchev_f
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    _force_cumulative(monkeypatch, True)

    B, P, S = 2, 4, 6
    torch.manual_seed(99)
    head_raw = torch.randn(B, P, S, 7) * 2.5
    last_bg = torch.tensor([72.0, 190.0])
    _, median = assemble_quantiles(head_raw, last_bg)

    anchor = kovatchev_f(last_bg.clamp(BG_CLAMP_MIN, BG_CLAMP_MAX))   # (B,)
    assert torch.allclose(median[:, 0, 0], anchor, atol=1e-6), (
        "first forecast step must pin EXACTLY to the anchor")
    print(f"\n[DUMP] R1 first-step pinning | median[:,0,0]-anchor max "
          f"{float((median[:,0,0]-anchor).abs().max()):.3e} ✓")


def test_assemble_quantiles_cumulative_off_is_bit_identical(monkeypatch):
    """R1 OFF: q_tau and median are torch.equal to the literal OLD m=anchor+delta
    assembly (the bit-identical-when-off regression invariant)."""
    from utils import assemble_quantiles, kovatchev_f
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    _force_cumulative(monkeypatch, False)

    B, P, S = 2, 4, 6
    torch.manual_seed(7)
    head_raw = torch.randn(B, P, S, 7)
    last_bg = torch.tensor([95.0, 210.0])
    q_tau, median = assemble_quantiles(head_raw, last_bg)

    # Reference: literal old formula m = anchor + delta.
    anchor = kovatchev_f(last_bg.clamp(BG_CLAMP_MIN, BG_CLAMP_MAX)).view(B, 1, 1)
    m_ref = anchor + head_raw[..., 0]
    assert torch.equal(median, m_ref), "OFF median must be bit-identical to anchor+delta"
    assert torch.equal(q_tau[..., 3], m_ref), "OFF q_tau[...,3] must == reference median"
    print("\n[DUMP] R1 OFF | median torch.equal to old anchor+delta ✓")


def test_assemble_quantiles_cumulative_p1_edge(monkeypatch):
    """R1 ON, P==1: no seams; no crash; median[...,0,0]==anchor exactly; and the
    single-patch curve == anchor + (delta - delta[...,0:1])."""
    from utils import assemble_quantiles, kovatchev_f
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    _force_cumulative(monkeypatch, True)

    B, P, S = 2, 1, 6
    torch.manual_seed(5)
    head_raw = torch.randn(B, P, S, 7) * 1.3
    last_bg = torch.tensor([100.0, 175.0])
    _, median = assemble_quantiles(head_raw, last_bg)

    anchor = kovatchev_f(last_bg.clamp(BG_CLAMP_MIN, BG_CLAMP_MAX)).view(B, 1, 1)
    delta = head_raw[..., 0]
    expected = anchor + (delta - delta[..., :1])
    assert torch.allclose(median, expected, atol=1e-6), "P==1 cumulative semantics"
    assert torch.allclose(median[:, 0, 0], anchor.view(B), atol=1e-6)
    print("\n[DUMP] R1 P==1 edge | anchor-pinned, no seams ✓")


def test_assemble_quantiles_cumulative_median_equals_q_tau3_and_ascending(monkeypatch):
    """R1 ON: median == q_tau[...,3] and q_tau strictly ascending along the τ axis."""
    from utils import assemble_quantiles
    _force_cumulative(monkeypatch, True)

    B, P, S = 2, 4, 6
    torch.manual_seed(31)
    head_raw = torch.randn(B, P, S, 7) * 2.0
    last_bg = torch.tensor([80.0, 220.0])
    q_tau, median = assemble_quantiles(head_raw, last_bg)

    assert torch.allclose(median, q_tau[..., 3], atol=1e-6), "median must == q_tau[...,3]"
    diffs = q_tau[..., 1:] - q_tau[..., :-1]
    assert (diffs > 0).all(), f"fan not strictly ascending: min {float(diffs.min()):.3e}"
    print("\n[DUMP] R1 ON | median==q3, fan strictly ascending ✓")


def test_assemble_quantiles_cumulative_init_is_persistence(monkeypatch):
    """R1 ON at init (zero head): median == anchor everywhere (persistence). Also
    holds OFF (the property is flag-agnostic for the zero head)."""
    from utils import assemble_quantiles, kovatchev_f, kovatchev_f_inv

    B, P, S = 2, 4, 6
    head_raw = torch.zeros(B, P, S, 7)
    last_bg = torch.tensor([95.0, 160.0])
    for flag in (True, False):
        _force_cumulative(monkeypatch, flag)
        _, median = assemble_quantiles(head_raw, last_bg)
        anchor = kovatchev_f(last_bg).view(B, 1, 1).expand(B, P, S)
        assert torch.allclose(median, anchor, atol=1e-6), (
            f"zero-head median must == anchor (flag={flag})")
        recovered = kovatchev_f_inv(median)
        assert torch.allclose(
            recovered, last_bg.view(B, 1, 1).expand(B, P, S), atol=1e-1)
    print("\n[DUMP] R1 init persistence | zero head -> median==anchor (both flags) ✓")


# ---------------------------------------------------------------------------
# R3 — GLOBAL smooth-basis median (BG_HEAD_MEDIAN_MODE='global', the default).
# Projects the per-patch median delta onto a fixed low-frequency DCT-II subspace
# over the FULL P*S horizon, killing R1's unconstrained-integrator drift.
# ---------------------------------------------------------------------------

def _force_global(monkeypatch):
    _force_mode(monkeypatch, 'global')


def _r1_cumulative_op(delta, anchor):
    """The R1 cumulative per-patch offset o[p] = m[:,p,0]-anchor, in risk space —
    the documented [0, 0.24, 0.43, 0.63]-style LINEAR ramp the projection must NOT
    reproduce.  o[p] = exclusive cumsum over patches of the within-patch rise."""
    rise = delta[..., -1] - delta[..., 0]               # (B,P)
    o = torch.cumsum(rise, dim=1) - rise                # (B,P) exclusive
    return o                                             # m[:,p,0]-anchor == o[:,p] (d_rel[...,0]==0)


def test_assemble_quantiles_global_non_accumulation(monkeypatch):
    """R3 KEY REGRESSION (reproduces scratch/overshoot_diag.py's o[p] measurement):
    under 'global', the per-patch offset o[p]=median[:,p,0]-anchor does NOT drift like
    R1's amplifying [0,0.24,0.43,0.63] ramp. Two witnesses on RANDOM coeffs:

    (A) for zero-mean random deltas, mean |o[p]| over the batch is NON-MONOTONE in p
        (a projection of unbiased noise has no preferred direction — it cannot ramp);
    (B) on the SAME random deltas, R1's cumulative integrator AMPLIFIES the offset in
        the far patches: its mean |o[p]| at the last patch strictly exceeds global's
        (the cumsum accumulates while the projection contracts). This is the structural
        contrast scratch/overshoot_diag.py measured on the trained R1 checkpoint."""
    from utils import assemble_quantiles, kovatchev_f
    _force_global(monkeypatch)

    B, P, S = 256, 4, 6
    for seed in (0, 1, 7, 123):
        torch.manual_seed(seed)
        head_raw = torch.randn(B, P, S, 7) * 1.2          # zero-mean random deltas
        last_bg = torch.full((B,), 120.0)
        _, median = assemble_quantiles(head_raw, last_bg)
        anchor = kovatchev_f(last_bg).view(B, 1, 1)
        op = (median[:, :, 0] - anchor.squeeze(-1))       # (B,P) o[p], global
        mean_abs_op = op.abs().mean(0)                     # (P,)

        # R1 cumulative o[p] on the SAME deltas.
        r1_op = _r1_cumulative_op(head_raw[..., 0], anchor).abs().mean(0)  # (P,)

        # (A) global |o[p]| is NON-monotone (no drift direction for unbiased noise).
        diffs = mean_abs_op[1:] - mean_abs_op[:-1]
        assert not bool((diffs > 0).all()), (
            f"seed {seed}: global |o[p]| monotone-increasing ({mean_abs_op.tolist()}) — drift!")
        # (A') bounded: every |o[p]| stays within the delta scale (L2 contraction).
        assert float(mean_abs_op.max()) <= 3.0 * float(head_raw[..., 0].abs().mean()), (
            f"seed {seed}: global o[p] unbounded vs delta scale ({mean_abs_op.tolist()})")
        # (B) R1 AMPLIFIES at the far horizon vs global (the integrator drift).
        assert float(r1_op[-1]) > float(mean_abs_op[-1]), (
            f"seed {seed}: R1 far-patch |o| ({float(r1_op[-1]):.3f}) should exceed "
            f"global ({float(mean_abs_op[-1]):.3f}) — drift contrast")
        # R1 also ramps monotonically here (its documented defect).
        if seed == 0:
            print(f"\n[DUMP] R3 non-accumulation | global |o[p]| = "
                  f"{[round(float(x),4) for x in mean_abs_op]} (bounded, non-monotone)  "
                  f"vs R1 cumulative |o[p]| = {[round(float(x),4) for x in r1_op]} "
                  f"(R1 amplifies at the far horizon) ✓")


def test_assemble_quantiles_global_is_projection_contraction(monkeypatch):
    """The structural guarantee: ||delta_global|| <= ||delta_flat|| (an orthogonal
    projection is an L2 contraction) — so the median cannot amplify or drift."""
    from utils import assemble_quantiles, kovatchev_f, get_global_median_basis
    from config import BG_HEAD_MEDIAN_GLOBAL_DIM, BG_HEAD_STEP_BASIS_TYPE
    _force_global(monkeypatch)

    B, P, S = 16, 4, 6
    torch.manual_seed(3)
    head_raw = torch.randn(B, P, S, 7) * 2.0
    last_bg = torch.full((B,), 130.0)
    _, median = assemble_quantiles(head_raw, last_bg)
    anchor = kovatchev_f(last_bg).view(B, 1, 1)
    delta_global = (median - anchor).reshape(B, P * S)
    delta_flat = head_raw[..., 0].reshape(B, P * S)
    nf = delta_flat.norm(dim=1)
    ng = delta_global.norm(dim=1)
    assert (ng <= nf + 1e-5).all(), (
        f"projection must contract: max ||proj||-||x|| = {float((ng-nf).max()):.3e}")
    print(f"\n[DUMP] R3 contraction | mean ||proj||/||x|| = {float((ng/nf.clamp_min(1e-9)).mean()):.3f} <= 1 ✓")


def test_assemble_quantiles_global_smoothness(monkeypatch):
    """R3 SMOOTHNESS: the low-pass leaves no high-frequency seam sawtooth — the
    risk-space |Δ²m| at the patch seams {S,2S,3S} is NOT larger than the interior
    |Δ²m| (the seam/interior ratio ~1), unlike 'independent' where seams spike."""
    from utils import assemble_quantiles, kovatchev_f
    _force_global(monkeypatch)

    B, P, S = 32, 4, 6
    torch.manual_seed(9)
    head_raw = torch.randn(B, P, S, 7) * 1.5
    last_bg = torch.full((B,), 110.0)
    _, median = assemble_quantiles(head_raw, last_bg)
    m = median.reshape(B, P * S)                          # patch-major flat
    d2 = (m[:, 2:] - 2.0 * m[:, 1:-1] + m[:, :-2]).abs()  # (B, P*S-2), index i -> flat i+1
    seam_flat = [k * S for k in range(1, P)]              # {6,12,18}
    seam_idx = [i - 1 for i in seam_flat]                 # into d2
    interior_idx = [i for i in range(d2.shape[1]) if i not in seam_idx]
    seam_d2 = d2[:, seam_idx].mean()
    interior_d2 = d2[:, interior_idx].mean()
    ratio = float(seam_d2 / interior_d2.clamp_min(1e-9))
    assert ratio < 2.0, (
        f"seam |Δ²| spikes vs interior (ratio {ratio:.2f}) — sawtooth not removed")

    # Contrast: 'independent' lets the seams spike (free per-patch curves).
    _force_mode(monkeypatch, 'independent')
    _, med_ind = assemble_quantiles(head_raw, last_bg)
    mi = med_ind.reshape(B, P * S)
    d2i = (mi[:, 2:] - 2.0 * mi[:, 1:-1] + mi[:, :-2]).abs()
    ratio_ind = float(d2i[:, seam_idx].mean() / d2i[:, interior_idx].mean().clamp_min(1e-9))
    print(f"\n[DUMP] R3 smoothness | global seam/interior |Δ²| = {ratio:.2f} "
          f"(< independent {ratio_ind:.2f}) ✓")


def test_assemble_quantiles_global_init_persistence(monkeypatch):
    """R3 persistence-at-init: head_raw≈0 (BG_HEAD_INIT_SCALE scale) ⇒ z≈0 ⇒
    delta_global≈0 ⇒ median≈anchor everywhere."""
    from utils import assemble_quantiles, kovatchev_f
    _force_global(monkeypatch)

    B, P, S = 2, 4, 6
    last_bg = torch.tensor([95.0, 160.0])
    anchor = kovatchev_f(last_bg).view(B, 1, 1).expand(B, P, S)
    for scale in (0.0, 1e-2):
        torch.manual_seed(1)
        head_raw = torch.randn(B, P, S, 7) * scale
        _, median = assemble_quantiles(head_raw, last_bg)
        tol = 1e-6 if scale == 0.0 else 5e-2
        assert torch.allclose(median, anchor, atol=tol), (
            f"global median must ≈ anchor at init scale {scale}")
    print("\n[DUMP] R3 init persistence | global zero/tiny head -> median≈anchor ✓")


def test_assemble_quantiles_global_median_eq_qtau3(monkeypatch):
    """R3: median == q_tau[...,3] exactly under 'global'."""
    from utils import assemble_quantiles
    _force_global(monkeypatch)

    B, P, S = 3, 4, 6
    torch.manual_seed(17)
    head_raw = torch.randn(B, P, S, 7) * 2.0
    last_bg = torch.tensor([80.0, 150.0, 220.0])
    q_tau, median = assemble_quantiles(head_raw, last_bg)
    assert torch.equal(median, q_tau[..., 3]), "global median must == q_tau[...,3]"
    print("\n[DUMP] R3 | global median == q_tau[...,3] ✓")


def test_assemble_quantiles_global_fan_ascending(monkeypatch):
    """R3: the quantile fan is strictly ascending in τ under 'global' (the spread
    structure is unchanged by the median projection)."""
    from utils import assemble_quantiles
    _force_global(monkeypatch)

    B, P, S = 2, 4, 6
    torch.manual_seed(21)
    head_raw = torch.randn(B, P, S, 7) * 2.0
    last_bg = torch.tensor([85.0, 205.0])
    q_tau, _ = assemble_quantiles(head_raw, last_bg)
    diffs = q_tau[..., 1:] - q_tau[..., :-1]
    assert (diffs > 0).all(), f"fan not ascending: min {float(diffs.min()):.3e}"
    print("\n[DUMP] R3 | global fan strictly ascending ✓")


def test_assemble_quantiles_global_spreads_unchanged(monkeypatch):
    """R3: with two head_raw differing ONLY in col0 (the median delta), the band is
    a pure function of head_raw[...,1:] and the median shift — the global projection
    touches only `m`, never the spread fan. Witness: the per-quantile offset from the
    median, q[...,k]-m, is IDENTICAL across the two assemblies (every q carries ±m,
    so the comparison is done against each assembly's own median; the residual is the
    softplus+cumsum band, bit-identical since cols 1: are shared)."""
    from utils import assemble_quantiles
    _force_global(monkeypatch)

    B, P, S = 2, 4, 6
    torch.manual_seed(33)
    head_a = torch.randn(B, P, S, 7)
    head_b = head_a.clone()
    head_b[..., 0] = torch.randn(B, P, S)               # differ ONLY in col0
    last_bg = torch.tensor([100.0, 175.0])
    qa, ma = assemble_quantiles(head_a, last_bg)
    qb, mb = assemble_quantiles(head_b, last_bg)
    # The band offset from each assembly's OWN median is the softplus+cumsum of the
    # shared cols 1:, so it is median-independent (to fp tolerance: q = m ± cumsum, and
    # (m±c)∓m fp-cancels through the differing m, hence allclose not equal).
    rel_a = qa - ma.unsqueeze(-1)
    rel_b = qb - mb.unsqueeze(-1)
    assert torch.allclose(rel_a, rel_b, atol=1e-4), (
        f"band offset from median must be col0-independent (max diff "
        f"{float((rel_a - rel_b).abs().max()):.2e})")
    # And the medians genuinely differ (the projection of two different deltas).
    assert not torch.equal(ma, mb), "the two medians must differ (sanity)"
    print("\n[DUMP] R3 | band offset q-median is a pure function of head_raw[...,1:] ✓")
    print("\n[DUMP] R3 | band gaps are a pure function of head_raw[...,1:] (median-independent) ✓")


def test_assemble_quantiles_global_p1_edge(monkeypatch):
    """R3 P==1 edge: n=S=6, G clamps to <=6; no seams; within-patch low-pass; no
    crash; median==q_tau[...,3]."""
    from utils import assemble_quantiles
    _force_global(monkeypatch)

    B, P, S = 2, 1, 6
    torch.manual_seed(5)
    head_raw = torch.randn(B, P, S, 7) * 1.3
    last_bg = torch.tensor([100.0, 175.0])
    q_tau, median = assemble_quantiles(head_raw, last_bg)
    assert median.shape == (B, P, S)
    assert torch.equal(median, q_tau[..., 3])
    assert torch.isfinite(median).all()
    print("\n[DUMP] R3 P==1 edge | within-patch low-pass, median==q3, finite ✓")


def test_assemble_quantiles_mode_cumulative_bit_identical(monkeypatch):
    """Legacy preserved: 'cumulative' reproduces the EXACT R1 cumulative median
    (torch.equal vs the R1 reference formula)."""
    from utils import assemble_quantiles, kovatchev_f
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    _force_mode(monkeypatch, 'cumulative')

    B, P, S = 3, 4, 6
    torch.manual_seed(123)
    head_raw = torch.randn(B, P, S, 7) * 1.7
    last_bg = torch.tensor([88.0, 140.0, 205.0])
    q_tau, median = assemble_quantiles(head_raw, last_bg)

    # Match the code's EXACT op grouping (delta_cum = o + d_rel; m = anchor + delta_cum)
    # so torch.equal holds bit-for-bit (a different association reorders fp rounding).
    anchor = kovatchev_f(last_bg.detach().clamp(BG_CLAMP_MIN, BG_CLAMP_MAX)).view(-1, 1, 1)
    delta = head_raw[..., 0]
    d_rel = delta - delta[..., :1]
    rise = delta[..., -1] - delta[..., 0]
    o = torch.cumsum(rise, dim=1) - rise
    delta_cum = o.unsqueeze(-1) + d_rel
    m_ref = anchor + delta_cum
    assert torch.equal(median, m_ref), "'cumulative' must be bit-identical to R1"
    assert torch.equal(q_tau[..., 3], m_ref)
    print("\n[DUMP] R3 legacy | 'cumulative' torch.equal to R1 reference ✓")


def test_assemble_quantiles_mode_independent_bit_identical(monkeypatch):
    """Legacy preserved: 'independent' reproduces m=anchor+delta bit-identically."""
    from utils import assemble_quantiles, kovatchev_f
    from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    _force_mode(monkeypatch, 'independent')

    B, P, S = 2, 4, 6
    torch.manual_seed(7)
    head_raw = torch.randn(B, P, S, 7)
    last_bg = torch.tensor([95.0, 210.0])
    q_tau, median = assemble_quantiles(head_raw, last_bg)

    anchor = kovatchev_f(last_bg.clamp(BG_CLAMP_MIN, BG_CLAMP_MAX)).view(B, 1, 1)
    m_ref = anchor + head_raw[..., 0]
    assert torch.equal(median, m_ref), "'independent' must be bit-identical to anchor+delta"
    assert torch.equal(q_tau[..., 3], m_ref)
    print("\n[DUMP] R3 legacy | 'independent' torch.equal to anchor+delta ✓")


def test_global_median_basis_orthonormal_and_cached():
    """get_global_median_basis: orthonormal columns (BgᵀBg≈I), col0 is the constant
    DC mode, and a second call with the same key returns the cached basis."""
    from utils import get_global_median_basis
    from config import PREDICTION_PATCHES, PATCH_SIZE, BG_HEAD_MEDIAN_GLOBAL_DIM

    n = PREDICTION_PATCHES * PATCH_SIZE
    G = min(BG_HEAD_MEDIAN_GLOBAL_DIM, n)
    Bg = get_global_median_basis(n, G, 'dct')
    gram = Bg.T @ Bg
    assert torch.allclose(gram, torch.eye(G), atol=1e-5), (
        f"columns not orthonormal: max off-diag {float((gram - torch.eye(G)).abs().max()):.3e}")
    # col0 is the constant DC mode (all equal).
    col0 = Bg[:, 0]
    assert torch.allclose(col0, col0[0].expand_as(col0), atol=1e-6), "col0 must be the DC mode"
    # Cached: same underlying canonical tensor (same values; same object pre-move on CPU).
    Bg2 = get_global_median_basis(n, G, 'dct')
    assert torch.equal(Bg, Bg2)
    print(f"\n[DUMP] R3 basis | ({n},{G}) orthonormal, col0 DC (val {float(col0[0]):.3f}), cached ✓")


def test_global_median_dim_scales_with_span_length():
    """``G_L = max(1, ceil(G * L / PREDICTION_PATCHES))`` at EVERY ``L`` the
    sampler can draw, reproducing the configured ``G`` exactly at
    ``L == PREDICTION_PATCHES``, non-decreasing in ``L``, and always a strict
    projection (``1 <= G_L < n = L * PATCH_SIZE``).

    ``MASK_SPAN_LENGTHS`` is retuned between runs, so the table of dimensions it
    produces is computed from the formula rather than written down — a dict keyed
    on one span ceiling is a ``KeyError`` at the next.

    A FIXED ``G`` is a defect here, not an approximation: at ``L = 1`` it would be
    ``min(G, PATCH_SIZE)`` columns over ``PATCH_SIZE`` points — the identity — so
    the L2 contraction that kills the median drift is ABSENT, not weakened, and
    every fan assert still passes.  This test measures that difference directly."""
    from config import (BG_HEAD_MEDIAN_GLOBAL_DIM, MASK_SPAN_LENGTHS, PATCH_SIZE,
                        PREDICTION_PATCHES, BG_HEAD_STEP_BASIS_TYPE)
    from utils import get_global_median_basis, global_median_dim

    cutoffs = []
    dims = []
    for L in MASK_SPAN_LENGTHS:
        n = L * PATCH_SIZE
        g = global_median_dim(L)
        want = max(1, math.ceil(BG_HEAD_MEDIAN_GLOBAL_DIM * L / PREDICTION_PATCHES))
        assert g == want, f"G_L at L={L} is {g}, expected {want}"
        assert 1 <= g < n, f"G_L={g} must satisfy 1 <= G_L < n={n} at L={L}"
        dims.append(g)
        cutoffs.append(2.0 * n / g)
    assert dims == sorted(dims), f"G_L must be non-decreasing in L, got {dims}"
    assert global_median_dim(PREDICTION_PATCHES) == BG_HEAD_MEDIAN_GLOBAL_DIM, \
        "G_L must reproduce the configured G at L == PREDICTION_PATCHES"

    # The L = 1 contrast: the fixed-G projection is the identity, the scaled one
    # is not.  ``x - P x`` is exactly zero for the identity.
    torch.manual_seed(0)
    n1 = 1 * PATCH_SIZE
    x = torch.randn(64, n1)
    fixed = get_global_median_basis(n1, min(BG_HEAD_MEDIAN_GLOBAL_DIM, n1),
                                    BG_HEAD_STEP_BASIS_TYPE)
    scaled = get_global_median_basis(n1, global_median_dim(1), BG_HEAD_STEP_BASIS_TYPE)
    resid_fixed = float((x - (x @ fixed) @ fixed.T).abs().max())
    resid_scaled = float((x - (x @ scaled) @ scaled.T).abs().max())
    if BG_HEAD_MEDIAN_GLOBAL_DIM >= n1:
        assert resid_fixed < 1e-5, (
            f"a fixed G at L=1 should be the identity to 1e-5, residual {resid_fixed:.2e}")
    assert resid_scaled > 1e-2, (
        f"G_1 must actually contract at L=1, residual {resid_scaled:.2e}")
    print(f"\n[DUMP] G_L | {dims} over "
          f"L={list(MASK_SPAN_LENGTHS)}; cutoff 2n/G_L = "
          f"{[round(c, 1) for c in cutoffs]} steps; L=1 identity residual "
          f"fixed={resid_fixed:.2e} vs scaled={resid_scaled:.2e} ✓")


def test_assembled_median_basis_has_rank_g_l_per_span():
    """PER SPAN, the assembled median lives in a ``G_L``-dimensional subspace.

    Measured through ``assemble_quantiles`` itself, not on the basis alone: for
    each span length L a batch of random head outputs is assembled, the span's
    ``median - anchor`` block is stacked ``(B, L*PATCH_SIZE)`` with ``B >> n``, and
    its numerical rank must be exactly ``G_L``, computed from the formula rather
    than written down — ``MASK_SPAN_LENGTHS`` is retuned between runs.  A fixed
    ``G`` reports rank ``BG_HEAD_MEDIAN_GLOBAL_DIM`` at every L — full rank at
    L = 1, where the projection has silently become the identity."""
    import config
    from config import (MASK_SPAN_LENGTHS, MAX_MASKED_PATCHES, PATCH_SIZE,
                        N_SPREADS)
    from utils import assemble_quantiles, kovatchev_f, global_median_dim

    assert config.BG_HEAD_MEDIAN_MODE == 'global', \
        "the rank property belongs to the 'global' median mode"

    B, M, S = 96, MAX_MASKED_PATCHES, PATCH_SIZE
    ranks = {}
    padded = []
    for L in MASK_SPAN_LENGTHS:
        torch.manual_seed(100 + L)
        head_raw = torch.randn(B, M, S, 1 + 2 * N_SPREADS) * 1.5
        # One span of L patches starting at patch 3; the surplus slots are padded
        # and gather patch 0, which can never continue a span.
        mask_idx = torch.zeros(B, M, dtype=torch.int64)
        valid = torch.zeros(B, M, dtype=torch.bool)
        mask_idx[:, :L] = torch.arange(3, 3 + L)
        valid[:, :L] = True
        anchor_bg = torch.full((B, M), 120.0)

        _, median = assemble_quantiles(head_raw, anchor_bg, mask_idx, valid)
        anchor = kovatchev_f(anchor_bg[:, :1]).unsqueeze(-1)          # (B,1,1)
        block = (median[:, :L] - anchor).reshape(B, L * S)            # (B, n)
        # RELATIVE cutoff, not absolute.  The projection leaves fp32 rounding in
        # its null space, and that residue scales with the energy the basis
        # KEEPS — so an absolute threshold is a threshold on
        # BG_HEAD_MEDIAN_GLOBAL_DIM and on the head's output scale, not on rank.
        # At G = 6 the residue sat just under 1e-6 and at G = 12 just over it,
        # which reported rank G_L + 1 for a projection that is still exactly
        # rank G_L.  The spectrum has a seven-decade cliff at G_L (sv[G_L-1] ~ 8,
        # sv[G_L] ~ 1e-6, ratio ~6e-8), so any rtol between fp32 epsilon and the
        # signal separates them at every G.
        r = int(torch.linalg.matrix_rank(block.double(), rtol=1e-6))
        ranks[L] = r
        assert r == global_median_dim(L), (
            f"span of L={L} assembled a rank-{r} median, expected G_L="
            f"{global_median_dim(L)} over n={L * S} steps")
        # Padded slots carry no median of their own: they sit exactly at anchor.
        # At L == M the span fills every slot and there is no padding to check.
        if L < M:
            pad = median[:, L:] - anchor
            assert float(pad.abs().max()) == 0.0, \
                "a padded slot's median must be exactly its anchor (no head gradient)"
            padded.append(L)
    assert padded, "no L leaves a padded slot: the pad-at-anchor property is untested"
    print(f"\n[DUMP] per-span rank | L->rank {ranks} == G_L "
          f"{{{', '.join(f'{L}: {global_median_dim(L)}' for L in MASK_SPAN_LENGTHS)}}}; "
          f"pad-at-anchor over L={padded} of M={M} ✓")


def test_reshape_consistency_with_to_patch_major():
    """RESHAPE INVARIANT (highest-severity latent bug guard): the delta.reshape(B,P*S)
    used inside assemble_quantiles's global path equals risk_loss._to_patch_major(delta)
    — both C-contiguous patch-major (flat=p*S+s) — so the global basis and the
    DILATE median term index the time axis identically."""
    from risk_loss import _to_patch_major
    from config import PREDICTION_PATCHES, PATCH_SIZE

    B, P, S = 5, PREDICTION_PATCHES, PATCH_SIZE
    torch.manual_seed(2)
    delta = torch.randn(B, P, S)
    assert torch.equal(delta.reshape(B, P * S), _to_patch_major(delta)), (
        "assemble_quantiles reshape must match _to_patch_major (patch-major C-contiguous)")
    print("\n[DUMP] R3 reshape | delta.reshape(B,P*S) == _to_patch_major(delta) ✓")


# ---------------------------------------------------------------------------
# ModelEMA — the finiteness guard: one NaN in the live weights must not
# permanently poison the shadow (decay*NaN + ... == NaN forever otherwise).
# ---------------------------------------------------------------------------

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
