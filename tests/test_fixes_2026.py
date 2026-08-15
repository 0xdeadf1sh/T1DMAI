"""Targeted tests for the 2026 fix batch — properties the unit suites for the
individual modules don't naturally cover end-to-end.

Each block pins one frozen-contract item:

* C-signbalance — ``sign_balance@h`` (fraction of truth strictly below the
  median, target 0.5) and ``inner50_cov@h`` (coverage of [τ.25, τ.75], target
  0.5) are computed correctly by ``train.compute_learning_metrics``.
* C-anchor — training (``data._build_sample`` ``last_bg = bg[pred_start-1]``) and
  inference (``utils.last_bg_mgdl_from_context``) read the SAME last raw context
  BG, so the model is anchored identically at train and deployment.
* C-assemble / C-rolling-phantom — ``inference.predict_rolling`` produces a
  band whose half-width is monotone non-decreasing across roll boundaries (the
  ``carry_spread`` accumulation), and re-feeds the zero-RAW carb/insulin
  baseline rather than a phantom z=0 dose.
* C-leak — the cache pool is carved into DISJOINT train/val/cal slabs so the
  +10M val / +2M cal seed bands can never reproject onto a train cache row.

The loss is the learned Kendall-Gal combine of pinball + DILATE; the median-curvature
(L_smooth) penalty and the seam (L_seam) penalty both stay retired (the smooth-basis
head and the global-median low-pass carry the anti-oscillation structurally).
"""
import math

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# C-signbalance — sign_balance@h and inner50_cov@h correctness.
# ---------------------------------------------------------------------------

def _bg_formula(true_bg: torch.Tensor) -> dict:
    """Minimal bg_formula_data for compute_learning_metrics (dt defaults to 5min
    via _dt_minutes when no dt_minutes key is packed)."""
    B, T = true_bg.shape
    return {
        'true_bg_trajectory': true_bg,
        'last_bg': true_bg[:, 0].clone(),
    }


def test_sign_balance_and_inner50_counts():
    """sign_balance counts true BG strictly below the median; inner50_cov counts
    true BG inside [τ.25, τ.75].  Construct a batch with a known split at the
    30-min horizon (h_idx=5 at dt=5min) and verify the emitted sums/counts."""
    from train import compute_learning_metrics
    from config import PREDICTION_PATCHES, PATCH_SIZE

    P, S = PREDICTION_PATCHES, PATCH_SIZE
    T = P * S
    B = 8
    h_idx = 30 // 5 - 1  # == 5

    pred_bg = torch.full((B, T), 120.0)            # median forecast, flat
    true_bg = torch.full((B, T), 120.0)
    # Make exactly 3 of 8 samples have truth strictly BELOW the median at 30 min.
    true_bg[:3, h_idx] = 100.0                     # below
    true_bg[3:, h_idx] = 140.0                     # above (not strictly below)

    # Inner band [110, 130] at 30 min: 5 of 8 samples have truth inside.
    inner_lo = torch.full((B, T), 110.0)
    inner_hi = torch.full((B, T), 130.0)
    true_bg[:5, h_idx + 0] = 120.0  # ensure handled below; reset for clarity
    # Put 5 inside [110,130], 3 outside at 30min — re-set deterministically.
    inside_vals = torch.tensor([115.0, 120.0, 125.0, 112.0, 128.0])  # 5 inside
    outside_vals = torch.tensor([90.0, 150.0, 200.0])                # 3 outside
    true_bg[:5, h_idx] = inside_vals
    true_bg[5:, h_idx] = outside_vals

    # Recompute the below-median count under the FINAL true_bg.
    expected_below = int((true_bg[:, h_idx] < pred_bg[:, h_idx]).sum())
    expected_inside = int(((true_bg[:, h_idx] >= 110.0) & (true_bg[:, h_idx] <= 130.0)).sum())

    # hypo_lo / hyper_hi are the (now required) clinical band-edge detector inputs;
    # here they sit in-range so they leave the sign_balance / inner50 counts alone.
    q_mgdl = {'lo': torch.full((B, T), 80.0), 'hi': torch.full((B, T), 200.0),
              'inner_lo': inner_lo, 'inner_hi': inner_hi,
              'hypo_lo': torch.full((B, T), 100.0), 'hyper_hi': torch.full((B, T), 150.0)}
    out = compute_learning_metrics(pred_bg, q_mgdl, _bg_formula(true_bg), P)

    assert out['sign_balance@30_below'] == float(expected_below)
    assert out['sign_balance@30_cnt'] == float(B)
    assert out['inner50_cov@30_hit'] == float(expected_inside)
    assert out['inner50_cov@30_cnt'] == float(B)
    print(f"\n[DUMP] sign_balance/inner50 | below={expected_below}/{B} "
          f"inside={expected_inside}/{B} ✓")


def test_inner50_absent_when_no_inner_band():
    """Without inner_lo/inner_hi keys the inner50 counts are zeroed (no crash) —
    the metric degrades gracefully when only the headline band is supplied."""
    from train import compute_learning_metrics
    from config import PREDICTION_PATCHES, PATCH_SIZE

    P, S = PREDICTION_PATCHES, PATCH_SIZE
    T = P * S
    B = 4
    pred_bg = torch.full((B, T), 120.0)
    true_bg = torch.full((B, T), 110.0)
    # No inner_lo/hi (the inner50 diagnostic degrades gracefully), but the clinical
    # band-edge detector inputs hypo_lo/hyper_hi are still required.
    q_mgdl = {'lo': torch.full((B, T), 80.0), 'hi': torch.full((B, T), 200.0),
              'hypo_lo': torch.full((B, T), 100.0), 'hyper_hi': torch.full((B, T), 150.0)}
    out = compute_learning_metrics(pred_bg, q_mgdl, _bg_formula(true_bg), P)
    assert out['inner50_cov@30_hit'] == 0.0 and out['inner50_cov@30_cnt'] == 0.0
    # sign_balance still works (median always available).
    assert out['sign_balance@30_cnt'] == float(B)
    print("[DUMP] inner50 | absent inner band -> zeroed, sign_balance unaffected ✓")


# ---------------------------------------------------------------------------
# C-band-edge — the clinical hypo/hyper detectors key off the band EDGES.
# ---------------------------------------------------------------------------

def test_hypo_hyper_detection_keys_off_band_edges():
    """CHANGE 3: hypo/hyper RECALL keys off the band edges, not the median.  Hypo
    fires when the LOWER band edge dips below the hypo threshold; hyper fires
    when the UPPER band edge rises above the hyper threshold.  A median that
    sits comfortably in range must NOT hide a band edge that has crossed — so a batch
    whose truth is out-of-range but whose median is in-range still scores full recall
    once the corresponding edge crosses.

    The band edges reach ``compute_learning_metrics`` as the q_mgdl ``'hypo_lo'``
    (== ``f_inv(q_tau[τ=HYPO_ALARM_QUANTILE_TAU])``) and ``'hyper_hi'``
    (== ``f_inv(q_tau[τ=HYPER_ALARM_QUANTILE_TAU])``) mg/dL keys that
    ``_run_validation`` fills at the config-tau indices.  We construct a mg/dL
    quantile fan directly and read those edges at the config taus (index via
    ``QUANTILE_LEVELS.index`` — never a bare literal)."""
    import config
    from train import compute_learning_metrics
    from config import (PREDICTION_PATCHES, PATCH_SIZE, QUANTILE_LEVELS,
                        BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD)

    lo_idx = QUANTILE_LEVELS.index(config.HYPO_ALARM_QUANTILE_TAU)   # config-selectable lower-band idx
    hi_idx = QUANTILE_LEVELS.index(config.HYPER_ALARM_QUANTILE_TAU)  # config-selectable upper-band idx
    assert QUANTILE_LEVELS[lo_idx] < 0.5 < QUANTILE_LEVELS[hi_idx], (lo_idx, hi_idx)

    P, S = PREDICTION_PATCHES, PATCH_SIZE
    T = P * S
    B = 4

    inner_lo_idx = QUANTILE_LEVELS.index(0.25)   # inner-50 band is FIXED, independent of the alarm τ
    inner_hi_idx = QUANTILE_LEVELS.index(0.75)

    def _edges(fan: torch.Tensor) -> dict:
        """Wrap a mg/dL (B,T,7) fan into the q_mgdl dict compute_learning_metrics
        consumes: headline τ.05/.95 as lo/hi, the FIXED τ.25/.75 as inner_lo/inner_hi,
        and the SELECTABLE clinical detector edges hypo_lo / hyper_hi at the config τ."""
        return {'lo': fan[..., 0], 'hi': fan[..., -1],
                'inner_lo': fan[..., inner_lo_idx], 'inner_hi': fan[..., inner_hi_idx],
                'hypo_lo': fan[..., lo_idx], 'hyper_hi': fan[..., hi_idx]}

    # --- hypo: truth IS hypo, the median is squarely in range, but the LOWER band
    # edge dips below the hypo threshold. A median<70 detector (pred==120) would
    # score ZERO recall; the band edge must catch every true hypo.
    offs = torch.tensor([-70.0, -65.0, -55.0, 0.0, 5.0, 10.0, 15.0])  # median at idx3
    fan_hypo = torch.full((B, T), 120.0).unsqueeze(-1) + offs          # (B,T,7) ascending
    median_hypo = fan_hypo[..., 3]                                     # == 120, in range
    assert (fan_hypo[..., lo_idx] < BG_HYPO_THRESHOLD).all()           # lower band edge < 70
    assert (median_hypo >= BG_HYPO_THRESHOLD).all()                    # median NOT hypo
    true_hypo_bg = torch.full((B, T), 55.0)                            # every cell true hypo
    out = compute_learning_metrics(
        median_hypo, _edges(fan_hypo), _bg_formula(true_hypo_bg), P)
    assert out['hypo_true'] == float(B * T)
    assert out['hypo_recall_hit'] == out['hypo_true'], (
        "hypo recall must fire off the lower band edge, not the in-range median")
    assert out['hypo_pred'] == float(B * T), "every step's lower band edge is < 70"

    # --- hyper: symmetric on the UPPER band edge.
    fan_hyper = torch.full((B, T), 120.0).unsqueeze(-1) + offs        # median 120 < 180
    median_hyper = fan_hyper[..., 3]
    # Lift the upper half so the upper band edge crosses the hyper threshold while
    # the median stays in range.
    fan_hyper = fan_hyper.clone()
    fan_hyper[..., hi_idx] = 190.0                                    # upper band edge > 180
    fan_hyper[..., -1] = 200.0                                        # keep ascending
    assert (fan_hyper[..., hi_idx] > BG_HYPER_THRESHOLD).all()
    assert (median_hyper <= BG_HYPER_THRESHOLD).all()
    true_hyper_bg = torch.full((B, T), 200.0)                        # every cell true hyper
    out_y = compute_learning_metrics(
        median_hyper, _edges(fan_hyper), _bg_formula(true_hyper_bg), P)
    assert out_y['hyper_true'] == float(B * T)
    assert out_y['hyper_recall_hit'] == out_y['hyper_true'], (
        "hyper recall must fire off the upper band edge, not the in-range median")
    assert out_y['hyper_pred'] == float(B * T), "every step's upper band edge is > 180"
    print(f"\n[DUMP] band-edge detector | hypo off lower τ={config.HYPO_ALARM_QUANTILE_TAU} "
          f"(idx {lo_idx}), hyper off upper τ={config.HYPER_ALARM_QUANTILE_TAU} (idx {hi_idx}); "
          f"median in-range yet full recall ✓")


def test_precision_tolerance_forgives_near_boundary():
    """Precision (not recall) carries a ±EXCURSION_PRECISION_TOLERANCE_MGDL forgiveness
    band: a predicted hypo whose band edge is within tol of a (non-hypo) true value is
    NOT a false alarm, so near-threshold CGM noise does not deflate precision; a far
    false alarm is still penalized, and recall stays strict."""
    import config
    from train import compute_learning_metrics
    from config import (PREDICTION_PATCHES, PATCH_SIZE, QUANTILE_LEVELS,
                        BG_HYPO_THRESHOLD, EXCURSION_PRECISION_TOLERANCE_MGDL)
    tol = EXCURSION_PRECISION_TOLERANCE_MGDL
    assert tol > 0, "test assumes the default nonzero precision tolerance"
    P, S = PREDICTION_PATCHES, PATCH_SIZE
    T, B = P * S, 4
    lo_idx = QUANTILE_LEVELS.index(config.HYPO_ALARM_QUANTILE_TAU)
    hi_idx = QUANTILE_LEVELS.index(config.HYPER_ALARM_QUANTILE_TAU)
    inner_lo_idx = QUANTILE_LEVELS.index(0.25)
    inner_hi_idx = QUANTILE_LEVELS.index(0.75)

    def _edges(fan: torch.Tensor) -> dict:
        return {'lo': fan[..., 0], 'hi': fan[..., -1],
                'inner_lo': fan[..., inner_lo_idx], 'inner_hi': fan[..., inner_hi_idx],
                'hypo_lo': fan[..., lo_idx], 'hyper_hi': fan[..., hi_idx]}

    # Median in range (120); the lower alarm edge (fan[..., lo_idx]) sits at 69
    # (< BG_HYPO_THRESHOLD -> the alarm FIRES).  Build an ascending fan whose
    # alarm edge lands at 69 for whatever HYPO_ALARM_QUANTILE_TAU resolves to,
    # rather than hardcoding a single quantile-level layout.
    med_idx = QUANTILE_LEVELS.index(0.5)
    med_val, edge_val = 120.0, 69.0
    offs_list = []
    for k in range(len(QUANTILE_LEVELS)):
        if k < lo_idx:                       # below the alarm edge
            offs_list.append(edge_val - 4.0 * (lo_idx - k) - med_val)
        elif k == lo_idx:                    # the alarm edge itself -> 69
            offs_list.append(edge_val - med_val)
        elif k < med_idx:                    # between alarm edge and median
            frac = (k - lo_idx) / (med_idx - lo_idx)
            offs_list.append((edge_val + frac * (med_val - edge_val)) - med_val)
        elif k == med_idx:                   # the median -> 0 offset
            offs_list.append(0.0)
        else:                                # above the median
            offs_list.append(5.0 * (k - med_idx))
    offs = torch.tensor(offs_list)
    fan = torch.full((B, T), 120.0).unsqueeze(-1) + offs
    assert (fan[..., lo_idx] < BG_HYPO_THRESHOLD).all() and (fan[..., med_idx] >= BG_HYPO_THRESHOLD).all()

    # NEAR: true just above 70, exactly tol from the 69 edge -> forgiven, NOT a false alarm.
    near = torch.full((B, T), 69.0 + tol)
    o_near = compute_learning_metrics(fan[..., med_idx], _edges(fan), _bg_formula(near), P)
    assert o_near['hypo_pred'] == float(B * T) and o_near['hypo_true'] == 0.0
    assert o_near['hypo_prec_hit'] == float(B * T), "near-boundary false alarm must be forgiven"
    assert o_near['hypo_recall_hit'] == 0.0, "recall stays strict — no true hypo, no recall hit"

    # FAR: true well outside the band -> a genuine false alarm (not forgiven).
    far = torch.full((B, T), 69.0 + tol + 40.0)
    o_far = compute_learning_metrics(fan[..., med_idx], _edges(fan), _bg_formula(far), P)
    assert o_far['hypo_prec_hit'] == 0.0, "far false alarm must NOT be forgiven"
    print(f"\n[DUMP] precision tolerance ±{tol:g} | near forgiven "
          f"({o_near['hypo_prec_hit']:.0f}/{o_near['hypo_pred']:.0f}), far not "
          f"({o_far['hypo_prec_hit']:.0f}/{o_far['hypo_pred']:.0f}) ✓")


# ---------------------------------------------------------------------------
# C-leak — DISJOINT train/val/cal cache slabs across master seeds.
# ---------------------------------------------------------------------------

def test_cache_slabs_disjoint_and_cover():
    """The three partition slabs are pairwise disjoint and exactly tile the pool."""
    from data import _cache_slab_geometry, CACHE_PARTITIONS

    for pool in (300_000, 1_000_000, 7, 9):
        bands = {p: _cache_slab_geometry(pool, p) for p in CACHE_PARTITIONS}
        spans = sorted((s, s + n) for s, n in bands.values())
        # Cover [0, pool) with no gap and no overlap.
        assert spans[0][0] == 0, f"slabs must start at 0: {spans}"
        assert spans[-1][1] == pool, f"slabs must end at pool={pool}: {spans}"
        for (lo0, hi0), (lo1, hi1) in zip(spans, spans[1:]):
            assert hi0 == lo1, f"slabs must be contiguous & disjoint: {spans}"
        for _p, (_s, _n) in bands.items():
            assert _n >= 1, f"slab {_p} empty at pool={pool}"
    print("\n[DUMP] cache slabs | disjoint + tiling for several pool sizes ✓")


def test_val_cal_rows_never_reproject_onto_train():
    """For several master seeds, the cache_idx a val/cal sample maps to can never
    equal a train sample's cache_idx — the leak the partition fix closes.  Mirror
    the dataset's ``cache_idx = slab_start + patient_seed % slab_size`` mapping and
    check the train slab and the val/cal slabs are non-overlapping for every drawn
    seed (val/cal bands are master_seed + {10M, 2M})."""
    from data import _cache_slab_geometry
    from utils import compute_patient_seed

    pool = 500_000
    train_lo, train_n = _cache_slab_geometry(pool, 'train')
    val_lo, val_n = _cache_slab_geometry(pool, 'val')
    cal_lo, cal_n = _cache_slab_geometry(pool, 'cal')
    train_hi = train_lo + train_n

    def cache_idx(seed: int, slab_lo: int, slab_n: int) -> int:
        return slab_lo + int(seed % slab_n)

    train_hits, val_hits, cal_hits = set(), set(), set()
    for master in (0, 1, 42, 1234, 999_983):
        for step in range(64):
            for pos in range(8):
                ts = compute_patient_seed(master, step, pos)
                vs = compute_patient_seed(master + 10_000_000, step, pos)
                cs = compute_patient_seed(master + 2_000_000, step, pos)
                train_hits.add(cache_idx(ts, train_lo, train_n))
                val_hits.add(cache_idx(vs, val_lo, val_n))
                cal_hits.add(cache_idx(cs, cal_lo, cal_n))

    # Every realized index lands in its own slab band.
    assert all(train_lo <= i < train_hi for i in train_hits)
    assert all(val_lo <= i < val_lo + val_n for i in val_hits)
    assert all(cal_lo <= i < cal_lo + cal_n for i in cal_hits)
    # The slabs are disjoint, so the realized index sets cannot intersect.
    assert train_hits.isdisjoint(val_hits), "val cache rows leaked into train"
    assert train_hits.isdisjoint(cal_hits), "cal cache rows leaked into train"
    assert val_hits.isdisjoint(cal_hits), "val/cal cache rows overlap"
    print(f"\n[DUMP] no-reproject | train={len(train_hits)} val={len(val_hits)} "
          f"cal={len(cal_hits)} realized idx, all disjoint ✓")


# (hypo-emphasis loss removed — its term and the tests that pinned it are retired)


def test_train_inference_anchor_identical():
    """C-anchor: training and inference must compute the SAME anchor — a mismatch
    makes the head learn a delta against an anchor it never sees at deployment,
    wrecking short-horizon accuracy.

    No-smoothing pipeline: inputs/target/anchor are raw post-noise signals.
    ``data._build_sample`` reads each slot's anchor off the raw mg/dL array at
    ``anchor_step``; inference has no raw array and reconstructs the same cell out
    of the normalized window through ``utils.last_bg_mgdl_from_context``, so the
    two agree to a round-trip ulp.  We build REAL samples and assert the parity on
    every valid slot.

    The claim is per SLOT, not on ``last_bg`` alone.  Reading the context's last
    cell is legal only while that patch is VISIBLE — feat 0 of a masked patch is a
    legal-looking ``z`` decoding to ~142 mg/dL on the balanced pool — and masking
    is not positional, so the last context patch is masked on a real share of
    windows.  ``anchor_step`` never points at one: the mandatory separator makes
    the patch left of a span visible, and a span at patch 0 reads its right
    neighbour instead.  The right-edge case is asserted separately, gated on that
    visibility, because it is the one the deployed forecast uses.
    """
    import os
    import numpy as np
    from utils import last_bg_mgdl_from_context
    from data import (_anchor_step_for_span, _build_sample, _make_simulator,
                      simulate_discard_warmup)
    from config import PATCH_SIZE, N_INPUT_FEATURES, PREDICTION_PATCHES
    from normalization import load_normalization_stats, NORM_STATS_FILE

    if not os.path.exists(NORM_STATS_FILE):
        pytest.skip("normalization_stats.json required")
    stats = load_normalization_stats()

    sim = _make_simulator(patient_seed=4242, uniform_skills=False)
    data = simulate_discard_warmup(sim, 33.0)
    icr = float(sim.patient.icr)

    n_slots = n_right_edge = 0
    # Several windows: the parity must hold for every drawn n_ctx / pred_start and
    # every masked set the sampler places in it.
    for seed in range(16):
        sample = _build_sample(data=data, icr=icr, stats=stats,
                               rng=np.random.default_rng(seed))
        n_ctx = int(sample['n_context_patches'])
        bf = sample['bg_formula_data']
        mask_idx, valid = bf['mask_idx'], bf['valid']
        seq_len = n_ctx + PREDICTION_PATCHES
        window = sample['patches'].reshape(seq_len, PATCH_SIZE, N_INPUT_FEATURES)

        # Spans are the maximal runs of adjacent masked patches, which is how
        # `utils._span_layout` recovers them too; each carries one anchor step.
        idx = mask_idx[valid].tolist()
        steps: list[int] = []
        run_start = idx[0]
        for prev, cur in zip(idx, idx[1:] + [None]):
            if cur != prev + 1:
                steps += [_anchor_step_for_span(run_start, prev - run_start + 1)] \
                    * (prev - run_start + 1)
                run_start = cur
        assert len(steps) == len(idx)

        a_train = bf['anchor_bg'][valid]
        a_infer = last_bg_mgdl_from_context(
            window, stats,
            patch_idx=np.asarray(steps) // PATCH_SIZE,
            step_idx=np.asarray(steps) % PATCH_SIZE).numpy()
        worst = float(np.abs(a_infer - a_train).max())
        assert worst < 1e-2, (seed, worst, a_train.tolist(), a_infer.tolist())
        n_slots += len(idx)

        # The deployed right-edge read, where the last context patch is visible.
        if (n_ctx - 1) not in idx:
            n_right_edge += 1
            ctx = window[:n_ctx]
            edge = float(last_bg_mgdl_from_context(ctx, stats).item())
            assert abs(edge - float(bf['last_bg'])) < 1e-2, (seed, edge, bf['last_bg'])

    assert n_right_edge > 0, \
        "no window left its context edge visible — the right-edge read is untested"
    print(f"[DUMP] anchor parity | {n_slots} slots over 16 windows agree to <1e-2 "
          f"mg/dL; {n_right_edge} of 16 had a visible context edge and matched "
          f"last_bg there ✓")


# ---------------------------------------------------------------------------
# C-assemble / C-rolling-phantom — predict_rolling band monotonicity + baseline.
# ---------------------------------------------------------------------------

def _rolling_stats():
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def test_predict_rolling_band_halfwidth_monotone():
    """The rolling band's terminal half-width is monotone NON-DECREASING across
    roll boundaries (the carry_spread accumulation keeps the fan from
    sawtooth-resetting at every new context).  We read the per-roll terminal-step
    (τ.95 − τ.05)/2 from the concatenated risk-space q_tau and assert it never
    shrinks roll-over-roll."""
    from inference import predict_rolling
    from model import T1DMAI
    from config import (PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES,
                        MIN_CONTEXT_PATCHES, QUANTILE_LEVELS)

    torch.manual_seed(0)
    model = T1DMAI()
    model.eval()
    stats = _rolling_stats()

    n_ctx = MIN_CONTEXT_PATCHES
    context = torch.randn(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)
    n_rolls = 4
    result = predict_rolling(model, context, patient_seed=42, n_rolls=n_rolls,
                             normalization_stats=stats)
    q_tau = result['q_tau']  # (n_rolls*PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES) risk
    assert q_tau.shape[0] == n_rolls * PREDICTION_PATCHES

    # Per-roll terminal-step half-width in risk space.
    half_widths = []
    for r in range(n_rolls):
        last_patch = (r + 1) * PREDICTION_PATCHES - 1
        last_step = q_tau[last_patch, -1]          # (N_QUANTILES,)
        hw = float((last_step[-1] - last_step[0]).clamp_min(0.0) * 0.5)
        half_widths.append(hw)

    for a, b in zip(half_widths, half_widths[1:]):
        assert b >= a - 1e-6, (
            f"rolling band half-width shrank across a boundary: {half_widths}")
    # Strictly grows overall (carry is strictly positive once the model emits any
    # spread, which it does — softplus floor BG_QUANTILE_SPREAD_MIN > 0).
    assert half_widths[-1] > half_widths[0], (
        f"band must widen over rolls, got {half_widths}")
    print(f"\n[DUMP] rolling band | terminal half-widths {['%.3f' % h for h in half_widths]} "
          "non-decreasing ✓")


def test_predict_rolling_phantom_baseline_not_z_zero():
    """C-rolling-phantom: re-fed carb/insulin context slots use the zero-RAW
    normalized baseline (``normalize(0)`` per channel), NOT torch.zeros — z=0
    would decode to a phantom ~0.39 g / ~0.14 U dose.  We confirm the module's
    computed baselines differ from 0.0 (sparse log1p channels have nonzero
    -mean/std) and that ``denormalize`` of the baseline is ~0 g / ~0 U."""
    import numpy as np
    from normalization import normalize, denormalize, CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS
    from config import CHANNEL_TO_FEAT

    stats = _rolling_stats()
    zero_raw = normalize(np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), stats)[0]
    carb_feat = CHANNEL_TO_FEAT[0]
    insulin_feat = CHANNEL_TO_FEAT[1]
    carb_z = float(zero_raw[carb_feat])
    insulin_z = float(zero_raw[insulin_feat])

    # The sparse channels are log1p z-scored, so the zero-dose baseline is -mean/std
    # which is NOT 0 (the phantom-dose trap).  At least one must be non-zero.
    assert abs(carb_z) > 1e-3 or abs(insulin_z) > 1e-3, (
        "zero-RAW baseline collapsed to z=0 — the phantom-dose guard is moot")

    # Decoding the baseline recovers ~0 physical dose (the intended re-feed).
    # The cache->input gather keeps the channel order, so the bg/carb/insulin
    # input feat index equals its CHANNEL_NAMES index (carb -> feat 1 -> channel 1).
    carb_name = CHANNEL_NAMES[carb_feat]  # feat 1 -> channel 'carb_intake'
    assert carb_name in SPARSE_LOG1P_CHANNELS, "carb must be a sparse log1p channel"
    back = denormalize(np.array([[carb_z]], dtype=np.float32), stats,
                       channel_names=[carb_name])[0, 0]
    assert abs(float(back)) < 1e-2, f"baseline must decode to ~0 dose, got {back}"
    print(f"\n[DUMP] rolling phantom | carb_z={carb_z:.4f} insulin_z={insulin_z:.4f} "
          f"!= 0, decode->~0 dose ✓")


# ---------------------------------------------------------------------------
# DILATE loss knobs (restored, replacing TILDE-Q) + the learned Kendall-Gal combine.
# The median-curvature (L_smooth) and seam (L_seam) penalties stay retired.
# ---------------------------------------------------------------------------

def test_dilate_knobs_in_valid_range():
    """DILATE is restored on the median: DILATE_ALPHA is the shape/TDI mix weight
    (alpha*shape + (1-alpha)*TDI, in [0, 1]), DILATE_GAMMA is the softmin softness
    (> 0) and DILATE_TDI_FD_EPS the TDI finite-difference step (> 0). The learned
    Kendall-Gal weighting is restored (KENDALL_LOGVAR_INIT); the median-curvature
    (L_smooth), TILDE-Q, and seam (L_seam) tunables must all be gone."""
    import config
    assert 0.0 <= config.DILATE_ALPHA <= 1.0, (
        f"DILATE_ALPHA must be in [0, 1], got {config.DILATE_ALPHA}")
    assert config.DILATE_GAMMA > 0.0, (
        f"DILATE_GAMMA must be > 0, got {config.DILATE_GAMMA}")
    assert config.DILATE_TDI_FD_EPS > 0.0, (
        f"DILATE_TDI_FD_EPS must be > 0, got {config.DILATE_TDI_FD_EPS}")
    assert hasattr(config, 'KENDALL_LOGVAR_INIT'), (
        "KENDALL_LOGVAR_INIT must exist (learned Kendall-Gal weighting restored)")
    for gone in ('TILDEQ_ALPHA', 'TILDEQ_GAMMA',
                 'MEDIAN_SEAM_PENALTY_ENABLED', 'MEDIAN_SEAM_PENALTY_WEIGHT',
                 'MEDIAN_SMOOTHNESS_ENABLED', 'MEDIAN_SMOOTHNESS_WEIGHT',
                 'PINBALL_LOSS_WEIGHT', 'DILATE_LOSS_WEIGHT'):
        assert not hasattr(config, gone), f"config.{gone} must be retired"
    print(f"\n[DUMP] DILATE knobs | alpha={config.DILATE_ALPHA} "
          f"gamma={config.DILATE_GAMMA} tdi_eps={config.DILATE_TDI_FD_EPS}; "
          f"Kendall-Gal restored, L_smooth/TILDE-Q/seam knobs retired ✓")


def test_loss_components_have_no_retired_keys():
    """``risk_total_loss`` exposes the DILATE + Kendall-Gal log-σ component keys and
    NONE of the retired TILDE-Q / seam / L_smooth keys — the components dict is the
    cross-owner contract the CSV/console writers key off."""
    from risk_loss import risk_total_loss, KendallGalWeighting
    from config import PREDICTION_PATCHES, PATCH_SIZE

    torch.manual_seed(0)
    B = 2
    median = 0.5 * torch.randn(B, PREDICTION_PATCHES, PATCH_SIZE)
    offs = torch.tensor([-.3, -.2, -.1, 0., .1, .2, .3])
    q = median.unsqueeze(-1) + offs
    true_bg = torch.full((B, PREDICTION_PATCHES, PATCH_SIZE), 120.0)
    _, parts = risk_total_loss(q, median, true_bg, KendallGalWeighting())

    for key in ('loss_Q', 'loss_D', 'loss_D_shape', 'loss_D_tdi',
                'log_sigma_Q', 'log_sigma_D'):
        assert key in parts, f"components dict missing DILATE/Kendall-Gal key {key!r}"
    for gone in ('loss_T', 'loss_T_ashift', 'loss_T_phase', 'loss_T_amp',
                 'loss_seam', 'log_sigma_T', 'loss_smooth'):
        assert gone not in parts, f"retired key {gone!r} must be gone from components"
    print(f"\n[DUMP] loss components | DILATE/Kendall-Gal keys present, "
          f"TILDE-Q/seam/L_smooth keys absent: {sorted(parts.keys())} ✓")


def test_cumulative_median_propagates_through_model_and_inference(monkeypatch):
    """The chokepoint propagation check: under BG_HEAD_MEDIAN_MODE='cumulative' (R1,
    forced here) model.forward produces C0-continuous medians at the patch seams, and
    the loss + backward flow finitely (no NaN). (The R3 default 'global' does NOT pin
    C0 — that is intentional, so this property test forces the cumulative mode.)"""
    import config
    from model import T1DMAI
    from risk_loss import risk_total_loss, KendallGalWeighting
    from config import PREDICTION_PATCHES, PATCH_SIZE, MAX_CONTEXT_PATCHES
    from tests.forward_inputs import right_edge_inputs
    monkeypatch.setattr(config, "BG_HEAD_MEDIAN_MODE", 'cumulative', raising=False)

    if PREDICTION_PATCHES <= 1:
        pytest.skip("no interior seams with PREDICTION_PATCHES==1")

    torch.manual_seed(0)
    model = T1DMAI().train()
    B = 3
    # ONE right-edge span: all PREDICTION_PATCHES slots belong to a single span, so
    # the cumulative median carries its offset across every interior seam.  Across
    # two spans the offset restarts at each span's own anchor, by design.
    patches, attn, anchor_bg, mask_idx = right_edge_inputs(
        B, n_ctx=MAX_CONTEXT_PATCHES, seed=0)

    q_tau, median = model(patches, attn, anchor_bg, mask_idx)
    # C0 continuity at every interior seam (forward path).
    end = median[:, :-1, PATCH_SIZE - 1]
    start = median[:, 1:, 0]
    max_gap = float((end - start).detach().abs().max())
    assert torch.allclose(end, start, atol=1e-5), (
        f"model.forward C0 violated: max |gap| {max_gap:.3e}")
    assert torch.allclose(median, q_tau[..., 3], atol=1e-6)

    # Loss + backward flow finitely.
    true_bg = torch.full((B, PREDICTION_PATCHES, PATCH_SIZE), 120.0)
    total, parts = risk_total_loss(q_tau, median, true_bg, KendallGalWeighting())
    assert torch.isfinite(total), "loss must be finite"
    total.backward()
    g = model.bg_head[-1].weight.grad
    assert g is not None and torch.isfinite(g).all(), "bg_head grad must be finite"
    print(f"\n[DUMP] R1 propagation | model.forward C0 (max seam |Δ| "
          f"{max_gap:.3e}), loss finite, grad finite ✓")
