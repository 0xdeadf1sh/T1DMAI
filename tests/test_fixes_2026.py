"""End-to-end properties the per-module unit suites do not naturally cover.

* ``sign_balance@h`` (truth strictly below the median, target 0.5) and
  ``inner50_cov@h`` (coverage of [τ.25, τ.75], target 0.5);
* the training and inference anchors read the SAME raw context BG;
* ``predict_rolling``'s band widens monotonically across roll boundaries and
  re-feeds the zero-RAW dose baseline rather than a phantom z = 0;
* the cache pool is carved into DISJOINT train/val/cal slabs, so the +10M val and
  +2M cal seed bands cannot reproject onto a train row.
"""
import math

import numpy as np
import pytest
import torch


def _bg_formula(true_bg: torch.Tensor) -> dict:
    """Minimal bg_formula_data; with no ``dt_minutes`` key, ``_dt_minutes`` gives 5."""
    B, T = true_bg.shape
    return {
        'true_bg_trajectory': true_bg,
        'last_bg': true_bg[:, 0].clone(),
    }


def test_sign_balance_and_inner50_counts():
    """A known split at the 30-minute horizon, h_idx = 5 at dt = 5 min."""
    from train import compute_learning_metrics
    from config import PREDICTION_PATCHES, PATCH_SIZE

    P, S = PREDICTION_PATCHES, PATCH_SIZE
    T = P * S
    B = 8
    h_idx = 30 // 5 - 1

    pred_bg = torch.full((B, T), 120.0)            # flat median forecast
    true_bg = torch.full((B, T), 120.0)
    true_bg[:3, h_idx] = 100.0                     # below the median
    true_bg[3:, h_idx] = 140.0                     # above, so not strictly below

    inner_lo = torch.full((B, T), 110.0)
    inner_hi = torch.full((B, T), 130.0)
    true_bg[:5, h_idx + 0] = 120.0
    inside_vals = torch.tensor([115.0, 120.0, 125.0, 112.0, 128.0])  # inside [110, 130]
    outside_vals = torch.tensor([90.0, 150.0, 200.0])                # outside it
    true_bg[:5, h_idx] = inside_vals
    true_bg[5:, h_idx] = outside_vals

    # counted under the FINAL true_bg
    expected_below = int((true_bg[:, h_idx] < pred_bg[:, h_idx]).sum())
    expected_inside = int(((true_bg[:, h_idx] >= 110.0) & (true_bg[:, h_idx] <= 130.0)).sum())

    # hypo_lo / hyper_hi are the required band-edge detector inputs; in range here, so
    # they leave the sign_balance and inner50 counts alone
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
    """Without inner_lo/inner_hi the inner50 counts zero rather than raising."""
    from train import compute_learning_metrics
    from config import PREDICTION_PATCHES, PATCH_SIZE

    P, S = PREDICTION_PATCHES, PATCH_SIZE
    T = P * S
    B = 4
    pred_bg = torch.full((B, T), 120.0)
    true_bg = torch.full((B, T), 110.0)
    # no inner_lo/hi, but hypo_lo/hyper_hi are still required
    q_mgdl = {'lo': torch.full((B, T), 80.0), 'hi': torch.full((B, T), 200.0),
              'hypo_lo': torch.full((B, T), 100.0), 'hyper_hi': torch.full((B, T), 150.0)}
    out = compute_learning_metrics(pred_bg, q_mgdl, _bg_formula(true_bg), P)
    assert out['inner50_cov@30_hit'] == 0.0 and out['inner50_cov@30_cnt'] == 0.0
    # sign_balance still works: the median is always available
    assert out['sign_balance@30_cnt'] == float(B)
    print("[DUMP] inner50 | absent inner band -> zeroed, sign_balance unaffected ✓")


def test_hypo_hyper_detection_keys_off_band_edges():
    """Hypo/hyper recall keys off the band EDGES, so an in-range median must not hide
    an edge that has crossed.

    The edges arrive as the q_mgdl ``hypo_lo`` / ``hyper_hi`` keys, ``f_inv`` of
    ``q_tau`` at the config taus, which are indexed through ``QUANTILE_LEVELS.index``
    and never a bare literal.
    """
    import config
    from train import compute_learning_metrics
    from config import (PREDICTION_PATCHES, PATCH_SIZE, QUANTILE_LEVELS,
                        BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD)

    lo_idx = QUANTILE_LEVELS.index(config.HYPO_ALARM_QUANTILE_TAU)
    hi_idx = QUANTILE_LEVELS.index(config.HYPER_ALARM_QUANTILE_TAU)
    assert QUANTILE_LEVELS[lo_idx] < 0.5 < QUANTILE_LEVELS[hi_idx], (lo_idx, hi_idx)

    P, S = PREDICTION_PATCHES, PATCH_SIZE
    T = P * S
    B = 4

    inner_lo_idx = QUANTILE_LEVELS.index(0.25)   # the inner-50 band is fixed, not the alarm τ
    inner_hi_idx = QUANTILE_LEVELS.index(0.75)

    def _edges(fan: torch.Tensor) -> dict:
        """A mg/dL (B, T, 7) fan as the q_mgdl dict: τ.05/.95 as lo/hi, the fixed
        τ.25/.75 as inner_lo/inner_hi, the selectable config τ as hypo_lo/hyper_hi."""
        return {'lo': fan[..., 0], 'hi': fan[..., -1],
                'inner_lo': fan[..., inner_lo_idx], 'inner_hi': fan[..., inner_hi_idx],
                'hypo_lo': fan[..., lo_idx], 'hyper_hi': fan[..., hi_idx]}

    # truth IS hypo and the median is in range, but the lower edge dips below the
    # threshold: a median<70 detector would score zero recall here
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

    # symmetric, on the upper edge
    fan_hyper = torch.full((B, T), 120.0).unsqueeze(-1) + offs        # median 120 < 180
    median_hyper = fan_hyper[..., 3]
    # lift the upper half so its edge crosses while the median stays in range
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
    """Precision, not recall, carries a ±EXCURSION_PRECISION_TOLERANCE_MGDL band: an
    edge within tol of a non-hypo truth is no false alarm, so near-threshold CGM noise
    cannot deflate it. A far false alarm still counts, and recall stays strict."""
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

    # median 120, alarm edge 69, so the alarm fires; built to land at 69 for whatever
    # HYPO_ALARM_QUANTILE_TAU resolves to rather than one hardcoded fan layout
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

    # NEAR: truth exactly tol from the 69 edge, so forgiven
    near = torch.full((B, T), 69.0 + tol)
    o_near = compute_learning_metrics(fan[..., med_idx], _edges(fan), _bg_formula(near), P)
    assert o_near['hypo_pred'] == float(B * T) and o_near['hypo_true'] == 0.0
    assert o_near['hypo_prec_hit'] == float(B * T), "near-boundary false alarm must be forgiven"
    assert o_near['hypo_recall_hit'] == 0.0, "recall stays strict — no true hypo, no recall hit"

    # FAR: truth well outside the band, a genuine false alarm
    far = torch.full((B, T), 69.0 + tol + 40.0)
    o_far = compute_learning_metrics(fan[..., med_idx], _edges(fan), _bg_formula(far), P)
    assert o_far['hypo_prec_hit'] == 0.0, "far false alarm must NOT be forgiven"
    print(f"\n[DUMP] precision tolerance ±{tol:g} | near forgiven "
          f"({o_near['hypo_prec_hit']:.0f}/{o_near['hypo_pred']:.0f}), far not "
          f"({o_far['hypo_prec_hit']:.0f}/{o_far['hypo_pred']:.0f}) ✓")


def test_cache_slabs_disjoint_and_cover():
    """The three partition slabs are pairwise disjoint and exactly tile the pool."""
    from data import _cache_slab_geometry, CACHE_PARTITIONS

    for pool in (300_000, 1_000_000, 7, 9):
        bands = {p: _cache_slab_geometry(pool, p) for p in CACHE_PARTITIONS}
        spans = sorted((s, s + n) for s, n in bands.values())
        # cover [0, pool) with no gap and no overlap
        assert spans[0][0] == 0, f"slabs must start at 0: {spans}"
        assert spans[-1][1] == pool, f"slabs must end at pool={pool}: {spans}"
        for (lo0, hi0), (lo1, hi1) in zip(spans, spans[1:]):
            assert hi0 == lo1, f"slabs must be contiguous & disjoint: {spans}"
        for _p, (_s, _n) in bands.items():
            assert _n >= 1, f"slab {_p} empty at pool={pool}"
    print("\n[DUMP] cache slabs | disjoint + tiling for several pool sizes ✓")


def test_val_cal_rows_never_reproject_onto_train():
    """A val or cal sample's cache_idx can never equal a train sample's.

    Mirrors the dataset's ``cache_idx = slab_start + patient_seed % slab_size`` over
    the val/cal seed bands, master_seed + {10M, 2M}.
    """
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

    # every realized index lands in its own slab band
    assert all(train_lo <= i < train_hi for i in train_hits)
    assert all(val_lo <= i < val_lo + val_n for i in val_hits)
    assert all(cal_lo <= i < cal_lo + cal_n for i in cal_hits)
    # the slabs are disjoint, so the realized index sets cannot intersect
    assert train_hits.isdisjoint(val_hits), "val cache rows leaked into train"
    assert train_hits.isdisjoint(cal_hits), "cal cache rows leaked into train"
    assert val_hits.isdisjoint(cal_hits), "val/cal cache rows overlap"
    print(f"\n[DUMP] no-reproject | train={len(train_hits)} val={len(val_hits)} "
          f"cal={len(cal_hits)} realized idx, all disjoint ✓")


def test_train_inference_anchor_identical():
    """Train and inference must compute the SAME anchor, or the head learns a delta
    against an anchor it never sees at deployment.

    ``_build_sample`` reads it off the raw mg/dL array at ``anchor_step``; inference
    reconstructs the same cell from the normalized window, so the two agree to a
    round-trip ulp. The claim is per SLOT: feat 0 of a masked patch is a legal-looking
    ``z`` decoding to ~142 mg/dL, so the right-edge read is asserted separately, gated
    on the context edge being visible.
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

    # ON_THE_FLY_SIM_HOURS, never a literal: it is what data.py requests for ONE
    # sample, so it follows MAX_CONTEXT_PATCHES. A literal stops covering the floor
    # when the window widens, and _build_sample then raises "No prediction window
    # found" instead of reporting an anchor mismatch.
    from data import ON_THE_FLY_SIM_HOURS

    sim = _make_simulator(patient_seed=4242, uniform_skills=False)
    data = simulate_discard_warmup(sim, ON_THE_FLY_SIM_HOURS)
    icr = float(sim.patient.icr)

    n_slots = n_right_edge = 0
    # the parity must hold for every drawn n_ctx and every masked set in it
    for seed in range(16):
        sample = _build_sample(data=data, icr=icr, stats=stats,
                               rng=np.random.default_rng(seed))
        n_ctx = int(sample['n_context_patches'])
        bf = sample['bg_formula_data']
        mask_idx, valid = bf['mask_idx'], bf['valid']
        seq_len = n_ctx + PREDICTION_PATCHES
        window = sample['patches'].reshape(seq_len, PATCH_SIZE, N_INPUT_FEATURES)

        # spans are the maximal runs of adjacent masked patches, as
        # ``utils._span_layout`` recovers them; each carries one anchor step
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

        # the deployed right-edge read, where the last context patch is visible
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


def _rolling_stats():
    import os
    from normalization import (compute_normalization_stats,
                               load_normalization_stats, NORM_STATS_FILE)
    if os.path.exists(NORM_STATS_FILE):
        return load_normalization_stats()
    return compute_normalization_stats(master_seed=42, n_patients=10, n_hours=72)


def test_predict_rolling_band_halfwidth_monotone():
    """The carry keeps the fan from sawtooth-resetting at each new context, so the
    per-roll terminal-step (τ.95 − τ.05)/2 never shrinks roll over roll."""
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
    q_tau = result['q_tau']  # (n_rolls*PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES), risk
    assert q_tau.shape[0] == n_rolls * PREDICTION_PATCHES

    # per-roll terminal-step half-width, in risk space
    half_widths = []
    for r in range(n_rolls):
        last_patch = (r + 1) * PREDICTION_PATCHES - 1
        last_step = q_tau[last_patch, -1]          # (N_QUANTILES,)
        hw = float((last_step[-1] - last_step[0]).clamp_min(0.0) * 0.5)
        half_widths.append(hw)

    for a, b in zip(half_widths, half_widths[1:]):
        assert b >= a - 1e-6, (
            f"rolling band half-width shrank across a boundary: {half_widths}")
    # strictly growing overall: the carry is positive once the model emits any spread,
    # which the softplus floor BG_QUANTILE_SPREAD_MIN > 0 guarantees
    assert half_widths[-1] > half_widths[0], (
        f"band must widen over rolls, got {half_widths}")
    # in QUADRATURE, not linearly: this model's native fan is near-identical on every
    # roll, so after n rolls the terminal half-width is √n × the first's. An additive
    # carry gives n×, twice too wide by the fourth roll and pinned to the rails soon after.
    for r in range(1, n_rolls):
        want = half_widths[0] * math.sqrt(r + 1)
        assert abs(half_widths[r] - want) / want < 0.02, (
            f"roll {r} terminal half-width {half_widths[r]:.3f} is not √{r + 1}× the "
            f"first roll's ({want:.3f}) — the carry is not composing in quadrature")
    print(f"\n[DUMP] rolling band | terminal half-widths {['%.3f' % h for h in half_widths]} "
          "non-decreasing, √n growth ✓")


def test_predict_rolling_carry_is_per_level():
    """ONE carry PER LEVEL, so a level resumes at its own width across a seam.

    Every level's offset is non-decreasing over a seam, and a roll's first-step
    .25/.75 pair still sits INSIDE the .05/.95 pair the previous roll ended on. One
    scalar carry seeded from the outermost level puts the inner pair outside at every
    seam, and two seams later the fan is one slab.
    """
    from inference import predict_rolling
    from model import T1DMAI
    from config import (PREDICTION_PATCHES, PATCH_SIZE, N_INPUT_FEATURES,
                        MIN_CONTEXT_PATCHES, QUANTILE_LEVELS, N_SPREADS)

    torch.manual_seed(0)
    model = T1DMAI()
    model.eval()
    stats = _rolling_stats()

    context = torch.randn(MIN_CONTEXT_PATCHES, PATCH_SIZE, N_INPUT_FEATURES)
    n_rolls = 3
    result = predict_rolling(model, context, patient_seed=42, n_rolls=n_rolls,
                             normalization_stats=stats)
    q_tau = result['q_tau']            # (n_rolls*PREDICTION_PATCHES, PATCH_SIZE, N_QUANTILES)
    med = QUANTILE_LEVELS.index(0.5)

    for r in range(n_rolls - 1):
        last = q_tau[(r + 1) * PREDICTION_PATCHES - 1, -1]   # terminal step of roll r
        first = q_tau[(r + 1) * PREDICTION_PATCHES, 0]       # first step of roll r+1
        for k in range(1, N_SPREADS + 1):
            up_prev = float(last[med + k] - last[med])
            up_next = float(first[med + k] - first[med])
            dn_prev = float(last[med] - last[med - k])
            dn_next = float(first[med] - first[med - k])
            assert up_next >= up_prev - 1e-6 and dn_next >= dn_prev - 1e-6, (
                f"level {k} narrowed across seam {r}: "
                f"up {up_prev:.4f}->{up_next:.4f}, dn {dn_prev:.4f}->{dn_next:.4f}")
        outer_prev_up = float(last[-1] - last[med])
        inner_next_up = float(first[med + 1] - first[med])
        outer_prev_dn = float(last[med] - last[0])
        inner_next_dn = float(first[med] - first[med - 1])
        assert inner_next_up < outer_prev_up, (
            f"seam {r}: the .75 edge ({inner_next_up:.4f}) left the previous roll's "
            f".95 edge ({outer_prev_up:.4f}) behind — the carry is shared across levels")
        assert inner_next_dn < outer_prev_dn, (
            f"seam {r}: the .25 edge ({inner_next_dn:.4f}) left the previous roll's "
            f".05 edge ({outer_prev_dn:.4f}) behind — the carry is shared across levels")
    print(f"\n[DUMP] rolling carry | per level, fan nested across {n_rolls - 1} seams ✓")


def test_predict_rolling_phantom_baseline_not_z_zero():
    """Re-fed carb/insulin slots take ``normalize(0)`` per channel, never ``torch.zeros``:
    z = 0 decodes to a phantom ~0.39 g / ~0.14 U dose."""
    import numpy as np
    from normalization import normalize, denormalize, CHANNEL_NAMES, SPARSE_LOG1P_CHANNELS
    from config import CHANNEL_TO_FEAT

    stats = _rolling_stats()
    zero_raw = normalize(np.zeros((1, len(CHANNEL_NAMES)), dtype=np.float32), stats)[0]
    carb_feat = CHANNEL_TO_FEAT[0]
    insulin_feat = CHANNEL_TO_FEAT[1]
    carb_z = float(zero_raw[carb_feat])
    insulin_z = float(zero_raw[insulin_feat])

    # the sparse channels are log1p z-scored, so the zero-dose baseline is -mean/std,
    # not 0 — at least one must be non-zero or the guard is moot
    assert abs(carb_z) > 1e-3 or abs(insulin_z) > 1e-3, (
        "zero-RAW baseline collapsed to z=0 — the phantom-dose guard is moot")

    # the cache->input gather keeps the channel order, so an input feat index equals
    # its CHANNEL_NAMES index: carb -> feat 1 -> channel 1
    carb_name = CHANNEL_NAMES[carb_feat]
    assert carb_name in SPARSE_LOG1P_CHANNELS, "carb must be a sparse log1p channel"
    back = denormalize(np.array([[carb_z]], dtype=np.float32), stats,
                       channel_names=[carb_name])[0, 0]
    assert abs(float(back)) < 1e-2, f"baseline must decode to ~0 dose, got {back}"
    print(f"\n[DUMP] rolling phantom | carb_z={carb_z:.4f} insulin_z={insulin_z:.4f} "
          f"!= 0, decode->~0 dose ✓")


def test_dilate_knobs_in_valid_range():
    """``DILATE_ALPHA`` is the shape/TDI mix, ``alpha*shape + (1-alpha)*TDI`` in [0, 1];
    ``DILATE_GAMMA`` the softmin softness; ``DILATE_TDI_FD_EPS`` the TDI finite
    difference step."""
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
    """The components dict is the cross-owner contract the CSV and console writers key
    off."""
    from risk_loss import risk_total_loss, KendallGalWeighting
    from config import PREDICTION_PATCHES, PATCH_SIZE

    torch.manual_seed(0)
    B = 2
    median = 0.5 * torch.randn(B, PREDICTION_PATCHES, PATCH_SIZE)
    offs = torch.tensor([-.3, -.2, -.1, 0., .1, .2, .3])
    q = median.unsqueeze(-1) + offs
    true_bg = torch.full((B, PREDICTION_PATCHES, PATCH_SIZE), 120.0)
    _, parts = risk_total_loss(q, median, true_bg, KendallGalWeighting())

    for key in ('loss_Q', 'loss_D', 'loss_D_shape', 'loss_D_tdi', 'loss_M',
                'log_sigma_Q', 'log_sigma_D'):
        assert key in parts, f"components dict missing DILATE/MSE/Kendall-Gal key {key!r}"
    for gone in ('loss_T', 'loss_T_ashift', 'loss_T_phase', 'loss_T_amp',
                 'loss_seam', 'log_sigma_T', 'loss_smooth'):
        assert gone not in parts, f"retired key {gone!r} must be gone from components"
    print(f"\n[DUMP] loss components | DILATE/Kendall-Gal keys present, "
          f"TILDE-Q/seam/L_smooth keys absent: {sorted(parts.keys())} ✓")


def test_crossing_alarm_auc_gives_tied_probabilities_average_ranks():
    """A saturated head ties every probability; ordinal ranks made the AUC 0 or 1 by row order."""
    from train import _crossing_alarm_metrics

    def _auc(pos_at: int) -> float:
        t = torch.zeros(4, 2, dtype=torch.bool)
        t[pos_at, 0] = True
        p = torch.ones(4, 2)                       # sigmoid saturated: every row identical
        return _crossing_alarm_metrics('xh', [p], [t], 0, 0.5)['xh_auc']

    aucs = [_auc(i) for i in range(4)]
    print(f"\n[DUMP] all-tied AUC by positive-row position: {aucs} (want 0.5 each)")
    assert aucs == [0.5, 0.5, 0.5, 0.5], "tied probabilities must score the midpoint"

    # untied ranking is unchanged: a perfect separator is still 1.0
    t = torch.zeros(4, 2, dtype=torch.bool)
    t[3, 0] = True
    p = torch.tensor([[0.1, 0.], [0.2, 0.], [0.3, 0.], [0.9, 0.]])
    assert _crossing_alarm_metrics('xh', [p], [t], 0, 0.5)['xh_auc'] == 1.0


def test_exercise_kernel_horizon_is_minutes_like_its_siblings():
    """``_*_KERNEL_MIN`` is a minutes horizon; the exercise kernel padded by raw step count."""
    from metrics.core.features import CARB_KERNEL, BOLUS_KERNEL, EXERCISE_KERNEL
    from T1DMSIM.simulator import DT_MINUTES

    n = 240 // DT_MINUTES
    print(f"\n[DUMP] kernel lengths carb/bolus/exercise: "
          f"{len(CARB_KERNEL)}/{len(BOLUS_KERNEL)}/{len(EXERCISE_KERNEL)} (want {n} each)")

    assert len(CARB_KERNEL) == len(BOLUS_KERNEL) == len(EXERCISE_KERNEL) == n
    assert abs(float(EXERCISE_KERNEL.sum()) - 1.0) < 1e-12
