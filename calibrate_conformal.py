"""Fit region-binned split-conformal quantile corrections on the reserved partition.

Post-hoc calibration step (run AFTER training): loads a checkpoint, runs the model
over the disjoint ``CALIBRATION_RESERVE_*`` simulator partition, fits the
per-horizon / per-quantile ``delta`` ONCE PER REGION BIN (``mondrian.fit_mondrian``),
stores it back into the checkpoint under ``conformal_delta``, and reports
raw-vs-marginal-vs-binned excursion-peak coverage on a disjoint test band so the
effect is visible.

THIS IS THE BAND THAT SHIPS. ``ckpt['conformal_delta']`` is what
``metrics/core/report.py`` attaches to the model and what every ``inference.predict``
call applies, so a marginal fit here leaves the deployed correction marginal
however many evaluation paths are binned.

TWO PROTOCOLS, ONE OF THEM SHIPPED. The forecast protocol (the trailing masked
span) is what feeds the alarm path, and it is the only protocol whose delta is
written to ``conformal_delta``. The infill protocol — an interior masked span,
bracketed by visible evidence on both sides — is fitted separately, announces
every marginal fallback on stdout, and is stored under ``conformal_delta_infill``
with ``shipped = False``. Infill residuals are the easier ones (its slots sit at
two-sided ``d`` = 1..2), and folding them into the shipped band would buy an
apparently tighter interval the forecast cannot honour.

The corrections live in mg/dL and are fit on the distribution they will serve; re-fit
on a held-out slice of that cohort (coverage validity needs cal/test exchangeability).

Usage:
    python calibrate_conformal.py --checkpoint checkpoints/t1dmai_best.pt [--n-cal 64]
    python calibrate_conformal.py --checkpoint checkpoints_blind/t1dmai_best.pt --blind

``--blind`` fits under the unconditioned policy ``train_blind.py`` trains, and the
flag must agree with the checkpoint's own ``masked_channel_policy`` — a mismatch
is refused, because the delta that would ship was fitted on a distribution the
model does not run in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics/sim"))

import config
import conformal
import mondrian
from model import T1DMAI
from data import blind_masked_doses, masked_channel_policy, zero_dose_fill
from inference import predict
from normalization import load_normalization_stats

H = config.PREDICTION_PATCHES * config.PATCH_SIZE
LEVELS = config.QUANTILE_LEVELS
MED = LEVELS.index(0.5)
LO, HI, H10 = LEVELS.index(0.05), LEVELS.index(0.95), LEVELS.index(0.10)

# --------------------------------------------------------------------------- #
# The infill protocol
# --------------------------------------------------------------------------- #
# One interior span, in the middle of the context, alongside the mandatory
# forecast span — ``inference._resolve_mask_spans`` requires the whole future zone
# to be masked whatever else is, since it carries no observed BG at all.  The
# interior span is bracketed on both sides, so its slots sit at two-sided
# ``d`` = min(j+1, L-j): 1, 2, 2, 1 at L = 4.  That is the easy regime, and it is
# exactly why this protocol's residuals are fitted and stored apart from the
# forecast's rather than pooled with them.
INFILL_SPAN_LEN = 4
INFILL_START_PATCH = config.MAX_CONTEXT_PATCHES // 2
assert INFILL_SPAN_LEN + config.PREDICTION_PATCHES <= config.MAX_MASKED_PATCHES, (
    "infill + forecast spans exceed the head's MAX_MASKED_PATCHES slots")
assert 0 < INFILL_START_PATCH and \
    INFILL_START_PATCH + INFILL_SPAN_LEN < config.MAX_CONTEXT_PATCHES, (
        "the infill span must be interior — a visible patch on each side")

# Per-STEP ``d`` of the infill span, patch-major: patch j is at min(j+1, L-j)
# patches from the nearest visible evidence on either side.
INFILL_D_PER_STEP = np.repeat(
    np.array([min(j + 1, INFILL_SPAN_LEN - j) for j in range(INFILL_SPAN_LEN)]),
    config.PATCH_SIZE)

ANNOUNCE = (0, 1, 2)                       # carb, insulin, exercise
# Every announceable channel is announced, and that is checked rather than left to
# read correctly: an announced set short of ``CHANNEL_TO_FEAT`` leaves the dropped
# slot at ``normalize(0)``, which for exercise_equiv is a legal "no session" value
# (−0.139 z on the balanced pool). The correction would then be fitted on a regime
# training never saw, and stored into the checkpoint as if it were the model's.
assert ANNOUNCE == tuple(config.CHANNEL_TO_FEAT), (
    f"announced set {ANNOUNCE} != announceable set {tuple(config.CHANNEL_TO_FEAT)}")


def _check_policy(ckpt: dict, blind: bool) -> str:
    """The checkpoint's masked-channel policy, or ``SystemExit`` if it is not
    the one this fit would run.

    A conformal delta is valid only under cal/test exchangeability with the
    distribution the model runs in, and the two policies ARE different
    distributions — one announces the future plan, the other withholds it. The
    delta lands in ``ckpt['conformal_delta']``, which is the band that ships, so
    a fit under the wrong policy is not a bad number on a page: it is a wrong
    interval on the phone.

    An absent key reads as announced. The key was introduced with the blind
    trainer, which always stamps it, so nothing that lacks it is blind.

    Args:
        ckpt: the loaded checkpoint dict.
        blind: whether ``--blind`` was passed.

    Returns:
        The checkpoint's policy string, when it matches.
    """
    stored = str((ckpt.get('training_config') or {}).get(
        'masked_channel_policy', masked_channel_policy(blind=False)))
    wanted = masked_channel_policy(blind=blind)
    if stored != wanted:
        raise SystemExit(
            f"masked_channel_policy mismatch: the checkpoint was trained "
            f"{stored!r} and this fit would run {wanted!r}. The delta lands in "
            "ckpt['conformal_delta'] and every inference call handed it applies "
            "it, so a band fitted on the other regime would ship. "
            + ('Drop --blind.' if blind else 'Pass --blind.')
        )
    return stored


def _blind_context(ctx, fill: dict[int, float]):
    """A copy of ``ctx`` with the infill span's dose channels withheld.

    The interior span this script masks sits INSIDE the context, so its doses
    arrive through ``context_window`` rather than through an override, and
    ``inference._build_patches_tensor`` withholds only bg there — the announced
    policy. Under the blind policy they go with the bg, and this is where.

    Args:
        ctx: ``(n_ctx, PATCH_SIZE, N_INPUT_FEATURES)`` context patches.
        fill: ``data.zero_dose_fill``'s ``{feat: z}``.

    Returns:
        A new tensor of the same shape; ``ctx`` is not modified.
    """
    import torch
    n_ctx = ctx.shape[0]
    flat = ctx.reshape(n_ctx, config.PATCH_SIZE * config.N_INPUT_FEATURES).clone()
    masked = torch.zeros(n_ctx, dtype=torch.bool)
    masked[INFILL_START_PATCH:INFILL_START_PATCH + INFILL_SPAN_LEN] = True
    blind_masked_doses(flat, masked, fill)
    return flat.reshape(n_ctx, config.PATCH_SIZE, config.N_INPUT_FEATURES)


def _collect(model, seeds, stats, device, infill: bool = True,
             blind: bool = False) -> dict:
    """Run the model over fresh sim patients under both protocols.

    Returns a dict of stacked arrays:

    ``q`` ``(N,H,K)`` / ``true`` ``(N,H)`` / ``peak`` ``(N,)`` / ``exc`` ``(N,)`` /
    ``patient`` ``(N,)`` — the FORECAST protocol (the trailing masked span), and
    ``iq`` ``(Ni,H,K)`` / ``itrue`` ``(Ni,H)`` / ``ipatient`` ``(Ni,)`` — the INFILL
    protocol, empty when ``infill`` is False.

    The infill pass masks the interior span AND the forecast zone in one forward
    (the future zone carries no observed BG, so it is never left visible), then
    keeps the interior rows only: ``mask_idx < MAX_CONTEXT_PATCHES``.

    ``blind`` calibrates under the unconditioned policy ``train_blind.py`` trains:
    nothing is announced in the future zone (which leaves it at the same
    ``normalize(0)`` the blind fill is), and the interior span's doses are
    withheld with its bg. Split conformal is only valid on the distribution the
    model runs under, so this has to match the checkpoint — ``main`` refuses a
    mismatch rather than fitting a band on the wrong regime.
    """
    import sim_data as S
    from sim_data import build_sim_feature_stack, _smooth_sim_bg, _future_overrides
    from metrics.core.features import context_window
    from metrics.core.calibrate import CTX_STEPS, PRED_STEPS
    n_ctx = config.MAX_CONTEXT_PATCHES
    spans = [(INFILL_START_PATCH, INFILL_SPAN_LEN), (n_ctx, config.PREDICTION_PATCHES)]
    # Window-relative first step of the infill span; the context block ends at ps.
    infill_step0 = -CTX_STEPS + INFILL_START_PATCH * config.PATCH_SIZE
    fill = zero_dose_fill(stats) if blind else None
    Q, T, J, E, P = [], [], [], [], []
    IQ, IT, IP = [], [], []
    for pid, d in S.make_sim_runs(seeds, 96.0):
        feats = build_sim_feature_stack(d, stats)
        cgm = _smooth_sim_bg(d['bg_observed'])
        n = (len(cgm) // config.PATCH_SIZE) * config.PATCH_SIZE
        for ps in range(CTX_STEPS, n - PRED_STEPS + 1, 4 * config.PATCH_SIZE):
            tr = cgm[ps:ps + H]
            lb = float(cgm[ps - 1])
            if len(tr) < H:
                continue
            ctx = context_window(feats, ps, n_ctx)
            # Blind: announce nothing, so every future dose slot stays at the
            # zero-RAW baseline the blind fill IS.
            ov = None if blind else _future_overrides(feats, ps, ANNOUNCE)
            out = predict(model, ctx, normalization_stats=stats, device=device,
                          overrides=ov)
            Q.append(out['bands'].detach().cpu().numpy().reshape(H, config.N_QUANTILES))
            T.append(tr)
            J.append(int(np.argmax(np.abs(tr - lb))))
            E.append(tr.max() - tr.min() > 25)
            P.append(pid)
            if infill:
                ictx = _blind_context(ctx, fill) if fill is not None else ctx
                iout = predict(model, ictx, normalization_stats=stats, device=device,
                               overrides=ov, mask_spans=spans)
                midx = iout['mask_idx'].detach().cpu().numpy()
                keep = np.flatnonzero(midx < n_ctx)
                assert keep.size == INFILL_SPAN_LEN, (
                    f"infill rows {keep.size} != span length {INFILL_SPAN_LEN}")
                bands = iout['bands'].detach().cpu().numpy()[keep]
                IQ.append(bands.reshape(-1, config.N_QUANTILES))
                a = ps + infill_step0
                itr = cgm[a:a + INFILL_SPAN_LEN * config.PATCH_SIZE]
                assert len(itr) == INFILL_SPAN_LEN * config.PATCH_SIZE, (
                    f"infill truth {len(itr)} steps at ps={ps}")
                IT.append(itr)
                IP.append(pid)
    res = {'q': np.asarray(Q), 'true': np.asarray(T), 'peak': np.asarray(J),
           'exc': np.asarray(E), 'patient': list(P),
           'iq': np.asarray(IQ), 'itrue': np.asarray(IT), 'ipatient': list(IP)}
    return res


def _peak_coverage(q, true, j, exc):
    """Excursion-peak coverage, lower-edge escape, hypo-edge escape and MEAN WIDTH.

    The width travels with the coverage because the two are traded against each
    other: a correction that widens every interval raises coverage and says
    nothing on its own.
    """
    rows = [i for i in range(len(q)) if exc[i]]
    c90 = np.mean([q[i, j[i], LO] <= true[i, j[i]] <= q[i, j[i], HI] for i in rows])
    below = np.mean([true[i, j[i]] < q[i, j[i], LO] for i in rows])
    hypo = np.mean([true[i, j[i]] < q[i, j[i], H10] for i in rows])
    width = np.mean([q[i, j[i], HI] - q[i, j[i], LO] for i in rows])
    return float(c90), float(below), float(hypo), float(width)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n-cal', type=int, default=512,
                    help="reserved-partition patients to fit on. The fitted band's OWN "
                         "coverage is a random variable in this: over 60 random splits, "
                         "test coverage has sd 0.036 at 64 (p5-p95 0.838-0.946), 0.020 "
                         "at 256, 0.015 at 512, 0.011 at 1024. At 64 the mondrian "
                         "low-BG bin also draws ~14 windows, under its own "
                         "min_n_own_fit of 39, and silently falls back to the marginal "
                         "delta")
    ap.add_argument('--no-write', action='store_true', help='report only; do not modify the checkpoint')
    ap.add_argument('--no-infill', action='store_true',
                    help='skip the infill protocol (halves the forward passes); the '
                         'shipped forecast delta is unaffected either way')
    ap.add_argument('--blind', action='store_true',
                    help="fit under the unconditioned policy train_blind.py trains: "
                         "nothing announced in the future zone, and the infill span's "
                         "doses withheld with its bg. Must match the checkpoint's "
                         "masked_channel_policy — a mismatch is refused")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = T1DMAI().to(device)
    sd = ckpt['model_state_dict']
    ema = ckpt.get('model_ema_state_dict')
    model.load_state_dict({k: ema.get(k, v) for k, v in sd.items()} if ema else sd, strict=True)
    model.eval()
    stats = ckpt.get('normalization_stats') or load_normalization_stats()

    print(f"masked_channel_policy: {_check_policy(ckpt, args.blind)}")

    ms = config.MASTER_SEED
    cal_seeds = tuple(ms + config.CALIBRATION_RESERVE_SEED_OFFSET + i for i in range(args.n_cal))
    test_seeds = tuple(range(8000, 8012))   # disjoint test band (the sim eval seeds)
    print(f"fitting conformal on {args.n_cal} reserved patients; evaluating on {len(test_seeds)} test patients…")
    do_infill = not args.no_infill
    cal = _collect(model, cal_seeds, stats, device, infill=do_infill, blind=args.blind)
    test = _collect(model, test_seeds, stats, device, infill=do_infill, blind=args.blind)

    # ------------------------------------------------------------------ #
    # The FORECAST protocol — the only one that ships.
    # ------------------------------------------------------------------ #
    cal_bin = mondrian.region_bin(mondrian.forecast_destination(cal['q'], MED))
    test_bin = mondrian.region_bin(mondrian.forecast_destination(test['q'], MED))
    delta, marginal, meta = mondrian.fit_mondrian(
        cal['q'], cal['true'], cal_bin, LEVELS, MED,
        patients=cal['patient'], protocol='forecast', verbose=True)

    tq_marg = conformal.apply_quantile_conformal(test['q'], marginal, MED)
    tq_mond = mondrian.apply_mondrian(test['q'], delta, test_bin, MED)

    # All three arms measured in THIS run: an archived figure predates the
    # kovatchev_f_inv clamp change, so a stored raw baseline is not comparable.
    r = _peak_coverage(test['q'], test['true'], test['peak'], test['exc'])
    m = _peak_coverage(tq_marg, test['true'], test['peak'], test['exc'])
    c = _peak_coverage(tq_mond, test['true'], test['peak'], test['exc'])
    n_exc = int(np.sum(test['exc']))
    n_exc_pat = len({p for p, e in zip(test['patient'], test['exc']) if e})
    print(f"\nexcursion-peak, n={n_exc} windows from {n_exc_pat} patients "
          f"(of {len(test['q'])} windows, {len(set(test['patient']))} patients)")
    print(f"  {'':<28}{'raw':>9}{'marginal':>10}{'binned':>9}   target")
    print(f"  {'90% coverage':<28}{r[0]:9.3f}{m[0]:10.3f}{c[0]:9.3f}    0.90")
    print(f"  {'mean 90% width (mg/dL)':<28}{r[3]:9.1f}{m[3]:10.1f}{c[3]:9.1f}      —")
    print(f"  {'truth below lower-05 edge':<28}{r[1]:9.3f}{m[1]:10.3f}{c[1]:9.3f}    0.05")
    print(f"  {'truth below tau=0.10 (hypo)':<28}{r[2]:9.3f}{m[2]:10.3f}{c[2]:9.3f}    0.10")

    fc_report = mondrian.bin_report(
        {'raw': test['q'], 'marginal': tq_marg, 'binned': tq_mond},
        test['true'], test_bin, LO, HI, patients=test['patient'],
        step_groups=mondrian.forecast_d_step_groups(config.PREDICTION_PATCHES,
                                                    config.PATCH_SIZE))
    mondrian.print_bin_report(fc_report, 0.90,
                              "forecast protocol, per region bin and per d")

    # ------------------------------------------------------------------ #
    # The INFILL protocol — its own coarse fit, never written to the band.
    # ------------------------------------------------------------------ #
    idelta = imeta = None
    if do_infill and len(cal['iq']) and len(test['iq']):
        print(f"\ninfill protocol: interior span ({INFILL_START_PATCH}, {INFILL_SPAN_LEN}), "
              f"two-sided d per patch {sorted(set(INFILL_D_PER_STEP.tolist()))}")
        ical_bin = mondrian.region_bin(mondrian.forecast_destination(cal['iq'], MED))
        itest_bin = mondrian.region_bin(mondrian.forecast_destination(test['iq'], MED))
        idelta, imarg, imeta = mondrian.fit_infill_conformal(
            cal['iq'], cal['itrue'], ical_bin, LEVELS, MED, patients=cal['ipatient'])
        iq_marg = conformal.apply_quantile_conformal(test['iq'], imarg, MED)
        iq_mond = mondrian.apply_mondrian(test['iq'], idelta, itest_bin, MED)
        iarms = {'raw': test['iq'], 'marginal': iq_marg, 'binned': iq_mond}
        # Reported per d (never pooled, never on span length): pooling over d mixes
        # the bracketed slots with the ones a forecast would have to reach.
        mondrian.print_bin_report(
            mondrian.bin_report(iarms, test['itrue'], itest_bin, LO, HI,
                                patients=test['ipatient'],
                                step_groups=mondrian.d_step_groups(INFILL_D_PER_STEP)),
            0.90, "infill protocol, per region bin and per d (NOT shipped)")

    if not args.no_write:
        # Store as a TORCH tensor, not a numpy array: torch.load(weights_only=True)
        # (the secure default since PyTorch 2.6, used by load_model and the whole
        # metrics pipeline) refuses to unpickle numpy's _reconstruct, so a
        # numpy delta makes the checkpoint unloadable on the safe path. Tensors and
        # plain python types (the meta below) are weights_only-safe.
        #
        # ``conformal_delta`` is now (n_bins, S, K) and is selected by the window's
        # own region, so a consumer must group its windows by bin and call
        # ``conformal.apply_quantile_conformal`` once per group with that bin's
        # (S, K) slice — ``mondrian.apply_mondrian`` does exactly that. A consumer
        # that passes the whole stack straight through fails the 2-D assert in
        # ``conformal.apply_quantile_conformal`` rather than mis-broadcasting.
        assert meta['shipped'] is True and meta['protocol'] == 'forecast'
        ckpt['conformal_delta'] = torch.from_numpy(delta.astype(np.float32))
        ckpt['conformal_delta_marginal'] = torch.from_numpy(marginal.astype(np.float32))
        ckpt['conformal_meta'] = {**meta, 'n_cal_patients': args.n_cal,
                                  'space': 'mgdl', 'source': 'sim'}
        if idelta is not None:
            # Never merged into the band: infill's residuals are the easy ones.
            assert imeta['shipped'] is False and imeta['protocol'] == 'infill'
            ckpt['conformal_delta_infill'] = torch.from_numpy(idelta.astype(np.float32))
            ckpt['conformal_meta_infill'] = {
                **imeta, 'space': 'mgdl', 'source': 'sim',
                'span': [INFILL_START_PATCH, INFILL_SPAN_LEN],
                'd_per_step': INFILL_D_PER_STEP.tolist()}
        torch.save(ckpt, args.checkpoint)
        print(f"\n  stored conformal_delta {tuple(delta.shape)} (n_bins, S, K) in "
              f"{args.checkpoint}; marginal baseline under conformal_delta_marginal"
              + ("" if idelta is None else
                 f"; infill under conformal_delta_infill {tuple(idelta.shape)} "
                 f"(shipped=False)"))


if __name__ == '__main__':
    main()
