"""
What-if effectiveness — does an announced dose move the forecast the way the
physiology requires?

The model is always conditioned: whatever carbohydrate and insulin the caller
writes into the prediction zone is what it forecasts against, which makes the
what-if question ("what happens to my BG if I eat this / inject this") a
first-class deployment capability rather than a special mode. This script asks
whether the response to a declared dose is directionally right, proportionate,
monotone in dose, and quiet when nothing is declared.

Every number here is a response measured against the model's OWN baseline
forecast for the same window, never against a ground truth: a real record has no
observable counterfactual — a window was either dosed or it was not. The
simulator cannot supply one either, since ``T1DMSimulator.generate`` draws from
its RNG conditionally on BG (meal-bolus gating, the correction loop), so
re-running a patient with one extra dose injected diverges in its entire
subsequent event schedule. Read this as a physiologic-response characterization,
not an accuracy score. The in-domain twin is train.py's ``cf_*`` validation
probe, which runs the same idea on simulator windows.

A dose is injected as a point event at the forecast origin and convolved with the
meal-appearance / insulin-action kernels the real-data bridge already owns
(``realdata.features.CARB_KERNEL`` / ``BOLUS_KERNEL``), so the perturbation
enters as a curve rather than an amount dropped into one bucket. Both kernels run
240 min, well past the 2 h forecast window, so only the leading fraction of a
dose can act inside the horizon; ``_meta.kernel_mass_in_horizon`` reports that
fraction, without which every magnitude below reads low for a reason that has
nothing to do with the model.

Four blocks per cohort, from 11 forwards per window:

  ``carb`` / ``insulin`` — a dose ladder at (0, ¼, ½, 1, 2)× the ``CF_*`` dose the
    training probe uses. Per level: mean ΔBG and correct-sign fraction at each
    reported horizon, and the mean peak of the intended-direction response.
    Across the ladder: the per-window sensitivity slope through the origin
    (mg/dL per gram, mg/dL per unit, fitted on the terminal step), the onset
    latency of the reference dose and its peak latency over the windows that
    reach that onset, the mean ``adverse`` excursion — the
    largest move the reference dose makes AGAINST its own direction, which the
    per-horizon sign fractions otherwise only hint at — and monotonicity in three
    strengths: of the terminal response, of every horizon step at once, and as
    the plain fraction of (rung, step) pairs that hold.

    ``rescue`` asks whether a dose clears an excursion the baseline forecast
    predicts: baseline-hypo windows lifted back over ``BG_HYPO_THRESHOLD`` by
    carbohydrate, baseline-hyper windows brought under ``BG_HYPER_THRESHOLD`` by
    insulin. Both eligibility and success are scored only from
    ``RESCUE_LAG_MIN`` onward — a dose taken at the origin has no mass before
    then, so scoring the whole horizon would count every window whose nadir
    precedes the response as an unrescuable failure. The graded companion is the
    mean time spent beyond the threshold per rung.

  ``empty_future`` — the same window with every carbohydrate and bolus event
    inside the prediction zone stripped out. Basal and the tails of past events
    stay: neither can be un-injected. Reports turning points (reversals
    retracing more than ``TURN_MIN_MGDL``, so numerical wiggle well under CGM
    noise is not counted as shape), the largest counter-trend swing, the net
    drift, and the monotone fraction. Monotonicity is only the physiologic
    expectation on the ``quiet`` subset (nothing logged in the trailing context
    either) — a window opening mid-absorption may legitimately turn once, so the
    two subsets are reported apart.

  ``null_rail`` — the 0-dose arm reconstructs its overrides through the same
    raw → log1p → z path as a dosed arm, so its forecast must reproduce the
    baseline built by ``_future_overrides`` from the feature stack. A non-zero
    ``max_abs_dbg`` is a plumbing fault in the override path, not a model result.

Runs on the live best checkpoint (checkpoints/t1dmai_best.pt). GPU if available.
Writes metrics/whatif.json.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)                              # curves (local load_model/split_segments)
torch.set_num_threads(8)

from config import (PATCH_SIZE, PREDICTION_PATCHES, MAX_CONTEXT_PATCHES,
                    BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD,
                    CF_CARB_BOLUS_G, CF_INSULIN_BOLUS_U)
from realdata import load_dataset
from realdata.features import (build_feature_stack, context_window, segment_to_channels,
                               CARB_KERNEL, BOLUS_KERNEL, _convolve)
from realdata.calibrate import _future_overrides
from realdata.horizons import HORIZONS, HORIZON_IDX, GRID_MIN
from curves import load_model, split_segments
from inference import predict
import figstyle as F
from figstyle import plt

F.style()

PRED = PREDICTION_PATCHES * PATCH_SIZE
CTX = MAX_CONTEXT_PATCHES * PATCH_SIZE
DATASETS = ('ohiot1dm', 'azt1d', 'shanghai')
STRIDE = 8 * PATCH_SIZE
CAP = 40                                   # windows/segment
ANNOUNCE = (0, 1)
# Dose ladders anchored on the training probe's dose scale so the two probes read
# on one axis; the 1.0x rung IS the CF_* dose.
LADDER = (0.0, 0.25, 0.5, 1.0, 2.0)
CARB_DOSES = tuple(f * CF_CARB_BOLUS_G for f in LADDER)
INSULIN_DOSES = tuple(f * CF_INSULIN_BOLUS_U for f in LADDER)
REF_IDX = LADDER.index(1.0)
QUIET_CTX_MIN = 180                        # trailing context a 'quiet' window must have no events in
QUIET_STEPS = QUIET_CTX_MIN // GRID_MIN
TURN_MIN_MGDL = 2.0                        # retracement a reversal must exceed to count as a turn
TURN_HIST_MAX = 4                          # turning-point histogram bins 0..MAX-1, then MAX+
ONSET_EPS = 5.0                            # mg/dL a response must reach to count as begun
RESCUE_LAG_MIN = 30                        # a dose cannot act before this; rescue is scored after it
RESCUE_LAG = RESCUE_LAG_MIN // GRID_MIN
TERM = HORIZON_IDX[HORIZONS[-1]]           # terminal forecast step (120 min at the default geometry)
KERNEL_MASS = {'carb': float(CARB_KERNEL[:PRED].sum()),
               'insulin': float(BOLUS_KERNEL[:PRED].sum())}


def _dose_curve(dose: float, kernel: np.ndarray) -> np.ndarray:
    """Appearance/action curve of one ``dose`` injected at the window origin, (PRED,)."""
    events = np.zeros(PRED, dtype=np.float64)
    events[0] = dose
    return _convolve(events, kernel)


def _renorm(raw: np.ndarray, stats: dict, name: str) -> torch.Tensor:
    """Raw per-step (PRED,) sparse channel → normalized (P, S) override tensor.

    Mirrors ``build_feature_stack``'s log1p+z branch exactly, so a zero-dose arm
    reproduces the override sliced from the feature stack.
    """
    z = (np.log1p(np.maximum(raw, 0.0)) - stats[name]['mean']) / (stats[name]['std'] + 1e-8)
    return torch.from_numpy(z.reshape(PREDICTION_PATCHES, PATCH_SIZE).astype(np.float32))


def _turning_points(y: np.ndarray) -> int:
    """Direction reversals that retrace more than ``TURN_MIN_MGDL`` from the running extreme.

    A per-step sign flip would count numerical wiggle far below CGM noise as a
    turn; this walks the series and only commits to a reversal once the pullback
    is large enough to be a shape a reader would see.
    """
    turns, direction = 0, 0
    hi = lo = float(y[0])
    for v in y[1:]:
        v = float(v)
        hi, lo = max(hi, v), min(lo, v)
        if direction <= 0 and v - lo > TURN_MIN_MGDL:
            turns += int(direction != 0); direction = 1; hi = lo = v
        elif direction >= 0 and hi - v > TURN_MIN_MGDL:
            turns += int(direction != 0); direction = -1; hi = lo = v
    return turns


def _reversal(y: np.ndarray) -> float:
    """Largest swing against the window's net direction (mg/dL)."""
    if y[-1] >= y[0]:
        return float(np.max(np.maximum.accumulate(y) - y))
    return float(np.max(y - np.minimum.accumulate(y)))


def _latency(delta: np.ndarray, sign: int) -> tuple[float | None, float]:
    """(onset, peak) minutes of the INTENDED-direction response.

    Both read ``sign * delta``, not ``|delta|`` — a dose that first pushes BG the
    wrong way must not have that excursion counted as its own onset.  ``onset`` is
    None where the intended response never reaches ``ONSET_EPS``.
    """
    signed = sign * delta
    hit = np.nonzero(signed >= ONSET_EPS)[0]
    onset = float((hit[0] + 1) * GRID_MIN) if hit.size else None
    return onset, float((int(np.argmax(signed)) + 1) * GRID_MIN)


def _pct(v: list[float], q: float) -> float | None:
    return round(float(np.percentile(v, q)), 3) if v else None


def _frac(hits: int, n: int) -> float | None:
    return round(hits / n, 3) if n else None


def run(model, stats: dict, device, segs: list) -> dict:
    """Probe every strided window of ``segs``; returns the cohort's summary dict."""
    carb_d: list[np.ndarray] = []           # per window: (L, PRED) ΔBG vs the null arm
    ins_d: list[np.ndarray] = []
    nulls: list[np.ndarray] = []            # per window: (PRED,) null-arm forecast
    empties: list[np.ndarray] = []
    quiet: list[bool] = []
    rail: list[float] = []

    for seg in segs:
        n = (len(seg) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX + PRED:
            continue
        feats = build_feature_stack(seg, stats)
        ch = segment_to_channels(seg)
        carb_raw = np.clip(ch['carb'], 0.0, None)
        ins_raw = np.clip(ch['insulin'], 0.0, None)
        cnt = 0
        for ps in range(CTX, n - PRED + 1, STRIDE):
            if cnt >= CAP:
                break
            cnt += 1
            ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)

            def _fc(carb_t: torch.Tensor, ins_t: torch.Tensor) -> np.ndarray:
                out = predict(model, ctx, normalization_stats=stats, device=device,
                              overrides={0: carb_t, 1: ins_t})
                return out['median_bg'].detach().cpu().numpy().astype(np.float64)

            ov = _future_overrides(feats, ps, ANNOUNCE)
            base = _fc(ov[0], ov[1])

            c_fut = carb_raw[ps:ps + PRED]
            i_fut = ins_raw[ps:ps + PRED]
            c_null_t = _renorm(c_fut, stats, 'carb_intake')
            i_null_t = _renorm(i_fut, stats, 'insulin_combined')
            null = _fc(c_null_t, i_null_t)
            rail.append(float(np.max(np.abs(base - null))))

            cd = np.zeros((len(CARB_DOSES), PRED))
            for j, g in enumerate(CARB_DOSES):
                if j != 0:
                    cd[j] = _fc(_renorm(c_fut + _dose_curve(g, CARB_KERNEL),
                                        stats, 'carb_intake'), i_null_t) - null
            idl = np.zeros((len(INSULIN_DOSES), PRED))
            for j, u in enumerate(INSULIN_DOSES):
                if j != 0:
                    idl[j] = _fc(c_null_t, _renorm(i_fut + _dose_curve(u, BOLUS_KERNEL),
                                                   stats, 'insulin_combined')) - null

            # Strip only the events whose ONSET falls inside the prediction zone;
            # past-event tails and basal are physiologically un-retractable.
            c_empty = np.clip(c_fut - _convolve(seg.carb_grams[ps:ps + PRED], CARB_KERNEL), 0.0, None)
            i_empty = np.clip(i_fut - _convolve(seg.bolus_units[ps:ps + PRED], BOLUS_KERNEL), 0.0, None)
            empty = _fc(_renorm(c_empty, stats, 'carb_intake'),
                        _renorm(i_empty, stats, 'insulin_combined'))

            lo = max(0, ps - QUIET_STEPS)
            quiet.append(bool(np.all(seg.carb_grams[lo:ps + PRED] <= 0)
                              and np.all(seg.bolus_units[lo:ps + PRED] <= 0)))
            carb_d.append(cd); ins_d.append(idl); nulls.append(null); empties.append(empty)

    return _summarize(carb_d, ins_d, nulls, empties, quiet, rail)


def _side(deltas: list[np.ndarray], doses: tuple[float, ...], nulls: list[np.ndarray],
          sign: int, thr: float) -> dict:
    """Summarize one dose ladder. ``sign`` +1 expects BG up (carb), -1 down (insulin)."""
    n = len(deltas)
    if n == 0:
        return {'n': 0}
    D = np.stack(deltas)                                     # (N, L, PRED)
    dose_arr = np.asarray(doses[1:])
    resp = D[:, 1:, TERM]                                    # (N, L-1) terminal response
    slopes = (resp @ dose_arr) / float(dose_arr @ dose_arr)

    mono_step = np.diff(D, axis=1) * sign                    # (N, L-1, PRED)
    term_ok = np.all(mono_step[:, :, TERM] >= -1e-6, axis=1)
    point_ok = np.all(mono_step >= -1e-6, axis=(1, 2))
    inversions = np.count_nonzero(mono_step[:, :, TERM] < -1e-6, axis=1)

    # Peak is read only where an onset was reached: argmax over a response that
    # never materializes locates nothing, and would drag the median toward step 0.
    onsets, peaks = [], []
    for d in D[:, REF_IDX, :]:
        o, p = _latency(d, sign)
        if o is not None:
            onsets.append(o); peaks.append(p)
    adverse = np.maximum(-sign * D[:, REF_IDX, :], 0.0).max(axis=1)

    # A dose injected at the origin cannot act instantly, so a rescue is scored
    # only over the steps where its curve has appreciable mass; scoring the whole
    # horizon would fail every window whose baseline nadir precedes the response.
    N = np.stack(nulls)[:, RESCUE_LAG:]
    Dr = D[:, :, RESCUE_LAG:]
    exc = (N.min(axis=1) < thr) if sign > 0 else (N.max(axis=1) > thr)
    n_exc = int(exc.sum())
    if n_exc:
        arms = N[exc][:, None, :] + Dr[exc]                  # (M, L, PRED') absolute forecasts
        beyond = (arms < thr) if sign > 0 else (arms > thr)
        cleared = ~beyond.any(axis=2)
        minutes = beyond.sum(axis=2) * GRID_MIN              # (M, L) time beyond the threshold
        rescue = {'frac_by_dose': [round(float(c), 3) for c in cleared.mean(axis=0)],
                  'minutes_beyond_by_dose': [round(float(m), 1) for m in minutes.mean(axis=0)]}
    else:
        rescue = {'frac_by_dose': None, 'minutes_beyond_by_dose': None}

    return {
        'n': n,
        'doses': [round(d, 3) for d in doses],
        'mean_dbg': {str(h): [round(float(v), 2) for v in D[:, :, HORIZON_IDX[h]].mean(axis=0)]
                     for h in HORIZONS},
        # The 0-rung has no direction to be right about; null rather than a
        # spurious 0.0, so the array still aligns index-for-index with ``doses``.
        'correct_sign_frac': {str(h): [None] + [round(float(v), 3) for v in
                                                (sign * D[:, 1:, HORIZON_IDX[h]] > 0).mean(axis=0)]
                              for h in HORIZONS},
        'mean_peak_dbg': [round(float(v), 2) for v in
                          (sign * D).max(axis=2).mean(axis=0)],
        'mean_dbg_curve': [[round(float(v), 3) for v in row] for row in D.mean(axis=0)],
        'slope': {'median': round(float(np.median(slopes)), 4),
                  'p25': round(float(np.percentile(slopes, 25)), 4),
                  'p75': round(float(np.percentile(slopes, 75)), 4)},
        'ref_dose': round(doses[REF_IDX], 3),
        'mean_adverse_dbg': round(float(adverse.mean()), 2),
        'onset_min': {'median': _pct(onsets, 50), 'reached_frac': _frac(len(onsets), n)},
        'peak_min': {'median': _pct(peaks, 50)},
        'monotone_frac': {'terminal': round(float(term_ok.mean()), 3),
                          'pointwise': round(float(point_ok.mean()), 3),
                          'step': round(float((mono_step >= -1e-6).mean()), 3)},
        'mean_terminal_inversions': round(float(inversions.mean()), 3),
        'rescue': {'n': n_exc, 'scored_from_min': RESCUE_LAG_MIN, **rescue},
    }


def _empty_block(empties: list[np.ndarray]) -> dict:
    if not empties:
        return {'n': 0}
    turns = [_turning_points(y) for y in empties]
    rev = [_reversal(y) for y in empties]
    hist = [int(np.count_nonzero([t == k for t in turns])) for k in range(TURN_HIST_MAX)]
    hist.append(int(np.count_nonzero([t >= TURN_HIST_MAX for t in turns])))
    return {
        'n': len(empties),
        'turning_point_hist': hist,
        'mean_turning_points': round(float(np.mean(turns)), 3),
        'monotone_frac': round(float(np.mean([t == 0 for t in turns])), 3),
        'reversal_mgdl': {'mean': round(float(np.mean(rev)), 2), 'p90': _pct(rev, 90)},
        'net_drift_mgdl': round(float(np.mean([y[-1] - y[0] for y in empties])), 2),
    }


def _summarize(carb_d: list[np.ndarray], ins_d: list[np.ndarray], nulls: list[np.ndarray],
               empties: list[np.ndarray], quiet: list[bool], rail: list[float]) -> dict:
    q = np.asarray(quiet, dtype=bool)
    return {
        'n_windows': len(nulls),
        'n_quiet': int(q.sum()),
        'carb': _side(carb_d, CARB_DOSES, nulls, +1, BG_HYPO_THRESHOLD),
        'insulin': _side(ins_d, INSULIN_DOSES, nulls, -1, BG_HYPER_THRESHOLD),
        'empty_future': {
            'all': _empty_block(empties),
            'quiet': _empty_block([e for e, k in zip(empties, quiet) if k]),
        },
        'null_rail': {'max_abs_dbg': round(float(np.max(rail)), 4) if rail else None,
                      'mean_abs_dbg': round(float(np.mean(rail)), 4) if rail else None},
    }


def _report(ds: str, r: dict) -> None:
    print(f"\n== {ds} ==  windows={r['n_windows']}  quiet={r['n_quiet']}  "
          f"null-rail max|Δ|={r['null_rail']['max_abs_dbg']} mg/dL")
    for side, unit in (('carb', 'g'), ('insulin', 'U')):
        b = r[side]
        if not b['n']:
            continue
        ladder = '  '.join(f"{d:g}{unit}:{v:+.1f}" for d, v in
                           zip(b['doses'], b['mean_dbg'][str(HORIZONS[-1])]))
        sgn = ' / '.join(f"{b['correct_sign_frac'][str(h)][REF_IDX]:.2f}" for h in HORIZONS)
        s = b['slope']
        m = b['monotone_frac']
        print(f"  {side:7} Δ@{HORIZONS[-1]}m  {ladder}")
        print(f"          sign@{'/'.join(str(h) for h in HORIZONS)}m: {sgn}   "
              f"slope {s['median']:+.3f} mg/dL/{unit} (IQR {s['p25']:+.3f}..{s['p75']:+.3f})   "
              f"adverse {b['mean_adverse_dbg']:.1f} mg/dL")
        print(f"          onset {b['onset_min']['median']} min (reached {b['onset_min']['reached_frac']})"
              f"  peak {b['peak_min']['median']} min  "
              f"monotone term/point/step: {m['terminal']} / {m['pointwise']} / {m['step']}")
        resc = b['rescue']
        if resc['frac_by_dose'] is not None:
            arm = 'hypo' if side == 'carb' else 'hyper'
            cells = '  '.join(f"{d:g}{unit}:{f:.2f}/{mn:.0f}m" for d, f, mn in
                              zip(b['doses'], resc['frac_by_dose'], resc['minutes_beyond_by_dose']))
            print(f"          {arm} rescue after {resc['scored_from_min']}m "
                  f"(n={resc['n']}, cleared/time-beyond): {cells}")
    for tag in ('all', 'quiet'):
        e = r['empty_future'][tag]
        if e['n']:
            print(f"  empty {tag:5} (n={e['n']:4}): turns {e['mean_turning_points']:.2f}  "
                  f"monotone {e['monotone_frac']:.2f}  "
                  f"reversal {e['reversal_mgdl']['mean']:.1f} mg/dL (p90 {e['reversal_mgdl']['p90']:.1f})  "
                  f"net {e['net_drift_mgdl']:+.1f}")


def _panel_curves(ax, block: dict, ramp: tuple[str, ...], unit: str, title: str,
                  loc: str = 'upper left') -> None:
    """Mean ΔBG against the null arm over the horizon, one line per dose rung."""
    t = (np.arange(PRED) + 1) * GRID_MIN
    F.zeroline(ax)
    for j in range(1, len(block['doses'])):
        y = np.asarray(block['mean_dbg_curve'][j])
        ax.plot(t, y, color=ramp[j], label=f"{block['doses'][j]:g} {unit}")
        if j == len(block['doses']) - 1:       # direct-label the top rung only
            ax.annotate(f"{block['doses'][j]:g} {unit}", xy=(t[-1], y[-1]), xytext=(4, 0),
                        textcoords='offset points', va='center', fontsize=8, color=F.INK2)
    ax.set_title(title)
    ax.set_xlabel('minutes ahead'); ax.set_ylabel('ΔBG vs no added dose (mg/dL)')
    ax.set_xlim(0, t[-1] * 1.12)
    F.legend(ax, loc=loc, ncol=2)


def _panel_ladder(ax, r: dict) -> None:
    """Terminal response against dose, both channels indexed to their reference dose."""
    F.zeroline(ax)
    for block, color, name in ((r['carb'], F.SERIES[0], 'carbohydrate'),
                               (r['insulin'], F.SERIES[1], 'insulin')):
        y = [block['mean_dbg'][str(HORIZONS[-1])][j] for j in range(len(LADDER))]
        ax.plot(LADDER, y, color=color, marker='o', markersize=6,
                markeredgecolor=F.SURFACE, markeredgewidth=2,
                label=f"{name} (1× = {block['ref_dose']:g} {'g' if name[0] == 'c' else 'U'})")
    ax.set_title(f'response at {HORIZONS[-1]} min vs dose')
    ax.set_xlabel('dose (× reference)'); ax.set_ylabel('ΔBG (mg/dL)')
    ax.set_xticks(LADDER); ax.set_xticklabels([f'{m:g}×' for m in LADDER])
    F.legend(ax, loc='upper left')


def _panel_sign(ax, r: dict) -> None:
    """Fraction of windows moving the clinically correct way, by horizon."""
    for block, color, name in ((r['carb'], F.SERIES[0], 'carbohydrate raises BG'),
                               (r['insulin'], F.SERIES[1], 'insulin lowers BG')):
        y = [block['correct_sign_frac'][str(h)][REF_IDX] for h in HORIZONS]
        ax.plot(HORIZONS, y, color=color, marker='o', markersize=6,
                markeredgecolor=F.SURFACE, markeredgewidth=2, label=name)
        ax.annotate(f'{y[-1]:.2f}', xy=(HORIZONS[-1], y[-1]), xytext=(5, 0),
                    textcoords='offset points', va='center', fontsize=8, color=F.INK2)
    F.threshold(ax, 0.5, 'coin flip', side='left')
    ax.set_title(f'correct-sign fraction at the {LADDER[REF_IDX]:g}× rung')
    ax.set_xlabel('horizon (min)'); ax.set_ylabel('fraction of windows')
    ax.set_xticks(list(HORIZONS)); ax.set_ylim(0, 1.05)
    F.legend(ax, loc='lower right')


def _panel_rescue(ax, r: dict) -> None:
    """Fraction of forecast excursions each rung clears, scored past the dose's onset."""
    drawn = False
    for block, color, name in ((r['carb'], F.SERIES[0], 'hypo cleared by carbohydrate'),
                               (r['insulin'], F.SERIES[1], 'hyper cleared by insulin')):
        resc = block['rescue']
        if resc['frac_by_dose'] is None:
            continue
        ax.plot(LADDER, resc['frac_by_dose'], color=color, marker='o', markersize=6,
                markeredgecolor=F.SURFACE, markeredgewidth=2, label=f"{name}  (n={resc['n']})")
        drawn = True
    ax.set_title(f"excursion rescue, scored from {r['carb']['rescue']['scored_from_min']} min")
    ax.set_xlabel('dose (× reference)'); ax.set_ylabel('fraction cleared')
    ax.set_xticks(LADDER); ax.set_xticklabels([f'{m:g}×' for m in LADDER]); ax.set_ylim(0, 1.05)
    if drawn:
        F.legend(ax, loc='center right')
    else:
        ax.annotate('no baseline excursions to rescue', xy=(0.5, 0.5), xycoords='axes fraction',
                    ha='center', fontsize=9, color=F.MUTED)


def _panel_empty(ax, r: dict) -> None:
    """How often an untouched future turns, all windows vs the quiet subset."""
    e_all, e_q = r['empty_future']['all'], r['empty_future']['quiet']
    x = np.arange(TURN_HIST_MAX + 1)
    ticks = [str(k) for k in range(TURN_HIST_MAX)] + [f'{TURN_HIST_MAX}+']
    for k, (e, color, name) in enumerate(((e_all, F.PAIR[0], 'all windows'),
                                          (e_q, F.PAIR[1], 'quiet windows'))):
        if not e['n']:
            continue
        frac = np.asarray(e['turning_point_hist']) / e['n']
        ax.bar(x + (k - 0.5) * 0.42, frac, width=0.4, color=color,
               edgecolor=F.SURFACE, linewidth=1.2, label=f"{name} (n={e['n']})")
    ax.set_title('turning points with no future dose announced')
    ax.set_xlabel(f'reversals over {HORIZONS[-1]} min (> {TURN_MIN_MGDL:g} mg/dL retrace)')
    ax.set_ylabel('fraction of windows')
    ax.set_xticks(x); ax.set_xticklabels(ticks)
    F.ygrid(ax)
    F.legend(ax, loc='upper right')


def _fig_cohort(ds: str, r: dict, step) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))
    _panel_curves(axes[0, 0], r['carb'], F.DOSE_CARB, 'g', 'carbohydrate announced at the origin')
    _panel_curves(axes[0, 1], r['insulin'], F.DOSE_INS, 'U', 'insulin announced at the origin',
                  loc='lower left')
    _panel_ladder(axes[0, 2], r)
    _panel_sign(axes[1, 0], r)
    _panel_rescue(axes[1, 1], r)
    _panel_empty(axes[1, 2], r)
    fig.suptitle(f"What-if dose response — {F.label(ds)}   "
                 f"({r['n_windows']} windows, {r['n_quiet']} quiet · step {step})",
                 x=0.005, ha='left', fontsize=12, color=F.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return F.save(fig, f'whatif_{ds}.png')


def _fig_summary(res: dict, step) -> str:
    dss = [d for d in DATASETS if d in res]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    for ax, side, unit in ((axes[0, 0], 'carb', 'g'), (axes[0, 1], 'insulin', 'U')):
        x = np.arange(len(dss))
        med = np.array([res[d][side]['slope']['median'] for d in dss])
        p25 = np.array([res[d][side]['slope']['p25'] for d in dss])
        p75 = np.array([res[d][side]['slope']['p75'] for d in dss])
        bars = ax.bar(x, med, width=0.34, color=[F.cohort_color(d) for d in dss],
                      edgecolor=F.SURFACE, linewidth=1.2)
        ax.errorbar(x, med, yerr=[np.abs(med - p25), np.abs(p75 - med)], fmt='none',
                    ecolor=F.INK2, elinewidth=1.2, capsize=4)
        F.bar_labels(ax, bars, '{:+.2f}', dy=6.0,
                     tops=np.where(med >= 0, p75, p25))
        F.zeroline(ax); F.ygrid(ax)
        ax.set_title(f'{side} sensitivity at {HORIZONS[-1]} min (median, IQR)')
        ax.set_ylabel(f'mg/dL per {unit}')
        ax.set_xticks(x); ax.set_xticklabels([F.label(d) for d in dss])
        ax.margins(y=0.16)

    w = 0.2
    for ax, vals_of, title, ticks in (
            (axes[1, 0],
             lambda d: [res[d][s]['correct_sign_frac'][str(HORIZONS[-1])][REF_IDX]
                        for s in ('carb', 'insulin')],
             f'correct-sign fraction at {HORIZONS[-1]} min', ['carbohydrate', 'insulin']),
            (axes[1, 1],
             lambda d: [res[d]['empty_future'][t]['monotone_frac'] for t in ('all', 'quiet')],
             'windows with no reversal when nothing is announced',
             ['all windows', 'quiet windows'])):
        for k, d in enumerate(dss):
            bars = ax.bar(np.arange(2) + (k - 1) * w, vals_of(d), width=w - 0.04,
                          color=F.cohort_color(d), edgecolor=F.SURFACE, linewidth=1.2)
            F.bar_labels(ax, bars, '{:.2f}')
        F.ygrid(ax)
        ax.set_title(title)
        ax.set_ylabel('fraction of windows'); ax.set_ylim(0, 1.12)
        ax.set_xticks(range(2)); ax.set_xticklabels(ticks)
    F.threshold(axes[1, 0], 0.5, 'coin flip', side='left')

    fig.suptitle(f'What-if dose response — cohort summary (step {step})',
                 x=0.007, ha='left', fontsize=12, color=F.INK)
    F.cohort_legend(fig, dss)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return F.save(fig, 'whatif_summary.png')


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, stats, step = load_model(device)
    model = model.to(device); model.eval()
    print(f"[whatif] model step={step} device={device}  "
          f"kernel mass in horizon: carb {KERNEL_MASS['carb']:.2f} insulin {KERNEL_MASS['insulin']:.2f}")
    res = {'_meta': {'step': step, 'carb_doses_g': list(CARB_DOSES),
                     'insulin_doses_u': list(INSULIN_DOSES),
                     'kernel_mass_in_horizon': KERNEL_MASS,
                     'horizon_min': list(HORIZONS), 'quiet_context_min': QUIET_CTX_MIN}}
    for ds in DATASETS:
        segs = load_dataset(ds)
        _, test = split_segments(segs, ds)
        r = run(model, stats, device, test)
        res[ds] = r
        _report(ds, r)
    with open(os.path.join(HERE, 'whatif.json'), 'w') as f:
        json.dump(res, f, indent=2)
    paths = [_fig_cohort(ds, res[ds], step) for ds in DATASETS if ds in res]
    paths.append(_fig_summary(res, step))
    print(f"\n[whatif] wrote whatif.json and {len(paths)} figures under "
          f"{os.path.dirname(paths[-1])}/")


if __name__ == '__main__':
    main()
