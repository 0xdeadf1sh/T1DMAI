"""
48 h day-figure machinery: the tiled 2 h / 8 h forecast curves and the plotting
that renders them.

This module is the shared half — the model loader, the sigma accumulation, the
per-day forecast assembly and ``plot_day``. ``curves_sim.py`` is the driver that
feeds it fresh T1DMSIM patients and writes the figures.

Every forecast here runs the FORECAST protocol and takes its masked set from
``metrics.protocols`` rather than deriving a trailing zone from position.

Runs on the live best checkpoint (checkpoints/t1dmai_best.pt). GPU if available.
"""
from __future__ import annotations
import os, sys, json
from dataclasses import replace
from datetime import timedelta
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)                          # protocols
torch.set_num_threads(8)

from config import (PATCH_SIZE, PREDICTION_PATCHES, MAX_CONTEXT_PATCHES,
                    NIGHT_LONG_HORIZON_PATCHES, NIGHT_LONG_HORIZON_HOURS,
                    _PATCHES_PER_HOUR, PREDICTION_HORIZON_HOURS,
                    BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD, CHANNEL_TO_FEAT,
                    TIME_PROBE_N_BINS)
from normalization import load_normalization_stats
from model import T1DMAI
from metrics.core.calibrate import _future_overrides, CTX_STEPS
from metrics.core.features import build_feature_stack, context_window
from metrics.core.schema import GRID_MIN, MIN_SEGMENT_STEPS
from inference import predict, predict_rolling
from utils import time_of_day_decode_bins
from clock_face import draw_clock_axis
from T1DMSIM.simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
import protocols as PR


def smooth_bg_truth(bg_raw) -> np.ndarray:
    """RAW BG, bg-clamped (the ground truth every metric/figure scores against).

    The pipeline consumes raw post-noise signals (``data._build_sample``): the
    scored ground-truth BG is the raw CGM that produced the model input, physical-
    range clamped. Name kept for its call-sites; no FIR smoothing is applied.
    """
    return np.clip(np.asarray(bg_raw, dtype=np.float64),
                   BG_CLAMP_MIN, BG_CLAMP_MAX).astype(float)

# Every forecast on this page runs the FORECAST protocol, and takes its masked
# set from ``metrics.protocols`` rather than deriving a trailing zone from
# position: ``PR.forecast_masked_set(n_ctx)`` names the span, the head slots it
# occupies, its per-slot d and the rows those slots occupy in ``predict``'s
# output. The INFILL protocol is the other fixed one and is not drawn here.
# ``protocols`` carries the d rule and the pooling prohibition; nothing on this
# page restates them.
PRED = PR.SPAN_STEPS                              # steps in one masked forecast span
DAY_PATCHES = 48 * _PATCHES_PER_HOUR              # 48 h day window (fixed figure span)
H8_PATCHES = NIGHT_LONG_HORIZON_PATCHES           # NIGHT_LONG_HORIZON_HOURS long-horizon window
CTX = MAX_CONTEXT_PATCHES * PATCH_SIZE
ANNOUNCE = (0, 1, 2)                              # carb, insulin, exercise

# Every announceable channel is announced, and that is checked here rather than
# left to read correctly: an announced set short of ``CHANNEL_TO_FEAT`` leaves the
# dropped slot at ``normalize(0)``, which for exercise_equiv is a legal "no
# session" value (−0.139 z on the balanced pool), so the eval silently scores a
# regime training never saw.
assert ANNOUNCE == tuple(CHANNEL_TO_FEAT), (
    f"announced set {ANNOUNCE} != announceable set {tuple(CHANNEL_TO_FEAT)}")

# Output-channel → input-feature index (carb 0→feat 1, insulin 1→feat 2,
# exercise 2→feat 3) for the per-roll night-onset conditioning, derived from the
# shared ``CHANNEL_TO_FEAT`` so it tracks the canonical layout (matching
# ``calibrate._future_overrides``).
_ANNOUNCE_FEAT_IDX = {ch: CHANNEL_TO_FEAT[ch] for ch in ANNOUNCE}

CKPT = os.path.join(ROOT, 'checkpoints', 't1dmai_best.pt')


def load_model(device, path: str = CKPT):
    """Load a checkpoint, building the model to match the WEIGHTS it carries.

    Reimplemented here (rather than imported from the foundation report module,
    which depends on deleted config symbols) so the deep-dive scripts import
    cleanly. Returns ``(model, normalization_stats, step)``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    m = T1DMAI().to(device)
    sd = ckpt['model_state_dict']
    ema = ckpt.get('model_ema_state_dict')
    m.load_state_dict({k: ema.get(k, v) for k, v in sd.items()} if ema else sd, strict=True)
    m.eval()
    return m, ckpt.get('normalization_stats') or load_normalization_stats(), ckpt.get('step')


def _slice_seg(seg, a: int, b: int):
    """A sub-Segment over steps [a, b), with t0 advanced accordingly.

    Must name every length-N array, the optional pre-resolved curves included:
    ``replace`` copies un-named fields through at full length, which would
    misalign them against the sliced CGM.
    """
    return replace(
        seg, t0=seg.t0 + timedelta(minutes=GRID_MIN * a),
        cgm=seg.cgm[a:b], carb_grams=seg.carb_grams[a:b], bolus_units=seg.bolus_units[a:b],
        basal_rate=seg.basal_rate[a:b], exercise=seg.exercise[a:b],
        carb_curve=(None if seg.carb_curve is None else seg.carb_curve[a:b]),
        insulin_curve=(None if seg.insulin_curve is None else seg.insulin_curve[a:b]))




def _pick_day_start(seg) -> int | None:
    """Patch-aligned day-window start nearest midnight with ``DAY_PATCHES`` ahead
    and ``MAX_CONTEXT_PATCHES`` of context behind. ``None`` if the segment is too
    short. (Reimplemented locally — the foundation report module dropped its
    day-figure helpers with the risk-space redesign.)"""
    Lp = len(seg) // PATCH_SIZE
    lo_p, hi_p = MAX_CONTEXT_PATCHES, Lp - DAY_PATCHES
    if hi_p <= lo_p:
        return None
    hod = seg.hour_of_day()
    best = min(range(lo_p, hi_p + 1),
               key=lambda ip: min(hod[ip * PATCH_SIZE], 24.0 - hod[ip * PATCH_SIZE]))
    return best * PATCH_SIZE


def _make_night_overrides_fn(feats: np.ndarray, pred_start: int,
                             announce: tuple[int, ...], stats: dict):
    """Per-roll announced carb(0)/insulin(1)/exercise(2) overrides for ``predict_rolling``.

    Roll ``r`` masks the same right-edge span, advanced by one horizon; this slices
    the announced channels (normalized) from ``feats`` over that span. The raw side
    of the tuple is unused by the risk-space rolling path so a zero placeholder is
    returned. ``None`` once a roll runs past the segment.

    Past roll 0 the span's left neighbour is the previous roll's own output, so its
    slots are again d = 1..PREDICTION_PATCHES but measured from a FABRICATED
    reading, not an observed one.
    """
    n = feats.shape[0]

    def fn(roll_idx: int, mu_np, abs_n_ctx: int):
        a = pred_start + roll_idx * PRED
        b = a + PRED
        if b > n:
            return None
        ov_norm: dict[int, np.ndarray] = {}
        ov_raw: dict[int, np.ndarray] = {}
        for ch in announce:
            fidx = _ANNOUNCE_FEAT_IDX[ch]
            col = feats[a:b, fidx].astype(np.float32)
            ov_norm[ch] = col.reshape(PREDICTION_PATCHES, PATCH_SIZE).copy()
            ov_raw[ch] = np.zeros_like(ov_norm[ch])
        return ov_norm, ov_raw
    return fn

def _onsets(x: np.ndarray) -> int:
    x = np.asarray(x) > 0
    return int(x[0]) + int(np.sum(x[1:] & ~x[:-1]))




def _bg(out) -> np.ndarray:
    """Headline BG forecast of a ROLLED pass as a flat mg/dL array.

    ``predict_rolling`` concatenates one forecast-protocol span per roll and
    returns them as ``pred_bg``; the rolls are contiguous by construction, so the
    flat array is already in step order.
    """
    pb = out['pred_bg'] if 'pred_bg' in out else out['median_bg']
    pb = pb.detach().cpu().numpy() if hasattr(pb, 'detach') else np.asarray(pb)
    return pb.flatten()


def _protocol_bg(out, ms) -> np.ndarray:
    """Headline BG of a SINGLE pass, ordered by the protocol's scored steps.

    ``predict`` returns one row per masked patch in head-slot order, which is a
    trailing zone only because the FORECAST protocol's masked set happens to be
    one. Selecting the scored rows through ``ms.scored_rows()`` and sorting by
    ``ms.scored_steps()`` reads the same array for the forecast protocol and the
    right array for any other masked set.
    """
    med = out['median_bg'].detach().cpu().numpy().reshape(-1, PATCH_SIZE)
    vals = med[ms.scored_rows()].ravel()
    return vals[np.argsort(ms.scored_steps())]


# --- BG forecast uncertainty. The model emits per-τ risk-space quantile bands,
# but here we plot a CALIBRATED empirical per-horizon error envelope instead (it
# is directly comparable across the tiled origins): slide windows over the cohort,
# and at each horizon step h accumulate the squared error of the headline
# ``median_bg`` forecast vs true CGM. The per-step RMSE √E[(pred−true)²] is the
# ±1σ half-width (a predictive-error band that widens with horizon).
# The horizon axis IS the distance-to-evidence axis: within one masked span, step
# h sits in patch ceil(h / PATCH_SIZE) and therefore at d = that patch number,
# one-sided. Past a single pass the forecast is rolled and d restarts at 1 per
# roll, against the previous roll's own output rather than an observed reading.

def _sigma_accumulate(model, stats, feats, cgm, horizon_patches, stride_steps, acc,
                      max_windows, report=None):
    """Accumulate per-horizon squared error of the headline forecast for one trajectory.

    Always conditioned: each window announces its true future carbs/insulin/exercise.

    ``horizon_patches`` is covered by ``nr`` rolls of the FORECAST protocol's
    masked set, so the envelope is binned on d within each roll — never on span
    length and never on arm. ``report`` (a ``protocols.RunReport``) records each
    window's realised d when supplied."""
    H = horizon_patches * PATCH_SIZE
    nr = horizon_patches // PREDICTION_PATCHES
    n = (len(cgm) // PATCH_SIZE) * PATCH_SIZE
    ms = PR.forecast_masked_set(MAX_CONTEXT_PATCHES)
    for ts in range(CTX, n - H + 1, stride_steps):
        if acc['count'] >= max_windows:
            return
        ctx = context_window(feats, ts, MAX_CONTEXT_PATCHES)
        if nr == 1:
            ov = _future_overrides(feats, ts, ANNOUNCE)
            out = predict(model, ctx, normalization_stats=stats, overrides=ov,
                          mask_spans=ms.spans)
            bg_all = _protocol_bg(out, ms)
        else:
            fn = _make_night_overrides_fn(feats, ts, ANNOUNCE, stats)
            out = predict_rolling(model, ctx, n_rolls=nr, normalization_stats=stats, overrides_fn=fn)
            bg_all = _bg(out)
        if report is not None:
            for _ in range(nr):
                report.observe(ms)
        bg = bg_all[:H]
        true = cgm[ts:ts + H].astype(float)
        m = min(len(bg), len(true))
        d = bg[:m] - true[:m]
        acc['se'][:m] += d * d
        acc['n'][:m] += 1
        acc['count'] += 1




def _tile_band(curve: np.ndarray, sig: np.ndarray, horizon_patches: int):
    """Tile a per-horizon σ across the (tiled) day curve → (lo1, hi1, lo2, hi2)."""
    hsteps = horizon_patches * PATCH_SIZE
    T = len(curve)
    sig_t = np.tile(np.asarray(sig)[:hsteps], T // hsteps + 1)[:T]
    return curve - sig_t, curve + sig_t, curve - 2 * sig_t, curve + 2 * sig_t


def day_curves(model, stats, seg, day_start: int, sig2=None, sig8=None) -> dict:
    """24 h true CGM + tiled 2 h and 8 h conditioned forecasts.

    Every forecast origin (each tile start ts >= day_start) is fed the full
    MAX_CONTEXT_PATCHES = 24 h of REAL history via context_window — far beyond the
    8 h (MIN_CONTEXT_PATCHES) floor; context_window requires a full window, so a
    short context would raise rather than silently truncate. Every window
    announces its true future carbs/insulin/exercise (the model is always
    conditioned).
    """
    assert day_start >= CTX, (
        f"origin needs MAX_CONTEXT_PATCHES of context behind it; "
        f"day_start={day_start} < {CTX}")
    feats = build_feature_stack(seg, stats)
    T = DAY_PATCHES * PATCH_SIZE
    # The plotted "true CGM" the forecast is scored against is the raw (bg-clamped)
    # CGM (one space; never the future). Logged-event overlays below stay raw.
    true = smooth_bg_truth(seg.cgm)[day_start:day_start + T]
    hod = seg.hour_of_day()
    hod0 = float(hod[day_start])
    hours = hod0 + np.arange(T) * (GRID_MIN / 60.0)

    # Model time-of-day probe read-out per 2 h tile origin, collected inside the
    # PREDICTION_PATCHES pass only (the 8 h pass leaves it untouched). Populated
    # only when the probe is enabled; NaN otherwise, so plot_day can skip cleanly.
    tod: dict[str, list] = {'pred_hour': [], 'true_hour': [], 'R': [], 'bin_probs': []}

    ms = PR.forecast_masked_set(MAX_CONTEXT_PATCHES)

    def tile(horizon_patches):
        hsteps = horizon_patches * PATCH_SIZE
        nr = horizon_patches // PREDICTION_PATCHES
        collect_tod = horizon_patches == PREDICTION_PATCHES
        curve = np.full(T, np.nan)
        for c in range(DAY_PATCHES // horizon_patches):
            ts = day_start + c * hsteps
            ctx = context_window(feats, ts, MAX_CONTEXT_PATCHES)
            if collect_tod:
                # collect_tod ==> horizon_patches == PREDICTION_PATCHES ==> nr == 1
                # (single pass): the same forward yields BOTH the forecast bg and the
                # time-of-day bin logits, so read the probe off THIS forward (no
                # separate predict_origin_hour pass).
                ov = _future_overrides(feats, ts, ANNOUNCE)
                out = predict(model, ctx, normalization_stats=stats, overrides=ov,
                              return_time=True, mask_spans=ms.spans)
                tp = out.get('time_pred')
                tod['bin_probs'].append(None if tp is None else torch.softmax(tp, -1).cpu().numpy())
                if tp is None:
                    pred_hour, R = float('nan'), float('nan')
                else:
                    h_t, r_t = time_of_day_decode_bins(tp[0:1, :], TIME_PROBE_N_BINS)
                    pred_hour, R = float(h_t.item()), float(r_t.item())
                tod['pred_hour'].append(pred_hour)
                tod['true_hour'].append(float(hod[ts]))
                tod['R'].append(R)
            elif nr == 1:
                ov = _future_overrides(feats, ts, ANNOUNCE)
                out = predict(model, ctx, normalization_stats=stats, overrides=ov,
                              mask_spans=ms.spans)
            else:
                fn = _make_night_overrides_fn(feats, ts, ANNOUNCE, stats)
                out = predict_rolling(model, ctx, n_rolls=nr, normalization_stats=stats,
                                      overrides_fn=fn)
            bg = _bg(out) if nr > 1 else _protocol_bg(out, ms)
            curve[c * hsteps:c * hsteps + len(bg)] = bg[:hsteps]
        return curve

    # raw logged events inside the day (grams / IU per step — discrete spikes, NOT
    # the convolved absorption curves the model ingests). basal_day is the
    # piecewise-constant basal RATE (IU/hour) per step; basal IU = rate × GRID/60.
    carb_day = np.asarray(seg.carb_grams[day_start:day_start + T], dtype=float)
    bol_day = np.asarray(seg.bolus_units[day_start:day_start + T], dtype=float)
    basal_day = np.asarray(seg.basal_rate[day_start:day_start + T], dtype=float)
    spec = {
        'patient': seg.patient, 'hours': hours, 'true': true,
        'p2': tile(PREDICTION_PATCHES),
        'p8': tile(H8_PATCHES),
        'tile2_h': [hod0 + c * PRED * (GRID_MIN / 60.0) for c in range(DAY_PATCHES // PREDICTION_PATCHES + 1)],
        'tile8_h': [hod0 + c * H8_PATCHES * PATCH_SIZE * (GRID_MIN / 60.0) for c in range(DAY_PATCHES // H8_PATCHES + 1)],
        'carb_steps': carb_day, 'bolus_steps': bol_day, 'basal_steps': basal_day,
        'carb_total': float(np.sum(carb_day)), 'bolus_total': float(np.sum(bol_day)),
        'basal_total': float(np.sum(basal_day)) * (GRID_MIN / 60.0),
    }
    # TOD probe read-out, one entry per 2 h tile (aligned with the tile2_h
    # boundaries). Emitted only when at least one origin returned a finite hour
    # (the probe is on); plot_day gates its render on this presence + finiteness.
    tod_pred = np.asarray(tod['pred_hour'], dtype=float)
    if tod_pred.size and np.isfinite(tod_pred).any():
        spec['tile2_pred_hour'] = tod_pred
        spec['tile2_true_hour'] = np.asarray(tod['true_hour'], dtype=float)
        spec['tile2_R'] = np.asarray(tod['R'], dtype=float)
        bp = tod['bin_probs']
        if bp and all(b is not None for b in bp):
            # (n_tiles, P, TIME_PROBE_N_BINS) per-2h-tile per-patch softmax beliefs
            spec['tile2_time_probs'] = np.stack(bp)
    _attach_bg_bands(spec, sig2, sig8)
    return spec


def _attach_bg_bands(spec: dict, sig2, sig8) -> None:
    """Add ±1σ/±2σ envelope keys for the conditioned BG curves to ``spec`` (in place)."""
    for key, sig, hp in (('p2', sig2, PREDICTION_PATCHES),
                         ('p8', sig8, H8_PATCHES)):
        if sig is None:
            continue
        lo1, hi1, lo2, hi2 = _tile_band(spec[key], sig, hp)
        spec[f'{key}_lo1'], spec[f'{key}_hi1'] = lo1, hi1
        spec[f'{key}_lo2'], spec[f'{key}_hi2'] = lo2, hi2


def _hhmm(x, _):
    """Format an hour-of-day-continuous x value (hod0 + elapsed) as clock HH:MM."""
    c = x % 24.0
    return f"{int(c):02d}:{int(round((c % 1) * 60)) % 60:02d}"


def plot_day(spec, pretty, path):
    """Two-panel 48 h BG figure (2 h and 8 h tiled forecasts).

    The dynamics-channel panels (carbs/insulin/IS/HGO) were removed with the
    risk-space redesign — those are no longer model outputs.
    """
    fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True)
    h = spec['hours']
    carb, bol = spec['carb_steps'], spec['bolus_steps']
    basal = spec.get('basal_steps')                       # IU/hour per step (real only)
    ci, bi = np.where(carb > 0)[0], np.where(bol > 0)[0]
    ev_max = max(120.0, float(carb.max()) if carb.size else 0.0)

    # --- rows 0,1: BG forecasts (2 h, 8 h) with logged-event overlay ---
    for ax, (tag, ck, tiles, hp) in zip(axes[:2], [
        ('2 h prediction windows (BG)', 'p2', 'tile2_h', PREDICTION_PATCHES),
        ('8 h prediction windows (BG)', 'p8', 'tile8_h', H8_PATCHES)]):
        ax.axhspan(BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD, color='#e8f5e9', zorder=0)
        ax.axhline(BG_HYPO_THRESHOLD, color='#c0392b', lw=0.7, ls=':')
        ax.axhline(BG_HYPER_THRESHOLD, color='#e67e22', lw=0.7, ls=':')
        for tb in spec[tiles]:
            ax.axvline(tb, color='#bbbbbb', lw=0.5, zorder=1)
        # empirical per-horizon error envelope on the conditioned forecast
        # (derived — the head has no native BG σ; see _sigma_accumulate)
        if f'{ck}_lo1' in spec:
            ax.fill_between(h, spec[f'{ck}_lo2'], spec[f'{ck}_hi2'], color='#d62728',
                            alpha=0.10, zorder=2, label='±2σ (emp.)')
            ax.fill_between(h, spec[f'{ck}_lo1'], spec[f'{ck}_hi1'], color='#d62728',
                            alpha=0.22, zorder=3, label='±1σ (emp.)')
        ax.plot(h, spec['true'], color='black', lw=2.0, label='true CGM', zorder=5)
        ax.plot(h, spec[ck], color='#d62728', lw=1.4, label='forecast (announced)', zorder=4)
        # Time-of-day probe read-out (2 h panel only): the model's predicted origin
        # clock per tile, coloured by confidence R (true clock is the x-axis). Only
        # present when TIME_PROBE_ENABLED — the spec keys are absent otherwise.
        if ck == 'p2' and 'tile2_pred_hour' in spec:
            hsteps_tod = hp * PATCH_SIZE
            for c, (ph_c, R_c) in enumerate(zip(spec['tile2_pred_hour'], spec['tile2_R'])):
                if not np.isfinite(ph_c):
                    continue
                xc = h[c * hsteps_tod + hsteps_tod // 2]
                col = plt.cm.viridis(min(max(float(R_c), 0.0), 1.0))
                ax.text(xc, 350.0, f"{_hhmm(ph_c, None)} (R{R_c:.1f})", ha='center', va='top',
                        fontsize=5, color=col, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6))
        # Native per-patch time-of-day clocks (2 h panel only): one clock per MASKED
        # patch, no rotation, in a micro-strip across each tile's top band (data
        # coords; ylim is 40-360). Dense (n_tiles*P clocks) so kept tiny. The row
        # count comes from the probe output, not from a fixed zone length, so a
        # masked set of another size draws its own clocks rather than 2 h of them.
        if ck == 'p2' and spec.get('tile2_time_probs') is not None:
            tp_all = spec['tile2_time_probs']       # (n_tiles, P, TIME_PROBE_N_BINS)
            th = spec['tile2_h']
            for c in range(len(tp_all)):
                x_lo, x_hi = th[c], th[c + 1]
                n_masked = len(tp_all[c])
                w = (x_hi - x_lo) / n_masked
                for p in range(n_masked):
                    cax = ax.inset_axes([x_lo + p * w, 305.0, w, 48.0],
                                        transform=ax.transData)
                    draw_clock_axis(cax, tp_all[c][p], show_hand=True)
        title = tag + ('  ·  grey nums = basal IU/h per window' if basal is not None else '')
        if ck == 'p2' and 'tile2_pred_hour' in spec:
            title += '  ·  coloured HH:MM (R) = model clock probe'
        ax.set_title(title, fontsize=10, loc='left', pad=8)
        ax.set_ylabel('BG (mg/dL)'); ax.set_ylim(40, 360)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        # per-window basal level (mean IU/h over each prediction window), as small
        # grey numbers along the bottom of the panel (logged-event sources only).
        if basal is not None:
            hsteps = hp * PATCH_SIZE
            for c in range(len(h) // hsteps):
                seg_b = basal[c * hsteps:(c + 1) * hsteps]
                if seg_b.size:
                    ax.text(h[c * hsteps + hsteps // 2], 43.5, f"{float(np.mean(seg_b)):.1f}",
                            ha='center', va='bottom', fontsize=5.5, color='#777777', zorder=3)
        # logged-event overlay; skipped when no discrete events are
        # supplied (e.g. simulator runs, where the full carb/insulin truth is in
        # the channel panels instead). Each event carries its dose UNDER the marker:
        # carb grams under the carb point, bolus IU under the bolus point.
        if ci.size or bi.size:
            ax2 = ax.twinx()
            # lower bound is pushed negative so short bolus lollipops (U is tiny on
            # a carb-g scale) lift off the bottom edge, leaving room for the dose
            # label UNDER each marker without colliding with the clock tick labels.
            ax2.set_ylim(-0.28 * ev_max, ev_max * 1.18)
            ax2.set_ylabel('carb g / bolus U', fontsize=8)
            ax2.axhline(0, color='#cccccc', lw=0.5, zorder=1)
            bbox = dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.55)
            if ci.size:
                ax2.vlines(h[ci], 0, carb[ci], color='#2e7d32', lw=1.3, alpha=0.75, zorder=2)
                ax2.scatter(h[ci], carb[ci], marker='^', s=18, color='#2e7d32', zorder=3, label='carb (g)')
                for x_i, g in zip(h[ci], carb[ci]):
                    ax2.annotate(f"{g:.0f}g", (x_i, g), textcoords='offset points', xytext=(0, -3),
                                 ha='center', va='top', fontsize=6, color='#1b5e20', zorder=5, bbox=bbox)
            if bi.size:
                ax2.vlines(h[bi], 0, bol[bi], color='#6a1b9a', lw=1.3, alpha=0.75, zorder=2)
                ax2.scatter(h[bi], bol[bi], marker='v', s=18, color='#6a1b9a', zorder=3, label='bolus (U)')
                for x_i, u in zip(h[bi], bol[bi]):
                    ax2.annotate(f"{u:.1f}U", (x_i, u), textcoords='offset points', xytext=(0, -3),
                                 ha='center', va='top', fontsize=6, color='#4a148c', zorder=5, bbox=bbox)
            ax2.legend(loc='upper left', fontsize=7)

    # The predicted-dynamics-channel panels (carbs/insulin/IS/HGO μ ± σ vs GT)
    # were removed: those are no longer model outputs under the risk-space design.
    # HH:MM clock x-axis, hour-by-hour (major every 1 h, minor every 30 min) +
    # midnight markers. labelbottom=True on EVERY panel (not just the bottom one)
    # so each window reads its own clock — `sharex` would otherwise hide all but
    # the last; the 48 hourly labels are rotated to stay legible.
    first_mid = (int(h[0] // 24) + 1) * 24
    mids = [m for m in (first_mid, first_mid + 24, first_mid + 48) if h[0] <= m <= h[-1]]
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(FuncFormatter(_hhmm))
        ax.tick_params(axis='x', labelbottom=True, labelsize=5.5, rotation=90)
        for m in mids:
            ax.axvline(m, color='#777777', lw=0.9, alpha=0.55, zorder=1)
    axes[-1].set_xlabel('time of day (HH:MM)  ·  grey lines = midnight')

    # Title ABOVE all content: reserve top margin, then draw a bold heading + a
    # smaller caption clear of the first panel (the old suptitle collided with it).
    bt = spec.get('basal_total')
    dose = (f"logged 48 h: {spec['carb_total']:.0f} g carb · {spec['bolus_total']:.1f} U bolus"
            + (f" · {bt:.1f} U basal" if bt is not None else ""))
    if bt is None:                                        # simulator: combined insulin only
        dose = f"48 h totals: {spec['carb_total']:.0f} g carb · {spec['bolus_total']:.1f} U insulin"
    ctx_h = MAX_CONTEXT_PATCHES * PATCH_SIZE * GRID_MIN / 60.0
    caption = ("Headline median_bg forecast (future carbs/insulin/exercise announced); each origin fed "
               f"{ctx_h:g} h context; BG band = empirical per-horizon error envelope  ·  {dose}")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.text(0.5, 0.992, f"{pretty} · {spec['patient']} · 48 h", ha='center', va='top',
             fontsize=14, fontweight='bold')
    fig.text(0.5, 0.967, caption, ha='center', va='top', fontsize=8.5)
    fig.savefig(path, dpi=330)
    plt.close(fig)










