"""48 h day-figure machinery: tiled 2 h / 8 h forecast curves and ``plot_day``.
Loader, sigma accumulation, per-day assembly, plotting; ``curves_sim.py`` drives it.
Forecast protocol only, from ``metrics.protocols``, never derived from position.
Runs on checkpoints/t1dmai_best.pt, GPU if available.
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
    """RAW BG, bg-clamped — the truth every metric and figure scores against.

    No smoothing despite the name: scored truth is the raw CGM that produced the model input.
    """
    return np.clip(np.asarray(bg_raw, dtype=np.float64),
                   BG_CLAMP_MIN, BG_CLAMP_MAX).astype(float)

# ``PR.forecast_masked_set(n_ctx)`` names the span, head slots, per-slot d, predict's output rows.
PRED = PR.SPAN_STEPS                              # steps in one masked forecast span
DAY_PATCHES = 48 * _PATCHES_PER_HOUR              # 48 h day window (fixed figure span)
H8_PATCHES = NIGHT_LONG_HORIZON_PATCHES           # NIGHT_LONG_HORIZON_HOURS long-horizon window
CTX = MAX_CONTEXT_PATCHES * PATCH_SIZE
ANNOUNCE = (0, 1, 2)                              # carb, insulin, exercise

# A short set leaves the slot at normalize(0) — exercise_equiv's −0.139 z, an untrained regime.
assert ANNOUNCE == tuple(CHANNEL_TO_FEAT), (
    f"announced set {ANNOUNCE} != announceable set {tuple(CHANNEL_TO_FEAT)}")

# output channel → input feature: carb 0→1, insulin 1→2, exercise 2→3, off ``CHANNEL_TO_FEAT``
_ANNOUNCE_FEAT_IDX = {ch: CHANNEL_TO_FEAT[ch] for ch in ANNOUNCE}

CKPT = os.path.join(ROOT, 'checkpoints', 't1dmai_best.pt')


def load_model(device, path: str = CKPT):
    """Load a checkpoint, build model to match the WEIGHTS it carries -> (model, stats, step)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    m = T1DMAI().to(device)
    sd = ckpt['model_state_dict']
    ema = ckpt.get('model_ema_state_dict')
    m.load_state_dict({k: ema.get(k, v) for k, v in sd.items()} if ema else sd, strict=True)
    m.eval()
    return m, ckpt.get('normalization_stats') or load_normalization_stats(), ckpt.get('step')


def _slice_seg(seg, a: int, b: int):
    """A sub-Segment over steps ``[a, b)``, ``t0`` advanced to match.

    Every length-N array must be named (curves included): ``replace`` copies an unnamed field
    through at FULL length, misaligned against the sliced CGM.
    """
    return replace(
        seg, t0=seg.t0 + timedelta(minutes=GRID_MIN * a),
        cgm=seg.cgm[a:b], carb_grams=seg.carb_grams[a:b], bolus_units=seg.bolus_units[a:b],
        basal_rate=seg.basal_rate[a:b], exercise=seg.exercise[a:b],
        carb_curve=(None if seg.carb_curve is None else seg.carb_curve[a:b]),
        insulin_curve=(None if seg.insulin_curve is None else seg.insulin_curve[a:b]))




def _pick_day_start(seg) -> int | None:
    """Day start, patch-aligned, nearest midnight; DAY_PATCHES ahead, MAX_CONTEXT_PATCHES behind.

    ``None`` when the segment is too short.
    """
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

    Roll r masks the right-edge span advanced by one horizon, sliced normalized from ``feats``. Raw
    half unused, returned zeroed; None once a roll runs past segment. Past roll 0, left neighbour
    is the previous roll's own output — d = 1..PREDICTION_PATCHES again, but FABRICATED."""
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
    """Headline BG of a ROLLED pass, flat mg/dL.

    ``predict_rolling`` concatenates one span per roll, rolls contiguous, so this is step order.
    """
    pb = out['pred_bg'] if 'pred_bg' in out else out['median_bg']
    pb = pb.detach().cpu().numpy() if hasattr(pb, 'detach') else np.asarray(pb)
    return pb.flatten()


def _protocol_bg(out, ms) -> np.ndarray:
    """Headline BG of a SINGLE pass, ordered by the protocol's scored steps.

    ``predict`` returns one row per masked patch in head-slot order — a trailing zone since this
    protocol's masked set is one. ``ms.scored_rows()`` / ``ms.scored_steps()`` read the right array
    for any masked set."""
    med = out['median_bg'].detach().cpu().numpy().reshape(-1, PATCH_SIZE)
    vals = med[ms.scored_rows()].ravel()
    return vals[np.argsort(ms.scored_steps())]


# Band is EMPIRICAL per-horizon error envelope, not the fan; ±1σ half-width = RMSE √E[(pred−true)²].

# Horizon step h is patch ceil(h/PATCH_SIZE) = d, one-sided; d resets to 1 each roll.

def _sigma_accumulate(model, stats, feats, cgm, horizon_patches, stride_steps, acc,
                      max_windows, report=None):
    """Per-horizon squared error of the headline forecast, one trajectory, into ``acc``.

    Always conditioned. ``horizon_patches`` covers ``nr`` rolls of the forecast masked set; envelope
    bins on d WITHIN each roll, never on span length. ``report`` records each window's realised d.
    """
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
    """Tile a per-horizon σ across the day curve -> ``(lo1, hi1, lo2, hi2)``."""
    hsteps = horizon_patches * PATCH_SIZE
    T = len(curve)
    sig_t = np.tile(np.asarray(sig)[:hsteps], T // hsteps + 1)[:T]
    return curve - sig_t, curve + sig_t, curve - 2 * sig_t, curve + 2 * sig_t


def day_curves(model, stats, seg, day_start: int, sig2=None, sig8=None) -> dict:
    """True CGM plus tiled 2 h and 8 h conditioned forecasts.

    Every origin is fed the full ``MAX_CONTEXT_PATCHES`` of REAL history, well past the
    ``MIN_CONTEXT_PATCHES`` floor; ``context_window`` raises on a short context rather than
    truncating quietly. Always conditioned on the true future doses."""
    assert day_start >= CTX, (
        f"origin needs MAX_CONTEXT_PATCHES of context behind it; "
        f"day_start={day_start} < {CTX}")
    feats = build_feature_stack(seg, stats)
    T = DAY_PATCHES * PATCH_SIZE
    # the plotted truth is the raw bg-clamped CGM; the event overlays below stay raw
    true = smooth_bg_truth(seg.cgm)[day_start:day_start + T]
    hod = seg.hour_of_day()
    hod0 = float(hod[day_start])
    hours = hod0 + np.arange(T) * (GRID_MIN / 60.0)

    # probe read-out per 2 h tile origin, PREDICTION_PATCHES pass only; NaN with the probe off
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
                # collect_tod ⇒ nr == 1, so this one forward carries both the bg and the bin logits
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

    # raw logged events, discrete spikes, NOT the model's curves; basal_day is RATE in IU/hour.
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
    # One per 2 h tile, emitted only if some origin returned a finite hour; gates plot_day's render.
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
    """Add ±1σ / ±2σ envelope keys for the conditioned BG curves to ``spec``, in place."""
    for key, sig, hp in (('p2', sig2, PREDICTION_PATCHES),
                         ('p8', sig8, H8_PATCHES)):
        if sig is None:
            continue
        lo1, hi1, lo2, hi2 = _tile_band(spec[key], sig, hp)
        spec[f'{key}_lo1'], spec[f'{key}_hi1'] = lo1, hi1
        spec[f'{key}_lo2'], spec[f'{key}_hi2'] = lo2, hi2


def _hhmm(x, _):
    """A continuous hour-of-day x value (hod0 + elapsed) as HH:MM."""
    c = x % 24.0
    return f"{int(c):02d}:{int(round((c % 1) * 60)) % 60:02d}"


def plot_day(spec, pretty, path):
    """Two-panel 48 h BG figure: 2 h and 8 h tiled forecasts."""
    fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True)
    h = spec['hours']
    carb, bol = spec['carb_steps'], spec['bolus_steps']
    basal = spec.get('basal_steps')                       # IU/hour per step (real only)
    ci, bi = np.where(carb > 0)[0], np.where(bol > 0)[0]
    ev_max = max(120.0, float(carb.max()) if carb.size else 0.0)

    # rows 0,1: BG forecasts (2 h, 8 h) with the logged-event overlay
    for ax, (tag, ck, tiles, hp) in zip(axes[:2], [
        ('2 h prediction windows (BG)', 'p2', 'tile2_h', PREDICTION_PATCHES),
        ('8 h prediction windows (BG)', 'p8', 'tile8_h', H8_PATCHES)]):
        ax.axhspan(BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD, color='#e8f5e9', zorder=0)
        ax.axhline(BG_HYPO_THRESHOLD, color='#c0392b', lw=0.7, ls=':')
        ax.axhline(BG_HYPER_THRESHOLD, color='#e67e22', lw=0.7, ls=':')
        for tb in spec[tiles]:
            ax.axvline(tb, color='#bbbbbb', lw=0.5, zorder=1)
        # empirical per-horizon error envelope, derived; see _sigma_accumulate
        if f'{ck}_lo1' in spec:
            ax.fill_between(h, spec[f'{ck}_lo2'], spec[f'{ck}_hi2'], color='#d62728',
                            alpha=0.10, zorder=2, label='±2σ (emp.)')
            ax.fill_between(h, spec[f'{ck}_lo1'], spec[f'{ck}_hi1'], color='#d62728',
                            alpha=0.22, zorder=3, label='±1σ (emp.)')
        ax.plot(h, spec['true'], color='black', lw=2.0, label='true CGM', zorder=5)
        ax.plot(h, spec[ck], color='#d62728', lw=1.4, label='forecast (announced)', zorder=4)
        # 2 h panel only: predicted origin clock per tile, by confidence R; absent with probe off.
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
        # One clock per MASKED patch in a strip across tile top (ylim 40-360); count from the probe.
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
        # mean IU/h per prediction window, grey numbers along the bottom; logged-event sources only
        if basal is not None:
            hsteps = hp * PATCH_SIZE
            for c in range(len(h) // hsteps):
                seg_b = basal[c * hsteps:(c + 1) * hsteps]
                if seg_b.size:
                    ax.text(h[c * hsteps + hsteps // 2], 43.5, f"{float(np.mean(seg_b)):.1f}",
                            ha='center', va='bottom', fontsize=5.5, color='#777777', zorder=3)
        # Skipped when a source has no discrete events; each dose labelled under its marker.
        if ci.size or bi.size:
            ax2 = ax.twinx()
            # Negative lower bound lifts bolus lollipops off the edge, room for label underneath.
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

    # labelbottom=True on EVERY panel: `sharex` would otherwise hide all but the last.
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

    # Title ABOVE all content: reserves top margin, heading and caption clear of the first panel.
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










