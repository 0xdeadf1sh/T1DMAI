"""
Report assembly for the in-domain T1DMSIM evaluation: the metric README/JSON
renderer and the actual-vs-predicted figure driver.

``metrics/sim/`` is the entry point — fresh simulator patients scored with each
window's future carbohydrate, insulin and exercise announced to the model, which
is the deployment what-if regime. This is the model's training distribution, so
the numbers are an in-domain reference rather than a peer comparison.

``load_model`` builds ``T1DMAI()`` from the live ``config.py``, loads the state
dict and attaches ``conformal_delta`` off the checkpoint for its callers to
forward. It is not the only loader: ``metrics/day_curves.py`` has its own, which
builds the model to match the WEIGHTS the checkpoint carries rather than the live
config, and that is the one the day figures use.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from config import (
    MAX_CONTEXT_PATCHES, PATCH_SIZE, PREDICTION_PATCHES,
    PREDICTION_HORIZON_HOURS, NIGHT_LONG_HORIZON_HOURS,
    NOCTURNAL_START_HOUR, NOCTURNAL_END_HOUR,
    BG_HYPO_THRESHOLD, BG_HYPER_THRESHOLD, TIME_PROBE_ENABLED, TIME_PROBE_N_BINS,
    METRIC_BAND_TAU_LO, METRIC_BAND_TAU_HI,
    HYPO_ALARM_QUANTILE_TAU, HYPER_ALARM_QUANTILE_TAU,
)
from model import T1DMAI
from normalization import load_normalization_stats
from inference import predict, predict_rolling
from utils import time_of_day_decode_bins
from .run_eval import _make_night_overrides_fn
from .calibrate import _future_overrides, CTX_STEPS, PRED_STEPS
from .features import build_feature_stack, context_window, smoothed_cgm
from .suite import HORIZONS

# Per-horizon excursion display targets for the SOTA-target markdown table. The
# risk-space redesign removed ``config.EXCURSION_TARGET_*`` (this package was out of
# scope for that config cycle); these mirror the prior tuples
# ``(base@30min, slope_per_30min, floor)`` and are DISPLAY-ONLY (row colour /
# Target text), never the loss or any selection.
EXCURSION_TARGET_HYPO_RECALL = (0.70, 0.10, 0.30)
EXCURSION_TARGET_HYPO_PRECISION = (0.70, 0.10, 0.30)
EXCURSION_TARGET_HYPER_RECALL = (0.80, 0.10, 0.40)
EXCURSION_TARGET_HYPER_PRECISION = (0.80, 0.10, 0.40)

# Repository root is three levels up: metrics/core/report.py. Two levels reaches
# metrics/, which is where this landed silently when the package moved here.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(_ROOT, 'checkpoints', 't1dmai_best.pt')



# --------------------------------------------------------------------------- #
# Model + evaluation drivers.
# --------------------------------------------------------------------------- #
def load_model(device, path: str = CKPT):
    """Load a checkpoint into a freshly constructed T1DMAI model.

    The EMA shadow is merged over the live weights below.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    m = T1DMAI().to(device)
    sd = ckpt['model_state_dict']
    ema = ckpt.get('model_ema_state_dict')
    merged = {k: ema.get(k, v) for k, v in sd.items()} if ema else dict(sd)
    # The diagnostic-only time-of-day probe changed geometry (per-patch categorical
    # bins); a checkpoint from an earlier probe shape must still load its FORECAST
    # weights exactly. Drop only mismatched ``time_head`` keys (re-inited under the
    # RNG-neutral construction) — a shape mismatch on any other tensor is fatal.
    model_sd = m.state_dict()
    dropped = [k for k, v in merged.items() if k in model_sd and model_sd[k].shape != v.shape]
    assert all(k.startswith('time_head.') for k in dropped), \
        f"shape mismatch on non-probe weights: {[k for k in dropped if not k.startswith('time_head.')]}"
    for k in dropped:
        del merged[k]
    res = m.load_state_dict(merged, strict=False)
    leftover = [k for k in list(res.missing_keys) + list(res.unexpected_keys)
                if not k.startswith('time_head.')]
    assert not leftover, f"unexpected state_dict mismatch (non-probe): {leftover}"
    m.eval()
    # Attach the stored conformal correction as an attribute so callers can pass it to
    # ``predict(..., conformal_delta=...)`` without changing this return tuple (~10 call
    # sites). ``None`` when the checkpoint was never calibrated. NOTE: the stored delta
    # is SIM-fit (calibrate_conformal.py) — valid for the simulator path only; real
    # a different distribution must RE-FIT its own delta (run_eval), since conformal validity needs
    # calibration/test exchangeability.
    cd = ckpt.get('conformal_delta')
    if cd is None:
        m.conformal_delta = None
    else:
        # The delta is stored as a torch tensor (weights_only-safe) and loaded onto
        # ``device`` (possibly CUDA); bring it back to a host numpy array, which is what
        # conformal.apply / predict expect. Tolerate a legacy numpy delta too.
        m.conformal_delta = cd.detach().cpu().numpy() if torch.is_tensor(cd) else np.asarray(cd)
    return m, ckpt.get('normalization_stats') or load_normalization_stats(), ckpt.get('step')




# --------------------------------------------------------------------------- #
# README rendering (neutral, public).
# --------------------------------------------------------------------------- #
def _fmt(x, nd=1):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _has_bands(res: dict) -> bool:
    """True when every horizon of ``res['metrics']`` carries the band-scoring keys
    (``band_cov50`` / ``median_line``), i.e. the suite was computed against the
    quantile band. False for a band-less run or a stats JSON predating it."""
    m = res.get('metrics', {})
    return all(isinstance(m.get(str(h)), dict) and 'band_cov50' in m[str(h)]
               and isinstance(m[str(h)].get('median_line'), dict) for h in HORIZONS)


def _suite_table(res: dict) -> str:
    m = res['metrics']
    band = _has_bands(res)
    cols = ["Horizon", "RMSE pt", "RMSE wm", "MAE pt", "MAE wm", "MARD %", "Clarke A %",
            "Clarke A+B %", "Clarke E %", "hypo rec", "hypo prec", "hyper rec", "hyper prec"]
    if band:
        cols += ["band cov50 %", "band width"]
    rows = ["| " + " | ".join(cols) + " |",
            "|---|" + "".join("--:|" for _ in cols[1:])]
    for h in HORIZONS:
        d = m[str(h)]; hy, yp = d['hypo'], d['hyper']
        row = (
            f"| {h} min | {d['rmse_point']:.1f} | {d['rmse_winmean']:.1f} | {d['mae_point']:.1f} | "
            f"{d['mae_winmean']:.1f} | {d['mard']:.1f} | {d['clarke_A']:.1f} | {d['clarke_AB']:.1f} | "
            f"{d['clarke_E']:.2f} | {_fmt(hy['recall'],2)} | {_fmt(hy['precision'],2)} | "
            f"{_fmt(yp['recall'],2)} | {_fmt(yp['precision'],2)} |")
        if band:
            cov = d.get('band_cov50')
            row += (f" {_fmt(100 * cov, 1) if isinstance(cov, (int, float)) else '—'} | "
                    f"{_fmt(d.get('band_width'), 1)} |")
        rows.append(row)
    note = "  ".join(
        f"@{h}m hypo {m[str(h)]['hypo']['n_true']} true / {m[str(h)]['hypo']['n_pred']} pred, "
        f"hyper {m[str(h)]['hyper']['n_true']} true / {m[str(h)]['hyper']['n_pred']} pred"
        for h in HORIZONS)
    return "\n".join(rows) + f"\n\nEvent counts (recall/precision denominators): {note}"


def _selected_offset_section(res: dict) -> str:
    so = res.get('selected_offsets')
    if not so or not so.get('hypo'):
        return ""
    floor = so.get('min_precision')
    series = (f"the τ={METRIC_BAND_TAU_LO:.2f} lower band edge" if _has_bands(res)
              else "the median forecast")
    caption = (
        f"Per-horizon hypo decision offset, selected on the calibration split under a "
        f"precision floor of {floor:.2f}: the highest-recall offset whose calibration "
        f"precision clears the floor (the strict crossing where none does). The alarm "
        f"fires when {series} falls below {BG_HYPO_THRESHOLD:.0f} + offset mg/dL. Recall and precision "
        f"below are measured on the disjoint test split at the selected offset.")
    rows = ["| Horizon | offset (mg/dL) | cal recall | cal prec | test recall | test prec |",
            "|---|--:|--:|--:|--:|--:|"]
    for h in HORIZONS:
        d = so['hypo'][str(h)]
        rows.append(
            f"| {h} min | {d['offset']:.1f} | {_fmt(d['cal_recall'],2)} | "
            f"{_fmt(d['cal_precision'],2)} | {_fmt(d['test_recall'],2)} | "
            f"{_fmt(d['test_precision'],2)} |")
    return f"{caption}\n\n" + "\n".join(rows)


def _night_onset_section(res: dict) -> str:
    """Per-night nocturnal-excursion recall/precision (bedtime → night end) for the
    conditioned forecast (announced overnight carbs+insulin). Returns "" when the
    metric is absent (e.g. the night fits in a single forward pass, or no
    night-onset window was available)."""
    no = res.get('night_onset')
    if not no or not no.get('n_nights'):
        return ""

    def cell(side_d, k):
        return _fmt(side_d.get(k), 2) if side_d else "—"

    caption = (
        f"Night-onset nocturnal-excursion prediction over {no['n_nights']} nights. At each "
        f"bedtime origin (hour-of-day near {NOCTURNAL_START_HOUR:.0f}:00) the forecast is "
        f"autoregressively rolled to night end (hour {NOCTURNAL_END_HOUR:.0f}:00); a night is "
        f"a true hypo/hyper if the true CGM crosses {BG_HYPO_THRESHOLD:.0f} / {BG_HYPER_THRESHOLD:.0f} mg/dL anywhere in the night, and "
        f"flagged likewise from the rolled forecast. Recall is the fraction of true-excursion "
        f"nights flagged; precision the fraction of flagged nights that truly had one. The "
        f"forecast is conditioned on the night's logged overnight carbohydrate and insulin, "
        f"fed to each roll.")
    rows = ["| hypo recall | hypo prec | hypo nights (true/pred) | "
            "hyper recall | hyper prec | hyper nights (true/pred) |",
            "|--:|--:|--:|--:|--:|--:|"]
    hy, yp = no.get('hypo', {}), no.get('hyper', {})
    rows.append(
        f"| {cell(hy,'recall')} | {cell(hy,'precision')} | "
        f"{hy.get('n_true','—')}/{hy.get('n_pred','—')} | "
        f"{cell(yp,'recall')} | {cell(yp,'precision')} | "
        f"{yp.get('n_true','—')}/{yp.get('n_pred','—')} |")
    return f"{caption}\n\n" + "\n".join(rows)


def _baseline_table(res: dict) -> str:
    m = res['metrics']; c = res['conformal']
    rows = ["| Horizon | persistence RMSE pt | persistence RMSE wm | RMSE skill vs persistence (pt) % | "
            "macro RMSE pt | conformal 90% half-width | conformal coverage |",
            "|---|--:|--:|--:|--:|--:|--:|"]
    for h in HORIZONS:
        d = m[str(h)]
        rows.append(
            f"| {h} min | {d['rmse_persist_point']:.1f} | {d['rmse_persist_winmean']:.1f} | "
            f"{100*d['skill_point']:+.1f} | {d['rmse_macro']:.1f} | ±{c[str(h)]['half_width']:.1f} | "
            f"{100*c[str(h)]['coverage']:.0f}% |")
    cqr = _conformal_cqr_table(res)
    return "\n".join(rows) + (("\n\n" + cqr) if cqr else "")


def _conformal_cqr_table(res: dict) -> str | None:
    """Calibrated-band-coverage table from the quantile-CQR re-fit (additive).

    Renders the raw-vs-calibrated central-90% band coverage and the τ=0.10
    hypo-edge escape rate per horizon. Guarded on ``res['conformal_cqr']`` so old
    result JSONs (and any path without bands) render the baseline table unchanged.
    Neutral observed numbers only — no remediation prose.
    """
    cq = res.get('conformal_cqr')
    if not cq:
        return None
    rows = ["| Horizon | band 90% coverage (raw) | band 90% coverage (calibrated) | "
            "τ=0.10 hypo-edge escape (raw) | τ=0.10 hypo-edge escape (calibrated) | n (cal/test) |",
            "|---|--:|--:|--:|--:|--:|"]
    for h in HORIZONS:
        d = cq.get(str(h))
        if d is None:
            continue
        rows.append(
            f"| {h} min | {100*d['raw_cov90']:.0f}% | {100*d['cal_cov90']:.0f}% | "
            f"{100*d['raw_hypo_escape']:.0f}% | {100*d['cal_hypo_escape']:.0f}% | "
            f"{d['n_cal']}/{d['n_test']} |")
    return ("Quantile-CQR calibrated band coverage (split-conformal, re-fit on this "
            "run's calibration split):\n\n" + "\n".join(rows))




# --------------------------------------------------------------------------- #
# SOTA-target validation table (markdown mirror of train.py's terminal table).
#
# The bars below MIRROR the values hard-coded in train.py's
# ``_render_validation_table`` — that table is the source of truth; keep these in
# sync by hand if it changes. The per-horizon excursion bars are NOT duplicated:
# they read the module-local ``EXCURSION_TARGET_*`` constants through ``_exc_target``
# (train.py's ``_excursion_target`` formula). Only the rows this report can score are
# emitted; the per-channel NLL/σ, event-detection, signal-attribution, TIR, and
# rate-of-change rows have no counterpart here and are omitted.
# --------------------------------------------------------------------------- #
_VT_BG_RMSE_SOTA = {30: 15.0, 60: 25.0, 120: 36.0}
_VT_MARD_SOTA = {30: 7.0, 60: 12.0, 120: 19.0}
_VT_CLARKE_A_SOTA = {30: 95.0, 60: 85.0, 120: 72.0}
_VT_BG_MAE_SOTA = {30: 9.7, 60: 16.5}
_VT_CGEGA_AP_SOTA = {'hypo': 80.0, 'eu': 90.0, 'hyper': 85.0}
_VT_CGEGA_EP_SOTA = {'hypo': 10.0, 'eu': 2.0, 'hyper': 5.0}
_VT_NIGHT_ONSET_SOTA = {'hypo': (70.0, 50.0), 'hyper': (70.0, 60.0)}  # (recall, precision)
_VT_EXC_TARGET = {
    ('hypo', 'recall'): EXCURSION_TARGET_HYPO_RECALL,
    ('hypo', 'precision'): EXCURSION_TARGET_HYPO_PRECISION,
    ('hyper', 'recall'): EXCURSION_TARGET_HYPER_RECALL,
    ('hyper', 'precision'): EXCURSION_TARGET_HYPER_PRECISION,
}

_VT_CAPTION = (
    "SOTA-target validation table — the markdown counterpart of the training-time "
    "table (`train.py`), over the subset of rows this report can score. The forecast is "
    "the announced-event quantile fan (the model's future carbohydrate/insulin are always "
    "announced). The RMSE, MAE, MARD, Clarke and CG-EGA rows are band-scored — the truth "
    f"projected onto the τ={METRIC_BAND_TAU_LO:.2f}–τ={METRIC_BAND_TAU_HI:.2f} band — whereas "
    "the SOTA bars they are tagged against are published point-forecast values, so the two "
    "columns rest on different bases. Every value is tagged against the project's SOTA bar — "
    "`pass` meets it, `near` is within the warn band, `miss` is beyond — the same "
    "horizon-dependent bars `train.py` uses. Rows with no real-CGM counterpart (per-channel "
    "uncertainty, event detection, signal attribution, time-in-range, rate-of-change) are omitted.")


def _exc_target(spec: tuple[float, float, float], h_min: int) -> float:
    """Per-horizon excursion bar (train.py's ``_excursion_target``):
    ``max(floor, base − slope·(h/30−1))`` over an ``EXCURSION_TARGET_*`` tuple."""
    base, slope, floor = spec
    return max(floor, base - slope * (h_min / 30.0 - 1.0))


def _vt_tag(val, sota: float, higher: bool, warn_edge: float) -> str:
    """pass / near / miss against a SOTA bar, replicating train.py's
    green/yellow/red tiers (green_edge = sota, red_edge = warn_edge)."""
    if val is None:
        return ''
    if higher:
        return 'pass' if val >= sota else ('near' if val >= warn_edge else 'miss')
    return 'pass' if val <= sota else ('near' if val <= warn_edge else 'miss')


def _vt_cell(val, sota: float, higher: bool, warn_edge: float, nd: int) -> str:
    if not isinstance(val, (int, float)):
        return '—'
    return f"{val:.{nd}f} {_vt_tag(val, sota, higher, warn_edge)}"


def _validation_table(res: dict) -> str:
    """Markdown SOTA-target validation table (Metric | Forecast | Target) for one
    source, mirroring train.py's terminal validation table over the rows this report
    can score. The Forecast column reads the announced-event ``res['metrics']`` /
    ``res['cgega']``. Night-onset rows read the per-night conditioned calls carried
    in ``res['night_onset']``."""
    cm = res.get('metrics', {})
    cg_c = res.get('cgega') or {}
    no = res.get('night_onset') or {}
    rows: list[tuple[str, str, str]] = []

    def section(title: str) -> None:
        rows.append((f"**{title}**", "", ""))

    def hget(m: dict, h: int, *path: str):
        d = m.get(str(h))
        for p in path:
            if not isinstance(d, dict):
                return None
            d = d.get(p)
        return d

    def row(label, val, sota, higher, nd, unit, warn_edge, scale=1.0) -> None:
        v = val * scale if isinstance(val, (int, float)) else None
        if v is None:
            return
        tgt = f"{'>' if higher else '<'} {sota:.{nd}f}{unit}"
        rows.append((label, _vt_cell(v, sota, higher, warn_edge, nd), tgt))

    section('BG Forecast (RMSE)')
    for h in (30, 60, 120):
        s = _VT_BG_RMSE_SOTA[h]
        row(f'bg_rmse @{h}m', hget(cm, h, 'rmse_point'), s, False, 1, ' mg/dL', s * 1.5)

    section('Relative Error (MARD)')
    for h in (30, 60, 120):
        s = _VT_MARD_SOTA[h]
        row(f'mard @{h}m', hget(cm, h, 'mard'), s, False, 2, '%', s * 2.0)

    section('Clinical Error Grid (Clarke)')
    for h in (30, 60, 120):
        s = _VT_CLARKE_A_SOTA[h]
        row(f'clarke_A @{h}m', hget(cm, h, 'clarke_A'), s, True, 2, '%', s - 2.0)
    for h in (30, 60, 120):
        row(f'clarke_A+B @{h}m', hget(cm, h, 'clarke_AB'), 98.0, True, 2, '%', 96.0)
    for h in (30, 60, 120):
        row(f'clarke_D @{h}m', hget(cm, h, 'clarke_D'), 0.5, False, 3, '%', 2.0)
    for h in (30, 60, 120):
        row(f'clarke_E @{h}m', hget(cm, h, 'clarke_E'), 0.0, False, 3, '%', 0.1)

    section('Clinical Accuracy (CG-EGA)')
    for reg in ('hypo', 'eu', 'hyper'):
        s = _VT_CGEGA_AP_SOTA[reg]
        row(f'cgega_AP @{reg}', cg_c.get(f'ap_{reg}'), s, True, 2, '%', s - 10.0)
    for reg in ('hypo', 'eu', 'hyper'):
        s = _VT_CGEGA_EP_SOTA[reg]
        row(f'cgega_EP @{reg}', cg_c.get(f'ep_{reg}'), s, False, 2, '%', s * 2.0)

    section('Excursions by Horizon')
    for side in ('hypo', 'hyper'):
        for metric in ('recall', 'precision'):
            spec = _VT_EXC_TARGET[(side, metric)]
            for h in (30, 60, 120):
                s = _exc_target(spec, h)
                row(f'{side}_{metric} @{h}m', hget(cm, h, side, metric),
                    s, True, 2, '%', s - 10.0, scale=100.0)

    section('Supplementary (MAE)')
    for h in (30, 60):
        s = _VT_BG_MAE_SOTA[h]
        row(f'bg_mae @{h}m', hget(cm, h, 'mae_point'), s, False, 1, ' mg/dL', s * 2.0)

    if no.get('n_nights'):
        section(f"Night-onset Excursion ({no['n_nights']} nights)")
        for side in ('hypo', 'hyper'):
            rec_s, prec_s = _VT_NIGHT_ONSET_SOTA[side]
            cd = no.get(side, {})
            row(f'night-onset {side} recall', cd.get('recall'),
                rec_s, True, 2, '%', rec_s - 15.0, scale=100.0)
            row(f'night-onset {side} precision', cd.get('precision'),
                prec_s, True, 2, '%', prec_s - 15.0, scale=100.0)

    head = ["| Metric | Forecast | Target |", "|---|--:|---|"]
    return "\n".join(head + [f"| {m} | {c} | {t} |" for (m, c, t) in rows])


def _validation_section(res: dict) -> str:
    return f"### Validation table (SOTA-target indexed)\n\n{_VT_CAPTION}\n\n{_validation_table(res)}"












def _band_series_label(median_curve: list[float] | None, suffix: str = "") -> str:
    """Legend label for the model's band-scored RMSE series: qualified as band-scored
    only when the median-line companion series is also drawn."""
    base = f"T1DMAI{suffix}"
    return f"{base} (band-scored)" if median_curve is not None else base


def _median_hour_curve(rbh: dict, hs: list[int], key: str) -> list[float] | None:
    """Median-line companion to an ``rmse_by_hour`` series (``rmse_point`` →
    ``rmse_point_median``), as a list over ``hs``. ``None`` when the stats JSON
    predates the band basis and carries no median series."""
    mk = f"{key}_median"
    if not all(isinstance(rbh.get(str(h)), dict) and mk in rbh[str(h)] for h in hs):
        return None
    return [rbh[str(h)][mk] for h in hs]


def _median_suite_curve(m: dict, hs: list[int], key: str) -> list[float] | None:
    """Median-line companion to a ``metrics`` suite series, read from
    ``metrics[h]['median_line'][key]`` over ``hs``. ``None`` when the suite was
    computed without bands and carries no median-line block."""
    if not all(isinstance(m.get(str(h)), dict)
               and isinstance(m[str(h)].get('median_line'), dict)
               and key in m[str(h)]['median_line'] for h in hs):
        return None
    return [m[str(h)]['median_line'][key] for h in hs]




# --------------------------------------------------------------------------- #
# In-domain simulator report.
# --------------------------------------------------------------------------- #
_SIM_METRIC_DEFS = f"""## Metric definitions

- **Scoring basis — band-projected.** The level metrics score the truth against the band
  between τ={METRIC_BAND_TAU_LO:.2f} and τ={METRIC_BAND_TAU_HI:.2f}: the scored forecast is
  the true value clipped to that band, so the reported error is the distance from the truth to
  the nearer band edge and is zero whenever the truth falls inside the band. RMSE, MAE, MARD,
  the Clarke grid and CG-EGA all read that projected series as the FORECAST; the truth stays
  the reference on every axis, so a CG-EGA point's glycaemic region and its rate-dependent
  acceptance widening are the truth's own throughout. Wherever the truth lies inside the band
  the projection equals it, so the scored series carries the truth's derivative there and the
  rate grid registers an error only where the band actually missed. The
  same metrics on the quantile median alone are kept in `stats.json` under
  `metrics[h]["median_line"]`.
- **RMSE — point (pt) / window-mean (wm):** error at the single PH-ahead step, and RMSE
  pooled over steps 0…PH. **MAE** likewise. **MARD:** |pred−true|/true at the horizon point.
- **Clarke A / A+B / E:** percentage of points in Clarke Error Grid zone A, zones A∪B, and
  zone E (Clarke et al. 1987), at the horizon point.
- **band cov50 %:** the realized fraction of true values inside the
  τ={METRIC_BAND_TAU_LO:.2f}–τ={METRIC_BAND_TAU_HI:.2f} band at the horizon point, as a percentage
  (nominal 50 %, the band being the model's inner-50 % interval). **band width:** the mean
  edge-to-edge width of that band at the horizon point, mg/dL — the scale the band-projected
  errors are relative to.
- **hypo / hyper recall, precision:** detection of true crossings below {BG_HYPO_THRESHOLD:.0f} / above {BG_HYPER_THRESHOLD:.0f} mg/dL
  at the horizon point, off the forecast band edges (the τ={HYPO_ALARM_QUANTILE_TAU:.2f} lower edge for hypo, the
  τ={HYPER_ALARM_QUANTILE_TAU:.2f} upper edge for hyper — their own knobs, independent of the scoring band above) with
  strict recall and a small precision tolerance (denominators under the table).
- **Excursion decision offset:** a per-horizon hypo decision offset δ added to the
  {BG_HYPO_THRESHOLD:.0f} mg/dL alarm level: the alarm fires when the τ={METRIC_BAND_TAU_LO:.2f} lower band edge
  (or the median line where no fan is available — the series named in the table caption) falls below
  {BG_HYPO_THRESHOLD:.0f} + δ, with δ selected on calibration under a precision floor.
  **Persistence:** the naive last-value (flat) forecast; it carries no band and is scored as a
  point forecast. **RMSE skill vs persistence:** (persistence − model) / persistence, %, so a
  band-scored numerator over a point baseline. **Conformal 90%:** split-conformal half-width
  fit on calibration, coverage measured on test, both splits on the band-projected series.
- **Night-onset excursion recall / precision:** a per-night binary call — from a bedtime
  origin the forecast is rolled across the whole night and flagged hypo/hyper if it crosses
  {BG_HYPO_THRESHOLD:.0f} / {BG_HYPER_THRESHOLD:.0f} mg/dL anywhere in the night, scored against whether the true CGM did. The roll is
  conditioned on the night's overnight carbohydrate/insulin."""


def render_sim_readme(R: dict) -> str:
    """Public README for the in-domain T1DMSIM report — numbers only."""
    s = R['sim']
    meta = R.get('_meta', {})
    step = meta.get('step')
    step_line = (f" The checkpoint evaluated here is at training step {step:,} (training was "
                 f"ongoing at generation time)." if step else "")
    return f"""# T1DMAI on the T1DMSIM simulator — in-domain reference

This report tabulates the T1DMAI model's blood-glucose-prediction metrics on fresh
patients drawn from the T1DMSIM behavioral simulator — the model's **training
distribution**, so it is an in-domain reference rather than a measure of
generalisation. It reports numbers only and draws no judgement of relative
quality.{step_line}

## Method

- **Patients.** {s['n_patients']} simulated patients at distinct seeds — each seed draws a
  distinct random patient — {meta.get('hours', '?')} h per patient after a
  {meta.get('warmup', '?')} h warmup discard. Calibration and test patients are disjoint
  seed pools.
- **Inputs.** The announced-event (what-if) regime: each window's future carbohydrate,
  insulin and exercise over the forecast horizon are given to the model. The model
  consumes only CGM, carbohydrate, insulin and the
  carbohydrate-equivalent exercise channel (time-of-day is inferred, not an input).
- **Forecast.** The model emits a risk-space quantile fan, inverted to mg/dL; the headline
  level metrics score the truth against the τ={METRIC_BAND_TAU_LO:.2f}–τ={METRIC_BAND_TAU_HI:.2f}
  band (definition below), with the same metrics on the quantile median kept alongside in
  `stats.json`. No per-patient or trend-gain calibration is applied — the quantile head is
  calibrated directly, and being in-domain the simulator needs none.
  The test split is disjoint from calibration.
- **Validation table.** A SOTA-target-indexed validation table mirroring the
  training-time table (`train.py`) scores the announced-event forecast against the
  same horizon-dependent bars. Being in-domain, these are upper-anchor numbers.

{_SIM_METRIC_DEFS}

## T1DMSIM

Calibration windows: {s['n_cal_windows']}; test windows: {s['n_test_windows']}; patients: {s['n_patients']}.

{_suite_table(s)}

{_selected_offset_section(s)}

{_night_onset_section(s)}

{_validation_section(s)}

Baselines and calibrated intervals:

{_baseline_table(s)}

## Figures

Generated by `make_comparison_figures.py` on the current checkpoint: example test-window
trajectories (simulated CGM, context + future, vs predicted BG); a predicted-vs-true parity
scatter; and a Clarke Error Grid. The parity and Clarke panels are reported hour-by-hour
(one panel per 30 min and then per hour out to the night long horizon, from a rolled
forecast). The BG panels are the announced-event median forecast. RMSE-by-horizon
(hour-by-hour) vs persistence is in `figures/rmse_vs_horizon.png`, carrying both the
band-scored and the median-line series.

![T1DMSIM — simulated CGM vs predicted BG, example windows](figures/sim_trajectories.png)
![T1DMSIM — predicted vs true parity](figures/sim_parity.png)
![T1DMSIM — Clarke Error Grid](figures/sim_clarke.png)

## Caveats

- In-domain: the model was trained on this simulator, so these numbers are an upper
  reference and say nothing about generalisation beyond it.
- The prediction-horizon carbohydrate, insulin and exercise are announced to the model.
- Per-window counts are capped (test windows shown above).
- Hypo/hyper recall and precision rest on the event counts noted under the suite table.

_Generated by `build_report.py`. Raw numbers in `stats.json`._
"""


def render_sim_figure(R: dict, path: str):
    """RMSE vs horizon: model (point + window-mean) against persistence — no real
    peers. Hour-by-hour from the rolled ``rmse_by_hour`` when present, else the
    3-point ``metrics`` suite. The solid model series are band-scored; the dashed
    ones are the median line, drawn only when the stats JSON carries it."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    s = R['sim']
    rbh = s.get('rmse_by_hour')
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if rbh:
        hs = sorted(int(h) for h in rbh)
        col = lambda key: [rbh[str(h)][key] for h in hs]
        med_p = _median_hour_curve(rbh, hs, 'rmse_point')
        med_w = _median_hour_curve(rbh, hs, 'rmse_winmean')
        ax.plot(hs, col('rmse_point'), 's-', label=_band_series_label(med_p, ' point'))
        ax.plot(hs, col('rmse_winmean'), 'o-', label=_band_series_label(med_w, ' window-mean'))
        ax.plot(hs, col('rmse_persist_point'), ':', color='gray', label='persistence (point)')
    else:
        m = s['metrics']; hs = list(HORIZONS)
        med_p = _median_suite_curve(m, hs, 'rmse_point')
        med_w = _median_suite_curve(m, hs, 'rmse_winmean')
        ax.plot(hs, [m[str(h)]['rmse_point'] for h in hs], 's-',
                label=_band_series_label(med_p, ' point'))
        ax.plot(hs, [m[str(h)]['rmse_winmean'] for h in hs], 'o-',
                label=_band_series_label(med_w, ' window-mean'))
        ax.plot(hs, [m[str(h)]['rmse_persist_point'] for h in hs], ':', color='gray', label='persistence (point)')
    if med_p is not None:
        ax.plot(hs, med_p, '--', color='C0', alpha=0.8, label='T1DMAI point (median line)')
    if med_w is not None:
        ax.plot(hs, med_w, '--', color='C1', alpha=0.8, label='T1DMAI window-mean (median line)')
    ax.set_xlabel('prediction horizon (min)'); ax.set_ylabel('RMSE (mg/dL)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title('T1DMSIM blood-glucose RMSE vs horizon (in-domain)')
    fig.tight_layout(); fig.savefig(path, dpi=360); plt.close(fig)


# --------------------------------------------------------------------------- #
# Comparison-figure driver (CPU-friendly; conditional BG panels + latent panels).
# --------------------------------------------------------------------------- #
CTX = MAX_CONTEXT_PATCHES * PATCH_SIZE
PRED = PREDICTION_PATCHES * PATCH_SIZE
# Parity/clarke forecasts are rolled out to the night long horizon so the figures
# read hour-by-hour past the single ``PREDICTION_HORIZON_HOURS`` forward pass; the trajectory examples then
# span the same window. Collapses to a single pass when no rolling is configured.
FIG_ROLLS = max(1, math.ceil(NIGHT_LONG_HORIZON_HOURS / PREDICTION_HORIZON_HOURS))
FIG_STEPS = FIG_ROLLS * PRED                             # rolled forecast length
CAP = {'sim': 30}                                        # test windows per patient






