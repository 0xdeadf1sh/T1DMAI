"""
Shared real-data report assembly: the metric README/JSON renderer and the
actual-vs-predicted figure driver, both parameterized by evaluation mode.

Two metrics entry points consume this module:

  * ``metrics/real/``      — evaluate on the unmodified real records, announcing
                             each window's LOGGED future carbs/insulin (the
                             deployment what-if regime).
  * ``metrics/augmented/`` — same, on records whose unlogged meals/boluses have
                             been reconstructed and injected (``augment_fn``),
                             then announced.

Both are CONDITIONAL evaluations (future carbs/insulin given to the model); the
only difference is the ``augment_fn`` applied to each Segment after load. The
rendered README is a public artifact — observed numbers beside published peers,
no proposed changes.
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
from . import load_dataset
from .run_eval import split_segments, evaluate_dataset, _make_night_overrides_fn
from .calibrate import _future_overrides, CTX_STEPS, PRED_STEPS
from .features import build_feature_stack, context_window, smoothed_cgm
from .metrics import HORIZONS
from .figures import trajectory_grid, parity_scatter, clarke_grid

# Per-horizon excursion display targets for the SOTA-target markdown table. The
# risk-space redesign removed ``config.EXCURSION_TARGET_*`` (realdata was out of
# scope for that config cycle); these mirror the prior tuples
# ``(base@30min, slope_per_30min, floor)`` and are DISPLAY-ONLY (row colour /
# Target text), never the loss or any selection.
EXCURSION_TARGET_HYPO_RECALL = (0.70, 0.10, 0.30)
EXCURSION_TARGET_HYPO_PRECISION = (0.70, 0.10, 0.30)
EXCURSION_TARGET_HYPER_RECALL = (0.80, 0.10, 0.40)
EXCURSION_TARGET_HYPER_PRECISION = (0.80, 0.10, 0.40)

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'checkpoints', 't1dmai_best.pt')
PRETTY = {'ohiot1dm': 'OhioT1DM', 'azt1d': 'AZT1D', 'shanghai': 'ShanghaiT1DM'}

# Published OhioT1DM peers (LITREVIEW.md). 'window_mean' = RMSE pooled over steps
# 0..PH; 'point' = RMSE at the single PH-ahead step. The peers forecast WITHOUT a
# future-input announcement; T1DMAI here is evaluated WITH it (stated in-report).
PEERS = {
    'window_mean': {
        'Crossformer': {'train': 'DCLP3 (real T1D)', 'test': 'OhioT1DM (external)',
                        'rmse': {30: 15.81, 60: 26.32, 120: 36.12}},
        'PatchTST': {'train': 'DCLP3 (real T1D)', 'test': 'OhioT1DM (external)',
                     'rmse': {30: 16.70, 60: 24.62, 120: 36.10}},
    },
    'point': {
        'DRNN': {'train': 'OhioT1DM', 'test': 'OhioT1DM', 'rmse': {30: 18.9}},
        'CRNN': {'train': 'OhioT1DM', 'test': 'OhioT1DM', 'rmse': {30: 21.07, 60: 33.27}},
        'DCRNet': {'train': 'OhioT1DM', 'test': 'OhioT1DM', 'rmse': {30: 20.72, 60: 34.41}},
        'BiT-MAML': {'train': 'OhioT1DM', 'test': 'OhioT1DM', 'rmse': {30: 24.89, 60: 30.58}},
        'KD-Seq2Seq (student)': {'train': 'OhioT1DM', 'test': 'OhioT1DM',
                                 'rmse': {30: 18.24, 60: 29.56, 120: 43.59}},
    },
}


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
    # cohorts must RE-FIT a per-cohort delta (run_eval), since conformal validity needs
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


def evaluate_all(device, conditional: bool = True, augment_fn=None) -> dict:
    """Evaluate every dataset with the current checkpoint.

    The model is always conditioned (each window's future carbs/insulin are
    announced); ``conditional`` is a deprecated no-op kept for call-compatibility.
    """
    model, stats, step = load_model(device)
    results = {'_meta': {'step': step, 'conditional': True,
                         'augmented': augment_fn is not None}}
    for ds in ('ohiot1dm', 'azt1d', 'shanghai'):
        print(f"evaluating {ds}…")
        results[ds] = evaluate_dataset(ds, model, stats, device,
                                       cal_stride=8, cal_cap=24, test_stride=4, test_cap=60,
                                       augment_fn=augment_fn)
    return results


# --------------------------------------------------------------------------- #
# README rendering (neutral, public).
# --------------------------------------------------------------------------- #
def _fmt(x, nd=1):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _has_bands(res: dict) -> bool:
    """True when every horizon of ``res['metrics']`` carries the band-scoring keys
    (``band_cov50`` / ``median_line``), i.e. the suite was computed against the
    quantile band. False for a band-less cohort or a stats JSON predating it."""
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
            "cohort's calibration split):\n\n" + "\n".join(rows))


def _peer_basis_table(res: dict, basis: str) -> str:
    m = res['metrics']
    key = 'rmse_point' if basis == 'point' else 'rmse_winmean'
    pkey = 'rmse_persist_point' if basis == 'point' else 'rmse_persist_winmean'
    label = 'point' if basis == 'point' else 'window-mean'
    band = _has_bands(res)
    rows = ["| Model | Trained on | Tested on | RMSE @30 | RMSE @60 | RMSE @120 |",
            "|---|---|---|--:|--:|--:|"]
    rows.append(f"| **T1DMAI ({label}{', band-scored' if band else ''})** | "
                f"T1DMSIM (behavioral sim) | OhioT1DM | "
                + " | ".join(f"{m[str(h)][key]:.1f}" for h in HORIZONS) + " |")
    if band:
        rows.append(f"| **T1DMAI ({label}, median line)** | T1DMSIM (behavioral sim) | OhioT1DM | "
                    + " | ".join(f"{m[str(h)]['median_line'][key]:.1f}" for h in HORIZONS) + " |")
    rows.append(f"| Naive persistence ({label}) | — | OhioT1DM | "
                + " | ".join(f"{m[str(h)][pkey]:.1f}" for h in HORIZONS) + " |")
    for name, v in PEERS[basis].items():
        r = v['rmse']
        rows.append(f"| {name} | {v['train']} | {v['test']} | "
                    + " | ".join(f"{r[h]:.1f}" if h in r else "—" for h in HORIZONS) + " |")
    caption = (f"\n\nThe band-scored row projects the truth onto the model's τ={METRIC_BAND_TAU_LO:.2f}–"
               f"τ={METRIC_BAND_TAU_HI:.2f} band; the published peers and persistence emit a single "
               f"trajectory with no band, so the median-line row is the like-for-like basis for this "
               f"table." if band else "")
    return "\n".join(rows) + caption


# --------------------------------------------------------------------------- #
# SOTA-target validation table (markdown mirror of train.py's terminal table).
#
# The bars below MIRROR the values hard-coded in train.py's
# ``_render_validation_table`` — that table is the source of truth; keep these in
# sync by hand if it changes. The per-horizon excursion bars are NOT duplicated:
# they read the module-local ``EXCURSION_TARGET_*`` constants through ``_exc_target``
# (train.py's ``_excursion_target`` formula). Only the rows real CGM can score are
# emitted; the per-channel NLL/σ, event-detection, signal-attribution, TIR, and
# rate-of-change rows have no real-data counterpart and are omitted.
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
    "table (`train.py`), over the subset of rows real CGM can score. The forecast is "
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
    dataset, mirroring train.py's terminal validation table over the rows real CGM
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


# --------------------------------------------------------------------------- #
# Consolidated comparison against the surveyed literature (LITREVIEW.md).
# --------------------------------------------------------------------------- #
# Best comparable peer per metric, OhioT1DM real CGM unless the in-silico column
# says otherwise, drawn verbatim from LITREVIEW.md. The aggregation basis is given
# per row (point-at-horizon vs window-mean) so each cell compares like with like
# (LITREVIEW §3a‡). Spec: (metric, horizon, path-into-metrics[h], basis, nd, scale,
# real-data peer, in-silico peer, LITREVIEW section).
_LIT_SPEC = [
    ("RMSE @30m",  30, ('rmse_point',),   "point",       1, 1.0,  "18.9 (DRNN)",            "6.45 (DCRNet)",  "§3"),
    ("RMSE @60m",  60, ('rmse_point',),   "point",       1, 1.0,  "30.58 (BiT-MAML)",       "17.73 (DCRNet)", "§3"),
    ("RMSE @120m", 120, ('rmse_point',),  "point",       1, 1.0,  "— (no firm point peer)", "—",              "§4"),
    ("RMSE @30m",  30, ('rmse_winmean',), "window-mean", 1, 1.0,  "15.81 (Crossformer)",    "—",              "§3a"),
    ("RMSE @60m",  60, ('rmse_winmean',), "window-mean", 1, 1.0,  "24.62 (PatchTST)",       "—",              "§3a"),
    ("RMSE @120m", 120, ('rmse_winmean',),"window-mean", 1, 1.0,  "36.10 (PatchTST)",       "—",              "§3a"),
    ("MAE @30m",   30, ('mae_point',),    "point",       1, 1.0,  "10.03 (DA-CMTL, pooled)","—",              "§3"),
    ("MAE @30m",   30, ('mae_winmean',),  "window-mean", 1, 1.0,  "9.67 (Crossformer)",     "—",              "§3a"),
    ("MAE @60m",   60, ('mae_winmean',),  "window-mean", 1, 1.0,  "16.38 (PatchTST)",       "—",              "§3a"),
    ("MARD @30m",  30, ('mard',),         "point / wm*", 2, 1.0,  "7.02 (Crossformer)*",    "4.8 (DRNN)",     "§3a,§3"),
    ("MARD @60m",  60, ('mard',),         "point / wm*", 2, 1.0,  "12.04 (PatchTST)*",      "—",              "§3a"),
    ("MARD @120m", 120, ('mard',),        "point / wm*", 2, 1.0,  "18.70 (PatchTST)*",      "—",              "§3a"),
    ("Clarke-A @30m", 30, ('clarke_A',),  "point",       2, 1.0,  "92.93 (DA-CMTL, Ohio)",  "—",              "§3"),
    ("hypo recall @30m", 30, ('hypo', 'recall'), "point", 2, 100.0, "92.13 (DA-CMTL, pooled)", "—",          "§3"),
]


def _litreview_comparison(R: dict, mode: str) -> str:
    """Place T1DMAI's OhioT1DM numbers beside the best comparable peer per metric
    surveyed in LITREVIEW.md, basis-matched. The level rows read the announced-event
    MEDIAN-LINE forecast, the like-for-like basis against published point forecasters
    (the surveyed peers forecast history-only, an input-regime difference noted below);
    the hypo-recall row keys off the forecast band edge, as the detector does.
    OhioT1DM-anchored because the surveyed peers are reported there (or pooled);
    numbers only, no judgement."""
    o = R['ohiot1dm']
    cm = o.get('metrics', {})
    band = _has_bands(o)

    def fetch(m: dict, h: int, path: tuple[str, ...], scale: float):
        d = m.get(str(h))
        if isinstance(d, dict):
            ml = d.get('median_line')
            if isinstance(ml, dict) and path[0] in ml:
                d = ml
        for p in path:
            if not isinstance(d, dict):
                return None
            d = d.get(p)
        return d * scale if isinstance(d, (int, float)) else None

    def num(v, nd):
        return '—' if not isinstance(v, (int, float)) else f"{v:.{nd}f}"

    head = ["| Metric | Basis | T1DMAI | Best real-data peer | Best in-silico peer | LITREVIEW |",
            "|---|---|--:|---|---|---|"]
    body = []
    for label, h, path, basis, nd, scale, real, sim, sec in _LIT_SPEC:
        cv = num(fetch(cm, h, path, scale), nd)
        body.append(f"| {label} | {basis} | {cv} | {real} | {sim} | {sec} |")

    aug = (" These are the **augmented** (upper-bound) numbers — meals/boluses missing "
           "from the source logs are reconstructed and announced — so they sit above the "
           "deployment-faithful values in `metrics/real/`."
           if mode == 'augmented' else "")
    intro = (
        "`LITREVIEW.md` surveys published deep-learning T1DM glucose forecasters. Its "
        "headline finding is that the honest real-CGM 30-min RMSE floor is ~15.7–18.9 mg/dL "
        "(Crossformer 15.8, PatchTST 16.7, KD-Seq2Seq 18.2, DRNN 18.9), rising to ~24.6–34.4 "
        "@60 min and ~36.1–43.6 @120 min; in-silico (UVA/Padova) peers run roughly 2× lower "
        "(DCRNet 6.45 @30 min). Real-T1D MARD < 7.5 % appears only at 30 min (Crossformer 7.02, "
        "PatchTST 7.44, window-mean), and Clarke-A peaks at ~92.9–98.7 % @30 min (DA-CMTL). Hypo "
        "detection tops out near 92 % recall (DA-CMTL); hyper recall, hypo precision, CG-EGA, and "
        "prediction-model TIR error are essentially unreported across the surveyed peers, so those "
        "rows have no head-to-head value here." + aug)
    method = (
        "The table anchors on OhioT1DM (the shared benchmark; AZT1D and ShanghaiT1DM enter the "
        "survey only pooled, in DA-CMTL) and matches each row by aggregation basis — point-at-"
        "horizon vs window-mean (LITREVIEW §3a‡). T1DMAI announces the horizon's carbohydrate and "
        "insulin, whereas the surveyed peers forecast history-only, so the input regimes differ. "
        "Peer values are real CGM unless the in-silico column "
        "states otherwise; T1DMSIM is a behavioral simulator distinct from the UVA/Padova glucose-"
        "insulin simulator the in-silico peers use, so that column is cross-simulator and only "
        "suggestive (LITREVIEW §1).")
    basis_note = (
        f"\n\n**Forecast basis.** The RMSE, MAE, MARD and Clarke-A rows read the median line of the "
        f"quantile forecast, not the band-scored suite in the per-dataset tables above: the surveyed "
        f"peers are point forecasters, and a band-scored error is ≤ its median-line value by "
        f"construction. The hypo-recall row is unaffected by the basis: the detector fires off the "
        f"τ={HYPO_ALARM_QUANTILE_TAU:.2f} lower band edge on both." if band else "")
    footnotes = (
        "\\* **MARD basis.** T1DMAI MARD is point-at-horizon; the Crossformer/PatchTST peer MARD is "
        "window-mean (averaged over steps 0…PH), which reads lower (LITREVIEW §3a‡) — the comparison "
        "is across bases. The RMSE rows reproduce the per-model OhioT1DM peer tables above."
        + basis_note)
    return ("## Comparison with the literature (LITREVIEW.md)\n\n"
            f"{intro}\n\n{method}\n\n" + "\n".join(head + body) + f"\n\n{footnotes}")


def _intro(mode: str) -> str:
    """Mode-specific opening (input regime + the augmentation method/caveat)."""
    common = (
        "T1DMAI is trained on T1DMSIM, a behavioral Type-1-diabetes simulator that models "
        "patient behavior as the primary driver of blood glucose. The model emits a "
        "risk-space quantile forecast of blood glucose (mg/dL after inversion); the headline "
        f"metrics score the truth against its τ={METRIC_BAND_TAU_LO:.2f}–τ={METRIC_BAND_TAU_HI:.2f} "
        "band, with the same metrics on the quantile median reported alongside. "
        "Every number below is computed on **real** CGM data the model did not see during "
        "training.")
    announce = (
        " This report evaluates the model in the **announced-event (what-if) regime**: at each "
        "prediction window the patient's future carbohydrate and insulin over the forecast "
        "horizon are given to the model, as they are at deployment when a meal or dose is "
        "declared. The forecast is conditioned on those announcements.")
    if mode == 'augmented':
        aug = (
            " The records are additionally **augmented**: meals and boluses that the source "
            "logs omit are reconstructed from the CGM record and injected before evaluation "
            "(an unexplained hyperglycemic onset receives a carbohydrate event; an unexplained "
            "hypoglycemic onset receives a bolus), then announced like any other event. Because "
            "the injected amounts are sized from the same CGM the model is then scored against, "
            "these numbers are an **upper bound** on the announced-event regime — the value "
            "attainable if every meal and dose were logged and declared.")
        return common + announce + aug
    return common + announce


def render_readme(R: dict, mode: str = 'real') -> str:
    o = R['ohiot1dm']
    meta = R.get('_meta', {})
    step = meta.get('step')
    step_line = (f" The checkpoint evaluated here is at training step {step:,} (training was "
                 f"ongoing at generation time)." if step else "")
    title = ("T1DMAI on real CGM datasets — announced-event comparison" if mode == 'real'
             else "T1DMAI on augmented real CGM datasets — announced-event comparison")
    return f"""# {title}

This report tabulates the T1DMAI model's blood-glucose-prediction metrics on real
continuous-glucose-monitoring (CGM) datasets alongside published deep-learning
forecasters. It reports numbers only and draws no judgement of relative quality.
{_intro(mode)}{step_line}

## Method

- **Datasets.** OhioT1DM (2018 cohort, canonical per-patient train/test split), AZT1D,
  and ShanghaiT1DM (intra-record temporal split). Per-window counts are capped.
- **Inputs.** CGM, carbohydrate, and insulin only — the channels the published
  forecasters also receive. Carbohydrate/insulin events are convolved with the
  simulator's absorption/action kernels. The prediction-horizon carbohydrate and insulin
  are announced to the model.
- **Forecast.** The model emits a risk-space quantile fan, inverted to mg/dL. The headline
  level metrics score the truth against the band between τ={METRIC_BAND_TAU_LO:.2f} and
  τ={METRIC_BAND_TAU_HI:.2f} (definition below); the same metrics on the quantile median
  alone are reported beside the published peers and kept in `stats.json` under
  `metrics[h]["median_line"]`. No per-patient or trend-gain calibration is applied — the
  quantile head is calibrated directly. A per-horizon excursion decision-offset is
  fit on the calibration split (below); the test split is disjoint from it.
- **Validation table.** Each dataset carries a SOTA-target-indexed validation table
  mirroring the training-time table (`train.py`), scoring the announced-event forecast
  against the same horizon-dependent bars. A consolidated comparison against the
  surveyed literature (`LITREVIEW.md`) closes the report.

## Metric definitions

- **Scoring basis — band-projected.** The forecast is a quantile fan, so the level metrics
  score the truth against the band between τ={METRIC_BAND_TAU_LO:.2f} and τ={METRIC_BAND_TAU_HI:.2f}
  rather than against a single line: the scored forecast is the true value clipped to that
  band, so the reported error is the distance from the truth to the nearer band edge, and is
  zero whenever the truth falls inside the band. RMSE, MAE, MARD, the Clarke grid and CG-EGA
  all read that projected series as the FORECAST; the truth stays the reference on every
  axis, so a CG-EGA point's glycaemic region and its rate-dependent acceptance widening are
  the truth's own throughout. Wherever the truth lies inside the band the projection equals
  it, so the scored series carries the truth's derivative there and the rate grid registers
  an error only where the band actually missed. The same metrics computed on the
  quantile median alone are kept in `stats.json` under `metrics[h]["median_line"]` and carry
  the published-peer and literature tables.
- **RMSE — point (pt):** error at the single step exactly PH minutes ahead.
- **RMSE — window-mean (wm):** RMSE pooled over every step from 5 min to PH (steps
  0…PH). For one model this is lower than its point value, as it averages in the easier
  near-term steps.
- **MAE:** mean absolute error (point and window-mean as above).
- **MARD:** mean absolute relative difference, |pred−true|/true, at the horizon point.
- **Clarke A / A+B / E:** percentage of points in Clarke Error Grid zone A, zones A∪B,
  and zone E (Clarke et al. 1987), at the horizon point.
- **band cov50 %:** the realized fraction of true values falling inside the
  τ={METRIC_BAND_TAU_LO:.2f}–τ={METRIC_BAND_TAU_HI:.2f} band at the horizon point, as a
  percentage. The band is the model's nominal inner-50 % interval, so the nominal value is 50 %.
- **band width:** the mean edge-to-edge width of that band at the horizon point, mg/dL. It is
  the scale the band-projected errors above are relative to.
- **hypo / hyper recall, precision:** detection of true CGM crossings below {BG_HYPO_THRESHOLD:.0f} / above
  {BG_HYPER_THRESHOLD:.0f} mg/dL at the horizon point. The alarm fires off the forecast band edges — the
  τ={HYPO_ALARM_QUANTILE_TAU:.2f} lower edge for hypo, the τ={HYPER_ALARM_QUANTILE_TAU:.2f} upper edge for
  hyper — with strict recall and a small precision tolerance (denominators given under each table).
  These two taus are their own knobs, independent of the scoring band above.
- **Excursion decision offset:** a per-horizon hypo decision offset δ added to the
  {BG_HYPO_THRESHOLD:.0f} mg/dL alarm level: the alarm fires when the τ={METRIC_BAND_TAU_LO:.2f} lower
  band edge — the same edge the level metrics project onto, or the median line where no fan is
  available — falls below {BG_HYPO_THRESHOLD:.0f} + δ, the series named in the caption of each table.
  That edge sits at or below the median, so the alarm is at least as sensitive as the same rule read
  off the median line. δ is selected on the calibration split under a precision floor; the test-split
  recall/precision at it are reported.
- **Night-onset excursion recall / precision:** a per-night binary call. From a bedtime
  origin the forecast is rolled across the whole night; a night is flagged hypo/hyper if
  the forecast crosses {BG_HYPO_THRESHOLD:.0f} / {BG_HYPER_THRESHOLD:.0f} mg/dL anywhere in the night, scored against whether the
  true CGM did. The roll is conditioned on the night's logged overnight carbohydrate/insulin.
- **Persistence:** the naive last-CGM-value (flat) forecast, on each basis. It is a single
  trajectory and carries no band, so it is scored as a point forecast.
- **RMSE skill vs persistence:** (persistence RMSE − model RMSE) / persistence RMSE, %. The
  numerator is band-scored and the baseline is a point forecast, so the two rest on different
  bases.
- **Conformal 90% interval:** split-conformal half-width fit on the calibration split,
  coverage measured on the test split. Both splits use the band-projected series, so the
  half-width is width on top of the band rather than around a line.

## OhioT1DM

**Training source:** T1DMSIM (behavioral simulator). **Evaluation source:** OhioT1DM
2018 cohort, canonical per-patient test split (real CGM).

Calibration windows: {o['n_cal_windows']}; test windows: {o['n_test_windows']}; patients: {o['n_patients']}.

{_suite_table(o)}

{_selected_offset_section(o)}

{_night_onset_section(o)}

{_validation_section(o)}

Baselines and calibrated intervals:

{_baseline_table(o)}

### Published forecasters — point-at-horizon basis (OhioT1DM)

RMSE at the single PH-ahead point (the convention verified for DRNN and CRNN, and
standard for the OhioT1DM / BGLP benchmark).

{_peer_basis_table(o, 'point')}

### Published forecasters — window-mean basis (OhioT1DM)

RMSE averaged over steps 0…PH (the convention verified from the Crossformer/PatchTST
source, arXiv:2505.08821).

{_peer_basis_table(o, 'window_mean')}

Provenance and transfer setting (per the Trained-on / Tested-on columns above). Three
distinct settings appear: the point-basis peers (DRNN [J. Healthcare Inf. Res. 2020],
CRNN [arXiv:1807.03043], DCRNet, BiT-MAML) are trained and tested on OhioT1DM (in-domain);
Crossformer and PatchTST (arXiv:2505.08821) are trained on DCLP3 (n = 112 real T1D) and
externally tested on OhioT1DM (cross-real-dataset, window-mean basis); T1DMAI is trained
on the T1DMSIM behavioral simulator and tested on OhioT1DM (simulator → real). The
published peers forecast without a future-input announcement, whereas T1DMAI here is given
the horizon's carbohydrate and insulin; the settings therefore differ in input regime as
well as in train/test provenance. KD-Seq2Seq (Sci. Reports 2026) is personalized, and its
per-horizon RMSE basis is not stated explicitly in the source.

## AZT1D

**Training source:** T1DMSIM (behavioral simulator). **Evaluation source:** AZT1D,
intra-record temporal test split (real CGM).

Test windows: {R['azt1d']['n_test_windows']}; patients: {R['azt1d']['n_patients']}.

{_suite_table(R['azt1d'])}

{_selected_offset_section(R['azt1d'])}

{_night_onset_section(R['azt1d'])}

{_validation_section(R['azt1d'])}

{_baseline_table(R['azt1d'])}

## ShanghaiT1DM

**Training source:** T1DMSIM (behavioral simulator). **Evaluation source:** ShanghaiT1DM,
intra-record temporal test split (real CGM).

Test windows: {R['shanghai']['n_test_windows']}; patients: {R['shanghai']['n_patients']}.

{_suite_table(R['shanghai'])}

{_selected_offset_section(R['shanghai'])}

{_night_onset_section(R['shanghai'])}

{_validation_section(R['shanghai'])}

{_baseline_table(R['shanghai'])}

## Figures

Actual-vs-predicted BG, generated by `make_comparison_figures.py` on the current
checkpoint. For each dataset: example test-window trajectories (real CGM, context +
future, vs the model's predicted BG); a predicted-vs-true parity scatter (identity and
±20% lines); and a Clarke Error Grid. The parity and Clarke panels are reported
hour-by-hour — one panel per 30 min and then per hour out to the night long horizon —
from a rolled forecast (the far horizons use only the windows that reach them). These BG
panels are the announced-event median forecast. RMSE-by-horizon (hour-by-hour) vs naive
persistence is in `figures/rmse_vs_horizon.png`, carrying both the band-scored and the
median-line series; the published peers are tabulated above rather than plotted.

### OhioT1DM

![OhioT1DM — real CGM vs predicted BG, example windows](figures/ohiot1dm_trajectories.png)
![OhioT1DM — predicted vs true parity](figures/ohiot1dm_parity.png)
![OhioT1DM — Clarke Error Grid](figures/ohiot1dm_clarke.png)

### AZT1D

![AZT1D — real CGM vs predicted BG, example windows](figures/azt1d_trajectories.png)
![AZT1D — predicted vs true parity](figures/azt1d_parity.png)
![AZT1D — Clarke Error Grid](figures/azt1d_clarke.png)

### ShanghaiT1DM

![ShanghaiT1DM — real CGM vs predicted BG, example windows](figures/shanghai_trajectories.png)
![ShanghaiT1DM — predicted vs true parity](figures/shanghai_parity.png)
![ShanghaiT1DM — Clarke Error Grid](figures/shanghai_clarke.png)

{_litreview_comparison(R, mode)}

## Caveats

- Numbers are computed on real CGM; the model is simulator-trained and saw no real CGM.
- The prediction-horizon carbohydrate and insulin are announced to the model; the published
  peers forecast without that announcement, so the input regimes differ.{_augment_caveat(mode)}
- Point and window-mean RMSE are different aggregations; rows compare only within a basis.
- The published OhioT1DM peer numbers come from differing test protocols and splits;
  Crossformer/PatchTST are cross-dataset external tests (trained on DCLP3).
- Per-window counts are capped (test windows per dataset shown above); the published
  peers use their full test sets.
- AZT1D is a closed-loop automated-insulin-delivery cohort with between-subject insulin
  scale variation; ShanghaiT1DM carbohydrate is estimated from free-text dietary entries
  and is the lowest-fidelity input channel (CGM cadence 15 min).
- Hypo/hyper recall and precision rest on the event counts noted under each suite table.

_Generated by `build_report.py`. Raw numbers in `stats.json`._
"""


def _augment_caveat(mode: str) -> str:
    if mode != 'augmented':
        return ""
    return (
        "\n- Records are augmented with reconstructed meals/boluses sized from the CGM "
        "record; because the injected amount is derived from the same CGM the forecast is "
        "scored against, the numbers are an upper bound on the announced-event regime.")


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


def render_figure(R: dict, path: str):
    """OhioT1DM RMSE vs horizon (point + window-mean) against naive persistence.
    Hour-by-hour from the rolled ``rmse_by_hour`` when present, falling back to the
    3-point ``metrics`` suite. The solid model series is band-scored; the dashed one
    is the median line, drawn only when the stats JSON carries it. The published
    peers are tabulated in the README's peer-basis tables, not plotted here."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    o = R['ohiot1dm']
    rbh = o.get('rmse_by_hour')
    fig, (axp, axw) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    if rbh:
        hs = sorted(int(h) for h in rbh)
        col = lambda key: [rbh[str(h)][key] for h in hs]
        med_p = _median_hour_curve(rbh, hs, 'rmse_point')
        med_w = _median_hour_curve(rbh, hs, 'rmse_winmean')
        axp.plot(hs, col('rmse_point'), 's-', label=_band_series_label(med_p))
        axp.plot(hs, col('rmse_persist_point'), ':', color='gray', label='persistence')
        axw.plot(hs, col('rmse_winmean'), 's-', label=_band_series_label(med_w))
        axw.plot(hs, col('rmse_persist_winmean'), ':', color='gray', label='persistence')
    else:
        m = o['metrics']; hs = list(HORIZONS)
        med_p = _median_suite_curve(m, hs, 'rmse_point')
        med_w = _median_suite_curve(m, hs, 'rmse_winmean')
        axp.plot(hs, [m[str(h)]['rmse_point'] for h in hs], 's-', label=_band_series_label(med_p))
        axp.plot(hs, [m[str(h)]['rmse_persist_point'] for h in hs], ':', color='gray', label='persistence')
        axw.plot(hs, [m[str(h)]['rmse_winmean'] for h in hs], 's-', label=_band_series_label(med_w))
        axw.plot(hs, [m[str(h)]['rmse_persist_winmean'] for h in hs], ':', color='gray', label='persistence')
    if med_p is not None:
        axp.plot(hs, med_p, '--', color='C0', alpha=0.8, label='T1DMAI (median line)')
    if med_w is not None:
        axw.plot(hs, med_w, '--', color='C0', alpha=0.8, label='T1DMAI (median line)')
    axp.set_title('point-at-horizon basis'); axw.set_title('window-mean basis')
    for ax in (axp, axw):
        ax.set_xlabel('prediction horizon (min)'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axp.set_ylabel('RMSE (mg/dL)')
    fig.suptitle('OhioT1DM blood-glucose RMSE vs horizon (real-data evaluation)')
    fig.tight_layout(); fig.savefig(path, dpi=360); plt.close(fig)


# --------------------------------------------------------------------------- #
# In-domain simulator report (single cohort, no real-CGM peers).
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
    """Public README for the in-domain T1DMSIM report — one cohort, numbers only."""
    s = R['sim']
    meta = R.get('_meta', {})
    step = meta.get('step')
    step_line = (f" The checkpoint evaluated here is at training step {step:,} (training was "
                 f"ongoing at generation time)." if step else "")
    return f"""# T1DMAI on the T1DMSIM simulator — in-domain reference

This report tabulates the T1DMAI model's blood-glucose-prediction metrics on fresh
patients drawn from the T1DMSIM behavioral simulator — the model's **training
distribution**. It is therefore an in-domain reference (an upper anchor for the
real-data reports under `metrics/real/` and `metrics/augmented/`), not a peer
comparison: the published real-CGM forecasters are evaluated on real data and are not
comparable here, so they are omitted. It reports numbers only and draws no judgement of
relative quality.{step_line}

## Method

- **Cohort.** {s['n_patients']} simulated patients at distinct seeds — each seed draws a
  distinct random patient — {meta.get('hours', '?')} h per patient after a
  {meta.get('warmup', '?')} h warmup discard. Calibration and test patients are disjoint
  seed pools.
- **Inputs.** The announced-event (what-if) regime: each window's future carbohydrate and
  insulin over the forecast horizon are given to the model, as in the real-data reports.
  The model consumes only CGM, carbohydrate, and insulin (time-of-day is inferred, not an input).
- **Forecast.** The model emits a risk-space quantile fan, inverted to mg/dL; the headline
  level metrics score the truth against the τ={METRIC_BAND_TAU_LO:.2f}–τ={METRIC_BAND_TAU_HI:.2f}
  band (definition below), with the same metrics on the quantile median kept alongside in
  `stats.json`. No per-patient or trend-gain calibration is applied — the quantile head is
  calibrated directly, and being in-domain the simulator needs none.
  The test split is disjoint from calibration.
- **Validation table.** A SOTA-target-indexed validation table mirroring the
  training-time table (`train.py`) scores the announced-event forecast against the
  same horizon-dependent bars; being in-domain, these are the upper-anchor numbers
  for the real-data reports. No real-CGM peer comparison is shown (the published
  forecasters are not comparable in-silico).

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
  reference, not real-data performance — read them beside `metrics/real/`.
- The prediction-horizon carbohydrate and insulin are announced to the model.
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
CAP = {'ohiot1dm': 30, 'azt1d': 12, 'shanghai': 20}      # test windows per patient


def _fmt_clock(hour: float | None) -> str:
    """Fractional hour-of-day in ``[0, 24)`` → a ``HH:MM`` wall-clock string.

    Returns ``"—"`` for ``None`` / non-finite input (e.g. the time-of-day probe is
    disabled, so the origin hour decoded to ``nan``).
    """
    if hour is None or not math.isfinite(hour):
        return "—"
    total = int(round(hour * 60.0)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def _collect(model, stats, name, device, augment_fn, announce=(0, 1)):
    """Per-window announced-event BG forecasts (trajectory/parity/clarke source).

    The forecast is the model's risk-space median (``median_bg`` for the single
    forward pass, the rolled ``pred_bg`` when extended). Forecasts are rolled out
    to ``FIG_STEPS`` (the night long horizon) so the parity/clarke panels read
    hour-by-hour. Windows are admitted with as little as a single
    ``PREDICTION_HORIZON_HOURS`` forward pass of trailing truth (``CTX + PRED``);
    ``true`` is then NaN-padded to ``FIG_STEPS``
    and the per-horizon plotters drop the unfilled tail, so the near-term horizons
    keep every window while the long ones use only the windows that reach them.
    """
    segs = load_dataset(name)
    if augment_fn is not None:
        segs = [augment_fn(s) for s in segs]
    _, test_segs = split_segments(segs, name)
    rows = []
    for seg in test_segs:
        n = (len(seg) // PATCH_SIZE) * PATCH_SIZE
        if n < CTX + PRED:
            continue
        feats = build_feature_stack(seg, stats)
        # The forecast is compared against the raw (bg-clamped) CGM (the model lives
        # in raw post-noise space); both the plotted truth and the context tail are
        # sliced from the raw ``seg.cgm`` (physical-range clamped).
        cgm_smooth = smoothed_cgm(seg.cgm)
        seg_hours = seg.hour_of_day()   # (len(seg),) true wall-clock hour per step, for the TOD probe overlay
        cnt = 0
        for ps in range(CTX, n - PRED + 1, 8 * PATCH_SIZE):
            if cnt >= CAP[name]:
                break
            ctx = context_window(feats, ps, MAX_CONTEXT_PATCHES)
            if FIG_ROLLS == 1:
                overrides = _future_overrides(feats, ps, announce)
                out = predict(model, ctx, normalization_stats=stats,
                              overrides=overrides, return_time=True)
                pred_t = out['median_bg']
            else:
                fn = _make_night_overrides_fn(feats, ps, announce, stats)
                out = predict_rolling(model, ctx, n_rolls=FIG_ROLLS,
                                      normalization_stats=stats, overrides_fn=fn,
                                      return_time=True)
                pred_t = out['pred_bg']
            pred = pred_t.detach().cpu().numpy()[:FIG_STEPS]
            tr = cgm_smooth[ps:ps + FIG_STEPS].astype(float)
            if len(tr) < FIG_STEPS:
                tr = np.concatenate([tr, np.full(FIG_STEPS - len(tr), np.nan)])
            # Read-only auxiliary time-of-day probe at this forecast origin. The
            # per-patch bin logits ride on the SAME forward as the forecast
            # (``return_time=True``); the scalar origin clock decodes from patch 0.
            # A probe-off checkpoint emits ``time_pred is None`` => NaN / None.
            tp = out.get('time_pred') if TIME_PROBE_ENABLED else None
            time_probs = None if tp is None else torch.softmax(tp, -1).cpu().numpy()
            if tp is None:
                pred_hour, tod_r = float('nan'), float('nan')
            else:
                h_t, r_t = time_of_day_decode_bins(tp[0:1, :], TIME_PROBE_N_BINS)
                pred_hour, tod_r = float(h_t.item()), float(r_t.item())
            rows.append({
                'patient': seg.patient,
                'pred': pred,
                'true': tr,
                'ctx_tail': cgm_smooth[max(0, ps - 12):ps].astype(float),
                'pred_hour': pred_hour,
                'true_hour': float(seg_hours[ps]),
                'tod_R': tod_r,
                'time_probs': time_probs,
            })
            cnt += 1
    return rows


def build_figures(figdir: str, device, conditional: bool = True, augment_fn=None,
                  announce: tuple[int, ...] = (0, 1)):
    """Write {dataset}_{trajectories,parity,clarke}.png (announced-event median BG
    forecast) into ``figdir``, on the current checkpoint.

    The model is always conditioned (the horizon's carbohydrate/insulin are
    announced); ``conditional`` is a deprecated no-op kept for call-compatibility.

    The risk-space redesign removed the model's dynamics outputs, so the former
    24 h carb/insulin/IS/HGO channel panels are gone — the model now emits only a
    BG quantile forecast, and these figures render the median BG forecast against
    the true CGM.
    """
    model, stats, step = load_model(device)
    model = model.to(device)
    os.makedirs(figdir, exist_ok=True)
    print(f"model step={step}; generating figures (device={device})…")
    for name in ('ohiot1dm', 'azt1d', 'shanghai'):
        rows = _collect(model, stats, name, device, augment_fn, announce)
        if not rows:
            print(f"  {name}: no windows"); continue
        pred = np.stack([r['pred'] for r in rows]); true = np.stack([r['true'] for r in rows])
        motion = np.nanmax(true, 1) - np.nanmin(true, 1)   # true has a NaN-padded tail
        order = np.argsort(motion)
        picks = [order[int(q * (len(order) - 1))] for q in
                 (0.97, 0.88, 0.78, 0.62, 0.5, 0.38, 0.22, 0.12, 0.04)]
        examples = []
        for i in picks:
            r = rows[i]
            label = f"{r['patient']} · swing {motion[i]:.0f} mg/dL"
            ph = r.get('pred_hour')
            if ph is not None and math.isfinite(ph):   # time-of-day probe on: append the clock read-out
                label += (f" · clock {_fmt_clock(ph)} (R{r['tod_R']:.1f})"
                          f" / true {_fmt_clock(r['true_hour'])}")
            examples.append({'ctx_tail': r['ctx_tail'], 'true_future': r['true'],
                             'pred_future': r['pred'], 'label': label,
                             'time_probs': r.get('time_probs')})
        pretty = PRETTY[name]
        trajectory_grid(examples, os.path.join(figdir, f'{name}_trajectories.png'),
                        f'{pretty}: real CGM vs predicted BG (example test windows)')
        parity_scatter(pred, true, os.path.join(figdir, f'{name}_parity.png'),
                       f'{pretty}: predicted vs true BG (test windows)')
        clarke_grid(pred, true, os.path.join(figdir, f'{name}_clarke.png'),
                    f'{pretty}: Clarke Error Grid (test windows)')
        print(f"  {name}: {len(rows)} windows → 3 figures")
    print("done")
