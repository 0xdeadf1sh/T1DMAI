"""Personal fine-tune: adapt a pretrained T1DMAI checkpoint to one's own record.

The published-cohort scripts hold out a PATIENT. A personal record has only one, so
this holds out a DAY — and specifically a *representative* day, the one whose
glycaemic variability sits nearest the median of every complete day the record
contains. Selecting on the calmest day would flatter the model; selecting on the
most chaotic would condemn it. The day nearest the middle is the honest choice, and
the ranking is printed in full so the choice can be inspected rather than trusted.

The validated machinery in ``finetune.py`` is reused wholesale — checkpoint loading
and its architecture guard, the windowed dataset, the Muon/AdamW split, the LR
schedule, the EMA shadow, the resilient step, the held-out evaluator and every
tunable. Only the data source and the split policy are new.

**Why a three-day reservation.** A prediction window's context is
``MAX_CONTEXT_PATCHES`` patches — a full 24 h. Scoring a held-out day therefore
reads the whole preceding day as context, and calibrating the conformal bands reads
the day before that. Reserving the held-out day alone would leave both contexts
inside the training span, so the reservation is three consecutive days:

    [ ctx day ][ cal day ][ val day ]
       context    conformal      reporting
                  calibration
                  + step selection

**The step is selected on the calibration day, never on the reported one.** A
fine-tune is evaluated every ``--eval-interval`` steps and the best snapshot kept;
keeping the minimum of dozens of noisy scores and then quoting that same score is
selection on the test set, and it biases the reported gain by an unknown amount. The
scored day is therefore never read by the selector: the checkpoint is chosen on the
calibration day's point error and the validation day is scored once per eval for
reporting only. The two are adjacent and so correlated — this bounds the leak rather
than eliminating it, which one record of a fortnight cannot do.

Training draws only from the record either side of that block, so no step the model
trained on is ever read as context by a calibration or scoring window. Windows are
strided densely (one patch by default, against four and eight for the published
cohorts) because a personal record is small; the resulting overlap makes neighbouring
windows highly correlated, which weakens the exchangeability the split-conformal fit
assumes — the coverage numbers are indicative, not guaranteed.

**The record is never modified.** ``--skip-hours`` drops the opening hours at load
time rather than by deleting rows, which keeps the action of any event preceding the
cut — a long-acting basal spans a full day — reaching correctly into what remains.
See ``realdata/personal.py``.

Runs from either ``finetune/`` or the repo root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from datetime import date, datetime, timedelta
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)              # so ``import finetune`` resolves the sibling

import numpy as np
import torch
from torch.utils.data import DataLoader

import finetune as F                   # also prepends REPO_ROOT to sys.path

from realdata.calibrate import collect_windows
from realdata.personal import DEFAULT_SKIP_HOURS, load as load_personal
from realdata.run_eval import _slice, evaluate_from_windows
from realdata.schema import GRID_MIN, Segment

# ---------------------------------------------------------------------------- #
# Split policy (each overridable by the matching CLI flag).
# ---------------------------------------------------------------------------- #
STEPS_PER_DAY: int = 24 * 60 // GRID_MIN          # 288 five-minute steps
PERSONAL_EVAL_STRIDE_PATCHES: int = 1             # dense: a personal record is small
CHAOS_METRICS: tuple[str, ...] = ('cv', 'sd', 'roc')
DEFAULT_CHAOS_METRIC: str = 'cv'

# Smallest span that yields even one training window.
MIN_TRAIN_STEPS: int = F._CTX_MIN_STEPS + F._PRED_STEPS


# ---------------------------------------------------------------------------- #
# Day enumeration + variability ranking.
# ---------------------------------------------------------------------------- #
def day_spans(seg: Segment) -> list[tuple[date, int, int]]:
    """Local calendar days wholly contained in ``seg``, as ``(day, lo, hi)`` spans.

    ``seg.t0`` is local wall clock (every adapter stamps it so), hence day
    boundaries are plain local midnights. A partial day at either end is omitted:
    a day is listed only if all ``STEPS_PER_DAY`` of its steps are present.

    Interpolation is invisible at this point — ``segment_grid`` has already bridged
    every gap up to ``MAX_INTERP_GAP_MIN`` and split the record at longer ones — so a
    listed day may contain a few reconstructed steps, bounded by that ceiling per gap.

    Returns:
        Spans in chronological order; ``seg.cgm[lo:hi]`` is that day's mg/dL trace.
    """
    n = len(seg)
    midnight = seg.t0.replace(hour=0, minute=0, second=0, microsecond=0)
    steps_past_midnight = int((seg.t0 - midnight).total_seconds() // 60) // GRID_MIN
    out: list[tuple[date, int, int]] = []
    i = (-steps_past_midnight) % STEPS_PER_DAY
    while i + STEPS_PER_DAY <= n:
        day = (seg.t0 + timedelta(minutes=GRID_MIN * i)).date()
        out.append((day, i, i + STEPS_PER_DAY))
        i += STEPS_PER_DAY
    return out


def day_chaos(cgm: np.ndarray, metric: str = DEFAULT_CHAOS_METRIC) -> float:
    """Score one day's glycaemic variability; larger is more chaotic.

    Args:
        cgm: (STEPS_PER_DAY,) mg/dL for a single local day.
        metric: ``cv`` — the coefficient of variation as a percentage, the
            variability figure a Stats block reports; ``sd`` — its standard
            deviation in mg/dL; ``roc`` — mean absolute step-to-step change in
            mg/dL per five minutes, which weights churn over dispersion.

    Returns:
        The scalar, or NaN if the day is too short to score.
    """
    assert metric in CHAOS_METRICS, f"unknown chaos metric {metric!r}, expected {CHAOS_METRICS}"
    v = np.asarray(cgm, dtype=np.float64)
    if v.size < 2:
        return float('nan')
    if metric == 'roc':
        return float(np.abs(np.diff(v)).mean())
    sd = float(v.std(ddof=1))
    if metric == 'sd':
        return sd
    mean = float(v.mean())
    return float('nan') if mean <= 0.0 else 100.0 * sd / mean


def rank_days(seg: Segment, metric: str) -> list[tuple[date, int, int, float]]:
    """Every wholly-contained local day with its variability score."""
    return [(day, lo, hi, day_chaos(seg.cgm[lo:hi], metric))
            for day, lo, hi in day_spans(seg)]


def choose_validation_day(
    ranked: list[tuple[date, int, int, float]],
    all_scores: list[float] | None = None,
) -> tuple[date, int, int]:
    """Pick the eligible day whose variability sits nearest the record's median.

    Two different populations, deliberately:

    * the median is taken over ``all_scores`` — every complete day in the WHOLE record,
      including days in segments that cannot host the reservation — because that is what
      makes it a statement about this person rather than about one contiguous run;
    * the choice is restricted to days that can host the three-day reservation, which
      needs two full days before it, for the calibration day and that day's own context.

    Keeping them apart matters: a record split by a CGM outage leaves a candidate set
    skewed toward whichever run happens to be longest, and letting that set define the
    median would drag the notion of "average" along with it.

    Args:
        ranked: ``(day, lo, hi, chaos)`` for the host segment, whose indices the
            reservation is expressed in.
        all_scores: chaos of every complete day across every segment. Defaults to the
            host's own scores when omitted.

    Returns:
        ``(day, lo, hi)`` of the chosen validation day.
    """
    scored = [r for r in ranked if math.isfinite(r[3])]
    assert scored, "no complete local day could be scored"
    pool = [s for s in (all_scores if all_scores is not None else [r[3] for r in scored])
            if math.isfinite(s)]
    assert pool, "no complete local day could be scored anywhere in the record"
    median = float(np.median(pool))

    eligible = [r for r in scored if r[1] >= 2 * STEPS_PER_DAY]
    assert eligible, (
        f"no day has the two preceding full days a leak-free reservation needs "
        f"(the longest run holds {len(scored)} complete day(s); at least three are required)"
    )
    day, lo, hi, _ = min(eligible, key=lambda r: (abs(r[3] - median), r[0]))
    return day, lo, hi


def _host_for_day(segs: list[Segment], wanted: date) -> Segment | None:
    """The segment in which ``wanted`` is a complete local day, or None.

    A record broken by a CGM outage has complete days in more than one segment, and
    the reservation has to be placed in whichever one holds the requested day — the
    longest-segment default can only ever score days in that one run, which for a
    leave-one-day-out sweep silently caps the fold set at the host's own days.
    """
    for seg in segs:
        if any(day == wanted for day, _lo, _hi in day_spans(seg)):
            return seg
    return None


def _find_day(ranked: list[tuple[date, int, int, float]], wanted: date
              ) -> tuple[date, int, int]:
    for day, lo, hi, _ in ranked:
        if day == wanted:
            return day, lo, hi
    raise SystemExit(
        f"day {wanted} is not a complete local day in the record; complete days are "
        f"{[str(r[0]) for r in ranked]}"
    )


# ---------------------------------------------------------------------------- #
# Split construction.
# ---------------------------------------------------------------------------- #
def build_splits(
    segs: list[Segment], metric: str,
    val_day: date | None = None, cal_day: date | None = None,
) -> dict[str, Any]:
    """Reserve the calibration and validation days, and train on what is left.

    The segment holding ``val_day`` hosts the reservation; absent a request, the
    longest one does (a personal record is normally one contiguous run). Any other
    segment is training data in full. Each reserved day
    is withheld together with the day before it, which its windows read as context —
    so the reserved index set is a 2-day block per role, coincident when the
    calibration day immediately precedes the validation day (the default).

    Days outside the host are still SCORED, and still count toward the median that
    picks the validation day, even though the reservation cannot be placed in them.

    Returns:
        Dict with ``train``/``cal``/``test`` Segment lists plus the ``ranked`` day
        table for the host, ``off_host`` for the rest, the chosen
        ``val_day``/``cal_day``, and the reserved spans.
    """
    assert segs, "no segments to split"
    if val_day is None:
        host = max(segs, key=len)
    else:
        found = _host_for_day(segs, val_day)
        if found is None:
            raise SystemExit(
                f"--val-day {val_day} is not a complete local day in any segment; "
                f"complete days are "
                f"{[str(d) for s in segs for d, _lo, _hi in day_spans(s)]}"
            )
        host = found
    others = [s for s in segs if s is not host]

    ranked = rank_days(host, metric)
    off_host = [(s_i, day, chaos)
                for s_i, s in enumerate(others)
                for day, _lo, _hi, chaos in rank_days(s, metric)]
    all_scores = [r[3] for r in ranked] + [o[2] for o in off_host]

    if val_day is not None:
        v_day, v_lo, v_hi = _find_day(ranked, val_day)
        assert v_lo >= 2 * STEPS_PER_DAY, (
            f"--val-day {v_day} lacks the two preceding full days the reservation needs"
        )
    else:
        v_day, v_lo, v_hi = choose_validation_day(ranked, all_scores)

    if cal_day is not None:
        c_day, c_lo, c_hi = _find_day(ranked, cal_day)
        assert c_lo >= STEPS_PER_DAY, (
            f"--cal-day {c_day} lacks the preceding full day its windows read as context"
        )
        # A calibration day AFTER the scored day would have the validation day itself as
        # its context, fitting the conformal half-width on the very windows it widens.
        assert c_day < v_day, (
            f"--cal-day {c_day} must precede --val-day {v_day}; a later calibration day "
            "reads the scored day as its own context and contaminates the conformal fit"
        )
    else:
        c_day, c_lo, c_hi = _find_day(ranked, v_day - timedelta(days=1))

    # Each role reserves its day plus the day before it (that day's context).
    cal_block = (c_lo - STEPS_PER_DAY, c_hi)
    val_block = (v_lo - STEPS_PER_DAY, v_hi)
    cal_segs = [_slice(host, *cal_block)]
    test_segs = [_slice(host, *val_block)]

    reserved = np.zeros(len(host), dtype=bool)
    for lo, hi in (cal_block, val_block):
        reserved[lo:hi] = True

    train_spans: list[tuple[int, int]] = []
    i = 0
    n = len(host)
    while i < n:
        if reserved[i]:
            i += 1
            continue
        j = i
        while j < n and not reserved[j]:
            j += 1
        if j - i >= MIN_TRAIN_STEPS:
            train_spans.append((i, j))
        i = j
    train_segs = [_slice(host, a, b) for a, b in train_spans] + others

    return {
        'train': train_segs, 'cal': cal_segs, 'test': test_segs,
        'ranked': ranked, 'off_host': off_host, 'median': float(np.median(all_scores)),
        'val_day': v_day, 'cal_day': c_day,
        'cal_block': cal_block, 'val_block': val_block,
        'train_spans': train_spans, 'host_steps': n, 'n_other_segs': len(others),
    }


def _eval_report_and_select(
    model: Any, ema: Any, stats: dict[str, Any],
    cal_segs: list[Segment], test_segs: list[Segment], device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score the validation day for REPORTING and the calibration day for SELECTION.

    Windows are collected once under the EMA shadow and scored twice with the roles
    swapped, so the two suites cost one set of forward passes rather than two.

    The selector reads only the point metric ``F.SELECTION_METRIC`` off the returned
    selection suite, which is computed on the median line and does not depend on the
    conformal fit — so it is immaterial that the swapped call fits that suite's bands
    on the validation windows. What matters is the converse: the REPORTED suite is
    scored on windows no checkpoint decision ever looked at.

    Returns:
        ``(report, select)`` — ``evaluate_from_windows`` results for the validation
        and calibration blocks respectively.
    """
    with ema.apply_to(model):
        model.eval()
        cal_w = collect_windows(
            model, stats, cal_segs, device,
            stride_patches=F.FINETUNE_EVAL_CAL_STRIDE_PATCHES,
            max_per_patient=F.FINETUNE_EVAL_MAX_PER_PATIENT)
        test_w = collect_windows(
            model, stats, test_segs, device,
            stride_patches=F.FINETUNE_EVAL_TEST_STRIDE_PATCHES,
            max_per_patient=F.FINETUNE_EVAL_MAX_PER_PATIENT)
        report = evaluate_from_windows(cal_w, test_w)
        select = evaluate_from_windows(test_w, cal_w)
        model.train()
    return report, select


def _db_fingerprint(path: str) -> dict[str, Any]:
    """Basename, size and sha256 of the source record.

    Deliberately NOT the absolute path: a personal checkpoint should identify the
    record it was fitted to without carrying a home directory into a file that may
    be copied elsewhere.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return {'name': os.path.basename(path), 'bytes': os.path.getsize(path),
            'sha256': h.hexdigest()}


# ---------------------------------------------------------------------------- #
# CLI.
# ---------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a T1DMAI checkpoint on a personal T1DMSERVER record.")
    p.add_argument('--checkpoint', required=True, help="pretrained checkpoint .pt path")
    p.add_argument('--db', required=True,
                   help="path to the personal sync database (read-only; never modified)")
    p.add_argument('--skip-hours', type=float, default=DEFAULT_SKIP_HOURS,
                   help="hours dropped from the START of the record at load time")
    p.add_argument('--exclude-range', nargs=2, action='append', default=None,
                   metavar=('START', 'END'),
                   help="drop the CGM readings in this half-open LOCAL wall-clock range "
                        "(ISO, e.g. 2026-07-30T11:10 2026-07-31T08:10); events keep their "
                        "curves, so the range reads as a CGM outage. Repeatable.")
    p.add_argument('--val-day', default=None,
                   help="ISO date to hold out (default: the day nearest median variability)")
    p.add_argument('--cal-day', default=None,
                   help="ISO date for the conformal fit (default: the day before --val-day)")
    p.add_argument('--chaos-metric', default=DEFAULT_CHAOS_METRIC, choices=list(CHAOS_METRICS))
    p.add_argument('--eval-stride-patches', type=int, default=PERSONAL_EVAL_STRIDE_PATCHES)
    p.add_argument('--patient', default='self', help="provenance label on every Segment")
    p.add_argument('--steps', type=int, default=F.FINETUNE_TOTAL_STEPS)
    p.add_argument('--batch-size', type=int, default=F.FINETUNE_BATCH_SIZE)
    p.add_argument('--lr-scale', type=float, default=F.FINETUNE_LR_SCALE)
    p.add_argument('--warmup', type=int, default=F.FINETUNE_WARMUP_STEPS)
    p.add_argument('--eval-interval', type=int, default=F.FINETUNE_EVAL_INTERVAL)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=F.FINETUNE_SEED)
    p.add_argument('--out', default=None, help="output checkpoint path (default: auto)")
    p.add_argument('--write-best', action='store_true',
                   help="also write checkpoints/t1dmai_best.pt for the metrics/ suite")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    total_steps = int(args.steps)
    batch_size = int(args.batch_size)
    lr_scale = float(args.lr_scale)
    warmup_steps = int(args.warmup)
    eval_interval = int(args.eval_interval)
    seed = int(args.seed)
    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    # F.eval_heldout reads these at call time; a personal record is far smaller than a
    # published cohort, so both eval splits are strided densely rather than at the
    # cohort defaults (4 / 8 patches). Set here so the EMA-shadow discipline in
    # F.eval_heldout is reused rather than reimplemented.
    stride = int(args.eval_stride_patches)
    assert stride >= 1, f"--eval-stride-patches must be >= 1, got {stride}"
    F.FINETUNE_EVAL_CAL_STRIDE_PATCHES = stride
    F.FINETUNE_EVAL_TEST_STRIDE_PATCHES = stride

    # --- Record ----------------------------------------------------------- #
    exclude = [(datetime.fromisoformat(lo), datetime.fromisoformat(hi))
               for lo, hi in (args.exclude_range or [])]
    segs_all = load_personal(args.db, skip_hours=float(args.skip_hours),
                             patient=str(args.patient), exclude=exclude)
    assert len(segs_all) > 0, (
        f"no segments loaded from {args.db} — the record may be shorter than "
        f"--skip-hours, or every run shorter than the minimum segment length"
    )
    split = build_splits(
        segs_all, args.chaos_metric,
        val_day=date.fromisoformat(args.val_day) if args.val_day else None,
        cal_day=date.fromisoformat(args.cal_day) if args.cal_day else None,
    )
    ft_segs, cal_segs, test_segs = split['train'], split['cal'], split['test']

    assert len(ft_segs) > 0, (
        "the reservation left no training span long enough for a window "
        f"(need >= {MIN_TRAIN_STEPS} steps); the record is too short to hold out a day"
    )
    if not F._has_eval_window(test_segs):
        raise SystemExit(
            f"validation day {split['val_day']} yields no eval window (need "
            f"{F._CTX_EVAL_STEPS + F._PRED_STEPS} contiguous steps) — the reservation "
            "did not land on a contiguous run"
        )

    print(f"[personal] db={os.path.basename(args.db)} skip_hours={args.skip_hours} "
          f"device={device.type} metric={args.chaos_metric}")
    for lo, hi in exclude:
        print(f"[personal] excluded readings: {lo} .. {hi} "
              f"({(hi - lo).total_seconds() / 3600:.2f} h)")
    print(f"[personal] record: {len(segs_all)} segment(s), host {split['host_steps']} steps "
          f"({split['host_steps'] * GRID_MIN / 60:.1f} h)")
    print(f"[personal] per-day {args.chaos_metric} (* = eligible to hold out; "
          f"record median {split['median']:.2f}):")
    for day, lo, _hi, chaos in split['ranked']:
        mark = '*' if lo >= 2 * STEPS_PER_DAY else ' '
        role = ('  <- validation' if day == split['val_day']
                else '  <- calibration' if day == split['cal_day'] else '')
        print(f"           {mark} {day}  {chaos:6.2f}{role}")
    # Days in shorter segments count toward the median but cannot host the reservation;
    # printing them keeps the table an account of the whole record, not of one run.
    for s_i, day, chaos in split['off_host']:
        print(f"             {day}  {chaos:6.2f}  (segment {s_i + 2}, training only)")
    print(f"[personal] reserved cal={split['cal_block']} val={split['val_block']}; "
          f"train spans={split['train_spans']} (+{split['n_other_segs']} other segment(s))")

    # --- Model ------------------------------------------------------------ #
    model, ckpt, stats = F.load_checkpoint(args.checkpoint, device)
    arch_version = ckpt.get('arch_version')
    loss_schema = ckpt.get('loss_schema')

    # Learned Kendall-Gal weighting for the pinball/DILATE combine; its two
    # log-variance scalars train in a dedicated AdamW group (build_optimizers) and are
    # EMA-excluded by living off-model (never passed to ModelEMA).
    weighting = F.KendallGalWeighting().to(device)

    # --- Training data ---------------------------------------------------- #
    dataset = F.RealSegmentDataset(ft_segs, stats, seed=seed)
    assert len(dataset) > 0, (
        f"no training windows from {len(ft_segs)} segment(s) — spans shorter than "
        f"{MIN_TRAIN_STEPS} steps"
    )
    if len(dataset) < batch_size:
        print(f"[personal] training windows ({len(dataset)}) < batch_size ({batch_size}); "
              f"clamping batch_size to {len(dataset)}")
        batch_size = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0, collate_fn=F.collate_fn,
                        pin_memory=(device.type == 'cuda'))

    # --- Optimizers / schedule / EMA -------------------------------------- #
    muon_lr = F.config.MUON_LR * lr_scale
    adam_lr = F.config.ADAM_LR * lr_scale
    muon_opt, adam_opt = F.build_optimizers(model, weighting, muon_lr, adam_lr)
    ema = F.ModelEMA(model, decay=F.FINETUNE_EMA_DECAY).to(device)

    # --- Logging ---------------------------------------------------------- #
    log_path = os.path.join(F.REPO_ROOT, 'finetune', 'finetune_personal_log.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    header = ['step', 'loss_ema']
    for h in F.EVAL_HORIZONS:
        header += [f'rmse_point_{h}', f'mae_point_{h}', f'mard_{h}', f'skill_point_{h}',
                   f'clarke_AB_{h}', f'hypo_recall_{h}']
    log_writer.writerow(header)
    log_file.flush()

    print(f"[personal] ft_segs={len(ft_segs)} windows={len(dataset)} "
          f"cal_segs={len(cal_segs)} test_segs={len(test_segs)} "
          f"steps={total_steps} bs={batch_size} lr_scale={lr_scale} eval_stride={stride}")

    best_sel: float | None = None
    best_step: int = -1
    best_model_sd: dict[str, torch.Tensor] | None = None
    best_ema_sd: dict[str, torch.Tensor] | None = None
    best_weighting_sd: dict[str, torch.Tensor] | None = None
    best_summ: dict[str, Any] | None = None
    baseline_summ: dict[str, Any] | None = None

    label = F._output_label(args.checkpoint)
    out_default = os.path.join(F.REPO_ROOT, 'finetune', f"{label}-finetune-personal.pt")
    out_path = args.out if args.out is not None else out_default
    db_fp = _db_fingerprint(args.db)

    def _write_output() -> None:
        assert (best_model_sd is not None and best_ema_sd is not None
                and best_weighting_sd is not None)
        save_dict = {
            'arch_version': arch_version, 'loss_schema': loss_schema, 'step': best_step,
            'model_state_dict': best_model_sd, 'model_ema_state_dict': best_ema_sd,
            'weighting_state_dict': best_weighting_sd,
            'normalization_stats': stats,
            'finetune_meta': {
                'dataset': 'personal', 'mode': 'personalize',
                'source_db': db_fp, 'skip_hours': float(args.skip_hours),
                'excluded_ranges': [[lo.isoformat(), hi.isoformat()] for lo, hi in exclude],
                'chaos_metric': args.chaos_metric,
                'val_day': str(split['val_day']), 'cal_day': str(split['cal_day']),
                'eval_stride_patches': stride,
                'base_checkpoint': os.path.abspath(args.checkpoint),
                'total_steps': total_steps, 'lr_scale': lr_scale,
                'baseline_heldout': baseline_summ, 'best_heldout': best_summ,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        torch.save(save_dict, out_path)
        if args.write_best:
            bd = os.path.join(F.REPO_ROOT, 'checkpoints')
            os.makedirs(bd, exist_ok=True)
            torch.save(save_dict, os.path.join(bd, 't1dmai_best.pt'))

    loss_ema: float | None = None

    def _log_eval(step: int, res: dict[str, Any], sel_res: dict[str, Any]) -> None:
        """Log the REPORTED (validation-day) suite; ``sel_res`` supplies the selector's own scalar.

        The console carries the four point metrics — RMSE, MAE, MARD and skill against
        persistence — because they are the ones this held-out set can actually resolve.
        Clarke A+B and hypo recall stay in the CSV but off the console: one validation
        day gives 45 heavily overlapping windows and two distinct excursions below
        70 mg/dL, so recall moves in steps of 0.5 and Clarke in steps of 1/45. Printing
        them each eval invites reading a coin flip as a trend.
        """
        row: list[Any] = [step, '' if loss_ema is None else f"{loss_ema:.4f}"]
        cells: list[str] = []
        for h in F.EVAL_HORIZONS:
            rmse = F._metric(res, h, 'rmse_point'); mae = F._metric(res, h, 'mae_point')
            mard = F._metric(res, h, 'mard'); skill = F._metric(res, h, 'skill_point')
            cab = F._metric(res, h, 'clarke_AB'); hyp = F._metric(res, h, 'hypo', 'recall')
            row += ['' if v is None else f"{v:.6f}"
                    for v in (rmse, mae, mard, skill, cab, hyp)]
            cells.append(f"h{h} rmse={F._fmt(rmse)} mae={F._fmt(mae)} "
                         f"mard={F._fmt(mard)} skill={F._fmt(skill)}")
        log_writer.writerow(row); log_file.flush()
        sel = F._selection_scalar(sel_res)
        tag = 'baseline' if step == 0 else f"step {step}"
        print(f"[eval {tag}] loss_ema={F._fmt(loss_ema)} n_test={res.get('n_test_windows')} "
              f"sel_cal={F._fmt(sel)} | " + " | ".join(cells))

    def _maybe_select(step: int, res: dict[str, Any], sel_res: dict[str, Any]) -> None:
        nonlocal best_sel, best_step, best_model_sd, best_ema_sd, best_weighting_sd, best_summ
        sel = F._selection_scalar(sel_res)
        if sel is None:
            return
        if best_sel is None or sel < best_sel:
            best_sel = sel; best_step = step
            best_model_sd = F._cpu_state(model.state_dict())
            best_ema_sd = F._cpu_state(ema.state_dict())
            best_weighting_sd = F._cpu_state(weighting.state_dict())
            best_summ = F._summarize_heldout(res)
            _write_output()
            print(f"  [best] new minimum {F.SELECTION_METRIC}@{F.SELECTION_HORIZON}={sel:.3f} "
                  f"on the CAL day at step {step} -> wrote {out_path}")

    try:
        # --- Baseline (pre-finetune) eval at step 0 ----------------------- #
        res0, sel0 = _eval_report_and_select(model, ema, stats, cal_segs, test_segs, device)
        baseline_summ = F._summarize_heldout(res0)
        _log_eval(0, res0, sel0)
        _maybe_select(0, res0, sel0)

        # --- Fine-tune loop ----------------------------------------------- #
        data_iter = iter(loader)
        consecutive_nan = 0
        step = 0
        while step < total_steps:
            model.train()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            patches = batch['patches'].to(device, non_blocking=True)
            attn_mask = batch['attn_mask'].to(device, non_blocking=True)
            last_bg = batch['bg_formula_data']['last_bg'].to(device).float()
            true_bg_full = (batch['bg_formula_data']['true_bg_trajectory'][:, :F._PRED_STEPS]
                            .to(device).float()
                            .reshape(-1, F.config.PREDICTION_PATCHES, F.config.PATCH_SIZE))

            # Cross-window probe input (window k+1) — skipped when the penalty is off.
            next_window = None
            if F.config.TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0:
                _nw = batch.get('next_window')
                if _nw is not None:
                    next_window = {
                        'patches': _nw['patches'].to(device, non_blocking=True),
                        'last_bg': _nw['last_bg'].to(device, non_blocking=True).float(),
                        'valid': _nw['valid'].to(device, non_blocking=True),
                    }

            def _halve_optimizer_state() -> None:
                for grp in muon_opt.param_groups:
                    for p in grp['params']:
                        st = muon_opt.state.get(p, {})
                        if 'momentum_buffer' in st:
                            st['momentum_buffer'].mul_(0.5)
                for grp in adam_opt.param_groups:
                    for p in grp['params']:
                        st = adam_opt.state.get(p, {})
                        if 'exp_avg' in st:
                            st['exp_avg'].mul_(0.5)

            def _maybe_restore_from_ema(reason: str) -> bool:
                nonlocal consecutive_nan
                if consecutive_nan >= F.CONSECUTIVE_NAN_RESTORE:
                    model.load_state_dict(ema.state_dict(), strict=False)
                    muon_opt.state.clear()
                    adam_opt.state.clear()
                    print(f"  [RECOVERY] {consecutive_nan} consecutive {reason} — "
                          f"restored from EMA")
                    consecutive_nan = 0
                    return True
                return False

            def _skip_nonfinite_step(reason: str) -> None:
                nonlocal consecutive_nan, loss_ema
                consecutive_nan += 1
                print(f"  [WARNING] {reason} at step {step} "
                      f"(consecutive: {consecutive_nan}) — skipping")
                _maybe_restore_from_ema(reason)
                muon_opt.zero_grad(set_to_none=True)
                adam_opt.zero_grad(set_to_none=True)
                _halve_optimizer_state()
                loss_ema = 1.0 if loss_ema is None else 0.98 * loss_ema + 0.02 * 1.0

            try:
                q_tau, median, time_pred = model(patches, attn_mask, last_bg, return_time=True)
                q_tau = q_tau.float()
                median = median.float()
                loss_total, _parts = F.risk_total_loss(q_tau, median, true_bg_full, weighting)

                # Time-of-day probe co-trains the shared trunk (TIME_PROBE_DETACH=False),
                # mirroring pretraining; isolated from the logged/EMA/selection scalar.
                _tod_extra = loss_total.new_zeros(())
                _tod_loss_val = float('nan')
                _psh = batch['bg_formula_data'].get('pred_start_hour')
                if time_pred is not None and _psh is not None:
                    P = time_pred.shape[1]
                    adv = F.config.PATCH_SIZE * 5.0 / 60.0
                    base = _psh.to(time_pred.device).float().reshape(-1)                       # (B,)
                    offs = torch.arange(P, device=time_pred.device, dtype=torch.float32) * adv  # (P,)
                    tgt_hours = (base[:, None] + offs[None, :]) % 24.0                          # (B,P)
                    _tod_ce = F.time_of_day_bin_ce(
                        time_pred, tgt_hours, F.config.TIME_PROBE_N_BINS,
                        F.config.TIME_PROBE_LABEL_SMOOTH_BINS,
                    )
                    _tod_loss = _tod_ce
                    _tod_loss_val = float(_tod_ce.detach())   # logged CE (pre cross-window)

                    # Cross-window (paired-window) phase-advance penalty (2nd forward
                    # on batch['next_window']) — same coupling pretraining applies.
                    if (next_window is not None
                            and F.config.TIME_PROBE_CROSS_WINDOW_WEIGHT > 0.0
                            and bool(next_window['valid'].any())):
                        B_nw = next_window['patches'].shape[0]
                        n_sub = (B_nw if F.config.TIME_PROBE_CROSS_WINDOW_FRACTION >= 1.0
                                 else max(1, math.ceil(
                                     F.config.TIME_PROBE_CROSS_WINDOW_FRACTION * B_nw)))
                        nw_valid_s = next_window['valid'][:n_sub]
                        if bool(nw_valid_s.any()):
                            nw_mask = attn_mask[:n_sub] if attn_mask.dim() == 3 else attn_mask
                            _, _, time_pred_next = model(
                                next_window['patches'][:n_sub], nw_mask,
                                next_window['last_bg'][:n_sub], return_time=True,
                            )
                            _tod_xwin = F.time_cross_window_consistency_loss(
                                time_pred[:n_sub], time_pred_next, F.config.TIME_PROBE_N_BINS,
                                F.config.PREDICTION_HORIZON_HOURS, valid=nw_valid_s,
                            )
                            if torch.isfinite(_tod_xwin):
                                _tod_loss = (_tod_loss
                                             + F.config.TIME_PROBE_CROSS_WINDOW_WEIGHT * _tod_xwin)
                    if torch.isfinite(_tod_loss):
                        _tod_extra = F.config.TIME_PROBE_LOSS_WEIGHT * _tod_loss
                loss_backward = loss_total + _tod_extra

                if not torch.isfinite(loss_backward):
                    _skip_nonfinite_step("NaN/Inf total loss")
                    step += 1
                    continue
                loss_backward.backward()
            except RuntimeError as exc:
                muon_opt.zero_grad(set_to_none=True)
                adam_opt.zero_grad(set_to_none=True)
                _skip_nonfinite_step(f"forward/loss/backward RuntimeError ({exc})")
                step += 1
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(weighting.parameters()),
                F.config.GRADIENT_CLIP_NORM, error_if_nonfinite=False)

            if torch.isfinite(grad_norm):
                F.update_lr(muon_opt, adam_opt, step, muon_lr, adam_lr,
                            warmup_steps, total_steps, F.FINETUNE_LR_MIN_RATIO)
                muon_opt.step()
                adam_opt.step()
                ema.update(model)
                consecutive_nan = 0
                loss_val = float(loss_total.item())
                loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
            else:
                consecutive_nan += 1
                print(f"  [WARNING] NaN/Inf gradient at step {step} "
                      f"(consecutive: {consecutive_nan})")
                _maybe_restore_from_ema("NaN gradients")
                _halve_optimizer_state()
            muon_opt.zero_grad(set_to_none=True)
            adam_opt.zero_grad(set_to_none=True)

            step += 1
            if step % F.FINETUNE_LOG_INTERVAL == 0:
                print(f"  step {step}/{total_steps} loss_ema={F._fmt(loss_ema)} "
                      f"loss_tod={F._fmt(_tod_loss_val)} "
                      f"lr_muon={muon_opt.param_groups[0]['lr']:.2e}")
            if step % eval_interval == 0 and step < total_steps:
                res, sel_res = _eval_report_and_select(
                    model, ema, stats, cal_segs, test_segs, device)
                _log_eval(step, res, sel_res)
                _maybe_select(step, res, sel_res)

        # --- Final eval --------------------------------------------------- #
        res_final, sel_final = _eval_report_and_select(
            model, ema, stats, cal_segs, test_segs, device)
        _log_eval(total_steps, res_final, sel_final)
        _maybe_select(total_steps, res_final, sel_final)
    finally:
        log_file.close()

    if best_model_sd is not None:
        _write_output()

    # --- Final report ----------------------------------------------------- #
    print()
    print("=" * 72)
    print(f"output checkpoint: {out_path}")
    print(f"best step: {best_step}  ({F.SELECTION_METRIC}@{F.SELECTION_HORIZON}="
          f"{F._fmt(best_sel)} on the CAL day {split['cal_day']})")
    print(f"held-out day: {split['val_day']}  (never read by the selector; conformal fit "
          f"on {split['cal_day']}, {args.chaos_metric} nearest the record median)")
    print("per-horizon held-out  baseline -> best (Δ):")
    _REPORTED = (('rmse_point', 'rmse'), ('mae_point', 'mae'),
                 ('mard', 'mard'), ('skill_point', 'skill'))
    for h in F.EVAL_HORIZONS:
        for key, label in _REPORTED:
            base = None if baseline_summ is None else baseline_summ[str(h)].get(key)
            best = None if best_summ is None else best_summ[str(h)].get(key)
            delta = (None if (base is None or best is None) else best - base)
            print(f"  h{h:>3} {label:<6}: {F._fmt(base):>8} -> {F._fmt(best):>8}"
                  + ("" if delta is None else f"  (Δ {delta:+.3f})"))
    print("-" * 72)
    print("One day of one person is a narrow test: it fixes a single day's meals, doses "
          "and sleep, so a gain here is evidence of adaptation, not of generalization. "
          "Re-run with a different --val-day to see how much of it is the day.")
    if args.write_best:
        print()
        print("checkpoints/t1dmai_best.pt was overwritten with a SINGLE-PERSON model. "
              "metrics/rebuild_all.sh reads that path and does not record which checkpoint "
              "produced its numbers, so any cohort figure regenerated from here on describes "
              "this one record — restore a cohort checkpoint before publishing one.")
    print("=" * 72)


if __name__ == '__main__':
    main()
