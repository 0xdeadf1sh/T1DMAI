"""The simulator's own dose response and noise-free tail roughness, measured.

Owns the counterfactual probe's dose definitions, so probe and reference cannot drift.
Writes ``sim_dose_reference.json``, every ``ref`` the validation table shows against a
``cf_*`` row. Regenerate after any T1DMSIM change."""

import argparse
import concurrent.futures as _futures
import copy
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np

from config import (
    CF_CARB_BOLUS_G, CF_INSULIN_BOLUS_U, PATCH_SIZE, PREDICTION_PATCHES,
)
from T1DMSIM.simulator import (
    BG_SCALE_FACTOR, DT_MINUTES, bolus_pk_for_dose, gamma_curve,
)
from T1DMSIM.cache_simulator import (
    DEFAULT_CONTEXT_STEPS, DEFAULT_WARMUP_HOURS, DEFAULT_WARMUP_OFFSET_STEPS,
    EVENT_REFRACTORY_MIN, RowConfig, TAIL_ARMS, boundary_dose, simulate_to_boundary,
)

REFERENCE_FILE = 'sim_dose_reference.json'
SCHEMA = 'sim-dose-reference-v1'

# Dose ladder, in multiples of the CF_* dose; every rung carries its own dose-scaled curve.
_CF_LADDER = (0.5, 1.0, 2.0)
_CF_REF_RUNG = _CF_LADDER.index(1.0)
_CF_ONSET_MGDL = 5.0                                # response reached = |ΔBG| past this
_CF_PRE_ACTION_STEPS = 15 // DT_MINUTES             # window in which a bolus cannot yet act
_CF_LINEARITY_FLOOR_MGDL = 5.0                      # |Δ| at the 1× rung below which no ratio

# The injected arm: carbohydrate at the top of the GI range, insulin into a perfect site.
_CF_CARB_GI = 100.0
_CF_BOLUS_ANALOGUE = 'aspart'
_CF_BOLUS_SITE_FACTOR = 1.0

# Every figure the table reads, keyed by the val_metrics column it is the reference for.
REFERENCE_KEYS: "tuple[str, ...]" = (
    'cf_carb_gain', 'cf_insulin_gain',
    'cf_carb_linearity', 'cf_insulin_linearity',
    'cf_meal_coverage',
    'cf_carb_onset_lag_min', 'cf_insulin_onset_lag_min',
    'cf_insulin_preaction_dbg',
    'median_roughness', 'median_roughness_far',
)

# The probe's one mean-over-windows figure; every other is a median over rows.
_MEAN_FIGURES = frozenset({'cf_insulin_preaction_dbg'})


def _cf_bolus_curve(channel: str, total: float, n_steps: int) -> np.ndarray:
    """Per-step curve for a counterfactual bolus of total, truncated to n_steps.
    Carb at GI 100, insulin under dose-scaled PK (SPEC/invariants.md §5), so the probe
    injects the shape the model was pretrained on.
    """
    if channel == 'carb':
        curve = gamma_curve(total, 2.0, 15.0, 120.0)
    elif channel == 'insulin':
        curve = gamma_curve(total, *bolus_pk_for_dose(total))
    else:
        raise ValueError(f"unknown counterfactual channel {channel!r}")
    out = np.zeros(n_steps, dtype=np.float32)
    n = min(n_steps, int(curve.shape[0]))
    out[:n] = curve[:n]
    return out


def _cf_onset_step(signed: np.ndarray) -> "int | None":
    """First step where the intended-direction response reaches _CF_ONSET_MGDL, else None."""
    hit = np.nonzero(signed >= _CF_ONSET_MGDL)[0]
    return int(hit[0]) if hit.size else None


def _cf_median(v: "list[float]") -> "float | None":
    return float(np.median(v)) if v else None


def load_reference(path: str = REFERENCE_FILE) -> "dict[str, Any] | None":
    """The measured reference, or None when the file is absent or not this schema."""
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
        return None
    return doc


def figure(doc: "dict[str, Any] | None", key: str) -> "dict[str, Any] | None":
    """One figure of a loaded reference, or None where it carries no finite value."""
    if not doc:
        return None
    fig = (doc.get('figures') or {}).get(key)
    if not isinstance(fig, dict) or not isinstance(fig.get('value'), (int, float)):
        return None
    return fig


class _ZeroNoiseRng:
    """A generator whose every normal draw is its own mean; everything else delegates."""

    def __init__(self, rng: Any) -> None:
        self._rng = rng

    def __getattr__(self, name: str) -> Any:
        return getattr(self._rng, name)

    def normal(self, loc: Any = 0.0, scale: Any = 1.0, size: Any = None) -> Any:
        return np.full(size, loc, dtype=np.float64) if size is not None else float(loc)


def row_config(context_steps: int = DEFAULT_CONTEXT_STEPS,
               seed_salt: int = 42) -> RowConfig:
    """The cache geometry the reference is measured on: its rows, its tail, its warm-up."""
    return RowConfig(
        warmup_steps=int(DEFAULT_WARMUP_HOURS * 60 / DT_MINUTES),
        context_steps=int(context_steps), max_attempts=1,
        rail_high=500.0, rail_low=0.0, hypo_prob=0.0, hypo_min_frac=0.0,
        hypo_threshold=70.0, seed_salt=int(seed_salt),
        event_refractory_steps=max(1, int(round(EVENT_REFRACTORY_MIN / DT_MINUTES))),
        events=False, tail_steps=PREDICTION_PATCHES * PATCH_SIZE,
        warmup_offset_steps=DEFAULT_WARMUP_OFFSET_STEPS)


def _run_tail(sim: Any, n_steps: int, inject: Any = None,
              zero_noise: bool = False) -> np.ndarray:
    """One behaviour-off tail of ``n_steps`` off a deep copy of ``sim``, as observed BG."""
    c = copy.deepcopy(sim)
    c.behaviour_enabled = False
    if zero_noise:
        c.rng = _ZeroNoiseRng(c.rng)
    if inject is not None:
        inject(c)
    return np.array([c.generate()['bg_observed'] for _ in range(n_steps)],
                    dtype=np.float64)


def _roughness_sums(tail: np.ndarray, far0: int) -> np.ndarray:
    """``|second difference|`` sum and count, whole horizon then last patch, as the table's."""
    from utils import kovatchev_f_np
    y = kovatchev_f_np(tail)
    d2 = np.abs(y[2:] - 2.0 * y[1:-1] + y[:-2])
    return np.array([d2.sum(), d2.size, d2[far0:].sum(), d2[far0:].size],
                    dtype=np.float64)


def measure_row(seed: int, cfg: RowConfig) -> "dict[str, Any]":
    """One patient's figures: the dose ladder against an undosed twin, then four quiet arms."""
    sim, _channels, stats, draws = simulate_to_boundary(seed, cfg)
    icr = float(stats['icr'])
    n = int(cfg.tail_steps)
    term = n - 1
    far0 = (PREDICTION_PATCHES - 1) * PATCH_SIZE - 1

    carb_ref = [BG_SCALE_FACTOR * np.cumsum(_cf_bolus_curve('carb', f * CF_CARB_BOLUS_G, n))
                for f in _CF_LADDER]
    ins_ref_per_icr = [
        BG_SCALE_FACTOR * np.cumsum(_cf_bolus_curve('insulin', f * CF_INSULIN_BOLUS_U, n))
        for f in _CF_LADDER]

    def _carbs(grams: float) -> Any:
        return lambda c: c.inject_carbs(grams, _CF_CARB_GI, grams, _CF_CARB_GI)

    def _bolus(units: float) -> Any:
        return lambda c: c.inject_bolus(units, _CF_BOLUS_ANALOGUE, _CF_BOLUS_SITE_FACTOR)

    base = _run_tail(sim, n)
    carb_d = [_run_tail(sim, n, _carbs(f * CF_CARB_BOLUS_G)) - base for f in _CF_LADDER]
    ins_d = [_run_tail(sim, n, _bolus(f * CF_INSULIN_BOLUS_U)) - base for f in _CF_LADDER]
    carb_1 = carb_d[_CF_REF_RUNG]
    ins_1 = ins_d[_CF_REF_RUNG]

    out: "dict[str, Any]" = {k: None for k in REFERENCE_KEYS}
    out['cf_carb_gain'] = float(carb_1[term] / carb_ref[_CF_REF_RUNG][term])
    out['cf_insulin_preaction_dbg'] = float(ins_1[:_CF_PRE_ACTION_STEPS].mean())
    if abs(carb_1[term]) >= _CF_LINEARITY_FLOOR_MGDL:
        out['cf_carb_linearity'] = float(carb_d[-1][term] / carb_1[term])
    if abs(ins_1[term]) >= _CF_LINEARITY_FLOOR_MGDL:
        out['cf_insulin_linearity'] = float(ins_d[-1][term] / ins_1[term])

    carb_onset = _cf_onset_step(carb_1)
    carb_onset_ref = _cf_onset_step(carb_ref[_CF_REF_RUNG])
    if carb_onset is not None and carb_onset_ref is not None:
        out['cf_carb_onset_lag_min'] = float((carb_onset - carb_onset_ref) * DT_MINUTES)

    if icr > 0.0:
        out['cf_insulin_gain'] = float(
            -ins_1[term] / (icr * ins_ref_per_icr[_CF_REF_RUNG][term]))
        ins_onset = _cf_onset_step(-ins_1)
        ins_onset_ref = _cf_onset_step(icr * ins_ref_per_icr[_CF_REF_RUNG])
        if ins_onset is not None and ins_onset_ref is not None:
            out['cf_insulin_onset_lag_min'] = float(
                (ins_onset - ins_onset_ref) * DT_MINUTES)
        matched_u = CF_CARB_BOLUS_G / icr

        def _meal(c: Any) -> None:
            _carbs(CF_CARB_BOLUS_G)(c)
            _bolus(matched_u)(c)

        meal_d = _run_tail(sim, n, _meal) - base
        if carb_1[term] >= _CF_LINEARITY_FLOOR_MGDL:
            out['cf_meal_coverage'] = float(1.0 - meal_d[term] / carb_1[term])

    dose = boundary_dose(sim, draws)
    sums = np.zeros(4, dtype=np.float64)
    for arm_name in TAIL_ARMS:
        def _arm(c: Any, name: str = arm_name) -> None:
            if 'bolus' in name:
                c.inject_bolus(dose.bolus_u, dose.analogue, dose.site_factor)
            if 'carbs' in name:
                c.inject_carbs(dose.carb_g, dose.carb_gi,
                               dose.logged_carb_g, dose.logged_carb_gi)

        sums += _roughness_sums(_run_tail(sim, n, _arm, zero_noise=True), far0)
    out['roughness_sums'] = sums.tolist()
    return out


def _spread(values: "list[float]", aggregate: str) -> "dict[str, Any]":
    a = np.asarray(values, dtype=np.float64)
    centre = float(a.mean()) if aggregate == 'mean' else float(np.median(a))
    return {'value': centre, 'p10': float(np.percentile(a, 10.0)),
            'p90': float(np.percentile(a, 90.0)), 'n': int(a.size)}


def _t1dmsim_commit() -> "dict[str, Any]":
    """The T1DMSIM checkout the measurement ran on, and whether it carried edits."""
    root = os.path.dirname(os.path.abspath(sys.modules['T1DMSIM.simulator'].__file__))
    try:
        head = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', root, 'status', '--porcelain'],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {'t1dmsim_commit': None, 't1dmsim_dirty': None}
    return {'t1dmsim_commit': head, 't1dmsim_dirty': dirty}


def measure(n_rows: int, workers: int, cfg: RowConfig) -> "dict[str, Any]":
    """The whole reference document over ``n_rows`` cache rows, across ``workers``."""
    from T1DMSIM.cache_simulator import _row_seed

    seeds = [_row_seed(i, 0, cfg.seed_salt) for i in range(n_rows)]
    if workers > 1:
        with _futures.ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(measure_row, seeds, [cfg] * n_rows, chunksize=4))
    else:
        rows = [measure_row(s, cfg) for s in seeds]

    figures: "dict[str, Any]" = {}
    for key in REFERENCE_KEYS:
        if key.startswith('median_roughness'):
            continue
        vals = [r[key] for r in rows if r.get(key) is not None]
        figures[key] = (_spread(vals, 'mean' if key in _MEAN_FIGURES else 'median')
                        if vals else {'value': None, 'p10': None, 'p90': None, 'n': 0})
    totals = np.sum([r['roughness_sums'] for r in rows], axis=0)
    for key, s, c in (('median_roughness', 0, 1), ('median_roughness_far', 2, 3)):
        figures[key] = {'value': float(totals[s] / totals[c]), 'p10': None,
                        'p90': None, 'n': int(totals[c])}

    return {
        'schema': SCHEMA,
        **_t1dmsim_commit(),
        'n_rows': int(n_rows),
        'tail_steps': int(cfg.tail_steps),
        'context_steps': int(cfg.context_steps),
        'seed_salt': int(cfg.seed_salt),
        'carb_bolus_g': float(CF_CARB_BOLUS_G),
        'insulin_bolus_u': float(CF_INSULIN_BOLUS_U),
        'ladder': list(_CF_LADDER),
        'figures': figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Measure the simulator dose reference.')
    parser.add_argument('--rows', type=int, default=240)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--context-steps', type=int, default=DEFAULT_CONTEXT_STEPS)
    parser.add_argument('--seed-salt', type=int, default=42)
    parser.add_argument('--out', type=str, default=REFERENCE_FILE)
    args = parser.parse_args()

    doc = measure(args.rows, max(1, min(8, args.workers)),
                  row_config(args.context_steps, args.seed_salt))
    with open(args.out, 'w') as fh:
        json.dump(doc, fh, indent=2)
        fh.write('\n')
    for key in REFERENCE_KEYS:
        f = doc['figures'][key]
        span = '' if f['p10'] is None else f"  p10-p90 {f['p10']:+.4f} {f['p90']:+.4f}"
        print(f"{key:28s} {f['value']:+.4f}  n={f['n']}{span}")
    print(f"wrote {args.out} at T1DMSIM {doc['t1dmsim_commit']}"
          f"{' (dirty)' if doc['t1dmsim_dirty'] else ''}")


if __name__ == '__main__':
    for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(_var, '1')
    main()
