"""
Meal statistics — simulator vs real (unaugmented & augmented).

Per cohort, meals are extracted, classified into breakfast / lunch / dinner / other
by time of day, and summarized: meals-per-day, mean±std hour, mean±std grams, plus
overall meals/day and carbs/day. The point is to compare the simulator's meal
SCHEDULE (timing + portion distribution it was trained on) against the real
cohorts — both as-logged and after metrics/augmented reconstruction injects the
meals the records omit.

Meal extraction
  sim    : the generator emits discrete meals; we capture every 'carb' pending
           event whose label starts 'Meal' (protein/fat tails and rescue
           correction_carb are excluded), grouped by meal step (components share
           one meal_idx), grams = curve area (gamma_curve integrates to grams).
  real   : carb-gram spikes clustered within MEAL_GAP_MIN into one meal; grams =
           sum, hour = carb-weighted mean; meals below MIN_MEAL_G dropped.
  aug    : augment_segment() first, then the real extractor.

Time windows (hour of day): breakfast [4,11) · lunch [11,16) · dinner [16,22) ·
other otherwise. Off-hours rescue carbs land in 'other'.
"""
from __future__ import annotations
import os, sys, json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch
torch.set_num_threads(8)

from realdata import load_dataset
from realdata.schema import GRID_MIN
from data import _make_simulator
from config import SIMULATOR_WARMUP_HOURS
from T1DMSIM.simulator import DT_MINUTES
from metrics.augmented.augment import augment_segment
from metrics import figstyle as F           # only ROOT is on sys.path here, so import by package

plt = F.plt
F.style()

SIM_SEEDS = list(range(8000, 8016))
SIM_HOURS = 120.0
SPD = int(24 * 60 / DT_MINUTES)                 # steps per day (288)
MEAL_GAP_MIN = 30                               # carb spikes within this = one meal
MIN_MEAL_G = 5.0                                # drop sub-meal carb specks
CLASSES = ('breakfast', 'lunch', 'dinner', 'other')


def classify(h: float) -> str:
    if 4.0 <= h < 11.0:
        return 'breakfast'
    if 11.0 <= h < 16.0:
        return 'lunch'
    if 16.0 <= h < 22.0:
        return 'dinner'
    return 'other'


class _RecordingEvents(list):
    """A drop-in _pending_events that also logs (meal_idx, grams) for meal carbs."""
    def __init__(self, *a):
        super().__init__(*a)
        self.meals: list[tuple[int, float]] = []

    def append(self, item):
        try:
            idx, etype, data = item
            if etype == 'carb' and isinstance(data, dict) \
                    and str(data.get('label', '')).startswith('Meal'):
                self.meals.append((int(idx), float(np.sum(data['curve']))))
        except Exception:
            pass
        super().append(item)


def sim_meals():
    """Return (meals, days): meals = list of (hour_of_day, grams) over the sim cohort."""
    n_warm = int(SIMULATOR_WARMUP_HOURS * 60 / DT_MINUTES)
    meals, days = [], 0.0
    for s in SIM_SEEDS:
        sim = _make_simulator(int(s), uniform_skills=False)
        rec = _RecordingEvents()
        sim._pending_events = rec
        raw = sim.generate_hours(SIM_HOURS + SIMULATOR_WARMUP_HOURS)
        hod = raw['hour_of_day']
        by_idx = defaultdict(float)
        for idx, g in rec.meals:
            by_idx[idx] += g
        for idx, g in by_idx.items():
            if idx < n_warm or idx >= len(hod) or g < MIN_MEAL_G:
                continue
            meals.append((float(hod[idx]), g))
        days += (len(hod) - n_warm) / SPD
    return meals, days


def real_meals(segs):
    """Cluster carb spikes into meals; return (meals, days) for a list of Segments."""
    gap = MEAL_GAP_MIN // GRID_MIN
    meals, days = [], 0.0
    for seg in segs:
        days += len(seg) / SPD
        carb = np.asarray(seg.carb_grams, dtype=float)
        hod = seg.hour_of_day()
        idx = np.where(carb > 0)[0]
        if len(idx) == 0:
            continue
        groups, cur = [], [idx[0]]
        for i in idx[1:]:
            if i - cur[-1] <= gap:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        for grp in groups:
            grp = np.asarray(grp)
            g = float(carb[grp].sum())
            if g < MIN_MEAL_G:
                continue
            meals.append((float(np.average(hod[grp], weights=carb[grp])), g))
    return meals, days


def summarize(meals, days):
    by = defaultdict(list)
    for h, g in meals:
        by[classify(h)].append((h, g))
    rows = {}
    for cls in CLASSES:
        items = by.get(cls, [])
        if items:
            H = np.array([h for h, _ in items]); G = np.array([g for _, g in items])
            rows[cls] = dict(per_day=len(items) / days, hour_mean=float(H.mean()),
                             hour_std=float(H.std()), g_mean=float(G.mean()), g_std=float(G.std()))
        else:
            rows[cls] = dict(per_day=0.0, hour_mean=float('nan'), hour_std=float('nan'),
                             g_mean=float('nan'), g_std=float('nan'))
    G = np.array([g for _, g in meals]) if meals else np.array([])
    rows['ALL'] = dict(per_day=len(meals) / days, carbs_per_day=float(G.sum() / days) if G.size else 0.0,
                       g_mean=float(G.mean()) if G.size else float('nan'))
    return rows


def _hhmm(h: float) -> str:
    if not np.isfinite(h):
        return '   —  '
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


REAL = ('ohiot1dm', 'azt1d', 'shanghai')
MEALS = ('breakfast', 'lunch', 'dinner')


def _figure(cohorts: dict) -> str:
    """Simulator meal schedule against the real cohorts, as logged and reconstructed.

    The simulator is the distribution the model trained on, not a fourth cohort, so
    it is drawn as a labelled reference rule rather than given a categorical hue —
    which also keeps the cohort key at three, inside the all-pairs series cap.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    x = np.arange(len(REAL))
    w = 0.24

    for ax, key, title, ylabel, fmt in (
            (axes[0, 0], 'per_day', 'meals per day', 'meals/day', '{:.2f}'),
            (axes[0, 1], 'carbs_per_day', 'carbohydrate per day', 'g/day', '{:.0f}')):
        for k, suffix in enumerate(('', '+aug')):
            vals = [cohorts[d + suffix]['ALL'][key] for d in REAL]
            bars = ax.bar(x + (k - 0.5) * w, vals, width=w - 0.04, color=F.PAIR[k],
                          edgecolor=F.SURFACE, linewidth=1.2)
            F.bar_labels(ax, bars, fmt)
        sim = cohorts['T1DMSIM']['ALL'][key]
        F.threshold(ax, sim, f'T1DMSIM {fmt.format(sim)}', side='left')
        F.ygrid(ax)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.margins(y=0.22)
        ax.set_xticks(x); ax.set_xticklabels([F.label(d) for d in REAL])

    # Meal size and clock hour are both strictly positive and right-skewed, so a
    # symmetric ± sd arm can reach below zero on the wide cohorts. The lower arm is
    # held at the origin rather than drawn into negative grams.
    def _err(m, s):
        return np.vstack([np.minimum(s, m), s])

    for ax, key, std_key, title, ylabel in (
            (axes[1, 0], 'hour_mean', 'hour_std', 'meal clock time (mean ± sd)', 'hour of day'),
            (axes[1, 1], 'g_mean', 'g_std', 'portion size (mean ± sd, clipped at zero)', 'grams')):
        xs = np.arange(len(MEALS))
        for k, d in enumerate(REAL):
            m = np.array([cohorts[d][c][key] for c in MEALS])
            s = np.array([cohorts[d][c][std_key] for c in MEALS])
            ax.errorbar(xs + (k - 1) * 0.16, m, yerr=_err(m, s), fmt='o', markersize=7,
                        color=F.cohort_color(d), ecolor=F.cohort_color(d), elinewidth=1.4,
                        capsize=4, markeredgecolor=F.SURFACE, markeredgewidth=2, zorder=3)
        m = np.array([cohorts['T1DMSIM'][c][key] for c in MEALS])
        s = np.array([cohorts['T1DMSIM'][c][std_key] for c in MEALS])
        ax.errorbar(xs + 2 * 0.16, m, yerr=_err(m, s), fmt='D', markersize=6, color=F.REFERENCE,
                    ecolor=F.REFERENCE, elinewidth=1.4, capsize=4, markeredgecolor=F.SURFACE,
                    markeredgewidth=2, zorder=3, label='T1DMSIM')
        F.ygrid(ax)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_xticks(xs); ax.set_xticklabels(MEALS); ax.margins(x=0.14)
        F.legend(ax, loc='best')

    F.pair_legend(axes[0, 0], 'as logged', 'reconstructed', loc='upper left')
    fig.suptitle('Meal schedule — simulator vs real cohorts', x=0.006, ha='left',
                 fontsize=12, color=F.INK)
    F.cohort_legend(fig, list(REAL), y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return F.save(fig, 'meal_stats.png')


def main():
    cohorts = {}
    print("[meal_stats] simulator…", flush=True)
    cohorts['T1DMSIM'] = summarize(*sim_meals())
    for ds in ('ohiot1dm', 'azt1d', 'shanghai'):
        print(f"[meal_stats] {ds}…", flush=True)
        segs = load_dataset(ds)
        cohorts[ds] = summarize(*real_meals(segs))
        cohorts[f'{ds}+aug'] = summarize(*real_meals([augment_segment(s) for s in segs]))

    order = ['T1DMSIM', 'ohiot1dm', 'ohiot1dm+aug', 'azt1d', 'azt1d+aug', 'shanghai', 'shanghai+aug']

    for cls in ('breakfast', 'lunch', 'dinner'):
        print(f"\n=== {cls.upper()} ===")
        print(f"{'cohort':14} {'/day':>5}  {'hour (mean±std)':>20}  {'carbs g (mean±std)':>20}")
        print('-' * 66)
        for name in order:
            r = cohorts[name][cls]
            hh = f"{_hhmm(r['hour_mean'])} ± {r['hour_std']*60:4.0f}m" if np.isfinite(r['hour_mean']) else '—'
            gg = f"{r['g_mean']:5.1f} ± {r['g_std']:4.1f}" if np.isfinite(r['g_mean']) else '—'
            print(f"{name:14} {r['per_day']:5.2f}  {hh:>20}  {gg:>20}")

    print("\n=== SUMMARY (all meals incl. snacks/other) ===")
    print(f"{'cohort':14} {'meals/day':>9} {'carbs g/day':>12} {'mean meal g':>12}")
    print('-' * 52)
    for name in order:
        a = cohorts[name]['ALL']
        print(f"{name:14} {a['per_day']:9.2f} {a['carbs_per_day']:12.1f} {a['g_mean']:12.1f}")

    with open(os.path.join(HERE, 'meal_stats.json'), 'w') as f:
        json.dump(cohorts, f, indent=2)
    path = _figure(cohorts)
    print(f"\n[meal_stats] wrote meal_stats.json and {path}")


if __name__ == '__main__':
    main()
