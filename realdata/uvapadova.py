"""
UVA/Padova adapter — generates virtual-patient trajectories from the ``simglucose``
port of the FDA-accepted UVA/Padova T1D simulator and builds canonical Segments.

This is a DIFFERENT simulator from the repo's own ``T1DMSIM``: the model was
trained on T1DMSIM, so UVA/Padova is an out-of-distribution (cross-simulator)
test cohort. simglucose emits 3-minute samples where ``CHO``/``insulin`` are RATES
(g/min, U/min); they are converted to per-5-min-step amounts (mass-conserving) and
CGM is resampled to the 5-min grid, then wrapped as :class:`~realdata.schema.Segment`
so the model's real-data bridge (kernel convolution in ``realdata.features``)
treats UVA/Padova exactly like a real cohort.

Generation is cached to ``finetune/uvapadova_cache/*.npz``; ``load`` reads that
cache. Run ``python realdata/uvapadova.py --generate`` (or call :func:`generate`)
once before ``load``.
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings
from datetime import datetime, timedelta

import numpy as np

try:
    from .schema import Segment
except ImportError:  # run as a script (python realdata/uvapadova.py) — no package parent
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from realdata.schema import Segment

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(_REPO, 'finetune', 'uvapadova_cache')
_T0 = datetime(2020, 1, 1, 0, 0, 0)


def generate(cache_dir: str = DEFAULT_CACHE, patients: list[str] | None = None,
             days: int = 7, base_seed: int = 100) -> str:
    """Simulate ``patients`` for ``days`` each and cache per-patient 5-min arrays.

    Args:
        cache_dir: output directory for the ``<patient>.npz`` files.
        patients: simglucose patient names (default: all 30 vpatients).
        days: post-start trajectory length per patient.
        base_seed: patient ``i`` uses ``base_seed + i`` for sensor + scenario.

    Returns:
        ``cache_dir``.
    """
    warnings.filterwarnings('ignore')
    import pandas as pd
    import simglucose
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.controller.basal_bolus_ctrller import BBController
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.actuator.pump import InsulinPump
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.simulation.scenario_gen import RandomScenario
    from simglucose.simulation.sim_engine import SimObj, sim

    if patients is None:
        vp = os.path.join(os.path.dirname(simglucose.__file__), 'params', 'vpatient_params.csv')
        patients = pd.read_csv(vp)['Name'].tolist()

    os.makedirs(cache_dir, exist_ok=True)
    resdir = os.path.join(cache_dir, '_simout')
    os.makedirs(resdir, exist_ok=True)

    for i, name in enumerate(patients):
        seed = base_seed + i
        env = T1DSimEnv(T1DPatient.withName(name), CGMSensor.withName('Dexcom', seed=seed),
                        InsulinPump.withName('Insulet'), RandomScenario(start_time=_T0, seed=seed))
        df = sim(SimObj(env, BBController(), timedelta(days=days), animate=False, path=resdir))
        step_min = (df.index[1] - df.index[0]).total_seconds() / 60.0
        cgm5 = df['CGM'].resample('5min').mean().interpolate(limit_direction='both')
        carb5 = (df['CHO'] * step_min).resample('5min').sum().reindex(cgm5.index, fill_value=0.0)
        ins5 = (df['insulin'] * step_min).resample('5min').sum().reindex(cgm5.index, fill_value=0.0)
        cgm = cgm5.to_numpy(np.float64)
        assert np.isfinite(cgm).all(), f"{name}: non-finite CGM after resample"
        np.savez(os.path.join(cache_dir, name.replace('#', '_') + '.npz'),
                 name=name, t0=_T0.isoformat(), cgm=cgm,
                 carb=carb5.to_numpy(np.float64), insulin=ins5.to_numpy(np.float64))
        print(f"[uvapadova] {name}: {len(cgm)} steps, "
              f"{carb5.sum()/days:.0f} g/day, {ins5.sum()/days:.1f} U/day")
    return cache_dir


def load(root_dir: str | None = None) -> list[Segment]:
    """Load cached UVA/Padova runs into Segments (one per patient).

    Insulin is folded into ``bolus_units`` (per-5-min U) with ``basal_rate = 0``,
    matching the real-data CSII treatment (all insulin convolved with the rapid
    kernel by ``realdata.features``).
    """
    root = root_dir or DEFAULT_CACHE
    segs: list[Segment] = []
    for f in sorted(glob.glob(os.path.join(root, '*.npz'))):
        z = np.load(f, allow_pickle=True)
        cgm = z['cgm'].astype(np.float64)
        n = len(cgm)
        segs.append(Segment(
            dataset='uvapadova', patient=str(z['name']),
            t0=datetime.fromisoformat(str(z['t0'])),
            cgm=cgm, carb_grams=z['carb'].astype(np.float64),
            bolus_units=z['insulin'].astype(np.float64),
            basal_rate=np.zeros(n, dtype=np.float64), exercise=np.zeros(n, dtype=np.float64)))
    return segs


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Generate the UVA/Padova (simglucose) cache.")
    p.add_argument('--generate', action='store_true')
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--cache', default=DEFAULT_CACHE)
    args = p.parse_args()
    if args.generate:
        generate(cache_dir=args.cache, days=args.days)
    else:
        segs = load(args.cache)
        print(f"loaded {len(segs)} UVA/Padova segments, "
              f"patients={sorted({s.patient for s in segs})}")
