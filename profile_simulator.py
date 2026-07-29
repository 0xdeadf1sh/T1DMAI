"""
profile_simulator.py — measure where T1DMSimulator spends its time.
=====================================================================

The on-the-fly DataLoader path runs the simulator inside every worker; if
the worker can't keep up with batch consumption the GPU starves.  This
script profiles a single ``simulate_discard_warmup`` call so you can see
which simulator hot path dominates per-sample wall time and decide whether
to (a) reduce the requested simulation hours or (b) raise NUM_WORKERS.

We profile at ``HOURS = 720.0`` to expose cold-path cost; the on-the-fly
training path uses ``ON_THE_FLY_SIM_HOURS`` (~55 h) per sample, so the
actual training cost will look like roughly 1/13th of the figures reported
here.
"""
import cProfile
import pstats
from data import _make_simulator, simulate_discard_warmup

PATIENT_SEED = 12345
HOURS = 720.0  # heaviest realistic simulator request

sim = _make_simulator(PATIENT_SEED, uniform_skills=False)

# Warm-up call so import / lazy-init costs don't pollute the profile.  Using
# a *different* simulator instance below ensures the warm-up doesn't advance
# the RNG state of the profiled run.
_ = simulate_discard_warmup(sim, 24.0)

sim = _make_simulator(PATIENT_SEED, uniform_skills=False)

profiler = cProfile.Profile()
profiler.enable()
data = simulate_discard_warmup(sim, HOURS)
profiler.disable()

print("=== Top 40 by cumulative time ===")
pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(40)

print("\n=== Top 25 by tottime (self time, excludes callees) ===")
pstats.Stats(profiler).strip_dirs().sort_stats('tottime').print_stats(25)
