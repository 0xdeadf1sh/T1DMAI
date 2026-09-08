"""Profile one ``simulate_discard_warmup`` call: where T1DMSimulator spends its time."""
import cProfile
import pstats
from data import _make_simulator, simulate_discard_warmup

PATIENT_SEED = 12345
HOURS = 720.0  # heaviest realistic request; training's ON_THE_FLY_SIM_HOURS (~200h) is ~1/3 of this

sim = _make_simulator(PATIENT_SEED, uniform_skills=False)

# Fresh simulator below: the warm-up must not advance the profiled run's RNG.
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
