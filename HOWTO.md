# HOWTO

Running a pretraining job: choosing a capacity, choosing a context window,
pairing the normalization statistics to the pool, and starting the run.

`README.md` describes what the model is. This file is the operating procedure.

---

## The short version

```bash
cd ~/Desktop/T1DMAI

# 1. capacity — writes config.py
venv/bin/python resize_model.py --d-model 128 --heads 8 --layers 8

# 2. statistics must match the pool you are about to train on
cp ~/Desktop/T1DMSIM/cache_balanced_cf/normalization_stats.json normalization_stats.json

# 3. gate: refuses to pass if anything is mismatched
venv/bin/python scratch/preflight.py ~/Desktop/T1DMSIM/cache_balanced_cf

# 4. run
venv/bin/python train.py --cache-path ~/Desktop/T1DMSIM/cache_balanced_cf
```

Always `venv/bin/python`. A bare `python` has no `blosc2` — it dies only once it
reaches the cache — and runs a different torch than the one the venv pins.

---

## 1. Capacity

Three rungs. `resize_model.py` rewrites `config.py`; there is no CLI flag,
because every module reads its dimensions from `config` at import.

| rung | command | parameters | buffers |
| --- | --- | ---: | ---: |
| nano | `--d-model 32 --heads 2 --layers 2` | 38,241 | 18 |
| small | `--d-model 64 --heads 4 --layers 4` | 279,457 | 18 |
| medium | `--d-model 128 --heads 8 --layers 8` | 2,157,345 | 18 |

```bash
venv/bin/python resize_model.py --d-model 32 --heads 2 --layers 2
```

`FFN_DIM`, `BG_HEAD_HIDDEN` and `TIME_PROBE_HIDDEN` are multiples of `D_MODEL`
and follow it; `HEAD_DIM = D_MODEL // N_HEADS` is 16 at all three rungs. The
buffer column is the `step_basis`, which carries no gradient.

**`--report-only` prints the projection and writes nothing.** Use it to compare
rungs without touching the tree:

```bash
venv/bin/python resize_model.py --d-model 32 --heads 2 --layers 2 --report-only
```

Without that flag the file is rewritten, which is the intended behaviour and
also the way an unattended command has twice replaced a working configuration.

A resize invalidates `scratch/bitident_ref_v25.pt`, which is frozen at one
capacity. The gate reports the mismatch rather than passing; re-freeze once the
capacity is settled:

```bash
venv/bin/python scratch/bitident.py freeze
```

---

## 2. Context window

A patch is 30 minutes, so the window is set in patches.

| window | `--max-context-patches` | `MAX_SEQ_LEN` | `d ≥ 3` share | Ohio segments kept |
| --- | ---: | ---: | ---: | ---: |
| 24 h | 48 | 52 | 1.94% | 82 of 156 |
| 48 h | 96 | 100 | 1.26% | 51 of 156 |
| 72 h | 144 | 148 | 0.96% | 29 of 156 |

```bash
venv/bin/python resize_model.py --min-context-patches 48 --max-context-patches 96
```

Three properties of that table are worth knowing before choosing.

**Pass `--min-context-patches` as well.** Training draws `n_ctx` uniformly from
`[MIN, MAX]`, and every evaluation entry point scores at `MAX_CONTEXT_PATCHES`.
Raising only the ceiling moves the *mean* training context to 28 h at 96 patches
and 40 h at 144, while the reported number stays pinned to the ceiling — a
mixture scored only at its widest point. Moving both keeps the two together.

**48, 96 and 144 are the arithmetically clean widths.** Hour-of-day coverage is
exactly uniform when `n_candidates % 48 == 0`, where
`n_candidates = N/PATCH_SIZE − max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES) − n_ctx + 1`.
At the pools' 1242 steps that holds at 48, 96 and 144 and nowhere else.

**Real-cohort evaluation sets shrink with the window.** `realdata/calibrate.py`
drops any segment shorter than the context plus the horizon, and segments are cut
at every CGM gap over 30 minutes. Ohio falls from 82 usable segments to 29 across
this table; Shanghai keeps all 16 but loses 22% of its windows. Real-cohort
figures from two different context widths describe two different window sets.

Attention cost is quadratic in `MAX_SEQ_LEN`: 100 is about 3.7× the attention
memory of 52, and 148 about 8.1×.

---

## 3. Pool and statistics

Two pools, both at 1242 steps and 10⁶ rows:

| pool | `hypo_oversample` | steps below 70 mg/dL |
| --- | ---: | ---: |
| `cache_balanced_cf` | 0.0 | 5.90% |
| `cache_hypo_cf` | 0.5 | 11.29% |

`normalization_stats.json` in the repository root is **per pool**, and `train.py`
loads whatever sits there regardless of `--cache-path`. Nothing raises on a
mismatch: the file yields a fully formed sample against either pool, so a run
trains in a z-space that does not describe its own data. `|z_bal − z_hypo|`
reaches 0.767 at `f(10)` — largest exactly where hypoglycaemia lives.

Switching pools therefore means switching two things:

```bash
cp ~/Desktop/T1DMSIM/cache_hypo_cf/normalization_stats.json normalization_stats.json
venv/bin/python scratch/preflight.py ~/Desktop/T1DMSIM/cache_hypo_cf
```

### What preflight checks

`scratch/preflight.py <cache>` exits 0 when a run is safe to start. It covers the
four conditions that produce a complete and plausible validation table when they
are wrong:

| check | what is otherwise silent |
| --- | --- |
| statistics match this pool | no code path opens `<cache>/normalization_stats.json`; the CWD file wins |
| pool identity | `data.py` compares `sim_hours`, warmup, `dt_minutes`, channels and format — not `hypo_oversample` or `hypo_min_frac`, the only fields that distinguish the two pools |
| pool is complete | the `.partial` staging files are full-size from the first byte, so a half-built cache looks finished by size alone |
| statistics well-formed | one entry per channel, each `std > 0`; a zero std divides by `0 + 1e-8` and scales that channel by about 1e8 |

It is advisory. Nothing calls it for you.

---

## 4. Sampler arm

The arm fixes `MASK_SPAN_LENGTHS`, `MAX_MASKED_PATCHES` and `D_BALANCED_LOSS`
together. It is a flag, not a `config.py` edit:

```bash
venv/bin/python train.py --arm c --cache-path ~/Desktop/T1DMSIM/cache_balanced_cf
```

`venv/bin/python arms.py` prints the table with the measured differences. Arm `a`
is the default. On the archived runs, `a` scores highest on the hypo pool and `c`
on the balanced pool — one seed each at 5000 steps, with 4 to 76 events per
horizon, so the ordering is a starting point rather than a tuned result.

The arm resolves at `config` import from `$T1DMAI_ARM`; `train.py` reads `--arm`
out of `argv` and sets that variable above the import. An unknown name raises
before the run starts. The resolved arm is stamped into the checkpoint, and
`finetune/` refuses a checkpoint whose arm disagrees with the live one.

---

## 5. Running

```bash
venv/bin/python train.py --cache-path ~/Desktop/T1DMSIM/cache_balanced_cf
```

Useful overrides — CLI beats `config.py`, and the run prints the resolved
configuration and its source before it starts:

```bash
venv/bin/python train.py \
    --cache-path ~/Desktop/T1DMSIM/cache_balanced_cf \
    --arm a \
    --total-steps 5000 \
    --batch-size 128 \
    --validation-interval 500 \
    --log-interval 25 \
    --warmup-steps 200
```

`--batch-size` defaults to 512. On an 8 GB card, start at 128 and work up.

**`logs/` is a single directory with no per-run isolation.** Two concurrent runs
interleave their CSVs, and the damage looks like a short file rather than an
error. Run one at a time, and move `logs/` aside between runs worth keeping.

### Output

| path | contents |
| --- | --- |
| `checkpoints/t1dmai_best.pt` | lowest `val_loss_total` so far |
| `checkpoints/t1dmai_step_N.pt` | periodic and final |
| `logs/training_log.csv` | per-step losses |
| `logs/validation_log.csv` | one row per validation |

The validation table prints every `--validation-interval` steps, with per-`d`
scoring rows. `@30/@60/@90/@120` minutes are `d = 1/2/3/4` — the distance in
patches to the nearest visible evidence. A parenthesised count such as `(76st)`
is that row's event count; some bins are small enough that the count changes how
the percentage reads.

Selection is on `val_loss_total`, computed from the full-window forward.

### Checking a finished run

```bash
venv/bin/python scratch/smoke_report.py
```

Reports whether every logged loss stayed finite, whether `loss_ema` fell, whether
the per-`d` columns are populated, and whether a best checkpoint was written. A
missing checkpoint is the visible symptom of a NaN reaching validation, since
`nan < best` is False against an initial infinity.

---

## Worked example: nano at 48 h on the balanced pool

```bash
cd ~/Desktop/T1DMAI

venv/bin/python resize_model.py \
    --d-model 32 --heads 2 --layers 2 \
    --min-context-patches 48 --max-context-patches 96

cp ~/Desktop/T1DMSIM/cache_balanced_cf/normalization_stats.json normalization_stats.json
venv/bin/python scratch/preflight.py ~/Desktop/T1DMSIM/cache_balanced_cf

venv/bin/python train.py \
    --cache-path ~/Desktop/T1DMSIM/cache_balanced_cf \
    --total-steps 5000 --batch-size 128 \
    --validation-interval 500 --log-interval 25

venv/bin/python scratch/smoke_report.py
```

---

## Tests

```bash
venv/bin/python -m pytest tests/ -v -s
```

`-s` is worth keeping: the `[DUMP]` lines carry numbers that catch a silent
numerical change. Run from the repository root and name `tests/` — a bare
`pytest` also collects the `T1DMSIM` symlink's own suite.
