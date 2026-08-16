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
| 24 h | 48 | 52 | 27.90% | 82 of 156 |
| 48 h | 96 | 100 | 27.11% | 51 of 156 |
| 72 h | 144 | 148 | 26.87% | 29 of 156 |

The `d ≥ 3` share is `d_balance.d_distribution` at the live sampler constants,
with `--min-context-patches` at half the ceiling on every row. It barely moves
across the table because the right-edge quota, not the window width, is what
supplies the far horizons: at quota 0 the same three rows read 22.79 / 21.72 /
21.39%. Under the retired four-patch span ceiling they read 1.94 / 1.26 / 0.96%,
at the floor of 16 patches that ceiling was paired with rather than at half the
ceiling.

```bash
venv/bin/python resize_model.py --min-context-patches 48 --max-context-patches 96
```

Three properties of that table are worth knowing before choosing.

**Pass `--min-context-patches` as well.** Training draws `n_ctx` uniformly from
`[MIN, MAX]`, and every evaluation entry point scores at `MAX_CONTEXT_PATCHES`.
Raising only the ceiling, from the live floor of 48 patches, moves the *mean*
training context to 36 h at 96 patches and 48 h at 144, while the reported number
stays pinned to the ceiling — a mixture scored only at its widest point. Moving
both keeps the two together.

**48, 96 and 144 are the arithmetically clean widths.** Hour-of-day coverage is
exactly uniform when `n_candidates % 48 == 0`, where
`n_candidates = N/PATCH_SIZE − max(PREDICTION_PATCHES, NIGHT_LONG_HORIZON_PATCHES) − n_ctx + 1`.
At the accepted 2394 steps, `n_candidates = 384 − n_ctx`, so it holds at every
multiple of 48 up to the 336-patch ceiling — 48, 96, 144, 192, 240, 288 and 336.

**Real-cohort evaluation sets shrink with the window.** `realdata/calibrate.py`
drops any segment shorter than the context plus the horizon, and segments are cut
at every CGM gap over 30 minutes. Ohio falls from 82 usable segments to 29 across
this table; Shanghai keeps all 16 but loses 22% of its windows. Real-cohort
figures from two different context widths describe two different window sets.

Attention cost is quadratic in `MAX_SEQ_LEN`: 100 is about 3.7× the attention
memory of 52, and 148 about 8.1×.

---

## 3. Pool and statistics

Two pools, both at 10⁶ rows. Both were built at the retired 1242-step geometry
and no longer load: `data.py` accepts 2394 steps (see `ON_THE_FLY_SIM_HOURS`) and
rejects any other length at first row read. Rebuild with
`--sim-hours 199.5` before use.

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

## 4. Mask sampler

`MASK_SPAN_LENGTHS`, `MAX_MASKED_PATCHES` and `MASK_RIGHT_EDGE_QUOTA` are plain
`config.py` constants, bound at import and fixed for a whole run. There is no
flag: edit `config.py`.

```bash
venv/bin/python d_balance.py    # the d histogram the live constants produce
```

Moving any of the three invalidates `metrics/protocols.py`'s
`SAMPLER_REFERENCE`, which is enumerated from `d_balance.d_distribution` at the
knobs it names; `sampler_reference_applies()` then refuses the comparison rather
than reporting against a mixture the sampler does not draw. Re-enumerate and
update it in the same change.

The three are stamped into every checkpoint, and `finetune/` refuses a checkpoint
whose recorded sampler is not the live one — no parameter shape depends on the
sampler, so a strict state-dict load would accept the weights and nothing else
would notice.

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

The validation table prints every `--validation-interval` steps. `@30/@60/@90/@120`
minutes are `d = 1/2/3/4` — the distance in patches to the nearest visible
evidence. A parenthesised count such as `(76st)` is that row's event count; some
bins are small enough that the count changes how the percentage reads.

The table is a reading surface, not the record: `logs/validation_log.csv` carries
every metric on every row, including the families the table does not print.

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
