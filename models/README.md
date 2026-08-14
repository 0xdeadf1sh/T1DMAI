# models/

Per-model training and evaluation output, and a script that compares checkpoints
against one another.

The tree holds no checkpoints. What is present under each capacity is the training
logs, figures and metric reports of a run; `compare.py` reads a populated tree and
writes a single cross-model comparison under `comparison/`.

## Layout

```
models/
├── compare.py                  cross-model comparison script
├── README.md                   this file
├── nano/  small/  medium/  large/
│   ├── weights_<variant>.pt    the checkpoint compare.py reads for that cell
│   ├── finetune_log.csv        ohio fine-tuning trace
│   ├── finetune_multi_log.csv  multi fine-tuning trace
│   ├── figures/                per-model training figures and summary.json
│   ├── logs/                   training and validation logs, resolved config
│   └── metrics_sim/  metrics_ohio/  metrics_multi/
│       ├── real/               evaluation on the three real cohorts
│       ├── sim/                evaluation on the simulator
│       └── *.json  *.py        probe results and the scripts that produced them
└── comparison/                 written by compare.py
    ├── figures/                27 PNG figures at 300 dpi
    └── data/                   10 JSON files
```

`.gitignore` excludes the per-model directories and `comparison/`; `compare.py` and
this README are not excluded. The excluded paths are produced by training, evaluation
and `compare.py` respectively.

## The capacity ladder

| Size | Architecture | Parameters | Buffers | Total |
| --- | --- | ---: | ---: | ---: |
| nano | D=32, 2L, 2H, FFN=128 | 38,241 | 18 | 38,259 |
| small | D=64, 4L, 4H, FFN=256 | 279,457 | 18 | 279,475 |
| medium | D=128, 8L, 8H, FFN=512 | 2,157,345 | 18 | 2,157,363 |
| large | D=256, 16L, 16H, FFN=1024 | 16,999,969 | 18 | 16,999,987 |

The parameter column is trainable parameters, measured at `PATCH_DIM = PATCH_SIZE ×
N_INPUT_FEATURES = 30` by instantiating each capacity; the buffer column is the
`step_basis`, which is fixed and carries no gradient. Each capacity is set by
`D_MODEL`, `N_LAYERS` and `N_HEADS` alone — `FFN_DIM`, `BG_HEAD_HIDDEN` and
`TIME_PROBE_HIDDEN` are multiples of `D_MODEL` and follow it — and `resize_model.py`
prints the count for any of them. Wall-clock and peak-memory figures are not listed
because none has been measured against this architecture: position is RoPE alone,
the attention mask reaches SDPA as a bool rather than as a per-layer additive
float, and the BG head gathers `MAX_MASKED_PATCHES` slots by index.

## The three variants

| Variant | Training | Held-out |
| --- | --- | --- |
| `sim` | pretrained on the simulator corpus only | — |
| `ohio` | `sim` transferred to OhioT1DM | patient 591 |
| `multi` | `sim` transferred to OhioT1DM, AZT1D and ShanghaiT1DM | 591, AZ23, 1003 |

All three share one pretrained base per capacity; `ohio` and `multi` fine-tune from it
at a reduced learning rate.

## Evaluation cohorts

Four: the real cohorts OhioT1DM, AZT1D and ShanghaiT1DM, plus the T1DMSIM synthetic
corpus. Headline horizons are 30, 60 and 120 minutes; the horizon sweep extends to 480
minutes. Every model is evaluated on every cohort.

## compare.py

```bash
python compare.py                      # writes comparison/
python compare.py --out DIR --dpi 300
```

Reads the checkpoints, the metric directories, the training logs and the fine-tuning
logs; writes figures and JSON. It produces no markdown. Palette and figure chrome come
from `metrics/figstyle.py`, the suite's single copy.

A cell whose checkpoint is absent is skipped rather than reported: its parameter
count, architecture version and fine-tune provenance are null in `models.json`.

### Reporting basis

`realdata/metrics.py` scores each horizon on two bases, and they are not
interchangeable:

- **median line** — the point forecast `f_inv(median)`. This is the basis published
  point-forecast numbers use, and the basis `compare.py` reports as headline accuracy
  for RMSE, MAE, MARD, Clarke zones and skill.
- **band-scored** — computed on `clip(true, q_lo, q_hi)`, which charges zero error
  wherever the truth falls inside the 50% band. A band-geometry diagnostic, not
  comparable to a point forecast.

Two families have no median-line counterpart and are reported band-scored: CG-EGA,
which scores the band projection, and hypo/hyper detection, which keys off the τ alarm
band edges. Every record in `comparison/data/` carries a `basis` field naming which of
these it was measured on.

### Figures

| File | Contents |
| --- | --- |
| `fig01_capacity_and_cost` | parameters, wall-clock, throughput, peak memory |
| `fig02_pretrain_dynamics` | loss, validation loss, overfit ratio, gradient norm, step time |
| `fig03_rmse` | median-line point RMSE across the grid |
| `fig04_mard` | median-line MARD across the grid |
| `fig05_skill_vs_persistence` | skill against a last-value baseline |
| `fig06_rmse_vs_horizon` | RMSE over the 30–480 minute sweep, with persistence |
| `fig07_reporting_basis_gap` | band-scored headline against the median line |
| `fig08_scaling_law` | RMSE against capacity, fitted as a power law |
| `fig09_clarke_zones` | Clarke zones A, A+B and D at 60 minutes |
| `fig10_cgega` | CG-EGA by glycaemic region |
| `fig11_hypo_detection` | hypoglycaemia recall and precision |
| `fig12_hyper_detection` | hyperglycaemia recall and precision |
| `fig13_threshold_sweep` | hypo precision–recall over the alarm offset sweep |
| `fig14_night_onset` | overnight excursion onset |
| `fig15_event_dependence` | recall with and without a logged event in context |
| `fig16_conformal` | split-conformal interval width and realized coverage |
| `fig17_cqr_calibration` | raw against CQR-calibrated coverage and hypo escape |
| `fig18_band_geometry` | 50% band coverage and width |
| `fig19_amplitude_and_trend` | amplitude ratio, direction correlation, trend gain |
| `fig20_time_of_day` | clock-phase probe against chance |
| `fig21_whatif_dose_response` | counterfactual carbohydrate and insulin dose response |
| `fig22_whatif_quality` | sign correctness and monotonicity of that response |
| `fig23_empty_future` | trajectory shape under an event-free future |
| `fig24_finetune_curves` | fine-tuning traces on the held-out patient |
| `fig25_transfer_delta` | held-out RMSE before and after transfer |
| `fig26_reality_gap` | simulator accuracy against pooled real-cohort accuracy |
| `fig27_ranking` | every model in the grid ranked by pooled real-cohort RMSE |

### Data

| File | Contents |
| --- | --- |
| `index.json` | grid definition, basis glossary, exclusions, file inventory |
| `models.json` | architecture, parameters, optimization settings, cost, variant provenance |
| `metrics_long.json` | tidy long-form table; one row per size, variant, cohort, horizon, metric and basis |
| `metrics_pooled.json` | window-weighted pooling across the three real cohorts |
| `rankings.json` | models ranked per cohort and per horizon |
| `scaling.json` | fitted power-law exponent of RMSE against parameter count |
| `transfer.json` | held-out metrics before and after fine-tuning, on the identical split |
| `uncertainty.json` | conformal, CQR and band-geometry results |
| `behaviour.json` | amplitude, time-of-day, counterfactual and event-dependence probes |
| `corpus.json` | corpus properties that do not vary with the model, with an invariance check |

Transfer deltas pair `baseline_heldout` against `best_heldout` from the same
checkpoint, so both sides describe the same held-out patient and calibration split. The
larger cohort-wide evaluation under `metrics_*/` is a different set and is not mixed
into that comparison.

### Not included in the comparison

The CGM 15-minute shift probe (`shift15.json`) and the augmentation bound
(`augexp.json`) are present in the per-model metric directories but are not read by
`compare.py`. `index.json` records the exclusion.

The `augmented/` evaluation regime is not part of this tree. `metrics/augmented/build_report.py`
in the repository rebuilds it from a checkpoint.
