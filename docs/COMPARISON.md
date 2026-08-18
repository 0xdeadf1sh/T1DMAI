# Cross-model comparison

`compare.py` at the repository root reads the trained-model tree under
`new_models/`, one directory per capacity, and writes a single comparison under
`comparison/`. Keeping one trained-checkpoint root is what stops the GUI and this
script from disagreeing about which checkpoint `medium` names.

## Layout

```
new_models/
└── nano/  small/  medium/
    ├── checkpoints/            t1dmai_best.pt and the periodic step snapshots
    ├── figures/                per-model training figures and summary.json
    ├── logs/                   training and validation logs, resolved config
    └── metrics_sim/
        ├── sim/                evaluation on the simulator
        └── *.json              probe results

compare.py                      cross-model comparison script (repository root)
comparison/                     written by compare.py
├── figures/                    20 PNG figures at 300 dpi
└── data/                       7 JSON files
```

`.gitignore` excludes the trained-model trees and `comparison/`; `compare.py` and
this document are not excluded. The excluded paths are produced by training,
evaluation and `compare.py` respectively.

## The capacity ladder

| Size | Architecture | Parameters | Buffers | Total |
| --- | --- | ---: | ---: | ---: |
| nano | D=32, 2L, 2H, FFN=128 | 38,934 | 36 | 38,970 |
| small | D=64, 4L, 4H, FFN=256 | 280,822 | 36 | 280,858 |
| medium | D=128, 8L, 8H, FFN=512 | 2,160,054 | 36 | 2,160,090 |

The parameter column is trainable parameters, measured at `PATCH_DIM = PATCH_SIZE ×
N_INPUT_FEATURES = 30` by instantiating each capacity; the buffer column is the
`step_basis`, which is fixed and carries no gradient. Each capacity is set by
`D_MODEL`, `N_LAYERS` and `N_HEADS` alone — `FFN_DIM`, `BG_HEAD_HIDDEN` and
`TIME_PROBE_HIDDEN` are multiples of `D_MODEL` and follow it — and `resize_model.py`
prints the count for any of them. Wall-clock and peak-memory figures are not listed
because none has been measured against this architecture: position is RoPE alone,
the attention mask reaches SDPA as a bool rather than as a per-layer additive
float, and the BG head gathers `MAX_MASKED_PATCHES` slots by index.

## Evaluation source

Fresh T1DMSIM patients, drawn at seeds disjoint from the calibration pool. Headline
horizons are 30, 60 and 120 minutes; the horizon sweep extends to 480 minutes.

## compare.py

```bash
python compare.py                      # writes comparison/
python compare.py --out DIR --dpi 300
```

Reads the checkpoints, the metric directories and the training logs; writes figures
and JSON. It produces no markdown. Palette and figure chrome come from
`metrics/figstyle.py`, the suite's single copy.

A capacity whose checkpoint is absent is skipped rather than reported: its parameter
count and architecture version are null in `models.json`.

### Reporting basis

`metrics/core/suite.py` scores each horizon on two bases, and they are not
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
| `fig03_rmse` | median-line point RMSE across the ladder |
| `fig04_mard` | median-line MARD across the ladder |
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
| `fig16_conformal` | split-conformal interval width and realized coverage |
| `fig17_cqr_calibration` | raw against CQR-calibrated coverage and hypo escape |
| `fig18_band_geometry` | 50% band coverage and width |
| `fig21_whatif_dose_response` | counterfactual carbohydrate and insulin dose response |
| `fig22_whatif_quality` | sign correctness and monotonicity of that response |
| `fig27_ranking` | every capacity ranked by pooled RMSE |

The numbering has gaps. It is the filename, and a stable filename outlives the
figure beside it — renumbering would silently repoint every link that ever named
one. `fig23_empty_future` is written only where a source carries raw carb/bolus
events to strip; on simulator segments the arm is not probed and the figure is
skipped rather than shipped blank.

### Data

| File | Contents |
| --- | --- |
| `index.json` | ladder definition, basis glossary, file inventory |
| `models.json` | architecture, parameters, optimization settings, cost |
| `metrics_long.json` | tidy long-form table; one row per size, horizon, metric and basis |
| `metrics_pooled.json` | window-weighted pooling across the evaluation sources |
| `rankings.json` | capacities ranked per horizon |
| `scaling.json` | fitted power-law exponent of RMSE against parameter count |
| `uncertainty.json` | conformal, CQR and band-geometry results |
