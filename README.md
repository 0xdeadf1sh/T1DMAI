# T1DMAI

> [!NOTE]
> Most of the code in this repo was written by AI. If you don't like that, feel free to leave.

This repo is a zoo. Here I train & test many blood glucose forecasting models (including my own)
on both my own data and metabonet (public).

My model is a small transformer that forecasts blood glucose 2 hours ahead. I've made two variants: D16 DILATE and D16 MSE
which use DILATE and MSE loss for training, respectively. D16 refers to the width of the model (16). They are both
trained on synthetic traces, and then a LoRA adapter is added for light fine-tuning, but I also test them without the adapter.

Feast thine eyes on some of their forecasts (these have been trained on synthetic traces, then fine-tuned with a LoRA adapter on my training split):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-d16-dilate-1-dark.png">
  <img alt="T1DMAI D16 DILATE forecasting my glucose, test days block 1" src="screenshots/personal-days-d16-dilate-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-d16-dilate-2-dark.png">
  <img alt="T1DMAI D16 DILATE forecasting my glucose, test days block 2" src="screenshots/personal-days-d16-dilate-2.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-d16-rmse-1-dark.png">
  <img alt="T1DMAI D16 MSE forecasting my glucose, test days block 1" src="screenshots/personal-days-d16-rmse-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-d16-rmse-2-dark.png">
  <img alt="T1DMAI D16 MSE forecasting my glucose, test days block 2" src="screenshots/personal-days-d16-rmse-2.png">
</picture>

> [!CAUTION]
> **Research and educational use only - not a medical device.** T1DMAI is a
> research artifact. It is **not** clinically validated, has **no** regulatory
> clearance, and its output is a forecast of research signals, not medical
> advice. It **must not** be used to make medical, diagnostic, or treatment
> decisions, to calculate or adjust insulin doses, or to manage diabetes in any
> way. For medical guidance consult a qualified healthcare professional. The
> software is provided "as is", without warranty of any kind (see
> [LICENSE](LICENSE)).

## Contents

1. [Background](#1-background)
2. [Main results](#2-main-results)
3. [Models](#3-models)
4. [Data](#4-data)
5. [Method](#5-method)
6. [Results](#6-results)
7. [Mistral models](#7-mistral-models)
8. [Limitations](#8-limitations)
9. [Terms](#9-terms)
10. [Full tables](#10-full-tables)
11. [Reproduce](#11-reproduce)
12. [Published scores](#12-published-scores)
13. [References](#13-references)

## 1. Background

I have type 1 diabetes and I wanna know if I can train a BG forecasting model
to help me with my daily management. To this end I've made some auxiliary tools:

- [T1DMSIM](https://github.com/0xdeadf1sh/T1DMSIM) creates synthetic CGM traces since UVA/Padova was too slow on my laptop.
- [T1DMDROID](https://github.com/0xdeadf1sh/T1DMDROID) runs the models on my phone, also has some neat functionality like reading BG from multiple CGMs at once.

For evaluation, I use the recommendations by Jacobs et al., IEEE Reviews in Biomedical Engineering, 2024 [[1]](#13-references). It's a nice paper that you should probably read.

## 2. Main results

All models suck. On my data XGBoost is the best one but my own model comes close. On MetaboNet
fine-tuned Timestral wins. More details below:

1. On my data, D16 DILATE + LoRA comes 3rd of 34 (22.8), behind XGBoost and LightGBM
   (both 22.4). Wilcoxon over test days cannot tell it from the best model at any horizon.
2. On MetaboNet, D16 DILATE + LoRA comes 13th (34.6). Fine-tuned Timestral 209M has the lowest
   RMSE at every horizon (32.6 at 60 min), and Wilcoxon ties no other model to it.
3. At 60 min a 3771 parameter LoRA adapter helps both models on both datasets: 26.1 to 22.8 and
   27.1 to 23.5 on my data, 36.6 to 34.6 and 36.7 to 35.1 on MetaboNet. At 120 min on
   MetaboNet it makes D16 DILATE worse (48.0 to 50.4).
4. On MetaboNet, LoRA also fixes the bands. The 90% band holds the truth 75-78% of the time at
   60 min with sim only and 81-85% with LoRA. On my data both are 83-85%.
5. T1DMAI + LoRA reacts earliest on my data: 40-50 min time gain at 60 min, against 30 for
   XGBoost.
6. On my data it catches 40-47% of lows at 60 min, against 28% for XGBoost and 36-50% for
   fine-tuned Timestral. On MetaboNet every low-RMSE model misses nearly all lows (XGBoost 0%,
   T1DMAI 6-8%, fine-tuned Timestral 10-12%), while linear extrapolation catches 58%.
7. LightGBM matches XGBoost within 0.2 mg/dL everywhere. An LSTM with 15 hours of history comes
   8th on MetaboNet (33.9 at 60 min), against 35.9 for the 2 hour one.
8. Zero-shot Chronos-2 and TimesFM 2.5 beat zero-shot Chronos-Bolt by 1.4-2.7 mg/dL at 60 min
   on both datasets, but trail fine-tuned Chronos-Bolt.
9. Zero-shot Timestral beats Chronos-2 and TimesFM 2.5 on my data (27.8-28.4 against 29.1-29.4
   at 60 min) and ties them on MetaboNet (35.0-35.2 against 35.0-35.3). Size barely matters:
   5M is within 0.4 mg/dL of 209M at 60 min on both datasets.
10. One epoch of full fine-tuning helps every Timestral size on both datasets: 27.8-28.4 to
    24.2-25.4 on my data and 35.0-35.2 to 32.6-33.8 on MetaboNet at 60 min. On MetaboNet size
    now matters: 209M leads 5M by 1.2 mg/dL. On my data 28M and 123M do best (24.2).

## 3. Models

| Model | Family | Description |
| --- | --- | --- |
| Persistence | Baseline | Last reading carried forward. |
| Linear extrapolation | Baseline | Straight line through the last few readings. |
| AR (ridge) | Classical | Linear regression on recent glucose. |
| ARX (ridge) | Classical | Linear regression on recent glucose, insulin and carbs. |
| ARIMA | Classical | Glucose only. |
| SARIMAX | Classical | With insulin and carbs. |
| Kalman filter | Classical | Damped local linear trend. |
| SVR | Classical | Glucose, insulin and carbs. |
| Sparse GP | Classical | 1000 inducing points. |
| XGBoost | Classical | Glucose, insulin and carbs [18]. |
| LightGBM | Classical | Same features, depth, rate and rounds as XGBoost [19]. |
| LSTM | Deep | 2 hours of history [5]. |
| LSTM (15 h) | Deep | Two layers of 128 over 15 hours, as in MetaboNet-Bench [3]. |
| GluNet | Deep | Dilated causal convolutions over 4 hours [6]. |
| PatchTST | Deep | [13] |
| N-HiTS | Deep | [14] |
| Chronos-Bolt tiny and mini | Foundation | 9M and 21M parameters, glucose only, zero-shot and fine-tuned [15]. |
| Chronos-2 | Foundation | 120M parameters, zero-shot. Past insulin and carbs; on my data also the future ones [16]. |
| TimesFM 2.5 | Foundation | 200M parameters, zero-shot. Glucose only on MetaboNet; on my data future insulin and carbs through in-context regression [17]. |
| Timestral 5M, 28M, 123M and 209M | Foundation | Zero-shot and fine-tuned. Past insulin and carbs; on my data also the future ones [27]. |
| T1DMAI D16 DILATE and D16 MSE | T1DMAI | 31k parameters. Glucose, carbs and insulin. |

Numbers in brackets point to [References](#13-references). Everything except the baselines and
the foundation models was trained from scratch on each dataset's training split. T1DMAI comes in two versions: **sim only** is the checkpoint in `models/`, and
**sim + LoRA** adds a frozen-base rank 1 adapter [23] (3771 parameters) trained on the real training
split.

<details>
<summary>T1DMAI details</summary>

| | |
| --- | --- |
| Architecture | encoder-only transformer, 30 min patches, one attention head, RoPE [26], SwiGLU |
| Size | 16 layers of width 16, 31k parameters |
| Loss | 7 quantile pinball loss plus a median term (DILATE [25] on soft-DTW [24], or MSE) |
| Inputs | CGM in a risk-scaled space, carb absorption curve, insulin action curve, mask bit |
| Output | 7 quantiles (5% to 95%) for every 5 minute step of the next 2 hours |
| History | up to 7 days |
| Simulator training | 1M simulated people, 7 days each, 10000 steps of batch 512 |
| Runtime | ExecuTorch or LiteRT, on the phone in T1DMDROID |

More in [docs/USAGE.md](docs/USAGE.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

</details>

## 4. Data

**My data.** 115 days from T1DMDROID: days 1 to 80 for training, 81 to 115 for testing.
Note that the model sees future inputs like insulin & carbs. I should probably also
train the adapters while withholding those inputs.

| Source | Subjects | Person-days | Forecasts | CGM coverage % | Mean mg/dL | CV % | < 54 % | < 70 % | 70-180 % | > 180 % | > 250 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t1dmdroid | 1 | 35 | 9764 | 97 | 130 | 28 | 0.4 | 3.0 | 89 | 8 | 0 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-glucose-dark.png">
  <img alt="Distribution of my CGM readings in the 35 test days" src="screenshots/personal-glucose.png">
</picture>

**MetaboNet** [[2]](#13-references). Official split, 279 test people from 9 studies. Blind, because most of them use
closed-loop pumps whose future insulin reacts to glucose the model hasn't seen.

| Source | Subjects | Person-days | Forecasts | CGM coverage % | Mean mg/dL | CV % | < 54 % | < 70 % | 70-180 % | > 180 % | > 250 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AZT1D | 4 | 131 | 775 | 99 | 141 | 30 | 0.3 | 1.3 | 82 | 17 | 2 |
| BrisT1D | 2 | 312 | 1953 | 77 | 144 | 33 | 0.2 | 1.8 | 78 | 20 | 3 |
| CTR3 | 5 | 1157 | 1052 | 51 | 141 | 38 | 0.7 | 3.5 | 78 | 19 | 4 |
| HUPA-UCM | 4 | 315 | 896 | 100 | 132 | 36 | 0.9 | 6.8 | 78 | 16 | 2 |
| IOBP2 | 62 | 4737 | 14653 | 84 | 159 | 37 | 0.4 | 2.0 | 68 | 30 | 8 |
| Loop | 159 | 47438 | 203538 | 93 | 139 | 40 | 0.7 | 3.9 | 77 | 19 | 5 |
| PEDAP | 6 | 2070 | 6321 | 85 | 158 | 38 | 0.4 | 2.4 | 68 | 29 | 9 |
| ReplaceBG | 35 | 7967 | 35683 | 87 | 158 | 41 | 1.0 | 4.2 | 64 | 32 | 9 |
| ShanghaiT1DM | 2 | 19 | 28 | 33 | 159 | 39 | 1.0 | 3.8 | 63 | 33 | 7 |
| All | 279 | 64146 | 264899 | 90 | 143 | 40 | 0.7 | 3.7 | 75 | 22 | 6 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-glucose-dark.png">
  <img alt="Distribution of MetaboNet CGM readings in the test split" src="screenshots/metabonet-glucose.png">
</picture>

## 5. Method

| | My data | MetaboNet |
| --- | --- | --- |
| Forecasts | every 5 min of the test days | every 10th row of the leaderboard template |
| Horizons | 30, 60, 90, 120 min | 30, 60, 90, 120 min |
| Training windows, other models | 50000 | 338000 (135000 for fine-tuned Chronos and Timestral) |
| LoRA steps | 2000 | 6000 |

- LoRA checkpoints are picked by mean RMSE on the last 10 days of each subject's training data.
- The other models never train on those 10 days. The neural ones pick their epoch on the first 5.
- A training window ends on a measured reading and is scored on measured readings only.
- All models forecast at the same moments.
- 95% intervals: bootstrap over people (MetaboNet) or days (my data).
- Significance: Wilcoxon signed-rank against the best model on the mean paired absolute-error
  difference per person (MetaboNet) or per day (my data), Bonferroni corrected.
- Clinical error grids: DTS [20], Clarke [22], CG-EGA [21].

## 6. Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-rmse-dark.png">
  <img alt="RMSE per model on my data" src="screenshots/personal-rmse.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-rmse-dark.png">
  <img alt="RMSE per model on MetaboNet" src="screenshots/metabonet-rmse.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-horizon-rmse-dark.png">
  <img alt="RMSE by horizon, my data" src="screenshots/personal-horizon-rmse.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-horizon-rmse-dark.png">
  <img alt="RMSE by horizon, MetaboNet" src="screenshots/metabonet-horizon-rmse.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-ci-dark.png">
  <img alt="RMSE at 60 min with 95% intervals, my data" src="screenshots/personal-ci.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-ci-dark.png">
  <img alt="RMSE at 60 min with 95% intervals, MetaboNet" src="screenshots/metabonet-ci.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-significance-dark.png">
  <img alt="Wilcoxon test over days against the lowest-RMSE model, my data" src="screenshots/personal-significance.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-significance-dark.png">
  <img alt="Wilcoxon test over people against the lowest-RMSE model, MetaboNet" src="screenshots/metabonet-significance.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-spread-dark.png">
  <img alt="RMSE at 60 min per test day, my data" src="screenshots/personal-spread.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-spread-dark.png">
  <img alt="RMSE at 60 min per person, MetaboNet" src="screenshots/metabonet-spread.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-strata-dark.png">
  <img alt="RMSE at 60 min by situation, my data" src="screenshots/personal-strata.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-strata-dark.png">
  <img alt="RMSE at 60 min by situation, MetaboNet" src="screenshots/metabonet-strata.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-horizon-hypo-sens-dark.png">
  <img alt="Share of lows caught by horizon, my data" src="screenshots/personal-horizon-hypo-sens.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-horizon-hypo-sens-dark.png">
  <img alt="Share of lows caught by horizon, MetaboNet" src="screenshots/metabonet-horizon-hypo-sens.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-heat-mcc-dark.png">
  <img alt="Low detection MCC by model and horizon, my data" src="screenshots/personal-heat-mcc.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-heat-mcc-dark.png">
  <img alt="Low detection MCC by model and horizon, MetaboNet" src="screenshots/metabonet-heat-mcc.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-horizon-hypo-bias-dark.png">
  <img alt="Mean error when the real value is below 70 mg/dL, my data" src="screenshots/personal-horizon-hypo-bias.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-horizon-hypo-bias-dark.png">
  <img alt="Mean error when the real value is below 70 mg/dL, MetaboNet" src="screenshots/metabonet-horizon-hypo-bias.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-heat-dts-dark.png">
  <img alt="DTS error grid zone A by model and horizon, my data" src="screenshots/personal-heat-dts.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-heat-dts-dark.png">
  <img alt="DTS error grid zone A by model and horizon, MetaboNet" src="screenshots/metabonet-heat-dts.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-calibration-dark.png">
  <img alt="Calibration of the T1DMAI bands at 60 min, my data" src="screenshots/personal-calibration.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-calibration-dark.png">
  <img alt="Calibration of the T1DMAI bands at 60 min, MetaboNet" src="screenshots/metabonet-calibration.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-traces-dark.png">
  <img alt="Example forecasts on my data" src="screenshots/personal-traces.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-traces-dark.png">
  <img alt="Example forecasts on MetaboNet" src="screenshots/metabonet-traces.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-clarke-dark.png">
  <img alt="Clarke error grid by horizon, my data" src="screenshots/personal-clarke.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-clarke-dark.png">
  <img alt="Clarke error grid by horizon, MetaboNet" src="screenshots/metabonet-clarke.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-paths-dark.png">
  <img alt="T1DMAI with LoRA on and off, my data" src="screenshots/personal-paths.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-paths-dark.png">
  <img alt="T1DMAI with LoRA on and off, MetaboNet" src="screenshots/metabonet-paths.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-cost-dark.png">
  <img alt="Accuracy against CPU time per forecast" src="screenshots/personal-cost.png">
</picture>

| Model | Parameter count | Microseconds per forecast | History used |
| --- | ---: | ---: | ---: |
| Persistence | 0 | 0.0042 | 5 min |
| Linear extrapolation | 0 | 0.033 | 30 min |
| AR (ridge) | 648 | 0.13 | 2 h |
| SARIMAX | 8,088 | 0.21 | 8 h |
| ARIMA | 4,632 | 0.27 | 4 h |
| ARX (ridge) | 1,116 | 0.36 | 6 h |
| Kalman filter | 864 | 0.48 | 3 h |
| SVR | 69,772 | 4.2 | 6 h |
| Sparse GP | 66,134 | 6.1 | 6 h |
| N-HiTS | 417,772 | 6.5 | 6 h |
| LSTM | 103,192 | 33 | 2 h |
| GluNet | 65,368 | 71 | 4 h |
| PatchTST | 182,266 | 74 | 6 h |
| XGBoost | 2,692,860 | 506 | 6 h |
| Chronos-Bolt tiny (fine-tuned) | 8,669,901 | 1,346 | 24 h |
| Chronos-Bolt tiny (zero-shot) | 8,669,901 | 1,361 | 24 h |
| Chronos-Bolt mini (zero-shot) | 21,256,141 | 3,871 | 24 h |
| Chronos-Bolt mini (fine-tuned) | 21,256,141 | 3,889 | 24 h |
| T1DMAI D16 DILATE | 31,253 | 5,679 | 168 h |
| T1DMAI D16 MSE | 31,253 | 7,342 | 168 h |

## 7. Mistral models

Timestral [[27]](#13-references) on two 48 h spans of my test days, zero-shot and fine-tuned on my
training split. Bands are the raw 50% and 90% quantiles of the model.

### Timestral, zero-shot

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-5m-1-dark.png">
  <img alt="Timestral 5M (zero-shot) forecasting my glucose, test days block 1" src="screenshots/personal-days-timestral-5m-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-5m-2-dark.png">
  <img alt="Timestral 5M (zero-shot) forecasting my glucose, test days block 2" src="screenshots/personal-days-timestral-5m-2.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-28m-1-dark.png">
  <img alt="Timestral 28M (zero-shot) forecasting my glucose, test days block 1" src="screenshots/personal-days-timestral-28m-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-28m-2-dark.png">
  <img alt="Timestral 28M (zero-shot) forecasting my glucose, test days block 2" src="screenshots/personal-days-timestral-28m-2.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-123m-1-dark.png">
  <img alt="Timestral 123M (zero-shot) forecasting my glucose, test days block 1" src="screenshots/personal-days-timestral-123m-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-123m-2-dark.png">
  <img alt="Timestral 123M (zero-shot) forecasting my glucose, test days block 2" src="screenshots/personal-days-timestral-123m-2.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-209m-1-dark.png">
  <img alt="Timestral 209M (zero-shot) forecasting my glucose, test days block 1" src="screenshots/personal-days-timestral-209m-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-209m-2-dark.png">
  <img alt="Timestral 209M (zero-shot) forecasting my glucose, test days block 2" src="screenshots/personal-days-timestral-209m-2.png">
</picture>

### Timestral, fine-tuned

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-5m-ft-1-dark.png">
  <img alt="Timestral 5M (fine-tuned) forecasting my glucose, test days block 1" src="screenshots/personal-days-timestral-5m-ft-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-5m-ft-2-dark.png">
  <img alt="Timestral 5M (fine-tuned) forecasting my glucose, test days block 2" src="screenshots/personal-days-timestral-5m-ft-2.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-28m-ft-1-dark.png">
  <img alt="Timestral 28M (fine-tuned) forecasting my glucose, test days block 1" src="screenshots/personal-days-timestral-28m-ft-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-28m-ft-2-dark.png">
  <img alt="Timestral 28M (fine-tuned) forecasting my glucose, test days block 2" src="screenshots/personal-days-timestral-28m-ft-2.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-123m-ft-1-dark.png">
  <img alt="Timestral 123M (fine-tuned) forecasting my glucose, test days block 1" src="screenshots/personal-days-timestral-123m-ft-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-123m-ft-2-dark.png">
  <img alt="Timestral 123M (fine-tuned) forecasting my glucose, test days block 2" src="screenshots/personal-days-timestral-123m-ft-2.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-209m-ft-1-dark.png">
  <img alt="Timestral 209M (fine-tuned) forecasting my glucose, test days block 1" src="screenshots/personal-days-timestral-209m-ft-1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-days-timestral-209m-ft-2-dark.png">
  <img alt="Timestral 209M (fine-tuned) forecasting my glucose, test days block 2" src="screenshots/personal-days-timestral-209m-ft-2.png">
</picture>

## 8. Limitations

- My data is N=1 over 35 days.
- The test sets differ a lot (in range 89% vs 75%, CV 28% vs 40%), so don't compare numbers
  across them.
- Lows are rare: 292 forecasts on my data, 8347 on MetaboNet.
- T1DMAI fell back to the last reading on 0.7% of my forecasts and 0.4% on MetaboNet, including
  all 28 ShanghaiT1DM ones. The other models never fall back.
- The bands are raw quantiles, no conformal correction.
- The sparse GP's own uncertainty isn't shown. Its noise fit hit the bottom of its grid, giving
  a band about 1 mg/dL wide.
- Time gain is only computed for my data.
- Unlogged meals count as "no meal".
- CPU timings are single runs on one laptop core and varied by up to 45% between runs.
- LightGBM, LSTM (15 h), Chronos-2, TimesFM 2.5 and Timestral have no CPU timing yet.

## 9. Terms

| Term | Meaning |
| --- | --- |
| Non-blind | The model gets the carbs and insulin logged after the forecast moment. |
| Blind | It doesn't. |
| Time gain | Horizon minus the lag of peak correlation between forecast and CGM. |
| Example forecasts | Median case of: falls into a low, rises over 40 mg/dL after a meal, calm nights. |
| After a meal | At least 5 g of carbs logged in the 2 hours before. |
| Parameter count | Stored weights and constants in the exported model. |

## 10. Full tables

A * marks the lowest RMSE and the models not significantly worse than it.

<details>
<summary>My data</summary>

#### Accuracy: RMSE, MARD, bias (mg/dL, %)

| Model | Family | RMSE 30 | RMSE 60 | RMSE 90 | RMSE 120 | gRMSE 60 | MARD 30 | MARD 60 | MARD 120 | ME 60 | MRE 60 % | TG 30 | TG 60 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | Classical | 14.6 * | 22.4 * | 27.5 * | 31.6 * | 24.3 | 8.5 | 13.4 | 20.1 | +1.1 | +3.3 | 15 | 30 |
| LightGBM | Classical | 14.6 * | 22.4 * | 27.6 * | 31.5 * | 24.4 | 8.5 | 13.4 | 20.2 | +1.1 | +3.3 | 15 | 30 |
| T1DMAI D16 DILATE (sim + LoRA) | T1DMAI | 15.2 * | 22.8 * | 27.4 * | 31.7 * | 24.7 | 8.8 | 13.5 | 20.3 | -1.6 | +1.2 | 15 | 40 |
| LSTM (15 h) | Deep | 15.1 | 23.2 | 28.6 * | 32.9 | 24.9 | 8.9 | 14.4 | 21.9 | +2.5 | +4.4 | 10 | 20 |
| PatchTST | Deep | 15.1 | 23.3 | 28.8 * | 33.0 | 25.0 | 8.9 | 14.3 | 21.8 | +1.5 | +3.4 | 10 | 20 |
| SVR | Classical | 15.2 * | 23.4 * | 28.4 * | 31.8 * | 25.3 | 8.6 | 13.7 | 20.0 | -0.9 | +1.6 | 10 | 20 |
| T1DMAI D16 MSE (sim + LoRA) | T1DMAI | 15.1 * | 23.5 * | 28.4 * | 32.4 * | 25.4 | 8.7 | 13.9 | 20.3 | -0.2 | +2.0 | 20 | 50 |
| Sparse GP | Classical | 15.4 | 23.7 | 28.9 * | 32.8 * | 25.8 | 9.0 | 14.4 | 21.2 | +0.7 | +3.1 | 10 | 20 |
| GluNet | Deep | 15.4 | 24.0 | 29.8 | 34.5 | 25.6 | 8.9 | 14.4 | 22.3 | +1.0 | +2.8 | 10 | 20 |
| Timestral 28M (fine-tuned) | Foundation | 15.4 * | 24.2 * | 29.8 * | 33.6 * | 26.5 | 8.7 | 14.1 | 21.0 | -1.6 | +1.2 | 10 | 25 |
| Timestral 123M (fine-tuned) | Foundation | 15.3 * | 24.2 | 30.5 | 34.7 | 26.3 | 8.7 | 14.3 | 21.8 | -0.6 | +1.8 | 15 | 30 |
| LSTM | Deep | 15.3 | 24.2 | 30.6 | 35.7 | 25.7 | 8.8 | 14.5 | 22.8 | +0.4 | +2.0 | 10 | 20 |
| ARX (ridge) | Classical | 15.8 | 24.3 | 28.9 * | 31.8 * | 26.5 | 9.4 | 15.2 | 21.3 | +0.9 | +3.9 | 5 | 15 |
| N-HiTS | Deep | 15.5 | 24.7 | 30.6 | 34.6 | 26.2 | 9.1 | 15.0 | 22.1 | +2.7 | +3.9 | 10 | 20 |
| Timestral 209M (fine-tuned) | Foundation | 15.3 * | 24.7 | 31.2 | 35.4 | 26.8 | 8.7 | 14.6 | 22.4 | -0.1 | +2.2 | 15 | 30 |
| Timestral 5M (fine-tuned) | Foundation | 16.2 | 25.4 | 30.8 | 34.4 | 28.1 | 9.1 | 14.9 | 22.3 | -1.6 | +1.6 | 10 | 20 |
| T1DMAI D16 DILATE (sim only) | T1DMAI | 16.2 | 26.1 | 33.2 | 39.9 | 28.6 | 9.6 | 16.0 | 25.2 | +0.6 | +3.9 | 10 | 15 |
| T1DMAI D16 MSE (sim only) | T1DMAI | 16.8 | 27.1 | 33.9 | 37.5 | 29.8 | 9.9 | 16.8 | 24.4 | +1.2 | +4.8 | 10 | 15 |
| AR (ridge) | Classical | 17.2 | 27.5 | 32.5 | 34.7 | 30.3 | 10.3 | 17.2 | 23.1 | -0.0 | +4.1 | 5 | 5 |
| Chronos-Bolt tiny (fine-tuned) | Foundation | 17.5 | 27.5 | 33.1 | 36.1 | 30.1 | 9.8 | 16.4 | 23.5 | -2.2 | +1.6 | 5 | 10 |
| Chronos-Bolt mini (fine-tuned) | Foundation | 17.5 | 27.6 | 33.1 | 36.1 | 30.1 | 9.8 | 16.5 | 23.7 | -1.8 | +1.9 | 5 | 10 |
| Timestral 123M (zero-shot) | Foundation | 16.9 | 27.8 | 34.4 | 38.2 | 30.1 | 9.6 | 16.5 | 24.1 | -2.8 | +0.7 | 5 | 10 |
| Timestral 5M (zero-shot) | Foundation | 17.1 | 28.0 | 34.4 | 37.7 | 30.4 | 9.8 | 16.7 | 24.1 | -2.7 | +1.0 | 5 | 10 |
| Timestral 209M (zero-shot) | Foundation | 17.1 | 28.4 | 35.2 | 39.2 | 30.5 | 9.6 | 16.7 | 24.6 | -2.4 | +0.9 | 5 | 10 |
| Timestral 28M (zero-shot) | Foundation | 17.2 | 28.4 | 35.3 | 39.2 | 30.6 | 9.8 | 16.8 | 24.8 | -2.8 | +0.7 | 5 | 10 |
| TimesFM 2.5 (zero-shot) | Foundation | 18.4 | 29.1 | 35.8 | 39.4 | 31.5 | 10.5 | 17.4 | 25.1 | -1.5 | +2.0 | 5 | 5 |
| Chronos-2 (zero-shot) | Foundation | 17.8 | 29.4 | 36.4 | 40.0 | 31.8 | 10.2 | 17.7 | 25.5 | -1.0 | +2.5 | 5 | 10 |
| SARIMAX | Classical | 17.7 | 29.8 | 38.7 | 45.5 | 31.6 | 10.2 | 17.9 | 28.8 | -3.8 | -0.9 | 5 | 5 |
| Chronos-Bolt tiny (zero-shot) | Foundation | 18.7 | 30.8 | 38.3 | 42.0 | 32.9 | 10.8 | 18.5 | 27.1 | +0.8 | +3.5 | 5 | 5 |
| Chronos-Bolt mini (zero-shot) | Foundation | 18.7 | 30.9 | 38.3 | 42.4 | 32.9 | 10.7 | 18.5 | 27.2 | +1.0 | +3.6 | 5 | 10 |
| Kalman filter | Classical | 18.5 | 31.3 | 39.6 | 45.0 | 33.2 | 10.6 | 18.7 | 28.6 | +0.2 | +2.7 | 0 | 5 |
| ARIMA | Classical | 18.7 | 31.5 | 39.8 | 45.1 | 33.3 | 10.7 | 18.8 | 28.7 | +0.2 | +2.7 | 0 | 5 |
| Persistence | Baseline | 20.0 | 32.1 | 39.7 | 44.7 | 34.2 | 11.7 | 19.4 | 28.5 | +0.2 | +3.3 | 0 | 0 |
| Linear extrapolation | Baseline | 24.3 | 48.9 | 71.0 | 89.2 | 50.1 | 13.3 | 27.9 | 56.4 | +0.1 | -0.3 | 10 | 15 |

#### Clinical: DTS, Clarke, CG-EGA (% of points)

| Model | DTS A 30 | DTS A 60 | DTS A 120 | DTS A+B 120 | Clarke A 60 | Clarke A+B 120 | CG-EGA hypo AP 30 | hypo EP 60 | eu EP 60 | hyper EP 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 91.3 | 77.9 | 62.7 | 96.9 | 78.3 | 96.6 | 78.4 | 51.3 | 0.0 | 0.1 |
| LightGBM | 91.4 | 78.0 | 62.1 | 96.8 | 78.3 | 96.6 | 77.7 | 57.7 | 0.0 | 0.1 |
| T1DMAI D16 DILATE (sim + LoRA) | 90.7 | 78.0 | 60.7 | 96.8 | 78.3 | 97.0 | 83.2 | 38.4 | 0.0 | 0.1 |
| LSTM (15 h) | 90.9 | 75.0 | 58.1 | 96.6 | 75.3 | 96.7 | 81.9 | 49.4 | 0.0 | 0.1 |
| PatchTST | 90.8 | 75.8 | 57.8 | 96.7 | 76.1 | 96.8 | 82.9 | 46.8 | 0.0 | 0.7 |
| SVR | 91.2 | 77.4 | 61.2 | 97.1 | 77.8 | 97.0 | 81.6 | 45.8 | 0.0 | 0.5 |
| T1DMAI D16 MSE (sim + LoRA) | 91.2 | 76.4 | 60.0 | 96.7 | 76.7 | 97.0 | 88.1 | 35.5 | 0.0 | 0.3 |
| Sparse GP | 90.2 | 75.2 | 58.4 | 96.8 | 75.6 | 96.8 | 76.8 | 57.4 | 0.0 | 0.3 |
| GluNet | 90.5 | 75.4 | 55.2 | 96.3 | 75.6 | 96.8 | 85.8 | 41.3 | 0.0 | 0.3 |
| Timestral 28M (fine-tuned) | 90.9 | 76.1 | 59.0 | 96.6 | 76.3 | 96.8 | 81.6 | 43.5 | 0.0 | 0.7 |
| Timestral 123M (fine-tuned) | 91.1 | 76.0 | 57.7 | 96.1 | 76.4 | 96.8 | 83.2 | 37.4 | 0.0 | 0.4 |
| LSTM | 91.1 | 75.4 | 55.4 | 95.4 | 75.8 | 97.1 | 89.7 | 31.9 | 0.0 | 0.1 |
| ARX (ridge) | 89.9 | 73.8 | 59.1 | 96.7 | 74.1 | 96.7 | 72.9 | 69.7 | 0.0 | 0.1 |
| N-HiTS | 90.7 | 73.4 | 56.6 | 96.4 | 73.8 | 97.0 | 85.2 | 35.8 | 0.0 | 0.1 |
| Timestral 209M (fine-tuned) | 91.4 | 74.9 | 56.4 | 96.2 | 75.1 | 96.9 | 83.5 | 42.9 | 0.0 | 0.7 |
| Timestral 5M (fine-tuned) | 90.0 | 74.1 | 57.5 | 95.8 | 74.5 | 96.5 | 79.4 | 51.6 | 0.0 | 1.4 |
| T1DMAI D16 DILATE (sim only) | 89.0 | 72.1 | 50.8 | 93.0 | 72.4 | 95.8 | 73.5 | 62.6 | 0.1 | 0.4 |
| T1DMAI D16 MSE (sim only) | 88.2 | 69.7 | 54.0 | 93.9 | 70.1 | 96.2 | 69.4 | 71.9 | 0.1 | 0.5 |
| AR (ridge) | 87.1 | 67.6 | 55.4 | 95.9 | 67.7 | 96.5 | 55.5 | 96.8 | 0.0 | 0.3 |
| Chronos-Bolt tiny (fine-tuned) | 88.2 | 69.6 | 54.0 | 95.7 | 69.9 | 96.4 | 75.5 | 69.4 | 0.0 | 0.3 |
| Chronos-Bolt mini (fine-tuned) | 88.1 | 69.4 | 54.1 | 95.4 | 69.7 | 96.4 | 81.3 | 71.0 | 0.0 | 0.4 |
| Timestral 123M (zero-shot) | 88.9 | 69.2 | 52.9 | 94.9 | 69.6 | 96.3 | 82.9 | 55.5 | 0.0 | 0.1 |
| Timestral 5M (zero-shot) | 88.5 | 69.0 | 52.4 | 94.9 | 69.3 | 96.4 | 78.4 | 60.0 | 0.0 | 0.4 |
| Timestral 209M (zero-shot) | 88.7 | 68.8 | 52.1 | 94.5 | 69.2 | 96.3 | 85.8 | 51.6 | 0.0 | 0.1 |
| Timestral 28M (zero-shot) | 88.4 | 68.1 | 52.0 | 94.6 | 68.5 | 96.4 | 81.3 | 55.2 | 0.0 | 0.4 |
| TimesFM 2.5 (zero-shot) | 86.1 | 67.3 | 51.8 | 93.9 | 67.6 | 96.5 | 78.4 | 61.0 | 0.0 | 0.3 |
| Chronos-2 (zero-shot) | 87.1 | 66.6 | 51.6 | 93.8 | 66.9 | 95.8 | 82.9 | 63.5 | 0.0 | 0.4 |
| SARIMAX | 87.5 | 64.8 | 43.5 | 89.6 | 65.3 | 97.4 | 91.3 | 26.5 | 0.0 | 0.3 |
| Chronos-Bolt tiny (zero-shot) | 86.0 | 64.4 | 49.5 | 92.4 | 64.7 | 95.8 | 79.4 | 53.9 | 0.0 | 0.3 |
| Chronos-Bolt mini (zero-shot) | 86.2 | 64.6 | 49.9 | 92.3 | 64.9 | 95.6 | 82.6 | 54.5 | 0.0 | 0.3 |
| Kalman filter | 86.1 | 64.0 | 47.1 | 90.8 | 64.3 | 96.1 | 85.2 | 46.8 | 0.1 | 0.3 |
| ARIMA | 85.9 | 63.9 | 47.0 | 90.8 | 64.3 | 96.0 | 84.5 | 46.5 | 0.1 | 0.3 |
| Persistence | 82.9 | 62.5 | 47.5 | 91.0 | 63.1 | 96.1 | 79.7 | 51.3 | 0.0 | 0.3 |
| Linear extrapolation | 78.7 | 50.5 | 25.9 | 65.9 | 51.3 | 83.2 | 91.9 | 23.2 | 2.0 | 2.6 |

#### Hypoglycaemia (< 70 mg/dL at the horizon)

| Model | Sens 30 | Sens 60 | Spec 60 | PPV 60 | MCC 30 | MCC 60 | RMSE<70 30 | ME<70 30 | ME<70 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.49 | 0.28 | 0.996 | 0.71 | 0.63 | 0.43 | 12.2 | +8.3 | +17.1 |
| LightGBM | 0.52 | 0.24 | 0.996 | 0.64 | 0.64 | 0.38 | 12.3 | +8.7 | +17.8 |
| T1DMAI D16 DILATE (sim + LoRA) | 0.63 | 0.40 | 0.993 | 0.65 | 0.69 | 0.50 | 11.3 | +6.2 | +13.0 |
| LSTM (15 h) | 0.56 | 0.35 | 0.992 | 0.58 | 0.63 | 0.44 | 10.9 | +6.2 | +13.9 |
| PatchTST | 0.59 | 0.40 | 0.990 | 0.54 | 0.64 | 0.45 | 11.2 | +4.7 | +11.5 |
| SVR | 0.59 | 0.34 | 0.992 | 0.55 | 0.64 | 0.42 | 11.3 | +6.6 | +14.1 |
| T1DMAI D16 MSE (sim + LoRA) | 0.66 | 0.47 | 0.992 | 0.64 | 0.70 | 0.54 | 10.3 | +4.7 | +10.9 |
| Sparse GP | 0.53 | 0.24 | 0.995 | 0.61 | 0.61 | 0.37 | 12.6 | +8.8 | +17.9 |
| GluNet | 0.66 | 0.50 | 0.987 | 0.55 | 0.67 | 0.51 | 11.0 | +3.6 | +9.0 |
| Timestral 28M (fine-tuned) | 0.63 | 0.42 | 0.990 | 0.58 | 0.65 | 0.48 | 10.9 | +5.7 | +13.4 |
| Timestral 123M (fine-tuned) | 0.65 | 0.50 | 0.991 | 0.62 | 0.68 | 0.54 | 11.4 | +5.6 | +13.4 |
| LSTM | 0.70 | 0.53 | 0.983 | 0.49 | 0.66 | 0.49 | 10.1 | +1.4 | +5.0 |
| ARX (ridge) | 0.41 | 0.20 | 0.995 | 0.53 | 0.51 | 0.31 | 12.9 | +9.5 | +20.7 |
| N-HiTS | 0.65 | 0.51 | 0.987 | 0.54 | 0.66 | 0.51 | 11.6 | +3.3 | +8.6 |
| Timestral 209M (fine-tuned) | 0.63 | 0.48 | 0.993 | 0.67 | 0.66 | 0.55 | 10.4 | +5.4 | +12.8 |
| Timestral 5M (fine-tuned) | 0.53 | 0.36 | 0.992 | 0.57 | 0.58 | 0.44 | 12.6 | +7.4 | +17.8 |
| T1DMAI D16 DILATE (sim only) | 0.51 | 0.21 | 0.997 | 0.69 | 0.64 | 0.37 | 13.2 | +9.2 | +20.2 |
| T1DMAI D16 MSE (sim only) | 0.42 | 0.14 | 0.997 | 0.62 | 0.58 | 0.28 | 14.1 | +10.5 | +22.3 |
| AR (ridge) | 0.23 | 0.00 | 0.999 | 0.00 | 0.33 | -0.01 | 15.8 | +13.9 | +30.3 |
| Chronos-Bolt tiny (fine-tuned) | 0.49 | 0.12 | 0.991 | 0.30 | 0.51 | 0.18 | 12.5 | +8.6 | +21.8 |
| Chronos-Bolt mini (fine-tuned) | 0.53 | 0.10 | 0.991 | 0.25 | 0.54 | 0.14 | 11.9 | +7.9 | +22.1 |
| Timestral 123M (zero-shot) | 0.58 | 0.31 | 0.983 | 0.36 | 0.55 | 0.31 | 11.6 | +6.0 | +17.4 |
| Timestral 5M (zero-shot) | 0.53 | 0.27 | 0.987 | 0.39 | 0.54 | 0.31 | 12.4 | +7.6 | +18.9 |
| Timestral 209M (zero-shot) | 0.61 | 0.37 | 0.980 | 0.37 | 0.56 | 0.35 | 11.1 | +5.3 | +15.9 |
| Timestral 28M (zero-shot) | 0.59 | 0.33 | 0.982 | 0.36 | 0.54 | 0.33 | 11.6 | +6.2 | +17.1 |
| TimesFM 2.5 (zero-shot) | 0.54 | 0.27 | 0.987 | 0.39 | 0.55 | 0.31 | 13.3 | +7.0 | +19.1 |
| Chronos-2 (zero-shot) | 0.59 | 0.24 | 0.983 | 0.31 | 0.53 | 0.25 | 11.7 | +5.7 | +21.5 |
| SARIMAX | 0.72 | 0.58 | 0.963 | 0.33 | 0.60 | 0.41 | 9.1 | +2.0 | +6.7 |
| Chronos-Bolt tiny (zero-shot) | 0.55 | 0.34 | 0.980 | 0.34 | 0.51 | 0.32 | 12.3 | +6.4 | +17.8 |
| Chronos-Bolt mini (zero-shot) | 0.60 | 0.29 | 0.982 | 0.34 | 0.54 | 0.30 | 11.9 | +5.8 | +18.5 |
| Kalman filter | 0.63 | 0.40 | 0.974 | 0.33 | 0.56 | 0.34 | 10.7 | +5.6 | +15.0 |
| ARIMA | 0.62 | 0.40 | 0.974 | 0.32 | 0.56 | 0.34 | 10.8 | +5.8 | +14.9 |
| Persistence | 0.55 | 0.36 | 0.981 | 0.36 | 0.54 | 0.34 | 12.8 | +8.1 | +18.1 |
| Linear extrapolation | 0.79 | 0.72 | 0.872 | 0.15 | 0.47 | 0.29 | 11.1 | -1.8 | -3.5 |

9661 forecasts at 60 min. * means lowest RMSE, or not significantly worse than it (Wilcoxon signed-rank test on the mean absolute-error difference per day, alpha 0.05/33).

</details>

<details>
<summary>MetaboNet</summary>

#### Accuracy: RMSE, MARD, bias (mg/dL, %)

| Model | Family | RMSE 30 | RMSE 60 | RMSE 90 | RMSE 120 | gRMSE 60 | MARD 30 | MARD 60 | MARD 120 | ME 60 | MRE 60 % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Timestral 209M (fine-tuned) | Foundation | 19.4 * | 32.6 * | 40.3 * | 44.7 * | 39.0 | 10.0 | 17.2 | 24.0 | -2.4 | +2.9 |
| Timestral 123M (fine-tuned) | Foundation | 19.5 | 32.9 | 40.5 | 45.0 | 39.3 | 10.0 | 17.2 | 24.1 | -3.0 | +2.4 |
| XGBoost | Classical | 19.8 | 33.2 | 41.3 | 46.0 | 40.3 | 10.6 | 18.6 | 27.1 | +0.5 | +6.3 |
| LightGBM | Classical | 19.8 | 33.2 | 41.3 | 46.0 | 40.3 | 10.6 | 18.7 | 27.1 | +0.6 | +6.4 |
| Timestral 28M (fine-tuned) | Foundation | 19.8 | 33.3 | 41.0 | 45.5 | 39.9 | 10.2 | 17.4 | 24.3 | -3.1 | +2.3 |
| Sparse GP | Classical | 20.4 | 33.7 | 41.6 | 46.2 | 40.8 | 10.9 | 18.9 | 27.2 | +0.4 | +6.2 |
| Timestral 5M (fine-tuned) | Foundation | 20.0 | 33.8 | 41.6 | 46.0 | 40.4 | 10.3 | 17.7 | 24.6 | -3.2 | +2.3 |
| LSTM (15 h) | Deep | 20.6 | 33.9 | 41.8 | 46.5 | 40.7 | 10.9 | 18.7 | 27.0 | -0.1 | +5.3 |
| SVR | Classical | 20.4 | 34.0 | 42.0 | 46.8 | 41.0 | 10.6 | 18.0 | 25.2 | -3.3 | +2.8 |
| N-HiTS | Deep | 20.6 | 34.3 | 42.7 | 47.8 | 40.8 | 11.0 | 18.9 | 27.5 | +0.5 | +5.4 |
| PatchTST | Deep | 20.4 | 34.5 | 42.9 | 48.0 | 40.7 | 10.7 | 19.0 | 27.7 | +0.8 | +5.6 |
| Chronos-Bolt mini (fine-tuned) | Foundation | 21.0 | 34.5 | 42.0 | 46.2 | 40.8 | 10.8 | 18.1 | 25.0 | -2.1 | +3.0 |
| T1DMAI D16 DILATE (sim + LoRA) | T1DMAI | 21.5 | 34.6 | 41.0 | 50.4 | 41.2 | 10.9 | 18.0 | 25.2 | -3.5 | +2.0 |
| Chronos-Bolt tiny (fine-tuned) | Foundation | 21.1 | 34.6 | 42.1 | 46.2 | 41.0 | 10.9 | 18.2 | 25.0 | -2.3 | +2.9 |
| GluNet | Deep | 20.5 | 34.8 | 43.6 | 49.0 | 41.1 | 10.8 | 18.9 | 27.8 | +0.4 | +5.0 |
| Timestral 5M (zero-shot) | Foundation | 20.5 | 35.0 | 43.5 | 48.5 | 41.4 | 10.5 | 18.2 | 25.8 | -2.8 | +2.2 |
| TimesFM 2.5 (zero-shot) | Foundation | 20.7 | 35.0 | 43.3 | 47.9 | 41.4 | 10.7 | 18.6 | 26.1 | -1.7 | +3.2 |
| T1DMAI D16 MSE (sim + LoRA) | T1DMAI | 21.8 | 35.1 | 41.0 | 46.8 | 41.3 | 11.2 | 18.1 | 25.1 | -3.8 | +1.3 |
| Timestral 123M (zero-shot) | Foundation | 20.5 | 35.1 | 43.6 | 48.5 | 41.4 | 10.5 | 18.2 | 25.7 | -3.2 | +1.8 |
| Timestral 209M (zero-shot) | Foundation | 20.4 | 35.1 | 43.7 | 48.7 | 41.2 | 10.5 | 18.3 | 26.1 | -2.3 | +2.4 |
| Timestral 28M (zero-shot) | Foundation | 20.5 | 35.2 | 43.8 | 48.9 | 41.4 | 10.5 | 18.2 | 25.9 | -2.9 | +1.9 |
| Chronos-2 (zero-shot) | Foundation | 21.0 | 35.3 | 43.4 | 48.0 | 41.6 | 10.8 | 18.7 | 26.2 | -1.3 | +3.6 |
| ARX (ridge) | Classical | 21.3 | 35.4 | 43.4 | 47.9 | 42.3 | 11.6 | 20.4 | 29.4 | +2.5 | +8.0 |
| AR (ridge) | Classical | 21.3 | 35.5 | 43.6 | 48.0 | 42.4 | 11.7 | 20.6 | 29.9 | +3.1 | +8.4 |
| LSTM | Deep | 21.2 | 35.9 | 45.2 | 51.0 | 41.9 | 11.1 | 19.4 | 28.6 | +0.5 | +4.7 |
| T1DMAI D16 DILATE (sim only) | T1DMAI | 23.4 | 36.6 | 44.0 | 48.0 | 44.0 | 12.2 | 20.4 | 28.8 | +0.0 | +6.0 |
| T1DMAI D16 MSE (sim only) | T1DMAI | 23.4 | 36.7 | 44.1 | 48.2 | 43.8 | 13.1 | 21.7 | 29.7 | +4.4 | +9.7 |
| SARIMAX | Classical | 21.8 | 36.9 | 46.4 | 52.3 | 42.9 | 11.5 | 20.4 | 30.3 | +1.5 | +5.7 |
| Chronos-Bolt tiny (zero-shot) | Foundation | 22.8 | 37.7 | 46.4 | 51.3 | 43.3 | 11.9 | 20.1 | 28.3 | +0.7 | +4.4 |
| Chronos-Bolt mini (zero-shot) | Foundation | 22.9 | 37.7 | 46.3 | 51.4 | 43.4 | 11.8 | 19.9 | 28.2 | +0.1 | +3.8 |
| Kalman filter | Classical | 22.2 | 38.4 | 48.9 | 55.6 | 43.9 | 11.4 | 20.3 | 30.3 | +0.2 | +3.5 |
| ARIMA | Classical | 22.5 | 39.3 | 50.2 | 57.4 | 44.7 | 11.6 | 20.9 | 31.5 | +0.2 | +3.4 |
| Persistence | Baseline | 25.2 | 40.0 | 49.4 | 55.5 | 46.2 | 13.1 | 21.3 | 30.3 | +0.3 | +4.4 |
| Linear extrapolation | Baseline | 30.7 | 59.2 | 81.8 | 98.8 | 64.0 | 15.6 | 31.4 | 56.8 | +0.1 | +0.7 |

#### Clinical: DTS, Clarke, CG-EGA (% of points)

| Model | DTS A 30 | DTS A 60 | DTS A 120 | DTS A+B 120 | Clarke A 60 | Clarke A+B 120 | CG-EGA hypo AP 30 | hypo EP 60 | eu EP 60 | hyper EP 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Timestral 209M (fine-tuned) | 87.2 | 69.7 | 54.8 | 94.4 | 69.9 | 93.5 | 65.1 | 78.6 | 0.1 | 2.9 |
| Timestral 123M (fine-tuned) | 87.2 | 69.6 | 54.6 | 94.3 | 69.8 | 93.5 | 67.5 | 76.8 | 0.1 | 2.9 |
| XGBoost | 85.9 | 67.0 | 49.8 | 93.0 | 67.1 | 93.1 | 33.1 | 99.6 | 0.1 | 3.1 |
| LightGBM | 85.9 | 66.9 | 49.8 | 93.0 | 67.0 | 93.1 | 33.2 | 99.5 | 0.1 | 3.1 |
| Timestral 28M (fine-tuned) | 86.8 | 69.0 | 54.0 | 94.2 | 69.1 | 93.4 | 67.4 | 76.5 | 0.1 | 3.1 |
| Sparse GP | 85.3 | 66.4 | 49.8 | 92.9 | 66.5 | 93.1 | 43.3 | 98.1 | 0.1 | 3.1 |
| Timestral 5M (fine-tuned) | 86.6 | 68.3 | 53.2 | 93.9 | 68.6 | 93.3 | 68.8 | 76.1 | 0.1 | 3.3 |
| LSTM (15 h) | 85.2 | 66.6 | 49.8 | 93.0 | 66.8 | 93.3 | 53.9 | 86.0 | 0.1 | 2.9 |
| SVR | 86.0 | 68.0 | 52.1 | 93.6 | 68.2 | 92.9 | 62.4 | 89.4 | 0.1 | 3.3 |
| N-HiTS | 85.1 | 66.3 | 49.3 | 92.4 | 66.5 | 93.3 | 57.4 | 80.0 | 0.1 | 2.9 |
| PatchTST | 85.7 | 65.9 | 48.8 | 92.3 | 66.1 | 93.3 | 63.0 | 80.8 | 0.1 | 2.6 |
| Chronos-Bolt mini (fine-tuned) | 85.3 | 67.7 | 53.1 | 93.7 | 67.9 | 93.3 | 64.1 | 78.7 | 0.1 | 3.0 |
| T1DMAI D16 DILATE (sim + LoRA) | 84.9 | 67.8 | 49.5 | 93.1 | 68.0 | 91.8 | 71.7 | 79.8 | 0.1 | 3.2 |
| Chronos-Bolt tiny (fine-tuned) | 85.1 | 67.5 | 53.0 | 93.7 | 67.7 | 93.3 | 59.8 | 78.6 | 0.1 | 3.1 |
| GluNet | 85.5 | 66.2 | 48.5 | 92.0 | 66.4 | 93.1 | 65.0 | 76.2 | 0.1 | 2.8 |
| Timestral 5M (zero-shot) | 86.0 | 67.1 | 51.5 | 92.6 | 67.4 | 93.3 | 69.0 | 71.6 | 0.1 | 3.4 |
| TimesFM 2.5 (zero-shot) | 85.6 | 66.6 | 51.3 | 92.7 | 66.8 | 93.3 | 67.6 | 73.8 | 0.1 | 3.1 |
| T1DMAI D16 MSE (sim + LoRA) | 84.4 | 67.7 | 51.5 | 94.0 | 67.9 | 92.4 | 68.5 | 74.8 | 0.1 | 2.9 |
| Timestral 123M (zero-shot) | 86.1 | 67.2 | 51.6 | 92.7 | 67.4 | 93.2 | 71.4 | 69.1 | 0.1 | 3.4 |
| Timestral 209M (zero-shot) | 86.1 | 67.1 | 51.2 | 92.4 | 67.3 | 93.2 | 71.2 | 69.2 | 0.1 | 3.2 |
| Timestral 28M (zero-shot) | 86.0 | 67.1 | 51.3 | 92.5 | 67.3 | 93.2 | 71.4 | 68.6 | 0.1 | 3.3 |
| Chronos-2 (zero-shot) | 85.3 | 66.3 | 51.0 | 92.7 | 66.6 | 93.2 | 68.9 | 74.6 | 0.1 | 3.3 |
| ARX (ridge) | 83.9 | 62.2 | 45.5 | 92.0 | 62.4 | 93.2 | 52.0 | 93.3 | 0.1 | 2.9 |
| AR (ridge) | 83.7 | 61.8 | 45.1 | 91.7 | 62.0 | 93.2 | 51.7 | 93.6 | 0.1 | 2.8 |
| LSTM | 84.8 | 65.0 | 47.7 | 91.1 | 65.3 | 92.8 | 67.5 | 71.3 | 0.2 | 2.8 |
| T1DMAI D16 DILATE (sim only) | 82.3 | 62.4 | 45.3 | 92.4 | 62.6 | 92.7 | 56.0 | 86.3 | 0.1 | 3.3 |
| T1DMAI D16 MSE (sim only) | 80.1 | 59.2 | 45.0 | 91.8 | 59.3 | 93.3 | 34.6 | 93.9 | 0.1 | 3.1 |
| SARIMAX | 83.9 | 63.6 | 46.9 | 89.9 | 63.8 | 92.7 | 62.7 | 75.6 | 0.2 | 2.6 |
| Chronos-Bolt tiny (zero-shot) | 82.8 | 63.9 | 49.2 | 90.9 | 64.1 | 92.8 | 66.9 | 67.9 | 0.3 | 3.0 |
| Chronos-Bolt mini (zero-shot) | 83.1 | 64.2 | 49.4 | 91.0 | 64.4 | 92.7 | 70.7 | 66.9 | 0.3 | 3.1 |
| Kalman filter | 83.9 | 63.3 | 46.3 | 88.5 | 63.6 | 92.2 | 74.7 | 61.0 | 0.3 | 2.9 |
| ARIMA | 83.5 | 61.8 | 44.4 | 87.6 | 62.1 | 91.8 | 75.2 | 59.9 | 0.3 | 3.0 |
| Persistence | 79.6 | 61.0 | 46.1 | 88.7 | 61.5 | 92.1 | 57.3 | 68.3 | 0.2 | 3.4 |
| Linear extrapolation | 74.9 | 48.6 | 26.8 | 67.0 | 49.4 | 78.0 | 83.5 | 34.3 | 3.6 | 7.8 |

#### Hypoglycaemia (< 70 mg/dL at the horizon)

| Model | Sens 30 | Sens 60 | Spec 60 | PPV 60 | MCC 30 | MCC 60 | RMSE<70 30 | ME<70 30 | ME<70 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Timestral 209M (fine-tuned) | 0.38 | 0.10 | 0.995 | 0.40 | 0.43 | 0.19 | 21.5 | +14.3 | +33.8 |
| Timestral 123M (fine-tuned) | 0.41 | 0.11 | 0.994 | 0.37 | 0.44 | 0.19 | 20.9 | +13.6 | +32.6 |
| XGBoost | 0.06 | 0.00 | 1.000 | 0.71 | 0.20 | 0.03 | 26.4 | +22.3 | +45.7 |
| LightGBM | 0.07 | 0.00 | 1.000 | 0.65 | 0.21 | 0.03 | 26.5 | +22.4 | +45.8 |
| Timestral 28M (fine-tuned) | 0.41 | 0.11 | 0.993 | 0.35 | 0.44 | 0.18 | 21.0 | +13.5 | +32.5 |
| Sparse GP | 0.18 | 0.00 | 1.000 | 0.30 | 0.29 | 0.03 | 25.8 | +20.4 | +44.4 |
| Timestral 5M (fine-tuned) | 0.43 | 0.12 | 0.992 | 0.32 | 0.43 | 0.18 | 20.7 | +12.9 | +32.2 |
| LSTM (15 h) | 0.31 | 0.05 | 0.997 | 0.32 | 0.37 | 0.12 | 24.9 | +17.7 | +39.0 |
| SVR | 0.34 | 0.03 | 0.998 | 0.34 | 0.39 | 0.09 | 22.6 | +15.6 | +37.6 |
| N-HiTS | 0.34 | 0.10 | 0.992 | 0.28 | 0.39 | 0.15 | 24.4 | +16.3 | +37.0 |
| PatchTST | 0.38 | 0.09 | 0.992 | 0.27 | 0.41 | 0.14 | 23.0 | +14.7 | +36.3 |
| Chronos-Bolt mini (fine-tuned) | 0.38 | 0.09 | 0.994 | 0.34 | 0.40 | 0.16 | 22.2 | +14.6 | +34.1 |
| T1DMAI D16 DILATE (sim + LoRA) | 0.42 | 0.06 | 0.997 | 0.38 | 0.41 | 0.15 | 20.6 | +13.3 | +33.4 |
| Chronos-Bolt tiny (fine-tuned) | 0.35 | 0.09 | 0.993 | 0.31 | 0.39 | 0.15 | 22.8 | +15.7 | +34.1 |
| GluNet | 0.41 | 0.13 | 0.988 | 0.26 | 0.42 | 0.16 | 22.8 | +13.9 | +34.9 |
| Timestral 5M (zero-shot) | 0.44 | 0.16 | 0.985 | 0.26 | 0.43 | 0.18 | 20.8 | +12.4 | +30.7 |
| TimesFM 2.5 (zero-shot) | 0.44 | 0.14 | 0.987 | 0.26 | 0.43 | 0.17 | 21.4 | +12.8 | +32.6 |
| T1DMAI D16 MSE (sim + LoRA) | 0.40 | 0.08 | 0.996 | 0.37 | 0.40 | 0.16 | 21.7 | +14.2 | +32.1 |
| Timestral 123M (zero-shot) | 0.47 | 0.18 | 0.983 | 0.25 | 0.44 | 0.19 | 20.4 | +11.4 | +29.7 |
| Timestral 209M (zero-shot) | 0.47 | 0.18 | 0.983 | 0.25 | 0.44 | 0.19 | 20.5 | +11.5 | +30.0 |
| Timestral 28M (zero-shot) | 0.47 | 0.18 | 0.982 | 0.25 | 0.44 | 0.19 | 20.3 | +11.3 | +29.5 |
| Chronos-2 (zero-shot) | 0.46 | 0.14 | 0.986 | 0.26 | 0.43 | 0.17 | 21.6 | +12.0 | +33.0 |
| ARX (ridge) | 0.28 | 0.03 | 0.996 | 0.19 | 0.33 | 0.07 | 24.3 | +17.4 | +42.6 |
| AR (ridge) | 0.28 | 0.03 | 0.996 | 0.19 | 0.33 | 0.06 | 24.0 | +17.4 | +42.9 |
| LSTM | 0.45 | 0.17 | 0.981 | 0.22 | 0.42 | 0.17 | 22.8 | +13.1 | +33.0 |
| T1DMAI D16 DILATE (sim only) | 0.32 | 0.07 | 0.991 | 0.20 | 0.34 | 0.10 | 24.9 | +16.9 | +39.3 |
| T1DMAI D16 MSE (sim only) | 0.18 | 0.03 | 0.997 | 0.22 | 0.27 | 0.07 | 28.6 | +22.5 | +45.1 |
| SARIMAX | 0.40 | 0.14 | 0.984 | 0.22 | 0.39 | 0.15 | 23.3 | +14.3 | +36.5 |
| Chronos-Bolt tiny (zero-shot) | 0.46 | 0.20 | 0.976 | 0.21 | 0.41 | 0.18 | 23.2 | +12.7 | +31.9 |
| Chronos-Bolt mini (zero-shot) | 0.49 | 0.20 | 0.976 | 0.21 | 0.41 | 0.18 | 22.4 | +11.5 | +31.0 |
| Kalman filter | 0.52 | 0.26 | 0.963 | 0.19 | 0.43 | 0.19 | 20.7 | +9.8 | +28.3 |
| ARIMA | 0.54 | 0.28 | 0.959 | 0.18 | 0.43 | 0.20 | 20.7 | +9.3 | +28.0 |
| Persistence | 0.37 | 0.19 | 0.975 | 0.19 | 0.35 | 0.16 | 26.1 | +16.3 | +33.6 |
| Linear extrapolation | 0.72 | 0.58 | 0.861 | 0.12 | 0.41 | 0.21 | 27.2 | -1.4 | +12.3 |

264893 forecasts at 60 min. * means lowest RMSE, or not significantly worse than it (Wilcoxon signed-rank test on the mean absolute-error difference per subject, alpha 0.05/33).

</details>

Raw numbers are in [results/](results/).

## 11. Reproduce

```bash
python -m zoo.run MODEL --cache CACHE --template template.parquet --truth truth.parquet [--causal]
python finetune.py --checkpoint CKPT --cache CACHE --lora-rank 1 --val-days 10 --select-on rmse \
  [--blind | --no-blind]
python export_submission.py --checkpoint CKPT --template template.parquet --cache CACHE \
  --max-context 336 --fill ladder --quantiles [--future-doses zero]
python leaderboard.py MANIFEST.json --out results
```

Chronos-2, TimesFM 2.5 and Timestral run under `.venv-zoo` (`requirements-zoo.txt`).

## 12. Published scores

RMSE (mg/dL) that each model's paper reports, beside the same model on my data and on
MetaboNet. The papers use other people, splits, inputs and history lengths, so a gap between the
two sides is not a like-for-like comparison. My data is non-blind, MetaboNet blind (§8).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/published-rmse-dark.png">
  <img alt="Published RMSE at 30 and 60 min against the same models on my data and MetaboNet" src="screenshots/published-rmse.png">
</picture>

| Model | Paper | Paper data | Paper inputs | Paper 30 | Paper 60 | My data 30 | My data 60 | MetaboNet 30 | MetaboNet 60 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | [7](https://ceur-ws.org/Vol-2148/) | OhioT1DM 2018 (6) | CGM, doses, meals, activity | 20.4 | - | 14.6 | 22.4 | 19.8 | 33.2 |
| LightGBM | [3](https://arxiv.org/abs/2606.18640) | MetaboNet | CGM, carbs, insulin, time | 20.8 | 34.5 | 14.6 | 22.4 | 19.8 | 33.2 |
| T1DMAI D16 DILATE (sim + LoRA) | - | - | - | - | - | 15.2 | 22.8 | 21.5 | 34.6 |
| LSTM (15 h) | [3](https://arxiv.org/abs/2606.18640) | MetaboNet | CGM, carbs, insulin | 20.1 | 33.4 | 15.1 | 23.2 | 20.6 | 33.9 |
| PatchTST | [9](https://arxiv.org/abs/2505.08821) | DCLP3 train, OhioT1DM test (12) | CGM, carbs, insulin | 16.7 | 24.6 | 15.1 | 23.3 | 20.4 | 34.5 |
| SVR | [6](https://doi.org/10.1109/JBHI.2019.2931842) | OhioT1DM 2018 (6) | CGM, carbs, insulin | 21.8 | 34.3 | 15.2 | 23.4 | 20.4 | 34.0 |
| T1DMAI D16 MSE (sim + LoRA) | - | - | - | - | - | 15.1 | 23.5 | 21.8 | 35.1 |
| GluNet | [6](https://doi.org/10.1109/JBHI.2019.2931842) | OhioT1DM 2018 (6) | CGM, carbs, insulin | 19.3 | 31.8 | 15.4 | 24.0 | 20.5 | 34.8 |
| LSTM | [5](https://doi.org/10.1007/s41666-019-00059-y) | OhioT1DM 2018 (6) | CGM | 18.9 | 31.4 | 15.3 | 24.2 | 21.2 | 35.9 |
| ARX | [6](https://doi.org/10.1109/JBHI.2019.2931842) | OhioT1DM 2018 (6) | CGM, carbs, insulin | 21.5 | 39.5 | 15.8 | 24.3 | 21.3 | 35.4 |
| T1DMAI D16 DILATE (sim only) | - | - | - | - | - | 16.2 | 26.1 | 23.4 | 36.6 |
| T1DMAI D16 MSE (sim only) | - | - | - | - | - | 16.8 | 27.1 | 23.4 | 36.7 |
| TimesFM 2.5 (zero-shot) | [4](https://arxiv.org/abs/2606.06881) | 12 datasets, mixed | CGM | 18.8 | 31.8 | 18.4 | 29.1 | 20.7 | 35.0 |
| Chronos-2 (zero-shot) | [4](https://arxiv.org/abs/2606.06881) | 12 datasets, mixed | CGM | 19.2 | 32.0 | 17.8 | 29.4 | 21.0 | 35.3 |
| ARIMA | [8](https://ceur-ws.org/Vol-2675/) | OhioT1DM 2020 (6) | CGM, doses, meals, exercise | 23.7 | 36.4 | 18.7 | 31.5 | 22.5 | 39.3 |
| Persistence | [3](https://arxiv.org/abs/2606.18640) | MetaboNet | CGM | 26.3 | 41.5 | 20.0 | 32.1 | 25.2 | 40.0 |

RMSE in mg/dL at the single step 30 or 60 min ahead, as printed in each paper. T1DMAI has no published score.

Chronos-Bolt (Zhang et al. 2026) and N-HiTS (Potosnak et al. 2025; Sergazinov et al. 2024) are published only as RMSE averaged over every step up to the horizon, which is not the same quantity, so they have no row.

## 13. References

1. Jacobs PG, Herrero P, Facchinetti A, Vehi J, et al. Artificial Intelligence and Machine Learning for Improving Glycemic Control in Diabetes: Best Practices, Pitfalls, and Opportunities. IEEE Reviews in Biomedical Engineering 17:19-41, 2024. <https://doi.org/10.1109/RBME.2023.3331297>
2. Wolff MK, Calhoun P, Aiello EM, Qin Y, Royston SF. MetaboNet: The Largest Publicly Available Consolidated Dataset for Type 1 Diabetes Management. arXiv:2601.11505, 2026. <https://arxiv.org/abs/2601.11505>
3. Jeffries N, Wolff M, Royston S, Healey E, Mayer C, Klonoff D, Snyder M, Wang T. MetaboNet-Bench: A Multi-modal Benchmark for Glucose Forecasting in Type 1 Diabetes. arXiv:2606.18640, 2026. <https://arxiv.org/abs/2606.18640>
4. Lu B, Liang Z, Pontius R, Tang S, Prioleau T. GlucoFM-Bench: Benchmarking Time-Series Foundation Models for Blood Glucose Forecasting. arXiv:2606.06881, 2026. <https://arxiv.org/abs/2606.06881>
5. Martinsson J, Schliep A, Eliasson B, Mogren O. Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks. Journal of Healthcare Informatics Research 4(1):1-18, 2020. <https://doi.org/10.1007/s41666-019-00059-y>
6. Li K, Liu C, Zhu T, Herrero P, Georgiou P. GluNet: A Deep Learning Framework for Accurate Glucose Forecasting. IEEE Journal of Biomedical and Health Informatics 24(2):414-423, 2020. <https://doi.org/10.1109/JBHI.2019.2931842>
7. Midroni C, Leimbigler PJ, Baruah G, Kolla M, Whitehead AJ, Fossat Y. Predicting Glycemia in Type 1 Diabetes Patients: Experiments with XGBoost. KHD@IJCAI-ECAI, CEUR-WS 2148:79-84, 2018. <https://ceur-ws.org/Vol-2148/>
8. McShinsky R, Marshall B. Comparison of Forecasting Algorithms for Type 1 Diabetic Glucose Prediction on 30 and 60-Minute Prediction Horizons. KDH@ECAI, CEUR-WS 2675, 2020. <https://ceur-ws.org/Vol-2675/>
9. Altin Karagoz M, Breton MD, El Fathi A. A Comparative Study of Transformer-Based Models for Multi-Horizon Blood Glucose Prediction. IFAC EDT, 2025. arXiv:2505.08821. <https://arxiv.org/abs/2505.08821>
10. Zhang B, Cheng H-W, Yang H, et al. Evaluating Time-Series Foundation Models and Multimodal Dietary Context for CGM Forecasting. arXiv:2609.11872, 2026. <https://arxiv.org/abs/2609.11872>
11. Potosnak W, Challu C, Olivares KG, Dufendach KA, Dubrawski A. Global Deep Forecasting with Patient-Specific Pharmacokinetics. CHIL, PMLR 287, 2025. arXiv:2309.13135. <https://arxiv.org/abs/2309.13135>
12. Sergazinov R, Chun E, Rogovchenko V, Fernandes N, Kasman N, Gaynanova I. GlucoBench: Curated List of Continuous Glucose Monitoring Datasets with Prediction Benchmarks. ICLR, 2024. arXiv:2410.05780. <https://arxiv.org/abs/2410.05780>
13. Nie Y, Nguyen NH, Sinthong P, Kalagnanam J. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR, 2023. <https://arxiv.org/abs/2211.14730>
14. Challu C, Olivares KG, Oreshkin BN, Garza F, Mergenthaler-Canseco M, Dubrawski A. N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. AAAI 37(6):6989-6997, 2023. <https://doi.org/10.1609/aaai.v37i6.25854>
15. Ansari AF, Stella L, Turkmen C, et al. Chronos: Learning the Language of Time Series. TMLR, 2024. <https://arxiv.org/abs/2403.07815>
16. Ansari AF, Shchur O, Kuken J, Auer A, Han B, et al. Chronos-2: From Univariate to Universal Forecasting. arXiv:2510.15821, 2025. <https://arxiv.org/abs/2510.15821>
17. Das A, Kong W, Sen R, Zhou Y. A Decoder-Only Foundation Model for Time-Series Forecasting. ICML, PMLR 235:10148-10167, 2024. <https://arxiv.org/abs/2310.10688>
18. Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD, 785-794, 2016. <https://doi.org/10.1145/2939672.2939785>
19. Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, Ye Q, Liu T-Y. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30:3146-3154, 2017. <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree>
20. Klonoff DC, Freckmann G, Pleus S, et al. The Diabetes Technology Society Error Grid and Trend Accuracy Matrix for Glucose Monitors. Journal of Diabetes Science and Technology 18(6):1346-1361, 2024. <https://doi.org/10.1177/19322968241275701>
21. Kovatchev BP, Gonder-Frederick LA, Cox DJ, Clarke WL. Evaluating the Accuracy of Continuous Glucose-Monitoring Sensors: Continuous Glucose-Error Grid Analysis Illustrated by TheraSense Freestyle Navigator Data. Diabetes Care 27(8):1922-1928, 2004. <https://doi.org/10.2337/diacare.27.8.1922>
22. Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL. Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose. Diabetes Care 10(5):622-628, 1987. <https://doi.org/10.2337/diacare.10.5.622>
23. Hu EJ, Shen Y, Wallis P, Allen-Zhu Z, Li Y, Wang S, Wang L, Chen W. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022. <https://arxiv.org/abs/2106.09685>
24. Cuturi M, Blondel M. Soft-DTW: a Differentiable Loss Function for Time-Series. ICML, PMLR 70:894-903, 2017. <https://arxiv.org/abs/1703.01541>
25. Le Guen V, Thome N. Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models. NeurIPS, 2019. <https://arxiv.org/abs/1909.09020>
26. Su J, Ahmed M, Lu Y, Pan S, Bo W, Liu Y. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 568:127063, 2024. <https://doi.org/10.1016/j.neucom.2023.127063>
27. Mistral AI. Timestral. 2026. <https://mistral.ai/news/timestral/>

## License

Copyright 2026 0xdeadf1sh. [MIT](LICENSE). See also [T1DMSIM](https://github.com/0xdeadf1sh/T1DMSIM),
[T1DMDROID](https://github.com/0xdeadf1sh/T1DMDROID) and
[T1DMCOMMON](https://github.com/0xdeadf1sh/T1DMCOMMON).
