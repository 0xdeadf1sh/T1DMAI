# T1DMAI

Blood-glucose forecasting benchmark: T1DMAI against 18 reference models, on one person's
CGM record (non-blind) and on MetaboNet (blind). Metrics after Jacobs et al., *IEEE Rev. Biomed.
Eng.* 2024.

> [!CAUTION]
> **Research and educational use only — not a medical device.** T1DMAI is a
> research artifact. It is **not** clinically validated, has **no** regulatory
> clearance, and its output is a forecast of research signals, not medical
> advice. It **must not** be used to make medical, diagnostic, or treatment
> decisions, to calculate or adjust insulin doses, or to manage diabetes in any
> way. For medical guidance consult a qualified healthcare professional. The
> software is provided "as is", without warranty of any kind (see
> [LICENSE](LICENSE)).

## Contents

- [Key findings](#key-findings)
- [Owner track — non-blind](#owner-track--non-blind)
- [MetaboNet track — blind](#metabonet-track--blind)
- [T1DMAI](#t1dmai-1)
- [Protocol](#protocol)
- [Metrics](#metrics)
- [Reproduce](#reproduce)
- [All figures](#all-figures)

## Key findings

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | Simulator pretraining hurts on real data. MetaboNet, D32, RMSE at 60 min: sim only 36.0 → sim + LoRA 34.3 → no simulator 32.0 mg/dL. | [paths](#metabonet-paths) |
| 2 | T1DMAI trained on MetaboNet alone takes ranks 1–5. D32: 19.3 / 32.0 / 43.8 mg/dL at 30 / 60 / 120 min; XGBoost 19.8 / 33.2 / 46.0. Seeds 42 and 43 within 0.1 mg/dL; every from-scratch run beats XGBoost at 60–120 min. | [MetaboNet](#metabonet-track--blind) |
| 3 | 80 days of one person beat 1 M simulated patients. Owner, RMSE at 60 min: XGBoost on 80 days 22.2; T1DMAI sim only 25.9 mg/dL. | [owner](#owner-track--non-blind) |
| 4 | A LoRA adapter on the same 80 days closes the gap. T1DMAI D32 + LoRA: 14.7 / 22.4 / 31.3 mg/dL; not significantly worse than XGBoost at any horizon. | [paths](#owner-paths) |
| 5 | Lowest RMSE ≠ catching lows. MetaboNet, sensitivity for < 70 mg/dL at 60 min: T1DMAI from scratch 0.03, XGBoost 0.00, Kalman filter 0.26, linear extrapolation 0.58. | [hypo](#hypo) |
| 6 | Nearly every model over-forecasts lows. Mean error when CGM < 70 mg/dL, 60 min: owner +5.2 to +31.4 (linear extrapolation −3.5), MetaboNet +12.3 to +45.6 mg/dL. | [bias](#bias) |
| 7 | T1DMAI + LoRA lags least. Owner, time gain at 60 min: 45–50 min; XGBoost 30; persistence 0. | [owner tables](#owner-tables) |
| 8 | Zero-shot foundation model ≈ ARIMA. Owner, RMSE at 60 min: Chronos-Bolt zero-shot 30.8–30.9, ARIMA 31.4, Kalman 31.3 mg/dL. | [owner](#owner-track--non-blind) |

<a id="metabonet-paths"></a>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-paths-dark.png">
  <img alt="T1DMAI training paths on MetaboNet" src="screenshots/metabonet-paths.png">
</picture>

<a id="owner-paths"></a>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-paths-dark.png">
  <img alt="T1DMAI training paths on the owner record" src="screenshots/personal-paths.png">
</picture>

<a id="hypo"></a>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-heat-mcc-dark.png">
  <img alt="Hypo detection MCC by model and horizon, MetaboNet" src="screenshots/metabonet-heat-mcc.png">
</picture>

<a id="bias"></a>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-horizon-hypo-bias-dark.png">
  <img alt="Mean error when CGM is below 70 mg/dL, owner" src="screenshots/personal-horizon-hypo-bias.png">
</picture>

## Owner track — non-blind

1 person with T1D · 115 days · train days 1–80 · test days 81–115 · future carbs and insulin as logged.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-rmse-dark.png">
  <img alt="RMSE per model, owner" src="screenshots/personal-rmse.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-significance-dark.png">
  <img alt="Wilcoxon against the lowest-RMSE model, owner" src="screenshots/personal-significance.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/personal-horizon-hypo-sens-dark.png">
  <img alt="Hypo sensitivity by horizon, owner" src="screenshots/personal-horizon-hypo-sens.png">
</picture>

<a id="owner-tables"></a>
#### Accuracy — RMSE / MARD / bias (mg/dL, %)

| Model | Family | RMSE 30 | RMSE 60 | RMSE 90 | RMSE 120 | gRMSE 60 | MARD 30 | MARD 60 | MARD 120 | ME 60 | MRE 60 % | TG 30 | TG 60 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | Classical | 14.7 ◆ | 22.2 ◆ | 26.9 ◆ | 31.1 ◆ | 24.2 | 8.5 | 13.4 | 19.8 | +0.8 | +3.1 | 15 | 30 |
| T1DMAI D32 MSE · sim + LoRA | T1DMAI | 14.7 ◆ | 22.4 ◆ | 26.9 ◆ | 31.3 ◆ | 24.3 | 8.7 | 13.3 | 19.7 | +0.1 | +2.4 | 20 | 50 |
| T1DMAI D16 DILATE · sim + LoRA | T1DMAI | 14.9 | 22.5 ◆ | 27.4 ◆ | 32.2 | 24.3 | 8.7 | 13.5 | 20.9 | -0.3 | +2.1 | 15 | 45 |
| SVR | Classical | 15.1 ◆ | 23.0 | 27.8 | 31.6 | 24.9 | 8.5 | 13.6 | 20.0 | -0.7 | +1.8 | 10 | 25 |
| T1DMAI D16 MSE · sim + LoRA | T1DMAI | 15.1 ◆ | 23.1 | 27.8 | 31.8 | 24.8 | 8.8 | 14.0 | 20.4 | +0.7 | +2.7 | 15 | 45 |
| PatchTST | Deep | 15.1 | 23.3 | 28.9 | 33.1 | 25.0 | 8.9 | 14.3 | 21.8 | +1.3 | +3.3 | 10 | 25 |
| Sparse GP | Classical | 15.4 | 23.5 | 28.3 | 32.1 | 25.6 | 9.1 | 14.4 | 21.0 | +0.8 | +3.4 | 10 | 25 |
| GluNet | Deep | 15.4 | 23.6 | 29.6 | 34.4 | 25.3 | 8.9 | 14.3 | 22.3 | +1.0 | +2.8 | 10 | 25 |
| ARX (ridge) | Classical | 15.7 | 24.1 | 28.6 | 31.6 | 26.3 | 9.4 | 15.1 | 21.2 | +1.0 | +4.0 | 10 | 15 |
| LSTM | Deep | 15.4 | 24.3 | 30.6 | 35.8 | 25.7 | 8.9 | 14.6 | 22.9 | +0.2 | +1.9 | 10 | 20 |
| N-HiTS | Deep | 15.8 | 25.0 | 31.4 | 36.0 | 26.6 | 9.5 | 15.5 | 23.3 | +2.0 | +3.5 | 15 | 30 |
| T1DMAI D32 MSE · sim only | T1DMAI | 16.2 | 25.9 | 32.4 | 36.5 | 28.5 | 9.5 | 15.9 | 23.3 | +0.2 | +3.7 | 10 | 15 |
| T1DMAI D16 DILATE · sim only | T1DMAI | 16.2 | 26.1 | 33.2 | 39.9 | 28.6 | 9.6 | 16.0 | 25.2 | +0.6 | +3.9 | 10 | 15 |
| T1DMAI D16 MSE · sim only | T1DMAI | 16.8 | 27.1 | 33.9 | 37.5 | 29.8 | 9.9 | 16.8 | 24.4 | +1.2 | +4.8 | 10 | 15 |
| AR (ridge) | Classical | 17.2 | 27.3 | 32.3 | 34.5 | 30.3 | 10.3 | 17.2 | 23.0 | -0.1 | +4.1 | 5 | 5 |
| Chronos-Bolt tiny (fine-tuned) | Foundation | 17.3 | 27.3 | 33.0 | 36.0 | 30.0 | 9.8 | 16.4 | 23.4 | -2.2 | +1.7 | 5 | 10 |
| Chronos-Bolt mini (fine-tuned) | Foundation | 17.3 | 27.3 | 32.9 | 36.0 | 29.9 | 9.8 | 16.5 | 23.6 | -1.5 | +2.2 | 5 | 10 |
| SARIMAX | Classical | 17.7 | 29.8 | 38.6 | 45.4 | 31.6 | 10.2 | 17.9 | 28.8 | -3.5 | -0.6 | 5 | 5 |
| Chronos-Bolt tiny (zero-shot) | Foundation | 18.7 | 30.8 | 38.3 | 42.0 | 32.9 | 10.8 | 18.5 | 27.1 | +0.8 | +3.5 | 5 | 5 |
| Chronos-Bolt mini (zero-shot) | Foundation | 18.7 | 30.9 | 38.3 | 42.4 | 32.9 | 10.7 | 18.5 | 27.2 | +1.0 | +3.6 | 5 | 10 |
| Kalman filter | Classical | 18.5 | 31.3 | 39.6 | 45.0 | 33.2 | 10.6 | 18.7 | 28.6 | +0.2 | +2.7 | 0 | 5 |
| ARIMA | Classical | 18.6 | 31.4 | 39.7 | 45.1 | 33.3 | 10.7 | 18.8 | 28.7 | +0.2 | +2.7 | 0 | 5 |
| Persistence | Baseline | 20.0 | 32.1 | 39.7 | 44.7 | 34.2 | 11.7 | 19.4 | 28.5 | +0.2 | +3.3 | 0 | 0 |
| Linear extrapolation | Baseline | 24.3 | 48.9 | 71.0 | 89.2 | 50.1 | 13.3 | 27.9 | 56.4 | +0.1 | -0.3 | 10 | 15 |

#### Clinical — DTS / Clarke / CG-EGA (% of points)

| Model | DTS A 30 | DTS A 60 | DTS A 120 | DTS A+B 120 | Clarke A 60 | Clarke A+B 120 | CG-EGA hypo AP 30 | hypo EP 60 | eu EP 60 | hyper EP 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 91.5 | 78.3 | 62.8 | 96.9 | 78.6 | 96.6 | 77.7 | 51.9 | 0.0 | 0.1 |
| T1DMAI D32 MSE · sim + LoRA | 91.0 | 78.6 | 63.5 | 96.8 | 78.9 | 97.1 | 76.5 | 44.5 | 0.0 | 0.7 |
| T1DMAI D16 DILATE · sim + LoRA | 91.0 | 77.9 | 59.8 | 96.8 | 78.1 | 97.0 | 82.9 | 39.7 | 0.0 | 0.1 |
| SVR | 91.3 | 77.5 | 61.3 | 97.3 | 77.8 | 97.1 | 83.5 | 47.4 | 0.0 | 0.8 |
| T1DMAI D16 MSE · sim + LoRA | 90.8 | 76.6 | 61.6 | 96.8 | 76.8 | 97.0 | 85.2 | 36.5 | 0.0 | 0.3 |
| PatchTST | 90.9 | 75.0 | 57.6 | 96.5 | 75.3 | 96.9 | 82.3 | 48.1 | 0.0 | 0.4 |
| Sparse GP | 89.9 | 75.7 | 59.2 | 97.1 | 76.1 | 96.8 | 71.9 | 58.7 | 0.0 | 0.8 |
| GluNet | 90.7 | 75.8 | 56.3 | 96.0 | 76.1 | 96.8 | 86.8 | 42.3 | 0.0 | 0.1 |
| ARX (ridge) | 90.0 | 73.7 | 59.5 | 96.7 | 74.0 | 96.7 | 69.7 | 72.3 | 0.0 | 0.1 |
| LSTM | 90.8 | 75.6 | 55.3 | 95.5 | 75.9 | 97.1 | 89.4 | 31.6 | 0.0 | 0.1 |
| N-HiTS | 89.8 | 72.4 | 54.5 | 95.4 | 72.8 | 97.1 | 84.2 | 33.9 | 0.0 | 0.1 |
| T1DMAI D32 MSE · sim only | 89.2 | 71.8 | 55.0 | 94.8 | 72.1 | 96.1 | 72.6 | 69.7 | 0.0 | 0.9 |
| T1DMAI D16 DILATE · sim only | 89.0 | 72.1 | 50.8 | 93.0 | 72.4 | 95.8 | 73.5 | 62.6 | 0.1 | 0.4 |
| T1DMAI D16 MSE · sim only | 88.2 | 69.7 | 54.0 | 93.9 | 70.1 | 96.2 | 69.4 | 71.9 | 0.1 | 0.5 |
| AR (ridge) | 87.1 | 67.7 | 55.7 | 95.9 | 67.9 | 96.5 | 53.2 | 97.7 | 0.0 | 0.3 |
| Chronos-Bolt tiny (fine-tuned) | 88.3 | 69.8 | 54.1 | 95.8 | 70.0 | 96.4 | 71.6 | 76.5 | 0.0 | 0.3 |
| Chronos-Bolt mini (fine-tuned) | 88.2 | 69.5 | 54.0 | 95.6 | 69.7 | 96.4 | 73.9 | 80.6 | 0.0 | 0.4 |
| SARIMAX | 87.6 | 64.7 | 43.5 | 89.8 | 65.2 | 97.4 | 90.6 | 27.4 | 0.0 | 0.3 |
| Chronos-Bolt tiny (zero-shot) | 86.0 | 64.4 | 49.5 | 92.4 | 64.7 | 95.8 | 79.4 | 53.9 | 0.0 | 0.3 |
| Chronos-Bolt mini (zero-shot) | 86.2 | 64.6 | 49.9 | 92.3 | 64.9 | 95.6 | 82.6 | 54.5 | 0.0 | 0.3 |
| Kalman filter | 86.1 | 64.0 | 47.1 | 90.8 | 64.3 | 96.1 | 85.2 | 46.8 | 0.1 | 0.3 |
| ARIMA | 86.0 | 64.0 | 46.9 | 90.8 | 64.3 | 96.0 | 85.2 | 46.8 | 0.1 | 0.3 |
| Persistence | 82.9 | 62.5 | 47.5 | 91.0 | 63.1 | 96.1 | 79.7 | 51.3 | 0.0 | 0.3 |
| Linear extrapolation | 78.7 | 50.5 | 25.9 | 65.9 | 51.3 | 83.2 | 91.9 | 23.2 | 2.0 | 2.6 |

#### Hypoglycaemia (< 70 mg/dL at the horizon)

| Model | Sens 30 | Sens 60 | Spec 60 | PPV 60 | MCC 30 | MCC 60 | RMSE<70 30 | ME<70 30 | ME<70 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.49 | 0.29 | 0.997 | 0.73 | 0.62 | 0.45 | 13.1 | +9.1 | +17.6 |
| T1DMAI D32 MSE · sim + LoRA | 0.49 | 0.30 | 0.997 | 0.74 | 0.62 | 0.46 | 12.2 | +8.5 | +15.1 |
| T1DMAI D16 DILATE · sim + LoRA | 0.59 | 0.37 | 0.995 | 0.71 | 0.69 | 0.51 | 10.8 | +6.6 | +13.8 |
| SVR | 0.62 | 0.34 | 0.992 | 0.58 | 0.66 | 0.43 | 10.7 | +6.2 | +14.4 |
| T1DMAI D16 MSE · sim + LoRA | 0.62 | 0.45 | 0.992 | 0.63 | 0.66 | 0.52 | 10.2 | +5.3 | +11.8 |
| PatchTST | 0.57 | 0.39 | 0.989 | 0.52 | 0.63 | 0.43 | 11.7 | +4.9 | +11.7 |
| Sparse GP | 0.51 | 0.23 | 0.995 | 0.57 | 0.60 | 0.35 | 13.7 | +9.5 | +19.0 |
| GluNet | 0.66 | 0.49 | 0.987 | 0.55 | 0.66 | 0.50 | 11.1 | +3.3 | +8.5 |
| ARX (ridge) | 0.38 | 0.18 | 0.995 | 0.54 | 0.49 | 0.30 | 13.4 | +10.1 | +21.5 |
| LSTM | 0.70 | 0.53 | 0.983 | 0.50 | 0.65 | 0.50 | 10.3 | +1.5 | +5.2 |
| N-HiTS | 0.65 | 0.53 | 0.984 | 0.50 | 0.66 | 0.50 | 12.4 | +3.0 | +7.8 |
| T1DMAI D32 MSE · sim only | 0.45 | 0.17 | 0.998 | 0.74 | 0.60 | 0.35 | 13.6 | +10.3 | +21.4 |
| T1DMAI D16 DILATE · sim only | 0.51 | 0.21 | 0.997 | 0.69 | 0.64 | 0.37 | 13.2 | +9.2 | +20.2 |
| T1DMAI D16 MSE · sim only | 0.42 | 0.14 | 0.997 | 0.62 | 0.58 | 0.28 | 14.1 | +10.5 | +22.3 |
| AR (ridge) | 0.22 | 0.00 | 0.999 | 0.00 | 0.33 | -0.00 | 16.1 | +14.3 | +31.4 |
| Chronos-Bolt tiny (fine-tuned) | 0.43 | 0.07 | 0.993 | 0.24 | 0.48 | 0.11 | 13.4 | +10.1 | +23.4 |
| Chronos-Bolt mini (fine-tuned) | 0.46 | 0.06 | 0.994 | 0.23 | 0.51 | 0.11 | 12.9 | +9.5 | +24.4 |
| SARIMAX | 0.71 | 0.55 | 0.964 | 0.32 | 0.60 | 0.40 | 9.2 | +2.3 | +7.6 |
| Chronos-Bolt tiny (zero-shot) | 0.55 | 0.34 | 0.980 | 0.34 | 0.51 | 0.32 | 12.3 | +6.4 | +17.8 |
| Chronos-Bolt mini (zero-shot) | 0.60 | 0.29 | 0.982 | 0.34 | 0.54 | 0.30 | 11.9 | +5.8 | +18.5 |
| Kalman filter | 0.63 | 0.40 | 0.974 | 0.33 | 0.56 | 0.34 | 10.7 | +5.6 | +15.0 |
| ARIMA | 0.63 | 0.40 | 0.974 | 0.33 | 0.56 | 0.34 | 10.7 | +5.7 | +14.9 |
| Persistence | 0.55 | 0.36 | 0.981 | 0.36 | 0.54 | 0.34 | 12.8 | +8.1 | +18.1 |
| Linear extrapolation | 0.79 | 0.72 | 0.872 | 0.15 | 0.47 | 0.29 | 11.1 | -1.8 | -3.5 |

n = 9661 forecasts at 60 min; ◆ = best or not significantly worse (Wilcoxon on |error|, Bonferroni α = 0.05/23).

## MetaboNet track — blind

MetaboNet test split · 12 sub-datasets · future carbs and insulin withheld.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-rmse-dark.png">
  <img alt="RMSE per model, MetaboNet" src="screenshots/metabonet-rmse.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-significance-dark.png">
  <img alt="Wilcoxon against the lowest-RMSE model, MetaboNet" src="screenshots/metabonet-significance.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="screenshots/metabonet-horizon-hypo-bias-dark.png">
  <img alt="Mean error when CGM is below 70 mg/dL, MetaboNet" src="screenshots/metabonet-horizon-hypo-bias.png">
</picture>

#### Accuracy — RMSE / MARD / bias (mg/dL, %)

| Model | Family | RMSE 30 | RMSE 60 | RMSE 90 | RMSE 120 | gRMSE 60 | MARD 30 | MARD 60 | MARD 120 | ME 60 | MRE 60 % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1DMAI D32 MSE · from scratch, seed 43 | T1DMAI | 19.3 ◆ | 32.0 ◆ | 39.5 ◆ | 43.8 ◆ | 38.8 | 10.1 | 16.9 | 23.3 | -3.0 | +2.8 |
| T1DMAI D32 MSE · from scratch | T1DMAI | 19.3 ◆ | 32.0 | 39.6 | 43.9 | 38.9 | 10.0 | 16.9 | 23.3 | -3.2 | +2.7 |
| T1DMAI D16 MSE · from scratch, seed 43 | T1DMAI | 19.7 | 32.4 | 39.9 | 44.2 | 39.3 | 10.3 | 17.3 | 23.8 | -2.6 | +3.3 |
| T1DMAI D16 MSE · from scratch | T1DMAI | 19.6 | 32.5 | 40.2 | 44.5 | 39.6 | 10.2 | 17.2 | 23.6 | -3.7 | +2.6 |
| T1DMAI D16 DILATE · from scratch | T1DMAI | 20.1 | 32.8 | 40.2 | 44.8 | 39.2 | 10.3 | 17.2 | 24.3 | -2.3 | +2.9 |
| XGBoost | Classical | 19.8 | 33.2 | 41.3 | 46.0 | 40.3 | 10.6 | 18.6 | 27.2 | +0.5 | +6.3 |
| SVR | Classical | 20.4 | 33.9 | 41.9 | 46.6 | 40.9 | 10.5 | 18.1 | 25.5 | -2.5 | +3.5 |
| T1DMAI D32 MSE · sim + LoRA | T1DMAI | 21.3 | 34.3 | 40.2 | 46.1 | 40.3 | 10.9 | 17.7 | 24.4 | -2.8 | +2.0 |
| Sparse GP | Classical | 20.8 | 34.3 | 42.1 | 46.6 | 41.6 | 11.1 | 19.1 | 27.5 | +0.3 | +6.3 |
| N-HiTS | Deep | 20.6 | 34.4 | 42.7 | 47.8 | 40.8 | 11.0 | 19.0 | 27.4 | +0.5 | +5.4 |
| PatchTST | Deep | 20.3 | 34.4 | 42.9 | 48.0 | 40.7 | 10.7 | 19.0 | 27.7 | +0.7 | +5.5 |
| T1DMAI D16 DILATE · sim + LoRA | T1DMAI | 21.2 | 34.5 | 40.9 | 48.7 | 40.9 | 10.9 | 18.0 | 25.0 | -2.8 | +2.4 |
| Chronos-Bolt mini (fine-tuned) | Foundation | 21.1 | 34.5 | 42.0 | 46.2 | 40.9 | 10.8 | 18.1 | 25.0 | -2.6 | +2.6 |
| Chronos-Bolt tiny (fine-tuned) | Foundation | 21.2 | 34.6 | 42.2 | 46.3 | 41.0 | 10.9 | 18.2 | 25.0 | -2.4 | +2.8 |
| GluNet | Deep | 20.6 | 34.8 | 43.6 | 49.0 | 41.1 | 10.8 | 19.0 | 27.7 | +0.2 | +4.9 |
| T1DMAI D16 MSE · sim + LoRA | T1DMAI | 21.6 | 35.1 | 40.9 | 46.6 | 41.2 | 11.0 | 18.1 | 24.9 | -4.0 | +1.0 |
| ARX (ridge) | Classical | 21.3 | 35.4 | 43.5 | 47.9 | 42.3 | 11.7 | 20.5 | 29.5 | +2.6 | +8.1 |
| AR (ridge) | Classical | 21.4 | 35.6 | 43.6 | 48.1 | 42.4 | 11.7 | 20.7 | 29.9 | +3.3 | +8.5 |
| LSTM | Deep | 21.1 | 35.9 | 45.1 | 51.0 | 41.9 | 11.1 | 19.5 | 28.5 | +0.6 | +4.8 |
| T1DMAI D32 MSE · sim only | T1DMAI | 22.9 | 36.0 | 43.2 | 47.3 | 43.6 | 12.1 | 20.0 | 27.0 | -0.7 | +5.8 |
| T1DMAI D16 DILATE · sim only | T1DMAI | 23.4 | 36.6 | 44.0 | 48.0 | 44.0 | 12.2 | 20.4 | 28.8 | +0.0 | +6.0 |
| T1DMAI D16 MSE · sim only | T1DMAI | 23.4 | 36.7 | 44.1 | 48.2 | 43.8 | 13.1 | 21.7 | 29.7 | +4.4 | +9.7 |
| Chronos-Bolt tiny (zero-shot) | Foundation | 22.8 | 37.7 | 46.4 | 51.3 | 43.3 | 11.9 | 20.1 | 28.3 | +0.7 | +4.4 |
| Chronos-Bolt mini (zero-shot) | Foundation | 22.9 | 37.7 | 46.3 | 51.4 | 43.4 | 11.8 | 19.9 | 28.2 | +0.1 | +3.8 |
| Kalman filter | Classical | 22.2 | 38.4 | 48.9 | 55.6 | 43.9 | 11.4 | 20.3 | 30.3 | +0.2 | +3.5 |
| SARIMAX | Classical | 22.3 | 39.2 | 51.0 | 59.4 | 44.3 | 11.6 | 21.2 | 33.7 | +2.2 | +4.8 |
| Persistence | Baseline | 25.2 | 40.0 | 49.4 | 55.5 | 46.2 | 13.1 | 21.3 | 30.3 | +0.3 | +4.4 |
| ARIMA | Classical | 23.4 | 41.9 | 55.1 | 64.6 | 47.1 | 12.1 | 22.2 | 35.5 | +0.1 | +2.6 |
| Linear extrapolation | Baseline | 30.7 | 59.2 | 81.8 | 98.8 | 64.0 | 15.6 | 31.4 | 56.8 | +0.1 | +0.7 |

#### Clinical — DTS / Clarke / CG-EGA (% of points)

| Model | DTS A 30 | DTS A 60 | DTS A 120 | DTS A+B 120 | Clarke A 60 | Clarke A+B 120 | CG-EGA hypo AP 30 | hypo EP 60 | eu EP 60 | hyper EP 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1DMAI D32 MSE · from scratch, seed 43 | 87.2 | 70.3 | 55.3 | 95.1 | 70.5 | 93.1 | 54.2 | 86.8 | 0.0 | 3.2 |
| T1DMAI D32 MSE · from scratch | 87.3 | 70.2 | 55.3 | 95.1 | 70.4 | 93.2 | 54.0 | 88.3 | 0.0 | 3.3 |
| T1DMAI D16 MSE · from scratch, seed 43 | 86.7 | 69.5 | 54.6 | 94.9 | 69.6 | 93.0 | 50.3 | 90.6 | 0.0 | 3.3 |
| T1DMAI D16 MSE · from scratch | 86.9 | 69.5 | 54.6 | 94.9 | 69.7 | 93.0 | 50.5 | 90.4 | 0.0 | 3.6 |
| T1DMAI D16 DILATE · from scratch | 86.6 | 69.7 | 53.7 | 94.5 | 69.8 | 92.9 | 62.5 | 87.9 | 0.1 | 3.0 |
| XGBoost | 86.0 | 66.9 | 49.6 | 93.0 | 67.0 | 93.1 | 33.5 | 99.4 | 0.1 | 3.1 |
| SVR | 86.0 | 67.6 | 51.6 | 93.6 | 67.8 | 92.9 | 61.7 | 91.3 | 0.1 | 3.4 |
| T1DMAI D32 MSE · sim + LoRA | 85.0 | 68.5 | 52.6 | 94.5 | 68.7 | 92.3 | 73.0 | 73.7 | 0.1 | 2.7 |
| Sparse GP | 84.8 | 66.0 | 49.3 | 92.7 | 66.2 | 93.1 | 45.8 | 97.1 | 0.1 | 3.0 |
| N-HiTS | 85.0 | 66.1 | 49.1 | 92.4 | 66.3 | 93.3 | 57.5 | 80.0 | 0.1 | 2.9 |
| PatchTST | 85.7 | 66.0 | 48.8 | 92.3 | 66.2 | 93.3 | 63.9 | 79.7 | 0.1 | 2.7 |
| T1DMAI D16 DILATE · sim + LoRA | 85.1 | 67.9 | 51.0 | 93.6 | 68.0 | 92.0 | 69.2 | 79.1 | 0.1 | 3.0 |
| Chronos-Bolt mini (fine-tuned) | 85.4 | 67.7 | 53.1 | 93.7 | 67.9 | 93.3 | 65.6 | 77.4 | 0.1 | 3.1 |
| Chronos-Bolt tiny (fine-tuned) | 85.2 | 67.5 | 53.0 | 93.7 | 67.7 | 93.3 | 62.2 | 77.8 | 0.1 | 3.1 |
| GluNet | 85.4 | 66.0 | 48.7 | 91.9 | 66.2 | 93.1 | 64.2 | 75.9 | 0.1 | 2.8 |
| T1DMAI D16 MSE · sim + LoRA | 84.7 | 67.6 | 51.6 | 94.2 | 67.8 | 92.3 | 70.9 | 69.5 | 0.1 | 2.9 |
| ARX (ridge) | 83.8 | 62.0 | 45.3 | 91.9 | 62.2 | 93.2 | 51.1 | 93.4 | 0.1 | 2.8 |
| AR (ridge) | 83.7 | 61.8 | 45.1 | 91.7 | 61.9 | 93.2 | 52.1 | 93.5 | 0.1 | 2.8 |
| LSTM | 84.9 | 64.9 | 47.7 | 91.1 | 65.1 | 92.8 | 67.5 | 72.4 | 0.1 | 2.8 |
| T1DMAI D32 MSE · sim only | 82.5 | 63.1 | 48.7 | 93.0 | 63.3 | 93.0 | 45.7 | 93.0 | 0.1 | 3.7 |
| T1DMAI D16 DILATE · sim only | 82.3 | 62.4 | 45.3 | 92.4 | 62.6 | 92.7 | 56.0 | 86.3 | 0.1 | 3.3 |
| T1DMAI D16 MSE · sim only | 80.1 | 59.2 | 45.0 | 91.8 | 59.3 | 93.3 | 34.6 | 93.9 | 0.1 | 3.1 |
| Chronos-Bolt tiny (zero-shot) | 82.8 | 63.9 | 49.2 | 90.9 | 64.1 | 92.8 | 66.9 | 67.9 | 0.3 | 3.0 |
| Chronos-Bolt mini (zero-shot) | 83.1 | 64.2 | 49.4 | 91.0 | 64.4 | 92.7 | 70.7 | 66.9 | 0.3 | 3.1 |
| Kalman filter | 83.9 | 63.3 | 46.3 | 88.5 | 63.6 | 92.2 | 74.7 | 61.0 | 0.3 | 2.9 |
| SARIMAX | 83.6 | 61.5 | 42.0 | 86.5 | 61.8 | 91.1 | 74.2 | 61.2 | 0.4 | 2.7 |
| Persistence | 79.6 | 61.0 | 46.1 | 88.7 | 61.5 | 92.1 | 57.3 | 68.3 | 0.2 | 3.4 |
| ARIMA | 82.4 | 59.3 | 40.1 | 83.6 | 59.7 | 89.8 | 78.0 | 51.9 | 0.5 | 3.3 |
| Linear extrapolation | 74.9 | 48.6 | 26.8 | 67.0 | 49.4 | 78.0 | 83.5 | 34.3 | 3.6 | 7.8 |

#### Hypoglycaemia (< 70 mg/dL at the horizon)

| Model | Sens 30 | Sens 60 | Spec 60 | PPV 60 | MCC 30 | MCC 60 | RMSE<70 30 | ME<70 30 | ME<70 60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1DMAI D32 MSE · from scratch, seed 43 | 0.25 | 0.04 | 0.999 | 0.61 | 0.39 | 0.14 | 23.2 | +18.0 | +35.7 |
| T1DMAI D32 MSE · from scratch | 0.24 | 0.03 | 0.999 | 0.60 | 0.38 | 0.12 | 23.2 | +18.1 | +35.9 |
| T1DMAI D16 MSE · from scratch, seed 43 | 0.21 | 0.02 | 1.000 | 0.56 | 0.36 | 0.09 | 24.0 | +19.0 | +37.6 |
| T1DMAI D16 MSE · from scratch | 0.22 | 0.02 | 1.000 | 0.57 | 0.36 | 0.10 | 23.8 | +18.7 | +37.2 |
| T1DMAI D16 DILATE · from scratch | 0.30 | 0.02 | 0.999 | 0.52 | 0.40 | 0.11 | 21.7 | +16.0 | +35.7 |
| XGBoost | 0.06 | 0.00 | 1.000 | 0.74 | 0.20 | 0.03 | 26.5 | +22.3 | +45.6 |
| SVR | 0.34 | 0.02 | 0.999 | 0.33 | 0.38 | 0.08 | 22.6 | +15.8 | +38.4 |
| T1DMAI D32 MSE · sim + LoRA | 0.45 | 0.09 | 0.996 | 0.41 | 0.44 | 0.18 | 20.5 | +12.9 | +31.0 |
| Sparse GP | 0.22 | 0.01 | 0.999 | 0.27 | 0.31 | 0.04 | 25.9 | +19.9 | +44.5 |
| N-HiTS | 0.34 | 0.10 | 0.992 | 0.29 | 0.39 | 0.16 | 24.3 | +16.3 | +37.0 |
| PatchTST | 0.40 | 0.10 | 0.992 | 0.28 | 0.42 | 0.15 | 22.8 | +14.4 | +36.0 |
| T1DMAI D16 DILATE · sim + LoRA | 0.40 | 0.06 | 0.997 | 0.38 | 0.41 | 0.15 | 21.2 | +14.1 | +33.2 |
| Chronos-Bolt mini (fine-tuned) | 0.40 | 0.10 | 0.993 | 0.32 | 0.41 | 0.16 | 21.9 | +14.1 | +33.5 |
| Chronos-Bolt tiny (fine-tuned) | 0.37 | 0.10 | 0.993 | 0.30 | 0.40 | 0.16 | 22.5 | +15.0 | +33.7 |
| GluNet | 0.41 | 0.13 | 0.988 | 0.26 | 0.42 | 0.17 | 22.9 | +14.1 | +34.8 |
| T1DMAI D16 MSE · sim + LoRA | 0.42 | 0.12 | 0.992 | 0.35 | 0.42 | 0.19 | 20.9 | +13.4 | +30.4 |
| ARX (ridge) | 0.27 | 0.03 | 0.996 | 0.20 | 0.33 | 0.07 | 24.5 | +17.7 | +42.7 |
| AR (ridge) | 0.28 | 0.03 | 0.995 | 0.19 | 0.33 | 0.07 | 24.0 | +17.3 | +42.9 |
| LSTM | 0.44 | 0.16 | 0.982 | 0.23 | 0.42 | 0.17 | 22.7 | +13.0 | +33.2 |
| T1DMAI D32 MSE · sim only | 0.23 | 0.02 | 0.998 | 0.23 | 0.30 | 0.06 | 26.0 | +19.6 | +41.4 |
| T1DMAI D16 DILATE · sim only | 0.32 | 0.07 | 0.991 | 0.20 | 0.34 | 0.10 | 24.9 | +16.9 | +39.3 |
| T1DMAI D16 MSE · sim only | 0.18 | 0.03 | 0.997 | 0.22 | 0.27 | 0.07 | 28.6 | +22.5 | +45.1 |
| Chronos-Bolt tiny (zero-shot) | 0.46 | 0.20 | 0.976 | 0.21 | 0.41 | 0.18 | 23.2 | +12.7 | +31.9 |
| Chronos-Bolt mini (zero-shot) | 0.49 | 0.20 | 0.976 | 0.21 | 0.41 | 0.18 | 22.4 | +11.5 | +31.0 |
| Kalman filter | 0.52 | 0.26 | 0.963 | 0.19 | 0.43 | 0.19 | 20.7 | +9.8 | +28.3 |
| SARIMAX | 0.53 | 0.27 | 0.961 | 0.18 | 0.43 | 0.19 | 21.2 | +9.8 | +28.8 |
| Persistence | 0.37 | 0.19 | 0.975 | 0.19 | 0.35 | 0.16 | 26.1 | +16.3 | +33.6 |
| ARIMA | 0.59 | 0.37 | 0.938 | 0.16 | 0.43 | 0.21 | 20.7 | +7.3 | +23.8 |
| Linear extrapolation | 0.72 | 0.58 | 0.861 | 0.12 | 0.41 | 0.21 | 27.2 | -1.4 | +12.3 |

n = 264893 forecasts at 60 min; ◆ = best or not significantly worse (Wilcoxon on |error|, Bonferroni α = 0.05/28).

## T1DMAI

| | |
| --- | --- |
| Architecture | encoder-only transformer, 30-min patches, one head, RoPE, SwiGLU |
| Variants | D16 (16 layers × width 16, 31 k parameters) · D32 (32 × 32, 238 k) |
| Loss | 7-quantile pinball + median term: DILATE (shape and time) or MSE |
| Inputs | CGM (risk space), carbohydrate appearance curve, insulin action curve, mask bit |
| Output | 7 quantiles over masked spans; the trailing span is the 2-hour forecast |
| Context | 24–168 h |
| Simulator pretraining | 1 M T1DMSIM patients × 7 days, 10 000 steps × batch 512 |
| LoRA | rank 4 on every linear layer, base frozen, merged into the weights after training |
| Runtime | ExecuTorch / LiteRT, on-device in T1DMDROID |

Details: [docs/USAGE.md](docs/USAGE.md), [ARCHITECTURE.md](ARCHITECTURE.md).

## Protocol

| | Owner track | MetaboNet track |
| --- | --- | --- |
| Data | 1 adult with T1D, T1DMDROID backup, 115 days, 5-min CGM | MetaboNet `test.parquet`, 12 sub-datasets |
| Train | days 1–80 (all sensors worn) | `train.parquet` (finetune cache v5) |
| Test | days 81–115, every measured 5-min origin | live-leaderboard template, every 10th row |
| Future carbs / insulin | given, as logged | withheld (blind) |
| Horizons | 30, 60, 90, 120 min | 30, 60, 90, 120 min |
| Zoo training windows | 60 000 | 351 000 (200 000 for fine-tuned Chronos) |
| T1DMAI paths | sim only · sim + LoRA (rank 4, 2 000 steps) | sim only · sim + LoRA (rank 4) · from scratch (15 000 steps) |
| Checkpoint choice | last step, no test-set selection | last step, no test-set selection |

## Metrics

| Metric | Definition |
| --- | --- |
| RMSE, MAE | mg/dL, median forecast vs CGM at the horizon |
| MARD | mean \|error\| / CGM, % |
| ME, MRE | mean signed error (mg/dL) and relative error (%); > 0 = over-forecast |
| gRMSE | RMSE weighted by the Del Favero 2012 gMSE penalty (over-forecast in hypo, under-forecast in hyper) |
| DTS A, A+B | Diabetes Technology Society error grid zones, % of points |
| Clarke A, A+B | Clarke error grid zones, % of points |
| CG-EGA AP / EP | continuous glucose EGA, accurate / erroneous share per region; rate over the horizon |
| Sens, Spec, PPV, MCC | forecast < 70 vs CGM < 70 at the horizon |
| RMSE<70, ME<70 | error over points with CGM < 70 |
| TG | time gain: horizon − lag of peak forecast/CGM correlation, min |
| ◆ | lowest RMSE, or not worse by two-sided Wilcoxon signed-rank on paired \|error\| (Bonferroni α = 0.05 / (models − 1)) |

## Reproduce

```bash
python -m zoo.run MODEL --cache CACHE --template template.parquet --truth truth.parquet [--causal]
python finetune.py --checkpoint CKPT --cache CACHE --lora-rank 4 [--blind | --no-blind]
python export_submission.py --checkpoint CKPT --template template.parquet --cache CACHE \
  --max-context 336 --fill ladder [--future-doses zero]
python leaderboard.py MANIFEST.json --out results
```

Per-model metrics: [results/personal.json](results/personal.json),
[results/metabonet.json](results/metabonet.json).

## All figures

- [metabonet-dts](screenshots/metabonet-dts.png)
- [metabonet-heat-dts](screenshots/metabonet-heat-dts.png)
- [metabonet-heat-mcc](screenshots/metabonet-heat-mcc.png)
- [metabonet-horizon-hypo-bias](screenshots/metabonet-horizon-hypo-bias.png)
- [metabonet-horizon-hypo-sens](screenshots/metabonet-horizon-hypo-sens.png)
- [metabonet-horizon-rmse](screenshots/metabonet-horizon-rmse.png)
- [metabonet-paths](screenshots/metabonet-paths.png)
- [metabonet-rmse](screenshots/metabonet-rmse.png)
- [metabonet-significance](screenshots/metabonet-significance.png)
- [personal-dts](screenshots/personal-dts.png)
- [personal-heat-dts](screenshots/personal-heat-dts.png)
- [personal-heat-mcc](screenshots/personal-heat-mcc.png)
- [personal-horizon-hypo-bias](screenshots/personal-horizon-hypo-bias.png)
- [personal-horizon-hypo-sens](screenshots/personal-horizon-hypo-sens.png)
- [personal-horizon-rmse](screenshots/personal-horizon-rmse.png)
- [personal-paths](screenshots/personal-paths.png)
- [personal-rmse](screenshots/personal-rmse.png)
- [personal-significance](screenshots/personal-significance.png)

## License

Copyright 2026 0xdeadf1sh. [MIT](LICENSE). Related: [T1DMSIM](https://github.com/0xdeadf1sh/T1DMSIM) ·
[T1DMDROID](https://github.com/0xdeadf1sh/T1DMDROID) ·
[T1DMCOMMON](https://github.com/0xdeadf1sh/T1DMCOMMON).
