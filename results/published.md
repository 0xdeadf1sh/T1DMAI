| Model | Paper | Paper data | Paper inputs | Paper 30 | Paper 60 | My data 30 | My data 60 | MetaboNet 30 | MetaboNet 60 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PatchTST | [9](https://arxiv.org/abs/2505.08821) | DCLP3 train, OhioT1DM test (12) | CGM, carbs, insulin | 16.7 | 24.6 | 15.1 | 23.3 | 20.3 | 34.4 |
| LSTM | [5](https://doi.org/10.1007/s41666-019-00059-y) | OhioT1DM 2018 (6) | CGM | 18.9 | 31.4 | 15.4 | 24.3 | 21.1 | 35.9 |
| TimesFM 2.5 (zero-shot) | [4](https://arxiv.org/abs/2606.06881) | 12 datasets, mixed | CGM | 18.8 | 31.8 | 18.4 | 29.1 | 20.7 | 35.0 |
| GluNet | [6](https://doi.org/10.1109/JBHI.2019.2931842) | OhioT1DM 2018 (6) | CGM, carbs, insulin | 19.3 | 31.8 | 15.4 | 23.6 | 20.6 | 34.8 |
| Chronos-2 (zero-shot) | [4](https://arxiv.org/abs/2606.06881) | 12 datasets, mixed | CGM | 19.2 | 32.0 | 17.8 | 29.4 | 21.0 | 35.3 |
| LSTM (15 h) | [3](https://arxiv.org/abs/2606.18640) | MetaboNet | CGM, carbs, insulin | 20.1 | 33.4 | 15.2 | 23.2 | 20.3 | 33.6 |
| SVR | [6](https://doi.org/10.1109/JBHI.2019.2931842) | OhioT1DM 2018 (6) | CGM, carbs, insulin | 21.8 | 34.3 | 15.1 | 23.0 | 20.4 | 33.9 |
| LightGBM | [3](https://arxiv.org/abs/2606.18640) | MetaboNet | CGM, carbs, insulin, time | 20.8 | 34.5 | 14.7 | 22.4 | 19.8 | 33.2 |
| ARIMA | [8](https://ceur-ws.org/Vol-2675/) | OhioT1DM 2020 (6) | CGM, doses, meals, exercise | 23.7 | 36.4 | 18.6 | 31.4 | 23.4 | 41.9 |
| ARX | [6](https://doi.org/10.1109/JBHI.2019.2931842) | OhioT1DM 2018 (6) | CGM, carbs, insulin | 21.5 | 39.5 | 15.7 | 24.1 | 21.3 | 35.4 |
| Persistence | [3](https://arxiv.org/abs/2606.18640) | MetaboNet | CGM | 26.3 | 41.5 | 20.0 | 32.1 | 25.2 | 40.0 |
| XGBoost | [7](https://ceur-ws.org/Vol-2148/) | OhioT1DM 2018 (6) | CGM, doses, meals, activity | 20.4 | - | 14.7 | 22.2 | 19.8 | 33.2 |
| T1DMAI D16 DILATE (sim + LoRA) | - | - | - | - | - | 15.2 | 22.8 | 21.5 | 34.6 |
| T1DMAI D16 MSE (sim + LoRA) | - | - | - | - | - | 15.1 | 23.5 | 21.8 | 35.1 |
| T1DMAI D16 DILATE (sim only) | - | - | - | - | - | 16.2 | 26.1 | 23.4 | 36.6 |
| T1DMAI D16 MSE (sim only) | - | - | - | - | - | 16.8 | 27.1 | 23.4 | 36.7 |

RMSE in mg/dL at the single step 30 or 60 min ahead, as printed in each paper. T1DMAI has no published score.

Chronos-Bolt (Zhang et al. 2026) and N-HiTS (Potosnak et al. 2025; Sergazinov et al. 2024) are published only as RMSE averaged over every step up to the horizon, which is not the same quantity, so they have no row.
