| Model | RMSE 30 | 95% CI | RMSE 60 | 95% CI | RMSE 120 | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T1DMAI D32 MSE (from scratch, seed 43) | 19.3 | 18.7 to 20.0 | 32.0 | 30.9 to 33.0 | 43.8 | 42.4 to 45.3 |
| T1DMAI D32 MSE (from scratch) | 19.3 | 18.7 to 20.0 | 32.0 | 31.0 to 33.1 | 43.9 | 42.5 to 45.3 |
| T1DMAI D16 MSE (from scratch, seed 43) | 19.7 | 19.0 to 20.3 | 32.4 | 31.3 to 33.5 | 44.2 | 42.8 to 45.7 |
| T1DMAI D16 MSE (from scratch) | 19.6 | 18.9 to 20.2 | 32.5 | 31.4 to 33.6 | 44.5 | 43.0 to 45.9 |
| T1DMAI D16 DILATE (from scratch) | 20.1 | 19.4 to 20.8 | 32.8 | 31.7 to 33.9 | 44.8 | 43.3 to 46.3 |
| XGBoost | 19.8 | 19.1 to 20.5 | 33.2 | 32.1 to 34.3 | 46.0 | 44.6 to 47.5 |
| SVR | 20.4 | 19.7 to 21.2 | 33.9 | 32.8 to 35.1 | 46.6 | 45.0 to 48.2 |
| T1DMAI D32 MSE (sim + LoRA) | 21.3 | 20.5 to 22.0 | 34.3 | 33.1 to 35.5 | 46.1 | 44.6 to 47.7 |
| Sparse GP | 20.8 | 20.1 to 21.6 | 34.3 | 33.2 to 35.4 | 46.6 | 45.2 to 48.1 |
| N-HiTS | 20.6 | 20.0 to 21.3 | 34.4 | 33.3 to 35.5 | 47.8 | 46.3 to 49.3 |
| PatchTST | 20.3 | 19.6 to 21.0 | 34.4 | 33.3 to 35.6 | 48.0 | 46.5 to 49.5 |
| T1DMAI D16 DILATE (sim + LoRA) | 21.2 | 20.5 to 22.0 | 34.5 | 33.3 to 35.6 | 48.7 | 46.8 to 50.5 |
| Chronos-Bolt mini (fine-tuned) | 21.1 | 20.3 to 21.8 | 34.5 | 33.3 to 35.7 | 46.2 | 44.7 to 47.8 |
| Chronos-Bolt tiny (fine-tuned) | 21.2 | 20.4 to 21.9 | 34.6 | 33.5 to 35.8 | 46.3 | 44.8 to 47.8 |
| GluNet | 20.6 | 19.9 to 21.3 | 34.8 | 33.7 to 36.0 | 49.0 | 47.4 to 50.6 |
| T1DMAI D16 MSE (sim + LoRA) | 21.6 | 20.8 to 22.3 | 35.1 | 33.9 to 36.3 | 46.6 | 45.0 to 48.2 |
| ARX (ridge) | 21.3 | 20.6 to 22.1 | 35.4 | 34.3 to 36.6 | 47.9 | 46.6 to 49.4 |
| AR (ridge) | 21.4 | 20.6 to 22.1 | 35.6 | 34.5 to 36.7 | 48.1 | 46.7 to 49.4 |
| LSTM | 21.1 | 20.4 to 21.9 | 35.9 | 34.7 to 37.1 | 51.0 | 49.3 to 52.7 |
| T1DMAI D32 MSE (sim only) | 22.9 | 22.1 to 23.7 | 36.0 | 34.8 to 37.1 | 47.3 | 45.8 to 48.9 |
| T1DMAI D16 DILATE (sim only) | 23.4 | 22.5 to 24.2 | 36.6 | 35.4 to 37.8 | 48.0 | 46.5 to 49.4 |
| T1DMAI D16 MSE (sim only) | 23.4 | 22.6 to 24.2 | 36.7 | 35.5 to 37.8 | 48.2 | 46.8 to 49.6 |
| Chronos-Bolt tiny (zero-shot) | 22.8 | 22.0 to 23.6 | 37.7 | 36.4 to 38.9 | 51.3 | 49.6 to 52.9 |
| Chronos-Bolt mini (zero-shot) | 22.9 | 22.1 to 23.7 | 37.7 | 36.4 to 38.9 | 51.4 | 49.7 to 53.1 |
| Kalman filter | 22.2 | 21.4 to 23.0 | 38.4 | 37.1 to 39.8 | 55.6 | 53.8 to 57.4 |
| SARIMAX | 22.3 | 21.5 to 23.1 | 39.2 | 37.8 to 40.5 | 59.4 | 57.6 to 61.3 |
| Persistence | 25.2 | 24.2 to 26.1 | 40.0 | 38.6 to 41.4 | 55.5 | 53.7 to 57.4 |
| ARIMA | 23.4 | 22.6 to 24.3 | 41.9 | 40.5 to 43.4 | 64.6 | 62.5 to 66.7 |
| Linear extrapolation | 30.7 | 29.6 to 31.8 | 59.2 | 57.1 to 61.1 | 98.8 | 96.2 to 101.4 |
