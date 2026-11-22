# Cross-environment engineering corrosion prediction benchmarks (Fig. 10a–b, Eq. 16)

| Method | Weihai near-coastal RMSE (μm) | Weihai near-coastal endpoint error (%) | Indoor natatorium RMSE (μm) | Indoor natatorium endpoint error (%) | NASA KSC RMSE (μm) | NASA KSC endpoint error (%) | Mean absolute endpoint error (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Time-integrated laboratory baselines** | | | | | | | |
| BP † | 22.65 | +9.68 | 70.30 | +40.58 | 5.34 | -2.97 | 17.74 |
| RF † | 18.30 | +8.09 | 45.45 | -7.91 | 9.80 | -15.72 | 10.57 |
| XGBoost † | 20.91 | +10.48 | 54.79 | -13.39 | 6.70 | -9.86 | 11.25 |
| Sparse regression | 14.44 | +0.50 | 32.40 | -0.14 | 5.53 | -5.06 | 1.90 |
| gplearn | 22.95 | +9.38 | 43.46 | +25.92 | 7.69 | -3.91 | 13.07 |
| PySR | 22.69 | +11.46 | 32.98 | -0.44 | 5.94 | -6.06 | 5.99 |
| PhySO | 24.45 | +9.94 | 29.39 | +0.65 | 7.75 | -1.14 | 3.91 |
| **Empirical power-law and direct field machine-learning models** | | | | | | | |
| Empirical power-law model | 12.55 | +3.06 | 114.62 | -43.30 | 23.37 | -37.89 | 28.08 |
| Quadratic Ridge † | Diverged | Diverged | Diverged | Diverged | Diverged | Diverged | Diverged |
| Source RF † | 35.92 | -36.12 | 193.55 | -79.12 | 42.13 | -62.88 | 59.37 |
| Source XGBoost † | 13.68 | +3.32 | 132.71 | -43.16 | 38.68 | -62.56 | 36.35 |
| **Unsupervised domain-adaptation models** | | | | | | | |
| TCA–Ridge † | 93.03 | -58.52 | 177.61 | -67.37 | Diverged | Diverged | Diverged |
| CORAL–Ridge † | Diverged | Diverged | Diverged | +18.08 | Diverged | Diverged | Diverged |
| Domain-standardised RF † | 25.37 | -19.62 | 162.90 | -66.15 | 55.91 | -20.70 | 35.49 |
| Domain-standardised XGBoost † | 13.68 | +3.32 | 132.71 | -43.16 | 16.24 | +21.47 | 22.65 |
| DANN–MLP † | Diverged | Diverged | 195.16 | -82.38 | Diverged | Diverged | Diverged |
| **KGED + integration** | | | | | | | |
| Five-parameter KGED | 14.03 | -0.10 | 32.38 | -0.49 | 4.96 | -3.28 | 1.29 |
| Eight-parameter KGED | 18.36 | +4.41 | 32.35 | +18.69 | 4.87 | +2.32 | 8.47 |

† Frozen predictions loaded from `engineering/Table_conditioned_predictions.csv` and re-scored under Eq. (16) (3 integrated laboratory ML baselines + 8 direct field ML / domain-adaptation models; see `engineering/comparators/reproduce_field_comparators.py` for the 8 direct field ML / adaptation models).

*Diverged*: cell-wise RMSE > 200 μm or |endpoint error| > 100% (raw values in `Diverged_Models_Raw_Metrics.csv`).
