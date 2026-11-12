# Table 8. Cross-environment corrosion-depth prediction accuracy (Eq. 16)

| Method | Jinhai RMSE (μm) | Jinhai EP (%) | Swimming pool RMSE (μm) | Swimming pool EP (%) | NASA KSC RMSE (μm) | NASA KSC EP (%) | Mean \|EP\| (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Time-integrated laboratory baselines** | | | | | | | |
| BP † | 22.65 | +9.68 | 70.30 | +40.58 | 5.34 | -2.97 | 17.74 |
| RF † | 18.30 | +8.09 | 45.45 | -7.91 | 9.80 | -15.72 | 10.57 |
| XGBoost † | 20.91 | +10.48 | 54.79 | -13.39 | 6.70 | -9.86 | 11.25 |
| Sparse regression | 14.44 | +0.50 | 32.40 | -0.14 | 5.53 | -5.04 | 1.89 |
| gplearn | 22.90 | +9.33 | 43.44 | +25.91 | 7.74 | -4.17 | 13.13 |
| PySR | 22.71 | +11.48 | 32.97 | -0.44 | 5.93 | -5.90 | 5.94 |
| PhySO | 24.45 | +9.94 | 29.39 | +0.65 | 7.75 | -1.14 | 3.91 |
| **Empirical power-law and direct field machine-learning models** | | | | | | | |
| Empirical power-law model | 12.55 | +3.06 | 114.62 | -43.30 | 23.37 | -37.89 | 28.08 |
| Quadratic Ridge † | Diverged | Diverged | Diverged | Diverged | Diverged | Diverged | Diverged |
| Source RF † | 35.92 | -36.12 | 193.55 | -79.12 | 42.13 | -62.88 | 59.37 |
| Source XGBoost † | 13.68 | +3.32 | 132.71 | -43.16 | 38.68 | -62.56 | 36.35 |
| **Domain adaptation** | | | | | | | |
| TCA-Ridge † | 93.03 | -58.52 | 177.61 | -67.37 | Diverged | Diverged | Diverged |
| CORAL-Ridge † | Diverged | Diverged | Diverged | +18.08 | Diverged | Diverged | Diverged |
| Domain-standardised RF † | 25.37 | -19.62 | 162.90 | -66.15 | 55.91 | -20.70 | 35.49 |
| Domain-standardised XGBoost † | 13.68 | +3.32 | 132.71 | -43.16 | 16.24 | +21.47 | 22.65 |
| DANN-MLP † | Diverged | Diverged | 195.16 | -82.38 | Diverged | Diverged | Diverged |
| **KGED + integration** | | | | | | | |
| Five-parameter KGED | 14.05 | -0.07 | 32.38 | -0.48 | 4.98 | -2.94 | 1.16 |
| Eight-parameter KGED | 18.39 | +4.45 | 32.36 | +18.70 | 4.83 | +2.73 | 8.63 |

† Frozen predictions loaded from `engineering/Table_conditioned_predictions.csv` and re-scored under Eq. (16) (see `engineering/comparators/reproduce_field_comparators.py` for standalone training code).

*Diverged*: cell-wise RMSE > 200 μm or |endpoint error| > 100% (Table 8 Footnote a; raw values in `Diverged_Models_Raw_Metrics.csv`).
