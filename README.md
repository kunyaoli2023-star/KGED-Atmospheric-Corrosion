# KGED: Knowledge-Guided Equation Discovery for Atmospheric Steel Corrosion

This repository provides the verification test scripts for the discovered thin-film electrochemical rate equations and the cross-environment engineering corrosion-growth prediction module accompanying the manuscript:

> **Knowledge-guided multi-agent discovery of explicit kinetic equations for atmospheric corrosion of steel**

## Repository Structure

- `data/`
  - `data.json`: 168-condition thin-film electrochemical dataset on Q355B steel ($T \in [0, 50]\,^\circ\mathrm{C}$, $C \in [0, 26.5]\,\mathrm{wt\%\ NaCl}$, $H \in [50, 1000]\,\mu\mathrm{m}$).
  - `folds.json`: 23-fold grouped cross-validation split definitions (5 profile, 5 temperature, 7 film-thickness, and 6 salinity holdout folds).
  - `Frozen_models.json`: Full-precision parameters and analytical expressions for the retained KGED equations and symbolic-regression baselines (Table 4).
- `engineering/`
  - `engineering_integrator.py` & `engineering_state.py`: Reaction-clock-weighted cubic Bernstein state-evolution integrator and bounded linear least-squares source calibration solver.
  - `run_engineering_v6.py`: Cross-environment calibration (Linhai source site) and target extrapolation pipeline (Jinhai, humid chlorinated indoor swimming pool, and NASA KSC).
  - `verify_closed_forms.py`: Analytical verification of the frozen 5-parameter parsimonious and 8-parameter minimum-error KGED equations.
  - `inputs/`: Environmental forcing histories (`*.npz`), observation records (`*_labels.json`), and microenvironmental mapping parameters (`mapping.json`).
  - `PREDICTIONS_FROZEN.json`: Full-precision calibrated state amplitudes (Table 6) and predicted corrosion trajectories (Table 7 & Fig. 9).
- `tests/`
  - `test_laboratory_equations.py`: Reproduces full-dataset training log-RMSE (`0.507` and `0.496`) and analytical interior corrosion maxima ($C^* = 4.00\text{–}4.05\,\mathrm{wt\%}$, $H^* = 167.7\text{–}184.3\,\mu\mathrm{m}$, Table 5).
  - `test_engineering_prediction.py`: Verifies the source-calibrated Bernstein state amplitudes (Table 6) and target trajectory RMSE, normalised RMSE (`6.3%`), pointwise MAPE (`11.4%`), and endpoint errors across Jinhai, swimming pool, and NASA KSC (`1.16%` mean absolute endpoint error for the 5-parameter KGED law, Table 7).
  - `test_salinity_film_coupling_extension.py`: Post-discovery verification of the 7-parameter salinity-modulated film-length rate law ($\lambda(c) = \lambda_0(1+c)^\beta$, Section 3.4) across the 23-fold laboratory cross-validation benchmark (`CV = 0.461`) and the 4-site field integration pipeline (`mean nRMSE = 6.7%`, `mean endpoint APE = 2.94%`, `Jinhai RMSE = 13.07 um`).

## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the laboratory rate equation verification test:
```bash
python tests/test_laboratory_equations.py
```

Run the cross-environment engineering prediction test:
```bash
python tests/test_engineering_prediction.py
```

Run the post-discovery 7-parameter salinity-modulated film-length verification test:
```bash
python tests/test_salinity_film_coupling_extension.py
```

Run the full closed-form and Gauss-Legendre integration verification in `engineering/`:
```bash
python engineering/verify_closed_forms.py
```
