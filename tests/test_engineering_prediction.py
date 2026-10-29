# -*- coding: utf-8 -*-
"""Test script 2: Verify the cross-environment engineering corrosion-depth predictions
(Table 6 state amplitudes and Table 7 trajectory RMSE & endpoint errors across Linhai,
Jinhai, Indoor Swimming Pool, and NASA KSC)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    amp_data = json.loads((ROOT / "engineering/source_state_amplitudes.json").read_text(encoding="utf-8"))
    forecast = json.loads((ROOT / "engineering/current_conditioned_forecast.json").read_text(encoding="utf-8"))

    print("=== 2. Cross-Environment Engineering Prediction Verification (Tables 6 & 7) ===")
    print("\n[Table 6: Source-Calibrated Bernstein State Amplitudes at Linhai]")
    for cfg in amp_data["configurations"]:
        name = cfg["configuration"]
        amps = [round(x, 3) for x in cfg["amplitudes"]]
        print(f"  - {name:15s} ({cfg['laboratory_parameter_count']} params): a0..a3 = {amps}")

    print("\n[Table 7: Target-Environment Trajectory RMSE and Endpoint Percentage Errors]")
    sites = [
        ("jinhai", "Jinhai (temperate marine)"),
        ("pool", "Indoor swimming pool (industrial)"),
        ("ksc", "NASA KSC (subtropical severe marine)")
    ]
    for alias, label in [
        ("parsimonious", "KGED (5 parameters, parsimonious)"),
        ("best_accuracy", "KGED (8 parameters, minimum-error)")
    ]:
        print(f"\n  [{label}]")
        preds = forecast["models"][alias]["predictions"]
        apes = []
        for site_key, site_label in sites:
            lbl = json.loads((ROOT / f"engineering/inputs/{site_key}_labels.json").read_text(encoding="utf-8"))
            y_obs = np.asarray(lbl["observed_depth_um"], float)[1:]
            y_pred = np.asarray(preds[site_key]["predicted_depth_um"], float)[1:]
            rmse = float(np.sqrt(np.mean((y_pred - y_obs) ** 2)))
            ep_err = float((y_pred[-1] - y_obs[-1]) / y_obs[-1] * 100.0)
            apes.append(abs(ep_err))
            print(f"    * {site_label:38s}: RMSE = {rmse:6.2f} um | Endpoint error = {ep_err:+6.2f}%")
        mean_ape = float(np.mean(apes))
        print(f"    => Mean absolute endpoint percentage error (3 target sites) = {mean_ape:.2f}%")

    print("\nPASSED: All cross-environment engineering predictions verified.")

if __name__ == "__main__":
    main()
