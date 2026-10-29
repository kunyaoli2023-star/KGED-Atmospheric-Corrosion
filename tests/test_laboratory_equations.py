# -*- coding: utf-8 -*-
"""Test script 1: Independent closed-form verification of the 5-parameter parsimonious
and 8-parameter minimum-error KGED equations on the 168 thin-film electrochemical conditions."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    lab = json.loads((ROOT / "data/data.json").read_text(encoding="utf-8"))
    t = (np.asarray(lab["T_C"], float) + 273.15) / 298.15
    c = np.asarray(lab["C_wt"], float)
    h = np.asarray(lab["h_um"], float) / 200.0
    y_true = np.asarray(lab["log_i"], float)

    print("=== 1. Laboratory Rate Equation Closed-Form & CV Verification (168 Conditions) ===")
    for alias, label in [
        ("parsimonious", "KGED (5 parameters, parsimonious)"),
        ("best_accuracy", "KGED (8 parameters, minimum-error)")
    ]:
        r = json.loads((ROOT / f"engineering/{alias}_equation.json").read_text(encoding="utf-8"))
        p = {m["hypothesis"]: m["parameters"] for m in r["physical_modules"]}
        B, D = p["H2"]["p0"], p["H2"]["p1"]
        salt = np.log((1.0 + B * c * np.exp(-D * c)) / (1.0 + B * 3.5 * np.exp(-D * 3.5)))

        if alias == "parsimonious":
            a = p["H3"]["p0"]
            pred = r["intercept"] + p["H1"]["p0"] * (1.0 - 1.0 / t) + salt + np.log(h * (a + 1.0) / (a + h * h))
            c_peak = 1.0 / D
            h_peak = 200.0 * np.sqrt(a)
        else:
            A, K = p["H1"]["p0"], p["H1"]["p1"]
            E, n, L = p["H3"]["p0"], p["H3"]["p1"], p["H3"]["p2"]
            pred = r["intercept"] + A * (np.tanh(K * t) - np.tanh(K)) + salt + np.log((1.0 + E * (h ** n) * np.exp(-L * h)) / (1.0 + E * np.exp(-L)))
            c_peak = 1.0 / D
            h_peak = 200.0 * (n / L)

        max_diff = float(np.max(np.abs(pred - np.asarray(r["full_predictions"], float))))
        train_rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
        cv_rmse = float(r["score"]["cv_log_rmse"])

        print(f"\n[{label}]")
        print(f"  - Closed-form reconstruction max |delta| : {max_diff:.2e}")
        print(f"  - Full-dataset training log-RMSE (N=168) : {train_rmse:.3f}")
        print(f"  - 23-fold balanced grouped CV log-RMSE   : {cv_rmse:.3f}")
        print(f"  - Analytical stationary peaks (Table 5)  : C* = {c_peak:.2f} wt% NaCl, H* = {h_peak:.1f} um")
        np.testing.assert_allclose(pred, r["full_predictions"], atol=2e-11, rtol=0)

    print("\nPASSED: All laboratory rate equation checks verified to < 2e-11.")

if __name__ == "__main__":
    main()
