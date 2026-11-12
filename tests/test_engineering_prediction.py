# -*- coding: utf-8 -*-
"""Test script 2: Recompute and verify the cross-environment engineering corrosion-depth predictions
(Table 7 Linhai source-calibrated Bernstein state amplitudes and Table 8 Eq. (16) initial-depth-anchored
target trajectory RMSE, normalised RMSE, pointwise MAPE, and endpoint errors across Jinhai,
Indoor Swimming Pool, and NASA KSC)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engineering.engineering_integrator import CASES, Integrator
from engineering import engineering_state as state
from engineering.verify_closed_forms import ClosedFormLaw, require


def main():
    amp_data = json.loads((ROOT / "engineering/source_state_amplitudes.json").read_text(encoding="utf-8"))
    forecast = json.loads((ROOT / "engineering/current_conditioned_forecast.json").read_text(encoding="utf-8"))
    scores = json.loads((ROOT / "engineering/scores.json").read_text(encoding="utf-8"))
    mapping = json.loads((ROOT / "engineering/inputs/mapping.json").read_text(encoding="utf-8"))
    linhai_lbl = json.loads((ROOT / "engineering/inputs/linhai_labels.json").read_text(encoding="utf-8"))
    scores_map = {(e["model"], e["case"]): e for e in scores}

    env_data = {}
    for c_name in CASES:
        with np.load(ROOT / f"engineering/inputs/{c_name}_environment.npz") as z:
            env_data[c_name] = {k: z[k].copy() for k in z.files}

    # Verify post-initial target observation counts: 8 (Jinhai) + 8 (Pool) + 9 (KSC) = 25 points
    post_counts = [
        len(json.loads((ROOT / f"engineering/inputs/{s}_labels.json").read_text(encoding="utf-8"))["months"]) - 1
        for s in ("jinhai", "pool", "ksc")
    ]
    require(post_counts == [8, 8, 9], f"Expected [8, 8, 9] post-t1 target points, got {post_counts}")

    expected_amps = {
        "parsimonious": [7.178, 6.497, 4.851, 4.376],
        "best_accuracy": [7.426, 6.562, 4.771, 4.200],
    }
    expected_table8 = {
        "parsimonious": {
            "jinhai": (14.05, -0.07),
            "pool": (32.38, -0.48),
            "ksc": (4.98, -2.94),
            "mean_abs_ep": 1.16,
        },
        "best_accuracy": {
            "jinhai": (18.39, 4.45),
            "pool": (32.36, 18.70),
            "ksc": (4.83, 2.73),
            "mean_abs_ep": 8.63,
        },
    }

    print("=== 2. Cross-Environment Engineering Prediction Verification (Tables 7 & 8) ===")
    print("\n[Table 7: Source-Calibrated Bernstein State Amplitudes at Linhai (Recomputed + Archived)]")
    recomputed_fits = {}
    for cfg in amp_data["configurations"]:
        name = cfg["configuration"]
        eq_json = json.loads((ROOT / f"engineering/{name}_equation.json").read_text(encoding="utf-8"))
        law = ClosedFormLaw(name, eq_json)
        integrator = Integrator(env_data, law, mapping)
        fitted = state.fit(
            integrator,
            mapping["fixed_state_structure"],
            np.asarray(linhai_lbl["observed_depth_um"], float),
            np.asarray(linhai_lbl["months"], float),
            len(linhai_lbl["observed_depth_um"]),
        )
        require(fitted["eligible"] is True, f"Source fit ineligible for {name}")
        np.testing.assert_allclose(fitted["coefficients"], cfg["amplitudes"], atol=1e-9, rtol=0)
        for got_a, exp_a in zip(fitted["coefficients"], expected_amps[name]):
            require(abs(got_a - exp_a) < 5e-4 + 1e-9, f"Table 7 amplitude mismatch for {name}: {got_a} vs {exp_a}")
        amps = [round(x, 3) for x in fitted["coefficients"]]
        recomputed_fits[name] = (integrator, np.asarray(fitted["coefficients"], float))
        print(f"  - {name:15s} ({cfg['laboratory_parameter_count']} params): a0..a3 = {amps}")

    print("\n[Table 8: Target-Environment Trajectory RMSE, nRMSE, Pointwise MAPE, and Endpoint Errors]")
    sites = [
        ("jinhai", "Jinhai (near-coastal marine)"),
        ("pool", "Indoor swimming pool (humid chlorinated)"),
        ("ksc", "NASA KSC (subtropical severe marine)"),
    ]
    for alias, label in [
        ("parsimonious", "KGED (5 parameters, parsimonious)"),
        ("best_accuracy", "KGED (8 parameters, minimum-error)"),
    ]:
        print(f"\n  [{label}]")
        integrator, a_vec = recomputed_fits[alias]
        preds = forecast["models"][alias]["predictions"]
        apes, mapes, nrmses = [], [], []
        for site_key, site_label in sites:
            lbl = json.loads((ROOT / f"engineering/inputs/{site_key}_labels.json").read_text(encoding="utf-8"))
            y_obs_all = np.asarray(lbl["observed_depth_um"], float)
            b_mat = integrator.basis(site_key, mapping["fixed_state_structure"], 16)
            raw_pred = b_mat @ a_vec
            anchored_pred = y_obs_all[0] + (raw_pred - raw_pred[0])
            np.testing.assert_allclose(anchored_pred, preds[site_key]["predicted_depth_um"], atol=1e-9, rtol=0)

            y_obs = y_obs_all[1:]
            y_pred = anchored_pred[1:]
            err = y_pred - y_obs
            rmse = float(np.sqrt(np.mean(err ** 2)))
            ep_err = float((y_pred[-1] - y_obs[-1]) / y_obs[-1] * 100.0)
            mape = float(np.mean(np.abs(err / y_obs)) * 100.0)
            nrmse = float(rmse / y_obs[-1] * 100.0)
            apes.append(abs(ep_err))
            mapes.append(mape)
            nrmses.append(nrmse)

            exp_rmse, exp_ep = expected_table8[alias][site_key]
            require(abs(rmse - exp_rmse) < 0.005 + 1e-9, f"Table 8 RMSE mismatch ({alias}, {site_key}): {rmse:.4f} vs {exp_rmse}")
            require(abs(ep_err - exp_ep) < 0.005 + 1e-9, f"Table 8 EP mismatch ({alias}, {site_key}): {ep_err:.4f} vs {exp_ep}")
            sc = scores_map[(alias, site_key)]
            require(abs(sc["depth_rmse_um"] - exp_rmse) < 0.005 + 1e-9, f"Archived score RMSE mismatch ({alias}, {site_key})")
            require(abs(sc["endpoint_error_pct"] - exp_ep) < 0.005 + 1e-9, f"Archived score EP mismatch ({alias}, {site_key})")

            print(
                f"    * {site_label:40s}: RMSE = {rmse:6.2f} um | nRMSE = {nrmse:4.1f}% | "
                f"PointMAPE = {mape:5.1f}% | Endpoint error = {ep_err:+6.2f}%"
            )
        mean_abs_ep = float(np.mean(apes))
        exp_mean = expected_table8[alias]["mean_abs_ep"]
        require(
            abs(mean_abs_ep - exp_mean) < 0.005 + 1e-9,
            f"Table 8 mean abs endpoint error mismatch ({alias}): {mean_abs_ep:.4f} vs {exp_mean}",
        )
        print(
            f"    => Mean nRMSE = {np.mean(nrmses):.1f}% | Mean PointMAPE = {np.mean(mapes):.1f}% | "
            f"Mean absolute endpoint error = {mean_abs_ep:.2f}%"
        )

    print("\nPASSED: All cross-environment engineering predictions (Tables 7 & 8) recomputed and verified.")


def test_engineering_prediction():
    main()


class TestEngineeringPrediction(unittest.TestCase):
    def test_tables_7_and_8(self):
        main()


if __name__ == "__main__":
    main()
