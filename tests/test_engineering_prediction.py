# -*- coding: utf-8 -*-
"""Test script 2: Recompute and verify the cross-environment engineering corrosion-depth predictions
(Table 7 Weihai shoreline source-calibrated Bernstein state amplitudes, Figs. 9-10 Eq. (16)
initial-depth-anchored target trajectory RMSE and endpoint errors across Weihai near-coastal,
indoor natatorium, and NASA KSC, the 6 single-factor integrator ablations in Fig. 10a, c-e,
and the 12 component-wise rate-law ablations in Fig. 11b)."""
from __future__ import annotations

import json
import sys
import tempfile
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
import run_engineering_predictions as stage5


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

    # Verify post-initial target observation counts: 8 (Weihai near-coastal) + 8 (Indoor natatorium) + 9 (NASA KSC) = 25 points
    post_counts = [
        len(json.loads((ROOT / f"engineering/inputs/{s}_labels.json").read_text(encoding="utf-8"))["months"]) - 1
        for s in ("jinhai", "pool", "ksc")
    ]
    require(post_counts == [8, 8, 9], f"Expected [8, 8, 9] post-t1 target points, got {post_counts}")

    expected_amps = {
        "parsimonious": [7.176, 6.495, 4.849, 4.374],
        "best_accuracy": [7.423, 6.560, 4.770, 4.198],
    }
    expected_a3_over_a0 = {
        "parsimonious": 0.610,
        "best_accuracy": 0.566,
    }
    expected_targets = {
        "parsimonious": {
            "jinhai": (14.03, -0.10),
            "pool": (32.38, -0.49),
            "ksc": (4.96, -3.28),
            "mean_abs_ep": 1.29,
        },
        "best_accuracy": {
            "jinhai": (18.36, 4.41),
            "pool": (32.35, 18.69),
            "ksc": (4.87, 2.32),
            "mean_abs_ep": 8.47,
        },
    }

    print("=== 2. Cross-Environment Engineering Prediction Verification (Table 7 & Figs. 9-10) ===")
    print("\n[Table 7: Source-Calibrated Bernstein State Amplitudes at Weihai Shoreline (Recomputed + Archived)]")
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
        got_ratio = round(float(fitted["coefficients"][3]) / float(fitted["coefficients"][0]), 3)
        exp_ratio = expected_a3_over_a0[name]
        require(abs(got_ratio - exp_ratio) < 1e-3 + 1e-9, f"Table 7 a3/a0 mismatch for {name}: {got_ratio} vs {exp_ratio}")
        recomputed_fits[name] = (integrator, np.asarray(fitted["coefficients"], float))
        print(f"  - {name:15s} ({cfg['laboratory_parameter_count']} params): a0..a3={amps}, a3/a0={got_ratio:.3f}")

    print("\n[Figs. 9-10: Target-Environment Trajectory RMSE and Endpoint Errors (with Auxiliary nRMSE and Pointwise MAPE)]")
    sites = [
        ("jinhai", "Weihai near-coastal (near-coastal marine)"),
        ("pool", "Indoor natatorium (humid chlorinated)"),
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

            exp_rmse, exp_ep = expected_targets[alias][site_key]
            require(abs(rmse - exp_rmse) < 0.005 + 1e-9, f"Target RMSE mismatch ({alias}, {site_key}): {rmse:.4f} vs {exp_rmse}")
            require(abs(ep_err - exp_ep) < 0.005 + 1e-9, f"Target EP mismatch ({alias}, {site_key}): {ep_err:.4f} vs {exp_ep}")
            sc = scores_map[(alias, site_key)]
            require(abs(sc["depth_rmse_um"] - exp_rmse) < 0.005 + 1e-9, f"Archived score RMSE mismatch ({alias}, {site_key})")
            require(abs(sc["endpoint_error_pct"] - exp_ep) < 0.005 + 1e-9, f"Archived score EP mismatch ({alias}, {site_key})")

            print(
                f"    * {site_label:40s}: RMSE = {rmse:6.2f} um | nRMSE = {nrmse:4.1f}% | "
                f"PointMAPE = {mape:5.1f}% | Endpoint error = {ep_err:+6.2f}%"
            )
        mean_abs_ep = float(np.mean(apes))
        exp_mean = expected_targets[alias]["mean_abs_ep"]
        require(
            abs(mean_abs_ep - exp_mean) < 0.005 + 1e-9,
            f"Mean abs endpoint error mismatch ({alias}): {mean_abs_ep:.4f} vs {exp_mean}",
        )
        print(
            f"    => Mean nRMSE = {np.mean(nrmses):.1f}% | Mean PointMAPE = {np.mean(mapes):.1f}% | "
            f"Mean absolute endpoint error = {mean_abs_ep:.2f}%"
        )

    print("\nPASSED: All cross-environment engineering predictions (Table 7 & Figs. 9-10) recomputed and verified.")


def test_engineering_prediction():
    main()


class TestEngineeringPrediction(unittest.TestCase):
    def test_table7_and_target_trajectories(self):
        main()

    def test_full_stage5_benchmarks_and_ablations(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = stage5.run_all(Path(tmp), skip_plots=True)
            self.assertEqual(summary["status"], "SUCCESS")
            self.assertEqual(summary["manuscript_fig10ab_benchmarks_verified"], "18/18")
            self.assertEqual(summary["manuscript_fig10cde_single_factor_ablations_verified"], "6/6")
            self.assertAlmostEqual(summary["kged_5param_target_mean_abs_endpoint_error_pct"], 1.29, places=2)
            self.assertAlmostEqual(summary["kged_8param_target_mean_abs_endpoint_error_pct"], 8.47, places=2)

    def test_convergence_eq8_and_canonical_ast_nodes(self):
        kg_root = ROOT / "knowledge_graph"
        if str(kg_root) not in sys.path:
            sys.path.insert(0, str(kg_root))
        from maief import convergence, numerics

        self.assertEqual(convergence.EPSILON, 0.002)
        prev_front = [{"score": {"cv_log_rmse": 0.5700, "parameter_count": 5, "canonical_ast_nodes": 30}}]
        cur_within_eps = [{"score": {"cv_log_rmse": 0.5682, "parameter_count": 5, "canonical_ast_nodes": 30}}]
        cur_exceeds_eps = [{"score": {"cv_log_rmse": 0.5675, "parameter_count": 5, "canonical_ast_nodes": 30}}]
        self.assertTrue(convergence.stable_transition(prev_front, cur_within_eps))
        self.assertFalse(convergence.stable_transition(prev_front, cur_exceeds_eps))

        for alias, exp_can in [("parsimonious", 30), ("best_accuracy", 32)]:
            eq_json = json.loads((ROOT / f"engineering/{alias}_equation.json").read_text(encoding="utf-8"))
            mods = []
            for pm in eq_json["physical_modules"]:
                params = [
                    {"name": k, "domain": "POSITIVE"}
                    for k in sorted(pm["parameters"].keys())
                ]
                rec = {
                    "module_id": pm["module_id"],
                    "proposal": {
                        "hypothesis_id": pm["hypothesis"],
                        "expression": pm["expression"],
                        "parameters": params,
                        "conditions": [],
                    },
                }
                mods.append(numerics.Module(rec))
            can_total = 1 + sum(m.canonical_nodes for m in mods) + 1
            self.assertEqual(can_total, exp_can)
            self.assertEqual(eq_json["score"]["canonical_ast_nodes"], exp_can)


if __name__ == "__main__":
    unittest.main(verbosity=2)
