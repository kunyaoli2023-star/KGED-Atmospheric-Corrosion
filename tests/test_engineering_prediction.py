# -*- coding: utf-8 -*-
"""Test script 2: Recompute and verify the cross-environment engineering corrosion-depth predictions
(Table 7 Linhai source-calibrated Bernstein state amplitudes and Table 8 Eq. (16) initial-depth-anchored
target trajectory RMSE, normalised RMSE, pointwise MAPE, and endpoint errors across Jinhai,
Indoor Swimming Pool, and NASA KSC)."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engineering.engineering_integrator import CASES, Integrator
from engineering import engineering_state as state
from engineering.verify_closed_forms import ClosedFormLaw


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
        assert fitted["eligible"] is True
        np.testing.assert_allclose(fitted["coefficients"], cfg["amplitudes"], atol=1e-9, rtol=0)
        amps = [round(x, 3) for x in fitted["coefficients"]]
        assert amps == expected_amps[name], f"Table 7 amplitude mismatch for {name}: {amps} vs {expected_amps[name]}"
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
            # Eq. (16): D_pred(t) = D_obs(t_1) + sum_k a_k [B_k(t) - B_k(t_1)]
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
            assert round(rmse, 2) == exp_rmse, f"Table 8 RMSE mismatch ({alias}, {site_key}): {rmse:.2f} vs {exp_rmse}"
            assert round(ep_err, 2) == exp_ep, f"Table 8 endpoint error mismatch ({alias}, {site_key}): {ep_err:.2f} vs {exp_ep}"
            sc = scores_map[(alias, site_key)]
            assert round(sc["depth_rmse_um"], 2) == exp_rmse
            assert round(sc["endpoint_error_pct"], 2) == exp_ep

            print(
                f"    * {site_label:40s}: RMSE = {rmse:6.2f} um | nRMSE = {nrmse:4.1f}% | "
                f"PointMAPE = {mape:5.1f}% | Endpoint error = {ep_err:+6.2f}%"
            )
        mean_abs_ep = float(np.mean(apes))
        assert round(mean_abs_ep, 2) == expected_table8[alias]["mean_abs_ep"], (
            f"Table 8 mean abs endpoint error mismatch ({alias}): {mean_abs_ep:.2f} vs {expected_table8[alias]['mean_abs_ep']}"
        )
        print(
            f"    => Mean nRMSE = {np.mean(nrmses):.1f}% | Mean PointMAPE = {np.mean(mapes):.1f}% | "
            f"Mean absolute endpoint error = {mean_abs_ep:.2f}%"
        )

    print("\nPASSED: All cross-environment engineering predictions (Tables 7 & 8) recomputed and verified.")


if __name__ == "__main__":
    main()
