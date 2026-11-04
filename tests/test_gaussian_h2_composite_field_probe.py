# -*- coding: utf-8 -*-
"""
Standalone Reproducibility Script for Stage 5 Field-Integration Verification of:
1. Retained 5-parameter 1-SE KGED law (M01 + M08 + M35, k = 5, CV = 0.569623, kappa = 2.47)
2. Retained 8-parameter min-CV KGED law (M23 + M08 + M30, k = 8, CV = 0.566703, kappa = 83.10)
3. Exploratory 7-parameter Gaussian H2 composite (M01 + [ExpSat + Gauss_H2] + M35, k = 7, CV = 0.549823, kappa = 13.87)
4. Exploratory 11-parameter Power+Gaussian H2 + H4 composite (M01 + [Pow + Gauss_H2] + M30 + H4, k = 11, CV = 0.495068, kappa = 34.46)

All four equations are fitted from scratch on the 168 laboratory condition means across all 23 grouped CV folds
and integrated across the four field environments (Linhai source calibration + Jinhai, Swimming Pool, and NASA KSC targets).
Note on Linhai source metrics: `linhai_source_rmse_um` and `linhai_source_endpoint_err_pct` (e.g. 22.82 um / -15.36% for 5-param)
report the unconditioned raw BVLS trajectory (matching `scores.json`), whereas the three target sites (and the main-text Section 3.3.1
conditioned Linhai values of 24.17 um / -15.77% for 5-param and 24.33 um / -15.87% for 8-param) condition on the first observation D_obs(t_1).
Outputs:
- tests/genuine_stability_benchmark_20261103/gaussian_h2_composite_field_verification.csv
- tests/genuine_stability_benchmark_20261103/gaussian_h2_composite_field_verification.json
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
sys.path.insert(0, str(RELEASE_ROOT / "knowledge_graph"))

from maief import numerics, symbols, storage
import engineering
sys.modules["engineering.storage"] = storage
sys.modules["engineering.numerics"] = numerics
sys.modules["engineering.symbols"] = symbols
from engineering.engineering_integrator import CASES, Integrator
from engineering import engineering_state as state


def make_mod(mid: str, hyp: str, expr: str, n_params: int) -> dict:
    return {
        "module_id": mid,
        "proposal": {
            "proposal_id": mid,
            "hypothesis_id": hyp,
            "expression": expr,
            "parameters": [{"name": f"p{i}", "domain": "POSITIVE"} for i in range(n_params)],
            "conditions": [],
        },
    }


class FreshLaw:
    def __init__(self, fresh_result: dict, pool_dict: dict):
        self.result = fresh_result
        self.modules = []
        for p in fresh_result["physical_modules"]:
            m = numerics.Module(pool_dict[p["module_id"]])
            z = np.asarray([
                np.log(p["parameters"][q["name"]]) if q["domain"] == "POSITIVE" else p["parameters"][q["name"]]
                for q in m.params
            ])
            self.modules.append((p["hypothesis"], m, z))

    def predict_log(self, T, C, H):
        T, C, H = np.broadcast_arrays(
            np.atleast_1d(np.asarray(T, float)),
            np.atleast_1d(np.asarray(C, float)),
            np.atleast_1d(np.asarray(H, float)),
        )
        shape = T.shape
        x = {
            "tau": (T.ravel() + 273.15) / 298.15,
            "c": C.ravel(),
            "h": H.ravel() / 200.0,
        }
        y = np.full(T.size, self.result["intercept"], float)
        for hyp, m, z in self.modules:
            y += m.response(x, z)
        return y.reshape(shape)


def main():
    kg_inputs = RELEASE_ROOT / "knowledge_graph/inputs"
    for fn in ("data.json", "folds.json"):
        if not (kg_inputs / fn).exists() and (RELEASE_ROOT / "data" / fn).exists():
            (kg_inputs / fn).write_bytes((RELEASE_ROOT / "data" / fn).read_bytes())

    options = {
        "seed": 20261007,
        "screen_starts": 2,
        "final_starts": 4,
        "screen_nfev": 220,
        "final_nfev": 900,
        "constraint_penalty": 1000.0,
    }
    evaluator = numerics.Evaluator(kg_inputs, options)

    pool = {
        "M01": make_mod("M01", "H1", "p0 * (1 - 1 / tau)", 1),
        "M23": make_mod("M23", "H1", "p0 * tanh(p1 * tau)", 2),
        "M08": make_mod("M08", "H2", "log1p(p0 * c * exp(-p1 * c))", 2),
        "M35": make_mod("M35", "H3", "log(h) - log(p0 + h ** 2)", 1),
        "M30": make_mod("M30", "H3", "log1p(p0 * h ** p1 * exp(-p2 * h))", 3),
        "H2_ExpSat_Gauss": make_mod("H2_ExpSat_Gauss", "H2", "p0 * (1 - exp(-p1 * c)) + -(c - p2) ** 2 / p3 ** 2", 4),
        "H2_Pow_Gauss": make_mod("H2_Pow_Gauss", "H2", "p0 * c ** p1 + -(c - p2) ** 2 / p3 ** 2", 4),
        "H4_ThermalFilm": make_mod("H4_ThermalFilm", "H4", "p0 * (tau - 1) * h / (1 + p1 * h)", 2),
    }

    candidates = [
        ("k5_1SE_M01_M08_M35", "Retained 5-param 1-SE law (M01 + M08 + M35)", ["M01", "M08", "M35"]),
        ("k8_MinCV_M23_M08_M30", "Retained 8-param min-CV law (M23 + M08 + M30)", ["M23", "M08", "M30"]),
        ("k7_GaussH2_M01_ExpSatGauss_M35", "Exploratory 7-param Gaussian H2 composite (M01 + [ExpSat+Gauss_H2] + M35)", ["M01", "H2_ExpSat_Gauss", "M35"]),
        ("k11_GaussH2_M01_PowGauss_M30_H4", "Exploratory 11-param Power+Gaussian H2 + H4 composite (M01 + [Pow+Gauss_H2] + M30 + H4)", ["M01", "H2_Pow_Gauss", "M30", "H4_ThermalFilm"]),
    ]

    in_dir = RELEASE_ROOT / "engineering/inputs"
    mapping = json.loads((in_dir / "mapping.json").read_text(encoding="utf-8"))
    env_data = {}
    labels = {}
    for c in CASES:
        with np.load(in_dir / f"{c}_environment.npz") as npz:
            env_data[c] = {k: npz[k].copy() for k in npz.files}
        labels[c] = json.loads((in_dir / f"{c}_labels.json").read_text(encoding="utf-8"))

    linhai_obs = np.asarray(labels["linhai"]["observed_depth_um"], float)
    linhai_months = np.asarray(labels["linhai"]["months"], float)

    csv_rows = []
    json_records = []

    print("=" * 100)
    print("STAGE 5 FIELD-INTEGRATION VERIFICATION OF RETAINED LAWS VS. GAUSSIAN H2 COMPOSITE CHALLENGERS")
    print("=" * 100)

    for code, desc, mod_ids in candidates:
        res = evaluator.evaluate([pool[m] for m in mod_ids], fidelity="final")
        sc = res["score"]
        law = FreshLaw(res, pool)
        integrator = Integrator(env_data, law, mapping)
        state_cfg = dict(mapping["fixed_state_structure"])
        state_cfg["tau"] = 0.25
        state_cfg["lam"] = 0.03
        fitted_state = state.fit(integrator, state_cfg, linhai_obs, linhai_months, len(linhai_obs))
        amps = [round(float(x), 6) for x in fitted_state["coefficients"]]

        site_res = {}
        for c in CASES:
            b16 = integrator.basis(c, state_cfg, quadrature=16)
            pred_raw = b16 @ np.asarray(fitted_state["coefficients"], float)
            y_obs = np.asarray(labels[c]["observed_depth_um"], float)
            if c == "linhai":
                err = pred_raw - y_obs
                rmse = float(np.sqrt(np.mean(err ** 2)))
                ep_err = float((pred_raw[-1] - y_obs[-1]) / y_obs[-1] * 100.0)
                pred_final = float(pred_raw[-1])
            else:
                pred_cond = y_obs[0] + (pred_raw - pred_raw[0])
                err = pred_cond[1:] - y_obs[1:]
                rmse = float(np.sqrt(np.mean(err ** 2)))
                ep_err = float((pred_cond[-1] - y_obs[-1]) / y_obs[-1] * 100.0)
                pred_final = float(pred_cond[-1])
            site_res[c] = {
                "observed_final_um": round(float(y_obs[-1]), 4),
                "predicted_final_um": round(pred_final, 4),
                "endpoint_error_pct": round(ep_err, 4),
                "trajectory_rmse_um": round(rmse, 4),
            }

        target_mean_abs_ep = round(float(np.mean([abs(site_res[c]["endpoint_error_pct"]) for c in CASES[1:]])), 4)
        target_mean_rmse = round(float(np.mean([site_res[c]["trajectory_rmse_um"] for c in CASES[1:]])), 4)

        expr_summary = " | ".join(f"{m['hypothesis']}: {m['expression']}" for m in res["physical_modules"])
        param_summary = {
            m["hypothesis"]: {pk: round(float(pv), 6) for pk, pv in m["parameters"].items()}
            for m in res["physical_modules"]
        }

        print(f"\n[{desc}]")
        print(f"  Expression : {expr_summary}")
        print(f"  Intercept  : {res['intercept']:.6f} | Parameters: {param_summary}")
        print(f"  Lab Metrics: k = {sc['parameter_count']} | 23-fold CV = {sc['cv_log_rmse']:.6f} | Train RMSE = {sc['train_log_rmse']:.6f} | R2 = {sc['train_log_r2']:.6f} | kappa(J_theta) = {sc['scaled_condition_number']:.2f}")
        print(f"  Linhai Fit : a0..a3 = {[round(x, 4) for x in amps]} | Source RMSE = {site_res['linhai']['trajectory_rmse_um']:.2f} um ({site_res['linhai']['endpoint_error_pct']:+.2f}%)")
        print(f"  Target Fits: Jinhai = {site_res['jinhai']['endpoint_error_pct']:+.2f}% (RMSE {site_res['jinhai']['trajectory_rmse_um']:.2f} um) | Pool = {site_res['pool']['endpoint_error_pct']:+.2f}% (RMSE {site_res['pool']['trajectory_rmse_um']:.2f} um) | KSC = {site_res['ksc']['endpoint_error_pct']:+.2f}% (RMSE {site_res['ksc']['trajectory_rmse_um']:.2f} um)")
        print(f"  => 3-Target-Site Mean |Endpoint Error| = {target_mean_abs_ep:.2f}% | Mean Trajectory RMSE = {target_mean_rmse:.2f} um")

        row_dict = {
            "model_code": code,
            "description": desc,
            "parameter_count_k": sc["parameter_count"],
            "cv_log_rmse": round(float(sc["cv_log_rmse"]), 6),
            "train_log_rmse": round(float(sc["train_log_rmse"]), 6),
            "train_log_r2": round(float(sc["train_log_r2"]), 6),
            "jacobian_rank": sc["response_jacobian_rank"],
            "boundary_count": sc["boundary_count"],
            "condition_number_kappa": round(float(sc["scaled_condition_number"]), 4),
            "intercept_beta0": round(float(res["intercept"]), 6),
            "expressions": expr_summary,
            "fitted_parameters_json": json.dumps(param_summary, ensure_ascii=False),
            "linhai_state_amplitudes_a0_a3": json.dumps(amps),
            "linhai_source_endpoint_err_pct": site_res["linhai"]["endpoint_error_pct"],
            "linhai_source_rmse_um": site_res["linhai"]["trajectory_rmse_um"],
            "linhai_conditioning_note": "Unconditioned raw source trajectory (matching scores.json); main-text Section 3.3.1 reports first-point D_obs(t1)-conditioned Linhai trajectory (24.17 um / -15.77% for 5-param, 24.33 um / -15.87% for 8-param).",
            "jinhai_endpoint_err_pct": site_res["jinhai"]["endpoint_error_pct"],
            "jinhai_rmse_um": site_res["jinhai"]["trajectory_rmse_um"],
            "pool_endpoint_err_pct": site_res["pool"]["endpoint_error_pct"],
            "pool_rmse_um": site_res["pool"]["trajectory_rmse_um"],
            "ksc_endpoint_err_pct": site_res["ksc"]["endpoint_error_pct"],
            "ksc_rmse_um": site_res["ksc"]["trajectory_rmse_um"],
            "three_target_mean_abs_endpoint_err_pct": target_mean_abs_ep,
            "three_target_mean_trajectory_rmse_um": target_mean_rmse,
        }
        csv_rows.append(row_dict)
        json_records.append({**row_dict, "site_details": site_res, "fitted_parameters": param_summary})

    out_dir = RELEASE_ROOT / "tests/genuine_stability_benchmark_20261103"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gaussian_h2_composite_field_verification.csv"
    json_path = out_dir / "gaussian_h2_composite_field_verification.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path.write_text(json.dumps(json_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved verification CSV : {csv_path}")
    print(f"Saved verification JSON: {json_path}")


if __name__ == "__main__":
    main()
