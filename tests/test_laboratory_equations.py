# -*- coding: utf-8 -*-
"""Test script 1: Independent closed-form and 23-fold grouped cross-validation verification
of the 5-parameter parsimonious and 8-parameter minimum-error KGED equations on the 168
thin-film electrochemical conditions (Tables 4, 5, and 6)."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]


def eval_5param_from_latent(z, t, c, h):
    # latent parameters: [b5, ln(A5), ln(k5), ln(B5), ln(q5)]
    b5 = z[0]
    A5, k5, B5, q5 = np.exp(z[1:5])
    salt = np.log((1.0 + B5 * c * np.exp(-k5 * c)) / (1.0 + B5 * 3.5 * np.exp(-k5 * 3.5)))
    film = np.log(h * (q5 + 1.0) / (q5 + h * h))
    return b5 + A5 * (1.0 - 1.0 / t) + salt + film


def eval_8param_from_latent(z, t, c, h):
    # latent parameters follow physical_modules dict key order:
    # [b8, ln(A8), ln(s8), ln(k8), ln(B8), ln(lam8), ln(D8), ln(m8)]
    b8 = z[0]
    A8, s8, k8, B8, lam8, D8, m8 = np.exp(z[1:8])
    therm = A8 * (np.tanh(s8 * t) - np.tanh(s8))
    salt = np.log((1.0 + B8 * c * np.exp(-k8 * c)) / (1.0 + B8 * 3.5 * np.exp(-k8 * 3.5)))
    film = np.log((1.0 + D8 * (h ** m8) * np.exp(-lam8 * h)) / (1.0 + D8 * np.exp(-lam8)))
    return b8 + therm + salt + film


def main():
    lab = json.loads((ROOT / "data/data.json").read_text(encoding="utf-8"))
    folds = json.loads((ROOT / "data/folds.json").read_text(encoding="utf-8"))
    t = (np.asarray(lab["T_C"], float) + 273.15) / 298.15
    c = np.asarray(lab["C_wt"], float)
    h = np.asarray(lab["h_um"], float) / 200.0
    y_true = np.asarray(lab["log_i"], float)
    assert len(y_true) == 168, f"Expected 168 conditions, got {len(y_true)}"
    assert len(folds) == 23, f"Expected 23 grouped CV folds, got {len(folds)}"

    # Verify fold partition integrity across all 23 folds
    for f_idx, f_spec in enumerate(folds):
        tr = set(f_spec["train"])
        va = set(f_spec["validation"])
        assert tr.isdisjoint(va), f"Fold {f_idx} train/validation overlap!"
        assert tr | va == set(range(168)), f"Fold {f_idx} does not cover all 168 conditions!"

    print("=== 1. Laboratory Rate Equation Closed-Form & 23-Fold Grouped CV Verification (168 Conditions) ===")
    expected = {
        "parsimonious": {"train": 0.507, "cv": 0.570, "c_peak": 4.00, "h_peak": 167.7, "d_salt": 8.806, "d_h3": 1.703},
        "best_accuracy": {"train": 0.496, "cv": 0.567, "c_peak": 4.05, "h_peak": 184.3, "d_salt": 8.815, "d_h3": 2.473},
    }
    for alias, label in [
        ("parsimonious", "KGED (5 parameters, parsimonious)"),
        ("best_accuracy", "KGED (8 parameters, minimum-error)"),
    ]:
        r = json.loads((ROOT / f"engineering/{alias}_equation.json").read_text(encoding="utf-8"))
        p = {m["hypothesis"]: m["parameters"] for m in r["physical_modules"]}
        B, D = p["H2"]["p0"], p["H2"]["p1"]
        denom_salt = 1.0 + B * 3.5 * np.exp(-D * 3.5)
        salt = np.log((1.0 + B * c * np.exp(-D * c)) / denom_salt)

        if alias == "parsimonious":
            a = p["H3"]["p0"]
            denom_h3 = a + 1.0
            pred = r["intercept"] + p["H1"]["p0"] * (1.0 - 1.0 / t) + salt + np.log(h * denom_h3 / (a + h * h))
            c_peak = 1.0 / D
            h_peak = 200.0 * np.sqrt(a)
            eval_fn = eval_5param_from_latent
            lb = np.array([-30.0] + [-8.0] * 4)
            ub = np.array([5.0] + [6.0] * 4)
        else:
            A, K = p["H1"]["p0"], p["H1"]["p1"]
            E, n, L = p["H3"]["p0"], p["H3"]["p1"], p["H3"]["p2"]
            denom_h3 = 1.0 + E * np.exp(-L)
            pred = r["intercept"] + A * (np.tanh(K * t) - np.tanh(K)) + salt + np.log((1.0 + E * (h ** n) * np.exp(-L * h)) / denom_h3)
            c_peak = 1.0 / D
            h_peak = 200.0 * (n / L)
            eval_fn = eval_8param_from_latent
            lb = np.array([-30.0] + [-8.0] * 7)
            ub = np.array([5.0] + [6.0] * 7)

        max_diff = float(np.max(np.abs(pred - np.asarray(r["full_predictions"], float))))
        train_rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))

        # Recompute 23-fold grouped CV predictions & refits across all 23 folds
        axis_fold_rmses = {"shape_group": [], "T_C": [], "h_um": [], "C_wt": []}
        max_fold_pred_diff = 0.0
        for f_row, f_spec in zip(r["fold_rows"], folds):
            assert f_row["train_indices"] == f_spec["train"]
            assert f_row["validation_indices"] == f_spec["validation"]
            tr_idx = np.asarray(f_row["train_indices"], int)
            va_idx = np.asarray(f_row["validation_indices"], int)
            z_saved = np.asarray(f_row["latent_parameters"], float)

            # 1) Verify closed-form prediction from fold parameters on held-out validation indices
            va_pred = eval_fn(z_saved, t[va_idx], c[va_idx], h[va_idx])
            fold_diff = float(np.max(np.abs(va_pred - np.asarray(f_row["predicted"], float))))
            max_fold_pred_diff = max(max_fold_pred_diff, fold_diff)
            np.testing.assert_allclose(va_pred, f_row["predicted"], atol=1e-11, rtol=0)
            np.testing.assert_allclose(y_true[va_idx], f_row["observed"], atol=1e-12, rtol=0)
            fold_rmse_exact = float(np.sqrt(np.mean((va_pred - y_true[va_idx]) ** 2)))
            np.testing.assert_allclose(fold_rmse_exact, f_row["rmse"], atol=1e-12, rtol=0)

            # 2) Verify nonlinear least-squares refit optimality on training fold
            res = least_squares(
                lambda z: eval_fn(z, t[tr_idx], c[tr_idx], h[tr_idx]) - y_true[tr_idx],
                z_saved,
                bounds=(lb, ub),
                method="trf",
                x_scale="jac",
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            va_pred_refit = eval_fn(res.x, t[va_idx], c[va_idx], h[va_idx])
            fold_rmse_refit = float(np.sqrt(np.mean((va_pred_refit - y_true[va_idx]) ** 2)))
            np.testing.assert_allclose(fold_rmse_refit, f_row["rmse"], atol=5e-5, rtol=0)
            axis_fold_rmses[f_row["axis"]].append(fold_rmse_exact)

        recomputed_axis = {ax: float(np.mean(vals)) for ax, vals in axis_fold_rmses.items()}
        for ax, val in recomputed_axis.items():
            np.testing.assert_allclose(val, r["score"]["axis_rmse"][ax], atol=1e-6, rtol=0)
        cv_rmse = float(np.mean(list(recomputed_axis.values())))
        np.testing.assert_allclose(cv_rmse, r["score"]["cv_log_rmse"], atol=1e-6, rtol=0)

        exp = expected[alias]
        assert round(denom_salt, 3) == exp["d_salt"], f"Table 4 salt denom mismatch: {denom_salt:.3f} vs {exp['d_salt']}"
        assert round(denom_h3, 3) == exp["d_h3"], f"Table 4 film const mismatch: {denom_h3:.3f} vs {exp['d_h3']}"
        assert round(train_rmse, 3) == exp["train"], f"Train RMSE mismatch: {train_rmse:.3f} vs {exp['train']}"
        assert round(cv_rmse, 3) == exp["cv"], f"CV RMSE mismatch: {cv_rmse:.3f} vs {exp['cv']}"
        assert round(c_peak, 2) == exp["c_peak"], f"C* mismatch: {c_peak:.2f} vs {exp['c_peak']}"
        assert round(h_peak, 1) == exp["h_peak"], f"H* mismatch: {h_peak:.1f} vs {exp['h_peak']}"
        np.testing.assert_allclose(pred, r["full_predictions"], atol=2e-11, rtol=0)

        print(f"\n[{label}]")
        print(f"  - Closed-form reconstruction max |delta| : {max_diff:.2e} (23-fold max |delta|: {max_fold_pred_diff:.2e})")
        print(f"  - Table 4 reference constants            : salt denom = {denom_salt:.3f}, film const = {denom_h3:.3f}")
        print(f"  - Full-dataset training log-RMSE (N=168) : {train_rmse:.3f}")
        print(f"  - 23-fold refit balanced CV log-RMSE     : {cv_rmse:.4f} (reported: {cv_rmse:.3f})")
        print(f"  - Analytical stationary peaks (Table 5)  : C* = {c_peak:.2f} wt% NaCl, H* = {h_peak:.1f} um")

    print("\nPASSED: All laboratory rate equation closed-form and 23-fold CV refit checks verified.")


if __name__ == "__main__":
    main()
