"""Independent closed-form reconstruction and Gauss-Legendre quadrature verification
of the frozen 5-parameter parsimonious and 8-parameter minimum-error KGED equations."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engineering.engineering_integrator import CASES, Integrator

read = lambda p: json.loads(p.read_text(encoding="utf-8"))


class ClosedFormLaw:
    def __init__(self, alias, r):
        self.alias = alias
        self.r = r
        self.p = {m["hypothesis"]: m["parameters"] for m in r["physical_modules"]}

    def predict_log(self, T, C, H):
        T, C, H = np.broadcast_arrays(
            np.atleast_1d(np.asarray(T, float)),
            np.atleast_1d(np.asarray(C, float)),
            np.atleast_1d(np.asarray(H, float)),
        )
        t = (T + 273.15) / 298.15
        c = C
        h = H / 200.0
        B, D = self.p["H2"]["p0"], self.p["H2"]["p1"]
        salt = np.log((1.0 + B * c * np.exp(-D * c)) / (1.0 + B * 3.5 * np.exp(-D * 3.5)))
        if self.alias == "parsimonious":
            a = self.p["H3"]["p0"]
            return self.r["intercept"] + self.p["H1"]["p0"] * (1.0 - 1.0 / t) + salt + np.log(h * (a + 1.0) / (a + h * h))
        A, K = self.p["H1"]["p0"], self.p["H1"]["p1"]
        E, n, L = self.p["H3"]["p0"], self.p["H3"]["p1"], self.p["H3"]["p2"]
        return (
            self.r["intercept"]
            + A * (np.tanh(K * t) - np.tanh(K))
            + salt
            + np.log((1.0 + E * (h ** n) * np.exp(-L * h)) / (1.0 + E * np.exp(-L)))
        )


def main():
    lab = read(REPO_ROOT / "data/data.json")
    mapping = read(ROOT / "inputs/mapping.json")
    env_data = {}
    for c_name in CASES:
        with np.load(ROOT / f"inputs/{c_name}_environment.npz") as z:
            env_data[c_name] = {k: z[k].copy() for k in z.files}

    checks = {}
    for alias in ("parsimonious", "best_accuracy"):
        r = read(ROOT / f"{alias}_equation.json")
        law = ClosedFormLaw(alias, r)
        pred = law.predict_log(lab["T_C"], lab["C_wt"], lab["h_um"])
        error = float(np.max(np.abs(pred - np.asarray(r["full_predictions"], float))))
        np.testing.assert_allclose(pred, r["full_predictions"], atol=2e-11, rtol=0)

        integrator = Integrator(env_data, law, mapping)
        quad_diffs = {}
        for c_name in CASES:
            b16 = integrator.basis(c_name, mapping["fixed_state_structure"], 16)
            b32 = integrator.basis(c_name, mapping["fixed_state_structure"], 32)
            q_err = float(np.max(np.abs(b16 - b32)))
            assert q_err < 1e-12, f"Quadrature difference too large for {alias} on {c_name}: {q_err}"
            quad_diffs[c_name] = q_err

        checks[alias] = {
            "independent_closed_form_max_log_difference": error,
            "i_at_lab_reference_A_cm2": float(np.exp(r["intercept"])),
            "clock_reference_Istar_A_cm2": float(integrator.reference),
            "gauss_legendre_16_vs_32_max_basis_diff_um": quad_diffs,
        }
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
