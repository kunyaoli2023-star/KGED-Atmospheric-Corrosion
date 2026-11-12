# -*- coding: utf-8 -*-
"""Independent closed-form reconstruction and Gauss-Legendre quadrature verification
of the retained 5- and 8-parameter KGED rate equations."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engineering.engineering_integrator import CASES, Integrator

CLOSED_FORM_ATOL = 2e-11
QUADRATURE_REL_TOL = 1e-10

read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))


def require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)


class ClosedFormLaw:
    """Analytical closed-form evaluator for retained KGED rate laws."""

    def __init__(self, alias: str, r: dict):
        self.alias = alias
        self.r = r
        self.p = {m["hypothesis"]: m["parameters"] for m in r["physical_modules"]}

    def predict_log(self, T, C, H) -> np.ndarray:
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
        if self.alias == "best_accuracy":
            A, K = self.p["H1"]["p0"], self.p["H1"]["p1"]
            E, n, L = self.p["H3"]["p0"], self.p["H3"]["p1"], self.p["H3"]["p2"]
            return (
                self.r["intercept"]
                + A * (np.tanh(K * t) - np.tanh(K))
                + salt
                + np.log((1.0 + E * (h ** n) * np.exp(-L * h)) / (1.0 + E * np.exp(-L)))
            )
        raise ValueError(f"Unknown closed-form equation alias: {self.alias}")


def main():
    mapping = read(ROOT / "inputs/mapping.json")
    env_data = {}
    for c_name in CASES:
        with np.load(ROOT / f"inputs/{c_name}_environment.npz") as z:
            env_data[c_name] = {k: z[k].copy() for k in z.files}

    checks = {}
    for alias in ("parsimonious", "best_accuracy"):
        eq_path = ROOT / f"{alias}_equation.json"
        if not eq_path.exists():
            continue
        r = read(eq_path)
        law = ClosedFormLaw(alias, r)
        ref_pred = float(law.predict_log(25.0, 3.5, 200.0)[0])
        error = abs(ref_pred - float(r["intercept"]))
        require(error < CLOSED_FORM_ATOL, f"Closed-form reference error too large for {alias}: {error:.2e}")

        integrator = Integrator(env_data, law, mapping)
        quad_diffs = {}
        quad_rel_diffs = {}
        for c_name in CASES:
            b16 = integrator.basis(c_name, mapping["fixed_state_structure"], 16)
            b32 = integrator.basis(c_name, mapping["fixed_state_structure"], 32)
            q_abs = float(np.max(np.abs(b16 - b32)))
            q_rel = q_abs / max(1.0, float(np.max(np.abs(b32))))
            require(
                q_rel < QUADRATURE_REL_TOL,
                f"Quadrature relative difference too large for {alias} on {c_name}: {q_rel:.2e}",
            )
            quad_diffs[c_name] = q_abs
            quad_rel_diffs[c_name] = q_rel

        checks[alias] = {
            "independent_closed_form_max_log_difference": error,
            "i_at_lab_reference_A_cm2": float(np.exp(r["intercept"])),
            "clock_reference_Istar_A_cm2": float(integrator.reference),
            "gauss_legendre_16_vs_32_max_basis_diff_um": quad_diffs,
            "gauss_legendre_16_vs_32_max_relative_diff": quad_rel_diffs,
        }
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
