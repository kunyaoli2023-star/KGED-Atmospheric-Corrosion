# -*- coding: utf-8 -*-
"""Microenvironmental coordinate mapping (Eq. 11) and reaction-clock Bernstein
state-basis integration (Eqs. 12-16) for cross-environment corrosion prediction."""
from __future__ import annotations

import json
import numpy as np
from . import engineering_state as state

CASES = ("linhai", "jinhai", "pool", "ksc")
DEFAULT_MAPPING = {
    "RH0": 55.0,
    "alpha": 0.3,
    "J0": 20000.0,
    "Hmin": 50.0,
    "Hmax": 1000.0,
}


def fingerprint(c: dict) -> str:
    return json.dumps(c, sort_keys=True)


class Integrator:
    """Shared microenvironmental mapping (Eq. 11) and Gauss-Legendre state-basis integrator."""

    def __init__(self, data: dict, law, mapping: dict, **overrides):
        self.data = data
        self.law = law
        self.mapping = {**DEFAULT_MAPPING, **mapping, **overrides}
        m = self.mapping
        self.rh0 = float(m["RH0"])
        self.alpha = float(m["alpha"])
        self.j0 = float(m["J0"])
        self.hmin = float(m["Hmin"])
        self.hmax = float(m["Hmax"])
        # Reference current density I* = 0.5 * i_corr(T_ref=25 C, C_bg, H_field,ref=500 um)
        self.reference = 0.5 * float(np.exp(law.predict_log([25.0], [m["Cbg"]], [500.0]))[0])
        if not np.isfinite(self.reference) or self.reference <= 0.0:
            raise ValueError("INVALID_REFERENCE_CURRENT")
        self.cache: dict = {}

    def drivers(self, case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.data[case]
        m = self.mapping
        q = np.clip((d["RH_pct"] - self.rh0) / (100.0 - self.rh0), 0.0, 1.0) ** self.alpha
        c_eq = m["Cbg"] + (m["Ccap"] - m["Cbg"]) * d["chloride_deposition"] / (
            d["chloride_deposition"] + self.j0
        )
        h_eq = np.clip(self.hmax * q, self.hmin, self.hmax)
        return q, c_eq, h_eq

    def eq_salt_range(self, case: str) -> tuple[float, float]:
        _, c_eq, _ = self.drivers(case)
        return float(np.min(c_eq)), float(np.max(c_eq))

    def wet_mean_film_thickness(self, case: str) -> float:
        """Duration-weighted mean effective film thickness strictly over active wet intervals (q > 0)."""
        d = self.data[case]
        q, _, h_eq = self.drivers(case)
        mask = q > 0.0
        if not np.any(mask):
            return self.hmin
        return float(np.average(h_eq[mask], weights=d["duration_days"][mask]))

    def basis(self, case: str, c: dict, quadrature: int = 16) -> np.ndarray:
        key = (case, fingerprint(c), quadrature)
        if key in self.cache:
            return self.cache[key]
        d = self.data[case]
        q, c_eq, h_eq = self.drivers(case)
        current = np.exp(self.law.predict_log(d["T_C"], c_eq, h_eq))
        if not np.isfinite(current).all():
            raise ValueError("CURRENT_OVERFLOW_NO_CLIPPING_APPLIED")
        start, step = state.state_clock(q * current / self.reference, d["duration_days"])
        nodes, weights = np.polynomial.legendre.leggauss(quadrature)
        u = start[:, None] + step[:, None] * (0.5 + 0.5 * nodes)
        average = np.einsum("nqk,q->nk", state.state_functions(u, c), 0.5 * weights)
        base = self.mapping["Gamma"] * q * current * d["duration_days"] * 86400.0
        increments = base[:, None] * average
        result = np.column_stack(
            [
                np.bincount(
                    d["observation_index"].astype(int),
                    weights=increments[:, j],
                    minlength=len(d["months"]),
                ).cumsum()
                for j in range(c["k"])
            ]
        )
        self.cache[key] = result
        return result


def condition_on_initial_depth(raw, initial_depth_um: float) -> np.ndarray:
    """Apply Eq. (16) initial-depth anchoring: D_pred(t) = D_obs(t1) + D_raw(t) - D_raw(t1)."""
    raw_arr = np.asarray(raw, dtype=float)
    return float(initial_depth_um) + (raw_arr - raw_arr[0])


def metrics(y, p) -> dict:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    dy, dp = (y - y[0])[1:], (p - p[0])[1:]
    denominator = float(np.sum((dy - dy.mean()) ** 2))
    return {
        "increment_r2": float(1.0 - np.sum((dp - dy) ** 2) / denominator) if denominator > 0.0 else None,
        "actual_final_depth_um": float(y[-1]),
        "predicted_final_depth_um": float(p[-1]),
        "endpoint_error_pct": float(100.0 * (p[-1] / y[-1] - 1.0)),
        "increment_scored_points": len(dy),
    }
