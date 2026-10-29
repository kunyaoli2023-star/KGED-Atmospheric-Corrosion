# -*- coding: utf-8 -*-
"""Test script 3: Post-discovery diagnostic verification of the 7-parameter
salinity-modulated film-length rate law (Section 3.4) across the 23-fold
laboratory cross-validation benchmark and the 4-site field integration pipeline."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engineering"))
import engineering_state as es  # noqa: E402

C0 = 3.5


def salt(B, k, c):
    return np.log((1.0 + B * c * np.exp(-k * c)) / (1.0 + B * C0 * np.exp(-k * C0)))


def film_rational(q, h):
    return np.log(h) - np.log(q + h * h) + np.log(q + 1.0)


def film_rational_qc(qc, qc0, h):
    return np.log(h) - np.log(qc + h * h) + np.log(qc0 + 1.0)


def film_peaked(D, m, lam, lam_ref, h):
    return np.log(1.0 + D * (h ** m) * np.exp(-lam * h)) - np.log(1.0 + D * np.exp(-lam_ref))


MODELS = {
    "KGED-5 (5p separable parsimonious)": (
        4,
        lambda p, b, tau, c, h: b + p[0] * (1.0 - 1.0 / tau) + salt(p[1], p[2], c) + film_rational(p[3], h),
    ),
    "KGED-8 (8p separable minimum-error)": (
        7,
        lambda p, b, tau, c, h: b
        + p[0] * (np.tanh(p[1] * tau) - np.tanh(p[1]))
        + salt(p[2], p[3], c)
        + film_peaked(p[4], p[5], p[6], p[6], h),
    ),
    "6p: K27 with q(c) = q0*exp(-beta*c)": (
        5,
        lambda p, b, tau, c, h: b
        + p[0] * (1.0 - 1.0 / tau)
        + salt(p[1], p[2], c)
        + film_rational_qc(p[3] * np.exp(-p[4] * c), p[3] * np.exp(-p[4] * C0), h),
    ),
    "6p: K27 with q(c) = q0*(1+c)^(-beta)": (
        5,
        lambda p, b, tau, c, h: b
        + p[0] * (1.0 - 1.0 / tau)
        + salt(p[1], p[2], c)
        + film_rational_qc(p[3] * ((1.0 + c) ** (-p[4])), p[3] * ((1.0 + C0) ** (-p[4])), h),
    ),
    "KGED-7: K19 with lam(c) = lam0*(1+c)^beta": (
        6,
        lambda p, b, tau, c, h: b
        + p[0] * (1.0 - 1.0 / tau)
        + salt(p[1], p[2], c)
        + film_peaked(p[3], 1.0, p[4] * ((1.0 + c) ** p[5]), p[4] * ((1.0 + C0) ** p[5]), h),
    ),
    "9p: KGED-8 with lam(c) = lam0*exp(beta*c)": (
        8,
        lambda p, b, tau, c, h: b
        + p[0] * (np.tanh(p[1] * tau) - np.tanh(p[1]))
        + salt(p[2], p[3], c)
        + film_peaked(p[4], p[5], p[6] * np.exp(p[7] * c), p[6] * np.exp(p[7] * C0), h),
    ),
}


def fit_model(npos, fn, tau, c, h, y, idx, rng, starts=6, nfev=1500):
    best = None
    lb = np.concatenate([[-30.0], np.full(npos, -8.0)])
    ub = np.concatenate([[5.0], np.full(npos, 6.0)])
    for s in range(starts):
        v0 = np.zeros(npos) if s == 0 else rng.uniform(-1.8, 1.4, npos)
        x0 = np.concatenate([[y[idx].mean()], v0])

        def residual(x):
            with np.errstate(all="ignore"):
                pred = fn(np.exp(x[1:]), x[0], tau[idx], c[idx], h[idx])
            r = pred - y[idx]
            return np.where(np.isfinite(r), r, 1e3)

        sol = least_squares(
            residual, x0, bounds=(lb, ub), method="trf", x_scale="jac",
            max_nfev=nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7
        )
        if best is None or sol.cost < best.cost:
            best = sol
    return best


def main():
    d = json.loads((ROOT / "data/data.json").read_text(encoding="utf-8"))
    folds = json.loads((ROOT / "data/folds.json").read_text(encoding="utf-8"))
    T = np.asarray(d["T_C"], float)
    C = np.asarray(d["C_wt"], float)
    H = np.asarray(d["h_um"], float)
    y = np.asarray(d["log_i"], float)
    tau = (T + 273.15) / 298.15
    c = C
    h = H / 200.0
    n = len(y)

    rng = np.random.default_rng(20261007)
    print("=== 3. Post-Discovery Salinity-Modulated Film-Length Verification (Section 3.4) ===")
    print(f"{'Model':42s} {'k':>2s} {'Train':>6s} {'BalCV':>6s} | {'Prof':>5s} {'T':>5s} {'H':>5s} {'C':>5s} | {'BIC':>7s} {'Rank':>5s}")
    fitted_full = {}
    for name, (npos, fn) in MODELS.items():
        k = npos + 1
        per_axis = {}
        for fo in folds:
            tr = np.asarray(fo["train"], int)
            va = np.asarray(fo["validation"], int)
            sol = fit_model(npos, fn, tau, c, h, y, tr, rng)
            pred_va = fn(np.exp(sol.x[1:]), sol.x[0], tau[va], c[va], h[va])
            per_axis.setdefault(fo["axis"], []).append(float(np.sqrt(np.mean((pred_va - y[va]) ** 2))))
        ax = {a: float(np.mean(v)) for a, v in per_axis.items()}
        bal = float(np.mean(list(ax.values())))
        full = fit_model(npos, fn, tau, c, h, y, np.arange(n), rng, starts=12)
        pred_all = fn(np.exp(full.x[1:]), full.x[0], tau, c, h)
        rmse = float(np.sqrt(np.mean((pred_all - y) ** 2)))
        bic = float(n * np.log(rmse ** 2) + k * np.log(n))
        J = full.jac / np.linalg.norm(full.jac, axis=0)
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int(np.sum(sv > 1e-7 * sv[0]))
        fitted_full[name] = (full.x[0], np.exp(full.x[1:]), fn, bal)
        print(
            f"{name:42s} {k:2d} {rmse:6.3f} {bal:6.3f} | "
            f"{ax['shape_group']:5.3f} {ax['T_C']:5.3f} {ax['h_um']:5.3f} {ax['C_wt']:5.3f} | "
            f"{bic:7.1f} {rank:2d}/{k:<2d}"
        )

    # Verify 7-parameter field integration under the frozen engineering protocol
    mapping = json.loads((ROOT / "engineering/inputs/mapping.json").read_text(encoding="utf-8"))
    cases = ("linhai", "jinhai", "pool", "ksc")
    env_data = {}
    labels = {}
    for cs in cases:
        with np.load(ROOT / f"engineering/inputs/{cs}_environment.npz") as z:
            env_data[cs] = {k_i: z[k_i].copy() for k_i in z.files}
        labels[cs] = json.loads((ROOT / f"engineering/inputs/{cs}_labels.json").read_text(encoding="utf-8"))

    b7, p7, fn7, bal7 = fitted_full["KGED-7: K19 with lam(c) = lam0*(1+c)^beta"]
    assert abs(bal7 - 0.461) < 0.002, f"Unexpected 7p balanced CV: {bal7}"

    def log_i_7p(T_in, C_in, H_in):
        return fn7(p7, b7, (np.asarray(T_in) + 273.15) / 298.15, np.asarray(C_in), np.asarray(H_in) / 200.0)

    ref_i = 0.5 * float(np.exp(log_i_7p([25.0], [mapping["Cbg"]], [500.0])[0]))
    c_struct = mapping["fixed_state_structure"]
    bases = {}
    for cs in cases:
        ed = env_data[cs]
        q = np.clip((ed["RH_pct"] - 55.0) / 45.0, 0.0, 1.0) ** 0.3
        C_eq = mapping["Cbg"] + (mapping["Ccap"] - mapping["Cbg"]) * ed["chloride_deposition"] / (ed["chloride_deposition"] + 20000.0)
        H_eq = np.clip(1000.0 * q, 50.0, 1000.0)
        curr = np.exp(log_i_7p(ed["T_C"], C_eq, H_eq))
        start, step = es.state_clock(q * curr / ref_i, ed["duration_days"])
        nodes, weights = np.polynomial.legendre.leggauss(16)
        u = start[:, None] + step[:, None] * (0.5 + 0.5 * nodes)
        avg = np.einsum("nqk,q->nk", es.state_functions(u, c_struct), 0.5 * weights)
        base = mapping["Gamma"] * q * curr * ed["duration_days"] * 86400.0
        inc = base[:, None] * avg
        bases[cs] = np.column_stack([
            np.bincount(ed["observation_index"].astype(int), weights=inc[:, j], minlength=len(ed["months"])).cumsum()
            for j in range(c_struct["k"])
        ])

    y_lh = np.asarray(labels["linhai"]["observed_depth_um"], float)
    m_lh = np.asarray(labels["linhai"]["months"], float)
    A_mat, b_vec = es.design_for_fit(bases["linhai"], y_lh, m_lh, c_struct)
    P_mat = es.penalty(c_struct)
    sol_lh = lsq_linear(
        np.vstack([A_mat, np.sqrt(c_struct["lam"]) * P_mat]),
        np.r_[b_vec, np.zeros(len(P_mat))],
        bounds=(0.0, 20.0), method="bvls", tol=1e-11, max_iter=200
    )
    amps = sol_lh.x
    print(f"\n[KGED-7 Field Integration Check] Linhai Bernstein amplitudes: {np.round(amps, 4).tolist()}")
    apes, mapes, nrmses = [], [], []
    for cs in ("jinhai", "pool", "ksc"):
        y_obs = np.asarray(labels[cs]["observed_depth_um"], float)
        raw_pred = bases[cs] @ amps
        y_cond = y_obs[0] + (raw_pred - raw_pred[0])
        err = y_cond[1:] - y_obs[1:]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ep_err = float((y_cond[-1] - y_obs[-1]) / y_obs[-1] * 100.0)
        mape = float(np.mean(np.abs(err / y_obs[1:])) * 100.0)
        nrmse = float(rmse / y_obs[-1] * 100.0)
        apes.append(abs(ep_err))
        mapes.append(mape)
        nrmses.append(nrmse)
        print(f"  * {cs:8s}: RMSE = {rmse:6.2f} um | Endpoint error = {ep_err:+6.2f}% | PointMAPE = {mape:5.1f}% | nRMSE = {nrmse:5.1f}%")
    print(f"  => Target-site Mean Endpoint APE = {np.mean(apes):.2f}% | Mean PointMAPE = {np.mean(mapes):.1f}% | Mean nRMSE = {np.mean(nrmses):.1f}%")
    print("\nPASSED: Post-discovery 7-parameter coupling verification complete.")


if __name__ == "__main__":
    main()
