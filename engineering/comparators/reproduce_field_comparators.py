# -*- coding: utf-8 -*-
"""Reference reproduction script for the eight 14-feature direct field machine-learning
and unsupervised domain-adaptation baseline models (Supplementary Section M1.5.2 & Table S5).

Note:
    Exact numerical reproduction of tree ensembles, LAPACK generalized eigensolvers (TCA),
    and stochastic neural-network training (DANN-MLP) depends on library/BLAS versions.
    See ``requirements-comparators.txt`` for the recorded runtime environment used to
    generate the frozen predictions in ``engineering/Table_conditioned_predictions.csv``.
    The lightweight zero-argument ``run_engineering_predictions.py`` runner loads and
    re-scores those frozen predictions without requiring scikit-learn, xgboost, or PyTorch.

Usage:
    pip install -r engineering/comparators/requirements-comparators.txt
    python engineering/comparators/reproduce_field_comparators.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ENG_DIR = HERE.parent
REPO_ROOT = ENG_DIR.parent

METHODS = (
    "Source_Ridge_poly2",
    "Source_RF",
    "Source_XGBoost",
    "CORAL_Ridge",
    "TCA_Ridge",
    "DomainInvariant_RF",
    "DomainInvariant_XGBoost",
    "DANN_MLP",
)
TARGET_SITES = ("jinhai", "pool", "ksc")


def standardize_pair(xs: np.ndarray, xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = xs.mean(axis=0, keepdims=True)
    sig = xs.std(axis=0, keepdims=True) + 1e-8
    return (xs - mu) / sig, (xt - mu) / sig


def coral_transform(xs: np.ndarray, xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs0, xt0 = standardize_pair(xs, xt)
    cs = np.cov(xs0, rowvar=False) + np.eye(xs0.shape[1]) * 1e-3
    ct = np.cov(xt0, rowvar=False) + np.eye(xt0.shape[1]) * 1e-3
    es, us = np.linalg.eigh(cs)
    et, ut = np.linalg.eigh(ct)
    cs_inv_sqrt = us @ np.diag(1.0 / np.sqrt(np.maximum(es, 1e-8))) @ us.T
    ct_sqrt = ut @ np.diag(np.sqrt(np.maximum(et, 1e-8))) @ ut.T
    return xs0 @ cs_inv_sqrt @ ct_sqrt, xt0


def tca_transform(xs: np.ndarray, xt: np.ndarray, dim: int = 4, mu: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    xs0, xt0 = standardize_pair(xs, xt)
    x = np.vstack([xs0, xt0])
    ns, nt = len(xs0), len(xt0)
    n = ns + nt
    e = np.vstack([np.ones((ns, 1)) / ns, -np.ones((nt, 1)) / nt])
    mmd = e @ e.T
    h = np.eye(n) - np.ones((n, n)) / n
    k = x @ x.T
    a = k @ mmd @ k + mu * np.eye(n)
    b = k @ h @ k + 1e-6 * np.eye(n)
    vals, vecs = np.linalg.eig(np.linalg.pinv(a) @ b)
    order = np.argsort(vals.real)[::-1][: min(dim, n - 2)]
    z = (vecs[:, order].real.T @ k).T
    z = StandardScaler().fit_transform(z)
    return z[:ns], z[ns:]


def dann_predict(xs: np.ndarray, ys: np.ndarray, xt: np.ndarray) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(0)
    torch.set_num_threads(1)
    xsa, xta = standardize_pair(xs, xt)
    y_mean, y_std = float(ys.mean()), float(ys.std()) + 1e-8
    ys0 = (ys - y_mean) / y_std
    xs_t = torch.tensor(xsa, dtype=torch.float32)
    xt_t = torch.tensor(xta, dtype=torch.float32)
    y_t = torch.tensor(ys0[:, None], dtype=torch.float32)
    xd = torch.cat([xs_t, xt_t], dim=0)
    yd = torch.cat([torch.zeros(len(xs_t), 1), torch.ones(len(xt_t), 1)], dim=0)

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lambd):
            ctx.lambd = lambd
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambd * grad_output, None

    class Net(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
            self.reg = nn.Linear(8, 1)
            self.dom = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))

        def forward(self, x, lambd: float = 0.0):
            f = self.feat(x)
            return self.reg(f), self.dom(GradReverse.apply(f, lambd))

    net = Net(xs.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    for epoch in range(700):
        lambd = min(1.0, epoch / 250.0) * 0.2
        opt.zero_grad()
        pred_y, _ = net(xs_t, lambd=0.0)
        _, pred_d = net(xd, lambd=lambd)
        loss = mse(pred_y, y_t) + 0.1 * bce(pred_d, yd)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred, _ = net(xt_t, lambd=0.0)
    return pred.numpy().ravel() * y_std + y_mean


def fit_predict(method: str, xs: np.ndarray, ys: np.ndarray, xt: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    if method == "Source_Ridge_poly2":
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))
        return model.fit(xs, ys).predict(xt)
    if method == "Source_RF":
        model = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=1, n_jobs=1, random_state=0)
        return model.fit(xs, ys).predict(xt)
    if method == "Source_XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=1, random_state=0,
        )
        return model.fit(xs, ys).predict(xt)
    if method == "CORAL_Ridge":
        xsa, xta = coral_transform(xs, xt)
        model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))
        return model.fit(xsa, ys).predict(xta)
    if method == "TCA_Ridge":
        xsa, xta = tca_transform(xs, xt)
        return Ridge(alpha=1.0).fit(xsa, ys).predict(xta)
    if method == "DomainInvariant_RF":
        xsa, xta = StandardScaler().fit_transform(xs), StandardScaler().fit_transform(xt)
        model = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=1, n_jobs=1, random_state=0)
        return model.fit(xsa, ys).predict(xta)
    if method == "DomainInvariant_XGBoost":
        from xgboost import XGBRegressor
        xsa, xta = StandardScaler().fit_transform(xs), StandardScaler().fit_transform(xt)
        model = XGBRegressor(
            n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=1, random_state=0,
        )
        return model.fit(xsa, ys).predict(xta)
    if method == "DANN_MLP":
        return dann_predict(xs, ys, xt)
    raise ValueError(f"Unknown comparator: {method}")


def main(out_json: Path | None = None):
    feat_bundle = json.loads((HERE / "features_14d.json").read_text(encoding="utf-8"))
    feats = {c: np.asarray(v, float) for c, v in feat_bundle["sites"].items()}
    labels = {
        c: np.asarray(json.loads((ENG_DIR / f"inputs/{c}_labels.json").read_text(encoding="utf-8"))["observed_depth_um"], float)
        for c in ("linhai", *TARGET_SITES)
    }

    xs, ys = feats["linhai"], labels["linhai"]
    results = {}
    for method in METHODS:
        results[method] = {}
        for site in TARGET_SITES:
            xt, yt = feats[site], labels[site]
            raw = np.asarray(fit_predict(method, xs, ys, xt), float)
            cond = yt[0] + (raw - raw[0])
            rmse = float(np.sqrt(np.mean((cond[1:] - yt[1:]) ** 2)))
            ep = float((cond[-1] - yt[-1]) / yt[-1] * 100.0)
            results[method][site] = {
                "rmse_um": round(rmse, 2),
                "endpoint_error_pct": round(ep, 2),
                "conditioned_depth_um": cond.tolist(),
            }
        print(
            f"{method:<25s} | Jinhai: RMSE={results[method]['jinhai']['rmse_um']:9.2f}, EP={results[method]['jinhai']['endpoint_error_pct']:+9.2f}% "
            f"| Pool: RMSE={results[method]['pool']['rmse_um']:9.2f}, EP={results[method]['pool']['endpoint_error_pct']:+9.2f}% "
            f"| KSC: RMSE={results[method]['ksc']['rmse_um']:11.2f}, EP={results[method]['ksc']['endpoint_error_pct']:+11.2f}%"
        )

    if out_json is not None:
        out_json = Path(out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON path to write reproduced comparator trajectories")
    args = parser.parse_args()
    main(args.out)
