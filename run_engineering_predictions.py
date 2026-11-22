# -*- coding: utf-8 -*-
"""One-click Stage V cross-environment engineering corrosion-depth prediction pipeline.

Usage (zero required arguments):
    python run_engineering_predictions.py
    python run_engineering_predictions.py --out-dir engineering/results_reproduced

Recomputes live from scratch the 7 equation-based engineering models (retained 5- and
8-parameter KGED laws, Sparse regression, gplearn, PySR, PhySO, and empirical power-law model),
re-scores the 11 frozen trained ML / domain-adaptation baselines under Eq. (16) initial-depth
anchoring (verifying 18/18 exact Fig. 10a-b benchmark agreement with the manuscript),
recomputes all 6 single-factor integrator ablations live from scratch (Fig. 10a, c-e),
evaluates the 12 component-wise rate-law term-omission ablations (Fig. 11b), and renders
Main Figs. 9, 10 & 11 and Supplementary Figs. S3-S5 at 300 DPI.
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 9):
    sys.exit(f"KGED Stage V requires Python >= 3.9 (found {sys.version.split()[0]}).")

import argparse
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engineering.engineering_integrator import (
    CASES,
    DEFAULT_MAPPING,
    H_FIELD_REF,
    Q_WET_REF,
    SITE_LABELS,
    T_REF,
    Integrator,
)
from engineering import engineering_state as state
from engineering.verify_closed_forms import CLOSED_FORM_ATOL, QUADRATURE_REL_TOL, ClosedFormLaw, require

TARGET_SITES = ("jinhai", "pool", "ksc")
SITE_DISPLAY = {
    "linhai": "Weihai shoreline (50 m, source)",
    "jinhai": "Weihai near-coastal (1000 m)",
    "pool": "Indoor natatorium",
    "ksc": "NASA KSC",
}

# Divergence threshold for field-only / domain-adaptation benchmarks: RMSE > 200 um or |EP| > 100%
DIVERGENCE_RULE = {
    "rmse_um_max": 200.0,
    "abs_endpoint_pct_max": 100.0,
}

LIVE_METHODS = {"kged5", "kged8", "sparse", "gp", "pysr", "physo", "power_time"}

BENCHMARK_GROUPS = [
    (
        "Time-integrated laboratory baselines",
        [
            ("bp", "BP"),
            ("rf", "RF"),
            ("xgb", "XGBoost"),
            ("sparse", "Sparse regression"),
            ("gp", "gplearn"),
            ("pysr", "PySR"),
            ("physo", "PhySO"),
        ],
    ),
    (
        "Empirical power-law and direct field machine-learning models",
        [
            ("power_time", "Empirical power-law model"),
            ("Source_Ridge_poly2", "Quadratic Ridge"),
            ("Source_RF", "Source RF"),
            ("Source_XGBoost", "Source XGBoost"),
        ],
    ),
    (
        "Unsupervised domain-adaptation models",
        [
            ("TCA_Ridge", "TCA–Ridge"),
            ("CORAL_Ridge", "CORAL–Ridge"),
            ("DomainInvariant_RF", "Domain-standardised RF"),
            ("DomainInvariant_XGBoost", "Domain-standardised XGBoost"),
        ]
        + [("DANN_MLP", "DANN–MLP")],
    ),
    (
        "KGED + integration",
        [
            ("kged5", "Five-parameter KGED"),
            ("kged8", "Eight-parameter KGED"),
        ],
    ),
]

# Published Main Manuscript Fig. 10a-b & Section 3.3.2 benchmark values (None = "Diverged")
MANUSCRIPT_BENCHMARKS = {
    "bp": ("22.65", "+9.68", "70.30", "+40.58", "5.34", "-2.97", "17.74"),
    "rf": ("18.30", "+8.09", "45.45", "-7.91", "9.80", "-15.72", "10.57"),
    "xgb": ("20.91", "+10.48", "54.79", "-13.39", "6.70", "-9.86", "11.25"),
    "sparse": ("14.44", "+0.50", "32.40", "-0.14", "5.53", "-5.06", "1.90"),
    "gp": ("22.95", "+9.38", "43.46", "+25.92", "7.69", "-3.91", "13.07"),
    "pysr": ("22.69", "+11.46", "32.98", "-0.44", "5.94", "-6.06", "5.99"),
    "physo": ("24.45", "+9.94", "29.39", "+0.65", "7.75", "-1.14", "3.91"),
    "power_time": ("12.55", "+3.06", "114.62", "-43.30", "23.37", "-37.89", "28.08"),
    "Source_Ridge_poly2": (None, None, None, None, None, None, None),
    "Source_RF": ("35.92", "-36.12", "193.55", "-79.12", "42.13", "-62.88", "59.37"),
    "Source_XGBoost": ("13.68", "+3.32", "132.71", "-43.16", "38.68", "-62.56", "36.35"),
    "TCA_Ridge": ("93.03", "-58.52", "177.61", "-67.37", None, None, None),
    "CORAL_Ridge": (None, None, None, "+18.08", None, None, None),
    "DomainInvariant_RF": ("25.37", "-19.62", "162.90", "-66.15", "55.91", "-20.70", "35.49"),
    "DomainInvariant_XGBoost": ("13.68", "+3.32", "132.71", "-43.16", "16.24", "+21.47", "22.65"),
    "DANN_MLP": (None, None, "195.16", "-82.38", None, None, None),
    "kged5": ("14.03", "-0.10", "32.38", "-0.49", "4.96", "-3.28", "1.29"),
    "kged8": ("18.36", "+4.41", "32.35", "+18.69", "4.87", "+2.32", "8.47"),
}

# Published Main Manuscript Section 3.3.2 & Fig. 10a, c-e single-factor integrator ablations
MANUSCRIPT_SINGLE_FACTOR_ABLATIONS = {
    "elapsed_time": ("14.73", "+0.64", "113.41", "-44.59", "17.96", "-30.28", "25.17"),
    "const_icorr": ("22.62", "+11.01", "29.61", "+2.27", "6.00", "-6.41", "6.56"),
    "no_rust_decay": ("67.25", "+53.77", "105.57", "+58.60", "18.69", "+32.03", "48.14"),
    "no_field_scaling": ("86.27", "-83.15", "196.28", "-83.01", "43.64", "-69.52", "78.56"),
    "no_wetness": ("10.35", "-4.51", "131.85", "-52.68", "23.34", "-37.89", "31.69"),
    "binary_iso9223": ("97.65", "+80.47", "451.10", "+225.35", "23.93", "+53.78", "119.86"),
}


class SymbolicBaselineLaw:
    """Closed-form laboratory symbolic-regression baselines (Table 4 & Table S1).

    Selected outer seeds:
        sparse: deterministic (no seed) | gp (gplearn): seed 2718 | pysr: seed 31415 | physo: seed 2718
    All four expressions were fitted on centred log-current with constant reference offset.
    """

    SEEDS = {"sparse": None, "gp": 2718, "pysr": 31415, "physo": 2718}
    DEFAULT_OFFSET = -11.683134940903741

    def __init__(self, kind: str, offset: float = DEFAULT_OFFSET):
        self.kind = kind
        self.offset = float(offset)

    def predict_log(self, T, C, H) -> np.ndarray:
        T, C, H = np.broadcast_arrays(
            np.atleast_1d(np.asarray(T, float)),
            np.atleast_1d(np.asarray(C, float)),
            np.atleast_1d(np.asarray(H, float)),
        )
        tau = (T + 273.15) / 298.15
        z = C / 3.5
        u = H / 200.0
        if self.kind == "sparse":
            val = (
                -8.70090958951834
                + 8.633653796611851 * tau
                + 0.6505553004514903 * z
                - 0.1029282080692102 * (z ** 2)
                - 0.025333504417577844 * (u ** 2)
                - 0.01132940439751234 * z * u
            )
        elif self.kind == "gp":
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                num = 0.853617397433095 + np.log(tau) / 0.2361454210346554
                den = np.sqrt(np.abs(z)) * np.log(tau * 0.6229490192977711)
                ratio = np.where(np.abs(num) > 1e-3, den / num, 1.0)
                inv_r = np.where(np.abs(ratio) > 1e-3, 1.0 / ratio, 0.0)
                val = np.where(np.abs(inv_r) > 1e-3, np.log(np.abs(inv_r)), 0.0)
        elif self.kind == "pysr":
            val = (
                2.379962492898294
                / (
                    ((-1.4647022956172997 - z ** 2) / ((2.3037025134368534 / (z + 1.3202349807865026)) - 1.7355089125357204))
                    / 2.9720278143708603
                )
            ) - tau
        elif self.kind == "physo":
            c0 = 0.02024552121368723
            val = c0 * (2.0 * tau - (c0 - u + z ** 2))
        else:
            raise ValueError(f"Unknown symbolic baseline: {self.kind}")
        return self.offset + val


class ConstantCurrentLaw:
    """Constant log-current wrapper for rate-law / elapsed-time single-factor ablations."""

    def __init__(self, log_current_A_cm2: float):
        self.log_current_A_cm2 = float(log_current_A_cm2)

    def predict_log(self, T_C, C_wt, h_um) -> np.ndarray:
        T_arr = np.atleast_1d(np.asarray(T_C, float))
        return np.full_like(T_arr, self.log_current_A_cm2)


class SingleFactorWetnessIntegrator(Integrator):
    """Integrator variant for elapsed-time, q_wet=1, and binary ISO 9223 wetness ablations."""

    def __init__(self, data: dict, law, mapping: dict, mode: str, override_ref: float | None = None):
        super().__init__(data, law, mapping)
        self.mode = mode
        if override_ref is not None:
            self.reference = float(override_ref)

    def drivers(self, case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.data[case]
        q_cont, c_eq, h_eq = super().drivers(case)
        if self.mode == "elapsed_time":
            return np.ones_like(q_cont), np.full_like(c_eq, self.mapping["Cbg"]), np.full_like(h_eq, H_FIELD_REF)
        if self.mode == "no_wetness":
            return np.ones_like(q_cont), c_eq, h_eq
        if self.mode == "binary_iso9223":
            q_iso = ((d["RH_pct"] > 80.0) & (d["T_C"] > 0.0)).astype(float)
            return q_iso, c_eq, h_eq
        return q_cont, c_eq, h_eq


class AblatedLaw:
    """Component-wise hypothesis ablation wrapper holding omitted coordinates at field reference values."""

    def __init__(self, base_law, c_bg: float, keep_H1: bool = True, keep_H2: bool = True, keep_H3: bool = True):
        self.base_law = base_law
        self.c_bg = float(c_bg)
        self.keep_H1 = keep_H1
        self.keep_H2 = keep_H2
        self.keep_H3 = keep_H3

    def predict_log(self, T_C, C_wt, h_um) -> np.ndarray:
        T_arr = np.asarray(T_C, float)
        C_arr = np.asarray(C_wt, float)
        H_arr = np.asarray(h_um, float)
        T_in = T_arr if self.keep_H1 else np.full_like(T_arr, T_REF)
        C_in = C_arr if self.keep_H2 else np.full_like(C_arr, self.c_bg)
        H_in = H_arr if self.keep_H3 else np.full_like(H_arr, H_FIELD_REF)
        return self.base_law.predict_log(T_in, C_in, H_in)


def validate_mapping(mapping: dict):
    st = mapping.get("fixed_state_structure", {})
    checks = {
        "state family == bernstein": st.get("family") == "bernstein",
        "degree == 3 and k == 4": st.get("degree") == 3 and st.get("k") == 4,
        "tau_u == 0.25": abs(float(st.get("tau", -1.0)) - 0.25) < 1e-12,
        "lambda_a == 0.03": abs(float(st.get("lam", -1.0)) - 0.03) < 1e-12,
        "Cbg ~ 0.1139 wt%": abs(float(mapping.get("Cbg", -1.0)) - 0.1139197357) < 1e-6,
        "Ccap ~ 0.1644 wt%": abs(float(mapping.get("Ccap", -1.0)) - 0.1644) < 1e-4,
        "Gamma ~ 0.3687 um/(A s cm-2)": abs(float(mapping.get("Gamma", -1.0)) - 0.3686577318) < 1e-6,
    }
    bad = [k for k, ok in checks.items() if not ok]
    require(not bad, "mapping.json inconsistent with Section 2.6 & Supplementary Methods M1.4: " + "; ".join(bad))


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def evaluate_conditioned_series(months: np.ndarray, y_obs: np.ndarray, p_cond: np.ndarray) -> dict:
    p_cond = np.asarray(p_cond, float)
    finite = bool(np.all(np.isfinite(p_cond)))
    if finite:
        err = p_cond[1:] - y_obs[1:]
        sst = float(np.sum((y_obs[1:] - np.mean(y_obs[1:])) ** 2))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        r2 = float(1.0 - np.sum(err ** 2) / sst) if sst > 0.0 else None
        ep_pct = float((p_cond[-1] - y_obs[-1]) / y_obs[-1] * 100.0)
        neg_pts = int(np.sum(p_cond < 0.0))
        dec_steps = int(np.sum(np.diff(p_cond) < -1e-9))
    else:
        rmse = mae = ep_pct = float("inf")
        r2 = None
        neg_pts = dec_steps = -1

    rmse_div = (not finite) or (rmse > DIVERGENCE_RULE["rmse_um_max"])
    ep_div = (not finite) or (abs(ep_pct) > DIVERGENCE_RULE["abs_endpoint_pct_max"])
    return {
        "months": months,
        "observed_depth_um": y_obs,
        "conditioned_depth_um": p_cond,
        "rmse_um": rmse,
        "mae_um": mae,
        "r2": r2,
        "endpoint_error_pct": ep_pct,
        "negative_points": neg_pts,
        "decreasing_steps": dec_steps,
        "rmse_diverged": rmse_div,
        "ep_diverged": ep_div,
        "diverged": rmse_div or ep_div,
        "physically_inadmissible": (neg_pts != 0) or (dec_steps != 0),
    }


def fmt_cell(d_site: dict, is_ep: bool) -> str:
    if is_ep:
        return "Diverged" if d_site["ep_diverged"] else f"{d_site['endpoint_error_pct']:+.2f}"
    return "Diverged" if d_site["rmse_diverged"] else f"{d_site['rmse_um']:.2f}"


def _plot_supplementary_group(
    out_dir: Path,
    all_series: dict[str, dict[str, dict]],
    members: list[tuple[str, str]],
    fname: str,
    use_symlog: bool,
    ncol: int,
    plt,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.05), dpi=300)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.93, bottom=0.19, hspace=0.38, wspace=0.29)
    cmap = plt.get_cmap("tab10")
    for idx, (ax, case) in enumerate(zip(axes.flat, CASES)):
        d5 = all_series["kged5"][case]
        m_arr, y_arr = d5["months"], d5["observed_depth_um"]
        ax.plot(m_arr[1:], y_arr[1:], "ko", ms=4.0, label="Observed", zorder=5)
        for j, (m_key, m_lbl) in enumerate(members):
            p = np.asarray(all_series[m_key][case]["conditioned_depth_um"], float)
            p = np.where(np.isfinite(p), p, np.nan)
            ax.plot(m_arr, p, lw=1.25, alpha=0.85, color=cmap(j), label=m_lbl)
        ax.plot(m_arr, d5["conditioned_depth_um"], color="#2474B5", ls="-", lw=2.15, label="Five-parameter KGED", zorder=4)
        ax.plot(
            m_arr,
            all_series["kged8"][case]["conditioned_depth_um"],
            color="#BC4B32",
            ls="--",
            lw=2.15,
            label="Eight-parameter KGED",
            zorder=4,
        )
        ax.plot(
            m_arr[0],
            y_arr[0],
            marker="D",
            mfc="white",
            mec="black",
            ms=5.2,
            ls="none",
            label="Initial depth $D_{\\mathrm{obs}}(t_1)$",
            zorder=6,
        )
        if use_symlog:
            ax.set_yscale("symlog", linthresh=10.0)
            ax.axhline(0.0, color="#888888", lw=0.6, zorder=1)
            ylo, yhi = ax.get_ylim()
            max_neg = int(np.floor(np.log10(max(10.0, -ylo)))) if ylo < -10.0 else 0
            max_pos = int(np.floor(np.log10(max(10.0, yhi)))) if yhi > 10.0 else 0
            total_decades = max_neg + max_pos
            if total_decades > 7:
                step = 2 if total_decades <= 12 else 3
                start_exp = 1 if step == 2 else 2
                neg_ticks = [-(10.0 ** e) for e in range(start_exp, max_neg + 1, step)]
                pos_ticks = [10.0 ** e for e in range(start_exp, max_pos + 1, step)]
                ax.set_yticks(sorted(neg_ticks + [0.0] + pos_ticks))
        ax.set_title(f"({chr(97 + idx)}) {SITE_DISPLAY[case]}", loc="left", fontsize=10.2, fontweight="bold")
        ax.set_xlabel("Exposure time (months)", fontsize=9.2)
        ax.set_ylabel("Corrosion depth (μm)" + (" [symlog]" if use_symlog else ""), fontsize=9.2)
        ax.grid(alpha=0.22, lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    h_leg, l_leg = axes.flat[0].get_legend_handles_labels()
    fig.legend(h_leg, l_leg, loc="lower center", bbox_to_anchor=(0.52, 0.015), ncol=ncol, frameon=False, fontsize=8.0)
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    return out_path


def _render_fig10_baselines_and_ablations(
    out_dir: Path,
    all_series: dict[str, dict[str, dict]],
    sf_map: dict[str, dict],
    i_ref_uA_cm2: float,
    plt,
    GridSpec,
) -> Path:
    """Render 5-panel Main Fig. 10 live from computed benchmark and single-factor ablation metrics."""

    def mean_ep_of(m_key: str) -> float:
        return float(np.mean([abs(all_series[m_key][c]["endpoint_error_pct"]) for c in TARGET_SITES]))

    def is_diverged(m_key: str) -> bool:
        return any(all_series[m_key][c]["diverged"] for c in TARGET_SITES)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.titlesize": 10.6,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig = plt.figure(figsize=(11.8, 7.2), dpi=300)
        outer = GridSpec(2, 1, figure=fig, height_ratios=[1.22, 0.92], hspace=0.38)
        top_gs = outer[0].subgridspec(1, 2, wspace=0.52)
        bot_gs = outer[1].subgridspec(1, 3, wspace=0.76)

        # Panel (a): Field-only, time-only, and domain-adaptation baselines vs. KGED
        ax_a = fig.add_subplot(top_gs[0, 0])
        spec_a = [
            ("5-param KGED", mean_ep_of("kged5"), False, "#0072B2"),
            ("Domain-std XGBoost", mean_ep_of("DomainInvariant_XGBoost"), is_diverged("DomainInvariant_XGBoost"), "#56B4E9"),
            ("Elapsed-time + $g(u)$", sf_map["elapsed_time"]["mean_abs_endpoint_error_pct"], False, "#999999"),
            ("Empirical power-law", mean_ep_of("power_time"), is_diverged("power_time"), "#999999"),
            ("Domain-std RF", mean_ep_of("DomainInvariant_RF"), is_diverged("DomainInvariant_RF"), "#56B4E9"),
            ("Source XGBoost", mean_ep_of("Source_XGBoost"), is_diverged("Source_XGBoost"), "#E69F00"),
            ("Source RF", mean_ep_of("Source_RF"), is_diverged("Source_RF"), "#E69F00"),
            ("TCA–Ridge", mean_ep_of("TCA_Ridge"), is_diverged("TCA_Ridge"), "#56B4E9"),
            ("CORAL–Ridge", mean_ep_of("CORAL_Ridge"), is_diverged("CORAL_Ridge"), "#56B4E9"),
            ("Quadratic Ridge", mean_ep_of("Source_Ridge_poly2"), is_diverged("Source_Ridge_poly2"), "#E69F00"),
            ("DANN–MLP", mean_ep_of("DANN_MLP"), is_diverged("DANN_MLP"), "#56B4E9"),
        ]
        labels_a = [s[0] for s in spec_a]
        vals_a = [105.0 if s[2] else s[1] for s in spec_a]
        capped_a = [s[2] for s in spec_a]
        colors_a = [s[3] for s in spec_a]

        y_a = np.arange(len(labels_a))
        ax_a.grid(axis="x", color="#E5E5E5", linewidth=0.6, zorder=0)
        bars_a = ax_a.barh(y_a, vals_a, height=0.64, color=colors_a, alpha=0.92, zorder=3)
        for b, cap in zip(bars_a, capped_a):
            if cap:
                b.set_hatch("//")
                b.set_edgecolor("#333333")
                b.set_linewidth(0.6)
        for i, (v, cap) in enumerate(zip(vals_a, capped_a)):
            txt = ">100%" if cap else f"{v:.2f}%"
            fw = "bold" if (i == 0 or cap) else "normal"
            col = "#0072B2" if i == 0 else "#222222"
            ax_a.text(v + 2.2, i, txt, va="center", ha="left", fontsize=8.2, fontweight=fw, color=col)

        ax_a.axvline(100, color="#666666", linestyle="--", linewidth=0.8, alpha=0.75, zorder=2)
        ax_a.set_yticks(y_a)
        ax_a.set_yticklabels(labels_a)
        ax_a.invert_yaxis()
        ax_a.set_xlim(0, 134)
        ax_a.set_xlabel("Three-site mean abs. endpoint error (%)")
        ax_a.set_title("(a) Field-only & time-only baselines", loc="left", fontweight="bold", pad=8)

        # Panel (b): Integrated laboratory rate laws under shared q_wet & g(u)
        ax_b = fig.add_subplot(top_gs[0, 1])
        spec_b = [
            ("5-param KGED", "kged5", "#0072B2"),
            ("Sparse regression", "sparse", "#009E73"),
            ("PhySO", "physo", "#009E73"),
            ("PySR", "pysr", "#009E73"),
            ("8-param KGED", "kged8", "#CC79A7"),
            ("RF", "rf", "#D55E00"),
            ("XGBoost", "xgb", "#D55E00"),
            ("gplearn", "gp", "#009E73"),
            ("BP", "bp", "#D55E00"),
        ]
        labels_b = [s[0] for s in spec_b]
        err_b = [mean_ep_of(s[1]) for s in spec_b]
        rmse_nc_b = [all_series[s[1]]["jinhai"]["rmse_um"] for s in spec_b]
        colors_b = [s[2] for s in spec_b]

        y_b = np.arange(len(labels_b))
        ax_b.grid(axis="x", color="#E5E5E5", linewidth=0.6, zorder=0)
        ax_b.barh(y_b, err_b, height=0.62, color=colors_b, alpha=0.92, zorder=3)
        for i, (e, r) in enumerate(zip(err_b, rmse_nc_b)):
            fw = "bold" if i == 0 else "normal"
            col = "#0072B2" if i == 0 else "#222222"
            ax_b.text(e + 0.5, i, f"{e:.2f}% ({r:.2f} μm)", va="center", ha="left", fontsize=8.2, color=col, fontweight=fw)

        ax_b.set_yticks(y_b)
        ax_b.set_yticklabels(labels_b)
        ax_b.invert_yaxis()
        ax_b.set_xlim(0, 31.5)
        ax_b.set_xlabel("Mean abs. endpoint error (%) [near-coastal RMSE]")
        ax_b.set_title("(b) Integrated laboratory rate laws", loc="left", fontweight="bold", pad=8)

        site_colors = ("#0072B2", "#D55E00", "#009E73")
        site_names = ("Weihai near-coastal", "Indoor natatorium", "NASA KSC")

        def plot_grouped_ablation(ax, labels, wh, nat, ksc, xlim, title, cap_val=105.0):
            y = np.arange(len(labels))
            h = 0.23
            wh_plot = [min(v, cap_val) if v > 100 else v for v in wh]
            nat_plot = [min(v, cap_val) if v > 100 else v for v in nat]
            ksc_plot = [min(v, cap_val) if v > 100 else v for v in ksc]

            ax.grid(axis="x", color="#E5E5E5", linewidth=0.6, zorder=0)
            b1 = ax.barh(y - h, wh_plot, height=h, color=site_colors[0], label=site_names[0], zorder=3)
            b2 = ax.barh(y, nat_plot, height=h, color=site_colors[1], label=site_names[1], zorder=3)
            b3 = ax.barh(y + h, ksc_plot, height=h, color=site_colors[2], label=site_names[2], zorder=3)

            for bars, raw_vals, col, dy in (
                (b1, wh, site_colors[0], -h),
                (b2, nat, site_colors[1], 0.0),
                (b3, ksc, site_colors[2], +h),
            ):
                for idx, (bar, rv) in enumerate(zip(bars, raw_vals)):
                    if rv > 100:
                        bar.set_hatch("//")
                        bar.set_edgecolor("#222222")
                        bar.set_linewidth(0.5)
                        x_pos = cap_val + 2.5
                        ax.text(
                            x_pos,
                            y[idx] + dy,
                            f"{rv:+.1f}%",
                            va="center",
                            ha="left",
                            fontsize=7.6,
                            color=col,
                            fontweight="bold",
                        )
                    elif abs(rv) >= 1.2:
                        pad = 0.7 if xlim[1] < 40 else 2.2
                        if rv >= 0:
                            ax.text(rv + pad, y[idx] + dy, f"+{rv:.1f}%", va="center", ha="left", fontsize=7.6, color=col)
                        else:
                            ax.text(rv - pad, y[idx] + dy, f"−{abs(rv):.1f}%", va="center", ha="right", fontsize=7.6, color=col)

            ax.axvline(0, color="#222222", linewidth=0.9, zorder=4)
            if xlim[1] > 100:
                ax.axvline(100, color="#666666", linestyle="--", linewidth=0.8, alpha=0.65, zorder=2)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlim(*xlim)
            ax.set_xlabel("Signed endpoint error (%)")
            ax.set_title(title, loc="left", fontweight="bold", pad=8)

        k5_wh = all_series["kged5"]["jinhai"]["endpoint_error_pct"]
        k5_nat = all_series["kged5"]["pool"]["endpoint_error_pct"]
        k5_ksc = all_series["kged5"]["ksc"]["endpoint_error_pct"]
        k5_mean = mean_ep_of("kged5")

        ax_c = fig.add_subplot(bot_gs[0, 0])
        c_row = sf_map["const_icorr"]
        plot_grouped_ablation(
            ax_c,
            labels=[
                f"Full 5-param KGED\n(mean {k5_mean:.2f}%)",
                f"No kinetic variation\n($i_{{\\mathrm{{corr}}}}={i_ref_uA_cm2:.2f}\\ \\mu\\mathrm{{A\\ cm}}^{{-2}}$, {c_row['mean_abs_endpoint_error_pct']:.2f}%)",
            ],
            wh=[k5_wh, c_row["jinhai_endpoint_error_pct"]],
            nat=[k5_nat, c_row["pool_endpoint_error_pct"]],
            ksc=[k5_ksc, c_row["ksc_endpoint_error_pct"]],
            xlim=(-13.5, 19.5),
            title="(c) Rate-law ablation",
        )
        ax_c.set_ylim(1.55, -0.55)

        ax_d = fig.add_subplot(bot_gs[0, 1])
        d_nr = sf_map["no_rust_decay"]
        d_ns = sf_map["no_field_scaling"]
        plot_grouped_ablation(
            ax_d,
            labels=[
                f"Full $a_0\\phi_{{\\mathrm{{rust}}}}(u)$\n(mean {k5_mean:.2f}%)",
                f"No rust decay ($\\phi_{{\\mathrm{{rust}}}}=1$)\n($g=a_0$, mean {d_nr['mean_abs_endpoint_error_pct']:.2f}%)",
                f"No field scaling ($a_0=1$)\n($g=\\phi_{{\\mathrm{{rust}}}}$, mean {d_ns['mean_abs_endpoint_error_pct']:.2f}%)",
            ],
            wh=[k5_wh, d_nr["jinhai_endpoint_error_pct"], d_ns["jinhai_endpoint_error_pct"]],
            nat=[k5_nat, d_nr["pool_endpoint_error_pct"], d_ns["pool_endpoint_error_pct"]],
            ksc=[k5_ksc, d_nr["ksc_endpoint_error_pct"], d_ns["ksc_endpoint_error_pct"]],
            xlim=(-138, 99),
            title="(d) Field-scaling & rust-decay ablation",
        )

        ax_e = fig.add_subplot(bot_gs[0, 2])
        e_nw = sf_map["no_wetness"]
        e_iso = sf_map["binary_iso9223"]
        plot_grouped_ablation(
            ax_e,
            labels=[
                f"Continuous $q_{{\\mathrm{{wet}}}}$ (full)\n(mean {k5_mean:.2f}%)",
                f"No wetness ($q_{{\\mathrm{{wet}}}}=1$)\n(mean {e_nw['mean_abs_endpoint_error_pct']:.2f}%)",
                f"Binary ISO 9223 ($q_{{\\mathrm{{ISO}}}}$)\n(mean {e_iso['mean_abs_endpoint_error_pct']:.2f}%)",
            ],
            wh=[k5_wh, e_nw["jinhai_endpoint_error_pct"], e_iso["jinhai_endpoint_error_pct"]],
            nat=[k5_nat, e_nw["pool_endpoint_error_pct"], e_iso["pool_endpoint_error_pct"]],
            ksc=[k5_ksc, e_nw["ksc_endpoint_error_pct"], e_iso["ksc_endpoint_error_pct"]],
            xlim=(-110, 162),
            title="(e) Surface-wetness ablation",
        )

        handles, labels_leg = ax_c.get_legend_handles_labels()
        fig.legend(
            handles,
            labels_leg,
            loc="lower center",
            bbox_to_anchor=(0.54, -0.02),
            ncol=3,
            frameon=False,
            handlelength=1.6,
            columnspacing=2.2,
        )

        out_path = out_dir / "Fig_10_Baselines_and_Ablations.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
        return out_path


def render_figures(
    out_dir: Path,
    all_series: dict[str, dict[str, dict]],
    sf_map: dict[str, dict],
    i_ref_uA_cm2: float,
    ablation_rows: list[dict],
    r5_json: dict,
    r8_json: dict,
    ref_integrator: Integrator,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.85,
        }
    )
    saved_figs: list[Path] = []

    # 1. Main Fig. 9: Cross-environment trajectories (Weihai shoreline source + Weihai near-coastal, Indoor natatorium, NASA KSC)
    fig9, axes = plt.subplots(2, 2, figsize=(8.2, 6.45), dpi=300)
    fig9.subplots_adjust(left=0.095, right=0.98, bottom=0.165, top=0.93, hspace=0.38, wspace=0.28)
    for idx, (ax, case) in enumerate(zip(axes.flat, CASES)):
        d5 = all_series["kged5"][case]
        d8 = all_series["kged8"][case]
        dp = all_series["power_time"][case]
        m_arr = d5["months"]
        y_arr = d5["observed_depth_um"]

        ax.plot(m_arr[1:], y_arr[1:], "ko", ms=4.0, label="Observed", zorder=5)
        ax.plot(m_arr, d5["conditioned_depth_um"], color="#2474B5", ls="-", lw=2.0, label="Five-parameter KGED", zorder=4)
        ax.plot(m_arr, d8["conditioned_depth_um"], color="#BC4B32", ls="--", lw=2.0, label="Eight-parameter KGED", zorder=4)
        ax.plot(m_arr, dp["conditioned_depth_um"], color="#707070", ls=":", lw=1.6, label="Empirical power-law model", zorder=3)
        ax.plot(
            m_arr[0],
            y_arr[0],
            marker="D",
            mfc="white",
            mec="black",
            ms=5.2,
            ls="none",
            label="Initial depth $D_{\\mathrm{obs}}(t_1)$",
            zorder=6,
        )

        note = (
            f"5-param: RMSE {d5['rmse_um']:.2f} μm ({d5['endpoint_error_pct']:+.2f}%)\n"
            f"8-param: RMSE {d8['rmse_um']:.2f} μm ({d8['endpoint_error_pct']:+.2f}%)"
        )
        ax.text(
            0.03,
            0.96,
            note,
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
        )
        top = max(float(np.max(y_arr)), float(np.max(d5["conditioned_depth_um"])), float(np.max(d8["conditioned_depth_um"])))
        ax.set_ylim(0.0, top * 1.32)
        ax.set_title(f"({chr(97 + idx)}) {SITE_DISPLAY[case]}", loc="left", fontsize=10.0, fontweight="bold")
        ax.set_xlabel("Exposure time (months)", fontsize=9.2)
        ax.set_ylabel("Corrosion depth (μm)", fontsize=9.2)
        ax.grid(color="#E4E7EB", lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, leg_labels = axes.flat[0].get_legend_handles_labels()
    fig9.legend(handles, leg_labels, loc="lower center", bbox_to_anchor=(0.52, 0.02), ncol=3, frameon=False, fontsize=8.3)
    f9_path = out_dir / "Fig_9_Engineering_Trajectories.png"
    fig9.savefig(f9_path, dpi=300, facecolor="white")
    plt.close(fig9)
    saved_figs.append(f9_path)

    # 2. Main Fig. 10: External benchmarks & single-factor integrator ablations
    saved_figs.append(_render_fig10_baselines_and_ablations(out_dir, all_series, sf_map, i_ref_uA_cm2, plt, GridSpec))

    # 3. Main Fig. 11: Thick-film extrapolation mechanism & 12-bar rate-law ablations
    a_5 = next(m["parameters"]["p0"] for m in r5_json["physical_modules"] if m["hypothesis"] == "H3")
    p3_8 = next(m["parameters"] for m in r8_json["physical_modules"] if m["hypothesis"] == "H3")
    e_8, n_8, l_8 = p3_8["p0"], p3_8["p1"], p3_8["p2"]
    h_peak_5 = 200.0 * float(np.sqrt(a_5))
    h_peak_8 = 200.0 * float(n_8 / l_8)
    f3_5 = lambda h_val: (h_val / 200.0) * (a_5 + 1.0) / (a_5 + (h_val / 200.0) ** 2)
    f3_8 = lambda h_val: (1.0 + e_8 * ((h_val / 200.0) ** n_8) * np.exp(-l_8 * (h_val / 200.0))) / (1.0 + e_8 * np.exp(-l_8))

    h_linhai = ref_integrator.wet_mean_film_thickness("linhai")
    h_jinhai = ref_integrator.wet_mean_film_thickness("jinhai")
    h_ksc = ref_integrator.wet_mean_film_thickness("ksc")
    h_pool = ref_integrator.wet_mean_film_thickness("pool")

    fig11 = plt.figure(figsize=(13.20, 5.15), dpi=300)
    gs = GridSpec(1, 2, figure=fig11, width_ratios=[1.02, 1.08], wspace=0.20, left=0.060, right=0.985, top=0.90, bottom=0.14)
    ax_a = fig11.add_subplot(gs[0, 0])
    ax_b = fig11.add_subplot(gs[0, 1])

    h_grid = np.linspace(50.0, 1000.0, 600)
    y5_f = f3_5(h_grid)
    y8_f = f3_8(h_grid)

    ax_a.axvspan(min(h_peak_5, h_peak_8), max(h_peak_5, h_peak_8), color="#d95f02", alpha=0.14, zorder=1)
    ax_a.axvline(h_peak_5, color="#d95f02", ls="--", lw=1.0, alpha=0.85, zorder=2)
    ax_a.text(
        (h_peak_5 + h_peak_8) / 2.0,
        1.038,
        f"Thin-film peak ({h_peak_5:.1f}–{h_peak_8:.1f} μm)",
        color="#B85000",
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
    )

    ax_a.plot(h_grid, y5_f, color="#2474B5", lw=2.3, ls="-", label="Five-parameter KGED (rational tail)", zorder=4)
    ax_a.plot(h_grid, y8_f, color="#BC4B32", lw=2.3, ls="--", label="Eight-parameter KGED (saturating plateau)", zorder=4)
    mask_thick = h_grid >= h_linhai
    ax_a.fill_between(h_grid[mask_thick], y5_f[mask_thick], y8_f[mask_thick], color="#BC4B32", alpha=0.14, zorder=2)

    site_lines = [
        ("Weihai shoreline", h_linhai, "#222222", False),
        (f"Weihai near-coastal ({h_jinhai:.1f} μm)", h_jinhai, "#2E8B57", True),
        (f"NASA KSC ({h_ksc:.1f} μm)", h_ksc, "#6A3D9A", True),
        ("Indoor natatorium", h_pool, "#90301B", False),
    ]
    for lbl, hv, col, draw_vert_text in site_lines:
        ax_a.axvline(hv, color=col, ls="-.", lw=0.85, alpha=0.60, zorder=2)
        ax_a.scatter([hv], [float(f3_5(hv))], s=32, color="#2474B5", edgecolors="white", linewidths=0.7, zorder=5)
        ax_a.scatter([hv], [float(f3_8(hv))], s=32, color="#BC4B32", edgecolors="white", linewidths=0.7, zorder=5)
        if draw_vert_text:
            ax_a.text(hv - 7.0, 0.445, lbl, rotation=90, va="bottom", ha="right", fontsize=7.3, color=col)

    ax_a.annotate(
        f"Weihai shoreline ({h_linhai:.1f} μm)",
        xy=(h_linhai - 2.0, float(f3_5(h_linhai)) - 0.004),
        xytext=(485.0, 0.355),
        fontsize=7.6,
        fontweight="bold",
        color="#222222",
        arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 1.0},
    )

    y5_pool = float(f3_5(h_pool))
    y8_pool = float(f3_8(h_pool))
    ax_a.annotate(
        "",
        xy=(h_pool, y8_pool),
        xytext=(h_pool, y5_pool),
        arrowprops={"arrowstyle": "<->", "color": "#90301B", "lw": 1.35},
        zorder=6,
    )
    ax_a.annotate(
        f"Indoor natatorium ({h_pool:.1f} μm)\n+23.8% plateau (+18.69% at 48 mo)",
        xy=(h_pool - 3.0, (y5_pool + y8_pool) / 2.0 + 0.015),
        xytext=(638.0, 0.845),
        fontsize=7.6,
        fontweight="bold",
        color="#7A2212",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF7F5", "edgecolor": "#D89A90", "lw": 0.85},
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.16", "color": "#90301B", "lw": 0.95},
        zorder=6,
    )

    ax_a.set_xlim(40.0, 1025.0)
    ax_a.set_ylim(0.24, 1.12)
    ax_a.set_xlabel("Effective wet film thickness $H_{\\mathrm{eq}}$ (μm)", fontsize=9.8)
    ax_a.set_ylabel(
        "Film-thickness factor $\\phi_{H}(H_{\\mathrm{eq}}/200\\,\\mathrm{\\mu m})$\n(normalised to 1.0 at $H_{\\mathrm{ref}} = 200$ μm)",
        fontsize=9.4,
    )
    ax_a.set_title("(a) Thick-film response and extrapolation", loc="left", fontsize=10.5, fontweight="bold", pad=8)
    ax_a.grid(True, ls=":", alpha=0.35)
    ax_a.legend(loc="lower left", fontsize=8.0, frameon=True, facecolor="white", edgecolor="#DDDDDD")

    ab_cats = [
        "H1 + H2 + H3\n(full model)",
        "H1 + H3\n(no H2)",
        "H2 + H3\n(no H1)",
        "H3 only",
        "H1 only",
        "H1 + H2\n(no H3)",
    ]
    k5_ab = [r["mean_abs_ep_pct"] for r in ablation_rows if r["rate_law"] == "Five-parameter KGED"]
    k8_ab = [r["mean_abs_ep_pct"] for r in ablation_rows if r["rate_law"] == "Eight-parameter KGED"]
    x_pos = np.arange(len(ab_cats))
    bw = 0.37
    b1 = ax_b.bar(x_pos - bw / 2, k5_ab, bw, label="Five-parameter KGED", color="#2474B5", zorder=3)
    b2 = ax_b.bar(x_pos + bw / 2, k8_ab, bw, label="Eight-parameter KGED", color="#BC4B32", zorder=3)
    for rect in b1:
        hv = rect.get_height()
        ax_b.text(
            rect.get_x() + rect.get_width() / 2.0,
            hv + 0.25,
            f"{hv:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#1B5688",
        )
    for rect in b2:
        hv = rect.get_height()
        ax_b.text(
            rect.get_x() + rect.get_width() / 2.0,
            hv + 0.25,
            f"{hv:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#90301B",
        )
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(ab_cats, fontsize=8.5)
    ax_b.set_ylim(0.0, 17.6)
    ax_b.set_yticks([0, 4, 8, 12, 16])
    ax_b.set_ylabel("Mean absolute endpoint error (%)", fontsize=9.8)
    ax_b.set_title("(b) Effect of omitting individual response terms", loc="left", fontsize=10.5, fontweight="bold", pad=8)
    ax_b.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    ax_b.legend(loc="upper left", fontsize=8.4, frameon=True, facecolor="white", edgecolor="#DDDDDD")

    f11_path = out_dir / "Fig_11_ThickFilm_and_Ablation.png"
    fig11.savefig(f11_path, dpi=300, facecolor="white")
    plt.close(fig11)
    saved_figs.append(f11_path)

    # 4. Supplementary Figs. S3, S4, S5 (matching SI figure numbering)
    saved_figs.append(
        _plot_supplementary_group(
            out_dir, all_series, BENCHMARK_GROUPS[0][1], "Fig_S3_Laboratory_Baselines_Trajectories.png", False, 4, plt
        )
    )
    saved_figs.append(
        _plot_supplementary_group(
            out_dir, all_series, BENCHMARK_GROUPS[1][1], "Fig_S4_Direct_DataDriven_Trajectories.png", True, 4, plt
        )
    )
    saved_figs.append(
        _plot_supplementary_group(
            out_dir, all_series, BENCHMARK_GROUPS[2][1], "Fig_S5_Domain_Adaptation_Trajectories.png", True, 3, plt
        )
    )
    try:
        from PIL import Image

        for fp in saved_figs:
            with Image.open(fp) as im:
                if im.mode != "RGB":
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    if "A" in im.getbands():
                        bg.paste(im, mask=im.split()[-1])
                    else:
                        bg.paste(im.convert("RGB"))
                    bg.save(fp, format="PNG", dpi=(300, 300))
                else:
                    im.save(fp, format="PNG", dpi=(300, 300))
    except ImportError:
        pass
    return saved_figs


def run_all(out_dir: Path, skip_plots: bool = False) -> dict:
    t0 = time.perf_counter()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    eng_dir = REPO_ROOT / "engineering"
    in_dir = eng_dir / "inputs"
    mapping = json.loads((in_dir / "mapping.json").read_text(encoding="utf-8"))
    validate_mapping(mapping)

    r5_json = json.loads((eng_dir / "parsimonious_equation.json").read_text(encoding="utf-8"))
    r8_json = json.loads((eng_dir / "best_accuracy_equation.json").read_text(encoding="utf-8"))

    env_data = {}
    labels = {}
    for c in CASES:
        with np.load(in_dir / f"{c}_environment.npz") as npz:
            env_data[c] = {k: npz[k].copy() for k in npz.files}
        labels[c] = json.loads((in_dir / f"{c}_labels.json").read_text(encoding="utf-8"))

    post_initial_counts = [len(labels[c]["months"]) - 1 for c in TARGET_SITES]
    require(post_initial_counts == [8, 8, 9], f"Unexpected post-t1 observation counts: {post_initial_counts}")

    law5 = ClosedFormLaw("parsimonious", r5_json)
    law8 = ClosedFormLaw("best_accuracy", r8_json)
    state_cfg = mapping["fixed_state_structure"]

    print("=" * 98)
    print("KGED STAGE V: ONE-CLICK CROSS-ENVIRONMENT CORROSION PREDICTION")
    print("=" * 98)

    # ------------------------------------------------------------------
    # STEP 1: Closed-form reference & Gauss-Legendre quadrature check
    # ------------------------------------------------------------------
    quad_rel_max = {}
    cf_max = {}
    for alias, law, eq_json in [("parsimonious", law5, r5_json), ("best_accuracy", law8, r8_json)]:
        ref_pred = float(law.predict_log(25.0, 3.5, 200.0)[0])
        cf_err = abs(ref_pred - float(eq_json["intercept"]))
        require(cf_err < CLOSED_FORM_ATOL, f"Closed-form reference error exceeded tolerance for {alias}: {cf_err:.2e}")
        cf_max[alias] = cf_err
        integ_tmp = Integrator(env_data, law, mapping)
        rel_errs = []
        for c in CASES:
            b16 = integ_tmp.basis(c, state_cfg, 16)
            b32 = integ_tmp.basis(c, state_cfg, 32)
            rel_errs.append(float(np.max(np.abs(b16 - b32))) / max(1.0, float(np.max(np.abs(b32)))))
        quad_rel_max[alias] = max(rel_errs)
        require(quad_rel_max[alias] < QUADRATURE_REL_TOL, f"Quadrature relative diff too large for {alias}")

    print(
        f"[Step 1/5] Closed-form & quadrature verified (reference |dlog i| = {max(cf_max.values()):.2e}; "
        f"16- vs 32-pt Gauss-Legendre max rel |dB| = {max(quad_rel_max.values()):.2e})"
    )

    # Export Section 2.6 & Supplementary Methods M1.4 parameter provenance summary
    written_files.append(
        write_csv(
            out_dir / "Parameter_Provenance.csv",
            [
                {"parameter": "RH0_pct", "value": DEFAULT_MAPPING["RH0"], "role": "Wetting threshold (Eq. 11)"},
                {"parameter": "alpha", "value": DEFAULT_MAPPING["alpha"], "role": "Wetting exponent (Eq. 11)"},
                {"parameter": "J0_mg_m2_day", "value": DEFAULT_MAPPING["J0"], "role": "Chloride half-saturation flux (Eq. 11)"},
                {"parameter": "Hmin_um", "value": DEFAULT_MAPPING["Hmin"], "role": "Minimum effective film thickness"},
                {"parameter": "Hmax_um", "value": DEFAULT_MAPPING["Hmax"], "role": "Maximum effective film thickness"},
                {"parameter": "T_ref_C", "value": T_REF, "role": "Field reference temperature for I* and ablations (Eq. 13)"},
                {"parameter": "H_field_ref_um", "value": H_FIELD_REF, "role": "Field reference film thickness for I* and ablations (Eq. 13)"},
                {"parameter": "H_field_ref_over_Hmax", "value": Q_WET_REF, "role": "Mid-range reference film-thickness ratio H_field_ref/H_max in I* = 0.5 * i_corr(T_ref, C_bg, H_field_ref) (Eq. 13)"},
                {"parameter": "Cbg_wt_pct", "value": round(float(mapping["Cbg"]), 6), "role": "Background effective NaCl concentration"},
                {"parameter": "Ccap_wt_pct", "value": round(float(mapping["Ccap"]), 6), "role": "Saturation cap effective NaCl concentration"},
                {"parameter": "tau_u", "value": state_cfg["tau"], "role": "Fixed characteristic dose scale"},
                {"parameter": "lambda_a", "value": state_cfg["lam"], "role": "Fixed Tikhonov smoothness weight"},
                {"parameter": "s_a", "value": state.S_A, "role": "Amplitude scale in Tikhonov smoothness penalty (SI Section M1.4)"},
                {"parameter": "Gamma_um_cm2_per_A_s", "value": round(float(mapping["Gamma"]), 6), "role": "Divalent iron Faraday conversion factor (um [A s cm^-2]^-1)"},
            ],
        )
    )

    # ------------------------------------------------------------------
    # STEP 2: Source-site BVLS calibration at Weihai shoreline (Table 7)
    # ------------------------------------------------------------------
    lh_months = np.asarray(labels["linhai"]["months"], float)
    lh_obs = np.asarray(labels["linhai"]["observed_depth_um"], float)

    table7_rows = []
    kged_live = {}
    print("\n[Step 2/5] Table 7 -- Source-Calibrated Cubic Bernstein State Amplitudes at Weihai Shoreline (BVLS)")
    print("-" * 98)
    print(f"{'Equation':<28s} {'a0':>10s} {'a1':>10s} {'a2':>10s} {'a3':>10s} {'a3/a0':>8s} {'Calib RMSE (um)':>17s}")
    print("-" * 98)
    for m_key, disp_name, law in [
        ("kged5", "Five-parameter KGED", law5),
        ("kged8", "Eight-parameter KGED", law8),
    ]:
        integ = Integrator(env_data, law, mapping)
        fitted = state.fit(integ, state_cfg, lh_obs, lh_months, len(lh_obs))
        require(bool(fitted["eligible"]), f"Weihai shoreline BVLS source calibration ineligible for {m_key}")
        a_vec = np.asarray(fitted["coefficients"], float)
        calib_rmse = float(np.sqrt(fitted["source_SSE_um2"] / len(lh_obs)))
        kged_live[m_key] = (integ, a_vec)
        a3_over_a0 = float(a_vec[3]) / float(a_vec[0])
        table7_rows.append(
            {
                "Equation": disp_name,
                "a0": round(float(a_vec[0]), 6),
                "a1": round(float(a_vec[1]), 6),
                "a2": round(float(a_vec[2]), 6),
                "a3": round(float(a_vec[3]), 6),
                "a3_over_a0": round(a3_over_a0, 6),
                "unanchored_source_rmse_um": round(calib_rmse, 4),
            }
        )
        print(
            f"{disp_name:<28s} {a_vec[0]:>10.3f} {a_vec[1]:>10.3f} {a_vec[2]:>10.3f} "
            f"{a_vec[3]:>10.3f} {a3_over_a0:>8.3f} {calib_rmse:>17.2f}"
        )
    print("-" * 98)
    written_files.append(write_csv(out_dir / "Table_7_Source_State_Amplitudes.csv", table7_rows))

    # ------------------------------------------------------------------
    # STEP 3: Live integration of 7 equation-based methods + re-scoring of
    #         11 frozen ML/DA baselines, with 18/18 Fig. 10a-b verification
    # ------------------------------------------------------------------
    live_series: dict[str, dict[str, dict]] = {}
    for m_key, (integ, a_vec) in kged_live.items():
        live_series[m_key] = {}
        for c in CASES:
            m_arr = np.asarray(labels[c]["months"], float)
            y_arr = np.asarray(labels[c]["observed_depth_um"], float)
            raw = integ.basis(c, state_cfg, 16) @ a_vec
            p_cond = y_arr[0] + (raw - raw[0])
            live_series[m_key][c] = evaluate_conditioned_series(m_arr, y_arr, p_cond)

    for sr_key in ("sparse", "gp", "pysr", "physo"):
        sr_law = SymbolicBaselineLaw(sr_key)
        sr_integ = Integrator(env_data, sr_law, mapping)
        sr_fit = state.fit(sr_integ, state_cfg, lh_obs, lh_months, len(lh_obs))
        require(bool(sr_fit["eligible"]), f"Weihai shoreline BVLS fit ineligible for symbolic baseline {sr_key}")
        a_sr = np.asarray(sr_fit["coefficients"], float)
        live_series[sr_key] = {}
        for c in CASES:
            m_arr = np.asarray(labels[c]["months"], float)
            y_arr = np.asarray(labels[c]["observed_depth_um"], float)
            raw = sr_integ.basis(c, state_cfg, 16) @ a_sr
            p_cond = y_arr[0] + (raw - raw[0])
            live_series[sr_key][c] = evaluate_conditioned_series(m_arr, y_arr, p_cond)

    # Empirical power-law model fitted from scratch on Weihai shoreline: OLS of ln(max(D, 1e-6)) on ln(max(t, 1e-3))
    n_pow, ln_a_pow = np.polyfit(np.log(np.maximum(lh_months, 1e-3)), np.log(np.maximum(lh_obs, 1e-6)), 1)
    a_pow = float(np.exp(ln_a_pow))
    live_series["power_time"] = {}
    for c in CASES:
        m_arr = np.asarray(labels[c]["months"], float)
        y_arr = np.asarray(labels[c]["observed_depth_um"], float)
        raw = a_pow * (m_arr ** n_pow)
        p_cond = y_arr[0] + (raw - raw[0])
        live_series["power_time"][c] = evaluate_conditioned_series(m_arr, y_arr, p_cond)

    archived_pts: dict[str, dict[str, list[dict]]] = {}
    with (eng_dir / "Table_conditioned_predictions.csv").open("r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            archived_pts.setdefault(r["method"], {}).setdefault(r["case"], []).append(r)

    all_series: dict[str, dict[str, dict]] = {}
    traj_csv_rows = []
    for grp_name, members in BENCHMARK_GROUPS:
        for m_key, m_label in members:
            all_series[m_key] = {}
            for c in CASES:
                pts = archived_pts[m_key][c]
                m_arch = np.asarray([float(x["month"]) for x in pts], float)
                y_arch = np.asarray([float(x["observed_depth_um"]) for x in pts], float)
                p_arch = np.asarray([float(x["conditioned_depth_um"]) for x in pts], float)
                if m_key in LIVE_METHODS:
                    all_series[m_key][c] = live_series[m_key][c]
                    src_mode = "live_from_scratch"
                else:
                    all_series[m_key][c] = evaluate_conditioned_series(m_arch, y_arch, p_arch)
                    src_mode = "archived_frozen_prediction"

                d_cur = all_series[m_key][c]
                for idx_pt in range(len(d_cur["months"])):
                    traj_csv_rows.append(
                        {
                            "method": m_key,
                            "label": m_label,
                            "group": grp_name,
                            "computation_mode": src_mode,
                            "site": c,
                            "month": float(d_cur["months"][idx_pt]),
                            "observed_depth_um": float(d_cur["observed_depth_um"][idx_pt]),
                            "predicted_depth_um": float(d_cur["conditioned_depth_um"][idx_pt]),
                            "is_initial_anchor": idx_pt == 0,
                        }
                    )

    written_files.append(write_csv(out_dir / "Predicted_Trajectories_All_Methods.csv", traj_csv_rows))

    print("\n[Step 3/5] Fig. 10a-b -- Cross-Environment Benchmark Comparisons (Eq. 16)")
    print("-" * 116)
    print(
        f"{'Mode Method':<34s} {'Near-coastal RMSE':>17s} {'Near-coastal EP%':>16s} "
        f"{'Natatorium RMSE':>15s} {'Natatorium EP%':>14s} {'NASA KSC RMSE':>13s} {'NASA KSC EP%':>12s} {'Mean |EP|%':>11s}"
    )
    print("-" * 116)

    bench_rows = []
    diverged_raw_rows = []
    formatted_cells: dict[str, tuple[str, ...]] = {}
    md_lines = [
        "# Cross-environment engineering corrosion prediction benchmarks (Fig. 10a–b, Eq. 16)",
        "",
        "| Method | Weihai near-coastal RMSE (μm) | Weihai near-coastal endpoint error (%) | Indoor natatorium RMSE (μm) | Indoor natatorium endpoint error (%) | NASA KSC RMSE (μm) | NASA KSC endpoint error (%) | Mean absolute endpoint error (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for grp_name, members in BENCHMARK_GROUPS:
        print(f"  [{grp_name}]")
        md_lines.append(f"| **{grp_name}** | | | | | | | |")
        for m_key, m_label in members:
            dj, dp, dk = all_series[m_key]["jinhai"], all_series[m_key]["pool"], all_series[m_key]["ksc"]
            any_div = dj["diverged"] or dp["diverged"] or dk["diverged"]
            mean_abs_ep = float(np.mean([abs(dj["endpoint_error_pct"]), abs(dp["endpoint_error_pct"]), abs(dk["endpoint_error_pct"])]))
            mean_str = "Diverged" if any_div else f"{mean_abs_ep:.2f}"

            jr_s, je_s = fmt_cell(dj, False), fmt_cell(dj, True)
            pr_s, pe_s = fmt_cell(dp, False), fmt_cell(dp, True)
            kr_s, ke_s = fmt_cell(dk, False), fmt_cell(dk, True)
            formatted_cells[m_key] = (jr_s, je_s, pr_s, pe_s, kr_s, ke_s, mean_str)

            mode = "live_from_scratch" if m_key in LIVE_METHODS else "archived_frozen_prediction"
            tag = "[L]" if mode == "live_from_scratch" else "[A]"
            md_name = m_label if mode == "live_from_scratch" else f"{m_label} †"

            print(
                f"  {tag} {m_label:<28s} {jr_s:>17s} {je_s:>16s} {pr_s:>15s} {pe_s:>14s} "
                f"{kr_s:>13s} {ke_s:>12s} {mean_str:>11s}"
            )
            md_lines.append(f"| {md_name} | {jr_s} | {je_s} | {pr_s} | {pe_s} | {kr_s} | {ke_s} | {mean_str} |")

            bench_rows.append(
                {
                    "group": grp_name,
                    "method_code": m_key,
                    "method": m_label,
                    "computation_mode": mode,
                    "jinhai_rmse_um": round(dj["rmse_um"], 4),
                    "jinhai_endpoint_error_pct": round(dj["endpoint_error_pct"], 4),
                    "pool_rmse_um": round(dp["rmse_um"], 4),
                    "pool_endpoint_error_pct": round(dp["endpoint_error_pct"], 4),
                    "ksc_rmse_um": round(dk["rmse_um"], 4),
                    "ksc_endpoint_error_pct": round(dk["endpoint_error_pct"], 4),
                    "mean_abs_endpoint_error_pct": round(mean_abs_ep, 4),
                    "negative_points_total": dj["negative_points"] + dp["negative_points"] + dk["negative_points"],
                    "decreasing_steps_total": dj["decreasing_steps"] + dp["decreasing_steps"] + dk["decreasing_steps"],
                    "status": "Diverged" if any_div else "Valid",
                }
            )
            if any_div:
                diverged_raw_rows.append(
                    {
                        "method_code": m_key,
                        "method": m_label,
                        "jinhai_raw_rmse_um": round(dj["rmse_um"], 2),
                        "jinhai_raw_ep_pct": round(dj["endpoint_error_pct"], 2),
                        "jinhai_neg_pts": dj["negative_points"],
                        "pool_raw_rmse_um": round(dp["rmse_um"], 2),
                        "pool_raw_ep_pct": round(dp["endpoint_error_pct"], 2),
                        "pool_neg_pts": dp["negative_points"],
                        "ksc_raw_rmse_um": round(dk["rmse_um"], 2),
                        "ksc_raw_ep_pct": round(dk["endpoint_error_pct"], 2),
                        "ksc_neg_pts": dk["negative_points"],
                        "raw_mean_abs_ep_pct": round(mean_abs_ep, 2),
                    }
                )

    print("-" * 116)
    print("  [L] = recalibrated at Weihai shoreline and integrated live from scratch in this run (7 equation-based methods).")
    print("  [A] = frozen archived ML / domain-adaptation predictions re-scored under Eq. (16) (3 integrated laboratory ML")
    print("        baselines + 8 direct field ML / domain-adaptation models; see engineering/comparators/reproduce_field_comparators.py).")

    mismatches = []
    for m_key, expected in MANUSCRIPT_BENCHMARKS.items():
        exp_tuple = tuple("Diverged" if v is None else v for v in expected)
        if formatted_cells[m_key] != exp_tuple:
            mismatches.append((m_key, formatted_cells[m_key], exp_tuple))
    require(not mismatches, f"Fig. 10a-b benchmark cell mismatch against manuscript: {mismatches}")
    print("  [VERIFIED] 18/18 benchmark models (all 126 cells) match the published manuscript exactly.")

    md_lines += [
        "",
        "† Frozen predictions loaded from `engineering/Table_conditioned_predictions.csv` and re-scored under Eq. (16) "
        "(3 integrated laboratory ML baselines + 8 direct field ML / domain-adaptation models; see "
        "`engineering/comparators/reproduce_field_comparators.py` for the 8 direct field ML / adaptation models).",
        "",
        f"*Diverged*: cell-wise RMSE > {DIVERGENCE_RULE['rmse_um_max']:.0f} μm or |endpoint error| > "
        f"{DIVERGENCE_RULE['abs_endpoint_pct_max']:.0f}% (raw values in `Diverged_Models_Raw_Metrics.csv`).",
    ]
    written_files.append(write_csv(out_dir / "Fig_10ab_Engineering_Comparison.csv", bench_rows))
    written_files.append(write_csv(out_dir / "Diverged_Models_Raw_Metrics.csv", diverged_raw_rows))
    md_path = out_dir / "Fig_10ab_Engineering_Comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    written_files.append(md_path)

    # ------------------------------------------------------------------
    # STEP 4: Single-factor integrator ablations (6 ablations, Fig. 10a, c-e)
    # ------------------------------------------------------------------
    integ_k5, a_k5 = kged_live["kged5"]
    b_k5 = {c: integ_k5.basis(c, state_cfg, 16) for c in CASES}
    t_end_s = float(np.sum(env_data["linhai"]["duration_days"])) * 86400.0
    i_time = float(lh_obs[-1]) / (float(mapping["Gamma"]) * t_end_s)
    log_iref = float(law5.predict_log([T_REF], [mapping["Cbg"]], [H_FIELD_REF])[0])
    i_ref_uA_cm2 = float(np.exp(log_iref) * 1e6)

    single_factor_specs = [
        (
            "elapsed_time",
            "Elapsed-time + g(u)",
            SingleFactorWetnessIntegrator(env_data, ConstantCurrentLaw(np.log(i_time)), mapping, "elapsed_time", override_ref=i_time),
            None,
        ),
        (
            "const_icorr",
            f"Constant i_corr = {i_ref_uA_cm2:.2f} uA cm^-2",
            Integrator(env_data, ConstantCurrentLaw(log_iref), mapping),
            None,
        ),
        (
            "no_rust_decay",
            "No rust-layer attenuation (phi_rust = 1, g = a0)",
            integ_k5,
            np.full(4, a_k5[0]),
        ),
        (
            "no_field_scaling",
            "No laboratory-to-field scaling (a0 = 1, g = phi_rust)",
            integ_k5,
            a_k5 / a_k5[0],
        ),
        (
            "no_wetness",
            "No surface-wetness weighting (q_wet = 1)",
            SingleFactorWetnessIntegrator(env_data, law5, mapping, "no_wetness"),
            None,
        ),
        (
            "binary_iso9223",
            "Binary ISO 9223 wetness (RH > 80%, T > 0 C)",
            SingleFactorWetnessIntegrator(env_data, law5, mapping, "binary_iso9223"),
            None,
        ),
    ]

    print("\n[Step 4/5] Fig. 10a, c-e -- Single-Factor Integrator Ablations (6 Ablations, Live from Scratch)")
    print("-" * 116)
    print(
        f"{'Ablation':<52s} {'Near-coastal EP%':>16s} {'Natatorium EP%':>14s} "
        f"{'NASA KSC EP%':>12s} {'Mean |EP|%':>11s}"
    )
    print("-" * 116)

    sf_rows = []
    sf_map = {}
    sf_mismatches = []
    for sf_code, sf_label, sf_integ, fixed_a in single_factor_specs:
        if fixed_a is None:
            sf_fit = state.fit(sf_integ, state_cfg, lh_obs, lh_months, len(lh_obs))
            require(bool(sf_fit["eligible"]), f"Single-factor ablation BVLS fit ineligible for {sf_code}")
            a_sf = np.asarray(sf_fit["coefficients"], float)
        else:
            a_sf = np.asarray(fixed_a, float)

        b_sf = b_k5 if sf_integ is integ_k5 else {c: sf_integ.basis(c, state_cfg, 16) for c in CASES}
        site_res = {}
        for c in TARGET_SITES:
            m_arr = np.asarray(labels[c]["months"], float)
            y_arr = np.asarray(labels[c]["observed_depth_um"], float)
            raw = b_sf[c] @ a_sf
            p_cond = y_arr[0] + (raw - raw[0])
            site_res[c] = evaluate_conditioned_series(m_arr, y_arr, p_cond)

        dj, dp, dk = site_res["jinhai"], site_res["pool"], site_res["ksc"]
        mean_abs_ep = float(np.mean([abs(dj["endpoint_error_pct"]), abs(dp["endpoint_error_pct"]), abs(dk["endpoint_error_pct"])]))
        got_tup = (
            f"{dj['rmse_um']:.2f}",
            f"{dj['endpoint_error_pct']:+.2f}",
            f"{dp['rmse_um']:.2f}",
            f"{dp['endpoint_error_pct']:+.2f}",
            f"{dk['rmse_um']:.2f}",
            f"{dk['endpoint_error_pct']:+.2f}",
            f"{mean_abs_ep:.2f}",
        )
        exp_tup = MANUSCRIPT_SINGLE_FACTOR_ABLATIONS[sf_code]
        if got_tup != exp_tup:
            sf_mismatches.append((sf_code, got_tup, exp_tup))

        row_dict = {
            "ablation_code": sf_code,
            "ablation": sf_label,
            "a0": round(float(a_sf[0]), 6),
            "a1": round(float(a_sf[1]), 6),
            "a2": round(float(a_sf[2]), 6),
            "a3": round(float(a_sf[3]), 6),
            "jinhai_rmse_um": round(dj["rmse_um"], 4),
            "jinhai_endpoint_error_pct": round(dj["endpoint_error_pct"], 4),
            "pool_rmse_um": round(dp["rmse_um"], 4),
            "pool_endpoint_error_pct": round(dp["endpoint_error_pct"], 4),
            "ksc_rmse_um": round(dk["rmse_um"], 4),
            "ksc_endpoint_error_pct": round(dk["endpoint_error_pct"], 4),
            "mean_abs_endpoint_error_pct": round(mean_abs_ep, 4),
        }
        sf_rows.append(row_dict)
        sf_map[sf_code] = row_dict
        print(
            f"  {sf_label:<50s} {dj['endpoint_error_pct']:>+15.2f}% "
            f"{dp['endpoint_error_pct']:>+13.2f}% {dk['endpoint_error_pct']:>+11.2f}% {mean_abs_ep:>10.2f}%"
        )

    print("-" * 116)
    require(not sf_mismatches, f"Fig. 10a, c-e single-factor ablation mismatch against manuscript: {sf_mismatches}")
    print("  [VERIFIED] 6/6 single-factor integrator ablations match Section 3.3.2 & Fig. 10 exactly.")
    written_files.append(write_csv(out_dir / "Fig_10cde_Single_Factor_Ablations.csv", sf_rows))

    # ------------------------------------------------------------------
    # STEP 5: Component-wise rate-law ablations (12 combinations, Fig. 11b)
    # ------------------------------------------------------------------
    ablation_specs = [
        ("H1 + H2 + H3 (full model)", True, True, True),
        ("H1 + H3 (no H2, C=Cbg)", True, False, True),
        ("H2 + H3 (no H1, T=25C)", False, True, True),
        ("H3 only (T=25C, C=Cbg)", False, False, True),
        ("H1 only (C=Cbg, H=500um)", True, False, False),
        ("H1 + H2 (no H3, H=500um)", True, True, False),
    ]
    c_bg = float(mapping["Cbg"])
    ablation_rows = []
    print("\n[Step 5/5] Component-Wise Hypothesis Ablations (12 Combinations, Fig. 11b)")
    print("-" * 108)
    print(f"{'Rate Law':<24s} {'Variant':<32s} {'Near-coastal EP%':>16s} {'Natatorium EP%':>14s} {'NASA KSC EP%':>12s} {'Mean |EP|%':>11s}")
    print("-" * 108)
    for eq_label, base_law in [("Five-parameter KGED", law5), ("Eight-parameter KGED", law8)]:
        for var_name, k1, k2, k3 in ablation_specs:
            ab_law = AblatedLaw(base_law, c_bg, k1, k2, k3)
            ab_integ = Integrator(env_data, ab_law, mapping)
            ab_fit = state.fit(ab_integ, state_cfg, lh_obs, lh_months, len(lh_obs))
            require(bool(ab_fit["eligible"]), f"Ablation BVLS fit ineligible for {eq_label} / {var_name}")
            a_vec = np.asarray(ab_fit["coefficients"], float)
            s_metrics = {}
            for c in TARGET_SITES:
                raw = ab_integ.basis(c, state_cfg, 16) @ a_vec
                y_arr = np.asarray(labels[c]["observed_depth_um"], float)
                yp = y_arr[0] + (raw - raw[0])
                s_metrics[c] = (
                    float(np.sqrt(np.mean((yp[1:] - y_arr[1:]) ** 2))),
                    float((yp[-1] - y_arr[-1]) / y_arr[-1] * 100.0),
                )
            mean_ep = float(np.mean([abs(s_metrics[c][1]) for c in TARGET_SITES]))
            ablation_rows.append(
                {
                    "rate_law": eq_label,
                    "variant": var_name,
                    "a0": round(float(a_vec[0]), 6),
                    "a1": round(float(a_vec[1]), 6),
                    "a2": round(float(a_vec[2]), 6),
                    "a3": round(float(a_vec[3]), 6),
                    "jinhai_rmse_um": round(s_metrics["jinhai"][0], 4),
                    "jinhai_ep_pct": round(s_metrics["jinhai"][1], 4),
                    "pool_rmse_um": round(s_metrics["pool"][0], 4),
                    "pool_ep_pct": round(s_metrics["pool"][1], 4),
                    "ksc_rmse_um": round(s_metrics["ksc"][0], 4),
                    "ksc_ep_pct": round(s_metrics["ksc"][1], 4),
                    "mean_abs_ep_pct": round(mean_ep, 4),
                }
            )
            print(
                f"{eq_label:<24s} {var_name:<32s} "
                f"{s_metrics['jinhai'][1]:>+15.2f}% {s_metrics['pool'][1]:>+13.2f}% "
                f"{s_metrics['ksc'][1]:>+11.2f}% {mean_ep:>10.2f}%"
            )
    print("-" * 108)
    written_files.append(write_csv(out_dir / "Rate_Law_Ablations_12Rows.csv", ablation_rows))

    saved_figs: list[Path] = []
    if not skip_plots:
        saved_figs = render_figures(out_dir, all_series, sf_map, i_ref_uA_cm2, ablation_rows, r5_json, r8_json, kged_live["kged5"][0])
        written_files.extend(saved_figs)

    try:
        rel_out = out_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_out = out_dir.as_posix()

    summary_path = out_dir / "Engineering_Summary.json"
    all_generated_names = sorted({*(p.name for p in written_files), summary_path.name})
    summary = {
        "status": "SUCCESS",
        "output_directory": rel_out,
        "manuscript_fig10ab_benchmarks_verified": "18/18",
        "manuscript_fig10cde_single_factor_ablations_verified": "6/6",
        "divergence_rule": DIVERGENCE_RULE,
        "computation_modes": {
            "live_from_scratch": sorted(LIVE_METHODS),
            "archived_frozen_prediction": sorted(set(MANUSCRIPT_BENCHMARKS) - LIVE_METHODS),
        },
        "kged_5param_target_mean_abs_endpoint_error_pct": round(
            float(np.mean([abs(all_series["kged5"][c]["endpoint_error_pct"]) for c in TARGET_SITES])), 4
        ),
        "kged_8param_target_mean_abs_endpoint_error_pct": round(
            float(np.mean([abs(all_series["kged8"][c]["endpoint_error_pct"]) for c in TARGET_SITES])), 4
        ),
        "empirical_power_law_fit_linhai": {"A": round(a_pow, 6), "nu": round(n_pow, 6)},
        "table7_amplitudes": table7_rows,
        "single_factor_ablations": sf_rows,
        "software_versions": {
            "python": platform.python_version().split(".")[0] + "." + platform.python_version().split(".")[1],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "generated_files": all_generated_names,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 98)
    print(f"COMPLETED IN {elapsed:.2f}s -- ALL OUTPUT TABLES & FIGURES SAVED TO: {rel_out}/")
    for fn in all_generated_names:
        print(f"  * {fn}")
    if saved_figs:
        print(f"Generated {len(saved_figs)} high-resolution figures (300 DPI PNG: Figs. 9, 10, 11, S3, S4, S5).")
    print("=" * 98)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "engineering/results",
        help="Directory where all engineering prediction tables, JSONs, and figures are written (default: engineering/results)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip rendering PNG trajectory and ablation figures")
    args = parser.parse_args()
    run_all(args.out_dir, skip_plots=args.no_plots)
