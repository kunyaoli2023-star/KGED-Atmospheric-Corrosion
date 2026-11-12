# -*- coding: utf-8 -*-
"""One-click Stage 5 cross-environment engineering corrosion-depth prediction pipeline.

Usage (zero required arguments):
    python run_engineering_predictions.py
    python run_engineering_predictions.py --out-dir engineering/results_reproduced

Recomputes live from scratch the 7 equation-based engineering models (retained 5- and
8-parameter KGED laws, Sparse regression, gplearn, PySR, PhySO, and empirical power law),
re-scores the 11 frozen trained ML / domain-adaptation baselines under Eq. (16) initial-depth
anchoring, verifies 18/18 exact Table 8 agreement with the manuscript, evaluates the 12
component-wise rate-law ablations (Fig. 10b & Supplementary Section M1.5.1), and renders
Main Figs. 9-10 and Supplementary Figs. S3-S5 at 300 DPI.
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 9):
    sys.exit(f"KGED Stage 5 requires Python >= 3.9 (found {sys.version.split()[0]}).")

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

from engineering.engineering_integrator import CASES, DEFAULT_MAPPING, Integrator
from engineering import engineering_state as state
from engineering.verify_closed_forms import CLOSED_FORM_ATOL, QUADRATURE_REL_TOL, ClosedFormLaw, require

TARGET_SITES = ("jinhai", "pool", "ksc")
SITE_DISPLAY = {
    "linhai": "Linhai (source)",
    "jinhai": "Jinhai",
    "pool": "Indoor swimming pool",
    "ksc": "NASA KSC",
}

# Exact Table 8 Footnote a divergence threshold: RMSE > 200 um or |EP| > 100%
DIVERGENCE_RULE = {
    "rmse_um_max": 200.0,
    "abs_endpoint_pct_max": 100.0,
}

LIVE_METHODS = {"kged5", "kged8", "sparse", "gp", "pysr", "physo", "power_time"}

TABLE8_GROUPS = [
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
        "Domain adaptation",
        [
            ("TCA_Ridge", "TCA-Ridge"),
            ("CORAL_Ridge", "CORAL-Ridge"),
            ("DomainInvariant_RF", "Domain-standardised RF"),
            ("DomainInvariant_XGBoost", "Domain-standardised XGBoost"),
            ("DANN_MLP", "DANN-MLP"),
        ],
    ),
    (
        "KGED + integration",
        [
            ("kged5", "Five-parameter KGED"),
            ("kged8", "Eight-parameter KGED"),
        ],
    ),
]

# Published Main Manuscript Table 8 values (None = "Diverged")
MANUSCRIPT_TABLE8 = {
    "bp": ("22.65", "+9.68", "70.30", "+40.58", "5.34", "-2.97", "17.74"),
    "rf": ("18.30", "+8.09", "45.45", "-7.91", "9.80", "-15.72", "10.57"),
    "xgb": ("20.91", "+10.48", "54.79", "-13.39", "6.70", "-9.86", "11.25"),
    "sparse": ("14.44", "+0.50", "32.40", "-0.14", "5.53", "-5.04", "1.89"),
    "gp": ("22.90", "+9.33", "43.44", "+25.91", "7.74", "-4.17", "13.13"),
    "pysr": ("22.71", "+11.48", "32.97", "-0.44", "5.93", "-5.90", "5.94"),
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
    "kged5": ("14.05", "-0.07", "32.38", "-0.48", "4.98", "-2.94", "1.16"),
    "kged8": ("18.39", "+4.45", "32.36", "+18.70", "4.83", "+2.73", "8.63"),
}


class SymbolicBaselineLaw:
    """Closed-form laboratory symbolic-regression baselines (Table 4 & Table S2).

    Selected outer seeds:
        sparse: seed 0 | gp (gplearn): seed 2718 | pysr: seed 31415 | physo: seed 2718
    All four expressions were fitted on centred log-current with constant reference offset.
    """

    SEEDS = {"sparse": 0, "gp": 2718, "pysr": 31415, "physo": 2718}
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
                ratio = np.where(np.abs(den) > 1e-3, den / num, 1.0)
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
        T_in = T_arr if self.keep_H1 else np.full_like(T_arr, 25.0)
        C_in = C_arr if self.keep_H2 else np.full_like(C_arr, self.c_bg)
        H_in = H_arr if self.keep_H3 else np.full_like(H_arr, 500.0)
        return self.base_law.predict_log(T_in, C_in, H_in)


def validate_mapping(mapping: dict):
    st = mapping.get("fixed_state_structure", {})
    checks = {
        "state family == bernstein": st.get("family") == "bernstein",
        "degree == 3 and k == 4": st.get("degree") == 3 and st.get("k") == 4,
        "tau_u == 0.25": abs(float(st.get("tau", -1.0)) - 0.25) < 1e-12,
        "lambda_a == 0.03": abs(float(st.get("lam", -1.0)) - 0.03) < 1e-12,
        "Cbg ~ 0.1139 wt%": abs(float(mapping.get("Cbg", -1.0)) - 0.1139197357) < 1e-6,
        "Ccap ~ 0.5171 wt%": abs(float(mapping.get("Ccap", -1.0)) - 0.5170650867) < 1e-6,
        "Gamma ~ 0.3687 um/(A s cm-2)": abs(float(mapping.get("Gamma", -1.0)) - 0.3686577318) < 1e-6,
    }
    bad = [k for k, ok in checks.items() if not ok]
    require(not bad, "mapping.json inconsistent with Supplementary Table S4: " + "; ".join(bad))


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
        ax.plot(m_arr[0], y_arr[0], marker="D", mfc="white", mec="black", ms=5.2, ls="none", label="Initial depth", zorder=6)
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


def render_figures(
    out_dir: Path,
    all_series: dict[str, dict[str, dict]],
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

    # 1. Main Fig. 9: Cross-environment trajectories (Linhai source + Jinhai, Pool, NASA KSC)
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

    # 2. Main Fig. 10: Thick-film extrapolation mechanism & 12-bar rate-law ablations
    a_5 = next(m["parameters"]["p0"] for m in r5_json["physical_modules"] if m["hypothesis"] == "H3")
    p3_8 = next(m["parameters"] for m in r8_json["physical_modules"] if m["hypothesis"] == "H3")
    e_8, n_8, l_8 = p3_8["p0"], p3_8["p1"], p3_8["p2"]
    h_peak_5 = 200.0 * float(np.sqrt(a_5))
    h_peak_8 = 200.0 * float(n_8 / l_8)
    f3_5 = lambda h_val: (h_val / 200.0) * (a_5 + 1.0) / (a_5 + (h_val / 200.0) ** 2)
    f3_8 = lambda h_val: (1.0 + e_8 * ((h_val / 200.0) ** n_8) * np.exp(-l_8 * (h_val / 200.0))) / (1.0 + e_8 * np.exp(-l_8))

    site_wet_h = [
        ("Linhai", ref_integrator.wet_mean_film_thickness("linhai"), "#222222"),
        ("Jinhai", ref_integrator.wet_mean_film_thickness("jinhai"), "#2E8B57"),
        ("NASA KSC", ref_integrator.wet_mean_film_thickness("ksc"), "#6A3D9A"),
        ("Pool", ref_integrator.wet_mean_film_thickness("pool"), "#90301B"),
    ]

    fig10 = plt.figure(figsize=(13.0, 5.0), dpi=300)
    gs = GridSpec(1, 2, figure=fig10, width_ratios=[1.05, 1.15], wspace=0.22, left=0.065, right=0.985, top=0.90, bottom=0.16)
    ax_a = fig10.add_subplot(gs[0, 0])
    ax_b = fig10.add_subplot(gs[0, 1])

    h_grid = np.linspace(50.0, 1000.0, 600)
    y5_f = f3_5(h_grid)
    y8_f = f3_8(h_grid)
    ax_a.axvspan(min(h_peak_5, h_peak_8), max(h_peak_5, h_peak_8), color="#d95f02", alpha=0.15, zorder=1)
    min_site_h = min(v for _, v, _ in site_wet_h)
    ax_a.axvspan(min_site_h - 5.5, 1025.0, color="#7570b3", alpha=0.07, zorder=1)
    ax_a.plot(h_grid, y5_f, color="#2474B5", lw=2.3, ls="-", label="Five-parameter KGED (rational tail)", zorder=4)
    ax_a.plot(h_grid, y8_f, color="#BC4B32", lw=2.3, ls="--", label="Eight-parameter KGED (saturating plateau)", zorder=4)
    mask_thick = h_grid >= min_site_h
    ax_a.fill_between(h_grid[mask_thick], y5_f[mask_thick], y8_f[mask_thick], color="#BC4B32", alpha=0.15, zorder=2)
    for s_lbl, h_v, col in site_wet_h:
        ax_a.axvline(h_v, color=col, ls="-.", lw=0.85, alpha=0.55, zorder=2)
        ax_a.text(h_v - 5.5, 1.095, s_lbl, rotation=90, va="top", ha="right", fontsize=7.6, color=col, fontweight="bold")
        ax_a.scatter([h_v], [float(f3_5(h_v))], s=34, color="#2474B5", edgecolors="white", linewidths=0.7, zorder=5)
        ax_a.scatter([h_v], [float(f3_8(h_v))], s=34, color="#BC4B32", edgecolors="white", linewidths=0.7, zorder=5)
    ax_a.set_xlim(40.0, 1025.0)
    ax_a.set_ylim(0.24, 1.12)
    ax_a.set_xlabel("Effective wet film thickness $H_{\\mathrm{eq}}$ (μm)", fontsize=9.8)
    ax_a.set_ylabel("Normalised film-thickness factor $\\exp[m_{\\mathrm{H3}}(H_{\\mathrm{eq}})]$", fontsize=9.8)
    ax_a.set_title("(a) Thick-film response and extrapolation", loc="left", fontsize=10.5, fontweight="bold")
    ax_a.grid(True, ls=":", alpha=0.32)
    ax_a.legend(loc="lower left", fontsize=8.2, frameon=True, facecolor="white")

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
            hv + 0.24,
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
            hv + 0.24,
            f"{hv:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#90301B",
        )
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(ab_cats, fontsize=8.6)
    ax_b.set_ylim(0.0, max(max(k5_ab), max(k8_ab)) * 1.15)
    ax_b.set_ylabel("Mean absolute endpoint error (%)", fontsize=9.8)
    ax_b.set_title("(b) Effect of omitting individual response terms", loc="left", fontsize=10.5, fontweight="bold")
    ax_b.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    ax_b.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white")

    f10_path = out_dir / "Fig_10_ThickFilm_and_Ablation.png"
    fig10.savefig(f10_path, dpi=300, facecolor="white")
    plt.close(fig10)
    saved_figs.append(f10_path)

    # 3. Supplementary Figs. S3, S4, S5 (matching SI figure numbering exactly)
    saved_figs.append(
        _plot_supplementary_group(
            out_dir, all_series, TABLE8_GROUPS[0][1], "Fig_S3_Laboratory_Baselines_Trajectories.png", False, 4, plt
        )
    )
    saved_figs.append(
        _plot_supplementary_group(
            out_dir, all_series, TABLE8_GROUPS[1][1], "Fig_S4_Direct_DataDriven_Trajectories.png", True, 4, plt
        )
    )
    saved_figs.append(
        _plot_supplementary_group(
            out_dir, all_series, TABLE8_GROUPS[2][1], "Fig_S5_Domain_Adaptation_Trajectories.png", True, 3, plt
        )
    )
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
    print("KGED STAGE 5: ONE-CLICK CROSS-ENVIRONMENT CORROSION PREDICTION")
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
        f"[Step 1/4] Closed-form & quadrature verified (reference |dlog i| = {max(cf_max.values()):.2e}; "
        f"16- vs 32-pt Gauss-Legendre max rel |dB| = {max(quad_rel_max.values()):.2e})"
    )

    # Export SI Table S4 parameter summary
    written_files.append(
        write_csv(
            out_dir / "Table_S4_Parameter_Provenance.csv",
            [
                {"parameter": "RH0_pct", "value": DEFAULT_MAPPING["RH0"], "role": "Wetting threshold (Eq. 11)"},
                {"parameter": "alpha", "value": DEFAULT_MAPPING["alpha"], "role": "Wetting exponent (Eq. 11)"},
                {"parameter": "J0_mg_m2_day", "value": DEFAULT_MAPPING["J0"], "role": "Chloride half-saturation flux (Eq. 11)"},
                {"parameter": "Hmin_um", "value": DEFAULT_MAPPING["Hmin"], "role": "Minimum effective film thickness"},
                {"parameter": "Hmax_um", "value": DEFAULT_MAPPING["Hmax"], "role": "Maximum effective film thickness"},
                {"parameter": "Cbg_wt_pct", "value": round(float(mapping["Cbg"]), 6), "role": "Background effective NaCl concentration"},
                {"parameter": "Ccap_wt_pct", "value": round(float(mapping["Ccap"]), 6), "role": "Saturation cap effective NaCl concentration"},
                {"parameter": "tau_u", "value": state_cfg["tau"], "role": "Fixed characteristic dose scale"},
                {"parameter": "lambda_a", "value": state_cfg["lam"], "role": "Fixed Tikhonov smoothness weight"},
                {"parameter": "Gamma_um_per_A_s_cm2", "value": round(float(mapping["Gamma"]), 6), "role": "Fe(II) Faraday conversion factor"},
            ],
        )
    )

    # ------------------------------------------------------------------
    # STEP 2: Source-site BVLS calibration at Linhai (Table 7)
    # ------------------------------------------------------------------
    lh_months = np.asarray(labels["linhai"]["months"], float)
    lh_obs = np.asarray(labels["linhai"]["observed_depth_um"], float)

    table7_rows = []
    kged_live = {}
    print("\n[Step 2/4] Table 7 -- Source-Calibrated Cubic Bernstein State Amplitudes at Linhai (BVLS)")
    print("-" * 90)
    print(f"{'Equation':<28s} {'a0':>10s} {'a1':>10s} {'a2':>10s} {'a3':>10s} {'Calib RMSE (um)':>17s}")
    print("-" * 90)
    for m_key, disp_name, law in [
        ("kged5", "Five-parameter KGED", law5),
        ("kged8", "Eight-parameter KGED", law8),
    ]:
        integ = Integrator(env_data, law, mapping)
        fitted = state.fit(integ, state_cfg, lh_obs, lh_months, len(lh_obs))
        require(bool(fitted["eligible"]), f"Linhai BVLS source calibration ineligible for {m_key}")
        a_vec = np.asarray(fitted["coefficients"], float)
        calib_rmse = float(np.sqrt(fitted["source_SSE_um2"] / len(lh_obs)))
        kged_live[m_key] = (integ, a_vec)
        table7_rows.append(
            {
                "Equation": disp_name,
                "a0": round(float(a_vec[0]), 6),
                "a1": round(float(a_vec[1]), 6),
                "a2": round(float(a_vec[2]), 6),
                "a3": round(float(a_vec[3]), 6),
                "unanchored_source_rmse_um": round(calib_rmse, 4),
            }
        )
        print(f"{disp_name:<28s} {a_vec[0]:>10.3f} {a_vec[1]:>10.3f} {a_vec[2]:>10.3f} {a_vec[3]:>10.3f} {calib_rmse:>17.2f}")
    print("-" * 90)
    written_files.append(write_csv(out_dir / "Table_7_Source_State_Amplitudes.csv", table7_rows))

    # ------------------------------------------------------------------
    # STEP 3: Live integration of 7 equation-based methods + re-scoring of
    #         11 frozen ML/DA baselines, with 18/18 Table 8 verification
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
        require(bool(sr_fit["eligible"]), f"Linhai BVLS fit ineligible for symbolic baseline {sr_key}")
        a_sr = np.asarray(sr_fit["coefficients"], float)
        live_series[sr_key] = {}
        for c in CASES:
            m_arr = np.asarray(labels[c]["months"], float)
            y_arr = np.asarray(labels[c]["observed_depth_um"], float)
            raw = sr_integ.basis(c, state_cfg, 16) @ a_sr
            p_cond = y_arr[0] + (raw - raw[0])
            live_series[sr_key][c] = evaluate_conditioned_series(m_arr, y_arr, p_cond)

    # Empirical power-law model fitted from scratch on Linhai: OLS of ln(D) on ln(t)
    n_pow, ln_a_pow = np.polyfit(np.log(lh_months), np.log(lh_obs), 1)
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
    for grp_name, members in TABLE8_GROUPS:
        for m_key, m_label in members:
            all_series[m_key] = {}
            for c in CASES:
                pts = archived_pts[m_key][c]
                m_arch = np.asarray([float(x["month"]) for x in pts], float)
                y_arch = np.asarray([float(x["observed_depth_um"]) for x in pts], float)
                p_arch = np.asarray([float(x["conditioned_depth_um"]) for x in pts], float)
                if m_key in LIVE_METHODS:
                    max_traj_diff = float(np.max(np.abs(live_series[m_key][c]["conditioned_depth_um"] - p_arch)))
                    require(
                        max_traj_diff < 1e-8 * max(1.0, float(np.max(np.abs(p_arch)))),
                        f"Live vs archived trajectory mismatch on {m_key}/{c}: {max_traj_diff:.2e}",
                    )
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

    print("\n[Step 3/4] Table 8 -- Target-Environment Trajectory RMSE and Endpoint Errors (Eq. 16)")
    print("-" * 108)
    print(
        f"{'Mode Method':<34s} {'Jinhai RMSE':>11s} {'Jinhai EP%':>10s} "
        f"{'Pool RMSE':>10s} {'Pool EP%':>10s} {'KSC RMSE':>10s} {'KSC EP%':>10s} {'Mean |EP|%':>11s}"
    )
    print("-" * 108)

    table8_rows = []
    diverged_raw_rows = []
    formatted_cells: dict[str, tuple[str, ...]] = {}
    md_lines = [
        "# Table 8. Cross-environment corrosion-depth prediction accuracy (Eq. 16)",
        "",
        "| Method | Jinhai RMSE (μm) | Jinhai EP (%) | Swimming pool RMSE (μm) | Swimming pool EP (%) | NASA KSC RMSE (μm) | NASA KSC EP (%) | Mean \\|EP\\| (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for grp_name, members in TABLE8_GROUPS:
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

            print(f"  {tag} {m_label:<28s} {jr_s:>11s} {je_s:>10s} {pr_s:>10s} {pe_s:>10s} {kr_s:>10s} {ke_s:>10s} {mean_str:>11s}")
            md_lines.append(f"| {md_name} | {jr_s} | {je_s} | {pr_s} | {pe_s} | {kr_s} | {ke_s} | {mean_str} |")

            table8_rows.append(
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

    print("-" * 108)
    print("  [L] = recalibrated at Linhai and integrated live from scratch in this run (7 equation-based methods).")
    print("  [A] = frozen archived ML / domain-adaptation predictions re-scored under Eq. (16) (11 methods;")
    print("        training script & 14D features provided in engineering/comparators/).")

    mismatches = []
    for m_key, expected in MANUSCRIPT_TABLE8.items():
        exp_tuple = tuple("Diverged" if v is None else v for v in expected)
        if formatted_cells[m_key] != exp_tuple:
            mismatches.append((m_key, formatted_cells[m_key], exp_tuple))
    require(not mismatches, f"Table 8 cell mismatch against manuscript: {mismatches}")
    print("  [VERIFIED] 18/18 Table 8 rows (all 126 cells) match the published manuscript exactly.")

    md_lines += [
        "",
        "† Frozen predictions loaded from `engineering/Table_conditioned_predictions.csv` and re-scored under Eq. (16) "
        "(see `engineering/comparators/reproduce_field_comparators.py` for standalone training code).",
        "",
        f"*Diverged*: cell-wise RMSE > {DIVERGENCE_RULE['rmse_um_max']:.0f} μm or |endpoint error| > "
        f"{DIVERGENCE_RULE['abs_endpoint_pct_max']:.0f}% (Table 8 Footnote a; raw values in `Diverged_Models_Raw_Metrics.csv`).",
    ]
    written_files.append(write_csv(out_dir / "Table_8_Engineering_Comparison.csv", table8_rows))
    written_files.append(write_csv(out_dir / "Diverged_Models_Raw_Metrics.csv", diverged_raw_rows))
    md_path = out_dir / "Table_8_Engineering_Comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    written_files.append(md_path)

    # ------------------------------------------------------------------
    # STEP 4: Component-wise rate-law ablations (12 combinations, Fig. 10b & SI M1.5.1)
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
    print("\n[Step 4/4] Component-Wise Hypothesis Ablations (12 Combinations, Fig. 10b & SI M1.5.1)")
    print("-" * 98)
    print(f"{'Rate Law':<26s} {'Variant':<34s} {'Jinhai EP%':>10s} {'Pool EP%':>10s} {'KSC EP%':>9s} {'Mean |EP|%':>11s}")
    print("-" * 98)
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
                f"{eq_label:<26s} {var_name:<34s} "
                f"{s_metrics['jinhai'][1]:>+9.2f}% {s_metrics['pool'][1]:>+9.2f}% "
                f"{s_metrics['ksc'][1]:>+8.2f}% {mean_ep:>10.2f}%"
            )
    print("-" * 98)
    written_files.append(write_csv(out_dir / "Rate_Law_Ablations_12Rows.csv", ablation_rows))

    saved_figs: list[Path] = []
    if not skip_plots:
        saved_figs = render_figures(out_dir, all_series, ablation_rows, r5_json, r8_json, kged_live["kged5"][0])
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
        "manuscript_table8_rows_verified": "18/18",
        "divergence_rule": DIVERGENCE_RULE,
        "computation_modes": {
            "live_from_scratch": sorted(LIVE_METHODS),
            "archived_frozen_prediction": sorted(set(MANUSCRIPT_TABLE8) - LIVE_METHODS),
        },
        "kged_5param_target_mean_abs_endpoint_error_pct": round(
            float(np.mean([abs(all_series["kged5"][c]["endpoint_error_pct"]) for c in TARGET_SITES])), 4
        ),
        "kged_8param_target_mean_abs_endpoint_error_pct": round(
            float(np.mean([abs(all_series["kged8"][c]["endpoint_error_pct"]) for c in TARGET_SITES])), 4
        ),
        "empirical_power_law_fit_linhai": {"A": round(a_pow, 6), "n": round(n_pow, 6)},
        "table7_amplitudes": table7_rows,
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
        print(f"Generated {len(saved_figs)} high-resolution figures (300 DPI PNG: Figs. 9, 10, S3, S4, S5).")
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
