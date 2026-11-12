# -*- coding: utf-8 -*-
"""Apply Eq. (16) initial-depth conditioning to archived raw engineering growth forecasts.

Can be executed directly with zero arguments to verify conditioning in-memory, or with
``--out <dir>`` to export conditioned JSONs, CSV tables, Markdown summary, and plots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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

from engineering.engineering_integrator import CASES, condition_on_initial_depth, metrics

MODELS = [
    ("parsimonious", "#2474b5", "Five-parameter KGED"),
    ("best_accuracy", "#bc4b32", "Eight-parameter KGED"),
]


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: Path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source: Path = ROOT, out: Path | None = None):
    source = Path(source).resolve()
    original_path = source / "PREDICTIONS_FROZEN.json"
    original_sha = sha(original_path)
    original = read(original_path)
    labels = {c: read(source / "inputs" / f"{c}_labels.json") for c in CASES}
    initials = {
        c: {"month": labels[c]["months"][0], "depth_um": labels[c]["observed_depth_um"][0]}
        for c in CASES
    }

    rel_source = (
        (original_path.relative_to(REPO_ROOT)).as_posix()
        if original_path.is_relative_to(REPO_ROOT)
        else original_path.name
    )
    frozen = {
        "status": "FIRST_OBSERVATION_CONDITIONED_FORECAST",
        "source_run": rel_source,
        "source_forecast_sha256": original_sha,
        "formula": "D_cond(t) = D_obs(t1) + D_raw(t) - D_raw(t1)",
        "state_clock": "Original exposure clock retained; no reset at t1.",
        "target_initialization_observations_per_site": 1,
        "target_parameter_fit_observations": 0,
        "coefficient_refitting": False,
        "initial_conditions": initials,
        "mapping": original["mapping"],
        "models": {},
        "scope": (
            "Initial-depth-anchored conditioning of source-calibrated forecasts (Eq. 16). "
            "Zero target-site kinetic or state parameters are refitted."
        ),
    }

    verification = []
    for alias, _, _ in MODELS:
        old = original["models"][alias]
        predictions = {}
        for case in CASES:
            series = old["predictions"][case]
            np.testing.assert_array_equal(series["months"], labels[case]["months"])
            raw = np.asarray(series["predicted_depth_um"], float)
            conditioned = condition_on_initial_depth(raw, initials[case]["depth_um"])
            np.testing.assert_equal(conditioned[0], initials[case]["depth_um"])
            np.testing.assert_allclose(np.diff(conditioned), np.diff(raw), rtol=1e-12, atol=1e-12)
            if np.any(conditioned < 0) or np.any(np.diff(conditioned) < -1e-9):
                raise ValueError("INVALID_CONDITIONED_DEPTH")
            predictions[case] = {
                "months": series["months"],
                "predicted_depth_um": conditioned.tolist(),
            }
            verification.append(
                {"model": alias, "case": case, "first_point_exact": True, "growth_preserved": True}
            )
        frozen["models"][alias] = {
            "equation_key": old["equation_key"],
            "lab_score": old["lab_score"],
            "source_fit": old["source_fit"],
            "predictions": predictions,
        }

    archived_cond_path = source / "current_conditioned_forecast.json"
    if archived_cond_path.exists():
        archived_cond = read(archived_cond_path)
        for alias, _, _ in MODELS:
            for case in CASES:
                np.testing.assert_allclose(
                    frozen["models"][alias]["predictions"][case]["predicted_depth_um"],
                    archived_cond["models"][alias]["predictions"][case]["predicted_depth_um"],
                    atol=1e-12,
                    rtol=0,
                )

    scores, prediction_rows = [], []
    for case in CASES:
        y = np.asarray(labels[case]["observed_depth_um"], float)
        if len(y) < 2:
            raise ValueError("NO_FUTURE_SCORING_POINTS")
        for alias, _, _ in MODELS:
            pred = np.asarray(frozen["models"][alias]["predictions"][case]["predicted_depth_um"], float)
            raw = np.asarray(original["models"][alias]["predictions"][case]["predicted_depth_um"], float)
            residual = pred[1:] - y[1:]
            denom = float(np.sum((y[1:] - y[1:].mean()) ** 2))
            scores.append(
                {
                    "case": case,
                    "model": alias,
                    "role": "source_calibration_conditioned"
                    if case == "linhai"
                    else "target_first_observation_conditioned",
                    **metrics(y, pred),
                    "future_scored_points": len(y) - 1,
                    "future_depth_rmse_um": float(np.sqrt(np.mean(residual ** 2))),
                    "future_absolute_r2": float(1.0 - np.sum(residual ** 2) / denom) if denom else None,
                    "raw_future_depth_rmse_um": float(np.sqrt(np.mean((raw[1:] - y[1:]) ** 2))),
                    "raw_endpoint_error_pct": float(100.0 * (raw[-1] / y[-1] - 1.0)),
                }
            )
            for i, month in enumerate(labels[case]["months"]):
                prediction_rows.append(
                    {
                        "case": case,
                        "model": alias,
                        "month": month,
                        "observed_depth_um": float(y[i]),
                        "predicted_depth_um": float(pred[i]),
                        "raw_predicted_depth_um": float(raw[i]),
                        "is_initial_condition": i == 0,
                    }
                )

    assert sha(original_path) == original_sha
    macro = {
        alias: round(
            float(
                np.mean(
                    [abs(s["endpoint_error_pct"]) for s in scores if s["model"] == alias and s["case"] != "linhai"]
                )
            ),
            4,
        )
        for alias, _, _ in MODELS
    }

    if out is not None:
        out = Path(out).resolve()
        if out == source or out == ROOT or out == ROOT / "inputs":
            raise SystemExit("Refusing to write into the archived engineering/ directory; choose a separate --out directory.")
        out.mkdir(parents=True, exist_ok=True)
        inputs = out / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        save(inputs / "initial_conditions.json", initials)
        for alias, _, _ in MODELS:
            shutil.copy2(source / f"{alias}_equation.json", out / f"{alias}_equation.json")
        for case in CASES:
            shutil.copy2(source / "inputs" / f"{case}_labels.json", inputs / f"{case}_labels.json")
        save(out / "PREDICTIONS_CONDITIONED.json", frozen)
        conditioned_sha = sha(out / "PREDICTIONS_CONDITIONED.json")
        save(out / "scores.json", scores)
        write_csv(out / "scores.csv", scores)
        write_csv(out / "predictions.csv", prediction_rows)
        save(
            out / "VERIFICATION.json",
            {
                "original_forecast_unchanged": True,
                "source_forecast_sha256": original_sha,
                "conditioned_forecast_sha256": conditioned_sha,
                "matches_archived_conditioned_forecast": archived_cond_path.exists(),
                "checks": verification,
                "target_macro_endpoint_ape_pct": macro,
                "input_sha256": {p.name: sha(p) for p in sorted(inputs.iterdir()) if p.is_file()},
            },
        )
        snapshot = out / "implementation_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        for p in (Path(__file__), ROOT / "engineering_integrator.py"):
            shutil.copy2(p, snapshot / p.name)
        render(out, frozen, labels, scores)
        lines = [
            "# Initial-Depth-Anchored Cross-Environment Engineering Predictions (Eq. 16)",
            "",
            r"$$D_{\mathrm{pred}}(t) = D_{\mathrm{obs}}(t_1) + \widehat{D}(t) - \widehat{D}(t_1)$$",
            "",
            "Each target trajectory is anchored at the first observed depth $D_{\\mathrm{obs}}(t_1)$ while preserving the reaction-clock exposure history and zero target-site parameter refitting.",
            "",
            "| Site | Model | Unanchored EP (%) | Eq. (16) EP (%) | Post-t1 RMSE (μm) | Unanchored RMSE (μm) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for s in scores:
            lines.append(
                f"| {s['case']} | {s['model']} | {s['raw_endpoint_error_pct']:+.2f} | "
                f"{s['endpoint_error_pct']:+.2f} | {s['future_depth_rmse_um']:.3f} | {s['raw_future_depth_rmse_um']:.3f} |"
            )
        lines += [
            "",
            f"Three-site mean absolute endpoint error: Five-parameter KGED = {macro['parsimonious']:.2f}%; Eight-parameter KGED = {macro['best_accuracy']:.2f}%.",
            "",
            "![Initial-depth-anchored engineering trajectories](ENGINEERING_FIRST_OBSERVATION.png)",
            "",
        ]
        (out / "ENGINEERING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    result = {
        "status": "COMPLETE",
        "matches_archived_conditioned_forecast": archived_cond_path.exists(),
        "target_mean_abs_endpoint_error_pct": macro,
    }
    if out is not None:
        result["output_dir"] = (
            out.relative_to(REPO_ROOT).as_posix() if out.is_relative_to(REPO_ROOT) else str(out)
        )
    print(json.dumps(result, indent=2))
    return result


def render(out, frozen, labels, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.16, hspace=0.36, wspace=0.28)
    names = {
        "linhai": "Linhai (source calibration)",
        "jinhai": "Jinhai",
        "pool": "Indoor swimming pool",
        "ksc": "NASA KSC",
    }
    for index, (ax, case) in enumerate(zip(axes.flat, CASES)):
        month, depth = labels[case]["months"], labels[case]["observed_depth_um"]
        ax.plot(month[1:], depth[1:], "ko", ms=4, label="Observed")
        for alias, color, title in MODELS:
            p = frozen["models"][alias]["predictions"][case]
            ax.plot(
                month,
                p["predicted_depth_um"],
                color=color,
                lw=1.9,
                ls="-" if alias == "parsimonious" else "--",
                label=title,
            )
        ax.plot(
            month[0],
            depth[0],
            marker="D",
            color="black",
            mfc="white",
            ms=5.5,
            linestyle="none",
            label="Initial depth $D_{\\mathrm{obs}}(t_1)$",
            zorder=5,
        )
        site_scores = {s["model"]: s for s in scores if s["case"] == case}
        text = "\n".join(
            f"{title}: RMSE {site_scores[alias]['future_depth_rmse_um']:.2f} μm ({site_scores[alias]['endpoint_error_pct']:+.2f}%)"
            for alias, _, title in MODELS
        )
        ax.text(
            0.04,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            fontsize=7.8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        ax.set_title(f"({chr(97 + index)}) {names[case]}", loc="left", fontsize=10.5, fontweight="bold")
        top = max(depth + frozen["models"]["best_accuracy"]["predictions"][case]["predicted_depth_um"])
        ax.set(xlabel="Exposure time (months)", ylabel="Corrosion depth (μm)", ylim=(0, top * 1.28))
        ax.grid(alpha=0.2)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.52, 0.02), ncol=4, frameon=False, fontsize=8.5)
    fig.savefig(out / "ENGINEERING_FIRST_OBSERVATION.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT, help="Engineering directory containing PREDICTIONS_FROZEN.json")
    parser.add_argument("--out", type=Path, default=None, help="Optional output directory to write conditioned tables and plots")
    args = parser.parse_args()
    run(args.source, args.out)
