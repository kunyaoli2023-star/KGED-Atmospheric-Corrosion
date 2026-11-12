# -*- coding: utf-8 -*-
"""Reproduce the cross-environment engineering calibration (Linhai source site, Table 7)
and initial-depth-anchored target trajectory predictions (Jinhai, indoor swimming pool,
and NASA KSC, Eq. (16) and Table 8) from the frozen laboratory KGED rate equations."""
from __future__ import annotations

import argparse
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

from engineering.engineering_integrator import CASES, Integrator, metrics
from engineering import engineering_state as state
from engineering.verify_closed_forms import CLOSED_FORM_ATOL, QUADRATURE_REL_TOL, ClosedFormLaw, require

read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
save = lambda p, obj: Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(out: Path | None = None):
    field = ROOT / "inputs"
    chosen = {
        "parsimonious": read(ROOT / "parsimonious_equation.json"),
        "best_accuracy": read(ROOT / "best_accuracy_equation.json"),
    }
    mapping = read(field / "mapping.json")
    data = {}
    for c in CASES:
        with np.load(field / f"{c}_environment.npz") as z:
            data[c] = {k: z[k].copy() for k in z.files}

    source_labels = read(field / "linhai_labels.json")
    months = np.asarray(source_labels["months"], float)
    observed = np.asarray(source_labels["observed_depth_um"], float)
    require(np.array_equal(months, data["linhai"]["months"]), "Linhai label months do not match environment months.")

    labels = {c: read(field / f"{c}_labels.json") for c in CASES}
    scores = []
    amplitudes_summary = {}

    for alias, result in chosen.items():
        law = ClosedFormLaw(alias, result)
        ref_pred = float(law.predict_log(25.0, 3.5, 200.0)[0])
        cf_err = abs(ref_pred - float(result["intercept"]))
        require(cf_err < CLOSED_FORM_ATOL, f"Closed-form reference mismatch for {alias}: {cf_err:.2e}")
        model = Integrator(data, law, mapping)
        fitted = state.fit(model, mapping["fixed_state_structure"], observed, months, len(observed))
        require(bool(fitted["eligible"]), f"SOURCE_FIT_INELIGIBLE_{alias}")
        a_vec = np.asarray(fitted["coefficients"], float)
        amplitudes_summary[alias] = [round(float(x), 3) for x in a_vec]

        for c in CASES:
            b16 = model.basis(c, mapping["fixed_state_structure"], 16)
            b32 = model.basis(c, mapping["fixed_state_structure"], 32)
            q_rel = float(np.max(np.abs(b16 - b32))) / max(1.0, float(np.max(np.abs(b32))))
            require(q_rel < QUADRATURE_REL_TOL, f"Quadrature relative diff too large on {alias}/{c}: {q_rel:.2e}")
            raw_pred = b16 @ a_vec
            y = np.asarray(labels[c]["observed_depth_um"], float)
            if c == "linhai":
                yp = raw_pred
                depth_rmse = float(np.sqrt(np.mean((yp - y) ** 2)))
                scores.append(
                    {
                        "model": alias,
                        "case": c,
                        "role": "source_calibration",
                        **metrics(y, yp),
                        "depth_rmse_um": depth_rmse,
                    }
                )
            else:
                yp = y[0] + (raw_pred - raw_pred[0])
                err_post = yp[1:] - y[1:]
                depth_rmse = float(np.sqrt(np.mean(err_post ** 2)))
                ep_err = float((yp[-1] - y[-1]) / y[-1] * 100.0)
                scores.append(
                    {
                        "model": alias,
                        "case": c,
                        "role": "retrospective_target_prediction",
                        "evaluation_protocol": "Eq_16_initial_depth_anchored",
                        "actual_final_depth_um": float(y[-1]),
                        "predicted_final_depth_um": float(yp[-1]),
                        "endpoint_error_pct": ep_err,
                        "depth_rmse_um": depth_rmse,
                    }
                )

    summary = {"status": "COMPLETE", "Table7_amplitudes": amplitudes_summary, "scores": scores}
    if out is not None:
        out = Path(out).resolve()
        require(out != ROOT, "Refusing to overwrite archived engineering/ directory; choose a subdirectory for --out.")
        out.mkdir(parents=True, exist_ok=True)
        save(out / "scores_recomputed.json", scores)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Optional directory to write recomputed scores JSON")
    args = parser.parse_args()
    run(args.out)
