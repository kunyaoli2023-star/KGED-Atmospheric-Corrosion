"""Reproduce the cross-environment engineering calibration (Linhai source site, Table 7)
and initial-depth-anchored target trajectory predictions (Jinhai, indoor swimming pool,
and NASA KSC, Eq. (16) and Table 8) from the frozen laboratory KGED rate equations."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engineering.engineering_integrator import CASES, Integrator, metrics
from engineering import engineering_state as state
from engineering.verify_closed_forms import ClosedFormLaw

read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
save = lambda p, obj: Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(out=None):
    field = ROOT / "inputs"
    chosen = {
        "parsimonious": read(ROOT / "parsimonious_equation.json"),
        "best_accuracy": read(ROOT / "best_accuracy_equation.json"),
    }
    lab = read(REPO_ROOT / "data/data.json")
    mapping = read(field / "mapping.json")
    data = {}
    for c in CASES:
        with np.load(field / f"{c}_environment.npz") as z:
            data[c] = {k: z[k].copy() for k in z.files}

    source_labels = read(field / "linhai_labels.json")
    months = np.asarray(source_labels["months"], float)
    observed = np.asarray(source_labels["observed_depth_um"], float)
    np.testing.assert_array_equal(months, data["linhai"]["months"])

    labels = {c: read(field / f"{c}_labels.json") for c in CASES}
    scores = []
    amplitudes_summary = {}

    for alias, result in chosen.items():
        law = ClosedFormLaw(alias, result)
        reconstructed = law.predict_log(lab["T_C"], lab["C_wt"], lab["h_um"])
        np.testing.assert_allclose(reconstructed, result["full_predictions"], atol=2e-11, rtol=0)
        model = Integrator(data, law, mapping)
        fitted = state.fit(model, mapping["fixed_state_structure"], observed, months, len(observed))
        if not fitted["eligible"]:
            raise ValueError("SOURCE_FIT_INELIGIBLE_" + alias)
        a_vec = np.asarray(fitted["coefficients"], float)
        amplitudes_summary[alias] = [round(float(x), 3) for x in a_vec]

        for c in CASES:
            b16 = model.basis(c, mapping["fixed_state_structure"], 16)
            b32 = model.basis(c, mapping["fixed_state_structure"], 32)
            np.testing.assert_allclose(b16, b32, atol=2e-9, rtol=2e-12)
            raw_pred = b16 @ a_vec
            y = np.asarray(labels[c]["observed_depth_um"], float)
            if c == "linhai":
                yp = raw_pred
                depth_rmse = float(np.sqrt(np.mean((yp - y) ** 2)))
                scores.append({
                    "model": alias,
                    "case": c,
                    "role": "source_calibration",
                    **metrics(y, yp),
                    "depth_rmse_um": depth_rmse,
                })
            else:
                # Eq. (16): initial-depth-anchored target prediction
                yp = y[0] + (raw_pred - raw_pred[0])
                err_post = yp[1:] - y[1:]
                depth_rmse = float(np.sqrt(np.mean(err_post ** 2)))
                ep_err = float((yp[-1] - y[-1]) / y[-1] * 100.0)
                scores.append({
                    "model": alias,
                    "case": c,
                    "role": "retrospective_target_prediction",
                    "evaluation_protocol": "Eq_16_initial_depth_anchored",
                    "actual_final_depth_um": float(y[-1]),
                    "predicted_final_depth_um": float(yp[-1]),
                    "endpoint_error_pct": ep_err,
                    "depth_rmse_um": depth_rmse,
                })

    summary = {"status": "COMPLETE", "Table7_amplitudes": amplitudes_summary, "scores": scores}
    if out is not None:
        out = Path(out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        save(out / "scores_recomputed.json", scores)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Optional directory to write recomputed scores JSON")
    args = parser.parse_args()
    run(args.out)
