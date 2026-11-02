"""Order-robust, multi-resolution saturation audit of the 50-paper corpus."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CORPUS = ROOT / "llm方程"
OLD = CORPUS / "extracted_formulas_vision_old"


def paper_key(source: str) -> str:
    return source.removesuffix(".json")


def locate_paper(formula_id: str, papers: list[str]) -> int | None:
    matches = [index for index, name in enumerate(papers) if formula_id.startswith(name + "_Eq")]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous formula id: {formula_id}")
    return matches[0] if matches else None


def load_units(papers: list[str]) -> dict[str, list[set[str]]]:
    output = {name: [set() for _ in papers] for name in ("coarse", "medium", "fine")}
    semantic = json.loads((OLD / "llm_semantic_classification_safe.json").read_text(encoding="utf-8"))
    for row in semantic:
        signature = row.get("signature")
        if row.get("is_kinetic_model") is True and signature:
            index = locate_paper(row["id"], papers)
            if index is not None:
                output["coarse"][index].add(signature)
    clusters = json.loads((CORPUS / "IccorAgent" / "icorr_meta_models.json").read_text(encoding="utf-8"))
    for cluster in clusters["clusters"]:
        unit = cluster["cluster_name"]
        for formula_id in cluster["formula_ids_included"]:
            index = locate_paper(formula_id, papers)
            if index is not None:
                output["medium"][index].add(unit)
    fine = json.loads((OLD / "merged_signatures.json").read_text(encoding="utf-8"))
    mapping = fine.get("id_to_coarse_signature", fine.get("id_to_signature", {}))
    for formula_id, signature in mapping.items():
        if not signature:
            continue
        index = locate_paper(formula_id, papers)
        if index is not None:
            output["fine"][index].add(signature)
    return output


def chao2(incidence: Counter, observed: int) -> tuple[float, int, int]:
    q1 = sum(value == 1 for value in incidence.values())
    q2 = sum(value == 2 for value in incidence.values())
    estimate = observed + q1 * (q1 - 1) / (2 * (q2 + 1))
    return float(max(observed, estimate)), q1, q2


def analyse(name: str, by_paper: list[set[str]], protocol: dict, rng: np.random.Generator):
    n_papers = len(by_paper)
    all_units = set().union(*by_paper)
    incidence = Counter(unit for units in by_paper for unit in units)
    estimated, q1, q2 = chao2(incidence, len(all_units))
    replicates = protocol["permutation_replicates"]
    curves = np.empty((replicates, n_papers), dtype=np.int16)
    for replicate in range(replicates):
        seen: set[str] = set()
        for position, index in enumerate(rng.permutation(n_papers)):
            seen.update(by_paper[int(index)])
            curves[replicate, position] = len(seen)
    mean = curves.mean(axis=0)
    low, high = np.quantile(curves, [.025, .975], axis=0)
    marginal = np.diff(np.column_stack([np.zeros(replicates), curves]), axis=1).mean(axis=0)
    threshold = protocol["coverage_threshold"] * len(all_units)
    probability_coverage = np.mean(curves >= threshold, axis=0)
    saturation = None
    for position in range(n_papers):
        next_stop = min(n_papers, position + 6)
        next_gain = float(np.mean(marginal[position:next_stop]))
        if (mean[position] >= threshold and next_gain <= protocol["next_five_mean_marginal_threshold"]
                and probability_coverage[position] >= .95):
            saturation = position + 1
            break
    seen, observed_curve, observed_marginal = set(), [], []
    for units in by_paper:
        before = len(seen); seen.update(units)
        observed_curve.append(len(seen)); observed_marginal.append(len(seen)-before)
    completeness = len(all_units)/estimated if estimated else 1.0
    passes = bool(saturation is not None and completeness >= protocol["minimum_estimated_completeness"])
    rows = [{"resolution": name, "papers": i+1, "observed_order_units": observed_curve[i],
             "observed_order_marginal": observed_marginal[i], "permutation_mean_units": float(mean[i]),
             "permutation_ci95_low": float(low[i]), "permutation_ci95_high": float(high[i]),
             "permutation_mean_marginal": float(marginal[i]),
             "probability_at_least_95pct_observed_coverage": float(probability_coverage[i])}
            for i in range(n_papers)]
    result = {"resolution": name, "observed_unique_units": len(all_units),
              "singleton_units_q1": q1, "doubleton_units_q2": q2,
              "chao2_estimated_units": estimated, "estimated_completeness": completeness,
              "actual_order_last_novel_paper": max((i+1 for i, gain in enumerate(observed_marginal) if gain), default=None),
              "actual_order_new_units_last_10_papers": int(sum(observed_marginal[-10:])),
              "order_robust_saturation_paper_count": saturation, "passes_saturation_rule": passes}
    return result, rows


def main() -> None:
    protocol = json.loads((HERE / "KNOWLEDGE_SATURATION_PROTOCOL.json").read_text(encoding="utf-8"))
    out = HERE / "results" / "knowledge_saturation_audit_v1"
    if out.exists():
        raise SystemExit("refusing to overwrite knowledge saturation audit")
    out.mkdir(parents=True)
    merged = json.loads((OLD / "all_formulas_merged.json").read_text(encoding="utf-8"))
    papers = [paper_key(row["source_file"]) for row in merged]
    if len(papers) != 50 or len(set(papers)) != 50:
        raise RuntimeError("expected 50 unique source papers")
    units = load_units(papers)
    rng = np.random.default_rng(protocol["seed"])
    summaries, curve_rows = {}, []
    for name in ("coarse", "medium", "fine"):
        summaries[name], rows = analyse(name, units[name], protocol, rng)
        curve_rows.extend(rows)
    joint = all(row["passes_saturation_rule"] for row in summaries.values())
    result = {"schema": "rhe_v3_multiresolution_knowledge_saturation_v1",
              "paper_count": len(papers), "formula_count": sum(len(row.get("formulas", [])) for row in merged),
              "summaries": summaries, "joint_knowledge_saturation": joint,
              "interpretation": ("The curated corpus is saturated at every audited resolution." if joint else
                  "The curated corpus is not jointly saturated across resolutions; a coarse plateau cannot justify global knowledge closure."),
              "claim_boundary": protocol["claim_boundary"]}
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0])); writer.writeheader(); writer.writerows(curve_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=180, constrained_layout=True)
    for axis, name in zip(axes, ("coarse", "medium", "fine")):
        rows = [row for row in curve_rows if row["resolution"] == name]
        x = np.asarray([row["papers"] for row in rows]); mean = np.asarray([row["permutation_mean_units"] for row in rows])
        low = np.asarray([row["permutation_ci95_low"] for row in rows]); high = np.asarray([row["permutation_ci95_high"] for row in rows])
        observed = np.asarray([row["observed_order_units"] for row in rows])
        axis.fill_between(x, low, high, color="#8fbad9", alpha=.35, label="95% order interval")
        axis.plot(x, mean, color="#1769aa", lw=2, label="random-order mean")
        axis.plot(x, observed, color="#d1495b", lw=1.3, ls="--", label="stored order")
        axis.axhline(.95*summaries[name]["observed_unique_units"], color="#555", lw=1, ls=":")
        axis.set_title(f"{name}: {summaries[name]['observed_unique_units']} units")
        axis.set_xlabel("Papers sampled"); axis.grid(alpha=.2)
    axes[0].set_ylabel("Cumulative unique knowledge units"); axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Multi-resolution knowledge saturation audit")
    fig.savefig(out / "multiresolution_saturation.png", bbox_inches="tight")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
