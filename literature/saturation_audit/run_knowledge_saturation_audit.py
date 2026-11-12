"""Order-robust corrosion-model category saturation audit of the 50-paper corpus (Section M1.1.2 & Fig. S1)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LIT_DIR = REPO_ROOT / "literature"
KG_DIR = REPO_ROOT / "knowledge_graph"


def paper_key(source: str) -> str:
    return source.removesuffix(".json")


def locate_paper(formula_id: str, papers: list[str]) -> int | None:
    matches = [index for index, name in enumerate(papers) if formula_id.startswith(name + "_Eq")]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous formula id: {formula_id}")
    return matches[0] if matches else None


def load_category_units(papers: list[str]) -> tuple[list[set[str]], int]:
    by_paper = [set() for _ in papers]
    clusters = json.loads((KG_DIR / "literature_graph" / "icorr_meta_models.json").read_text(encoding="utf-8"))
    eq_count = 0
    for cluster in clusters["clusters"]:
        unit = cluster["cluster_name"]
        for formula_id in cluster["formula_ids_included"]:
            eq_count += 1
            index = locate_paper(formula_id, papers)
            if index is not None:
                by_paper[index].add(unit)
    return by_paper, eq_count


def analyse(by_paper: list[set[str]], protocol: dict, rng: np.random.Generator) -> tuple[dict, list[dict]]:
    n_papers = len(by_paper)
    all_units = set().union(*by_paper)
    replicates = protocol["permutation_replicates"]
    curves = np.empty((replicates, n_papers), dtype=np.int16)
    for replicate in range(replicates):
        seen: set[str] = set()
        for position, index in enumerate(rng.permutation(n_papers)):
            seen.update(by_paper[int(index)])
            curves[replicate, position] = len(seen)
    mean = curves.mean(axis=0)
    low, high = np.quantile(curves, [0.025, 0.975], axis=0)
    marginal = np.diff(np.column_stack([np.zeros(replicates), curves]), axis=1).mean(axis=0)
    threshold = protocol["coverage_threshold"] * len(all_units)
    probability_coverage = np.mean(curves >= threshold, axis=0)
    saturation = None
    for position in range(n_papers):
        next_stop = min(n_papers, position + 6)
        next_gain = float(np.mean(marginal[position:next_stop]))
        if (
            mean[position] >= threshold
            and next_gain <= protocol["next_five_mean_marginal_threshold"]
            and probability_coverage[position] >= 0.95
        ):
            saturation = position + 1
            break
    seen, observed_curve, observed_marginal = set(), [], []
    for units in by_paper:
        before = len(seen)
        seen.update(units)
        observed_curve.append(len(seen))
        observed_marginal.append(len(seen) - before)
    passes = bool(saturation is not None and int(sum(observed_marginal[-10:])) == 0)
    rows = [
        {
            "papers": i + 1,
            "observed_order_categories": observed_curve[i],
            "observed_order_marginal": observed_marginal[i],
            "permutation_mean_categories": float(mean[i]),
            "permutation_ci95_low": float(low[i]),
            "permutation_ci95_high": float(high[i]),
            "permutation_mean_marginal": float(marginal[i]),
            "probability_at_least_95pct_observed_coverage": float(probability_coverage[i]),
        }
        for i in range(n_papers)
    ]
    summary = {
        "observed_unique_categories": len(all_units),
        "actual_order_last_novel_paper": max((i + 1 for i, gain in enumerate(observed_marginal) if gain), default=None),
        "new_categories_last_10_papers": int(sum(observed_marginal[-10:])),
        "saturation_paper_count": saturation,
        "passes_saturation_rule": passes,
    }
    return summary, rows


def main() -> None:
    protocol = json.loads((HERE / "KNOWLEDGE_SATURATION_PROTOCOL.json").read_text(encoding="utf-8"))
    merged = json.loads((LIT_DIR / "extracted_formulas_302.json").read_text(encoding="utf-8"))
    papers = [paper_key(row["source_file"]) for row in merged]
    if len(papers) != 50 or len(set(papers)) != 50:
        raise RuntimeError("expected 50 unique source papers")
    by_paper, eq_count = load_category_units(papers)
    rng = np.random.default_rng(protocol["seed"])
    summary, curve_rows = analyse(by_paper, protocol, rng)
    result = {
        "schema": "kged_corrosion_model_category_saturation_v1",
        "paper_count": len(papers),
        "formula_count": sum(len(row.get("formulas", [])) for row in merged),
        "quantitative_corrosion_equation_count": eq_count,
        "category_saturation": summary,
        "claim_boundary": protocol["claim_boundary"],
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (HERE / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
