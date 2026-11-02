"""Verification test for the 50-paper literature corpus, saturation audit, and two-layer knowledge graph."""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_literature_corpus_and_saturation():
    lit_dir = REPO_ROOT / "literature"

    # 1. Verify 50-paper bibliographic inventory
    corpus_csv = lit_dir / "WOS_Golden_Corpus_Top50.csv"
    with corpus_csv.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 50, f"Expected 50 papers in WOS_Golden_Corpus_Top50.csv, got {len(rows)}"

    # 2. Verify 50-paper records and 302 extracted relationships
    env_50 = json.loads((lit_dir / "environment_knowledge_50papers.json").read_text(encoding="utf-8"))
    form_50 = json.loads((lit_dir / "extracted_formulas_302.json").read_text(encoding="utf-8"))
    assert len(env_50) == 50, f"Expected 50 paper records, got {len(env_50)}"
    assert len(form_50) == 50, f"Expected 50 paper records, got {len(form_50)}"
    total_formulas = sum(len(p["formulas"]) for p in form_50)
    assert total_formulas == 302, f"Expected 302 extracted relationships, got {total_formulas}"

    # 3. Verify 3-resolution knowledge saturation audit (Table S2 & Fig. S1)
    sat_res = json.loads((lit_dir / "saturation_audit/result.json").read_text(encoding="utf-8"))
    assert sat_res["paper_count"] == 50
    assert sat_res["formula_count"] == 302
    summaries = sat_res["summaries"]
    assert summaries["coarse"]["observed_unique_units"] == 17
    assert summaries["medium"]["observed_unique_units"] == 9
    assert summaries["medium"]["order_robust_saturation_paper_count"] == 42
    assert summaries["medium"]["passes_saturation_rule"] is True
    assert summaries["fine"]["observed_unique_units"] == 70


def test_two_layer_knowledge_graph():
    kg_dir = REPO_ROOT / "knowledge_graph"

    # 1. Verify raw 50-paper literature graph (50 paper nodes, 1,057 total nodes, 3,116 edges)
    rep = json.loads((kg_dir / "literature_graph/report.json").read_text(encoding="utf-8"))
    assert rep["num_papers"] == 50
    assert rep["num_nodes"] == 1057
    assert rep["num_edges"] == 3116
    assert (kg_dir / "literature_graph/graph.graphml").exists()

    # 2. Verify executable two-layer knowledge graph (Table S3, Table S4, Table S5)
    mg = json.loads((kg_dir / "math_graph.json").read_text(encoding="utf-8"))
    nodes = mg["nodes"]
    process_nodes = [k for k in nodes if k.startswith("process:")]
    template_nodes = [k for k in nodes if k.startswith("template:")]
    tag_nodes = [k for k in nodes if k.startswith("tag:")]
    hyp_nodes = [k for k in nodes if k.startswith("hypothesis:")]
    construction_nodes = [k for k in nodes if k.startswith("construction:")]

    assert len(process_nodes) == 12, f"Expected 12 physical process categories (Table S3), got {len(process_nodes)}"
    assert len(template_nodes) == 31, f"Expected 31 dimensionless log-response kernels (Table S4), got {len(template_nodes)}"
    assert len(tag_nodes) == 20, f"Expected 20 semantic tags (Table S5), got {len(tag_nodes)}"
    assert len(hyp_nodes) == 4, f"Expected 4 hypotheses (H1-H4, Table S1), got {len(hyp_nodes)}"
    assert len(construction_nodes) == 2

    # 3. Verify 41 candidate modules (M01-M41, Table S13)
    mods = json.loads((kg_dir / "modules_M01_M41_4round.json").read_text(encoding="utf-8"))
    assert len(mods) == 41, f"Expected 41 modules (M01-M41, Table S13), got {len(mods)}"
    r1_mods = [m for m in mods if m["round"] == 1]
    r2_4_mods = [m for m in mods if m["round"] in (2, 3, 4)]
    assert len(r1_mods) == 20, f"Expected 20 literature-direct baseline modules in Round 1 (M01-M20), got {len(r1_mods)}"
    assert len(r2_4_mods) == 21, f"Expected 21 LLM-generated modules in Rounds 2-4 (M21-M41), got {len(r2_4_mods)}"


if __name__ == "__main__":
    test_literature_corpus_and_saturation()
    test_two_layer_knowledge_graph()
    print("[PASS] 50-paper literature corpus, 302 extracted relationships, and 3-resolution saturation audit verified (Table S2 & Fig. S1).")
    print("[PASS] Raw literature graph (50 papers, 1,057 nodes, 3,116 edges) and executable two-layer knowledge graph (12 process categories, 31 kernels K01-K31, 20 tags, 41 modules M01-M41) verified (Tables S1, S3-S6, S13).")
