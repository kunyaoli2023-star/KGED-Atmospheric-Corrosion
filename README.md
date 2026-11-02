# KGED: Knowledge-Guided Equation Discovery for Atmospheric Steel Corrosion

This repository provides the 50-paper corrosion literature corpus, two-layer knowledge graph, verification test scripts for the discovered thin-film electrochemical rate equations, and cross-environment engineering corrosion-growth prediction module accompanying the manuscript:

> **Knowledge-guided multi-agent discovery of explicit kinetic equations for atmospheric corrosion of steel**

## Repository Structure

- `literature/` — **50-Paper Corrosion Literature Corpus & Saturation Audit (Tables S1–S2, Fig. S1)**
  - `WOS_Golden_Corpus_Top50.csv`: Bibliographic inventory of the 50 peer-reviewed atmospheric and thin-film corrosion papers.
  - `evidence.json` & `source_audit.json`: Curated literature evidence items (`H1`–`H4`, Table S1) and page-level source verification records.
  - `extracted_formulas_302.json` & `environment_knowledge_50papers.json`: The 302 extracted corrosion relationships and structured environmental/electrochemical annotations across the 50-paper corpus.
  - `semantic_classification/`: Multi-resolution semantic and symbolic classifications (`llm_semantic_classification_safe.json`, `merged_signatures.json`, `semantic_symbolic_results.json`, `formulas_global_clusters.json`, `canonical_templates.json`).
  - `saturation_audit/`: Knowledge coverage saturation protocol (`KNOWLEDGE_SATURATION_PROTOCOL.json`), audit script (`run_knowledge_saturation_audit.py`), 10,000-permutation accumulation curves (`curves.csv`, `result.json`), and multi-resolution saturation plot (`Fig_S1_Knowledge_Coverage_Clean.png`, Table S2 & Fig. S1).
- `knowledge_graph/` — **Two-Layer Corrosion Knowledge Graph & Candidate Modules (Tables S3–S6, Table S13)**
  - `literature_graph/`: Raw 50-paper literature knowledge graph (`graph.graphml` and `report.json`: 50 paper nodes, 1,057 total nodes, 3,116 edges) together with operator and symbol stability summaries (`stability_table.csv`, `top_symbols.csv`, `icorr_meta_models.json`, `icorr_semantic_signatures.json`, `global_ontology_mapping.json`).
  - `math_graph.json`, `validation.json`, `index.html`: Executable two-layer knowledge graph encoding the 12 physical process categories (Table S3), 8 mathematical families and 31 dimensionless log-response kernels `K01`–`K31` (Table S4), 20 semantic tags (Table S5), and admissible additive/subtractive composition rules.
  - `modules_M01_M41_4round.json` & `modules_M01_M41_4round.csv`: Complete inventory of all 41 candidate log-response modules (`M01`–`M41`, Table S13), partitioned into Stage 1 (Round 1: 20 literature-direct baseline modules `M01`–`M20`) and Stage 2 (Rounds 2–4: 21 residual-driven LLM-generated modules `M21`–`M41` from the 8-family mathematical library).
  - `inputs/`: Working hypothesis specifications (`hypotheses.json`), dimensionless coordinate registry (`symbol_registry.json`), and module compilation protocol (`module_protocol.json`).
  - `family_audit/`: Mathematical family audit and cross-round module provenance records (`FAMILY_AUDIT.md`, `family_stability.json`, `evidence_family_bridges.json`, `module_families.jsonl`, `literature_inventory_provisional.jsonl`).
  - `build_math_knowledge.py` & `maief/`: Python implementation for constructing, querying, and validating the two-layer knowledge graph and compiling candidate modules.
- `data/` — **Laboratory Thin-Film Electrochemical Dataset & Discovered Rate Laws (Tables 1, 4, 5)**
  - `data.json`: 168-condition thin-film electrochemical dataset on Q355B steel ($T \in [0, 50]\,^\circ\mathrm{C}$, $C \in [0, 26.5]\,\mathrm{wt\%\ NaCl}$, $H \in [50, 1000]\,\mu\mathrm{m}$).
  - `folds.json`: 23-fold grouped cross-validation split definitions (5 profile, 5 temperature, 7 film-thickness, and 6 salinity holdout folds).
  - `Frozen_models.json`: Full-precision parameters and analytical expressions for the retained KGED equations and symbolic-regression baselines (Table 4).
- `engineering/` — **Cross-Environment Corrosion-Growth Prediction Module (Tables 6–8, Fig. 9)**
  - `engineering_integrator.py` & `engineering_state.py`: Reaction-clock-weighted cubic Bernstein state-evolution integrator and bounded linear least-squares source calibration solver.
  - `run_engineering_v6.py`: Cross-environment calibration (Linhai source site) and target extrapolation pipeline (Jinhai, humid chlorinated indoor swimming pool, and NASA KSC).
  - `verify_closed_forms.py`: Analytical verification of the frozen 5-parameter parsimonious and 8-parameter minimum-error KGED equations.
  - `inputs/`: Environmental forcing histories (`*.npz`), observation records (`*_labels.json`), and microenvironmental mapping parameters (`mapping.json`).
  - `PREDICTIONS_FROZEN.json`: Full-precision calibrated state amplitudes (Table 6) and predicted corrosion trajectories (Table 7 & Fig. 9).
- `tests/` — **Automated Verification Suite**
  - `test_knowledge_graph_assets.py`: Verifies the 50-paper literature corpus, 302 extracted relationships, 3-resolution saturation audit (Table S2 & Fig. S1), raw literature graph (1,057 nodes, 3,116 edges), executable two-layer knowledge graph (12 process categories, 31 kernels `K01`–`K31`, 20 tags), and 41 candidate modules (`M01`–`M41`, Table S13).
  - `test_laboratory_equations.py`: Reproduces full-dataset training log-RMSE (`0.507` and `0.496`), 23-fold grouped CV log-RMSE (`0.570` and `0.567`), and analytical interior corrosion maxima ($C^* = 4.00\text{–}4.05\,\mathrm{wt\%}$, $H^* = 167.7\text{–}184.3\,\mu\mathrm{m}$, Table 5).
  - `test_engineering_prediction.py`: Verifies the source-calibrated Bernstein state amplitudes (Table 6) and target trajectory RMSE and endpoint errors across Jinhai, swimming pool, and NASA KSC (`1.16%` mean absolute endpoint error for the 5-parameter KGED law, Table 7).

## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Verify the 50-paper literature corpus, saturation audit, and two-layer knowledge graph:
```bash
python tests/test_knowledge_graph_assets.py
```

Run the laboratory rate equation verification test:
```bash
python tests/test_laboratory_equations.py
```

Run the cross-environment engineering prediction test:
```bash
python tests/test_engineering_prediction.py
```

Run the full closed-form and Gauss-Legendre integration verification in `engineering/`:
```bash
python engineering/verify_closed_forms.py
```
