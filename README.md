# KGED: Knowledge-Guided Equation Discovery for Atmospheric Steel Corrosion

Test data and code accompanying the manuscript:

> **Knowledge-guided multi-agent discovery of explicit kinetic equations for atmospheric corrosion of steel**

## Quick Start: Engineering Corrosion Prediction

All four field-exposure test datasets (`engineering/inputs/`: Linhai source site and Jinhai, indoor swimming pool, and NASA KSC target sites) and frozen rate-law coefficients are bundled directly in this repository. Requires **Python 3.9+** (`numpy`, `scipy`, `matplotlib`, and `sympy`).

```bash
python -m pip install -r requirements.txt
python run_engineering_predictions.py
```

Windows users can also double-click `run_engineering_predictions.bat`. All reproduced tables (`Tables 7–8`, `Table S4`), trajectory predictions, rate-law ablation results, and 300 DPI figures (`Figs. 9–10`, `Figs. S3–S5`) are written to `engineering/results/` (exact package versions used for verification are listed in `requirements-lock.txt`).

## Verification Tests

Verify the cross-environment engineering corrosion-depth predictions directly:

```bash
python tests/test_engineering_prediction.py
```

Or run the full offline verification suite:

```bash
python -m unittest discover -s tests -v
```

## Two-Layer Knowledge Graph & Corpus Assets

- **50-paper literature corpus & saturation audit (`literature/`)**: Bibliographic inventory `C01`–`C50` (`WOS_Golden_Corpus_Top50.csv`), 302 extracted relationships (`extracted_formulas_302.json`), semantic tags (`semantic_classification/`), and 9-category coverage saturation audit (`saturation_audit/`, `Fig. S1`).
- **Layer 1 — Literature evidence graph (`knowledge_graph/literature_graph/`)**: Raw provenance graph (`graph.graphml`, `nodes.csv`, `edges.csv`; 1,057 nodes and 3,116 edges across 50 papers) linking corrosion mechanisms, environmental variables, and published expressions to hypotheses `H1`–`H4`.
- **Layer 2 — Executable mathematical knowledge graph (`knowledge_graph/math_graph.json`)**: 12 physical process categories, 8 canonical mathematical families, 31 dimensionless elementary kernels (`K01`–`K31`), and construction rules built by `knowledge_graph/build_math_knowledge_graph.py`.
- **Round-wise candidate modules (`knowledge_graph/modules_M01_M41_4round.csv` / `.json`)**: Full inventory of all 41 dimensionless modules (`M01`–`M20` in Stage 2a Round 1; `M21`–`M41` in Stage 2b Rounds 2–4) and symbolic construction/validation scripts (`knowledge_graph/maief/`).

## Repository Structure

- `run_engineering_predictions.py` — One-click entry script reproducing Linhai source calibration, target-site corrosion-depth trajectories, model comparisons, rate-law ablations, and 300 DPI figures in `engineering/results/`.
- `engineering/` — Field-exposure test datasets (`inputs/`), frozen 5- and 8-parameter KGED rate laws, cumulative-dose state-basis integrator, baseline comparison scripts (`comparators/`), and generated outputs (`results/`).
- `data/` — Per-seed symbolic-regression baseline expressions and multi-round Pareto evaluation archives.
- `literature/` — 50-paper corrosion literature corpus, extracted relationships, semantic classifications, and coverage saturation audit.
- `knowledge_graph/` — Two-layer corrosion knowledge graph, dimensionless candidate modules (`M01`–`M41`), and symbolic construction/validation scripts (`maief/`).
- `tests/` — Offline verification tests and archived multi-agent prompt/schema repeatability records.

## License

Code is released under the [MIT License](LICENSE); field-exposure test datasets, curated knowledge-graph assets, and model archives are shared under CC BY 4.0 (see `LICENSE` for details).
