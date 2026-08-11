# Paper Analysis: Does the Simulator Predict Real Competition Outcomes?

Supporting code and figures for **"Does a Synthetic Aircraft Design-Space Simulator Predict Real Competition Outcomes? A Case Study on the DARPA Lift Challenge"** (Uri Kartoun, DBbun LLC, August 2026).

The paper reconstructs the real aircraft that competed in the 2026 DARPA Lift Challenge — the five teams that completed a scored course, plus DefendTex, the one crash with enough public data to reconstruct — as inputs to this repository's own, unmodified `darpa_lift_challenge_generator_v1_2.py`, and evaluates whether the simulator's assessment of those designs agrees with what actually happened in the real competition. Every script here imports the generator directly as a library; none of them edit it.

## Contents

- **code/** — every script that produced a number or figure in the paper, organized as:
  - `case_studies/` — real-design reconstructions and mission simulations for all six real aircraft
  - `dataset_qc/` — CSV integrity and schema validation
  - `population_analysis/` — top-tier and nearest-neighbor analysis of the generated design population
  - `monte_carlo_sensitivity.py` / `monte_carlo_sensitivity_v2.py` — the sensitivity analysis and its energy-capacity correction
  - `attribution_analysis.py` / `attribution_analysis_2.py` — controlled ablations isolating the source of Monte Carlo variance
  - `ratio_sweep_ablation.py` — controlled test of the ratio-driven stress-index mechanism
  - `population_percentile_benchmark.py` — the paper's headline population-consistency benchmark
  - `design_improvement_recommendations.py` / `verify_touchdown_cruise_levers.py` — discovery and controlled verification of design-improvement levers
  - remaining `*.py` files — the matching chart-generation scripts for each figure
- **figures/** — every chart referenced in the paper, as generated
- **notes/methodology_and_findings.md** — full working notes: methodology, every result, and the self-correction history behind the paper's final claims (including two findings that did not survive follow-up testing and are reported rather than omitted)

## Reproducing a result

Each script imports `darpa_lift_challenge_generator_v1_2.py` from the repository root as a library, e.g.:

```python
import sys
sys.path.insert(0, "..")
import darpa_lift_challenge_generator_v1_2 as gen
```

Run any script directly with Python 3; most write their output to stdout. Scripts under `population_analysis/` and `dataset_qc/` read from the DBbun DARPA Lift dataset (Hugging Face: https://huggingface.co/datasets/DBbun/DARPA_Lift_2026) and expect it available locally at the path referenced in each script.
