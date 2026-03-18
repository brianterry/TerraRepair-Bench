# TerraRepair-Bench

A research benchmark that measures how well large language models repair security misconfigurations in real-world Terraform Infrastructure-as-Code.

**Research question:** When LLMs are used to repair scanner-confirmed Terraform security misconfigurations, do they successfully eliminate the target vulnerability, and at what rate do they introduce new ones?

## Prior Work

- **Low et al. (2024)** — "Repairing Infrastructure-as-Code using Large Language Models" (IEEE SecDev 2024). Evaluated GPT-3.5 and GPT-4 on synthetic vulnerable codebases. This benchmark extends their work with real-world repos, reasoning models, full automation, and regression measurement.
- **TerraDS (Bühler et al., MSR 2025)** — 62,406 real-world Terraform repositories. [Zenodo](https://zenodo.org/records/14217386). We sample 200 modules with confirmed scanner findings.

## Primary Metrics

1. **Repair success rate** — Fraction of target findings resolved (same rule_id + resource_name no longer flagged).
2. **Regression rate** — Fraction of target findings for which new findings were introduced.
3. **Net delta** — resolved_count − introduced_count (positive = improvement).

Matching is deterministic on (rule_id, resource_name) pairs. No LLM judge.

## Models

| Model | Type |
|-------|------|
| Claude 3.5 Sonnet | Non-reasoning |
| Nova Pro | Non-reasoning |
| Nova Lite | Non-reasoning |
| Llama 3.3 70B | Non-reasoning |
| DeepSeek-R1 | Reasoning |
| Qwen3-32B | Reasoning |

All models via AWS Bedrock. Temperature 0.0 for deterministic repair.

## Quick Start

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install external tools
brew install trivy
pip install checkov

# Download TerraDS (SQLite + archive)
# Place TerraDS.sqlite in a known path

# Sample 200 modules
python scripts/sample_terrads.py --sqlite /path/to/TerraDS.sqlite --output data/terrads_sample

# Run experiment (6 models × 200 modules)
python scripts/run_experiment.py --config experiments/configs/full_experiment.yaml

# Analyze results
python scripts/analyze.py --results experiments/results/full_180_game/
```

## Environment

Copy `.env.example` to `.env` and set:

- `AWS_REGION`, `AWS_PROFILE` (or credentials)
- `TERRADS_SQLITE_PATH`
- `CORPUS_DIR`

## Target Venue

[RAID 2026](https://raid2026.org/) — International Symposium on Research in Attacks, Intrusions and Defenses (April 16, 2026 deadline).
