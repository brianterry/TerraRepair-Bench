"""
Main experiment runner. Runs all 1,200 games (6 models × 200 modules).

Usage: python scripts/run_experiment.py --config experiments/configs/full_experiment.yaml

For each module in the corpus:
  1. Run scanner stack on original code → findings_before
  2. For each model:
     a. Repair the module
     b. Run scanner stack on repaired code → findings_after
     c. Score the repair → GameResult
     d. Save to experiments/results/{experiment_id}/games/

Progress bar with ETA. Cost tracking per model.
Auto-resume: skip games where results/games/{module_id}_{model_id}.json exists.
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.repair.repair_agent import repair_module
from src.scanning.scanner_normalizer import Finding, ScannerStack
from src.scoring.delta_scorer import GameResult, score_repair
from src.utils.bedrock_client import BedrockClient


def _finding_to_dict(f: Finding) -> dict:
    return {
        "tool": f.tool,
        "rule_id": f.rule_id,
        "title": f.title,
        "severity": f.severity,
        "resource_name": f.resource_name,
        "resource_type": f.resource_type,
        "file_path": f.file_path,
        "start_line": f.start_line,
        "end_line": f.end_line,
    }


def _game_result_to_dict(g: GameResult) -> dict:
    return {
        "module_id": g.module_id,
        "model_id": g.model_id,
        "findings_before": [_finding_to_dict(f) for f in g.findings_before],
        "findings_after": [_finding_to_dict(f) for f in g.findings_after],
        "target_count": g.target_count,
        "resolved_count": g.resolved_count,
        "introduced_count": g.introduced_count,
        "unchanged_count": g.unchanged_count,
        "repair_success_rate": g.repair_success_rate,
        "regression_rate": g.regression_rate,
        "net_delta": g.net_delta,
        "class_breakdown": g.class_breakdown,
        "repair_cost_usd": g.repair_cost_usd,
        "repair_tokens": g.repair_tokens,
        "hcl_valid": g.hcl_valid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--region", default="us-east-1", help="AWS region for Bedrock")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment"]["name"]
    corpus_dir = Path(config["corpus"]["dir"])
    max_modules = config["corpus"].get("max_modules", 200)
    output_dir = Path(config["output_dir"]) / exp_name
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    # List modules (subdirs with .tf files)
    module_dirs = sorted([
        str(d) for d in corpus_dir.iterdir()
        if d.is_dir() and list(d.glob("*.tf"))
    ])[:max_modules]
    models = config["models"]
    scanner_stack = ScannerStack()
    bedrock = BedrockClient(region=args.region)

    total = len(module_dirs) * len(models)
    cost_by_model = {m["id"]: 0.0 for m in models}

    with tqdm(total=total, desc=exp_name, unit="game") as pbar:
        for module_dir in module_dirs:
            module_id = Path(module_dir).name
            findings_before = scanner_stack.run(module_dir)

            for model_cfg in models:
                model_id = model_cfg["id"]
                game_file = games_dir / f"{module_id}_{model_id}.json"
                if game_file.exists():
                    pbar.update(1)
                    continue

                # Repair
                result = repair_module(
                    module_dir=module_dir,
                    findings=findings_before,
                    model_id=model_id,
                    bedrock_client=bedrock,
                )
                cost_by_model[model_id] += result.repair_cost_usd

                # Scan repaired code
                findings_after = []
                if result.repaired_dir:
                    try:
                        findings_after = scanner_stack.run(result.repaired_dir)
                    finally:
                        if result.repaired_dir and Path(result.repaired_dir).exists():
                            shutil.rmtree(result.repaired_dir, ignore_errors=True)

                # Score
                game = score_repair(
                    module_id=module_id,
                    model_id=model_id,
                    findings_before=findings_before,
                    findings_after=findings_after,
                    repair_result=result,
                )

                game_file.write_text(json.dumps(_game_result_to_dict(game), indent=2))
                pbar.update(1)

    print("\nCost by model:")
    for mid, cost in cost_by_model.items():
        print(f"  {mid}: ${cost:.4f}")


if __name__ == "__main__":
    main()
