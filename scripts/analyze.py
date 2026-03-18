from __future__ import annotations
"""
Aggregate results and generate paper tables.

Usage: python scripts/analyze.py --results experiments/results/full_180_game/

Generates:
  - results_summary.json        Raw aggregates per model
  - table1_main_results.tex     LaTeX table: model × repair_success_rate × regression_rate
  - table2_class_breakdown.tex  LaTeX table: model × vulnerability class breakdown
  - table3_reasoning_comparison.tex  Reasoning vs non-reasoning aggregate stats
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_games(results_dir: Path) -> list[dict]:
    games = []
    games_path = results_dir / "games"
    if not games_path.exists():
        return games
    for f in games_path.glob("*.json"):
        games.append(json.loads(f.read_text()))
    return games


def bootstrap_ci(data: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    low = np.percentile(boot_means, (1 - ci) / 2 * 100)
    high = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return low, high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to experiment results dir")
    args = parser.parse_args()

    results_dir = Path(args.results)
    games = load_games(results_dir)
    if not games:
        print("No games found")
        return

    # Aggregate by model
    by_model: dict[str, list[dict]] = {}
    for g in games:
        mid = g["model_id"]
        if mid not in by_model:
            by_model[mid] = []
        by_model[mid].append(g)

    summary = {}
    for model_id, model_games in by_model.items():
        success_rates = [g["repair_success_rate"] for g in model_games]
        regression_rates = [g["regression_rate"] for g in model_games]
        net_deltas = [g["net_delta"] for g in model_games]
        summary[model_id] = {
            "n": len(model_games),
            "repair_success_rate_mean": float(np.mean(success_rates)),
            "repair_success_rate_std": float(np.std(success_rates)),
            "repair_success_rate_ci": bootstrap_ci(success_rates),
            "regression_rate_mean": float(np.mean(regression_rates)),
            "regression_rate_std": float(np.std(regression_rates)),
            "regression_rate_ci": bootstrap_ci(regression_rates),
            "net_delta_mean": float(np.mean(net_deltas)),
            "net_delta_std": float(np.std(net_deltas)),
            "net_delta_ci": bootstrap_ci(net_deltas),
            "total_cost_usd": sum(g["repair_cost_usd"] for g in model_games),
        }

    (results_dir / "results_summary.json").write_text(json.dumps(summary, indent=2))

    # Table 1: Main results
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Repair Success Rate & Regression Rate & Net $\Delta$ \\",
        r"\midrule",
    ]
    for mid, s in summary.items():
        lines.append(
            f"{mid} & "
            f"{s['repair_success_rate_mean']:.2f} $\\pm$ {s['repair_success_rate_std']:.2f} & "
            f"{s['regression_rate_mean']:.2f} $\\pm$ {s['regression_rate_std']:.2f} & "
            f"{s['net_delta_mean']:.1f} $\\pm$ {s['net_delta_std']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (results_dir / "table1_main_results.tex").write_text("\n".join(lines))

    # Table 2: Class breakdown (simplified - aggregate across models)
    class_totals = {}
    for g in games:
        for cls, counts in g.get("class_breakdown", {}).items():
            if cls not in class_totals:
                class_totals[cls] = {"resolved": 0, "introduced": 0, "unchanged": 0}
            for k, v in counts.items():
                class_totals[cls][k] += v

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Class & Resolved & Introduced & Unchanged \\",
        r"\midrule",
    ]
    for cls, c in class_totals.items():
        lines.append(f"{cls} & {c['resolved']} & {c['introduced']} & {c['unchanged']} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (results_dir / "table2_class_breakdown.tex").write_text("\n".join(lines))

    # Table 3: Reasoning vs non-reasoning
    reasoning_models = ["deepseek-r1", "qwen3-32b"]
    non_reasoning = [m for m in summary if m not in reasoning_models]
    reasoning_regressions = []
    non_reasoning_regressions = []
    for g in games:
        if g["model_id"] in reasoning_models:
            reasoning_regressions.append(g["regression_rate"])
        else:
            non_reasoning_regressions.append(g["regression_rate"])

    t_stat, p_value = stats.ttest_ind(reasoning_regressions, non_reasoning_regressions)
    cohens_d = (np.mean(reasoning_regressions) - np.mean(non_reasoning_regressions)) / (
        np.sqrt((np.var(reasoning_regressions) + np.var(non_reasoning_regressions)) / 2)
    ) if len(reasoning_regressions) and len(non_reasoning_regressions) else 0

    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Reasoning mean regression rate & {np.mean(reasoning_regressions):.2f} \\\\",
        f"Non-reasoning mean regression rate & {np.mean(non_reasoning_regressions):.2f} \\\\",
        f"Two-sample $t$-test $p$-value & {p_value:.4f} \\\\",
        f"Cohen's $d$ & {cohens_d:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (results_dir / "table3_reasoning_comparison.tex").write_text("\n".join(lines))

    print(f"Wrote results to {results_dir}")


if __name__ == "__main__":
    main()
