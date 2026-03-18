"""
One-time: query TerraDS SQLite, extract modules, write corpus.

Usage:
  python scripts/sample_terrads.py --sqlite /path/to/TerraDS.sqlite --output data/terrads_sample

Requires: TerraDS.sqlite, git, trivy, checkov.
"""

import argparse
import json
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sampling.terrads_sampler import (
    clone_and_extract_module,
    query_candidate_modules,
    select_final_corpus,
)
from src.scanning.scanner_normalizer import ScannerStack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True, help="Path to TerraDS.sqlite")
    parser.add_argument("--output", default="data/terrads_sample", help="Output corpus directory")
    parser.add_argument("--limit", type=int, default=1000, help="Max candidates to fetch")
    parser.add_argument("--target", type=int, default=200, help="Target module count")
    parser.add_argument("--min-findings", type=int, default=3, help="Min findings per module")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Querying TerraDS for candidate modules...")
    candidates = query_candidate_modules(
        sqlite_path=args.sqlite,
        min_resource_types=4,
        min_stars=5,
        provider="aws",
        limit=args.limit,
    )
    print(f"Found {len(candidates)} candidates")

    scanner_stack = ScannerStack()
    candidate_dirs = []
    for i, c in enumerate(candidates):
        if len(candidate_dirs) >= args.target * 2:  # Get extra for filtering
            break
        try:
            out_sub = output_path / f"module_{i:04d}"
            if out_sub.exists():
                candidate_dirs.append(str(out_sub))
                continue
            path = clone_and_extract_module(
                repo_clone_url=c["clone_url"],
                module_relative_path=c["relative_path"],
                output_dir=str(out_sub),
            )
            candidate_dirs.append(path)
        except Exception as e:
            print(f"  Skip {c.get('repo_name', '?')}: {e}")
            continue

    print(f"Cloned {len(candidate_dirs)} modules. Running scanners...")
    selected = select_final_corpus(
        candidate_dirs=candidate_dirs,
        min_findings=args.min_findings,
        target_count=args.target,
    )
    print(f"Selected {len(selected)} modules with >= {args.min_findings} findings")

    # Write metadata per module and collect manifest entries
    manifest_modules = []
    for mod_dir in selected:
        findings = scanner_stack.run(mod_dir)
        meta_path = Path(mod_dir) / "metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "findings_count": len(findings),
                    "findings": [
                        {
                            "tool": f.tool,
                            "rule_id": f.rule_id,
                            "title": f.title,
                            "resource_name": f.resource_name,
                        }
                        for f in findings[:20]
                    ],
                },
                indent=2,
            )
        )
        manifest_modules.append({
            "module_id": Path(mod_dir).name,
            "path": str(mod_dir),
            "findings_count": len(findings),
        })

    # Write corpus manifest
    corpus_manifest = {
        "selected_count": len(selected),
        "modules": manifest_modules,
    }
    (output_path / "corpus.json").write_text(json.dumps(corpus_manifest, indent=2))
    print(f"Corpus written to {output_path} ({len(selected)} modules, manifest: corpus.json)")


if __name__ == "__main__":
    main()
