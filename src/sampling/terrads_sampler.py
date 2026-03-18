"""
TerraDS Sampler — query SQLite metadata to find suitable AWS Terraform modules.

Selection criteria for the 200-module corpus:
- Provider: AWS (resource types starting with 'aws_')
- Complexity: >= 4 distinct resource types per module
- Stars: >= 5 (filters out toy repos)
- License: any permissive (already filtered in TerraDS)
- After selection: run Checkov + Trivy, keep only modules with >= 3 confirmed
  findings across both tools combined

Output: directory of 200 HCL modules, one subdirectory per module,
with metadata.json per module recording repo name, star count,
resource types, and initial scanner findings.
"""

import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path


def verify_terrads_schema(sqlite_path: str) -> dict[str, list[str]]:
    """
    Print and return the column names for each TerraDS table.

    Run this once against the downloaded TerraDS.sqlite to verify that
    column names match what query_candidate_modules expects before
    running the full sampling pipeline.

    Expected columns:
      repository: id, clone_url, stars, name, ...
      module: id, repository_id, relative_path, providers, ...
      resource: id, module_id, name, type, is_managed, ...

    Usage:
      python -c "
      from src.sampling.terrads_sampler import verify_terrads_schema
      verify_terrads_schema('/path/to/TerraDS.sqlite')
      "
    """
    conn = sqlite3.connect(sqlite_path)
    schema: dict[str, list[str]] = {}
    for table in ["Repositories", "Modules", "Resources"]:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [c[1] for c in cols]
        schema[table] = col_names
        print(f"{table}: {col_names}")
    conn.close()
    return schema


def query_candidate_modules(
    sqlite_path: str,
    min_resource_types: int = 4,
    min_stars: int = 5,
    provider: str = "aws",
    limit: int = 1000,
) -> list[dict]:
    """
    Query SQLite for modules meeting basic criteria.

    Returns list of dicts with keys: repository_id, clone_url, stars,
    relative_path, module_id, resource_types.
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    provider_prefix = f"{provider}_"
    cur.execute(
        """
        SELECT r.Id AS repository_id, r.CloneUrl AS clone_url, r.StarCount AS stars, r.Name AS repo_name,
               m.Id AS module_id, m.Path AS relative_path
        FROM Repositories r
        JOIN Modules m ON m.RepositoryId = r.Id
        JOIN Resources res ON res.ModuleId = m.Id
        WHERE res.Type LIKE ?
          AND r.StarCount >= ?
        GROUP BY r.Id, m.Id
        HAVING COUNT(DISTINCT res.Type) >= ?
        ORDER BY r.StarCount DESC
        LIMIT ?
        """,
        (f"{provider_prefix}%", min_stars, min_resource_types, limit),
    )

    rows = cur.fetchall()
    result = []

    for row in rows:
        cur.execute(
            "SELECT DISTINCT Type FROM Resources WHERE ModuleId = ?",
            (row["module_id"],),
        )
        resource_types = [r[0] for r in cur.fetchall()]
        result.append(
            {
                "repository_id": row["repository_id"],
                "clone_url": row["clone_url"],
                "stars": row["stars"],
                "repo_name": row["repo_name"],
                "module_id": row["module_id"],
                "relative_path": row["relative_path"],
                "resource_types": resource_types,
            }
        )

    conn.close()
    return result


def clone_and_extract_module(
    repo_clone_url: str,
    module_relative_path: str,
    output_dir: str,
) -> str:
    """
    Clone repo, extract the specific module directory, return path.

    Uses git clone --depth 1 for efficiency. Creates output_dir/module_xxx/
    with the module's .tf files.
    """
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", repo_clone_url, str(tmp)],
            check=True,
            capture_output=True,
        )

        module_src = tmp / module_relative_path
        if not module_src.exists():
            raise FileNotFoundError(f"Module path not found: {module_relative_path}")

        # output_dir is the target directory for this module
        out_subdir = Path(output_dir)
        out_subdir.mkdir(parents=True, exist_ok=True)

        # Copy .tf files
        for tf_file in module_src.glob("*.tf"):
            (out_subdir / tf_file.name).write_text(tf_file.read_text())

        return str(out_subdir)


def select_final_corpus(
    candidate_dirs: list[str],
    min_findings: int = 3,
    target_count: int = 200,
) -> list[str]:
    """
    Run scanners on candidates, select those with enough findings.

    Returns list of module directory paths that have >= min_findings
    combined Trivy + Checkov findings, up to target_count modules.
    """
    from src.scanning.trivy_runner import run_trivy
    from src.scanning.checkov_runner import run_checkov
    from src.scanning.scanner_normalizer import (
        normalize_trivy_findings,
        normalize_checkov_findings,
    )

    selected = []
    for module_dir in candidate_dirs:
        if len(selected) >= target_count:
            break
        try:
            trivy_raw = run_trivy(module_dir)
            checkov_raw = run_checkov(module_dir)
            trivy_findings = normalize_trivy_findings(trivy_raw)
            checkov_findings = normalize_checkov_findings(checkov_raw)
            total = len(trivy_findings) + len(checkov_findings)
            if total >= min_findings:
                selected.append(module_dir)
        except Exception:
            # Skip modules that fail to scan
            continue

    return selected
