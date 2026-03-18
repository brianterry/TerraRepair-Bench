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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.scanning.scanner_normalizer import ScannerStack


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
        SELECT r.id AS repository_id, r.clone_url, r.stars, r.name AS repo_name,
               m.id AS module_id, m.relative_path
        FROM repository r
        JOIN module m ON m.repository_id = r.id
        JOIN resource res ON res.module_id = m.id
        WHERE res.type LIKE ?
          AND r.stars >= ?
        GROUP BY r.id, m.id
        HAVING COUNT(DISTINCT res.type) >= ?
        ORDER BY r.stars DESC
        LIMIT ?
        """,
        (f"{provider_prefix}%", min_stars, min_resource_types, limit),
    )

    rows = cur.fetchall()
    conn.close()

    # Get resource types per module
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    result = []
    for row in rows:
        cur.execute(
            "SELECT DISTINCT type FROM resource WHERE module_id = ?",
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
    scanner_stack: "ScannerStack",
    min_findings: int = 3,
    target_count: int = 200,
) -> list[str]:
    """
    Run scanners on candidates, select those with enough findings.

    Returns list of module directory paths that have >= min_findings
    combined Trivy + Checkov findings, up to target_count modules.
    """
    selected = []
    for module_dir in candidate_dirs:
        if len(selected) >= target_count:
            break
        findings = scanner_stack.run(module_dir)
        if len(findings) >= min_findings:
            selected.append(module_dir)
    return selected
