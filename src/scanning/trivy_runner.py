"""
Run trivy config against a directory and parse JSON output.
"""

import json
import subprocess
from pathlib import Path


class ScannerError(Exception):
    """Raised when trivy not found or times out."""

    pass


def run_trivy(
    target_dir: str,
    severity: str = "LOW,MEDIUM,HIGH,CRITICAL",
    timeout: int = 60,
) -> list[dict]:
    """
    Execute: trivy config --format json --severity {severity} --quiet {target_dir}
    Parse the Results[].Misconfigurations[] structure.
    Return list of normalized findings (raw dicts for normalizer).
    Raises ScannerError if trivy not found or times out.
    """
    target = Path(target_dir)
    if not target.exists():
        return []

    cmd = [
        "trivy",
        "config",
        "--format",
        "json",
        "--severity",
        severity,
        "--quiet",
        str(target),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(target.parent) if target.is_dir() else str(target),
        )
    except FileNotFoundError as e:
        raise ScannerError("trivy not found. Install with: brew install trivy") from e
    except subprocess.TimeoutExpired as e:
        raise ScannerError(f"trivy timed out after {timeout}s") from e

    if result.returncode not in (0, 1):  # 1 = findings found
        raise ScannerError(f"trivy failed: {result.stderr}")

    raw = json.loads(result.stdout) if result.stdout else {}
    findings = []
    for res in raw.get("Results", []):
        target_file = res.get("Target", "")
        for misconfig in res.get("Misconfigurations", []):
            m = dict(misconfig)
            m["Target"] = m.get("Target", target_file)
            findings.append(m)
    return findings
