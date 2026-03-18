"""
Run checkov --directory and parse JSON output.
"""

import json
import subprocess
from pathlib import Path


class ScannerError(Exception):
    """Raised when checkov not found or fails."""

    pass


def run_checkov(
    target_dir: str,
    timeout: int = 120,
) -> list[dict]:
    """
    Execute: checkov --directory {target_dir} --output json --quiet --framework terraform
    Parse the results.failed_checks[] structure.
    Return list of raw findings for normalizer.
    """
    target = Path(target_dir)
    if not target.exists():
        return []

    cmd = [
        "checkov",
        "--directory",
        str(target),
        "--output",
        "json",
        "--quiet",
        "--framework",
        "terraform",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise ScannerError(
            "checkov not found. Install with: pip install checkov"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise ScannerError(f"checkov timed out after {timeout}s") from e

    if result.returncode not in (0, 1):  # 1 = findings
        raise ScannerError(f"checkov failed: {result.stderr}")

    raw = json.loads(result.stdout) if result.stdout else {}
    # Checkov may return array when scanning multiple frameworks
    if isinstance(raw, list):
        all_failed = []
        for item in raw:
            results = item.get("results", {})
            all_failed.extend(results.get("failed_checks", []))
        return all_failed
    results = raw.get("results", {})
    return results.get("failed_checks", [])
