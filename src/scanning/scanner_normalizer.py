"""
Unified finding format across Trivy and Checkov.

Normalizes raw scanner output into Finding dataclass.
Deduplicates by (rule_id, resource_name).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Finding:
    tool: str  # "trivy" or "checkov"
    rule_id: str  # "AVD-AWS-0086" or "CKV_AWS_19"
    title: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    resource_name: str  # "aws_s3_bucket.example"
    resource_type: str  # "aws_s3_bucket"
    file_path: str
    start_line: int
    end_line: int


def normalize_trivy_findings(raw: list[dict]) -> list[Finding]:
    """Convert raw Trivy Misconfigurations to Finding objects."""
    findings = []
    for item in raw:
        meta = item.get("CauseMetadata", {})
        resource = meta.get("Resource", "unknown")
        # Extract resource type (e.g., aws_s3_bucket from aws_s3_bucket.example)
        resource_type = resource.split(".")[0] if "." in resource else resource
        findings.append(
            Finding(
                tool="trivy",
                rule_id=item.get("ID", ""),
                title=item.get("Title", ""),
                severity=item.get("Severity", "UNKNOWN").upper(),
                resource_name=resource,
                resource_type=resource_type,
                file_path=item.get("Target", ""),
                start_line=meta.get("StartLine", 0),
                end_line=meta.get("EndLine", 0),
            )
        )
    return findings


def normalize_checkov_findings(raw: list[dict]) -> list[Finding]:
    """Convert raw Checkov failed_checks to Finding objects."""
    findings = []
    for item in raw:
        line_range = item.get("file_line_range", [0, 0])
        resource = item.get("resource", "unknown")
        resource_type = resource.split(".")[0] if "." in resource else resource
        severity = item.get("severity", item.get("check", {}).get("severity", "UNKNOWN"))
        findings.append(
            Finding(
                tool="checkov",
                rule_id=item.get("check_id", ""),
                title=item.get("check_name", ""),
                severity=(severity or "UNKNOWN").upper(),
                resource_name=resource,
                resource_type=resource_type,
                file_path=item.get("file_path", ""),
                start_line=line_range[0] if line_range else 0,
                end_line=line_range[1] if len(line_range) > 1 else line_range[0],
            )
        )
    return findings


def deduplicate_findings(findings: list[Finding]) -> list[Finding]:
    """
    Remove findings that describe the same issue on the same resource.

    A finding is a duplicate if: same resource_name AND same rule_id.
    Keeps first occurrence.
    """
    seen: set[tuple[str, str]] = set()
    result = []
    for f in findings:
        key = (f.resource_name, f.rule_id)
        if key not in seen:
            seen.add(key)
            result.append(f)
    return result


class ScannerStack:
    """
    Runs Trivy and Checkov, normalizes and deduplicates findings.
    """

    def __init__(self, trivy_runner=None, checkov_runner=None):
        if trivy_runner is None:
            from src.scanning.trivy_runner import run_trivy
            trivy_runner = run_trivy
        if checkov_runner is None:
            from src.scanning.checkov_runner import run_checkov
            checkov_runner = run_checkov
        self._run_trivy = trivy_runner
        self._run_checkov = checkov_runner

    def run(self, target_dir: str) -> list[Finding]:
        """Run both scanners, normalize, deduplicate. Returns combined findings."""
        all_findings: list[Finding] = []
        try:
            trivy_raw = self._run_trivy(target_dir)
            all_findings.extend(normalize_trivy_findings(trivy_raw))
        except Exception:
            pass  # Scanner may not be installed or may fail
        try:
            checkov_raw = self._run_checkov(target_dir)
            all_findings.extend(normalize_checkov_findings(checkov_raw))
        except Exception:
            pass
        return deduplicate_findings(all_findings)
