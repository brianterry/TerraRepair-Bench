"""
Core metric computation — the primary scientific contribution.

Deterministic matching on (rule_id, resource_name) pairs.
No LLM judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.repair.repair_agent import RepairResult
    from src.scanning.scanner_normalizer import Finding

# Three security property categories following the CIA triad framework.
# access_control  — who can reach what (IAM, SGs, public access, bucket policies)
# data_protection — protecting data content (encryption, secrets, backups)
# observability   — detecting what happened (logging, monitoring, audit trails)
VULNERABILITY_CLASSES = {
    "access_control": [],
    "data_protection": [],
    "observability": [],
    "other": [],
}

# Keyword patterns for classification. Checked against rule_id and title
# (lowercased). First match wins. Order matters — more specific first.
_ACCESS_CONTROL_KEYWORDS = [
    "public", "iam", "policy", "permission", "privilege", "role",
    "trust", "principal", "acl", "ingress", "egress", "cidr",
    "security_group", "securitygroup", "unrestricted", "open",
    "allow", "access_control", "rbac", "admin",
]

_DATA_PROTECTION_KEYWORDS = [
    "encrypt", "kms", "tls", "ssl", "https", "secret", "credential",
    "password", "token", "key_rotation", "certificate", "backup",
    "snapshot", "deletion_protection", "versioning", "mfa",
    "sensitive", "plaintext", "hardcoded",
]

_OBSERVABILITY_KEYWORDS = [
    "log", "logging", "audit", "trail", "cloudtrail", "cloudwatch",
    "monitor", "metric", "alarm", "flow_log", "flowlog", "insight",
    "retention", "event", "notification", "alert",
]


def classify_finding(finding: "Finding") -> str:
    """
    Classify a finding into one of three security property categories.

    Uses keyword matching against rule_id and title (both lowercased).
    Categories follow the security triad:
      access_control  — IAM, security groups, public access
      data_protection — encryption, secrets, backups
      observability   — logging, monitoring, audit trails
    """
    # Combine rule_id and title for matching surface
    text = f"{finding.rule_id} {finding.title}".lower()

    for keyword in _ACCESS_CONTROL_KEYWORDS:
        if keyword in text:
            return "access_control"

    for keyword in _DATA_PROTECTION_KEYWORDS:
        if keyword in text:
            return "data_protection"

    for keyword in _OBSERVABILITY_KEYWORDS:
        if keyword in text:
            return "observability"

    return "other"


@dataclass
class GameResult:
    """Result of one repair attempt: one module × one model."""

    module_id: str
    model_id: str

    # Scanner findings before repair
    findings_before: list["Finding"]

    # Scanner findings after repair
    findings_after: list["Finding"]

    # Derived metrics
    target_count: int  # len(findings_before)
    resolved_count: int  # findings in before but not after (same rule_id + resource)
    introduced_count: int  # findings in after but not before
    unchanged_count: int  # findings present in both

    # Primary metrics
    repair_success_rate: float  # resolved_count / target_count
    regression_rate: float  # introduced_count / target_count
    net_delta: int  # resolved_count - introduced_count (positive = improvement)

    # Breakdown by vulnerability class
    class_breakdown: dict[str, dict]  # {class: {resolved, introduced, unchanged}}

    # Metadata
    repair_cost_usd: float
    repair_tokens: int
    hcl_valid: bool  # Did repaired code pass basic HCL validation


def _finding_key(f: "Finding") -> tuple[str, str]:
    return (f.rule_id, f.resource_name)


def score_repair(
    module_id: str,
    model_id: str,
    findings_before: list["Finding"],
    findings_after: list["Finding"],
    repair_result: "RepairResult",
) -> GameResult:
    """
    Compute all metrics.

    A finding is RESOLVED if:
      finding_before.rule_id == X AND finding_before.resource_name == Y
      AND no finding_after has the same rule_id AND resource_name.

    A finding is INTRODUCED if:
      finding_after.rule_id == X AND finding_after.resource_name == Y
      AND no finding_before has the same rule_id AND resource_name.
    """
    hcl_valid = repair_result.hcl_valid

    # When HCL is invalid, no repaired code was written to disk and no scan
    # was performed.  An empty findings_after would be misleading (looks like
    # all issues resolved).  Treat it as "nothing changed".
    if not hcl_valid:
        findings_after = list(findings_before)

    before_keys = {_finding_key(f) for f in findings_before}
    after_keys = {_finding_key(f) for f in findings_after}

    resolved_count = len(before_keys - after_keys)
    introduced_count = len(after_keys - before_keys)
    unchanged_count = len(before_keys & after_keys)
    target_count = len(before_keys)

    repair_success_rate = resolved_count / target_count if target_count > 0 else 0.0
    regression_rate = introduced_count / target_count if target_count > 0 else 0.0
    net_delta = resolved_count - introduced_count

    # Class breakdown
    class_breakdown: dict[str, dict] = {}
    for cls in VULNERABILITY_CLASSES.keys():
        class_breakdown[cls] = {"resolved": 0, "introduced": 0, "unchanged": 0}

    for f in findings_before:
        cls = classify_finding(f)
        key = _finding_key(f)
        if key in after_keys:
            class_breakdown[cls]["unchanged"] += 1
        else:
            class_breakdown[cls]["resolved"] += 1

    for f in findings_after:
        cls = classify_finding(f)
        key = _finding_key(f)
        if key not in before_keys:
            class_breakdown[cls]["introduced"] += 1

    return GameResult(
        module_id=module_id,
        model_id=model_id,
        findings_before=findings_before,
        findings_after=findings_after,
        target_count=target_count,
        resolved_count=resolved_count,
        introduced_count=introduced_count,
        unchanged_count=unchanged_count,
        repair_success_rate=repair_success_rate,
        regression_rate=regression_rate,
        net_delta=net_delta,
        class_breakdown=class_breakdown,
        repair_cost_usd=repair_result.repair_cost_usd,
        repair_tokens=repair_result.repair_prompt_tokens + repair_result.repair_completion_tokens,
        hcl_valid=hcl_valid,
    )
