from __future__ import annotations
"""Test delta scorer: resolved, introduced, metrics."""

from unittest.mock import MagicMock

from src.scanning.scanner_normalizer import Finding
from src.scoring.delta_scorer import classify_finding, score_repair


def _mk_finding(rule_id: str, resource_name: str, tool: str = "trivy", title: str = "") -> Finding:
    return Finding(
        tool=tool,
        rule_id=rule_id,
        title=title,
        severity="HIGH",
        resource_name=resource_name,
        resource_type=resource_name.split(".")[0],
        file_path="main.tf",
        start_line=1,
        end_line=2,
    )


def _mk_repair_result(success: bool = True, cost: float = 0.01, hcl_valid: bool | None = None):
    r = MagicMock()
    r.success = success
    r.hcl_valid = hcl_valid if hcl_valid is not None else success
    r.repair_cost_usd = cost
    r.repair_prompt_tokens = 100
    r.repair_completion_tokens = 50
    return r


def test_perfect_repair():
    """All resolved, none introduced."""
    before = [
        _mk_finding("AVD-AWS-0086", "aws_s3_bucket.a"),
        _mk_finding("CKV_AWS_19", "aws_s3_bucket.b"),
    ]
    after = []
    result = score_repair("m1", "claude", before, after, _mk_repair_result())
    assert result.resolved_count == 2
    assert result.introduced_count == 0
    assert result.unchanged_count == 0
    assert result.repair_success_rate == 1.0
    assert result.regression_rate == 0.0
    assert result.net_delta == 2


def test_perfect_regression():
    """None resolved, all introduced (repair added new issues)."""
    before = [
        _mk_finding("AVD-AWS-0086", "aws_s3_bucket.a"),
        _mk_finding("CKV_AWS_19", "aws_s3_bucket.b"),
    ]
    after = [
        _mk_finding("AVD-AWS-0086", "aws_s3_bucket.a"),  # unchanged
        _mk_finding("CKV_AWS_19", "aws_s3_bucket.b"),  # unchanged
        _mk_finding("AVD-AWS-0104", "aws_security_group.x"),  # introduced
        _mk_finding("CKV_AWS_25", "aws_security_group.y"),  # introduced
    ]
    result = score_repair("m1", "claude", before, after, _mk_repair_result())
    assert result.resolved_count == 0
    assert result.introduced_count == 2
    assert result.unchanged_count == 2
    assert result.repair_success_rate == 0.0
    assert result.regression_rate == 1.0  # 2 introduced / 2 target
    assert result.net_delta == -2


def test_partial_repair_with_regression():
    """Some resolved, some introduced."""
    before = [
        _mk_finding("AVD-AWS-0086", "aws_s3_bucket.a"),
        _mk_finding("CKV_AWS_19", "aws_s3_bucket.b"),
    ]
    after = [
        _mk_finding("CKV_AWS_19", "aws_s3_bucket.b"),  # unchanged
        _mk_finding("AVD-AWS-0104", "aws_security_group.x"),  # introduced
    ]
    result = score_repair("m1", "claude", before, after, _mk_repair_result())
    assert result.resolved_count == 1  # aws_s3_bucket.a gone
    assert result.introduced_count == 1  # aws_security_group.x new
    assert result.unchanged_count == 1  # aws_s3_bucket.b
    assert result.repair_success_rate == 0.5
    assert result.regression_rate == 0.5
    assert result.net_delta == 0


def test_unchanged_findings():
    """All findings unchanged."""
    before = [_mk_finding("AVD-AWS-0086", "aws_s3_bucket.a")]
    after = [_mk_finding("AVD-AWS-0086", "aws_s3_bucket.a")]
    result = score_repair("m1", "claude", before, after, _mk_repair_result())
    assert result.resolved_count == 0
    assert result.introduced_count == 0
    assert result.unchanged_count == 1
    assert result.repair_success_rate == 0.0
    assert result.regression_rate == 0.0


def test_empty_findings_before():
    """Edge case: no findings before."""
    before = []
    after = [_mk_finding("AVD-AWS-0086", "aws_s3_bucket.a")]
    result = score_repair("m1", "claude", before, after, _mk_repair_result())
    assert result.target_count == 0
    assert result.repair_success_rate == 0.0
    assert result.regression_rate == 0.0
    assert result.introduced_count == 1


def test_target_count_uses_deduplicated_keys():
    """Cross-scanner duplicates (same rule_id+resource) count as one target."""
    before = [
        _mk_finding("AVD-AWS-0086", "aws_s3_bucket.a", tool="trivy"),
        _mk_finding("AVD-AWS-0086", "aws_s3_bucket.a", tool="checkov"),  # duplicate
    ]
    after = []
    result = score_repair("m1", "claude", before, after, _mk_repair_result())
    assert result.target_count == 1
    assert result.resolved_count == 1


def test_empty_findings_after():
    """Edge case: no findings after (perfect repair)."""
    before = [_mk_finding("AVD-AWS-0086", "aws_s3_bucket.a")]
    after = []
    result = score_repair("m1", "claude", before, after, _mk_repair_result())
    assert result.resolved_count == 1
    assert result.introduced_count == 0
    assert result.repair_success_rate == 1.0


def test_hcl_valid_from_repair_result():
    """GameResult.hcl_valid reflects repair_result.hcl_valid, not success."""
    before = [_mk_finding("AVD-AWS-0086", "aws_s3_bucket.a")]
    after = []
    r = _mk_repair_result(success=True, hcl_valid=False)  # LLM returned but invalid HCL
    result = score_repair("m1", "claude", before, after, r)
    assert result.hcl_valid is False


def test_classify_finding():
    # access_control: security group, ingress
    assert classify_finding(_mk_finding("AVD-AWS-0107", "x", title="An ingress security group rule allows traffic from /0")) == "access_control"
    # data_protection: encrypt
    assert classify_finding(_mk_finding("CKV_AWS_19", "x", title="Ensure all data stored in S3 bucket is encrypted")) == "data_protection"
    # observability: logging
    assert classify_finding(_mk_finding("CKV_AWS_86", "x", title="Ensure S3 bucket has access logging enabled")) == "observability"
    # other: no matching keywords
    assert classify_finding(_mk_finding("UNKNOWN-123", "x", title="Some other check")) == "other"
