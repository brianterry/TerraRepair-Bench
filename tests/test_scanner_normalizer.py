"""Test scanner normalizer: Trivy/Checkov JSON → Finding, deduplication."""

import pytest

from src.scanning.scanner_normalizer import (
    Finding,
    deduplicate_findings,
    normalize_checkov_findings,
    normalize_trivy_findings,
)


def test_normalize_trivy_findings():
    raw = [
        {
            "ID": "AVD-AWS-0086",
            "Title": "S3 bucket does not have versioning enabled",
            "Severity": "LOW",
            "Target": "main.tf",
            "CauseMetadata": {
                "Resource": "aws_s3_bucket.example",
                "StartLine": 12,
                "EndLine": 18,
            },
        }
    ]
    findings = normalize_trivy_findings(raw)
    assert len(findings) == 1
    f = findings[0]
    assert f.tool == "trivy"
    assert f.rule_id == "AVD-AWS-0086"
    assert f.resource_name == "aws_s3_bucket.example"
    assert f.resource_type == "aws_s3_bucket"
    assert f.start_line == 12
    assert f.end_line == 18


def test_normalize_checkov_findings():
    raw = [
        {
            "check_id": "CKV_AWS_19",
            "check_name": "Ensure S3 encryption",
            "severity": "HIGH",
            "file_path": "/main.tf",
            "resource": "aws_s3_bucket.example",
            "file_line_range": [1, 20],
        }
    ]
    findings = normalize_checkov_findings(raw)
    assert len(findings) == 1
    f = findings[0]
    assert f.tool == "checkov"
    assert f.rule_id == "CKV_AWS_19"
    assert f.title == "Ensure S3 encryption"
    assert f.severity == "HIGH"
    assert f.resource_name == "aws_s3_bucket.example"
    assert f.resource_type == "aws_s3_bucket"
    assert f.start_line == 1
    assert f.end_line == 20


def test_deduplicate_findings():
    f1 = Finding("trivy", "AVD-AWS-0086", "x", "LOW", "aws_s3_bucket.a", "aws_s3_bucket", "", 1, 2)
    f2 = Finding("checkov", "AVD-AWS-0086", "y", "LOW", "aws_s3_bucket.a", "aws_s3_bucket", "", 1, 2)
    f3 = Finding("trivy", "AVD-AWS-0086", "z", "LOW", "aws_s3_bucket.b", "aws_s3_bucket", "", 3, 4)
    deduped = deduplicate_findings([f1, f2, f3])
    # f1 and f2 have same (rule_id, resource_name) -> keep first
    assert len(deduped) == 2
    assert deduped[0].resource_name == "aws_s3_bucket.a"
    assert deduped[1].resource_name == "aws_s3_bucket.b"
