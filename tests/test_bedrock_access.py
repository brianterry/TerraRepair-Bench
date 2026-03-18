"""
Pre-flight checks: verify Bedrock access for all 6 models and validate
that each model produces usable output for the experiment pipeline.

Run before the full experiment:
    pytest tests/test_bedrock_access.py -v -s

These tests make real API calls and cost a small amount (~$0.01 total).
Skip with: pytest tests/ --ignore=tests/test_bedrock_access.py
"""
from __future__ import annotations

import pytest
import shutil
from pathlib import Path

from src.utils.bedrock_client import BedrockClient, SUPPORTED_MODELS
from src.utils.response_sanitizer import sanitize_repair_response, is_valid_hcl

TINY_HCL = """\
resource "aws_s3_bucket" "example" {
  bucket = "my-test-bucket"
}
"""

TINY_FINDING = """\
- [CKV_AWS_19] Ensure all data stored in the S3 bucket is securely encrypted at rest (severity: HIGH)
  Resource: aws_s3_bucket.example in main.tf:1-3"""

REPAIR_PROMPT = f"""\
You are a cloud security engineer. Repair this Terraform code to fix the identified misconfiguration.

MISCONFIGURATION:
{TINY_FINDING}

TERRAFORM CODE:
```hcl
{TINY_HCL}
```

Return ONLY the complete repaired Terraform code. No explanations, no markdown fences, no preamble."""


@pytest.fixture(scope="module")
def client():
    return BedrockClient()


class TestModelAccess:
    """Verify every model in SUPPORTED_MODELS responds to a trivial prompt."""

    REASONING_MODELS = {"deepseek-r1", "qwen3-32b"}

    @pytest.mark.parametrize("model_id", list(SUPPORTED_MODELS.keys()))
    def test_model_responds(self, client, model_id):
        # Reasoning models need more tokens — they consume budget on internal
        # chain-of-thought before producing visible output.
        tokens = 256 if model_id in self.REASONING_MODELS else 50
        text, usage = client.invoke(model_id, "Reply with only the word OK", max_tokens=tokens)
        assert text.strip(), f"{model_id} returned empty response"
        assert usage["input_tokens"] > 0, f"{model_id} reported 0 input tokens"
        assert usage["output_tokens"] > 0, f"{model_id} reported 0 output tokens"


class TestRepairOutput:
    """Verify each model can produce valid HCL from a minimal repair prompt."""

    @pytest.mark.parametrize("model_id", list(SUPPORTED_MODELS.keys()))
    def test_repair_produces_valid_hcl(self, client, model_id):
        raw_text, usage = client.invoke(model_id, REPAIR_PROMPT, max_tokens=1024)

        cleaned = sanitize_repair_response(raw_text, model_id)
        assert len(cleaned) > 50, f"{model_id} response too short after sanitization"

        assert is_valid_hcl(cleaned), (
            f"{model_id} produced invalid HCL.\n"
            f"First 200 chars: {cleaned[:200]}"
        )
        assert "aws_s3_bucket" in cleaned, f"{model_id} did not preserve the resource"

    @pytest.mark.parametrize("model_id", ["deepseek-r1"])
    def test_reasoning_model_think_blocks_stripped(self, client, model_id):
        raw_text, _ = client.invoke(model_id, REPAIR_PROMPT, max_tokens=2048)
        cleaned = sanitize_repair_response(raw_text, model_id)
        assert "<think>" not in cleaned, f"{model_id} still has <think> after sanitization"


class TestTokenCounting:
    """Verify token usage is reported for cost tracking."""

    @pytest.mark.parametrize("model_id", list(SUPPORTED_MODELS.keys()))
    def test_usage_nonzero(self, client, model_id):
        _, usage = client.invoke(model_id, "Say hello", max_tokens=20)
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0
