"""
Single LLM call to repair Terraform code given scanner findings.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.scanning.scanner_normalizer import Finding
    from src.utils.bedrock_client import BedrockClient

REPAIR_PROMPT = """You are a cloud security engineer. You have been given
Terraform code with confirmed security misconfigurations identified by
static analysis tools.

Your task: repair the Terraform code to fix the identified misconfigurations.

REQUIREMENTS:
1. Fix ALL identified misconfigurations
2. Preserve the original infrastructure intent and resource names exactly
3. Do not add resources, modules, or variables that weren't in the original
4. Generate valid, deployable Terraform (HCL) syntax
5. Do not add comments explaining your changes

MISCONFIGURATIONS TO FIX:
{findings_formatted}

TERRAFORM CODE:
```hcl
{terraform_code}
```

Return ONLY the complete repaired Terraform code. No explanations, no
markdown fences, no preamble. Start directly with the HCL content."""


@dataclass
class RepairResult:
    model_id: str
    module_id: str
    original_code: str
    repaired_code: str
    repaired_dir: str | None  # Temp dir with repaired .tf files; caller must clean up
    repair_prompt_tokens: int
    repair_completion_tokens: int
    repair_cost_usd: float
    success: bool  # False if LLM refused or returned invalid HCL
    hcl_valid: bool  # True if repaired code passes basic HCL structural validation
    error: str | None


# Approximate cost per 1K tokens (USD) - update as needed
MODEL_COSTS = {
    "claude-3-5-sonnet": (0.003, 0.015),
    "nova-pro": (0.001, 0.001),
    "nova-lite": (0.0001, 0.0002),
    "llama-3-3-70b": (0.0008, 0.0008),
    "deepseek-r1": (0.0014, 0.0028),
    "qwen3-32b": (0.0004, 0.0004),
}


def _format_findings(findings: list["Finding"]) -> str:
    lines = []
    for f in findings:
        lines.append(
            f"- [{f.rule_id}] {f.title} (severity: {f.severity})\n"
            f"  Resource: {f.resource_name} in {f.file_path}:{f.start_line}-{f.end_line}"
        )
    return "\n".join(lines)


def _read_module_code(module_dir: str) -> str:
    """Concatenate all .tf files in module_dir into a single string."""
    path = Path(module_dir)
    parts = []
    for tf_file in sorted(path.glob("*.tf")):
        parts.append(tf_file.read_text())
    return "\n\n".join(parts)


def repair_module(
    module_dir: str,
    findings: list["Finding"],
    model_id: str,
    bedrock_client: "BedrockClient",
    max_tokens: int = 8192,
) -> RepairResult:
    """
    Concatenate all .tf files in module_dir into a single string.
    Format findings into the prompt.
    Call the LLM via Bedrock.
    Parse and sanitize the response.
    Write repaired code to a temp directory.
    Return RepairResult with original code, repaired code, and metadata.
    """
    from src.utils.response_sanitizer import is_valid_hcl, sanitize_repair_response

    module_id = Path(module_dir).name
    original_code = _read_module_code(module_dir)
    findings_formatted = _format_findings(findings)

    prompt = REPAIR_PROMPT.format(
        findings_formatted=findings_formatted,
        terraform_code=original_code,
    )

    success = False
    repaired_code = ""
    repaired_dir = None
    hcl_valid = False
    error_msg = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        response_text, usage = bedrock_client.invoke(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        repaired_code = sanitize_repair_response(response_text, model_id)
        success = bool(repaired_code)  # True if LLM returned non-empty response
        hcl_valid = is_valid_hcl(repaired_code) if success else False

        if success and hcl_valid:
            # Write repaired code to temp dir for scanning
            tmp = tempfile.mkdtemp()
            (Path(tmp) / "main.tf").write_text(repaired_code)
            repaired_dir = tmp
        elif success and not hcl_valid:
            error_msg = "Repaired code failed HCL validation"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"

    # Cost estimation
    costs = MODEL_COSTS.get(model_id, (0.001, 0.001))
    cost_usd = (prompt_tokens / 1000 * costs[0]) + (completion_tokens / 1000 * costs[1])

    return RepairResult(
        model_id=model_id,
        module_id=module_id,
        original_code=original_code,
        repaired_code=repaired_code,
        repaired_dir=repaired_dir,
        repair_prompt_tokens=prompt_tokens,
        repair_completion_tokens=completion_tokens,
        repair_cost_usd=cost_usd,
        success=success,
        hcl_valid=hcl_valid,
        error=error_msg,
    )
