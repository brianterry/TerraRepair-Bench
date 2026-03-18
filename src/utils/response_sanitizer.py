"""
Handle model-specific output quirks before writing repaired code.

Strips <think> blocks, markdown fences, preambles.
"""

import re


def sanitize_repair_response(raw_response: str, model_id: str) -> str:
    """
    Clean LLM response to extract pure HCL content.

    Handles:
    - DeepSeek-R1: strip <think>...</think> blocks before HCL
    - Qwen3: strip ```hcl ... ``` markdown fences
    - All models: strip leading/trailing whitespace
    - All models: strip "Here is the repaired code:" preambles

    Returns clean HCL string ready to write to disk.
    Raises ValueError if response appears to be refusal or empty after cleaning.
    """
    content = raw_response.strip()

    # Strip DeepSeek <think>...</think> blocks
    if "deepseek" in model_id.lower():
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # Strip markdown code fences (```hcl ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:hcl)?\s*\n(.*?)```", content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1)

    # Strip common preambles
    preambles = [
        r"Here is the repaired code:?\s*",
        r"Here's the repaired code:?\s*",
        r"Below is the repaired code:?\s*",
        r"The repaired Terraform code:?\s*",
        r"Here is the fixed code:?\s*",
    ]
    for p in preambles:
        content = re.sub(p, "", content, flags=re.IGNORECASE)

    content = content.strip()

    # Refusal / empty check
    refusal_phrases = [
        "i cannot",
        "i can't",
        "i'm unable",
        "i am unable",
        "as an ai",
        "i don't have",
        "i do not have",
    ]
    lower = content.lower()
    if len(content) < 50:
        raise ValueError("Response too short or empty after sanitization")
    if any(p in lower[:200] for p in refusal_phrases):
        raise ValueError("Model refused the repair request")

    return content


def is_valid_hcl(content: str) -> bool:
    """
    Basic HCL validity check without requiring terraform CLI.
    Checks: balanced braces, at least one resource block present,
    no obvious non-HCL content (Python, JSON, etc.)
    """
    if not content or len(content) < 10:
        return False

    # Reject obvious non-HCL
    bad_starts = ["import ", "def ", "class ", "function ", "<?php", "<%"]
    if any(content.strip().lower().startswith(s) for s in bad_starts):
        return False

    # Check balanced braces
    open_braces = content.count("{") - content.count("}")
    if open_braces != 0:
        return False

    # Must have at least one resource/data/provider block
    if "resource " not in content and "data " not in content and "provider " not in content:
        return False

    # Basic structural check
    return "resource " in content or "data " in content
