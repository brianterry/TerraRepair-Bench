"""
AWS Bedrock wrapper supporting all six target models.

Handles cross-region inference profiles (us.anthropic, us.amazon, etc.).
"""
from __future__ import annotations

import json

import boto3
from botocore.config import Config as BotoConfig

SUPPORTED_MODELS = {
    # Non-reasoning (us. prefix = cross-region inference profile)
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "nova-pro": "us.amazon.nova-pro-v1:0",
    "nova-lite": "us.amazon.nova-lite-v1:0",
    "llama-3-3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    # Reasoning
    "deepseek-r1": "us.deepseek.r1-v1:0",
    "qwen3-32b": "qwen.qwen3-32b-v1:0",
}

# Models that need invoke_model instead of Converse
_INVOKE_MODEL_IDS = {"qwen"}


class BedrockClient:
    def __init__(self, region: str = "us-east-1", read_timeout: int = 300):
        config = BotoConfig(
            read_timeout=read_timeout,
            retries={"max_attempts": 2, "mode": "adaptive"},
        )
        self.client = boto3.client(
            "bedrock-runtime", region_name=region, config=config
        )

    @staticmethod
    def _extract_converse_text(content_blocks: list[dict]) -> str:
        """Extract text from Converse API content blocks.

        Reasoning models (DeepSeek-R1) may return reasoningContent blocks
        alongside or instead of text blocks. We prefer text blocks but fall
        back to reasoning text when the model exhausts tokens during reasoning.
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "reasoningContent" in block:
                rt = block["reasoningContent"].get("reasoningText", {})
                if isinstance(rt, dict):
                    reasoning_parts.append(rt.get("text", ""))
                elif isinstance(rt, str):
                    reasoning_parts.append(rt)
        return "".join(text_parts) if text_parts else "".join(reasoning_parts)

    def invoke(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> tuple[str, dict]:
        """
        Invoke Bedrock model. Returns (response_text, usage_dict).
        usage_dict contains input_tokens, output_tokens for cost tracking.
        Temperature 0.0 for deterministic repair.
        """
        bedrock_id = SUPPORTED_MODELS.get(model_id, model_id)

        if "claude" in bedrock_id:
            return self._invoke_claude(bedrock_id, prompt, max_tokens, temperature)

        if any(prefix in bedrock_id for prefix in _INVOKE_MODEL_IDS):
            return self._invoke_raw(bedrock_id, prompt, max_tokens, temperature)

        return self._invoke_converse(bedrock_id, prompt, max_tokens, temperature)

    def _invoke_claude(
        self, bedrock_id: str, prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, dict]:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self.client.invoke_model(
            modelId=bedrock_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        resp_body = json.loads(response["body"].read())
        text = ""
        for block in resp_body.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        usage = {
            "input_tokens": resp_body.get("usage", {}).get("input_tokens", 0),
            "output_tokens": resp_body.get("usage", {}).get("output_tokens", 0),
        }
        return text, usage

    def _invoke_converse(
        self, bedrock_id: str, prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, dict]:
        """Nova, Llama, DeepSeek via the Converse API."""
        response = self.client.converse(
            modelId=bedrock_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        content_blocks = response["output"]["message"]["content"]
        text = self._extract_converse_text(content_blocks)
        usage = {
            "input_tokens": response.get("usage", {}).get("inputTokens", 0),
            "output_tokens": response.get("usage", {}).get("outputTokens", 0),
        }
        return text, usage

    def _invoke_raw(
        self, bedrock_id: str, prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, dict]:
        """Fallback for models that don't support Converse (e.g. Qwen3)."""
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = self.client.invoke_model(
            modelId=bedrock_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        resp_body = json.loads(response["body"].read())
        # Qwen3 returns {"choices": [{"message": {"content": "..."}}]}
        choices = resp_body.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")
        else:
            text = resp_body.get("output", {}).get("text", "")
        usage_block = resp_body.get("usage", {})
        usage = {
            "input_tokens": usage_block.get("prompt_tokens", usage_block.get("input_tokens", 0)),
            "output_tokens": usage_block.get("completion_tokens", usage_block.get("output_tokens", 0)),
        }
        return text, usage
