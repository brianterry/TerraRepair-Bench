"""
AWS Bedrock wrapper supporting all six target models.

Handles cross-region inference profiles (us.anthropic, us.amazon, etc.).
"""

import json

import boto3

SUPPORTED_MODELS = {
    "claude-3-5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "nova-pro": "us.amazon.nova-pro-v1:0",
    "nova-lite": "us.amazon.nova-lite-v1:0",
    "llama-3-3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "deepseek-r1": "us.deepseek.deepseek-r1:0",
    "qwen3-32b": "us.alibaba.qwen3-32b-instruct:0",
}


class BedrockClient:
    def __init__(self, region: str = "us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)

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

        # Claude / Nova use messages API; others may use Converse
        if "claude" in bedrock_id or "nova" in bedrock_id:
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
        else:
            # Llama, DeepSeek, Qwen use Converse API
            body = {
                "inferenceConfig": {
                    "maxNewTokens": max_tokens,
                    "temperature": temperature,
                },
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
            }
            response = self.client.converse(
                modelId=bedrock_id,
                messages=body["messages"],
                inferenceConfig=body["inferenceConfig"],
            )
            text = response["output"]["message"]["content"][0]["text"]
            usage = {
                "input_tokens": response.get("usage", {}).get("inputTokens", 0),
                "output_tokens": response.get("usage", {}).get("outputTokens", 0),
            }

        return text, usage
