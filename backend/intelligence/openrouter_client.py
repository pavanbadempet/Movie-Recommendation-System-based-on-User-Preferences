"""Small OpenRouter client helpers.

OpenRouter model IDs change over time and free providers can be rate-limited.
Keeping the model list configurable prevents stale hard-coded IDs from slowing
down the recommendation path.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_MODELS = [
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
]


def configured_models(env_name: str) -> list[str]:
    """Return configured OpenRouter models for a feature."""
    raw_value = os.getenv(env_name, "") or os.getenv("OPENROUTER_MODELS", "") or ",".join(DEFAULT_OPENROUTER_MODELS)
    models = [model.strip() for model in raw_value.split(",") if model.strip()]
    return models or DEFAULT_OPENROUTER_MODELS


# Fast/cheap models for simple tasks (model routing optimization)
FAST_MODELS = [
    "google/gemma-3-27b-it:free",
    "openai/gpt-oss-20b:free",
]

# High-quality models for complex tasks
QUALITY_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


def openrouter_api_key() -> str | None:
    """Return the configured OpenRouter API key, if present."""
    value = os.getenv("OPENROUTER_API_KEY", "").strip()
    return value or None


def _error_detail(response: requests.Response | None) -> str:
    if response is None:
        return ""
    try:
        return f" Response: {response.text[:240]}"
    except Exception:
        return ""


def chat_completion(
    *,
    messages: list[dict[str, Any]],
    models: list[str],
    temperature: float,
    timeout_seconds: float,
    api_key: str,
    max_tokens: int | None = None,
    use_fast_model: bool = False,
    response_format: dict[str, str] | None = None,
    enable_prompt_caching: bool = True,
) -> str:
    """Call OpenRouter with model fallbacks and return assistant content.
    
    Args:
        use_fast_model: If True, prioritize fast/cheap models for simple tasks (model routing optimization).
        response_format: Optional response format for structured output (e.g., {"type": "json_object"}).
        enable_prompt_caching: If True, enable prompt caching for repeated system prompts (cost optimization).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/pavanbadempet/Movie-Recommendation-System",
        "X-Title": "Nova Recommendation Intelligence",
    }

    # Model routing: use fast models for simple tasks
    model_list = FAST_MODELS if use_fast_model else models

    last_error = None
    for model in model_list:
        response = None
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format
        if enable_prompt_caching:
            # Enable prompt caching for cost optimization (90% savings on cached prefixes)
            # This is supported by OpenRouter and compatible providers
            payload["cache"] = {"read": {"enabled": True}, "write": {"enabled": True}}
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            last_error = f"{exc}{_error_detail(response)}"
            logger.warning("OpenRouter model %s failed: %s", model, last_error)

    raise ValueError(f"OpenRouter API error. All model fallbacks failed. Last error: {last_error}")
