"""Token usage monitoring and tracking for LLM API calls.

This module provides utilities to track token usage across LLM calls,
enabling cost monitoring and optimization insights.
"""

from collections import defaultdict
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# In-memory token tracking (can be replaced with Redis/database for persistence)
_token_stats = defaultdict(
    lambda: {
        "total_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost_estimate": 0.0,
        "last_call_time": None,
    }
)


def estimate_tokens(text: str) -> int:
    """Token estimation with fallback to tiktoken if available."""
    try:
        import tiktoken

        # Use cl100k_base encoding (GPT-4, Claude, etc.)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback to rough estimation (approximately 4 chars per token)
        return len(text) // 4


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "default") -> float:
    """Estimate API cost based on token usage.

    Uses conservative pricing estimates for OpenRouter free tier models.
    Adjust these values based on your actual pricing.
    """
    # Conservative estimates (can be updated based on actual model pricing)
    input_cost_per_1m = 0.10  # $0.10 per million input tokens
    output_cost_per_1m = 0.30  # $0.30 per million output tokens

    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m

    return input_cost + output_cost


def track_token_usage(
    feature: str,
    input_text: str,
    output_text: str,
    model: str = "default",
) -> dict[str, Any]:
    """Track token usage for an LLM call.

    Args:
        feature: Name of the feature using LLM (e.g., "explanation", "rerank")
        input_text: Input prompt text
        output_text: Output response text
        model: Model name used

    Returns:
        Dictionary with token statistics for this call
    """
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    cost = estimate_cost(input_tokens, output_tokens, model)

    stats = _token_stats[feature]
    stats["total_calls"] += 1
    stats["total_input_tokens"] += input_tokens
    stats["total_output_tokens"] += output_tokens
    stats["total_cost_estimate"] += cost
    stats["last_call_time"] = time.time()

    call_stats = {
        "feature": feature,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_estimate": cost,
        "model": model,
    }

    logger.debug(f"Token usage - {feature}: {input_tokens} input, {output_tokens} output, ${cost:.6f}")

    return call_stats


def get_token_stats(feature: str | None = None) -> dict[str, Any]:
    """Get token usage statistics.

    Args:
        feature: If provided, return stats for specific feature only.
                 If None, return stats for all features.

    Returns:
        Dictionary with token statistics
    """
    if feature:
        return dict(_token_stats[feature])

    return {k: dict(v) for k, v in _token_stats.items()}


def reset_token_stats(feature: str | None = None):
    """Reset token usage statistics.

    Args:
        feature: If provided, reset stats for specific feature only.
                 If None, reset all stats.
    """
    if feature:
        _token_stats[feature] = defaultdict(
            lambda: {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_estimate": 0.0,
                "last_call_time": None,
            }
        )[feature]
    else:
        _token_stats.clear()


def log_token_summary():
    """Log a summary of token usage across all features."""
    total_cost = sum(stats["total_cost_estimate"] for stats in _token_stats.values())
    total_calls = sum(stats["total_calls"] for stats in _token_stats.values())

    logger.info(f"Token Usage Summary - Total calls: {total_calls}, Total cost: ${total_cost:.4f}")

    for feature, stats in _token_stats.items():
        if stats["total_calls"] > 0:
            avg_input = stats["total_input_tokens"] / stats["total_calls"]
            avg_output = stats["total_output_tokens"] / stats["total_calls"]
            logger.info(
                f"  {feature}: {stats['total_calls']} calls, "
                f"avg {avg_input:.0f} input / {avg_output:.0f} output tokens, "
                f"${stats['total_cost_estimate']:.4f}"
            )


# Optional: Persist stats to file for long-term tracking
def persist_token_stats(filepath: str = "data/token_stats.json"):
    """Persist token statistics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(get_token_stats(), f, indent=2)
        logger.info(f"Token stats persisted to {filepath}")
    except Exception as e:
        logger.warning(f"Failed to persist token stats: {e}")


def load_token_stats(filepath: str = "data/token_stats.json") -> dict[str, Any]:
    """Load token statistics from a JSON file."""
    try:
        with open(filepath) as f:
            stats = json.load(f)
        logger.info(f"Token stats loaded from {filepath}")
        return stats
    except FileNotFoundError:
        logger.debug(f"No token stats file found at {filepath}")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load token stats: {e}")
        return {}
