"""
LLM Explanation Generator Module.

This module generates personalized explanations for movie recommendations using
Large Language Models (LLMs). It implements a multi-layered caching strategy
to optimize performance and reduce API costs.

Key Features:
- Redis-backed distributed caching with in-memory LRU fallback
- Semantic caching for similar queries
- Token optimization through prompt compression
- Graceful degradation when LLM services are unavailable
"""

import collections
import hashlib
import logging
from typing import Any

from backend.intelligence.openrouter_client import chat_completion, configured_models, openrouter_api_key
from backend.intelligence.semantic_cache import get_semantic_cache, set_semantic_cache
from backend.intelligence.token_monitor import track_token_usage
from backend.serving.feature_store import feature_store

logger = logging.getLogger(__name__)

# ============================================================================
# Cache Configuration
# ============================================================================

# Maximum number of explanations to store in the in-memory LRU cache
# This provides a fast fallback when Redis is unavailable
_EXPLANATION_CACHE_MAX = 10_000

# In-memory LRU cache using OrderedDict for O(1) eviction of oldest entries
# OrderedDict maintains insertion order, allowing efficient LRU implementation
_explanation_cache: collections.OrderedDict[str, str] = collections.OrderedDict()


# ============================================================================
# Cache Management Functions
# ============================================================================


def _generate_cache_key(user_id: str, movie_id: int, signals_hash: str) -> str:
    """
    Generate a unique cache key for this user/movie/signal combination.

    The cache key incorporates the signals hash to ensure that if the
    recommendation rationale changes (e.g., due to model updates),
    a fresh explanation will be generated.

    Args:
        user_id: Unique identifier for the user
        movie_id: The movie ID being explained
        signals_hash: Hash of the retrieval signals that drove the recommendation

    Returns:
        A formatted cache key string
    """
    # Hash the signals so if the recommendation rationale changes, we get a new explanation
    return f"llm_expl:{user_id}:{movie_id}:{signals_hash}"


def _get_cached_explanation(cache_key: str) -> str | None:
    """
    Retrieve an explanation from cache, trying Redis first then falling back to memory.

    This implements a two-tier caching strategy:
    1. Redis (distributed, persistent) - checked first
    2. In-memory LRU cache (local, fast) - fallback if Redis fails

    Args:
        cache_key: The cache key to look up

    Returns:
        The cached explanation string if found, None otherwise
    """
    # Try Redis first (distributed cache)
    try:
        if feature_store.redis_client:
            val = feature_store.redis_client.get(cache_key)
            if val:
                return val.decode("utf-8")
    except Exception as e:
        logger.debug(f"Redis cache miss/error for explanations: {e}")

    # Fallback to in-memory cache
    return _explanation_cache.get(cache_key)


def _set_cached_explanation(cache_key: str, explanation: str, ttl_seconds: int = 86400 * 7):
    """
    Store an explanation in both Redis and in-memory cache.

    The explanation is stored in Redis with a TTL for persistence across
    restarts, and in the in-memory LRU cache for fast access.

    Args:
        cache_key: The cache key to store under
        explanation: The explanation text to cache
        ttl_seconds: Time-to-live for Redis cache (default: 7 days)
    """
    # Store in Redis for persistence across restarts
    try:
        if feature_store.redis_client:
            feature_store.redis_client.setex(cache_key, ttl_seconds, explanation)
    except Exception as e:
        logger.debug(f"Failed to set Redis cache for explanations: {e}")

    # Always keep in memory as a fast backup
    # Evict the oldest single entry when the cache is full (true LRU behaviour)
    if len(_explanation_cache) >= _EXPLANATION_CACHE_MAX:
        _explanation_cache.popitem(last=False)

    # Add to cache and move to end (mark as most recently used)
    _explanation_cache[cache_key] = explanation
    _explanation_cache.move_to_end(cache_key)


# ============================================================================
# Signal and Genre Formatting Functions
# ============================================================================


def _format_signals(movie: dict[str, Any]) -> str:
    """
    Extract and format retrieval signals for the LLM with token compression.

    This function compresses the retrieval signals to reduce token usage while
    preserving the most important information for generating explanations.

    Args:
        movie: Movie dictionary containing retrieval signals and explanation tags

    Returns:
        Compressed string representation of the signals
    """
    signals = movie.get("retrieval_signals", {})
    explanation_tags = movie.get("explanation", [])

    text = []

    # Compress: take first 2 tags only to reduce tokens
    if explanation_tags:
        text.append(f"Matches: {', '.join(explanation_tags[:2])}")

    # Compress: remove labels, use compact percentage format
    if "genre_overlap" in signals:
        text.append(f"{signals['genre_overlap'] * 100:.0f}% genre match")

    return " | ".join(text)


def _compress_genres(genres: str | list) -> str:
    """
    Compress genre list to reduce token usage in LLM prompts.

    Handles both string and list inputs, compressing long genre lists
    to a more compact representation.

    Args:
        genres: Either a comma-separated string or list of genre names

    Returns:
        Compressed genre string (e.g., "Action, Drama +3 more")
    """
    if not genres:
        return "various genres"

    # Convert list to string if needed
    if isinstance(genres, list):
        genres = ", ".join(genres) if genres else "various genres"

    if genres == "various genres":
        return genres

    genre_list = [g.strip() for g in genres.split(",")]

    # No compression needed for short lists
    if len(genre_list) <= 2:
        return genres

    # For long lists, take first 2 and add count
    return f"{', '.join(genre_list[:2])} +{len(genre_list) - 2} more"


# ============================================================================
# Main Explanation Generation Function
# ============================================================================


def generate_explanation(user_id: str, movie: dict[str, Any], user_context: str | None = None) -> str:
    """
    Generate a 1-2 sentence personalized explanation using an LLM.

    This function implements a multi-layered caching strategy:
    1. Exact cache lookup (user + movie + signals combination)
    2. Semantic cache lookup (similar movie/genre/signal combinations)
    3. LLM generation with prompt optimization
    4. Graceful fallback to template if LLM fails

    The function is designed to be fast (2.5s timeout) and cost-effective
    through token compression and caching.

    Args:
        user_id: Unique identifier for the user
        movie: Movie dictionary containing id, title, genres, and retrieval signals
        user_context: Optional user taste/context information for personalization

    Returns:
        A 1-2 sentence explanation string
    """
    # Extract movie information
    movie_id = movie.get("id")
    title = movie.get("title", "this movie")
    genres = _compress_genres(movie.get("genres", "various genres"))

    # Hash the signals so the cache invalidates if the recommendation rationale changes
    # usedforsecurity=False makes this safe on FIPS-enabled systems
    signals_str = _format_signals(movie)

    import os
    disable_llm = os.getenv("NOVA_DISABLE_LLM_EXPLANATIONS", "").strip().lower()
    is_tier3 = os.getenv("NOVA_SERVING_TIER", "").strip().lower() == "tier3"
    is_low_mem = os.getenv("NOVA_LOW_MEMORY", "").strip().lower() in {"1", "true", "yes", "on"}

    if disable_llm in {"1", "true", "yes", "on"} or (disable_llm == "" and (is_tier3 or is_low_mem)):
        return f"Recommended for you because: {signals_str}"

    signals_hash = hashlib.md5(signals_str.encode(), usedforsecurity=False).hexdigest()[:8]

    # Try exact cache first
    cache_key = _generate_cache_key(user_id, movie_id, signals_hash)
    cached = _get_cached_explanation(cache_key)
    if cached:
        return cached

    # Try semantic cache for similar queries
    semantic_key = f"{title}_{genres}_{signals_str}"
    semantic_cached = get_semantic_cache(semantic_key)
    if semantic_cached:
        logger.debug(f"Semantic cache hit for explanation: {semantic_key[:50]}...")
        _set_cached_explanation(cache_key, semantic_cached)  # Backfill exact cache
        return semantic_cached

    # Check if LLM API is available
    api_key = openrouter_api_key()
    if not api_key:
        return f"Recommended for you because: {signals_str}"

    # Construct system prompt with strict constraints for token optimization
    sys_prompt = (
        "Expert movie recommender. Explain why a user was recommended a movie in 1 concise sentence. "
        "Never mention 'vector', 'AI', 'algorithm'. Speak directly: 'Because you enjoyed X, you'll love Y'. "
        "Max 15 words. Return plain text only."
    )

    # Compress user prompt to reduce tokens
    user_prompt_parts = [f"Movie: {title}", f"Genres: {genres}"]
    if signals_str:
        user_prompt_parts.append(f"Why: {signals_str}")
    if user_context:
        user_prompt_parts.append(f"Taste: {user_context[:100]}")  # Truncate long context

    user_prompt = ". ".join(user_prompt_parts)

    # Prepare messages for LLM
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

    # Get configured models for explanation generation
    models = configured_models("NOVA_EXPLANATION_MODELS")

    try:
        # Use fast model for simple explanation tasks (model routing optimization)
        explanation = chat_completion(
            messages=messages,
            models=models,
            temperature=0.7,
            timeout_seconds=2.5,  # Fast timeout, we don't want to block the API for long
            api_key=api_key,
            max_tokens=50,  # Limit output tokens for cost savings
            use_fast_model=True,  # Use cheaper/faster models for simple tasks
            enable_prompt_caching=True,  # Enable prompt caching for 90% savings on repeated prefixes
        )

        # Clean up any quotes the LLM might have added
        explanation = explanation.strip("\"'")

        # Track token usage for monitoring and cost optimization
        input_text = sys_prompt + user_prompt
        track_token_usage("explanation", input_text, explanation, model=models[0] if models else "unknown")

        # Store in semantic cache for future similar queries
        semantic_key = f"{title}_{genres}_{signals_str}"
        set_semantic_cache(semantic_key, explanation)

        # Store in exact cache
        _set_cached_explanation(cache_key, explanation)
        return explanation

    except Exception as e:
        logger.warning(f"LLM explanation generation failed: {e}")
        # Graceful fallback to template
        return f"Recommended based on your preferences: {signals_str}"
