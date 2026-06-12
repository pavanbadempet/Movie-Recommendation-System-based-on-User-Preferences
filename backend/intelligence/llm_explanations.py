import collections
import hashlib
import logging
from typing import Any

from backend.intelligence.openrouter_client import chat_completion, configured_models, openrouter_api_key
from backend.intelligence.semantic_cache import get_semantic_cache, set_semantic_cache
from backend.intelligence.token_monitor import track_token_usage
from backend.serving.feature_store import feature_store

logger = logging.getLogger(__name__)

# Fallback in-memory LRU cache (OrderedDict gives O(1) eviction of oldest entry)
_EXPLANATION_CACHE_MAX = 10_000
_explanation_cache: collections.OrderedDict[str, str] = collections.OrderedDict()


def _generate_cache_key(user_id: str, movie_id: int, signals_hash: str) -> str:
    """Generate a unique cache key for this user/movie/signal combo."""
    # We hash the signals so if the recommendation rationale changes, we get a new explanation
    return f"llm_expl:{user_id}:{movie_id}:{signals_hash}"


def _get_cached_explanation(cache_key: str) -> str | None:
    """Try to retrieve from Redis, fallback to memory."""
    try:
        if feature_store.redis_client:
            val = feature_store.redis_client.get(cache_key)
            if val:
                return val.decode("utf-8")
    except Exception as e:
        logger.debug(f"Redis cache miss/error for explanations: {e}")

    return _explanation_cache.get(cache_key)


def _set_cached_explanation(cache_key: str, explanation: str, ttl_seconds: int = 86400 * 7):
    """Store in Redis for 7 days, fallback to memory."""
    try:
        if feature_store.redis_client:
            feature_store.redis_client.setex(cache_key, ttl_seconds, explanation)
    except Exception as e:
        logger.debug(f"Failed to set Redis cache for explanations: {e}")

    # Always keep in memory as a fast backup.
    # Evict the oldest single entry when the cache is full (true LRU behaviour).
    if len(_explanation_cache) >= _EXPLANATION_CACHE_MAX:
        _explanation_cache.popitem(last=False)
    _explanation_cache[cache_key] = explanation
    _explanation_cache.move_to_end(cache_key)


def _format_signals(movie: dict[str, Any]) -> str:
    """Extract and format the retrieval signals for the LLM with compression."""
    signals = movie.get("retrieval_signals", {})
    explanation_tags = movie.get("explanation", [])

    text = []
    if explanation_tags:
        # Compress: take first 2 tags only to reduce tokens
        text.append(f"Matches: {', '.join(explanation_tags[:2])}")

    if "genre_overlap" in signals:
        # Compress: remove labels, use compact format
        text.append(f"{signals['genre_overlap'] * 100:.0f}% genre match")

    return " | ".join(text)


def _compress_genres(genres: str | list) -> str:
    """Compress genre list to reduce tokens. Handles both string and list inputs."""
    if not genres:
        return "various genres"

    # Convert list to string if needed
    if isinstance(genres, list):
        genres = ", ".join(genres) if genres else "various genres"

    if genres == "various genres":
        return genres

    genre_list = [g.strip() for g in genres.split(",")]
    if len(genre_list) <= 2:
        return genres  # No compression needed for short lists

    # For long lists, take first 2 and add count
    return f"{', '.join(genre_list[:2])} +{len(genre_list) - 2} more"


def generate_explanation(user_id: str, movie: dict[str, Any], user_context: str | None = None) -> str:
    """
    Generate a 1-2 sentence personalized explanation using an LLM.
    Falls back to template if LLM is unavailable or fails.
    """
    movie_id = movie.get("id")
    title = movie.get("title", "this movie")
    genres = _compress_genres(movie.get("genres", "various genres"))

    # Hash the signals so the cache invalidates if the math changes.
    # usedforsecurity=False makes this safe on FIPS-enabled systems.
    signals_str = _format_signals(movie)
    signals_hash = hashlib.md5(signals_str.encode(), usedforsecurity=False).hexdigest()[:8]

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

    api_key = openrouter_api_key()
    if not api_key:
        return f"Recommended for you because: {signals_str}"

    # Construct prompt with token optimization and structured output
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

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

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

        # Track token usage for monitoring
        input_text = sys_prompt + user_prompt
        track_token_usage("explanation", input_text, explanation, model=models[0] if models else "unknown")

        # Store in semantic cache for future similar queries
        semantic_key = f"{title}_{genres}_{signals_str}"
        set_semantic_cache(semantic_key, explanation)

        _set_cached_explanation(cache_key, explanation)
        return explanation

    except Exception as e:
        logger.warning(f"LLM explanation generation failed: {e}")
        # Graceful fallback to template
        return f"Recommended based on your preferences: {signals_str}"
