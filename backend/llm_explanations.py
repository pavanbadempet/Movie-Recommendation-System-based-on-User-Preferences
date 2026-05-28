import logging
import json
import hashlib
from typing import Any, Dict, Optional

from backend.openrouter_client import chat_completion, configured_models, openrouter_api_key
from backend.feature_store import feature_store

logger = logging.getLogger(__name__)

# Fallback in-memory cache if Redis is completely unavailable
_explanation_cache: Dict[str, str] = {}

def _generate_cache_key(user_id: str, movie_id: int, signals_hash: str) -> str:
    """Generate a unique cache key for this user/movie/signal combo."""
    # We hash the signals so if the recommendation rationale changes, we get a new explanation
    return f"llm_expl:{user_id}:{movie_id}:{signals_hash}"

def _get_cached_explanation(cache_key: str) -> Optional[str]:
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
        
    # Always keep in memory as a fast backup, bound size to prevent leaks
    if len(_explanation_cache) > 10000:
        _explanation_cache.clear()
    _explanation_cache[cache_key] = explanation

def _format_signals(movie: Dict[str, Any]) -> str:
    """Extract and format the retrieval signals for the LLM."""
    signals = movie.get("retrieval_signals", {})
    explanation_tags = movie.get("explanation", [])
    
    text = []
    if explanation_tags:
        text.append(f"Matching factors: {', '.join(explanation_tags)}")
    
    if "genre_overlap" in signals:
        text.append(f"Genre Overlap Match: {signals['genre_overlap']*100:.0f}%")
        
    return " | ".join(text)

def generate_explanation(user_id: str, movie: Dict[str, Any], user_context: Optional[str] = None) -> str:
    """
    Generate a 1-2 sentence personalized explanation using an LLM.
    Falls back to template if LLM is unavailable or fails.
    """
    movie_id = movie.get("id")
    title = movie.get("title", "this movie")
    genres = movie.get("genres", "various genres")
    
    # Hash the signals so the cache invalidates if the math changes
    signals_str = _format_signals(movie)
    signals_hash = hashlib.md5(signals_str.encode()).hexdigest()[:8]
    
    cache_key = _generate_cache_key(user_id, movie_id, signals_hash)
    cached = _get_cached_explanation(cache_key)
    if cached:
        return cached
        
    api_key = openrouter_api_key()
    if not api_key:
        return f"Recommended for you because: {signals_str}"
        
    # Construct prompt
    sys_prompt = (
        "You are an expert movie recommender engine (like Netflix or Amazon Prime). "
        "Your job is to explain why a user was recommended a specific movie in 1 concise, engaging sentence. "
        "Do NOT mention 'vector similarity' or 'AI' or 'algorithm'. "
        "Speak directly to the user (e.g. 'Because you enjoyed X, you will love Y')."
    )
    
    user_prompt = f"Movie Recommended: {title}\nGenres: {genres}\nAlgorithmic Reasons: {signals_str}\n"
    if user_context:
        user_prompt += f"\nUser's Taste Profile: {user_context}"
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    models = configured_models("NOVA_EXPLANATION_MODELS")
    
    try:
        explanation = chat_completion(
            messages=messages,
            models=models,
            temperature=0.7,
            timeout_seconds=2.5, # Fast timeout, we don't want to block the API for long
            api_key=api_key
        )
        
        # Clean up any quotes the LLM might have added
        explanation = explanation.strip('"\'')
        
        _set_cached_explanation(cache_key, explanation)
        return explanation
        
    except Exception as e:
        logger.warning(f"LLM explanation generation failed: {e}")
        # Graceful fallback to template
        return f"Recommended based on your preferences: {signals_str}"
