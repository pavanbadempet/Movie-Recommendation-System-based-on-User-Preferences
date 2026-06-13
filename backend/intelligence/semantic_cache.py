"""Semantic caching for LLM queries.

This module provides semantic similarity-based caching for LLM queries,
allowing cache hits for semantically similar queries even if they're not exact matches.
"""

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Simple semantic cache using exact matching with normalization
# For production, consider using vector embeddings with similarity search
_semantic_cache: dict[str, Any] = {}
_semantic_cache_max = 5000


def _normalize_query(query: str) -> str:
    """Normalize query for semantic matching."""
    # Basic normalization: lowercase, remove extra whitespace
    normalized = " ".join(query.lower().split())
    return normalized


def _generate_semantic_key(prompt: str, context: str = "") -> str:
    """Generate a semantic cache key."""
    # Create a normalized representation
    normalized = _normalize_query(prompt + " " + context)
    # Use hash for efficient lookup
    return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()


def get_semantic_cache(prompt: str, context: str = "", threshold: float = 0.8) -> str | None:
    """Get cached response for semantically similar query.

    Args:
        prompt: The input prompt
        context: Additional context (optional)
        threshold: Similarity threshold (not used in simple implementation)

    Returns:
        Cached response if found, None otherwise
    """
    key = _generate_semantic_key(prompt, context)
    entry = _semantic_cache.get(key)
    if entry and isinstance(entry, dict):
        return entry.get("response")
    return entry


def set_semantic_cache(prompt: str, response: str, context: str = "", ttl_seconds: int = 3600):
    """Cache response for semantic matching.

    Args:
        prompt: The input prompt
        response: The LLM response to cache
        context: Additional context (optional)
        ttl_seconds: Time-to-live in seconds (not enforced in simple implementation)
    """
    key = _generate_semantic_key(prompt, context)

    # Evict oldest entries if cache is full
    if len(_semantic_cache) >= _semantic_cache_max:
        # Simple FIFO eviction (could be improved with LRU)
        oldest_key = next(iter(_semantic_cache))
        del _semantic_cache[oldest_key]

    _semantic_cache[key] = {
        "response": response,
        "prompt": prompt,
        "context": context,
        "timestamp": __import__("time").time(),
    }


def clear_semantic_cache():
    """Clear the semantic cache."""
    _semantic_cache.clear()
    logger.info("Semantic cache cleared")


def get_semantic_cache_stats() -> dict[str, Any]:
    """Get statistics about the semantic cache."""
    return {
        "size": len(_semantic_cache),
        "max_size": _semantic_cache_max,
        "utilization": len(_semantic_cache) / _semantic_cache_max if _semantic_cache_max > 0 else 0,
    }


def persist_semantic_cache(filepath: str = "data/semantic_cache.json"):
    """Persist semantic cache to disk."""
    try:
        import os

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(_semantic_cache, f, indent=2)
        logger.info(f"Semantic cache persisted to {filepath}")
    except Exception as e:
        logger.warning(f"Failed to persist semantic cache: {e}")


def load_semantic_cache(filepath: str = "data/semantic_cache.json"):
    """Load semantic cache from disk."""
    global _semantic_cache
    try:
        import os

        if os.path.exists(filepath):
            with open(filepath) as f:
                _semantic_cache = json.load(f)
            logger.info(f"Semantic cache loaded from {filepath} ({len(_semantic_cache)} entries)")
    except Exception as e:
        logger.warning(f"Failed to load semantic cache: {e}")


# Advanced: Vector-based semantic caching (requires sentence-transformers)
# Uncomment if you want to use embedding-based similarity
"""
def get_semantic_cache_vector(prompt: str, context: str = "", threshold: float = 0.85) -> str | None:
    \"\"\"Get cached response using vector similarity.\"\"\"
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if not hasattr(get_semantic_cache_vector, 'model'):
            get_semantic_cache_vector.model = SentenceTransformer('all-MiniLM-L6-v2')

        model = get_semantic_cache_vector.model
        query_embedding = model.encode(prompt + " " + context)

        best_match = None
        best_similarity = 0.0

        for key, entry in _semantic_cache.items():
            cached_embedding = entry.get('embedding')
            if cached_embedding is not None:
                similarity = np.dot(query_embedding, cached_embedding)
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = entry['response']

        return best_match
    except ImportError:
        logger.debug("sentence-transformers not available, falling back to simple cache")
        return get_semantic_cache(prompt, context, threshold)
    except Exception as e:
        logger.warning(f"Vector semantic cache failed: {e}")
        return get_semantic_cache(prompt, context, threshold)


def set_semantic_cache_vector(prompt: str, response: str, context: str = ""):
    \"\"\"Cache response with vector embedding.\"\"\"
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if not hasattr(set_semantic_cache_vector, 'model'):
            set_semantic_cache_vector.model = SentenceTransformer('all-MiniLM-L6-v2')

        model = set_semantic_cache_vector.model
        embedding = model.encode(prompt + " " + context)

        key = _generate_semantic_key(prompt, context)
        _semantic_cache[key] = {
            'response': response,
            'prompt': prompt,
            'context': context,
            'embedding': embedding,
            'timestamp': __import__('time').time(),
        }
    except Exception as e:
        logger.warning(f"Failed to store vector embedding: {e}, using simple cache")
        set_semantic_cache(prompt, response, context)
"""
