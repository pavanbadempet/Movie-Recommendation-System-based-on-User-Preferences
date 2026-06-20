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


def get_semantic_cache(prompt: str, context: str = "", threshold: float = 0.88) -> str | None:
    """Get cached response for semantically similar query using vector similarity.

    Falls back to exact key matching if sentence-transformers is unavailable
    or on error.
    """
    # 1. Try vector-based similarity matching
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if not hasattr(get_semantic_cache, "_model"):
            # Load a very small, fast model (all-MiniLM-L6-v2) for caching
            get_semantic_cache._model = SentenceTransformer("all-MiniLM-L6-v2")

        model = get_semantic_cache._model
        query_text = _normalize_query(prompt + " " + context)
        query_embedding = model.encode(query_text, convert_to_numpy=True)
        # Normalize for cosine similarity via dot product
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        best_match = None
        best_similarity = 0.0

        for key, entry in _semantic_cache.items():
            if not isinstance(entry, dict):
                continue
            cached_emb_raw = entry.get("embedding")
            if cached_emb_raw is not None:
                cached_emb = np.array(cached_emb_raw, dtype=np.float32)
                # Ensure it's normalized
                norm = np.linalg.norm(cached_emb)
                if norm > 0:
                    cached_emb = cached_emb / norm
                similarity = float(np.dot(query_embedding, cached_emb))
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = entry.get("response")

        if best_match is not None:
            logger.info("Semantic cache hit with similarity: %.4f", best_similarity)
            return best_match

    except Exception as exc:
        logger.debug("Vector-based cache lookup failed: %s; falling back to exact cache", exc)

    # 2. Fallback to exact key matching
    key = _generate_semantic_key(prompt, context)
    entry = _semantic_cache.get(key)
    if entry and isinstance(entry, dict):
        return entry.get("response")
    return entry


def set_semantic_cache(prompt: str, response: str, context: str = "", ttl_seconds: int = 3600):
    """Cache response with both exact key and vector embedding representation."""
    key = _generate_semantic_key(prompt, context)

    # Evict oldest entries if cache is full
    if len(_semantic_cache) >= _semantic_cache_max:
        oldest_key = next(iter(_semantic_cache))
        del _semantic_cache[oldest_key]

    embedding_list = None
    try:
        from sentence_transformers import SentenceTransformer

        if not hasattr(set_semantic_cache, "_model"):
            set_semantic_cache._model = SentenceTransformer("all-MiniLM-L6-v2")

        model = set_semantic_cache._model
        query_text = _normalize_query(prompt + " " + context)
        embedding = model.encode(query_text, convert_to_numpy=True)
        embedding_list = embedding.tolist()
    except Exception as exc:
        logger.debug("Vector-based cache embedding generation failed: %s", exc)

    _semantic_cache[key] = {
        "response": response,
        "prompt": prompt,
        "context": context,
        "embedding": embedding_list,
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
