"""
Benchmark caching state and helper functions.

Extracted from backend/main.py as part of the main.py decomposition (Requirement 10.1).
Provides a clean public API for semantic and recommendation benchmark caching with
TTL-based expiry, background computation threads, and double-checked locking.
"""

from datetime import UTC, datetime
import logging
import os
from threading import Lock, Thread
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level private state
# ---------------------------------------------------------------------------
_semantic_benchmark_cache: dict[int, tuple[float, dict]] = {}
_semantic_benchmark_threads: dict[int, Thread] = {}
_semantic_benchmark_cache_lock = Lock()
_semantic_benchmark_compute_lock = Lock()

_recommendation_benchmark_cache: dict[int, tuple[float, dict]] = {}
_recommendation_benchmark_threads: dict[int, Thread] = {}
_recommendation_benchmark_cache_lock = Lock()
_recommendation_benchmark_compute_lock = Lock()


# ---------------------------------------------------------------------------
# TTL helpers
# ---------------------------------------------------------------------------


def semantic_benchmark_ttl_seconds() -> int:
    """Return the TTL (in seconds) for cached semantic benchmark results."""
    return max(60, int(os.getenv("NOVA_SEMANTIC_BENCHMARK_CACHE_TTL_SECONDS", "3600")))


def recommendation_benchmark_ttl_seconds() -> int:
    """Return the TTL (in seconds) for cached recommendation benchmark results."""
    return max(60, int(os.getenv("NOVA_RECOMMENDATION_BENCHMARK_CACHE_TTL_SECONDS", "3600")))


# ---------------------------------------------------------------------------
# Warming report helpers
# ---------------------------------------------------------------------------


def warming_semantic_benchmark_report(k: int) -> dict:
    """Return a placeholder report indicating the semantic benchmark is still warming."""
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "warming",
        "reason": "Semantic benchmark is warming in the background. Retry shortly.",
        "case_count": 0,
        "evaluated_case_count": 0,
        "skipped_case_count": 0,
        "k": k,
        "metrics": {},
        "cases": [],
        "skipped": [],
    }


def warming_recommendation_benchmark_report(k: int) -> dict:
    """Return a placeholder report indicating the recommendation benchmark is still warming."""
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "warming",
        "reason": "Recommendation benchmark is warming in the background. Retry shortly.",
        "case_count": 0,
        "evaluated_case_count": 0,
        "skipped_case_count": 0,
        "k": k,
        "metrics": {},
        "cases": [],
        "skipped": [],
    }


# ---------------------------------------------------------------------------
# Semantic benchmark cache
# ---------------------------------------------------------------------------


def get_cached_semantic_benchmark(k: int) -> dict | None:
    """Return the cached semantic benchmark report for *k*, or None if absent/expired."""
    with _semantic_benchmark_cache_lock:
        cached = _semantic_benchmark_cache.get(k)
    if cached is None:
        return None
    cached_at, report = cached
    if time.time() - cached_at > semantic_benchmark_ttl_seconds():
        return None
    return report


def compute_semantic_benchmark_cached(rec, k: int) -> dict:
    """Compute (or return cached) semantic benchmark for *k*.

    Uses double-checked locking to avoid redundant computation when multiple
    threads race to populate the cache simultaneously.
    """
    from backend.metrics.semantic_benchmark import evaluate_semantic_benchmark

    cached = get_cached_semantic_benchmark(k)
    if cached is not None:
        return cached

    with _semantic_benchmark_compute_lock:
        cached = get_cached_semantic_benchmark(k)
        if cached is not None:
            return cached
        report = evaluate_semantic_benchmark(rec, k=k)
        with _semantic_benchmark_cache_lock:
            _semantic_benchmark_cache[k] = (time.time(), report)
        return report


def _background_semantic_benchmark(k: int) -> None:
    """Background thread target: compute and cache the semantic benchmark for *k*."""
    try:
        # Import lazily to avoid circular imports at module load time.
        from backend.pipeline.recommender import get_recommender

        rec = get_recommender()
        compute_semantic_benchmark_cached(rec, k=k)
    except Exception as exc:
        logger.exception("Background semantic benchmark failed: %s", exc)


def start_background_semantic_benchmark(k: int) -> None:
    """Start a background daemon thread to compute the semantic benchmark for *k*.

    If a thread for this *k* is already running, this is a no-op.
    """
    with _semantic_benchmark_cache_lock:
        thread = _semantic_benchmark_threads.get(k)
        if thread is not None and thread.is_alive():
            return
        thread = Thread(
            target=_background_semantic_benchmark,
            args=(k,),
            name=f"semantic-benchmark-{k}",
            daemon=True,
        )
        _semantic_benchmark_threads[k] = thread
        thread.start()


# ---------------------------------------------------------------------------
# Recommendation benchmark cache
# ---------------------------------------------------------------------------


def get_cached_recommendation_benchmark(k: int) -> dict | None:
    """Return the cached recommendation benchmark report for *k*, or None if absent/expired."""
    with _recommendation_benchmark_cache_lock:
        cached = _recommendation_benchmark_cache.get(k)
    if cached is None:
        return None
    cached_at, report = cached
    if time.time() - cached_at > recommendation_benchmark_ttl_seconds():
        return None
    return report


def compute_recommendation_benchmark_cached(rec, k: int) -> dict:
    """Compute (or return cached) recommendation benchmark for *k*.

    Uses double-checked locking to avoid redundant computation when multiple
    threads race to populate the cache simultaneously.
    """
    from backend.metrics.recommendation_benchmark import evaluate_recommendation_benchmark

    cached = get_cached_recommendation_benchmark(k)
    if cached is not None:
        return cached

    with _recommendation_benchmark_compute_lock:
        cached = get_cached_recommendation_benchmark(k)
        if cached is not None:
            return cached
        report = evaluate_recommendation_benchmark(rec, k=k)
        with _recommendation_benchmark_cache_lock:
            _recommendation_benchmark_cache[k] = (time.time(), report)
        return report


def _background_recommendation_benchmark(k: int) -> None:
    """Background thread target: compute and cache the recommendation benchmark for *k*."""
    try:
        from backend.pipeline.recommender import get_recommender

        rec = get_recommender()
        compute_recommendation_benchmark_cached(rec, k=k)
    except Exception as exc:
        logger.exception("Background recommendation benchmark failed: %s", exc)


def start_background_recommendation_benchmark(k: int) -> None:
    """Start a background daemon thread to compute the recommendation benchmark for *k*.

    If a thread for this *k* is already running, this is a no-op.
    """
    with _recommendation_benchmark_cache_lock:
        thread = _recommendation_benchmark_threads.get(k)
        if thread is not None and thread.is_alive():
            return
        thread = Thread(
            target=_background_recommendation_benchmark,
            args=(k,),
            name=f"recommendation-benchmark-{k}",
            daemon=True,
        )
        _recommendation_benchmark_threads[k] = thread
        thread.start()
