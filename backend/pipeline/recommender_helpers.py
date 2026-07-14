"""
Recommender helper functions extracted from backend/main.py.

Provides utilities for reloading the local recommender, refreshing artifact files,
background warmup, environment management, and small value-conversion helpers.

Call ``configure(get_rec, set_rec)`` once from ``main.py``'s lifespan to wire up
the module-level singleton accessors before any helper is invoked.

Requirement 11.1
"""

from collections.abc import Callable
from contextlib import contextmanager
import gc
import logging
import os
from threading import Lock, Thread

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level singleton accessors (wired up by configure() at startup)
# ---------------------------------------------------------------------------
def _get_recommender_default():
    return None


def _set_recommender_default(r):
    pass


_get_recommender: Callable = _get_recommender_default
_set_recommender: Callable = _set_recommender_default

# Warmup thread state
_warmup_thread: Thread | None = None
_warmup_thread_lock = Lock()


def configure(get_rec: Callable, set_rec: Callable) -> None:
    """Called once from main.py lifespan to wire up the _recommender singleton accessors."""
    global _get_recommender, _set_recommender
    _get_recommender = get_rec
    _set_recommender = set_rec


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


@contextmanager
def temporary_env(overrides: dict[str, str | None]):
    """Temporarily override environment variables for one operation."""
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def artifact_refresh_env(force_download: bool) -> dict[str, str]:
    """Return model-loader env overrides for artifact reload operations."""
    overrides: dict[str, str] = {"NOVA_REFRESH_PIPELINE_MANIFEST": "1"}
    if force_download:
        overrides["FORCE_MODEL_REFRESH"] = "1"
    return overrides


# ---------------------------------------------------------------------------
# Recommender reload / artifact refresh
# ---------------------------------------------------------------------------


def reload_local_recommender(force_download: bool):
    """Load a fresh recommender and atomically publish it to both singletons."""
    from backend.pipeline import recommender as recommender_module

    previous_main_recommender = _get_recommender()
    previous_module_recommender = recommender_module._recommender
    try:
        with temporary_env(artifact_refresh_env(force_download)):
            fresh_recommender = recommender_module.Recommender().load()
    except Exception:
        _set_recommender(previous_main_recommender)
        recommender_module._recommender = previous_module_recommender
        raise

    _set_recommender(fresh_recommender)
    recommender_module._recommender = fresh_recommender

    try:
        from backend.api.fast_cache import clear_all_caches

        clear_all_caches()
        from backend.api.browse_routes import clear_vectors_cache

        clear_vectors_cache()
    except Exception:
        pass

    gc.collect()
    return fresh_recommender


def refresh_artifact_files(force_download: bool) -> dict[str, bool]:
    """Refresh serving artifact files without rebuilding the in-memory recommender."""
    from backend.models.model_loader import default_artifacts_for_serving_profile, ensure_model_files
    from backend.pipeline import recommender as recommender_module

    with temporary_env(artifact_refresh_env(force_download)):
        return ensure_model_files(
            recommender_module.MODELS_DIR,
            selected_files=default_artifacts_for_serving_profile(),
        )


# ---------------------------------------------------------------------------
# Background warmup
# ---------------------------------------------------------------------------


def background_recommender_warmup() -> None:
    """Warm the recommender after startup without blocking health probes."""
    from backend.metrics.benchmark_cache import (
        compute_recommendation_benchmark_cached,
        compute_semantic_benchmark_cached,
    )

    def _env_truthy(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    try:
        logger.info("Starting background recommender warmup...")
        rec = _get_recommender()
        if rec is None:
            # Trigger lazy load via the configured getter (which may load on first call)
            # If the getter doesn't auto-load, we skip warmup gracefully.
            logger.warning("Background warmup: recommender not yet loaded; skipping benchmark precompute.")
            return
        if _env_truthy("NOVA_PRECOMPUTE_SEMANTIC_BENCHMARK"):
            k = int(os.getenv("NOVA_SEMANTIC_BENCHMARK_K", "10"))
            compute_semantic_benchmark_cached(rec, k=k)
        if _env_truthy("NOVA_PRECOMPUTE_RECOMMENDATION_BENCHMARK"):
            k = int(os.getenv("NOVA_RECOMMENDATION_BENCHMARK_K", "10"))
            compute_recommendation_benchmark_cached(rec, k=k)
        logger.info("Background recommender warmup completed.")
    except Exception as exc:
        logger.exception("Background recommender warmup failed: %s", exc)


def start_background_recommender_warmup() -> None:
    """Start one daemon warmup thread per process."""
    global _warmup_thread
    with _warmup_thread_lock:
        if _warmup_thread is not None and _warmup_thread.is_alive():
            return
        _warmup_thread = Thread(
            target=background_recommender_warmup,
            name="recommender-warmup",
            daemon=True,
        )
        _warmup_thread.start()


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------


def event_logging_enabled() -> bool:
    """Return whether recommendation serving should emit analytics events."""
    value = os.getenv("NOVA_RECOMMENDATION_EVENT_LOGGING", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def safe_float(value: object) -> float | None:
    """Safely convert *value* to a finite float, returning None on failure."""
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return round(number, 6)
