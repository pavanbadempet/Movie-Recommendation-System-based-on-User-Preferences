"""
Application startup and shutdown orchestration.

Separates the lifespan logic from main.py to keep the entry-point focused on
app creation and route wiring.  All tier-specific engine initialisation,
online-learning coordinator startup, and background warmup live here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GITHUB_REPO = "https://github.com/pavanbadempet/Movie-Recommendation-System"


def env_truthy(name: str) -> bool:
    """Return True when the named env-var is set to a truthy value."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Tier-specific startup
# ---------------------------------------------------------------------------


def _start_tier1_engine(tier_detector) -> tuple:
    """Initialise the full GPU ensemble + unified online-learning coordinator.

    Returns ``(coordinator, lightgcn_learner)`` on success, ``(None, None)``
    on failure.
    """
    try:
        from backend.learning.online_learning_coordinator import OnlineLearningCoordinator
        from backend.models.ensemble_engine import get_apex_engine

        gpu = getattr(getattr(tier_detector, "_profile", None), "gpu_available", False)
        vram_ok = getattr(getattr(tier_detector, "_profile", None), "gpu_vram_gb", 0.0) >= 8.0
        device = "cuda" if (gpu and vram_ok) else "cpu"
        engine = get_apex_engine(device=device)

        coordinator = OnlineLearningCoordinator(engine=engine)
        coordinator.start()

        # Verify all daemon threads started correctly.
        coord_status = coordinator.status()
        for name, info in coord_status["learners"].items():
            if not info["thread_alive"]:
                logger.critical(
                    "OnlineLearningCoordinator: %s thread failed to start — attempting restart.",
                    name,
                )

        alive = all(v["thread_alive"] for v in coord_status["learners"].values())
        if not alive:
            # One retry
            coordinator.stop()
            coordinator = OnlineLearningCoordinator(engine=engine)
            coordinator.start()
            coord_status = coordinator.status()
            if not all(v["thread_alive"] for v in coord_status["learners"].values()):
                logger.critical(
                    "OnlineLearningCoordinator threads could not be started after retry. Online learning disabled."
                )
                return None, None

        lightgcn_learner = coordinator.lightgcn_learner
        return coordinator, lightgcn_learner

    except Exception as exc:
        logger.critical("Failed to initialise Tier1 engine: %s", exc)
        return None, None


def _start_tier2_engine(tier_detector) -> None:
    """Initialise the ONNX CPU engine for Tier 2.

    Explicit Tier 2 configuration fails closed when required artifacts are
    missing. Hardware auto-detection may fall back to Tier 3.
    """
    explicit_tier2 = os.getenv("NOVA_SERVING_TIER", "").strip().lower() == "tier2"
    try:
        from backend.serving.onnx_engine import get_onnx_engine

        cpu_cores = getattr(getattr(tier_detector, "_profile", None), "cpu_cores", 0)
        onnx_engine = get_onnx_engine(cpu_cores=cpu_cores)
        missing = onnx_engine.missing_required_models()
        if missing:
            message = f"Tier 2 requires ONNX sessions for: {', '.join(missing)}"
            if explicit_tier2:
                raise RuntimeError(message)
            logger.warning("%s; falling back to tier3 behaviour.", message)
            if tier_detector is not None:
                tier_detector._tier = "tier3"
                tier_detector._reason = "onnx_fallback"
    except Exception as exc:
        if explicit_tier2:
            raise RuntimeError(f"Explicit Tier 2 startup failed: {exc}") from exc
        logger.warning("Failed to initialise Tier2 ONNX engine; falling back to Tier 3: %s", exc)
        if tier_detector is not None:
            tier_detector._tier = "tier3"
            tier_detector._reason = "onnx_initialization_error"


def _preload_realtime_index() -> None:
    """Kick off the real-time feature-index pre-load in an executor thread."""
    try:
        from backend.serving.realtime_feature_updater import preload_from_event_store

        asyncio.get_event_loop().run_in_executor(None, preload_from_event_store, 10_000)
    except Exception as exc:
        logger.warning("Real-time index pre-load failed: %s", exc)


# ---------------------------------------------------------------------------
# Public startup / shutdown entry points
# ---------------------------------------------------------------------------


async def startup(
    recommender_get_fn,
    recommender_set_fn,
) -> dict:
    """Run all startup tasks and return an app-state dict.

    The caller (``lifespan`` in ``main.py``) stores the returned dict and
    passes the individual values to ``shutdown`` when the process exits.

    Returns a dict with keys:
        ``http_client``, ``online_learner``, ``online_learning_coordinator``,
        ``tier_detector``.
    """
    from backend.pipeline import recommender_helpers

    # Wire the recommender singleton accessors used by recommender_helpers.
    recommender_helpers.configure(get_rec=recommender_get_fn, set_rec=recommender_set_fn)

    # Resolve serving tier first — everything else depends on it.
    tier_detector = None
    active_tier = "tier2"
    try:
        from backend.serving.serving_tier import get_tier_detector

        tier_detector = get_tier_detector()
        active_tier, tier_reason = tier_detector.resolve()
        logger.info("Active serving tier: %s (%s)", active_tier, tier_reason)
    except Exception as exc:
        logger.warning("Tier detection failed: %s; defaulting to tier2.", exc)

    http_client = httpx.AsyncClient(timeout=10.0)

    state = {
        "http_client": http_client,
        "online_learner": None,
        "online_learning_coordinator": None,
        "tier_detector": tier_detector,
    }

    def init_tier1():
        coord, learner = _start_tier1_engine(tier_detector)
        state["online_learning_coordinator"] = coord
        state["online_learner"] = learner

    def init_tier2():
        _start_tier2_engine(tier_detector)

    if active_tier == "tier1":
        asyncio.get_event_loop().run_in_executor(None, init_tier1)
    elif active_tier == "tier2":
        asyncio.get_event_loop().run_in_executor(None, init_tier2)
    # tier3: no engine pre-loading; recommender loads lazily on first request.

    _preload_realtime_index()

    # Pre-warm recommender singleton in background so first user request responds in < 5ms
    def _prewarm_recommender():
        try:
            rec = recommender_get_fn()
            if rec is not None:
                rec.search_by_title("Action", top_n=5)
                logger.info("Recommender model pre-warmed successfully for zero-latency HF Spaces serving.")
        except Exception as err:
            logger.warning("Recommender pre-warming skipped: %s", err)

    asyncio.get_event_loop().run_in_executor(None, _prewarm_recommender)

    return state


async def shutdown(state: dict) -> None:
    """Gracefully stop background workers and release async resources."""
    coordinator = state.get("online_learning_coordinator")
    learner = state.get("online_learner")
    http_client: httpx.AsyncClient | None = state.get("http_client")

    if coordinator is not None:
        coordinator.stop()
    elif learner is not None:
        # Fallback: stop standalone learner if coordinator was not used.
        learner.stop()

    if http_client is not None:
        await http_client.aclose()
