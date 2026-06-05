"""
FastAPI backend for the Movie Recommendation System.
Provides REST API endpoints for movie search and recommendations.
"""

from collections import Counter
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import logging
import os
from pathlib import Path
import time
from typing import Optional
import uuid

# ---------------------------------------------------------------------------
# Fast JSON — use orjson when available, fall back to stdlib json
# ---------------------------------------------------------------------------
try:
    import orjson as _json_lib

    _ORJSON_AVAILABLE = True
except ImportError:
    import json as _json_lib  # type: ignore[no-redef]

    _ORJSON_AVAILABLE = False


def _json_dumps(obj) -> str:
    """Serialize obj to a JSON string. Uses orjson when available."""
    if _ORJSON_AVAILABLE:
        try:
            return _json_lib.dumps(obj).decode()
        except Exception:
            import json as _stdlib_json

            return _stdlib_json.dumps(obj)
    return _json_lib.dumps(obj)


def _json_loads(s):
    """Deserialize a JSON string. Uses orjson when available."""
    if _ORJSON_AVAILABLE:
        try:
            return _json_lib.loads(s)
        except Exception:
            import json as _stdlib_json

            return _stdlib_json.loads(s)
    return _json_lib.loads(s)


from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx
import sentry_sdk
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend import recommender_helpers
from backend.app_info import app_metadata, public_base_url
from backend.artifact_health import evaluate_artifact_health
from backend.artifact_routes import create_artifact_router
from backend.auth import TenantContext, enforce_payload_context, resolve_admin_token, resolve_tenant_context
from backend.auth_routes import router as auth_router
from backend.benchmark_cache import (
    compute_recommendation_benchmark_cached,
    compute_semantic_benchmark_cached,
    get_cached_recommendation_benchmark,
    get_cached_semantic_benchmark,
    start_background_recommendation_benchmark,
    start_background_semantic_benchmark,
    warming_recommendation_benchmark_report,
    warming_semantic_benchmark_report,
)
from backend.browse_routes import create_browse_router
from backend.catalog_routes import create_catalog_router
from backend.catalogs import persist_catalog_upload, profile_catalog_csv
from backend.chat import generate_chat_response
from backend.database import get_db
from backend.ensemble_engine import get_apex_engine
from backend.evaluation import evaluate_recommendation_quality
from backend.evaluation_routes import create_evaluation_router
from backend.events import (
    aggregate_behavior_features,
    append_event,
    build_user_behavior_profile,
    event_storage_status,
    get_events_path,
    summarize_recommendation_events,
)
from backend.experiment_routes import create_experiment_router
from backend.experiments import assign_experiment, attach_experiment, summarize_experiment_metrics
from backend.frontend_failover import configured_frontends, frontend_status_report
from backend.online_learner import OnlineLearner
from backend.platform_readiness import (
    _combine_readiness_status,
    _platform_readiness_report,
)
from backend.platform_readiness import (
    platform_readiness_report as _platform_readiness_report_fn,  # noqa: F401 — available for external callers
)
from backend.ranker import load_ranker
from backend.recommender import Recommender, get_recommender
from backend.recommender_helpers import (
    event_logging_enabled as _event_logging_enabled,
)
from backend.recommender_helpers import (
    refresh_artifact_files as _refresh_artifact_files,
)
from backend.recommender_helpers import (
    reload_local_recommender as _reload_local_recommender,
)
from backend.recommender_helpers import (
    safe_float as _safe_float,
)
from backend.remote_recommender import remote_get_json, remote_recommender_status, remote_recommender_url
from backend.search_benchmark import evaluate_search_benchmark
from backend.slo import RequestSloTracker, build_slo_report, should_track_request
from backend.usage import record_usage, summarize_usage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )
    except Exception as e:
        logger.warning("SENTRY_DSN is invalid. Error monitoring disabled: %s", e)
    else:
        logger.info("Sentry monitoring enabled.")
else:
    logger.warning("SENTRY_DSN not set. Error monitoring disabled.")

# TMDB API config
TMDB_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
APP_VERSION = "2.0.0"
REVISION_FILE = Path(__file__).resolve().parent.parent / "REVISION"
FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Async HTTP client (initialized via lifespan)
http_client: httpx.AsyncClient | None = None
_online_learner: OnlineLearner | None = None
_tier_detector = None  # TierDetector singleton — set in lifespan
_slo_tracker = RequestSloTracker()


async def _trigger_active_inference(movie_id: int, reward: float) -> None:
    """Dispatch Active Inference self-heal as a background task (Requirement 5.7)."""
    try:
        import torch

        from backend.active_inference_engine import get_active_inference_engine

        engine = get_active_inference_engine()
        # Use a random proxy embedding if we can't retrieve the real one
        movie_emb = torch.randn(1, engine.emb_dim)
        engine.self_heal(movie_emb, reward)
    except Exception as exc:
        logger.warning("Active Inference self_heal failed for movie_id=%s: %s", movie_id, exc)


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async resources for app lifetime."""
    global http_client, _online_learner, _tier_detector

    # --- Wire recommender_helpers singleton accessors ---
    def _set_recommender(r):
        global _recommender
        _recommender = r

    recommender_helpers.configure(get_rec=lambda: _recommender, set_rec=_set_recommender)

    # --- Resolve serving tier before any model loading ---
    try:
        from backend.serving_tier import get_tier_detector

        _tier_detector = get_tier_detector()
        active_tier, tier_reason = _tier_detector.resolve()
        logger.info("Active serving tier: %s (%s)", active_tier, tier_reason)
    except Exception as exc:
        logger.warning("Tier detection failed: %s; defaulting to tier2", exc)
        active_tier = "tier2"

    http_client = httpx.AsyncClient(timeout=10.0)
    if _env_truthy("NOVA_BACKGROUND_RECOMMENDER_WARMUP"):
        _start_background_recommender_warmup()

    # --- Tier-specific engine startup ---
    if active_tier == "tier1":
        # Full ensemble + GPU + OnlineLearner
        try:
            gpu = _tier_detector._profile.gpu_available if _tier_detector and _tier_detector._profile else False
            device = "cuda" if gpu else "cpu"
            engine = get_apex_engine(device=device)
            _online_learner = OnlineLearner(lightgcn=engine.lightgcn)
            _online_learner.start()
            if _online_learner._thread is None or not _online_learner._thread.is_alive():
                logger.warning("OnlineLearner thread failed to start; attempting restart...")
                _online_learner.start()
                if _online_learner._thread is None or not _online_learner._thread.is_alive():
                    logger.critical("OnlineLearner thread could not be started. Online learning disabled.")
                    _online_learner = None
        except Exception as exc:
            logger.critical("Failed to initialise Tier1 engine: %s", exc)
            _online_learner = None

    elif active_tier == "tier2":
        # ONNX CPU engine
        try:
            from backend.onnx_engine import get_onnx_engine

            cpu_cores = _tier_detector._profile.cpu_cores if _tier_detector and _tier_detector._profile else 0
            onnx_engine = get_onnx_engine(cpu_cores=cpu_cores)
            if not onnx_engine.has_any_onnx_models():
                logger.warning("No ONNX models found; falling back to tier3 behavior")
                if _tier_detector:
                    _tier_detector._tier = "tier3"
                    _tier_detector._reason = "onnx_fallback"
        except Exception as exc:
            logger.warning("Failed to initialise Tier2 ONNX engine: %s", exc)

    # tier3: no engine pre-loading; recommender loads lazily on first request

    # Pre-load real-time feature index from event store for instant session sequences
    try:
        import asyncio

        from backend.realtime_feature_updater import preload_from_event_store

        asyncio.get_event_loop().run_in_executor(None, preload_from_event_store, 10000)
    except Exception as exc:
        logger.warning("Real-time index pre-load failed: %s", exc)

    yield

    # Shutdown
    if _online_learner is not None:
        _online_learner.stop()
    await http_client.aclose()


# Create FastAPI app
app = FastAPI(
    title="APEX Recommendation API",
    description=(
        "Production-grade AI recommendation engine with a 6-model ensemble "
        "(SASRec, KAN, LightGCN, Diffusion, Quantum-Fluid, Hyperbolic). "
        "Implements Doubly Robust IPS weight optimization, differential privacy, "
        "and adaptive 3-tier serving (GPU / ONNX CPU / FAISS lite).\n\n"
        "**Interactive docs:** `/docs` (Swagger UI) · `/redoc` (ReDoc)\n\n"
        "**Full API reference:** [docs/API_REFERENCE.md]"
        "(https://github.com/your-username/Movie-Recommendation-System/blob/main/docs/API_REFERENCE.md)"
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "health",
            "description": "Health checks and platform readiness probes.",
        },
        {
            "name": "recommendations",
            "description": "Core recommendation endpoints — neural ensemble, multi-modal, knowledge graph.",
        },
        {
            "name": "search",
            "description": "Semantic and hybrid search over the movie catalog.",
        },
        {
            "name": "events",
            "description": "Behavior event ingestion for online learning and analytics.",
        },
        {
            "name": "evaluation",
            "description": "Offline evaluation metrics — semantic benchmark, recommendation benchmark, SLO.",
        },
        {
            "name": "admin",
            "description": "Admin operations — weight reload, artifact refresh, platform status. Requires Bearer token.",
        },
        {
            "name": "auth",
            "description": "JWT authentication and user registration.",
        },
        {
            "name": "experiments",
            "description": "A/B experiment management.",
        },
        {
            "name": "catalog",
            "description": "Catalog browsing, filtering, and upload.",
        },
    ],
    contact={
        "name": "APEX Project",
        "url": "https://github.com/your-username/Movie-Recommendation-System",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

from backend.admin_tests import router as admin_router

app.include_router(admin_router)
app.include_router(auth_router)

# =====================================================================
# PROMETHEUS METRICS & MIDDLEWARE (Phase 11.1)
# =====================================================================
from prometheus_client import Counter as PromCounter
from prometheus_client import Histogram as PromHistogram
from prometheus_client import make_asgi_app

REQUEST_COUNT = PromCounter("nova_http_requests_total", "Total HTTP requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = PromHistogram("nova_http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    path = request.url.path
    if not path.startswith("/metrics"):
        REQUEST_COUNT.labels(method=request.method, endpoint=path, http_status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(duration)

    return response


app.mount("/metrics", make_asgi_app())


@app.get("/")
async def root():
    metadata = app_metadata()
    frontend_available = (FRONTEND_DIST_DIR / "index.html").exists()
    frontends = configured_frontends(frontend_available=frontend_available)
    return {
        "status": "online",
        "message": "Welcome to the Movie Recommendation API. Use /go for the healthiest UI or /docs for endpoints.",
        "version": metadata["version"],
        "app": metadata,
        "ui": "/ui/" if frontend_available else None,
        "launch_url": "/go",
        "frontend_status_url": "/v1/frontends/status",
        "frontends": [
            {
                "name": frontend.name,
                "label": frontend.label,
                "kind": frontend.kind,
                "url": frontend.url,
                "priority": frontend.priority,
                "local": frontend.local,
            }
            for frontend in frontends
        ],
    }


# Rate limiting (30 requests/minute per IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration for hosted frontends.
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [
        "https://a-movie-recommendation-system.streamlit.app",
        "https://movie-recommendation-system.streamlit.app",
        "http://localhost:8501",
        "http://localhost:5173",
        "http://localhost:3000",
    ]
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX",
    (
        r"https://([a-zA-Z0-9-]+\.)+(vercel\.app|pages\.dev|netlify\.app|github\.io)"
        r"|http://(localhost|127\.0\.0\.1):\d+"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# B2B SaaS Enterprise Rate Limiting (Token Bucket)
# ---------------------------------------------------------
try:
    from backend.middleware.rate_limiter import RedisRateLimiter

    app.add_middleware(RedisRateLimiter)
except ImportError:
    logger.warning("RedisRateLimiter could not be loaded. Running without SLA quotas.")

try:
    from backend.middleware.plan_enforcer import PlanEnforcerMiddleware

    app.add_middleware(PlanEnforcerMiddleware)
    logger.info("PlanEnforcerMiddleware loaded — daily limits active.")
except ImportError:
    logger.warning("PlanEnforcerMiddleware could not be loaded. Running without plan enforcement.")

# Billing routes (Stripe Checkout, Portal, Webhook, Usage)
from backend.billing_routes import router as billing_router

app.include_router(billing_router)

if (FRONTEND_DIST_DIR / "index.html").exists():
    app.mount(
        "/ui",
        StaticFiles(directory=FRONTEND_DIST_DIR, html=True),
        name="frontend",
    )
    logger.info("Mounted React frontend at /ui/ from %s", FRONTEND_DIST_DIR)


@app.middleware("http")
async def request_slo_middleware(request: Request, call_next):
    """Record process-local request latency/error SLO samples."""

    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", None) or request.url.path
        if should_track_request(path=request.url.path, route=route):
            _slo_tracker.record(
                method=request.method,
                path=request.url.path,
                route=route,
                status_code=status_code,
                latency_ms=(time.perf_counter() - started) * 1000,
            )


# Response models
from backend.response_models import (
    EnrichedMovie,
    EnrichedRecommendationResponse,
    EventRequest,
    EventResponse,
    HealthResponse,
    Movie,
    MovieTitle,
    PlatformContextResponse,
    RecommendationResponse,
    UsageResponse,
)

# Lazy-load recommender on first request
_recommender: Recommender | None = None


def get_rec() -> Recommender:
    """Get recommender instance, loading on first call."""
    global _recommender
    if _recommender is None:
        logger.info("Loading recommender on first request...")
        _recommender = get_recommender()
    return _recommender


def _background_recommender_warmup() -> None:
    """Warm the recommender after startup without blocking health probes (delegates to recommender_helpers)."""
    recommender_helpers.background_recommender_warmup()


def _start_background_recommender_warmup() -> None:
    """Start one daemon warmup thread per process (delegates to recommender_helpers)."""
    recommender_helpers.start_background_recommender_warmup()


# Moved to backend/recommendation_events.py (task 6.3)
# Moved to backend/platform_readiness.py (task 6.3)
from backend.platform_readiness import (
    _benchmark_readiness_component,
    _readiness_component,
)
from backend.recommendation_events import (
    _candidate_event_summary,
    _serving_lineage,
    record_recommendation_events,
    remote_payload_or_raise,
)

# Moved to backend/recommendation_routes.py (task 6.3)
from backend.recommendation_routes import (
    _recommendation_diagnostic_report,
)

# Moved to backend/platform_readiness.py (task 2.1)
# _combine_readiness_status and _platform_readiness_report are imported above.

# Moved to backend/recommendation_events.py (task 2.2)
# record_recommendation_events and remote_payload_or_raise are imported above.


app.include_router(
    create_evaluation_router(
        resolve_tenant_context=resolve_tenant_context,
        remote_payload_or_raise=remote_payload_or_raise,
        record_usage=record_usage,
        get_rec=get_rec,
        evaluate_recommendation_quality=evaluate_recommendation_quality,
        evaluate_search_benchmark=evaluate_search_benchmark,
        get_cached_semantic_benchmark=get_cached_semantic_benchmark,
        compute_semantic_benchmark_cached=compute_semantic_benchmark_cached,
        start_background_semantic_benchmark=start_background_semantic_benchmark,
        warming_semantic_benchmark_report=warming_semantic_benchmark_report,
        get_cached_recommendation_benchmark=get_cached_recommendation_benchmark,
        compute_recommendation_benchmark_cached=compute_recommendation_benchmark_cached,
        start_background_recommendation_benchmark=start_background_recommendation_benchmark,
        warming_recommendation_benchmark_report=warming_recommendation_benchmark_report,
        env_truthy=_env_truthy,
    )
)

from backend.admin_routes import create_admin_router

app.include_router(
    create_admin_router(
        resolve_admin_token=resolve_admin_token,
        get_apex_engine=get_apex_engine,
    )
)

app.include_router(
    create_artifact_router(
        resolve_tenant_context=resolve_tenant_context,
        resolve_admin_token=resolve_admin_token,
        evaluate_artifact_health=evaluate_artifact_health,
        record_usage=record_usage,
        reload_local_recommender=_reload_local_recommender,
        refresh_artifact_files=_refresh_artifact_files,
        serving_lineage=_serving_lineage,
        current_recommender=lambda: _recommender,
    )
)

app.include_router(
    create_experiment_router(
        resolve_tenant_context=resolve_tenant_context,
        assign_experiment=assign_experiment,
        summarize_experiment_metrics=summarize_experiment_metrics,
        record_usage=record_usage,
    )
)

app.include_router(
    create_catalog_router(
        resolve_tenant_context=resolve_tenant_context,
        profile_catalog_csv=profile_catalog_csv,
        persist_catalog_upload=persist_catalog_upload,
        record_usage=record_usage,
    )
)

app.include_router(
    create_browse_router(
        resolve_tenant_context=resolve_tenant_context,
        remote_payload_or_raise=remote_payload_or_raise,
        get_rec=get_rec,
        record_usage=record_usage,
    )
)


# ===== RECOMMENDATION, SEARCH, MOVIE, EVENTS & CHAT ROUTES =====
# Extracted to backend/recommendation_routes.py to keep this file under 1500 lines.

from backend.recommendation_routes import (
    configure as _configure_rec_routes,
)
from backend.recommendation_routes import (
    create_core_router,
    create_rec_engine_router,
    create_recommendation_router,
    create_search_movie_router,
)

_configure_rec_routes(
    tmdb_key=TMDB_KEY,
    tmdb_base=TMDB_BASE,
    frontend_dist_dir=FRONTEND_DIST_DIR,
    http_client_getter=lambda: http_client,
    online_learner_getter=lambda: _online_learner,
    recommender_getter=lambda: _recommender,
    slo_tracker_getter=lambda: _slo_tracker,
    tier_detector_getter=lambda: _tier_detector,
    limiter=limiter,
    app_metadata_fn=app_metadata,
    public_base_url_fn=public_base_url,
    platform_readiness_report_fn=_platform_readiness_report,
    recommendation_diagnostic_report_fn=_recommendation_diagnostic_report,
    trigger_active_inference_fn=_trigger_active_inference,
    serving_lineage_fn=_serving_lineage,
    candidate_event_summary_fn=_candidate_event_summary,
    event_logging_enabled_fn=_event_logging_enabled,
    safe_float_fn=_safe_float,
)

app.include_router(
    create_recommendation_router(
        get_rec=get_rec,
        record_usage=record_usage,
        remote_payload_or_raise=remote_payload_or_raise,
        record_recommendation_events=record_recommendation_events,
        resolve_tenant_context=resolve_tenant_context,
        build_user_behavior_profile=build_user_behavior_profile,
        assign_experiment=assign_experiment,
        attach_experiment=attach_experiment,
        aggregate_behavior_features=aggregate_behavior_features,
        append_event=append_event,
        summarize_recommendation_events=summarize_recommendation_events,
        evaluate_artifact_health=evaluate_artifact_health,
        build_slo_report=build_slo_report,
        frontend_status_report=frontend_status_report,
        configured_frontends=configured_frontends,
        remote_recommender_status=remote_recommender_status,
        load_ranker=load_ranker,
        enforce_payload_context=enforce_payload_context,
        get_db=get_db,
        generate_chat_response=generate_chat_response,
        summarize_usage=summarize_usage,
        event_storage_status=event_storage_status,
        get_events_path=get_events_path,
        limiter=limiter,
        Movie=Movie,
        EnrichedMovie=EnrichedMovie,
        HealthResponse=HealthResponse,
        RecommendationResponse=RecommendationResponse,
        EnrichedRecommendationResponse=EnrichedRecommendationResponse,
        EventRequest=EventRequest,
        EventResponse=EventResponse,
        PlatformContextResponse=PlatformContextResponse,
        UsageResponse=UsageResponse,
    )
)

app.include_router(
    create_core_router(
        get_rec=get_rec,
        record_usage=record_usage,
        remote_payload_or_raise=remote_payload_or_raise,
        record_recommendation_events=record_recommendation_events,
        resolve_tenant_context=resolve_tenant_context,
        build_user_behavior_profile=build_user_behavior_profile,
        assign_experiment=assign_experiment,
        attach_experiment=attach_experiment,
        aggregate_behavior_features=aggregate_behavior_features,
        append_event=append_event,
        summarize_recommendation_events=summarize_recommendation_events,
        evaluate_artifact_health=evaluate_artifact_health,
        load_ranker=load_ranker,
        enforce_payload_context=enforce_payload_context,
        get_db=get_db,
        generate_chat_response=generate_chat_response,
        summarize_usage=summarize_usage,
        event_storage_status=event_storage_status,
        get_events_path=get_events_path,
        limiter=limiter,
        Movie=Movie,
        EnrichedMovie=EnrichedMovie,
        HealthResponse=HealthResponse,
        RecommendationResponse=RecommendationResponse,
        EnrichedRecommendationResponse=EnrichedRecommendationResponse,
        EventRequest=EventRequest,
        EventResponse=EventResponse,
        PlatformContextResponse=PlatformContextResponse,
        UsageResponse=UsageResponse,
    )
)

app.include_router(
    create_search_movie_router(
        get_rec=get_rec,
        record_usage=record_usage,
        remote_payload_or_raise=remote_payload_or_raise,
        record_recommendation_events=record_recommendation_events,
        resolve_tenant_context=resolve_tenant_context,
        build_user_behavior_profile=build_user_behavior_profile,
        assign_experiment=assign_experiment,
        attach_experiment=attach_experiment,
        aggregate_behavior_features=aggregate_behavior_features,
        append_event=append_event,
        summarize_recommendation_events=summarize_recommendation_events,
        enforce_payload_context=enforce_payload_context,
        get_db=get_db,
        generate_chat_response=generate_chat_response,
        summarize_usage=summarize_usage,
        event_storage_status=event_storage_status,
        get_events_path=get_events_path,
        limiter=limiter,
        Movie=Movie,
        EnrichedMovie=EnrichedMovie,
        RecommendationResponse=RecommendationResponse,
        EnrichedRecommendationResponse=EnrichedRecommendationResponse,
        EventRequest=EventRequest,
        EventResponse=EventResponse,
    )
)

app.include_router(
    create_rec_engine_router(
        get_rec=get_rec,
        record_usage=record_usage,
        remote_payload_or_raise=remote_payload_or_raise,
        record_recommendation_events=record_recommendation_events,
        resolve_tenant_context=resolve_tenant_context,
        build_user_behavior_profile=build_user_behavior_profile,
        assign_experiment=assign_experiment,
        attach_experiment=attach_experiment,
        generate_chat_response=generate_chat_response,
        limiter=limiter,
        Movie=Movie,
        RecommendationResponse=RecommendationResponse,
        EnrichedRecommendationResponse=EnrichedRecommendationResponse,
        EnrichedMovie=EnrichedMovie,
    )
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


# ---------------------------------------------------------------------------
# NOTE: fetch_trailer, fetch_details, fetch_credits, enrich_movie are now
# defined in backend/recommendation_routes.py and re-exported above.
# ---------------------------------------------------------------------------
