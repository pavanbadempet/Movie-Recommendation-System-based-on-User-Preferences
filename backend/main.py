"""
FastAPI backend for the APEX Movie Recommendation System.
Provides REST API endpoints for movie search and recommendations.
"""

# =====================================================================
# 1. STDLIB IMPORTS
# =====================================================================
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import logging
import os
from pathlib import Path
import time
from typing import Optional

# =====================================================================
# 2. FAST JSON — use orjson when available, fall back to stdlib json
# =====================================================================
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


# =====================================================================
# 3. THIRD-PARTY IMPORTS
# =====================================================================
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx
from prometheus_client import Counter as PromCounter
from prometheus_client import Histogram as PromHistogram
from prometheus_client import make_asgi_app
import sentry_sdk
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# =====================================================================
# 4. INTERNAL IMPORTS — backend package
# =====================================================================
from backend.api.admin_routes import create_admin_router
from backend.api.admin_tests import router as admin_router
from backend.api.artifact_routes import create_artifact_router
from backend.api.auth_routes import router as auth_router
from backend.api.billing_routes import router as billing_router
from backend.api.browse_routes import create_browse_router
from backend.api.catalog_routes import create_catalog_router
from backend.api.chat import generate_chat_response
from backend.api.evaluation_routes import create_evaluation_router
from backend.api.experiment_routes import create_experiment_router
from backend.api.recommendation_routes import (
    _recommendation_diagnostic_report,
    create_core_router,
    create_rec_engine_router,
    create_recommendation_router,
    create_search_movie_router,
)
from backend.api.recommendation_routes import (
    configure as _configure_rec_routes,
)
from backend.data.auth import (
    TenantContext,
    enforce_payload_context,
    resolve_admin_token,
    resolve_tenant_context,
)
from backend.data.catalogs import persist_catalog_upload, profile_catalog_csv
from backend.data.database import get_db
from backend.data.experiments import (
    assign_experiment,
    attach_experiment,
    summarize_experiment_metrics,
)
from backend.data.frontend_failover import configured_frontends, frontend_status_report
from backend.data.remote_recommender import (
    remote_get_json,
    remote_recommender_status,
    remote_recommender_url,
)
from backend.data.usage import record_usage, summarize_usage
from backend.events import (
    aggregate_behavior_features,
    append_event,
    build_user_behavior_profile,
    event_storage_status,
    get_events_path,
    summarize_recommendation_events,
)
from backend.events.recommendation_events import (
    _candidate_event_summary,
    _serving_lineage,
    record_recommendation_events,
    remote_payload_or_raise,
)
from backend.intelligence.active_inference_engine import get_active_inference_engine
from backend.learning.online_learner import OnlineLearner
from backend.metrics.benchmark_cache import (
    _recommendation_benchmark_cache,
    _semantic_benchmark_cache,
    compute_recommendation_benchmark_cached,
    compute_semantic_benchmark_cached,
    get_cached_recommendation_benchmark,
    get_cached_semantic_benchmark,
    start_background_recommendation_benchmark,
    start_background_semantic_benchmark,
    warming_recommendation_benchmark_report,
    warming_semantic_benchmark_report,
)
from backend.metrics.evaluation import evaluate_recommendation_quality
from backend.metrics.recommendation_benchmark import load_recommendation_benchmark
from backend.metrics.search_benchmark import evaluate_search_benchmark

_start_background_semantic_benchmark = start_background_semantic_benchmark
_start_background_recommendation_benchmark = start_background_recommendation_benchmark

# Optional middleware — imported conditionally in the middleware section below
from backend.models.ensemble_engine import get_apex_engine
from backend.pipeline import recommender_helpers
from backend.pipeline.ranker import load_ranker
from backend.pipeline.recommender import Recommender, get_recommender
from backend.pipeline.recommender_helpers import (
    event_logging_enabled as _event_logging_enabled,
)
from backend.pipeline.recommender_helpers import (
    refresh_artifact_files as _refresh_artifact_files,
)
from backend.pipeline.recommender_helpers import (
    reload_local_recommender as _reload_local_recommender,
)
from backend.pipeline.recommender_helpers import (
    safe_float as _safe_float,
)
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
from backend.serving.app_info import app_metadata, public_base_url
from backend.serving.app_startup import env_truthy as _env_truthy
from backend.serving.app_startup import shutdown as _app_shutdown
from backend.serving.app_startup import startup as _app_startup
from backend.serving.artifact_health import evaluate_artifact_health
from backend.serving.platform_readiness import (
    _benchmark_readiness_component,
    _combine_readiness_status,
    _platform_readiness_report,
    _readiness_component,
)
from backend.serving.platform_readiness import (
    platform_readiness_report as _platform_readiness_report_fn,
)
from backend.serving.slo import RequestSloTracker, build_slo_report, should_track_request

# =====================================================================
# 5. LOGGING & CONSTANTS
# =====================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TMDB_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
APP_VERSION = "2.0.0"
REVISION_FILE = Path(__file__).resolve().parent.parent / "REVISION"
FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Sentry error monitoring
_SENTRY_DSN = os.getenv("SENTRY_DSN")
if _SENTRY_DSN:
    try:
        sentry_sdk.init(dsn=_SENTRY_DSN, traces_sample_rate=1.0, profiles_sample_rate=1.0)
    except Exception as e:
        logger.warning("SENTRY_DSN is invalid. Error monitoring disabled: %s", e)
    else:
        logger.info("Sentry monitoring enabled.")
else:
    logger.warning("SENTRY_DSN not set. Error monitoring disabled.")

# =====================================================================
# 6. RUNTIME SINGLETONS — initialised in lifespan, read-only after startup
# =====================================================================
http_client: httpx.AsyncClient | None = None
_online_learner: OnlineLearner | None = None
_online_learning_coordinator = None
_tier_detector = None
_slo_tracker = RequestSloTracker()
_recommender: Recommender | None = None

# =====================================================================
# 7. MODULE-LEVEL HELPERS
# =====================================================================


async def _trigger_active_inference(movie_id: int, reward: float) -> None:
    """Dispatch Active Inference self-heal as a background task."""
    try:
        import torch

        engine = get_active_inference_engine()
        movie_emb = torch.randn(1, engine.emb_dim)
        engine.self_heal(movie_emb, reward)
    except Exception as exc:
        logger.warning("Active Inference self_heal failed for movie_id=%s: %s", movie_id, exc)


def get_rec() -> Recommender:
    """Get recommender instance, loading on first call."""
    global _recommender
    if _recommender is None:
        logger.info("Loading recommender on first request...")
        _recommender = get_recommender()
    return _recommender


def set_rec(r: Recommender) -> None:
    """Set global recommender instance."""
    global _recommender
    _recommender = r


def _background_recommender_warmup() -> None:
    """Warm the recommender after startup without blocking health probes."""
    recommender_helpers.background_recommender_warmup()


def _start_background_recommender_warmup() -> None:
    """Start one daemon warmup thread per process."""
    recommender_helpers.start_background_recommender_warmup()


# =====================================================================
# 8. LIFESPAN
# =====================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Delegate startup/shutdown to ``backend.serving.app_startup``."""
    global http_client, _online_learner, _online_learning_coordinator, _tier_detector

    def _set_recommender(r):
        global _recommender
        _recommender = r

    state = await _app_startup(
        recommender_get_fn=lambda: _recommender,
        recommender_set_fn=_set_recommender,
    )

    http_client = state["http_client"]
    _online_learner = state["online_learner"]
    _online_learning_coordinator = state["online_learning_coordinator"]
    _tier_detector = state["tier_detector"]

    if _env_truthy("NOVA_BACKGROUND_RECOMMENDER_WARMUP"):
        _start_background_recommender_warmup()

    yield

    await _app_shutdown(state)


# =====================================================================
# 9. FASTAPI APP
# =====================================================================
app = FastAPI(
    title="APEX Recommendation API",
    description=(
        "Production-grade AI recommendation engine with a 6-model ensemble "
        "(SASRec, KAN, LightGCN, Diffusion, Quantum-Fluid, Hyperbolic). "
        "Implements Doubly Robust IPS weight optimization, differential privacy, "
        "and adaptive 3-tier serving (GPU / ONNX CPU / FAISS lite).\n\n"
        "**Interactive docs:** `/docs` (Swagger UI) · `/redoc` (ReDoc)\n\n"
        "**Full API reference:** [docs/API_REFERENCE.md]"
        "(https://github.com/pavanbadempet/Movie-Recommendation-System/blob/main/docs/API_REFERENCE.md)"
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Health checks and platform readiness probes."},
        {
            "name": "recommendations",
            "description": "Core recommendation endpoints — neural ensemble, multi-modal, knowledge graph.",
        },
        {"name": "search", "description": "Semantic and hybrid search over the movie catalog."},
        {"name": "events", "description": "Behavior event ingestion for online learning and analytics."},
        {
            "name": "evaluation",
            "description": "Offline evaluation metrics — semantic benchmark, recommendation benchmark, SLO.",
        },
        {
            "name": "admin",
            "description": "Admin operations — weight reload, artifact refresh, platform status. Requires Bearer token.",
        },
        {"name": "auth", "description": "JWT authentication and user registration."},
        {"name": "experiments", "description": "A/B experiment management."},
        {"name": "catalog", "description": "Catalog browsing, filtering, and upload."},
    ],
    contact={
        "name": "APEX Project",
        "url": "https://github.com/pavanbadempet/Movie-Recommendation-System",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# =====================================================================
# 10. PROMETHEUS METRICS & MIDDLEWARE
# =====================================================================
REQUEST_COUNT = PromCounter("apex_http_requests_total", "Total HTTP requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = PromHistogram("apex_http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])


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

# =====================================================================
# 11. RATE LIMITING & CORS
# =====================================================================
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    r"https://(pavanbadempet\.github\.io)" r"|http://(localhost|127\.0\.0\.1):\d+",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enterprise rate limiting (token bucket via Redis)
try:
    from backend.middleware.rate_limiter import RedisRateLimiter as _RedisRateLimiter

    app.add_middleware(_RedisRateLimiter)
except ImportError:
    logger.warning("RedisRateLimiter could not be loaded. Running without SLA quotas.")

try:
    from backend.middleware.plan_enforcer import PlanEnforcerMiddleware as _PlanEnforcer

    app.add_middleware(_PlanEnforcer)
    logger.info("PlanEnforcerMiddleware loaded — daily limits active.")
except ImportError:
    logger.warning("PlanEnforcerMiddleware could not be loaded. Running without plan enforcement.")

# =====================================================================
# 12. SLO TRACKING MIDDLEWARE
# =====================================================================


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


# =====================================================================
# 13. STATIC FILES & ROOT ENDPOINT
# =====================================================================
if (FRONTEND_DIST_DIR / "index.html").exists():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
    logger.info("Mounted React frontend at /ui/ from %s", FRONTEND_DIST_DIR)


@app.get("/")
async def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/ui/")

    metadata = app_metadata()
    frontend_available = (FRONTEND_DIST_DIR / "index.html").exists()
    frontends = configured_frontends(frontend_available=frontend_available)
    return {
        "status": "online",
        "message": "Welcome to the APEX Recommendation API. Use /go for the healthiest UI or /docs for endpoints.",
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


# =====================================================================
# 14. ROUTE REGISTRATION — all router wiring in one place
# =====================================================================

# Wire the recommender helpers with lazy-load accessors
recommender_helpers.configure(get_rec, set_rec)


def _register_routes() -> None:
    """Register all routers and configure module-level route state.

    Extracted from module scope to keep app construction declarative and
    make the route wiring easy to audit in a single location.
    """

    # --- Auth & admin test routers (plain routers, no factory) ---
    app.include_router(admin_router, dependencies=[Depends(resolve_admin_token)])
    app.include_router(auth_router)
    app.include_router(billing_router)

    # --- Configure recommendation route module globals ---
    _configure_rec_routes(
        tmdb_key=TMDB_KEY,
        tmdb_base=TMDB_BASE,
        frontend_dist_dir=FRONTEND_DIST_DIR,
        http_client_getter=lambda: http_client,
        online_learner_getter=lambda: _online_learner,
        online_learning_coordinator_getter=lambda: _online_learning_coordinator,
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

    # --- Evaluation router ---
    app.include_router(
        create_evaluation_router(
            resolve_tenant_context=resolve_tenant_context,
            remote_payload_or_raise=lambda *a, **kw: remote_payload_or_raise(*a, **kw),
            record_usage=lambda *a, **kw: record_usage(*a, **kw),
            get_rec=lambda: get_rec(),
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

    # --- Admin router ---
    app.include_router(
        create_admin_router(
            resolve_admin_token=resolve_admin_token,
            get_apex_engine=get_apex_engine,
        )
    )

    # --- Artifact router ---
    app.include_router(
        create_artifact_router(
            resolve_tenant_context=resolve_tenant_context,
            resolve_admin_token=resolve_admin_token,
            evaluate_artifact_health=evaluate_artifact_health,
            record_usage=lambda *a, **kw: record_usage(*a, **kw),
            reload_local_recommender=_reload_local_recommender,
            refresh_artifact_files=_refresh_artifact_files,
            serving_lineage=_serving_lineage,
            current_recommender=lambda: _recommender,
        )
    )

    # --- Experiment router ---
    app.include_router(
        create_experiment_router(
            resolve_tenant_context=resolve_tenant_context,
            assign_experiment=assign_experiment,
            summarize_experiment_metrics=summarize_experiment_metrics,
            record_usage=lambda *a, **kw: record_usage(*a, **kw),
        )
    )

    # --- Catalog router ---
    app.include_router(
        create_catalog_router(
            resolve_tenant_context=resolve_tenant_context,
            profile_catalog_csv=profile_catalog_csv,
            persist_catalog_upload=persist_catalog_upload,
            record_usage=lambda *a, **kw: record_usage(*a, **kw),
        )
    )

    # --- Browse router ---
    app.include_router(
        create_browse_router(
            resolve_tenant_context=resolve_tenant_context,
            remote_payload_or_raise=lambda *a, **kw: remote_payload_or_raise(*a, **kw),
            get_rec=lambda: get_rec(),
            record_usage=lambda *a, **kw: record_usage(*a, **kw),
        )
    )

    # --- Core recommendation routers ---
    # Common callables shared across all recommendation factories
    def _rec():
        return get_rec()

    def _ru(*a, **kw):
        return record_usage(*a, **kw)

    def _rp(*a, **kw):
        return remote_payload_or_raise(*a, **kw)

    def _re(*a, **kw):
        return record_recommendation_events(*a, **kw)

    # Full recommendation router (SLO, frontend status, remote recommender)
    app.include_router(
        create_recommendation_router(
            get_rec=_rec,
            record_usage=_ru,
            remote_payload_or_raise=_rp,
            record_recommendation_events=_re,
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

    # Core router
    app.include_router(
        create_core_router(
            get_rec=_rec,
            record_usage=_ru,
            remote_payload_or_raise=_rp,
            record_recommendation_events=_re,
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

    # Search/movie router
    app.include_router(
        create_search_movie_router(
            get_rec=_rec,
            record_usage=_ru,
            remote_payload_or_raise=_rp,
            record_recommendation_events=_re,
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

    # Rec engine router
    app.include_router(
        create_rec_engine_router(
            get_rec=_rec,
            record_usage=_ru,
            remote_payload_or_raise=_rp,
            record_recommendation_events=_re,
            resolve_tenant_context=resolve_tenant_context,
            build_user_behavior_profile=build_user_behavior_profile,
            assign_experiment=assign_experiment,
            attach_experiment=attach_experiment,
            generate_chat_response=generate_chat_response,
            limiter=limiter,
            Movie=Movie,
            EnrichedMovie=EnrichedMovie,
            RecommendationResponse=RecommendationResponse,
            EnrichedRecommendationResponse=EnrichedRecommendationResponse,
        )
    )


# Execute route registration
_register_routes()


# =====================================================================
# 15. ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
