"""
RouterDeps — shared dependency container for FastAPI router factories.

Replaces the pattern of passing ~40 keyword arguments to each
create_*_router() factory call in main.py. All router factories accept
a single `RouterDeps` instance instead.

Usage in main.py:
    deps = RouterDeps(
        get_rec=get_rec,
        record_usage=record_usage,
        ...
    )
    app.include_router(create_recommendation_router(deps))
    app.include_router(create_search_movie_router(deps))

This is a purely mechanical refactor — no behaviour changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class RouterDeps:
    """
    Shared dependencies injected into all FastAPI router factories.

    All fields are callables or configuration objects — no state is stored
    here at request time. The dataclass is constructed once at startup in
    main.py and passed to every create_*_router() factory.
    """

    # Core recommender
    get_rec: Callable
    record_usage: Callable

    # Auth & tenant resolution
    resolve_tenant_context: Callable
    enforce_payload_context: Callable
    get_db: Callable

    # Remote fallback recommender
    remote_payload_or_raise: Callable

    # Event pipeline
    record_recommendation_events: Callable
    build_user_behavior_profile: Callable
    aggregate_behavior_features: Callable
    append_event: Callable
    summarize_recommendation_events: Callable
    event_storage_status: Callable
    get_events_path: Callable

    # A/B experiments
    assign_experiment: Callable
    attach_experiment: Callable

    # Artifact health
    evaluate_artifact_health: Callable

    # ML pipeline
    load_ranker: Callable

    # LLM chat
    generate_chat_response: Callable

    # Usage
    summarize_usage: Callable

    # Rate limiter
    limiter: Any

    # Response model classes (passed as types, not instances)
    Movie: type = field(default=object)  # type: ignore[type-arg]
    EnrichedMovie: type = field(default=object)  # type: ignore[type-arg]
    HealthResponse: type = field(default=object)  # type: ignore[type-arg]
    RecommendationResponse: type = field(default=object)  # type: ignore[type-arg]
    EnrichedRecommendationResponse: type = field(default=object)  # type: ignore[type-arg]
    EventRequest: type = field(default=object)  # type: ignore[type-arg]
    EventResponse: type = field(default=object)  # type: ignore[type-arg]
    PlatformContextResponse: type = field(default=object)  # type: ignore[type-arg]
    UsageResponse: type = field(default=object)  # type: ignore[type-arg]

    # Optional — not all routers use these
    build_slo_report: Callable | None = None
    frontend_status_report: Callable | None = None
    configured_frontends: Callable | None = None
    remote_recommender_status: Callable | None = None
