from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class RouterDeps:
    # Core handlers / getters
    get_rec: Callable
    record_usage: Callable
    resolve_tenant_context: Callable
    remote_payload_or_raise: Callable
    record_recommendation_events: Callable
    build_user_behavior_profile: Callable
    assign_experiment: Callable
    attach_experiment: Callable
    aggregate_behavior_features: Callable
    append_event: Callable
    summarize_recommendation_events: Callable
    evaluate_artifact_health: Callable
    load_ranker: Callable
    enforce_payload_context: Callable
    get_db: Callable
    generate_chat_response: Callable
    summarize_usage: Callable
    event_storage_status: Callable
    get_events_path: Callable
    limiter: Any

    # Extra recommendation / SLO deps
    build_slo_report: Callable | None = None
    frontend_status_report: Callable | None = None
    configured_frontends: Any = None
    remote_recommender_status: Callable | None = None

    # Evaluation deps
    evaluate_recommendation_quality: Callable | None = None
    evaluate_search_benchmark: Callable | None = None
    get_cached_semantic_benchmark: Callable | None = None
    compute_semantic_benchmark_cached: Callable | None = None
    start_background_semantic_benchmark: Callable | None = None
    warming_semantic_benchmark_report: Callable | None = None
    get_cached_recommendation_benchmark: Callable | None = None
    compute_recommendation_benchmark_cached: Callable | None = None
    start_background_recommendation_benchmark: Callable | None = None
    warming_recommendation_benchmark_report: Callable | None = None
    env_truthy: Callable | None = None

    # Admin / Auth / Artifact deps
    resolve_admin_token: Callable | None = None
    get_apex_engine: Callable | None = None
    reload_local_recommender: Callable | None = None
    refresh_artifact_files: Callable | None = None
    serving_lineage: Callable | None = None
    current_recommender: Callable | None = None

    # Experiment / Catalog / Browse deps
    summarize_experiment_metrics: Callable | None = None
    profile_catalog_csv: Callable | None = None
    persist_catalog_upload: Callable | None = None
