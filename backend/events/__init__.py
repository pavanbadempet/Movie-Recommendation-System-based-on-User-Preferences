"""
Event ingestion sub-package for APEX.

Modules:
    events.py                — JSONL/Postgres behavior event store
    recommendation_events.py — Recommendation lineage event logging

All public symbols from events.py are re-exported here so that
``from backend.events import append_event`` continues to work
alongside the new ``from backend.events.events import append_event`` form.
"""

from backend.events.events import (
    aggregate_behavior_features,
    append_event,
    build_user_behavior_profile,
    event_storage_status,
    get_events_path,
    get_event_store_mode,
    iter_events,
    normalize_event,
    summarize_recommendation_events,
    utc_now,
)

__all__ = [
    "aggregate_behavior_features",
    "append_event",
    "build_user_behavior_profile",
    "event_storage_status",
    "get_events_path",
    "get_event_store_mode",
    "iter_events",
    "normalize_event",
    "summarize_recommendation_events",
    "utc_now",
]
