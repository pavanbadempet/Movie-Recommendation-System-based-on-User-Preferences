"""
Real-Time Feature Updater for APEX.

Provides millisecond-latency feature freshness by maintaining an in-memory
user event index that is updated synchronously on every event write,
rather than waiting for the 5-minute background rebuild.

This is the same pattern used by Netflix's Flink-based feature pipeline
and Spotify's real-time personalization system.

Architecture:
  - Every event write calls update_user_index(event) immediately
  - The ensemble engine's _get_session_sequence reads from this index
  - No file I/O on the hot path — pure in-memory dict operations

Integration:
  - Called from backend/main.py record_event handler
  - Replaces the 5-minute TTL background rebuild for active users
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory real-time user event index
# ---------------------------------------------------------------------------

_realtime_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
_realtime_index_lock = threading.Lock()
_MAX_EVENTS_PER_USER = 200  # Keep last 200 events per user in memory
_MAX_USERS_IN_INDEX = 50_000  # Cap total users to prevent unbounded memory


def update_user_index(event: dict[str, Any]) -> None:
    """
    Update the real-time user event index with a new event.
    Called synchronously on every event write — O(1) operation.

    Args:
        event: Normalized event dict with at least user_id, movie_id, event_ts, event_type
    """
    et = str(event.get("event_type", "")).lower()
    if et not in {"click", "rating", "view"}:
        return

    uid = event.get("user_id")
    mid = event.get("movie_id")
    if uid is None or mid is None:
        return

    try:
        mid = int(mid)
    except (TypeError, ValueError):
        return

    ts = str(event.get("event_ts") or "")
    uid_str = str(uid)

    with _realtime_index_lock:
        # Evict oldest user if at capacity
        if uid_str not in _realtime_index and len(_realtime_index) >= _MAX_USERS_IN_INDEX:
            # Remove the user with the oldest most-recent event
            try:
                oldest_uid = min(
                    _realtime_index.keys(),
                    key=lambda u: _realtime_index[u][-1][0] if _realtime_index[u] else "",
                )
                del _realtime_index[oldest_uid]
            except Exception:
                pass

        user_events = _realtime_index[uid_str]
        user_events.append((ts, mid))

        # Keep only the most recent N events, sorted by timestamp
        if len(user_events) > _MAX_EVENTS_PER_USER:
            user_events.sort(key=lambda x: x[0])
            _realtime_index[uid_str] = user_events[-_MAX_EVENTS_PER_USER:]


def get_user_session_sequence(
    user_id: str | int,
    max_len: int = 50,
) -> list[int] | None:
    """
    Get the user's real-time session sequence from the in-memory index.

    Returns None if the user has no real-time events (fall back to JSONL scan).
    Returns a list of movie_ids ordered chronologically (oldest first).
    """
    uid_str = str(user_id)
    with _realtime_index_lock:
        events = _realtime_index.get(uid_str)
        if not events:
            return None
        # Sort by timestamp and return last max_len movie IDs
        sorted_events = sorted(events, key=lambda x: x[0])
        return [mid for _, mid in sorted_events[-max_len:]]


def get_index_stats() -> dict[str, Any]:
    """Return stats about the real-time index for monitoring."""
    with _realtime_index_lock:
        total_users = len(_realtime_index)
        total_events = sum(len(v) for v in _realtime_index.values())
        return {
            "total_users": total_users,
            "total_events": total_events,
            "max_users_capacity": _MAX_USERS_IN_INDEX,
            "utilization_pct": round(total_users / _MAX_USERS_IN_INDEX * 100, 1),
        }


def preload_from_event_store(max_users: int = 10_000) -> int:
    """
    Pre-populate the real-time index from the JSONL event store at startup.
    Loads the most recent events for the most active users.

    Args:
        max_users: Maximum number of users to pre-load

    Returns:
        Number of users pre-loaded
    """
    try:
        from backend.events import iter_events
        logger.info("Pre-loading real-time index from event store (max %d users)...", max_users)

        # Collect all events grouped by user
        user_events_raw: dict[str, list[tuple[str, int]]] = defaultdict(list)
        INTERACTION_TYPES = {"click", "rating", "view"}

        for event in iter_events():
            et = str(event.get("event_type", "")).lower()
            if et not in INTERACTION_TYPES:
                continue
            uid = event.get("user_id")
            mid = event.get("movie_id")
            if uid is None or mid is None:
                continue
            try:
                mid = int(mid)
            except (TypeError, ValueError):
                continue
            ts = str(event.get("event_ts") or "")
            user_events_raw[str(uid)].append((ts, mid))

        # Sort users by activity (most active first) and load top max_users
        sorted_users = sorted(
            user_events_raw.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:max_users]

        with _realtime_index_lock:
            for uid, events in sorted_users:
                sorted_evts = sorted(events, key=lambda x: x[0])
                _realtime_index[uid] = sorted_evts[-_MAX_EVENTS_PER_USER:]

        loaded = len(sorted_users)
        logger.info("Pre-loaded real-time index: %d users", loaded)
        return loaded

    except Exception as exc:
        logger.warning("Failed to pre-load real-time index: %s", exc)
        return 0
