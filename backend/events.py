"""
User behavior event capture and feature aggregation.

This module is intentionally storage-light: local JSONL is enough for the
current product loop, while the function boundary can later be backed by Kafka,
S3, Delta, or a warehouse table without changing API callers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVENTS_PATH = REPO_ROOT / "data" / "events" / "movie_events.jsonl"
DEFAULT_TENANT_ID = os.getenv("NOVA_TENANT_ID", "demo-media-co")
DEFAULT_CATALOG_ID = os.getenv("NOVA_CATALOG_ID", "tmdb-movies")
DEFAULT_EVENT_TABLE = "nova_content_events"
EVENT_STORE_MODES = {"jsonl", "postgres", "dual"}
_POSTGRES_TABLE_READY: set[str] = set()

ALLOWED_EVENT_TYPES = {
    "view",
    "search",
    "click",
    "rating",
    "recommendation_impression",
}


def utc_now() -> str:
    """Return a compact UTC timestamp for persisted event records."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def get_events_path(event_path: str | Path | None = None) -> Path:
    """Resolve the event log path from an explicit argument, env var, or default."""
    if event_path is not None:
        return Path(event_path)

    configured_path = os.getenv("EVENT_LOG_PATH")
    if configured_path:
        return Path(configured_path)

    return DEFAULT_EVENTS_PATH


def get_event_store_mode() -> str:
    """Return the configured behavior-event store mode."""
    mode = os.getenv("NOVA_EVENT_STORE", "jsonl").strip().lower()
    if mode not in EVENT_STORE_MODES:
        logger.warning("Invalid NOVA_EVENT_STORE=%s; falling back to jsonl", mode)
        return "jsonl"
    return mode


def get_event_database_url() -> str | None:
    """Return the durable event database URL, when configured."""
    return os.getenv("NOVA_EVENT_DATABASE_URL") or os.getenv("DATABASE_URL")


def get_event_table_name() -> str:
    """Return a safe Postgres table name for product events."""
    table_name = os.getenv("NOVA_EVENT_TABLE", DEFAULT_EVENT_TABLE).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        logger.warning("Invalid NOVA_EVENT_TABLE=%s; using %s", table_name, DEFAULT_EVENT_TABLE)
        return DEFAULT_EVENT_TABLE
    return table_name


def event_storage_status(event_path: str | Path | None = None) -> dict[str, Any]:
    """Describe the currently configured event persistence layer."""
    mode = "jsonl" if event_path is not None else get_event_store_mode()
    database_url = get_event_database_url()
    durable = mode in {"postgres", "dual"} and bool(database_url)
    return {
        "event_store": mode,
        "durable": durable,
        "postgres_configured": bool(database_url),
        "event_table": get_event_table_name() if mode in {"postgres", "dual"} else None,
        "event_log_path": str(get_events_path(event_path)),
    }


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a raw behavior event before persistence."""
    event_type = str(event.get("event_type", "")).strip().lower()
    if event_type not in ALLOWED_EVENT_TYPES:
        allowed = ", ".join(sorted(ALLOWED_EVENT_TYPES))
        raise ValueError(f"event_type must be one of: {allowed}")

    normalized: dict[str, Any] = {
        "event_id": str(event.get("event_id") or uuid.uuid4()),
        "event_ts": str(event.get("event_ts") or utc_now()),
        "event_type": event_type,
        "tenant_id": str(event.get("tenant_id") or DEFAULT_TENANT_ID),
        "catalog_id": str(event.get("catalog_id") or DEFAULT_CATALOG_ID),
    }

    if event.get("movie_id") is not None:
        movie_id = _coerce_int(event["movie_id"], "movie_id")
        normalized["movie_id"] = movie_id
        normalized.setdefault("source_content_id", str(movie_id))

    if event.get("content_id") is not None:
        normalized["content_id"] = str(event["content_id"])

    if event.get("source_content_id") is not None:
        normalized["source_content_id"] = str(event["source_content_id"])

    if event.get("query_text") is not None:
        query_text = str(event["query_text"]).strip()
        if query_text:
            normalized["query_text"] = query_text

    if event.get("rating") is not None:
        rating = _coerce_float(event["rating"], "rating")
        if not 1 <= rating <= 5:
            raise ValueError("rating must be between 1 and 5")
        normalized["rating"] = rating

    for field_name in ("user_id", "session_id", "request_id", "source"):
        if event.get(field_name) is not None:
            normalized[field_name] = str(event[field_name])

    metadata = event.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        normalized["metadata"] = metadata

    return normalized


def _append_event_jsonl(normalized: dict[str, Any], event_path: str | Path | None = None) -> Path:
    """Append one normalized event to the JSONL event log."""
    path = get_events_path(event_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(normalized, sort_keys=True, ensure_ascii=True))
        fh.write("\n")

    return path


def _get_psycopg():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Postgres event storage requires psycopg. Install requirements.txt or run "
            "`pip install psycopg[binary]`."
        ) from exc
    return psycopg


def _connect_postgres(database_url: str):
    psycopg = _get_psycopg()
    return psycopg.connect(database_url)


def _ensure_postgres_events_table(conn: Any, table_name: str) -> None:
    cache_key = table_name
    if cache_key in _POSTGRES_TABLE_READY:
        return

    index_prefix = table_name[:40]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                event_id uuid PRIMARY KEY,
                event_ts timestamptz NOT NULL,
                tenant_id text NOT NULL,
                catalog_id text NOT NULL,
                event_type text NOT NULL,
                movie_id bigint,
                content_id text,
                source_content_id text,
                user_id text,
                session_id text,
                query_text text,
                rating double precision,
                request_id text,
                source text,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                raw_event jsonb NOT NULL
            )
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {index_prefix}_tenant_ts_idx
            ON {table_name} (tenant_id, catalog_id, event_ts DESC)
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {index_prefix}_type_ts_idx
            ON {table_name} (event_type, event_ts DESC)
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {index_prefix}_user_ts_idx
            ON {table_name} (user_id, event_ts DESC)
            """
        )
    _POSTGRES_TABLE_READY.add(cache_key)


def _append_event_postgres(normalized: dict[str, Any]) -> None:
    database_url = get_event_database_url()
    if not database_url:
        raise RuntimeError("NOVA_EVENT_DATABASE_URL or DATABASE_URL is required for Postgres event storage")

    table_name = get_event_table_name()
    metadata = json.dumps(normalized.get("metadata") or {}, ensure_ascii=True)
    raw_event = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
    with _connect_postgres(database_url) as conn:
        _ensure_postgres_events_table(conn, table_name)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {table_name} (
                    event_id, event_ts, tenant_id, catalog_id, event_type,
                    movie_id, content_id, source_content_id, user_id, session_id,
                    query_text, rating, request_id, source, metadata, raw_event
                )
                VALUES (
                    %s::uuid, %s::timestamptz, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s::jsonb, %s::jsonb
                )
                ON CONFLICT (event_id) DO NOTHING
                """,
                (
                    normalized["event_id"],
                    normalized["event_ts"],
                    normalized["tenant_id"],
                    normalized["catalog_id"],
                    normalized["event_type"],
                    normalized.get("movie_id"),
                    normalized.get("content_id"),
                    normalized.get("source_content_id"),
                    normalized.get("user_id"),
                    normalized.get("session_id"),
                    normalized.get("query_text"),
                    normalized.get("rating"),
                    normalized.get("request_id"),
                    normalized.get("source"),
                    metadata,
                    raw_event,
                ),
            )


def append_event(event: dict[str, Any], event_path: str | Path | None = None) -> dict[str, Any]:
    """Append one normalized behavior event to the configured event store."""
    normalized = normalize_event(event)
    mode = "jsonl" if event_path is not None else get_event_store_mode()
    wrote_jsonl = False
    wrote_postgres = False
    errors = []

    if mode in {"jsonl", "dual"}:
        _append_event_jsonl(normalized, event_path)
        wrote_jsonl = True

    if mode in {"postgres", "dual"}:
        try:
            _append_event_postgres(normalized)
            wrote_postgres = True
        except Exception as exc:
            if mode == "postgres":
                raise
            logger.warning("Postgres event write failed; JSONL fallback retained event: %s", exc)
            errors.append(str(exc))

    if mode == "postgres" and not wrote_postgres:
        raise RuntimeError("Postgres event write did not complete")
    if mode == "dual" and not wrote_jsonl and not wrote_postgres:
        _append_event_jsonl(normalized, event_path)
        wrote_jsonl = True

    result = dict(normalized)
    result["event_store"] = "postgres" if wrote_postgres and not wrote_jsonl else mode
    result["durable"] = wrote_postgres
    result["event_log_path"] = str(get_events_path(event_path))
    if errors:
        result["persistence_warnings"] = errors
    return result


def _iter_jsonl_events(event_path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield parsed JSONL events, skipping malformed lines with a warning."""
    path = get_events_path(event_path)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed event line %s in %s", line_number, path)
                continue

            if isinstance(parsed, dict):
                yield parsed


def _iter_postgres_events(limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield persisted events from the configured Postgres event table."""
    database_url = get_event_database_url()
    if not database_url:
        return

    table_name = get_event_table_name()
    query_limit = limit or int(os.getenv("NOVA_EVENT_READ_LIMIT", "100000"))
    with _connect_postgres(database_url) as conn:
        _ensure_postgres_events_table(conn, table_name)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT raw_event
                FROM {table_name}
                ORDER BY event_ts ASC
                LIMIT %s
                """,
                (query_limit,),
            )
            for row in cur:
                raw_event = row[0]
                if isinstance(raw_event, dict):
                    yield raw_event
                elif isinstance(raw_event, str):
                    try:
                        parsed = json.loads(raw_event)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        yield parsed


def iter_events(event_path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield events from the configured store, or a specific JSONL file."""
    if event_path is not None:
        yield from _iter_jsonl_events(event_path)
        return

    mode = get_event_store_mode()
    if mode in {"postgres", "dual"} and get_event_database_url():
        try:
            yielded = False
            for event in _iter_postgres_events():
                yielded = True
                yield event
            if yielded or mode == "postgres":
                return
        except Exception as exc:
            logger.warning("Postgres event read failed; falling back to JSONL: %s", exc)

    yield from _iter_jsonl_events()


def aggregate_behavior_features(
    event_path: str | Path | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Aggregate product behavior into lightweight recommender features."""
    path = get_events_path(event_path)
    storage = event_storage_status(event_path)
    event_type_counts: Counter[str] = Counter()
    search_counts: Counter[str] = Counter()
    movie_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "movie_id": None,
            "event_count": 0,
            "views": 0,
            "clicks": 0,
            "ratings": 0,
            "rating_sum": 0.0,
            "impressions": 0,
            "last_event_ts": None,
        }
    )

    total_events = 0
    for event in iter_events(event_path):
        event_type = str(event.get("event_type", "")).lower()
        if event_type not in ALLOWED_EVENT_TYPES:
            continue

        total_events += 1
        event_type_counts[event_type] += 1

        query_text = event.get("query_text")
        if event_type == "search" and query_text:
            search_counts[str(query_text).strip().lower()] += 1

        movie_id = event.get("source_content_id") or event.get("movie_id") or event.get("content_id")
        if movie_id is None:
            continue

        try:
            movie_key = str(_coerce_int(movie_id, "movie_id"))
        except ValueError:
            movie_key = str(movie_id)

        stats = movie_stats[movie_key]
        try:
            stats["movie_id"] = int(movie_key)
        except ValueError:
            stats["movie_id"] = movie_key
        stats["content_id"] = str(event.get("content_id") or movie_key)
        stats["tenant_id"] = str(event.get("tenant_id") or DEFAULT_TENANT_ID)
        stats["catalog_id"] = str(event.get("catalog_id") or DEFAULT_CATALOG_ID)
        stats["event_count"] += 1
        stats["last_event_ts"] = max(
            stats["last_event_ts"] or "",
            str(event.get("event_ts") or ""),
        ) or None

        if event_type == "view":
            stats["views"] += 1
        elif event_type == "click":
            stats["clicks"] += 1
        elif event_type == "recommendation_impression":
            stats["impressions"] += 1
        elif event_type == "rating" and event.get("rating") is not None:
            rating = _coerce_float(event["rating"], "rating")
            stats["ratings"] += 1
            stats["rating_sum"] += rating

    ranked_movies = sorted(
        movie_stats.values(),
        key=lambda item: (
            item["event_count"],
            item["clicks"],
            item["views"],
            item["rating_sum"],
        ),
        reverse=True,
    )[: max(limit, 0)]

    trending_movies: dict[str, dict[str, Any]] = {}
    for item in ranked_movies:
        item = dict(item)
        rating_sum = float(item.pop("rating_sum", 0.0))
        item["avg_rating"] = (
            round(rating_sum / item["ratings"], 3)
            if item["ratings"]
            else None
        )
        trending_movies[str(item["movie_id"])] = item

    top_searches = [
        {"query_text": query_text, "count": count}
        for query_text, count in search_counts.most_common(max(limit, 0))
    ]

    return {
        "generated_at": utc_now(),
        "event_store": storage["event_store"],
        "durable": storage["durable"],
        "event_table": storage["event_table"],
        "event_log_path": str(path),
        "total_events": total_events,
        "event_type_counts": dict(event_type_counts),
        "trending_movies": trending_movies,
        "top_searches": top_searches,
    }


def build_user_behavior_profile(
    user_id: str,
    event_path: str | Path | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Build a lightweight implicit-feedback profile from local event logs."""
    user_id = str(user_id)
    weighted_events = []
    negative_movie_ids = []
    query_counts: Counter[str] = Counter()
    event_weights = {
        "rating": 1.4,
        "click": 1.2,
        "view": 1.0,
        "recommendation_impression": 0.4,
    }

    for event in iter_events(event_path):
        if str(event.get("user_id")) != user_id:
            continue

        event_type = str(event.get("event_type", "")).lower()
        if event_type == "search" and event.get("query_text"):
            query_counts[str(event["query_text"]).strip().lower()] += 1
            continue

        movie_id = event.get("movie_id") or event.get("source_content_id")
        if movie_id is None:
            continue

        try:
            movie_id = int(movie_id)
        except (TypeError, ValueError):
            continue

        rating = event.get("rating")
        weight = event_weights.get(event_type, 0.5)
        is_negative = False
        if rating is not None:
            try:
                rating_value = max(0.0, min(float(rating), 5.0))
                weight += rating_value / 5.0
                is_negative = rating_value <= 2.0
            except (TypeError, ValueError):
                pass

        if is_negative:
            negative_movie_ids.append(movie_id)

        weighted_events.append(
            {
                "movie_id": movie_id,
                "event_type": event_type,
                "event_ts": str(event.get("event_ts") or ""),
                "weight": round(weight, 4),
                "rating": rating,
                "negative": is_negative,
            }
        )

    weighted_events.sort(key=lambda item: (item["event_ts"], item["weight"]), reverse=True)

    seen_ids = []
    seen_set = set()
    for item in weighted_events:
        if item.get("negative"):
            continue
        if item["movie_id"] in seen_set:
            continue
        seen_set.add(item["movie_id"])
        seen_ids.append(item["movie_id"])
        if len(seen_ids) >= limit:
            break

    return {
        "user_id": user_id,
        "generated_at": utc_now(),
        "seed_movie_ids": seen_ids,
        "negative_movie_ids": sorted(set(negative_movie_ids)),
        "recent_events": weighted_events[:limit],
        "top_searches": [
            {"query_text": query_text, "count": count}
            for query_text, count in query_counts.most_common(limit)
        ],
    }
