"""
Recommendation event helpers — extracted from backend/main.py.

Provides serving lineage, candidate summaries, diagnostic reports,
and the record_recommendation_events function.
"""

from __future__ import annotations

from collections import Counter
import logging
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def serving_lineage(rec) -> dict:
    """Return compact model/artifact lineage for recommendation events."""
    if rec is None:
        return {"serving_path": "remote_gateway"}

    artifact_status = dict(getattr(rec, "_artifact_status", {}) or {})
    manifest = dict(getattr(rec, "_artifact_manifest", {}) or {})
    ranker = getattr(rec, "_learned_ranker", None)
    ranker_metadata = dict(getattr(ranker, "metadata", {}) or {}) if ranker is not None else {}
    return {
        "serving_path": "local",
        "manifest_run_id": artifact_status.get("manifest_run_id") or manifest.get("run_id"),
        "manifest_run_date": artifact_status.get("manifest_run_date") or manifest.get("run_date"),
        "vector_artifacts_ready": artifact_status.get("vector_artifacts_ready"),
        "movie_count": artifact_status.get("movie_count"),
        "vector_count": artifact_status.get("vector_count"),
        "faiss_index_count": artifact_status.get("faiss_index_count"),
        "ranker_available": ranker is not None,
        "ranker_training_mode": ranker_metadata.get("training_mode"),
        "ranker_promoted_at": ranker_metadata.get("promoted_at"),
    }


def candidate_event_summary(candidate: dict, rank: int, safe_float_fn) -> dict:
    """Return the event-safe summary for one ranked recommendation."""
    return {
        "rank": rank,
        "movie_id": candidate.get("id"),
        "title": candidate.get("title"),
        "retrieval_stage": candidate.get("retrieval_stage"),
        "similarity_score": safe_float_fn(candidate.get("similarity_score")),
        "ranker_score": safe_float_fn(candidate.get("ranker_score")),
        "retrieval_signals": candidate.get("retrieval_signals") or {},
    }


def candidate_diagnostic_summary(candidate: dict, rank: int, safe_float_fn) -> dict:
    """Return recommendation evidence safe to expose to product/debug clients."""
    return {
        "rank": rank,
        "id": candidate.get("id"),
        "title": candidate.get("title"),
        "score": safe_float_fn(candidate.get("similarity_score")),
        "ranker_score": safe_float_fn(candidate.get("ranker_score")),
        "retrieval_stage": candidate.get("retrieval_stage") or "unknown",
        "explanation": candidate.get("explanation") or [],
        "explanation_text": candidate.get("explanation_text"),
        "retrieval_signals": candidate.get("retrieval_signals") or {},
    }


def recommendation_diagnostic_report(
    *,
    context,
    rec,
    query_movie: dict,
    recommendations: list[dict],
    k: int,
    app_metadata_fn,
    safe_float_fn,
    find_benchmark_case_fn,
    load_benchmark_fn,
    evaluate_case_fn,
) -> dict:
    """Build a compact per-seed report for ranking explainability and support."""
    diagnostic_items = [
        candidate_diagnostic_summary(candidate, rank, safe_float_fn)
        for rank, candidate in enumerate(recommendations[:k], start=1)
    ]
    stage_counts = Counter(item["retrieval_stage"] for item in diagnostic_items)
    explained_count = sum(1 for item in diagnostic_items if item.get("explanation") or item.get("explanation_text"))
    scores = [item["score"] for item in diagnostic_items if item.get("score") is not None]

    benchmark_case = find_benchmark_case_fn(query_movie, cases=load_benchmark_fn())
    benchmark_summary = None
    if benchmark_case is not None:
        benchmark_summary = evaluate_case_fn(recommendations, benchmark_case, k=k, seed_movie=query_movie)
        benchmark_summary.pop("_aggregate", None)

    return {
        "status": "ok",
        "app": app_metadata_fn(),
        "tenant_id": context.tenant_id,
        "catalog_id": context.catalog_id,
        "query_movie": {
            "id": query_movie.get("id"),
            "title": query_movie.get("title"),
            "genres": query_movie.get("genres"),
            "release_date": query_movie.get("release_date"),
            "vote_average": query_movie.get("vote_average"),
            "vote_count": query_movie.get("vote_count"),
        },
        "lineage": serving_lineage(rec),
        "diagnostics": {
            "requested_k": k,
            "result_count": len(diagnostic_items),
            "stage_distribution": dict(sorted(stage_counts.items())),
            "explanation_coverage": round(explained_count / max(len(diagnostic_items), 1), 4),
            "average_similarity_score": (round(sum(scores) / len(scores), 6) if scores else None),
            "benchmark_case_available": benchmark_summary is not None,
            "benchmark_case_passed": (benchmark_summary.get("passed") if benchmark_summary is not None else None),
        },
        "benchmark_case": benchmark_summary,
        "recommendations": diagnostic_items,
    }


def record_recommendation_events(
    *,
    endpoint: str,
    context,
    query_movie: dict,
    recommendations: list[dict],
    rec,
    request_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    event_logging_enabled_fn,
    append_event_fn,
    safe_float_fn,
) -> str:
    """Persist request and impression events for offline analysis and training labels."""
    resolved_request_id = request_id or str(uuid.uuid4())
    if not event_logging_enabled_fn():
        return resolved_request_id

    try:
        lineage = serving_lineage(rec)
        ranked_candidates = [
            candidate_event_summary(candidate, rank, safe_float_fn)
            for rank, candidate in enumerate(recommendations, start=1)
        ]
        stage_counts = Counter(str(candidate.get("retrieval_stage") or "unknown") for candidate in recommendations)
        common_payload = {
            "tenant_id": context.tenant_id,
            "catalog_id": context.catalog_id,
            "user_id": user_id,
            "session_id": session_id,
            "request_id": resolved_request_id,
            "source": "recommendation_api",
        }
        append_event_fn(
            {
                **common_payload,
                "event_type": "recommendation_request",
                "movie_id": query_movie.get("id"),
                "metadata": {
                    "endpoint": endpoint,
                    "query_movie": {
                        "id": query_movie.get("id"),
                        "title": query_movie.get("title"),
                    },
                    "requested_count": len(recommendations),
                    "candidate_ids": [c.get("movie_id") for c in ranked_candidates],
                    "retrieval_stage_counts": dict(stage_counts),
                    "lineage": lineage,
                },
            }
        )
        for candidate in ranked_candidates:
            append_event_fn(
                {
                    **common_payload,
                    "event_type": "recommendation_impression",
                    "movie_id": candidate.get("movie_id"),
                    "metadata": {
                        "endpoint": endpoint,
                        "seed_movie_id": query_movie.get("id"),
                        "seed_title": query_movie.get("title"),
                        "lineage": lineage,
                        **candidate,
                    },
                }
            )
    except Exception as exc:
        logger.warning("Recommendation event logging skipped: %s", exc)

    return resolved_request_id


# ---------------------------------------------------------------------------
# Private-named aliases used by backend/main.py (kept here after extraction)
# ---------------------------------------------------------------------------
from backend.recommender_helpers import safe_float as _safe_float


def _serving_lineage(rec) -> dict:
    """Return compact model/artifact lineage for recommendation events."""
    return serving_lineage(rec)


def _candidate_event_summary(candidate: dict, rank: int) -> dict:
    """Return the event-safe summary for one ranked recommendation."""
    return candidate_event_summary(candidate, rank, safe_float_fn=_safe_float)


# ---------------------------------------------------------------------------
# Private-named wrappers moved from backend/main.py (task 2.2)
# These preserve the exact call signatures that main.py and routers expect.
# ---------------------------------------------------------------------------
import os
import uuid as _uuid
from collections import Counter as _Counter

from backend.events import append_event as _append_event
from backend.recommender_helpers import event_logging_enabled as _event_logging_enabled


def record_recommendation_events(
    *,
    endpoint: str,
    context,
    query_movie: dict,
    recommendations: list[dict],
    rec,
    request_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str:
    """Persist request and impression events for offline analysis and training labels."""
    resolved_request_id = request_id or str(_uuid.uuid4())
    if not _event_logging_enabled():
        return resolved_request_id

    try:
        lineage = _serving_lineage(rec)
        ranked_candidates = [
            _candidate_event_summary(candidate, rank) for rank, candidate in enumerate(recommendations, start=1)
        ]
        stage_counts = _Counter(str(candidate.get("retrieval_stage") or "unknown") for candidate in recommendations)
        common_payload = {
            "tenant_id": context.tenant_id,
            "catalog_id": context.catalog_id,
            "user_id": user_id,
            "session_id": session_id,
            "request_id": resolved_request_id,
            "source": "recommendation_api",
        }
        _append_event(
            {
                **common_payload,
                "event_type": "recommendation_request",
                "movie_id": query_movie.get("id"),
                "metadata": {
                    "endpoint": endpoint,
                    "query_movie": {
                        "id": query_movie.get("id"),
                        "title": query_movie.get("title"),
                    },
                    "requested_count": len(recommendations),
                    "candidate_ids": [candidate.get("movie_id") for candidate in ranked_candidates],
                    "retrieval_stage_counts": dict(stage_counts),
                    "lineage": lineage,
                },
            }
        )
        for candidate in ranked_candidates:
            _append_event(
                {
                    **common_payload,
                    "event_type": "recommendation_impression",
                    "movie_id": candidate.get("movie_id"),
                    "metadata": {
                        "endpoint": endpoint,
                        "seed_movie_id": query_movie.get("id"),
                        "seed_title": query_movie.get("title"),
                        "lineage": lineage,
                        **candidate,
                    },
                }
            )
    except Exception as exc:
        logger.warning("Recommendation event logging skipped: %s", exc)

    return resolved_request_id


async def remote_payload_or_raise(
    path: str,
    params: dict | None = None,
    context=None,
) -> object | None:
    """Return remote recommender payload when configured, otherwise None."""
    from fastapi import HTTPException

    from backend.remote_recommender import remote_get_json, remote_recommender_url

    def _env_truthy(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    remote_response = await remote_get_json(path, params=params, context=context)
    if remote_response is None:
        if _env_truthy("NOVA_REMOTE_RECOMMENDER_REQUIRED") and remote_recommender_url():
            raise HTTPException(status_code=503, detail="Remote recommender unavailable")
        return None
    if remote_response.status_code >= 400:
        detail = remote_response.payload
        if isinstance(remote_response.payload, dict) and "detail" in remote_response.payload:
            detail = remote_response.payload["detail"]
        raise HTTPException(status_code=remote_response.status_code, detail=detail)
    return remote_response.payload
