"""
Tests for behavior event capture and feature aggregation.
"""

import pytest

from backend.events import (
    aggregate_behavior_features,
    append_event,
    build_user_behavior_profile,
    event_storage_status,
    iter_events,
    normalize_event,
    summarize_recommendation_events,
)
from backend.recommender import Recommender


def test_append_event_and_aggregate_features(tmp_path):
    event_path = tmp_path / "movie_events.jsonl"

    append_event({"event_type": "view", "movie_id": 100}, event_path)
    append_event({"event_type": "click", "movie_id": "100"}, event_path)
    append_event({"event_type": "rating", "movie_id": 100, "rating": 5}, event_path)
    append_event({"event_type": "search", "query_text": "space drama"}, event_path)

    events = list(iter_events(event_path))
    assert len(events) == 4
    assert all("event_id" in event for event in events)

    features = aggregate_behavior_features(event_path)
    assert features["total_events"] == 4
    assert features["event_type_counts"]["view"] == 1
    assert features["event_type_counts"]["click"] == 1
    assert features["event_type_counts"]["rating"] == 1
    assert features["event_type_counts"]["search"] == 1

    movie_stats = features["trending_movies"]["100"]
    assert movie_stats["event_count"] == 3
    assert movie_stats["views"] == 1
    assert movie_stats["clicks"] == 1
    assert movie_stats["ratings"] == 1
    assert movie_stats["avg_rating"] == 5.0
    assert movie_stats["tenant_id"] == "demo-media-co"
    assert movie_stats["catalog_id"] == "tmdb-movies"
    assert features["top_searches"] == [{"query_text": "space drama", "count": 1}]


def test_recommendation_request_is_counted_without_trending_bias(tmp_path):
    event_path = tmp_path / "movie_events.jsonl"

    append_event(
        {
            "event_type": "recommendation_request",
            "movie_id": 100,
            "request_id": "request-1",
            "metadata": {
                "candidate_ids": [200],
                "numpy_like_score": 0.91,
            },
        },
        event_path,
    )
    append_event(
        {
            "event_type": "recommendation_impression",
            "movie_id": 200,
            "request_id": "request-1",
            "metadata": {"rank": 1},
        },
        event_path,
    )

    features = aggregate_behavior_features(event_path)

    assert features["total_events"] == 2
    assert features["event_type_counts"]["recommendation_request"] == 1
    assert features["event_type_counts"]["recommendation_impression"] == 1
    assert "100" not in features["trending_movies"]
    assert features["trending_movies"]["200"]["impressions"] == 1


def test_summarize_recommendation_events(tmp_path):
    event_path = tmp_path / "movie_events.jsonl"

    append_event(
        {
            "event_type": "recommendation_request",
            "movie_id": 100,
            "request_id": "request-1",
            "metadata": {
                "query_movie": {"id": 100, "title": "Seed"},
                "retrieval_stage_counts": {"semantic": 2},
            },
        },
        event_path,
    )
    append_event(
        {
            "event_type": "recommendation_impression",
            "movie_id": 200,
            "request_id": "request-1",
            "metadata": {"rank": 1, "retrieval_stage": "semantic"},
        },
        event_path,
    )
    append_event(
        {
            "event_type": "click",
            "movie_id": 200,
            "request_id": "request-1",
        },
        event_path,
    )

    summary = summarize_recommendation_events(event_path)

    assert summary["request_count"] == 1
    assert summary["distinct_request_count"] == 1
    assert summary["impression_count"] == 1
    assert summary["click_count"] == 1
    assert summary["click_through_rate"] == 1.0
    assert summary["avg_impressions_per_request"] == 1.0
    assert summary["top_seed_movies"] == [{"movie_id": "100", "request_count": 1}]
    assert summary["top_recommended_movies"] == [{"movie_id": "200", "impression_count": 1}]
    assert summary["rank_position_counts"]["1"] == 1
    assert summary["retrieval_stage_counts"]["semantic"] == 1


def test_content_event_uses_product_identifiers(tmp_path):
    event_path = tmp_path / "content_events.jsonl"

    event = append_event(
        {
            "event_type": "view",
            "tenant_id": "ott-startup",
            "catalog_id": "short-films",
            "content_id": "content-123",
            "session_id": "session-1",
        },
        event_path,
    )

    assert event["tenant_id"] == "ott-startup"
    assert event["catalog_id"] == "short-films"
    assert event["content_id"] == "content-123"

    features = aggregate_behavior_features(event_path)
    assert features["trending_movies"]["content-123"]["content_id"] == "content-123"


def test_dual_event_store_falls_back_to_jsonl_without_database_url(tmp_path, monkeypatch):
    event_path = tmp_path / "dual_events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))
    monkeypatch.setenv("NOVA_EVENT_STORE", "dual")
    monkeypatch.delenv("NOVA_EVENT_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    event = append_event({"event_type": "view", "movie_id": 100})

    assert event["event_store"] == "dual"
    assert event["durable"] is False
    assert event_path.exists()
    features = aggregate_behavior_features()
    assert features["event_store"] == "dual"
    assert features["durable"] is False
    assert features["total_events"] == 1


def test_postgres_event_store_requires_database_url(monkeypatch):
    monkeypatch.setenv("NOVA_EVENT_STORE", "postgres")
    monkeypatch.delenv("NOVA_EVENT_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="NOVA_EVENT_DATABASE_URL"):
        append_event({"event_type": "view", "movie_id": 100})


def test_event_storage_status_for_explicit_jsonl_path(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_EVENT_STORE", "postgres")
    monkeypatch.setenv("NOVA_EVENT_DATABASE_URL", "postgresql://example")

    status = event_storage_status(tmp_path / "events.jsonl")

    assert status["event_store"] == "jsonl"
    assert status["durable"] is False
    assert status["postgres_configured"] is True


def test_build_user_behavior_profile(tmp_path):
    event_path = tmp_path / "user_events.jsonl"

    append_event({"event_type": "view", "movie_id": 100, "user_id": "u1"}, event_path)
    append_event({"event_type": "rating", "movie_id": 200, "user_id": "u1", "rating": 5}, event_path)
    append_event({"event_type": "search", "query_text": "space drama", "user_id": "u1"}, event_path)
    append_event({"event_type": "view", "movie_id": 300, "user_id": "u2"}, event_path)

    profile = build_user_behavior_profile("u1", event_path)

    assert profile["user_id"] == "u1"
    assert set(profile["seed_movie_ids"]) == {100, 200}
    assert profile["top_searches"] == [{"query_text": "space drama", "count": 1}]


def test_normalize_event_rejects_invalid_rating():
    with pytest.raises(ValueError, match="rating must be between 1 and 5"):
        normalize_event({"event_type": "rating", "movie_id": 100, "rating": 8})


def test_recommender_behavior_boost_is_bounded():
    recommender = Recommender()
    recommender._behavior_features = {
        "trending_movies": {
            "100": {
                "event_count": 20,
                "views": 12,
                "clicks": 5,
                "ratings": 3,
                "avg_rating": 4.7,
            }
        }
    }

    boost, reasons = recommender._behavior_boost(100)

    assert 0 < boost <= 0.10
    assert reasons[0] == "Trending with viewers (20 recent events)"
    assert reasons[1] == "Audience signal (4.7/5)"
