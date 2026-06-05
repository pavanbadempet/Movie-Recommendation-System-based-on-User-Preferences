"""
Tests for lightweight search intent extraction.
"""

from backend.query_understanding import intent_score, parse_query_intent


def test_parse_query_intent_extracts_genre_and_recency():
    intent = parse_query_intent("latest sci fi space adventure")

    assert "science fiction" in intent["genres"]
    assert "adventure" in intent["genres"]
    assert intent["recent"] is True


def test_intent_score_rewards_matching_movie():
    intent = parse_query_intent("best recent sci fi")
    score, reasons = intent_score(
        {
            "genres": "Science Fiction, Adventure",
            "release_date": "2024-01-01",
            "vote_average": 8.0,
            "vote_count": 500,
        },
        intent,
        current_year=2026,
    )

    assert score > 0
    assert reasons
