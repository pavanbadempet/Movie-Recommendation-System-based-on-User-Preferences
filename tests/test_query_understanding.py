import pytest
from backend.intelligence.query_understanding import parse_query_intent, intent_score, VIBE_PROFILES

def test_parse_standard_intent():
    intent = parse_query_intent("latest mind-bending sci-fi movies")
    assert "science fiction" in intent["genres"]
    assert intent["recent"] is True
    assert "mind_bending" in intent.get("vibes", [])

def test_parse_cyberpunk_vibe():
    intent = parse_query_intent("neon cyberpunk thriller with synth music")
    assert "cyberpunk" in intent.get("vibes", [])
    assert "thriller" in intent["genres"]

def test_parse_cozy_melancholic_vibe():
    intent = parse_query_intent("rainy cozy studio ghibli feeling romance")
    assert "cozy_melancholic" in intent.get("vibes", [])
    assert "romance" in intent["genres"] or "animation" in intent["genres"]

def test_intent_scoring_boost():
    intent = parse_query_intent("cyberpunk sci-fi action")
    movie = {
        "title": "Blade Runner 2049",
        "genres": "Science Fiction, Drama",
        "overview": "A cyberpunk neon dystopia exploring human identity.",
        "release_date": "2017-10-06",
        "vote_average": 8.0,
        "vote_count": 12000
    }
    score, reasons = intent_score(movie, intent, current_year=2026)
    assert score > 0
    assert any("genre" in r or "vibe" in r for r in reasons)
