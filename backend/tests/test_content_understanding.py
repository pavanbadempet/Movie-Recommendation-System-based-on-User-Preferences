from unittest.mock import MagicMock

from backend.intelligence.content_understanding import ContentUnderstandingEngine
from backend.intelligence.knowledge_graph import KnowledgeGraphEngine


def test_content_understanding_mocked():
    """Test that the classification pipeline filters low confidence scores and correctly formats themes/moods."""
    engine = ContentUnderstandingEngine(device="cpu")

    # Mock the HF pipeline outputs to avoid downloading massive models during tests
    engine.classifier = MagicMock()
    engine.classifier.return_value = {
        "labels": ["moral dilemma", "coming-of-age", "time travel"],
        "scores": [0.95, 0.88, 0.10],  # time travel should be filtered out (< 0.5)
    }

    engine.ner_pipeline = MagicMock()
    engine.ner_pipeline.return_value = [
        {"entity_group": "LOC", "word": "Gotham"},
        {"entity_group": "PER", "word": "Bruce Wayne"},
        {"entity_group": "LOC", "word": "A"},  # Too short, should be filtered
    ]

    # Mock load to bypass real model initialization
    engine._load_models = MagicMock()

    overview = "A dark billionaire fights crime in Gotham."

    # Test Theme/Mood Extraction
    res = engine.extract_themes_and_moods(overview)
    assert "moral dilemma" in res["themes"]
    assert "coming-of-age" in res["themes"]
    assert "time travel" not in res["themes"]  # Filtered

    # Test Entity Extraction
    ents = engine.extract_entities(overview)
    assert "Gotham" in ents["LOC"]
    assert "A" not in ents["LOC"]  # Filtered by length
    assert "Bruce Wayne" in ents["PER"]


def test_knowledge_graph_multi_hop():
    """Ensure the graph can find 2-hop connected movies through shared semantics."""
    kg = KnowledgeGraphEngine()

    # Mock data
    movies = [{"id": 1, "title": "The Dark Knight"}, {"id": 2, "title": "Inception"}, {"id": 3, "title": "Toy Story"}]

    parsed = {
        1: {"themes": ["moral dilemma", "corruption"], "moods": ["dark"]},
        2: {"themes": ["moral dilemma", "time travel"], "moods": ["dark", "tense"]},
        3: {"themes": ["coming-of-age"], "moods": ["uplifting"]},
    }

    kg.build_graph(movies, parsed)

    # "The Dark Knight" (1) shares "moral dilemma" and "dark" with "Inception" (2)
    # It shares nothing with "Toy Story" (3)
    similar = kg.find_thematically_similar(1)

    assert len(similar) > 0

    # The first element of the first tuple should be movie ID 2
    top_movie_id, score = similar[0]
    assert top_movie_id == 2
    assert score == 2.0  # 1 point for shared theme, 1 point for shared mood
