"""Tests for deterministic semantic item twins."""

from backend.semantic_twin import build_semantic_twin, compare_semantic_twins


def test_semantic_twin_extracts_structured_signals():
    twin = build_semantic_twin(
        {
            "id": 19995,
            "title": "Avatar",
            "overview": "A marine travels to an alien moon and joins a native civilization in an ecological conflict.",
            "genres": "Action, Adventure, Science Fiction",
            "vote_count": 12000,
            "vote_average": 7.6,
        }
    )

    assert twin["item_id"] == 19995
    assert "science fiction" in twin["genres"]
    assert "alien" in twin["concepts"]
    assert "world_immersion" in twin["viewer_jobs"]
    assert twin["generated_by"]["llm_in_hot_path"] is False


def test_semantic_affinity_rewards_concept_and_viewer_job_overlap():
    query = build_semantic_twin(
        {
            "id": 1,
            "title": "Space Epic",
            "overview": "A team explores an alien planet and protects a hidden civilization during a war.",
            "genres": "Adventure, Science Fiction",
            "vote_count": 1000,
        }
    )
    good = build_semantic_twin(
        {
            "id": 2,
            "title": "Alien World",
            "overview": "Explorers discover a planet civilization and fight to protect its future.",
            "genres": "Adventure, Science Fiction",
            "vote_count": 1000,
        }
    )
    weak = build_semantic_twin(
        {
            "id": 3,
            "title": "Office Comedy",
            "overview": "Friends navigate work jokes and romance in a city office.",
            "genres": "Comedy, Romance",
            "vote_count": 1000,
        }
    )

    good_score = compare_semantic_twins(query, good)
    weak_score = compare_semantic_twins(query, weak)

    assert good_score.score > weak_score.score
    assert good_score.shared_concepts
    assert good_score.reasons
