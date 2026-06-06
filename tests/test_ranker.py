"""
Tests for Nova's learned ranking layer.
"""

import pandas as pd

from backend.pipeline.ranker import candidate_features, load_ranker
from backend.pipeline.ranker_training import build_item_feedback, build_training_frame, promotion_decision, train_nova_ranker


def sample_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [100, 200, 300, 400],
            "title": ["Action One", "Space One", "Comedy One", "Drama One"],
            "overview": ["a" * 40, "b" * 40, "c" * 40, "d" * 40],
            "genres": ["Action", "Sci-Fi", "Comedy", "Drama"],
            "vote_average": [7.5, 8.2, 6.8, 7.1],
            "vote_count": [1000, 2000, 400, 700],
            "popularity": [100.0, 150.0, 40.0, 60.0],
            "release_date": ["2020-01-01", "2022-01-01", "2019-01-01", "2018-01-01"],
        }
    )


def test_candidate_features_are_stable_length():
    features = candidate_features(
        {
            "similarity_score": 0.8,
            "vote_average": 7.5,
            "vote_count": 1000,
            "popularity": 100,
            "release_date": "2022-01-01",
            "retrieval_signals": {"dense": 0.9, "sparse": 0.2, "metadata": 0.6, "behavior": 0.1},
        }
    )

    assert len(features) == 11
    assert features[0] == 0.8
    assert 0 <= features[-1] <= 1


def test_build_item_feedback_aggregates_implicit_events():
    feedback = build_item_feedback(
        [
            {"event_type": "view", "movie_id": 100},
            {"event_type": "click", "movie_id": 100},
            {"event_type": "rating", "movie_id": 200, "rating": 5},
        ]
    )

    assert feedback[100] > 0
    assert feedback[200] == 1.0


def test_train_ranker_saves_loadable_artifact(tmp_path):
    events = [
        {"event_type": "view", "movie_id": 100},
        {"event_type": "click", "movie_id": 200},
        {"event_type": "rating", "movie_id": 200, "rating": 5},
        {"event_type": "view", "movie_id": 300},
    ]
    artifact_path = tmp_path / "nova_ranker.joblib"

    report = train_nova_ranker(sample_movies(), events, artifact_path)
    ranker = load_ranker(artifact_path)

    assert artifact_path.exists()
    assert report["metadata"]["training_mode"] == "implicit_feedback"
    assert report["metadata"]["evaluation"]["ndcg_at_k"] >= 0
    assert ranker is not None

    candidates = [
        {"id": 100, "similarity_score": 0.4, "vote_average": 7.5, "vote_count": 1000, "popularity": 100},
        {"id": 200, "similarity_score": 0.4, "vote_average": 8.2, "vote_count": 2000, "popularity": 150},
    ]
    reranked = ranker.rerank(candidates)

    assert "ranker_score" in reranked[0]
    assert "learned_ranker" in reranked[0]["retrieval_stage"]


def test_build_training_frame_bootstraps_without_events():
    features, labels, metadata = build_training_frame(sample_movies(), [])

    assert len(features) == len(labels) == 4
    assert metadata["training_mode"] == "catalog_bootstrap"
    assert labels.max() > labels.min()


def test_promotion_gate_promotes_safe_candidate(tmp_path):
    candidate_path = tmp_path / "nova_ranker.candidate.joblib"
    production_path = tmp_path / "nova_ranker.joblib"

    report = train_nova_ranker(
        sample_movies(),
        [{"event_type": "click", "movie_id": 200}, {"event_type": "rating", "movie_id": 200, "rating": 5}],
        candidate_path,
        promotion_gate=True,
        production_path=production_path,
    )

    assert report["promotion"]["decision"] == "promote"
    assert report["promoted"] is True
    assert production_path.exists()
    assert "baseline_evaluation" in report["metadata"]


def test_promotion_decision_rejects_clear_regression():
    decision = promotion_decision(
        candidate={"ndcg_at_k": 0.5, "recall_at_k": 0.1},
        baseline={"ndcg_at_k": 0.9, "recall_at_k": 0.5},
    )

    assert decision["decision"] == "reject"
    assert decision["reasons"]
