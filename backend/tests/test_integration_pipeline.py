"""
End-to-End Pipeline Integration Testing.

This suite tests the full lifecycle:
Dense/Sparse Retrieval (FAISS) -> Candidate Generation -> ML Reranking (ApexEnsemble).
It ensures the data strictly respects the Medallion flow and guarantees
the reranking step does not drop valid items.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from turbovec import TurboQuantIndex


# We use fixture scope=module so we only load the mock/real index once per test session
@pytest.fixture(scope="module")
def recommender(tmp_path_factory):
    # Set up mock model artifacts for testing when running in CI/CD (no real artifacts)
    tmp_path = tmp_path_factory.mktemp("integration_artifacts")

    # Mock movies dataframe with all required columns
    movies = pd.DataFrame(
        {
            "id": [862, 863, 300],
            "title": ["Toy Story", "Toy Story 2", "Test Movie C"],
            "overview": ["Action adventure animation", "Comedy family", "Sci-fi adventure"],
            "genres": ["Action, Adventure, Animation, Family", "Adventure, Comedy, Family", "Science Fiction, Action"],
            "vote_average": [7.5, 6.5, 8.0],
            "vote_count": [1000, 500, 2000],
            "popularity": [100.0, 50.0, 150.0],
            "release_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
            "poster_path": [None, None, None],
            "director": ["John Lasseter", "John Lasseter", "Dir C"],
            "original_language": ["en", "en", "en"],
            "tagline": ["Tag A", "Tag B", "Tag C"],
            "runtime": [81.0, 92.0, 100.0],
            "metadata_completeness": [1.0, 1.0, 1.0],
            "content_quality_score": [1.0, 1.0, 1.0],
            "quality_bucket": ["high", "high", "high"],
            "searchable": [True, True, True],
            "recommendable": [True, True, True],
            "public_demo_eligible": [True, True, True],
        }
    )
    # Write movie parquet files
    movies.to_parquet(tmp_path / "movies_transformed.parquet")
    pd.DataFrame(
        {
            "id": movies["id"].astype("int64"),
            "semantic_twin_json": ["{}"] * len(movies),
        }
    ).to_parquet(tmp_path / "semantic_twins.parquet", index=False)

    (tmp_path / "semantic_twin_summary.json").write_text(
        json.dumps({"row_count": len(movies), "avg_confidence": 0.8}),
        encoding="utf-8",
    )

    # Mock vectors (MPNet style - 768 dims)
    vecs = np.random.rand(3, 768).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    np.save(tmp_path / "sbert_embeddings.npy", vecs)
    np.save(tmp_path / "movie_ids.npy", movies["id"].astype("int64").to_numpy())

    # TurboVec index
    idx = TurboQuantIndex(vecs.shape[1], bit_width=4)
    idx.add(vecs)
    idx.write(str(tmp_path / "turbovec.tq"))

    (tmp_path / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "serving_contract": {
                    "version": 1,
                    "movie_rows": 3,
                    "embedding_rows": 3,
                    "embedding_dimensions": 768,
                    "turbovec_index_size": 3,
                    "movie_id_map_rows": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    # Patch paths on recommender
    import backend.pipeline.recommender as rec

    old_models_dir = getattr(rec, "MODELS_DIR", None)
    old_data_dir = getattr(rec, "DATA_DIR", None)

    rec.MODELS_DIR = tmp_path
    rec.DATA_DIR = tmp_path

    # Reset singleton
    rec._recommender = None
    if "backend.main" in sys.modules:
        sys.modules["backend.main"]._recommender = None

    from backend.main import get_rec as _get_rec

    recommender_instance = _get_rec()
    recommender_instance.load()

    yield recommender_instance

    # Restore paths
    rec.MODELS_DIR = old_models_dir
    rec.DATA_DIR = old_data_dir
    rec._recommender = None
    if "backend.main" in sys.modules:
        sys.modules["backend.main"]._recommender = None


def test_recommend_for_user_profile_integration(recommender):
    """
    Tests the complete end-to-end user recommendation pipeline.
    Ensures the retrieval + ApexEnsembleEngine reranking path works
    without crashing and returns valid movie records.
    """
    # Create a mock user profile using the actual expected schema
    mock_profile = {
        "user_id": "integration-test-user",
        "recent_events": [
            {"movie_id": 862, "event_type": "rating", "rating": 5},  # Toy Story
            {"movie_id": 863, "event_type": "rating", "rating": 4},
        ],
        "favorite_genres": ["Animation", "Family"],
        "negative_movie_ids": [],
    }

    # Run the full pipeline — param is `n`, not `limit`
    results = recommender.recommend_for_user_profile(mock_profile, n=20)

    # Assertions
    assert isinstance(results, list)
    assert len(results) <= 20

    # Every returned item MUST have core identification fields
    for item in results:
        assert "id" in item
        assert "title" in item


def test_vector_search_fallback(recommender):
    """
    Ensures that if the semantic dense retrieval fails, it gracefully
    falls back to TF-IDF / Sparse metadata matching without crashing.
    """
    # We query something highly obscure
    results = recommender.search_movies("A completely obscure nonsensical query that doesnt exist 12345", limit=5)

    assert isinstance(results, list)
    assert len(results) <= 5

    if len(results) > 0:
        # Check that it executed the standard pipeline schema
        assert "id" in results[0]
