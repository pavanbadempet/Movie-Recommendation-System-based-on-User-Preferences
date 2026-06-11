"""Shared pytest configuration for backend/tests/."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from turbovec import TurboQuantIndex


@pytest.fixture(autouse=True)
def _backend_test_env(tmp_path, monkeypatch):
    """Ensure backend tests have consistent environment variables and mock artifacts."""
    monkeypatch.setenv("NOVA_DISABLE_MODEL_DOWNLOADS", "1")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-ci-only")

    # Mock movies
    movies = pd.DataFrame(
        {
            "id": [100, 200, 300],
            "title": ["Test Movie A", "Test Movie B", "Test Movie C"],
            "overview": ["Action thriller", "Comedy romance", "Sci-fi adventure"],
            "genres": ["Action", "Comedy", "Sci-Fi"],
            "vote_average": [7.5, 6.5, 8.0],
            "vote_count": [1000, 500, 2000],
            "popularity": [100.0, 50.0, 150.0],
            "release_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
            "poster_path": [None, None, None],
        }
    )
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

    # Patch paths
    import backend.pipeline.recommender as rec

    monkeypatch.setattr(rec, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(rec, "DATA_DIR", tmp_path)
    monkeypatch.setenv("NOVA_USAGE_PATH", str(tmp_path / "api_usage.jsonl"))
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.delenv("NOVA_API_KEYS", raising=False)

    # Reset singleton
    rec._recommender = None
    if "backend.main" in sys.modules:
        sys.modules["backend.main"]._recommender = None
