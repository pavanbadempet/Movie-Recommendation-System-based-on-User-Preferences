"""
API integration tests for FastAPI backend
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def mock_artifacts(tmp_path, monkeypatch):
    """Set up mock model artifacts for testing."""
    import faiss
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Mock movies
    movies = pd.DataFrame({
        "id": [100, 200, 300],
        "title": ["Test Movie A", "Test Movie B", "Test Movie C"],
        "overview": ["Action thriller", "Comedy romance", "Sci-fi adventure"],
        "genres": ["Action", "Comedy", "Sci-Fi"],
        "vote_average": [7.5, 6.5, 8.0],
        "vote_count": [1000, 500, 2000],
        "popularity": [100.0, 50.0, 150.0],
        "release_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
        "poster_path": [None, None, None],
    })
    movies.to_parquet(tmp_path / "movies_transformed.parquet")
    
    # Mock vectors (MPNet style - 768 dims)
    vecs = np.random.rand(3, 768).astype(np.float32)
    
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    
    np.save(tmp_path / "sbert_embeddings.npy", vecs)
    
    # FAISS index
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    faiss.write_index(idx, str(tmp_path / "faiss.index"))
    
    # Patch paths
    import backend.recommender as rec
    monkeypatch.setattr(rec, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(rec, "DATA_DIR", tmp_path)
    
    # Reset singleton
    rec._recommender = None
    
    return tmp_path


class TestHealthEndpoint:
    def test_health_returns_ok(self, mock_artifacts):
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["movie_count"] == 3


class TestSearchEndpoint:
    def test_search_finds_movie(self, mock_artifacts):
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/search", params={"q": "Test Movie A"})
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert results[0]["title"] == "Test Movie A"

    def test_search_empty_returns_empty(self, mock_artifacts):
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/search", params={"q": "xyz123nonexistent"})
        assert resp.status_code == 200
        assert resp.json() == []


class TestMoviesEndpoint:
    def test_movies_returns_list(self, mock_artifacts):
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/movies", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        movies = resp.json()
        assert len(movies) == 2

    def test_movies_pagination(self, mock_artifacts):
        from backend.main import app
        client = TestClient(app)
        resp1 = client.get("/movies", params={"limit": 1, "offset": 0})
        resp2 = client.get("/movies", params={"limit": 1, "offset": 1})
        
        m1 = resp1.json()[0]
        m2 = resp2.json()[0]
        assert m1["id"] != m2["id"]


class TestRecommendEndpoints:
    def test_recommend_by_id(self, mock_artifacts):
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/recommend/id/100", params={"n": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert "query_movie" in data
        assert "recommendations" in data
        assert data["query_movie"]["id"] == 100

    def test_recommend_by_id_not_found(self, mock_artifacts):
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/recommend/id/999999")
        assert resp.status_code == 404

    def test_recommend_by_title(self, mock_artifacts):
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/recommend/title/Test Movie B", params={"n": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query_movie"]["title"] == "Test Movie B"
