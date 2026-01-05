"""
Tests for ETL modules: config, ingest, transform, index
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# ----- Config Tests -----

def test_paths_dataclass():
    """Paths dataclass creates directories."""
    from etl.config import Paths
    with tempfile.TemporaryDirectory() as tmp:
        p = Paths(
            raw_data=Path(tmp) / "raw",
            processed_data=Path(tmp) / "processed",
            models=Path(tmp) / "models",
            logs=Path(tmp) / "logs",
        )
        assert p.raw_data.exists()
        assert p.processed_data.exists()

def test_data_config_defaults():
    """DataConfig has sensible defaults."""
    from etl.config import DataConfig
    cfg = DataConfig()
    assert cfg.min_vote_count == 50
    assert cfg.tfidf_max_features == 5000
    assert cfg.n_recommendations == 10


# ----- Ingest Tests -----

class TestIngest:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "title": ["Movie A", "Movie B", "Movie C"],
            "overview": ["Good story", "Another story", "Third one"],
            "genres": ["[]", "[]", "[]"],
            "vote_average": [7.5, 6.0, 8.0],
            "vote_count": [100, 20, 150],
            "popularity": [50.0, 30.0, 70.0],
            "release_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
            "poster_path": ["/a.jpg", "/b.jpg", "/c.jpg"],
        })

    def test_filter_movies_removes_low_votes(self, sample_df):
        """filter_movies removes movies below vote threshold."""
        from etl.ingest import filter_movies
        result = filter_movies(sample_df)
        assert len(result) == 2  # id 2 has only 20 votes
        assert 2 not in result["id"].values

    def test_filter_movies_removes_nulls(self):
        """filter_movies removes rows with null title/overview."""
        from etl.ingest import filter_movies
        df = pd.DataFrame({
            "id": [1, 2],
            "title": ["Movie", None],
            "overview": ["Story", "Another"],
            "vote_count": [100, 100],
        })
        result = filter_movies(df)
        assert len(result) == 1

    def test_quality_checks_returns_metrics(self, sample_df):
        """run_quality_checks returns dict with expected keys."""
        from etl.ingest import run_quality_checks
        metrics = run_quality_checks(sample_df)
        assert "total_rows" in metrics
        assert "null_titles" in metrics
        assert metrics["total_rows"] == 3


# ----- Transform Tests -----

class TestTransform:
    def test_parse_json_column_list(self):
        """parse_json_column extracts names from list of dicts."""
        from etl.transform import parse_json_column
        val = "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Drama'}]"
        result = parse_json_column(val)
        assert result == ["Action", "Drama"]

    def test_parse_json_column_empty(self):
        """parse_json_column returns [] for empty values."""
        from etl.transform import parse_json_column
        assert parse_json_column("") == []
        assert parse_json_column(None) == []
        assert parse_json_column("[]") == []

    def test_parse_json_column_comma_separated(self):
        """parse_json_column handles comma-separated as fallback."""
        from etl.transform import parse_json_column
        result = parse_json_column("Action, Comedy, Drama")
        assert result == ["Action", "Comedy", "Drama"]

    def test_clean_text(self):
        """clean_text lowercases and removes special chars."""
        from etl.transform import clean_text
        assert clean_text("Hello, World!") == "hello world"
        assert clean_text("Test@123") == "test 123"
        assert clean_text(None) == ""

    def test_generate_tags_creates_column(self):
        """generate_tags adds 'tags' column."""
        from etl.transform import generate_tags
        df = pd.DataFrame({
            "id": [1],
            "title": ["Test Movie"],
            "overview": ["A great adventure story."],
            "genres": ["[{'name': 'Adventure'}]"],
        })
        result = generate_tags(df)
        assert "tags" in result.columns
        assert "adventure" in result.iloc[0]["tags"]

    def test_build_sbert_embeddings(self, monkeypatch):
        """build_sbert_embeddings returns model and normalized embeddings."""
        import etl.transform as t
        from unittest.mock import MagicMock
        
        # Mock SentenceTransformer
        mock_model = MagicMock()
        # Return random 384-dim vectors
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        
        mock_cls = MagicMock(return_value=mock_model)
        monkeypatch.setattr(t, "SentenceTransformer", mock_cls)
        
        tags = pd.Series(["action movie", "comedy", "drama"])
        model, vecs = t.build_sbert_embeddings(tags)
        
        assert model == mock_model
        assert vecs.shape == (3, 384)
        # Check normalization (roughly)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


# ----- Index Tests -----

class TestIndex:
    def test_normalize_vectors(self):
        """normalize_vectors produces unit vectors."""
        from etl.index import normalize_vectors
        vecs = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        normed = normalize_vectors(vecs)
        # L2 norm should be 1
        np.testing.assert_almost_equal(np.linalg.norm(normed[0]), 1.0)
        np.testing.assert_almost_equal(np.linalg.norm(normed[1]), 1.0)

    def test_build_faiss_index(self):
        """build_faiss_index creates index with correct count."""
        from etl.index import build_faiss_index
        vecs = np.random.rand(50, 128).astype(np.float32)
        idx = build_faiss_index(vecs)
        assert idx.ntotal == 50

    def test_faiss_search(self):
        """FAISS search returns correct results."""
        from etl.index import build_faiss_index
        vecs = np.eye(10, dtype=np.float32)  # identity matrix
        idx = build_faiss_index(vecs)
        # Search for first vector
        query = vecs[0:1]
        dists, indices = idx.search(query, 3)
        assert indices[0][0] == 0  # first match is itself


# ----- Recommender Tests -----

class TestRecommender:
    @pytest.fixture
    def mock_recommender(self, tmp_path):
        """Create recommender with mock data."""
        import faiss
        import joblib
        
        # Create mock data
        movies = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "title": ["Avatar", "Titanic", "Inception", "Interstellar", "Dunkirk"],
            "overview": ["blue aliens", "ship sinks", "dreams within dreams", "space travel", "war movie"],
        })
        movies.to_parquet(tmp_path / "movies_transformed.parquet")
        
        # Create random vectors (Simulating SBERT embeddings)
        # SBERT is typically 384 dimensions
        vecs = np.random.rand(5, 384).astype(np.float32)
        
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        
        # Save SBERT embeddings
        np.save(tmp_path / "sbert_embeddings.npy", vecs)
        
        # Build Index
        idx = faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        faiss.write_index(idx, str(tmp_path / "faiss.index"))
        
        return tmp_path

    def test_recommender_load(self, mock_recommender, monkeypatch):
        """Recommender loads all artifacts."""
        import backend.recommender as rec
        monkeypatch.setattr(rec, "MODELS_DIR", mock_recommender)
        monkeypatch.setattr(rec, "DATA_DIR", mock_recommender)
        
        r = rec.Recommender().load()
        assert r._index is not None
        assert r._movies is not None
        assert len(r.movies) == 5
        assert r._vectors is not None
        assert r._vectors.shape == (5, 384)

    def test_search_movies(self, mock_recommender, monkeypatch):
        """search_movies finds by title."""
        import backend.recommender as rec
        monkeypatch.setattr(rec, "MODELS_DIR", mock_recommender)
        monkeypatch.setattr(rec, "DATA_DIR", mock_recommender)
        
        r = rec.Recommender().load()
        results = r.search_movies("avatar")
        assert len(results) == 1
        assert results[0]["title"] == "Avatar"

    def test_recommend_by_id(self, mock_recommender, monkeypatch):
        """recommend_by_id returns similar movies."""
        import backend.recommender as rec
        monkeypatch.setattr(rec, "MODELS_DIR", mock_recommender)
        monkeypatch.setattr(rec, "DATA_DIR", mock_recommender)
        
        r = rec.Recommender().load()
        recs = r.recommend_by_id(1, n=2)  # Avatar
        assert len(recs) == 2
        assert all("similarity_score" in m for m in recs)
