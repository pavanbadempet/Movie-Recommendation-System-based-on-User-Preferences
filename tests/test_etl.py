"""
Tests for Pandas ETL module (consolidated).
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

# ----- Config Tests -----


def test_paths_dataclass():
    """Paths dataclass creates directories."""
    from etl.config import Paths

    with tempfile.TemporaryDirectory() as tmp:
        p = Paths(
            raw_data=Path(tmp) / "raw",
            processed_data=Path(tmp) / "processed",
            bronze_data=Path(tmp) / "bronze",
            silver_data=Path(tmp) / "silver",
            gold_data=Path(tmp) / "gold",
            models=Path(tmp) / "models",
            logs=Path(tmp) / "logs",
            quality_reports=Path(tmp) / "quality",
            manifests=Path(tmp) / "manifests",
        )
        assert p.raw_data.exists()
        assert p.processed_data.exists()
        assert p.bronze_data.exists()
        assert p.silver_data.exists()
        assert p.gold_data.exists()
        assert p.quality_reports.exists()
        assert p.manifests.exists()


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
        return pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Movie A", "Movie B", "Movie C"],
                "overview": ["Good story", "Another story", "Third one"],
                "genres": ["[]", "[]", "[]"],
                "vote_average": [7.5, 6.0, 8.0],
                "vote_count": [100, 20, 150],
                "popularity": [50.0, 30.0, 70.0],
                "release_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
                "poster_path": ["/a.jpg", "/b.jpg", "/c.jpg"],
            }
        )

    def test_filter_movies_keeps_low_vote_long_tail(self, sample_df):
        """filter_movies keeps low-vote movies and scores them as long-tail content."""
        from etl.pandas_etl import filter_movies

        result = filter_movies(sample_df)
        assert len(result) == 3
        assert 2 in result["id"].values
        assert "content_quality_score" in result.columns
        assert "quality_bucket" in result.columns

    def test_filter_movies_removes_null_title_but_keeps_missing_overview(self):
        """filter_movies removes rows without identity but keeps weak metadata."""
        from etl.pandas_etl import filter_movies

        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Movie", None, "No Overview"],
                "overview": ["Story", "Another", None],
                "vote_count": [100, 100, 0],
            }
        )
        result = filter_movies(df)
        assert len(result) == 2
        assert 3 in result["id"].values

    def test_filter_movies_deduplicates_by_highest_signal(self):
        """filter_movies keeps one deterministic row per movie id."""
        from etl.pandas_etl import filter_movies

        df = pd.DataFrame(
            {
                "id": [1, 1, 2],
                "title": ["Movie A Low", "Movie A High", "Movie B"],
                "overview": ["Story", "Better story", "Another"],
                "vote_count": [100, 500, 100],
                "popularity": [10.0, 40.0, 20.0],
            }
        )

        result = filter_movies(df)

        assert len(result) == 2
        assert result[result["id"] == 1].iloc[0]["title"] == "Movie A High"

    def test_quality_checks_returns_metrics(self, sample_df):
        """run_quality_checks returns dict with expected keys."""
        from etl.pandas_etl import run_quality_checks

        metrics = run_quality_checks(sample_df)
        assert "total_rows" in metrics
        assert "null_titles" in metrics
        assert metrics["total_rows"] == 3
        assert metrics["duplicate_ids"] == 0
        assert metrics["title_completeness"] == 1.0


# ----- Transform Tests -----


class TestTransform:
    def test_parse_json_column_list(self):
        """parse_json_column extracts names from list of dicts."""
        from etl.pandas_etl import parse_json_column

        val = "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Drama'}]"
        result = parse_json_column(val)
        assert result == ["Action", "Drama"]

    def test_parse_json_column_empty(self):
        """parse_json_column returns [] for empty values."""
        from etl.pandas_etl import parse_json_column

        assert parse_json_column("") == []
        assert parse_json_column(None) == []
        assert parse_json_column("[]") == []

    def test_parse_json_column_comma_separated(self):
        """parse_json_column handles comma-separated as fallback."""
        from etl.pandas_etl import parse_json_column

        result = parse_json_column("Action, Comedy, Drama")
        assert result == ["Action", "Comedy", "Drama"]

    def test_clean_text(self):
        """clean_text preserves punctuation for SBERT."""
        from etl.pandas_etl import clean_text

        # Logic changed to preserve punctuation and case for better embeddings
        assert clean_text("Hello, World!") == "Hello, World!"
        assert clean_text("Test@123") == "Test 123"
        assert clean_text(None) == ""

    def test_generate_tags_creates_column(self):
        """generate_tags adds 'tags' column."""
        from etl.pandas_etl import generate_tags

        df = pd.DataFrame(
            {
                "id": [1],
                "title": ["Test Movie"],
                "overview": ["A great adventure story."],
                "genres": ["[{'name': 'Adventure'}]"],
            }
        )
        result = generate_tags(df)
        assert "tags" in result.columns
        assert "adventure" in result.iloc[0]["tags"]
        assert result.iloc[0]["genres"] == "Adventure"

    def test_build_sbert_embeddings(self, monkeypatch):
        """build_sbert_embeddings returns model and normalized embeddings."""
        from unittest.mock import MagicMock

        import etl.pandas_etl as t

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
    def test_build_faiss_index(self):
        """build_faiss_index creates index with correct count."""

        from etl.pandas_etl import build_faiss_index

        # Override data_config for test if needed, but build_faiss_index uses it
        # Just ensure we test logic

        vecs = np.random.rand(50, 128).astype(np.float32)
        idx = build_faiss_index(vecs)
        assert idx.ntotal == 50

    def test_atomic_parquet_write_replaces_existing_file(self, tmp_path):
        """atomic_write_parquet replaces only after a complete write."""
        from etl.pandas_etl import atomic_write_parquet

        output_path = tmp_path / "movies.parquet"
        atomic_write_parquet(pd.DataFrame({"id": [1], "title": ["Old"]}), output_path)
        atomic_write_parquet(pd.DataFrame({"id": [2], "title": ["New"]}), output_path)

        result = pd.read_parquet(output_path)
        assert result.to_dict(orient="records") == [{"id": 2, "title": "New"}]

    def test_batch_invariants_reject_duplicate_ids(self):
        """assert_batch_invariants fails before bad serving artifacts are published."""
        from etl.pandas_etl import assert_batch_invariants

        df = pd.DataFrame(
            {
                "id": [1, 1],
                "title": ["A", "A duplicate"],
                "overview": ["Story", "Story again"],
            }
        )

        with pytest.raises(ValueError, match="duplicate movie ids"):
            assert_batch_invariants(df, stage="silver")

    def test_batch_invariants_reject_misaligned_movie_id_map(self):
        """The ETL contract catches row-order drift before artifacts are published."""
        from etl.pandas_etl import assert_batch_invariants

        df = pd.DataFrame(
            {
                "id": [1, 2],
                "title": ["Avatar", "Titanic"],
                "overview": ["Blue aliens", "Ship sinks"],
            }
        )
        vectors = np.random.rand(2, 8).astype(np.float32)

        with pytest.raises(ValueError, match="movie id map order"):
            assert_batch_invariants(df, vectors=vectors, movie_ids=np.array([2, 1]), stage="gold")

    # Removed faiss_search test as search logic is inside faiss index mostly,
    # and we removed index.search wrapper function (it was just idx.search).
    # Recommender tests cover search.


# ----- Recommender Tests -----


class TestRecommender:
    @pytest.fixture
    def mock_recommender(self, tmp_path):
        """Create recommender with mock data."""
        import faiss

        # Create mock data
        movies = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "title": ["Avatar", "Titanic", "Inception", "Interstellar", "Dunkirk"],
                "overview": ["blue aliens", "ship sinks", "dreams within dreams", "space travel", "war movie"],
                "genres": [
                    "[{'name': 'Action'}]",
                    "[{'name': 'Drama'}]",
                    "[{'name': 'Action'}]",
                    "[{'name': 'Sci-Fi'}]",
                    "[{'name': 'War'}]",
                ],
                "popularity": [10.0, 20.0, 15.0, 25.0, 5.0],
            }
        )
        movies.to_parquet(tmp_path / "movies_transformed.parquet")

        # Create random vectors
        vecs = np.random.rand(5, 384).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms

        np.save(tmp_path / "sbert_embeddings.npy", vecs)
        np.save(tmp_path / "movie_ids.npy", movies["id"].astype("int64").to_numpy())

        # As recommender uses SBERT now, we skip scaler/tfidf

        # Build Index
        idx = faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        faiss.write_index(idx, str(tmp_path / "faiss.index"))
        (tmp_path / "pipeline_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "test-run",
                    "serving_contract": {
                        "version": 1,
                        "movie_rows": 5,
                        "embedding_rows": 5,
                        "embedding_dimensions": 384,
                        "faiss_index_size": 5,
                        "movie_id_map_rows": 5,
                    },
                }
            ),
            encoding="utf-8",
        )

        return tmp_path

    def test_recommender_load(self, mock_recommender, monkeypatch):
        """Recommender loads all artifacts."""
        import backend.pipeline.recommender as rec

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
        import backend.pipeline.recommender as rec

        monkeypatch.setattr(rec, "MODELS_DIR", mock_recommender)
        monkeypatch.setattr(rec, "DATA_DIR", mock_recommender)

        r = rec.Recommender().load()
        results = r.search_movies("avatar")
        assert len(results) == 1
        assert results[0]["title"] == "Avatar"

    def test_search_movies_handles_minimal_catalog_columns(self):
        """Search should degrade gracefully when optional serving columns are absent."""
        import backend.pipeline.recommender as rec

        r = rec.Recommender()
        r._movies = pd.DataFrame(
            {
                "id": [1, 2],
                "title": ["Avatar", "Untitled Drama"],
            }
        )

        results = r.search_movies("avatar")

        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["title"] == "Avatar"
        assert "relevance" in results[0]

    def test_search_promotes_canonical_franchise_over_weak_duplicate_titles(self):
        """Exact-title duplicates should not bury high-signal franchise continuations."""
        import backend.pipeline.recommender as rec

        r = rec.Recommender()
        r._movies = pd.DataFrame(
            {
                "id": [19995, 1096978, 83533, 76600, 1132450],
                "title": [
                    "Avatar",
                    "Avatar",
                    "Avatar: Fire and Ash",
                    "Avatar: The Way of Water",
                    "Avataro Sentai Donbrothers",
                ],
                "overview": [""] * 5,
                "genres": ["Science Fiction"] * 5,
                "popularity": [30.47, 4.63, 210.38, 27.04, 4.76],
                "vote_count": [33849, 46, 2774, 14019, 13],
            }
        )

        results = r.search_movies("Avatar", limit=4)

        assert [item["id"] for item in results[:3]] == [19995, 83533, 76600]
        assert 1096978 not in [item["id"] for item in results[:3]]
        assert 1132450 not in [item["id"] for item in results[:3]]

    def test_search_normalizes_title_punctuation(self):
        """Search should match user punctuation to catalog punctuation variants."""
        import backend.pipeline.recommender as rec

        r = rec.Recommender()
        r._movies = pd.DataFrame(
            {
                "id": [10681, 22192],
                "title": ["WALL-E", "The Jonsson Gang Turns Up Again"],
                "overview": ["Robot love story", "Gang comedy"],
                "genres": ["Animation, Science Fiction", "Comedy"],
                "popularity": [28.0, 1.0],
                "vote_count": [19000, 100],
            }
        )

        hyphen_results = r.search_movies("WALL-E", limit=1)
        space_results = r.search_movies("wall e", limit=1)

        assert hyphen_results[0]["id"] == 10681
        assert space_results[0]["id"] == 10681

    def test_quality_gate_drops_low_rated_recommendation_drift(self):
        """MMR should not rescue weak low-rated candidates when enough strong matches exist."""
        import backend.pipeline.recommender as rec

        r = rec.Recommender()
        query = {"title": "Avatar", "genres": "Science Fiction, Action, Adventure"}
        candidates = [
            {
                "id": 1,
                "title": "Strong Sci-Fi Match",
                "genres": "Science Fiction, Adventure",
                "vote_average": 7.8,
                "vote_count": 5000,
                "retrieval_signals": {"semantic_twin": 0.7},
            },
            {
                "id": 2,
                "title": "Low Rated Drift",
                "genres": "Action, Adventure, Fantasy",
                "vote_average": 4.6,
                "vote_count": 4000,
                "retrieval_signals": {"semantic_twin": 0.65},
            },
            {
                "id": 3,
                "title": "Another Sci-Fi Match",
                "genres": "Science Fiction, Action",
                "vote_average": 6.8,
                "vote_count": 900,
                "retrieval_signals": {"semantic_twin": 0.64},
            },
        ]

        gated = r._quality_gate_item_recommendations(candidates, query, n=2)

        assert [item["title"] for item in gated] == ["Strong Sci-Fi Match", "Another Sci-Fi Match"]

    def test_recommend_by_id(self, mock_recommender, monkeypatch):
        """recommend_by_id returns similar movies."""
        import backend.pipeline.recommender as rec

        monkeypatch.setattr(rec, "MODELS_DIR", mock_recommender)
        monkeypatch.setattr(rec, "DATA_DIR", mock_recommender)

        r = rec.Recommender().load()
        recs = r.recommend_by_id(1, n=2)  # Avatar
        assert len(recs) >= 1
        assert all("similarity_score" in m for m in recs)

    def test_llm_rerank_is_disabled_by_default(self, mock_recommender, monkeypatch):
        """OpenRouter reranking must not run unless explicitly enabled."""
        import backend.pipeline.recommender as rec

        monkeypatch.setattr(rec, "MODELS_DIR", mock_recommender)
        monkeypatch.setattr(rec, "DATA_DIR", mock_recommender)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.delenv("NOVA_ENABLE_LLM_RERANK", raising=False)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("LLM reranking should be opt-in")

        monkeypatch.setattr(rec.Recommender, "_rerank_with_llm", fail_if_called)

        r = rec.Recommender().load()
        recs = r.recommend_by_id(1, n=2)

        assert len(recs) >= 1

    def test_mismatched_vector_artifacts_fall_back_to_sparse_content(self, mock_recommender, monkeypatch):
        """Serving must not trust FAISS vectors when row counts differ from the catalog."""
        import faiss

        import backend.pipeline.recommender as rec

        monkeypatch.setattr(rec, "MODELS_DIR", mock_recommender)
        monkeypatch.setattr(rec, "DATA_DIR", mock_recommender)

        bad_vecs = np.random.rand(6, 384).astype(np.float32)
        bad_vecs = bad_vecs / np.linalg.norm(bad_vecs, axis=1, keepdims=True)
        np.save(mock_recommender / "sbert_embeddings.npy", bad_vecs)
        bad_index = faiss.IndexFlatIP(bad_vecs.shape[1])
        bad_index.add(bad_vecs)
        faiss.write_index(bad_index, str(mock_recommender / "faiss.index"))

        r = rec.Recommender().load()
        recs = r.recommend_by_id(1, n=2)

        assert r._vectors is None
        assert r._index is None
        assert r._artifact_status["vector_artifacts_ready"] is False
        assert "vector" in r._artifact_status["disabled_reason"]
        assert len(recs) >= 1
        assert all(item["retrieval_stage"] == "content_sparse_fallback" for item in recs)

    def test_vector_artifacts_require_movie_id_map(self, mock_recommender, monkeypatch):
        """Serving must not trust row-position vectors without an explicit movie id map."""
        import backend.pipeline.recommender as rec

        monkeypatch.setattr(rec, "MODELS_DIR", mock_recommender)
        monkeypatch.setattr(rec, "DATA_DIR", mock_recommender)
        monkeypatch.delenv("NOVA_ALLOW_LEGACY_ROW_ALIGNED_VECTORS", raising=False)
        (mock_recommender / "movie_ids.npy").unlink()

        r = rec.Recommender().load()

        assert r._vectors is None
        assert r._index is None
        assert r._artifact_status["vector_artifacts_ready"] is False
        assert "movie_ids.npy is required" in r._artifact_status["disabled_reason"]
