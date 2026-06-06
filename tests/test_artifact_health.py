"""Tests for cheap serving artifact diagnostics."""

import json

import numpy as np
import pandas as pd

from backend.serving.artifact_health import evaluate_artifact_health, movie_id_sha256


def test_artifact_health_reports_ready_when_catalog_ids_and_semantic_twins_align(tmp_path):
    movies = pd.DataFrame({"id": [10, 20], "title": ["A", "B"]})
    movies.to_parquet(tmp_path / "movies_transformed.parquet", index=False)
    pd.DataFrame({"id": [10, 20]}).to_parquet(tmp_path / "semantic_twins.parquet", index=False)
    (tmp_path / "semantic_twin_summary.json").write_text(
        json.dumps({"row_count": 2, "avg_confidence": 0.7}),
        encoding="utf-8",
    )
    movie_ids = np.array([10, 20], dtype=np.int64)
    np.save(tmp_path / "movie_ids.npy", movie_ids)
    (tmp_path / "sbert_embeddings.npy").write_bytes(b"embedding-bytes")
    (tmp_path / "faiss.index").write_bytes(b"faiss-bytes")
    (tmp_path / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "run_date": "2026-05-07",
                "serving_contract": {
                    "movie_rows": 2,
                    "movie_id_map_rows": 2,
                    "movie_id_sha256": movie_id_sha256(movie_ids),
                },
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_artifact_health(models_dir=tmp_path, data_dir=tmp_path)

    assert report["status"] == "ready"
    assert report["checks"]["catalog_vector_aligned"] is True
    assert report["checks"]["semantic_catalog_aligned"] is True
    assert report["row_counts"]["semantic_twins"] == 2


def test_artifact_health_degrades_when_semantic_twins_are_missing(tmp_path):
    movies = pd.DataFrame({"id": [10], "title": ["A"]})
    movies.to_parquet(tmp_path / "movies_transformed.parquet", index=False)

    report = evaluate_artifact_health(models_dir=tmp_path, data_dir=tmp_path)

    assert report["status"] == "degraded"
    assert report["checks"]["metadata_ready"] is True
    assert report["checks"]["semantic_files_ready"] is False
    assert report["recommendations"]
