"""Tests for batch semantic-twin artifact generation."""

import json

import pandas as pd

from etl.semantic_artifacts import build_semantic_twin_frame, semantic_twin_quality_gate, write_semantic_artifacts


def sample_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [19995, 27205],
            "title": ["Avatar", "Inception"],
            "overview": [
                "A marine travels to an alien moon and protects a native civilization.",
                "A thief enters dreams and builds a dangerous mind-bending mission.",
            ],
            "genres": ["Action, Adventure, Science Fiction", "Action, Science Fiction, Thriller"],
            "vote_count": [12000, 10000],
            "content_quality_score": [0.92, 0.90],
        }
    )


def test_build_semantic_twin_frame_preserves_catalog_order():
    movies = sample_movies()
    twins = build_semantic_twin_frame(movies)

    assert twins["id"].tolist() == [19995, 27205]
    assert "semantic_twin_json" in twins.columns
    assert "alien" in json.loads(twins.iloc[0]["concepts"])
    assert semantic_twin_quality_gate(movies, twins)["id_order_matches_catalog"] is True


def test_write_semantic_artifacts_outputs_parquet_and_summary(tmp_path):
    result = write_semantic_artifacts(sample_movies(), tmp_path, run_id="test-run", run_date="2026-05-07")

    assert result["semantic_twins_path"].exists()
    assert result["semantic_twin_summary_path"].exists()
    assert result["summary"]["row_count"] == 2
    assert result["quality_gate"]["semantic_twin_rows"] == 2
