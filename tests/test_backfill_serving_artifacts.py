"""Tests for Hugging Face serving metadata artifact backfill."""

import json

import numpy as np
import pandas as pd

from scripts.backfill_serving_metadata_artifacts import build_backfill_artifacts


def test_build_backfill_artifacts_creates_alignment_contract(tmp_path):
    movies_path = tmp_path / "movies_transformed.parquet"
    movies = pd.DataFrame(
        {
            "id": [10, 20],
            "title": ["Avatar", "Dune"],
            "overview": [
                "A marine discovers an alien world and a conflict over nature.",
                "A desert planet epic about prophecy, politics, and survival.",
            ],
            "genres": ["Action, Adventure, Science Fiction", "Adventure, Science Fiction"],
            "vote_count": [100, 100],
            "content_quality_score": [0.8, 0.8],
        }
    )
    movies.to_parquet(movies_path, index=False)

    result = build_backfill_artifacts(movies, movies_path, tmp_path / "out")

    movie_ids = np.load(result["paths"]["movie_ids"])
    manifest = json.loads(result["paths"]["pipeline_manifest"].read_text(encoding="utf-8"))
    semantic_summary = json.loads(result["paths"]["semantic_twin_summary"].read_text(encoding="utf-8"))

    assert movie_ids.tolist() == [10, 20]
    assert manifest["serving_contract"]["movie_rows"] == 2
    assert manifest["serving_contract"]["movie_id_map_rows"] == 2
    assert manifest["serving_contract"]["movie_id_sha256"] == result["movie_id_sha256"]
    assert semantic_summary["row_count"] == 2
    assert result["paths"]["semantic_twins"].exists()

