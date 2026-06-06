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
    assert "embedding_rows" not in manifest["serving_contract"]
    assert "turbovec_index_size" not in manifest["serving_contract"]
    assert semantic_summary["row_count"] == 2
    assert result["paths"]["semantic_twins"].exists()


def test_build_backfill_artifacts_can_include_heavy_artifact_contract(tmp_path):
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
            "tags": ["avatar pandora ecology", "dune prophecy desert"],
            "vote_count": [100, 100],
            "content_quality_score": [0.8, 0.8],
        }
    )
    movies.to_parquet(movies_path, index=False)

    embeddings = np.zeros((2, 8), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    embeddings_path = tmp_path / "sbert_embeddings.npy"
    np.save(embeddings_path, embeddings)

    from turbovec import TurboQuantIndex
    index = TurboQuantIndex(embeddings.shape[1], bit_width=4)
    index.add(embeddings)
    turbovec_path = tmp_path / "turbovec.tq"
    index.write(str(turbovec_path))

    result = build_backfill_artifacts(
        movies,
        movies_path,
        tmp_path / "out",
        embeddings_path=embeddings_path,
        turbovec_path=turbovec_path,
    )

    manifest = json.loads(result["paths"]["pipeline_manifest"].read_text(encoding="utf-8"))

    assert manifest["serving_contract"]["embedding_rows"] == 2
    assert manifest["serving_contract"]["embedding_dimensions"] == 8
    assert manifest["serving_contract"]["turbovec_index_size"] == 2
    assert "sbert_embeddings.npy" in manifest["artifact_checksums"]
    assert "turbovec.tq" in manifest["artifact_checksums"]
