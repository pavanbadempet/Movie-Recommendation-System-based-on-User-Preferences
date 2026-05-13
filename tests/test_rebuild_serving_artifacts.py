"""Tests for rebuilding aligned serving artifacts from the current catalog."""

import json

import numpy as np
import pandas as pd

from scripts.rebuild_serving_artifacts import rebuild_serving_artifacts


class FakeEncoder:
    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for idx, _ in enumerate(texts):
            vectors[idx, idx % 4] = 1.0
            vectors[idx, (idx + 1) % 4] = 0.5
        return vectors


def test_rebuild_serving_artifacts_writes_aligned_outputs(tmp_path):
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    movies_path = processed_dir / "movies_transformed.parquet"
    movies = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "title": ["Avatar", "Dune", "Interstellar"],
            "overview": ["Pandora conflict", "Desert prophecy", "Space survival"],
            "genres": ["Action, Sci-Fi", "Adventure, Sci-Fi", "Adventure, Sci-Fi"],
            "tags": ["avatar pandora ecology", "dune prophecy desert", "interstellar space survival"],
            "vote_count": [100, 200, 300],
            "content_quality_score": [0.8, 0.9, 0.85],
        }
    )
    movies.to_parquet(movies_path, index=False)

    result = rebuild_serving_artifacts(
        movies_path=movies_path,
        models_dir=models_dir,
        processed_dir=processed_dir,
        encoder=FakeEncoder(),
    )

    manifest = json.loads((models_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
    movie_ids = np.load(models_dir / "movie_ids.npy")
    vectors = np.load(models_dir / "sbert_embeddings.npy")

    assert result["row_count"] == 3
    assert movie_ids.tolist() == [10, 20, 30]
    assert vectors.shape == (3, 4)
    assert manifest["serving_contract"]["movie_rows"] == 3
    assert manifest["serving_contract"]["embedding_rows"] == 3
    assert manifest["serving_contract"]["faiss_index_size"] == 3
    assert manifest["serving_contract"]["movie_id_map_rows"] == 3
    assert (processed_dir / "semantic_twins.parquet").exists()
