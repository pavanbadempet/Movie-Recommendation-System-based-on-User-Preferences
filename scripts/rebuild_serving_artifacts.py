"""Rebuild aligned vector serving artifacts from the current serving catalog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from etl.pandas_etl import (
    atomic_save_npy,
    atomic_write_turbovec_index,
    build_turbovec_index,
    movie_id_vector,
)
from scripts.backfill_serving_metadata_artifacts import build_backfill_artifacts, upload_artifacts

DEFAULT_MODEL_NAME = "all-mpnet-base-v2"


def encode_catalog_tags(
    movies: pd.DataFrame,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    encoder: Any | None = None,
) -> np.ndarray:
    """Encode serving tags into normalized float32 vectors."""
    if "tags" not in movies.columns:
        raise ValueError("movies_transformed.parquet must contain a tags column")

    tags = movies["tags"].fillna("").astype(str)
    if (tags.str.strip() == "").any():
        raise ValueError("movies_transformed.parquet contains empty tags; rerun the transform stage first")

    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_name)

    vectors = encoder.encode(
        tags.tolist(),
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def rebuild_serving_artifacts(
    movies_path: Path,
    models_dir: Path,
    processed_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Rebuild embeddings, TurboVec, ids, semantic twins, and manifest from one catalog snapshot."""
    movies_path = Path(movies_path)
    models_dir = Path(models_dir)
    processed_dir = Path(processed_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    movies = pd.read_parquet(movies_path)
    if "id" not in movies.columns:
        raise ValueError("movies_transformed.parquet must contain an id column")

    vectors = encode_catalog_tags(movies, model_name=model_name, batch_size=batch_size, encoder=encoder)
    movie_ids = movie_id_vector(movies)

    embeddings_path = models_dir / "sbert_embeddings.npy"
    movie_ids_path = models_dir / "movie_ids.npy"
    turbovec_path = models_dir / "turbovec.tq"

    atomic_save_npy(vectors, embeddings_path)
    atomic_save_npy(movie_ids, movie_ids_path)
    index = build_turbovec_index(vectors)
    atomic_write_turbovec_index(index, turbovec_path)

    artifacts = build_backfill_artifacts(
        movies=movies,
        movies_path=movies_path,
        output_dir=models_dir,
        semantic_output_dir=processed_dir,
        embeddings_path=embeddings_path,
        turbovec_path=turbovec_path,
    )
    return {
        **artifacts,
        "vector_dimensions": int(vectors.shape[1]) if len(vectors.shape) > 1 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies-path", type=Path, default=Path("data/processed/movies_transformed.parquet"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--hf-repo", default="pavanbadempet/movie-recs-models")
    parser.add_argument("--hf-repo-type", default="model")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    artifacts = rebuild_serving_artifacts(
        movies_path=args.movies_path,
        models_dir=args.models_dir,
        processed_dir=args.processed_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    if args.upload_to_hf:
        upload_artifacts(args.hf_repo, args.hf_repo_type, args.hf_token, artifacts)

    print(
        json.dumps(
            {
                "run_id": artifacts["run_id"],
                "run_date": artifacts["run_date"],
                "row_count": artifacts["row_count"],
                "vector_dimensions": artifacts["vector_dimensions"],
                "movie_id_sha256": artifacts["movie_id_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
