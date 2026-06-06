"""Backfill small serving metadata artifacts from the current HF movie catalog.

This is a recovery path for cases where the heavy Kaggle embedding job already
left usable movies/embeddings/FAISS artifacts in Hugging Face, but small
alignment artifacts are missing.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from etl.semantic_artifacts import write_semantic_artifacts

DEFAULT_REPO = "pavanbadempet/movie-recs-models"
MODEL_NAME = "all-mpnet-base-v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def movie_id_sha256(movie_ids: np.ndarray) -> str:
    ids = np.asarray(movie_ids, dtype=np.int64).astype("<i8", copy=False)
    return hashlib.sha256(ids.tobytes()).hexdigest()


def describe_file(path: Path) -> dict[str, Any]:
    return {
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def describe_embedding_artifact(path: Path) -> dict[str, Any]:
    vectors = np.load(path, mmap_mode="r")
    shape = tuple(int(value) for value in vectors.shape)
    return {
        **describe_file(path),
        "rows": int(shape[0]) if shape else 0,
        "dimensions": int(shape[1]) if len(shape) > 1 else None,
    }


def describe_turbovec_artifact(path: Path) -> dict[str, Any]:
    from turbovec import TurboQuantIndex

    index = TurboQuantIndex.load(str(path))
    return {
        **describe_file(path),
        "rows": len(index),
    }


def load_movies_from_hf(repo_id: str, repo_type: str, token: str | None, cache_dir: Path) -> tuple[pd.DataFrame, Path]:
    movies_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="movies_transformed.parquet",
            repo_type=repo_type,
            token=token,
            cache_dir=cache_dir,
        )
    )
    return pd.read_parquet(movies_path), movies_path


def build_backfill_artifacts(
    movies: pd.DataFrame,
    movies_path: Path,
    output_dir: Path,
    *,
    semantic_output_dir: Path | None = None,
    embeddings_path: Path | None = None,
    turbovec_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_output_dir = semantic_output_dir or output_dir
    semantic_output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(UTC).replace(microsecond=0)
    run_id = f"backfill-{run_ts.strftime('%Y%m%dT%H%M%SZ')}"
    run_date = run_ts.date().isoformat()

    if "id" not in movies.columns:
        raise ValueError("movies_transformed.parquet must contain an id column")

    movie_ids = pd.to_numeric(movies["id"], errors="raise").astype("int64").to_numpy()
    movie_ids_path = output_dir / "movie_ids.npy"
    np.save(movie_ids_path, movie_ids)

    semantic = write_semantic_artifacts(movies, semantic_output_dir, run_id=run_id, run_date=run_date)
    semantic_twins_path = Path(semantic["semantic_twins_path"])
    semantic_summary_path = Path(semantic["semantic_twin_summary_path"])

    row_count = int(len(movie_ids))
    id_hash = movie_id_sha256(movie_ids)
    quality_report = {
        "run_id": run_id,
        "run_date": run_date,
        "source": "serving_metadata_backfill",
        "serving_rows": row_count,
        "movie_rows": row_count,
        "movie_id_map_rows": row_count,
        "movie_id_sha256": id_hash,
        "semantic_twin_rows": int(semantic["summary"]["row_count"]),
        "semantic_twin_avg_confidence": semantic["summary"]["avg_confidence"],
    }
    serving_contract: dict[str, Any] = {
        "version": 1,
        "model_name": MODEL_NAME,
        "movie_rows": row_count,
        "movie_id_map_rows": row_count,
        "movie_id_sha256": id_hash,
    }

    artifact_checksums = {
        "movies_transformed.parquet": describe_file(movies_path),
        movie_ids_path.name: describe_file(movie_ids_path),
        semantic_twins_path.name: describe_file(semantic_twins_path),
        semantic_summary_path.name: describe_file(semantic_summary_path),
    }

    if embeddings_path is not None:
        embedding_info = describe_embedding_artifact(embeddings_path)
        quality_report["embedding_rows"] = embedding_info["rows"]
        serving_contract["embedding_rows"] = embedding_info["rows"]
        if embedding_info["dimensions"] is not None:
            serving_contract["embedding_dimensions"] = embedding_info["dimensions"]
        artifact_checksums[embeddings_path.name] = {
            "sha256": embedding_info["sha256"],
            "size_bytes": embedding_info["size_bytes"],
        }

    if turbovec_path is not None:
        turbovec_info = describe_turbovec_artifact(turbovec_path)
        quality_report["turbovec_index_size"] = turbovec_info["rows"]
        serving_contract["turbovec_index_size"] = turbovec_info["rows"]
        artifact_checksums[turbovec_path.name] = {
            "sha256": turbovec_info["sha256"],
            "size_bytes": turbovec_info["size_bytes"],
        }

    quality_path = output_dir / "quality_report.json"
    quality_path.write_text(json.dumps(quality_report, indent=2, sort_keys=True), encoding="utf-8")
    artifact_checksums[quality_path.name] = describe_file(quality_path)

    manifest = {
        "run_id": run_id,
        "run_date": run_date,
        "generated_at": run_ts.isoformat().replace("+00:00", "Z"),
        "source": "serving_metadata_backfill",
        "model_name": MODEL_NAME,
        "artifacts": {
            "movies": "movies_transformed.parquet",
            "embeddings": "sbert_embeddings.npy",
            "turbovec_index": "turbovec.tq",
            "movie_ids": movie_ids_path.name,
            "quality_report": quality_path.name,
            "semantic_twins": semantic_twins_path.name,
            "semantic_twin_summary": semantic_summary_path.name,
        },
        "artifact_checksums": artifact_checksums,
        "serving_contract": serving_contract,
        "quality": quality_report,
        "semantic_twins": semantic["summary"],
    }
    manifest_path = output_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_date": run_date,
        "row_count": row_count,
        "movie_id_sha256": id_hash,
        "paths": {
            "movies": movies_path,
            "embeddings": embeddings_path,
            "turbovec_index": turbovec_path,
            "movie_ids": movie_ids_path,
            "semantic_twins": semantic_twins_path,
            "semantic_twin_summary": semantic_summary_path,
            "quality_report": quality_path,
            "pipeline_manifest": manifest_path,
        },
    }


def upload_artifacts(repo_id: str, repo_type: str, token: str | None, artifacts: dict[str, Any]) -> None:
    if not token:
        raise ValueError("HF token is required for upload")
    api = HfApi(token=token)
    for path in artifacts["paths"].values():
        if path is None:
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=Path(path).name,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
        )
        print(f"Uploaded {Path(path).name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--upload-to-hf", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        output_dir = args.output_dir or (Path(tmp) / "backfill")
        movies, movies_path = load_movies_from_hf(args.repo, args.repo_type, args.token, cache_dir)
        artifacts = build_backfill_artifacts(movies, movies_path, output_dir)
        if args.upload_to_hf:
            upload_artifacts(args.repo, args.repo_type, args.token, artifacts)
        print(json.dumps({k: v for k, v in artifacts.items() if k != "paths"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
