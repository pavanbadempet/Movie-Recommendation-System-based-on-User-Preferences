"""Validate the Hugging Face serving artifact contract without downloading huge vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pandas as pd

REQUIRED_FILES = {
    "movies_transformed.parquet",
    "sbert_embeddings.npy",
    "turbovec.tq",
    "movie_ids.npy",
    "pipeline_manifest.json",
    "semantic_twins.parquet",
    "semantic_twin_summary.json",
}

HEAVY_ARTIFACT_CONTRACTS = {
    "embedding_rows": "sbert_embeddings.npy",
    "turbovec_index_size": "turbovec.tq",
}


def movie_id_sha256(movie_ids: np.ndarray) -> str:
    ids = np.asarray(movie_ids, dtype=np.int64).astype("<i8", copy=False)
    return hashlib.sha256(ids.tobytes()).hexdigest()


def contract_value(manifest: dict, key: str):
    contract = manifest.get("serving_contract") or {}
    quality = manifest.get("quality") or {}
    return contract.get(key) or quality.get(key)


def _remote_size_map(info) -> dict[str, int | None]:
    return {sibling.rfilename: getattr(sibling, "size", None) for sibling in getattr(info, "siblings", [])}


def _validate_heavy_artifact_contract(
    manifest: dict,
    remote_sizes: dict[str, int | None],
    allow_metadata_only_vectors: bool = False,
) -> dict[str, dict]:
    """Check that declared vector/index row contracts are backed by artifact metadata."""
    checks = {}
    artifact_checksums = manifest.get("artifact_checksums") or {}

    for contract_key, filename in HEAVY_ARTIFACT_CONTRACTS.items():
        expected_rows = contract_value(manifest, contract_key)
        checksum_entry = artifact_checksums.get(filename)
        if expected_rows is None:
            if allow_metadata_only_vectors:
                checks[filename] = {"contract_rows": None, "checksum_present": bool(checksum_entry)}
                continue
            raise RuntimeError(
                f"manifest missing {contract_key}; run the full vector rebuild instead of metadata-only backfill"
            )

        if not checksum_entry:
            raise RuntimeError(f"manifest declares {contract_key} but lacks checksum metadata for {filename}")

        expected_size = checksum_entry.get("size_bytes")
        remote_size = remote_sizes.get(filename)
        if expected_size is not None and remote_size is not None and int(expected_size) != int(remote_size):
            raise RuntimeError(f"manifest size for {filename} ({expected_size}) != Hugging Face size ({remote_size})")

        checks[filename] = {
            "contract_rows": int(expected_rows),
            "checksum_present": True,
            "manifest_size_bytes": expected_size,
            "remote_size_bytes": remote_size,
        }

    return checks


def validate(
    repo_id: str,
    token: str | None = None,
    repo_type: str = "model",
    allow_metadata_only_vectors: bool = False,
) -> dict:
    api = HfApi(token=token)
    info = api.model_info(repo_id) if repo_type == "model" else api.repo_info(repo_id, repo_type=repo_type)
    remote_files = {sibling.rfilename for sibling in info.siblings}
    remote_sizes = _remote_size_map(info)
    missing = sorted(REQUIRED_FILES - remote_files)
    if missing:
        raise RuntimeError(f"Hugging Face repo is missing serving artifacts: {missing}")

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        manifest_path = hf_hub_download(
            repo_id=repo_id,
            filename="pipeline_manifest.json",
            repo_type=repo_type,
            token=token,
            cache_dir=cache_dir,
        )
        movie_ids_path = hf_hub_download(
            repo_id=repo_id,
            filename="movie_ids.npy",
            repo_type=repo_type,
            token=token,
            cache_dir=cache_dir,
        )
        movies_path = hf_hub_download(
            repo_id=repo_id,
            filename="movies_transformed.parquet",
            repo_type=repo_type,
            token=token,
            cache_dir=cache_dir,
        )
        semantic_twins_path = hf_hub_download(
            repo_id=repo_id,
            filename="semantic_twins.parquet",
            repo_type=repo_type,
            token=token,
            cache_dir=cache_dir,
        )

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        movie_ids = np.load(movie_ids_path)
        movies = pd.read_parquet(movies_path, columns=["id"])
        semantic_twins = pd.read_parquet(semantic_twins_path, columns=["id"])

    heavy_artifact_checks = _validate_heavy_artifact_contract(
        manifest,
        remote_sizes,
        allow_metadata_only_vectors=allow_metadata_only_vectors,
    )
    expected_movie_rows = contract_value(manifest, "movie_rows") or contract_value(manifest, "serving_rows")
    expected_embedding_rows = contract_value(manifest, "embedding_rows")
    expected_index_size = contract_value(manifest, "turbovec_index_size") or contract_value(
        manifest, "faiss_index_size"
    )
    expected_id_rows = contract_value(manifest, "movie_id_map_rows")
    expected_id_hash = contract_value(manifest, "movie_id_sha256")

    checks = {
        "movie_rows": len(movies),
        "movie_id_map_rows": len(movie_ids),
        "manifest_movie_rows": int(expected_movie_rows) if expected_movie_rows is not None else None,
        "manifest_embedding_rows": int(expected_embedding_rows) if expected_embedding_rows is not None else None,
        "manifest_turbovec_index_size": int(expected_index_size) if expected_index_size is not None else None,
        "manifest_movie_id_map_rows": int(expected_id_rows) if expected_id_rows is not None else None,
        "movie_id_sha256": movie_id_sha256(movie_ids),
        "manifest_movie_id_sha256": expected_id_hash,
        "semantic_twin_rows": len(semantic_twins),
        "heavy_artifacts": heavy_artifact_checks,
        "run_id": manifest.get("run_id"),
        "run_date": manifest.get("run_date"),
    }

    if len(movies) != len(movie_ids):
        raise RuntimeError(f"movies rows ({len(movies)}) != movie_ids rows ({len(movie_ids)})")
    if not np.array_equal(movies["id"].astype("int64").to_numpy(), movie_ids.astype("int64")):
        raise RuntimeError("movie_ids.npy order does not match movies_transformed.parquet id order")

    count_expectations = {
        "movie_rows": expected_movie_rows,
        "embedding_rows": expected_embedding_rows,
        "turbovec_index_size": expected_index_size,
        "movie_id_map_rows": expected_id_rows,
    }
    for name, expected in count_expectations.items():
        if expected is not None and int(expected) != len(movie_ids):
            raise RuntimeError(f"manifest {name} ({expected}) != serving rows ({len(movie_ids)})")

    if expected_id_hash and expected_id_hash != checks["movie_id_sha256"]:
        raise RuntimeError("manifest movie_id_sha256 does not match movie_ids.npy")
    if len(semantic_twins) != len(movies):
        raise RuntimeError(f"semantic twin rows ({len(semantic_twins)}) != movie rows ({len(movies)})")
    if not np.array_equal(semantic_twins["id"].astype("int64").to_numpy(), movie_ids.astype("int64")):
        raise RuntimeError("semantic_twins.parquet id order does not match movie_ids.npy")

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="pavanbadempet/movie-recs-models")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument(
        "--allow-metadata-only-vectors",
        action="store_true",
        help="Allow manifests that intentionally omit heavy vector/index row counts.",
    )
    args = parser.parse_args()

    result = validate(
        args.repo,
        token=args.token,
        repo_type=args.repo_type,
        allow_metadata_only_vectors=args.allow_metadata_only_vectors,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
