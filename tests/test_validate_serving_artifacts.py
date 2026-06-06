"""Tests for Hugging Face serving artifact contract validation."""

import pytest

from scripts.validate_serving_artifacts import _validate_heavy_artifact_contract


def test_heavy_artifact_contract_rejects_metadata_only_manifest_by_default():
    manifest = {
        "serving_contract": {
            "movie_rows": 2,
            "movie_id_map_rows": 2,
        }
    }

    with pytest.raises(RuntimeError, match="missing embedding_rows"):
        _validate_heavy_artifact_contract(manifest, remote_sizes={})


def test_heavy_artifact_contract_can_allow_metadata_only_manifest():
    manifest = {
        "serving_contract": {
            "movie_rows": 2,
            "movie_id_map_rows": 2,
        }
    }

    checks = _validate_heavy_artifact_contract(
        manifest,
        remote_sizes={},
        allow_metadata_only_vectors=True,
    )

    assert checks["sbert_embeddings.npy"]["contract_rows"] is None
    assert checks["turbovec.tq"]["contract_rows"] is None


def test_heavy_artifact_contract_requires_checksum_when_rows_are_declared():
    manifest = {
        "serving_contract": {
            "embedding_rows": 2,
            "turbovec_index_size": 2,
        },
        "artifact_checksums": {},
    }

    with pytest.raises(RuntimeError, match="lacks checksum metadata"):
        _validate_heavy_artifact_contract(manifest, remote_sizes={})


def test_heavy_artifact_contract_rejects_remote_size_mismatch():
    manifest = {
        "serving_contract": {
            "embedding_rows": 2,
            "turbovec_index_size": 2,
        },
        "artifact_checksums": {
            "sbert_embeddings.npy": {"size_bytes": 100},
            "turbovec.tq": {"size_bytes": 200},
        },
    }

    with pytest.raises(RuntimeError, match="Hugging Face size"):
        _validate_heavy_artifact_contract(
            manifest,
            remote_sizes={
                "sbert_embeddings.npy": 101,
                "turbovec.tq": 200,
            },
        )


def test_heavy_artifact_contract_accepts_declared_rows_with_matching_sizes():
    manifest = {
        "serving_contract": {
            "embedding_rows": 2,
            "turbovec_index_size": 2,
        },
        "artifact_checksums": {
            "sbert_embeddings.npy": {"size_bytes": 100},
            "turbovec.tq": {"size_bytes": 200},
        },
    }

    checks = _validate_heavy_artifact_contract(
        manifest,
        remote_sizes={
            "sbert_embeddings.npy": 100,
            "turbovec.tq": 200,
        },
    )

    assert checks["sbert_embeddings.npy"]["contract_rows"] == 2
    assert checks["turbovec.tq"]["contract_rows"] == 2
