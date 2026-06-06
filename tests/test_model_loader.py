"""Tests for external artifact loader behavior."""

import json

import numpy as np


def test_ensure_model_files_accepts_small_valid_npy_fixture(tmp_path, monkeypatch):
    """Small test ID maps are valid .npy files and must not be overwritten."""
    import backend.models.model_loader as loader

    calls = []
    np.save(tmp_path / "movie_ids.npy", np.array([1, 2, 3], dtype=np.int64))
    monkeypatch.delenv("NOVA_DISABLE_MODEL_DOWNLOADS", raising=False)
    monkeypatch.setattr(loader, "download_file", lambda *args, **kwargs: calls.append(args) or False)

    result = loader.ensure_model_files(tmp_path, selected_files={"movie_ids.npy"})

    assert result["movie_ids.npy"] is True
    assert calls == []


def test_ensure_model_files_can_disable_external_downloads(tmp_path, monkeypatch):
    """Tests and offline runs can prevent accidental network artifact pulls."""
    import backend.models.model_loader as loader

    monkeypatch.setenv("NOVA_DISABLE_MODEL_DOWNLOADS", "1")

    def fail_download(*args, **kwargs):
        raise AssertionError("download_file should not be called")

    monkeypatch.setattr(loader, "download_file", fail_download)

    result = loader.ensure_model_files(tmp_path, selected_files={"movie_ids.npy"})

    assert result["movie_ids.npy"] is False


def test_low_memory_profile_still_downloads_serving_metadata(monkeypatch):
    """Lite hosts still need small alignment and semantic health artifacts."""
    import backend.models.model_loader as loader

    monkeypatch.setenv("NOVA_SERVING_PROFILE", "lite")

    selected = loader.default_artifacts_for_serving_profile()

    assert "movies_transformed.parquet" in selected
    assert "semantic_twins.parquet" in selected
    assert "semantic_twin_summary.json" in selected
    assert "movie_ids.npy" in selected
    assert "sbert_embeddings.npy" not in selected
    assert "turbovec.tq" not in selected


def test_ensure_model_files_redownloads_vectors_when_manifest_rows_do_not_match(tmp_path, monkeypatch):
    """Stale vector files must be refreshed when the manifest contract disagrees."""
    import backend.models.model_loader as loader

    np.save(tmp_path / "sbert_embeddings.npy", np.ones((2, 4), dtype=np.float32))
    (tmp_path / "pipeline_manifest.json").write_text(
        """
        {
          "serving_contract": {
            "embedding_rows": 3
          }
        }
        """,
        encoding="utf-8",
    )

    calls = []
    monkeypatch.delenv("NOVA_DISABLE_MODEL_DOWNLOADS", raising=False)
    monkeypatch.setattr(loader, "download_file", lambda *args, **kwargs: calls.append(args) or True)

    result = loader.ensure_model_files(tmp_path, selected_files={"sbert_embeddings.npy"})

    assert result["sbert_embeddings.npy"] is True
    assert len(calls) == 1


def test_ensure_model_files_redownloads_turbovec_when_manifest_rows_do_not_match(tmp_path, monkeypatch):
    """Stale TurboVec indices must be refreshed when the manifest contract disagrees."""
    from turbovec import TurboQuantIndex

    import backend.models.model_loader as loader

    index = TurboQuantIndex(8, bit_width=4)
    index.add(np.ones((2, 8), dtype=np.float32))
    index.write(str(tmp_path / "turbovec.tq"))
    (tmp_path / "pipeline_manifest.json").write_text(
        """
        {
          "serving_contract": {
            "turbovec_index_size": 3
          }
        }
        """,
        encoding="utf-8",
    )

    calls = []
    monkeypatch.delenv("NOVA_DISABLE_MODEL_DOWNLOADS", raising=False)
    monkeypatch.setattr(loader, "download_file", lambda *args, **kwargs: calls.append(args) or True)

    result = loader.ensure_model_files(tmp_path, selected_files={"turbovec.tq"})

    assert result["turbovec.tq"] is True
    assert len(calls) == 1


def test_force_refresh_downloads_when_manifest_checksum_does_not_match(tmp_path, monkeypatch):
    """Manifest checksum mismatches must not be bypassed by same-size remote files."""
    import backend.models.model_loader as loader

    local_ids = np.array([1, 2, 3], dtype=np.int64)
    np.save(tmp_path / "movie_ids.npy", local_ids)
    local_size = (tmp_path / "movie_ids.npy").stat().st_size
    (tmp_path / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "artifact_checksums": {
                    "movie_ids.npy": {
                        "size_bytes": local_size,
                        "sha256": "not-the-local-checksum",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    calls = []
    monkeypatch.delenv("NOVA_DISABLE_MODEL_DOWNLOADS", raising=False)
    monkeypatch.setenv("FORCE_MODEL_REFRESH", "1")
    monkeypatch.setattr(loader, "download_file", lambda *args, **kwargs: calls.append(args) or True)
    monkeypatch.setattr(
        loader.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("HEAD should not run on manifest mismatch")),
    )

    result = loader.ensure_model_files(tmp_path, selected_files={"movie_ids.npy"})

    assert result["movie_ids.npy"] is True
    assert len(calls) == 1
