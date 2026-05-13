"""Tests for external artifact loader behavior."""

import numpy as np


def test_ensure_model_files_accepts_small_valid_npy_fixture(tmp_path, monkeypatch):
    """Small test ID maps are valid .npy files and must not be overwritten."""
    import backend.model_loader as loader

    calls = []
    np.save(tmp_path / "movie_ids.npy", np.array([1, 2, 3], dtype=np.int64))
    monkeypatch.delenv("NOVA_DISABLE_MODEL_DOWNLOADS", raising=False)
    monkeypatch.setattr(loader, "download_file", lambda *args, **kwargs: calls.append(args) or False)

    result = loader.ensure_model_files(tmp_path, selected_files={"movie_ids.npy"})

    assert result["movie_ids.npy"] is True
    assert calls == []


def test_ensure_model_files_can_disable_external_downloads(tmp_path, monkeypatch):
    """Tests and offline runs can prevent accidental network artifact pulls."""
    import backend.model_loader as loader

    monkeypatch.setenv("NOVA_DISABLE_MODEL_DOWNLOADS", "1")

    def fail_download(*args, **kwargs):
        raise AssertionError("download_file should not be called")

    monkeypatch.setattr(loader, "download_file", fail_download)

    result = loader.ensure_model_files(tmp_path, selected_files={"movie_ids.npy"})

    assert result["movie_ids.npy"] is False

