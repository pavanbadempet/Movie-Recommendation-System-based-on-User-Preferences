import contextlib
import os
from pathlib import Path
import shutil
import time

import backend.serving.video_cache as vc


def test_cache_cleanup():
    # Setup temp cache dir for testing
    test_dir = Path("data/test_video_cache")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Patch the CACHE_DIR and MAX_CACHE_FILES
    orig_dir = vc.CACHE_DIR
    orig_max = vc.MAX_CACHE_FILES

    vc.CACHE_DIR = test_dir
    vc.MAX_CACHE_FILES = 3

    try:
        # Create some dummy mp4 files
        for i in range(5):
            f = test_dir / f"video_{i}.mp4"
            f.write_text("dummy content")
            # Set artificial modification times so they are ordered
            mtime = time.time() - (5 - i) * 10
            os.utime(f, (mtime, mtime))

        # Run cleanup
        vc.cleanup_cache()

        # Verify that only the 3 newest remain (video_2, video_3, video_4)
        remaining = [f.name for f in test_dir.glob("*.mp4")]
        assert len(remaining) == 3
        assert "video_0.mp4" not in remaining
        assert "video_1.mp4" not in remaining
        assert "video_2.mp4" in remaining
        assert "video_3.mp4" in remaining
        assert "video_4.mp4" in remaining

    finally:
        # Restore and cleanup
        vc.CACHE_DIR = orig_dir
        vc.MAX_CACHE_FILES = orig_max
        if test_dir.exists():
            with contextlib.suppress(Exception):
                shutil.rmtree(test_dir)


def test_video_stream_route(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app

    dummy_file = tmp_path / "dummy_video.mp4"
    dummy_file.write_text("dummy video file contents")

    # Mock the get_or_download_video function
    async def mock_get_or_download(youtube_id):
        if youtube_id == "abc123_DEF-":
            return dummy_file
        return None

    monkeypatch.setattr(vc, "get_or_download_video", mock_get_or_download)

    client = TestClient(app)

    # 1. Test success case
    response = client.get("/v1/videos/stream/abc123_DEF-")
    assert response.status_code == 200
    assert response.text == "dummy video file contents"

    # 2. Test fail case
    response = client.get("/v1/videos/stream/Zyx987-AB_c")
    assert response.status_code == 404

    # 3. Invalid YouTube IDs are rejected before cache path construction.
    response = client.get("/v1/videos/stream/bad..id")
    assert response.status_code == 400


def test_video_cache_status_route(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app

    # Patch CACHE_DIR to use our temp path
    monkeypatch.setattr(vc, "CACHE_DIR", tmp_path)

    # 1. Check status when file doesn't exist
    client = TestClient(app)
    response = client.get("/v1/videos/cache-status/abc123_DEF-")
    assert response.status_code == 200
    assert response.json() == {"youtube_id": "abc123_DEF-", "cached": False}

    # 2. Check status when file does exist
    video_file = tmp_path / "Zyx987-AB_c.mp4"
    video_file.write_text("mp4 data")

    response = client.get("/v1/videos/cache-status/Zyx987-AB_c")
    assert response.status_code == 200
    assert response.json() == {"youtube_id": "Zyx987-AB_c", "cached": True}

    response = client.get("/v1/videos/cache-status/bad..id")
    assert response.status_code == 400


def test_latest_movies_endpoint(monkeypatch):
    import backend.api.recommendation_routes as rr
    import backend.pipeline.recommender as rec

    # Mock get_rec() to return a mock recommender
    class MockRecommender:
        def get_all_movies(self):
            return [
                {
                    "id": 1,
                    "title": "Movie In",
                    "original_language": "hi",
                    "poster_path": "hi.jpg",
                    "release_date": "2026-01-01",
                },
                {
                    "id": 2,
                    "title": "Movie Us",
                    "original_language": "en",
                    "poster_path": "us.jpg",
                    "release_date": "2025-01-01",
                },
                {
                    "id": 3,
                    "title": "Movie Jp",
                    "original_language": "ja",
                    "poster_path": "jp.jpg",
                    "release_date": "2024-01-01",
                },
            ]

        def get_movie_by_id(self, movie_id):
            return None

    import backend.main as main
    monkeypatch.setattr(rec, "_recommender", MockRecommender())
    monkeypatch.setattr(main, "_recommender", MockRecommender())
    monkeypatch.setattr(rr, "_TMDB_KEY", None)  # Force fallback path

    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)

    # Test query parameter
    res1 = client.get("/movies/latest?country=IN")
    assert res1.status_code == 200
    movies = res1.json()
    assert len(movies) > 0
    # Since we passed country=IN, "Movie In" (language="hi") should be boosted to first place
    assert movies[0]["title"] == "Movie In"

    # Test request header cf-ipcountry
    res2 = client.get("/movies/latest", headers={"cf-ipcountry": "JP"})
    assert res2.status_code == 200
    movies2 = res2.json()
    assert movies2[0]["title"] == "Movie Jp"
