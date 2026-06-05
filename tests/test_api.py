"""
API integration tests for FastAPI backend
"""

import json
import sys
import uuid

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_artifacts(tmp_path, monkeypatch):
    """Set up mock model artifacts for testing."""
    import faiss

    # Mock movies
    movies = pd.DataFrame(
        {
            "id": [100, 200, 300],
            "title": ["Test Movie A", "Test Movie B", "Test Movie C"],
            "overview": ["Action thriller", "Comedy romance", "Sci-fi adventure"],
            "genres": ["Action", "Comedy", "Sci-Fi"],
            "vote_average": [7.5, 6.5, 8.0],
            "vote_count": [1000, 500, 2000],
            "popularity": [100.0, 50.0, 150.0],
            "release_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
            "poster_path": [None, None, None],
        }
    )
    movies.to_parquet(tmp_path / "movies_transformed.parquet")
    pd.DataFrame(
        {
            "id": movies["id"].astype("int64"),
            "semantic_twin_json": ["{}"] * len(movies),
        }
    ).to_parquet(tmp_path / "semantic_twins.parquet", index=False)
    (tmp_path / "semantic_twin_summary.json").write_text(
        json.dumps({"row_count": len(movies), "avg_confidence": 0.8}),
        encoding="utf-8",
    )

    # Mock vectors (MPNet style - 768 dims)
    vecs = np.random.rand(3, 768).astype(np.float32)

    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    np.save(tmp_path / "sbert_embeddings.npy", vecs)
    np.save(tmp_path / "movie_ids.npy", movies["id"].astype("int64").to_numpy())

    # FAISS index
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    faiss.write_index(idx, str(tmp_path / "faiss.index"))
    (tmp_path / "pipeline_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "serving_contract": {
                    "version": 1,
                    "movie_rows": 3,
                    "embedding_rows": 3,
                    "embedding_dimensions": 768,
                    "faiss_index_size": 3,
                    "movie_id_map_rows": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    # Patch paths
    import backend.recommender as rec

    monkeypatch.setattr(rec, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(rec, "DATA_DIR", tmp_path)
    monkeypatch.setenv("NOVA_USAGE_PATH", str(tmp_path / "api_usage.jsonl"))
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.delenv("NOVA_API_KEYS", raising=False)

    # Reset singleton
    rec._recommender = None
    if "backend.main" in sys.modules:
        sys.modules["backend.main"]._recommender = None

    return tmp_path


class TestHealthEndpoint:
    def test_health_returns_ok(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["movie_count"] == 3
        assert data["app_version"] == "2.0.0"

    def test_health_includes_app_commit_when_available(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_APP_COMMIT", "abcdef1234567890")

        from backend.main import app

        client = TestClient(app)
        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["app_commit"] == "abcdef123456"

    def test_app_metadata_prefers_revision_file_over_host_commit(self, tmp_path, monkeypatch):
        import backend.main as main

        revision_path = tmp_path / "REVISION"
        revision_path.write_text("sourceabcdef1234567890", encoding="utf-8")
        monkeypatch.setattr(main, "REVISION_FILE", revision_path)
        monkeypatch.delenv("NOVA_APP_COMMIT", raising=False)
        monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
        monkeypatch.delenv("SOURCE_VERSION", raising=False)
        monkeypatch.delenv("GITHUB_SHA", raising=False)
        monkeypatch.setenv("SOURCE_VERSION", "hostabcdef1234567890")
        monkeypatch.setenv("COMMIT_SHA", "spaceabcdef1234567890")

        metadata = main.app_metadata()

        assert metadata["commit"] == "sourceabcdef"
        assert metadata["source"] == "REVISION"

    def test_health_without_recommender_load_reports_catalog_count(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_HEALTH_LOAD_RECOMMENDER", "false")

        from backend.main import app

        client = TestClient(app)

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["movie_count"] == 3


class TestPlatformEndpoint:
    def test_platform_context_public_demo(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/platform/context")

        assert resp.status_code == 200
        assert resp.json()["mode"] == "public-demo"
        assert resp.json()["tenant_id"] == "demo-media-co"

    def test_platform_context_requires_key_when_configured(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_API_KEYS", "secret-key:acme:main:free")

        from backend.main import app

        client = TestClient(app)

        missing_resp = client.get("/v1/platform/context")
        assert missing_resp.status_code == 401

        valid_resp = client.get("/v1/platform/context", headers={"X-Nova-API-Key": "secret-key"})
        assert valid_resp.status_code == 200
        assert valid_resp.json()["tenant_id"] == "acme"
        assert valid_resp.json()["catalog_id"] == "main"
        assert valid_resp.json()["mode"] == "authenticated"

    def test_platform_status(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/platform/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["app"]["version"] == "2.0.0"
        assert "personalization_v2" in data["capabilities"]
        assert "recommendation_benchmark" in data["capabilities"]
        assert "event_store" in data

    def test_platform_readiness_reports_component_status(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/platform/readiness", params={"k": 3})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in {"ready", "degraded"}
        assert data["strict"] is False
        assert data["summary"]["component_count"] >= 6
        components = {component["name"]: component for component in data["components"]}
        assert components["catalog"]["status"] == "ok"
        assert components["artifact_health"]["status"] == "ok"
        assert components["vector_serving"]["status"] == "ok"
        assert components["recommendation_smoke"]["status"] == "ok"

    def test_platform_readiness_strict_degrades_when_benchmark_cache_is_missing(self, mock_artifacts):
        import backend.main as main
        from backend.main import app

        main._semantic_benchmark_cache.clear()
        main._recommendation_benchmark_cache.clear()

        client = TestClient(app)
        resp = client.get("/v1/platform/readiness", params={"strict": True, "k": 3})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        components = {component["name"]: component for component in data["components"]}
        assert components["semantic_benchmark_cache"]["status"] == "warming"
        assert components["recommendation_benchmark_cache"]["status"] == "warming"

    def test_platform_readiness_starts_background_benchmark_warmers(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_ASYNC_EVALUATION_CACHE", "true")

        import backend.main as main
        from backend.main import app

        started = []
        main._semantic_benchmark_cache.clear()
        main._recommendation_benchmark_cache.clear()
        monkeypatch.setattr(main, "_start_background_semantic_benchmark", lambda k: started.append(("semantic", k)))
        monkeypatch.setattr(
            main, "_start_background_recommendation_benchmark", lambda k: started.append(("recommendation", k))
        )

        client = TestClient(app)
        resp = client.get("/v1/platform/readiness", params={"strict": True, "k": 3})

        assert resp.status_code == 200
        assert ("semantic", 3) in started
        assert ("recommendation", 3) in started

    def test_platform_readiness_can_proxy_to_remote_service(self, mock_artifacts, monkeypatch):
        import backend.main as main
        from backend.main import app
        from backend.remote_recommender import RemoteResponse

        async def fake_remote_get_json(path, params=None, context=None):
            assert path == "/v1/platform/readiness"
            assert params["strict"] is True
            return RemoteResponse(
                status_code=200,
                payload={"status": "ready", "remote": True, "components": []},
            )

        def fail_local_load():
            raise AssertionError("Gateway readiness should proxy to the vector service")

        monkeypatch.setattr(main, "remote_get_json", fake_remote_get_json)
        monkeypatch.setattr(main, "get_rec", fail_local_load)

        client = TestClient(app)
        resp = client.get("/v1/platform/readiness", params={"strict": True})

        assert resp.status_code == 200
        assert resp.json()["remote"] is True

    def test_platform_status_can_proxy_to_remote_service(self, mock_artifacts, monkeypatch):
        import backend.main as main
        from backend.main import app
        from backend.remote_recommender import RemoteResponse

        async def fake_remote_get_json(path, params=None, context=None):
            assert path == "/v1/platform/status"
            assert params is None
            return RemoteResponse(
                status_code=200,
                payload={"status": "ready", "remote": True, "movie_count": 75247},
            )

        def fail_local_load():
            raise AssertionError("Gateway status should proxy to the vector service")

        monkeypatch.setattr(main, "remote_get_json", fake_remote_get_json)
        monkeypatch.setattr(main, "get_rec", fail_local_load)

        client = TestClient(app)
        resp = client.get("/v1/platform/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["remote"] is True
        assert data["movie_count"] == 75247
        assert data["gateway"]["status"] == "ready"
        assert data["gateway"]["remote_recommender"]["configured"] is False

    @pytest.mark.parametrize(
        ("request_url", "expected_path", "expected_params", "payload"),
        [
            (
                "/v1/evaluation/recommendations?sample_size=2&k=3",
                "/v1/evaluation/recommendations",
                {"sample_size": 2, "k": 3},
                {"status": "ready", "remote": True},
            ),
            (
                "/v1/evaluation/semantic-benchmark?k=3",
                "/v1/evaluation/semantic-benchmark",
                {"k": 3},
                {"status": "ready", "remote": True},
            ),
            (
                "/v1/ranker/status",
                "/v1/ranker/status",
                None,
                {"available": True, "remote": True},
            ),
            (
                "/v1/semantic-twins/id/100",
                "/v1/semantic-twins/id/100",
                None,
                {"id": 100, "remote": True},
            ),
            (
                "/v1/recommendations/user/setup-test?n=3",
                "/v1/recommendations/user/setup-test",
                {"n": 3},
                [
                    {
                        "id": 100,
                        "title": "Test Movie A",
                        "overview": "Action thriller",
                        "genres": "Action",
                        "vote_average": 7.5,
                        "vote_count": 1000,
                        "popularity": 100.0,
                        "release_date": "2020-01-01",
                    }
                ],
            ),
        ],
    )
    def test_gateway_heavy_endpoints_proxy_to_remote_service(
        self,
        mock_artifacts,
        monkeypatch,
        request_url,
        expected_path,
        expected_params,
        payload,
    ):
        import backend.main as main
        from backend.main import app
        from backend.remote_recommender import RemoteResponse

        calls = []

        async def fake_remote_get_json(path, params=None, context=None):
            calls.append((path, params))
            assert path == expected_path
            if expected_params is None:
                assert params is None
            else:
                for key, value in expected_params.items():
                    assert params[key] == value
            return RemoteResponse(status_code=200, payload=payload)

        def fail_local_load():
            raise AssertionError("Gateway endpoint should proxy to the vector service")

        monkeypatch.setattr(main, "remote_get_json", fake_remote_get_json)
        monkeypatch.setattr(main, "get_rec", fail_local_load)
        monkeypatch.setattr(main, "record_usage", lambda *args, **kwargs: None)
        monkeypatch.setattr(main, "record_recommendation_events", lambda *args, **kwargs: "test-request")

        client = TestClient(app)
        resp = client.get(request_url)

        assert resp.status_code == 200
        assert calls

    def test_required_remote_recommender_fails_fast_without_local_fallback(self, mock_artifacts, monkeypatch):
        import backend.main as main
        from backend.main import app

        async def fake_remote_get_json(path, params=None, context=None):
            return None

        def fail_local_load():
            raise AssertionError("Render gateway should not fall back to local recommendation serving")

        monkeypatch.setenv("NOVA_REMOTE_RECOMMENDER_REQUIRED", "true")
        monkeypatch.setattr(main, "remote_recommender_url", lambda: "https://remote.example")
        monkeypatch.setattr(main, "remote_get_json", fake_remote_get_json)
        monkeypatch.setattr(main, "get_rec", fail_local_load)

        client = TestClient(app)
        resp = client.get("/v1/recommendations/id/100", params={"n": 2})

        assert resp.status_code == 503
        assert resp.json()["detail"] == "Remote recommender unavailable"


class TestCorsPolicy:
    def test_github_pages_origin_is_allowed_by_default(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.options(
            "/v1/search",
            headers={
                "Origin": "https://pavanbadempet.github.io",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "https://pavanbadempet.github.io"

    def test_malicious_origin_is_blocked_by_default(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.options(
            "/v1/search",
            headers={
                "Origin": "https://malicious-domain.vercel.app",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert resp.status_code == 400
        assert "access-control-allow-origin" not in resp.headers

    def test_local_vite_dev_origin_is_allowed_by_default(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.options(
            "/v1/search",
            headers={
                "Origin": "http://127.0.0.1:5174",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"


class TestAuthEndpoints:
    def test_register_and_login_requires_valid_password(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        username = f"auth-user-{uuid.uuid4().hex[:8]}"
        password = "verysecure123"
        register_resp = client.post(
            "/v1/auth/register",
            json={"username": username, "password": password},
        )
        assert register_resp.status_code == 200

        bad_login_resp = client.post(
            "/v1/auth/token",
            data={"username": username, "password": "wrong-password"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert bad_login_resp.status_code == 401

        good_login_resp = client.post(
            "/v1/auth/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert good_login_resp.status_code == 200
        payload = good_login_resp.json()
        assert payload["token_type"] == "bearer"
        assert payload["access_token"]

    def test_register_rejects_short_password(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.post(
            "/v1/auth/register",
            json={"username": "short-pass-user", "password": "1234567"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Password must be at least 8 characters"


class TestSearchEndpoint:
    def test_movie_titles_limit_for_readiness_probe(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/movies/titles", params={"limit": 1})

        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_movie_titles_accepts_full_catalog_limit(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/movies/titles", params={"limit": 100000})

        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_search_finds_movie(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/search", params={"q": "Test Movie A"})
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert results[0]["title"] == "Test Movie A"

    def test_search_empty_returns_empty(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/search", params={"q": "xyz123nonexistent"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_v1_search_alias(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/v1/search", params={"q": "Test Movie A"})
        assert resp.status_code == 200
        assert resp.json()[0]["title"] == "Test Movie A"

    def test_v1_search_sanitizes_nan_optional_fields(self, monkeypatch):
        import backend.main as main
        from backend.main import app
        from backend.recommender import Recommender

        rec = Recommender()
        rec._movies = pd.DataFrame(
            {
                "id": [19995],
                "title": ["Avatar"],
                "overview": ["Alien world adventure"],
                "genres": [np.nan],
                "poster_path": [np.nan],
                "popularity": [100.0],
            }
        )
        monkeypatch.setattr(main, "get_rec", lambda: rec)

        client = TestClient(app)
        resp = client.get("/v1/search", params={"q": "avatar"})

        assert resp.status_code == 200
        assert resp.json()[0]["genres"] is None
        assert resp.json()[0]["poster_path"] is None

    def test_v1_ai_search_uses_hybrid_retrieval(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_ENABLE_DENSE_QUERY", "false")

        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/search/ai", params={"q": "sci fi adventure", "top_k": 2})

        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert len(results) <= 2
        assert results[0]["retrieval_stage"] == "sparse_metadata"
        assert "retrieval_signals" in results[0]
        assert results[0]["explanation"]

    def test_v1_ai_search_can_proxy_to_remote_recommender(self, mock_artifacts, monkeypatch):
        import backend.main as main
        from backend.main import app
        from backend.remote_recommender import RemoteResponse

        async def fake_remote_get_json(path, params=None, context=None):
            assert path == "/v1/search/ai"
            assert params["q"] == "space"
            assert params["top_k"] == 1
            return RemoteResponse(
                status_code=200,
                payload=[
                    {
                        "id": 300,
                        "title": "Remote Space Movie",
                        "overview": "Returned by the model service",
                        "retrieval_stage": "remote_hybrid",
                        "retrieval_signals": {"remote": True},
                    }
                ],
            )

        def fail_local_load():
            raise AssertionError("Render gateway should not load the local recommender")

        monkeypatch.setattr(main, "remote_get_json", fake_remote_get_json)
        monkeypatch.setattr(main, "get_rec", fail_local_load)

        client = TestClient(app)
        resp = client.get("/v1/search/ai", params={"q": "space", "top_k": 1})

        assert resp.status_code == 200
        results = resp.json()
        assert results[0]["title"] == "Remote Space Movie"
        assert results[0]["retrieval_stage"] == "remote_hybrid"

    def test_semantic_twin_endpoint_returns_structured_catalog_evidence(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/semantic-twins/id/300")

        assert resp.status_code == 200
        data = resp.json()
        assert data["item_id"] == 300
        assert "concepts" in data
        assert data["generated_by"]["llm_in_hot_path"] is False

    def test_semantic_benchmark_endpoint_is_available(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/evaluation/semantic-benchmark", params={"k": 3})

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "case_count" in data

    def test_search_benchmark_endpoint_is_available(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/evaluation/search-benchmark", params={"k": 3})

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "case_count" in data
        assert "top1_hit_rate" in data["metrics"]

    def test_recommendation_benchmark_endpoint_is_available(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/evaluation/recommendation-benchmark", params={"k": 3})

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "case_count" in data
        assert "case_pass_rate" in data["metrics"]

    def test_semantic_benchmark_async_cache_returns_warming(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_ASYNC_EVALUATION_CACHE", "true")

        import backend.main as main
        from backend.main import app

        main._semantic_benchmark_cache.clear()
        monkeypatch.setattr(main, "_start_background_semantic_benchmark", lambda k: None)

        client = TestClient(app)
        resp = client.get("/v1/evaluation/semantic-benchmark", params={"k": 3})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "warming"
        assert data["k"] == 3

    def test_semantic_benchmark_sync_bypasses_async_warming(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_ASYNC_EVALUATION_CACHE", "true")

        import backend.main as main
        from backend.main import app

        main._semantic_benchmark_cache.clear()
        monkeypatch.setattr(
            main,
            "_start_background_semantic_benchmark",
            lambda k: (_ for _ in ()).throw(AssertionError("sync benchmark should not start async warmer")),
        )

        client = TestClient(app)
        resp = client.get("/v1/evaluation/semantic-benchmark", params={"k": 3, "sync": True})

        assert resp.status_code == 200
        assert resp.json()["status"] != "warming"

    def test_recommendation_benchmark_async_cache_returns_warming(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_ASYNC_EVALUATION_CACHE", "true")

        import backend.main as main
        from backend.main import app

        main._recommendation_benchmark_cache.clear()
        monkeypatch.setattr(main, "_start_background_recommendation_benchmark", lambda k: None)

        client = TestClient(app)
        resp = client.get("/v1/evaluation/recommendation-benchmark", params={"k": 3})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "warming"
        assert data["k"] == 3

    def test_recommendation_benchmark_sync_bypasses_async_warming(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_ASYNC_EVALUATION_CACHE", "true")

        import backend.main as main
        from backend.main import app

        main._recommendation_benchmark_cache.clear()
        monkeypatch.setattr(
            main,
            "_start_background_recommendation_benchmark",
            lambda k: (_ for _ in ()).throw(AssertionError("sync benchmark should not start async warmer")),
        )

        client = TestClient(app)
        resp = client.get("/v1/evaluation/recommendation-benchmark", params={"k": 3, "sync": True})

        assert resp.status_code == 200
        assert resp.json()["status"] != "warming"

    def test_artifact_health_endpoint_reports_alignment(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/artifacts/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["checks"]["catalog_vector_aligned"] is True
        assert data["checks"]["semantic_catalog_aligned"] is True

    def test_artifact_reload_is_disabled_without_admin_token(self, mock_artifacts, monkeypatch):
        monkeypatch.delenv("NOVA_ADMIN_TOKEN", raising=False)

        from backend.main import app

        client = TestClient(app)
        resp = client.post("/v1/artifacts/reload", params={"force_download": False})

        assert resp.status_code == 404

    def test_artifact_reload_requires_valid_admin_token(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_ADMIN_TOKEN", "admin-secret")

        from backend.main import app

        client = TestClient(app)
        resp = client.post(
            "/v1/artifacts/reload",
            params={"force_download": False},
            headers={"X-Nova-Admin-Token": "wrong"},
        )

        assert resp.status_code == 401

    def test_artifact_reload_swaps_loaded_recommender(self, mock_artifacts, monkeypatch):
        monkeypatch.setenv("NOVA_ADMIN_TOKEN", "admin-secret")

        import backend.main as main
        from backend.main import app

        client = TestClient(app)
        resp = client.post(
            "/v1/artifacts/reload",
            params={"force_download": False, "load": True},
            headers={"X-Nova-Admin-Token": "admin-secret"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reloaded"
        assert data["artifact_health"]["status"] == "ready"
        assert data["lineage"]["movie_count"] == 3
        assert main._recommender is not None


class TestMoviesEndpoint:
    def test_movies_returns_list(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/movies", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        movies = resp.json()
        assert len(movies) == 2

    def test_movies_pagination(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)
        resp1 = client.get("/movies", params={"limit": 1, "offset": 0})
        resp2 = client.get("/movies", params={"limit": 1, "offset": 1})

        m1 = resp1.json()[0]
        m2 = resp2.json()[0]
        assert m1["id"] != m2["id"]


class TestEventsEndpoint:
    def test_record_event_and_read_features(self, tmp_path, monkeypatch, mock_artifacts):
        monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))

        from backend.main import app

        client = TestClient(app)
        resp = client.post(
            "/events",
            json={"event_type": "view", "movie_id": 100, "user_id": "test-user"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert "event_id" in data

        features_resp = client.get("/events/features")
        assert features_resp.status_code == 200
        features = features_resp.json()
        assert features["total_events"] == 1
        assert features["trending_movies"]["100"]["views"] == 1

    def test_v1_event_alias_accepts_content_id(self, tmp_path, monkeypatch, mock_artifacts):
        monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))

        from backend.main import app

        client = TestClient(app)
        resp = client.post(
            "/v1/events",
            json={
                "event_type": "view",
                "tenant_id": "ott-startup",
                "catalog_id": "short-films",
                "content_id": "content-123",
            },
        )

        assert resp.status_code == 200
        features_resp = client.get("/v1/events/features")
        assert features_resp.status_code == 200
        assert features_resp.json()["trending_movies"]["content-123"]["tenant_id"] == "ott-startup"

    def test_event_rejects_cross_tenant_payload_when_api_key_configured(self, tmp_path, monkeypatch, mock_artifacts):
        monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))
        monkeypatch.setenv("NOVA_API_KEYS", "secret-key:acme:main:free")

        from backend.main import app

        client = TestClient(app)
        resp = client.post(
            "/v1/events",
            headers={"X-Nova-API-Key": "secret-key"},
            json={
                "event_type": "view",
                "tenant_id": "other",
                "catalog_id": "main",
                "content_id": "content-123",
            },
        )

        assert resp.status_code == 403

    def test_record_event_requires_movie_id_for_movie_events(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))

        from backend.main import app

        client = TestClient(app)
        resp = client.post("/events", json={"event_type": "click"})

        assert resp.status_code == 400
        assert resp.json()["detail"] == "movie_id or content_id is required for content events"

    def test_record_event_validates_rating_range(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))

        from backend.main import app

        client = TestClient(app)
        resp = client.post(
            "/events",
            json={"event_type": "rating", "movie_id": 100, "rating": 6},
        )

        assert resp.status_code == 400
        assert resp.json()["detail"] == "rating must be between 1 and 5"


class TestRecommendEndpoints:
    def test_recommend_by_id(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/recommend/id/100", params={"n": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"]
        assert "query_movie" in data
        assert "recommendations" in data
        assert data["query_movie"]["id"] == 100

    def test_recommend_by_id_writes_request_and_impression_events(self, tmp_path, monkeypatch, mock_artifacts):
        event_path = tmp_path / "recommendation_events.jsonl"
        monkeypatch.setenv("EVENT_LOG_PATH", str(event_path))

        from backend.events import iter_events
        from backend.main import app

        client = TestClient(app)
        resp = client.get(
            "/v1/recommendations/id/100",
            params={
                "n": 2,
                "request_id": "req-test-1",
                "user_id": "user-1",
                "session_id": "session-1",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["request_id"] == "req-test-1"

        events = list(iter_events(event_path))
        assert [event["event_type"] for event in events].count("recommendation_request") == 1
        assert [event["event_type"] for event in events].count("recommendation_impression") == 2
        request_event = next(event for event in events if event["event_type"] == "recommendation_request")
        assert request_event["request_id"] == "req-test-1"
        assert request_event["user_id"] == "user-1"
        assert request_event["session_id"] == "session-1"
        assert request_event["metadata"]["endpoint"] == "recommendations.id"
        assert request_event["metadata"]["query_movie"]["id"] == 100
        assert request_event["metadata"]["candidate_ids"]

        impression_events = [event for event in events if event["event_type"] == "recommendation_impression"]
        assert [event["metadata"]["rank"] for event in impression_events] == [1, 2]
        assert all(event["metadata"]["retrieval_stage"] for event in impression_events)

        analytics_resp = client.get("/v1/events/recommendation-analytics")
        assert analytics_resp.status_code == 200
        analytics = analytics_resp.json()
        assert analytics["request_count"] == 1
        assert analytics["impression_count"] == 2
        assert analytics["top_seed_movies"][0]["movie_id"] == "100"

    def test_recommend_by_id_not_found(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/recommend/id/999999")
        assert resp.status_code == 404

    def test_recommendation_diagnostics_exposes_ranking_evidence(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/diagnostics/recommendations/100", params={"n": 2})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["query_movie"]["id"] == 100
        assert data["lineage"]["serving_path"] == "local"
        assert data["diagnostics"]["result_count"] == 2
        assert data["diagnostics"]["stage_distribution"]
        assert len(data["recommendations"]) == 2
        assert "retrieval_stage" in data["recommendations"][0]

    def test_recommendation_diagnostics_includes_labeled_case_summary(self, mock_artifacts, monkeypatch):
        import backend.main as main
        from backend.main import app

        class FakeRec:
            _artifact_status = {"vector_artifacts_ready": True}
            _artifact_manifest = {}
            _learned_ranker = None

            def get_movie_by_id(self, movie_id):
                if movie_id == 100:
                    return {"id": 100, "title": "Seed Movie", "genres": "Drama"}
                return None

            def recommend_by_id(self, movie_id, n=10):
                return [
                    {
                        "id": 200,
                        "title": "Good Match",
                        "similarity_score": 0.91,
                        "retrieval_stage": "vector_semantic_ranked",
                        "explanation": ["shared themes"],
                    }
                ][:n]

        monkeypatch.setattr(main, "get_rec", lambda: FakeRec())
        monkeypatch.setattr(
            main,
            "load_recommendation_benchmark",
            lambda: [
                {
                    "case_id": "seed_case",
                    "seed": {"id": 100, "title": "Seed Movie"},
                    "min_good_hits": 1,
                    "good_matches": [{"id": 200, "title": "Good Match"}],
                    "bad_matches": [{"id": 300, "title": "Bad Drift"}],
                }
            ],
        )

        client = TestClient(app)
        resp = client.get("/v1/diagnostics/recommendations/100", params={"n": 1})

        assert resp.status_code == 200
        data = resp.json()
        assert data["diagnostics"]["benchmark_case_available"] is True
        assert data["diagnostics"]["benchmark_case_passed"] is True
        assert data["benchmark_case"]["case_id"] == "seed_case"
        assert data["benchmark_case"]["good_hit_count"] == 1

    def test_recommend_by_title(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/recommend/title/Test Movie B", params={"n": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query_movie"]["title"] == "Test Movie B"

    def test_recommend_for_user_from_events(self, tmp_path, monkeypatch, mock_artifacts):
        monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))

        from backend.main import app

        client = TestClient(app)
        event_resp = client.post(
            "/v1/events",
            json={"event_type": "view", "movie_id": 100, "user_id": "user-1"},
        )
        assert event_resp.status_code == 200

        resp = client.get("/v1/recommendations/user/user-1", params={"top_k": 1})

        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert results[0]["retrieval_stage"].startswith("personalized_v2")
        assert "variant" in results[0]["retrieval_signals"]


class TestEvaluationEndpoint:
    def test_recommendation_quality_report(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/evaluation/recommendations", params={"sample_size": 2, "k": 2})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["movie_count"] == 3
        assert data["vectors"]["available"] is True
        assert data["vectors"]["index_rows_match_catalog"] is True
        assert data["recommendations"]["available"] is True

    def test_ranker_status_without_artifact(self, mock_artifacts, monkeypatch):
        import backend.main as main
        from backend.main import app

        def fail_local_load():
            raise AssertionError("Ranker status should not load the recommender")

        main._recommender = None
        monkeypatch.setattr(main, "get_rec", fail_local_load)
        client = TestClient(app)

        resp = client.get("/v1/ranker/status")

        assert resp.status_code == 200
        assert resp.json()["available"] is False


class TestExperimentsEndpoint:
    def test_experiment_assignment_is_available(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        resp = client.get("/v1/experiments/assignment", params={"user_id": "user-1"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["experiment"]
        assert data["variant"]

    def test_experiment_metrics_from_events(self, tmp_path, monkeypatch, mock_artifacts):
        monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))

        from backend.main import app

        client = TestClient(app)
        event_resp = client.post(
            "/v1/events",
            json={
                "event_type": "recommendation_impression",
                "movie_id": 100,
                "metadata": {"experiment": "ranker", "variant": "control"},
            },
        )
        assert event_resp.status_code == 200

        resp = client.get("/v1/experiments/metrics")

        assert resp.status_code == 200
        rows = resp.json()["experiments"]
        assert rows[0]["experiment"] == "ranker"
        assert rows[0]["impressions"] == 1


class TestCatalogOnboardingEndpoints:
    def test_catalog_preview_profiles_csv(self, mock_artifacts):
        from backend.main import app

        client = TestClient(app)

        csv_text = (
            "id,title,overview,genres\n"
            "1,Arrival,A linguist communicates with alien visitors in a tense science fiction story,Sci-Fi\n"
            "2,,Too short,Drama\n"
        )
        resp = client.post(
            "/v1/catalog/preview",
            json={"filename": "catalog.csv", "csv_text": csv_text, "sample_size": 2},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows_profiled"] == 2
        assert data["valid_rows"] == 1
        assert data["missing_title_rows"] == 1
        assert data["samples"][0]["title"] == "Arrival"

    def test_catalog_upload_stores_manifest(self, tmp_path, monkeypatch, mock_artifacts):
        monkeypatch.setenv("NOVA_CATALOG_UPLOAD_PATH", str(tmp_path))

        from backend.main import app

        client = TestClient(app)

        csv_text = (
            "id,title,overview,genres\n"
            "1,Arrival,A linguist communicates with alien visitors in a tense science fiction story,Sci-Fi\n"
        )
        resp = client.post(
            "/v1/catalog/upload",
            json={"filename": "catalog.csv", "csv_text": csv_text},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"
        assert data["upload_id"]
        assert data["profile"]["ready_for_ingestion"] is True
