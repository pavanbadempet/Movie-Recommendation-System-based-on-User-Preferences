from fastapi.testclient import TestClient


def test_slo_report_detects_latency_or_error_violation(monkeypatch):
    from backend.slo import RequestSloTracker, build_slo_report

    monkeypatch.setenv("NOVA_SLO_MIN_REQUESTS", "2")
    monkeypatch.setenv("NOVA_SLO_MIN_ROUTE_REQUESTS", "1")
    monkeypatch.setenv("NOVA_SLO_LATENCY_P95_MS", "100")
    monkeypatch.setenv("NOVA_SLO_ERROR_RATE", "0.10")
    monkeypatch.setenv("NOVA_SLO_ROUTE_LATENCY_BUDGETS", "/health:100,/v1/search:100")
    tracker = RequestSloTracker(max_events=10)
    tracker.record(method="GET", path="/health", route="/health", status_code=200, latency_ms=25)
    tracker.record(method="GET", path="/v1/search", route="/v1/search", status_code=500, latency_ms=250)

    report = build_slo_report(
        tracker=tracker,
        app={"version": "test"},
        dependencies={
            "artifacts": {"status": "ready"},
            "remote_recommender": {"circuit": {"state": "closed"}},
            "frontends": {"status": "skipped"},
        },
    )

    assert report["status"] == "violated"
    assert report["traffic"]["request_count"] == 2
    assert report["slo"]["latency_p95_ms"]["passed"] is False
    assert report["slo"]["error_rate"]["passed"] is False
    assert report["slo"]["latency_p95_ms"]["route_violations"][0]["route"] == "/v1/search"


def test_internal_quality_routes_are_excluded_from_serving_slo(monkeypatch):
    from backend.slo import should_track_request

    monkeypatch.delenv("NOVA_SLO_EXCLUDED_ROUTE_PREFIXES", raising=False)

    assert should_track_request(path="/v1/search", route="/v1/search") is True
    assert should_track_request(path="/v1/recommendations/id/19995", route="/v1/recommendations/id/{movie_id}") is True
    assert should_track_request(path="/v1/evaluation/search-benchmark", route="/v1/evaluation/search-benchmark") is False
    assert should_track_request(path="/v1/platform/readiness", route="/v1/platform/readiness") is False


def test_platform_slo_endpoint_is_lightweight(monkeypatch):
    import backend.main as main

    main._slo_tracker.clear()
    main._slo_tracker.record(
        method="GET",
        path="/health",
        route="/health",
        status_code=200,
        latency_ms=20,
    )
    monkeypatch.setenv("NOVA_SLO_MIN_REQUESTS", "1")
    monkeypatch.setenv("NOVA_SLO_MIN_ROUTE_REQUESTS", "1")
    monkeypatch.setattr(
        main,
        "evaluate_artifact_health",
        lambda **kwargs: {
            "status": "ready",
            "row_counts": {"movies": 3, "embeddings": 3},
            "alignment": {"status": "ok"},
        },
    )
    monkeypatch.setattr(
        main,
        "remote_recommender_status",
        lambda: {"configured": False, "circuit": {"state": "closed"}},
    )

    client = TestClient(main.app)
    response = client.get("/v1/platform/slo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["traffic"]["request_count"] >= 1
    assert payload["dependencies"]["artifacts"]["status"] == "ready"
    assert payload["dependencies"]["frontends"]["status"] == "skipped"
