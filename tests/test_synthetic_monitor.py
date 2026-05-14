from argparse import Namespace

from scripts import synthetic_monitor as monitor


def _good_payload(path: str):
    if path == "/health":
        return {"status": "healthy", "movie_count": 75253}
    if path.startswith("/v1/frontends/status?"):
        return {
            "status": "ready",
            "selected": {"name": "react", "absolute_url": "https://api.example/ui/"},
        }
    if path.startswith("/v1/platform/slo?"):
        return {"status": "ok", "traffic": {"request_count": 10}}
    if path.startswith("/v1/search?"):
        return [{"id": 19995, "title": "Avatar"}]
    if path.startswith("/v1/recommendations/id/19995?"):
        return {
            "recommendations": [
                {"title": "Avatar: The Way of Water"},
                {"title": "The Abyss"},
                {"title": "Pacific Rim"},
            ]
        }
    raise AssertionError(f"Unexpected path: {path}")


def test_synthetic_monitor_accepts_good_product_paths(monkeypatch):
    monkeypatch.setattr(monitor, "_get_json", lambda base_url, path, timeout: _good_payload(path))

    report = monitor.evaluate_synthetic_monitor(
        Namespace(
            base_url=["https://api.example"],
            timeout=1,
            skip_recommendations=False,
        )
    )

    assert report["status"] == "ok"
    assert report["failures"] == []
    assert {check["name"] for check in report["targets"][0]["checks"]} == {
        "health",
        "frontends",
        "slo",
        "search_avatar",
        "recommend_avatar",
    }


def test_synthetic_monitor_rejects_known_recommendation_drift(monkeypatch):
    def fake_get_json(base_url, path, timeout):
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Small Soldiers"},
                    {"title": "Supergirl"},
                    {"title": "Barbarella"},
                ]
            }
        return _good_payload(path)

    monkeypatch.setattr(monitor, "_get_json", fake_get_json)

    report = monitor.evaluate_synthetic_monitor(
        Namespace(
            base_url=["https://api.example"],
            timeout=1,
            skip_recommendations=False,
        )
    )

    assert report["status"] == "failed"
    assert any("known drift titles" in failure for failure in report["failures"])
