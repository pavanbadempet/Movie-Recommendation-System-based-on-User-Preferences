"""Frontend failover routing tests."""

from fastapi.testclient import TestClient


def test_frontend_status_reports_streamlit_and_react_without_remote_probe(tmp_path, monkeypatch):
    import backend.frontend_failover as frontend_failover
    import backend.main as main

    frontend_failover._HEALTH_CACHE.clear()
    (tmp_path / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_DIST_DIR", tmp_path)
    monkeypatch.setenv("NOVA_FRONTEND_STREAMLIT_URL", "https://streamlit.example")
    monkeypatch.setenv("NOVA_FRONTEND_PRIORITY", "streamlit,react")

    client = TestClient(main.app)
    response = client.get("/v1/frontends/status", params={"include_remote": "false"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "multi_frontend_failover"
    assert payload["status"] == "degraded"
    statuses = {item["name"]: item["status"] for item in payload["frontends"]}
    assert statuses == {"streamlit": "unknown", "react": "ok"}
    assert payload["selected"]["name"] == "react"
    assert payload["launch_url"] == "http://testserver/ui/"


def test_frontend_launch_redirects_to_healthy_react_backup(tmp_path, monkeypatch):
    import backend.frontend_failover as frontend_failover
    import backend.main as main

    frontend_failover._HEALTH_CACHE.clear()
    (tmp_path / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_DIST_DIR", tmp_path)
    monkeypatch.setenv("NOVA_FRONTEND_STREAMLIT_URL", "https://streamlit.example")
    monkeypatch.setenv("NOVA_FRONTEND_PRIORITY", "streamlit,react")

    client = TestClient(main.app)
    response = client.get(
        "/go",
        params={"include_remote": "false"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["location"] == "http://testserver/ui/"


def test_frontend_launch_uses_forwarded_https_for_same_origin_backup(tmp_path, monkeypatch):
    import backend.frontend_failover as frontend_failover
    import backend.main as main

    frontend_failover._HEALTH_CACHE.clear()
    (tmp_path / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_DIST_DIR", tmp_path)
    monkeypatch.setenv("NOVA_FRONTEND_STREAMLIT_URL", "https://streamlit.example")
    monkeypatch.setenv("NOVA_FRONTEND_PRIORITY", "streamlit,react")

    client = TestClient(main.app)
    response = client.get(
        "/go",
        params={"include_remote": "false"},
        headers={
            "x-forwarded-proto": "https",
            "x-forwarded-host": "api.example",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["location"] == "https://api.example/ui/"


def test_frontend_status_honors_healthy_primary_streamlit(tmp_path, monkeypatch):
    import backend.frontend_failover as frontend_failover

    frontend_failover._HEALTH_CACHE.clear()
    (tmp_path / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setenv("NOVA_FRONTEND_STREAMLIT_URL", "https://streamlit.example")
    monkeypatch.setenv("NOVA_FRONTEND_PRIORITY", "streamlit,react")

    async def fake_probe(target, *, frontend_dist_dir, include_remote):
        if target.name == "streamlit":
            return {
                "name": target.name,
                "label": target.label,
                "kind": target.kind,
                "url": target.url,
                "health_url": target.health_url,
                "priority": target.priority,
                "local": target.local,
                "status": "ok",
                "http_status": 200,
                "latency_ms": 42.0,
                "error": None,
            }
        return {
            "name": target.name,
            "label": target.label,
            "kind": target.kind,
            "url": target.url,
            "health_url": target.health_url,
            "priority": target.priority,
            "local": target.local,
            "status": "ok",
            "http_status": None,
            "latency_ms": 0,
            "error": None,
        }

    monkeypatch.setattr(frontend_failover, "probe_frontend", fake_probe)

    import asyncio

    report = asyncio.run(
        frontend_failover.frontend_status_report(
            frontend_dist_dir=tmp_path,
            base_url="https://api.example/",
            include_remote=True,
            app={"version": "test"},
        )
    )

    assert report["status"] == "ready"
    assert report["selected"]["name"] == "streamlit"
    assert report["launch_url"] == "https://streamlit.example"
