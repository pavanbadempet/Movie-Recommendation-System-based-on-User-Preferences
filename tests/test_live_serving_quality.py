from argparse import Namespace
import json

from scripts import evaluate_live_serving as live


def _args(
    skip_semantic_benchmark: bool = False,
    allow_degraded_artifact_health: bool = False,
    skip_search_benchmark: bool = True,
    skip_recommendation_diagnostics: bool = False,
    skip_recommendation_benchmark: bool = False,
    search_benchmark_path=None,
) -> Namespace:
    return Namespace(
        base_url="https://example.test",
        timeout=1,
        retries=1,
        retry_delay_seconds=0,
        expected_app_commit="",
        k=10,
        search_query="Avatar",
        search_limit=5,
        min_search_results=1,
        expected_search_title="Avatar",
        required_search_titles="Avatar: Fire and Ash,Avatar: The Way of Water",
        min_required_search_hits=2,
        search_benchmark_path=search_benchmark_path,
        search_benchmark_k=5,
        search_benchmark_mode="endpoint",
        min_search_top1_hit_rate=0.98,
        min_search_hit_rate=1.0,
        max_search_blocked_hit_case_rate=0.0,
        skip_search_benchmark=skip_search_benchmark,
        recommendation_smoke_movie_id=19995,
        recommendation_smoke_k=10,
        min_recommendation_results=5,
        required_recommendation_titles=(
            "Avatar: The Way of Water,Avatar: Fire and Ash,The Abyss,Pacific Rim,Dune"
        ),
        min_required_recommendation_hits=2,
        blocked_recommendation_titles="Small Soldiers,Supergirl,Barbarella,The Last Airbender",
        recommendation_diagnostics_k=5,
        min_recommendation_diagnostic_results=5,
        require_diagnostic_benchmark_case=True,
        skip_recommendation_diagnostics=skip_recommendation_diagnostics,
        min_recommendation_benchmark_pass_rate=0.80,
        min_recommendation_benchmark_hit_rate=0.90,
        max_recommendation_benchmark_bad_case_rate=0.0,
        skip_recommendation_benchmark=skip_recommendation_benchmark,
        max_bad_match_rate=0.05,
        min_hit_rate=0.95,
        min_mrr=0.35,
        min_ndcg=0.25,
        min_explanation_coverage=0.90,
        skip_semantic_benchmark=skip_semantic_benchmark,
        allow_degraded_artifact_health=allow_degraded_artifact_health,
        fail_on_threshold=True,
    )


def _healthy_payload(path: str):
    if path == "/health":
        return {"status": "healthy", "movie_count": 75253, "app_commit": "abcdef123456"}
    if path == "/v1/artifacts/health":
        return {"status": "ready"}
    if path.startswith("/v1/search?"):
        return [
            {"id": 19995, "title": "Avatar"},
            {"id": 83533, "title": "Avatar: Fire and Ash"},
            {"id": 76600, "title": "Avatar: The Way of Water"},
        ]
    if path.startswith("/v1/evaluation/semantic-benchmark?"):
        return {
            "status": "ok",
            "metrics": {
                "bad_match_rate_at_k": 0.0,
                "hit_rate_at_k": 1.0,
                "mrr_at_k": 0.9,
                "ndcg_at_k": 0.6,
                "explanation_coverage": 1.0,
            },
        }
    if path.startswith("/v1/evaluation/search-benchmark?"):
        return {
            "status": "ok",
            "case_count": 1,
            "evaluated_case_count": 1,
            "metrics": {
                "top1_hit_rate": 1.0,
                "hit_rate_at_k": 1.0,
                "blocked_hit_case_rate": 0.0,
            },
        }
    if path.startswith("/v1/evaluation/recommendation-benchmark?"):
        return {
            "status": "ok",
            "case_count": 1,
            "evaluated_case_count": 1,
            "metrics": {
                "case_pass_rate": 1.0,
                "good_hit_case_rate": 1.0,
                "bad_case_rate_at_k": 0.0,
                "mrr_at_k": 0.9,
                "ndcg_at_k": 0.8,
            },
        }
    if path.startswith("/v1/diagnostics/recommendations/19995?"):
        return {
            "status": "ok",
            "diagnostics": {
                "result_count": 5,
                "benchmark_case_available": True,
                "benchmark_case_passed": True,
                "explanation_coverage": 1.0,
            },
            "recommendations": [
                {"title": "Avatar: The Way of Water", "retrieval_stage": "vector"},
                {"title": "The Abyss", "retrieval_stage": "vector"},
                {"title": "Pacific Rim", "retrieval_stage": "vector"},
                {"title": "Dune", "retrieval_stage": "vector"},
                {"title": "Prometheus", "retrieval_stage": "vector"},
            ],
        }
    raise AssertionError(f"Unexpected path: {path}")


def test_live_serving_gate_accepts_good_recommendation_smoke(monkeypatch):
    def fake_get_json(base_url, path, timeout):
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(_args())

    assert report["status"] == "ok"
    assert report["recommendation_smoke_summary"]["required_hit_count"] == 4
    assert report["recommendation_smoke_summary"]["blocked_hits"] == []


def test_live_serving_gate_rejects_bad_recommendation_drift(monkeypatch):
    def fake_get_json(base_url, path, timeout):
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Small Soldiers"},
                    {"title": "Supergirl"},
                    {"title": "Barbarella"},
                    {"title": "Mystery Men"},
                    {"title": "Kids Next Door: Operation Z.E.R.O."},
                ]
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(_args())

    assert report["status"] == "failed"
    assert any("required semantic hits" in failure for failure in report["failures"])
    assert any("blocked drift titles" in failure for failure in report["failures"])


def test_live_serving_gate_rejects_recommendation_benchmark_regression(monkeypatch):
    def fake_get_json(base_url, path, timeout):
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        if path.startswith("/v1/evaluation/recommendation-benchmark?"):
            return {
                "status": "needs_attention",
                "evaluated_case_count": 35,
                "metrics": {
                    "case_pass_rate": 0.51,
                    "good_hit_case_rate": 0.82,
                    "bad_case_rate_at_k": 0.14,
                },
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(_args())

    assert report["status"] == "failed"
    assert any("recommendation_benchmark_case_pass_rate" in failure for failure in report["failures"])
    assert any("recommendation_benchmark_bad_case_rate_at_k" in failure for failure in report["failures"])


def test_live_serving_gate_rejects_recommendation_diagnostics_regression(monkeypatch):
    def fake_get_json(base_url, path, timeout):
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        if path.startswith("/v1/diagnostics/recommendations/19995?"):
            return {
                "status": "ok",
                "diagnostics": {
                    "result_count": 3,
                    "benchmark_case_available": False,
                    "benchmark_case_passed": None,
                },
                "recommendations": [{"title": "Avatar: The Way of Water"}],
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(_args())

    assert report["status"] == "failed"
    assert any("recommendation diagnostics returned 3 results" in failure for failure in report["failures"])
    assert any("did not include a labeled benchmark case" in failure for failure in report["failures"])


def test_live_serving_gate_rejects_search_title_regression(monkeypatch):
    def fake_get_json(base_url, path, timeout):
        if path.startswith("/v1/search?"):
            return [
                {"id": 19995, "title": "Avatar"},
                {"id": 1096978, "title": "Avatar"},
                {"id": 282908, "title": "Avatar"},
            ]
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(_args())

    assert report["status"] == "failed"
    assert any("/v1/search found 0 required title hits" in failure for failure in report["failures"])


def test_live_serving_gate_runs_search_benchmark(monkeypatch, tmp_path):
    benchmark_path = tmp_path / "search_benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "avatar",
                        "query": "Avatar",
                        "expected_results": [{"id": 19995, "title": "Avatar"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_get_json(base_url, path, timeout):
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(
        _args(skip_search_benchmark=False, search_benchmark_path=benchmark_path)
    )

    assert report["status"] == "ok"
    assert report["search_benchmark"]["metrics"]["top1_hit_rate"] == 1.0


def test_live_serving_gate_rejects_stale_deploy_revision(monkeypatch):
    def fake_get_json(base_url, path, timeout):
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)
    args = _args()
    args.expected_app_commit = "fffffff999999999"

    report = live.evaluate_live_serving(args)

    assert report["status"] == "failed"
    assert any("does not match expected" in failure for failure in report["failures"])


def test_live_serving_gate_can_skip_semantic_benchmark_for_lite_gateway(monkeypatch):
    seen_paths = []

    def fake_get_json(base_url, path, timeout):
        seen_paths.append(path)
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(_args(skip_semantic_benchmark=True))

    assert report["status"] == "ok"
    assert report["semantic_benchmark"]["status"] == "skipped"
    assert not any(path.startswith("/v1/evaluation/semantic-benchmark") for path in seen_paths)


def test_live_serving_gate_can_skip_recommendation_benchmark_for_lite_gateway(monkeypatch):
    seen_paths = []

    def fake_get_json(base_url, path, timeout):
        seen_paths.append(path)
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(_args(skip_recommendation_benchmark=True))

    assert report["status"] == "ok"
    assert report["recommendation_benchmark"]["status"] == "skipped"
    assert not any(path.startswith("/v1/evaluation/recommendation-benchmark") for path in seen_paths)


def test_live_serving_gate_can_skip_recommendation_diagnostics_for_lite_gateway(monkeypatch):
    seen_paths = []

    def fake_get_json(base_url, path, timeout):
        seen_paths.append(path)
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(_args(skip_recommendation_diagnostics=True))

    assert report["status"] == "ok"
    assert report["recommendation_diagnostics"]["status"] == "skipped"
    assert not any(path.startswith("/v1/diagnostics/recommendations") for path in seen_paths)


def test_live_serving_gate_can_allow_degraded_artifacts_for_gateway(monkeypatch):
    def fake_get_json(base_url, path, timeout):
        if path == "/v1/artifacts/health":
            return {"status": "degraded"}
        if path.startswith("/v1/recommendations/id/19995?"):
            return {
                "recommendations": [
                    {"title": "Avatar: The Way of Water"},
                    {"title": "The Abyss"},
                    {"title": "Pacific Rim"},
                    {"title": "Dune"},
                    {"title": "Prometheus"},
                ]
            }
        return _healthy_payload(path)

    monkeypatch.setattr(live, "_get_json", fake_get_json)

    report = live.evaluate_live_serving(
        _args(skip_semantic_benchmark=True, allow_degraded_artifact_health=True)
    )

    assert report["status"] == "ok"
    assert report["artifact_health"]["status"] == "degraded"
