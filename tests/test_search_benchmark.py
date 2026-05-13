import json

from backend.search_benchmark import evaluate_search_benchmark, load_search_benchmark


def test_load_search_benchmark_reads_cases(tmp_path):
    path = tmp_path / "search_benchmark.json"
    path.write_text(
        json.dumps({"cases": [{"case_id": "avatar", "query": "Avatar"}]}),
        encoding="utf-8",
    )

    cases = load_search_benchmark(path)

    assert cases == [{"case_id": "avatar", "query": "Avatar"}]


def test_evaluate_search_benchmark_reports_top1_and_required_hits(tmp_path):
    path = tmp_path / "search_benchmark.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "avatar",
                        "query": "Avatar",
                        "expected_results": [{"id": 19995, "title": "Avatar"}],
                        "required_results": [{"id": 76600, "title": "Avatar: The Way of Water"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def search_fn(query: str, k: int):
        return [
            {"id": 19995, "title": "Avatar"},
            {"id": 76600, "title": "Avatar: The Way of Water"},
        ][:k]

    report = evaluate_search_benchmark(search_fn, path, k=5)

    assert report["status"] == "ok"
    assert report["metrics"]["top1_hit_rate"] == 1.0
    assert report["metrics"]["hit_rate_at_k"] == 1.0
    assert report["metrics"]["required_hit_case_rate"] == 1.0


def test_evaluate_search_benchmark_flags_regressions(tmp_path):
    path = tmp_path / "search_benchmark.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "avatar",
                        "query": "Avatar",
                        "expected_results": [{"id": 19995, "title": "Avatar"}],
                        "blocked_results": [{"id": 1096978, "title": "Avatar"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def search_fn(query: str, k: int):
        return [{"id": 1096978, "title": "Avatar"}]

    report = evaluate_search_benchmark(search_fn, path, k=5)

    assert report["status"] == "needs_attention"
    assert report["metrics"]["top1_hit_rate"] == 0.0
    assert report["metrics"]["blocked_hit_case_rate"] == 1.0

