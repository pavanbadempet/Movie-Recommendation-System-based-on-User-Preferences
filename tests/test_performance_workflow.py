import json
from pathlib import Path


def _benchmark_payload(mean: float) -> dict:
    return {
        "benchmarks": [
            {
                "fullname": "tests/test_performance_regression.py::test_ranker_throughput",
                "stats": {"mean": mean},
            }
        ]
    }


def test_performance_checker_reports_regression(tmp_path):
    from scripts.check_performance_regression import compare_benchmarks

    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    current.write_text(json.dumps(_benchmark_payload(0.012)), encoding="utf-8")
    baseline.write_text(json.dumps(_benchmark_payload(0.010)), encoding="utf-8")

    report = compare_benchmarks(current, baseline, threshold=0.10)

    assert report["passed"] is False
    assert report["regressions"][0]["change_ratio"] == 0.2


def test_performance_checker_accepts_benchmark_within_budget(tmp_path):
    from scripts.check_performance_regression import compare_benchmarks

    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    current.write_text(json.dumps(_benchmark_payload(0.0105)), encoding="utf-8")
    baseline.write_text(json.dumps(_benchmark_payload(0.010)), encoding="utf-8")

    report = compare_benchmarks(current, baseline, threshold=0.10)

    assert report["passed"] is True
    assert report["regressions"] == []


def test_performance_workflow_targets_real_benchmarks_and_existing_scripts():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "performance-regression.yml").read_text(encoding="utf-8")

    assert "tests/test_performance_regression.py" in workflow
    assert "tests/test_pure_unit_tests.py" not in workflow
    assert "scripts/check_performance_regression.py" in workflow
    assert (root / "scripts" / "check_performance_regression.py").is_file()
    assert (root / ".github" / "baselines" / "benchmark-baseline.json").is_file()
