"""Compare pytest-benchmark output against a committed performance baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _means(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError(f"No benchmark results found in {path}")
    result: dict[str, float] = {}
    for benchmark in benchmarks:
        name = benchmark.get("fullname") or benchmark.get("name")
        stats = benchmark.get("stats") if isinstance(benchmark.get("stats"), dict) else {}
        mean = stats.get("mean")
        if not name or mean is None:
            raise ValueError(f"Malformed benchmark entry in {path}: {benchmark}")
        result[str(name)] = float(mean)
    return result


def compare_benchmarks(current_path: Path, baseline_path: Path, threshold: float = 0.10) -> dict:
    """Return a deterministic regression report for matching benchmark names."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    current = _means(Path(current_path))
    baseline = _means(Path(baseline_path))
    missing = sorted(set(baseline) - set(current))
    regressions = []
    comparisons = []
    for name in sorted(set(baseline) & set(current)):
        baseline_mean = baseline[name]
        current_mean = current[name]
        if baseline_mean <= 0:
            raise ValueError(f"Baseline mean must be positive for {name}")
        change_ratio = (current_mean - baseline_mean) / baseline_mean
        item = {
            "name": name,
            "baseline_mean_seconds": baseline_mean,
            "current_mean_seconds": current_mean,
            "change_ratio": round(change_ratio, 6),
        }
        comparisons.append(item)
        if change_ratio > threshold:
            regressions.append(item)
    return {
        "passed": not missing and not regressions,
        "threshold": threshold,
        "missing_benchmarks": missing,
        "regressions": regressions,
        "comparisons": comparisons,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = compare_benchmarks(args.current, args.baseline, args.threshold)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
