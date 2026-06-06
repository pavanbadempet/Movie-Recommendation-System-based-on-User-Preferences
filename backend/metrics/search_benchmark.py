"""Human-labeled search relevance benchmark utilities."""

from __future__ import annotations

import json

# Fast JSON
try:
    import orjson as _orjson

    def _jloads(s):
        return _orjson.loads(s)

    def _jdumps(obj, **kw) -> str:
        return _orjson.dumps(obj).decode()
except ImportError:

    def _jloads(s):
        return json.loads(s)

    def _jdumps(obj, **kw) -> str:
        return json.dumps(obj, **kw)


from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_SEARCH_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "search_quality_benchmark.json"
)


def _canonical_title(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _case_items(items: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            normalized.append({"title": str(item)})
    return normalized


def _matches_item(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    if expected.get("id") is not None and result.get("id") is not None:
        try:
            return int(expected["id"]) == int(result["id"])
        except (TypeError, ValueError):
            return False
    expected_title = _canonical_title(expected.get("title"))
    result_title = _canonical_title(result.get("title"))
    return bool(expected_title and result_title and expected_title == result_title)


def load_search_benchmark(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load search benchmark cases from JSON."""
    benchmark_path = Path(path) if path is not None else DEFAULT_SEARCH_BENCHMARK_PATH
    if not benchmark_path.exists():
        return []
    payload = _jloads(benchmark_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("cases") or [])
    if isinstance(payload, list):
        return payload
    return []


def evaluate_search_benchmark(
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    benchmark_path: Path | str | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate search output against expected canonical results."""
    cases = load_search_benchmark(benchmark_path)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    if not cases:
        return {
            "generated_at": generated_at,
            "status": "unavailable",
            "reason": "No search benchmark cases found",
            "case_count": 0,
        }

    evaluated = []
    skipped = []
    top1_hits = 0
    hit_cases = 0
    required_hit_cases = 0
    blocked_hit_cases = 0
    reciprocal_ranks = []

    for case in cases:
        query = str(case.get("query") or "").strip()
        if not query:
            skipped.append({"case_id": case.get("case_id"), "reason": "missing query"})
            continue

        expected_items = _case_items(case.get("expected_results") or case.get("expected") or [])
        required_items = _case_items(case.get("required_results") or [])
        blocked_items = _case_items(case.get("blocked_results") or [])
        try:
            raw_results = search_fn(query, max(k, 1))
        except Exception as exc:
            evaluated.append(
                {
                    "case_id": case.get("case_id"),
                    "query": query,
                    "error": str(exc),
                    "hit_rank": None,
                    "top_results": [],
                    "required_hits": [],
                    "blocked_hits": [],
                }
            )
            reciprocal_ranks.append(0.0)
            continue

        results = [item for item in raw_results[:k] if isinstance(item, dict)]
        hit_rank = None
        for rank, result in enumerate(results, start=1):
            if any(_matches_item(result, expected) for expected in expected_items):
                hit_rank = rank
                break

        if hit_rank == 1:
            top1_hits += 1
        if hit_rank is not None:
            hit_cases += 1
        reciprocal_ranks.append(0.0 if hit_rank is None else 1.0 / hit_rank)

        required_hits = [
            {"id": result.get("id"), "title": result.get("title")}
            for result in results
            if any(_matches_item(result, expected) for expected in required_items)
        ]
        blocked_hits = [
            {"id": result.get("id"), "title": result.get("title")}
            for result in results
            if any(_matches_item(result, expected) for expected in blocked_items)
        ]
        if required_items and len(required_hits) >= min(len(required_items), k):
            required_hit_cases += 1
        if blocked_hits:
            blocked_hit_cases += 1

        evaluated.append(
            {
                "case_id": case.get("case_id"),
                "query": query,
                "hit_rank": hit_rank,
                "required_hit_count": len(required_hits),
                "blocked_hit_count": len(blocked_hits),
                "required_hits": required_hits,
                "blocked_hits": blocked_hits,
                "top_results": [
                    {"id": result.get("id"), "title": result.get("title")} for result in results[: min(k, 5)]
                ],
            }
        )

    evaluated_count = len(evaluated)
    metrics = {
        "top1_hit_rate": round(top1_hits / max(evaluated_count, 1), 4),
        "hit_rate_at_k": round(hit_cases / max(evaluated_count, 1), 4),
        "mrr_at_k": round(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1), 4),
        "required_hit_case_rate": round(required_hit_cases / max(evaluated_count, 1), 4),
        "blocked_hit_case_rate": round(blocked_hit_cases / max(evaluated_count, 1), 4),
        "top1_hit_count": top1_hits,
        "hit_case_count": hit_cases,
        "required_hit_case_count": required_hit_cases,
        "blocked_hit_case_count": blocked_hit_cases,
    }
    status = "ok" if evaluated_count else "unavailable"
    if evaluated_count and (metrics["top1_hit_rate"] < 0.95 or metrics["blocked_hit_case_rate"] > 0):
        status = "needs_attention"

    return {
        "generated_at": generated_at,
        "status": status,
        "case_count": len(cases),
        "evaluated_case_count": evaluated_count,
        "skipped_case_count": len(skipped),
        "k": k,
        "metrics": metrics,
        "cases": evaluated,
        "skipped": skipped,
    }
