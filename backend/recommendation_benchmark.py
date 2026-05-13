"""Human-labeled recommendation benchmark utilities."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_RECOMMENDATION_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "evaluation" / "recommendation_quality_benchmark.json"
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


def _find_seed_movie(recommender: Any, case: dict[str, Any]) -> dict[str, Any] | None:
    seed = case.get("seed") or {}
    if isinstance(seed, str):
        seed = {"title": seed}

    if seed.get("id") is not None:
        try:
            movie = recommender.get_movie_by_id(int(seed["id"]))
            if movie:
                return movie
        except (TypeError, ValueError):
            pass

    title = str(seed.get("title") or "").strip()
    if title:
        matches = recommender.search_movies(title, limit=5)
        for match in matches:
            if _canonical_title(match.get("title")) == _canonical_title(title):
                return match
        if matches:
            return matches[0]
    return None


def load_recommendation_benchmark(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load recommendation benchmark cases from JSON."""
    benchmark_path = Path(path) if path is not None else DEFAULT_RECOMMENDATION_BENCHMARK_PATH
    if not benchmark_path.exists():
        return []
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("cases") or [])
    if isinstance(payload, list):
        return payload
    return []


def evaluate_recommendation_benchmark(
    recommender: Any,
    benchmark_path: Path | str | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate recommendation output against product-quality labeled cases."""
    cases = load_recommendation_benchmark(benchmark_path)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    if not cases:
        return {
            "generated_at": generated_at,
            "status": "unavailable",
            "reason": "No recommendation benchmark cases found",
            "case_count": 0,
        }

    evaluated = []
    skipped = []
    pass_count = 0
    total_good_hits = 0
    total_bad_hits = 0
    total_good_labels = 0
    good_hit_cases = 0
    bad_hit_cases = 0
    reciprocal_ranks = []
    ndcg_scores = []
    explanation_hits = 0
    stage_counts: dict[str, int] = {}
    total_results = 0

    for case in cases:
        seed_movie = _find_seed_movie(recommender, case)
        if not seed_movie:
            skipped.append({"case_id": case.get("case_id"), "reason": "seed item not found"})
            continue

        good_items = _case_items(case.get("good_matches") or [])
        bad_items = _case_items(case.get("bad_matches") or [])
        min_good_hits = int(case.get("min_good_hits") or (1 if good_items else 0))
        max_bad_hits = int(case.get("max_bad_hits", 0))

        recommendations = recommender.recommend_by_id(int(seed_movie["id"]), n=max(k, 1))
        top_recommendations = [item for item in recommendations[:k] if isinstance(item, dict)]

        good_hits = []
        bad_hits = []
        first_good_rank = None
        dcg = 0.0
        for rank, rec in enumerate(top_recommendations, start=1):
            total_results += 1
            if rec.get("explanation") or rec.get("explanation_text"):
                explanation_hits += 1
            stage = str(rec.get("retrieval_stage") or "unknown")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

            if any(_matches_item(rec, item) for item in good_items):
                good_hits.append({"id": rec.get("id"), "title": rec.get("title"), "rank": rank})
                if first_good_rank is None:
                    first_good_rank = rank
                dcg += 1.0 / math.log2(rank + 1)
            if any(_matches_item(rec, item) for item in bad_items):
                bad_hits.append({"id": rec.get("id"), "title": rec.get("title"), "rank": rank})

        case_passed = len(good_hits) >= min_good_hits and len(bad_hits) <= max_bad_hits
        if case_passed:
            pass_count += 1
        if good_hits:
            good_hit_cases += 1
        if bad_hits:
            bad_hit_cases += 1
        reciprocal_ranks.append(0.0 if first_good_rank is None else 1.0 / first_good_rank)
        ideal_hits = min(len(good_items), k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg_scores.append(0.0 if ideal_dcg <= 0 else dcg / ideal_dcg)

        total_good_hits += len(good_hits)
        total_bad_hits += len(bad_hits)
        total_good_labels += len(good_items)
        evaluated.append(
            {
                "case_id": case.get("case_id"),
                "seed": {"id": seed_movie.get("id"), "title": seed_movie.get("title")},
                "intent": case.get("intent"),
                "k": k,
                "passed": case_passed,
                "min_good_hits": min_good_hits,
                "max_bad_hits": max_bad_hits,
                "good_hit_count": len(good_hits),
                "bad_hit_count": len(bad_hits),
                "good_hits": good_hits,
                "bad_hits": bad_hits,
                "top_results": [
                    {
                        "id": rec.get("id"),
                        "title": rec.get("title"),
                        "score": rec.get("similarity_score"),
                        "retrieval_stage": rec.get("retrieval_stage"),
                        "explanation": rec.get("explanation"),
                    }
                    for rec in top_recommendations[:5]
                ],
            }
        )

    evaluated_count = len(evaluated)
    pass_rate = pass_count / max(evaluated_count, 1)
    bad_rate = total_bad_hits / max(evaluated_count * k, 1)
    good_recall = total_good_hits / max(total_good_labels, 1)
    hit_rate = good_hit_cases / max(evaluated_count, 1)
    bad_case_rate = bad_hit_cases / max(evaluated_count, 1)
    status = "ok" if evaluated_count else "unavailable"
    if evaluated_count and (pass_rate < 0.8 or bad_case_rate > 0.05):
        status = "needs_attention"

    return {
        "generated_at": generated_at,
        "status": status,
        "case_count": len(cases),
        "evaluated_case_count": evaluated_count,
        "skipped_case_count": len(skipped),
        "k": k,
        "metrics": {
            "case_pass_rate": round(pass_rate, 4),
            "case_pass_count": pass_count,
            "good_hit_case_rate": round(hit_rate, 4),
            "bad_case_rate_at_k": round(bad_case_rate, 4),
            "bad_match_rate_at_k": round(bad_rate, 4),
            "good_recall_at_k": round(good_recall, 4),
            "mrr_at_k": round(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1), 4),
            "ndcg_at_k": round(sum(ndcg_scores) / max(len(ndcg_scores), 1), 4),
            "good_hit_count": total_good_hits,
            "bad_hit_count": total_bad_hits,
            "stage_distribution": dict(sorted(stage_counts.items())),
            "explanation_coverage": round(explanation_hits / max(total_results, 1), 4),
        },
        "cases": evaluated,
        "skipped": skipped,
    }
