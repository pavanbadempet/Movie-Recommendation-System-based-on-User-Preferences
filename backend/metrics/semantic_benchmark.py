"""Human-readable semantic benchmark evaluation for recommendations."""

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


from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "semantic_similarity_benchmark.json"
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


def _matches_item(movie: dict[str, Any], expected: dict[str, Any]) -> bool:
    if expected.get("id") is not None and movie.get("id") is not None:
        try:
            if int(expected["id"]) == int(movie["id"]):
                return True
        except (TypeError, ValueError):
            pass
    expected_title = _canonical_title(expected.get("title"))
    movie_title = _canonical_title(movie.get("title"))
    return bool(expected_title and movie_title and expected_title == movie_title)


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


def load_semantic_benchmark(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load benchmark cases from JSON."""
    benchmark_path = Path(path) if path is not None else DEFAULT_BENCHMARK_PATH
    if not benchmark_path.exists():
        return []
    payload = _jloads(benchmark_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("cases") or [])
    if isinstance(payload, list):
        return payload
    return []


def evaluate_semantic_benchmark(
    recommender: Any,
    benchmark_path: Path | str | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate recommender output against small human-labeled semantic cases."""
    cases = load_semantic_benchmark(benchmark_path)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    if not cases:
        return {
            "generated_at": generated_at,
            "status": "unavailable",
            "reason": "No semantic benchmark cases found",
            "case_count": 0,
        }

    evaluated = []
    skipped = []
    total_good_hits = 0
    total_bad_hits = 0
    total_good_labels = 0
    reciprocal_ranks = []
    ndcg_scores = []
    good_hit_cases = 0
    bad_hit_cases = 0
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
        recommendations = recommender.recommend_by_id(int(seed_movie["id"]), n=max(k, 1))
        top_recommendations = recommendations[:k]

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
    bad_rate = total_bad_hits / max(evaluated_count * k, 1)
    good_recall = total_good_hits / max(total_good_labels, 1)
    precision = total_good_hits / max(evaluated_count * k, 1)
    hit_rate = good_hit_cases / max(evaluated_count, 1)
    bad_case_rate = bad_hit_cases / max(evaluated_count, 1)
    status = "ok" if evaluated_count else "unavailable"
    if evaluated_count and bad_rate > 0.15:
        status = "needs_attention"
    if evaluated_count and hit_rate < 0.5:
        status = "needs_attention"

    return {
        "generated_at": generated_at,
        "status": status,
        "case_count": len(cases),
        "evaluated_case_count": evaluated_count,
        "skipped_case_count": len(skipped),
        "k": k,
        "metrics": {
            "good_recall_at_k": round(good_recall, 4),
            "precision_at_k": round(precision, 4),
            "hit_rate_at_k": round(hit_rate, 4),
            "mrr_at_k": round(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1), 4),
            "ndcg_at_k": round(sum(ndcg_scores) / max(len(ndcg_scores), 1), 4),
            "bad_match_rate_at_k": round(bad_rate, 4),
            "bad_case_rate_at_k": round(bad_case_rate, 4),
            "good_hit_count": total_good_hits,
            "bad_hit_count": total_bad_hits,
            "stage_distribution": dict(sorted(stage_counts.items())),
            "explanation_coverage": round(explanation_hits / max(total_results, 1), 4),
        },
        "cases": evaluated,
        "skipped": skipped,
    }
