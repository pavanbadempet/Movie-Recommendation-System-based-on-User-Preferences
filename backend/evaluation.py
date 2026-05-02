"""
Recommendation quality evaluation for the Nova demo/product console.

These metrics are label-free and cheap enough for free-tier hosting. They do not
replace a real A/B test, but they make the AI layer measurable: vector health,
catalog coverage, genre consistency, and diversity.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


def _genre_set(value: Any) -> set[str]:
    return {part.strip().lower() for part in str(value or "").split(",") if part.strip()}


def evaluate_recommendation_quality(recommender: Any, sample_size: int = 25, k: int = 10) -> dict[str, Any]:
    """Evaluate current recommendation artifacts without external labels."""
    movies: pd.DataFrame = recommender.movies
    vectors = getattr(recommender, "_vectors", None)
    index = getattr(recommender, "_index", None)

    if movies is None or len(movies) == 0:
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "status": "unavailable",
            "reason": "No catalog rows loaded",
        }

    metrics: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "ok",
        "movie_count": int(len(movies)),
        "sample_size": int(min(max(sample_size, 1), len(movies))),
        "k": int(max(k, 1)),
        "catalog": {},
        "vectors": {},
        "recommendations": {},
    }

    if "genres" in movies.columns:
        all_genres = set()
        for value in movies["genres"].dropna().head(10000):
            all_genres.update(_genre_set(value))
        metrics["catalog"]["unique_genres"] = len(all_genres)
        metrics["catalog"]["genre_coverage_sample"] = sorted(all_genres)[:30]

    if "original_language" in movies.columns:
        metrics["catalog"]["language_count"] = int(movies["original_language"].dropna().nunique())

    required_serving_columns = {"id", "title", "overview", "genres"}
    metrics["catalog"]["required_serving_columns_present"] = sorted(required_serving_columns & set(movies.columns))
    metrics["catalog"]["missing_serving_columns"] = sorted(required_serving_columns - set(movies.columns))

    if vectors is None or index is None:
        metrics["status"] = "partial"
        metrics["vectors"] = {
            "available": False,
            "artifact_status": getattr(recommender, "_artifact_status", {}),
        }
        metrics["recommendations"] = {
            "available": True,
            "mode": "content_sparse_fallback",
            "note": "Vector artifacts are unavailable or failed alignment checks; sparse content fallback is serving recommendations.",
        }
        return metrics

    vector_count = int(getattr(index, "ntotal", 0))
    vector_dim = int(vectors.shape[1]) if len(vectors.shape) == 2 else None
    norm_sample = np.asarray(vectors[: min(len(vectors), 1000)], dtype=np.float32)
    norms = np.linalg.norm(norm_sample, axis=1) if len(norm_sample) else np.array([])

    metrics["vectors"] = {
        "available": True,
        "vector_count": vector_count,
        "dimension": vector_dim,
        "norm_mean": round(float(norms.mean()), 6) if len(norms) else None,
        "norm_std": round(float(norms.std()), 6) if len(norms) else None,
        "index_rows_match_catalog": vector_count == len(movies),
    }

    sample_n = metrics["sample_size"]
    top_k = min(metrics["k"] + 1, vector_count)
    if sample_n <= 0 or top_k <= 1:
        metrics["recommendations"]["available"] = False
        return metrics

    sample_indices = np.linspace(0, min(len(movies), vector_count) - 1, sample_n, dtype=int)
    recommended_ids: set[int] = set()
    genre_overlap_hits = 0
    genre_overlap_total = 0
    diversity_scores = []
    self_match_hits = 0

    for movie_idx in sample_indices:
        query_vector = np.asarray(vectors[movie_idx], dtype=np.float32).reshape(1, -1)
        distances, indices = index.search(np.ascontiguousarray(query_vector), top_k)
        if len(indices[0]) and int(indices[0][0]) == int(movie_idx):
            self_match_hits += 1

        query_genres = _genre_set(movies.iloc[int(movie_idx)].get("genres", ""))
        result_genres: set[str] = set()
        result_count = 0

        for rec_idx in indices[0]:
            rec_idx = int(rec_idx)
            if rec_idx < 0 or rec_idx == int(movie_idx) or rec_idx >= len(movies):
                continue

            result_count += 1
            recommended_ids.add(rec_idx)
            candidate_genres = _genre_set(movies.iloc[rec_idx].get("genres", ""))
            result_genres.update(candidate_genres)

            if query_genres and candidate_genres:
                genre_overlap_total += 1
                if not query_genres.isdisjoint(candidate_genres):
                    genre_overlap_hits += 1

            if result_count >= metrics["k"]:
                break

        if result_count:
            diversity_scores.append(len(result_genres) / result_count)

    total_possible = sample_n * metrics["k"]
    metrics["recommendations"] = {
        "available": True,
        "self_match_rate": round(self_match_hits / sample_n, 4),
        "catalog_coverage_at_k": round(len(recommended_ids) / max(total_possible, 1), 4),
        "genre_overlap_rate": round(genre_overlap_hits / max(genre_overlap_total, 1), 4),
        "avg_genre_diversity_per_list": round(float(np.mean(diversity_scores)), 4) if diversity_scores else None,
        "unique_recommended_items": len(recommended_ids),
    }
    return metrics
