"""Tests for the human-labeled semantic benchmark evaluator."""

import json

import pandas as pd

from backend.semantic_benchmark import evaluate_semantic_benchmark
from scripts.evaluate_semantic_benchmark import build_offline_recommender


class FakeRecommender:
    def __init__(self):
        self.movies = pd.DataFrame(
            [
                {"id": 1, "title": "Avatar"},
                {"id": 2, "title": "Dune"},
                {"id": 3, "title": "Small Soldiers"},
            ]
        )

    def get_movie_by_id(self, movie_id):
        matches = self.movies[self.movies["id"] == movie_id]
        if len(matches) == 0:
            return None
        return matches.iloc[0].to_dict()

    def search_movies(self, title, limit=5):
        title = title.lower()
        matches = self.movies[self.movies["title"].str.lower().str.contains(title, regex=False)]
        return matches.head(limit).to_dict(orient="records")

    def recommend_by_id(self, movie_id, n=10):
        return [
            {"id": 2, "title": "Dune", "similarity_score": 0.91, "explanation": ["good match"]},
            {"id": 3, "title": "Small Soldiers", "similarity_score": 0.12, "explanation": ["bad drift"]},
        ][:n]


def test_semantic_benchmark_reports_good_and_bad_hits(tmp_path):
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "avatar",
                        "seed": {"id": 1, "title": "Avatar"},
                        "good_matches": [{"id": 2, "title": "Dune"}],
                        "bad_matches": [{"id": 3, "title": "Small Soldiers"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_semantic_benchmark(FakeRecommender(), benchmark_path=benchmark_path, k=2)

    assert report["evaluated_case_count"] == 1
    assert report["metrics"]["good_hit_count"] == 1
    assert report["metrics"]["bad_hit_count"] == 1
    assert report["metrics"]["hit_rate_at_k"] == 1.0
    assert report["metrics"]["mrr_at_k"] == 1.0
    assert report["metrics"]["ndcg_at_k"] == 1.0
    assert report["metrics"]["explanation_coverage"] == 1.0
    assert report["cases"][0]["good_hits"][0]["title"] == "Dune"
    assert report["cases"][0]["good_hits"][0]["rank"] == 1


def test_offline_benchmark_recommender_uses_catalog_without_vectors(tmp_path):
    movies_path = tmp_path / "movies_transformed.parquet"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Avatar", "Dune", "Small Soldiers"],
            "overview": [
                "Alien world adventure with ecological conflict and a native civilization.",
                "Desert planet civilization conflict with prophecy and political war.",
                "Toy soldiers cause suburban action comedy chaos.",
            ],
            "genres": ["Action, Adventure, Science Fiction", "Adventure, Science Fiction", "Comedy, Family"],
            "vote_average": [7.6, 8.0, 6.0],
            "vote_count": [10000, 9000, 100],
            "popularity": [100.0, 95.0, 30.0],
            "release_date": ["2009-01-01", "2021-01-01", "1998-01-01"],
            "poster_path": ["", "", ""],
        }
    ).to_parquet(movies_path, index=False)

    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "avatar",
                        "seed": {"id": 1, "title": "Avatar"},
                        "good_matches": [{"id": 2, "title": "Dune"}],
                        "bad_matches": [{"id": 3, "title": "Small Soldiers"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rec = build_offline_recommender(movies_path)
    report = evaluate_semantic_benchmark(rec, benchmark_path=benchmark_path, k=2)

    assert report["evaluated_case_count"] == 1
    assert report["metrics"]["good_hit_count"] >= 1
