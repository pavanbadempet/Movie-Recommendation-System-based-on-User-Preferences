from backend.metrics.recommendation_benchmark import (
    evaluate_recommendation_benchmark,
    evaluate_recommendation_case,
    find_recommendation_benchmark_case,
)


class FakeRecommender:
    def __init__(self, movies, recommendations):
        self.movies = movies
        self.recommendations = recommendations

    def get_movie_by_id(self, movie_id):
        return self.movies.get(int(movie_id))

    def search_movies(self, query, limit=5):
        normalized = query.lower()
        return [movie for movie in self.movies.values() if normalized in movie["title"].lower()][:limit]

    def recommend_by_id(self, movie_id, n=10):
        return list(self.recommendations.get(int(movie_id), []))[:n]


def test_recommendation_benchmark_scores_good_and_bad_hits(tmp_path):
    benchmark_path = tmp_path / "recommendation_quality_benchmark.json"
    benchmark_path.write_text(
        """
        {
          "cases": [
            {
              "case_id": "avatar_case",
              "seed": {"id": 19995, "title": "Avatar"},
              "min_good_hits": 2,
              "max_bad_hits": 0,
              "good_matches": [
                {"id": 76600, "title": "Avatar: The Way of Water"},
                {"id": 438631, "title": "Dune"}
              ],
              "bad_matches": [
                {"id": 11551, "title": "Small Soldiers"}
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    recommender = FakeRecommender(
        movies={19995: {"id": 19995, "title": "Avatar"}},
        recommendations={
            19995: [
                {
                    "id": 76600,
                    "title": "Avatar: The Way of Water",
                    "retrieval_stage": "vector",
                    "explanation": ["same world"],
                },
                {"id": 438631, "title": "Dune", "retrieval_stage": "vector", "explanation": ["desert epic"]},
            ]
        },
    )

    report = evaluate_recommendation_benchmark(recommender, benchmark_path=benchmark_path, k=2)

    assert report["status"] == "ok"
    assert report["metrics"]["case_pass_rate"] == 1.0
    assert report["metrics"]["good_hit_case_rate"] == 1.0
    assert report["metrics"]["bad_case_rate_at_k"] == 0.0
    assert report["metrics"]["explanation_coverage"] == 1.0
    assert report["cases"][0]["passed"] is True


def test_recommendation_benchmark_uses_id_before_title_match(tmp_path):
    benchmark_path = tmp_path / "recommendation_quality_benchmark.json"
    benchmark_path.write_text(
        """
        {
          "cases": [
            {
              "case_id": "id_case",
              "seed": {"title": "Avatar"},
              "min_good_hits": 1,
              "good_matches": [{"id": 76600, "title": "Avatar: The Way of Water"}],
              "bad_matches": []
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    recommender = FakeRecommender(
        movies={19995: {"id": 19995, "title": "Avatar"}},
        recommendations={
            19995: [
                {"id": 999999, "title": "Avatar: The Way of Water", "retrieval_stage": "vector"},
            ]
        },
    )

    report = evaluate_recommendation_benchmark(recommender, benchmark_path=benchmark_path, k=1)

    assert report["status"] == "needs_attention"
    assert report["metrics"]["good_hit_case_rate"] == 0.0
    assert report["cases"][0]["good_hits"] == []
    assert report["cases"][0]["passed"] is False


def test_single_case_diagnostics_can_be_reused_by_api():
    case = {
        "case_id": "seed_case",
        "seed": {"id": 100, "title": "Seed Movie"},
        "min_good_hits": 1,
        "good_matches": [{"id": 200, "title": "Good Match"}],
        "bad_matches": [{"id": 300, "title": "Bad Drift"}],
    }

    found = find_recommendation_benchmark_case(
        {"id": 100, "title": "Seed Movie"},
        cases=[case],
    )
    summary = evaluate_recommendation_case(
        [
            {"id": 200, "title": "Good Match", "retrieval_stage": "vector", "explanation": ["shared genre"]},
            {"id": 400, "title": "Neutral Match", "retrieval_stage": "vector"},
        ],
        found,
        k=2,
        seed_movie={"id": 100, "title": "Seed Movie"},
    )

    assert found["case_id"] == "seed_case"
    assert summary["passed"] is True
    assert summary["good_hit_count"] == 1
    assert summary["bad_hit_count"] == 0
    assert summary["_aggregate"]["stage_counts"] == {"vector": 2}
