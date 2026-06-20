import numpy as np

from backend.metrics.debiased_metrics import ips_ndcg_at_k
from backend.pipeline.multi_objective_ranker import pareto_rank
from backend.pipeline.recommender_core import build_rl_state


def test_ranker_throughput(benchmark):
    candidates = [
        {
            "id": index,
            "similarity_score": 1.0 - index / 500,
            "genres": "Action,Drama" if index % 2 else "Comedy",
            "vote_average": 7.5,
            "vote_count": 100 + index,
        }
        for index in range(200)
    ]
    popularity = {index: (index + 1) / 20_000 for index in range(200)}

    result = benchmark(
        pareto_rank,
        candidates,
        set(range(0, 50, 3)),
        {"action": 10.0, "drama": 4.0},
        popularity,
        20,
    )

    assert len(result) == 20


def test_rl_state_throughput(benchmark):
    profile = {"total_ratings": 200, "avg_rating": 4.2, "click_count": 80, "view_count": 300}
    embedding = np.linspace(-1.0, 1.0, 16, dtype=np.float32)

    state = benchmark(build_rl_state, profile, embedding)

    assert tuple(state.shape) == (1, 20)


def test_ips_ndcg_throughput(benchmark):
    ranked = list(range(100))
    ground_truth = set(range(0, 100, 7))
    popularity = {index: (index + 1) / 10_000 for index in range(100)}

    score = benchmark(ips_ndcg_at_k, ranked, ground_truth, popularity, 50)

    assert 0.0 <= score <= 1.0
