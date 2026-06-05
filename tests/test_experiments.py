"""
Tests for Nova experiment assignment and metrics.
"""

from backend.experiments import assign_experiment, attach_experiment, summarize_experiment_metrics


def test_assignment_is_stable_for_same_subject():
    first = assign_experiment("user-1", variants={"control": 50, "personalized_v2": 50})
    second = assign_experiment("user-1", variants={"control": 50, "personalized_v2": 50})

    assert first["variant"] == second["variant"]
    assert first["experiment"] == second["experiment"]


def test_attach_experiment_adds_response_metadata():
    assignment = assign_experiment("user-1", variants={"control": 100})
    candidates = attach_experiment([{"id": 100, "retrieval_signals": {"behavior": 0.1}}], assignment)

    assert candidates[0]["retrieval_signals"]["variant"] == "control"
    assert candidates[0]["retrieval_signals"]["experiment"] == assignment["experiment"]


def test_summarize_experiment_metrics():
    metrics = summarize_experiment_metrics(
        [
            {
                "event_type": "recommendation_impression",
                "metadata": {"experiment": "ranker", "variant": "control"},
            },
            {
                "event_type": "click",
                "metadata": {"experiment": "ranker", "variant": "control"},
            },
            {
                "event_type": "rating",
                "rating": 5,
                "metadata": {"experiment": "ranker", "variant": "control"},
            },
        ]
    )

    row = metrics["experiments"][0]
    assert row["experiment"] == "ranker"
    assert row["variant"] == "control"
    assert row["ctr"] == 1.0
    assert row["avg_rating"] == 5.0
