"""Tests for Multi-Armed Bandit Exploration Engine."""

import pytest
from backend.intelligence.bandit_engine import ThompsonSamplingBandit, UCB1Bandit


def test_thompson_sampling_bandit():
    bandit = ThompsonSamplingBandit()
    bandit.record_feedback(item_id=100, reward=1.0)
    bandit.record_feedback(item_id=100, reward=1.0)
    bandit.record_feedback(item_id=200, reward=0.0)

    candidates = [
        {"id": 100, "title": "High CTR Movie", "similarity_score": 0.8},
        {"id": 200, "title": "Low CTR Movie", "similarity_score": 0.8},
    ]

    reranked = bandit.rank_candidates(candidates, exploration_weight=0.5)
    assert len(reranked) == 2
    assert "bandit_score" in reranked[0]


def test_ucb1_bandit():
    bandit = UCB1Bandit(c_parameter=1.414)
    bandit.record_feedback(item_id=1, reward=1.0)
    bandit.record_feedback(item_id=2, reward=0.0)

    score_1 = bandit.score_item(1)
    score_2 = bandit.score_item(2)

    assert score_1 > score_2
