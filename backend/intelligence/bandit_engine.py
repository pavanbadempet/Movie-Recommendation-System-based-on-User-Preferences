"""Multi-Armed Bandit Online Exploration & Exploitation Engine.

Implements Thompson Sampling (Beta Distribution) and UCB1 (Upper Confidence Bound)
to dynamic balance high-performing recommendations with new catalog item discovery.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import numpy as np


class ThompsonSamplingBandit:
    """Thompson Sampling Bandit using Beta Priors for CTR optimization."""

    def __init__(self):
        # Maps item_id -> {"successes": int, "failures": int}
        self.item_stats: Dict[int, Dict[str, int]] = {}

    def record_feedback(self, item_id: int, reward: float):
        """Record positive (click/conversion) or negative signal."""
        if item_id not in self.item_stats:
            self.item_stats[item_id] = {"successes": 1, "failures": 1}

        if reward > 0.0:
            self.item_stats[item_id]["successes"] += 1
        else:
            self.item_stats[item_id]["failures"] += 1

    def sample_score(self, item_id: int) -> float:
        """Sample from Beta distribution prior B(α, β)."""
        stats = self.item_stats.get(item_id, {"successes": 1, "failures": 1})
        alpha = max(1, stats["successes"])
        beta = max(1, stats["failures"])
        return float(np.random.beta(alpha, beta))

    def rank_candidates(self, candidates: List[Dict[str, Any]], exploration_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Rank candidate items by combining exploitation relevance with Thompson Sampling exploration."""
        reranked = []
        for item in candidates:
            item_id = int(item.get("id", 0))
            exploit_score = float(item.get("similarity_score") or item.get("vote_average", 5.0) / 10.0)
            sample_score = self.sample_score(item_id)

            combined_score = (1.0 - exploration_weight) * exploit_score + exploration_weight * sample_score
            item_copy = dict(item)
            item_copy["bandit_score"] = combined_score
            item_copy["exploration_sample"] = sample_score
            reranked.append(item_copy)

        reranked.sort(key=lambda x: x["bandit_score"], reverse=True)
        return reranked


class UCB1Bandit:
    """Upper Confidence Bound (UCB1) Bandit for deterministic optimism in the face of uncertainty."""

    def __init__(self, c_parameter: float = 1.414):
        self.c_parameter = c_parameter
        self.item_counts: Dict[int, int] = {}
        self.item_rewards: Dict[int, float] = {}
        self.total_pulls = 0

    def record_feedback(self, item_id: int, reward: float):
        self.total_pulls += 1
        self.item_counts[item_id] = self.item_counts.get(item_id, 0) + 1
        self.item_rewards[item_id] = self.item_rewards.get(item_id, 0.0) + reward

    def score_item(self, item_id: int) -> float:
        count = self.item_counts.get(item_id, 0)
        if count == 0 or self.total_pulls == 0:
            return float("inf")

        mean_reward = self.item_rewards.get(item_id, 0.0) / count
        confidence_bound = self.c_parameter * math.sqrt(math.log(self.total_pulls) / count)
        return mean_reward + confidence_bound
