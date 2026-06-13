"""
Contextual Bandit Engine for Recommendation Exploration.

Implements exploration strategies (Thompson Sampling, UCB, Epsilon-Greedy) to
prevent filter bubbles and solve the cold-start problem for new items.
This perfectly balances Exploitation (MMoE Ranker) with Exploration (Bandits).
"""

from collections import defaultdict
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BanditEngine:
    def __init__(self):
        # In-memory storage for bandit states.
        # In production, this syncs with Redis or PostgreSQL.
        self.item_impressions = defaultdict(int)
        self.item_clicks = defaultdict(int)
        self.total_impressions = 0

    def inject_priors(self, movies_df):
        """
        Bootstrap the bandit with historical data so it doesn't start completely blind.
        We use vote_count as impressions and (vote_average/10 * vote_count) as clicks.
        """
        logger.info("Injecting historical priors into Bandit Engine...")
        try:
            for row in movies_df.to_dict(orient="records"):
                movie_id = int(row["id"])
                votes = int(row.get("vote_count") or 0)
                rating = float(row.get("vote_average") or 0.0)

                # Scale down slightly to allow new data to easily overtake history
                prior_impressions = min(votes, 1000)
                prior_clicks = int(prior_impressions * (rating / 10.0))

                self.item_impressions[movie_id] = prior_impressions
                self.item_clicks[movie_id] = prior_clicks
                self.total_impressions += prior_impressions
            logger.info(f"Bandit initialized with priors for {len(self.item_impressions)} items.")
        except Exception as e:
            logger.error(f"Failed to inject bandit priors: {e}")

    def update_reward(self, movie_id: int, clicked: bool):
        """
        Updates the bandit state based on real user feedback.
        Called asynchronously by the /v1/events endpoint.
        """
        self.item_impressions[movie_id] += 1
        self.total_impressions += 1
        if clicked:
            self.item_clicks[movie_id] += 1

    def get_ucb_score(self, movie_id: int, base_score: float, c: float = 0.5) -> float:
        """
        Upper Confidence Bound (UCB1).
        Favors items with high base scores AND items with very few impressions.
        c controls the degree of exploration.
        """
        impressions = self.item_impressions.get(movie_id, 0)
        if impressions == 0:
            return base_score + 100.0  # Force explore completely unseen items

        # The UCB exploration term
        exploration_bonus = c * np.sqrt(np.log(max(self.total_impressions, 1)) / impressions)
        return base_score + exploration_bonus

    def get_thompson_sample(self, movie_id: int, base_score: float) -> float:
        """
        Thompson Sampling using Beta distribution.
        Draws a random sample from the item's posterior distribution.
        If an item is uncertain (low impressions), the distribution is wide, allowing
        it to occasionally sample a very high score and get shown.
        """
        impressions = self.item_impressions.get(movie_id, 0)
        clicks = self.item_clicks.get(movie_id, 0)

        # Beta(alpha, beta) where alpha = successes + 1, beta = failures + 1
        alpha = clicks + 1
        beta_param = (impressions - clicks) + 1

        # Draw sample
        ts_multiplier = np.random.beta(alpha, beta_param)

        # Blend TS sample with the base score (base score from MMoE)
        return base_score * (0.5 + ts_multiplier)

    def apply_exploration(
        self, candidates: list[dict[str, Any]], strategy: str = "thompson", epsilon: float = 0.1
    ) -> list[dict[str, Any]]:
        """
        Modifies candidate scores to inject exploration.
        """
        if not candidates:
            return candidates

        if strategy == "epsilon_greedy":
            # Epsilon-Greedy: With probability epsilon, randomly boost an item
            for candidate in candidates:
                if np.random.random() < epsilon:
                    # Massive boost to force it to the top
                    candidate["similarity_score"] = float(candidate.get("similarity_score", 0.0)) + 5.0
                    if "explanation" not in candidate:
                        candidate["explanation"] = []
                    candidate["explanation"].insert(0, "Exploration: Epsilon-Greedy Random Selection")

        elif strategy == "ucb":
            for candidate in candidates:
                movie_id = int(candidate.get("id", 0))
                base_score = float(candidate.get("similarity_score", 0.0))

                new_score = self.get_ucb_score(movie_id, base_score)
                if new_score > base_score + 0.1:  # Only label if it actually got a big UCB boost
                    if "explanation" not in candidate:
                        candidate["explanation"] = []
                    candidate["explanation"].insert(0, "Exploration: Upper Confidence Bound (UCB) Boost")
                candidate["similarity_score"] = new_score

        elif strategy == "thompson":
            for candidate in candidates:
                movie_id = int(candidate.get("id", 0))
                base_score = float(candidate.get("similarity_score", 0.0))

                new_score = self.get_thompson_sample(movie_id, base_score)
                candidate["similarity_score"] = new_score
                # Thompson is stochastic, hard to explicitly label as "explored",
                # but we can note it in the metrics
                candidate.setdefault("metrics", {})["thompson_applied"] = True

        # Re-sort after exploration adjustments
        candidates.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return candidates


# Singleton Instance
_bandit_engine = None


def get_bandit_engine() -> BanditEngine:
    global _bandit_engine
    if _bandit_engine is None:
        _bandit_engine = BanditEngine()
    return _bandit_engine
