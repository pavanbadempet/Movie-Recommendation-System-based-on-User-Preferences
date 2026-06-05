import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RLRewardEngine:
    """
    Defines the Markov Decision Process (MDP) for the Recommendation Agent.
    Calculates delayed rewards prioritizing long-term user retention over short-term clicks.
    """

    def __init__(self):
        # Hyperparameters for reward shaping
        self.w_click = 0.1  # Small reward for immediate engagement
        self.w_rating = 0.4  # Medium reward for explicitly liking the content
        self.w_session = 0.2  # Reward for watching multiple movies in one sitting
        self.w_retention = 1.0  # Massive reward if the user returns within 7 days
        self.penalty_dislike = -2.0  # Massive penalty for recommending something they hate

    def calculate_reward(self, interaction: dict[str, Any], future_interactions: list[dict[str, Any]]) -> float:
        """
        Calculates the Q-value reward for a specific recommendation action.

        Args:
            interaction: The immediate event (e.g., click, rating)
            future_interactions: Look-ahead events for this user to calculate 7-day retention
        """
        event_type = interaction.get("event_type", "view")
        rating = float(interaction.get("rating", 0.0))

        reward = 0.0

        # 1. Immediate Reward
        if event_type == "click":
            reward += self.w_click

        if event_type == "rating":
            if rating >= 4.0:
                reward += self.w_rating * rating
            elif rating <= 2.0:
                # Negative feedback
                reward += self.penalty_dislike

        # 2. Delayed Reward (Retention)
        # Check if the user had another session between 1 and 7 days after this interaction
        current_time = interaction.get("timestamp", 0)
        for future_event in future_interactions:
            dt = future_event.get("timestamp", 0) - current_time
            if 86400 <= dt <= (86400 * 7):  # Between 1 and 7 days
                reward += self.w_retention
                break  # Only reward once for retention

        return reward

    def build_state_representation(
        self, user_profile: dict[str, Any], recent_history: list[dict[str, Any]], user_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Constructs the State (S_t) for the Actor-Critic network.
        Fuses static embeddings with dynamic temporal features.
        Returns a flat 1D numpy array.
        """
        # User base embedding (e.g., 768d SBERT aggregation)
        base_emb = np.array(user_embedding, dtype=np.float32).flatten()

        # Dynamic feature: Average recent rating
        recent_ratings = [e.get("rating", 3.0) for e in recent_history if e.get("event_type") == "rating"]
        avg_rating = np.mean(recent_ratings) if recent_ratings else 0.0

        # Dynamic feature: Interaction velocity (events in last 24h)
        # Assume history is sorted, check timestamps
        velocity = min(len(recent_history) / 10.0, 1.0)  # Normalized 0-1

        # Time context (e.g. weekend vs weekday, could affect willingness to watch long movies)
        # We'll mock a simple binary feature for weekend
        is_weekend = 1.0 if user_profile.get("is_weekend", False) else 0.0

        dynamic_features = np.array([avg_rating, velocity, is_weekend], dtype=np.float32)

        # Concat into final State vector
        state = np.concatenate([base_emb, dynamic_features])
        return state
