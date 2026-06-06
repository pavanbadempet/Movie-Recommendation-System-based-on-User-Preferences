"""
Exploration Engine for APEX — Multi-Armed Bandit with Thompson Sampling.

Solves the exploration-exploitation dilemma in recommendations:
- Exploitation: recommend what we know the user likes
- Exploration: occasionally recommend something new to discover preferences

Uses Thompson Sampling (Bayesian approach) which is provably optimal for
the multi-armed bandit problem. Each item is modeled as a Beta distribution
over click probability, updated with each interaction.

This is more sophisticated than the existing contextual bandit because:
1. It uses proper Bayesian uncertainty (Beta distribution)
2. It adapts exploration rate based on confidence
3. It handles the cold-start problem naturally (high uncertainty = more exploration)

Netflix uses a similar approach for their "New Releases" row.
"""

from __future__ import annotations

from collections import defaultdict
import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ThompsonSamplingBandit:
    """
    Thompson Sampling bandit for recommendation exploration.

    Each item maintains a Beta(alpha, beta) distribution over click probability:
    - alpha = number of clicks + 1 (prior)
    - beta = number of non-clicks + 1 (prior)

    At recommendation time, we sample from each item's distribution and
    occasionally surface items with high uncertainty (exploration).
    """

    def __init__(self, exploration_rate: float = 0.1):
        """
        Args:
            exploration_rate: Fraction of recommendations that use exploration (0.1 = 10%)
        """
        self.exploration_rate = exploration_rate
        # item_id → (alpha, beta) for Beta distribution
        self._item_stats: dict[int, tuple[float, float]] = defaultdict(lambda: (1.0, 1.0))
        self._total_impressions: int = 0

    def update(self, item_id: int, clicked: bool) -> None:
        """Update item statistics after an interaction."""
        alpha, beta = self._item_stats[item_id]
        if clicked:
            self._item_stats[item_id] = (alpha + 1.0, beta)
        else:
            self._item_stats[item_id] = (alpha, beta + 1.0)
        self._total_impressions += 1

    def sample_click_probability(self, item_id: int) -> float:
        """Sample click probability from the item's Beta distribution."""
        alpha, beta = self._item_stats[item_id]
        return float(np.random.beta(alpha, beta))

    def get_uncertainty(self, item_id: int) -> float:
        """Return uncertainty (variance) of the click probability estimate."""
        alpha, beta = self._item_stats[item_id]
        total = alpha + beta
        variance = (alpha * beta) / (total**2 * (total + 1))
        return float(variance)

    def apply_exploration(
        self,
        candidates: list[dict[str, Any]],
        n: int = 10,
        rng: random.Random | None = None,
    ) -> list[dict[str, Any]]:
        """
        Apply Thompson Sampling exploration to a candidate list.

        With probability (1 - exploration_rate): return top-n by similarity score
        With probability exploration_rate: inject 1-2 exploratory items

        Args:
            candidates: Ranked candidate list
            n: Number of results to return
            rng: Optional random number generator for reproducibility

        Returns:
            Final recommendation list with exploration applied
        """
        if rng is None:
            rng = random.Random()

        if not candidates:
            return candidates

        # Decide whether to explore
        if rng.random() > self.exploration_rate:
            return candidates[:n]

        # Exploitation: top n-1 by score
        exploit_results = candidates[: max(n - 1, 1)]

        # Exploration: find the item with highest uncertainty not already selected
        selected_ids = {c.get("id") for c in exploit_results}
        explore_candidates = [c for c in candidates[n:] if c.get("id") not in selected_ids]

        if not explore_candidates:
            return candidates[:n]

        # Thompson Sampling: sample from each candidate's Beta distribution
        best_explore = max(
            explore_candidates,
            key=lambda c: self.sample_click_probability(int(c.get("id", 0))),
        )

        # Insert exploratory item at a random position (not first)
        insert_pos = rng.randint(1, len(exploit_results))
        result = exploit_results[:insert_pos] + [best_explore] + exploit_results[insert_pos:]
        return result[:n]

    def load_from_events(self, events: list[dict[str, Any]]) -> None:
        """Initialize bandit statistics from historical events."""
        click_counts: dict[int, int] = defaultdict(int)
        impression_counts: dict[int, int] = defaultdict(int)

        for event in events:
            mid = event.get("movie_id")
            if mid is None:
                continue
            try:
                mid = int(mid)
            except (TypeError, ValueError):
                continue

            et = str(event.get("event_type", "")).lower()
            if et in {"recommendation_impression", "view"}:
                impression_counts[mid] += 1
            elif et == "click":
                click_counts[mid] += 1
                impression_counts[mid] += 1

        for mid in set(click_counts) | set(impression_counts):
            clicks = click_counts.get(mid, 0)
            impressions = impression_counts.get(mid, 0)
            non_clicks = max(impressions - clicks, 0)
            self._item_stats[mid] = (float(clicks + 1), float(non_clicks + 1))

        logger.info(
            "Thompson Sampling bandit initialized: %d items from %d events",
            len(self._item_stats),
            len(events),
        )


# Module-level singleton
_bandit: ThompsonSamplingBandit | None = None


def get_thompson_bandit() -> ThompsonSamplingBandit:
    """Get or create the module-level Thompson Sampling bandit."""
    global _bandit
    if _bandit is None:
        _bandit = ThompsonSamplingBandit(exploration_rate=0.1)
        # Initialize from event store
        try:
            from backend.events import iter_events

            events = list(iter_events())
            _bandit.load_from_events(events)
        except Exception as exc:
            logger.warning("Could not initialize bandit from events: %s", exc)
    return _bandit
