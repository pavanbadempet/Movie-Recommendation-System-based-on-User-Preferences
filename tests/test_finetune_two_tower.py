"""
Property-based and unit tests for scripts/finetune_two_tower.py pair construction.

Feature: apex-peak-capability, Property 5 & 6: Fine-Tuning Negative Ratio
and Positive Pair Filter
Validates: Requirements 4.1, 4.2
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Ensure scripts/ is importable
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.finetune_two_tower import extract_positive_pairs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EVENT_TYPES = ["rating", "click", "view", "search", "recommendation_request"]


def _make_event(event_type: str, rating: float | None = None, user_id: str = "u1", movie_id: int = 1) -> dict:
    e = {"event_type": event_type, "user_id": user_id, "movie_id": movie_id}
    if rating is not None:
        e["rating"] = rating
    return e


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(
    events=st.lists(
        st.fixed_dictionaries({
            "event_type": st.sampled_from(EVENT_TYPES),
            "rating": st.one_of(st.none(), st.floats(min_value=1.0, max_value=5.0, allow_nan=False)),
            "user_id": st.just("u1"),
            "movie_id": st.integers(min_value=1, max_value=1000),
        }),
        min_size=0,
        max_size=50,
    )
)
@settings(max_examples=100, deadline=None)
def test_positive_pair_filter_property(events: list[dict]):
    """
    Feature: apex-peak-capability, Property 6
    Only rating>=3.5 or click events produce positive pairs; no other types.
    """
    with patch("scripts.finetune_two_tower.iter_events", return_value=iter(events)):
        pairs = extract_positive_pairs()

    # Verify each pair came from a qualifying event
    qualifying_movie_ids = set()
    for e in events:
        et = e.get("event_type", "")
        r = e.get("rating")
        mid = e.get("movie_id")
        uid = e.get("user_id")
        if mid is None or uid is None:
            continue
        if et == "click":
            qualifying_movie_ids.add(mid)
        elif et == "rating" and r is not None and float(r) >= 3.5:
            qualifying_movie_ids.add(mid)

    for _, mid in pairs:
        assert mid in qualifying_movie_ids, (
            f"movie_id {mid} in pairs but not from a qualifying event"
        )


@given(
    num_positives=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=50, deadline=None)
def test_negative_ratio_property(num_positives: int):
    """
    Feature: apex-peak-capability, Property 5
    TwoTowerDataset produces exactly num_negatives=4 negatives per positive.
    """
    from scripts.train_two_tower import TwoTowerDataset
    import pandas as pd

    num_negatives = 4
    num_items = 200

    # Build minimal features
    user_features = {0: np.zeros(18, dtype=np.float32)}
    item_features = {i: np.zeros(20, dtype=np.float32) for i in range(num_items)}

    rows = [(0, i, 4.0) for i in range(num_positives)]
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])

    dataset = TwoTowerDataset(
        ratings_df=df,
        user_features=user_features,
        item_features=item_features,
        num_negatives=num_negatives,
    )

    for i in range(len(dataset)):
        _, _, neg_feats = dataset[i]
        assert neg_feats.shape[0] == num_negatives, (
            f"Expected {num_negatives} negatives, got {neg_feats.shape[0]}"
        )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestPositivePairExtraction:
    def test_50_qualifying_events_returns_50_pairs(self):
        """50 qualifying events → 50 pairs returned."""
        events = [_make_event("click", movie_id=i) for i in range(1, 51)]
        with patch("scripts.finetune_two_tower.iter_events", return_value=iter(events)):
            pairs = extract_positive_pairs()
        assert len(pairs) == 50

    def test_click_without_rating_is_included(self):
        """click event with no rating → included as positive pair."""
        events = [_make_event("click", rating=None, movie_id=42)]
        with patch("scripts.finetune_two_tower.iter_events", return_value=iter(events)):
            pairs = extract_positive_pairs()
        assert len(pairs) == 1
        assert pairs[0][1] == 42

    def test_rating_below_threshold_excluded(self):
        """rating=3.0 → NOT included."""
        events = [_make_event("rating", rating=3.0, movie_id=10)]
        with patch("scripts.finetune_two_tower.iter_events", return_value=iter(events)):
            pairs = extract_positive_pairs()
        assert len(pairs) == 0

    def test_view_events_excluded(self):
        """view events → never produce positive pairs."""
        events = [_make_event("view", movie_id=5)]
        with patch("scripts.finetune_two_tower.iter_events", return_value=iter(events)):
            pairs = extract_positive_pairs()
        assert len(pairs) == 0

    def test_rating_at_threshold_included(self):
        """rating=3.5 (boundary) → included."""
        events = [_make_event("rating", rating=3.5, movie_id=99)]
        with patch("scripts.finetune_two_tower.iter_events", return_value=iter(events)):
            pairs = extract_positive_pairs()
        assert len(pairs) == 1

    def test_search_events_excluded(self):
        """search events → never produce positive pairs."""
        events = [{"event_type": "search", "user_id": "u1", "movie_id": 1, "query_text": "action"}]
        with patch("scripts.finetune_two_tower.iter_events", return_value=iter(events)):
            pairs = extract_positive_pairs()
        assert len(pairs) == 0

    def test_none_user_id_excluded(self):
        """Events with None user_id → excluded."""
        events = [{"event_type": "click", "user_id": None, "movie_id": 1}]
        with patch("scripts.finetune_two_tower.iter_events", return_value=iter(events)):
            pairs = extract_positive_pairs()
        assert len(pairs) == 0

    def test_none_movie_id_excluded(self):
        """Events with None movie_id → excluded."""
        events = [{"event_type": "click", "user_id": "u1", "movie_id": None}]
        with patch("scripts.finetune_two_tower.iter_events", return_value=iter(events)):
            pairs = extract_positive_pairs()
        assert len(pairs) == 0
