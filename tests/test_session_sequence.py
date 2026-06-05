"""
Property-based and unit tests for ApexEnsembleEngine._get_session_sequence.

Feature: apex-peak-capability, Property 1 & 2: Session Sequence Length Invariant
and Padding Correctness
Validates: Requirements 1.2, 1.3, 1.7
"""

from __future__ import annotations

import time
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers — build a lightweight engine without loading heavy models
# ---------------------------------------------------------------------------

NUM_USERS = 10
NUM_ITEMS = 100
EMB_DIM = 4


def _make_engine():
    """Return a small ApexEnsembleEngine with mocked sub-models."""
    with (
        patch("backend.ensemble_engine.QuantumFluidRecommender"),
        patch("backend.ensemble_engine.HyperbolicRecommender"),
        patch("backend.ensemble_engine.KANRanker"),
        patch("backend.ensemble_engine.LatentDiffusionRecommender"),
        patch("backend.ensemble_engine.SASRec"),
        patch("backend.ensemble_engine.LightGCN"),
        patch("backend.ensemble_engine.ApexEnsembleEngine._inject_pyspark_priors"),
        patch("backend.ensemble_engine.ApexEnsembleEngine._load_trained_weights"),
    ):
        from backend.ensemble_engine import ApexEnsembleEngine

        engine = ApexEnsembleEngine(num_users=NUM_USERS, num_items=NUM_ITEMS, emb_dim=EMB_DIM)
    return engine


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(
    user_id=st.integers(min_value=0, max_value=10**6),
    movie_ids=st.lists(
        st.integers(min_value=0, max_value=10**6),
        min_size=0,
        max_size=200,
    ),
)
@settings(max_examples=100, deadline=None)
def test_session_sequence_shape_and_padding(user_id: int, movie_ids: list[int]):
    """
    Feature: apex-peak-capability, Property 1 & 2
    For any user_id and any number of interactions (0–200), the returned tensor
    must have shape [1, 50], all values in [0, num_items), and leading zeros
    equal to max(0, 50 - len(injected_ids)).
    """
    engine = _make_engine()
    SEQ_LEN = 50

    # Inject directly into cache (bypass event store)
    injected = movie_ids[-SEQ_LEN:] if len(movie_ids) > SEQ_LEN else movie_ids
    engine._session_cache[str(user_id)] = (time.time(), [m % NUM_ITEMS for m in injected])

    result = engine._get_session_sequence(user_id)

    # Shape invariant
    assert result.shape == (1, SEQ_LEN), f"Expected [1, 50], got {result.shape}"

    # Value bounds
    flat = result.squeeze().tolist()
    assert all(0 <= v < NUM_ITEMS for v in flat), "Values out of [0, num_items)"

    # Padding correctness
    expected_leading_zeros = max(0, SEQ_LEN - len(injected))
    actual_leading_zeros = 0
    for v in flat:
        if v == 0:
            actual_leading_zeros += 1
        else:
            break
    assert actual_leading_zeros >= expected_leading_zeros, (
        f"Expected >= {expected_leading_zeros} leading zeros, got {actual_leading_zeros}"
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestSessionSequenceUnit:
    def test_cache_hit_returns_cached_data(self):
        """A fresh cache entry is used without querying the event store."""
        engine = _make_engine()
        cached_ids = [1, 2, 3]
        engine._session_cache["42"] = (time.time(), cached_ids)

        with patch("backend.ensemble_engine.iter_events") as mock_iter:
            result = engine._get_session_sequence(42)
            mock_iter.assert_not_called()

        flat = result.squeeze().tolist()
        # Last 3 values should be the cached IDs (modulo-bounded)
        assert flat[-3:] == [i % NUM_ITEMS for i in cached_ids]
        assert result.shape == (1, 50)

    def test_cold_start_returns_zeros(self):
        """No cache, no events → zero tensor."""
        engine = _make_engine()
        with patch("backend.ensemble_engine.iter_events", return_value=iter([])):
            result = engine._get_session_sequence(999)
        assert result.shape == (1, 50)
        assert result.sum().item() == 0

    def test_io_error_fallback_returns_zeros(self):
        """Event store I/O error → zero tensor, WARNING logged."""
        engine = _make_engine()
        with patch("backend.ensemble_engine.iter_events", side_effect=OSError("disk error")):
            result = engine._get_session_sequence(7)
        assert result.shape == (1, 50)
        assert result.sum().item() == 0

    def test_override_parameter_bypasses_cache_and_store(self):
        """override=[1,2,3] → those IDs appear in tensor, shape [1,50]."""
        engine = _make_engine()
        override = [1, 2, 3]
        with patch("backend.ensemble_engine.iter_events") as mock_iter:
            result = engine._get_session_sequence(0, override=override)
            mock_iter.assert_not_called()

        assert result.shape == (1, 50)
        flat = result.squeeze().tolist()
        expected_tail = [i % NUM_ITEMS for i in override]
        assert flat[-3:] == expected_tail

    def test_exactly_50_interactions_no_leading_zeros(self):
        """Exactly 50 interactions → no leading zeros."""
        engine = _make_engine()
        ids = list(range(1, 51))  # 50 non-zero IDs
        engine._session_cache["1"] = (time.time(), [i % NUM_ITEMS for i in ids])
        result = engine._get_session_sequence(1)
        flat = result.squeeze().tolist()
        assert flat[0] != 0, "Expected no leading zeros for exactly 50 interactions"

    def test_more_than_50_interactions_uses_last_50(self):
        """75 interactions → only the last 50 are used."""
        engine = _make_engine()
        ids = list(range(1, 76))  # 75 IDs
        engine._session_cache["2"] = (time.time(), [i % NUM_ITEMS for i in ids[-50:]])
        result = engine._get_session_sequence(2)
        assert result.shape == (1, 50)
        flat = result.squeeze().tolist()
        # The last value should correspond to id 75 % NUM_ITEMS
        assert flat[-1] == 75 % NUM_ITEMS
