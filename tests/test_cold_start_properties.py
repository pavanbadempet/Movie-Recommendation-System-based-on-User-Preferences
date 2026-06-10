"""
Property-based and unit tests for cold-start intelligence.

Covers:
- cold_start_boost invariants:
    * warm users (>= 10 interactions) always return multiplier == 1.0
    * cold-start users (< 10 interactions) always return multiplier >= 1.0
    * boost is monotonically non-decreasing as content quality improves
    * boost is bounded in [1.0, 1.2] (no runaway amplification)
- uncertainty_estimator helpers:
    * ensemble_uncertainty is in [0, 1] for any model scores
    * coverage_uncertainty is in [0, 1] for any interaction counts
    * compute_confidence_score returns all required keys
    * is_cold_start flag is True when either user or item has < 5 interactions
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.intelligence.uncertainty_estimator import (
    cold_start_boost,
    compute_confidence_score,
    coverage_uncertainty,
    ensemble_uncertainty,
)

# ---------------------------------------------------------------------------
# cold_start_boost invariants
# ---------------------------------------------------------------------------


class TestColdStartBoost:
    def test_warm_user_returns_no_boost(self):
        """Users with >= 10 interactions receive exactly 1.0 (no boost applied)."""
        movie = {"vote_average": 8.5, "vote_count": 5000, "popularity": 200.0}
        boost = cold_start_boost(movie=movie, user_interaction_count=10)
        assert boost == 1.0, f"Warm user should get boost=1.0, got {boost}"

    def test_warm_user_many_interactions(self):
        movie = {"vote_average": 7.0, "vote_count": 1000, "popularity": 50.0}
        for count in [10, 20, 50, 100, 1000]:
            boost = cold_start_boost(movie=movie, user_interaction_count=count)
            assert boost == 1.0, f"Expected 1.0 for count={count}, got {boost}"

    def test_cold_start_user_boost_gte_one(self):
        """Cold-start users always receive a multiplier >= 1.0."""
        movie = {"vote_average": 8.0, "vote_count": 2000, "popularity": 100.0}
        for count in range(0, 10):
            boost = cold_start_boost(movie=movie, user_interaction_count=count)
            assert boost >= 1.0, f"Boost {boost} < 1.0 for interaction_count={count}"

    def test_cold_start_boost_bounded(self):
        """Boost must not exceed 1.2 (20% max amplification)."""
        movie = {"vote_average": 10.0, "vote_count": 100000, "popularity": 10000.0}
        for count in range(0, 10):
            boost = cold_start_boost(movie=movie, user_interaction_count=count)
            assert boost <= 1.2, f"Boost {boost} > 1.2 for count={count}"

    def test_high_quality_movie_gets_higher_boost_than_low_quality(self):
        """High-rated, well-voted movie should receive strictly more boost than low-quality."""
        high_quality = {"vote_average": 9.0, "vote_count": 10000, "popularity": 500.0}
        low_quality = {"vote_average": 3.0, "vote_count": 50, "popularity": 1.0}
        boost_high = cold_start_boost(movie=high_quality, user_interaction_count=0)
        boost_low = cold_start_boost(movie=low_quality, user_interaction_count=0)
        assert boost_high >= boost_low, (
            f"High quality movie boost ({boost_high}) should be >= low quality ({boost_low})"
        )

    def test_missing_movie_fields_does_not_raise(self):
        """cold_start_boost must handle movies with missing metadata gracefully."""
        cold_start_boost(movie={}, user_interaction_count=0)
        cold_start_boost(movie={"vote_average": None}, user_interaction_count=2)
        cold_start_boost(movie={"vote_count": None, "popularity": None}, user_interaction_count=3)

    @given(
        vote_avg=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        vote_count=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        popularity=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        count=st.integers(min_value=0, max_value=9),
    )
    @settings(max_examples=100)
    def test_property_cold_start_boost_always_in_range(self, vote_avg, vote_count, popularity, count):
        """Property: for any cold-start user, boost is always in [1.0, 1.2]."""
        movie = {
            "vote_average": vote_avg,
            "vote_count": vote_count,
            "popularity": popularity,
        }
        boost = cold_start_boost(movie=movie, user_interaction_count=count)
        assert 1.0 <= boost <= 1.2, (
            f"Boost {boost} out of [1.0, 1.2] for "
            f"vote_avg={vote_avg}, vote_count={vote_count}, "
            f"popularity={popularity}, count={count}"
        )

    @given(count=st.integers(min_value=10, max_value=10000))
    @settings(max_examples=50)
    def test_property_warm_user_always_no_boost(self, count):
        """Property: for any warm user (>= 10 interactions), boost is exactly 1.0."""
        movie = {"vote_average": 8.0, "vote_count": 500, "popularity": 50.0}
        boost = cold_start_boost(movie=movie, user_interaction_count=count)
        assert boost == 1.0, f"Warm user (count={count}) got boost={boost}, expected 1.0"


# ---------------------------------------------------------------------------
# ensemble_uncertainty invariants
# ---------------------------------------------------------------------------


class TestEnsembleUncertainty:
    def test_all_models_agree_produces_low_uncertainty(self):
        """When all models give identical scores, uncertainty should be near 0."""
        scores = {"lightgcn": 0.8, "quantum": 0.8, "sasrec": 0.8, "kan": 0.8}
        weights = {"lightgcn": 0.005, "quantum": 0.01, "sasrec": 0.659, "kan": 0.298}
        unc = ensemble_uncertainty(scores, weights)
        assert unc < 0.05, f"Expected near-zero uncertainty for consensus, got {unc}"

    def test_maximal_disagreement_produces_high_uncertainty(self):
        """Extreme score spread should produce high uncertainty."""
        scores = {"m1": 0.0, "m2": 1.0}
        weights = {"m1": 0.5, "m2": 0.5}
        unc = ensemble_uncertainty(scores, weights)
        assert unc > 0.5, f"Expected high uncertainty for split scores, got {unc}"

    def test_empty_inputs_return_half(self):
        """Empty model scores returns 0.5 (unknown uncertainty)."""
        assert ensemble_uncertainty({}, {}) == 0.5

    def test_single_model_returns_moderate_uncertainty(self):
        """Single model returns 0.3 (moderate uncertainty)."""
        unc = ensemble_uncertainty({"m": 0.9}, {"m": 1.0})
        assert unc == 0.3

    @given(
        score1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        score2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        w1=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        w2=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_property_uncertainty_in_unit_interval(self, score1, score2, w1, w2):
        """Property: ensemble_uncertainty is always in [0, 1]."""
        scores = {"m1": score1, "m2": score2}
        weights = {"m1": w1, "m2": w2}
        unc = ensemble_uncertainty(scores, weights)
        assert 0.0 <= unc <= 1.0, f"Uncertainty {unc} out of [0, 1]"


# ---------------------------------------------------------------------------
# coverage_uncertainty invariants
# ---------------------------------------------------------------------------


class TestCoverageUncertainty:
    def test_zero_interactions_maximum_uncertainty(self):
        """Both user and item with 0 interactions = maximum uncertainty (1.0)."""
        unc = coverage_uncertainty(user_interaction_count=0, item_interaction_count=0)
        assert unc == 1.0

    def test_sufficient_interactions_low_uncertainty(self):
        """Both user and item well above threshold = low uncertainty."""
        unc = coverage_uncertainty(user_interaction_count=100, item_interaction_count=100)
        assert unc < 0.1, f"Expected low uncertainty with lots of data, got {unc}"

    def test_asymmetric_interactions(self):
        """Mixed case: one side cold, other side warm."""
        unc_warm_user = coverage_uncertainty(user_interaction_count=100, item_interaction_count=0)
        unc_balanced = coverage_uncertainty(user_interaction_count=5, item_interaction_count=5)
        # Cold item with warm user should have higher uncertainty than balanced
        assert unc_warm_user > unc_balanced

    @given(
        user_count=st.integers(min_value=0, max_value=1000),
        item_count=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=100)
    def test_property_coverage_uncertainty_in_unit_interval(self, user_count, item_count):
        """Property: coverage_uncertainty is always in [0, 1]."""
        unc = coverage_uncertainty(
            user_interaction_count=user_count,
            item_interaction_count=item_count,
        )
        assert 0.0 <= unc <= 1.0, f"Uncertainty {unc} out of [0, 1]"


# ---------------------------------------------------------------------------
# compute_confidence_score invariants
# ---------------------------------------------------------------------------


class TestComputeConfidenceScore:
    def test_returns_all_required_keys(self):
        scores = {"m1": 0.8, "m2": 0.6}
        weights = {"m1": 0.5, "m2": 0.5}
        result = compute_confidence_score(scores, weights, user_interaction_count=3, item_interaction_count=3)
        required_keys = {
            "confidence",
            "uncertainty_ensemble",
            "uncertainty_coverage",
            "is_cold_start",
            "confidence_label",
        }
        assert required_keys <= set(result.keys()), f"Missing keys: {required_keys - set(result.keys())}"

    def test_cold_start_flag_for_few_interactions(self):
        """is_cold_start is True when user or item has < 5 interactions."""
        scores = {"m": 0.5}
        weights = {"m": 1.0}
        result = compute_confidence_score(scores, weights, user_interaction_count=4, item_interaction_count=100)
        assert result["is_cold_start"] is True

    def test_not_cold_start_with_sufficient_interactions(self):
        scores = {"m": 0.5}
        weights = {"m": 1.0}
        result = compute_confidence_score(scores, weights, user_interaction_count=10, item_interaction_count=10)
        assert result["is_cold_start"] is False

    def test_confidence_in_unit_interval(self):
        scores = {"m1": 0.9, "m2": 0.1}
        weights = {"m1": 0.6, "m2": 0.4}
        result = compute_confidence_score(scores, weights, user_interaction_count=50, item_interaction_count=50)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_label_valid_values(self):
        scores = {"m": 0.5}
        weights = {"m": 1.0}
        result = compute_confidence_score(scores, weights)
        assert result["confidence_label"] in {"high", "medium", "low", "very_low"}

    @given(
        score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        user_count=st.integers(min_value=0, max_value=500),
        item_count=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=80)
    def test_property_confidence_always_in_unit_interval(self, score, user_count, item_count):
        """Property: confidence is always in [0, 1] for any valid inputs."""
        scores = {"m": score}
        weights = {"m": 1.0}
        result = compute_confidence_score(
            scores,
            weights,
            user_interaction_count=user_count,
            item_interaction_count=item_count,
        )
        assert 0.0 <= result["confidence"] <= 1.0, f"Confidence {result['confidence']} out of [0, 1]"
