"""
Property-based tests for RankingPipeline invariants.
# Feature: architecture-design-perfection, Property 4/5/6/7
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.pipeline_types import CandidateItem
from backend.ranking_pipeline import RankingConfig, RankingPipeline


def _candidate_strategy():
    return st.builds(
        CandidateItem,
        movie_id=st.integers(min_value=0, max_value=100000),
        retrieval_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        retrieval_source=st.sampled_from(["faiss", "tfidf", "knowledge_graph", "hybrid"]),
        metadata=st.just({}),
    )


def _make_pipeline():
    config = RankingConfig(use_neural_ensemble=False, use_learned_ranker=False)
    return RankingPipeline(ensemble_engine=None, learned_ranker=None, config=config)


# Property 4: Ranking Count Preservation
# Validates: Requirements 4.1
@given(st.lists(_candidate_strategy(), min_size=0, max_size=50, unique_by=lambda c: c.movie_id))
@settings(max_examples=100)
def test_ranking_count_preservation(candidates):
    """len(result) == len(candidates) for any input list."""
    # Feature: architecture-design-perfection, Property 4: Ranking Count Preservation
    pipeline = _make_pipeline()
    result = pipeline.rank(candidates, user_context={})
    assert len(result) == len(candidates), f"Expected {len(candidates)}, got {len(result)}"


# Property 5: Ranking Set-Identity Round-Trip
# Validates: Requirements 4.2
@given(st.lists(_candidate_strategy(), min_size=0, max_size=50, unique_by=lambda c: c.movie_id))
@settings(max_examples=100)
def test_ranking_set_identity_round_trip(candidates):
    """The set of movie_ids in the result equals the set of movie_ids in the input."""
    # Feature: architecture-design-perfection, Property 5: Ranking Set-Identity Round-Trip
    pipeline = _make_pipeline()
    result = pipeline.rank(candidates, user_context={})
    result_ids = {r.movie_id for r in result}
    candidate_ids = {c.movie_id for c in candidates}
    assert result_ids == candidate_ids, f"movie_id sets differ: result={result_ids}, candidates={candidate_ids}"


# Property 6: Ranking Ordering Invariant
# Validates: Requirements 4.3
@given(st.lists(_candidate_strategy(), min_size=2, max_size=50, unique_by=lambda c: c.movie_id))
@settings(max_examples=100)
def test_ranking_ordering_invariant(candidates):
    """Result is sorted descending by blended score — no two adjacent items are out of order.

    # Feature: architecture-design-perfection, Property 6: Ranking Ordering Invariant
    """
    pipeline = _make_pipeline()
    result = pipeline.rank(candidates, user_context={})
    if len(result) < 2:
        return
    scores = [r.ensemble_score + r.ranker_score for r in result]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1] - 1e-9, (
            f"Out-of-order at position {i}: score[{i}]={scores[i]} < score[{i + 1}]={scores[i + 1]}"
        )


# Property 7: Ranking Determinism
# Validates: Requirements 4.4
@given(st.lists(_candidate_strategy(), min_size=0, max_size=50, unique_by=lambda c: c.movie_id))
@settings(max_examples=100)
def test_ranking_determinism(candidates):
    """Calling rank() twice with identical inputs produces identical scores.

    # Feature: architecture-design-perfection, Property 7: Ranking Determinism
    """
    pipeline = _make_pipeline()
    result1 = pipeline.rank(candidates, user_context={})
    result2 = pipeline.rank(candidates, user_context={})
    ids1 = [r.movie_id for r in result1]
    ids2 = [r.movie_id for r in result2]
    assert ids1 == ids2, f"Non-deterministic ranking: {ids1} != {ids2}"
    scores1 = [r.ensemble_score for r in result1]
    scores2 = [r.ensemble_score for r in result2]
    for s1, s2 in zip(scores1, scores2, strict=True):
        assert abs(s1 - s2) < 1e-9, f"Score mismatch: {s1} != {s2}"
