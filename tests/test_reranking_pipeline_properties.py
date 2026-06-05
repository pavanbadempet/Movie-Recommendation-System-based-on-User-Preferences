"""
Property-based tests for RerankingPipeline invariants.
# Feature: architecture-design-perfection, Property 8/9
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.pipeline_types import RankedItem
from backend.reranking_pipeline import RerankingConfig, RerankingPipeline


def _ranked_item_strategy():
    return st.builds(
        RankedItem,
        movie_id=st.integers(min_value=0, max_value=100000),
        retrieval_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        retrieval_source=st.sampled_from(["faiss", "tfidf", "knowledge_graph", "hybrid"]),
        ensemble_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ranker_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        final_rank=st.integers(min_value=1, max_value=1000),
        retrieval_signals=st.just({}),
        metadata=st.just({}),
    )


def _make_pipeline():
    config = RerankingConfig(enable_llm_reranking=False, enable_rl_safety=False)
    return RerankingPipeline(rl_policy=None, llm_client=None, config=config)


# Property 8: Reranking No-Hallucination
# Validates: Requirements 5.1
@given(st.lists(_ranked_item_strategy(), min_size=0, max_size=50, unique_by=lambda r: r.movie_id))
@settings(max_examples=100)
def test_reranking_no_hallucination(ranked_items):
    """Output movie_ids must be a subset of input movie_ids — no hallucinated items.

    # Feature: architecture-design-perfection, Property 8: Reranking No-Hallucination
    """
    pipeline = _make_pipeline()
    result = pipeline.rerank(ranked_items, constraints={})
    input_ids = {r.movie_id for r in ranked_items}
    output_ids = {f.movie_id for f in result}
    assert output_ids <= input_ids, f"Hallucinated items: {output_ids - input_ids}"


# Property 9: Reranking Determinism
# Validates: Requirements 5.2
@given(st.lists(_ranked_item_strategy(), min_size=0, max_size=50, unique_by=lambda r: r.movie_id))
@settings(max_examples=100)
def test_reranking_determinism(ranked_items):
    """Calling rerank() twice with identical inputs produces identical ordered results.

    # Feature: architecture-design-perfection, Property 9: Reranking Determinism
    """
    pipeline = _make_pipeline()
    result1 = pipeline.rerank(ranked_items, constraints={})
    result2 = pipeline.rerank(ranked_items, constraints={})
    ids1 = [f.movie_id for f in result1]
    ids2 = [f.movie_id for f in result2]
    assert ids1 == ids2, f"Reranking is non-deterministic: first call={ids1}, second call={ids2}"


def test_reranking_determinism_empty_input():
    """rerank([], {}) returns [] without raising an exception.

    # Feature: architecture-design-perfection, Property 9: Reranking Determinism
    """
    pipeline = _make_pipeline()
    result = pipeline.rerank([], {})
    assert result == [], f"Expected [] for empty input, got {result}"
