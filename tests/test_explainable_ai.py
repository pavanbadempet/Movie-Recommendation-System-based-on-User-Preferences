"""
Unit tests for the Explainable AI (XAI) layer in the RerankingPipeline.
"""

import pandas as pd
import pytest

from backend.pipeline.pipeline_types import RankedItem
from backend.pipeline.reranking_pipeline import RerankingConfig, RerankingPipeline


@pytest.fixture
def sample_movie_df():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Toy Story", "Toy Story 2"],
            "genres": ["Animation, Family", "Animation, Family"],
        }
    )


@pytest.fixture
def sample_ranked_items():
    return [
        RankedItem(
            movie_id=1,
            retrieval_score=0.9,
            retrieval_source="turbovec",
            ensemble_score=0.85,
            ranker_score=0.8,
            final_rank=1,
            retrieval_signals={"genre_overlap": 1.0},
            metadata={"title": "Toy Story", "genres": "Animation, Family"},
        )
    ]


class MockLLMClient:
    def generate_explanation(self, item: RankedItem) -> str:
        return f"Custom explanation for {item.movie_id}"


def test_explainable_ai_fallback_template(sample_movie_df, sample_ranked_items):
    # Tests that when llm_client is None, it falls back to direct library call which
    # falls back to template when no OpenRouter API key is present.
    config = RerankingConfig(enable_llm_reranking=True, enable_rl_safety=False)
    pipeline = RerankingPipeline(rl_policy=None, llm_client=None, config=config, movie_df=sample_movie_df)

    profile = {
        "user_id": "test-user",
        "favorite_genres": ["Animation"],
        "recent_events": [{"movie_id": 2, "weight": 1.0}],
    }

    results = pipeline.rerank(sample_ranked_items, constraints={"profile": profile})
    assert len(results) == 1
    explanation = results[0].explanation
    assert explanation is not None
    # Fallback template string contains the formatted signals (e.g. "genre match")
    assert "Recommended" in explanation or "genre match" in explanation


def test_explainable_ai_custom_client(sample_movie_df, sample_ranked_items):
    # Tests that custom llm_client takes precedence and is invoked correctly.
    config = RerankingConfig(enable_llm_reranking=True, enable_rl_safety=False)
    mock_client = MockLLMClient()
    pipeline = RerankingPipeline(rl_policy=None, llm_client=mock_client, config=config, movie_df=sample_movie_df)

    results = pipeline.rerank(sample_ranked_items, constraints={})
    assert len(results) == 1
    assert results[0].explanation == "Custom explanation for 1"


def test_explainable_ai_disabled(sample_movie_df, sample_ranked_items):
    # Tests that when disabled, no explanations are generated.
    config = RerankingConfig(enable_llm_reranking=False, enable_rl_safety=False)
    mock_client = MockLLMClient()
    pipeline = RerankingPipeline(rl_policy=None, llm_client=mock_client, config=config, movie_df=sample_movie_df)

    results = pipeline.rerank(sample_ranked_items, constraints={})
    assert len(results) == 1
    assert results[0].explanation is None
