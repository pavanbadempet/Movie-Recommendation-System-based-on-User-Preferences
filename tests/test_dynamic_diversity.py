"""
Unit tests for Dynamic Causal Diversity Temperature Gating in the RerankingPipeline.
"""

import pandas as pd
import pytest

from backend.pipeline.pipeline_types import RankedItem
from backend.pipeline.reranking_pipeline import RerankingConfig, RerankingPipeline


@pytest.fixture
def sample_movie_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "title": [f"Movie {i}" for i in range(1, 11)],
            "genres": [
                "Action",                  # 1: Action
                "Action",                  # 2: Action
                "Action",                  # 3: Action
                "Action",                  # 4: Action
                "Comedy",                  # 5: Comedy
                "Drama",                   # 6: Drama
                "Sci-Fi",                  # 7: Sci-Fi
                "Romance",                 # 8: Romance
                "Action, Comedy",          # 9: Action & Comedy
                "Drama, Thriller",         # 10: Drama & Thriller
            ],
        }
    )


@pytest.fixture
def sample_ranked_items():
    return [
        RankedItem(
            movie_id=i,
            retrieval_score=0.9 - (i * 0.05),
            retrieval_source="turbovec",
            ensemble_score=0.8 - (i * 0.05),
            ranker_score=0.7 - (i * 0.05),
            final_rank=i,
            retrieval_signals={},
            metadata={"genres": "Action" if i <= 4 else "Drama"},
        )
        for i in range(1, 6)
    ]


def test_calculate_genre_entropy_low(sample_movie_df):
    config = RerankingConfig(enable_dynamic_diversity=True)
    pipeline = RerankingPipeline(rl_policy=None, llm_client=None, config=config, movie_df=sample_movie_df)

    # User profile with only Action movies watched
    profile = {
        "recent_events": [
            {"movie_id": 1, "weight": 1.0, "event_ts": "2026-06-20T12:00:00Z"},
            {"movie_id": 2, "weight": 1.0, "event_ts": "2026-06-20T12:00:00Z"},
        ]
    }
    
    entropy = pipeline._calculate_genre_entropy(profile)
    # Since only 1 genre ("action") was consumed, entropy should be 0.0
    assert entropy is not None
    assert pytest.approx(entropy, abs=1e-5) == 0.0

    # Test dynamic lambda: should be clamped to min_lambda
    lam = pipeline._get_dynamic_mmr_lambda({"profile": profile})
    assert lam == config.dynamic_diversity_min_lambda


def test_calculate_genre_entropy_high(sample_movie_df):
    config = RerankingConfig(enable_dynamic_diversity=True)
    pipeline = RerankingPipeline(rl_policy=None, llm_client=None, config=config, movie_df=sample_movie_df)

    # User profile with diverse genres watched equally
    profile = {
        "recent_events": [
            {"movie_id": 1, "weight": 1.0, "event_ts": "2026-06-20T12:00:00Z"},  # Action
            {"movie_id": 5, "weight": 1.0, "event_ts": "2026-06-20T12:00:00Z"},  # Comedy
            {"movie_id": 6, "weight": 1.0, "event_ts": "2026-06-20T12:00:00Z"},  # Drama
            {"movie_id": 7, "weight": 1.0, "event_ts": "2026-06-20T12:00:00Z"},  # Sci-Fi
        ]
    }
    
    entropy = pipeline._calculate_genre_entropy(profile)
    assert entropy is not None
    # 4 distinct genres of equal weight: -4 * (0.25 * log2(0.25)) = -4 * (0.25 * -2) = 2.0
    assert pytest.approx(entropy, abs=1e-5) == 2.0

    # Test dynamic lambda: should interpolate between H_min=1.0 and H_max=3.0
    # For H = 2.0, lambda is halfway between min (0.3) and max (0.85): 0.575
    lam = pipeline._get_dynamic_mmr_lambda({"profile": profile})
    expected_lam = 0.3 + 0.5 * (0.85 - 0.3)
    assert pytest.approx(lam, abs=1e-5) == expected_lam


def test_fallback_dynamic_diversity(sample_movie_df):
    config = RerankingConfig(enable_dynamic_diversity=True, mmr_lambda=0.7)
    pipeline = RerankingPipeline(rl_policy=None, llm_client=None, config=config, movie_df=sample_movie_df)

    # Missing constraints or profile should fallback to mmr_lambda config
    assert pipeline._get_dynamic_mmr_lambda({}) == 0.7
    assert pipeline._get_dynamic_mmr_lambda(None) == 0.7
    assert pipeline._get_dynamic_mmr_lambda({"profile": {}}) == 0.7


def test_invalid_movie_ids_ignored(sample_movie_df):
    config = RerankingConfig(enable_dynamic_diversity=True)
    pipeline = RerankingPipeline(rl_policy=None, llm_client=None, config=config, movie_df=sample_movie_df)

    # Profile containing non-existent movie IDs in catalog
    profile = {
        "recent_events": [
            {"movie_id": 9999, "weight": 1.0},
            {"movie_id": 1, "weight": 1.0},
        ]
    }

    entropy = pipeline._calculate_genre_entropy(profile)
    # The invalid ID should be skipped. Since only movie_id 1 is left, entropy should be 0.0
    assert entropy is not None
    assert pytest.approx(entropy, abs=1e-5) == 0.0


def test_dynamic_diversity_disabled(sample_movie_df):
    config = RerankingConfig(enable_dynamic_diversity=False, mmr_lambda=0.6)
    pipeline = RerankingPipeline(rl_policy=None, llm_client=None, config=config, movie_df=sample_movie_df)

    profile = {
        "recent_events": [
            {"movie_id": 1, "weight": 1.0},
        ]
    }
    # Should always return default mmr_lambda when disabled
    assert pipeline._get_dynamic_mmr_lambda({"profile": profile}) == 0.6
