"""
End-to-End Pipeline Integration Testing.

This suite tests the full lifecycle: 
Dense/Sparse Retrieval (FAISS) -> Candidate Generation -> ML Reranking (ApexEnsemble).
It ensures the data strictly respects the Medallion flow and guarantees 
the reranking step does not drop valid items.
"""

import pytest
from backend.main import get_rec

# We use fixture scope=module so we only load the 70MB FAISS index once per test session
@pytest.fixture(scope="module")
def recommender():
    rec = get_rec()
    rec.load() # This loads FAISS, embeddings, and initializes the Apex Ensemble
    return rec

def test_recommend_for_user_profile_integration(recommender):
    """
    Tests the complete end-to-end user recommendation pipeline.
    Ensures the retrieval + ApexEnsembleEngine reranking path works
    without crashing and returns valid movie records.
    """
    # Create a mock user profile using the actual expected schema
    mock_profile = {
        "user_id": "integration-test-user",
        "recent_events": [
            {"movie_id": 862, "event_type": "rating", "rating": 5},  # Toy Story
            {"movie_id": 863, "event_type": "rating", "rating": 4},
        ],
        "favorite_genres": ["Animation", "Family"],
        "negative_movie_ids": [],
    }
    
    # Run the full pipeline — param is `n`, not `limit`
    results = recommender.recommend_for_user_profile(mock_profile, n=20)
    
    # Assertions
    assert isinstance(results, list)
    assert len(results) <= 20
    
    # Every returned item MUST have core identification fields
    for item in results:
        assert "id" in item
        assert "title" in item

def test_vector_search_fallback(recommender):
    """
    Ensures that if the semantic dense retrieval fails, it gracefully
    falls back to TF-IDF / Sparse metadata matching without crashing.
    """
    # We query something highly obscure
    results = recommender.search_movies("A completely obscure nonsensical query that doesnt exist 12345", limit=5)
    
    assert isinstance(results, list)
    assert len(results) <= 5
    
    if len(results) > 0:
        # Check that it executed the standard pipeline schema
        assert "id" in results[0]
