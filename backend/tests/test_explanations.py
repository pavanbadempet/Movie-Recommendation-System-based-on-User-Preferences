import pytest
from unittest.mock import patch, MagicMock
from backend.llm_explanations import generate_explanation, _generate_cache_key, _explanation_cache

@pytest.fixture(autouse=True)
def clear_cache():
    _explanation_cache.clear()
    yield
    _explanation_cache.clear()

@pytest.fixture
def dummy_movie():
    return {
        "id": 123,
        "title": "Interstellar",
        "genres": ["Sci-Fi", "Drama"],
        "retrieval_signals": {
            "genre_overlap": 0.8
        },
        "explanation": ["Similar story", "High rating"]
    }

@patch("backend.llm_explanations.openrouter_api_key")
@patch("backend.llm_explanations.chat_completion")
def test_generate_explanation_success(mock_chat, mock_api_key, dummy_movie):
    """Test that the explanation generation correctly formats the prompt and returns the LLM response."""
    # Mock LLM response
    mock_api_key.return_value = "dummy_key"
    mock_chat.return_value = "Because you loved the mind-bending narrative of Inception, you'll love Interstellar."
    
    explanation = generate_explanation("user_1", dummy_movie, "user loves Sci-Fi with complex narratives")
    
    assert "Because you loved" in explanation
    assert "Interstellar" in explanation
    
    # Verify the LLM was called with correct context
    args, kwargs = mock_chat.call_args
    messages = kwargs.get("messages")
    user_prompt = messages[1]["content"]
    assert "Interstellar" in user_prompt
    assert "user loves Sci-Fi" in user_prompt

@patch("backend.llm_explanations.openrouter_api_key")
@patch("backend.llm_explanations.chat_completion")
def test_generate_explanation_fallback(mock_chat, mock_api_key, dummy_movie):
    """Test that it gracefully degrades to a template if the LLM fails or times out."""
    # Simulate a timeout or OpenRouter failure
    mock_api_key.return_value = "dummy_key"
    mock_chat.side_effect = Exception("OpenRouter API timeout")
    
    explanation = generate_explanation("user_1", dummy_movie)
    
    # Should contain the fallback text and the signals
    assert "Recommended based on your preferences" in explanation
    assert "Similar story" in explanation
    
def test_cache_key_generation():
    """Test that cache keys are unique but deterministic based on signals."""
    key1 = _generate_cache_key("user_1", 123, "hash_A")
    key2 = _generate_cache_key("user_1", 123, "hash_B")
    key3 = _generate_cache_key("user_2", 123, "hash_A")
    
    assert key1 != key2 # Different signals = different key
    assert key1 != key3 # Different user = different key
    assert key1 == "llm_expl:user_1:123:hash_A"
