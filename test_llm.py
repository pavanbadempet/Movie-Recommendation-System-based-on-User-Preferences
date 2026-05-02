import os

import pytest
from huggingface_hub import InferenceClient


def test_huggingface_chat_completion():
    """Optional smoke test for Hugging Face hosted chat models."""
    if os.getenv("RUN_LLM_TESTS") != "1":
        pytest.skip("Set RUN_LLM_TESTS=1 to run external LLM smoke tests.")

    client = InferenceClient(token=os.getenv("HF_TOKEN"))
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 'What is 2+2? Output as JSON: {"result": 4}'},
    ]

    response = client.chat_completion(
        messages=messages,
        model="Qwen/Qwen2.5-72B-Instruct",
        max_tokens=100,
        temperature=0.1,
    )

    assert response.choices[0].message.content
