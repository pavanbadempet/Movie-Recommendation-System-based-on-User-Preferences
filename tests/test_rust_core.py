"""Unit tests for expanded Rust SIMD core functions."""

import numpy as np
import pytest

try:
    import rust_core
except ImportError:
    rust_core = None

pytestmark = pytest.mark.skipif(
    rust_core is None, reason="rust_core binary not installed in this Python environment"
)


def test_rust_fast_feature_hash():
    tokens = ["action", "thriller", "sci-fi", "romance", "comedy"]
    hashes = rust_core.fast_feature_hash_rust(tokens, num_buckets=100)

    assert len(hashes) == len(tokens)
    assert all(0 <= h < 100 for h in hashes)


def test_rust_fast_cosine_similarity():
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_c = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    sim_identical = rust_core.fast_cosine_similarity_rust(vec_a, vec_b)
    sim_orthogonal = rust_core.fast_cosine_similarity_rust(vec_a, vec_c)

    assert pytest.approx(sim_identical, abs=1e-4) == 1.0
    assert pytest.approx(sim_orthogonal, abs=1e-4) == 0.0


def test_rust_tokenize_title():
    tokens = rust_core.rust_tokenize_title_rust("The Dark Knight (2008)")
    assert tokens == ["the", "dark", "knight", "2008"]


def test_rust_time_decay_score():
    now = 1000000.0
    timestamps = [now, now - 86400.0, now - 86400.0 * 2]
    scores = rust_core.rust_time_decay_score_rust(timestamps, reference_time=now, half_life_days=1.0)

    assert len(scores) == 3
    assert pytest.approx(scores[0], abs=1e-3) == 1.0
    assert pytest.approx(scores[1], abs=1e-3) == 0.5
    assert pytest.approx(scores[2], abs=1e-3) == 0.25


def test_rust_fast_json_parse_names():
    json_data = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    names = rust_core.fast_json_parse_names_rust(json_data)
    assert names == ["Action", "Adventure"]


def test_rust_fast_softmax():
    scores = [2.0, 1.0, 0.1]
    probabilities = rust_core.fast_softmax_rust(scores, temperature=1.0)

    assert len(probabilities) == 3
    assert pytest.approx(sum(probabilities), abs=1e-4) == 1.0
    assert probabilities[0] > probabilities[1] > probabilities[2]
