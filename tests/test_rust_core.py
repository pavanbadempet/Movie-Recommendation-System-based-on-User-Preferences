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
