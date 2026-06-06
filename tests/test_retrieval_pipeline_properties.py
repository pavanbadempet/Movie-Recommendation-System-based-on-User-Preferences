"""
Property-based tests for RetrievalPipeline invariants.
# Feature: architecture-design-perfection, Property 1/2/3
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.pipeline.pipeline_types import CandidateItem
from backend.pipeline.retrieval_pipeline import RetrievalConfig, RetrievalPipeline


def _make_mock_faiss(n_items: int):
    """Create a minimal mock FAISS index backed by numpy."""
    class MockFaissIndex:
        def __init__(self, n):
            self.ntotal = n
            self._vecs = np.random.rand(n, 64).astype(np.float32)

        def search(self, query, k):
            k = min(k, self.ntotal)
            dists = np.random.rand(1, k).astype(np.float32)
            idxs = np.random.choice(self.ntotal, size=k, replace=False).reshape(1, k).astype(np.int64)
            return dists, idxs

    return MockFaissIndex(n_items)


def _make_mock_movie_df(n_items: int):
    import pandas as pd
    return pd.DataFrame({
        "id": list(range(n_items)),
        "title": [f"Movie {i}" for i in range(n_items)],
        "overview": ["" for _ in range(n_items)],
        "genres": ["" for _ in range(n_items)],
    })


# Property 1: Retrieval Bounds Guarantee
# Validates: Requirements 1.1 — for non-empty catalog and n >= 1,
# result length is between 0 and n (inclusive).
@given(st.integers(min_value=1, max_value=50))
@settings(max_examples=100, deadline=None)
def test_retrieval_bounds_guarantee(n):
    """
    **Validates: Requirements 1.1**

    For non-empty catalog and n >= 1, result length is between 0 and n.
    # Feature: architecture-design-perfection, Property 1: Retrieval Bounds Guarantee
    """
    n_items = max(n + 10, 20)
    faiss_idx = _make_mock_faiss(n_items)
    movie_df = _make_mock_movie_df(n_items)
    config = RetrievalConfig(
        faiss_k=min(n * 2, n_items),
        tfidf_k=0,
        kg_k=0,
        enable_kg=False,
    )
    pipeline = RetrievalPipeline(
        faiss_index=faiss_idx,
        tfidf_index=None,
        kg_engine=None,
        movie_df=movie_df,
        config=config,
    )
    query_vector = np.random.rand(1, 64).astype(np.float32)
    result = pipeline.retrieve(query_vector, n=n)
    assert 0 <= len(result) <= n, f"Expected len <= {n}, got {len(result)}"


# Property 2: Retrieval Deduplication Invariant
# Validates: Requirements 1.2 — all movie_id values in the result are unique,
# even when multiple retrieval sources return overlapping candidates.
@given(st.integers(min_value=1, max_value=50))
@settings(max_examples=100, deadline=None)
def test_retrieval_deduplication_invariant(n):
    """
    **Validates: Requirements 1.2**

    When FAISS and TF-IDF sources return overlapping movie_id sets,
    the pipeline must deduplicate so that all movie_id values in the
    result are unique.
    # Feature: architecture-design-perfection, Property 2: Retrieval Deduplication Invariant
    """
    import pandas as pd
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize

    # Use a small fixed catalog so overlaps are guaranteed.
    n_items = max(n, 10)

    # Build a movie_df that uses "movie_id" column (distinct from row index).
    movie_df = pd.DataFrame({
        "movie_id": list(range(n_items)),
        "title": [f"Movie {i}" for i in range(n_items)],
        "overview": ["" for _ in range(n_items)],
        "genres": ["" for _ in range(n_items)],
    })

    # Mock FAISS index that always returns the SAME first min(n, n_items) indices
    # — guarantees overlap with TF-IDF which also returns the same indices.
    class OverlappingFaissIndex:
        def __init__(self, total):
            self.ntotal = total

        def search(self, query, k):
            k = min(k, self.ntotal)
            # Always return indices 0..k-1 so FAISS and TF-IDF fully overlap.
            idxs = np.arange(k, dtype=np.int64).reshape(1, k)
            dists = np.ones((1, k), dtype=np.float32)
            return dists, idxs

    faiss_idx = OverlappingFaissIndex(n_items)

    # Mock TF-IDF index: a sparse identity-like matrix so cosine similarity
    # returns non-zero scores for the same first n_items rows.
    dim = 64
    # Each document is a unit vector in dimension (i % dim) — simple but valid.
    rows, cols, data = [], [], []
    for i in range(n_items):
        rows.append(i)
        cols.append(i % dim)
        data.append(1.0)
    tfidf_matrix = normalize(csr_matrix((data, (rows, cols)), shape=(n_items, dim)), norm="l2")

    class MockVectorizer:
        pass  # cosine_similarity is called directly on the matrix; vectorizer unused.

    tfidf_index = (MockVectorizer(), tfidf_matrix)

    config = RetrievalConfig(
        faiss_k=min(n * 2, n_items),
        tfidf_k=min(n * 2, n_items),
        kg_k=0,
        enable_kg=False,
    )
    pipeline = RetrievalPipeline(
        faiss_index=faiss_idx,
        tfidf_index=tfidf_index,
        kg_engine=None,
        movie_df=movie_df,
        config=config,
    )

    query_vector = np.ones((1, dim), dtype=np.float32) / np.sqrt(dim)
    result = pipeline.retrieve(query_vector, n=n)

    movie_ids = [c.movie_id for c in result]
    assert len({c.movie_id for c in result}) == len(result), (
        f"Duplicate movie_ids found in result: {movie_ids}"
    )


# Property 3: Retrieval Source Tagging
# Validates: Requirements 1.3
@given(st.integers(min_value=1, max_value=50))
@settings(max_examples=100, deadline=None)
def test_retrieval_source_tagging(n):
    """Every CandidateItem.retrieval_source is in the valid set.

    # Feature: architecture-design-perfection, Property 3: Retrieval Source Tagging
    """
    VALID_SOURCES = {"faiss", "tfidf", "knowledge_graph", "hybrid"}
    n_items = max(n + 10, 20)
    faiss_idx = _make_mock_faiss(n_items)
    movie_df = _make_mock_movie_df(n_items)
    config = RetrievalConfig(faiss_k=min(n * 2, n_items), tfidf_k=0, kg_k=0, enable_kg=False)
    pipeline = RetrievalPipeline(
        faiss_index=faiss_idx, tfidf_index=None, kg_engine=None,
        movie_df=movie_df, config=config,
    )
    query_vector = np.random.rand(1, 64).astype(np.float32)
    result = pipeline.retrieve(query_vector, n=n)
    for item in result:
        assert item.retrieval_source in VALID_SOURCES, (
            f"Invalid retrieval_source: {item.retrieval_source!r}"
        )
