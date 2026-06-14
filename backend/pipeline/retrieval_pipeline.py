"""
Retrieval pipeline (Stage 1) for the APEX Movie Recommendation System.

This module implements the first stage of the 3-stage recommendation pipeline.
It queries up to three retrieval sources — FAISS ANN index, TF-IDF sparse index,
and a Knowledge Graph engine — merges the results via max-pool deduplication, and
returns the top-n candidates sorted by retrieval score.

Circular-import constraint
--------------------------
Only ``CandidateItem`` is imported from ``backend.pipeline_types`` at module
level.  No other ``backend/`` modules are imported here.

Typical usage
-------------
>>> from backend.pipeline.retrieval_pipeline import RetrievalPipeline, RetrievalConfig
>>> from backend.pipeline.pipeline_types import CandidateItem
>>>
>>> config = RetrievalConfig(faiss_k=100, tfidf_k=50, kg_k=20)
>>> pipeline = RetrievalPipeline(
...     faiss_index=faiss_index,
...     tfidf_index=(vectorizer, tfidf_matrix),
...     kg_engine=kg_engine,
...     movie_df=movie_df,
...     config=config,
... )
>>> candidates = pipeline.retrieve(query_embedding, n=20)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from backend.pipeline.pipeline_types import CandidateItem

logger = logging.getLogger(__name__)

# Valid retrieval source tags accepted by CandidateItem.
_VALID_SOURCES = frozenset({"faiss", "tfidf", "knowledge_graph", "hybrid"})


@dataclass
class RetrievalConfig:
    """Configuration knobs for :class:`RetrievalPipeline`.

    Attributes
    ----------
    faiss_k:
        Number of nearest-neighbour candidates to fetch from the FAISS index.
        Defaults to 100.
    tfidf_k:
        Number of top candidates to fetch from the TF-IDF sparse index.
        Defaults to 50.
    kg_k:
        Number of neighbours to fetch from the Knowledge Graph engine.
        Defaults to 20.
    low_memory:
        When ``True``, the TF-IDF retrieval step is skipped entirely to
        conserve memory (Tier-3 degradation path).  Defaults to ``False``.
    enable_kg:
        When ``False``, the Knowledge Graph retrieval step is skipped.
        Defaults to ``True``.
    """

    faiss_k: int = 100
    tfidf_k: int = 50
    kg_k: int = 20
    low_memory: bool = False
    enable_kg: bool = True


class RetrievalPipeline:
    """Stage 1 of the recommendation pipeline: multi-source candidate retrieval.

    Queries FAISS, TF-IDF, and/or a Knowledge Graph engine, deduplicates the
    results by ``movie_id`` using max-pool on ``retrieval_score``, and returns
    the top-n :class:`~backend.pipeline_types.CandidateItem` objects sorted
    descending by score.

    Parameters
    ----------
    faiss_index:
        A ``faiss.IndexFlatIP`` (or compatible) index.  May be ``None``; when
        ``None`` the FAISS retrieval step is skipped and the pipeline falls
        back to TF-IDF.
    tfidf_index:
        A ``(vectorizer, tfidf_matrix)`` tuple where *vectorizer* is a fitted
        ``sklearn.feature_extraction.text.TfidfVectorizer`` and *tfidf_matrix*
        is the corresponding sparse document-term matrix.  May be ``None``; when
        ``None`` (or when ``config.low_memory`` is ``True``) the TF-IDF step is
        skipped.
    kg_engine:
        An object with a ``get_neighbors(movie_id: int, n: int) -> list[int]``
        method.  May be ``None``; when ``None`` (or when ``config.enable_kg``
        is ``False``) the KG step is skipped.
    movie_df:
        A ``pandas.DataFrame`` containing movie metadata.  Row indices must
        align with the FAISS index (i.e. FAISS index *i* corresponds to
        ``movie_df.iloc[i]``).
    config:
        A :class:`RetrievalConfig` instance controlling retrieval parameters.
    """

    def __init__(
        self,
        faiss_index,
        tfidf_index,
        kg_engine,
        movie_df,
        config: RetrievalConfig,
    ) -> None:
        self.faiss_index = faiss_index
        self.tfidf_index = tfidf_index  # (vectorizer, tfidf_matrix) or None
        self.kg_engine = kg_engine
        self.movie_df = movie_df
        self.config = config
        # Fast lookup mapping to bypass slow DataFrame iloc operations on retrieval hot path
        self._movie_id_map = movie_df["id"].values if (movie_df is not None and "id" in movie_df.columns) else np.array([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query_embedding: np.ndarray, n: int, query_text: str | None = None) -> list[CandidateItem]:
        """Retrieve up to *n* candidate movies for the given query embedding.

        Retrieval steps (each step is skipped on error with a WARNING):

        1. **FAISS ANN search** — top-``faiss_k`` candidates tagged
           ``retrieval_source="faiss"``.  Skipped when ``faiss_index`` is
           ``None``.
        2. **TF-IDF sparse search** — additional candidates tagged
           ``retrieval_source="tfidf"``.  Skipped when ``low_memory=True``
           or ``tfidf_index`` is ``None``.
        3. **Knowledge Graph traversal** — additional candidates tagged
           ``retrieval_source="knowledge_graph"``.  Skipped when
           ``kg_engine`` is ``None`` or ``enable_kg=False``.
        4. **Deduplication** — candidates sharing the same ``movie_id`` are
           merged by keeping the maximum ``retrieval_score`` and tagging the
           merged item ``retrieval_source="hybrid"``.
        5. **Sort & truncate** — results are sorted descending by
           ``retrieval_score`` and the top-*n* items are returned.

        Parameters
        ----------
        query_embedding:
            1-D (or 2-D row) NumPy array representing the query in the
            embedding space.
        n:
            Maximum number of candidates to return.  Returns ``[]``
            immediately when ``n == 0``.

        Returns
        -------
        list[CandidateItem]
            Up to *n* candidates, sorted descending by ``retrieval_score``.
            Returns ``[]`` when *n* is 0 or all retrieval sources are
            unavailable / raise exceptions.

        Invariants
        ----------
        - ``len(result) <= n`` always.
        - All ``movie_id`` values in the result are unique.
        - All ``retrieval_source`` values are in
          ``{"faiss", "tfidf", "knowledge_graph", "hybrid"}``.
        """
        if n == 0:
            return []

        all_candidates: list[CandidateItem] = []

        # ----------------------------------------------------------------
        # Step 1: FAISS ANN retrieval
        # ----------------------------------------------------------------
        if self.faiss_index is not None:
            faiss_candidates = self._retrieve_faiss(query_embedding)
            all_candidates.extend(faiss_candidates)
        else:
            logger.debug("FAISS index not available; skipping FAISS retrieval.")

        # ----------------------------------------------------------------
        # Step 2: TF-IDF sparse retrieval
        # ----------------------------------------------------------------
        if self.config.low_memory:
            logger.debug("low_memory=True; skipping TF-IDF retrieval.")
        elif self.tfidf_index is None:
            logger.debug("TF-IDF index not available; skipping TF-IDF retrieval.")
        else:
            tfidf_candidates = self._retrieve_tfidf(query_embedding, query_text=query_text)
            all_candidates.extend(tfidf_candidates)

        # ----------------------------------------------------------------
        # Step 3: Knowledge Graph retrieval
        # ----------------------------------------------------------------
        if not self.config.enable_kg:
            logger.debug("enable_kg=False; skipping KG retrieval.")
        elif self.kg_engine is None:
            logger.debug("KG engine not available; skipping KG retrieval.")
        else:
            kg_candidates = self._retrieve_kg(query_embedding)
            all_candidates.extend(kg_candidates)

        # ----------------------------------------------------------------
        # Step 4: Deduplication via max-pool on retrieval_score
        # ----------------------------------------------------------------
        merged = self._deduplicate(all_candidates)

        if not merged:
            logger.warning("All retrieval sources returned no candidates for the given query.")
            return []

        # ----------------------------------------------------------------
        # Step 5: Sort descending by retrieval_score and truncate to n
        # ----------------------------------------------------------------
        merged.sort(key=lambda c: c.retrieval_score, reverse=True)
        result = merged[:n]

        # Enforce invariants (defensive assertions — should never fire in
        # production but help catch regressions during testing).
        assert len(result) <= n, f"Result length {len(result)} exceeds n={n}"
        movie_ids = [c.movie_id for c in result]
        assert len(movie_ids) == len(set(movie_ids)), "Duplicate movie_ids in result"
        for item in result:
            assert item.retrieval_source in _VALID_SOURCES, f"Invalid retrieval_source: {item.retrieval_source!r}"

        logger.debug(
            "retrieve() returning %d candidates (n=%d, sources=%s).",
            len(result),
            n,
            {c.retrieval_source for c in result},
        )
        return result

    # ------------------------------------------------------------------
    # Private retrieval helpers
    # ------------------------------------------------------------------

    def _retrieve_faiss(self, query_embedding: np.ndarray) -> list[CandidateItem]:
        """Query the FAISS index and return up to ``faiss_k`` candidates.

        Parameters
        ----------
        query_embedding:
            Query vector; reshaped to ``(1, d)`` and cast to ``float32``
            before passing to FAISS.

        Returns
        -------
        list[CandidateItem]
            Candidates tagged ``retrieval_source="faiss"``.  Returns ``[]``
            on any exception (logs WARNING).
        """
        try:
            k = self.config.faiss_k
            vec = query_embedding.reshape(1, -1).astype(np.float32)
            distances, indices = self.faiss_index.search(vec, k)

            candidates: list[CandidateItem] = []
            for dist, idx in zip(distances[0], indices[0], strict=False):
                if idx < 0 or idx >= len(self.movie_df):
                    # FAISS returns -1 for padding when fewer than k results exist.
                    continue
                movie_id = int(self._movie_id_map[idx]) if idx < len(self._movie_id_map) else int(idx)
                candidates.append(
                    CandidateItem(
                        movie_id=movie_id,
                        retrieval_score=float(dist),
                        retrieval_source="faiss",
                        metadata={"faiss_index": int(idx)},
                    )
                )

            logger.debug("FAISS retrieval returned %d candidates.", len(candidates))
            return candidates

        except Exception as exc:
            logger.warning(
                "FAISS retrieval failed with %s: %s — skipping FAISS source.",
                type(exc).__name__,
                exc,
            )
            return []

    def _retrieve_tfidf(self, query_embedding: np.ndarray, query_text: str | None = None) -> list[CandidateItem]:
        """Query the TF-IDF sparse index and return up to ``tfidf_k`` candidates.

        Uses cosine similarity between the vectorized query text and the TF-IDF
        document matrix.

        Parameters
        ----------
        query_embedding:
            Unused query embedding; kept for signature uniformity.
        query_text:
            Optional query search string to vectorize and query.

        Returns
        -------
        list[CandidateItem]
            Candidates tagged ``retrieval_source="tfidf"``.  Returns ``[]``
            on any exception (logs WARNING).
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity  # local import

            if not query_text:
                logger.debug("query_text not provided; skipping TF-IDF retrieval.")
                return []

            vectorizer, tfidf_matrix = self.tfidf_index
            k = self.config.tfidf_k

            query_sparse = vectorizer.transform([query_text])
            scores = cosine_similarity(query_sparse, tfidf_matrix).flatten()
            top_indices = np.argsort(scores)[::-1][:k]

            candidates: list[CandidateItem] = []
            for idx in top_indices:
                if idx < 0 or idx >= len(self.movie_df):
                    continue
                score = float(scores[idx])
                if score <= 0.0:
                    continue
                movie_id = int(self._movie_id_map[idx]) if idx < len(self._movie_id_map) else int(idx)
                candidates.append(
                    CandidateItem(
                        movie_id=movie_id,
                        retrieval_score=score,
                        retrieval_source="tfidf",
                        metadata={"tfidf_index": int(idx)},
                    )
                )

            logger.debug("TF-IDF retrieval returned %d candidates.", len(candidates))
            return candidates

        except Exception as exc:
            logger.warning(
                "TF-IDF retrieval failed with %s: %s — skipping TF-IDF source.",
                type(exc).__name__,
                exc,
            )
            return []

    def _retrieve_kg(self, query_embedding: np.ndarray) -> list[CandidateItem]:
        """Query the Knowledge Graph engine for neighbours of the query movie.

        The query movie is identified by finding the FAISS nearest neighbour
        (if available) or by falling back to the first row of ``movie_df``.
        The KG engine's ``get_neighbors(movie_id, n=kg_k)`` method is called
        to obtain related movie IDs.

        Parameters
        ----------
        query_embedding:
            Query vector used to identify the seed movie for KG traversal.

        Returns
        -------
        list[CandidateItem]
            Candidates tagged ``retrieval_source="knowledge_graph"``.
            Returns ``[]`` on any exception (logs WARNING).
        """
        try:
            k = self.config.kg_k

            # Identify the seed movie_id for KG traversal.
            seed_movie_id = self._get_seed_movie_id(query_embedding)
            if seed_movie_id is None:
                logger.warning("Could not determine seed movie_id for KG traversal; skipping KG source.")
                return []

            neighbor_ids: list[int] = self.kg_engine.get_neighbors(seed_movie_id, n=k)

            candidates: list[CandidateItem] = []
            for rank, movie_id in enumerate(neighbor_ids):
                # Assign a decaying score so KG candidates rank below FAISS/TF-IDF
                # by default but still participate in max-pool deduplication.
                score = 1.0 / (rank + 1)
                candidates.append(
                    CandidateItem(
                        movie_id=int(movie_id),
                        retrieval_score=score,
                        retrieval_source="knowledge_graph",
                        metadata={"kg_rank": rank, "seed_movie_id": seed_movie_id},
                    )
                )

            logger.debug(
                "KG retrieval returned %d candidates (seed_movie_id=%d).",
                len(candidates),
                seed_movie_id,
            )
            return candidates

        except Exception as exc:
            logger.warning(
                "KG retrieval failed with %s: %s — skipping KG source.",
                type(exc).__name__,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Private utility helpers
    # ------------------------------------------------------------------

    def _get_seed_movie_id(self, query_embedding: np.ndarray) -> int | None:
        """Determine the seed movie_id for KG traversal.

        Prefers the top FAISS nearest neighbour; falls back to the first row
        of ``movie_df`` when FAISS is unavailable.

        Parameters
        ----------
        query_embedding:
            Query vector.

        Returns
        -------
        int | None
            The seed ``movie_id``, or ``None`` if it cannot be determined.
        """
        if self.faiss_index is not None:
            try:
                vec = query_embedding.reshape(1, -1).astype(np.float32)
                _, indices = self.faiss_index.search(vec, 1)
                idx = int(indices[0][0])
                if 0 <= idx < len(self.movie_df):
                    return int(self._movie_id_map[idx]) if idx < len(self._movie_id_map) else int(idx)
            except Exception as exc:
                logger.debug(
                    "FAISS seed lookup failed (%s: %s); falling back to movie_df[0].",
                    type(exc).__name__,
                    exc,
                )

        # Fallback: use the first movie in the DataFrame.
        if len(self.movie_df) > 0:
            return int(self._movie_id_map[0]) if len(self._movie_id_map) > 0 else 0

        return None

    @staticmethod
    def _deduplicate(candidates: list[CandidateItem]) -> list[CandidateItem]:
        """Deduplicate candidates by ``movie_id`` using max-pool on score.

        When the same ``movie_id`` appears from multiple retrieval sources,
        the item with the highest ``retrieval_score`` is kept and its
        ``retrieval_source`` is set to ``"hybrid"`` to indicate that multiple
        sources contributed.

        Parameters
        ----------
        candidates:
            Raw (possibly duplicate) list of :class:`CandidateItem` objects.

        Returns
        -------
        list[CandidateItem]
            Deduplicated list; order is not guaranteed (caller should sort).
        """
        best: dict[int, CandidateItem] = {}

        for item in candidates:
            mid = item.movie_id
            if mid not in best:
                best[mid] = item
            else:
                existing = best[mid]
                if item.retrieval_score > existing.retrieval_score:
                    # Keep the higher score; mark as hybrid since multiple
                    # sources contributed.
                    merged_metadata = {**existing.metadata, **item.metadata}
                    best[mid] = CandidateItem(
                        movie_id=mid,
                        retrieval_score=item.retrieval_score,
                        retrieval_source="hybrid",
                        metadata=merged_metadata,
                    )
                elif existing.retrieval_source != item.retrieval_source:
                    # Same score but different source — still mark as hybrid.
                    merged_metadata = {**existing.metadata, **item.metadata}
                    best[mid] = CandidateItem(
                        movie_id=mid,
                        retrieval_score=existing.retrieval_score,
                        retrieval_source="hybrid",
                        metadata=merged_metadata,
                    )

        return list(best.values())
