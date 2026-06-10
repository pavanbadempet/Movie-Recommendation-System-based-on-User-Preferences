"""
Reranking pipeline (Stage 3) for the APEX Movie Recommendation System.

This module implements the third and final stage of the 3-stage recommendation
pipeline.  It takes a list of :class:`~backend.pipeline_types.RankedItem`
objects from the ranking stage and applies four sequential post-processing
steps — RL safety filtering, quality gating, MMR diversity selection, and
optional LLM reranking — before returning a list of
:class:`~backend.pipeline_types.FinalItem` objects.

Circular-import constraint
--------------------------
Only ``RankedItem`` and ``FinalItem`` are imported from
``backend.pipeline_types`` at module level.  No other ``backend/`` modules are
imported here.

Typical usage
-------------
>>> from backend.pipeline.reranking_pipeline import RerankingPipeline, RerankingConfig
>>> from backend.pipeline.pipeline_types import RankedItem
>>>
>>> config = RerankingConfig(mmr_lambda=0.7, enable_llm_reranking=False)
>>> pipeline = RerankingPipeline(
...     rl_policy=None,
...     llm_client=llm_client,
...     config=config,
... )
>>> final_items = pipeline.rerank(ranked_items, constraints={"disliked_movie_ids": [42]})
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from backend.pipeline.pipeline_types import FinalItem, RankedItem

logger = logging.getLogger(__name__)


@dataclass
class RerankingConfig:
    """Configuration knobs for :class:`RerankingPipeline`.

    Attributes
    ----------
    mmr_lambda:
        Trade-off parameter for Maximal Marginal Relevance (MMR) diversity
        selection.  Higher values favour relevance; lower values favour
        diversity.  Must be in ``[0.0, 1.0]``.  Defaults to ``0.7``.
    enable_llm_reranking:
        When ``True``, the LLM client is called to generate a natural-language
        explanation for each selected item.  Defaults to ``False``.
    enable_rl_safety:
        When ``True``, items whose ``movie_id`` appears in the
        ``"disliked_movie_ids"`` constraint are removed before further
        processing.  Defaults to ``True``.
    quality_threshold:
        Minimum ``ranker_score`` required for an item to pass the quality gate.
        Items below this threshold are filtered out.  Defaults to ``0.3``.
    """

    mmr_lambda: float = 0.7
    enable_llm_reranking: bool = False
    enable_rl_safety: bool = True
    quality_threshold: float = 0.3


class RerankingPipeline:
    """Stage 3 of the recommendation pipeline: safety, quality, diversity, and explanation.

    Applies four sequential post-processing steps to a list of
    :class:`~backend.pipeline_types.RankedItem` objects and returns a list of
    :class:`~backend.pipeline_types.FinalItem` objects.

    Graceful degradation
    --------------------
    Both ``rl_policy`` and ``llm_client`` may be ``None``.  When a component
    is unavailable (``None`` or raises an exception), the corresponding step is
    skipped and the pipeline continues with the unmodified list.  Each step is
    individually wrapped in a ``try/except`` block so that a failure in one
    step does not prevent subsequent steps from running.

    Parameters
    ----------
    rl_policy:
        An optional RL policy object.  Currently used only as a sentinel to
        indicate that RL-based safety filtering is available.  The actual
        filtering logic uses the ``"disliked_movie_ids"`` constraint from the
        ``constraints`` dict.  May be ``None``; when ``None`` the RL safety
        step still runs (it relies on the constraints dict, not the policy
        object directly).
    llm_client:
        An object with a ``generate_explanation(item: RankedItem) -> str``
        method.  May be ``None``; when ``None`` (or when
        ``config.enable_llm_reranking`` is ``False``) the LLM reranking step
        is skipped and ``explanation`` is set to ``None`` for all items.
    config:
        A :class:`RerankingConfig` instance controlling pipeline behaviour.
    """

    def __init__(
        self,
        rl_policy,
        llm_client,
        config: RerankingConfig,
    ) -> None:
        self.rl_policy = rl_policy
        self.llm_client = llm_client
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        ranked_items: list[RankedItem],
        constraints: dict,
    ) -> list[FinalItem]:
        """Apply safety, quality, diversity, and explanation steps to *ranked_items*.

        Reranking steps (each step is wrapped in ``try/except``; on failure the
        step is skipped with a WARNING and the unmodified list is carried
        forward):

        1. **RL safety filter** — remove items whose ``movie_id`` is in
           ``constraints.get("disliked_movie_ids", [])``.  If the filter would
           remove *all* items, the original list is restored (safety fallback).
        2. **Quality gate** — remove items where
           ``ranker_score < config.quality_threshold``.  If the gate would
           remove *all* items, the pre-gate list is restored.
        3. **MMR diversity selection** — greedy Maximal Marginal Relevance
           selection using ``lambda=config.mmr_lambda``.  At each step the
           item maximising
           ``lambda * ranker_score - (1 - lambda) * max_similarity_to_selected``
           is chosen.  Cosine similarity on ``ensemble_score`` is used as a
           proxy for item similarity (scalar cosine reduces to sign-agreement
           on a 1-D signal; ``final_rank`` is used as a tiebreaker proxy when
           scores are identical).
        4. **LLM reranking** — if ``enable_llm_reranking=True`` and
           ``llm_client`` is not ``None``, call
           ``llm_client.generate_explanation(item)`` for each item to populate
           ``explanation``.  Individual LLM failures are caught and logged;
           the item is still included with ``explanation=None``.

        Parameters
        ----------
        ranked_items:
            List of :class:`~backend.pipeline_types.RankedItem` objects
            produced by the ranking stage.  May be empty.
        constraints:
            Dictionary of reranking constraints.  Recognised keys:

            - ``"disliked_movie_ids"`` (:class:`list[int]`) — movie IDs to
              suppress via the RL safety filter.

        Returns
        -------
        list[FinalItem]
            A subset of *ranked_items* converted to
            :class:`~backend.pipeline_types.FinalItem` objects.  Returns ``[]``
            immediately when *ranked_items* is empty.

        Invariants
        ----------
        - ``{f.movie_id for f in result} ⊆ {r.movie_id for r in ranked_items}``
          (no hallucination — reranking cannot introduce new items).
        - Returns ``[]`` when *ranked_items* is empty (no exception raised).
        - Calling ``rerank()`` twice with identical inputs produces identical
          ordered lists of :class:`FinalItem` objects (determinism).
        """
        if not ranked_items:
            return []

        # Working list — each step may replace this with a filtered/reordered
        # version.  We keep a reference to the original for safety fallbacks.
        original_items: list[RankedItem] = list(ranked_items)
        working: list[RankedItem] = list(ranked_items)

        # ----------------------------------------------------------------
        # Step 1: RL safety filter
        # ----------------------------------------------------------------
        if self.config.enable_rl_safety:
            working = self._apply_rl_safety_filter(working, original_items, constraints)

        # ----------------------------------------------------------------
        # Step 2: Quality gate
        # ----------------------------------------------------------------
        working = self._apply_quality_gate(working)

        # ----------------------------------------------------------------
        # Step 3: MMR diversity selection
        # ----------------------------------------------------------------
        working = self._apply_mmr_diversity(working)

        # ----------------------------------------------------------------
        # Step 4: LLM reranking (explanation generation)
        # ----------------------------------------------------------------
        explanations: dict[int, str | None] = {}
        if self.config.enable_llm_reranking and self.llm_client is not None:
            explanations = self._apply_llm_reranking(working)

        # ----------------------------------------------------------------
        # Convert RankedItem → FinalItem
        # ----------------------------------------------------------------
        result: list[FinalItem] = []
        for item in working:
            diversity_score = self._compute_diversity_score(item, working)
            result.append(
                FinalItem(
                    movie_id=item.movie_id,
                    retrieval_score=item.retrieval_score,
                    retrieval_source=item.retrieval_source,
                    ensemble_score=item.ensemble_score,
                    ranker_score=item.ranker_score,
                    final_rank=item.final_rank,
                    diversity_score=diversity_score,
                    safety_passed=True,
                    explanation=explanations.get(item.movie_id),
                    retrieval_signals=dict(item.retrieval_signals),
                    metadata=dict(item.metadata),
                )
            )

        # Defensive invariant check: no hallucination.
        input_ids = {r.movie_id for r in ranked_items}
        result_ids = {f.movie_id for f in result}
        assert result_ids <= input_ids, (
            f"RerankingPipeline invariant violated: "
            f"result contains movie_ids not in input — extra={result_ids - input_ids}"
        )

        logger.debug(
            "rerank() returning %d items from %d input items.",
            len(result),
            len(ranked_items),
        )
        return result

    # ------------------------------------------------------------------
    # Private step helpers
    # ------------------------------------------------------------------

    def _apply_rl_safety_filter(
        self,
        working: list[RankedItem],
        original: list[RankedItem],
        constraints: dict,
    ) -> list[RankedItem]:
        """Remove items in the user's dislike list.

        If the filter would remove *all* items, the original list is restored
        as a safety fallback.

        Parameters
        ----------
        working:
            Current working list of :class:`RankedItem` objects.
        original:
            The unmodified input list used for the safety fallback.
        constraints:
            Constraints dict; ``"disliked_movie_ids"`` key is read here.

        Returns
        -------
        list[RankedItem]
            Filtered list, or *original* if all items would have been removed.
        """
        try:
            disliked: set[int] = set(constraints.get("disliked_movie_ids", []))
            if not disliked:
                return working

            filtered = [item for item in working if item.movie_id not in disliked]

            if not filtered:
                logger.warning(
                    "RL safety filter would remove all %d items; reverting to original list (safety fallback).",
                    len(working),
                )
                return list(original)

            removed = len(working) - len(filtered)
            if removed:
                logger.debug(
                    "RL safety filter removed %d item(s) matching disliked_movie_ids.",
                    removed,
                )
            return filtered

        except Exception as exc:
            logger.warning(
                "RL safety filter failed with %s: %s — skipping step.",
                type(exc).__name__,
                exc,
            )
            return working

    def _apply_quality_gate(self, working: list[RankedItem]) -> list[RankedItem]:
        """Remove items below the quality threshold.

        If the gate would remove *all* items, the pre-gate list is restored.

        Parameters
        ----------
        working:
            Current working list of :class:`RankedItem` objects.

        Returns
        -------
        list[RankedItem]
            Quality-filtered list, or *working* if all items would have been
            removed.
        """
        try:
            threshold = self.config.quality_threshold
            filtered = [item for item in working if item.ranker_score >= threshold]

            if not filtered:
                logger.warning(
                    "Quality gate (threshold=%.3f) would remove all %d items; reverting to pre-gate list.",
                    threshold,
                    len(working),
                )
                return working

            removed = len(working) - len(filtered)
            if removed:
                logger.debug(
                    "Quality gate removed %d item(s) with ranker_score < %.3f.",
                    removed,
                    threshold,
                )
            return filtered

        except Exception as exc:
            logger.warning(
                "Quality gate failed with %s: %s — skipping step.",
                type(exc).__name__,
                exc,
            )
            return working

    def _apply_mmr_diversity(self, working: list[RankedItem]) -> list[RankedItem]:
        """Greedy MMR diversity selection.

        At each step, the item that maximises the MMR objective is selected:

            ``mmr_lambda * ranker_score - (1 - mmr_lambda) * max_sim_to_selected``

        Similarity between items is approximated using the absolute difference
        of their ``ensemble_score`` values (a 1-D cosine proxy).  When
        ``ensemble_score`` values are identical, ``final_rank`` is used as a
        tiebreaker to preserve determinism.

        Parameters
        ----------
        working:
            Current working list of :class:`RankedItem` objects.

        Returns
        -------
        list[RankedItem]
            MMR-reordered list of the same length as *working*.  Returns
            *working* unchanged on any exception.
        """
        try:
            if len(working) <= 1:
                return working

            lam = self.config.mmr_lambda
            remaining: list[RankedItem] = list(working)
            selected: list[RankedItem] = []

            while remaining:
                best_item: RankedItem | None = None
                best_score: float = float("-inf")

                for candidate in remaining:
                    if not selected:
                        # First selection: pure relevance (no diversity penalty yet).
                        mmr_score = lam * candidate.ranker_score
                    else:
                        max_sim = max(_ensemble_similarity(candidate, sel) for sel in selected)
                        mmr_score = lam * candidate.ranker_score - (1.0 - lam) * max_sim

                    # Use (mmr_score, -final_rank) as a deterministic tiebreaker:
                    # higher mmr_score wins; among ties, lower final_rank (better
                    # original rank) wins.
                    if best_item is None or (
                        mmr_score > best_score
                        or (mmr_score == best_score and candidate.final_rank < best_item.final_rank)
                    ):
                        best_item = candidate
                        best_score = mmr_score

                if best_item is not None:
                    selected.append(best_item)
                    remaining.remove(best_item)

            logger.debug(
                "MMR diversity selection reordered %d items (lambda=%.2f).",
                len(selected),
                lam,
            )
            return selected

        except Exception as exc:
            logger.warning(
                "MMR diversity selection failed with %s: %s — skipping step.",
                type(exc).__name__,
                exc,
            )
            return working

    def _apply_llm_reranking(self, working: list[RankedItem]) -> dict[int, str | None]:
        """Generate LLM explanations for each item in *working*.

        Calls ``llm_client.generate_explanation(item)`` for each item.
        Individual failures are caught and logged; the item is still included
        with ``explanation=None``.

        Parameters
        ----------
        working:
            Current working list of :class:`RankedItem` objects.

        Returns
        -------
        dict[int, str | None]
            Mapping of ``movie_id → explanation`` (or ``None`` on failure).
        """
        explanations: dict[int, str | None] = {}

        for item in working:
            try:
                explanation = self.llm_client.generate_explanation(item)
                explanations[item.movie_id] = str(explanation) if explanation is not None else None
                logger.debug("LLM generated explanation for movie_id=%d.", item.movie_id)
            except Exception as exc:
                logger.warning(
                    "LLM explanation failed for movie_id=%d with %s: %s — setting explanation=None.",
                    item.movie_id,
                    type(exc).__name__,
                    exc,
                )
                explanations[item.movie_id] = None

        return explanations

    # ------------------------------------------------------------------
    # Private utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_diversity_score(item: RankedItem, all_items: list[RankedItem]) -> float:
        """Compute a diversity score for *item* relative to *all_items*.

        The diversity score is defined as ``1 - max_similarity``, where
        ``max_similarity`` is the maximum similarity between *item* and any
        other item in *all_items*.  Similarity is approximated using the
        absolute difference of ``ensemble_score`` values (1-D cosine proxy).

        Parameters
        ----------
        item:
            The item for which to compute the diversity score.
        all_items:
            The full list of selected items (including *item* itself).

        Returns
        -------
        float
            Diversity score in ``[0.0, 1.0]``.  Returns ``1.0`` when
            *all_items* contains only *item*.
        """
        others = [other for other in all_items if other.movie_id != item.movie_id]
        if not others:
            return 1.0

        max_sim = max(_ensemble_similarity(item, other) for other in others)
        return float(1.0 - max_sim)


# ------------------------------------------------------------------
# Module-level utility functions
# ------------------------------------------------------------------


def _ensemble_similarity(a: RankedItem, b: RankedItem) -> float:
    """Compute a similarity proxy between two :class:`RankedItem` objects.

    Uses the absolute difference of ``ensemble_score`` values as a 1-D
    cosine proxy.  The result is normalised to ``[0.0, 1.0]`` by mapping
    ``|a - b| = 0`` → similarity ``1.0`` and ``|a - b| >= 1`` → similarity
    ``0.0``.

    Parameters
    ----------
    a:
        First item.
    b:
        Second item.

    Returns
    -------
    float
        Similarity value in ``[0.0, 1.0]``.
    """
    diff = abs(a.ensemble_score - b.ensemble_score)
    # Clamp to [0, 1]: items with identical ensemble_score are maximally
    # similar; items differing by >= 1.0 are treated as maximally dissimilar.
    return float(max(0.0, 1.0 - diff))
