"""
Ranking pipeline (Stage 2) for the APEX Movie Recommendation System.

This module implements the second stage of the 3-stage recommendation pipeline.
It takes a list of :class:`~backend.pipeline_types.CandidateItem` objects from
the retrieval stage, scores them via a neural ensemble engine and/or a learned
ranker, blends the scores, and returns a sorted list of
:class:`~backend.pipeline_types.RankedItem` objects.

Circular-import constraint
--------------------------
Only ``CandidateItem`` and ``RankedItem`` are imported from
``backend.pipeline_types`` at module level.  No other ``backend/`` modules are
imported here.

Typical usage
-------------
>>> from backend.ranking_pipeline import RankingPipeline, RankingConfig
>>> from backend.pipeline_types import CandidateItem
>>>
>>> config = RankingConfig(ensemble_weight=0.7, ranker_weight=0.3)
>>> pipeline = RankingPipeline(
...     ensemble_engine=ensemble_engine,
...     learned_ranker=learned_ranker,
...     config=config,
... )
>>> ranked = pipeline.rank(candidates, user_context={"user_id": 42})
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from backend.pipeline_types import CandidateItem, RankedItem

logger = logging.getLogger(__name__)


@dataclass
class RankingConfig:
    """Configuration knobs for :class:`RankingPipeline`.

    Attributes
    ----------
    ensemble_weight:
        Weight applied to the ensemble score in the blended final score.
        Must be in ``[0.0, 1.0]``.  Defaults to ``0.7``.
    ranker_weight:
        Weight applied to the learned ranker score in the blended final score.
        Must be in ``[0.0, 1.0]``.  Defaults to ``0.3``.
    use_neural_ensemble:
        When ``True``, the ensemble engine is queried for scores.  When
        ``False`` (e.g. Tier-3 degradation), the retrieval score is used as
        the ensemble score.  Defaults to ``True``.
    use_learned_ranker:
        When ``True``, the learned ranker is applied on top of the ensemble
        score.  When ``False``, the ensemble score is used as the ranker score
        and ``ranker_weight`` is effectively set to ``0``.  Defaults to
        ``True``.
    """

    ensemble_weight: float = 0.7
    ranker_weight: float = 0.3
    use_neural_ensemble: bool = True
    use_learned_ranker: bool = True


class RankingPipeline:
    """Stage 2 of the recommendation pipeline: ensemble scoring and learned ranking.

    Scores each :class:`~backend.pipeline_types.CandidateItem` using a neural
    ensemble engine and/or a learned ranker, blends the scores, sorts the
    results descending by blended score, and returns a
    :class:`~backend.pipeline_types.RankedItem` list of the same length as the
    input.

    Both ``ensemble_engine`` and ``learned_ranker`` may be ``None``; the
    pipeline degrades gracefully by falling back to the retrieval score when
    the ensemble engine is unavailable, and to the ensemble score when the
    learned ranker is unavailable.

    Parameters
    ----------
    ensemble_engine:
        An object with a
        ``predict_ensemble(user_id: int, candidate_ids: list[int]) -> dict[int, float]``
        method.  May be ``None``; when ``None`` (or when
        ``config.use_neural_ensemble`` is ``False``) the retrieval score is
        used as the ensemble score.
    learned_ranker:
        An object with a
        ``predict(user_id: int, candidate_ids: list[int]) -> dict[int, float]``
        method.  May be ``None``; when ``None`` (or when
        ``config.use_learned_ranker`` is ``False``) the ensemble score is used
        as the ranker score and ``ranker_weight`` is set to ``0``.
    config:
        A :class:`RankingConfig` instance controlling blend weights and
        feature flags.
    """

    def __init__(
        self,
        ensemble_engine,
        learned_ranker,
        config: RankingConfig,
    ) -> None:
        self.ensemble_engine = ensemble_engine
        self.learned_ranker = learned_ranker
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        candidates: list[CandidateItem],
        user_context: dict,
    ) -> list[RankedItem]:
        """Score, blend, sort, and rank a list of candidate movies.

        Ranking steps:

        1. **Ensemble scoring** — query ``ensemble_engine.predict_ensemble()``
           for all candidates if ``use_neural_ensemble=True`` and the engine is
           available.  Falls back to each candidate's ``retrieval_score`` on
           failure or when the engine is unavailable.
        2. **Learned ranker scoring** — query ``learned_ranker.predict()`` if
           ``use_learned_ranker=True`` and the ranker is available.  Falls back
           to the ensemble score (and sets ``ranker_weight=0``) when the ranker
           is unavailable.
        3. **Score blending** —
           ``final_score = ensemble_weight * ensemble_score + ranker_weight * ranker_score``
        4. **Sort** — results are sorted descending by ``final_score``.
        5. **Rank assignment** — ``final_rank`` is assigned 1-indexed
           (``1`` = best).
        6. **Return** — a :class:`~backend.pipeline_types.RankedItem` list of
           exactly ``len(candidates)`` items.

        Parameters
        ----------
        candidates:
            List of :class:`~backend.pipeline_types.CandidateItem` objects
            from the retrieval stage.  May be empty.
        user_context:
            Arbitrary context dict.  The key ``"user_id"`` (``int``) is used
            when querying the ensemble engine and learned ranker; defaults to
            ``0`` if absent.

        Returns
        -------
        list[RankedItem]
            Ranked items sorted descending by blended score.  Always has
            exactly ``len(candidates)`` elements.

        Invariants
        ----------
        - ``len(result) == len(candidates)`` always.
        - ``{r.movie_id for r in result} == {c.movie_id for c in candidates}``
          (set-identity — no items added or dropped).
        - Items are sorted descending by blended score.
        - Calling ``rank()`` twice with identical inputs produces identical
          scores (determinism).
        """
        if not candidates:
            return []

        user_id: int = int(user_context.get("user_id", 0))
        candidate_ids: list[int] = [c.movie_id for c in candidates]

        # ----------------------------------------------------------------
        # Step 1: Ensemble scoring
        # ----------------------------------------------------------------
        ensemble_scores = self._get_ensemble_scores(user_id, candidate_ids, candidates)

        # ----------------------------------------------------------------
        # Step 2: Learned ranker scoring
        # ----------------------------------------------------------------
        ranker_scores, effective_ranker_weight = self._get_ranker_scores(user_id, candidate_ids, ensemble_scores)

        # ----------------------------------------------------------------
        # Step 3: Blend scores
        # ----------------------------------------------------------------
        effective_ensemble_weight = self.config.ensemble_weight

        blended: list[tuple[CandidateItem, float, float, float]] = []
        for candidate in candidates:
            mid = candidate.movie_id
            ens_score = ensemble_scores.get(mid, candidate.retrieval_score)
            rnk_score = ranker_scores.get(mid, ens_score)
            final_score = effective_ensemble_weight * ens_score + effective_ranker_weight * rnk_score
            blended.append((candidate, ens_score, rnk_score, final_score))

        # ----------------------------------------------------------------
        # Step 4: Sort descending by final_score (stable sort for determinism)
        # ----------------------------------------------------------------
        blended.sort(key=lambda t: t[3], reverse=True)

        # ----------------------------------------------------------------
        # Step 5: Assign final_rank and build RankedItem list
        # ----------------------------------------------------------------
        result: list[RankedItem] = []
        for rank_idx, (candidate, ens_score, rnk_score, _final_score) in enumerate(blended, start=1):
            result.append(
                RankedItem(
                    movie_id=candidate.movie_id,
                    retrieval_score=candidate.retrieval_score,
                    retrieval_source=candidate.retrieval_source,
                    ensemble_score=ens_score,
                    ranker_score=rnk_score,
                    final_rank=rank_idx,
                    retrieval_signals={},
                    metadata=dict(candidate.metadata),
                )
            )

        # Enforce invariants (defensive assertions — should never fire in
        # production but help catch regressions during testing).
        assert len(result) == len(candidates), f"rank() length invariant violated: {len(result)} != {len(candidates)}"
        input_ids = {c.movie_id for c in candidates}
        output_ids = {r.movie_id for r in result}
        assert input_ids == output_ids, (
            f"rank() set-identity invariant violated: added={output_ids - input_ids}, dropped={input_ids - output_ids}"
        )

        logger.debug(
            "rank() returning %d RankedItems for user_id=%d (ensemble_weight=%.2f, ranker_weight=%.2f).",
            len(result),
            user_id,
            effective_ensemble_weight,
            effective_ranker_weight,
        )
        return result

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _get_ensemble_scores(
        self,
        user_id: int,
        candidate_ids: list[int],
        candidates: list[CandidateItem],
    ) -> dict[int, float]:
        """Obtain ensemble scores for all candidates.

        Queries ``ensemble_engine.predict_ensemble()`` when
        ``use_neural_ensemble=True`` and the engine is available.  Falls back
        to each candidate's ``retrieval_score`` on any exception or when the
        engine is unavailable.

        Parameters
        ----------
        user_id:
            User identifier forwarded to the ensemble engine.
        candidate_ids:
            Ordered list of ``movie_id`` values to score.
        candidates:
            Original :class:`CandidateItem` list (used for fallback scores).

        Returns
        -------
        dict[int, float]
            Mapping of ``movie_id`` → ensemble score.
        """
        # Build a retrieval-score fallback map first (used on any failure path).
        fallback: dict[int, float] = {c.movie_id: c.retrieval_score for c in candidates}

        if not self.config.use_neural_ensemble:
            logger.debug("use_neural_ensemble=False; using retrieval_score as ensemble_score.")
            return fallback

        if self.ensemble_engine is None:
            logger.warning("Ensemble engine is None; falling back to retrieval_score for all candidates.")
            return fallback

        try:
            scores: dict[int, float] = self.ensemble_engine.predict_ensemble(user_id, candidate_ids)
            # Fill in any missing movie_ids with the retrieval-score fallback.
            for mid in candidate_ids:
                if mid not in scores:
                    logger.debug(
                        "Ensemble engine returned no score for movie_id=%d; using retrieval_score fallback.",
                        mid,
                    )
                    scores[mid] = fallback[mid]
            logger.debug(
                "Ensemble engine scored %d/%d candidates for user_id=%d.",
                len(scores),
                len(candidate_ids),
                user_id,
            )
            return scores

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Ensemble engine predict_ensemble() failed with %s: %s — "
                "falling back to retrieval_score for all candidates.",
                type(exc).__name__,
                exc,
            )
            return fallback

    def _get_ranker_scores(
        self,
        user_id: int,
        candidate_ids: list[int],
        ensemble_scores: dict[int, float],
    ) -> tuple[dict[int, float], float]:
        """Obtain learned ranker scores for all candidates.

        Queries ``learned_ranker.predict()`` when ``use_learned_ranker=True``
        and the ranker is available.  Falls back to the ensemble score (and
        returns ``ranker_weight=0``) on any exception or when the ranker is
        unavailable.

        Parameters
        ----------
        user_id:
            User identifier forwarded to the learned ranker.
        candidate_ids:
            Ordered list of ``movie_id`` values to score.
        ensemble_scores:
            Pre-computed ensemble scores used as fallback values.

        Returns
        -------
        tuple[dict[int, float], float]
            A ``(scores_dict, effective_ranker_weight)`` pair where
            ``scores_dict`` maps ``movie_id`` → ranker score and
            ``effective_ranker_weight`` is either ``config.ranker_weight``
            (ranker available) or ``0.0`` (ranker unavailable / disabled).
        """
        # Fallback: use ensemble scores and zero out ranker weight.
        fallback_scores: dict[int, float] = dict(ensemble_scores)
        zero_weight: float = 0.0

        if not self.config.use_learned_ranker:
            logger.debug("use_learned_ranker=False; using ensemble_score as ranker_score with ranker_weight=0.")
            return fallback_scores, zero_weight

        if self.learned_ranker is None:
            logger.warning("Learned ranker is None; using ensemble_score as ranker_score with ranker_weight=0.")
            return fallback_scores, zero_weight

        try:
            scores: dict[int, float] = self.learned_ranker.predict(user_id, candidate_ids)
            # Fill in any missing movie_ids with the ensemble-score fallback.
            for mid in candidate_ids:
                if mid not in scores:
                    logger.debug(
                        "Learned ranker returned no score for movie_id=%d; using ensemble_score fallback.",
                        mid,
                    )
                    scores[mid] = ensemble_scores.get(mid, 0.0)
            logger.debug(
                "Learned ranker scored %d/%d candidates for user_id=%d.",
                len(scores),
                len(candidate_ids),
                user_id,
            )
            return scores, self.config.ranker_weight

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Learned ranker predict() failed with %s: %s — "
                "using ensemble_score as ranker_score with ranker_weight=0.",
                type(exc).__name__,
                exc,
            )
            return fallback_scores, zero_weight
