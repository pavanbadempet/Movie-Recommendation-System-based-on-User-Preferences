"""
Standalone core functions extracted from Recommender for the architecture-design-perfection
refactor.  These functions contain the heavy logic previously inlined in
``Recommender._apply_query_mmr`` and ``Recommender._apply_learned_ranker``.

By moving the logic here, ``recommender.py`` becomes a thin orchestrator that
delegates to these helpers, keeping its line count well below 600 lines.

All functions are pure (no class state) and receive the data they need as
explicit parameters, making them independently testable.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def apply_query_mmr(
    candidates: list[dict],
    n: int,
    vectors: np.ndarray | None,
    index_for_movie_id_fn: Callable[[Any], int | None],
    lambda_param: float = 0.72,
) -> list[dict]:
    """Diversify query search results using Maximal Marginal Relevance (MMR).

    Standalone version of ``Recommender._apply_query_mmr``.  Selects up to
    *n* candidates from *candidates* by greedily maximising the MMR objective:

        ``mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity``

    where *relevance* is ``candidate["similarity_score"]`` and *max_similarity*
    is the maximum cosine similarity (dot product of L2-normalised SBERT
    vectors) between the candidate and any already-selected item.

    Parameters
    ----------
    candidates:
        Ordered list of candidate movie dicts (highest relevance first).
        Each dict must have an ``"id"`` key and a ``"similarity_score"`` key.
    n:
        Maximum number of items to return.
    vectors:
        Memory-mapped SBERT embedding matrix (shape ``[num_movies, dim]``).
        When ``None``, the function falls back to returning ``candidates[:n]``
        without diversity re-ranking.
    index_for_movie_id_fn:
        Callable that maps a ``movie_id`` value to its row index in *vectors*,
        or ``None`` when the movie has no embedding.  Typically
        ``Recommender._index_for_movie_id``.
    lambda_param:
        MMR trade-off parameter.  Higher values favour relevance; lower values
        favour diversity.  Defaults to ``0.72``.

    Returns
    -------
    list[dict]
        Up to *n* diversified candidates.
    """
    if len(candidates) <= n or vectors is None:
        return candidates[:n]

    selected: list[dict] = []
    remaining = candidates.copy()
    selected.append(remaining.pop(0))
    vector_cache: dict[int, np.ndarray] = {}

    def candidate_vector(movie: dict) -> np.ndarray | None:
        row_idx = index_for_movie_id_fn(movie.get("id"))
        if row_idx is None:
            return None
        if row_idx not in vector_cache:
            vector_cache[row_idx] = np.asarray(vectors[row_idx], dtype=np.float32)
        return vector_cache[row_idx]

    while remaining and len(selected) < n:
        best_idx = 0
        best_score = -float("inf")
        for idx, candidate in enumerate(remaining):
            candidate_vec = candidate_vector(candidate)
            if candidate_vec is None:
                continue
            relevance = float(candidate.get("similarity_score") or 0)

            max_similarity = 0.0
            for chosen in selected:
                chosen_vec = candidate_vector(chosen)
                if chosen_vec is None:
                    continue
                max_similarity = max(max_similarity, float(np.dot(candidate_vec, chosen_vec)))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(remaining.pop(best_idx))

    return selected[:n]


def apply_learned_ranker(
    candidates: list[dict[str, Any]],
    user_id: int = 0,
    precomputed_ensemble_scores: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Apply the APEX MMoE neural ranking pipeline to *candidates*.

    Standalone version of ``Recommender._apply_learned_ranker``.  Scores each
    candidate using the APEX ensemble engine and the MMoE ONNX ranker, then
    blends the scores using the YouTube-style formulation:

        ``final_score = base_score * 0.3 + ensemble_score * 0.4 + (ctr * watch * sat) * 0.3``

    After scoring, applies:
      - Differential Privacy on the user embedding (Gaussian ε-DP, GDPR/EU AI Act)
      - Long-horizon RL score adjustment (churn risk + preference stability)
      - Cold-start content boost (uncertainty_estimator)
      - Contextual bandit exploration (Thompson Sampling or UCB)

    Parameters
    ----------
    candidates:
        List of candidate movie dicts.  Each dict must have an ``"id"`` key
        and a ``"similarity_score"`` key.  Modified in-place (scores updated).
    user_id:
        Integer user identifier used for ensemble and ranker lookups.
        Defaults to ``0`` (anonymous / cold-start user).
    precomputed_ensemble_scores:
        Optional pre-computed ensemble scores keyed by *safe* item id
        (``item_id % 9724``).  When provided, the ensemble forward pass is
        skipped to avoid redundant computation.

    Returns
    -------
    list[dict[str, Any]]
        Re-scored and re-sorted candidates.  Returns *candidates* unchanged
        when the neural stack is unavailable (graceful degradation).
    """
    if not candidates:
        return candidates

    try:
        import torch

        from backend.models.ensemble_engine import get_apex_engine
        from backend.models.mmoe_ranker import get_mmoe_ranker  # noqa: F401 — kept for parity

        # The models were trained on specific vocab sizes. We enforce safe modulus.
        apex_engine = get_apex_engine(num_users=610, num_items=9724)

        item_ids = []
        for c in candidates:
            try:
                item_ids.append(int(c.get("id", 0)))
            except Exception:
                item_ids.append(0)

        safe_user_id_apex = user_id % 610
        safe_item_ids_apex = [i % 9724 for i in item_ids]

        safe_user_id_mmoe = user_id % 611
        safe_item_ids_mmoe = [i % 193610 for i in item_ids]

        # -----------------------------------------------------------------
        # Differential Privacy: privatize the user embedding before scoring.
        # Applies Gaussian (ε, δ)-DP noise to a per-request copy of the user's
        # latent representation so individual preferences cannot be
        # reverse-engineered from the embedding space — required for GDPR /
        # EU AI Act compliance.
        #
        # Thread-safety note: we operate on a local numpy copy of the
        # embedding and pass it directly to predict_ensemble via the
        # session_sequence override pathway.  The shared embedding table
        # is never modified, eliminating the race condition that would
        # arise from concurrent requests for the same user_id.
        # -----------------------------------------------------------------
        dp_epsilon = float(__import__("os").getenv("APEX_DP_EPSILON", "1.0"))
        privatized_user_emb_tensor = None
        try:
            from backend.privacy.privacy_preserving_ml import privatize_user_embedding

            with torch.no_grad():
                raw_user_emb = (
                    apex_engine.lightgcn.user_embedding(torch.tensor([safe_user_id_apex], dtype=torch.long))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

            privatized_np = privatize_user_embedding(
                raw_user_emb,
                epsilon=dp_epsilon,
                delta=1e-5,
                mechanism="gaussian",
            )
            privatized_user_emb_tensor = torch.tensor(privatized_np, dtype=torch.float32)
            logger.debug(
                "DP Gaussian noise applied to user embedding copy (user_id=%d, ε=%.2f)",
                user_id,
                dp_epsilon,
            )
        except Exception as dp_exc:
            logger.debug("Differential privacy embedding noise skipped: %s", dp_exc)

        # -----------------------------------------------------------------
        # Long-horizon RL: load user events for churn risk + stability scoring
        # -----------------------------------------------------------------
        user_events: list[dict] = []
        try:
            from backend.events import iter_events

            user_events = [e for e in iter_events() if str(e.get("user_id", "")) == str(user_id)]
        except Exception as evt_exc:
            logger.debug("Could not load user events for long-horizon RL: %s", evt_exc)

        churn_risk = 0.5
        preference_stability = 0.5
        if user_events:
            try:
                from backend.intelligence.long_horizon_rl import compute_preference_stability, estimate_churn_risk

                churn_risk = estimate_churn_risk(user_events, lookback_days=30)
                preference_stability = compute_preference_stability(user_events, window_days=90)
            except Exception as lh_exc:
                logger.debug("Long-horizon RL metrics unavailable: %s", lh_exc)

        # -----------------------------------------------------------------
        # Cold-start detection for content boost
        # -----------------------------------------------------------------
        user_interaction_count = len(user_events)
        is_cold_start = user_interaction_count < 5

        # 1. Get Ensemble Scores — use precomputed if available to avoid redundant forward pass.
        # When a privatized embedding was computed above, pass it as user_emb_override so the
        # ensemble uses the DP-noised vector without touching the shared embedding table.
        if precomputed_ensemble_scores is not None:
            ensemble_scores = precomputed_ensemble_scores
        else:
            ensemble_scores = apex_engine.predict_ensemble(
                safe_user_id_apex,
                safe_item_ids_apex,
                user_emb_override=privatized_user_emb_tensor,
            )

        from backend.serving.onnx_engine import get_onnx_engine

        onnx = get_onnx_engine()

        # 2. Get MMoE Task Predictions via High-Speed ONNX Runtime (No PyTorch Overhead)
        u_mmoe = np.array([safe_user_id_mmoe] * len(item_ids), dtype=np.int64)
        i_mmoe = np.array(safe_item_ids_mmoe, dtype=np.int64)

        p_ctr, p_watch, p_sat = onnx.predict_mmoe(u_mmoe, i_mmoe)

        # 3. Final Unified Ranking Equation + long-horizon RL + cold-start boost
        for idx, candidate in enumerate(candidates):
            base_score = float(candidate.get("similarity_score") or 0.0)
            ens_score = ensemble_scores.get(safe_item_ids_apex[idx], 0.0)

            ctr = float(p_ctr[idx])
            watch = float(p_watch[idx])
            sat = float(p_sat[idx])

            # The YouTube formulation: (P(click) * P(watch) * P(satisfaction)) + Structural Prior
            # We blend the retrieval base score to ensure query relevance isn't lost
            final_score = (base_score * 0.3) + (ens_score * 0.4) + (ctr * watch * sat * 0.3)

            # Apply long-horizon RL score adjustment (churn risk + preference stability)
            if user_events:
                try:
                    from backend.intelligence.long_horizon_rl import long_horizon_score_adjustment

                    lh_adjustment = long_horizon_score_adjustment(
                        candidate=candidate,
                        user_events=user_events,
                        churn_risk=churn_risk,
                        preference_stability=preference_stability,
                    )
                    final_score += lh_adjustment
                except Exception as lh_exc:
                    logger.debug("Long-horizon score adjustment failed for item %s: %s", candidate.get("id"), lh_exc)

            # Apply cold-start boost: boost content signals for users with < 5 interactions
            if is_cold_start:
                try:
                    from backend.intelligence.uncertainty_estimator import cold_start_boost

                    boost_multiplier = cold_start_boost(
                        movie=candidate,
                        user_interaction_count=user_interaction_count,
                    )
                    final_score *= boost_multiplier
                except Exception as cs_exc:
                    logger.debug("Cold-start boost failed for item %s: %s", candidate.get("id"), cs_exc)

            candidate["similarity_score"] = final_score
            candidate["metrics"] = {
                "p_click": round(ctr, 4),
                "p_watch": round(watch, 4),
                "p_satisfaction": round(sat, 4),
                "ensemble_prior": round(ens_score, 4),
                "churn_risk": round(churn_risk, 4),
                "preference_stability": round(preference_stability, 4),
                "cold_start": is_cold_start,
            }

            if "explanation" not in candidate:
                candidate["explanation"] = []
            candidate["explanation"].insert(0, "DeepSeek-inspired MMoE Multi-Task Ranker")

        # Apply Contextual Bandit Exploration (Thompson Sampling / UCB)
        from backend.intelligence.contextual_bandit import get_bandit_engine

        bandit = get_bandit_engine()
        # 10% chance to run UCB (cold-start discoverability), otherwise Thompson Sampling
        strategy = "ucb" if torch.rand(1).item() < 0.1 else "thompson"
        candidates = bandit.apply_exploration(candidates, strategy=strategy)

        # Final Safety Sort
        candidates.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        return candidates

    except Exception as exc:
        logger.error("MMoE Neural Ranker failed, falling back to base retrieval: %s", exc)
        return candidates


# ---------------------------------------------------------------------------
# Fallback implementations extracted from Recommender to keep recommender.py
# under 600 lines. Each function accepts the Recommender instance as `rec`
# and delegates back through its public/private methods as needed.
# ---------------------------------------------------------------------------

import re as _re


def sparse_search_movies(rec, query: str, limit: int = 20) -> list:
    """TF-IDF + relevance-scoring fallback for Recommender.search_movies."""
    import numpy as _np
    import pandas as _pd
    import logging

    if not query or rec._movies is None:
        return []

    q_lower = query.lower().strip()
    q_norm = _re.sub(r"[^a-z0-9]+", " ", q_lower).strip()

    def text_column(column, index_subset=None):
        if column not in rec._movies.columns:
            target_index = index_subset if index_subset is not None else rec._movies.index
            return _pd.Series("", index=target_index, dtype="string")
        series = rec._movies[column]
        if index_subset is not None:
            series = series.loc[index_subset]
        return series.astype(object).fillna("").astype(str)

    def normalized_text_column(column, index_subset=None):
        frame_id = id(rec._movies)
        if rec._search_text_cache_frame_id != frame_id:
            rec._search_text_cache.clear()
            rec._search_text_cache_frame_id = frame_id
        
        cached = rec._search_text_cache.get(column)
        if cached is None or not cached.index.equals(rec._movies.index):
            normalized = (
                text_column(column)
                .str.lower()
                .str.replace(r"[^a-z0-9]+", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            rec._search_text_cache[column] = normalized
            cached = normalized
            
        if index_subset is not None:
            return cached.loc[index_subset]
        return cached

    def numeric_column(column, index_subset=None):
        if column not in rec._movies.columns:
            target_index = index_subset if index_subset is not None else rec._movies.index
            return _pd.Series(0.0, index=target_index, dtype="float32")
        series = rec._movies[column]
        if index_subset is not None:
            series = series.loc[index_subset]
        return _pd.to_numeric(series, errors="coerce").fillna(0.0)

    # Attempt vectorized TF-IDF search
    use_vectorized = False
    try:
        rec._ensure_sparse_retrieval_index()
        if rec._tfidf_matrix is not None and rec._vectorizer is not None:
            use_vectorized = True
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to warm sparse TF-IDF index: %s", exc)

    if use_vectorized:
        from sklearn.preprocessing import normalize
        query_vec = rec._vectorizer.transform([query])
        query_vec_norm = normalize(query_vec, norm='l2', axis=1)
        scores = rec._tfidf_matrix.dot(query_vec_norm.T).toarray().flatten()
        
        # Take a candidate pool of limit * 10, capped at catalog length
        top_k = min(limit * 10, len(rec._movies))
        if len(scores) <= top_k:
            top_indices = _np.argsort(scores)[::-1]
        else:
            top_indices = _np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[_np.argsort(scores[top_indices])[::-1]]
            
        top_indices = [idx for idx in top_indices if scores[idx] > 0.0]
        if not top_indices:
            return []
            
        matches = rec._movies.iloc[top_indices].copy()
        matches_index = matches.index
        
        # Relevance starts with TF-IDF base score scaled
        matches["relevance"] = scores[top_indices] * 20.0
    else:
        # Fallback: scan all movies using pandas regex (original behavior)
        titles = text_column("title")
        overviews = text_column("overview")
        genres = text_column("genres")
        normalized_titles = normalized_text_column("title")
        normalized_overviews = normalized_text_column("overview")
        normalized_genres = normalized_text_column("genres")

        mask_title = titles.str.lower().str.contains(q_lower, regex=False, na=False)
        if q_norm:
            mask_title = mask_title | normalized_titles.str.contains(q_norm, regex=False, na=False)
        mask_overview = overviews.str.lower().str.contains(q_lower, regex=False, na=False)
        if q_norm:
            mask_overview = mask_overview | normalized_overviews.str.contains(q_norm, regex=False, na=False)
        mask_genre = genres.str.lower().str.contains(q_lower, regex=False, na=False)
        if q_norm:
            mask_genre = mask_genre | normalized_genres.str.contains(q_norm, regex=False, na=False)

        matches = rec._movies[mask_title | mask_overview | mask_genre].copy()
        if len(matches) == 0:
            return []
        matches["relevance"] = 0.0
        matches_index = matches.index

    # Run original relevance boosting logic only on the matches subset
    titles = text_column("title", matches_index)
    overviews = text_column("overview", matches_index)
    genres = text_column("genres", matches_index)
    normalized_titles = normalized_text_column("title", matches_index)
    normalized_overviews = normalized_text_column("overview", matches_index)
    normalized_genres = normalized_text_column("genres", matches_index)

    mask_title = titles.str.lower().str.contains(q_lower, regex=False, na=False)
    if q_norm:
        mask_title = mask_title | normalized_titles.str.contains(q_norm, regex=False, na=False)
    mask_overview = overviews.str.lower().str.contains(q_lower, regex=False, na=False)
    if q_norm:
        mask_overview = mask_overview | normalized_overviews.str.contains(q_norm, regex=False, na=False)
    mask_genre = genres.str.lower().str.contains(q_lower, regex=False, na=False)
    if q_norm:
        mask_genre = mask_genre | normalized_genres.str.contains(q_norm, regex=False, na=False)

    m_title = titles.str.lower()
    m_title_norm = normalized_titles
    exact_title = m_title == q_lower
    if q_norm:
        exact_title = exact_title | (m_title_norm == q_norm)
    starts_with_boundary = (
        exact_title
        | m_title.str.startswith(f"{q_lower} ", na=False)
        | m_title.str.startswith(f"{q_lower}:", na=False)
        | m_title.str.startswith(f"{q_lower}-", na=False)
    )
    if q_norm:
        starts_with_boundary = starts_with_boundary | m_title_norm.str.startswith(f"{q_norm} ", na=False)
    starts_with_prefix = m_title.str.startswith(q_lower, na=False) & ~starts_with_boundary
    if q_norm:
        starts_with_prefix = starts_with_prefix | (
            m_title_norm.str.startswith(q_norm, na=False) & ~starts_with_boundary
        )
    matches.loc[exact_title, "relevance"] += 50.0
    matches.loc[starts_with_boundary, "relevance"] += 20.0
    matches.loc[starts_with_prefix, "relevance"] += 8.0
    matches.loc[m_title.str.contains(q_lower, regex=False), "relevance"] += 10.0
    if q_norm:
        matches.loc[m_title_norm.str.contains(q_norm, regex=False), "relevance"] += 10.0
    matches.loc[mask_genre, "relevance"] += 5.0
    matches.loc[mask_overview, "relevance"] += 3.0

    popularity = numeric_column("popularity", matches_index).clip(lower=0)
    vote_count = numeric_column("vote_count", matches_index).clip(lower=0)
    matches["relevance"] += _np.log1p(popularity) * 2.0
    matches["relevance"] += _np.log1p(vote_count) * 0.8

    strong_exact_exists = bool((exact_title & ((vote_count >= 500) | (popularity >= 20))).any())
    if strong_exact_exists:
        weak_exact_duplicate = exact_title & (vote_count < 100) & (popularity < 15)
        matches.loc[weak_exact_duplicate, "relevance"] -= 55.0

    franchise_continuation = (
        m_title.str.startswith(f"{q_lower}: ", na=False)
        | m_title.str.startswith(f"{q_lower} - ", na=False)
        | (m_title_norm.str.startswith(f"{q_norm} ", na=False) if q_norm else False)
    ) & ((vote_count >= 250) | (popularity >= 20))
    matches.loc[franchise_continuation, "relevance"] += 16.0

    matches = matches.sort_values("relevance", ascending=False).head(limit)

    # Pre-cache optimization check: use cached dict records if matching indexes to avoid calling to_dict() per row!
    response_records = []
    if hasattr(rec, "_movie_records") and rec._movie_records:
        for idx in matches.index:
            if idx < len(rec._movie_records):
                response_records.append(rec._clean_response_record(rec._movie_records[idx]))
            else:
                response_records.append(rec._clean_response_record(rec._movies.iloc[idx].to_dict()))
    else:
        response_records = [rec._clean_response_record(record) for record in matches.to_dict(orient="records")]

    return response_records


def metadata_recommend_by_index(rec, movie_idx: int, n: int = 10) -> list:
    """Content-based fallback recommender when vector artifacts are unavailable."""
    import numpy as _np
    import pandas as _pd

    if rec._movies is None or movie_idx < 0 or movie_idx >= len(rec._movies):
        return []

    rec.refresh_behavior_features()
    try:
        rec._ensure_item_retrieval_index()
        content_scores = rec._item_tfidf_matrix[movie_idx].dot(rec._item_tfidf_matrix.T).toarray().ravel()
    except Exception as exc:
        logger.warning("Item sparse similarity unavailable; using metadata-only scoring: %s", exc)
        content_scores = _np.zeros(len(rec._movies), dtype=_np.float32)

    query_movie = rec.get_movie_by_index(movie_idx)
    q_genres = rec._genre_set(query_movie)
    q_director = str(query_movie.get("director") or "").strip().lower()
    q_language = str(query_movie.get("original_language") or "").strip().lower()
    query_votes = float(query_movie.get("vote_count") or 0)
    query_runtime = float(query_movie.get("runtime") or 0)
    q_title_tokens = {
        token
        for token in str(query_movie.get("title") or "").lower().replace(":", " ").replace("-", " ").split()
        if len(token) >= 4 and token not in {"movie", "part", "chapter"}
    }

    scores = _np.asarray(content_scores, dtype=_np.float32) * 0.76

    def numeric_array(column, default=0.0):
        if column not in rec._movies.columns:
            return _np.full(len(rec._movies), default, dtype=_np.float32)
        return _pd.to_numeric(rec._movies[column], errors="coerce").fillna(default).to_numpy(dtype=_np.float32)

    genre_overlap = _np.zeros(len(rec._movies), dtype=_np.float32)
    if q_genres and "genres" in rec._movies.columns:
        genres_col = rec._movies["genres"].astype(object).fillna("").astype(str).str.lower()
        for genre in q_genres:
            genre_mask = genres_col.str.contains(genre, regex=False, na=False).to_numpy()
            genre_overlap += genre_mask.astype(_np.float32)
        genre_ratio = genre_overlap / max(len(q_genres), 1)
        scores += genre_ratio * 0.12
        scores += _np.minimum(genre_overlap, 2) * 0.02

    if q_director and q_director != "unknown" and "director" in rec._movies.columns:
        directors = rec._movies["director"].astype(object).fillna("").astype(str).str.lower()
        scores += directors.eq(q_director).to_numpy().astype(_np.float32) * 0.08

    if q_language and "original_language" in rec._movies.columns:
        languages = rec._movies["original_language"].astype(str).str.lower()
        scores += languages.eq(q_language).to_numpy().astype(_np.float32) * 0.025

    if "content_quality_score" in rec._movies.columns:
        scores += _np.clip(numeric_array("content_quality_score"), 0, 1) * 0.12
    else:
        vote_average = numeric_array("vote_average")
        vote_count_arr = numeric_array("vote_count")
        confidence = _np.minimum(1.0, _np.log1p(_np.maximum(vote_count_arr, 0)) / 8.0)
        scores += _np.clip(vote_average / 10.0, 0, 1) * confidence * 0.10

    if "vote_count" in rec._movies.columns:
        vc = numeric_array("vote_count")
        scores += _np.minimum(1.0, _np.log1p(_np.maximum(vc, 0)) / 10.0) * 0.06

    if "popularity" in rec._movies.columns:
        pop = numeric_array("popularity")
        scores += _np.minimum(1.0, _np.log1p(_np.maximum(pop, 0)) / 8.0) * 0.08

    scores[movie_idx] = -_np.inf
    if len(scores) <= 1:
        return []

    candidate_count = min(max(n * 40, 250), len(scores) - 1)
    candidate_indices = _np.argpartition(scores, -candidate_count)[-candidate_count:]
    candidate_indices = candidate_indices[_np.argsort(scores[candidate_indices])[::-1]]

    results = []
    for idx in candidate_indices:
        idx = int(idx)
        if len(results) >= n:
            break
        if not _np.isfinite(scores[idx]) or scores[idx] <= 0:
            continue

        movie = rec.get_movie_by_index(idx)
        movie_genres = rec._genre_set(movie)
        content_score = float(content_scores[idx]) if len(content_scores) > idx else 0.0
        shared_genres = q_genres & movie_genres
        candidate_votes = float(movie.get("vote_count") or 0)
        candidate_runtime = float(movie.get("runtime") or 0)
        semantic_affinity = rec._semantic_affinity_for_indices(movie_idx, idx)
        semantic_score = float(semantic_affinity["score"])

        if movie.get("public_demo_eligible") is False:
            continue
        if movie.get("recommendable") is False:
            continue
        if query_runtime >= 60 and 0 < candidate_runtime < 60:
            continue
        if "documentary" in movie_genres and "documentary" not in q_genres:
            continue
        if "tv movie" in movie_genres and "tv movie" not in q_genres:
            continue
        if not shared_genres and content_score < 0.10:
            continue
        min_votes = 500 if query_votes >= 5000 else 50
        if candidate_votes < min_votes and content_score < 0.16:
            continue
        if {"animation", "family"} & movie_genres and not ({"animation", "family"} & q_genres):
            scores[idx] -= 0.12
        if "comedy" in movie_genres and "comedy" not in q_genres and content_score < 0.16:
            continue

        scores[idx] += semantic_score * 0.18

        title_tokens = {
            token
            for token in str(movie.get("title") or "").lower().replace(":", " ").replace("-", " ").split()
            if len(token) >= 4
        }
        if q_title_tokens & title_tokens and "documentary" not in movie_genres:
            scores[idx] += 0.14

        reasons = []
        if shared_genres:
            reasons.append(f"Shared genres: {', '.join(sorted(g.title() for g in shared_genres)[:2])}")
        if content_score >= 0.18:
            reasons.append("Similar story and setting")
        reasons.extend(semantic_affinity.get("reasons") or [])
        if q_director and str(movie.get("director") or "").strip().lower() == q_director:
            reasons.append(f"Same director ({movie.get('director')})")
        if q_language and str(movie.get("original_language") or "").strip().lower() == q_language:
            reasons.append("Same catalog language")
        if float(movie.get("vote_average") or 0) >= 7.5:
            reasons.append(f"Strong audience rating ({float(movie.get('vote_average') or 0):.1f}/10)")
        if not reasons:
            reasons.append("Closest content and catalog-quality match")

        behavior_boost, behavior_reasons = rec._behavior_boost(movie.get("id"))
        score = float(scores[idx] + behavior_boost)
        movie["similarity_score"] = score
        movie["retrieval_stage"] = "content_sparse_fallback"
        movie["retrieval_signals"] = {
            "content_sparse": round(content_score, 4),
            "semantic_twin": round(semantic_score, 4),
            "semantic_twin_details": semantic_affinity,
            "genre_overlap": round(float(len(shared_genres) / max(len(q_genres), 1)), 4),
            "metadata": round(float(scores[idx]), 4),
            "behavior": round(behavior_boost, 4),
            "vector_artifacts_loaded": False,
            "vector_artifact_status": rec._artifact_status,
        }
        movie["explanation"] = (reasons + behavior_reasons)[:5]
        movie["explanation_text"] = " | ".join(movie["explanation"])
        results.append(movie)

    results.sort(key=lambda item: float(item.get("similarity_score") or 0), reverse=True)
    results = rec._apply_learned_ranker(results)
    return rec._quality_gate_item_recommendations(results, query_movie, n)[:n]


def legacy_ai_search(rec, query: str, n: int = 10, fetch_k: int = 80) -> list:
    """Legacy multi-stage AI search: SBERT encoding + FAISS + cross-encoder + MMR."""
    import os as _os

    import numpy as _np

    from backend.intelligence.query_understanding import intent_score, parse_query_intent

    rec.refresh_behavior_features()
    query_intent = parse_query_intent(query)
    fetch_k = max(n, min(fetch_k, len(rec._movies)))
    dense_scores: dict = {}
    sparse_scores: dict = {}
    dense_error = None

    rec._ensure_sparse_retrieval_index()
    if rec._tfidf_matrix is not None and rec._vectorizer is not None:
        from sklearn.metrics.pairwise import cosine_similarity

        query_sparse = rec._vectorizer.transform([query])
        sparse_similarities = cosine_similarity(query_sparse, rec._tfidf_matrix).ravel()
        sparse_indices = _np.argsort(sparse_similarities)[::-1][:fetch_k]
        sparse_scores = {
            int(idx): float(sparse_similarities[int(idx)])
            for idx in sparse_indices
            if sparse_similarities[int(idx)] > 0
        }

    if rec._dense_query_enabled() and rec._index is not None and rec._vectors is not None:
        try:
            encoder = rec._get_query_encoder()
            query_embedding = encoder.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding / _np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_embedding = query_embedding.astype(_np.float32)
            distances, indices = rec._index.search(query_embedding, fetch_k)
            dense_scores = {
                int(idx): float(distances[0][rank])
                for rank, idx in enumerate(indices[0])
                if 0 <= idx < len(rec._movies)
            }
        except Exception as exc:
            dense_error = str(exc)
            logger.warning("Dense query retrieval skipped: %s", exc)

    normalized_sparse = rec._normalize_score_map(sparse_scores)
    normalized_dense = rec._normalize_score_map(dense_scores)
    candidate_indices = set(normalized_sparse) | set(normalized_dense)
    if not candidate_indices:
        return rec.search_movies(query, limit=n)

    alpha = 0.62 if normalized_dense else 0.0
    ranked_candidates = []
    for idx in candidate_indices:
        movie = rec._clean_response_record(rec._movies.iloc[idx].to_dict())
        sparse_score = normalized_sparse.get(idx, 0.0)
        dense_score = normalized_dense.get(idx, 0.0)
        metadata_score = rec._popularity_quality_score(movie)
        behavior_boost, behavior_reasons = rec._behavior_boost(movie.get("id"))
        intent_boost, intent_reasons = intent_score(movie, query_intent)
        hybrid_score = (
            alpha * dense_score + (1 - alpha) * sparse_score + 0.10 * metadata_score + behavior_boost + intent_boost
        )
        explanation = []
        if dense_score > 0:
            explanation.append("semantic meaning match")
        if sparse_score > 0:
            explanation.append("keyword/entity match")
        if metadata_score > 0.5:
            explanation.append("strong catalog quality signal")
        explanation.extend(intent_reasons)
        explanation.extend(behavior_reasons)
        if dense_error and not normalized_dense:
            explanation.append("dense query model unavailable; sparse fallback used")
        if not explanation:
            explanation.append("best available catalog match")
        movie["similarity_score"] = float(hybrid_score)
        movie["retrieval_stage"] = "hybrid" if normalized_dense else "sparse_metadata"
        movie["retrieval_signals"] = {
            "dense": round(dense_score, 4),
            "sparse": round(sparse_score, 4),
            "metadata": round(metadata_score, 4),
            "behavior": round(behavior_boost, 4),
            "intent": round(intent_boost, 4),
            "intent_features": query_intent,
        }
        movie["explanation"] = explanation[:4]
        movie["explanation_text"] = " | ".join(movie["explanation"])
        ranked_candidates.append(movie)

    ranked_candidates.sort(key=lambda item: item["similarity_score"], reverse=True)

    if rec._cross_encoder_enabled() and len(ranked_candidates) > 1:
        try:
            reranker = rec._get_cross_encoder()
            rerank_window = ranked_candidates[
                : min(len(ranked_candidates), int(_os.getenv("NOVA_RERANK_WINDOW", "30")))
            ]
            pairs = [
                [query, f"{item.get('title', '')}. {item.get('genres', '')}. {item.get('overview', '')}"]
                for item in rerank_window
            ]
            rerank_scores = reranker.predict(pairs)
            for item, rerank_score in zip(rerank_window, rerank_scores, strict=False):
                item["retrieval_signals"]["cross_encoder"] = round(float(rerank_score), 4)
                item["similarity_score"] = 0.75 * float(item["similarity_score"]) + 0.25 * float(rerank_score)
                item["retrieval_stage"] = f"{item['retrieval_stage']}_cross_encoder"
                item["explanation"] = ["neural reranker selected this match"] + item["explanation"][:3]
                item["explanation_text"] = " | ".join(item["explanation"])
            ranked_candidates = rerank_window + ranked_candidates[len(rerank_window) :]
            ranked_candidates.sort(key=lambda item: item["similarity_score"], reverse=True)
        except Exception as exc:
            logger.warning("Cross-encoder reranking skipped: %s", exc)

    ranked_candidates = rec._apply_learned_ranker(ranked_candidates)
    return rec._apply_query_mmr(ranked_candidates, n=n)


def recommend_batch(rec, movie_ids: list, n: int = 10) -> dict:
    """Batch recommendations — more efficient than N individual calls."""
    import contextlib
    import time as _time

    import numpy as _np

    if not movie_ids or rec._vectors is None or rec._index is None:
        return {mid: rec.recommend_by_id(mid, n) for mid in movie_ids}

    results: dict = {}
    uncached_ids: list = []
    if not hasattr(rec, "_rec_cache"):
        rec._rec_cache = {}
    for mid in movie_ids:
        cached = rec._rec_cache.get((mid, n))
        if cached is not None and _time.time() - cached[0] < 300:
            results[mid] = cached[1]
        else:
            uncached_ids.append(mid)

    if not uncached_ids:
        return results

    valid_pairs: list = []
    for mid in uncached_ids:
        idx = rec._index_for_movie_id(mid)
        if idx is not None:
            valid_pairs.append((mid, idx))

    if not valid_pairs:
        for mid in uncached_ids:
            results[mid] = []
        return results

    fetch_k = min(100, getattr(rec._index, "ntotal", len(rec._movies)))
    if hasattr(rec._index, "nprobe"):
        rec._index.nprobe = min(50, getattr(rec._index, "nlist", 10))
    if hasattr(rec._index, "hnsw"):
        rec._index.hnsw.efSearch = 200

    batch_vectors = _np.vstack(
        [_np.ascontiguousarray(rec._vectors[idx].reshape(1, -1).astype(_np.float32)) for _, idx in valid_pairs]
    )
    all_distances, all_indices = rec._index.search(batch_vectors, fetch_k)

    for i, (mid, movie_idx) in enumerate(valid_pairs):
        _ = all_distances[i : i + 1], all_indices[i : i + 1]
        result = rec.recommend_by_index(movie_idx, n)
        results[mid] = result
        if len(rec._rec_cache) >= 500:
            with contextlib.suppress(StopIteration):
                rec._rec_cache.pop(next(iter(rec._rec_cache)))
        rec._rec_cache[(mid, n)] = (_time.time(), result)

    for mid in uncached_ids:
        if mid not in results:
            results[mid] = []

    return results


def get_all_titles(rec, limit: int = 100000) -> list:
    """Return lightweight movie ID + title list for autocomplete."""
    import pandas as _pd

    if rec._movies is None:
        return []

    cols = ["id", "title"]
    for col in ("release_date", "popularity", "genres"):
        if col in rec._movies.columns:
            cols.append(col)

    titles_df = rec._movies[cols].copy()

    if "release_date" in titles_df.columns:
        years = _pd.to_datetime(titles_df["release_date"], errors="coerce").dt.year
        mask = years.notna() & (years > 0)
        titles_df.loc[mask, "title"] = titles_df.loc[mask, "title"] + " (" + years[mask].astype(int).astype(str) + ")"

    if "genres" in titles_df.columns:
        mask = titles_df["genres"].notna() & (titles_df["genres"] != "")

        def get_top_genres(g_str):
            try:
                parts = str(g_str).split(",")
                return ", ".join(p.strip() for p in parts[:2])
            except Exception:
                return str(g_str)

        top_genres = titles_df.loc[mask, "genres"].apply(get_top_genres)
        titles_df.loc[mask, "title"] = titles_df.loc[mask, "title"] + " - " + top_genres

    if "popularity" in titles_df.columns:
        titles_df = titles_df.sort_values("popularity", ascending=False)
    else:
        titles_df = titles_df.sort_values("title")

    if limit and limit > 0:
        titles_df = titles_df.head(limit)

    return titles_df[["id", "title"]].to_dict(orient="records")


def user_profile_fallback(rec, profile: dict, result_limit: int) -> list:
    """Metadata-only fallback for recommend_for_user_profile (no pipeline available)."""
    negative_ids = {int(mid) for mid in profile.get("negative_movie_ids") or []}
    seed_events = [
        ev for ev in (profile.get("recent_events") or []) if ev.get("movie_id") is not None and not ev.get("negative")
    ]
    genre_affinity = rec._genre_affinity_from_profile(profile)
    scored: dict = {}

    def add_candidate(candidate, score, reason, stage):
        cid = candidate.get("id")
        if cid is None:
            return
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            return
        if cid in negative_ids:
            return
        item = dict(candidate)
        genre_boost = min(0.15, sum(genre_affinity.get(g, 0.0) for g in rec._genre_set(item)) * 0.045)
        final_score = float(score) + genre_boost
        current = scored.get(cid)
        if current is None or final_score > float(current.get("similarity_score") or 0):
            explanations = list(item.get("explanation") or [])
            explanations.insert(0, reason)
            if genre_boost > 0:
                explanations.insert(1, "matches your genre affinity")
            item["similarity_score"] = final_score
            item["retrieval_stage"] = stage
            item["retrieval_signals"] = {
                **(item.get("retrieval_signals") or {}),
                "personalization": round(final_score, 4),
                "genre_affinity": round(genre_boost, 4),
            }
            item["explanation"] = explanations[:5]
            item["explanation_text"] = " | ".join(item["explanation"])
            scored[cid] = item

    seen_seed_ids: set = set()
    for event_rank, event in enumerate(seed_events[:8]):
        seed_movie_id = int(event["movie_id"])
        seen_seed_ids.add(seed_movie_id)
        event_weight = float(event.get("weight") or 1.0)
        recency = rec._event_recency_decay(event.get("event_ts"))
        seed_weight = event_weight * recency / (event_rank + 1)
        for candidate in rec.recommend_by_id(seed_movie_id, n=min(30, max(result_limit * 4, 12))):
            if candidate.get("id") in seen_seed_ids:
                continue
            add_candidate(
                candidate,
                score=float(candidate.get("similarity_score") or 0.0) * seed_weight,
                reason=f"personalized from recent {event.get('event_type', 'interaction')}",
                stage="personalized_v2_seed_blend",
            )

    for search in (profile.get("top_searches") or [])[:3]:
        query_text = str(search.get("query_text") or "").strip()
        if not query_text:
            continue
        count_weight = min(1.0, float(search.get("count") or 1) / 3.0)
        for candidate in rec.ai_search(query_text, n=min(12, max(result_limit * 2, 8))):
            add_candidate(
                candidate,
                score=float(candidate.get("similarity_score") or 0.0) * 0.42 * count_weight,
                reason=f"matches your search intent: {query_text}",
                stage="personalized_v2_search_blend",
            )

    if not scored:
        behavior = rec.refresh_behavior_features()
        for item in (behavior.get("trending_movies") or {}).values():
            movie_id = item.get("movie_id")
            if movie_id is None:
                continue
            movie = rec.get_movie_by_id(int(movie_id)) if isinstance(movie_id, int) else None
            if movie:
                add_candidate(
                    movie,
                    score=min(1.0, float(item.get("event_count") or 0) / 20.0),
                    reason=f"trending with viewers ({item.get('event_count')} recent events)",
                    stage="personalized_v2_trending_fallback",
                )

    results = sorted(scored.values(), key=lambda item: float(item.get("similarity_score") or 0), reverse=True)
    return results[:result_limit]


# ---------------------------------------------------------------------------
# Second-pass extractions: load sub-loaders, behavior, index builders, etc.
# ---------------------------------------------------------------------------


def load_vector_artifacts(rec) -> None:
    """Load FAISS index, SBERT embeddings, movie ID map, and pipeline manifest."""
    import json as _json
    import pathlib as _pathlib
    import sys as _sys

    import numpy as _np
    import torch as _torch

    from backend.models.diffusion_recommender import LatentDiffusionRecommender
    from backend.models.model_loader import ensure_model_files
    from backend.serving.feature_store import feature_store

    recommender_module = _sys.modules.get(rec.__class__.__module__)
    MODELS_DIR = (
        (recommender_module and getattr(recommender_module, "MODELS_DIR", None))
        or getattr(rec, "MODELS_DIR", None)
        or _pathlib.Path(__file__).parent.parent / "models"
    )

    try:
        feature_store.load()
    except Exception as e:
        logger.warning("Could not load Feature Store: %s", e)

    rec.diffusion_model = None
    try:
        diffusion_path = MODELS_DIR / "diffusion_recommender.pth"
        if diffusion_path.exists() and not rec._low_memory:
            model = LatentDiffusionRecommender(emb_dim=384, num_timesteps=100)
            model.load_state_dict(_torch.load(diffusion_path, map_location="cpu", weights_only=True))
            model.eval()
            rec.diffusion_model = model
            logger.info("Loaded Generative Diffusion Recommender.")
    except Exception as e:
        logger.warning("Could not load Diffusion Recommender: %s", e)

    import os as _os

    selected_artifacts = {
        "movies_transformed.parquet",
        "semantic_twins.parquet",
        "semantic_twin_summary.json",
        "pipeline_manifest.json",
        "nova_ranker.joblib",
        "nova_ranker.joblib.metadata.json",
    }
    if not rec._low_memory or _os.getenv("NOVA_FORCE_VECTOR_ARTIFACTS", "").lower() in {"1", "true", "yes", "on"}:
        selected_artifacts.update({"sbert_embeddings.npy", "turbovec.tq", "movie_ids.npy"})
    ensure_model_files(MODELS_DIR, selected_files=selected_artifacts)

    def _env_truthy(name):
        return _os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    turbovec_path = MODELS_DIR / "turbovec.tq"
    faiss_path = MODELS_DIR / "faiss.index"

    if rec._low_memory and not _env_truthy("NOVA_FORCE_VECTOR_ARTIFACTS"):
        logger.info("Skipping TurboVec index load in low-memory serving profile.")
    elif turbovec_path.exists():
        from turbovec import TurboQuantIndex

        rec._index = TurboQuantIndex.load(str(turbovec_path))
        logger.info("Loaded TurboVec index with %s vectors", f"{len(rec._index):,}")
    elif faiss_path.exists():
        logger.warning(
            "turbovec.tq not found at %s; faiss.index found but will not be loaded. "
            "Run scripts/migrate_faiss_to_turbovec.py to migrate. "
            "Falling back to metadata-only serving.",
            turbovec_path,
        )
        rec._artifact_status["vector_artifacts_ready"] = False
    else:
        raise FileNotFoundError(f"turbovec.tq not found at {turbovec_path}. Run the ETL pipeline first.")

    vectors_path = MODELS_DIR / "sbert_embeddings.npy"
    if rec._low_memory and not _env_truthy("NOVA_FORCE_VECTOR_ARTIFACTS"):
        logger.info("Skipping embedding matrix load in low-memory serving profile.")
    elif vectors_path.exists():
        rec._vectors = _np.load(vectors_path, mmap_mode="r")
        logger.info("Loaded SBERT embeddings with shape %s (memory-mapped)", rec._vectors.shape)
    else:
        vectors_path = MODELS_DIR / "tfidf_vectors.npy"
        if vectors_path.exists():
            rec._vectors = _np.load(vectors_path, mmap_mode="r")
            logger.warning("SBERT embeddings not found, using TF-IDF vectors.")
        else:
            logger.warning("No vectors found.")

    movie_ids_path = MODELS_DIR / "movie_ids.npy"
    if movie_ids_path.exists():
        rec._artifact_movie_ids = _np.load(movie_ids_path, mmap_mode="r")
        logger.info("Loaded vector movie id map with %s ids", len(rec._artifact_movie_ids))

    manifest_path = MODELS_DIR / "pipeline_manifest.json"
    if manifest_path.exists():
        try:
            rec._artifact_manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
            rec._artifact_status.update(
                {
                    "manifest_run_id": rec._artifact_manifest.get("run_id"),
                    "manifest_run_date": rec._artifact_manifest.get("run_date"),
                }
            )
        except Exception as exc:
            logger.warning("Could not read pipeline manifest %s: %s", manifest_path, exc)


def load_movie_catalog(rec) -> None:
    """Load movie metadata parquet and build lookup maps."""
    import pathlib as _pathlib
    import sys as _sys

    import pandas as _pd

    recommender_module = _sys.modules.get(rec.__class__.__module__)
    DATA_DIR = (
        (recommender_module and getattr(recommender_module, "DATA_DIR", None))
        or getattr(rec, "DATA_DIR", None)
        or _pathlib.Path(__file__).parent.parent / "data" / "processed"
    )

    movies_path = DATA_DIR / "movies_transformed.parquet"
    if not movies_path.exists():
        movies_path = DATA_DIR / "movies.parquet"
    if not movies_path.exists():
        raise FileNotFoundError("Movie data not found. Run the ETL pipeline first.")

    essential_cols = [
        "id",
        "title",
        "overview",
        "genres",
        "vote_average",
        "vote_count",
        "popularity",
        "release_date",
        "poster_path",
        "director",
        "original_language",
        "tagline",
        "runtime",
        "metadata_completeness",
        "content_quality_score",
        "quality_bucket",
        "searchable",
        "recommendable",
        "public_demo_eligible",
    ]
    if not rec._low_memory:
        essential_cols.append("cast")
    import polars as _pl
    try:
        rec._movies = _pl.read_parquet(movies_path, columns=essential_cols).to_pandas()
    except (KeyError, ValueError, Exception):
        try:
            rec._movies = _pl.read_parquet(movies_path).to_pandas()
        except Exception:
            rec._movies = _pd.read_parquet(movies_path)
    rec._optimize_movie_frame()
    rec._rebuild_lookup_maps()
    rec._validate_vector_artifacts()
    logger.info("Loaded %s movies", f"{len(rec._movies):,}")


def load_ranker_and_behavior(rec) -> None:
    """Load learned ranker, build sparse index, and warm behavior features."""
    import os as _os
    import pathlib as _pathlib
    import sys as _sys

    from backend.pipeline.ranker import load_ranker

    recommender_module = _sys.modules.get(rec.__class__.__module__)
    MODELS_DIR = (
        (recommender_module and getattr(recommender_module, "MODELS_DIR", None))
        or getattr(rec, "MODELS_DIR", None)
        or _pathlib.Path(__file__).parent.parent / "models"
    )

    def _env_truthy(name):
        return _os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    if rec._low_memory and not _env_truthy("NOVA_BUILD_SPARSE_ON_LOAD"):
        logger.info("Deferring sparse retrieval index build until first AI search.")
    else:
        rec._build_sparse_retrieval_index()
    rec._learned_ranker = load_ranker(models_dir=MODELS_DIR)
    rec.refresh_behavior_features(force=True)
    try:
        from backend.intelligence.contextual_bandit import get_bandit_engine

        get_bandit_engine().inject_priors(rec._movies)
    except Exception as e:
        logger.warning("Failed to initialize bandit engine: %s", e)


def load_optional_models(rec) -> None:
    """Load multi-modal index, KG, Two-Tower fine-tune, and RL policy."""
    import pathlib as _pathlib
    import sys as _sys

    import torch as _torch

    from backend.intelligence.multimodal_fusion import MultiModalFusionIndex

    recommender_module = _sys.modules.get(rec.__class__.__module__)
    MODELS_DIR = (
        (recommender_module and getattr(recommender_module, "MODELS_DIR", None))
        or getattr(rec, "MODELS_DIR", None)
        or _pathlib.Path(__file__).parent.parent / "models"
    )

    try:
        rec.multimodal_index = MultiModalFusionIndex()
        rec.multimodal_index.load()
    except Exception as e:
        rec.multimodal_index = None
        logger.warning("Failed to load Multi-Modal index: %s", e)

    try:
        loaded = rec.kg_engine.load()
        if not loaded or not hasattr(rec.kg_engine, "graph") or rec.kg_engine.graph is None or len(rec.kg_engine.graph) < 100:
            logger.info("Knowledge Graph is empty or mock. Rebuilding dynamically from catalog...")
            DATA_DIR = (
                (recommender_module and getattr(recommender_module, "DATA_DIR", None))
                or getattr(rec, "DATA_DIR", None)
                or _pathlib.Path(__file__).parent.parent / "data" / "processed"
            )
            twins_path = DATA_DIR / "semantic_twins.parquet"
            rec.kg_engine.rebuild_from_catalog(rec._movies, twins_path)

        try:
            from backend.intelligence.cross_domain_kg import enrich_knowledge_graph_with_cross_domain

            enrich_knowledge_graph_with_cross_domain(rec.kg_engine)
        except Exception as exc:
            logger.warning("Cross-domain KG enrichment skipped: %s", exc)
    except Exception as e:
        logger.warning("Failed to load/rebuild Knowledge Graph: %s", e)

    try:
        from backend.models.two_tower import TwoTowerModel

        two_tower_finetuned_path = MODELS_DIR / "two_tower_finetuned.pth"
        if two_tower_finetuned_path.exists():
            if not hasattr(rec, "_two_tower_model") or rec._two_tower_model is None:
                rec._two_tower_model = TwoTowerModel(
                    user_input_dim=18, item_input_dim=20, embedding_dim=128, temperature=0.07
                )
            state_dict = _torch.load(two_tower_finetuned_path, map_location="cpu", weights_only=True)
            rec._two_tower_model.load_state_dict(state_dict)
            logger.info("Loaded fine-tuned Two-Tower weights from %s", two_tower_finetuned_path)
    except Exception as exc:
        logger.warning("Could not load fine-tuned Two-Tower weights: %s — using base weights", exc)

    try:
        from backend.learning.rl_policy import ActorCriticPolicy

        rl_policy_path = MODELS_DIR / "rl_policy.pth"
        if not rl_policy_path.exists():
            logger.debug("rl_policy.pth not found; RL score adjustment disabled.")
            rec._rl_policy = None
        else:
            policy = ActorCriticPolicy(state_dim=20, action_dim=16)
            state_dict = _torch.load(rl_policy_path, map_location="cpu", weights_only=True)
            policy.load_state_dict(state_dict)
            policy.eval()
            rec._rl_policy = policy
            logger.info("Loaded ActorCriticPolicy from %s", rl_policy_path)
    except RuntimeError as exc:
        logger.warning("Could not load RL policy (state_dim mismatch or corrupt file): %s", exc)
        rec._rl_policy = None
    except Exception as exc:
        logger.warning("Could not load RL policy: %s", exc)
        rec._rl_policy = None


def wire_pipelines(rec, is_tier3: bool) -> None:
    """Wire RetrievalPipeline, RankingPipeline, and RerankingPipeline."""
    import os as _os

    try:
        from backend.pipeline.ranking_pipeline import RankingConfig, RankingPipeline
        from backend.pipeline.reranking_pipeline import RerankingConfig, RerankingPipeline
        from backend.pipeline.retrieval_pipeline import RetrievalConfig, RetrievalPipeline

        def _serving_profile():
            profile = _os.getenv("NOVA_SERVING_PROFILE", "auto").strip().lower()
            return profile if profile in {"full", "lite", "light", "low-memory", "metadata"} else "auto"

        tfidf_idx = (rec._vectorizer, rec._tfidf_matrix) if rec._vectorizer is not None else None
        kg = rec.kg_engine if hasattr(rec, "kg_engine") and rec.kg_engine is not None else None
        rec._retrieval_pipeline = RetrievalPipeline(
            faiss_index=rec._index,
            tfidf_index=tfidf_idx,
            kg_engine=kg,
            movie_df=rec._movies,
            config=RetrievalConfig(low_memory=rec._low_memory, enable_kg=not is_tier3),
        )
        # Wire the pre-trained neural ensemble engine (ApexEnsembleEngine)
        try:
            from backend.models.ensemble_engine import get_apex_engine
            ensemble = get_apex_engine(num_users=610, num_items=9724)
        except Exception as exc:
            logger.warning("Could not load get_apex_engine: %s", exc)
            ensemble = None

        rec._ranking_pipeline = RankingPipeline(
            ensemble_engine=ensemble,
            learned_ranker=rec._learned_ranker,
            config=RankingConfig(
                use_neural_ensemble=(not is_tier3 or _serving_profile() == "full"),
                use_learned_ranker=False, # Disable learned ranker until retraining fix
            ),
        )
        rec._reranking_pipeline = RerankingPipeline(
            rl_policy=rec._rl_policy,
            llm_client=None,
            config=RerankingConfig(),
            movie_df=rec._movies,
        )
        logger.info("Pipeline modules wired: RetrievalPipeline, RankingPipeline, RerankingPipeline")
    except Exception as exc:
        logger.warning("Failed to wire pipeline modules: %s", exc)
        rec._retrieval_pipeline = None
        rec._ranking_pipeline = None
        rec._reranking_pipeline = None


def refresh_behavior_features(rec, force: bool = False) -> dict:
    """Refresh aggregated behavior features (thread-safe, TTL-cached)."""
    from datetime import UTC
    from datetime import datetime as _datetime
    import os as _os

    ttl_seconds = int(_os.getenv("BEHAVIOR_FEATURE_TTL_SECONDS", "60"))
    now = _datetime.now(UTC)
    if (
        not force
        and rec._behavior_features_refreshed_at is not None
        and (now - rec._behavior_features_refreshed_at).total_seconds() < ttl_seconds
    ):
        return rec._behavior_features
    with rec._behavior_features_lock:
        now = _datetime.now(UTC)
        if (
            not force
            and rec._behavior_features_refreshed_at is not None
            and (now - rec._behavior_features_refreshed_at).total_seconds() < ttl_seconds
        ):
            return rec._behavior_features
        try:
            from backend.events import aggregate_behavior_features

            rec._behavior_features = aggregate_behavior_features(limit=100)
            rec._behavior_features_refreshed_at = now
        except Exception as exc:
            logger.warning("Behavior feature refresh skipped: %s", exc)
            rec._behavior_features = {}
            rec._behavior_features_refreshed_at = now
    return rec._behavior_features


def optimize_movie_frame(rec) -> None:
    """Reduce the in-memory footprint of the serving catalog."""
    import pandas as _pd

    if rec._movies is None:
        return
    for column in ("id", "vote_count"):
        if column in rec._movies.columns:
            rec._movies[column] = _pd.to_numeric(rec._movies[column], errors="coerce", downcast="integer")
    for column in ("vote_average", "popularity", "metadata_completeness", "content_quality_score"):
        if column in rec._movies.columns:
            rec._movies[column] = _pd.to_numeric(rec._movies[column], errors="coerce", downcast="float")
    for column in ("searchable", "recommendable", "public_demo_eligible"):
        if column in rec._movies.columns:
            rec._movies[column] = rec._movies[column].fillna(False).astype(bool)
    for column in ("quality_bucket", "original_language"):
        if column in rec._movies.columns:
            rec._movies[column] = rec._movies[column].fillna("").astype("category")


def build_sparse_retrieval_index(rec) -> None:
    """Build a TF-IDF recall index for hybrid search and cold-start resilience."""
    import os as _os

    import numpy as _np
    import pandas as _pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    text_parts = []
    for column in ("title", "overview", "genres", "director", "cast", "original_language"):
        if column in rec._movies.columns:
            text_parts.append(rec._movies[column].astype(object).fillna("").astype(str))
    if not text_parts:
        rec._content_text = _pd.Series([""] * len(rec._movies), index=rec._movies.index)
    else:
        content_text = text_parts[0]
        for part in text_parts[1:]:
            content_text = content_text + ". " + part
        rec._content_text = content_text
    default_features = "12000" if rec._low_memory else "50000"
    max_features = int(_os.getenv("NOVA_TFIDF_MAX_FEATURES", default_features))
    ngram_range = (1, 1) if rec._low_memory else (1, 2)
    rec._vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=1,
        dtype=_np.float32,
    )
    rec._tfidf_matrix = rec._vectorizer.fit_transform(rec._content_text)
    logger.info("Built sparse TF-IDF retrieval index with %s features", len(rec._vectorizer.vocabulary_))
    rec._content_text = None


def build_item_retrieval_index(rec) -> None:
    """Build a plot/genre-focused sparse index for item-to-item recommendations."""
    import os as _os

    import numpy as _np
    from sklearn.feature_extraction.text import TfidfVectorizer

    if rec._movies is None:
        return

    def text_column(column):
        if column not in rec._movies.columns:
            import pandas as _pd

            return _pd.Series([""] * len(rec._movies), index=rec._movies.index)
        return rec._movies[column].astype(object).fillna("").astype(str)

    item_text = (
        text_column("overview")
        + ". "
        + text_column("tagline")
        + ". Genres "
        + text_column("genres").str.replace(",", " ", regex=False)
        + ". Language "
        + text_column("original_language")
    )
    default_features = "18000" if rec._low_memory else "40000"
    max_features = int(_os.getenv("NOVA_ITEM_TFIDF_MAX_FEATURES", default_features))
    rec._item_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        dtype=_np.float32,
    )
    rec._item_tfidf_matrix = rec._item_vectorizer.fit_transform(item_text)
    logger.info("Built item-to-item sparse retrieval index with %s features", len(rec._item_vectorizer.vocabulary_))


def behavior_boost(rec, movie_id) -> tuple:
    """Return a bounded score nudge from recent product behavior."""
    if movie_id is None:
        return 0.0, []
    try:
        movie_key = str(int(movie_id))
    except (TypeError, ValueError):
        return 0.0, []
    trending_movies = rec._behavior_features.get("trending_movies", {})
    if not isinstance(trending_movies, dict):
        return 0.0, []
    stats = trending_movies.get(movie_key)
    if not isinstance(stats, dict):
        return 0.0, []
    event_count = int(stats.get("event_count") or 0)
    views = int(stats.get("views") or 0)
    clicks = int(stats.get("clicks") or 0)
    ratings = int(stats.get("ratings") or 0)
    avg_rating = stats.get("avg_rating")
    boost = min(0.08, event_count * 0.005 + views * 0.003 + clicks * 0.01)
    if avg_rating is not None and ratings > 0 and float(avg_rating) >= 4.0:
        boost += min(0.02, ratings * 0.005)
    boost = min(0.10, boost)
    reasons = []
    if event_count:
        reasons.append(f"Trending with viewers ({event_count} recent events)")
    if avg_rating is not None and ratings > 0:
        reasons.append(f"Audience signal ({float(avg_rating):.1f}/5)")
    return boost, reasons[:2]


def quality_gate_item_recommendations(rec, candidates: list, query_movie: dict, n: int) -> list:
    """Drop obvious low-quality or genre-drift candidates when enough alternatives exist."""
    if len(candidates) <= n:
        return candidates
    query_genres = rec._genre_set(query_movie)
    gated = []
    for candidate in candidates:
        rating = float(candidate.get("vote_average") or 0.0)
        votes = float(candidate.get("vote_count") or 0.0)
        candidate_genres = rec._genre_set(candidate)
        shared_genres = query_genres & candidate_genres
        signals = candidate.get("retrieval_signals") or {}
        semantic_score = float(signals.get("semantic_twin") or 0.0)
        if votes >= 500 and 0 < rating < 5.5:
            continue
        if "science fiction" in query_genres and "science fiction" not in candidate_genres:
            if len(shared_genres) < 2 and semantic_score < 0.62:
                continue
        gated.append(candidate)
    return gated if len(gated) >= n else candidates


def genre_affinity_from_profile(rec, profile: dict) -> dict:
    """Build genre affinity weights from positive user events."""
    affinity: dict = {}
    for event in profile.get("recent_events") or []:
        if event.get("negative"):
            continue
        movie = rec.get_movie_by_id(int(event.get("movie_id"))) if event.get("movie_id") is not None else None
        if not movie:
            continue
        weight = float(event.get("weight") or 1.0) * rec._event_recency_decay(event.get("event_ts"))
        for genre in rec._genre_set(movie):
            affinity[genre] = affinity.get(genre, 0.0) + weight
    if not affinity:
        return {}
    max_weight = max(affinity.values())
    if max_weight <= 0:
        return {}
    return {genre: round(weight / max_weight, 4) for genre, weight in affinity.items()}


def visual_search(rec, movie_id: int, n: int = 10) -> list:
    """Multi-Modal similarity search using Text + Visual (Poster) embeddings."""
    import pathlib as _pathlib
    import sys as _sys

    import numpy as _np

    recommender_module = _sys.modules.get(rec.__class__.__module__)
    MODELS_DIR = (
        (recommender_module and getattr(recommender_module, "MODELS_DIR", None))
        or getattr(rec, "MODELS_DIR", None)
        or _pathlib.Path(__file__).parent.parent / "models"
    )

    if rec.multimodal_index is None or rec.multimodal_index.index is None:
        logger.warning("Visual search requested, but multimodal index is not loaded.")
        return []
    text_emb_path = MODELS_DIR / "sbert_embeddings.npy"
    text_ids_path = MODELS_DIR / "movie_ids.npy"
    vision_emb_path = MODELS_DIR / "poster_embeddings.npy"
    vision_ids_path = MODELS_DIR / "poster_movie_ids.npy"
    if not all(p.exists() for p in (text_emb_path, text_ids_path, vision_emb_path, vision_ids_path)):
        return []
    text_vectors = _np.load(text_emb_path, mmap_mode="r")
    text_ids = _np.load(text_ids_path, mmap_mode="r")
    vision_vectors = _np.load(vision_emb_path, mmap_mode="r")
    vision_ids = _np.load(vision_ids_path, mmap_mode="r")
    t_loc = _np.where(text_ids == movie_id)[0]
    v_loc = _np.where(vision_ids == movie_id)[0]
    if len(t_loc) == 0 or len(v_loc) == 0:
        logger.warning("Movie %s is missing text or vision vectors.", movie_id)
        return []
    results = rec.multimodal_index.search(text_vectors[t_loc[0]], vision_vectors[v_loc[0]], top_k=n * 2)
    final_results = []
    for res_id, dist in results:
        if res_id == movie_id or len(final_results) >= n:
            continue
        movie = rec.get_movie_by_id(int(res_id))
        if movie:
            movie["similarity_score"] = float(dist)
            movie["retrieval_stage"] = "multi_modal_fusion"
            movie["explanation"] = ["Aesthetically and thematically similar (Text + Vision matching)"]
            movie["explanation_text"] = movie["explanation"][0]
            final_results.append(movie)
    return final_results


def candidate_to_dict(rec, item) -> dict:
    """Convert a FinalItem from the pipeline to the response dict shape."""
    movie = rec.get_movie_by_id(item.movie_id)
    if movie is None:
        movie = dict(item.metadata)
        movie.setdefault("id", item.movie_id)

    retrieval_score = getattr(item, "retrieval_score", 0.0)
    ensemble_score = getattr(item, "ensemble_score", 0.0)
    ranker_score = getattr(item, "ranker_score", None)
    diversity_score = getattr(item, "diversity_score", 0.0)

    movie["similarity_score"] = float(ranker_score if ranker_score is not None else retrieval_score)
    movie["retrieval_stage"] = getattr(item, "retrieval_source", "unknown")

    retrieval_signals = getattr(item, "retrieval_signals", {}) or {}
    movie["retrieval_signals"] = {
        "dense": round(float(retrieval_score), 4),
        "ensemble": round(float(ensemble_score), 4),
        "ranker": round(float(ranker_score) if ranker_score is not None else 0.0, 4),
        "diversity": round(float(diversity_score), 4),
        **retrieval_signals,
    }

    explanation = getattr(item, "explanation", None)
    if explanation:
        movie["explanation"] = [explanation]
        movie["explanation_text"] = explanation
    else:
        movie.setdefault("explanation", ["Similar themes and plot"])
        movie.setdefault("explanation_text", "Similar themes and plot")
    return movie


def popularity_quality_score(movie: dict) -> float:
    """Small bounded business score from popularity and quality."""
    import math as _math

    import numpy as _np

    if movie.get("content_quality_score") is not None:
        try:
            score = float(movie.get("content_quality_score"))
            if not _math.isnan(score):
                return max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            pass
    completeness = float(movie.get("metadata_completeness") or 0.0)
    popularity = float(movie.get("popularity") or 0)
    rating = float(movie.get("vote_average") or 0)
    votes = float(movie.get("vote_count") or 0)
    popularity_score = min(1.0, _np.log1p(max(popularity, 0)) / 8.0)
    confidence = min(1.0, _np.log1p(max(votes, 0)) / 8.0)
    quality_score = (rating / 10.0) * confidence if rating > 0 else 0.0
    return float(0.35 * popularity_score + 0.35 * quality_score + 0.30 * completeness)


def semantic_affinity_for_indices(rec, query_idx: int, candidate_idx: int) -> dict:
    """Compare query/candidate semantic twins and return serializable signals."""
    import contextlib as _contextlib

    from backend.intelligence.semantic_twin import compare_semantic_twins

    if not hasattr(rec, "_affinity_cache"):
        rec._affinity_cache = {}
    pair_key = (query_idx, candidate_idx)
    cached = rec._affinity_cache.get(pair_key)
    if cached is not None:
        return cached
    affinity = compare_semantic_twins(
        rec._semantic_twin_for_index(query_idx),
        rec._semantic_twin_for_index(candidate_idx),
    )
    result = affinity.as_dict()
    if len(rec._affinity_cache) >= 20000:
        with _contextlib.suppress(StopIteration):
            rec._affinity_cache.pop(next(iter(rec._affinity_cache)))
    rec._affinity_cache[pair_key] = result
    return result


# ---------------------------------------------------------------------------
# Module-level helpers extracted from recommender.py
# ---------------------------------------------------------------------------


def render_like_environment() -> bool:
    """Detect constrained PaaS runtimes where the full vector stack can exceed memory."""
    import os as _os

    return any(
        _os.getenv(name)
        for name in (
            "RENDER",
            "RENDER_SERVICE_ID",
            "RENDER_SERVICE_NAME",
            "RENDER_EXTERNAL_URL",
            "RENDER_EXTERNAL_HOSTNAME",
        )
    )


def serving_profile() -> str:
    """Resolve the serving profile for this process."""
    import os as _os

    profile = _os.getenv("NOVA_SERVING_PROFILE", "auto").strip().lower()
    return profile if profile in {"full", "lite", "light", "low-memory", "metadata"} else "auto"


def low_memory_serving_enabled() -> bool:
    """Return true when serving should avoid loading heavyweight vector artifacts."""
    import os as _os

    if _os.getenv("NOVA_LOW_MEMORY", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if _os.getenv("NOVA_LOW_MEMORY", "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    profile = serving_profile()
    if profile in {"lite", "light", "low-memory", "metadata"}:
        return True
    if profile == "full":
        return False
    return render_like_environment()


def safe_float(val, default: float = 0.0) -> float:
    """Convert val to float safely, returning default on error or non-finite."""
    import math as _math

    try:
        v = float(val)
        return v if _math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def build_rl_state(behavior_profile: dict, als_user_embedding, state_dim: int = 20):
    """Build a fixed-length RL state vector from user behavior profile."""
    import math as _math

    import numpy as _np
    import torch as _torch

    def _sf(val, default=0.0):
        try:
            v = float(val)
            return v if _math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    total_ratings = _sf(behavior_profile.get("total_ratings", 0))
    avg_rating = _sf(behavior_profile.get("avg_rating", 0))
    click_count = _sf(behavior_profile.get("click_count", 0))
    view_count = _sf(behavior_profile.get("view_count", 0))

    scalars = [
        _math.log1p(max(total_ratings, 0)) / _math.log1p(1000),
        avg_rating / 5.0,
        _math.log1p(max(click_count, 0)) / _math.log1p(500),
        _math.log1p(max(view_count, 0)) / _math.log1p(500),
    ]
    if als_user_embedding is not None:
        emb = _np.asarray(als_user_embedding, dtype=_np.float32).flatten()[:16]
        emb_list = emb.tolist() + [0.0] * (16 - len(emb.tolist()))
    else:
        emb_list = [0.0] * 16

    tensor = _torch.tensor(scalars + emb_list, dtype=_torch.float32).unsqueeze(0)
    return _torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def learned_ranker_enabled(rec) -> bool:
    """Return whether the learned ranker has enough signal to influence serving."""
    import os as _os

    value = _os.getenv("NOVA_ENABLE_LEARNED_RANKER", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    metadata = getattr(rec._learned_ranker, "metadata", {}) or {}
    training_mode = str(metadata.get("training_mode") or "").lower()
    try:
        feedback_count = int(metadata.get("feedback_item_count") or 0)
    except (TypeError, ValueError):
        feedback_count = 0
    min_feedback = int(_os.getenv("NOVA_MIN_RANKER_FEEDBACK_ITEMS", "100"))
    if training_mode == "catalog_bootstrap" and feedback_count < min_feedback:
        return False
    return feedback_count >= min_feedback
