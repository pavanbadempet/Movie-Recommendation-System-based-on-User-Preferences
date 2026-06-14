"""
Learned ranking layer for Nova recommendations.

This module is deliberately free-tier friendly: a small scikit-learn model is
loaded from a joblib artifact when available. If no artifact exists, the
hand-built hybrid ranker remains the serving path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "base_similarity",
    "dense_score",
    "sparse_score",
    "metadata_score",
    "behavior_score",
    "cross_encoder_score",
    "vote_average_norm",
    "vote_confidence",
    "popularity_norm",
    "release_year_norm",
    "is_recent",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _release_year(value: Any) -> int | None:
    try:
        raw = str(value or "")[:4]
        if not raw:
            return None
        year = int(raw)
        if 1800 <= year <= 2100:
            return year
    except (TypeError, ValueError):
        return None
    return None


def candidate_features(candidate: dict[str, Any], current_year: int | None = None) -> list[float]:
    """Extract ranking features from a recommendation/search candidate."""
    current_year = current_year or datetime.now(UTC).year
    signals = candidate.get("retrieval_signals") or {}
    release_year = _release_year(candidate.get("release_date"))
    years_old = current_year - release_year if release_year else None

    vote_average = _safe_float(candidate.get("vote_average"))
    vote_count = _safe_float(candidate.get("vote_count"))
    popularity = _safe_float(candidate.get("popularity"))

    return [
        _safe_float(candidate.get("similarity_score")),
        _safe_float(signals.get("dense")),
        _safe_float(signals.get("sparse")),
        _safe_float(signals.get("metadata")),
        _safe_float(signals.get("behavior")),
        _safe_float(signals.get("cross_encoder")),
        min(1.0, max(0.0, vote_average / 10.0)),
        min(1.0, math.log1p(max(vote_count, 0.0)) / 10.0),
        min(1.0, math.log1p(max(popularity, 0.0)) / 8.0),
        min(1.0, max(0.0, ((release_year or 1900) - 1900) / 140.0)),
        1.0 if years_old is not None and years_old <= 5 else 0.0,
    ]


@dataclass
class NovaRanker:
    """Loaded learned ranker artifact."""

    model: Any
    feature_columns: list[str]
    metadata: dict[str, Any]
    _movie_df: Any = None
    _movie_lookup: dict = None

    @property
    def movie_df(self):
        if not hasattr(self, "_movie_df") or self._movie_df is None:
            try:
                import pandas as pd
                from pathlib import Path
                model_path = Path(self.metadata.get("artifact_path", ""))
                if model_path.exists():
                    df_path = model_path.parent / "movies_transformed.parquet"
                else:
                    df_path = Path(__file__).resolve().parent.parent / "models" / "movies_transformed.parquet"
                
                if df_path.exists():
                    logger.info("NovaRanker lazy loading movie DataFrame from %s", df_path)
                    df = pd.read_parquet(df_path)
                    self.movie_df = df
                else:
                    logger.warning("NovaRanker could not find movie DataFrame at %s", df_path)
                    self._movie_df = None
                    self._movie_lookup = {}
            except Exception as e:
                logger.warning("NovaRanker failed to lazy load movie DataFrame: %s", e)
                self._movie_df = None
                self._movie_lookup = {}
        return self._movie_df

    @movie_df.setter
    def movie_df(self, df):
        self._movie_df = df
        if df is not None:
            required_cols = {'id', 'title', 'vote_average', 'vote_count', 'popularity', 'release_date'}
            cols = [c for c in df.columns if c in required_cols]
            self._movie_lookup = {
                int(row['id']): row for row in df[cols].to_dict(orient='records')
                if 'id' in row and pd.notna(row['id'])
            }
        else:
            self._movie_lookup = {}

    def predict(
        self,
        user_id_or_candidates: Any,
        candidate_ids: list[int] | None = None,
        candidates: list[Any] | None = None,
    ) -> np.ndarray | dict[int, float]:
        if candidate_ids is not None:
            # Pipeline predict(user_id, candidate_ids) mode:
            candidate_map = {}
            if candidates:
                candidate_map = {c.movie_id: c for c in candidates}
            
            # Trigger lazy load if movie_df not initialized
            _ = self.movie_df
            lookup = getattr(self, "_movie_lookup", {}) or {}
            
            candidates_list = []
            for mid in candidate_ids:
                movie_rec = lookup.get(mid, {})
                c_item = candidate_map.get(mid)
                score = c_item.retrieval_score if c_item else 0.0
                source = c_item.retrieval_source if c_item else "candidate"
                signals = c_item.metadata if (c_item and c_item.metadata) else {}
                
                candidate_dict = {
                    "similarity_score": score,
                    "vote_average": movie_rec.get("vote_average"),
                    "vote_count": movie_rec.get("vote_count"),
                    "popularity": movie_rec.get("popularity"),
                    "release_date": movie_rec.get("release_date"),
                    "retrieval_signals": {
                        "dense": signals.get("dense_score", score if source in ("faiss", "turbovec") else 0.0),
                        "sparse": signals.get("sparse_score", score if source == "tfidf" else 0.0),
                        "metadata": signals.get("metadata_score", score if source == "knowledge_graph" else 0.0),
                        "behavior": signals.get("behavior_score", score if source == "behavior" else 0.0),
                        "cross_encoder": signals.get("cross_encoder_score", 0.0),
                    }
                }
                candidates_list.append(candidate_dict)
            
            if not candidates_list:
                return {}
                
            features = pd.DataFrame(
                [candidate_features(c) for c in candidates_list],
                columns=self.feature_columns,
            )
            scores = self.model.predict(features)
            return {mid: float(s) for mid, s in zip(candidate_ids, scores)}
        else:
            # Legacy/candidates mode:
            candidates_data = user_id_or_candidates
            if not candidates_data:
                return np.array([], dtype=np.float32)
            features = pd.DataFrame(
                [candidate_features(candidate) for candidate in candidates_data],
                columns=self.feature_columns,
            )
            scores = self.model.predict(features)
            return np.asarray(scores, dtype=np.float32)

    def rerank(self, candidates: list[dict[str, Any]], blend_weight: float = 0.72) -> list[dict[str, Any]]:
        """Blend learned ranker scores into existing candidate scores and resort."""
        if not candidates:
            return []

        scores = self.predict(candidates)
        if len(scores) == 0:
            return candidates

        min_score = float(scores.min())
        max_score = float(scores.max())
        if max_score > min_score:
            normalized = (scores - min_score) / (max_score - min_score)
        else:
            normalized = np.ones_like(scores, dtype=np.float32)

        reranked = []
        for candidate, ranker_score in zip(candidates, normalized, strict=False):
            item = dict(candidate)
            previous_score = _safe_float(item.get("similarity_score"))
            learned_score = float(ranker_score)
            item["ranker_score"] = round(learned_score, 6)
            item["similarity_score"] = float(blend_weight * learned_score + (1 - blend_weight) * previous_score)
            stage = str(item.get("retrieval_stage") or "candidate")
            if "learned_ranker" not in stage:
                item["retrieval_stage"] = f"{stage}_learned_ranker"
            explanation = list(item.get("explanation") or [])
            if "learned feedback ranker" not in explanation:
                explanation.insert(0, "learned feedback ranker")
            item["explanation"] = explanation[:5]
            item["explanation_text"] = " | ".join(item["explanation"])
            reranked.append(item)

        reranked.sort(key=lambda item: _safe_float(item.get("similarity_score")), reverse=True)
        return reranked


def default_ranker_path(models_dir: Path | None = None) -> Path:
    configured = os.getenv("NOVA_RANKER_PATH")
    if configured:
        return Path(configured)
    models_dir = models_dir or Path(__file__).resolve().parent.parent / "models"
    return models_dir / "nova_ranker.joblib"


def load_ranker(path: Path | str | None = None, models_dir: Path | None = None) -> NovaRanker | None:
    """Load a Nova ranker artifact if one is available."""
    artifact_path = Path(path) if path is not None else default_ranker_path(models_dir)
    if not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return None

    try:
        payload = joblib.load(artifact_path)
    except Exception as exc:
        logger.warning("Could not load Nova ranker artifact %s: %s", artifact_path, exc)
        return None

    if not isinstance(payload, dict) or "model" not in payload:
        logger.warning("Nova ranker artifact %s has an invalid payload", artifact_path)
        return None

    feature_columns = list(payload.get("feature_columns") or FEATURE_COLUMNS)
    metadata = dict(payload.get("metadata") or {})
    metadata["artifact_path"] = str(artifact_path)
    logger.info("Loaded Nova learned ranker from %s", artifact_path)
    return NovaRanker(
        model=payload["model"],
        feature_columns=feature_columns,
        metadata=metadata,
    )


def save_ranker(
    model: Any,
    output_path: Path | str,
    metadata: dict[str, Any],
    feature_columns: list[str] | None = None,
) -> Path:
    """Persist a learned ranker artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns or FEATURE_COLUMNS,
            "metadata": metadata,
        },
        output_path,
    )
    return output_path
