"""
Training and offline evaluation for Nova's learned ranker.

The first model is intentionally small and cheap: it learns from implicit
feedback when events exist, otherwise it can bootstrap from catalog quality
signals. The artifact is a joblib file that the backend loads opportunistically.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import contextlib
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score

from backend.pipeline.ranker import FEATURE_COLUMNS, candidate_features, load_ranker, save_ranker

EVENT_WEIGHTS = {
    "recommendation_impression": 0.15,
    "view": 0.45,
    "click": 1.0,
    "rating": 1.2,
}


def _event_label(event: dict[str, Any]) -> float:
    event_type = str(event.get("event_type") or "").lower()
    label = EVENT_WEIGHTS.get(event_type, 0.0)
    if event_type == "rating" and event.get("rating") is not None:
        with contextlib.suppress(TypeError, ValueError):
            label += max(0.0, min(float(event["rating"]), 5.0)) / 5.0
    return label


def build_item_feedback(events: Iterable[dict[str, Any]]) -> dict[int, float]:
    """Aggregate implicit feedback into movie-level relevance labels."""
    scores: dict[int, float] = defaultdict(float)
    for event in events:
        movie_id = event.get("movie_id") or event.get("source_content_id")
        if movie_id is None:
            continue
        try:
            movie_id = int(movie_id)
        except (TypeError, ValueError):
            continue
        scores[movie_id] += _event_label(event)
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return dict.fromkeys(scores, 0.0)
    return {movie_id: round(score / max_score, 6) for movie_id, score in scores.items()}


def _catalog_quality_label(row: pd.Series) -> float:
    if row.get("content_quality_score") is not None:
        try:
            score = float(row.get("content_quality_score"))
            if not np.isnan(score):
                return max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            pass
    vote_average = float(row.get("vote_average") or 0)
    vote_count = float(row.get("vote_count") or 0)
    popularity = float(row.get("popularity") or 0)
    quality = (vote_average / 10.0) * min(1.0, np.log1p(max(vote_count, 0)) / 8.0)
    popularity_score = min(1.0, np.log1p(max(popularity, 0)) / 8.0)
    return float(0.55 * popularity_score + 0.45 * quality)


def _row_to_candidate(row: pd.Series, label_hint: float = 0.0) -> dict[str, Any]:
    metadata_score = _catalog_quality_label(row)
    return {
        "id": int(row.get("id")),
        "similarity_score": 0.35 * metadata_score + 0.65 * label_hint,
        "vote_average": row.get("vote_average"),
        "vote_count": row.get("vote_count"),
        "popularity": row.get("popularity"),
        "release_date": row.get("release_date"),
        "retrieval_signals": {
            "dense": 0.0,
            "sparse": 0.0,
            "metadata": metadata_score,
            "behavior": label_hint,
            "cross_encoder": 0.0,
        },
    }


def build_training_frame(
    movies: pd.DataFrame,
    events: Iterable[dict[str, Any]],
    min_behavior_items: int = 3,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Create a supervised ranking frame from catalog metadata and events."""
    feedback = build_item_feedback(events)
    has_behavior = len(feedback) >= min_behavior_items

    features = []
    labels = []
    for _, row in movies.iterrows():
        movie_id = int(row.get("id"))
        behavior_label = float(feedback.get(movie_id, 0.0))
        bootstrap_label = _catalog_quality_label(row)
        if has_behavior:
            label = 0.78 * behavior_label + 0.22 * bootstrap_label
        else:
            label = bootstrap_label
        features.append(candidate_features(_row_to_candidate(row, behavior_label)))
        labels.append(label)

    metadata = {
        "training_mode": "implicit_feedback" if has_behavior else "catalog_bootstrap",
        "movie_count": int(len(movies)),
        "feedback_item_count": int(len(feedback)),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    return pd.DataFrame(features, columns=FEATURE_COLUMNS), pd.Series(labels), metadata


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10, positive_threshold: float = 0.2) -> float:
    positives = set(np.where(y_true >= positive_threshold)[0])
    if not positives:
        return 0.0
    top_k = set(np.argsort(y_score)[::-1][:k])
    return round(len(positives & top_k) / len(positives), 6)


def evaluate_scores(labels: pd.Series | np.ndarray, scores: np.ndarray, k: int = 10) -> dict[str, Any]:
    """Evaluate already-computed ranking scores with simple offline top-k metrics."""
    if isinstance(labels, pd.Series):
        labels_array = labels.to_numpy(dtype=np.float32)
    else:
        labels_array = np.asarray(labels, dtype=np.float32)
    predictions = np.asarray(scores, dtype=np.float32)
    top_k = min(k, len(labels_array))
    if top_k <= 0:
        return {"recall_at_k": 0.0, "ndcg_at_k": 0.0, "top_k": 0}

    ndcg = ndcg_score([labels_array], [predictions], k=top_k) if np.any(labels_array > 0) else 0.0
    return {
        "recall_at_k": recall_at_k(labels_array, predictions, k=top_k),
        "ndcg_at_k": round(float(ndcg), 6),
        "top_k": int(top_k),
        "prediction_min": round(float(predictions.min()), 6),
        "prediction_max": round(float(predictions.max()), 6),
    }


def evaluate_ranker(model: Any, features: pd.DataFrame, labels: pd.Series, k: int = 10) -> dict[str, Any]:
    """Evaluate ranking quality with simple offline top-k metrics."""
    predictions = np.asarray(model.predict(features), dtype=np.float32)
    return evaluate_scores(labels, predictions, k=k)


def baseline_scores(features: pd.DataFrame) -> np.ndarray:
    """Score candidates with Nova's hand-built serving signals."""
    return np.asarray(
        0.55 * features["base_similarity"] + 0.30 * features["metadata_score"] + 0.15 * features["behavior_score"],
        dtype=np.float32,
    )


def promotion_decision(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    current: dict[str, Any] | None = None,
    min_ndcg_lift: float = 0.0,
    max_ndcg_regression: float = 0.002,
    max_recall_regression: float = 0.02,
) -> dict[str, Any]:
    """Decide whether a candidate ranker is safe to promote."""
    candidate_ndcg = float(candidate.get("ndcg_at_k") or 0.0)
    candidate_recall = float(candidate.get("recall_at_k") or 0.0)
    baseline_ndcg = float(baseline.get("ndcg_at_k") or 0.0)
    baseline_recall = float(baseline.get("recall_at_k") or 0.0)
    reference = current or baseline
    reference_name = "current_production" if current else "baseline"
    reference_ndcg = float(reference.get("ndcg_at_k") or 0.0)
    reference_recall = float(reference.get("recall_at_k") or 0.0)

    checks = {
        "beats_baseline_ndcg": candidate_ndcg >= baseline_ndcg + min_ndcg_lift - max_ndcg_regression,
        "no_reference_ndcg_regression": candidate_ndcg >= reference_ndcg - max_ndcg_regression,
        "no_reference_recall_regression": candidate_recall >= reference_recall - max_recall_regression,
    }
    promote = all(checks.values())
    reasons = []
    if promote:
        reasons.append(f"candidate is safe against {reference_name}")
    else:
        if not checks["beats_baseline_ndcg"]:
            reasons.append("candidate NDCG is below baseline gate")
        if not checks["no_reference_ndcg_regression"]:
            reasons.append(f"candidate NDCG regressed against {reference_name}")
        if not checks["no_reference_recall_regression"]:
            reasons.append(f"candidate recall regressed against {reference_name}")

    return {
        "decision": "promote" if promote else "reject",
        "reference": reference_name,
        "checks": checks,
        "candidate_ndcg_lift_vs_baseline": round(candidate_ndcg - baseline_ndcg, 6),
        "candidate_recall_lift_vs_baseline": round(candidate_recall - baseline_recall, 6),
        "reasons": reasons,
    }


def train_nova_ranker(
    movies: pd.DataFrame,
    events: Iterable[dict[str, Any]],
    output_path: Path | str,
    random_state: int = 42,
    promotion_gate: bool = False,
    production_path: Path | str | None = None,
) -> dict[str, Any]:
    """Train and persist Nova's learned ranker artifact."""
    features, labels, metadata = build_training_frame(movies, events)
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features, labels)

    baseline_evaluation = evaluate_scores(labels, baseline_scores(features))
    evaluation = evaluate_ranker(model, features, labels)
    metadata["evaluation"] = evaluation
    metadata["baseline_evaluation"] = baseline_evaluation
    metadata["feature_importances"] = {
        column: round(float(importance), 6)
        for column, importance in zip(FEATURE_COLUMNS, getattr(model, "feature_importances_", []), strict=False)
    }
    artifact_path = save_ranker(
        model=model,
        output_path=output_path,
        metadata=metadata,
        feature_columns=FEATURE_COLUMNS,
    )

    report = {
        "artifact_path": str(artifact_path),
        "metadata": metadata,
    }

    if promotion_gate:
        production_path = Path(production_path) if production_path is not None else artifact_path
        current_evaluation = None
        current_ranker = load_ranker(production_path)
        if current_ranker is not None:
            current_evaluation = evaluate_ranker(current_ranker.model, features, labels)
        decision = promotion_decision(
            candidate=evaluation,
            baseline=baseline_evaluation,
            current=current_evaluation,
        )
        metadata["current_evaluation"] = current_evaluation
        metadata["promotion"] = decision
        report["promotion"] = decision
        report["promoted"] = decision["decision"] == "promote"
        if report["promoted"]:
            production_path.parent.mkdir(parents=True, exist_ok=True)
            save_ranker(
                model=model,
                output_path=production_path,
                metadata=metadata,
                feature_columns=FEATURE_COLUMNS,
            )
            report["promoted_artifact_path"] = str(production_path)
        else:
            report["promoted_artifact_path"] = None

    report_path = Path(str(artifact_path) + ".metadata.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if promotion_gate and report.get("promoted_artifact_path"):
        production_report_path = Path(str(report["promoted_artifact_path"]) + ".metadata.json")
        production_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["metadata_path"] = str(report_path)
    return report
