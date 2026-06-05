"""Batch semantic-twin artifacts for the recommendation lakehouse."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.semantic_twin import build_semantic_twin

SEMANTIC_TWIN_ARTIFACT_VERSION = 1


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_semantic_twin_frame(movies: pd.DataFrame) -> pd.DataFrame:
    """Build one deterministic semantic-twin row per serving catalog item."""
    rows = []
    for movie in movies.to_dict(orient="records"):
        twin = build_semantic_twin(movie)
        rows.append(
            {
                "id": int(movie["id"]),
                "title": movie.get("title"),
                "genres": _json_dump(twin.get("genres") or []),
                "concepts": _json_dump(twin.get("concepts") or []),
                "emotional_arcs": _json_dump(twin.get("emotional_arcs") or []),
                "viewer_jobs": _json_dump(twin.get("viewer_jobs") or []),
                "risk_tags": _json_dump(twin.get("risk_tags") or []),
                "confidence": float(twin.get("confidence") or 0.0),
                "semantic_twin_json": _json_dump(twin),
            }
        )
    return pd.DataFrame(rows)


def _iter_json_list(values: pd.Series) -> list[str]:
    output: list[str] = []
    for value in values.dropna():
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            output.extend(str(item) for item in parsed if item)
    return output


def summarize_semantic_twins(twins: pd.DataFrame, run_id: str, run_date: str) -> dict[str, Any]:
    """Summarize semantic-twin coverage and risk signals."""
    concept_counts = Counter(_iter_json_list(twins.get("concepts", pd.Series(dtype=str))))
    arc_counts = Counter(_iter_json_list(twins.get("emotional_arcs", pd.Series(dtype=str))))
    job_counts = Counter(_iter_json_list(twins.get("viewer_jobs", pd.Series(dtype=str))))
    risk_counts = Counter(_iter_json_list(twins.get("risk_tags", pd.Series(dtype=str))))

    return {
        "artifact_version": SEMANTIC_TWIN_ARTIFACT_VERSION,
        "run_id": run_id,
        "run_date": run_date,
        "row_count": int(len(twins)),
        "avg_confidence": round(float(twins["confidence"].mean()), 6) if len(twins) else 0.0,
        "coverage": {
            "rows_with_concepts": int((twins["concepts"] != "[]").sum()) if "concepts" in twins else 0,
            "rows_with_emotional_arcs": int((twins["emotional_arcs"] != "[]").sum())
            if "emotional_arcs" in twins
            else 0,
            "rows_with_viewer_jobs": int((twins["viewer_jobs"] != "[]").sum()) if "viewer_jobs" in twins else 0,
            "rows_with_risk_tags": int((twins["risk_tags"] != "[]").sum()) if "risk_tags" in twins else 0,
        },
        "top_concepts": dict(concept_counts.most_common(30)),
        "top_emotional_arcs": dict(arc_counts.most_common(20)),
        "top_viewer_jobs": dict(job_counts.most_common(20)),
        "risk_tags": dict(risk_counts.most_common(20)),
    }


def semantic_twin_quality_gate(movies: pd.DataFrame, twins: pd.DataFrame) -> dict[str, Any]:
    """Validate that semantic-twin artifacts align with serving catalog rows."""
    if len(movies) != len(twins):
        raise ValueError(f"semantic twin rows ({len(twins)}) do not match movie rows ({len(movies)})")
    expected_ids = pd.to_numeric(movies["id"], errors="raise").astype("int64").to_numpy()
    actual_ids = pd.to_numeric(twins["id"], errors="raise").astype("int64").to_numpy()
    if not (expected_ids == actual_ids).all():
        raise ValueError("semantic twin id order does not match serving catalog order")
    if "semantic_twin_json" not in twins.columns or twins["semantic_twin_json"].isna().any():
        raise ValueError("semantic twin artifact contains null semantic_twin_json values")
    return {
        "stage": "semantic_twins",
        "rows": int(len(twins)),
        "semantic_twin_rows": int(len(twins)),
        "id_order_matches_catalog": True,
    }


def write_semantic_artifacts(
    movies: pd.DataFrame,
    output_dir: Path | str,
    run_id: str,
    run_date: str,
) -> dict[str, Any]:
    """Write semantic-twin parquet and summary JSON artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    twins = build_semantic_twin_frame(movies)
    quality_gate = semantic_twin_quality_gate(movies, twins)
    summary = summarize_semantic_twins(twins, run_id=run_id, run_date=run_date)
    summary["quality_gate"] = quality_gate

    twins_path = output_dir / "semantic_twins.parquet"
    summary_path = output_dir / "semantic_twin_summary.json"
    twins.to_parquet(twins_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "twins": twins,
        "summary": summary,
        "semantic_twins_path": twins_path,
        "semantic_twin_summary_path": summary_path,
        "quality_gate": quality_gate,
    }
