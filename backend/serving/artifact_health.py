"""Serving artifact health checks.

The API needs a cheap way to verify catalog/vector/semantic artifact alignment
without loading the full recommendation engine or large embedding arrays.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import orjson as _orjson

    def _jloads(s):
        return _orjson.loads(s)
except ImportError:

    def _jloads(s):
        return json.loads(s)


import pandas as pd


def movie_id_sha256(movie_ids: np.ndarray) -> str:
    """Hash the exact ordered int64 movie-id vector."""
    ids = np.asarray(movie_ids, dtype=np.int64).astype("<i8", copy=False)
    return hashlib.sha256(ids.tobytes()).hexdigest()


def _file_report(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": int(path.stat().st_size) if exists and path.is_file() else 0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _jloads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def _contract_value(manifest: dict[str, Any], key: str) -> Any:
    contract = manifest.get("serving_contract") or {}
    quality = manifest.get("quality") or {}
    return contract.get(key) if contract.get(key) is not None else quality.get(key)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evaluate_artifact_health(models_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Return artifact availability and row-alignment diagnostics."""
    models_dir = Path(models_dir)
    data_dir = Path(data_dir)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    paths = {
        "movies": data_dir / "movies_transformed.parquet",
        "semantic_twins": data_dir / "semantic_twins.parquet",
        "semantic_twin_summary": data_dir / "semantic_twin_summary.json",
        "embeddings": models_dir / "sbert_embeddings.npy",
        "turbovec_index": models_dir / "turbovec.tq",
        "movie_ids": models_dir / "movie_ids.npy",
        "pipeline_manifest": models_dir / "pipeline_manifest.json",
    }
    files = {name: _file_report(path) for name, path in paths.items()}
    manifest = _read_json(paths["pipeline_manifest"])
    semantic_summary = _read_json(paths["semantic_twin_summary"])

    checks: dict[str, Any] = {
        "metadata_ready": files["movies"]["exists"],
        "movie_id_map_ready": files["movie_ids"]["exists"],
        "manifest_ready": files["pipeline_manifest"]["exists"] and "_error" not in manifest,
        "vector_files_ready": files["embeddings"]["exists"] and files["turbovec_index"]["exists"],
        "semantic_files_ready": files["semantic_twins"]["exists"] and files["semantic_twin_summary"]["exists"],
        "catalog_vector_aligned": None,
        "semantic_catalog_aligned": None,
        "manifest_counts_aligned": None,
        "semantic_summary_aligned": None,
    }
    row_counts: dict[str, int | None] = {
        "movies": None,
        "movie_ids": None,
        "semantic_twins": None,
    }
    recommendations: list[str] = []
    errors: list[str] = []

    movies_ids = None
    movie_ids = None
    semantic_ids = None

    if files["movies"]["exists"]:
        try:
            movies_ids = pd.read_parquet(paths["movies"], columns=["id"])["id"].astype("int64").to_numpy()
            row_counts["movies"] = len(movies_ids)
        except Exception as exc:
            errors.append(f"movies_transformed.parquet could not be read: {exc}")
            checks["metadata_ready"] = False

    if files["movie_ids"]["exists"]:
        try:
            movie_ids = np.load(paths["movie_ids"]).astype("int64")
            row_counts["movie_ids"] = len(movie_ids)
        except Exception as exc:
            errors.append(f"movie_ids.npy could not be read: {exc}")
            checks["movie_id_map_ready"] = False

    if files["semantic_twins"]["exists"]:
        try:
            semantic_ids = pd.read_parquet(paths["semantic_twins"], columns=["id"])["id"].astype("int64").to_numpy()
            row_counts["semantic_twins"] = len(semantic_ids)
        except Exception as exc:
            errors.append(f"semantic_twins.parquet could not be read: {exc}")
            checks["semantic_files_ready"] = False

    if movies_ids is not None and movie_ids is not None:
        checks["catalog_vector_aligned"] = bool(
            len(movies_ids) == len(movie_ids) and np.array_equal(movies_ids, movie_ids)
        )
        if not checks["catalog_vector_aligned"]:
            recommendations.append(
                "Regenerate artifacts: movies_transformed.parquet and movie_ids.npy are not aligned."
            )

    if movies_ids is not None and semantic_ids is not None:
        checks["semantic_catalog_aligned"] = bool(
            len(movies_ids) == len(semantic_ids) and np.array_equal(movies_ids, semantic_ids)
        )
        if not checks["semantic_catalog_aligned"]:
            recommendations.append("Regenerate semantic_twins.parquet from the same serving catalog snapshot.")

    if manifest and "_error" in manifest:
        errors.append(f"pipeline_manifest.json could not be parsed: {manifest['_error']}")

    expected_rows = _safe_int(_contract_value(manifest, "movie_rows") or _contract_value(manifest, "serving_rows"))
    expected_id_rows = _safe_int(_contract_value(manifest, "movie_id_map_rows"))
    expected_hash = _contract_value(manifest, "movie_id_sha256")
    if expected_rows is not None and row_counts["movies"] is not None:
        checks["manifest_counts_aligned"] = expected_rows == row_counts["movies"]
    if expected_id_rows is not None and row_counts["movie_ids"] is not None:
        checks["manifest_movie_ids_aligned"] = expected_id_rows == row_counts["movie_ids"]
    if expected_hash and movie_ids is not None:
        checks["manifest_movie_id_hash_aligned"] = expected_hash == movie_id_sha256(movie_ids)

    summary_rows = _safe_int(semantic_summary.get("row_count"))
    if summary_rows is not None and row_counts["semantic_twins"] is not None:
        checks["semantic_summary_aligned"] = summary_rows == row_counts["semantic_twins"]
    if semantic_summary and "_error" in semantic_summary:
        errors.append(f"semantic_twin_summary.json could not be parsed: {semantic_summary['_error']}")

    if not files["semantic_twins"]["exists"]:
        recommendations.append("Run the updated Kaggle refresh so semantic_twins.parquet is published to Hugging Face.")
    if not files["semantic_twin_summary"]["exists"]:
        recommendations.append(
            "Run the updated Kaggle refresh so semantic_twin_summary.json is published to Hugging Face."
        )
    if not checks["vector_files_ready"]:
        recommendations.append("Vector artifacts are missing; full TurboVec serving will fall back or be unavailable.")
    if not checks["manifest_ready"]:
        recommendations.append("pipeline_manifest.json is missing or invalid; artifact cache validation is weaker.")

    semantic_ready = bool(
        checks["semantic_files_ready"]
        and checks["semantic_catalog_aligned"] is not False
        and checks["semantic_summary_aligned"] is not False
    )
    vector_ready = bool(
        checks["vector_files_ready"] and checks["movie_id_map_ready"] and checks["catalog_vector_aligned"] is not False
    )

    if not checks["metadata_ready"]:
        status = "unavailable"
    elif semantic_ready and vector_ready:
        status = "ready"
    else:
        status = "degraded"

    return {
        "generated_at": generated_at,
        "status": status,
        "run_id": manifest.get("run_id"),
        "run_date": manifest.get("run_date"),
        "model_name": manifest.get("model_name") or _contract_value(manifest, "model_name"),
        "files": files,
        "row_counts": row_counts,
        "checks": checks,
        "semantic_summary": {
            "row_count": semantic_summary.get("row_count"),
            "avg_confidence": semantic_summary.get("avg_confidence"),
            "coverage": semantic_summary.get("coverage"),
        },
        "recommendations": sorted(set(recommendations)),
        "errors": errors,
    }
