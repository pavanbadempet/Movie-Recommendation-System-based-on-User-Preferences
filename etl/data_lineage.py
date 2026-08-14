"""OpenLineage Data Lineage Tracker for Medallion Lakehouse."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class LineageTracker:
    def __init__(self):
        self._nodes = [
            {"id": "bronze_movies_raw", "name": "movies_raw", "layer": "bronze", "type": "table"},
            {"id": "silver_movies_curated", "name": "movies_curated", "layer": "silver", "type": "table"},
            {"id": "gold_movies_features", "name": "movies_features", "layer": "gold", "type": "table"},
            {"id": "neon_pgvector_shards", "name": "neon_shards", "layer": "serving", "type": "database"},
        ]
        self._edges = [
            {"source": "bronze_movies_raw", "target": "silver_movies_curated", "transformation": "data_quality_gates"},
            {"source": "silver_movies_curated", "target": "gold_movies_features", "transformation": "scd2_merge_clustering"},
            {"source": "gold_movies_features", "target": "neon_pgvector_shards", "transformation": "hash_partitioned_export"},
        ]

    def to_graph_dict(self) -> dict[str, Any]:
        return {
            "nodes": self._nodes,
            "edges": self._edges,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_openlineage_event(self, job_name: str = "pyspark_medallion_etl") -> dict[str, Any]:
        return {
            "eventType": "COMPLETE",
            "eventTime": datetime.now(timezone.utc).isoformat(),
            "job": {
                "namespace": "apex.movie.rec",
                "name": job_name,
            },
            "inputs": [
                {"namespace": "apex.default", "name": "movies_raw"}
            ],
            "outputs": [
                {"namespace": "apex.default", "name": "movies_features"}
            ],
            "producer": "https://github.com/pavanbadempet/AI-Recommendation-System/etl",
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
        }


_TRACKER: LineageTracker | None = None


def get_lineage_tracker() -> LineageTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = LineageTracker()
    return _TRACKER
