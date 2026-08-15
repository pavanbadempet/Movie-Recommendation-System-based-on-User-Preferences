"""OpenLineage Data Lineage & Provenance Tracker.

Tracks transformation DAGs across Bronze -> Silver -> Gold Medallion pipelines,
generating OpenLineage-compliant JSON specs and interactive DAG graph models.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import Any
import uuid

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class DataLineageNode:
    """Represents a dataset node in the OpenLineage DAG graph."""

    def __init__(self, namespace: str, name: str, layer: str, schema_columns: list[str] | None = None):
        self.node_id = f"{namespace}.{name}"
        self.namespace = namespace
        self.name = name
        self.layer = layer
        self.schema_columns = schema_columns or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "namespace": self.namespace,
            "name": self.name,
            "layer": self.layer,
            "schema_columns": self.schema_columns,
        }


class DataLineageEdge:
    """Represents a transformation lineage edge connecting source and target dataset nodes."""

    def __init__(self, source_id: str, target_id: str, job_name: str, transformation_type: str = "ETL_PIPELINE"):
        self.source_id = source_id
        self.target_id = target_id
        self.job_name = job_name
        self.transformation_type = transformation_type
        self.created_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "job_name": self.job_name,
            "transformation_type": self.transformation_type,
            "created_at": self.created_at,
        }


class DataLineageTracker:
    """OpenLineage DAG Tracker emitting dataset provenance metadata."""

    def __init__(self):
        self.nodes: dict[str, DataLineageNode] = {}
        self.edges: list[DataLineageEdge] = []
        self._bootstrap_medallion_lineage()

    def _bootstrap_medallion_lineage(self):
        """Bootstrap default OpenLineage Medallion DAG graph."""
        bronze = self.add_node("main.recommendations", "movies_raw", "bronze", ["id", "title", "overview"])
        silver = self.add_node(
            "main.recommendations", "movies_curated", "silver", ["id", "title", "genres", "vote_average"]
        )
        gold_features = self.add_node(
            "main.recommendations", "movies_features", "gold", ["id", "feature_vector", "cluster"]
        )
        gold_scd = self.add_node(
            "main.recommendations", "dim_movie_scd", "gold", ["id", "is_current", "valid_from", "valid_to"]
        )

        self.add_edge(bronze.node_id, silver.node_id, "pyspark_silver_curation_job", "CONTRACT_VALIDATION_TYPECAST")
        self.add_edge(
            silver.node_id, gold_features.node_id, "pyspark_gold_feature_engineering", "VECTOR_EMBEDDING_GENERATION"
        )
        self.add_edge(silver.node_id, gold_scd.node_id, "pyspark_scd2_merge", "HISTORICAL_SCD2_HASH_DRIFT")

    def add_node(
        self, namespace: str, name: str, layer: str, schema_columns: list[str] | None = None
    ) -> DataLineageNode:
        node = DataLineageNode(namespace, name, layer, schema_columns)
        self.nodes[node.node_id] = node
        return node

    def add_edge(
        self, source_id: str, target_id: str, job_name: str, transformation_type: str = "ETL_PIPELINE"
    ) -> DataLineageEdge:
        edge = DataLineageEdge(source_id, target_id, job_name, transformation_type)
        self.edges.append(edge)
        logger.info(f"Added lineage edge: {source_id} -> {target_id} via {job_name}")
        return edge

    def get_openlineage_event(self, job_name: str) -> dict[str, Any]:
        """Generate OpenLineage 1.0 specification payload."""
        return {
            "eventType": "COMPLETE",
            "eventTime": _utc_now(),
            "run": {"runId": str(uuid.uuid4())},
            "job": {
                "namespace": "apex.medallion.pipeline",
                "name": job_name,
            },
            "producer": "https://github.com/pavanbadempet/AI-Recommendation-System",
            "inputs": [node.to_dict() for node in self.nodes.values() if node.layer in ("bronze", "silver")],
            "outputs": [node.to_dict() for node in self.nodes.values() if node.layer == "gold"],
        }

    def to_graph_dict(self) -> dict[str, Any]:
        """Export interactive graph dict containing nodes and edges for UI rendering."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }


# Global singleton instance
_lineage_tracker: DataLineageTracker | None = None


def get_lineage_tracker() -> DataLineageTracker:
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = DataLineageTracker()
    return _lineage_tracker
