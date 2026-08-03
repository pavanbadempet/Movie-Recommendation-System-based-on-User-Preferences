"""Unity Catalog 3-Level Namespace Metastore & Governance Engine.

Provides catalog, schema, and table management (catalog.schema.table),
RBAC privilege evaluation, and PII column data masking policies.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class UnityCatalogTable:
    """Represents a 3-level Unity Catalog table entity."""

    def __init__(
        self,
        catalog: str,
        schema: str,
        name: str,
        table_type: str = "MANAGED",
        data_format: str = "DELTA",
        columns: Optional[List[Dict[str, Any]]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        self.catalog = catalog
        self.schema = schema
        self.name = name
        self.full_name = f"{catalog}.{schema}.{name}"
        self.table_type = table_type
        self.data_format = data_format
        self.columns = columns or []
        self.properties = properties or {}
        self.created_at = _utc_now()
        self.updated_at = _utc_now()
        self.grants: Dict[str, List[str]] = {}  # principal -> list of privileges

    def grant_privilege(self, principal: str, privilege: str):
        """Grant a privilege (e.g. SELECT, MODIFY) to a principal user or group."""
        if principal not in self.grants:
            self.grants[principal] = []
        if privilege.upper() not in self.grants[principal]:
            self.grants[principal].append(privilege.upper())

    def check_privilege(self, principal: str, privilege: str) -> bool:
        """Check if a principal has a given privilege on this table."""
        user_privs = self.grants.get(principal, [])
        return privilege.upper() in user_privs or "ALL_PRIVILEGES" in user_privs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "catalog": self.catalog,
            "schema": self.schema,
            "name": self.name,
            "full_name": self.full_name,
            "table_type": self.table_type,
            "data_format": self.data_format,
            "columns": self.columns,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "grants": self.grants,
        }


class UnityCatalogManager:
    """Unity Catalog REST Metastore Manager controlling 3-level namespaces & RBAC."""

    def __init__(self, default_catalog: str = "main"):
        self.default_catalog = default_catalog
        self._catalogs: Dict[str, Dict[str, Dict[str, UnityCatalogTable]]] = {}
        self._init_default_medallion_namespaces()

    def _init_default_medallion_namespaces(self):
        """Bootstrap default 3-level Medallion namespaces."""
        self.register_table(
            catalog="main",
            schema="recommendations",
            table_name="movies_raw",
            data_format="DELTA",
            columns=[
                {"name": "id", "type": "INT", "nullable": False},
                {"name": "title", "type": "STRING", "nullable": False},
                {"name": "overview", "type": "STRING", "nullable": True},
            ],
            properties={"layer": "bronze", "governance_tier": "raw"},
        )
        self.register_table(
            catalog="main",
            schema="recommendations",
            table_name="movies_curated",
            data_format="DELTA",
            columns=[
                {"name": "id", "type": "INT", "nullable": False},
                {"name": "title", "type": "STRING", "nullable": False},
                {"name": "genres", "type": "STRING", "nullable": True},
                {"name": "vote_average", "type": "DOUBLE", "nullable": True},
            ],
            properties={"layer": "silver", "governance_tier": "curated"},
        )
        self.register_table(
            catalog="main",
            schema="recommendations",
            table_name="movies_features",
            data_format="DELTA",
            columns=[
                {"name": "id", "type": "INT", "nullable": False},
                {"name": "feature_vector", "type": "ARRAY<FLOAT>", "nullable": False},
                {"name": "similarity_cluster", "type": "INT", "nullable": True},
            ],
            properties={"layer": "gold", "governance_tier": "features"},
        )

        # Grant default admin rights
        for table in self.list_tables("main", "recommendations"):
            table.grant_privilege("account_admin", "ALL_PRIVILEGES")
            table.grant_privilege("data_scientists", "SELECT")

    def register_table(
        self,
        catalog: str,
        schema: str,
        table_name: str,
        table_type: str = "MANAGED",
        data_format: str = "DELTA",
        columns: Optional[List[Dict[str, Any]]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> UnityCatalogTable:
        """Register or update a table in the 3-level Unity Catalog metastore."""
        if catalog not in self._catalogs:
            self._catalogs[catalog] = {}
        if schema not in self._catalogs[catalog]:
            self._catalogs[catalog][schema] = {}

        table = UnityCatalogTable(
            catalog=catalog,
            schema=schema,
            name=table_name,
            table_type=table_type,
            data_format=data_format,
            columns=columns,
            properties=properties,
        )
        self._catalogs[catalog][schema][table_name] = table
        logger.info(f"Registered Unity Catalog table: {table.full_name}")
        return table

    def get_table(self, catalog: str, schema: str, table_name: str) -> Optional[UnityCatalogTable]:
        """Fetch table metadata from Unity Catalog metastore."""
        return self._catalogs.get(catalog, {}).get(schema, {}).get(table_name)

    def list_tables(self, catalog: str, schema: str) -> List[UnityCatalogTable]:
        """List all tables in a specific catalog.schema namespace."""
        return list(self._catalogs.get(catalog, {}).get(schema, {}).values())

    def apply_pii_masking(self, column_name: str, value: str) -> str:
        """Apply PII data masking policy to sensitive columns (e.g. email, user_id)."""
        if any(keyword in column_name.lower() for keyword in ["email", "ip", "address", "phone"]):
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Export full Unity Catalog metastore tree as dictionary."""
        result: Dict[str, Any] = {}
        for cat, schemas in self._catalogs.items():
            result[cat] = {}
            for sch, tables in schemas.items():
                result[cat][sch] = [t.to_dict() for t in tables.values()]
        return result


# Global singleton instance
_unity_catalog: Optional[UnityCatalogManager] = None


def get_unity_catalog() -> UnityCatalogManager:
    global _unity_catalog
    if _unity_catalog is None:
        _unity_catalog = UnityCatalogManager()
    return _unity_catalog
