"""Unity Catalog Metastore & Governance Engine."""

from __future__ import annotations

import hashlib
from typing import Any


class CatalogTable:
    def __init__(self, catalog: str, schema: str, name: str, columns: list[str] | None = None):
        self.catalog = catalog
        self.schema = schema
        self.name = name
        self.columns = columns or ["id", "title", "vote_average", "tags"]
        self.full_name = f"{catalog}.{schema}.{name}"

    def check_privilege(self, role: str, privilege: str) -> bool:
        if role in ("account_admin", "admin"):
            return True
        if role in ("data_scientists", "analysts") and privilege in ("SELECT", "READ"):
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.schema,
            "catalog": self.catalog,
            "full_name": self.full_name,
            "columns": self.columns,
        }


class UnityCatalog:
    def __init__(self):
        self.default_catalog = "main"
        self._tables: dict[str, CatalogTable] = {
            "main.recommendations.movies_raw": CatalogTable("main", "recommendations", "movies_raw"),
            "main.recommendations.movies_curated": CatalogTable("main", "recommendations", "movies_curated"),
            "main.recommendations.movies_features": CatalogTable("main", "recommendations", "movies_features"),
            "main.recommendations.user_events": CatalogTable("main", "recommendations", "user_events"),
        }

    def get_table(self, catalog: str, schema: str, table_name: str) -> CatalogTable | None:
        full_name = f"{catalog}.{schema}.{table_name}"
        return self._tables.get(full_name)

    def list_tables(self, catalog: str = "main", schema: str = "recommendations") -> list[CatalogTable]:
        prefix = f"{catalog}.{schema}."
        return [t for k, t in self._tables.items() if k.startswith(prefix)]

    def apply_pii_masking(self, column_name: str, value: str) -> str:
        if "email" in column_name.lower():
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "main": {
                "recommendations": [t.to_dict() for t in self.list_tables("main", "recommendations")]
            }
        }


_INSTANCE: UnityCatalog | None = None


def get_unity_catalog() -> UnityCatalog:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = UnityCatalog()
    return _INSTANCE
