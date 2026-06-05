"""
Customer catalog onboarding helpers.

This module intentionally uses the Python standard library rather than a paid
database or object store. It supports the zero-capital path: preview a CSV,
surface quality issues, and persist a raw upload plus manifest locally.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_ROOT = REPO_ROOT / "data" / "customer_catalogs"

DEFAULT_COLUMN_MAPPING = {
    "source_content_id": "id",
    "title": "title",
    "description": "overview",
    "genres": "genres",
    "people": "cast",
    "language": "original_language",
    "release_date": "release_date",
    "rating": "vote_average",
    "popularity": "popularity",
}


def catalog_root() -> Path:
    configured = os.getenv("NOVA_CATALOG_UPLOAD_PATH")
    return Path(configured) if configured else DEFAULT_CATALOG_ROOT


def safe_identifier(value: str) -> str:
    """Create a filesystem-safe tenant/catalog identifier."""
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip()).strip("-")
    return cleaned or "unknown"


def normalize_mapping(column_mapping: dict[str, str] | None = None) -> dict[str, str]:
    """Merge user-provided mapping with default catalog column names."""
    mapping = dict(DEFAULT_COLUMN_MAPPING)
    for key, value in (column_mapping or {}).items():
        if value:
            mapping[str(key)] = str(value)
    return mapping


def _read_csv_rows(csv_text: str, max_rows: int = 5000) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_text or not csv_text.strip():
        raise ValueError("csv_text is empty")

    stream = io.StringIO(csv_text.lstrip("\ufeff"))
    reader = csv.DictReader(stream)
    if not reader.fieldnames:
        raise ValueError("CSV header row is required")

    rows: list[dict[str, str]] = []
    for idx, row in enumerate(reader):
        if idx >= max_rows:
            break
        rows.append({str(key): value for key, value in row.items() if key is not None})
    return list(reader.fieldnames), rows


def _field(row: dict[str, str], mapping: dict[str, str], field_name: str, default: str = "") -> str:
    source_column = mapping.get(field_name)
    if not source_column:
        return default
    value = row.get(source_column, default)
    if value is None:
        return default
    return str(value).strip()


def _content_id(tenant_id: str, catalog_id: str, source_content_id: str) -> str:
    raw = f"{tenant_id}||{catalog_id}||{source_content_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def profile_catalog_csv(
    csv_text: str,
    tenant_id: str,
    catalog_id: str,
    column_mapping: dict[str, str] | None = None,
    sample_size: int = 20,
    max_rows: int = 5000,
) -> dict[str, Any]:
    """Profile a customer catalog CSV and return quality/onboarding metadata."""
    mapping = normalize_mapping(column_mapping)
    columns, rows = _read_csv_rows(csv_text, max_rows=max_rows)

    missing_mapped_columns = sorted({source for source in mapping.values() if source and source not in columns})
    missing_required_mapped_columns = sorted(
        {mapping[field] for field in ("title", "description") if mapping.get(field) and mapping[field] not in columns}
    )
    seen_source_ids: set[str] = set()
    duplicate_source_ids = 0
    missing_title = 0
    weak_description = 0
    valid_rows = 0
    samples: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        source_content_id = _field(row, mapping, "source_content_id") or str(index)
        title = _field(row, mapping, "title")
        description = _field(row, mapping, "description")
        genres = _field(row, mapping, "genres")
        people = _field(row, mapping, "people")
        language = _field(row, mapping, "language")
        release_date = _field(row, mapping, "release_date")

        if source_content_id in seen_source_ids:
            duplicate_source_ids += 1
        seen_source_ids.add(source_content_id)

        row_is_valid = True
        if not title:
            missing_title += 1
            row_is_valid = False
        if len(description) < 20:
            weak_description += 1
            row_is_valid = False
        if row_is_valid:
            valid_rows += 1

        if len(samples) < sample_size:
            samples.append(
                {
                    "content_id": _content_id(tenant_id, catalog_id, source_content_id),
                    "source_content_id": source_content_id,
                    "title": title,
                    "description_length": len(description),
                    "genres": genres,
                    "people": people,
                    "language": language,
                    "release_date": release_date,
                    "valid": row_is_valid,
                }
            )

    total_rows = len(rows)
    invalid_rows = total_rows - valid_rows
    quality_score = round(valid_rows / total_rows, 4) if total_rows else 0.0
    warnings = []
    if missing_required_mapped_columns:
        warnings.append("mapped_columns_missing")
    elif missing_mapped_columns:
        warnings.append("optional_mapped_columns_missing")
    if duplicate_source_ids:
        warnings.append("duplicate_source_content_ids")
    if missing_title:
        warnings.append("missing_titles")
    if weak_description:
        warnings.append("weak_descriptions")

    return {
        "tenant_id": tenant_id,
        "catalog_id": catalog_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "mapping": mapping,
        "columns": columns,
        "missing_mapped_columns": missing_mapped_columns,
        "missing_required_mapped_columns": missing_required_mapped_columns,
        "total_rows_profiled": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "missing_title_rows": missing_title,
        "weak_description_rows": weak_description,
        "duplicate_source_content_ids": duplicate_source_ids,
        "quality_score": quality_score,
        "ready_for_ingestion": total_rows > 0 and quality_score >= 0.8 and not missing_required_mapped_columns,
        "warnings": warnings,
        "samples": samples,
        "profile_limit_reached": total_rows == max_rows,
    }


def persist_catalog_upload(
    csv_text: str,
    tenant_id: str,
    catalog_id: str,
    filename: str,
    column_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Persist a raw customer CSV and manifest to the local free-tier filesystem."""
    profile = profile_catalog_csv(
        csv_text,
        tenant_id=tenant_id,
        catalog_id=catalog_id,
        column_mapping=column_mapping,
    )
    upload_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tenant_dir = safe_identifier(tenant_id)
    catalog_dir = safe_identifier(catalog_id)
    upload_dir = catalog_root() / tenant_dir / catalog_dir / "uploads" / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    raw_path = upload_dir / "raw.csv"
    manifest_path = upload_dir / "manifest.json"
    raw_path.write_text(csv_text, encoding="utf-8")

    manifest = {
        "upload_id": upload_id,
        "filename": filename,
        "tenant_id": tenant_id,
        "catalog_id": catalog_id,
        "raw_path": str(raw_path),
        "manifest_path": str(manifest_path),
        "profile": profile,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
