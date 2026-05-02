"""
Slowly changing dimension helpers for curated movie metadata.

The recommendation model uses the latest movie attributes for serving, but an
interview-grade data platform should also be able to explain how historical
dimension changes are tracked. This module implements SCD Type 2 in Pandas so
the behavior is deterministic, unit-testable, and easy to port to PySpark or
SQL MERGE statements later.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Iterable

import pandas as pd


SCD_START_COL = "effective_start_at"
SCD_END_COL = "effective_end_at"
SCD_CURRENT_COL = "is_current"
SCD_HASH_COL = "record_hash"
DEFAULT_HIGH_DATE = "9999-12-31T00:00:00"


def _as_list(columns: Iterable[str]) -> list[str]:
    return list(columns)


def _normalize_value(value: object) -> str:
    if pd.isna(value):
        return "<NULL>"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value).strip()


def build_record_hash(row: pd.Series, tracked_columns: Iterable[str]) -> str:
    """Create a stable hash for the attributes that should trigger SCD changes."""
    payload = "||".join(_normalize_value(row[col]) for col in tracked_columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _key_tuple(row: pd.Series, key_columns: list[str]) -> tuple:
    return tuple(row[col] for col in key_columns)


def _validate_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _prepare_new_versions(
    incoming: pd.DataFrame,
    tracked_columns: list[str],
    effective_ts: str,
    high_date: str,
) -> pd.DataFrame:
    versions = incoming.copy()
    versions[SCD_START_COL] = effective_ts
    versions[SCD_END_COL] = high_date
    versions[SCD_CURRENT_COL] = True
    versions[SCD_HASH_COL] = versions.apply(build_record_hash, axis=1, tracked_columns=tracked_columns)
    return versions


def apply_scd_type2(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
    key_columns: Iterable[str],
    tracked_columns: Iterable[str],
    effective_ts: str | datetime,
    high_date: str = DEFAULT_HIGH_DATE,
) -> pd.DataFrame:
    """
    Apply SCD Type 2 changes to a dimension table.

    Args:
        existing: Current historical dimension table. Can be empty or None.
        incoming: Latest snapshot of source records.
        key_columns: Business key columns, for example ["id"].
        tracked_columns: Columns whose changes should create a new version.
        effective_ts: Timestamp for this pipeline run.
        high_date: Open-ended end timestamp for current records.

    Returns:
        A dimension table containing historical and current records.
    """
    key_columns = _as_list(key_columns)
    tracked_columns = _as_list(tracked_columns)
    effective_ts = effective_ts.isoformat() if isinstance(effective_ts, datetime) else str(effective_ts)

    _validate_columns(incoming, key_columns + tracked_columns, "incoming")

    if incoming.empty:
        return existing.copy() if existing is not None else pd.DataFrame()

    if existing is None or existing.empty:
        return _prepare_new_versions(incoming, tracked_columns, effective_ts, high_date).reset_index(drop=True)

    existing = existing.copy()
    _validate_columns(existing, key_columns + tracked_columns, "existing")

    for scd_col in [SCD_START_COL, SCD_END_COL, SCD_CURRENT_COL, SCD_HASH_COL]:
        if scd_col not in existing.columns:
            if scd_col == SCD_CURRENT_COL:
                existing[scd_col] = True
            elif scd_col == SCD_END_COL:
                existing[scd_col] = high_date
            elif scd_col == SCD_START_COL:
                existing[scd_col] = effective_ts
            else:
                existing[scd_col] = existing.apply(build_record_hash, axis=1, tracked_columns=tracked_columns)

    current_mask = existing[SCD_CURRENT_COL].astype(bool)
    current = existing[current_mask].copy()
    current["_scd_key"] = current.apply(_key_tuple, axis=1, key_columns=key_columns)

    incoming_versions = _prepare_new_versions(incoming, tracked_columns, effective_ts, high_date)
    incoming_versions["_scd_key"] = incoming_versions.apply(_key_tuple, axis=1, key_columns=key_columns)

    current_by_key = {
        row["_scd_key"]: row
        for _, row in current.iterrows()
    }
    rows_to_insert = []
    keys_to_expire = set()

    for _, row in incoming_versions.iterrows():
        key = row["_scd_key"]
        if key not in current_by_key:
            rows_to_insert.append(row.drop(labels=["_scd_key"]))
            continue

        current_row = current_by_key[key]
        if current_row[SCD_HASH_COL] != row[SCD_HASH_COL]:
            keys_to_expire.add(key)
            rows_to_insert.append(row.drop(labels=["_scd_key"]))

    if keys_to_expire:
        existing_keys = existing.apply(_key_tuple, axis=1, key_columns=key_columns)
        expire_mask = current_mask & existing_keys.isin(list(keys_to_expire))
        existing.loc[expire_mask, SCD_CURRENT_COL] = False
        existing.loc[expire_mask, SCD_END_COL] = effective_ts

    if rows_to_insert:
        inserts = pd.DataFrame(rows_to_insert)
        existing = pd.concat([existing, inserts], ignore_index=True, sort=False)

    return existing.reset_index(drop=True)
