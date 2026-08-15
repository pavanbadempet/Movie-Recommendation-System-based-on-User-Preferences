"""Lightweight dataset contract loading and DataFrame validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"


def load_contract(contract_name: str) -> dict[str, Any]:
    """Load a contract JSON document from the repository contracts directory."""
    contract_path = CONTRACTS_DIR / f"{contract_name}.schema.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract file not found: {contract_path}")
    return json.loads(contract_path.read_text(encoding="utf-8"))


def _validate_series_type(series: pd.Series, type_name: str) -> bool:
    values = series.dropna()
    if values.empty:
        return True

    if type_name == "integer":
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            return False
        return bool(((numeric % 1) == 0).all())

    if type_name == "number":
        return bool(pd.to_numeric(values, errors="coerce").notna().all())

    if type_name == "string":
        return bool(values.map(lambda value: isinstance(value, str)).all())

    if type_name == "boolean":
        return bool(values.map(lambda value: isinstance(value, bool)).all())

    if type_name == "datetime":
        return bool(pd.to_datetime(values, errors="coerce").notna().all())

    raise ValueError(f"Unsupported contract type: {type_name}")


def validate_dataframe_contract(
    df: pd.DataFrame,
    contract_name: str,
    *,
    stage: str | None = None,
) -> dict[str, Any]:
    """Validate a DataFrame against a machine-readable dataset contract."""
    contract = load_contract(contract_name)
    dataset_name = str(contract.get("name", contract_name))
    stage_label = stage or dataset_name

    required_columns = set(contract.get("required_columns", []))
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{stage_label} missing contract columns: {missing_columns}")

    columns = contract.get("columns", {})
    for column_name, rules in columns.items():
        if column_name not in df.columns:
            continue

        series = df[column_name]
        nullable = bool(rules.get("nullable", True))
        if not nullable and series.isna().any():
            raise ValueError(f"{stage_label} column '{column_name}' contains nulls")

        type_name = rules.get("type")
        if type_name and not _validate_series_type(series, type_name):
            raise ValueError(f"{stage_label} column '{column_name}' violates type '{type_name}'")

        if "min" in rules or "max" in rules:
            numeric = pd.to_numeric(series.dropna(), errors="coerce")
            if numeric.isna().any():
                raise ValueError(f"{stage_label} column '{column_name}' is not fully numeric")
            minimum = rules.get("min")
            maximum = rules.get("max")
            if minimum is not None and (numeric < minimum).any():
                raise ValueError(f"{stage_label} column '{column_name}' is below minimum {minimum}")
            if maximum is not None and (numeric > maximum).any():
                raise ValueError(f"{stage_label} column '{column_name}' is above maximum {maximum}")

        allowed_values = rules.get("enum")
        if allowed_values is not None:
            invalid = sorted(set(series.dropna()) - set(allowed_values))
            if invalid:
                raise ValueError(f"{stage_label} column '{column_name}' has invalid values: {invalid}")

    primary_key = list(contract.get("primary_key", []))
    if primary_key:
        duplicates = df.duplicated(subset=primary_key, keep=False)
        if duplicates.any():
            raise ValueError(f"{stage_label} violates primary key {primary_key}: duplicate rows found")

    return {
        "contract_name": dataset_name,
        "contract_version": int(contract.get("version", 1)),
        "rows": len(df),
        "primary_key": primary_key,
    }
