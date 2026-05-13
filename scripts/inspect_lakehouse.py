"""Inspect local medallion/lakehouse snapshots and SCD history."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etl.config import paths
from etl.lakehouse import (
    SCD_CURRENT_COL,
    as_of_scd,
    compare_scd_as_of,
    list_table_versions,
    load_table_version,
)


DEFAULT_TABLES = (
    ("bronze", "movies_raw", "bronze_data"),
    ("silver", "movies_curated", "silver_data"),
    ("gold", "movies_features", "gold_data"),
    ("gold", "dim_movie_scd", "gold_data"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_path(value: Path | str) -> str:
    return str(value)


def _version_record(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": manifest.get("run_id"),
        "run_date": manifest.get("run_date"),
        "row_count": manifest.get("row_count"),
        "data_size_bytes": manifest.get("data_size_bytes"),
        "data_sha256": manifest.get("data_sha256"),
        "manifest_path": manifest.get("manifest_path"),
        "data_path": manifest.get("data_path"),
    }


def summarize_versioned_table(base_path: Path | str, table_name: str) -> dict[str, Any]:
    """Return a compact operational summary for a local versioned table."""
    table_root = Path(base_path) / table_name
    summary: dict[str, Any] = {
        "table": table_name,
        "base_path": _safe_path(base_path),
        "table_path": _safe_path(table_root),
        "status": "missing",
        "version_count": 0,
        "latest": None,
        "versions": [],
    }
    if not table_root.exists():
        return summary

    versions = list_table_versions(base_path, table_name)
    if not versions:
        summary["status"] = "empty"
        return summary

    summary["status"] = "ready"
    summary["version_count"] = len(versions)
    summary["latest"] = _version_record(versions[-1])
    summary["versions"] = [_version_record(version) for version in versions]
    return summary


def summarize_scd_table(
    base_path: Path | str,
    table_name: str = "dim_movie_scd",
    as_of_ts: str | None = None,
    compare_from: str | None = None,
    compare_to: str | None = None,
) -> dict[str, Any]:
    """Return row counts and optional as-of comparison for an SCD Type 2 table."""
    summary = summarize_versioned_table(base_path, table_name)
    if summary["status"] != "ready":
        summary["scd"] = {
            "current_rows": 0,
            "historical_versions": 0,
            "as_of": None,
            "comparison": None,
        }
        return summary

    history = load_table_version(base_path, table_name)
    current_rows = int(history[SCD_CURRENT_COL].astype(bool).sum()) if SCD_CURRENT_COL in history.columns else 0
    scd_summary: dict[str, Any] = {
        "current_rows": current_rows,
        "historical_versions": int(len(history) - current_rows),
        "total_versions": int(len(history)),
        "business_keys": int(history["id"].nunique()) if "id" in history.columns else None,
        "as_of": None,
        "comparison": None,
    }

    if as_of_ts:
        as_of_view = as_of_scd(history, as_of_ts)
        scd_summary["as_of"] = {
            "timestamp": as_of_ts,
            "active_rows": int(len(as_of_view)),
        }

    if compare_from and compare_to:
        scd_summary["comparison"] = compare_scd_as_of(history, compare_from, compare_to)

    summary["scd"] = scd_summary
    return summary


def inspect_lakehouse(
    base_paths: dict[str, Path | str] | None = None,
    as_of_ts: str | None = None,
    compare_from: str | None = None,
    compare_to: str | None = None,
) -> dict[str, Any]:
    """Inspect default Bronze/Silver/Gold local tables."""
    base_paths = base_paths or {
        "bronze": paths.bronze_data,
        "silver": paths.silver_data,
        "gold": paths.gold_data,
    }

    tables: dict[str, dict[str, Any]] = {}
    for layer, table_name, path_attr in DEFAULT_TABLES:
        base_path = base_paths.get(layer) or getattr(paths, path_attr)
        key = f"{layer}.{table_name}"
        if table_name == "dim_movie_scd":
            tables[key] = summarize_scd_table(
                base_path,
                table_name=table_name,
                as_of_ts=as_of_ts,
                compare_from=compare_from,
                compare_to=compare_to,
            )
        else:
            tables[key] = summarize_versioned_table(base_path, table_name)
        tables[key]["layer"] = layer

    ready_count = sum(1 for table in tables.values() if table["status"] == "ready")
    if ready_count == len(tables):
        status = "ready"
    elif ready_count:
        status = "partial"
    else:
        status = "empty"

    return {
        "generated_at": _utc_now(),
        "status": status,
        "ready_table_count": ready_count,
        "table_count": len(tables),
        "tables": tables,
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"Lakehouse status: {report['status']} ({report['ready_table_count']}/{report['table_count']} ready)")
    for key, table in report["tables"].items():
        latest = table.get("latest") or {}
        run_id = latest.get("run_id") or "-"
        row_count = latest.get("row_count")
        row_text = "-" if row_count is None else f"{row_count:,}"
        print(f"- {key}: {table['status']} | versions={table['version_count']} | latest={run_id} | rows={row_text}")
        scd = table.get("scd")
        if scd:
            print(
                f"  SCD: current={scd['current_rows']:,} | historical={scd['historical_versions']:,} | keys={scd.get('business_keys')}"
            )
            if scd.get("as_of"):
                print(f"  As of {scd['as_of']['timestamp']}: active={scd['as_of']['active_rows']:,}")
            if scd.get("comparison"):
                comparison = scd["comparison"]
                print(
                    "  Compare: "
                    f"changed={comparison['changed_count']} | "
                    f"new={comparison['new_count']} | "
                    f"removed={comparison['removed_count']}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--as-of", dest="as_of_ts", help="Timestamp/date for SCD as-of active rows.")
    parser.add_argument("--compare-from", help="Start timestamp/date for SCD comparison.")
    parser.add_argument("--compare-to", help="End timestamp/date for SCD comparison.")
    parser.add_argument("--bronze-base", default=paths.bronze_data)
    parser.add_argument("--silver-base", default=paths.silver_data)
    parser.add_argument("--gold-base", default=paths.gold_data)
    args = parser.parse_args()

    report = inspect_lakehouse(
        base_paths={
            "bronze": Path(args.bronze_base),
            "silver": Path(args.silver_base),
            "gold": Path(args.gold_base),
        },
        as_of_ts=args.as_of_ts,
        compare_from=args.compare_from,
        compare_to=args.compare_to,
    )

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)


if __name__ == "__main__":
    main()
