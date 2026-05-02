"""
Lakehouse data models, versioned snapshots, and local time travel helpers.

Delta Lake gives this behavior through the transaction log in production Spark
runs. These helpers provide the same project contracts for local/Kaggle parquet
artifacts: canonical table models, per-run manifests, and as-of reads.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from etl.scd import DEFAULT_HIGH_DATE, SCD_CURRENT_COL, SCD_END_COL, SCD_HASH_COL, SCD_START_COL


@dataclass(frozen=True)
class ColumnSpec:
    """Column-level data contract for a lakehouse table."""

    name: str
    dtype: str
    nullable: bool = True
    description: str = ""


@dataclass(frozen=True)
class TableModel:
    """Table-level data model metadata."""

    name: str
    layer: str
    grain: str
    primary_key: tuple[str, ...]
    partition_columns: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if not column.nullable)


MOVIE_RAW_MODEL = TableModel(
    name="movies_raw",
    layer="bronze",
    grain="One source TMDB movie row per daily batch run.",
    primary_key=("id",),
    partition_columns=("run_date", "run_id"),
    columns=(
        ColumnSpec("id", "long", nullable=False, description="TMDB movie id."),
        ColumnSpec("title", "string", nullable=True),
        ColumnSpec("overview", "string", nullable=True),
        ColumnSpec("genres", "string", nullable=True),
        ColumnSpec("vote_average", "double", nullable=True),
        ColumnSpec("vote_count", "double", nullable=True),
        ColumnSpec("popularity", "double", nullable=True),
        ColumnSpec("release_date", "string", nullable=True),
        ColumnSpec("poster_path", "string", nullable=True),
        ColumnSpec("adult", "boolean", nullable=True),
    ),
)

MOVIE_CURATED_MODEL = TableModel(
    name="movies_curated",
    layer="silver",
    grain="One validated, deduplicated movie row per TMDB movie id.",
    primary_key=("id",),
    partition_columns=("run_date", "run_id"),
    columns=(
        ColumnSpec("id", "long", nullable=False),
        ColumnSpec("title", "string", nullable=False),
        ColumnSpec("overview", "string", nullable=False),
        ColumnSpec("genres", "string", nullable=True),
        ColumnSpec("vote_average", "double", nullable=True),
        ColumnSpec("vote_count", "double", nullable=True),
        ColumnSpec("popularity", "double", nullable=True),
        ColumnSpec("release_date", "string", nullable=True),
        ColumnSpec("poster_path", "string", nullable=True),
    ),
)

MOVIE_FEATURE_MODEL = TableModel(
    name="movies_features",
    layer="gold",
    grain="One ML-ready recommendation feature row per TMDB movie id.",
    primary_key=("id",),
    partition_columns=("run_date", "run_id"),
    columns=(
        ColumnSpec("id", "long", nullable=False),
        ColumnSpec("title", "string", nullable=False),
        ColumnSpec("overview", "string", nullable=False),
        ColumnSpec("genres", "string", nullable=True),
        ColumnSpec("tags", "string", nullable=False, description="Text used to generate semantic embeddings."),
        ColumnSpec("vote_average", "double", nullable=True),
        ColumnSpec("vote_count", "double", nullable=True),
        ColumnSpec("popularity", "double", nullable=True),
    ),
)

MOVIE_SCD_MODEL = TableModel(
    name="dim_movie_scd",
    layer="gold",
    grain="One historical movie attribute version per TMDB movie id and effective interval.",
    primary_key=("id", SCD_START_COL),
    partition_columns=("is_current",),
    columns=(
        ColumnSpec("id", "long", nullable=False),
        ColumnSpec("title", "string", nullable=False),
        ColumnSpec("overview", "string", nullable=False),
        ColumnSpec("genres", "string", nullable=True),
        ColumnSpec("vote_average", "double", nullable=True),
        ColumnSpec("vote_count", "double", nullable=True),
        ColumnSpec("popularity", "double", nullable=True),
        ColumnSpec("release_date", "string", nullable=True),
        ColumnSpec(SCD_HASH_COL, "string", nullable=False),
        ColumnSpec(SCD_START_COL, "timestamp", nullable=False),
        ColumnSpec(SCD_END_COL, "timestamp", nullable=False),
        ColumnSpec(SCD_CURRENT_COL, "boolean", nullable=False),
    ),
)

FACT_MOVIE_EVENT_MODEL = TableModel(
    name="fact_movie_event",
    layer="gold",
    grain="One user behavior event emitted by the application.",
    primary_key=("event_id",),
    partition_columns=("event_date",),
    columns=(
        ColumnSpec("event_id", "string", nullable=False),
        ColumnSpec("event_ts", "timestamp", nullable=False),
        ColumnSpec("event_type", "string", nullable=False),
        ColumnSpec("movie_id", "long", nullable=True),
        ColumnSpec("user_id", "string", nullable=True),
        ColumnSpec("rating", "double", nullable=True),
        ColumnSpec("query_text", "string", nullable=True),
    ),
)

TABLE_MODELS: dict[str, TableModel] = {
    model.name: model
    for model in (
        MOVIE_RAW_MODEL,
        MOVIE_CURATED_MODEL,
        MOVIE_FEATURE_MODEL,
        MOVIE_SCD_MODEL,
        FACT_MOVIE_EVENT_MODEL,
    )
}


def utc_now() -> str:
    """Return an ISO UTC timestamp safe for manifests."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _atomic_write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(output_path)
    return output_path


def _atomic_write_parquet(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        df.to_parquet(temp_path, index=False)
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return output_path


def _file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_run_date(run_date: str | date | datetime) -> str:
    if isinstance(run_date, datetime):
        return run_date.date().isoformat()
    if isinstance(run_date, date):
        return run_date.isoformat()
    return str(run_date)


def normalize_as_of_ts(value: str | date | datetime) -> str:
    """Normalize date/datetime inputs for lexical ISO interval comparison."""
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return f"{value.isoformat()}T23:59:59Z"
    value = str(value)
    if len(value) == 10:
        return f"{value}T23:59:59Z"
    return value


def get_table_model(table_name: str) -> TableModel:
    try:
        return TABLE_MODELS[table_name]
    except KeyError as exc:
        known = ", ".join(sorted(TABLE_MODELS))
        raise ValueError(f"Unknown table model '{table_name}'. Known tables: {known}") from exc


def validate_table_contract(
    df: pd.DataFrame,
    model: TableModel | str,
    require_primary_key_unique: bool = True,
) -> dict[str, Any]:
    """Validate a DataFrame against a table model's core contract."""
    model = get_table_model(model) if isinstance(model, str) else model
    missing_columns = [column for column in model.required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{model.name} missing required columns: {missing_columns}")

    null_columns = [
        column
        for column in model.required_columns
        if column in df.columns and df[column].isna().any()
    ]
    if null_columns:
        raise ValueError(f"{model.name} has nulls in required columns: {null_columns}")

    primary_key_columns = [column for column in model.primary_key if column in df.columns]
    duplicate_keys = 0
    if require_primary_key_unique and primary_key_columns:
        duplicate_keys = int(df.duplicated(subset=primary_key_columns).sum())
        if duplicate_keys:
            raise ValueError(f"{model.name} has duplicate primary keys: {duplicate_keys}")

    return {
        "table": model.name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "required_columns": list(model.required_columns),
        "duplicate_keys": duplicate_keys,
    }


def write_versioned_snapshot(
    df: pd.DataFrame,
    base_path: Path | str,
    table_name: str,
    run_id: str,
    run_date: str | date | datetime,
    model: TableModel | str | None = None,
    validate_contract: bool = True,
) -> dict[str, Any]:
    """
    Write a run-scoped parquet snapshot and manifest.

    Layout:
      {base_path}/{table_name}/run_date=YYYY-MM-DD/run_id={run_id}/data.parquet
      {base_path}/{table_name}/run_date=YYYY-MM-DD/run_id={run_id}/_manifest.json
      {base_path}/{table_name}/_latest.json
    """
    model = get_table_model(model or table_name) if isinstance(model or table_name, str) else model
    if model is None:
        raise ValueError("model could not be resolved")

    if validate_contract:
        contract = validate_table_contract(df, model)
    else:
        contract = {"table": model.name, "rows": int(len(df)), "columns": list(df.columns)}

    run_date = _normalize_run_date(run_date)
    table_root = Path(base_path) / table_name
    version_dir = table_root / f"run_date={run_date}" / f"run_id={run_id}"
    data_path = _atomic_write_parquet(df, version_dir / "data.parquet")

    manifest = {
        "table": table_name,
        "layer": model.layer,
        "grain": model.grain,
        "run_id": run_id,
        "run_date": run_date,
        "created_at": utc_now(),
        "row_count": int(len(df)),
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "data_size_bytes": int(data_path.stat().st_size),
        "model": asdict(model),
        "contract": contract,
    }

    manifest_path = _atomic_write_json(manifest, version_dir / "_manifest.json")
    manifest["manifest_path"] = str(manifest_path)
    _atomic_write_json(manifest, table_root / "_latest.json")
    return manifest


def list_table_versions(base_path: Path | str, table_name: str) -> list[dict[str, Any]]:
    """List version manifests for a versioned local table."""
    table_root = Path(base_path) / table_name
    manifests = []
    for manifest_path in table_root.glob("run_date=*/run_id=*/_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["manifest_path"] = str(manifest_path)
        manifests.append(manifest)
    return sorted(manifests, key=lambda item: (item["run_date"], item["run_id"]))


def resolve_table_version(
    base_path: Path | str,
    table_name: str,
    run_id: str | None = None,
    as_of_date: str | date | datetime | None = None,
) -> dict[str, Any]:
    """Resolve a table version by run_id, as-of date, or latest manifest."""
    versions = list_table_versions(base_path, table_name)
    if not versions:
        raise FileNotFoundError(f"No versions found for table {table_name}")

    if run_id is not None:
        for version in versions:
            if version["run_id"] == run_id:
                return version
        raise FileNotFoundError(f"No version found for table {table_name} and run_id={run_id}")

    if as_of_date is not None:
        target_date = _normalize_run_date(as_of_date)
        eligible = [version for version in versions if version["run_date"] <= target_date]
        if not eligible:
            raise FileNotFoundError(f"No version found for table {table_name} as of {target_date}")
        return eligible[-1]

    return versions[-1]


def load_table_version(
    base_path: Path | str,
    table_name: str,
    run_id: str | None = None,
    as_of_date: str | date | datetime | None = None,
) -> pd.DataFrame:
    """Load a versioned parquet table by run_id, as-of date, or latest."""
    manifest = resolve_table_version(base_path, table_name, run_id=run_id, as_of_date=as_of_date)
    return pd.read_parquet(manifest["data_path"])


def as_of_scd(
    history: pd.DataFrame,
    as_of_ts: str | date | datetime,
    key_columns: Iterable[str] = ("id",),
    start_col: str = SCD_START_COL,
    end_col: str = SCD_END_COL,
) -> pd.DataFrame:
    """Return SCD records active at a given timestamp."""
    if history.empty:
        return history.copy()

    as_of = normalize_as_of_ts(as_of_ts)
    result = history.copy()
    start_values = result[start_col].astype(str)
    end_values = result[end_col].fillna(DEFAULT_HIGH_DATE).astype(str)
    active = (start_values <= as_of) & (end_values > as_of)
    result = result[active].copy()

    key_columns = list(key_columns)
    if key_columns and not result.empty:
        result = result.sort_values(start_col, ascending=False).drop_duplicates(subset=key_columns, keep="first")

    return result.reset_index(drop=True)


def compare_scd_as_of(
    history: pd.DataFrame,
    from_ts: str | date | datetime,
    to_ts: str | date | datetime,
    key_columns: Iterable[str] = ("id",),
    hash_col: str = SCD_HASH_COL,
) -> dict[str, Any]:
    """Compare two SCD as-of views and return changed/new/removed key counts."""
    key_columns = list(key_columns)
    before = as_of_scd(history, from_ts, key_columns=key_columns)
    after = as_of_scd(history, to_ts, key_columns=key_columns)

    if not key_columns:
        raise ValueError("key_columns must not be empty")

    before_map = {
        tuple(row[column] for column in key_columns): row.get(hash_col)
        for row in before.to_dict(orient="records")
    }
    after_map = {
        tuple(row[column] for column in key_columns): row.get(hash_col)
        for row in after.to_dict(orient="records")
    }

    before_keys = set(before_map)
    after_keys = set(after_map)
    new_keys = sorted(after_keys - before_keys)
    removed_keys = sorted(before_keys - after_keys)
    changed_keys = sorted(
        key for key in before_keys & after_keys if before_map[key] != after_map[key]
    )

    return {
        "from_ts": normalize_as_of_ts(from_ts),
        "to_ts": normalize_as_of_ts(to_ts),
        "before_count": len(before),
        "after_count": len(after),
        "new_count": len(new_keys),
        "removed_count": len(removed_keys),
        "changed_count": len(changed_keys),
        "new_keys": new_keys,
        "removed_keys": removed_keys,
        "changed_keys": changed_keys,
    }
