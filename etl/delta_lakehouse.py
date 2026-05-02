"""
PySpark + Delta Lake table models and time-travel helpers.

This is the canonical DE/lakehouse layer for the project. Pandas helpers are
only local fallbacks; production batch semantics belong here: Spark schemas,
Delta table paths, Delta history, and version/timestamp time travel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, concat_ws, coalesce, current_timestamp, desc, lit, row_number, sha2, when
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window

from etl.config import paths


@dataclass(frozen=True)
class DeltaTableModel:
    """Spark/Delta data model definition."""

    name: str
    layer: str
    path: str
    schema: StructType
    primary_key: tuple[str, ...]
    partition_columns: tuple[str, ...]
    description: str

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.schema.fields if not field.nullable)


def _path_join(base_path: Path | str, *parts: str) -> str:
    if isinstance(base_path, Path):
        return str(base_path.joinpath(*parts))

    path = base_path.rstrip("/")
    for part in parts:
        path += f"/{part.strip('/')}"
    return path


def _normalize_timestamp(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return f"{value.isoformat()} 23:59:59"
    return str(value)


BRONZE_MOVIES_SCHEMA = StructType(
    [
        StructField("id", LongType(), False),
        StructField("title", StringType(), True),
        StructField("overview", StringType(), True),
        StructField("genres", StringType(), True),
        StructField("vote_average", DoubleType(), True),
        StructField("vote_count", DoubleType(), True),
        StructField("popularity", DoubleType(), True),
        StructField("release_date", StringType(), True),
        StructField("poster_path", StringType(), True),
        StructField("adult", BooleanType(), True),
        StructField("run_date", StringType(), False),
        StructField("run_id", StringType(), False),
        StructField("ingestion_ts", TimestampType(), False),
    ]
)

SILVER_MOVIES_SCHEMA = StructType(
    [
        StructField("id", LongType(), False),
        StructField("title", StringType(), False),
        StructField("overview", StringType(), False),
        StructField("genres", StringType(), True),
        StructField("vote_average", DoubleType(), True),
        StructField("vote_count", DoubleType(), True),
        StructField("popularity", DoubleType(), True),
        StructField("release_date", StringType(), True),
        StructField("poster_path", StringType(), True),
        StructField("release_year", StringType(), True),
        StructField("tags", StringType(), False),
        StructField("metadata_completeness", DoubleType(), True),
        StructField("content_quality_score", DoubleType(), True),
        StructField("quality_bucket", StringType(), True),
        StructField("searchable", BooleanType(), True),
        StructField("recommendable", BooleanType(), True),
        StructField("is_adult_content", BooleanType(), True),
        StructField("public_demo_eligible", BooleanType(), True),
        StructField("run_date", StringType(), False),
        StructField("run_id", StringType(), False),
        StructField("ingestion_ts", TimestampType(), False),
    ]
)

GOLD_MOVIES_FEATURES_SCHEMA = StructType(
    [
        StructField("id", LongType(), False),
        StructField("title", StringType(), False),
        StructField("overview", StringType(), False),
        StructField("genres", StringType(), True),
        StructField("tags", StringType(), False),
        StructField("vector", ArrayType(FloatType()), True),
        StructField("popularity_score", DoubleType(), True),
        StructField("quality_score", DoubleType(), True),
        StructField("engagement_score", DoubleType(), True),
        StructField("metadata_completeness", DoubleType(), True),
        StructField("content_quality_score", DoubleType(), True),
        StructField("quality_bucket", StringType(), True),
        StructField("searchable", BooleanType(), True),
        StructField("recommendable", BooleanType(), True),
        StructField("is_popular", IntegerType(), True),
        StructField("is_high_rated", IntegerType(), True),
        StructField("is_recent", IntegerType(), True),
        StructField("run_date", StringType(), False),
        StructField("run_id", StringType(), False),
        StructField("ingestion_ts", TimestampType(), False),
    ]
)

DIM_MOVIE_SCD_SCHEMA = StructType(
    [
        StructField("id", LongType(), False),
        StructField("title", StringType(), False),
        StructField("overview", StringType(), False),
        StructField("genres", StringType(), True),
        StructField("vote_average", DoubleType(), True),
        StructField("vote_count", DoubleType(), True),
        StructField("popularity", DoubleType(), True),
        StructField("release_date", StringType(), True),
        StructField("poster_path", StringType(), True),
        StructField("director", StringType(), True),
        StructField("cast", StringType(), True),
        StructField("original_language", StringType(), True),
        StructField("record_hash", StringType(), False),
        StructField("effective_start_at", StringType(), False),
        StructField("effective_end_at", StringType(), False),
        StructField("is_current", BooleanType(), False),
    ]
)

FACT_MOVIE_EVENT_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), False),
        StructField("event_ts", TimestampType(), False),
        StructField("event_type", StringType(), False),
        StructField("movie_id", LongType(), True),
        StructField("user_id", StringType(), True),
        StructField("rating", FloatType(), True),
        StructField("query_text", StringType(), True),
        StructField("event_date", StringType(), False),
    ]
)

MOVIE_EMBEDDING_JOBS_SCHEMA = StructType(
    [
        StructField("job_id", StringType(), False),
        StructField("movie_id", LongType(), False),
        StructField("title", StringType(), True),
        StructField("tags", StringType(), False),
        StructField("tags_hash", StringType(), False),
        StructField("change_type", StringType(), False),
        StructField("source_run_date", StringType(), False),
        StructField("source_run_id", StringType(), False),
        StructField("model_name", StringType(), False),
        StructField("job_status", StringType(), False),
        StructField("created_at", TimestampType(), False),
        StructField("completed_at", TimestampType(), True),
        StructField("error_message", StringType(), True),
    ]
)

PIPELINE_RUN_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), False),
        StructField("run_date", StringType(), False),
        StructField("pipeline_name", StringType(), False),
        StructField("status", StringType(), False),
        StructField("started_at", TimestampType(), True),
        StructField("finished_at", TimestampType(), True),
        StructField("input_rows", LongType(), True),
        StructField("output_rows", LongType(), True),
        StructField("embedding_jobs", LongType(), True),
        StructField("error_message", StringType(), True),
    ]
)

QUARANTINE_MOVIES_SCHEMA = StructType(
    [
        StructField("id", LongType(), True),
        StructField("title", StringType(), True),
        StructField("overview", StringType(), True),
        StructField("failure_reason", StringType(), False),
        StructField("run_date", StringType(), False),
        StructField("run_id", StringType(), False),
        StructField("quarantined_at", TimestampType(), False),
    ]
)

TENANT_CATALOG_SCHEMA = StructType(
    [
        StructField("tenant_id", StringType(), False),
        StructField("catalog_id", StringType(), False),
        StructField("catalog_name", StringType(), False),
        StructField("industry", StringType(), True),
        StructField("default_language", StringType(), True),
        StructField("status", StringType(), False),
        StructField("created_at", TimestampType(), False),
        StructField("updated_at", TimestampType(), False),
    ]
)

CONTENT_ITEMS_SCHEMA = StructType(
    [
        StructField("tenant_id", StringType(), False),
        StructField("catalog_id", StringType(), False),
        StructField("content_id", StringType(), False),
        StructField("source_system", StringType(), False),
        StructField("source_content_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("description", StringType(), True),
        StructField("content_type", StringType(), False),
        StructField("genres", StringType(), True),
        StructField("people", StringType(), True),
        StructField("language", StringType(), True),
        StructField("release_date", StringType(), True),
        StructField("rating", DoubleType(), True),
        StructField("popularity", DoubleType(), True),
        StructField("tags", StringType(), False),
        StructField("run_date", StringType(), False),
        StructField("run_id", StringType(), False),
        StructField("ingestion_ts", TimestampType(), False),
    ]
)

CONTENT_FEATURES_SCHEMA = StructType(
    [
        StructField("tenant_id", StringType(), False),
        StructField("catalog_id", StringType(), False),
        StructField("content_id", StringType(), False),
        StructField("source_content_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("content_type", StringType(), False),
        StructField("tags", StringType(), False),
        StructField("vector", ArrayType(FloatType()), True),
        StructField("popularity_score", DoubleType(), True),
        StructField("quality_score", DoubleType(), True),
        StructField("engagement_score", DoubleType(), True),
        StructField("run_date", StringType(), False),
        StructField("run_id", StringType(), False),
        StructField("ingestion_ts", TimestampType(), False),
    ]
)

DIM_CONTENT_SCD_SCHEMA = StructType(
    [
        StructField("tenant_id", StringType(), False),
        StructField("catalog_id", StringType(), False),
        StructField("content_id", StringType(), False),
        StructField("source_content_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("description", StringType(), True),
        StructField("content_type", StringType(), False),
        StructField("genres", StringType(), True),
        StructField("people", StringType(), True),
        StructField("language", StringType(), True),
        StructField("release_date", StringType(), True),
        StructField("record_hash", StringType(), False),
        StructField("effective_start_at", StringType(), False),
        StructField("effective_end_at", StringType(), False),
        StructField("is_current", BooleanType(), False),
    ]
)

CONTENT_EVENT_SCHEMA = StructType(
    [
        StructField("tenant_id", StringType(), False),
        StructField("catalog_id", StringType(), False),
        StructField("event_id", StringType(), False),
        StructField("event_ts", TimestampType(), False),
        StructField("event_type", StringType(), False),
        StructField("content_id", StringType(), True),
        StructField("source_content_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("session_id", StringType(), True),
        StructField("request_id", StringType(), True),
        StructField("rating", FloatType(), True),
        StructField("query_text", StringType(), True),
        StructField("source", StringType(), True),
        StructField("event_date", StringType(), False),
    ]
)

CONTENT_BEHAVIOR_DAILY_SCHEMA = StructType(
    [
        StructField("tenant_id", StringType(), False),
        StructField("catalog_id", StringType(), False),
        StructField("content_id", StringType(), False),
        StructField("event_date", StringType(), False),
        StructField("views", LongType(), False),
        StructField("clicks", LongType(), False),
        StructField("impressions", LongType(), False),
        StructField("ratings", LongType(), False),
        StructField("avg_rating", DoubleType(), True),
        StructField("behavior_score", DoubleType(), False),
        StructField("run_id", StringType(), False),
        StructField("ingestion_ts", TimestampType(), False),
    ]
)

DATA_QUALITY_OBSERVATION_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), False),
        StructField("run_date", StringType(), False),
        StructField("table_name", StringType(), False),
        StructField("metric_name", StringType(), False),
        StructField("metric_value", DoubleType(), False),
        StructField("threshold_value", DoubleType(), True),
        StructField("status", StringType(), False),
        StructField("observed_at", TimestampType(), False),
    ]
)

DELTA_TABLES: dict[str, DeltaTableModel] = {
    "bronze.movies": DeltaTableModel(
        name="bronze.movies",
        layer="bronze",
        path=_path_join(paths.bronze_data, "movies"),
        schema=BRONZE_MOVIES_SCHEMA,
        primary_key=("id", "run_id"),
        partition_columns=("run_date",),
        description="Raw TMDB movie snapshot landed once per batch run.",
    ),
    "silver.movies": DeltaTableModel(
        name="silver.movies",
        layer="silver",
        path=_path_join(paths.silver_data, "movies"),
        schema=SILVER_MOVIES_SCHEMA,
        primary_key=("id", "run_id"),
        partition_columns=("run_date",),
        description="Validated and deduplicated movie snapshot.",
    ),
    "gold.movies_features": DeltaTableModel(
        name="gold.movies_features",
        layer="gold",
        path=_path_join(paths.gold_data, "movies_features"),
        schema=GOLD_MOVIES_FEATURES_SCHEMA,
        primary_key=("id", "run_id"),
        partition_columns=("run_date",),
        description="ML-ready feature table for recommendation artifact generation.",
    ),
    "gold.dim_movie_scd": DeltaTableModel(
        name="gold.dim_movie_scd",
        layer="gold",
        path=_path_join(paths.gold_data, "dim_movie_scd"),
        schema=DIM_MOVIE_SCD_SCHEMA,
        primary_key=("id", "effective_start_at"),
        partition_columns=("is_current",),
        description="SCD Type 2 movie dimension with historical attribute versions.",
    ),
    "gold.fact_movie_event": DeltaTableModel(
        name="gold.fact_movie_event",
        layer="gold",
        path=_path_join(paths.gold_data, "fact_movie_event"),
        schema=FACT_MOVIE_EVENT_SCHEMA,
        primary_key=("event_id",),
        partition_columns=("event_date",),
        description="Application behavior events for analytics and personalization features.",
    ),
    "gold.movie_embedding_jobs": DeltaTableModel(
        name="gold.movie_embedding_jobs",
        layer="gold",
        path=_path_join(paths.gold_data, "movie_embedding_jobs"),
        schema=MOVIE_EMBEDDING_JOBS_SCHEMA,
        primary_key=("job_id",),
        partition_columns=("source_run_date",),
        description="Incremental embedding work queue for new or changed movie tags.",
    ),
    "gold.pipeline_run": DeltaTableModel(
        name="gold.pipeline_run",
        layer="gold",
        path=_path_join(paths.gold_data, "pipeline_run"),
        schema=PIPELINE_RUN_SCHEMA,
        primary_key=("run_id",),
        partition_columns=("run_date",),
        description="Batch observability table for pipeline status and row-count metrics.",
    ),
    "silver.movies_quarantine": DeltaTableModel(
        name="silver.movies_quarantine",
        layer="silver",
        path=_path_join(paths.silver_data, "movies_quarantine"),
        schema=QUARANTINE_MOVIES_SCHEMA,
        primary_key=("run_id", "id", "failure_reason"),
        partition_columns=("run_date",),
        description="Rows rejected by batch quality gates with failure reasons.",
    ),
    "gold.tenant_catalog": DeltaTableModel(
        name="gold.tenant_catalog",
        layer="gold",
        path=_path_join(paths.gold_data, "tenant_catalog"),
        schema=TENANT_CATALOG_SCHEMA,
        primary_key=("tenant_id", "catalog_id"),
        partition_columns=("tenant_id",),
        description="Registered customer catalogs served by the recommendation platform.",
    ),
    "bronze.content_items": DeltaTableModel(
        name="bronze.content_items",
        layer="bronze",
        path=_path_join(paths.bronze_data, "content_items"),
        schema=CONTENT_ITEMS_SCHEMA,
        primary_key=("tenant_id", "catalog_id", "content_id", "run_id"),
        partition_columns=("tenant_id", "catalog_id", "run_date"),
        description="Raw catalog items normalized into the platform content contract.",
    ),
    "silver.content_items": DeltaTableModel(
        name="silver.content_items",
        layer="silver",
        path=_path_join(paths.silver_data, "content_items"),
        schema=CONTENT_ITEMS_SCHEMA,
        primary_key=("tenant_id", "catalog_id", "content_id", "run_id"),
        partition_columns=("tenant_id", "catalog_id", "run_date"),
        description="Validated, deduplicated customer content catalog records.",
    ),
    "gold.content_features": DeltaTableModel(
        name="gold.content_features",
        layer="gold",
        path=_path_join(paths.gold_data, "content_features"),
        schema=CONTENT_FEATURES_SCHEMA,
        primary_key=("tenant_id", "catalog_id", "content_id", "run_id"),
        partition_columns=("tenant_id", "catalog_id", "run_date"),
        description="Tenant-aware semantic and ranking features for serving.",
    ),
    "gold.dim_content_scd": DeltaTableModel(
        name="gold.dim_content_scd",
        layer="gold",
        path=_path_join(paths.gold_data, "dim_content_scd"),
        schema=DIM_CONTENT_SCD_SCHEMA,
        primary_key=("tenant_id", "catalog_id", "content_id", "effective_start_at"),
        partition_columns=("tenant_id", "catalog_id", "is_current"),
        description="Tenant-aware SCD Type 2 content dimension for customer catalog history.",
    ),
    "gold.fact_content_event": DeltaTableModel(
        name="gold.fact_content_event",
        layer="gold",
        path=_path_join(paths.gold_data, "fact_content_event"),
        schema=CONTENT_EVENT_SCHEMA,
        primary_key=("tenant_id", "catalog_id", "event_id"),
        partition_columns=("tenant_id", "catalog_id", "event_date"),
        description="Product behavior events for personalization and customer analytics.",
    ),
    "gold.content_behavior_daily": DeltaTableModel(
        name="gold.content_behavior_daily",
        layer="gold",
        path=_path_join(paths.gold_data, "content_behavior_daily"),
        schema=CONTENT_BEHAVIOR_DAILY_SCHEMA,
        primary_key=("tenant_id", "catalog_id", "content_id", "event_date"),
        partition_columns=("tenant_id", "catalog_id", "event_date"),
        description="Daily behavior aggregates that can be joined into ranking features.",
    ),
    "gold.data_quality_observation": DeltaTableModel(
        name="gold.data_quality_observation",
        layer="gold",
        path=_path_join(paths.gold_data, "data_quality_observation"),
        schema=DATA_QUALITY_OBSERVATION_SCHEMA,
        primary_key=("run_id", "table_name", "metric_name"),
        partition_columns=("run_date",),
        description="Data quality metrics for audits, SLAs, and customer-facing reliability.",
    ),
}


def get_delta_table(table_name: str) -> DeltaTableModel:
    """Resolve a canonical Delta table model by name."""
    try:
        return DELTA_TABLES[table_name]
    except KeyError as exc:
        known = ", ".join(sorted(DELTA_TABLES))
        raise ValueError(f"Unknown Delta table '{table_name}'. Known tables: {known}") from exc


def require_delta_table() -> Any:
    """Import DeltaTable with a clear error message for ETL-only dependencies."""
    try:
        from delta.tables import DeltaTable
    except ImportError as exc:
        raise RuntimeError(
            "Delta Lake operations require delta-spark. Install requirements-etl.txt "
            "or run on a Spark cluster with Delta Lake jars configured."
        ) from exc
    return DeltaTable


def add_batch_metadata(df: DataFrame, run_date: str, run_id: str) -> DataFrame:
    """Add standard batch lineage columns before writing a Delta table."""
    return df.withColumn("run_date", lit(run_date)) \
        .withColumn("run_id", lit(run_id)) \
        .withColumn("ingestion_ts", current_timestamp())


def _string_column_or_default(df: DataFrame, column_name: str, default: str | None = None):
    """Return a string column expression, or a typed literal when the source column is absent."""
    if column_name in df.columns:
        return coalesce(col(column_name).cast("string"), lit(default))
    return lit(default).cast("string")


def _double_column_or_null(df: DataFrame, column_name: str):
    """Return a double column expression when available, otherwise NULL."""
    if column_name in df.columns:
        return col(column_name).cast("double")
    return lit(None).cast("double")


def movie_snapshot_to_content_items(
    df: DataFrame,
    tenant_id: str,
    catalog_id: str,
    source_system: str,
    content_type: str = "movie",
) -> DataFrame:
    """
    Convert TMDB movie rows into the tenant-aware platform content contract.

    This keeps the public product model generic enough for OTT, e-learning,
    publisher, and marketplace catalogs while still supporting the current movie
    dataset as the first vertical.
    """
    source_content_id = _string_column_or_default(df, "id", "unknown")
    title = _string_column_or_default(df, "title", "Untitled")
    description = _string_column_or_default(df, "overview", "")
    genres = _string_column_or_default(df, "genres")
    people = concat_ws(
        ", ",
        _string_column_or_default(df, "director"),
        _string_column_or_default(df, "cast"),
    )
    language = _string_column_or_default(df, "original_language")
    release_date = _string_column_or_default(df, "release_date")
    tags = (
        _string_column_or_default(df, "tags", "")
        if "tags" in df.columns
        else concat_ws(". ", title, description, genres, people, lit(content_type))
    )

    return df.select(
        lit(tenant_id).alias("tenant_id"),
        lit(catalog_id).alias("catalog_id"),
        sha2(concat_ws("||", lit(tenant_id), lit(catalog_id), source_content_id), 256).alias("content_id"),
        lit(source_system).alias("source_system"),
        source_content_id.alias("source_content_id"),
        title.alias("title"),
        description.alias("description"),
        lit(content_type).alias("content_type"),
        genres.alias("genres"),
        people.alias("people"),
        language.alias("language"),
        release_date.alias("release_date"),
        _double_column_or_null(df, "vote_average").alias("rating"),
        _double_column_or_null(df, "popularity").alias("popularity"),
        tags.alias("tags"),
        "run_date",
        "run_id",
        "ingestion_ts",
    )


def movie_features_to_content_features(
    df: DataFrame,
    tenant_id: str,
    catalog_id: str,
    source_system: str,
    content_type: str = "movie",
) -> DataFrame:
    """Project movie feature rows into the product-level content feature table."""
    source_content_id = _string_column_or_default(df, "id", "unknown")
    return df.select(
        lit(tenant_id).alias("tenant_id"),
        lit(catalog_id).alias("catalog_id"),
        sha2(concat_ws("||", lit(tenant_id), lit(catalog_id), source_content_id), 256).alias("content_id"),
        source_content_id.alias("source_content_id"),
        _string_column_or_default(df, "title", "Untitled").alias("title"),
        lit(content_type).alias("content_type"),
        _string_column_or_default(df, "tags", "").alias("tags"),
        col("vector"),
        _double_column_or_null(df, "popularity_score").alias("popularity_score"),
        _double_column_or_null(df, "quality_score").alias("quality_score"),
        _double_column_or_null(df, "engagement_score").alias("engagement_score"),
        "run_date",
        "run_id",
        "ingestion_ts",
    )


def write_pipeline_run(
    spark: SparkSession,
    run_id: str,
    run_date: str,
    pipeline_name: str,
    status: str,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    input_rows: int | None = None,
    output_rows: int | None = None,
    embedding_jobs: int | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Append one audit row for a batch pipeline run."""
    now = datetime.now(UTC)
    row = [
        {
            "run_id": run_id,
            "run_date": run_date,
            "pipeline_name": pipeline_name,
            "status": status,
            "started_at": started_at or now,
            "finished_at": finished_at,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "embedding_jobs": embedding_jobs,
            "error_message": error_message,
        }
    ]
    df = spark.createDataFrame(row, schema=PIPELINE_RUN_SCHEMA)
    return write_delta_table(df, "gold.pipeline_run", mode="append", validate_contract=True)


def write_data_quality_observation(
    spark: SparkSession,
    run_id: str,
    run_date: str,
    table_name: str,
    metric_name: str,
    metric_value: float,
    status: str,
    threshold_value: float | None = None,
) -> dict[str, Any]:
    """Append one machine-readable data quality observation."""
    row = [
        {
            "run_id": run_id,
            "run_date": run_date,
            "table_name": table_name,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "threshold_value": threshold_value,
            "status": status,
            "observed_at": datetime.now(UTC),
        }
    ]
    df = spark.createDataFrame(row, schema=DATA_QUALITY_OBSERVATION_SCHEMA)
    return write_delta_table(df, "gold.data_quality_observation", mode="append", validate_contract=True)


def validate_delta_contract(df: DataFrame, table: DeltaTableModel | str) -> dict[str, Any]:
    """Validate required columns and partition columns before Delta writes."""
    table = get_delta_table(table) if isinstance(table, str) else table
    missing_required = [column for column in table.required_columns if column not in df.columns]
    missing_partitions = [column for column in table.partition_columns if column not in df.columns]

    if missing_required:
        raise ValueError(f"{table.name} missing required columns: {missing_required}")
    if missing_partitions:
        raise ValueError(f"{table.name} missing partition columns: {missing_partitions}")

    return {
        "table": table.name,
        "path": table.path,
        "required_columns": list(table.required_columns),
        "partition_columns": list(table.partition_columns),
        "input_columns": list(df.columns),
    }


def write_delta_table(
    df: DataFrame,
    table: DeltaTableModel | str,
    mode: str = "append",
    merge_schema: bool = True,
    validate_contract: bool = True,
    enable_change_data_feed: bool = True,
    replace_where: str | None = None,
) -> dict[str, Any]:
    """Write a DataFrame to a canonical Delta table."""
    table = get_delta_table(table) if isinstance(table, str) else table
    contract = validate_delta_contract(df, table) if validate_contract else {}

    writer = df.write.format("delta").mode(mode)
    if replace_where:
        writer = writer.option("replaceWhere", replace_where)
    if enable_change_data_feed:
        writer = writer.option("delta.enableChangeDataFeed", "true")
    if merge_schema:
        writer = writer.option("mergeSchema", "true")
    if table.partition_columns:
        writer = writer.partitionBy(*table.partition_columns)
    writer.save(table.path)

    return {
        "table": table.name,
        "path": table.path,
        "format": "delta",
        "mode": mode,
        "replace_where": replace_where,
        "change_data_feed": enable_change_data_feed,
        "contract": contract,
    }


def read_delta_table(
    spark: SparkSession,
    table: DeltaTableModel | str,
    version_as_of: int | None = None,
    timestamp_as_of: str | date | datetime | None = None,
) -> DataFrame:
    """Read a Delta table, optionally using Delta Lake time travel."""
    table = get_delta_table(table) if isinstance(table, str) else table
    if version_as_of is not None and timestamp_as_of is not None:
        raise ValueError("Use either version_as_of or timestamp_as_of, not both")

    reader = spark.read.format("delta")
    if version_as_of is not None:
        reader = reader.option("versionAsOf", int(version_as_of))
    if timestamp_as_of is not None:
        reader = reader.option("timestampAsOf", _normalize_timestamp(timestamp_as_of))

    return reader.load(table.path)


def read_delta_changes(
    spark: SparkSession,
    table: DeltaTableModel | str,
    starting_version: int | None = None,
    ending_version: int | None = None,
    starting_timestamp: str | date | datetime | None = None,
    ending_timestamp: str | date | datetime | None = None,
) -> DataFrame:
    """Read Delta Change Data Feed for downstream incremental processing."""
    table = get_delta_table(table) if isinstance(table, str) else table
    if starting_version is not None and starting_timestamp is not None:
        raise ValueError("Use either starting_version or starting_timestamp, not both")
    if ending_version is not None and ending_timestamp is not None:
        raise ValueError("Use either ending_version or ending_timestamp, not both")

    reader = spark.read.format("delta").option("readChangeFeed", "true")
    if starting_version is not None:
        reader = reader.option("startingVersion", int(starting_version))
    if ending_version is not None:
        reader = reader.option("endingVersion", int(ending_version))
    if starting_timestamp is not None:
        reader = reader.option("startingTimestamp", _normalize_timestamp(starting_timestamp))
    if ending_timestamp is not None:
        reader = reader.option("endingTimestamp", _normalize_timestamp(ending_timestamp))

    return reader.load(table.path)


def delta_table_exists(spark: SparkSession, table: DeltaTableModel | str) -> bool:
    """Return whether a path is currently a Delta table."""
    table = get_delta_table(table) if isinstance(table, str) else table
    DeltaTable = require_delta_table()
    try:
        return bool(DeltaTable.isDeltaTable(spark, table.path))
    except Exception:
        return False


def delta_table_history(spark: SparkSession, table: DeltaTableModel | str, limit: int = 20) -> DataFrame:
    """Return Delta transaction history for audit/debug/time-travel discovery."""
    table = get_delta_table(table) if isinstance(table, str) else table
    DeltaTable = require_delta_table()
    return DeltaTable.forPath(spark, table.path).history(limit)


def restore_delta_table_to_version(
    spark: SparkSession,
    table: DeltaTableModel | str,
    version: int,
) -> DataFrame:
    """Restore a Delta table to a previous transaction version."""
    table = get_delta_table(table) if isinstance(table, str) else table
    DeltaTable = require_delta_table()
    return DeltaTable.forPath(spark, table.path).restoreToVersion(int(version))


def vacuum_delta_table(
    spark: SparkSession,
    table: DeltaTableModel | str,
    retention_hours: int = 168,
) -> DataFrame:
    """Vacuum old Delta files after retention. Keep retention high to preserve time travel."""
    table = get_delta_table(table) if isinstance(table, str) else table
    return spark.sql(f"VACUUM delta.`{table.path}` RETAIN {int(retention_hours)} HOURS")


def optimize_delta_table(spark: SparkSession, table: DeltaTableModel | str) -> DataFrame:
    """Run Delta OPTIMIZE when available on the target Spark runtime."""
    table = get_delta_table(table) if isinstance(table, str) else table
    return spark.sql(f"OPTIMIZE delta.`{table.path}`")


def add_tags_hash(df: DataFrame, tags_col: str = "tags", output_col: str = "tags_hash") -> DataFrame:
    """Hash recommendation tags so embeddings can be incrementally refreshed."""
    return df.withColumn(output_col, sha2(coalesce(col(tags_col).cast("string"), lit("<NULL>")), 256))


def build_embedding_jobs(
    current_features: DataFrame,
    previous_features: DataFrame | None,
    run_date: str,
    run_id: str,
    model_name: str,
) -> DataFrame:
    """
    Compare current movie features against previous features and emit embedding jobs.

    A job is created only when a movie is new or its `tags` changed. This is the
    AI-era cost/performance win: daily batch does not need to re-embed unchanged
    movies.
    """
    current = add_tags_hash(
        current_features.select("id", "title", "tags"),
    ).select(
        col("id").alias("movie_id"),
        "title",
        "tags",
        "tags_hash",
    )

    if previous_features is None:
        changed = current.withColumn("change_type", lit("new"))
    else:
        previous_features = latest_rows_by_key(previous_features, key_columns=("id",))
        previous = add_tags_hash(
            previous_features.select("id", "tags"),
        ).select(
            col("id").alias("movie_id"),
            col("tags_hash").alias("previous_tags_hash"),
        )
        joined = current.join(previous, "movie_id", "left")
        changed = joined.filter(
            col("previous_tags_hash").isNull() | (col("tags_hash") != col("previous_tags_hash"))
        ).withColumn(
            "change_type",
            when(col("previous_tags_hash").isNull(), lit("new")).otherwise(lit("changed")),
        ).drop("previous_tags_hash")

    return changed.withColumn(
        "job_id",
        sha2(concat_ws("||", col("movie_id").cast("string"), col("tags_hash"), lit(run_id), lit(model_name)), 256),
    ).withColumn("source_run_date", lit(run_date)) \
        .withColumn("source_run_id", lit(run_id)) \
        .withColumn("model_name", lit(model_name)) \
        .withColumn("job_status", lit("pending")) \
        .withColumn("created_at", current_timestamp()) \
        .withColumn("completed_at", lit(None).cast(TimestampType())) \
        .withColumn("error_message", lit(None).cast(StringType())) \
        .select(
            "job_id",
            "movie_id",
            "title",
            "tags",
            "tags_hash",
            "change_type",
            "source_run_date",
            "source_run_id",
            "model_name",
            "job_status",
            "created_at",
            "completed_at",
            "error_message",
        )


def load_previous_features_for_incremental(
    spark: SparkSession,
    table: DeltaTableModel | str = "gold.movies_features",
    previous_version: int | None = None,
    previous_timestamp: str | date | datetime | None = None,
) -> DataFrame | None:
    """Load the previous feature table view used for incremental embedding comparison."""
    table = get_delta_table(table) if isinstance(table, str) else table
    if previous_version is None and previous_timestamp is None:
        previous_version = latest_delta_version(spark, table)
    if previous_version is not None and previous_version < 0:
        return None
    if previous_version is None and previous_timestamp is None:
        return None

    try:
        return read_delta_table(
            spark,
            table,
            version_as_of=previous_version,
            timestamp_as_of=previous_timestamp,
        )
    except Exception:
        return None


def write_embedding_jobs(
    jobs_df: DataFrame,
    mode: str = "append",
    replace_where: str | None = None,
) -> dict[str, Any]:
    """Persist embedding jobs to the canonical Delta work queue."""
    return write_delta_table(
        jobs_df,
        "gold.movie_embedding_jobs",
        mode=mode,
        validate_contract=True,
        enable_change_data_feed=True,
        replace_where=replace_where,
    )


def latest_delta_version(spark: SparkSession, table: DeltaTableModel | str) -> int | None:
    """Return the latest Delta transaction version when the table exists."""
    try:
        history = delta_table_history(spark, table, limit=1)
        rows = history.select("version").collect()
    except Exception:
        return None
    if not rows:
        return None
    return int(rows[0]["version"])


def latest_rows_by_key(df: DataFrame, key_columns: tuple[str, ...] = ("id",)) -> DataFrame:
    """Return the latest row per key from an append-only Delta snapshot table."""
    order_columns = [
        desc(column)
        for column in ("run_date", "run_id", "ingestion_ts")
        if column in df.columns
    ]
    if not order_columns:
        return df.dropDuplicates(list(key_columns))

    window = Window.partitionBy(*[col(column) for column in key_columns]).orderBy(*order_columns)
    return df.withColumn("_latest_row_number", row_number().over(window)) \
        .filter(col("_latest_row_number") == 1) \
        .drop("_latest_row_number")
