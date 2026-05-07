"""
PySpark ETL - processes TMDB movie data using Spark.

Canonical batch/lakehouse pipeline for the project.
"""
import ast
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat_ws, coalesce, current_timestamp, desc, expr, greatest, length, lit, row_number, sha2, udf, when
from pyspark.sql.window import Window

from etl.config import paths
from etl.delta_lakehouse import (
    add_batch_metadata,
    build_embedding_jobs,
    get_delta_table,
    latest_rows_by_key,
    load_previous_features_for_incremental,
    movie_features_to_content_features,
    movie_snapshot_to_content_items,
    write_delta_table,
    write_data_quality_observation,
    write_embedding_jobs,
    write_pipeline_run,
)
from etl.semantic_artifacts import write_semantic_artifacts

logger = logging.getLogger(__name__)

SCD_START_COL = "effective_start_at"
SCD_END_COL = "effective_end_at"
SCD_CURRENT_COL = "is_current"
SCD_HASH_COL = "record_hash"
SCD_HIGH_DATE = "9999-12-31T00:00:00Z"

MOVIE_KEY_COLUMNS = ("id",)
MOVIE_SCD_TRACKED_COLUMNS = (
    "title",
    "overview",
    "genres",
    "vote_average",
    "vote_count",
    "popularity",
    "release_date",
    "poster_path",
    "director",
    "cast",
    "original_language",
)
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_TENANT_ID = "demo-media-co"
DEFAULT_CATALOG_ID = "tmdb-movies"
DEFAULT_SOURCE_SYSTEM = "tmdb_kaggle"


def _path_join(base_path: Path | str, *parts: str) -> str:
    """Join local/cloud paths without corrupting URI-style paths."""
    if isinstance(base_path, Path):
        return str(base_path.joinpath(*parts))
    path = base_path.rstrip("/")
    for part in parts:
        path += f"/{part.strip('/')}"
    return path


def _path_exists(path: Path | str) -> bool:
    """Best-effort local path existence check."""
    if isinstance(path, Path):
        return path.exists()
    if path.startswith(("s3://", "gs://", "abfs://")):
        return False
    return Path(path).exists()


def create_spark_session(
    app_name: str = "MovieETL",
    master: str = "local[*]",
    enable_delta: bool = True,
):
    """Create a local Spark session with AQE enabled and Delta Lake support."""
    # MEMORY SAFETY for Machine Learning (SBERT runs off-heap)
    # Prevent Arrow batches from exploding memory during UDF transfer
    builder = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.python.worker.memory", "2g") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")

    if enable_delta:
        builder = builder \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        try:
            from delta import configure_spark_with_delta_pip

            builder = configure_spark_with_delta_pip(builder)
        except ImportError:
            logger.warning(
                "delta-spark is not installed. Delta writes require the package or cluster-level Delta jars."
            )

    return builder.getOrCreate()


def dedupe_latest_movies(df: DataFrame, key_columns: tuple[str, ...] = MOVIE_KEY_COLUMNS) -> DataFrame:
    """Keep one deterministic row per movie key before SCD comparison."""
    order_columns = [
        expr(f"try_cast({column} as double) desc nulls last")
        for column in ("vote_count", "popularity")
        if column in df.columns
    ]
    if not order_columns:
        order_columns = [desc(key_columns[0])]

    window = Window.partitionBy(*[col(column) for column in key_columns]).orderBy(*order_columns)
    return df.withColumn("_scd_row_number", row_number().over(window)) \
        .filter(col("_scd_row_number") == 1) \
        .drop("_scd_row_number")


def add_scd_record_hash(df: DataFrame, tracked_columns: tuple[str, ...]) -> DataFrame:
    """Hash tracked attributes so changed records can be detected cheaply."""
    hash_inputs = []
    for column in tracked_columns:
        if column in df.columns:
            hash_inputs.append(coalesce(col(column).cast("string"), lit("<NULL>")))
        else:
            hash_inputs.append(lit("<MISSING>"))
    return df.withColumn(SCD_HASH_COL, sha2(concat_ws("||", *hash_inputs), 256))


def _ensure_scd_columns(
    existing_df: DataFrame,
    tracked_columns: tuple[str, ...],
    effective_ts: str,
    high_date: str,
) -> DataFrame:
    """Backfill SCD columns for a historical table created before this code existed."""
    result = existing_df
    if SCD_HASH_COL not in result.columns:
        result = add_scd_record_hash(result, tracked_columns)
    if SCD_START_COL not in result.columns:
        result = result.withColumn(SCD_START_COL, lit(effective_ts))
    if SCD_END_COL not in result.columns:
        result = result.withColumn(SCD_END_COL, lit(high_date))
    if SCD_CURRENT_COL not in result.columns:
        result = result.withColumn(SCD_CURRENT_COL, lit(True))
    return result


def apply_spark_scd_type2(
    incoming_df: DataFrame,
    existing_df: DataFrame | None = None,
    key_columns: tuple[str, ...] = MOVIE_KEY_COLUMNS,
    tracked_columns: tuple[str, ...] = MOVIE_SCD_TRACKED_COLUMNS,
    effective_ts: str | None = None,
    high_date: str = SCD_HIGH_DATE,
) -> DataFrame:
    """Apply SCD Type 2 semantics to a latest movie snapshot."""
    effective_ts = effective_ts or datetime.now(timezone.utc).isoformat(timespec="seconds")
    missing_keys = [column for column in key_columns if column not in incoming_df.columns]
    if missing_keys:
        raise ValueError(f"Incoming movie snapshot missing key columns: {missing_keys}")

    incoming_versions = add_scd_record_hash(
        dedupe_latest_movies(incoming_df, key_columns),
        tracked_columns,
    ).withColumn(SCD_START_COL, lit(effective_ts)) \
        .withColumn(SCD_END_COL, lit(high_date)) \
        .withColumn(SCD_CURRENT_COL, lit(True))

    if existing_df is None:
        return incoming_versions

    existing_df = _ensure_scd_columns(existing_df, tracked_columns, effective_ts, high_date)
    current_df = existing_df.filter(col(SCD_CURRENT_COL) == lit(True)) \
        .select(*key_columns, col(SCD_HASH_COL).alias("_existing_record_hash"))

    joined_df = incoming_versions.alias("incoming").join(current_df.alias("current"), list(key_columns), "left")
    changed_or_new_df = joined_df.filter(
        col("_existing_record_hash").isNull() | (col(SCD_HASH_COL) != col("_existing_record_hash"))
    )

    insert_columns = [col(f"incoming.{column}").alias(column) for column in incoming_versions.columns]
    inserts_df = changed_or_new_df.select(*insert_columns)

    keys_to_expire_df = joined_df.filter(
        col("_existing_record_hash").isNotNull() & (col(SCD_HASH_COL) != col("_existing_record_hash"))
    ).select(*key_columns).distinct()

    existing_marked_df = existing_df.join(
        keys_to_expire_df.withColumn("_expire_current", lit(True)),
        list(key_columns),
        "left",
    )
    should_expire = col("_expire_current").isNotNull() & (col(SCD_CURRENT_COL) == lit(True))

    updated_existing_df = existing_marked_df \
        .withColumn(SCD_CURRENT_COL, when(should_expire, lit(False)).otherwise(col(SCD_CURRENT_COL))) \
        .withColumn(SCD_END_COL, when(should_expire, lit(effective_ts)).otherwise(col(SCD_END_COL))) \
        .drop("_expire_current")

    return updated_existing_df.unionByName(inserts_df, allowMissingColumns=True)


def write_table(
    df: DataFrame,
    output_path: str,
    sink_format: str = "delta",
    mode: str = "overwrite",
    partition_columns: list[str] | None = None,
) -> str:
    """Write a Spark table using Delta in lakehouse runs or parquet as a portable fallback."""
    writer = df.write.mode(mode)
    if partition_columns:
        writer = writer.partitionBy(*partition_columns)

    if sink_format == "delta":
        writer = writer.option("delta.autoOptimize.optimizeWrite", "true") \
            .option("delta.autoOptimize.autoCompact", "true") \
            .format("delta")
    else:
        writer = writer.format(sink_format)

    writer.save(output_path)
    return output_path


def load_table_if_exists(spark: SparkSession, input_path: str, sink_format: str = "delta") -> DataFrame | None:
    """Load a table when it exists; return None for first-run SCD bootstraps."""
    if not _path_exists(input_path):
        return None
    try:
        return spark.read.format(sink_format).load(input_path)
    except Exception as exc:
        logger.warning("Could not load existing table at %s: %s", input_path, exc)
        return None


def _optional_long(df: DataFrame, column_name: str):
    if column_name in df.columns:
        return col(column_name).cast("long")
    return lit(None).cast("long")


def _optional_string(df: DataFrame, column_name: str):
    if column_name in df.columns:
        return col(column_name).cast("string")
    return lit(None).cast("string")


def parse_metadata_name_list(value) -> str:
    """Normalize Kaggle list/dict metadata strings into comma-separated names."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ", ".join(part.strip() for part in text.split(",") if part.strip())

    if isinstance(parsed, list):
        names = []
        for item in parsed:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                if name:
                    names.append(name)
            elif item:
                names.append(str(item).strip())
        return ", ".join(names)
    if isinstance(parsed, dict):
        return str(parsed.get("name") or "").strip()
    return str(parsed).strip()


def split_valid_and_quarantined_movies(
    df: DataFrame,
    run_date: str,
    run_id: str,
) -> tuple[DataFrame, DataFrame]:
    """
    Apply row-level validity gates and return usable rows plus a quarantine table.

    The pipeline still aborts for source-wide failures, but row-level failures
    are preserved for auditability instead of disappearing behind filters.
    Weak metadata is retained and scored later because long-tail catalog coverage
    matters for a content-based product.
    """
    failure_reason = lit(None).cast("string")

    if "id" in df.columns:
        failure_reason = when(col("id").isNull(), lit("missing_id")).otherwise(failure_reason)
    else:
        failure_reason = lit("missing_id")

    if "title" in df.columns:
        failure_reason = when(
            failure_reason.isNull() & (col("title").isNull() | (length(col("title")) == 0)),
            lit("missing_title"),
        ).otherwise(failure_reason)
    else:
        failure_reason = when(failure_reason.isNull(), lit("missing_title")).otherwise(failure_reason)

    checked_df = df.withColumn("_failure_reason", failure_reason)
    valid_df = checked_df.filter(col("_failure_reason").isNull()).drop("_failure_reason")
    quarantine_df = checked_df.filter(col("_failure_reason").isNotNull()).select(
        _optional_long(checked_df, "id").alias("id"),
        _optional_string(checked_df, "title").alias("title"),
        _optional_string(checked_df, "overview").alias("overview"),
        col("_failure_reason").alias("failure_reason"),
        lit(run_date).alias("run_date"),
        lit(run_id).alias("run_id"),
        current_timestamp().alias("quarantined_at"),
    )
    return valid_df, quarantine_df


def add_catalog_coverage_features(df: DataFrame) -> DataFrame:
    """Add coverage-first quality features without dropping long-tail movies."""
    result = df
    title_len = length(coalesce(col("title").cast("string"), lit("")))
    overview_len = length(coalesce(col("overview").cast("string"), lit("")))
    genres_len = length(coalesce(col("genres").cast("string"), lit("")))
    release_len = length(coalesce(col("release_date").cast("string"), lit("")))
    poster_len = length(coalesce(col("poster_path").cast("string"), lit("")))
    vote_count = expr("coalesce(try_cast(vote_count as double), 0.0)") if "vote_count" in result.columns else lit(0.0)
    vote_average = expr("coalesce(try_cast(vote_average as double), 0.0)") if "vote_average" in result.columns else lit(0.0)
    popularity = expr("coalesce(try_cast(popularity as double), 0.0)") if "popularity" in result.columns else lit(0.0)

    result = result.withColumn(
        "metadata_completeness",
        (when(title_len > 0, lit(0.20)).otherwise(lit(0.0))
         + when(overview_len >= 20, lit(0.25)).otherwise(when(overview_len > 0, lit(0.10)).otherwise(lit(0.0)))
         + when(genres_len > 0, lit(0.15)).otherwise(lit(0.0))
         + when(vote_count > 0, lit(0.15)).otherwise(lit(0.0))
         + when(popularity > 0, lit(0.10)).otherwise(lit(0.0))
         + when(release_len >= 4, lit(0.10)).otherwise(lit(0.0))
         + when(poster_len > 0, lit(0.05)).otherwise(lit(0.0))),
    )
    result = result.withColumn("vote_confidence", expr("least(1.0, log1p(coalesce(try_cast(vote_count as double), 0.0)) / 8.0)"))
    result = result.withColumn("popularity_norm", expr("least(1.0, log1p(coalesce(try_cast(popularity as double), 0.0)) / 8.0)"))
    result = result.withColumn(
        "content_quality_score",
        greatest(
            lit(0.0),
            (col("metadata_completeness") * lit(0.45))
            + ((vote_average / lit(10.0)) * col("vote_confidence") * lit(0.30))
            + (col("popularity_norm") * lit(0.25)),
        ),
    )
    result = result.withColumn(
        "quality_bucket",
        when(col("content_quality_score") >= 0.70, lit("premium"))
        .when(col("content_quality_score") >= 0.45, lit("standard"))
        .when(col("metadata_completeness") >= 0.35, lit("long_tail"))
        .otherwise(lit("thin_metadata")),
    )
    result = result.withColumn("searchable", title_len > 0)
    result = result.withColumn("recommendable", (overview_len >= 20) | (genres_len > 0) | (col("metadata_completeness") >= 0.45))
    if "adult" in result.columns:
        result = result.withColumn(
            "is_adult_content",
            expr("lower(cast(adult as string)) in ('true', '1', 'yes')"),
        )
    else:
        result = result.withColumn("is_adult_content", lit(False))
    result = result.withColumn("public_demo_eligible", ~col("is_adult_content"))
    return result


def upsert_movie_scd_dimension(
    spark: SparkSession,
    incoming_df: DataFrame,
    dimension_path: str | None = None,
    run_date: str | None = None,
    sink_format: str = "delta",
) -> dict:
    """Build or update the movie SCD Type 2 dimension table."""
    effective_ts = f"{run_date}T00:00:00Z" if run_date else datetime.now(timezone.utc).isoformat(timespec="seconds")
    dimension_path = dimension_path or get_delta_table("gold.dim_movie_scd").path

    if sink_format == "delta":
        try:
            from delta.tables import DeltaTable

            incoming_versions = add_scd_record_hash(
                dedupe_latest_movies(incoming_df, MOVIE_KEY_COLUMNS),
                MOVIE_SCD_TRACKED_COLUMNS,
            ).withColumn(SCD_START_COL, lit(effective_ts)) \
                .withColumn(SCD_END_COL, lit(SCD_HIGH_DATE)) \
                .withColumn(SCD_CURRENT_COL, lit(True))

            is_existing_delta = False
            try:
                is_existing_delta = DeltaTable.isDeltaTable(spark, dimension_path)
            except Exception:
                is_existing_delta = False

            if not is_existing_delta:
                write_table(
                    incoming_versions,
                    dimension_path,
                    sink_format=sink_format,
                    mode="overwrite",
                    partition_columns=[SCD_CURRENT_COL],
                )
            else:
                delta_table = DeltaTable.forPath(spark, dimension_path)
                merge_condition = " AND ".join([f"target.{key} = source.{key}" for key in MOVIE_KEY_COLUMNS])
                delta_table.alias("target").merge(
                    incoming_versions.alias("source"),
                    f"{merge_condition} AND target.{SCD_CURRENT_COL} = true AND target.{SCD_HASH_COL} <> source.{SCD_HASH_COL}",
                ).whenMatchedUpdate(
                    set={
                        SCD_CURRENT_COL: "false",
                        SCD_END_COL: f"'{effective_ts}'",
                    }
                ).execute()

                current_df = spark.read.format("delta").load(dimension_path) \
                    .filter(col(SCD_CURRENT_COL) == lit(True)) \
                    .select(*MOVIE_KEY_COLUMNS, SCD_HASH_COL)
                inserts_df = incoming_versions.join(
                    current_df,
                    list(MOVIE_KEY_COLUMNS) + [SCD_HASH_COL],
                    "left_anti",
                )
                inserts_df.write.format("delta").mode("append").save(dimension_path)

            scd_df = spark.read.format("delta").load(dimension_path)
        except ImportError as exc:
            raise RuntimeError(
                "Delta SCD upserts require delta-spark or cluster-level Delta Lake jars. "
                "Use sink_format='parquet' for local fallback runs."
            ) from exc
    else:
        existing_df = load_table_if_exists(spark, dimension_path, sink_format)
        if existing_df is not None:
            existing_df = spark.createDataFrame(existing_df.collect(), existing_df.schema)
        scd_df = apply_spark_scd_type2(
            incoming_df=incoming_df,
            existing_df=existing_df,
            effective_ts=effective_ts,
        )
        write_table(scd_df, dimension_path, sink_format=sink_format, mode="overwrite")

    current_count = scd_df.filter(col(SCD_CURRENT_COL) == lit(True)).count()
    total_versions = scd_df.count()
    return {
        "path": dimension_path,
        "format": sink_format,
        "current_rows": int(current_count),
        "total_versions": int(total_versions),
        "effective_ts": effective_ts,
    }

def run_spark_etl(
    input_path: str = None,
    run_date: str = None,
    sink_format: str = "delta",
    run_id: str | None = None,
    tenant_id: str = DEFAULT_TENANT_ID,
    catalog_id: str = DEFAULT_CATALOG_ID,
    source_system: str = DEFAULT_SOURCE_SYSTEM,
):
    """
    Run the Spark ETL pipeline with Medallion Architecture.
    Args:
        input_path: Path to raw CSV (defaults to data/raw/TMDB_all_movies.csv)
        run_date: Date string (YYYY-MM-DD) for partitioning
        sink_format: Output format ('parquet', 'delta', 'snowflake')
    """
    logger.info("Starting Spark ETL...")
    started_at = datetime.now(timezone.utc)

    if input_path is None:
        input_path = str(paths.raw_data / "TMDB_all_movies.csv")
    run_date = run_date or datetime.now(timezone.utc).date().isoformat()
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    spark = create_spark_session(enable_delta=(sink_format == "delta"))

    # Load data
    logger.info(f"Reading from {input_path}")
    df = spark.read.option("mode", "DROPMALFORMED") \
        .csv(input_path, header=True, inferSchema=True)
    raw_df = df

    initial_count = df.count()
    logger.info(f"Loaded {initial_count:,} rows")

    # ---------------------------------------------------------
    # DATA QUALITY GATES (Abort if data is garbage)
    # ---------------------------------------------------------
    if initial_count == 0:
        logger.error("DQ FAILURE: Input dataset is empty.")
        spark.stop()
        raise ValueError("Input dataset is empty")

    # Check null rate for critical columns
    null_titles = df.filter(col("title").isNull()).count()
    null_rate = null_titles / initial_count

    if null_rate > 0.5: # Hard limit: if >50% movies have no title, source is broken
        logger.error(f"DQ FAILURE: Null title rate {null_rate:.2%} exceeds 50% threshold.")
        spark.stop()
        raise ValueError(f"Data Quality Error: Too many null titles ({null_rate:.2%})")

    logger.info("DQ Success: Input data passed basic quality gates.")

    # Row-level quality gates. Invalid records are preserved in quarantine
    # so customer-facing SLAs can explain what was rejected and why.
    df, quarantine_df = split_valid_and_quarantined_movies(
        df,
        run_date=run_date,
        run_id=run_id,
    )
    parse_names_udf = udf(parse_metadata_name_list, StringType())
    for metadata_column in ("genres", "keywords", "production_companies"):
        if metadata_column in df.columns:
            df = df.withColumn(metadata_column, parse_names_udf(col(metadata_column)))
    valid_count = df.count()
    quarantine_count = quarantine_df.count()
    logger.info("DQ Success: %s valid rows, %s quarantined rows", valid_count, quarantine_count)

    # Create Tags Column (Simple concatenation for now, Spark SQL is fast)
    # Note: We duplicate simple tag generation here for full Spark pipeline
    df = df.withColumn("tags",
        expr("concat_ws('. ', title, coalesce(overview, ''), 'Movie')")
    )

    # ---------------------------------------------------------
    # DISTRIBUTED MODEL INFERENCE (The "Pro" Move)
    # ---------------------------------------------------------
    logger.info("Generating Embeddings using Pandas UDF...")

    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, FloatType
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    # Broadcast model isn't efficient for large weights, better to load on executors once
    # We use a Scalar Iterator UDF to amortize model loading cost across a batch

    @pandas_udf(ArrayType(FloatType()))
    def predict_embeddings(iterator):
        # Load model once per partition/iterator
        model = SentenceTransformer('all-mpnet-base-v2')
        model.eval() # Inference mode

        for series in iterator:
            # Series is a batch of strings
            # Encode batch
            embeddings = model.encode(
                series.tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            # Normalize
            import numpy as np
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            yield pd.Series(list(embeddings))

    def add_embeddings(df: DataFrame) -> DataFrame:
        """Generate embeddings for only the rows that need fresh vectors."""
        return df.repartition(10).withColumn("vector", predict_embeddings(col("tags")))

    # ---------------------------------------------------------
    # MEDALLION ARCHITECTURE DATA SINK
    # ---------------------------------------------------------
    def delta_literal(value: str) -> str:
        """Escape a string for Delta replaceWhere predicates."""
        return str(value).replace("'", "''")

    run_replace_where = f"run_date = '{delta_literal(run_date)}'"
    content_replace_where = (
        f"tenant_id = '{delta_literal(tenant_id)}' AND "
        f"catalog_id = '{delta_literal(catalog_id)}' AND "
        f"run_date = '{delta_literal(run_date)}'"
    )

    def write_bronze(df, format_type="delta", mode=None, **kwargs):
        """Write raw data to Bronze layer."""
        logger.info(f"Writing data to Bronze layer: format={format_type}")

        if format_type == "delta":
            table = get_delta_table("bronze.movies")
            output_df = add_batch_metadata(df, kwargs["run_date"], kwargs["run_id"])
            return write_delta_table(
                output_df,
                table,
                mode=mode or "overwrite",
                validate_contract=True,
                replace_where=run_replace_where,
            )["path"]

        path = str(paths.bronze_data / "movies")
        if kwargs.get("run_date"):
            path += f"/run_date={kwargs.get('run_date')}"

        writer = df.write.mode(mode or "overwrite")
        writer.format(format_type).save(path)
        return path

    def transform_to_silver(df):
        """Transform raw data to Silver layer (cleaned, validated, enriched)."""
        logger.info("Transforming data to Silver layer")

        # Data Quality: Handle missing values in critical fields
        df = df.withColumn("title", when(col("title").isNull(), "Unknown").otherwise(col("title")))
        df = df.withColumn("overview", when(col("overview").isNull(), "").otherwise(col("overview")))

        # Data Enrichment: Create additional features
        df = df.withColumn("release_year",
                          when(col("release_date").isNotNull(),
                               expr("substring(release_date, 1, 4)")).otherwise(None))

        # Create more sophisticated tags for better recommendations
        df = df.withColumn("tags",
            expr("concat_ws('. ', " +
                 "coalesce(title, ''), " +
                 "coalesce(overview, ''), " +
                 "coalesce(genres, ''), " +
                 "coalesce(cast, ''), " +
                 "coalesce(director, ''), " +
                 "'Movie')")
        )

        # Data Standardization: trim display text without destroying customer-facing casing.
        df = df.withColumn("title", expr("trim(title)"))
        df = df.withColumn("overview", expr("trim(overview)"))
        df = add_catalog_coverage_features(df)

        # Add data quality metrics
        df = df.withColumn("title_completeness", when(col("title") != "Unknown", 1.0).otherwise(0.0))
        df = df.withColumn("overview_completeness", when(length(col("overview")) > 0, 1.0).otherwise(0.0))

        return df

    def write_silver(df, format_type="delta", mode=None, **kwargs):
        """Write cleaned data to Silver layer."""
        logger.info(f"Writing data to Silver layer: format={format_type}")

        if format_type == "delta":
            table = get_delta_table("silver.movies")
            output_df = add_batch_metadata(df, kwargs["run_date"], kwargs["run_id"])
            return write_delta_table(
                output_df,
                table,
                mode=mode or "overwrite",
                validate_contract=True,
                replace_where=run_replace_where,
            )["path"]

        path = str(paths.silver_data / "movies")
        if kwargs.get("run_date"):
            path += f"/run_date={kwargs.get('run_date')}"

        writer = df.write.mode(mode or "overwrite")
        writer.format(format_type).save(path)
        return path

    def write_quarantine(df, format_type="delta", mode=None):
        """Write rejected rows to the Silver quarantine table."""
        logger.info(f"Writing quarantined records: format={format_type}")

        if format_type == "delta":
            return write_delta_table(
                df,
                "silver.movies_quarantine",
                mode=mode or "overwrite",
                validate_contract=True,
                replace_where=run_replace_where,
            )["path"]

        path = str(paths.silver_data / "movies_quarantine")
        if run_date:
            path += f"/run_date={run_date}"

        writer = df.write.mode(mode or "overwrite")
        writer.format(format_type).save(path)
        return path

    def transform_to_gold(df):
        """Transform Silver data to Gold layer (business-level aggregations and ML-ready features)."""
        logger.info("Transforming data to Gold layer")

        # Business Logic: Create ML-ready features
        df = df.withColumn("popularity_score",
                          coalesce(col("popularity").cast("double"), lit(0.0)) * (coalesce(col("vote_average").cast("double"), lit(0.0)) / 10.0))

        df = df.withColumn("quality_score",
                          coalesce(col("content_quality_score"), lit(0.0)))

        # Create features for recommendation system
        df = df.withColumn("is_popular", when(coalesce(col("popularity").cast("double"), lit(0.0)) > 50, 1).otherwise(0))
        df = df.withColumn("is_high_rated", when(coalesce(col("vote_average").cast("double"), lit(0.0)) >= 7.5, 1).otherwise(0))
        df = df.withColumn("is_recent", when(col("release_year") >= "2015", 1).otherwise(0))

        # Create genre features for better recommendations
        if "genres" in df.columns:
            # Extract top 3 genres
            df = df.withColumn("top_genre",
                              expr("split(genres, ',')[0]"))
            df = df.withColumn("second_genre",
                              expr("case when size(split(genres, ',')) > 1 then split(genres, ',')[1] else null end"))
            df = df.withColumn("third_genre",
                              expr("case when size(split(genres, ',')) > 2 then split(genres, ',')[2] else null end"))

        # Add business metrics
        df = df.withColumn("engagement_score",
                          (col("popularity_score") * 0.6) +
                          (col("quality_score") * 0.4))

        return df

    def write_gold(df, format_type="delta", mode=None, **kwargs):
        """Write business-level data to Gold layer."""
        logger.info(f"Writing data to Gold layer: format={format_type}")

        if format_type == "delta":
            table = get_delta_table("gold.movies_features")
            output_df = add_batch_metadata(df, kwargs["run_date"], kwargs["run_id"])
            return write_delta_table(
                output_df,
                table,
                mode=mode or "overwrite",
                validate_contract=True,
                replace_where=run_replace_where,
            )["path"]

        path = str(paths.gold_data / "movies")
        if kwargs.get("run_date"):
            path += f"/run_date={kwargs.get('run_date')}"

        writer = df.write.mode(mode or "overwrite")
        writer.format(format_type).save(path)
        return path

    def write_sink(df, format_type="delta", mode="overwrite", **kwargs):
        """
        Legacy data sink for backward compatibility.
        Supports:
        - 'parquet': Local/S3 (Standard)
        - 'delta': Delta Lake
        - 'snowflake': Snowflake Data Cloud
        """
        logger.info(f"Writing data to legacy sink: format={format_type}")

        if format_type == "delta":
            # Write to Silver layer by default for backward compatibility
            return write_silver(df, format_type, mode, **kwargs)

        elif format_type == "snowflake":
            # SNOWFLAKE INTEGRATION
            # Requires spark-snowflake connector
            writer = df.write.mode(mode)
            writer \
                .format("net.snowflake.spark.snowflake") \
                .options(**{
                    "sfUrl": kwargs.get("sfUrl"),
                    "sfUser": kwargs.get("sfUser"),
                    "sfPassword": kwargs.get("sfPassword"),
                    "sfDatabase": "MOVIE_DB",
                    "sfSchema": "PUBLIC",
                    "sfWarehouse": "COMPUTE_WH"
                }) \
                .option("dbtable", "MOVIES_PROCESSED") \
                .save()

        else:
            # DEFAULT: Parquet to Silver layer
            path = str(paths.silver_data / "movies")
            if kwargs.get("run_date"):
                path += f"/run_date={kwargs.get('run_date')}"
            df.write.mode(mode).format(format_type).save(path)
            return path

    # ---------------------------------------------------------
    # MEDALLION ARCHITECTURE PIPELINE
    # ---------------------------------------------------------
    # Write to Bronze layer (raw ingested data)
    bronze_path = write_bronze(raw_df, format_type=sink_format, run_date=run_date, run_id=run_id)
    logger.info(f"Bronze layer data written to: {bronze_path}")

    if quarantine_count:
        quarantine_path = write_quarantine(quarantine_df, format_type=sink_format)
        logger.info("Silver quarantine rows written to: %s", quarantine_path)

    # Write to Silver layer (cleaned, validated, filtered data)
    silver_df = transform_to_silver(df)
    silver_path = write_silver(silver_df, format_type=sink_format, run_date=run_date, run_id=run_id)
    logger.info(f"Silver layer data written to: {silver_path}")

    if sink_format == "delta":
        content_silver_df = movie_snapshot_to_content_items(
            add_batch_metadata(silver_df, run_date, run_id),
            tenant_id=tenant_id,
            catalog_id=catalog_id,
            source_system=source_system,
        )
        write_delta_table(
            content_silver_df,
            "silver.content_items",
            mode="overwrite",
            validate_contract=True,
            replace_where=content_replace_where,
        )
        logger.info(
            "Tenant-aware content items written: tenant_id=%s catalog_id=%s",
            tenant_id,
            catalog_id,
        )

    # Build Gold feature rows, then identify only new/changed rows for embedding.
    gold_df = transform_to_gold(silver_df)
    previous_features = (
        load_previous_features_for_incremental(spark, "gold.movies_features")
        if sink_format == "delta"
        else None
    )
    embedding_jobs_df = build_embedding_jobs(
        current_features=gold_df,
        previous_features=previous_features,
        run_date=run_date,
        run_id=run_id,
        model_name=EMBEDDING_MODEL_NAME,
    )
    embedding_job_count = embedding_jobs_df.count()
    logger.info("Embedding jobs generated for %s new/changed movies", embedding_job_count)

    if sink_format == "delta":
        write_embedding_jobs(
            embedding_jobs_df,
            mode="overwrite",
            replace_where=f"source_run_date = '{delta_literal(run_date)}'",
        )
        logger.info("Embedding job queue written to Delta table: %s", get_delta_table("gold.movie_embedding_jobs").path)

    fresh_vectors_df = add_embeddings(
        embedding_jobs_df.select(col("movie_id").alias("id"), "tags")
    ).select("id", col("vector").alias("fresh_vector"))
    gold_df = gold_df.join(fresh_vectors_df, "id", "left")

    if previous_features is not None and "vector" in previous_features.columns:
        previous_vectors_df = latest_rows_by_key(previous_features, key_columns=("id",)) \
            .select("id", col("vector").alias("previous_vector"))
        gold_df = gold_df.join(previous_vectors_df, "id", "left")
        gold_df = gold_df.withColumn("vector", coalesce(col("fresh_vector"), col("previous_vector"))) \
            .drop("fresh_vector", "previous_vector")
    else:
        gold_df = gold_df.withColumnRenamed("fresh_vector", "vector")

    final_count = gold_df.count()
    logger.info(f"Prepared Gold features for {final_count:,} rows with incremental embeddings")

    # Write to Gold layer (business-level data with aggregations and ML-ready features)
    gold_path = write_gold(gold_df, format_type=sink_format, run_date=run_date, run_id=run_id)
    logger.info(f"Gold layer data written to: {gold_path}")

    if sink_format == "delta":
        content_features_df = movie_features_to_content_features(
            add_batch_metadata(gold_df, run_date, run_id),
            tenant_id=tenant_id,
            catalog_id=catalog_id,
            source_system=source_system,
        )
        write_delta_table(
            content_features_df,
            "gold.content_features",
            mode="overwrite",
            validate_contract=True,
            replace_where=content_replace_where,
        )
        logger.info(
            "Tenant-aware content features written: tenant_id=%s catalog_id=%s",
            tenant_id,
            catalog_id,
        )

    # Maintain a historical movie dimension for explainable daily changes.
    if sink_format in {"delta", "parquet"}:
        scd_info = upsert_movie_scd_dimension(
            spark=spark,
            incoming_df=silver_df,
            run_date=run_date,
            sink_format=sink_format,
        )
        logger.info(
            "Movie SCD Type 2 dimension written: path=%s current_rows=%s total_versions=%s",
            scd_info["path"],
            scd_info["current_rows"],
            scd_info["total_versions"],
        )
    else:
        logger.info("Skipping SCD dimension for unsupported sink format: %s", sink_format)

    if sink_format == "delta":
        write_data_quality_observation(
            spark,
            run_id=run_id,
            run_date=run_date,
            table_name="bronze.movies",
            metric_name="input_rows",
            metric_value=float(initial_count),
            status="pass",
        )
        write_data_quality_observation(
            spark,
            run_id=run_id,
            run_date=run_date,
            table_name="silver.movies_quarantine",
            metric_name="quarantined_rows",
            metric_value=float(quarantine_count),
            threshold_value=float(initial_count) * 0.5,
            status="pass" if quarantine_count <= initial_count * 0.5 else "warn",
        )
        write_pipeline_run(
            spark,
            run_id=run_id,
            run_date=run_date,
            pipeline_name="catalog_batch_refresh",
            status="success",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            input_rows=int(initial_count),
            output_rows=int(final_count),
            embedding_jobs=int(embedding_job_count),
        )

    logger.info("Medallion Architecture data write complete.")

    # ---------------------------------------------------------
    # ARTIFACT GENERATION (Bridge to Backend)
    # ---------------------------------------------------------
    logger.info("Collecting vectors for FAISS index (Reference Architecture Pattern)...")

    # Collect to driver (Acceptable for <1M rows, otherwise use specialized tools)
    try:
        # Read from Gold layer for artifact generation to ensure we use business-ready data
        if sink_format == "delta":
            gold_path = get_delta_table("gold.movies_features").path
            gold_df = spark.read.format("delta").load(gold_path).filter(col("run_id") == run_id)
        else:
            gold_path = str(paths.gold_data / "movies")
            if run_date:
                gold_path += f"/run_date={run_date}"
            gold_df = spark.read.format(sink_format).load(gold_path)
        serving_df = gold_df.orderBy("id")
        serving_columns = [
            column
            for column in [
                "id",
                "title",
                "overview",
                "genres",
                "vote_average",
                "vote_count",
                "popularity",
                "release_date",
                "poster_path",
                "director",
                "cast",
                "original_language",
                "tags",
                "metadata_completeness",
                "content_quality_score",
                "quality_bucket",
                "searchable",
                "recommendable",
                "is_adult_content",
                "public_demo_eligible",
            ]
            if column in serving_df.columns
        ]
        serving_pdf = serving_df.select(*serving_columns).toPandas()
        paths.processed_data.mkdir(parents=True, exist_ok=True)
        serving_pdf.to_parquet(paths.processed_data / "movies_transformed.parquet", index=False)
        semantic_artifacts = write_semantic_artifacts(
            serving_pdf,
            paths.processed_data,
            run_id=run_id,
            run_date=run_date,
        )

        rows = serving_df.select("id", "vector").collect()

        import numpy as np
        import faiss

        # COMPRESSION (Precision Engineering):
        # Convert to float16 (Half Precision) to save 50% RAM/Disk/Network
        movie_ids = np.array([int(r['id']) for r in rows]).astype('int64')
        vectors = np.array([r['vector'] for r in rows]).astype('float16')
        if len(movie_ids) != vectors.shape[0]:
            raise ValueError(
                f"movie id map rows ({len(movie_ids)}) != vector rows ({vectors.shape[0]})"
            )
        movie_id_hash = hashlib.sha256(movie_ids.astype("<i8", copy=False).tobytes()).hexdigest()

        # Save for Backend
        np.save(str(paths.models / "sbert_embeddings.npy"), vectors)
        np.save(str(paths.models / "movie_ids.npy"), movie_ids)
        logger.info(f"Saved {paths.models / 'sbert_embeddings.npy'} (Size: {vectors.nbytes / 1024 / 1024:.2f} MB)")

        # Build FAISS Index with Quantization
        # Use SQfp16 (Scalar Quantizer Float16) to match the storage format
        # This reduces index size by 50% with negligible accuracy loss
        d = vectors.shape[1]

        # Note: FAISS training/adding usually expects float32 input,
        # but stores internally as defined by factory string.
        vectors_f32 = vectors.astype('float32')

        index = faiss.index_factory(d, "SQfp16", faiss.METRIC_INNER_PRODUCT)
        index.train(vectors_f32)
        index.add(vectors_f32)
        if index.ntotal != len(movie_ids):
            raise ValueError(f"FAISS index rows ({index.ntotal}) != movie id rows ({len(movie_ids)})")

        faiss.write_index(index, str(paths.models / "faiss.index"))
        manifest = {
            "run_date": run_date,
            "pipeline": "nova-pyspark-etl",
            "model_name": EMBEDDING_MODEL_NAME,
            "serving_contract": {
                "version": 1,
                "movie_rows": int(len(movie_ids)),
                "embedding_rows": int(vectors.shape[0]),
                "embedding_dimensions": int(vectors.shape[1]) if len(vectors.shape) > 1 else 0,
                "faiss_index_size": int(index.ntotal),
                "movie_id_map_rows": int(len(movie_ids)),
                "movie_id_sha256": movie_id_hash,
            },
            "artifacts": {
                "movies": "movies_transformed.parquet",
                "semantic_twins": "semantic_twins.parquet",
                "semantic_twin_summary": "semantic_twin_summary.json",
                "embeddings": "sbert_embeddings.npy",
                "faiss_index": "faiss.index",
                "movie_ids": "movie_ids.npy",
            },
            "semantic_twins": semantic_artifacts["summary"],
        }
        (paths.models / "pipeline_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info(f"Saved {paths.models / 'faiss.index'} (Compressed SQfp16)")

    except Exception as e:
        logger.warning(f"Could not build local artifacts (maybe running on pure cluster without shared FS?): {e}")

    spark.stop()
    return final_count

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Spark ETL Pipeline")
    parser.add_argument("--date", type=str, help="Run date (YYYY-MM-DD)", default=None)
    parser.add_argument("--run-id", type=str, help="Unique batch run id", default=None)
    parser.add_argument("--sink", type=str, help="Output format (parquet, delta, snowflake)", default="delta")
    parser.add_argument("--tenant-id", type=str, help="Customer/tenant identifier", default=DEFAULT_TENANT_ID)
    parser.add_argument("--catalog-id", type=str, help="Customer catalog identifier", default=DEFAULT_CATALOG_ID)
    parser.add_argument("--source-system", type=str, help="Upstream catalog source system", default=DEFAULT_SOURCE_SYSTEM)
    args = parser.parse_args()

    run_spark_etl(
        run_date=args.date,
        run_id=args.run_id,
        sink_format=args.sink,
        tenant_id=args.tenant_id,
        catalog_id=args.catalog_id,
        source_system=args.source_system,
    )
