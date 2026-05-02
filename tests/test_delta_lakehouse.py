"""
Tests for PySpark/Delta table models and time-travel helpers.

These are deliberately lightweight: they validate the Delta contracts and read
options without needing a running Spark cluster or Delta transaction log.
"""

from datetime import date

import pytest

pyspark_sql = pytest.importorskip("pyspark.sql", reason="Delta lakehouse tests require PySpark.")

from etl.delta_lakehouse import (
    DELTA_TABLES,
    get_delta_table,
    read_delta_changes,
    read_delta_table,
    validate_delta_contract,
    write_delta_table,
)


class FakeDataFrame:
    def __init__(self, columns):
        self.columns = columns
        self.write = FakeWriter()


class FakeWriter:
    def __init__(self):
        self.format_name = None
        self.mode_name = None
        self.options = {}
        self.partition_columns = ()
        self.saved_path = None

    def format(self, value):
        self.format_name = value
        return self

    def mode(self, value):
        self.mode_name = value
        return self

    def option(self, key, value):
        self.options[key] = value
        return self

    def partitionBy(self, *columns):
        self.partition_columns = columns
        return self

    def save(self, path):
        self.saved_path = path


class FakeReader:
    def __init__(self):
        self.format_name = None
        self.options = {}

    def format(self, value):
        self.format_name = value
        return self

    def option(self, key, value):
        self.options[key] = value
        return self

    def load(self, path):
        return {
            "format": self.format_name,
            "options": self.options,
            "path": path,
        }


class FakeSpark:
    def __init__(self):
        self.read = FakeReader()


def test_delta_table_models_capture_medallion_and_scd_contracts():
    assert set(DELTA_TABLES) >= {
        "bronze.movies",
        "silver.movies",
        "gold.movies_features",
        "gold.dim_movie_scd",
        "gold.fact_movie_event",
        "gold.movie_embedding_jobs",
        "gold.pipeline_run",
        "silver.movies_quarantine",
        "gold.tenant_catalog",
        "silver.content_items",
        "gold.content_features",
        "gold.dim_content_scd",
        "gold.fact_content_event",
        "gold.content_behavior_daily",
        "gold.data_quality_observation",
    }

    scd_table = get_delta_table("gold.dim_movie_scd")
    scd_fields = {field.name for field in scd_table.schema.fields}

    assert scd_table.layer == "gold"
    assert scd_table.partition_columns == ("is_current",)
    assert scd_table.primary_key == ("id", "effective_start_at")
    assert {"record_hash", "effective_start_at", "effective_end_at", "is_current"}.issubset(scd_fields)

    jobs_table = get_delta_table("gold.movie_embedding_jobs")
    jobs_fields = {field.name for field in jobs_table.schema.fields}
    assert jobs_table.partition_columns == ("source_run_date",)
    assert {"job_id", "movie_id", "tags_hash", "change_type", "job_status"}.issubset(jobs_fields)


def test_validate_delta_contract_requires_partition_and_required_columns():
    table = get_delta_table("silver.movies")
    valid_columns = set(table.required_columns) | set(table.partition_columns)
    validation = validate_delta_contract(FakeDataFrame(valid_columns), table)

    assert validation["table"] == "silver.movies"
    assert validation["partition_columns"] == ["run_date"]

    with pytest.raises(ValueError, match="missing required columns"):
        validate_delta_contract(FakeDataFrame(["id", "run_date"]), table)


def test_read_delta_table_uses_version_time_travel_option():
    spark = FakeSpark()
    result = read_delta_table(spark, "gold.dim_movie_scd", version_as_of=7)

    assert result["format"] == "delta"
    assert result["options"] == {"versionAsOf": 7}
    assert result["path"] == get_delta_table("gold.dim_movie_scd").path


def test_read_delta_table_uses_timestamp_time_travel_option():
    spark = FakeSpark()
    result = read_delta_table(spark, "gold.dim_movie_scd", timestamp_as_of=date(2026, 5, 2))

    assert result["options"] == {"timestampAsOf": "2026-05-02 23:59:59"}


def test_read_delta_changes_uses_change_data_feed_options():
    spark = FakeSpark()
    result = read_delta_changes(
        spark,
        "gold.movies_features",
        starting_version=3,
        ending_version=5,
    )

    assert result["format"] == "delta"
    assert result["options"] == {
        "readChangeFeed": "true",
        "startingVersion": 3,
        "endingVersion": 5,
    }


def test_write_delta_table_enables_change_data_feed_by_default():
    table = get_delta_table("gold.movie_embedding_jobs")
    columns = set(table.required_columns) | set(table.partition_columns)
    df = FakeDataFrame(columns)
    result = write_delta_table(df, table, mode="append")

    assert result["change_data_feed"] is True
    assert df.write.format_name == "delta"
    assert df.write.mode_name == "append"
    assert df.write.options["delta.enableChangeDataFeed"] == "true"
    assert df.write.partition_columns == ("source_run_date",)


def test_write_delta_table_supports_run_scoped_partition_replacement():
    table = get_delta_table("silver.content_items")
    columns = set(table.required_columns) | set(table.partition_columns)
    df = FakeDataFrame(columns)
    result = write_delta_table(
        df,
        table,
        mode="overwrite",
        replace_where="tenant_id = 'demo' AND catalog_id = 'movies' AND run_date = '2026-05-02'",
    )

    assert result["replace_where"] == "tenant_id = 'demo' AND catalog_id = 'movies' AND run_date = '2026-05-02'"
    assert df.write.options["replaceWhere"] == result["replace_where"]


def test_read_delta_table_rejects_two_time_travel_modes():
    with pytest.raises(ValueError, match="either version_as_of or timestamp_as_of"):
        read_delta_table(
            FakeSpark(),
            "gold.dim_movie_scd",
            version_as_of=1,
            timestamp_as_of="2026-05-02 00:00:00",
        )
