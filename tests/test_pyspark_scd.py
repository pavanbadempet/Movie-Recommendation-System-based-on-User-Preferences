"""
Tests for Spark SCD Type 2 helpers.

These tests skip automatically when PySpark/Java is not available, so the
serving app can stay lightweight while the ETL path remains testable.
"""

import pytest
from functools import reduce


def test_parse_metadata_name_list_normalizes_kaggle_jsonish_values():
    from etl.metadata_parsing import parse_metadata_name_list

    assert parse_metadata_name_list("[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Drama'}]") == "Action, Drama"
    assert parse_metadata_name_list("Action, Comedy") == "Action, Comedy"
    assert parse_metadata_name_list(None) == ""


@pytest.fixture(scope="module")
def spark_session():
    pytest.importorskip("pyspark.sql", reason="PySpark is only required for Spark ETL tests.")

    from etl.pyspark_etl import create_spark_session

    try:
        spark = create_spark_session(app_name="MovieSCDTests", master="local[1]", enable_delta=False)
    except Exception as exc:
        pytest.skip(f"Spark session could not start in this environment: {exc}")

    yield spark
    spark.stop()


def _movie_snapshot(spark, rows):
    from pyspark.sql import functions as F

    columns = list(rows[0].keys())
    for row in rows[1:]:
        for column in row:
            if column not in columns:
                columns.append(column)

    def _column_type(column):
        values = [row.get(column) for row in rows if row.get(column) is not None]
        if values and all(isinstance(value, bool) for value in values):
            return "boolean"
        if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            return "bigint"
        if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            return "double"
        return "string"

    column_types = {column: _column_type(column) for column in columns}

    frames = [
        spark.range(1).select(
            *[
                F.lit(row.get(column)).cast(column_types[column]).alias(column)
                for column in columns
            ]
        )
        for row in rows
    ]
    return reduce(lambda left, right: left.unionByName(right), frames)


def test_apply_spark_scd_type2_tracks_changed_and_new_movies(spark_session):
    from etl.pyspark_etl import SCD_CURRENT_COL, SCD_END_COL, apply_spark_scd_type2

    first_snapshot = _movie_snapshot(
        spark_session,
        [
            {
                "id": 1,
                "title": "Matrix",
                "overview": "Red pill blue pill",
                "genres": "Sci-Fi",
                "vote_average": 8.7,
                "vote_count": 1000.0,
                "popularity": 90.0,
            },
            {
                "id": 2,
                "title": "Inception",
                "overview": "Dreams within dreams",
                "genres": "Sci-Fi",
                "vote_average": 8.8,
                "vote_count": 2000.0,
                "popularity": 95.0,
            },
        ],
    )
    history = apply_spark_scd_type2(first_snapshot, effective_ts="2026-05-01T00:00:00Z")

    second_snapshot = _movie_snapshot(
        spark_session,
        [
            {
                "id": 1,
                "title": "Matrix",
                "overview": "Red pill blue pill updated",
                "genres": "Sci-Fi",
                "vote_average": 8.7,
                "vote_count": 1000.0,
                "popularity": 90.0,
            },
            {
                "id": 2,
                "title": "Inception",
                "overview": "Dreams within dreams",
                "genres": "Sci-Fi",
                "vote_average": 8.8,
                "vote_count": 2000.0,
                "popularity": 95.0,
            },
            {
                "id": 3,
                "title": "Interstellar",
                "overview": "Space travel data",
                "genres": "Sci-Fi",
                "vote_average": 8.6,
                "vote_count": 1500.0,
                "popularity": 91.0,
            },
        ],
    )
    updated = apply_spark_scd_type2(
        second_snapshot,
        existing_df=history,
        effective_ts="2026-05-02T00:00:00Z",
    )

    rows = [
        row.asDict()
        for row in updated.select("id", "overview", SCD_CURRENT_COL, SCD_END_COL).collect()
    ]

    assert len(rows) == 4
    assert sum(1 for row in rows if row["id"] == 1 and row[SCD_CURRENT_COL]) == 1
    assert sum(1 for row in rows if row["id"] == 1 and not row[SCD_CURRENT_COL]) == 1
    assert any(row["id"] == 1 and row[SCD_END_COL] == "2026-05-02T00:00:00Z" for row in rows)
    assert sum(1 for row in rows if row[SCD_CURRENT_COL]) == 3


def test_spark_quality_gates_keep_long_tail_movies(spark_session):
    from etl.pyspark_etl import add_catalog_coverage_features, split_valid_and_quarantined_movies

    snapshot = _movie_snapshot(
        spark_session,
        [
            {"id": 1, "title": "Obscure Gem", "overview": "", "vote_count": 0.0, "popularity": 0.0},
            {"id": 2, "title": None, "overview": "Missing title", "vote_count": 100.0, "popularity": 10.0},
        ],
    )

    valid, quarantine = split_valid_and_quarantined_movies(snapshot, run_date="2026-05-02", run_id="test-run")
    enriched = add_catalog_coverage_features(valid)
    rows = enriched.select("id", "quality_bucket", "searchable").collect()

    assert valid.count() == 1
    assert quarantine.count() == 1
    assert rows[0]["id"] == 1
    assert rows[0]["searchable"] is True


def test_upsert_movie_scd_dimension_with_parquet_fallback(spark_session, tmp_path):
    from etl.pyspark_etl import upsert_movie_scd_dimension

    dimension_path = str(tmp_path / "dim_movie_scd")
    first_snapshot = _movie_snapshot(
        spark_session,
        [
            {"id": 1, "title": "Matrix", "overview": "Red pill", "vote_count": 100.0, "popularity": 10.0},
            {"id": 2, "title": "Inception", "overview": "Dreams", "vote_count": 200.0, "popularity": 20.0},
        ],
    )
    first_result = upsert_movie_scd_dimension(
        spark_session,
        first_snapshot,
        dimension_path=dimension_path,
        run_date="2026-05-01",
        sink_format="parquet",
    )

    second_snapshot = _movie_snapshot(
        spark_session,
        [
            {"id": 1, "title": "Matrix", "overview": "Red pill updated", "vote_count": 100.0, "popularity": 10.0},
            {"id": 2, "title": "Inception", "overview": "Dreams", "vote_count": 200.0, "popularity": 20.0},
            {"id": 3, "title": "Interstellar", "overview": "Space", "vote_count": 300.0, "popularity": 30.0},
        ],
    )
    second_result = upsert_movie_scd_dimension(
        spark_session,
        second_snapshot,
        dimension_path=dimension_path,
        run_date="2026-05-02",
        sink_format="parquet",
    )

    assert first_result["current_rows"] == 2
    assert first_result["total_versions"] == 2
    assert second_result["current_rows"] == 3
    assert second_result["total_versions"] == 4
