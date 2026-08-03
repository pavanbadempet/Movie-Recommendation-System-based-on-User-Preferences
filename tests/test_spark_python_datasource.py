"""Unit tests for Spark 4.2 Python Data Source API v2."""

import pytest
from etl.spark_python_datasource import MovieRecommendationDataSource, MovieRecommendationDataSourceReader


def test_spark_python_datasource_name_and_schema():
    ds = MovieRecommendationDataSource()
    assert ds.name() == "movie_rec"
    assert "movie_id INT" in ds.schema()


def test_spark_python_datasource_reader_partitions():
    reader = MovieRecommendationDataSourceReader()
    partitions = reader.partitions()
    assert len(partitions) == 2


def test_spark_python_datasource_reader_stream():
    reader = MovieRecommendationDataSourceReader()
    partitions = reader.partitions()
    records = list(reader.read(partitions[0]))

    assert len(records) == 2
    assert records[0][1] == "Movie_0_A"
