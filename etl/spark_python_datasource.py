"""Spark 4.2 Python Data Source API v2 Implementation."""

from __future__ import annotations

from collections.abc import Iterator
import logging

from pyspark.sql.datasource import DataSource, DataSourceReader, InputPartition

logger = logging.getLogger(__name__)


class MovieRecommendationPartition(InputPartition):
    """Represents an input partition for movie recommendation streams."""

    def __init__(self, partition_id: int):
        self.partition_id = partition_id


class MovieRecommendationDataSourceReader(DataSourceReader):
    """Python Data Source Reader v2 for PySpark 4.2."""

    def partitions(self) -> list[InputPartition]:
        return [MovieRecommendationPartition(0), MovieRecommendationPartition(1)]

    def read(self, partition: InputPartition) -> Iterator[tuple]:
        p_id = getattr(partition, "partition_id", 0)
        yield (p_id * 10 + 1, f"Movie_{p_id}_A", "Action", 4.5)
        yield (p_id * 10 + 2, f"Movie_{p_id}_B", "Sci-Fi", 4.8)


class MovieRecommendationDataSource(DataSource):
    """
    Custom Python Data Source v2 for Apache Spark 4.2.
    Allows PySpark 4.2 to query custom recommendation streams via standard `spark.read.format("movie_rec").load()`.
    """

    def __init__(self, options: dict | None = None):
        super().__init__(options=options or {})

    @classmethod
    def name(cls) -> str:
        return "movie_rec"

    def schema(self) -> str:
        return "movie_id INT, title STRING, genres STRING, rating DOUBLE"

    def reader(self, schema) -> DataSourceReader:
        return MovieRecommendationDataSourceReader()
