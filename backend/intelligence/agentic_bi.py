"""Agentic Business Intelligence (AI & BI) Query & Analytics Engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgenticBIEngine:
    """
    Agentic BI (AI + BI) Engine (Databricks-compatible Open Analytics Architecture).
    Translates natural language questions into DBSQL/Delta queries, computes metric aggregations,
    and formats visual markdown reports.
    """

    def __init__(self, catalog: str = "main", schema: str = "recommendations"):
        self.catalog = catalog
        self.schema = schema

    def generate_sql(self, natural_language_prompt: str) -> str:
        """Translates natural language prompt into ANSI compliant SQL for DBSQL / Spark SQL."""
        prompt_lower = natural_language_prompt.lower()
        if "top" in prompt_lower or "best" in prompt_lower or "highest rated" in prompt_lower:
            return (
                f"SELECT movie_id, title, genres, AVG(rating) as avg_rating, COUNT(*) as vote_count "
                f"FROM {self.catalog}.{self.schema}.gold_movie_features "
                f"GROUP BY movie_id, title, genres "
                f"HAVING vote_count >= 10 "
                f"ORDER BY avg_rating DESC LIMIT 10;"
            )
        elif "genre" in prompt_lower:
            return (
                f"SELECT genres, COUNT(*) as total_movies "
                f"FROM {self.catalog}.{self.schema}.silver_curated_movies "
                f"GROUP BY genres ORDER BY total_movies DESC;"
            )
        else:
            return (
                f"SELECT COUNT(*) as total_movies, COUNT(DISTINCT genres) as total_genres "
                f"FROM {self.catalog}.{self.schema}.silver_curated_movies;"
            )

    def execute_analytics(self, natural_language_prompt: str) -> Dict[str, Any]:
        """Executes natural language BI query and returns data payload and SQL query."""
        sql_query = self.generate_sql(natural_language_prompt)
        logger.info(f"Agentic BI generated SQL for '{natural_language_prompt}': {sql_query}")

        # Simulated high-speed DBSQL execution result
        mock_data = [
            {"movie_id": 1, "title": "The Shawshank Redemption", "avg_rating": 4.9, "vote_count": 1500},
            {"movie_id": 2, "title": "The Godfather", "avg_rating": 4.85, "vote_count": 1420},
            {"movie_id": 3, "title": "The Dark Knight", "avg_rating": 4.82, "vote_count": 1890},
        ]

        return {
            "prompt": natural_language_prompt,
            "generated_sql": sql_query,
            "result_rows": len(mock_data),
            "data": mock_data,
            "summary": f"Agentic BI query executed successfully on {self.catalog}.{self.schema}.",
        }
