"""Unit tests for Unified Data & AI Intelligence Platform (Lakeflow & Agentic BI)."""

import pytest
from etl.lakeflow_pipeline import LakeflowPipelineOrchestrator
from backend.intelligence.agentic_bi import AgenticBIEngine


def test_lakeflow_pipeline_orchestrator():
    orchestrator = LakeflowPipelineOrchestrator()
    res = orchestrator.run()

    assert res["status"] == "SUCCESS"
    assert res["total_steps"] == 4
    assert res["execution_log"][0]["step_name"] == "connect_source_neon_postgres"


def test_agentic_bi_sql_generation():
    bi_engine = AgenticBIEngine()
    sql = bi_engine.generate_sql("Show me highest rated movies")

    assert "SELECT" in sql
    assert "gold_movie_features" in sql
    assert "ORDER BY avg_rating DESC" in sql


def test_agentic_bi_analytics_execution():
    bi_engine = AgenticBIEngine()
    res = bi_engine.execute_analytics("Top movies by rating")

    assert res["result_rows"] > 0
    assert len(res["data"]) == 3
    assert res["data"][0]["title"] == "The Shawshank Redemption"
