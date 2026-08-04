"""Unit tests for Spark Declarative Pipeline (SDP) Specification & Executor Engine."""

from etl.spark_declarative_pipeline import SparkDeclarativePipeline


def test_spark_declarative_pipeline_spec_loading():
    pipeline = SparkDeclarativePipeline()
    assert pipeline.spec["pipeline_id"] == "apex_movie_recommendation_pipeline"
    assert pipeline.catalog == "main"
    assert pipeline.schema == "recommendations"


def test_spark_declarative_pipeline_validation():
    pipeline = SparkDeclarativePipeline()
    assert pipeline.validate_spec() is True


def test_spark_declarative_pipeline_dag_compilation():
    pipeline = SparkDeclarativePipeline()
    dag_plan = pipeline.compile_dag()

    assert len(dag_plan) == 3
    layers = [step["layer"] for step in dag_plan]
    assert layers == ["BRONZE", "SILVER", "GOLD"]


def test_spark_declarative_pipeline_execution():
    pipeline = SparkDeclarativePipeline()
    result = pipeline.run()

    assert result["status"] == "SUCCESS"
    assert result["steps_executed"] == 3
