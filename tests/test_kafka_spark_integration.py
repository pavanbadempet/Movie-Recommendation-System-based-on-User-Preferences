"""
Tests for Kafka and Spark integration in Airflow DAGs.
"""

import pytest
import os

airflow_models = pytest.importorskip(
    "airflow.models",
    reason="Airflow DAG tests require apache-airflow to be installed.",
)
try:
    DagBag = airflow_models.DagBag
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(
        f"Airflow DAG tests require a compatible Airflow runtime: {exc}",
        allow_module_level=True,
    )

# Mock environment variables
os.environ["KAGGLE_KEY"] = "mock_key"
os.environ["KAGGLE_USERNAME"] = "mock_user"

def test_kafka_spark_integration_dag_loads():
    """Verify the Kafka-Spark integration DAG loads without import errors."""
    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)

    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"
    assert "kafka_spark_integration" in dag_bag.dags

def test_kafka_spark_integration_dag_structure():
    """Verify the Kafka-Spark integration DAG has the expected tasks."""
    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dag_bag.dags["kafka_spark_integration"]

    # Check tasks exist
    task_ids = set(dag.task_ids)
    expected_tasks = {
        "check_kafka_connection",
        "check_spark_connection",
        "create_movie_events_topic",
        "produce_movie_events",
        "process_events_with_spark",
        "run_spark_etl_with_delta"
    }
    assert expected_tasks.issubset(task_ids)

    # Check dependencies
    t0 = dag.get_task("check_kafka_connection")
    t1 = dag.get_task("check_spark_connection")
    t2 = dag.get_task("create_movie_events_topic")
    t3 = dag.get_task("produce_movie_events")
    t4 = dag.get_task("process_events_with_spark")
    t5 = dag.get_task("run_spark_etl_with_delta")

    assert t1 in t0.downstream_list
    assert t2 in t1.downstream_list
    assert t3 in t2.downstream_list
    assert t4 in t3.downstream_list
    assert t5 in t4.downstream_list

def test_movie_data_refresh_dag_has_delta_support():
    """Verify the main movie data refresh DAG has Delta Lake support."""
    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dag_bag.dags["movie_data_refresh"]

    # Check that Spark ETL is configured to write Delta Lake output
    assert "run_spark_etl" in dag.task_ids
    spark_task = dag.get_task("run_spark_etl")
    assert "--sink delta" in spark_task.bash_command

    # Check dependencies
    t2 = dag.get_task("run_spark_etl")
    t3 = dag.get_task("rebuild_index")

    assert t3 in t2.downstream_list

def test_dag_imports():
    """Test that the DAG files can import necessary modules."""
    pytest.importorskip("kafka", reason="Kafka client is only needed for integration runs.")
    pytest.importorskip("pyspark.sql", reason="PySpark is only needed for integration runs.")
