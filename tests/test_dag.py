"""
Tests for Airflow DAG integrity.
Ensures the DAG loads correctly and has the expected structure.
"""
import pytest
from airflow.models import DagBag
from pathlib import Path
import os

# Mock Airflow environment variables if not present
os.environ["KAGGLE_KEY"] = "mock_key"
os.environ["KAGGLE_USERNAME"] = "mock_user"

def test_dag_loads_with_no_errors():
    """Verify the DAG file loads without import errors."""
    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
    
    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"
    assert "movie_data_refresh" in dag_bag.dags

def test_dag_structure():
    """Verify tasks and dependencies."""
    dag_bag = DagBag(dag_folder="airflow/dags", include_examples=False)
    dag = dag_bag.dags["movie_data_refresh"]
    
    # Check tasks exist
    task_ids = set(dag.task_ids)
    expected_tasks = {"check_creds", "download_from_kaggle", "run_spark_etl", "rebuild_index"}
    assert expected_tasks.issubset(task_ids)
    
    # Check dependencies
    # check_creds >> download_from_kaggle >> run_spark_etl >> rebuild_index
    t0 = dag.get_task("check_creds")
    t1 = dag.get_task("download_from_kaggle")
    t2 = dag.get_task("run_spark_etl")
    t3 = dag.get_task("rebuild_index")
    
    assert t1 in t0.downstream_list
    assert t2 in t1.downstream_list
    assert t3 in t2.downstream_list
