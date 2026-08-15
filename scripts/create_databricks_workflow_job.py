import os

import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
GIT_URL = "https://github.com/pavanbadempet/AI-Recommendation-System"


def update_job_split_pipeline():
    """
    Split pipeline: 3 tasks on instant Standard Serverless + 1 isolated GPU task.
    Only the embedding generation step pays the GPU cold-start penalty.
    """
    print("Configuring 4-Task Split Pipeline (3x Standard Serverless + 1x GPU)...")

    payload = {
        "job_id": 303494851952917,
        "new_settings": {
            "name": "⭐ Production Movie Rec Pipeline (Kaggle -> PySpark ETL -> GPU Embeddings -> Neon Export)",
            "git_source": {"git_url": GIT_URL, "git_provider": "gitHub", "git_branch": "main"},
            "schedule": {"quartz_cron_expression": "0 0 0 * * ?", "timezone_id": "UTC", "pause_status": "UNPAUSED"},
            "performance_target": "PERFORMANCE_OPTIMIZED",
            "tasks": [
                {
                    # Task 1: Kaggle Download → Standard Serverless (instant)
                    "task_key": "step_00_kaggle_download",
                    "notebook_task": {"notebook_path": "databricks_notebooks/00_kaggle_download", "source": "GIT"},
                },
                {
                    # Task 2: Pure PySpark ETL → Standard Serverless (instant)
                    "task_key": "step_01_pyspark_etl",
                    "depends_on": [{"task_key": "step_00_kaggle_download"}],
                    "notebook_task": {"notebook_path": "databricks_notebooks/01_pyspark_etl", "source": "GIT"},
                },
                {
                    # Task 3: Embeddings Generation → Standard Serverless (instant execution, 0 GPU DBUs)
                    "task_key": "step_01c_gpu_embeddings",
                    "depends_on": [{"task_key": "step_01_pyspark_etl"}],
                    "notebook_task": {"notebook_path": "databricks_notebooks/01c_gpu_embeddings", "source": "GIT"},
                },
                {
                    # Task 4: Neon Export → Standard Serverless (instant)
                    "task_key": "step_02_export_to_neon",
                    "depends_on": [{"task_key": "step_01c_gpu_embeddings"}],
                    "notebook_task": {"notebook_path": "databricks_notebooks/02_export_to_neon", "source": "GIT"},
                },
            ],
        },
    }

    res = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/reset", headers=HEADERS, json=payload)
    print(f"Job Update Status: {res.status_code} - {res.text}")

    print("\nTriggering pipeline run...")
    r = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/run-now", headers=HEADERS, json={"job_id": 303494851952917})
    print(f"Run Trigger Status: {r.status_code} - {r.text}")


if __name__ == "__main__":
    update_job_split_pipeline()
