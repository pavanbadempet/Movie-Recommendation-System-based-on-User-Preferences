import os
import requests
import json

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

def create_workflow_job():
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/create"
    
    payload = {
        "name": "Daily Full Automated Movie Rec Pipeline (Kaggle -> GPU PySpark -> Neon Export)",
        "schedule": {
            "quartz_cron_expression": "0 0 0 * * ?",
            "timezone_id": "UTC",
            "pause_status": "UNPAUSED"
        },
        "tasks": [
            {
                "task_key": "step_00_kaggle_download",
                "environment_key": "cpu_env",
                "notebook_task": {
                    "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/00_kaggle_download",
                    "source": "WORKSPACE"
                }
            },
            {
                "task_key": "step_01_pyspark_etl",
                "depends_on": [{"task_key": "step_00_kaggle_download"}],
                "environment_key": "gpu_env",
                "notebook_task": {
                    "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/01_pyspark_etl",
                    "source": "WORKSPACE"
                }
            },
            {
                "task_key": "step_02_export_to_neon",
                "depends_on": [{"task_key": "step_01_pyspark_etl"}],
                "environment_key": "cpu_env",
                "notebook_task": {
                    "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/02_export_to_neon",
                    "source": "WORKSPACE"
                }
            }
        ],
        "environments": [
            {
                "environment_key": "cpu_env",
                "spec": {
                    "client": "1"
                }
            },
            {
                "environment_key": "gpu_env",
                "spec": {
                    "client": "1"
                }
            }
        ]
    }

    res = requests.post(url, headers=HEADERS, json=payload)
    print(f"Create Databricks Job Status: {res.status_code} - {res.text}")
    if res.status_code == 200:
        job_id = res.json().get("job_id")
        print(f"DATABRICKS WORKFLOW JOB UPDATED WITH GPU OPTIMIZATION! JOB ID: {job_id}")

if __name__ == "__main__":
    create_workflow_job()
