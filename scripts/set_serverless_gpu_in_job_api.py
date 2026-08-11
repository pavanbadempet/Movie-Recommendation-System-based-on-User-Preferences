import os
import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
GIT_URL = "https://github.com/pavanbadempet/AI-Recommendation-System"

def set_gpu_in_job_api():
    print("Setting Serverless GPU on Step 1 & Step 2...")

    payload = {
        "job_id": 303494851952917,
        "new_settings": {
            "name": "⭐ Production Movie Rec Pipeline (Kaggle Ingest -> PySpark Medallion -> Neon Export)",
            "git_source": {
                "git_url": GIT_URL,
                "git_provider": "gitHub",
                "git_branch": "main"
            },
            "schedule": {
                "quartz_cron_expression": "0 0 0 * * ?",
                "timezone_id": "UTC",
                "pause_status": "UNPAUSED"
            },
            "tasks": [
                {
                    "task_key": "step_00_kaggle_download",
                    "notebook_task": {
                        "notebook_path": "databricks_notebooks/00_kaggle_download",
                        "source": "GIT"
                    }
                },
                {
                    "task_key": "step_01_pyspark_etl",
                    "depends_on": [{"task_key": "step_00_kaggle_download"}],
                    "environment_key": "gpu_env",
                    "notebook_task": {
                        "notebook_path": "databricks_notebooks/01_pyspark_etl",
                        "source": "GIT"
                    }
                },
                {
                    "task_key": "step_02_export_to_neon",
                    "depends_on": [{"task_key": "step_01_pyspark_etl"}],
                    "environment_key": "gpu_env",
                    "notebook_task": {
                        "notebook_path": "databricks_notebooks/02_export_to_neon",
                        "source": "GIT"
                    }
                }
            ],
            "environments": [
                {
                    "environment_key": "gpu_env",
                    "spec": {
                        "client": "1"
                    }
                }
            ]
        }
    }

    res = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/reset", headers=HEADERS, json=payload)
    print(f"Set GPU Status: {res.status_code} - {res.text}")

    print("Triggering new Job Run with Serverless GPU settings...")
    r = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/run-now", headers=HEADERS, json={"job_id": 303494851952917})
    print(f"Run Trigger Status: {r.status_code} - {r.text}")

if __name__ == "__main__":
    set_gpu_in_job_api()
