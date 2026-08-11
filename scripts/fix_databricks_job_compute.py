import os
import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

def fix_job_compute(job_id=303494851952917):
    print(f"Fixing Databricks Job {job_id} compute to Standard Serverless (CPU)...")

    cancel_url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/cancel"
    requests.post(cancel_url, headers=HEADERS, json={"run_id": 19761387404303})

    reset_url = f"{DATABRICKS_HOST}/api/2.1/jobs/reset"
    payload = {
        "job_id": job_id,
        "new_settings": {
            "name": "⭐ Production Movie Rec Pipeline (Kaggle Ingest -> PySpark Medallion -> Neon Export)",
            "schedule": {
                "quartz_cron_expression": "0 0 0 * * ?",
                "timezone_id": "UTC",
                "pause_status": "UNPAUSED"
            },
            "tasks": [
                {
                    "task_key": "step_00_kaggle_download",
                    "notebook_task": {
                        "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/00_kaggle_download",
                        "source": "WORKSPACE"
                    }
                },
                {
                    "task_key": "step_01_pyspark_etl",
                    "depends_on": [{"task_key": "step_00_kaggle_download"}],
                    "notebook_task": {
                        "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/01_pyspark_etl",
                        "source": "WORKSPACE"
                    }
                },
                {
                    "task_key": "step_02_export_to_neon",
                    "depends_on": [{"task_key": "step_01_pyspark_etl"}],
                    "notebook_task": {
                        "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/02_export_to_neon",
                        "source": "WORKSPACE"
                    }
                }
            ]
        }
    }

    res = requests.post(reset_url, headers=HEADERS, json=payload)
    print(f"Reset Job Response: {res.status_code} - {res.text}")

    run_url = f"{DATABRICKS_HOST}/api/2.1/jobs/run-now"
    run_res = requests.post(run_url, headers=HEADERS, json={"job_id": job_id})
    print(f"New Run Triggered: {run_res.status_code} - {run_res.text}")

if __name__ == "__main__":
    fix_job_compute()
