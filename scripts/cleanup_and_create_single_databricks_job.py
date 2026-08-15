import os

import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}


def cleanup_and_create_single_job():
    print("Cleaning up duplicate Databricks jobs...")
    list_url = f"{DATABRICKS_HOST}/api/2.1/jobs/list"
    res = requests.get(list_url, headers=HEADERS)

    if res.status_code == 200:
        jobs = res.json().get("jobs", [])
        print(f"Found {len(jobs)} total jobs in Databricks.")

        for j in jobs:
            job_id = j.get("job_id")
            job_name = j.get("settings", {}).get("name", "")
            print(f"Deleting old job: {job_id} ({job_name})...")
            requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/delete", headers=HEADERS, json={"job_id": job_id})

    print("\nCreating SINGLE OFFICIAL PRODUCTION WORKFLOW JOB...")
    create_url = f"{DATABRICKS_HOST}/api/2.1/jobs/create"

    payload = {
        "name": "⭐ Production Movie Rec Pipeline (Kaggle Ingest -> PySpark Medallion -> Neon Export)",
        "schedule": {"quartz_cron_expression": "0 0 0 * * ?", "timezone_id": "UTC", "pause_status": "UNPAUSED"},
        "tasks": [
            {
                "task_key": "step_00_kaggle_download",
                "notebook_task": {
                    "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/00_kaggle_download",
                    "source": "WORKSPACE",
                },
            },
            {
                "task_key": "step_01_pyspark_etl",
                "depends_on": [{"task_key": "step_00_kaggle_download"}],
                "notebook_task": {
                    "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/01_pyspark_etl",
                    "source": "WORKSPACE",
                },
            },
            {
                "task_key": "step_02_export_to_neon",
                "depends_on": [{"task_key": "step_01_pyspark_etl"}],
                "notebook_task": {
                    "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/02_export_to_neon",
                    "source": "WORKSPACE",
                },
            },
        ],
    }

    c_res = requests.post(create_url, headers=HEADERS, json=payload)
    print(f"Create Job Response: {c_res.status_code} - {c_res.text}")
    if c_res.status_code == 200:
        new_job_id = c_res.json().get("job_id")
        print(f"\nSUCCESS! SINGLE CLEAN PRODUCTION JOB CREATED! JOB ID: {new_job_id}")


if __name__ == "__main__":
    cleanup_and_create_single_job()
