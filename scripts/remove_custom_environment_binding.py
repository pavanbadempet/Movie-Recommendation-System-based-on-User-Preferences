import os

import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
GIT_URL = "https://github.com/pavanbadempet/AI-Recommendation-System"


def remove_environment_binding():
    print("Removing custom Environment and Libraries binding from Databricks Job REST API...")

    payload = {
        "job_id": 303494851952917,
        "new_settings": {
            "name": "⭐ Production Movie Rec Pipeline (Kaggle Ingest -> PySpark Medallion -> Neon Export)",
            "git_source": {"git_url": GIT_URL, "git_provider": "gitHub", "git_branch": "main"},
            "schedule": {"quartz_cron_expression": "0 0 0 * * ?", "timezone_id": "UTC", "pause_status": "UNPAUSED"},
            "tasks": [
                {
                    "task_key": "step_00_kaggle_download",
                    "notebook_task": {"notebook_path": "databricks_notebooks/00_kaggle_download", "source": "GIT"},
                },
                {
                    "task_key": "step_01_pyspark_etl",
                    "depends_on": [{"task_key": "step_00_kaggle_download"}],
                    "notebook_task": {"notebook_path": "databricks_notebooks/01_pyspark_etl", "source": "GIT"},
                },
                {
                    "task_key": "step_02_export_to_neon",
                    "depends_on": [{"task_key": "step_01_pyspark_etl"}],
                    "notebook_task": {"notebook_path": "databricks_notebooks/02_export_to_neon", "source": "GIT"},
                },
            ],
            "environments": [],
        },
    }

    res = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/reset", headers=HEADERS, json=payload)
    print(f"Remove Environment Status: {res.status_code} - {res.text}")


if __name__ == "__main__":
    remove_environment_binding()
