import os

import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
GIT_URL = "https://github.com/pavanbadempet/AI-Recommendation-System"


def setup_github_integration():
    print("Setting up direct GitHub Integration for Databricks Jobs...")

    batch_payload = {
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
        },
    }

    r1 = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/reset", headers=HEADERS, json=batch_payload)
    print(f"Batch Job GitHub Integration Status: {r1.status_code} - {r1.text}")

    streaming_payload = {
        "job_id": 772367112113846,
        "new_settings": {
            "name": "⚡ Real-Time Continuous Streaming Ingestion (Clickstream -> Neon)",
            "git_source": {"git_url": GIT_URL, "git_provider": "gitHub", "git_branch": "main"},
            "continuous": {"pause_status": "UNPAUSED"},
            "tasks": [
                {
                    "task_key": "step_01b_streaming_events",
                    "notebook_task": {"notebook_path": "databricks_notebooks/01b_streaming_events", "source": "GIT"},
                }
            ],
        },
    }

    r2 = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/reset", headers=HEADERS, json=streaming_payload)
    print(f"Streaming Job GitHub Integration Status: {r2.status_code} - {r2.text}")

    print("\nTriggering fresh Job runs directly from GitHub main branch...")
    requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/run-now", headers=HEADERS, json={"job_id": 772367112113846})
    requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/run-now", headers=HEADERS, json={"job_id": 303494851952917})


if __name__ == "__main__":
    setup_github_integration()
