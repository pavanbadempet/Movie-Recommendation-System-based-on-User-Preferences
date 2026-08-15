import os

import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}


def create_streaming_job():
    print("Creating Real-Time Continuous Streaming Job in Databricks...")
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/create"

    payload = {
        "name": "⚡ Real-Time Continuous Streaming Ingestion (Clickstream -> Neon)",
        "continuous": {"pause_status": "UNPAUSED"},
        "tasks": [
            {
                "task_key": "step_01b_streaming_events",
                "notebook_task": {
                    "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/01b_streaming_events",
                    "source": "WORKSPACE",
                },
            }
        ],
    }

    res = requests.post(url, headers=HEADERS, json=payload)
    print(f"Create Streaming Job Response: {res.status_code} - {res.text}")
    if res.status_code == 200:
        streaming_job_id = res.json().get("job_id")
        print(f"CONTINUOUS STREAMING JOB CREATED! JOB ID: {streaming_job_id}")


if __name__ == "__main__":
    create_streaming_job()
