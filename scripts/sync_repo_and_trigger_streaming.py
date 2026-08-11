import os
import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}

def sync_and_trigger():
    print("Syncing Databricks Workspace Repo to latest main...")
    repo_url = f"{DATABRICKS_HOST}/api/2.0/repos/609827932595256"
    r = requests.patch(repo_url, headers=HEADERS, json={"branch": "main"})
    print(f"Repo Sync Status: {r.status_code} - {r.text}")

    print("Triggering Real-Time Continuous Streaming Job (Job ID: 772367112113846)...")
    job_url = f"{DATABRICKS_HOST}/api/2.1/jobs/run-now"
    r2 = requests.post(job_url, headers=HEADERS, json={"job_id": 772367112113846})
    print(f"Streaming Job Trigger: {r2.status_code} - {r2.text}")

if __name__ == "__main__":
    sync_and_trigger()
