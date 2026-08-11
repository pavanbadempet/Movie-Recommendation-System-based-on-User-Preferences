import os
import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}

def sync_databricks_workspace_repo():
    print("Fetching Databricks Workspace Git Repos...")
    list_url = f"{DATABRICKS_HOST}/api/2.0/repos"
    res = requests.get(list_url, headers=HEADERS)
    print(f"List Repos Status: {res.status_code}")

    if res.status_code == 200:
        repos = res.json().get("repos", [])
        print(f"Found {len(repos)} Git Repos in Workspace:")

        for r in repos:
            repo_id = r.get("id")
            path = r.get("path")
            branch = r.get("branch")
            print(f"• Repo ID: {repo_id} | Path: {path} | Branch: {branch}")

            print(f"Pulling latest GitHub 'main' branch into Databricks Workspace Repo ID {repo_id}...")
            patch_url = f"{DATABRICKS_HOST}/api/2.0/repos/{repo_id}"
            patch_res = requests.patch(patch_url, headers=HEADERS, json={"branch": "main"})
            print(f"Patch Response: {patch_res.status_code} - {patch_res.text}")

    print("\nRe-triggering Real-Time Continuous Streaming Job (Job ID: 772367112113846)...")
    requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/cancel", headers=HEADERS, json={"run_id": 136400498247464})

    run_res = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/run-now", headers=HEADERS, json={"job_id": 772367112113846})
    print(f"New Run Triggered: {run_res.status_code} - {run_res.text}")

if __name__ == "__main__":
    sync_databricks_workspace_repo()
