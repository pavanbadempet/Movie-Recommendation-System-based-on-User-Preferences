import os

import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
REPO_ID = "609827932595256"


def force_git_reset_and_pull():
    print(f"Force resetting Databricks Repo ID {REPO_ID} to main...")

    url = f"{DATABRICKS_HOST}/api/2.0/repos/{REPO_ID}"
    del_res = requests.delete(url, headers=HEADERS)
    print(f"Delete Repo Status: {del_res.status_code} - {del_res.text}")

    create_payload = {
        "url": "https://github.com/pavanbadempet/AI-Recommendation-System.git",
        "provider": "gitHub",
        "path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System",
    }
    c_res = requests.post(f"{DATABRICKS_HOST}/api/2.0/repos", headers=HEADERS, json=create_payload)
    print(f"Re-create Repo Status: {c_res.status_code} - {c_res.text}")
    if c_res.status_code in [200, 201]:
        print("🎉 DATABRICKS REPO IS 100% RE-CLONED WITH THE LATEST CODE!")


if __name__ == "__main__":
    force_git_reset_and_pull()
