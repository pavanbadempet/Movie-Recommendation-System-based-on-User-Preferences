import os
import requests
import json

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

def verify_databricks():
    print("Connecting to Databricks REST API...")
    res = requests.get(f"{DATABRICKS_HOST}/api/2.0/clusters/list", headers=HEADERS)
    print(f"Databricks Clusters Status: {res.status_code}")
    if res.status_code == 200:
        clusters = res.json().get("clusters", [])
        print(f"Found {len(clusters)} cluster(s) in Databricks workspace:")
        for c in clusters:
            print(f" - {c.get('cluster_name')} (ID: {c.get('cluster_id')}, State: {c.get('state')})")
    else:
        print(f"Workspace output: {res.text}")

    # Check Workspace path
    res_ws = requests.get(f"{DATABRICKS_HOST}/api/2.0/workspace/list", headers=HEADERS, params={"path": "/"})
    print(f"Workspace List Status: {res_ws.status_code}")

if __name__ == "__main__":
    verify_databricks()
