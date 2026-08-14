import os
import sys
import requests
import json

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

def query_databricks_live():
    print("=" * 60)
    print("LIVE DATABRICKS API & LAKEHOUSE INSPECTOR")
    print("=" * 60)
    print(f"Connecting to Host: {DATABRICKS_HOST}")
    
    # 1. Check Jobs
    r_jobs = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/list", headers=HEADERS)
    print(f"\nDatabricks Jobs (HTTP {r_jobs.status_code}):")
    if r_jobs.status_code == 200:
        jobs = r_jobs.json().get("jobs", [])
        print(f"Total Configured Workflow Jobs: {len(jobs)}")
        for j in jobs:
            print(f"  * Job ID: {j.get('job_id')} | Name: '{j.get('settings', {}).get('name')}' | Created: {j.get('created_time')}")
    else:
        print(f"  Error: {r_jobs.text}")

    # 2. Check Recent Runs
    r_runs = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/list?limit=5", headers=HEADERS)
    print(f"\nRecent Pipeline Execution Runs (HTTP {r_runs.status_code}):")
    if r_runs.status_code == 200:
        runs = r_runs.json().get("runs", [])
        print(f"Total Recent Runs: {len(runs)}")
        for r in runs:
            state = r.get("state", {})
            print(f"  * Run ID: {r.get('run_id')} | Job: {r.get('job_id')} | State: {state.get('life_cycle_state')} | Result: {state.get('result_state', 'IN_PROGRESS')}")
    else:
        print(f"  Error: {r_runs.text}")

    # 3. Check Workspace Notebooks
    r_ws = requests.get(f"{DATABRICKS_HOST}/api/2.0/workspace/list", headers=HEADERS, params={"path": "/Users"})
    print(f"\nWorkspace Directory Listing (HTTP {r_ws.status_code}):")
    if r_ws.status_code == 200:
        objects = r_ws.json().get("objects", [])
        for obj in objects:
            print(f"  * Path: {obj.get('path')} (Type: {obj.get('object_type')})")
    else:
        print(f"  Error: {r_ws.text}")

    # 4. Check Catalogs / Schemas (Unity Catalog)
    r_cat = requests.get(f"{DATABRICKS_HOST}/api/2.1/unity-catalog/catalogs", headers=HEADERS)
    print(f"\nUnity Catalog Status (HTTP {r_cat.status_code}):")
    if r_cat.status_code == 200:
        catalogs = r_cat.json().get("catalogs", [])
        print(f"Total Catalogs: {len(catalogs)}")
        for c in catalogs:
            print(f"  * Catalog: '{c.get('name')}' (Owner: {c.get('owner')})")
    else:
        print(f"  Unity Catalog Notice: {r_cat.text[:120]}")

    print("\n" + "=" * 60)
    print("DATABRICKS LIVE QUERY COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    query_databricks_live()
