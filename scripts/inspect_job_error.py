import os
import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}

run_id = 524994646162731
r = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id={run_id}", headers=HEADERS)
data = r.json()
tasks = data.get("tasks", [])

for t in tasks:
    task_run_id = t.get("run_id")
    task_key = t.get("task_key")
    print(f"\n==================================================")
    print(f"TASK: {task_key} (Task Run ID: {task_run_id})")
    print(f"==================================================")
    out_res = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get-output?run_id={task_run_id}", headers=HEADERS)
    out_data = out_res.json()
    print("ERROR MSG:", out_data.get("error"))
    print("ERROR TRACE:", out_data.get("error_trace"))
    if out_data.get("notebook_output"):
        print("NOTEBOOK OUTPUT:", out_data.get("notebook_output"))
