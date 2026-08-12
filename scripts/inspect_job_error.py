import os
import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}

# Get latest run for Job 303494851952917
r = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/list?job_id=303494851952917&limit=1", headers=HEADERS)
runs = r.json().get("runs", [])
if not runs:
    print("No runs found!")
    exit(0)

latest_run = runs[0]
run_id = latest_run.get("run_id")

# Fetch full run object including tasks array
full_run_res = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id={run_id}", headers=HEADERS)
full_run = full_run_res.json()
state = full_run.get("state", {})
print(f"LATEST RUN ID: {run_id} | Life Cycle: {state.get('life_cycle_state')} | Result: {state.get('result_state')}")

tasks = full_run.get("tasks", [])

for t in tasks:
    task_run_id = t.get("run_id")
    task_key = t.get("task_key")
    t_state = t.get("state", {})
    life = t_state.get('life_cycle_state')
    result = t_state.get('result_state')
    print(f"TASK: {task_key:<25} | LifeCycle: {life:<15} | Result: {result}")
    out_res = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get-output?run_id={task_run_id}", headers=HEADERS)
    out_data = out_res.json()
    if out_data.get("error"):
        print(f"   [ERROR] {out_data.get('error')}")
