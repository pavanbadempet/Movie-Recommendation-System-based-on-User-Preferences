import os
import requests
import json
import time

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

def trigger_and_monitor_job(job_id=538741510998535):
    print(f"Triggering Databricks Workflow Job (Job ID: {job_id})...")
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/run-now"
    res = requests.post(url, headers=HEADERS, json={"job_id": job_id})
    print(f"Trigger Status: {res.status_code} - {res.text}")

    if res.status_code == 200:
        run_id = res.json().get("run_id")
        print(f"DATABRICKS JOB RUN STARTED! RUN ID: {run_id}")
        
        status_url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id={run_id}"
        print(f"Live Monitoring Run ID {run_id}...")
        
        for _ in range(5):
            time.sleep(5)
            s_res = requests.get(status_url, headers=HEADERS)
            if s_res.status_code == 200:
                state = s_res.json().get("state", {})
                life_cycle_state = state.get("life_cycle_state")
                result_state = state.get("result_state", "PENDING")
                print(f"LifeCycle State: {life_cycle_state} | Result: {result_state}")
                if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                    break

if __name__ == "__main__":
    trigger_and_monitor_job()
