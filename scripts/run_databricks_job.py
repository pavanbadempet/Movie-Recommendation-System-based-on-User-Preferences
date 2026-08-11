import requests
import json
import time

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

def submit_notebook_job():
    print("Submitting Notebook 02 Execution Run to Databricks Serverless via API...")
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/submit"
    
    payload = {
        "run_name": "Multi-Shard Neon Vector Export Run",
        "tasks": [
            {
                "task_key": "export_to_neon_task",
                "notebook_task": {
                    "notebook_path": "/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/02_export_to_neon",
                    "source": "WORKSPACE"
                }
            }
        ]
    }

    res = requests.post(url, headers=HEADERS, json=payload)
    print(f"Submit Job Status: {res.status_code} - {res.text}")
    if res.status_code == 200:
        run_id = res.json().get("run_id")
        print(f"Job Run submitted successfully! Run ID: {run_id}")
        
        # Check run status
        for _ in range(5):
            time.sleep(3)
            status_res = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get", headers=HEADERS, params={"run_id": run_id})
            if status_res.status_code == 200:
                s_data = status_res.json()
                state = s_data.get("state", {})
                life_cycle = state.get("life_cycle_state")
                result_state = state.get("result_state")
                print(f"Run {run_id} Life Cycle: {life_cycle} | Result: {result_state}")
                if life_cycle in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                    break

if __name__ == "__main__":
    submit_notebook_job()
