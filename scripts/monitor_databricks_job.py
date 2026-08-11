import requests
import json
import time

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}
RUN_ID = 594775582600194

def monitor_run():
    print(f"Monitoring Databricks Job Run {RUN_ID}...")
    res = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get", headers=HEADERS, params={"run_id": RUN_ID})
    if res.status_code == 200:
        data = res.json()
        state = data.get("state", {})
        life_cycle = state.get("life_cycle_state")
        result_state = state.get("result_state")
        state_message = state.get("state_message", "")
        execution_duration = data.get("execution_duration", 0) / 1000.0

        print(f"\n==========================================")
        print(f" Databricks Serverless Run ID: {RUN_ID}")
        print(f" Life Cycle State: {life_cycle}")
        print(f" Result State: {result_state if result_state else 'IN_PROGRESS'}")
        print(f" Elapsed Time: {execution_duration:.1f} seconds")
        print(f" Message: {state_message}")
        print(f"==========================================\n")
    else:
        print(f"Error fetching status: {res.status_code} - {res.text}")

if __name__ == "__main__":
    monitor_run()
