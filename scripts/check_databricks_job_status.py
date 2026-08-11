import os
import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}

def check_run(run_id=392559988232409):
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id={run_id}"
    res = requests.get(url, headers=HEADERS)
    if res.status_code == 200:
        data = res.json()
        state = data.get("state", {})
        print(f"Databricks Run ID {run_id} Status:")
        print(f"LifeCycle State: {state.get('life_cycle_state')}")
        print(f"State Message: {state.get('state_message', 'Running...')}")

if __name__ == "__main__":
    check_run()
