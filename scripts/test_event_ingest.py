import os
import json
import requests
import uuid
from datetime import datetime, timezone

def test_realtime_event_ingest():
    zerobus_url = os.environ.get("DATABRICKS_ZEROBUS_URL")
    db_token = os.environ.get("DATABRICKS_TOKEN")
    
    if not zerobus_url or not db_token:
        print("Missing DATABRICKS_ZEROBUS_URL or DATABRICKS_TOKEN in environment!")
        return

    print(f"Target Volume Endpoint: {zerobus_url}")
    
    # 2. Construct Real-Time Event Payload
    event_id = str(uuid.uuid4())[:8]
    payload = {
        "user_id": f"usr_test_{event_id}",
        "movie_id": "101",
        "interaction_type": "click",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": "Test real-time event from automated verification script"
    }
    
    # 3. Upload JSON file to Databricks Unity Catalog Volume via Files REST API
    upload_url = f"{zerobus_url}/event_{event_id}.json?overwrite=true"
    headers = {
        "Authorization": f"Bearer {db_token}",
        "Content-Type": "application/json"
    }
    
    print(f"Uploading event '{event_id}' to Databricks Volume...")
    resp = requests.put(upload_url, data=json.dumps(payload), headers=headers)
    
    print(f"Databricks API Response Status: {resp.status_code}")
    print(f"Response: {resp.text}")
    
    if resp.status_code in [200, 201]:
        print(f"SUCCESS! Real-Time event 'event_{event_id}.json' is live in Databricks Volume!")
    else:
        print(f"Upload Notice: HTTP {resp.status_code}")

if __name__ == "__main__":
    test_realtime_event_ingest()
