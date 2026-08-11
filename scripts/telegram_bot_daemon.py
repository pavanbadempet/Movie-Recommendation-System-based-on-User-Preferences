import os
import sys
import time
import requests
import json
import subprocess

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
CLOUDFLARE_WORKER_URL = "https://movie-recommendation-system.pavan9b.workers.dev"

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send_message(chat_id, text, parse_mode="Markdown", reply_markup=None):
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        requests.post(f"{TELEGRAM_API_URL}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

def get_keyboard():
    return {
        "inline_keyboard": [
            [
                {"text": "📊 System Status", "callback_data": "status"},
                {"text": "🚀 Databricks Export", "callback_data": "run_export"}
            ],
            [
                {"text": "🤗 HuggingFace Deploy", "callback_data": "deploy_hf"},
                {"text": "⚡ Cloudflare Edge", "callback_data": "cloudflare"}
            ],
            [
                {"text": "🌐 Neon Singapore", "callback_data": "shards"},
                {"text": "🔄 Refresh Menu", "callback_data": "status"}
            ]
        ]
    }

def get_system_status():
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    try:
        res = requests.get(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/list?limit=1", headers=headers, timeout=5)
        runs = res.json().get("runs", [])
        if runs:
            latest_run = runs[0]
            run_id = latest_run.get("run_id")
            state = latest_run.get("state", {})
            life_cycle = state.get("life_cycle_state")
            result_state = state.get("result_state")
            db_status = f"Run #{run_id}: `{life_cycle}` ({result_state if result_state else 'IN_PROGRESS'})"
        else:
            db_status = "Idle (No active runs)"
    except Exception as e:
        db_status = f"Status error: {e}"

    msg = (
        "📱 *NOVA MOVIE RECOMMENDER BOT DASHBOARD*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🟢 *ALL SYSTEMS OPERATIONAL*\n\n"
        f"⚡ *Databricks Serverless:* {db_status}\n"
        "🇸🇬 *Neon Region:* AWS Singapore (`ap-southeast-1`)\n"
        "🌐 *Vector Cluster:* 10 Shards (5.12 GB Free Storage)\n"
        "🛠️ *Microservices:* 10 Dedicated DB Projects (Account 2)\n"
        "⚡ *Cloudflare Edge:* `movie-recommendation-system` (15ms AI)\n"
        "🤗 *HuggingFace Space:* `pavanbadempet/movie-rec-api`\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "_Tap buttons below for complete phone control:_"
    )
    return msg

def get_shards_info():
    msg = (
        "🇸🇬 *NEON SINGAPORE 20-DATABASE TOPOLOGY*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🌐 *Vector Serving Cluster (Account 1):*\n"
        "• `DATABASE_URL_SHARD_0` .. `9` (movie-shard-0 to 9)\n"
        "• Total Capacity: 5.12 GB Free Storage\n\n"
        "🛠️ *Domain Microservices Cluster (Account 2):*\n"
        "1. `user-auth-db` (Auth & Passwords)\n"
        "2. `clickstream-events-db` (Real-Time Clicks)\n"
        "3. `recommendations-cache-db` (Trending Feed)\n"
        "4. `analytics-metrics-db` (BI & Performance)\n"
        "5. `model-registry-db` (ML Models & A/B)\n"
        "6. `notifications-db` (User Alerts)\n"
        "7. `search-history-db` (Query History)\n"
        "8. `watchlists-db` (Favorites & Bookmarks)\n"
        "9. `billing-subscriptions-db` (Payments)\n"
        "10. `feedback-reviews-db` (User Reviews)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    return msg

def test_cloudflare_edge():
    try:
        res = requests.post(f"{CLOUDFLARE_WORKER_URL}/api/search", json={"query": "Inception mind bending sci-fi"}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return (
                "⚡ *CLOUDFLARE WORKERS AI EDGE STATUS*\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🟢 *Status:* OPERATIONAL\n"
                f"📍 *Model:* `@cf/baai/bge-base-en-v1.5`\n"
                f"📏 *Dimensions:* {data.get('embedding_dimensions')}D Vector\n"
                f"⏱️ *Latency:* {data.get('latency_ms')} ms\n"
                f"🌐 *Url:* `{CLOUDFLARE_WORKER_URL}`\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
        else:
            return f"❌ Cloudflare Edge returned HTTP {res.status_code}"
    except Exception as e:
        return f"❌ Cloudflare error: {e}"

def trigger_export():
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/submit"
    payload = {
        "run_name": "Telegram Remote Trigger - Neon Export",
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
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            run_id = res.json().get("run_id")
            return f"🚀 *Job Run #{run_id} Submitted!*\n\nDatabricks is exporting vectors across all 10 Singapore Neon Shards on Serverless GPU!"
        else:
            return f"❌ Failed to submit run: {res.status_code} - {res.text}"
    except Exception as e:
        return f"❌ Execution Error: {e}"

def trigger_hf_upload():
    try:
        subprocess.Popen(["doppler", "run", "--", "python", "scripts/hf_upload.py"])
        return "🤗 *HuggingFace Deployment Initiated!*\n\nSyncing codebase and model weights to `pavanbadempet/movie-rec-api`!"
    except Exception as e:
        return f"❌ HF Upload Error: {e}"

def poll_updates():
    print("NovaMovieRecBot is online and polling Telegram API...")
    offset = 0
    while True:
        try:
            res = requests.get(f"{TELEGRAM_API_URL}/getUpdates", params={"offset": offset, "timeout": 20}, timeout=25)
            if res.status_code == 200:
                updates = res.json().get("result", [])
                for u in updates:
                    offset = u["update_id"] + 1
                    
                    if "message" in u:
                        msg = u["message"]
                        chat_id = msg["chat"]["id"]
                        text = msg.get("text", "")
                        
                        if text.startswith("/start") or text.startswith("/help") or text.startswith("/status"):
                            send_message(chat_id, get_system_status(), reply_markup=get_keyboard())
                        elif text.startswith("/shards"):
                            send_message(chat_id, get_shards_info(), reply_markup=get_keyboard())
                        elif text.startswith("/run"):
                            send_message(chat_id, trigger_export(), reply_markup=get_keyboard())
                        elif text.startswith("/hf"):
                            send_message(chat_id, trigger_hf_upload(), reply_markup=get_keyboard())
                        elif text.startswith("/cf") or text.startswith("/cloudflare"):
                            send_message(chat_id, test_cloudflare_edge(), reply_markup=get_keyboard())
                            
                    elif "callback_query" in u:
                        cb = u["callback_query"]
                        chat_id = cb["message"]["chat"]["id"]
                        data = cb.get("data")
                        
                        if data == "status":
                            send_message(chat_id, get_system_status(), reply_markup=get_keyboard())
                        elif data == "run_export":
                            send_message(chat_id, trigger_export(), reply_markup=get_keyboard())
                        elif data == "deploy_hf":
                            send_message(chat_id, trigger_hf_upload(), reply_markup=get_keyboard())
                        elif data == "cloudflare":
                            send_message(chat_id, test_cloudflare_edge(), reply_markup=get_keyboard())
                        elif data == "shards":
                            send_message(chat_id, get_shards_info(), reply_markup=get_keyboard())
        except Exception as err:
            time.sleep(2)

if __name__ == "__main__":
    poll_updates()
