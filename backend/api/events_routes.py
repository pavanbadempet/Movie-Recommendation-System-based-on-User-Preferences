import logging
import os

from fastapi import APIRouter, BackgroundTasks
import httpx
from pydantic import BaseModel

from backend.intelligence.contextual_bandit import get_bandit_engine

router = APIRouter(prefix="/events", tags=["events"])
logger = logging.getLogger(__name__)

# The Databricks Zerobus Ingest REST endpoint URL
# Example: https://<databricks-instance>/api/2.0/streaming/ingest/zerobus
ZEROBUS_REST_URL = os.getenv("DATABRICKS_ZEROBUS_URL")
ZEROBUS_TOKEN = os.getenv("DATABRICKS_TOKEN")


class UserInteractionEvent(BaseModel):
    user_id: str
    movie_id: str
    interaction_type: str  # e.g., "click", "like", "watch", "hover"
    timestamp: str
    metadata: dict | None = {}


async def send_to_zerobus(event: UserInteractionEvent):
    """
    Background task to stream the event to Databricks Zerobus via REST.
    This enables real-time Kappa streaming without Kafka.
    """
    if not ZEROBUS_REST_URL or not ZEROBUS_TOKEN:
        logger.warning("Databricks Zerobus URL or Token not configured. Skipping ingest.")
        return

    headers = {"Authorization": f"Bearer {ZEROBUS_TOKEN}", "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(ZEROBUS_REST_URL, json=event.dict(), headers=headers, timeout=5.0)
            response.raise_for_status()
            logger.info(f"Successfully streamed {event.interaction_type} event to Databricks.")
        except Exception as e:
            logger.error(f"Failed to stream event to Zerobus: {e}")


@router.post("/ingest")
async def ingest_event(event: UserInteractionEvent, background_tasks: BackgroundTasks):
    """
    Webhook for the Hugging Face / Flutter UI to send user interactions.
    Streams directly to Databricks Zerobus for real-time processing.
    """
    # 1. Update the in-memory Contextual Bandit (Reinforcement Learning) instantly
    if event.interaction_type in ["click", "like"]:
        try:
            get_bandit_engine().update_reward(int(event.movie_id), clicked=(event.interaction_type == "like"))
        except ValueError:
            pass  # ignore invalid movie ids

    # 2. Delegate to a background task so the UI doesn't block waiting for Databricks
    background_tasks.add_task(send_to_zerobus, event)
    return {"status": "accepted", "message": "Event processed for ML Bandits and queued for Databricks ingestion."}
