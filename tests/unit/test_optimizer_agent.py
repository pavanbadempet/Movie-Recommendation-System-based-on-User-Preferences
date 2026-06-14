from datetime import UTC, datetime
import os
import tempfile
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.agents.base_agent import BaseAgent
from backend.agents.recommender_optimizer_agent import RecommenderOptimizerAgent
from backend.data.database import Base, Tenant, User, UserEvent


# Setup in-memory SQLite database for testing
@pytest.fixture(name="db_session")
def fixture_db_session():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_base_agent_telemetry():
    """Verify BaseAgent metrics tracking and steps logs."""
    agent = BaseAgent(name="Base Test Agent")
    assert agent.status == "initialized"

    agent.start()
    assert agent.status == "running"

    agent.log_step("Step Action", "Step Result")
    assert len(agent.steps) == 2

    agent.estimate_tokens("Input content context", is_output=False)
    agent.estimate_tokens("Output recommendation parameters", is_output=True)
    assert agent.input_tokens_estimated > 0
    assert agent.output_tokens_estimated > 0

    agent.finish()
    assert agent.status == "completed"
    assert agent.duration >= 0.0


def test_base_agent_github_actions():
    """Verify BaseAgent GHA summaries and variables outputting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_file = os.path.join(tmpdir, "summary.md")
        output_file = os.path.join(tmpdir, "outputs.txt")

        custom_env = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_STEP_SUMMARY": summary_file,
            "GITHUB_OUTPUT": output_file,
        }

        with patch.dict(os.environ, custom_env):
            agent = BaseAgent("GHA Movie Agent")
            agent.start()
            agent.estimate_tokens("Input context description", is_output=False)
            agent.estimate_tokens("Output hyperparameter list", is_output=True)
            agent.finish()

            # Verify step summary written
            assert os.path.exists(summary_file)
            with open(summary_file, encoding="utf-8") as f:
                content = f.read()
                assert "GHA Movie Agent" in content

            # Verify outputs written
            assert os.path.exists(output_file)
            with open(output_file, encoding="utf-8") as f:
                lines = f.read().splitlines()
                assert "agent_status=completed" in lines


@pytest.mark.asyncio
async def test_optimizer_agent_no_data(db_session):
    """Verify RecommenderOptimizerAgent behavior on empty database."""
    agent = RecommenderOptimizerAgent(db_session, "Empty DB Optimizer")
    report, report_json = await agent.run(hours=24, dry_run=True)

    assert "Recommender System Optimization Report" in report
    assert "Click-Through Rate (CTR)" in report
    assert agent.status == "completed"
    assert len(agent.steps) == 6  # Init, Fetch, Calculate, Drift, LLM, Shutdown
    assert report_json["metrics"]["ctr"] == 0.0
    assert report_json["metrics"]["drift_detected"] is True


@pytest.mark.asyncio
async def test_optimizer_agent_with_data(db_session):
    """Verify RecommenderOptimizerAgent CTR calculations and report metrics."""
    # Seed Tenant
    tenant = Tenant(tenant_id="11111111-1111-1111-1111-111111111111", company_name="Test Co", plan_tier="pro")
    db_session.add(tenant)
    db_session.commit()

    # Seed User
    user = User(user_sk="22222222-2222-2222-2222-222222222222", tenant_id="11111111-1111-1111-1111-111111111111", external_user_id="ext-123")
    db_session.add(user)
    db_session.commit()

    # Seed mock events: 10 recommendations served, 2 clicks -> 20% CTR
    for _ in range(10):
        db_session.add(
            UserEvent(
                tenant_id="11111111-1111-1111-1111-111111111111",
                user_sk="22222222-2222-2222-2222-222222222222",
                event_type="recommendation_served",
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )

    for _ in range(2):
        db_session.add(
            UserEvent(
                tenant_id="11111111-1111-1111-1111-111111111111",
                user_sk="22222222-2222-2222-2222-222222222222",
                event_type="click",
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )

    db_session.commit()

    # Run agent
    agent = RecommenderOptimizerAgent(db_session, "Active DB Optimizer")
    report, report_json = await agent.run(hours=24, dry_run=True)

    assert "Recommender System Optimization Report" in report
    assert "20.00%" in report  # CTR
    assert "[HEURISTIC LOCAL ASSESSMENT" in report
    assert "SASRec=0.45" in report
    assert agent.status == "completed"
    assert agent.input_tokens_estimated > 0
    assert agent.output_tokens_estimated > 0
    assert report_json["metrics"]["ctr"] == 0.20
    assert report_json["metrics"]["drift_detected"] is False
    assert report_json["suggested_hyperparameters"]["ensemble_weights"]["sasrec"] == 0.45
