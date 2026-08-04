"""Unit tests for Multi-Agent Orchestrator Framework and Agentic AI reasoning."""

import pytest

from backend.agents.multi_agent_orchestrator import AgenticRecommendationTask, MultiAgentOrchestrator


@pytest.mark.asyncio
async def test_agentic_orchestrator_execution():
    orchestrator = MultiAgentOrchestrator()
    task = await orchestrator.execute_task("Find me mind-bending sci-fi movies like Inception")

    assert isinstance(task, AgenticRecommendationTask)
    assert len(task.trace) >= 3
    assert len(task.final_recommendations) > 0
    assert "Agentic AI" in task.explanation


@pytest.mark.asyncio
async def test_agentic_reasoning_intent():
    orchestrator = MultiAgentOrchestrator()
    intent = await orchestrator.reasoning_agent.analyze_intent(
        AgenticRecommendationTask(query="recent action thrillers")
    )

    assert intent["requires_action"] is True
    assert intent["requires_recency"] is True
