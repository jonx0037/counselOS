"""Integration tests for POST /api/chat using FastAPI TestClient."""
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app
from models.matter import (
    IntakeResult,
    MatterType,
    UrgencyLevel,
    AssignmentTier,
    RiskFlags,
)
from orchestrator.pipeline import CounselOSPipeline


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def seed_cache():
    """Put a result in the cache before each test."""
    result = IntakeResult(
        matter_id="MATTER-CHAT01",
        client_name="Test Corp",
        submitted_by="Test User",
        submitted_at=datetime.utcnow(),
        matter_type=MatterType.LITIGATION,
        matter_type_confidence=0.9,
        matter_summary="Employment dispute",
        urgency=UrgencyLevel.HIGH,
        risk_score=8.0,
        risk_flags=RiskFlags(time_sensitivity=True),
        recommended_tier=AssignmentTier.PARTNER,
        intake_summary="High-risk employment litigation.",
        suggested_next_steps=["Assign partner", "Review timeline"],
    )
    CounselOSPipeline.cache_result(result)
    yield
    CounselOSPipeline._result_cache.clear()


class TestChatEndpoint:
    @patch("main.chat_agent")
    def test_chat_returns_reply(self, mock_agent, client):
        from models.matter import ChatResponse
        mock_agent.answer.return_value = ChatResponse(
            reply="The risk is 8 because of time sensitivity.", sources_used=2
        )
        resp = client.post("/api/chat", json={
            "matter_id": "MATTER-CHAT01",
            "message": "Why is my risk high?",
            "conversation_history": [],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "The risk is 8 because of time sensitivity."
        assert data["sources_used"] == 2

    def test_chat_unknown_matter_returns_404(self, client):
        resp = client.post("/api/chat", json={
            "matter_id": "MATTER-UNKNOWN",
            "message": "Hello",
            "conversation_history": [],
        })
        assert resp.status_code == 404
