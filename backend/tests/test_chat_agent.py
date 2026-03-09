"""Tests for the ChatAgent — uses a mock LLM provider."""
from unittest.mock import MagicMock
from datetime import datetime

import pytest

from agents.chat import ChatAgent
from core.llm.base import BaseLLMProvider, LLMResponse
from models.matter import (
    ChatMessage,
    IntakeResult,
    MatterType,
    UrgencyLevel,
    AssignmentTier,
    RiskFlags,
)


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock(spec=BaseLLMProvider)
    llm.complete.return_value = LLMResponse(
        content="The risk score is 7 because of regulatory exposure.",
        model="mock-model",
        input_tokens=100,
        output_tokens=50,
    )
    return llm


@pytest.fixture
def sample_result() -> IntakeResult:
    return IntakeResult(
        matter_id="MATTER-TEST0001",
        client_name="Acme Corp",
        submitted_by="Jane Smith",
        submitted_at=datetime.utcnow(),
        matter_type=MatterType.CONTRACT,
        matter_type_confidence=0.85,
        matter_summary="Software licensing agreement review",
        urgency=UrgencyLevel.HIGH,
        risk_score=7.0,
        risk_flags=RiskFlags(regulatory_exposure=True, notes=["GDPR implications"]),
        recommended_tier=AssignmentTier.SENIOR_ASSOCIATE,
        intake_summary="Review needed for SaaS licensing with GDPR implications.",
        suggested_next_steps=["Review DPA", "Check conflict"],
    )


class TestChatAgent:
    def test_answer_returns_reply(
        self, mock_llm: MagicMock, sample_result: IntakeResult
    ) -> None:
        agent = ChatAgent(mock_llm)
        response = agent.answer(
            message="Why is my risk score 7?",
            result=sample_result,
            conversation_history=[],
        )
        assert response.reply == "The risk score is 7 because of regulatory exposure."
        assert response.sources_used >= 0

    def test_llm_receives_system_prompt_with_result(
        self, mock_llm: MagicMock, sample_result: IntakeResult
    ) -> None:
        agent = ChatAgent(mock_llm)
        agent.answer(
            message="Explain the risk",
            result=sample_result,
            conversation_history=[],
        )
        call_args = mock_llm.complete.call_args
        system_prompt = call_args.kwargs.get("system", "") or call_args[1].get("system", "")
        assert "MATTER-TEST0001" in system_prompt
        assert "7.0" in system_prompt or "7" in system_prompt

    def test_conversation_history_included_in_prompt(
        self, mock_llm: MagicMock, sample_result: IntakeResult
    ) -> None:
        history = [
            ChatMessage(role="user", content="What is my risk?"),
            ChatMessage(role="assistant", content="Your risk score is 7."),
        ]
        agent = ChatAgent(mock_llm)
        agent.answer(
            message="Why?",
            result=sample_result,
            conversation_history=history,
        )
        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs.get("prompt", "") or call_args[0][0]
        assert "What is my risk?" in prompt
        assert "Your risk score is 7." in prompt
        assert "Why?" in prompt
