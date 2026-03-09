"""
Unit tests for CounselOS pipeline models and state management.
These tests run without requiring an LLM API key or ChromaDB.
"""
from datetime import datetime

import pytest

from models.matter import (
    AssignmentTier,
    IntakeResult,
    MatterSubmission,
    MatterType,
    RiskFlags,
    UrgencyLevel,
)
from orchestrator.state import MatterState
from models.matter import ChatMessage, ChatRequest, ChatResponse


class TestChatModels:
    def test_chat_message_user_role(self) -> None:
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_rejects_invalid_role(self) -> None:
        with pytest.raises(ValueError):
            ChatMessage(role="admin", content="Hello")

    def test_chat_request_requires_matter_id_and_message(self) -> None:
        req = ChatRequest(
            matter_id="MATTER-TEST0001",
            message="Why is my risk score 7?",
            conversation_history=[],
        )
        assert req.matter_id == "MATTER-TEST0001"
        assert req.message == "Why is my risk score 7?"
        assert req.conversation_history == []

    def test_chat_response_has_reply_and_sources(self) -> None:
        resp = ChatResponse(reply="The risk is high because...", sources_used=2)
        assert resp.reply == "The risk is high because..."
        assert resp.sources_used == 2


@pytest.fixture
def sample_submission() -> MatterSubmission:
    return MatterSubmission(
        client_name="Acme Corporation",
        submitted_by="Jane Smith",
        matter_description=(
            "Acme Corporation requires review of a software licensing agreement "
            "with a third-party SaaS vendor. The contract includes auto-renewal "
            "clauses and a data processing agreement under GDPR."
        ),
        jurisdiction="Delaware",
        deadline="Response required within 5 business days",
    )


@pytest.fixture
def initial_state(sample_submission: MatterSubmission) -> MatterState:
    return MatterState(submission=sample_submission, matter_id="MATTER-TEST0001")


# ── MatterState tests ──────────────────────────────────────────────────────────

class TestMatterState:
    def test_initial_state_has_empty_agents_run(self, initial_state: MatterState) -> None:
        assert initial_state.agents_run == []

    def test_initial_state_has_zero_tokens(self, initial_state: MatterState) -> None:
        assert initial_state.total_tokens_used == 0

    def test_record_agent_appends_name(self, initial_state: MatterState) -> None:
        initial_state.record_agent("IntakeAgent", tokens_used=150, model="claude-sonnet-4-6")
        assert "IntakeAgent" in initial_state.agents_run

    def test_record_agent_accumulates_tokens(self, initial_state: MatterState) -> None:
        initial_state.record_agent("IntakeAgent", tokens_used=150)
        initial_state.record_agent("ClassificationAgent", tokens_used=200)
        assert initial_state.total_tokens_used == 350

    def test_record_agent_sets_model(self, initial_state: MatterState) -> None:
        initial_state.record_agent("IntakeAgent", model="claude-sonnet-4-6")
        assert initial_state.llm_model == "claude-sonnet-4-6"

    def test_errors_start_empty(self, initial_state: MatterState) -> None:
        assert initial_state.errors == []


# ── MatterSubmission validation tests ─────────────────────────────────────────

class TestMatterSubmission:
    def test_valid_submission_creates_ok(self, sample_submission: MatterSubmission) -> None:
        assert sample_submission.client_name == "Acme Corporation"

    def test_description_too_short_raises(self) -> None:
        with pytest.raises(ValueError):
            MatterSubmission(
                client_name="Test",
                submitted_by="Test",
                matter_description="Too short",  # < 20 chars
            )

    def test_submitted_at_defaults_to_now(self, sample_submission: MatterSubmission) -> None:
        assert isinstance(sample_submission.submitted_at, datetime)

    def test_optional_fields_default_none(self) -> None:
        sub = MatterSubmission(
            client_name="Test Corp",
            submitted_by="Test User",
            matter_description="A sufficiently long matter description for testing purposes.",
        )
        assert sub.jurisdiction is None
        assert sub.deadline is None


# ── Enum coverage ──────────────────────────────────────────────────────────────

class TestEnums:
    def test_all_matter_types_are_valid(self) -> None:
        expected = {
            "contract", "litigation", "mergers_acquisitions", "employment",
            "intellectual_property", "compliance", "real_estate",
            "corporate_governance", "unknown",
        }
        assert {t.value for t in MatterType} == expected

    def test_urgency_levels(self) -> None:
        assert UrgencyLevel("critical") == UrgencyLevel.CRITICAL
        assert UrgencyLevel("low") == UrgencyLevel.LOW

    def test_assignment_tiers(self) -> None:
        assert AssignmentTier("partner") == AssignmentTier.PARTNER
        assert AssignmentTier("paralegal") == AssignmentTier.PARALEGAL


# ── RiskFlags tests ────────────────────────────────────────────────────────────

class TestRiskFlags:
    def test_default_flags_are_false(self) -> None:
        flags = RiskFlags()
        assert not flags.conflict_of_interest
        assert not flags.jurisdiction_complexity
        assert not flags.regulatory_exposure
        assert not flags.time_sensitivity

    def test_notes_default_empty(self) -> None:
        assert RiskFlags().notes == []
