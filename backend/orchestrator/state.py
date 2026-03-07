"""
MatterState: the shared state object that flows through the agent pipeline.

The orchestrator owns this object. Each agent receives it, adds its output
fields, and returns the updated state. Agents never communicate directly.
"""
from typing import Any, Optional
from pydantic import BaseModel, Field

from models.matter import (
    AssignmentTier,
    MatterSubmission,
    MatterType,
    RiskFlags,
    UrgencyLevel,
)


class MatterState(BaseModel):
    """Mutable pipeline state. Grows as each agent adds its output."""

    # ── Input (set at pipeline entry) ─────────────────────────────────────────
    submission: MatterSubmission
    matter_id: str

    # ── Intake Agent output ───────────────────────────────────────────────────
    normalized_description: Optional[str] = None
    extracted_entities: dict[str, Any] = Field(default_factory=dict)

    # ── Classification Agent output ───────────────────────────────────────────
    matter_type: Optional[MatterType] = None
    matter_type_confidence: float = 0.0
    matter_summary: Optional[str] = None

    # ── RAG Agent output ──────────────────────────────────────────────────────
    retrieved_context: list[str] = Field(default_factory=list)

    # ── Risk Agent output ─────────────────────────────────────────────────────
    urgency: Optional[UrgencyLevel] = None
    risk_score: float = 0.0
    risk_flags: RiskFlags = Field(default_factory=RiskFlags)

    # ── Response Agent output ─────────────────────────────────────────────────
    recommended_tier: Optional[AssignmentTier] = None
    intake_summary: Optional[str] = None
    suggested_next_steps: list[str] = Field(default_factory=list)

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    agents_run: list[str] = Field(default_factory=list)
    total_tokens_used: int = 0
    llm_model: str = ""
    errors: list[str] = Field(default_factory=list)

    def record_agent(self, agent_name: str, tokens_used: int = 0, model: str = "") -> None:
        self.agents_run.append(agent_name)
        self.total_tokens_used += tokens_used
        if model:
            self.llm_model = model
