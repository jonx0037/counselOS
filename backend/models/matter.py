from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MatterType(str, Enum):
    CONTRACT = "contract"
    LITIGATION = "litigation"
    MERGERS_ACQUISITIONS = "mergers_acquisitions"
    EMPLOYMENT = "employment"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    COMPLIANCE = "compliance"
    REAL_ESTATE = "real_estate"
    CORPORATE_GOVERNANCE = "corporate_governance"
    UNKNOWN = "unknown"


class UrgencyLevel(str, Enum):
    CRITICAL = "critical"   # < 24 hours
    HIGH = "high"           # < 3 days
    STANDARD = "standard"  # standard queue
    LOW = "low"             # non-urgent


class AssignmentTier(str, Enum):
    PARTNER = "partner"
    SENIOR_ASSOCIATE = "senior_associate"
    ASSOCIATE = "associate"
    PARALEGAL = "paralegal"


# ── Input ──────────────────────────────────────────────────────────────────────

class MatterSubmission(BaseModel):
    """Raw client matter submission — the pipeline entry point."""
    client_name: str = Field(..., description="Client or company name")
    submitted_by: str = Field(..., description="Contact person submitting the matter")
    matter_description: str = Field(
        ...,
        min_length=20,
        description="Free-text description of the legal matter",
    )
    jurisdiction: Optional[str] = Field(None, description="Relevant jurisdiction, if known")
    deadline: Optional[str] = Field(None, description="Client-stated deadline or urgency note")
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


# ── Output ─────────────────────────────────────────────────────────────────────

class RiskFlags(BaseModel):
    conflict_of_interest: bool = False
    jurisdiction_complexity: bool = False
    regulatory_exposure: bool = False
    time_sensitivity: bool = False
    notes: list[str] = Field(default_factory=list)


class IntakeResult(BaseModel):
    """Structured output from the full CounselOS pipeline."""

    # Identity
    matter_id: str
    client_name: str
    submitted_by: str
    submitted_at: datetime

    # Classification
    matter_type: MatterType
    matter_type_confidence: float = Field(ge=0.0, le=1.0)
    matter_summary: str

    # RAG context used
    retrieved_context: list[str] = Field(default_factory=list)

    # Risk
    urgency: UrgencyLevel
    risk_score: float = Field(ge=0.0, le=10.0)
    risk_flags: RiskFlags

    # Recommendation
    recommended_tier: AssignmentTier
    intake_summary: str
    suggested_next_steps: list[str]

    # Pipeline metadata
    agents_run: list[str] = Field(default_factory=list)
    total_tokens_used: int = 0
    llm_model: str = ""
