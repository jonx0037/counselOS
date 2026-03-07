"""
Risk Agent
───────────
Assesses matter complexity and urgency. Flags potential
conflicts of interest, regulatory exposure, and time sensitivity.
"""
import json
import re

from agents.base import BaseAgent
from models.matter import RiskFlags, UrgencyLevel
from orchestrator.state import MatterState

SYSTEM = """You are a senior legal risk analyst at a corporate law firm.
Assess legal matters for risk, urgency, and complexity.
Respond ONLY with valid JSON — no preamble, no markdown fences."""

PROMPT_TEMPLATE = """
Assess the risk profile of this corporate legal matter.

MATTER TYPE: {matter_type}
MATTER SUMMARY: {summary}
CLIENT DEADLINE NOTE: {deadline}
JURISDICTION: {jurisdiction}

RELEVANT POLICY CONTEXT:
{context}

Return a JSON object with these exact keys:
{{
  "urgency": "critical | high | standard | low",
  "risk_score": 0.0 to 10.0 (overall complexity/risk score),
  "conflict_of_interest": true or false,
  "jurisdiction_complexity": true or false,
  "regulatory_exposure": true or false,
  "time_sensitivity": true or false,
  "risk_notes": ["list of specific risk observations, each as a short string"]
}}

Urgency guide:
  critical = deadline < 24 hours or statute of limitations imminent
  high = deadline < 3 days or significant financial exposure
  standard = normal intake queue
  low = no deadline stated, routine matter
"""


class RiskAgent(BaseAgent):
    def run(self, state: MatterState) -> MatterState:
        context_str = "\n".join(
            f"- {doc}" for doc in state.retrieved_context
        ) or "No relevant context retrieved."

        prompt = PROMPT_TEMPLATE.format(
            matter_type=state.matter_type.value if state.matter_type else "unknown",
            summary=state.matter_summary or state.submission.matter_description[:300],
            deadline=state.submission.deadline or "not specified",
            jurisdiction=state.submission.jurisdiction or "not specified",
            context=context_str,
        )

        response = self.llm.complete(prompt, system=SYSTEM)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)

            urgency_raw = parsed.get("urgency", "standard").lower()
            try:
                state.urgency = UrgencyLevel(urgency_raw)
            except ValueError:
                state.urgency = UrgencyLevel.STANDARD

            state.risk_score = float(parsed.get("risk_score", 5.0))
            state.risk_flags = RiskFlags(
                conflict_of_interest=bool(parsed.get("conflict_of_interest", False)),
                jurisdiction_complexity=bool(parsed.get("jurisdiction_complexity", False)),
                regulatory_exposure=bool(parsed.get("regulatory_exposure", False)),
                time_sensitivity=bool(parsed.get("time_sensitivity", False)),
                notes=parsed.get("risk_notes", []),
            )
        except (json.JSONDecodeError, ValueError):
            state.urgency = UrgencyLevel.STANDARD
            state.risk_score = 5.0
            state.errors.append("RiskAgent: JSON parse failed, using defaults")

        state.record_agent(
            "RiskAgent",
            tokens_used=response.input_tokens + response.output_tokens,
            model=response.model,
        )
        return state
