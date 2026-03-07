"""
Response Agent
───────────────
Synthesizes all upstream agent outputs into a structured intake
summary and recommended attorney assignment tier with next steps.
"""
import json
import re

from agents.base import BaseAgent
from models.matter import AssignmentTier
from orchestrator.state import MatterState

SYSTEM = """You are a senior legal intake coordinator at a corporate law firm.
Synthesize matter analysis into clear, actionable intake summaries for attorney assignment.
Respond ONLY with valid JSON — no preamble, no markdown fences."""

PROMPT_TEMPLATE = """
Produce a final intake summary and attorney assignment recommendation.

MATTER DETAILS:
- Type: {matter_type} (confidence: {confidence:.0%})
- Summary: {summary}
- Urgency: {urgency}
- Risk Score: {risk_score}/10
- Risk Flags: {risk_flags}

CLIENT INFO:
- Client: {client_name}
- Submitted by: {submitted_by}
- Jurisdiction: {jurisdiction}
- Deadline: {deadline}

RISK NOTES:
{risk_notes}

RELEVANT CONTEXT RETRIEVED:
{context}

Assignment tiers:
  partner = high-stakes, complex, regulatory risk, or urgent matters
  senior_associate = moderate complexity, standard commercial matters
  associate = routine matters, lower value, well-precedented
  paralegal = administrative, document collection, or very low complexity

Return a JSON object with these exact keys:
{{
  "recommended_tier": "partner | senior_associate | associate | paralegal",
  "intake_summary": "A clear 3-5 sentence intake summary written for the assigned attorney",
  "suggested_next_steps": ["ordered list of 3-5 specific next steps for the assigned attorney"]
}}
"""


class ResponseAgent(BaseAgent):
    def run(self, state: MatterState) -> MatterState:
        flags = state.risk_flags
        active_flags = [
            name for name, val in {
                "Conflict of Interest": flags.conflict_of_interest,
                "Jurisdiction Complexity": flags.jurisdiction_complexity,
                "Regulatory Exposure": flags.regulatory_exposure,
                "Time Sensitivity": flags.time_sensitivity,
            }.items() if val
        ]

        context_str = "\n".join(
            f"- {doc}" for doc in state.retrieved_context
        ) or "No context retrieved."

        prompt = PROMPT_TEMPLATE.format(
            matter_type=state.matter_type.value if state.matter_type else "unknown",
            confidence=state.matter_type_confidence,
            summary=state.matter_summary or state.submission.matter_description[:300],
            urgency=state.urgency.value if state.urgency else "standard",
            risk_score=state.risk_score,
            risk_flags=", ".join(active_flags) if active_flags else "none",
            client_name=state.submission.client_name,
            submitted_by=state.submission.submitted_by,
            jurisdiction=state.submission.jurisdiction or "not specified",
            deadline=state.submission.deadline or "not specified",
            risk_notes="\n".join(f"- {n}" for n in flags.notes) or "none",
            context=context_str,
        )

        response = self.llm.complete(prompt, system=SYSTEM)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)

            tier_raw = parsed.get("recommended_tier", "associate").lower()
            try:
                state.recommended_tier = AssignmentTier(tier_raw)
            except ValueError:
                state.recommended_tier = AssignmentTier.ASSOCIATE

            state.intake_summary = parsed.get("intake_summary", "")
            state.suggested_next_steps = parsed.get("suggested_next_steps", [])
        except (json.JSONDecodeError, ValueError):
            state.recommended_tier = AssignmentTier.ASSOCIATE
            state.errors.append("ResponseAgent: JSON parse failed")

        state.record_agent(
            "ResponseAgent",
            tokens_used=response.input_tokens + response.output_tokens,
            model=response.model,
        )
        return state
