"""
Intake Agent
─────────────
Normalizes the raw matter submission and extracts structured entities
(parties, dates, jurisdictions, referenced instruments) before the
classification stage.
"""
import json
import re

from agents.base import BaseAgent
from orchestrator.state import MatterState

SYSTEM = """You are a legal intake specialist at a corporate law firm.
Your job is to normalize and structure incoming matter submissions.
Respond ONLY with valid JSON — no preamble, no markdown fences."""

PROMPT_TEMPLATE = """
Analyze the following legal matter submission and extract structured information.

CLIENT: {client_name}
SUBMITTED BY: {submitted_by}
JURISDICTION: {jurisdiction}
DEADLINE NOTE: {deadline}
DESCRIPTION:
{description}

Return a JSON object with these exact keys:
{{
  "normalized_description": "A clear, professionally rewritten version of the description (2-4 sentences)",
  "parties": ["list of identified parties / entities"],
  "key_dates": ["list of mentioned dates or timeframes"],
  "referenced_instruments": ["contracts, statutes, or legal instruments mentioned"],
  "jurisdiction_notes": "any jurisdiction-specific observations or 'none identified'"
}}
"""


class IntakeAgent(BaseAgent):
    def run(self, state: MatterState) -> MatterState:
        s = state.submission
        prompt = PROMPT_TEMPLATE.format(
            client_name=s.client_name,
            submitted_by=s.submitted_by,
            jurisdiction=s.jurisdiction or "not specified",
            deadline=s.deadline or "not specified",
            description=s.matter_description,
        )

        response = self.llm.complete(prompt, system=SYSTEM)
        raw = response.content.strip()

        # Strip markdown fences if the model includes them despite instructions
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)
            state.normalized_description = parsed.get("normalized_description", s.matter_description)
            state.extracted_entities = {
                "parties": parsed.get("parties", []),
                "key_dates": parsed.get("key_dates", []),
                "referenced_instruments": parsed.get("referenced_instruments", []),
                "jurisdiction_notes": parsed.get("jurisdiction_notes", ""),
            }
        except json.JSONDecodeError:
            # Degrade gracefully — downstream agents use original description
            state.normalized_description = s.matter_description
            state.errors.append("IntakeAgent: JSON parse failed, using raw description")

        state.record_agent(
            "IntakeAgent",
            tokens_used=response.input_tokens + response.output_tokens,
            model=response.model,
        )
        return state
