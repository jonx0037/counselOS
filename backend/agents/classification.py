"""
Classification Agent
─────────────────────
Determines the matter type and produces a concise matter summary.
Uses the normalized description from the Intake Agent.
"""
import json
import re

from agents.base import BaseAgent
from models.matter import MatterType
from orchestrator.state import MatterState

SYSTEM = """You are a legal matter classification specialist.
Classify legal matters accurately and concisely.
Respond ONLY with valid JSON — no preamble, no markdown fences."""

MATTER_TYPES = ", ".join([t.value for t in MatterType if t != MatterType.UNKNOWN])

PROMPT_TEMPLATE = """
Classify the following corporate legal matter.

NORMALIZED DESCRIPTION:
{description}

EXTRACTED ENTITIES:
- Parties: {parties}
- Referenced Instruments: {instruments}
- Jurisdiction Notes: {jurisdiction_notes}

Valid matter types: {matter_types}

Return a JSON object with these exact keys:
{{
  "matter_type": "one of the valid matter types above",
  "confidence": 0.0 to 1.0 (your classification confidence),
  "matter_summary": "A 2-3 sentence summary of the matter suitable for attorney review"
}}
"""


class ClassificationAgent(BaseAgent):
    def run(self, state: MatterState) -> MatterState:
        entities = state.extracted_entities
        prompt = PROMPT_TEMPLATE.format(
            description=state.normalized_description or state.submission.matter_description,
            parties=", ".join(entities.get("parties", [])) or "not identified",
            instruments=", ".join(entities.get("referenced_instruments", [])) or "none",
            jurisdiction_notes=entities.get("jurisdiction_notes", "none"),
            matter_types=MATTER_TYPES,
        )

        response = self.llm.complete(prompt, system=SYSTEM)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)
            raw_type = parsed.get("matter_type", "unknown").lower().replace(" ", "_")
            try:
                state.matter_type = MatterType(raw_type)
            except ValueError:
                state.matter_type = MatterType.UNKNOWN

            state.matter_type_confidence = float(parsed.get("confidence", 0.5))
            state.matter_summary = parsed.get("matter_summary", "")
        except (json.JSONDecodeError, ValueError):
            state.matter_type = MatterType.UNKNOWN
            state.matter_type_confidence = 0.0
            state.errors.append("ClassificationAgent: JSON parse failed")

        state.record_agent(
            "ClassificationAgent",
            tokens_used=response.input_tokens + response.output_tokens,
            model=response.model,
        )
        return state
