"""
CounselOS Orchestrator — custom Python state machine.

No framework dependency. The orchestrator owns the MatterState object,
runs each agent in sequence, handles per-agent errors without killing
the pipeline, and returns a final IntakeResult.

Agent execution order:
    IntakeAgent → ClassificationAgent → RAGAgent → RiskAgent → ResponseAgent
"""
import uuid
import logging
from datetime import datetime

from models.matter import (
    IntakeResult,
    MatterSubmission,
    MatterType,
    UrgencyLevel,
    AssignmentTier,
    RiskFlags,
)
from orchestrator.state import MatterState
from agents.intake import IntakeAgent
from agents.classification import ClassificationAgent
from agents.rag import RAGAgent
from agents.risk import RiskAgent
from agents.response import ResponseAgent
from core.llm import get_llm_provider

logger = logging.getLogger(__name__)


class CounselOSPipeline:
    """
    Orchestrates the full matter intake pipeline.
    Instantiate once at app startup; call run() per request.
    """

    _result_cache: dict[str, "IntakeResult"] = {}

    @staticmethod
    def cache_result(result: IntakeResult) -> None:
        CounselOSPipeline._result_cache[result.matter_id] = result

    @staticmethod
    def get_cached_result(matter_id: str) -> IntakeResult | None:
        return CounselOSPipeline._result_cache.get(matter_id)

    def __init__(self) -> None:
        llm = get_llm_provider()
        self._agents = [
            IntakeAgent(llm),
            ClassificationAgent(llm),
            RAGAgent(llm),
            RiskAgent(llm),
            ResponseAgent(llm),
        ]
        logger.info(
            "CounselOS pipeline initialised with %d agents using %s",
            len(self._agents),
            llm.model_name(),
        )

    def run(self, submission: MatterSubmission) -> IntakeResult:
        """
        Execute the full agent pipeline for a matter submission.

        Each agent is run in sequence. If an agent raises, the error is
        recorded in state.errors and the pipeline continues with degraded
        output rather than failing entirely.
        """
        state = MatterState(
            submission=submission,
            matter_id=f"MATTER-{uuid.uuid4().hex[:8].upper()}",
        )

        logger.info("Pipeline started for matter %s", state.matter_id)

        for agent in self._agents:
            agent_name = agent.__class__.__name__
            try:
                logger.debug("Running %s", agent_name)
                state = agent.run(state)
            except Exception as exc:
                error_msg = f"{agent_name} failed: {exc}"
                logger.error(error_msg, exc_info=True)
                state.errors.append(error_msg)

        logger.info(
            "Pipeline complete for %s — %d agents, %d tokens, %d errors",
            state.matter_id,
            len(state.agents_run),
            state.total_tokens_used,
            len(state.errors),
        )

        result = self._build_result(state)
        CounselOSPipeline.cache_result(result)
        return result

    @staticmethod
    def _build_result(state: MatterState) -> IntakeResult:
        """Map final pipeline state to the API response model."""
        return IntakeResult(
            matter_id=state.matter_id,
            client_name=state.submission.client_name,
            submitted_by=state.submission.submitted_by,
            submitted_at=state.submission.submitted_at,
            matter_type=state.matter_type or MatterType.UNKNOWN,
            matter_type_confidence=state.matter_type_confidence,
            matter_summary=state.matter_summary or state.submission.matter_description[:200],
            retrieved_context=state.retrieved_context,
            urgency=state.urgency or UrgencyLevel.STANDARD,
            risk_score=state.risk_score,
            risk_flags=state.risk_flags,
            recommended_tier=state.recommended_tier or AssignmentTier.ASSOCIATE,
            intake_summary=state.intake_summary or "Summary unavailable.",
            suggested_next_steps=state.suggested_next_steps,
            agents_run=state.agents_run,
            total_tokens_used=state.total_tokens_used,
            llm_model=state.llm_model,
        )
