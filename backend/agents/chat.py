"""
Chat Agent
───────────
Answers follow-up questions about a matter's analysis using the
IntakeResult context and RAG-retrieved knowledge base documents.
"""
import logging

from core.llm.base import BaseLLMProvider
from models.matter import ChatMessage, ChatResponse, IntakeResult
from rag.store import retrieve

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """You are a legal intake assistant for CounselOS. You help users understand
their matter's analysis and answer questions using the firm's knowledge base.

== MATTER ANALYSIS ==
Matter ID: {matter_id}
Client: {client_name}
Type: {matter_type} (confidence: {confidence:.0%})
Urgency: {urgency}
Risk Score: {risk_score}/10
Risk Flags: {risk_flags}
Summary: {intake_summary}
Next Steps: {next_steps}

== KNOWLEDGE BASE ==
{rag_context}

Answer the user's question based on the matter analysis and knowledge base above.
Be specific, reference the actual data, and keep answers concise. If you don't know
something, say so — do not make up information."""


class ChatAgent:
    def __init__(self, llm: BaseLLMProvider) -> None:
        self.llm = llm

    def answer(
        self,
        message: str,
        result: IntakeResult,
        conversation_history: list[ChatMessage],
    ) -> ChatResponse:
        # Retrieve relevant knowledge base documents
        rag_query = f"{result.matter_type} {message}"
        rag_docs = retrieve(rag_query, n_results=3)
        rag_context = "\n\n".join(rag_docs) if rag_docs else "No relevant documents found."

        # Build active risk flags string
        flags = result.risk_flags
        active_flags = [
            name for name, active in [
                ("Conflict of Interest", flags.conflict_of_interest),
                ("Jurisdiction Complexity", flags.jurisdiction_complexity),
                ("Regulatory Exposure", flags.regulatory_exposure),
                ("Time Sensitivity", flags.time_sensitivity),
            ] if active
        ]
        flag_notes = flags.notes
        flags_str = ", ".join(active_flags) if active_flags else "None"
        if flag_notes:
            flags_str += f" (notes: {'; '.join(flag_notes)})"

        system = SYSTEM_TEMPLATE.format(
            matter_id=result.matter_id,
            client_name=result.client_name,
            matter_type=result.matter_type,
            confidence=result.matter_type_confidence,
            urgency=result.urgency,
            risk_score=result.risk_score,
            risk_flags=flags_str,
            intake_summary=result.intake_summary,
            next_steps="; ".join(result.suggested_next_steps),
            rag_context=rag_context,
        )

        # Build conversation prompt
        lines: list[str] = []
        for msg in conversation_history:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        lines.append(f"User: {message}")
        prompt = "\n".join(lines)

        response = self.llm.complete(prompt=prompt, system=system)

        return ChatResponse(
            reply=response.content,
            sources_used=len(rag_docs),
        )
