"""
RAG Agent
──────────
Retrieves relevant precedents, policies, and jurisdiction notes
from the ChromaDB knowledge base using the classified matter type
and normalized description as the query.
"""
from agents.base import BaseAgent
from orchestrator.state import MatterState
from rag.store import retrieve


class RAGAgent(BaseAgent):
    def run(self, state: MatterState) -> MatterState:
        # Build a targeted retrieval query from upstream agent output
        matter_type = state.matter_type.value if state.matter_type else ""
        description = state.normalized_description or state.submission.matter_description
        jurisdiction = state.submission.jurisdiction or ""

        query_parts = [p for p in [matter_type, description[:300], jurisdiction] if p]
        query = " | ".join(query_parts)

        docs = retrieve(query, n_results=3)
        state.retrieved_context = docs

        # RAG agent doesn't call the LLM — it's a retrieval step
        state.record_agent("RAGAgent", tokens_used=0)
        return state
