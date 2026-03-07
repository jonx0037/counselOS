# CounselOS — Architecture Notes

## Design Philosophy

CounselOS is built around three principles that mirror FutureSecure AI's Red-Zone approach:

1. **Agent isolation** — agents share state but never communicate directly. The orchestrator owns the `MatterState` object; agents read from it and write to it through defined output fields only.

2. **LLM agnosticism** — no agent imports an LLM SDK. All LLM calls go through `BaseLLMProvider`. Swapping from Anthropic to OpenAI (or a self-hosted model) requires one config change and one new class.

3. **Graceful degradation** — if any agent fails, the pipeline continues with degraded output and records the error in `state.errors`. A failing Classification Agent doesn't prevent the Response Agent from running.

---

## Agent Contracts

Each agent implements `BaseAgent.run(state: MatterState) -> MatterState`.

| Agent | Input fields consumed | Output fields written |
|---|---|---|
| IntakeAgent | `submission` | `normalized_description`, `extracted_entities` |
| ClassificationAgent | `normalized_description`, `extracted_entities` | `matter_type`, `matter_type_confidence`, `matter_summary` |
| RAGAgent | `matter_type`, `normalized_description`, `submission.jurisdiction` | `retrieved_context` |
| RiskAgent | `matter_type`, `matter_summary`, `retrieved_context`, `submission.deadline` | `urgency`, `risk_score`, `risk_flags` |
| ResponseAgent | all upstream fields | `recommended_tier`, `intake_summary`, `suggested_next_steps` |

---

## LLM Provider Interface

```python
class BaseLLMProvider(ABC):
    def complete(self, prompt: str, system: str = "") -> LLMResponse: ...
    def model_name(self) -> str: ...
```

Adding a new provider:
1. Create `backend/core/llm/<provider>.py` implementing `BaseLLMProvider`
2. Add a branch in `backend/core/llm/__init__.py:get_llm_provider()`
3. Set `LLM_PROVIDER=<provider>` in `.env`

No agent code changes required.

---

## Structured Output Pattern

All LLM-calling agents request JSON output via system prompt enforcement and parse with a try/except fallback. This prevents a malformed LLM response from crashing the pipeline.

```python
SYSTEM = "...Respond ONLY with valid JSON — no preamble, no markdown fences."
# After response:
raw = re.sub(r"^```(?:json)?\s*", "", raw)   # strip fences if model adds them anyway
parsed = json.loads(raw)                       # parse
```

---

## Extending the Pipeline

To add a new agent:
1. Create `backend/agents/<name>.py` implementing `BaseAgent`
2. Add it to the `self._agents` list in `CounselOSPipeline.__init__`
3. Add any new output fields to `MatterState`

The orchestrator loop handles the rest.

---

## RAG Knowledge Base

The ChromaDB store is seeded with `backend/rag/seed_data.py`. Documents cover:
- Contract review policies
- Litigation intake protocols
- M&A due diligence requirements
- Employment law edge cases
- IP enforcement notes
- Regulatory compliance thresholds
- Jurisdiction-specific notes (Delaware, cross-border)

To extend: add entries to the `DOCUMENTS` list in `seed_data.py` and re-run the seed script.
