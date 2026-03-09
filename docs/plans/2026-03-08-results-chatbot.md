# Results Chatbot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an interactive chatbot to the results page that lets users ask follow-up questions about their matter analysis and query the knowledge base.

**Architecture:** New `POST /api/chat` endpoint backed by a `ChatAgent` that builds context-aware prompts from cached `IntakeResult` objects + RAG retrieval. Frontend gets a slide-out `ChatPanel` component with dynamic starter chips, integrated into the existing results view.

**Tech Stack:** FastAPI, Pydantic, pytest (backend); React 18, TypeScript, Tailwind CSS (frontend). Reuses existing `BaseLLMProvider` and `rag.store.retrieve`.

---

### Task 1: Backend — Chat Models

**Files:**
- Modify: `backend/models/matter.py:90` (append after `IntakeResult`)
- Test: `backend/tests/test_models.py`

**Step 1: Write the failing tests**

Add to `backend/tests/test_models.py`:

```python
from models.matter import ChatMessage, ChatRequest, ChatResponse


class TestChatModels:
    def test_chat_message_user_role(self) -> None:
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_rejects_invalid_role(self) -> None:
        with pytest.raises(ValueError):
            ChatMessage(role="admin", content="Hello")

    def test_chat_request_requires_matter_id_and_message(self) -> None:
        req = ChatRequest(
            matter_id="MATTER-TEST0001",
            message="Why is my risk score 7?",
            conversation_history=[],
        )
        assert req.matter_id == "MATTER-TEST0001"
        assert req.message == "Why is my risk score 7?"
        assert req.conversation_history == []

    def test_chat_response_has_reply_and_sources(self) -> None:
        resp = ChatResponse(reply="The risk is high because...", sources_used=2)
        assert resp.reply == "The risk is high because..."
        assert resp.sources_used == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_models.py::TestChatModels -v`
Expected: FAIL with `ImportError: cannot import name 'ChatMessage'`

**Step 3: Write minimal implementation**

Append to `backend/models/matter.py` after the `IntakeResult` class (after line 90):

```python


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    matter_id: str
    message: str
    conversation_history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    sources_used: int = 0
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_models.py::TestChatModels -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add backend/models/matter.py backend/tests/test_models.py
git commit -m "feat: add chat request/response models"
```

---

### Task 2: Backend — Result Cache

**Files:**
- Modify: `backend/orchestrator/pipeline.py:34-88` (add cache to `CounselOSPipeline`)
- Test: `backend/tests/test_models.py`

**Step 1: Write the failing tests**

Add to `backend/tests/test_models.py`:

```python
from orchestrator.pipeline import CounselOSPipeline


class TestResultCache:
    def test_cache_result_and_retrieve(self) -> None:
        result = IntakeResult(
            matter_id="MATTER-CACHE01",
            client_name="Test Corp",
            submitted_by="Jane",
            submitted_at=datetime.utcnow(),
            matter_type=MatterType.CONTRACT,
            matter_type_confidence=0.9,
            matter_summary="Test summary",
            urgency=UrgencyLevel.STANDARD,
            risk_score=5.0,
            risk_flags=RiskFlags(),
            recommended_tier=AssignmentTier.ASSOCIATE,
            intake_summary="Summary text",
            suggested_next_steps=["Step 1"],
        )
        CounselOSPipeline.cache_result(result)
        cached = CounselOSPipeline.get_cached_result("MATTER-CACHE01")
        assert cached is not None
        assert cached.matter_id == "MATTER-CACHE01"

    def test_cache_miss_returns_none(self) -> None:
        result = CounselOSPipeline.get_cached_result("MATTER-NONEXIST")
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_models.py::TestResultCache -v`
Expected: FAIL with `AttributeError: type object 'CounselOSPipeline' has no attribute 'cache_result'`

**Step 3: Write minimal implementation**

In `backend/orchestrator/pipeline.py`, add a class-level cache and two static methods. Add at the top of `CounselOSPipeline` (after line 38):

```python
    _result_cache: dict[str, "IntakeResult"] = {}

    @staticmethod
    def cache_result(result: IntakeResult) -> None:
        CounselOSPipeline._result_cache[result.matter_id] = result

    @staticmethod
    def get_cached_result(matter_id: str) -> IntakeResult | None:
        return CounselOSPipeline._result_cache.get(matter_id)
```

Then in the `run` method, cache the result before returning. Change line 88 from:

```python
        return self._build_result(state)
```

to:

```python
        result = self._build_result(state)
        CounselOSPipeline.cache_result(result)
        return result
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_models.py::TestResultCache -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/orchestrator/pipeline.py backend/tests/test_models.py
git commit -m "feat: add in-memory result cache to pipeline"
```

---

### Task 3: Backend — ChatAgent

**Files:**
- Create: `backend/agents/chat.py`
- Create: `backend/tests/test_chat_agent.py`

**Step 1: Write the failing tests**

Create `backend/tests/test_chat_agent.py`:

```python
"""Tests for the ChatAgent — uses a mock LLM provider."""
from unittest.mock import MagicMock
from datetime import datetime

import pytest

from agents.chat import ChatAgent
from core.llm.base import BaseLLMProvider, LLMResponse
from models.matter import (
    ChatMessage,
    IntakeResult,
    MatterType,
    UrgencyLevel,
    AssignmentTier,
    RiskFlags,
)


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock(spec=BaseLLMProvider)
    llm.complete.return_value = LLMResponse(
        content="The risk score is 7 because of regulatory exposure.",
        model="mock-model",
        input_tokens=100,
        output_tokens=50,
    )
    return llm


@pytest.fixture
def sample_result() -> IntakeResult:
    return IntakeResult(
        matter_id="MATTER-TEST0001",
        client_name="Acme Corp",
        submitted_by="Jane Smith",
        submitted_at=datetime.utcnow(),
        matter_type=MatterType.CONTRACT,
        matter_type_confidence=0.85,
        matter_summary="Software licensing agreement review",
        urgency=UrgencyLevel.HIGH,
        risk_score=7.0,
        risk_flags=RiskFlags(regulatory_exposure=True, notes=["GDPR implications"]),
        recommended_tier=AssignmentTier.SENIOR_ASSOCIATE,
        intake_summary="Review needed for SaaS licensing with GDPR implications.",
        suggested_next_steps=["Review DPA", "Check conflict"],
    )


class TestChatAgent:
    def test_answer_returns_reply(
        self, mock_llm: MagicMock, sample_result: IntakeResult
    ) -> None:
        agent = ChatAgent(mock_llm)
        response = agent.answer(
            message="Why is my risk score 7?",
            result=sample_result,
            conversation_history=[],
        )
        assert response.reply == "The risk score is 7 because of regulatory exposure."
        assert response.sources_used >= 0

    def test_llm_receives_system_prompt_with_result(
        self, mock_llm: MagicMock, sample_result: IntakeResult
    ) -> None:
        agent = ChatAgent(mock_llm)
        agent.answer(
            message="Explain the risk",
            result=sample_result,
            conversation_history=[],
        )
        call_args = mock_llm.complete.call_args
        system_prompt = call_args.kwargs.get("system", "") or call_args[1].get("system", "")
        assert "MATTER-TEST0001" in system_prompt
        assert "7.0" in system_prompt or "7" in system_prompt

    def test_conversation_history_included_in_prompt(
        self, mock_llm: MagicMock, sample_result: IntakeResult
    ) -> None:
        history = [
            ChatMessage(role="user", content="What is my risk?"),
            ChatMessage(role="assistant", content="Your risk score is 7."),
        ]
        agent = ChatAgent(mock_llm)
        agent.answer(
            message="Why?",
            result=sample_result,
            conversation_history=history,
        )
        call_args = mock_llm.complete.call_args
        prompt = call_args.kwargs.get("prompt", "") or call_args[0][0]
        assert "What is my risk?" in prompt
        assert "Your risk score is 7." in prompt
        assert "Why?" in prompt
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_chat_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.chat'`

**Step 3: Write minimal implementation**

Create `backend/agents/chat.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_chat_agent.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add backend/agents/chat.py backend/tests/test_chat_agent.py
git commit -m "feat: add ChatAgent with context-aware prompting"
```

---

### Task 4: Backend — Chat Endpoint

**Files:**
- Modify: `backend/main.py:8-9` (update imports), append new endpoint after line 67
- Create: `backend/tests/test_chat_endpoint.py`

**Step 1: Write the failing tests**

Create `backend/tests/test_chat_endpoint.py`:

```python
"""Integration tests for POST /api/chat using FastAPI TestClient."""
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app
from models.matter import (
    IntakeResult,
    MatterType,
    UrgencyLevel,
    AssignmentTier,
    RiskFlags,
)
from orchestrator.pipeline import CounselOSPipeline


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def seed_cache():
    """Put a result in the cache before each test."""
    result = IntakeResult(
        matter_id="MATTER-CHAT01",
        client_name="Test Corp",
        submitted_by="Test User",
        submitted_at=datetime.utcnow(),
        matter_type=MatterType.LITIGATION,
        matter_type_confidence=0.9,
        matter_summary="Employment dispute",
        urgency=UrgencyLevel.HIGH,
        risk_score=8.0,
        risk_flags=RiskFlags(time_sensitivity=True),
        recommended_tier=AssignmentTier.PARTNER,
        intake_summary="High-risk employment litigation.",
        suggested_next_steps=["Assign partner", "Review timeline"],
    )
    CounselOSPipeline.cache_result(result)
    yield
    CounselOSPipeline._result_cache.clear()


class TestChatEndpoint:
    @patch("main.chat_agent")
    def test_chat_returns_reply(self, mock_agent, client):
        from models.matter import ChatResponse
        mock_agent.answer.return_value = ChatResponse(
            reply="The risk is 8 because of time sensitivity.", sources_used=2
        )
        resp = client.post("/api/chat", json={
            "matter_id": "MATTER-CHAT01",
            "message": "Why is my risk high?",
            "conversation_history": [],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "The risk is 8 because of time sensitivity."
        assert data["sources_used"] == 2

    def test_chat_unknown_matter_returns_404(self, client):
        resp = client.post("/api/chat", json={
            "matter_id": "MATTER-UNKNOWN",
            "message": "Hello",
            "conversation_history": [],
        })
        assert resp.status_code == 404
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_chat_endpoint.py -v`
Expected: FAIL with either `AttributeError` (no `chat_agent` on main) or 404 for unregistered route

**Step 3: Write minimal implementation**

Update `backend/main.py`. Add to imports (line 8-9):

```python
from models.matter import ChatRequest, ChatResponse, IntakeResult, MatterSubmission
from orchestrator.pipeline import CounselOSPipeline
from agents.chat import ChatAgent
from core.llm import get_llm_provider
```

Add module-level `chat_agent` after `pipeline` (line 18):

```python
chat_agent: ChatAgent | None = None
```

Update the `lifespan` function to initialize the chat agent:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, chat_agent
    logger.info("Initialising CounselOS pipeline...")
    pipeline = CounselOSPipeline()
    llm = get_llm_provider()
    chat_agent = ChatAgent(llm)
    logger.info("Pipeline ready.")
    yield
    logger.info("CounselOS shutting down.")
```

Append the new endpoint after the existing `/api/intake` endpoint:

```python
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Ask a follow-up question about a matter's analysis.
    Requires the matter to have been previously processed.
    """
    if chat_agent is None:
        raise HTTPException(status_code=503, detail="Chat agent not initialised.")

    result = CounselOSPipeline.get_cached_result(request.matter_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Matter {request.matter_id} not found in cache.")

    try:
        return chat_agent.answer(
            message=request.message,
            result=result,
            conversation_history=request.conversation_history,
        )
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_chat_endpoint.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_chat_endpoint.py
git commit -m "feat: add POST /api/chat endpoint"
```

---

### Task 5: Frontend — Types and ChatMessage Component

**Files:**
- Modify: `frontend/types/index.ts:49` (append chat types)
- Create: `frontend/components/ChatMessage.tsx`

**Step 1: Add chat types**

Append to `frontend/types/index.ts` after line 49:

```typescript

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  matter_id: string;
  message: string;
  conversation_history: ChatMessage[];
}

export interface ChatResponse {
  reply: string;
  sources_used: number;
}
```

**Step 2: Create ChatMessage component**

Create `frontend/components/ChatMessage.tsx`:

```tsx
"use client";

import type { ChatMessage as ChatMessageType } from "@/types";

interface Props {
  message: ChatMessageType;
}

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "bg-counsel-blue text-white rounded-br-md"
            : "bg-slate-100 text-slate-800 rounded-bl-md"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}
```

**Step 3: Verify manually**

Run: `cd frontend && npx tsc --noEmit`
Expected: No type errors

**Step 4: Commit**

```bash
git add frontend/types/index.ts frontend/components/ChatMessage.tsx
git commit -m "feat: add chat types and ChatMessage component"
```

---

### Task 6: Frontend — SuggestedChips Component

**Files:**
- Create: `frontend/components/SuggestedChips.tsx`

**Step 1: Create SuggestedChips with chip generation logic**

Create `frontend/components/SuggestedChips.tsx`:

```tsx
"use client";

import type { IntakeResult } from "@/types";

interface Props {
  result: IntakeResult;
  onSelect: (question: string) => void;
}

function generateChips(result: IntakeResult): string[] {
  const chips: string[] = ["What similar matters have we handled?"];

  const candidates: [boolean, string][] = [
    [result.risk_score >= 7, `Why is my risk score ${result.risk_score.toFixed(1)}/10?`],
    [
      result.risk_score >= 5 && result.risk_score < 7,
      "What's driving my risk assessment?",
    ],
    [
      result.risk_flags.conflict_of_interest,
      "Explain the conflict of interest flag",
    ],
    [
      result.risk_flags.jurisdiction_complexity,
      "Explain the jurisdiction complexity flag",
    ],
    [
      result.risk_flags.regulatory_exposure,
      "Explain the regulatory exposure flag",
    ],
    [
      result.risk_flags.time_sensitivity,
      "Explain the time sensitivity flag",
    ],
    [
      result.matter_type_confidence < 0.8,
      `Why was this classified as ${result.matter_type.replace(/_/g, " ")}?`,
    ],
    [
      result.recommended_tier === "partner" ||
        result.recommended_tier === "senior_associate",
      `Why does this need a ${result.recommended_tier === "partner" ? "partner" : "senior associate"}?`,
    ],
    [
      result.urgency === "critical" || result.urgency === "high",
      `What makes this ${result.urgency} urgency?`,
    ],
  ];

  for (const [condition, chip] of candidates) {
    if (chips.length >= 4) break;
    if (condition) chips.push(chip);
  }

  return chips;
}

export default function SuggestedChips({ result, onSelect }: Props) {
  const chips = generateChips(result);

  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((chip) => (
        <button
          key={chip}
          onClick={() => onSelect(chip)}
          className="text-xs bg-counsel-light text-counsel-blue px-3 py-1.5 rounded-full hover:bg-counsel-mid hover:text-white transition-colors"
        >
          {chip}
        </button>
      ))}
    </div>
  );
}
```

**Step 2: Verify manually**

Run: `cd frontend && npx tsc --noEmit`
Expected: No type errors

**Step 3: Commit**

```bash
git add frontend/components/SuggestedChips.tsx
git commit -m "feat: add SuggestedChips with dynamic chip generation"
```

---

### Task 7: Frontend — ChatPanel Component

**Files:**
- Create: `frontend/components/ChatPanel.tsx`

**Step 1: Create the slide-out panel**

Create `frontend/components/ChatPanel.tsx`:

```tsx
"use client";

import { useState, useRef, useEffect } from "react";
import type { IntakeResult, ChatMessage as ChatMessageType } from "@/types";
import ChatMessage from "./ChatMessage";
import SuggestedChips from "./SuggestedChips";

interface Props {
  result: IntakeResult;
  open: boolean;
  onClose: () => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ChatPanel({ result, open, onClose }: Props) {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;

    const userMsg: ChatMessageType = { role: "user", content: text.trim() };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput("");
    setLoading(true);

    try {
      const resp = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          matter_id: result.matter_id,
          message: text.trim(),
          conversation_history: messages,
        }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();
      setMessages([...updated, { role: "assistant", content: data.reply }]);
    } catch {
      setMessages([
        ...updated,
        { role: "assistant", content: "Sorry, something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  return (
    <div
      className={`fixed top-0 right-0 h-full w-[400px] bg-white border-l border-slate-200 shadow-2xl flex flex-col z-50 transition-transform duration-300 ease-in-out ${
        open ? "translate-x-0" : "translate-x-full"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 bg-counsel-navy">
        <div>
          <p className="text-white text-sm font-semibold">Chat</p>
          <p className="text-counsel-light text-xs font-mono">{result.matter_id}</p>
        </div>
        <button
          onClick={onClose}
          className="text-slate-300 hover:text-white text-lg leading-none"
        >
          &times;
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="space-y-4">
            <p className="text-sm text-slate-500">
              Ask me anything about this matter&apos;s analysis or our knowledge base.
            </p>
            <SuggestedChips result={result} onSelect={sendMessage} />
          </div>
        )}

        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} />
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-100 rounded-2xl rounded-bl-md px-4 py-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-3 border-t border-slate-200">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            disabled={loading}
            className="flex-1 text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-counsel-mid focus:border-transparent disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-counsel-blue text-white text-sm px-4 py-2 rounded-lg hover:bg-counsel-navy transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
```

**Step 2: Verify manually**

Run: `cd frontend && npx tsc --noEmit`
Expected: No type errors

**Step 3: Commit**

```bash
git add frontend/components/ChatPanel.tsx
git commit -m "feat: add ChatPanel slide-out component"
```

---

### Task 8: Frontend — Page Integration and Reset Behavior

**Files:**
- Modify: `frontend/app/page.tsx`

**Step 1: Integrate chat button and panel**

Replace `frontend/app/page.tsx` with:

```tsx
"use client";

import { useState } from "react";
import IntakeForm from "@/components/IntakeForm";
import IntakeResultView from "@/components/IntakeResult";
import ChatPanel from "@/components/ChatPanel";
import type { IntakeResult } from "@/types";

export default function HomePage() {
  const [result, setResult] = useState<IntakeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);

  const handleReset = () => {
    setResult(null);
    setChatOpen(false);
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-counsel-navy mb-2">
          Legal Matter Intake
        </h1>
        <p className="text-slate-500 max-w-2xl">
          CounselOS runs your matter submission through a five-agent AI pipeline —
          intake normalization, classification, RAG-augmented context retrieval,
          risk assessment, and structured attorney assignment recommendation.
        </p>
      </div>

      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-10 flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-counsel-light border-t-counsel-mid rounded-full animate-spin" />
          <div className="text-center">
            <p className="text-counsel-navy font-medium">Running intake pipeline...</p>
            <p className="text-sm text-slate-400 mt-1">
              Intake → Classification → RAG → Risk → Response
            </p>
          </div>
        </div>
      )}

      {!loading && !result && (
        <IntakeForm onResult={setResult} onLoading={setLoading} />
      )}

      {!loading && result && (
        <IntakeResultView result={result} onReset={handleReset} />
      )}

      {/* Chat button — visible when results are showing */}
      {!loading && result && !chatOpen && (
        <button
          onClick={() => setChatOpen(true)}
          className="fixed bottom-6 right-6 w-14 h-14 bg-counsel-blue text-white rounded-full shadow-lg hover:bg-counsel-navy transition-colors flex items-center justify-center z-40"
          title="Ask about this matter"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </button>
      )}

      {/* Chat panel */}
      {result && (
        <ChatPanel
          result={result}
          open={chatOpen}
          onClose={() => setChatOpen(false)}
        />
      )}
    </div>
  );
}
```

**Step 2: Verify manually**

Run: `cd frontend && npx tsc --noEmit`
Expected: No type errors

**Step 3: Verify the dev server runs**

Run: `cd frontend && npm run dev`
Expected: Compiles without errors. Results page shows chat button when results exist.

**Step 4: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat: integrate chat button and panel into results page"
```

---

### Task 9: End-to-End Smoke Test

**Step 1: Start backend and frontend**

Terminal 1: `cd backend && python -m uvicorn main:app --reload --port 8000`
Terminal 2: `cd frontend && npm run dev`

**Step 2: Verify the full flow**

1. Open `http://localhost:3000` and submit a legal matter.
2. Wait for results to appear.
3. Confirm the chat icon button appears in the bottom-right corner.
4. Click it — the chat panel should slide in from the right.
5. Verify starter chips appear (at least "What similar matters have we handled?").
6. Click a chip or type a question and send.
7. Verify typing indicator appears, then an assistant response.
8. Close the panel with the X button. Confirm it slides out.
9. Click "Submit another matter" — confirm the chat button disappears.

**Step 3: Commit any final fixes and tag**

```bash
git add -A
git commit -m "feat: results chatbot complete — end-to-end verified"
```
