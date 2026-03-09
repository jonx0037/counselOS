# Results Chatbot — Design Document

**Date:** 2026-03-08
**Status:** Approved

## Summary

Add an interactive chatbot to the results page that lets users ask follow-up questions about their matter analysis and query the firm's knowledge base. The chatbot appears as a slide-out side panel, preserving visibility of results during conversation.

## User Experience Flow

When results appear after pipeline execution, a chat icon button appears fixed in the bottom-right corner. Clicking it slides open a right-side panel (~400px wide) that overlays the page edge while results remain visible and scrollable on the left.

The panel opens with:

1. A greeting: "Ask me anything about this matter's analysis or our knowledge base."
2. 3-4 dynamic starter chips generated from the actual results (see Chip Logic below).

Clicking a chip sends it as a message. Users can also type freely. The conversation is ephemeral — it resets when they submit a new matter.

The panel has a header with the matter ID, a close button, and the chat history scrolls below. Input is a text field with a send button at the bottom.

## Chatbot Scope

The chatbot can answer about:

- **This matter's analysis** — "Why was my risk score 7?", "Explain the jurisdiction complexity flag", "What do the suggested next steps mean?"
- **Knowledge base queries** — "What similar cases have we handled?", "What's the typical timeline for this matter type?"

It does not act as a general legal assistant.

## Backend Architecture

### New Endpoint

**`POST /api/chat`** accepts:

- `matter_id` — identifies which matter's results to reference
- `message` — the user's question
- `conversation_history` — array of prior messages (frontend holds state)

### Processing

The endpoint builds a context-aware prompt by:

1. Pulling the matter's full `IntakeResult` — classification, risk score, flags, summary, next steps — so the LLM can explain its own analysis.
2. Querying ChromaDB with the user's question + matter context to retrieve relevant knowledge base documents.
3. Combining both into a system prompt: "You are a legal intake assistant. Here is the matter analysis: [result]. Here is relevant knowledge: [RAG docs]. Answer the user's question."

### Implementation Details

- New `ChatAgent` class following the existing agent pattern.
- Reuses existing `LLMProvider` abstraction and `RAGStore`.
- In-memory cache of recent `IntakeResult` objects keyed by `matter_id` (avoids re-running pipeline).
- Conversation history passed in each request (stateless chat pattern, no session storage).

## Frontend Components

### `ChatPanel.tsx`

The slide-out panel. Takes `IntakeResult` as a prop. Manages conversation state (`messages[]`) with `useState`. Open/close animation via CSS transform (`translateX`). Contains message list, input field, and header with matter ID + close button.

### `ChatMessage.tsx`

Renders a single message bubble. User messages align right (counsel-blue background, white text). Assistant messages align left (light gray background, dark text). Assistant messages support basic markdown rendering.

### `SuggestedChips.tsx`

Renders dynamic starter questions as clickable pill-shaped buttons. Receives `IntakeResult` and generates chips based on the logic below. Chips disappear once the user sends their first message.

### Integration

- `page.tsx` gets a floating chat button visible only when `result` is not null.
- Clicking toggles `ChatPanel`, which receives the current `IntakeResult`.
- Typing indicator (three animated dots) shows while waiting for response.
- Send button disables during loading.

## Dynamic Chip Generation Logic

A pure function takes an `IntakeResult` and returns 3-4 suggested questions.

### Always included (1 chip)

- "What similar matters have we handled?"

### Conditional chips (pick 2-3, evaluated top-to-bottom)

| Condition | Chip |
|---|---|
| `risk_score >= 7` | "Why is my risk score {score}/10?" |
| `risk_score >= 5 && risk_score < 7` | "What's driving my risk assessment?" |
| Any flag in `risk_flags` is `true` | "Explain the {flag_name} flag" (first active flag) |
| `matter_type_confidence < 0.8` | "Why was this classified as {matter_type}?" |
| `recommended_tier` is `partner` or `senior_associate` | "Why does this need a {tier}?" |
| `urgency` is `critical` or `high` | "What makes this {urgency} urgency?" |

Evaluation stops once 4 chips total are collected (including the static one).
