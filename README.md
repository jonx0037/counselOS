# CounselOS

**A multi-agent AI Co-Worker for corporate legal matter intake.**

CounselOS automates the triage, classification, and risk assessment of incoming legal matters — transforming an unstructured matter submission into a structured intake summary with recommended attorney assignment tier and urgency flags.

Built as a demonstration of Red-Zone enterprise AI architecture: a bespoke, LLM-agnostic multi-agent pipeline with a clean provider abstraction layer, RAG-augmented context retrieval, and a Next.js frontend.

---

## Architecture

```
Client Submission
       │
       ▼
 ┌─────────────┐
 │ Intake Agent│  Normalizes & structures raw matter input
 └──────┬──────┘
        │
        ▼
 ┌──────────────────┐
 │Classification Agent│  Categorizes matter type (contract, litigation, M&A, etc.)
 └──────┬─────────────┘
        │
        ▼
 ┌──────────────┐
 │   RAG Agent  │  Retrieves relevant precedents, policies & jurisdiction notes
 └──────┬───────┘
        │
        ▼
 ┌─────────────┐
 │  Risk Agent │  Scores complexity, urgency, flags conflicts of interest
 └──────┬──────┘
        │
        ▼
 ┌───────────────┐
 │ Response Agent│  Generates structured intake summary + assignment recommendation
 └───────────────┘
```

Each agent is stateless. Shared state flows through the pipeline via an immutable `MatterState` object managed by the orchestrator.

---

## Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | Custom Python state machine |
| LLM Provider | Anthropic Claude (provider-abstracted — swap via config) |
| RAG Store | ChromaDB |
| Backend API | FastAPI |
| Frontend | Next.js 14 (App Router) + TypeScript + Tailwind CSS |

---

## Project Structure

```
counselos/
├── backend/
│   ├── agents/          # Individual agent definitions
│   ├── core/llm/        # Provider-abstracted LLM interface
│   ├── models/          # Pydantic data models
│   ├── orchestrator/    # Pipeline state machine + shared state schema
│   ├── rag/             # ChromaDB store + seed knowledge base
│   └── main.py          # FastAPI entry point
├── frontend/
│   ├── app/             # Next.js App Router pages + API routes
│   └── components/      # React components
└── docs/
    └── architecture.md  # Design decisions & agent contract specs
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- Node.js 18+
- An Anthropic API key

### Backend

```bash
cd backend
pip install -e ".[dev]"
cp ../.env.example ../.env   # add your ANTHROPIC_API_KEY
python -m rag.seed_data      # seed the ChromaDB knowledge base
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## LLM Provider Abstraction

The LLM provider is configured via environment variable and never referenced directly in agent code:

```python
# .env
LLM_PROVIDER=anthropic      # swap to "openai" without touching agent logic
LLM_MODEL=claude-sonnet-4-6
```

All agents call `llm.complete(prompt, system)` through the base interface. Adding a new provider means implementing one class — nothing else changes.

---

## Author

Jonathan Rocha · [datasalt.ai](https://datasalt.ai) · [LinkedIn](https://linkedin.com/in/jonathanrocha)
