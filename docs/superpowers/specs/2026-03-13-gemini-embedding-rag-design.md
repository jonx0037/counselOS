# Design: Gemini Embedding 2 + ChromaDB RAG Upgrade

**Date:** 2026-03-13
**Status:** Approved

## Prerequisites

> **Python 3.14 + ChromaDB compatibility must be verified before implementation begins.**
> The project runs Python 3.14.2. ChromaDB 1.5.2 (currently installed) has a known incompatibility on Python 3.14 due to its internal use of `pydantic.v1.BaseSettings`. Implementer must confirm a working ChromaDB version on Python 3.14, or resolve by downgrading the runtime to Python 3.12, before writing any code. The version pin in `requirements.txt` must be updated to the verified version.

## Problem

CounselOS's current RAG pipeline (`rag/store.py`) uses TF-IDF + cosine similarity — a purely lexical (keyword-matching) approach. It has no semantic understanding, so a query like "employee fired without cause" can miss a document about "wrongful termination" unless those exact words appear.

Additionally, `chromadb` is declared in `requirements.txt` but never used — the store is backed by a plain JSON file.

## Goals

1. Replace TF-IDF with Gemini Embedding 2 for semantic retrieval
2. Wire up ChromaDB as the vector store (already a declared dependency)
3. Introduce a provider-agnostic `core/embeddings/` layer so the embedding model can be swapped via config — mirroring the existing `core/llm/` pattern

## Non-Goals

- Hybrid retrieval (TF-IDF + dense) — deferred until there is evidence dense-only misses results
- Replacing Anthropic with Google for LLM inference
- Any frontend changes

## Architecture

### `core/embeddings/` Module

New module mirroring `core/llm/`. The factory lives in `__init__.py` to match the existing `core/llm/` pattern exactly:

```
backend/core/embeddings/
├── __init__.py   # BaseEmbeddingProvider + get_embedding_provider() factory
├── base.py       # Abstract BaseEmbeddingProvider (re-exported from __init__)
└── gemini.py     # GeminiEmbeddingProvider
```

**`base.py`**
```python
class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
```

**`__init__.py`** — exports `BaseEmbeddingProvider` and `get_embedding_provider()`:
```python
def get_embedding_provider() -> BaseEmbeddingProvider:
    if settings.embedding_provider == "gemini":
        return GeminiEmbeddingProvider(
            model=settings.embedding_model,
            api_key=settings.google_api_key,
        )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
```

**`gemini.py`**

Uses the **`google-genai`** SDK (current, v1.x — not the legacy `google-generativeai`). Calls with task-type hints:
- `RETRIEVAL_DOCUMENT` when embedding documents at index time
- `RETRIEVAL_QUERY` when embedding the query at retrieval time

This distinction meaningfully improves retrieval accuracy for asymmetric search (short query vs. long document), which is the dominant pattern in CounselOS.

> **Action required:** Confirm the exact Gemini Embedding 2 model ID from the Google blog post. Update `embedding_model` default in `config.py` before implementation. The current placeholder `"gemini-embedding-exp-03-07"` is provisional.

### Config Changes (`core/config.py`)

Four new fields:

```python
# Embeddings
embedding_provider: str = "gemini"
embedding_model: str = "gemini-embedding-exp-03-07"  # confirm from blog post
google_api_key: str = ""
chroma_collection_name: str = "counsel_kb"
```

`chroma_collection_name` is a config field (not a hardcoded constant) because renaming it post-deployment would silently create a second empty collection on disk while leaving the original intact.

### RAG Store Rewrite (`rag/store.py`)

Full internal rewrite; **public API unchanged**:

| Function | Signature (unchanged) |
|----------|----------------------|
| `retrieve` | `(query: str, n_results: int = 3) -> list[str]` |
| `add_documents` | `(new_docs: list[dict]) -> int` |
| `document_count` | `() -> int` |

Internal changes:
- `TfidfVectorizer` + JSON file → ChromaDB `PersistentClient` + Gemini embeddings
- `add_documents()` pre-fetches existing IDs via `collection.get(ids=[...])` to compute the "newly added" count before upserting, preserving the current return semantics that `seed_data.py` relies on
- `retrieve()` embeds the query via `embed_query()` then calls `collection.query(query_embeddings=..., n_results=n_results)`. Defensive guard: if `collection.count() == 0`, return `[]` immediately — some ChromaDB versions raise on querying an empty collection
- `lru_cache` removed — ChromaDB handles persistence natively

Because the public API is unchanged, `agents/rag.py` requires **zero modifications**.

## Data Flow

```
[seed_data.py]
     │
     ▼
add_documents(docs)
     │
     ├─ collection.get(ids=[...])  ← determine which are new
     ▼
GeminiEmbeddingProvider.embed_documents(new_texts)
     │  task_type=RETRIEVAL_DOCUMENT
     ▼
ChromaDB.upsert(ids, embeddings, documents)
     │
     └─ return count of newly added docs

─────────────────────────────────────────

[RAGAgent.run(state)]
     │
     ▼
retrieve(query, n_results=3)
     │
     ├─ guard: if collection.count() == 0 → return []
     ▼
GeminiEmbeddingProvider.embed_query(query)
     │  task_type=RETRIEVAL_QUERY
     ▼
ChromaDB.query(query_embeddings, n_results)
     │
     ▼
state.retrieved_context = [doc_texts]
```

## Dependencies

| Package | Change |
|---------|--------|
| `chromadb` | Already declared — pin to verified Python 3.14-compatible version |
| `google-genai>=1.0.0` | New — current Google AI SDK (replaces legacy `google-generativeai`) |
| `scikit-learn`, `numpy` | No longer used by store — safe to remove in a follow-up cleanup |

## Environment

Add to `.env`:
```
GOOGLE_API_KEY=your_key_here
```

## Change Surface

| File | Type of Change |
|------|----------------|
| `core/embeddings/__init__.py` | New |
| `core/embeddings/base.py` | New |
| `core/embeddings/gemini.py` | New |
| `core/config.py` | +4 fields |
| `rag/store.py` | Full rewrite (same public API) |
| `requirements.txt` | +1 package (`google-genai`), pin chromadb to verified version |
| `.env` | +1 key (`GOOGLE_API_KEY`) |
| `agents/rag.py` | No changes |
| `rag/seed_data.py` | No changes |
| Frontend | No changes |

## Migration Steps

1. Verify ChromaDB version compatibility with Python 3.14; update pin in `requirements.txt`
2. Confirm Gemini Embedding 2 model ID; update `embedding_model` default in `config.py`
3. Add `GOOGLE_API_KEY` to `.env`
4. `pip install google-genai` (and reinstall chromadb at verified version)
5. Delete old `./data/chroma/documents.json`
6. Re-run `python -m rag.seed_data` to re-embed all documents into ChromaDB

## Testing

- Unit test `BaseEmbeddingProvider` contract against `GeminiEmbeddingProvider`
- Integration test: seed → retrieve round-trip returns semantically relevant documents (e.g., query "employee fired" retrieves `employment-001`)
- Regression: follow `test_llm_factory.py` pattern for structure. Patch targets mirror the existing split:
  - Factory / unknown-provider path: patch `"core.embeddings.settings"` (the import inside `__init__.py`)
  - `GeminiEmbeddingProvider` constructor tests: patch `"core.config.settings"` (the canonical object, same as `AnthropicProvider` tests today)
