# Design: Gemini Embedding 2 + ChromaDB RAG Upgrade

**Date:** 2026-03-13
**Status:** Approved

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

New module mirroring `core/llm/`:

```
backend/core/embeddings/
├── __init__.py
├── base.py       # Abstract BaseEmbeddingProvider
├── gemini.py     # GeminiEmbeddingProvider
└── factory.py    # get_embedding_provider() factory
```

**`base.py`**
```python
class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
```

**`gemini.py`**
Calls `google-generativeai` with task-type hints:
- `RETRIEVAL_DOCUMENT` when embedding documents at index time
- `RETRIEVAL_QUERY` when embedding the query at retrieval time

This distinction meaningfully improves retrieval accuracy for asymmetric search (short query vs. long document), which is the dominant pattern in CounselOS.

**`factory.py`**
Reads `settings.embedding_provider` and returns the configured implementation.

### Config Changes (`core/config.py`)

Three new fields:

```python
# Embeddings
embedding_provider: str = "gemini"
embedding_model: str = "gemini-embedding-exp-03-07"  # confirm exact ID from blog post
google_api_key: str = ""
```

> **Action required:** Confirm the exact Gemini Embedding 2 model ID from https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/ and update `embedding_model` default accordingly.

### RAG Store Rewrite (`rag/store.py`)

Full internal rewrite; **public API unchanged**:

| Function | Signature (unchanged) |
|----------|----------------------|
| `retrieve` | `(query: str, n_results: int = 3) -> list[str]` |
| `add_documents` | `(new_docs: list[dict]) -> int` |
| `document_count` | `() -> int` |

Internal changes:
- `TfidfVectorizer` + JSON file → ChromaDB `PersistentClient` + Gemini embeddings
- `add_documents()` embeds texts via `embed_documents()` and upserts into Chroma
- `retrieve()` embeds the query via `embed_query()` and calls `collection.query()`
- `lru_cache` removed — ChromaDB handles persistence natively

Because the public API is unchanged, `agents/rag.py` requires **zero modifications**.

## Data Flow

```
[seed_data.py]
     │
     ▼
add_documents(docs)
     │
     ▼
GeminiEmbeddingProvider.embed_documents(texts)
     │  task_type=RETRIEVAL_DOCUMENT
     ▼
ChromaDB.upsert(ids, embeddings, documents)

─────────────────────────────────────────

[RAGAgent.run(state)]
     │
     ▼
retrieve(query, n_results=3)
     │
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
| `chromadb>=0.5.0` | Already declared, now actually used |
| `google-generativeai>=0.8.0` | New — add to `requirements.txt` |
| `scikit-learn`, `numpy` | Still in requirements, but no longer used by store. Can be removed in a follow-up if no other code needs them. |

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
| `core/embeddings/factory.py` | New |
| `core/config.py` | +3 fields |
| `rag/store.py` | Full rewrite (same public API) |
| `requirements.txt` | +1 package (`google-generativeai`) |
| `.env` | +1 key (`GOOGLE_API_KEY`) |
| `agents/rag.py` | No changes |
| `rag/seed_data.py` | No changes |
| Frontend | No changes |

## Migration Steps

1. Add `GOOGLE_API_KEY` to `.env`
2. `pip install google-generativeai`
3. Delete old `./data/chroma/documents.json` (or leave it — Chroma ignores it)
4. Re-run `python -m rag.seed_data` to re-embed all documents into ChromaDB

## Testing

- Unit test `BaseEmbeddingProvider` contract against `GeminiEmbeddingProvider`
- Integration test: seed → retrieve round-trip returns semantically relevant documents
- Regression: existing `test_llm_factory.py` pattern as reference for structure
