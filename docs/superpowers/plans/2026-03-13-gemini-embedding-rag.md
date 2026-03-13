# Gemini Embedding 2 + ChromaDB RAG Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the TF-IDF + JSON RAG pipeline with Gemini Embedding 2 dense embeddings + ChromaDB vector store, behind a provider-agnostic `core/embeddings/` module.

**Architecture:** New `core/embeddings/` module mirrors the existing `core/llm/` factory pattern. `rag/store.py` is fully rewritten internally but its public API (`retrieve`, `add_documents`, `document_count`) stays identical so `agents/rag.py` and `rag/seed_data.py` require zero changes.

**Tech Stack:** `google-genai` SDK (v1.x), ChromaDB `PersistentClient`, Python 3.12 (local dev), Python 3.11.9 (Vercel deploy via `runtime.txt`)

**Spec:** `docs/superpowers/specs/2026-03-13-gemini-embedding-rag-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `backend/core/embeddings/__init__.py` | Re-export `BaseEmbeddingProvider` + `get_embedding_provider()` factory | Create |
| `backend/core/embeddings/base.py` | Abstract `BaseEmbeddingProvider` with `embed_documents()` and `embed_query()` | Create |
| `backend/core/embeddings/gemini.py` | `GeminiEmbeddingProvider` using `google-genai` SDK | Create |
| `backend/core/config.py` | Add 4 new settings fields | Modify (lines 12-14) |
| `backend/rag/store.py` | Full rewrite: ChromaDB + embedding provider | Rewrite |
| `backend/requirements.txt` | Add `google-genai>=1.0.0`, pin `chromadb>=1.5.5` | Modify |
| `backend/.env` (root `.env`) | Add `GOOGLE_API_KEY` | Modify |
| `backend/tests/test_embedding_factory.py` | Factory + provider unit tests | Create |
| `backend/tests/test_rag_store.py` | RAG store unit tests with mocked embeddings | Create |

**Unchanged files:** `backend/agents/rag.py`, `backend/rag/seed_data.py`, frontend (all).

---

## Chunk 1: Prerequisites and Foundation

### Task 0: Recreate venv with Python 3.12

ChromaDB (all current versions through 1.5.5) is incompatible with Python 3.14 due to `pydantic.v1.BaseSettings`. Python 3.12 is available locally at `/opt/homebrew/bin/python3.12`.

**Files:** None (environment setup only)

- [ ] **Step 1: Deactivate current venv and recreate with Python 3.12**

```bash
cd backend
deactivate 2>/dev/null
rm -rf .venv
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python --version  # expect: Python 3.12.x
```

- [ ] **Step 2: Install existing dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install successfully including ChromaDB.

- [ ] **Step 3: Verify ChromaDB works on Python 3.12**

```bash
python -c "import chromadb; c = chromadb.Client(); print('ChromaDB OK:', c.heartbeat())"
```

Expected: Prints `ChromaDB OK: ...` with no `pydantic.v1` errors.

---

### Task 1: Add `google-genai` dependency and config fields

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/core/config.py:4-22`
- Modify: `.env` (project root)

- [ ] **Step 1: Update requirements.txt**

Add `google-genai>=1.0.0` and pin chromadb:

In `backend/requirements.txt`, change `chromadb>=0.5.0` to `chromadb>=1.5.5` and add `google-genai>=1.0.0` after the `anthropic` line:

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0
pydantic-settings>=2.2.0
anthropic>=0.26.0
google-genai>=1.0.0
chromadb>=1.5.5
scikit-learn>=1.5.0    # No longer used by store — safe to remove in a follow-up cleanup
numpy>=1.26.0          # No longer used by store — safe to remove in a follow-up cleanup
python-dotenv>=1.0.0
httpx>=0.27.0
```

- [ ] **Step 2: Install new dependency**

```bash
cd backend && pip install -r requirements.txt
```

Expected: `google-genai` installs (latest is 1.67.0+).

- [ ] **Step 3: Add config fields to `core/config.py`**

Add these 4 fields to the `Settings` class after the `# LLM` block (before `# Server`):

```python
    # Embeddings
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-2-preview"  # verified: ai.google.dev/gemini-api/docs/embeddings
    google_api_key: str = ""
    chroma_collection_name: str = "counsel_kb"
```

> **Note:** The model ID `gemini-embedding-2-preview` was verified from the official Google AI documentation at `ai.google.dev/gemini-api/docs/embeddings`. The spec's provisional `"gemini-embedding-exp-03-07"` has been superseded.

The full `Settings` class should look like:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""

    # Embeddings
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-2-preview"  # verified: ai.google.dev/gemini-api/docs/embeddings
    google_api_key: str = ""
    chroma_collection_name: str = "counsel_kb"

    # Server
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]
```

- [ ] **Step 4: Add `GOOGLE_API_KEY` to `.env`**

Add after the `ANTHROPIC_API_KEY` line:

```
GOOGLE_API_KEY=your_google_api_key_here
```

- [ ] **Step 5: Verify config loads**

```bash
cd backend && python -c "from core.config import settings; print(settings.embedding_model)"
```

Expected: `gemini-embedding-2-preview`

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/core/config.py .env
git commit -m "feat: add google-genai dep and embedding config fields"
```

---

## Chunk 2: Embedding Provider Module (TDD)

### Task 2: Create `core/embeddings/base.py` — abstract interface

**Files:**
- Create: `backend/core/embeddings/base.py`
- Test: `backend/tests/test_embedding_factory.py`

- [ ] **Step 1: Write the failing test for the abstract interface**

Create `backend/tests/test_embedding_factory.py`:

```python
"""
Tests for the embedding provider factory.
Validates that the factory correctly resolves providers
and raises on unknown configurations — without making real API calls.
"""
import pytest
from unittest.mock import patch


class TestBaseEmbeddingProvider:
    def test_cannot_instantiate_base_directly(self) -> None:
        from core.embeddings.base import BaseEmbeddingProvider

        with pytest.raises(TypeError):
            BaseEmbeddingProvider()  # type: ignore[abstract]

    def test_base_has_required_methods(self) -> None:
        from core.embeddings.base import BaseEmbeddingProvider

        assert hasattr(BaseEmbeddingProvider, "embed_documents")
        assert hasattr(BaseEmbeddingProvider, "embed_query")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_embedding_factory.py::TestBaseEmbeddingProvider -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.embeddings'`

- [ ] **Step 3: Create the `core/embeddings/` package and `base.py`**

Create `backend/core/embeddings/__init__.py` (empty for now):

```python
```

Create `backend/core/embeddings/base.py`:

```python
"""
Abstract embedding provider interface.

All RAG components interact exclusively with this interface.
Swapping providers means implementing a new subclass and
updating EMBEDDING_PROVIDER in .env — no consumer code changes required.
"""
from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Provider-agnostic embedding interface."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of documents for indexing.

        Args:
            texts: Document texts to embed.

        Returns:
            List of embedding vectors (one per document).
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query for retrieval.

        Args:
            text: Query text to embed.

        Returns:
            Single embedding vector.
        """
        ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && python -m pytest tests/test_embedding_factory.py::TestBaseEmbeddingProvider -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add backend/core/embeddings/
git commit -m "feat: add BaseEmbeddingProvider abstract interface"
```

---

### Task 3: Create `core/embeddings/gemini.py` — Gemini provider

**Files:**
- Create: `backend/core/embeddings/gemini.py`
- Test: `backend/tests/test_embedding_factory.py` (append)

- [ ] **Step 1: Write the failing test for GeminiEmbeddingProvider**

Append to `backend/tests/test_embedding_factory.py`:

```python
class TestGeminiEmbeddingProvider:
    def test_provider_has_required_interface(self) -> None:
        """Mirrors test_llm_factory.py: import the class, check hasattr — no factory call."""
        from core.embeddings.gemini import GeminiEmbeddingProvider

        assert hasattr(GeminiEmbeddingProvider, "embed_documents")
        assert hasattr(GeminiEmbeddingProvider, "embed_query")

    def test_embed_documents_calls_api_with_correct_task_type(self) -> None:
        with patch("core.embeddings.gemini.genai") as mock_genai:
            mock_client = mock_genai.Client.return_value
            mock_response = type("Resp", (), {
                "embeddings": [
                    type("Emb", (), {"values": [0.1, 0.2, 0.3]})()
                ]
            })()
            mock_client.models.embed_content.return_value = mock_response

            from core.embeddings.gemini import GeminiEmbeddingProvider

            provider = GeminiEmbeddingProvider(
                model="gemini-embedding-2-preview",
                api_key="test-key",
            )
            result = provider.embed_documents(["hello world"])

            mock_client.models.embed_content.assert_called_once()
            call_kwargs = mock_client.models.embed_content.call_args
            assert "RETRIEVAL_DOCUMENT" in str(call_kwargs)
            assert result == [[0.1, 0.2, 0.3]]

    def test_embed_query_calls_api_with_correct_task_type(self) -> None:
        with patch("core.embeddings.gemini.genai") as mock_genai:
            mock_client = mock_genai.Client.return_value
            mock_response = type("Resp", (), {
                "embeddings": [
                    type("Emb", (), {"values": [0.4, 0.5, 0.6]})()
                ]
            })()
            mock_client.models.embed_content.return_value = mock_response

            from core.embeddings.gemini import GeminiEmbeddingProvider

            provider = GeminiEmbeddingProvider(
                model="gemini-embedding-2-preview",
                api_key="test-key",
            )
            result = provider.embed_query("test query")

            mock_client.models.embed_content.assert_called_once()
            call_kwargs = mock_client.models.embed_content.call_args
            assert "RETRIEVAL_QUERY" in str(call_kwargs)
            assert result == [0.4, 0.5, 0.6]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_embedding_factory.py::TestGeminiEmbeddingProvider -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.embeddings.gemini'`

- [ ] **Step 3: Implement `GeminiEmbeddingProvider`**

Create `backend/core/embeddings/gemini.py`:

```python
from google import genai
from google.genai import types

from core.embeddings.base import BaseEmbeddingProvider


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Gemini Embedding 2 implementation of the embedding provider interface."""

    def __init__(self, model: str, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )
        return [e.values for e in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )
        return response.embeddings[0].values
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && python -m pytest tests/test_embedding_factory.py::TestGeminiEmbeddingProvider -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add backend/core/embeddings/gemini.py backend/tests/test_embedding_factory.py
git commit -m "feat: add GeminiEmbeddingProvider with task-type hints"
```

---

### Task 4: Wire up the factory in `core/embeddings/__init__.py`

**Files:**
- Modify: `backend/core/embeddings/__init__.py`
- Test: `backend/tests/test_embedding_factory.py` (append)

- [ ] **Step 1: Write the failing test for the factory**

Append to `backend/tests/test_embedding_factory.py`:

```python
class TestEmbeddingProviderFactory:
    def test_gemini_provider_resolves(self) -> None:
        """Factory resolves 'gemini' → GeminiEmbeddingProvider.
        Patches genai.Client to avoid real API calls in CI."""
        with patch("core.embeddings.settings") as mock_settings, \
             patch("core.embeddings.gemini.genai") as _mock_genai:
            mock_settings.embedding_provider = "gemini"
            mock_settings.embedding_model = "gemini-embedding-2-preview"
            mock_settings.google_api_key = "test-key"

            from core.embeddings import get_embedding_provider

            provider = get_embedding_provider()
            assert hasattr(provider, "embed_documents")
            assert hasattr(provider, "embed_query")

    def test_unknown_provider_raises_value_error(self) -> None:
        with patch("core.embeddings.settings") as mock_settings:
            mock_settings.embedding_provider = "unknown_provider"

            from core.embeddings import get_embedding_provider

            with pytest.raises(ValueError, match="Unknown embedding provider"):
                get_embedding_provider()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_embedding_factory.py::TestEmbeddingProviderFactory -v
```

Expected: FAIL with `ImportError: cannot import name 'get_embedding_provider'`

- [ ] **Step 3: Implement the factory**

Replace `backend/core/embeddings/__init__.py` with:

```python
from core.config import settings
from core.embeddings.base import BaseEmbeddingProvider

__all__ = ["BaseEmbeddingProvider", "get_embedding_provider"]


def get_embedding_provider() -> BaseEmbeddingProvider:
    """
    Factory: returns the configured embedding provider.
    Add new providers here — consumer code never changes.
    """
    provider = settings.embedding_provider.lower()

    if provider == "gemini":
        from core.embeddings.gemini import GeminiEmbeddingProvider
        return GeminiEmbeddingProvider(
            model=settings.embedding_model,
            api_key=settings.google_api_key,
        )

    raise ValueError(
        f"Unknown embedding provider: {provider!r}. "
        "Set EMBEDDING_PROVIDER in .env (e.g. 'gemini')."
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && python -m pytest tests/test_embedding_factory.py::TestEmbeddingProviderFactory -v
```

Expected: 2 passed

- [ ] **Step 5: Run all embedding tests together**

```bash
cd backend && python -m pytest tests/test_embedding_factory.py -v
```

Expected: 7 passed (2 base + 3 gemini + 2 factory)

- [ ] **Step 6: Commit**

```bash
git add backend/core/embeddings/__init__.py backend/tests/test_embedding_factory.py
git commit -m "feat: add embedding provider factory mirroring core/llm/ pattern"
```

---

## Chunk 3: RAG Store Rewrite (TDD)

### Task 5: Write failing tests for the new RAG store

**Files:**
- Create: `backend/tests/test_rag_store.py`

- [ ] **Step 1: Write RAG store tests**

Create `backend/tests/test_rag_store.py`:

```python
"""
Tests for the RAG store (ChromaDB + embedding provider).
All embedding calls are mocked — no real API calls.

Strategy: patch module-level globals (`_client`, `_collection`, `_embedder`)
that are created at import time, rather than using importlib.reload.
This avoids fragile reimport issues and keeps tests deterministic.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestRagStore:
    @pytest.fixture(autouse=True)
    def _setup_store(self, tmp_path):
        """
        Patch the module-level ChromaDB client and embedding provider
        BEFORE importing rag.store, so tests never hit real infra.
        """
        mock_provider = MagicMock()
        mock_provider.embed_documents.side_effect = lambda texts: [
            [float(i)] * 3 for i in range(len(texts))
        ]
        mock_provider.embed_query.return_value = [0.0] * 3

        with patch("rag.store.settings") as mock_settings, \
             patch("rag.store.chromadb") as mock_chromadb, \
             patch("rag.store.get_embedding_provider", return_value=mock_provider):

            mock_settings.chroma_persist_dir = str(tmp_path / "chroma")
            mock_settings.chroma_collection_name = "test_kb"

            # Use a real in-memory ChromaDB client for realistic behavior
            import chromadb
            client = chromadb.Client()
            collection = client.get_or_create_collection(
                name="test_kb",
                metadata={"hnsw:space": "cosine"},
            )

            with patch("rag.store._client", client), \
                 patch("rag.store._collection", collection), \
                 patch("rag.store._embedder", mock_provider):

                import rag.store
                self.store = rag.store
                self.mock_provider = mock_provider
                yield

    def test_document_count_empty(self) -> None:
        assert self.store.document_count() == 0

    def test_add_documents_returns_count(self) -> None:
        docs = [
            {"id": "doc-1", "text": "First document"},
            {"id": "doc-2", "text": "Second document"},
        ]
        added = self.store.add_documents(docs)
        assert added == 2

    def test_add_documents_skips_duplicates(self) -> None:
        docs = [{"id": "doc-1", "text": "First document"}]
        self.store.add_documents(docs)

        # Add again — should skip
        added = self.store.add_documents(docs)
        assert added == 0

    def test_retrieve_returns_results(self) -> None:
        docs = [
            {"id": "doc-1", "text": "Contract review policy"},
            {"id": "doc-2", "text": "Employment law details"},
        ]
        self.store.add_documents(docs)

        results = self.store.retrieve("contract", n_results=1)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_retrieve_empty_collection_returns_empty(self) -> None:
        results = self.store.retrieve("anything")
        assert results == []

    def test_document_count_after_add(self) -> None:
        docs = [
            {"id": "doc-1", "text": "First"},
            {"id": "doc-2", "text": "Second"},
        ]
        self.store.add_documents(docs)
        assert self.store.document_count() == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_rag_store.py -v
```

Expected: FAIL (tests try to import `get_embedding_provider` from `rag.store` which doesn't exist yet)

- [ ] **Step 3: Commit the failing tests**

```bash
git add backend/tests/test_rag_store.py
git commit -m "test: add RAG store tests (red phase)"
```

---

### Task 6: Rewrite `rag/store.py` with ChromaDB + Gemini embeddings

**Files:**
- Rewrite: `backend/rag/store.py`

- [ ] **Step 1: Rewrite `rag/store.py`**

Replace the entire contents of `backend/rag/store.py` with:

```python
"""
RAG store backed by ChromaDB + configurable embedding provider.
Public API is unchanged from the TF-IDF implementation:
  - retrieve(query, n_results) -> list[str]
  - add_documents(new_docs) -> int
  - document_count() -> int
"""
import logging

import chromadb

from core.config import settings
from core.embeddings import get_embedding_provider

logger = logging.getLogger(__name__)

_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
_collection = _client.get_or_create_collection(
    name=settings.chroma_collection_name,
    metadata={"hnsw:space": "cosine"},
)
_embedder = get_embedding_provider()


def retrieve(query: str, n_results: int = 3) -> list[str]:
    if _collection.count() == 0:
        logger.warning("Knowledge base is empty. Run: python -m rag.seed_data")
        return []

    query_embedding = _embedder.embed_query(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    documents = results.get("documents", [[]])[0]
    return documents


def add_documents(new_docs: list[dict]) -> int:
    ids = [d["id"] for d in new_docs]
    texts = [d["text"] for d in new_docs]

    # Check which IDs already exist to compute "newly added" count
    existing = _collection.get(ids=ids)
    existing_ids = set(existing["ids"])
    to_add_indices = [i for i, doc_id in enumerate(ids) if doc_id not in existing_ids]

    if not to_add_indices:
        return 0

    new_ids = [ids[i] for i in to_add_indices]
    new_texts = [texts[i] for i in to_add_indices]

    embeddings = _embedder.embed_documents(new_texts)
    _collection.upsert(
        ids=new_ids,
        embeddings=embeddings,
        documents=new_texts,
    )
    return len(new_ids)


def document_count() -> int:
    return _collection.count()
```

- [ ] **Step 2: Run RAG store tests**

```bash
cd backend && python -m pytest tests/test_rag_store.py -v
```

Expected: 6 passed

- [ ] **Step 3: Run ALL tests to check for regressions**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: All tests pass (embedding factory + RAG store + existing tests)

- [ ] **Step 4: Commit**

```bash
git add backend/rag/store.py
git commit -m "feat: rewrite RAG store with ChromaDB + Gemini embeddings"
```

---

## Chunk 4: Cleanup, Seed, and Verify

### Task 7: Remove old JSON store and re-seed

**Files:**
- Delete: `backend/data/chroma/documents.json`
- No code changes

- [ ] **Step 1: Delete the old JSON document store**

```bash
rm -f backend/data/chroma/documents.json
```

- [ ] **Step 2: Add real `GOOGLE_API_KEY` to `.env`**

Replace the placeholder in `.env` with your actual Google API key. You can get one from https://aistudio.google.com/apikey

- [ ] **Step 3: Re-seed the knowledge base**

```bash
cd backend && python -m rag.seed_data
```

Expected: `Seeded 10 documents. Total: 10.`

- [ ] **Step 4: Verify semantic retrieval works**

```bash
cd backend && python -c "
from rag.store import retrieve
results = retrieve('employee fired without cause')
for r in results:
    print(r[:80] + '...')
"
```

Expected: Should return documents about employment law / wrongful termination (e.g., `employment-001`, `litigation-002`) — demonstrating semantic understanding that TF-IDF could not provide.

- [ ] **Step 5: Commit cleanup**

```bash
git add -A
git commit -m "chore: remove old JSON store, re-seed with Gemini embeddings"
```

---

### Task 8: Start the full stack and smoke test

**Files:** None (manual verification)

- [ ] **Step 1: Start the backend**

```bash
cd backend && uvicorn api.main:app --port 8000 --reload
```

Expected: Server starts without errors.

- [ ] **Step 2: Start the frontend (separate terminal)**

```bash
cd frontend && npm run dev
```

- [ ] **Step 3: Smoke test via the chat interface**

Open `http://localhost:3000`, enter a query like "What are the rules for executive severance?", and verify:
1. The chat returns a response (no 500 errors)
2. The response references relevant legal content (severance, non-compete, ADEA)

- [ ] **Step 4: Final commit and push**

```bash
git add -A
git commit -m "feat: complete Gemini Embedding 2 + ChromaDB RAG upgrade"
git push origin main
```

This push triggers the Vercel auto-deploy. Production uses Python 3.11.9 (per `runtime.txt`), which is compatible with all dependencies.
