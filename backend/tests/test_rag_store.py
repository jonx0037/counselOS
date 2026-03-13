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
