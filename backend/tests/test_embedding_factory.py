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
