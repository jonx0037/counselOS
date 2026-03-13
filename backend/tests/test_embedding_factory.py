"""
Tests for the embedding provider factory.
Validates that the factory correctly resolves providers
and raises on unknown configurations — without making real API calls.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestBaseEmbeddingProvider:
    def test_cannot_instantiate_base_directly(self) -> None:
        from core.embeddings.base import BaseEmbeddingProvider

        with pytest.raises(TypeError):
            BaseEmbeddingProvider()  # type: ignore[abstract]

    def test_base_has_required_methods(self) -> None:
        from core.embeddings.base import BaseEmbeddingProvider

        assert hasattr(BaseEmbeddingProvider, "embed_documents")
        assert hasattr(BaseEmbeddingProvider, "embed_query")


class TestGeminiEmbeddingProvider:
    def test_provider_has_required_interface(self) -> None:
        """Mirrors test_llm_factory.py: import the class, check hasattr — no factory call."""
        from core.embeddings.gemini import GeminiEmbeddingProvider

        assert hasattr(GeminiEmbeddingProvider, "embed_documents")
        assert hasattr(GeminiEmbeddingProvider, "embed_query")

    def test_embed_documents_calls_api_with_correct_task_type(self) -> None:
        with patch("core.embeddings.gemini.genai") as mock_genai:
            with patch("core.embeddings.gemini.types") as mock_types:
                mock_client = mock_genai.Client.return_value
                mock_response = type("Resp", (), {
                    "embeddings": [
                        type("Emb", (), {"values": [0.1, 0.2, 0.3]})()
                    ]
                })()
                mock_client.models.embed_content.return_value = mock_response
                
                # Capture the EmbedContentConfig call
                mock_config_instance = MagicMock()
                mock_types.EmbedContentConfig.return_value = mock_config_instance

                from core.embeddings.gemini import GeminiEmbeddingProvider

                provider = GeminiEmbeddingProvider(
                    model="gemini-embedding-2-preview",
                    api_key="test-key",
                )
                result = provider.embed_documents(["hello world"])

                mock_client.models.embed_content.assert_called_once()
                # Check that EmbedContentConfig was called with correct task_type
                mock_types.EmbedContentConfig.assert_called_once_with(
                    task_type="RETRIEVAL_DOCUMENT"
                )
                assert result == [[0.1, 0.2, 0.3]]

    def test_embed_query_calls_api_with_correct_task_type(self) -> None:
        with patch("core.embeddings.gemini.genai") as mock_genai:
            with patch("core.embeddings.gemini.types") as mock_types:
                mock_client = mock_genai.Client.return_value
                mock_response = type("Resp", (), {
                    "embeddings": [
                        type("Emb", (), {"values": [0.4, 0.5, 0.6]})()
                    ]
                })()
                mock_client.models.embed_content.return_value = mock_response
                
                # Capture the EmbedContentConfig call
                mock_config_instance = MagicMock()
                mock_types.EmbedContentConfig.return_value = mock_config_instance

                from core.embeddings.gemini import GeminiEmbeddingProvider

                provider = GeminiEmbeddingProvider(
                    model="gemini-embedding-2-preview",
                    api_key="test-key",
                )
                result = provider.embed_query("test query")

                mock_client.models.embed_content.assert_called_once()
                # Check that EmbedContentConfig was called with correct task_type
                mock_types.EmbedContentConfig.assert_called_once_with(
                    task_type="RETRIEVAL_QUERY"
                )
                assert result == [0.4, 0.5, 0.6]
