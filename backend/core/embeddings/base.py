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
