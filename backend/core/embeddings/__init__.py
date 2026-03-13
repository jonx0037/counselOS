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
