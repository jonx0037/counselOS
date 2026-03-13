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
