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
