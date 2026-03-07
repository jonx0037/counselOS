"""
Lightweight RAG store using TF-IDF + cosine similarity.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings

logger = logging.getLogger(__name__)
STORE_PATH = Path(settings.chroma_persist_dir) / "documents.json"

def _load_documents() -> list[dict]:
    if not STORE_PATH.exists():
        return []
    with open(STORE_PATH) as f:
        return json.load(f)

def _save_documents(docs: list[dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w") as f:
        json.dump(docs, f, indent=2)

@lru_cache(maxsize=1)
def _build_index() -> tuple:
    docs = _load_documents()
    if not docs:
        return None, None, None
    texts = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return docs, vectorizer, matrix

def retrieve(query: str, n_results: int = 3) -> list[str]:
    docs, vectorizer, matrix = _build_index()
    if docs is None:
        logger.warning("Knowledge base is empty. Run: python -m rag.seed_data")
        return []
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:n_results]
    return [docs[i]["text"] for i in top_indices if scores[i] > 0]

def add_documents(new_docs: list[dict]) -> int:
    existing = _load_documents()
    existing_ids = {d["id"] for d in existing}
    to_add = [d for d in new_docs if d["id"] not in existing_ids]
    if to_add:
        _save_documents(existing + to_add)
        _build_index.cache_clear()
    return len(to_add)

def document_count() -> int:
    return len(_load_documents())
