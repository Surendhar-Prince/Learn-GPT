"""
rag_pipeline/embeddings.py — Embedding Model Provider
───────────────────────────────────────────────────────
Provides a singleton SentenceTransformer model and a LangChain-compatible
wrapper so it can be plugged directly into ChromaDB via LangChain.

Singleton pattern ensures the model is loaded only once per process,
avoiding redundant memory usage on every request.

Usage:
    from rag_pipeline.embeddings import get_langchain_embeddings
    embeddings = get_langchain_embeddings()
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# ── Singleton storage ─────────────────────────────────────────────────────────

_model_instance: SentenceTransformer | None = None


# ── Raw SentenceTransformer access ────────────────────────────────────────────

def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load and return the shared SentenceTransformer instance.

    The model is downloaded once on first call and reused on subsequent calls.
    Thread-safe at module import level (Python GIL protects the assignment).
    """
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}' ...")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
    return _model_instance


# ── LangChain-compatible wrapper ──────────────────────────────────────────────

class SentenceTransformerEmbeddings(Embeddings):
    """
    LangChain Embeddings adapter wrapping the singleton SentenceTransformer.

    Implements the two methods required by LangChain:
      - embed_documents(texts)  → called when indexing chunks into ChromaDB
      - embed_query(text)       → called when embedding a search query
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents. Returns a list of float vectors."""
        model = get_embedding_model()
        vectors = model.encode(texts, show_progress_bar=False, batch_size=32)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string. Returns a float vector."""
        model = get_embedding_model()
        vector = model.encode([text], show_progress_bar=False)[0]
        return vector.tolist()


# ── Factory ───────────────────────────────────────────────────────────────────

def get_langchain_embeddings() -> SentenceTransformerEmbeddings:
    """
    Return a LangChain-compatible SentenceTransformerEmbeddings instance.

    Call this wherever a LangChain Embeddings object is expected,
    e.g. when creating or loading a Chroma vector store.
    """
    return SentenceTransformerEmbeddings()
