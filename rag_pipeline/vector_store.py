"""
rag_pipeline/vector_store.py — Shared ChromaDB Vector Store Manager
────────────────────────────────────────────────────────────────────
Architecture decision: ONE shared ChromaDB collection for all sessions.
Documents are tagged with {"session_id": <id>} metadata on ingestion.
Retrieval uses a metadata `where` filter to isolate per-session content.

This avoids the disk explosion caused by one DB folder per session and
is far more scalable for a growing number of sessions.

Singleton pattern: the ChromaDB client and LangChain wrapper are
initialised once per process and reused across all requests.

Usage:
    from rag_pipeline.vector_store import add_documents, session_has_documents
"""
from __future__ import annotations

import logging
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

from rag_pipeline.embeddings import get_langchain_embeddings
from config import VECTOR_DIR

logger = logging.getLogger(__name__)

# Name of the single shared ChromaDB collection
COLLECTION_NAME: str = "ai_tutor_docs"

# ── Singletons ────────────────────────────────────────────────────────────────

_chroma_client: chromadb.PersistentClient | None = None
_vector_store: Chroma | None = None


# ── Internal initialisation helpers ──────────────────────────────────────────

def _get_client() -> chromadb.PersistentClient:
    """
    Lazy-initialise and return the shared persistent ChromaDB client.

    The client persists data to VECTOR_DIR so it survives server restarts.
    Telemetry is disabled to keep it fully local and offline.
    """
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initialising ChromaDB persistent client at: {VECTOR_DIR}")
        _chroma_client = chromadb.PersistentClient(
            path=str(VECTOR_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client ready.")
    return _chroma_client


def get_vector_store() -> Chroma:
    """
    Return the shared LangChain Chroma wrapper (lazy singleton).

    Creates the ChromaDB collection on first call if it doesn't exist.
    """
    global _vector_store
    if _vector_store is None:
        client = _get_client()
        _vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=get_langchain_embeddings(),
        )
        logger.info(f"LangChain Chroma wrapper initialised for collection '{COLLECTION_NAME}'.")
    return _vector_store


# ── Public API ────────────────────────────────────────────────────────────────

def add_documents(session_id: str, chunks: List[str]) -> None:
    """
    Embed and store text chunks in ChromaDB, tagging each with session_id.

    Args:
        session_id: The session these chunks belong to (used for filtering).
        chunks: List of plain text strings to embed and store.
    """
    if not chunks:
        logger.warning(f"add_documents() called with empty chunk list for session '{session_id}'.")
        return

    store = get_vector_store()

    # Tag every chunk with its session_id for later filtering
    metadatas = [{"session_id": session_id} for _ in chunks]

    store.add_texts(texts=chunks, metadatas=metadatas)

    logger.info(f"Stored {len(chunks)} chunks for session '{session_id}' in ChromaDB.")


def session_has_documents(session_id: str) -> bool:
    """
    Check whether any documents have been indexed for this session.

    Used by the pipeline to decide between RAG path and plain LLM path.

    Returns:
        True if at least one chunk with this session_id exists, False otherwise.
    """
    try:
        client = _get_client()
        collection = client.get_collection(COLLECTION_NAME)
        result = collection.get(
            where={"session_id": session_id},
            limit=1,
        )
        has_docs = len(result.get("ids", [])) > 0
        logger.debug(f"session_has_documents('{session_id}'): {has_docs}")
        return has_docs
    except Exception as e:
        # Collection may not exist yet on a fresh server start
        logger.debug(f"session_has_documents check failed for '{session_id}': {e}")
        return False
