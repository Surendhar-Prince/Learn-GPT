"""
rag_pipeline/retriever.py — Semantic Retrieval from ChromaDB
──────────────────────────────────────────────────────────────
Queries the shared ChromaDB collection for the top-k most semantically
similar text chunks to the user's query, filtered by session_id.

Session isolation is achieved entirely through ChromaDB metadata filtering:
  where={"session_id": session_id}

This means only content uploaded in the current session is ever retrieved,
even though all sessions share the same underlying collection.
"""

import logging
from typing import List

from rag_pipeline.vector_store import get_vector_store
from config import RETRIEVAL_K

logger = logging.getLogger(__name__)


def retrieve(session_id: str, query: str) -> List[str]:
    """
    Retrieve the top-k text chunks most relevant to the query for a session.

    Args:
        session_id: The session whose documents to search within.
        query: The user's natural language question or query.

    Returns:
        A list of text strings (chunk content), ordered by relevance.
        Returns an empty list if no results are found or on error.
    """
    if not query.strip():
        logger.warning("retrieve() called with an empty query.")
        return []

    store = get_vector_store()

    try:
        results = store.similarity_search(
            query=query,
            k=RETRIEVAL_K,
            filter={"session_id": session_id},
        )

        chunks = [doc.page_content for doc in results]

        logger.info(
            f"Retrieved {len(chunks)} chunk(s) for session '{session_id}' "
            f"(requested top-{RETRIEVAL_K})."
        )
        return chunks

    except Exception as e:
        # Graceful degradation: log the error and return empty to fall back
        # to plain LLM response rather than raising a 500 error.
        logger.warning(
            f"Retrieval failed for session '{session_id}': {e}. "
            f"Falling back to empty context."
        )
        return []
