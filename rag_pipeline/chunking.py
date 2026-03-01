"""
rag_pipeline/chunking.py — Text Chunking for Embedding
────────────────────────────────────────────────────────
Splits raw extracted text into overlapping chunks before embedding.

Uses LangChain's RecursiveCharacterTextSplitter which tries to split on:
  1. Paragraph breaks (\n\n)
  2. Line breaks (\n)
  3. Sentence endings (. )
  4. Spaces
  5. Characters (last resort)

This hierarchy preserves semantic coherence across chunks.

Config values (from config.py):
  CHUNK_SIZE    — target characters per chunk (default: 500)
  CHUNK_OVERLAP — overlap between adjacent chunks (default: 60)
"""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_text(text: str) -> List[str]:
    """
    Split a plain text string into overlapping chunks for embedding.

    Args:
        text: Raw text extracted from a PDF or any source.

    Returns:
        A list of text chunk strings. Empty list if input is blank.
    """
    if not text or not text.strip():
        logger.warning("chunk_text() received empty or whitespace-only text.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)

    logger.info(
        f"Chunked text into {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
        f"input_length={len(text)} chars)."
    )

    return chunks
