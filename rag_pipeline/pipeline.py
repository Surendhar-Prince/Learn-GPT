"""
rag_pipeline/pipeline.py — RAG Orchestration Pipeline
───────────────────────────────────────────────────────
Central orchestrator that decides HOW to respond to a user message:

  Path A — RAG (documents uploaded):
    1. Retrieve top-k relevant chunks from ChromaDB (filtered by session)
    2. Build a grounded prompt: system + context + history + question
    3. Generate response from LLM

  Path B — Plain LLM (no documents uploaded):
    1. Build a plain prompt: system + history + question
    2. Generate response from LLM

The pipeline enforces a MAX_CONTEXT_CHUNKS budget to prevent the retrieved
context from overflowing the LLM's context window.

System persona:
  "You are an AI tutor helping students understand concepts clearly."
"""

import logging
from typing import List

from rag_pipeline.vector_store import session_has_documents
from rag_pipeline.retriever import retrieve
from rag_pipeline.llm import generate
from config import MAX_CONTEXT_CHUNKS

logger = logging.getLogger(__name__)

# ── System Persona ────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = (
    "You are an AI tutor helping students understand concepts clearly. "
    "Explain ideas step by step, use simple language, provide examples where helpful, "
    "and always encourage curiosity. If you are unsure about something, say so honestly "
    "rather than guessing."
)

# ── Prompt Builders ───────────────────────────────────────────────────────────

def _format_history(history: List[dict]) -> str:
    """
    Format the windowed chat history into a readable conversation transcript.

    Args:
        history: List of {"role": ..., "content": ...} message dicts.

    Returns:
        Multi-line string with each message prefixed by role label.
    """
    if not history:
        return "(No prior conversation.)"

    lines = []
    for msg in history:
        role_label = "Student" if msg["role"] == "user" else "Tutor"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)


def _build_rag_prompt(
    context_chunks: List[str],
    history: List[dict],
    user_message: str,
) -> str:
    """
    Build a fully grounded RAG prompt with retrieved document context.

    Limits injected context to MAX_CONTEXT_CHUNKS to stay within
    the LLM's token budget.
    """
    # Respect the token budget: cap injected context chunks
    capped_chunks = context_chunks[:MAX_CONTEXT_CHUNKS]
    context_block = "\n\n---\n\n".join(capped_chunks)
    history_block = _format_history(history)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Relevant Content from Uploaded Study Material:\n"
        f"{context_block}\n\n"
        f"### Conversation So Far:\n"
        f"{history_block}\n\n"
        f"### Student's Question:\n"
        f"{user_message}\n\n"
        f"### Tutor's Response:"
    )
    return prompt


def _build_plain_prompt(history: List[dict], user_message: str) -> str:
    """
    Build a plain conversational prompt without document context.

    Used when no PDFs have been uploaded for the session yet.
    """
    history_block = _format_history(history)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Conversation So Far:\n"
        f"{history_block}\n\n"
        f"### Student's Question:\n"
        f"{user_message}\n\n"
        f"### Tutor's Response:"
    )
    return prompt


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_pipeline(
    session_id: str,
    user_message: str,
    history: List[dict],
) -> str:
    """
    Execute the full AI Tutor pipeline for a single user message.

    Automatically selects RAG path or plain LLM path based on whether
    the session has any indexed documents in ChromaDB.

    Args:
        session_id: The active session identifier.
        user_message: The student's latest message/question.
        history: Windowed chat history (excluding the current message).

    Returns:
        The AI tutor's response as a plain string.

    Raises:
        RuntimeError: Propagated from llm.generate() if Ollama is unavailable.
    """
    if session_has_documents(session_id):
        logger.info(f"[Pipeline] RAG path → session '{session_id}'.")

        # Step 1: Retrieve semantically relevant chunks
        context_chunks = retrieve(session_id, user_message)

        if context_chunks:
            # Step 2a: Build grounded prompt with context
            prompt = _build_rag_prompt(context_chunks, history, user_message)
            logger.info(
                f"[Pipeline] Injecting {min(len(context_chunks), MAX_CONTEXT_CHUNKS)} "
                f"context chunk(s) into prompt."
            )
        else:
            # Retrieval returned nothing — fall back gracefully to plain prompt
            logger.warning(
                f"[Pipeline] RAG retrieval returned 0 chunks for session '{session_id}'. "
                f"Falling back to plain LLM prompt."
            )
            prompt = _build_plain_prompt(history, user_message)
    else:
        logger.info(
            f"[Pipeline] Plain LLM path → session '{session_id}' "
            f"(no documents uploaded yet)."
        )
        prompt = _build_plain_prompt(history, user_message)

    # Step 3: Generate and return response
    return generate(prompt)
