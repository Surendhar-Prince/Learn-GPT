"""
rag_pipeline/llm.py — Local LLM Interface via Ollama
──────────────────────────────────────────────────────
Sends prompts to a locally-running Ollama instance and returns the
generated text response.

Default model: llama3 (best for educational tutoring — clear reasoning,
structured explanations, low hallucination risk).

Configurable via:
  config.py → LLM_MODEL, OLLAMA_BASE_URL
  Environment variable overrides also supported.

To switch models, set LLM_MODEL in config.py or:
  export LLM_MODEL=mistral
  export LLM_MODEL=hermes-mistral
"""

import logging

import ollama

from config import LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


def generate(prompt: str) -> str:
    """
    Send a prompt string to Ollama and return the generated response.

    Args:
        prompt: The complete formatted prompt string (system + context + query).

    Returns:
        The LLM's text response as a plain string.

    Raises:
        RuntimeError: If Ollama is unreachable or the model is not available locally.
                      The error message includes actionable instructions for the user.
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    try:
        logger.debug(f"Sending prompt to Ollama model '{LLM_MODEL}' ({len(prompt)} chars).")

        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.generate(model=LLM_MODEL, prompt=prompt)

        answer = response["response"].strip()

        logger.debug(f"Received response from Ollama ({len(answer)} chars).")
        return answer

    except Exception as e:
        logger.error(f"Ollama LLM call failed: {e}")
        raise RuntimeError(
            f"LLM is unavailable. "
            f"Ensure Ollama is running and the model '{LLM_MODEL}' is pulled locally.\n"
            f"Fix: Run  →  ollama pull {LLM_MODEL}\n"
            f"Then restart Ollama and retry.\n"
            f"Internal error: {e}"
        ) from e


def prewarm() -> None:
    """
    Send a trivial prompt to Ollama to load the model weights into memory.

    Called once at server startup (in a background thread) to eliminate the
    5–15 second cold-start delay on the first real user request.

    This is a best-effort operation — failure is logged but not fatal.
    """
    try:
        logger.info(f"Pre-warming Ollama model '{LLM_MODEL}' ...")
        generate("Hello.")
        logger.info(f"Ollama pre-warm complete. Model '{LLM_MODEL}' is ready.")
    except RuntimeError as e:
        logger.warning(
            f"Ollama pre-warm failed (non-fatal — server will still start): {e}"
        )
