"""
config.py — Centralised Configuration for AI Tutor Backend
────────────────────────────────────────────────────────────
All tunable parameters live here.  Every value reads from an environment
variable first, then falls back to a sensible default.

To override without editing this file:
    export EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
    export LLM_MODEL=mistral
    export MEMORY_WINDOW=12
    ...
"""

import os
from pathlib import Path

# ── Directory Layout ──────────────────────────────────────────────────────────

# Root of the project (same folder as this file)
BASE_DIR: Path = Path(__file__).resolve().parent

# All persistent data lives under data/
BASE_DATA_DIR: Path = BASE_DIR / "data"

# Chat history JSON files:  data/chats/{session_id}.json
CHAT_DIR: Path = BASE_DATA_DIR / "chats"

# Single shared ChromaDB store:  data/vectors/
VECTOR_DIR: Path = BASE_DATA_DIR / "vectors"

# ── Embedding Model ───────────────────────────────────────────────────────────

# Best semantic accuracy for concept-dense educational content.
# Swap to "multi-qa-MiniLM-L6-cos-v1" for lower RAM / faster speed.
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")

# ── Text Chunking ─────────────────────────────────────────────────────────────

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))        # characters per chunk
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "60"))   # overlap between chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────

RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))              # top-k chunks fetched
MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "4")) # max injected into prompt

# ── LLM (Ollama) ─────────────────────────────────────────────────────────────

# Recommended for educational tutoring: clear reasoning, low hallucination.
# Switch to "hermes-mistral", "mistral", etc. via env var.
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")

# Ollama server address (default local)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Chat Memory ───────────────────────────────────────────────────────────────

# Number of recent messages (user + assistant combined) passed to LLM as context.
MEMORY_WINDOW: int = int(os.getenv("MEMORY_WINDOW", "10"))

# ── PDF Upload ────────────────────────────────────────────────────────────────

MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
MAX_PDF_BYTES: int = MAX_PDF_SIZE_MB * 1024 * 1024

# ── CORS ──────────────────────────────────────────────────────────────────────

# Comma-separated list of allowed origins, e.g. "http://localhost:3000,http://localhost:5173"
# Defaults to "*" for local development — RESTRICT THIS in production.
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Auto-create required directories ─────────────────────────────────────────

CHAT_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
