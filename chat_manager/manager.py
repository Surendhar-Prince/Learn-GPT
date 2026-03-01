"""
chat_manager/manager.py — Persistent Chat Session Manager
──────────────────────────────────────────────────────────
Responsibilities:
  - Create sessions (UUID-based)
  - Save messages atomically (filelock-protected JSON)
  - Load full history
  - Return windowed history for LLM context

Storage format:  data/chats/{session_id}.json
  [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]

Thread safety: Every read/write acquires a per-session filelock so that
concurrent requests to the same session cannot corrupt the JSON file.
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Optional

from filelock import FileLock

from config import CHAT_DIR, MEMORY_WINDOW

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _session_path(session_id: str) -> Path:
    """Return the JSON file path for a given session."""
    return CHAT_DIR / f"{session_id}.json"


def _lock_path(session_id: str) -> Path:
    """Return the lockfile path for a given session."""
    return CHAT_DIR / f"{session_id}.lock"


def _read_messages(session_id: str) -> list[dict]:
    """Read messages from disk. Must be called inside an acquired lock."""
    path = _session_path(session_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_messages(session_id: str, messages: list[dict]) -> None:
    """Write messages to disk. Must be called inside an acquired lock."""
    path = _session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


# ── Public API ────────────────────────────────────────────────────────────────

def create_session(session_id: Optional[str] = None) -> str:
    """
    Create a new chat session.

    Args:
        session_id: Optional custom UUID. If None, a new UUID is generated.

    Returns:
        The session_id string.
    """
    sid = session_id or str(uuid.uuid4())
    path = _session_path(sid)

    with FileLock(str(_lock_path(sid))):
        if not path.exists():
            _write_messages(sid, [])
            logger.info(f"Session created: {sid}")
        else:
            logger.debug(f"Session already exists, skipping init: {sid}")

    return sid


def session_exists(session_id: str) -> bool:
    """Return True if the session's JSON file exists on disk."""
    return _session_path(session_id).exists()


def save_message(session_id: str, role: str, content: str) -> None:
    """
    Append a single message to the session's chat history.

    Args:
        session_id: Target session identifier.
        role: Either "user" or "assistant".
        content: The message text.
    """
    if role not in {"user", "assistant"}:
        raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")

    lock = FileLock(str(_lock_path(session_id)))
    with lock:
        messages = _read_messages(session_id)
        messages.append({"role": role, "content": content})
        _write_messages(session_id, messages)

    logger.debug(f"Saved [{role}] message to session '{session_id}'.")


def load_history(session_id: str) -> list[dict]:
    """
    Load the complete chat history for a session.

    Returns:
        List of message dicts: [{"role": ..., "content": ...}, ...]
        Returns an empty list if the session does not exist.
    """
    with FileLock(str(_lock_path(session_id))):
        return _read_messages(session_id)


def get_windowed_history(session_id: str) -> list[dict]:
    """
    Return only the most recent MEMORY_WINDOW messages.

    This prevents the LLM context from growing unboundedly across long sessions.
    """
    history = load_history(session_id)
    windowed = history[-MEMORY_WINDOW:] if len(history) > MEMORY_WINDOW else history
    logger.debug(
        f"Returning {len(windowed)} of {len(history)} messages for session '{session_id}'."
    )
    return windowed


def delete_session(session_id: str) -> bool:
    """
    Delete a session's chat file from disk.

    Returns:
        True if the file was deleted, False if it didn't exist.
    """
    path = _session_path(session_id)
    lock_file = _lock_path(session_id)

    with FileLock(str(lock_file)):
        if path.exists():
            path.unlink()
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
