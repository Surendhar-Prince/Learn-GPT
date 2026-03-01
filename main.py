"""
main.py — AI Tutor Backend Entry Point
────────────────────────────────────────
FastAPI application with:
  - Structured logging (timestamp | level | module | message)
  - CORS middleware (configurable via CORS_ORIGINS in config.py)
  - Lifespan handler: Ollama model pre-warm on startup
  - Background PDF embedding via FastAPI BackgroundTasks
  - In-memory embedding status registry per session
  - Full exception handling with descriptive HTTP errors

Endpoints:
  POST /chat                   → Chat with AI Tutor (creates session if needed)
  POST /upload                 → Upload PDF for RAG (async background embedding)
  GET  /history/{session_id}   → Retrieve full chat history
  GET  /status/{session_id}    → Poll background PDF embedding status
  GET  /health                 → Health check

Run with:
  uvicorn main:app --reload --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat_manager.manager import (
    create_session,
    get_windowed_history,
    load_history,
    save_message,
    session_exists,
)
from config import CHAT_DIR, CORS_ORIGINS, MAX_PDF_BYTES
from rag_pipeline.chunking import chunk_text
from rag_pipeline.llm import prewarm
from rag_pipeline.pipeline import run_pipeline, run_pipeline_stream
from rag_pipeline.vector_store import add_documents
from utils.pdf_loader import extract_text_from_pdf

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── In-memory embedding status registry ──────────────────────────────────────
# Maps session_id → embedding status string
# Status values: "pending" | "ready" | "error"
# Note: This is per-process in-memory state. A server restart resets statuses,
# but vector data persists in ChromaDB. For multi-process deploys, move this
# to Redis or a DB in v2.
embedding_status: dict[str, str] = {}


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Pre-warm the Ollama LLM in a thread pool executor so it doesn't
             block the event loop. Server remains fully available during warm-up.
    Shutdown: Log clean exit.
    """
    logger.info("=" * 60)
    logger.info("AI Tutor Backend starting up ...")
    logger.info("=" * 60)

    # Fire-and-forget Ollama pre-warm in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, prewarm)

    yield  # Server is live here

    logger.info("AI Tutor Backend shutting down. Goodbye.")


# ── App initialisation ────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Tutor Backend",
    description=(
        "Production-ready RAG-powered AI tutoring API. "
        "Upload PDFs and chat with an AI tutor grounded in your study material."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins by default for local dev
# Override CORS_ORIGINS env var in production: "http://localhost:3000,https://yourdomain.com"
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files & templates ──────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session UUID. Omit to create a new session automatically.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="The student's message or question.",
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str


class UploadResponse(BaseModel):
    message: str
    session_id: str
    filename: str


class HistoryResponse(BaseModel):
    session_id: str
    message_count: int
    history: list[dict]


class StatusResponse(BaseModel):
    session_id: str
    embedding_status: str


# ── Background task: PDF embedding ───────────────────────────────────────────

def _embed_pdf_in_background(
    session_id: str,
    file_bytes: bytes,
    filename: str,
) -> None:
    """
    Background worker: extract → chunk → embed → store.

    Runs in FastAPI's thread pool executor (not the async event loop).
    Updates embedding_status[session_id] during each stage.
    """
    try:
        logger.info(f"[BG] Starting PDF embedding | session='{session_id}' | file='{filename}'")
        embedding_status[session_id] = "pending"

        # Step 1: Extract text
        text = extract_text_from_pdf(file_bytes)
        logger.info(f"[BG] Extracted {len(text):,} chars from '{filename}'.")

        # Step 2: Chunk text
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Chunking produced zero chunks. Check PDF content.")
        logger.info(f"[BG] Produced {len(chunks)} chunks from '{filename}'.")

        # Step 3: Embed + store in ChromaDB
        add_documents(session_id, chunks)

        embedding_status[session_id] = "ready"
        logger.info(
            f"[BG] Embedding complete | session='{session_id}' | "
            f"chunks={len(chunks)} | file='{filename}'"
        )

    except Exception as e:
        embedding_status[session_id] = "error"
        logger.error(
            f"[BG] PDF embedding FAILED | session='{session_id}' | "
            f"file='{filename}' | error={e}"
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend(request: Request):
    """Serve the frontend SPA."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", tags=["System"], summary="Health check")
async def health_check():
    """
    Returns 200 OK if the server is running.
    Use this to verify the backend is alive before making other requests.
    """
    return {"status": "ok", "service": "ai-tutor-backend", "version": "1.0.0"}


@app.post(
    "/chat",
    tags=["Chat"],
    summary="Send a message to the AI Tutor (streaming)",
)
async def chat(request: ChatRequest):
    """
    Send a student message and receive a real-time streamed AI Tutor response.

    - Response is delivered as a plain-text stream (text/plain) so the
      frontend can progressively render tokens as they arrive.
    - Session management and message persistence are unchanged.
    - The session_id is sent as the very first line of the stream:
        SESSION:<uuid>\n
      followed immediately by the LLM tokens. The frontend strips this
      header to extract the session_id without a separate JSON round-trip.
    """
    # ── Session management ────────────────────────────────────────────────────
    if not request.session_id:
        session_id = create_session()
        logger.info(f"Auto-created new session: {session_id}")
    else:
        session_id = request.session_id
        if not session_exists(session_id):
            create_session(session_id)
            logger.info(f"Auto-created session for provided ID: {session_id}")

    # ── Save user message ─────────────────────────────────────────────────────
    save_message(session_id, "user", request.message)

    # ── Build LLM context (windowed history, excluding current msg) ───────────
    full_window = get_windowed_history(session_id)
    history_for_llm = full_window[:-1]

    # ── Streaming generator ───────────────────────────────────────────────────
    def token_stream():
        """
        Yields:
          1. A session header line so the frontend can recover the session_id.
          2. Raw LLM token chunks as they arrive.
        After the stream is exhausted the full response is saved to chat history.
        """
        # Header: lets the client know the (possibly auto-created) session_id
        yield f"SESSION:{session_id}\n"

        collected = []
        try:
            for token in run_pipeline_stream(
                session_id=session_id,
                user_message=request.message,
                history=history_for_llm,
            ):
                collected.append(token)
                yield token
        except RuntimeError as e:
            logger.error(f"[Stream] Pipeline error for session '{session_id}': {e}")
            yield f"\n\n[ERROR] {e}"
            return
        except Exception as e:
            logger.error(
                f"[Stream] Unexpected pipeline error for session '{session_id}': {e}"
            )
            yield "\n\n[ERROR] An unexpected error occurred. Check server logs."
            return

        # ── Persist the full assembled response ───────────────────────────────
        full_response = "".join(collected).strip()
        if full_response:
            save_message(session_id, "assistant", full_response)
            logger.info(
                f"[Stream] Saved assistant response "
                f"({len(full_response)} chars) for session '{session_id}'."
            )

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["PDF"],
    summary="Upload a PDF for RAG",
)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload (max 50 MB)."),
    session_id: str = Form(..., description="Session UUID to associate this PDF with."),
):
    """
    Upload a PDF document to a session.

    Text extraction and embedding happen **asynchronously in the background**.
    This endpoint returns immediately. Poll **GET /status/{session_id}** to
    check when embedding is complete before sending RAG-dependent messages.

    Multiple PDFs can be uploaded to the same session — all will be indexed
    and searched during retrieval.
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read file bytes immediately — UploadFile is not safe across async boundaries
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MB size limit.",
        )

    # ── Ensure session exists ─────────────────────────────────────────────────
    if not session_exists(session_id):
        create_session(session_id)
        logger.info(f"Session auto-created for upload: {session_id}")

    # ── Queue background embedding ────────────────────────────────────────────
    background_tasks.add_task(
        _embed_pdf_in_background, session_id, file_bytes, file.filename
    )
    logger.info(
        f"PDF '{file.filename}' queued for background embedding | session='{session_id}'"
    )

    return UploadResponse(
        message=(
            "PDF received and queued for embedding. "
            f"Poll GET /status/{session_id} to check progress."
        ),
        session_id=session_id,
        filename=file.filename,
    )


@app.get(
    "/status/{session_id}",
    response_model=StatusResponse,
    tags=["PDF"],
    summary="Check PDF embedding status",
)
async def get_embedding_status(session_id: str):
    """
    Poll the background PDF embedding status for a session.

    Possible values:
    - `not_started` — No PDF has been uploaded yet (or server was restarted).
    - `pending`     — PDF is currently being processed.
    - `ready`       — Embedding complete; RAG is now active for this session.
    - `error`       — Embedding failed; check server logs for details.
    """
    status = embedding_status.get(session_id, "not_started")
    return StatusResponse(session_id=session_id, embedding_status=status)


@app.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    tags=["Chat"],
    summary="Retrieve full chat history",
)
async def get_history(session_id: str):
    """
    Retrieve the complete chat history for a session.

    Returns all messages (not windowed) so the frontend can display
    the full conversation. Structure: [{"role": "user"|"assistant", "content": "..."}, ...]
    """
    if not session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Start a chat first.",
        )

    history = load_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        message_count=len(history),
        history=history,
    )
