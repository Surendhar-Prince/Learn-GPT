"""
utils/pdf_loader.py — PDF Text Extractor
──────────────────────────────────────────
Extracts and cleans plain text from PDF files using PyMuPDF (fitz).

Handles:
  - Multi-page PDFs (iterates every page)
  - Binary bytes input (no temp files needed)
  - Whitespace normalisation and artifact removal
  - Image-only / scanned PDFs (raises a clear ValueError)
  - Corrupt or unreadable PDFs (raises a clear ValueError)

Deliberately not responsible for chunking or embedding —
those responsibilities live in rag_pipeline/chunking.py
and rag_pipeline/vector_store.py respectively.
"""

import re
import logging
from io import BytesIO

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract and clean all text from a PDF given its raw byte content.

    Args:
        file_bytes: Raw bytes of the PDF file (e.g. from UploadFile.read()).

    Returns:
        A single cleaned plain-text string containing all extractable text
        from all pages, separated by double newlines.

    Raises:
        ValueError: If the file cannot be opened as a PDF.
        ValueError: If the PDF contains no pages.
        ValueError: If no text could be extracted (image-only / scanned PDF).
    """
    # Open the PDF from bytes (no temp file needed)
    try:
        doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open file as a valid PDF: {e}") from e

    if doc.page_count == 0:
        doc.close()
        raise ValueError("PDF file has no pages.")

    logger.info(f"Extracting text from PDF ({doc.page_count} pages) ...")

    page_texts: list[str] = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text")  # "text" mode preserves layout
        cleaned = _clean_page_text(raw_text)
        if cleaned:
            page_texts.append(cleaned)

    doc.close()

    if not page_texts:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned/image-only document. "
            "Consider running OCR preprocessing first."
        )

    full_text = "\n\n".join(page_texts)

    logger.info(
        f"Extracted {len(full_text):,} characters from "
        f"{len(page_texts)} text-bearing page(s)."
    )

    return full_text


# ── Private Helpers ───────────────────────────────────────────────────────────

def _clean_page_text(text: str) -> str:
    """
    Clean raw text extracted from a single PDF page.

    Transformations applied:
      1. Remove lines that are purely whitespace or empty
      2. Collapse 3+ consecutive newlines to a double newline
      3. Strip leading/trailing whitespace from the whole block

    Args:
        text: Raw string from page.get_text("text").

    Returns:
        Cleaned string, or empty string if nothing remains after cleaning.
    """
    if not text:
        return ""

    # Remove lines that are only whitespace
    lines = [line for line in text.splitlines() if line.strip()]
    joined = "\n".join(lines)

    # Collapse excessive blank lines
    joined = re.sub(r"\n{3,}", "\n\n", joined)

    return joined.strip()
