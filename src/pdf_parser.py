"""Page-level text extraction for selectable-text PDF files."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pymupdf as fitz

from src.schemas import Page


logger = logging.getLogger(__name__)


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or read."""


def _normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to one space."""
    return re.sub(r"\s+", " ", text).strip()


def _stable_paper_id(pdf_path: Path) -> str:
    """Create a stable, compact identifier from the PDF file bytes."""
    digest = hashlib.sha256()
    try:
        with pdf_path.open("rb") as pdf_file:
            for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PDFExtractionError(f"Could not read PDF '{pdf_path}': {exc}") from exc
    return digest.hexdigest()[:16]


def extract_pdf_pages(
    pdf_path: str,
    paper_id: str | None = None,
) -> list[Page]:
    """Extract non-empty pages while retaining user-facing source metadata.

    Page numbers refer to positions in the original PDF and are always 1-indexed,
    even when an empty page is skipped.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise PDFExtractionError(f"PDF file does not exist: '{path}'")

    resolved_paper_id = paper_id.strip() if paper_id is not None else _stable_paper_id(path)
    if not resolved_paper_id:
        raise ValueError("paper_id must not be empty")

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise PDFExtractionError(f"Could not open PDF '{path}': {exc}") from exc

    pages: list[Page] = []
    try:
        if not document.is_pdf:
            raise PDFExtractionError(f"File is not a PDF: '{path}'")

        for index, pdf_page in enumerate(document):
            try:
                text = _normalize_whitespace(pdf_page.get_text("text"))
            except Exception as exc:
                raise PDFExtractionError(
                    f"Could not extract page {index + 1} from '{path}': {exc}"
                ) from exc

            if not text:
                continue

            pages.append(
                Page(
                    paper_id=resolved_paper_id,
                    filename=path.name,
                    page_number=index + 1,
                    text=text,
                )
            )
    finally:
        document.close()

    logger.info("Extracted %d non-empty pages from %s", len(pages), path.name)
    return pages
