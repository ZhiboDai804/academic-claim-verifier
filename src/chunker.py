"""Deterministic page-local word chunking."""

from __future__ import annotations

import logging
import re

from src.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
from src.schemas import Chunk, Page


logger = logging.getLogger(__name__)


def chunk_pages(
    pages: list[Page],
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Split each page independently into overlapping word windows."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")
    if overlap_words < 0:
        raise ValueError("overlap_words must be zero or greater")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    step = chunk_size_words - overlap_words

    for page in pages:
        word_matches = list(re.finditer(r"\S+", page.text))
        for chunk_index, start_word in enumerate(range(0, len(word_matches), step), start=1):
            end_word = min(start_word + chunk_size_words, len(word_matches))
            if start_word >= end_word:
                continue

            start_char = word_matches[start_word].start()
            end_char = word_matches[end_word - 1].end()
            chunk_id = f"{page.paper_id}_p{page.page_number}_c{chunk_index}"
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk ID generated: {chunk_id}")
            seen_ids.add(chunk_id)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=page.paper_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    text=page.text[start_char:end_char],
                    start_char=start_char,
                    end_char=end_char,
                )
            )

            if end_word == len(word_matches):
                break

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks
