"""High-level deterministic orchestration for ingestion and verification."""

from __future__ import annotations

import logging

from src.chunker import chunk_pages
from src.claim_extractor import extract_claims
from src.config import TOP_K
from src.pdf_parser import extract_pdf_pages
from src.retriever import retrieve_evidence
from src.schemas import ClaimVerificationReport, IngestionStats
from src.vector_store import PaperVectorStore
from src.verifier import verify_claim


logger = logging.getLogger(__name__)


class ClaimVerificationPipeline:
    """Compose the inspectable ingestion, retrieval, and verification stages."""

    def __init__(
        self,
        vector_store: PaperVectorStore | None = None,
        *,
        top_k: int = TOP_K,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._vector_store = vector_store or PaperVectorStore()
        self._top_k = top_k
        self._last_ingestion_stats = IngestionStats(papers=0, pages=0, chunks=0)

    @property
    def vector_store(self) -> PaperVectorStore:
        """Expose the store for evaluation without duplicating retrieval logic."""
        return self._vector_store

    @property
    def last_ingestion_stats(self) -> IngestionStats:
        return self._last_ingestion_stats

    def ingest_pdfs(self, pdf_paths: list[str]) -> None:
        """Parse, page-chunk, and rebuild the index from supplied PDFs."""
        if not pdf_paths:
            raise ValueError("At least one PDF path is required")

        pages = []
        for pdf_path in pdf_paths:
            pages.extend(extract_pdf_pages(pdf_path))

        if not pages:
            raise ValueError("The supplied PDFs contained no extractable text")

        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("The supplied PDFs produced no indexable chunks")

        self._vector_store.reset()
        self._vector_store.add_chunks(chunks)
        self._last_ingestion_stats = IngestionStats(
            papers=len(pdf_paths),
            pages=len(pages),
            chunks=len(chunks),
        )
        logger.info(
            "Ingested %d papers, %d pages, and %d chunks",
            len(pdf_paths),
            len(pages),
            len(chunks),
        )

    def verify_text(self, text: str) -> list[ClaimVerificationReport]:
        """Extract claims, retrieve evidence, and verify each claim in order."""
        claims = extract_claims(text)
        reports: list[ClaimVerificationReport] = []

        for claim in claims:
            evidence = retrieve_evidence(
                claim,
                self._vector_store,
                top_k=self._top_k,
            )
            verification = verify_claim(claim, evidence)
            reports.append(
                ClaimVerificationReport(
                    claim=claim,
                    verification=verification,
                    retrieved_evidence=evidence,
                )
            )

        logger.info("Verified %d claims", len(reports))
        return reports
