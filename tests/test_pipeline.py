from pathlib import Path
from uuid import uuid4

import chromadb
import pymupdf

from src.pipeline import ClaimVerificationPipeline
from src.schemas import (
    Chunk,
    Claim,
    Page,
    RetrievedEvidence,
    VerificationResult,
)
from src.vector_store import PaperVectorStore


class FakeVectorStore:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.added_chunks: list[Chunk] = []
        self.search_calls: list[tuple[str, int]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.added_chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        self.search_calls.append((query, top_k))
        return [
            RetrievedEvidence(
                chunk_id="paper_p4_c1",
                paper_id="paper",
                filename="paper.pdf",
                page_number=4,
                text=f"Evidence for {query}",
                score=0.87,
            )
        ]


def test_ingest_pdfs_rebuilds_index_and_records_stats(monkeypatch: object) -> None:
    store = FakeVectorStore()
    pipeline = ClaimVerificationPipeline(vector_store=store)  # type: ignore[arg-type]

    def fake_extract(path: str) -> list[Page]:
        return [
            Page(
                paper_id=Path(path).stem,
                filename=Path(path).name,
                page_number=1,
                text="Page text",
            )
        ]

    def fake_chunk(pages: list[Page]) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{page.paper_id}_p1_c1",
                paper_id=page.paper_id,
                filename=page.filename,
                page_number=page.page_number,
                text=page.text,
            )
            for page in pages
        ]

    monkeypatch.setattr("src.pipeline.extract_pdf_pages", fake_extract)  # type: ignore[attr-defined]
    monkeypatch.setattr("src.pipeline.chunk_pages", fake_chunk)  # type: ignore[attr-defined]

    pipeline.ingest_pdfs(["first.pdf", "second.pdf"])

    assert store.reset_calls == 1
    assert [chunk.filename for chunk in store.added_chunks] == [
        "first.pdf",
        "second.pdf",
    ]
    assert pipeline.last_ingestion_stats.model_dump() == {
        "papers": 2,
        "pages": 2,
        "chunks": 2,
    }


def test_verify_text_preserves_retrieved_source_metadata(
    monkeypatch: object,
) -> None:
    store = FakeVectorStore()
    pipeline = ClaimVerificationPipeline(vector_store=store, top_k=3)  # type: ignore[arg-type]
    claims = [
        Claim(claim_id="claim_001", text="First atomic claim."),
        Claim(claim_id="claim_002", text="Second atomic claim."),
    ]

    monkeypatch.setattr("src.pipeline.extract_claims", lambda text: claims)  # type: ignore[attr-defined]

    def fake_verify(
        claim: Claim,
        evidence: list[RetrievedEvidence],
    ) -> VerificationResult:
        assert evidence[0].filename == "paper.pdf"
        return VerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            label="SUPPORTED",
            reason="The supplied passage supports the claim.",
            evidence_ids=["E1"],
        )

    monkeypatch.setattr("src.pipeline.verify_claim", fake_verify)  # type: ignore[attr-defined]

    reports = pipeline.verify_text("A paragraph with two claims.")

    assert len(reports) == 2
    assert store.search_calls == [
        ("First atomic claim.", 3),
        ("Second atomic claim.", 3),
    ]
    assert reports[0].verification.label == "SUPPORTED"
    assert reports[0].retrieved_evidence[0].chunk_id == "paper_p4_c1"
    assert reports[0].retrieved_evidence[0].filename == "paper.pdf"
    assert reports[0].retrieved_evidence[0].page_number == 4
    assert reports[0].retrieved_evidence[0].score == 0.87


def test_verify_text_returns_empty_reports_for_no_claims(monkeypatch: object) -> None:
    store = FakeVectorStore()
    pipeline = ClaimVerificationPipeline(vector_store=store)  # type: ignore[arg-type]
    monkeypatch.setattr("src.pipeline.extract_claims", lambda text: [])  # type: ignore[attr-defined]

    assert pipeline.verify_text("No factual claims here.") == []
    assert store.search_calls == []


def test_deterministic_end_to_end_flow_preserves_pdf_citation(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pdf_path = tmp_path / "grounding-study.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Retrieval augmentation reduced unsupported factual generations.",
    )
    document.save(pdf_path)
    document.close()

    store = PaperVectorStore(
        collection_name=f"pipeline-integration-{uuid4().hex}",
        embedding_function=lambda texts: [[1.0, 0.0] for _ in texts],
        client=chromadb.Client(),
    )
    pipeline = ClaimVerificationPipeline(vector_store=store)
    pipeline.ingest_pdfs([str(pdf_path)])

    claim = Claim(claim_id="claim_001", text="RAG reduces hallucination.")
    monkeypatch.setattr("src.pipeline.extract_claims", lambda text: [claim])  # type: ignore[attr-defined]

    def fake_verify(
        input_claim: Claim,
        evidence: list[RetrievedEvidence],
    ) -> VerificationResult:
        assert evidence[0].filename == "grounding-study.pdf"
        assert evidence[0].page_number == 1
        return VerificationResult(
            claim_id=input_claim.claim_id,
            claim_text=input_claim.text,
            label="SUPPORTED",
            reason="The retrieved passage supports the claim.",
            evidence_ids=["E1"],
        )

    monkeypatch.setattr("src.pipeline.verify_claim", fake_verify)  # type: ignore[attr-defined]

    reports = pipeline.verify_text("RAG reduces hallucination.")

    assert reports[0].retrieved_evidence[0].filename == "grounding-study.pdf"
    assert reports[0].retrieved_evidence[0].page_number == 1
    assert reports[0].retrieved_evidence[0].chunk_id.endswith("_p1_c1")
