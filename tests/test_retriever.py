import pytest

from src.retriever import retrieve_evidence
from src.schemas import Claim, RetrievedEvidence


class FakeVectorStore:
    def __init__(self, results: list[RetrievedEvidence]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def _evidence(index: int) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=f"paper_p{index}_c1",
        paper_id="paper",
        filename="study.pdf",
        page_number=index,
        text=f"Evidence {index}",
        score=1.0 - index / 10,
    )


def test_retrieval_preserves_metadata_score_and_top_k() -> None:
    store = FakeVectorStore([_evidence(1), _evidence(2), _evidence(3)])
    claim = Claim(claim_id="claim_001", text="A testable claim")

    results = retrieve_evidence(claim, store, top_k=2)  # type: ignore[arg-type]

    assert store.calls == [("A testable claim", 2)]
    assert [item.chunk_id for item in results] == ["paper_p1_c1", "paper_p2_c1"]
    assert results[0].filename == "study.pdf"
    assert results[0].page_number == 1
    assert results[0].score == pytest.approx(0.9)


def test_retrieval_propagates_empty_results() -> None:
    store = FakeVectorStore([])
    claim = Claim(claim_id="claim_001", text="No matching evidence")

    assert retrieve_evidence(claim, store) == []  # type: ignore[arg-type]


def test_retrieval_rejects_invalid_top_k() -> None:
    store = FakeVectorStore([])
    claim = Claim(claim_id="claim_001", text="Claim")

    with pytest.raises(ValueError):
        retrieve_evidence(claim, store, top_k=0)  # type: ignore[arg-type]
