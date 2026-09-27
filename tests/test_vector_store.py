from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest

from src.schemas import Chunk
from src.vector_store import PaperVectorStore


def _chunk(
    chunk_id: str,
    text: str,
    *,
    paper_id: str,
    filename: str,
    page_number: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        filename=filename,
        page_number=page_number,
        text=text,
    )


def _deterministic_embeddings(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lower_text = text.lower()
        vectors.append(
            [
                float(any(term in lower_text for term in ("rag", "retrieval", "hallucination"))),
                float(any(term in lower_text for term in ("banana", "fruit", "potassium"))),
                0.1,
            ]
        )
    return vectors


@pytest.fixture
def store() -> PaperVectorStore:
    return PaperVectorStore(
        collection_name=f"paper-vector-store-tests-{uuid4().hex}",
        embedding_function=_deterministic_embeddings,
        client=chromadb.Client(),
    )


def test_search_returns_relevant_text_and_complete_metadata(
    store: PaperVectorStore,
) -> None:
    chunks = [
        _chunk(
            "rag_p7_c1",
            "Retrieval augmentation reduced unsupported factual generations.",
            paper_id="rag",
            filename="rag-study.pdf",
            page_number=7,
        ),
        _chunk(
            "fruit_p2_c1",
            "Bananas are a fruit containing potassium.",
            paper_id="fruit",
            filename="fruit-study.pdf",
            page_number=2,
        ),
    ]
    store.add_chunks(chunks)

    results = store.search("RAG reduces hallucination", top_k=2)

    assert results[0].chunk_id == "rag_p7_c1"
    assert results[0].paper_id == "rag"
    assert results[0].filename == "rag-study.pdf"
    assert results[0].page_number == 7
    assert results[0].text == chunks[0].text
    assert results[0].score == pytest.approx(1.0)
    assert isinstance(results, Sequence)


def test_empty_collection_and_reset_return_no_results(store: PaperVectorStore) -> None:
    assert store.search("RAG reduces hallucination") == []

    store.add_chunks(
        [
            _chunk(
                "rag_p1_c1",
                "RAG reduces hallucination.",
                paper_id="rag",
                filename="rag.pdf",
                page_number=1,
            )
        ]
    )
    store.reset()

    assert store.search("RAG reduces hallucination") == []


def test_add_chunks_is_noop_for_empty_input(store: PaperVectorStore) -> None:
    store.add_chunks([])
    assert store.search("anything") == []


def test_default_client_persists_records_between_instances(tmp_path: Path) -> None:
    collection_name = f"paper-vector-store-persistence-{uuid4().hex}"
    first_store = PaperVectorStore(
        persist_directory=tmp_path,
        collection_name=collection_name,
        embedding_function=_deterministic_embeddings,
    )
    first_store.add_chunks(
        [
            _chunk(
                "rag_p3_c1",
                "Retrieval systems can ground generated answers.",
                paper_id="rag",
                filename="rag.pdf",
                page_number=3,
            )
        ]
    )

    reopened_store = PaperVectorStore(
        persist_directory=tmp_path,
        collection_name=collection_name,
        embedding_function=_deterministic_embeddings,
    )

    assert reopened_store.search("RAG retrieval", top_k=1)[0].chunk_id == "rag_p3_c1"


def test_duplicate_chunk_ids_are_rejected(store: PaperVectorStore) -> None:
    chunk = _chunk(
        "duplicate",
        "text",
        paper_id="paper",
        filename="paper.pdf",
        page_number=1,
    )

    with pytest.raises(ValueError, match="unique"):
        store.add_chunks([chunk, chunk])


@pytest.mark.parametrize(("query", "top_k"), [("", 1), ("   ", 1), ("valid", 0)])
def test_invalid_search_arguments_are_rejected(
    store: PaperVectorStore,
    query: str,
    top_k: int,
) -> None:
    with pytest.raises(ValueError):
        store.search(query, top_k=top_k)
