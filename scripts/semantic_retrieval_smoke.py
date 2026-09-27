r"""Manual OpenAI + Chroma semantic retrieval smoke test.

Run from the repository root after configuring ``OPENAI_API_KEY``:

    .venv\Scripts\python.exe scripts\semantic_retrieval_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import chromadb


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas import Chunk  # noqa: E402
from src.vector_store import PaperVectorStore  # noqa: E402


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY before running this smoke test.")
        return 2

    chunks = [
        Chunk(
            chunk_id="rag_p6_c1",
            paper_id="rag",
            filename="rag-study.pdf",
            page_number=6,
            text="Retrieval augmentation reduced unsupported factual generations.",
        ),
        Chunk(
            chunk_id="fruit_p2_c1",
            paper_id="fruit",
            filename="fruit-study.pdf",
            page_number=2,
            text="Bananas are rich in potassium and are widely cultivated.",
        ),
        Chunk(
            chunk_id="systems_p4_c1",
            paper_id="systems",
            filename="systems-study.pdf",
            page_number=4,
            text="The caching layer reduced median request latency.",
        ),
    ]
    store = PaperVectorStore(
        collection_name=f"smoke-{uuid4().hex}",
        client=chromadb.Client(),
    )
    store.add_chunks(chunks)

    results = store.search("RAG reduces hallucination", top_k=3)
    for rank, evidence in enumerate(results, start=1):
        print(
            f"{rank}. {evidence.chunk_id} | {evidence.filename} "
            f"p.{evidence.page_number} | score={evidence.score:.4f}"
        )

    if not results or results[0].chunk_id != "rag_p6_c1":
        print("Smoke test failed: expected the RAG passage to rank first.")
        return 1

    print("Smoke test passed: the semantically related RAG passage ranked first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
