"""Chroma-backed storage and semantic search for paper chunks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY
from src.embeddings import embed_texts
from src.schemas import Chunk, RetrievedEvidence


EmbeddingFunction = Callable[[list[str]], list[list[float]]]

logger = logging.getLogger(__name__)


class PaperVectorStore:
    """Persist chunks in one cosine-distance Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
        *,
        embedding_function: EmbeddingFunction | None = None,
        client: Any | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")

        self._collection_name = collection_name
        self._embedding_function = embedding_function or embed_texts

        if client is None:
            storage_path = Path(persist_directory)
            storage_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(storage_path))
        else:
            self._client = client

        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks while retaining citation metadata."""
        if not chunks:
            return

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunks must have unique chunk IDs")

        documents = [chunk.text for chunk in chunks]
        embeddings = self._embedding_function(documents)
        if len(embeddings) != len(chunks):
            raise ValueError("embedding function returned the wrong number of vectors")

        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        ]
        self._collection.upsert(
            ids=chunk_ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Indexed %d paper chunks", len(chunks))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        """Return the closest chunks, ordered from most to least similar."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        collection_size = self._collection.count()
        if collection_size == 0:
            return []

        query_embedding = self._embedding_function([query])
        if len(query_embedding) != 1 or not query_embedding[0]:
            raise ValueError("embedding function must return one non-empty query vector")

        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection_size),
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        evidence: list[RetrievedEvidence] = []
        for index, record_id in enumerate(ids):
            metadata = metadatas[index] or {}
            distance = distances[index] if index < len(distances) else None
            score = None if distance is None else 1.0 - float(distance)
            evidence.append(
                RetrievedEvidence(
                    chunk_id=str(metadata.get("chunk_id", record_id)),
                    paper_id=str(metadata["paper_id"]),
                    filename=str(metadata["filename"]),
                    page_number=int(metadata["page_number"]),
                    text=str(documents[index]),
                    score=score,
                )
            )

        logger.info(
            "Retrieved chunk IDs for query: %s",
            [(item.chunk_id, item.score) for item in evidence],
        )
        return evidence

    def reset(self) -> None:
        """Delete every indexed record by replacing the collection."""
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._get_or_create_collection()
        logger.info("Reset Chroma collection %s", self._collection_name)
