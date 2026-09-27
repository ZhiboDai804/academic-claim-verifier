"""Observable evidence retrieval for one atomic claim."""

from __future__ import annotations

import logging

from src.config import TOP_K
from src.schemas import Claim, RetrievedEvidence
from src.vector_store import PaperVectorStore


logger = logging.getLogger(__name__)


def retrieve_evidence(
    claim: Claim,
    vector_store: PaperVectorStore,
    top_k: int = TOP_K,
) -> list[RetrievedEvidence]:
    """Retrieve evidence directly from the claim text without rewriting."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    evidence = vector_store.search(claim.text, top_k=top_k)
    logger.info(
        "Claim %s retrieved chunks: %s",
        claim.claim_id,
        [(item.chunk_id, item.score) for item in evidence],
    )
    return evidence
