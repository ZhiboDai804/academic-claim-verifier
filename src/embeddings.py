"""Small, centralized wrapper around the OpenAI embeddings API."""

from __future__ import annotations

from openai import OpenAI

from src.config import EMBEDDING_MODEL


def _validate_texts(texts: list[str]) -> None:
    if not texts:
        raise ValueError("texts must contain at least one item")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must contain only non-empty strings")


def embed_texts(
    texts: list[str],
    *,
    client: OpenAI | None = None,
    model: str = EMBEDDING_MODEL,
) -> list[list[float]]:
    """Embed a batch of non-empty texts in one API request.

    ``client`` and ``model`` are keyword-only seams for testing and controlled
    experiments; normal application code only needs to pass ``texts``.
    """
    _validate_texts(texts)
    if not model.strip():
        raise ValueError("model must not be empty")

    api_client = client or OpenAI()
    response = api_client.embeddings.create(model=model, input=texts)
    ordered_data = sorted(response.data, key=lambda item: item.index)
    embeddings = [list(item.embedding) for item in ordered_data]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Embedding API returned a different number of vectors than inputs"
        )
    if any(not embedding for embedding in embeddings):
        raise RuntimeError("Embedding API returned an empty vector")

    return embeddings


def embed_query(
    text: str,
    *,
    client: OpenAI | None = None,
    model: str = EMBEDDING_MODEL,
) -> list[float]:
    """Embed one query using the same model as document chunks."""
    return embed_texts([text], client=client, model=model)[0]
