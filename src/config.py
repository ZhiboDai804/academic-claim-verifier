"""Central configuration for the Academic Claim Verifier."""

import os

from dotenv import load_dotenv


load_dotenv()


def _int_from_env(name: str, default: int, *, minimum: int) -> int:
    """Read a bounded integer setting from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


CHUNK_SIZE_WORDS = _int_from_env("CHUNK_SIZE_WORDS", 350, minimum=1)
CHUNK_OVERLAP_WORDS = _int_from_env("CHUNK_OVERLAP_WORDS", 50, minimum=0)
TOP_K = _int_from_env("TOP_K", 5, minimum=1)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CLAIM_EXTRACTION_MODEL = os.getenv("CLAIM_EXTRACTION_MODEL", "gpt-4o-mini")
VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", "gpt-4o-mini")

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "academic_papers")
