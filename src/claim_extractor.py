"""Structured extraction of atomic claims from academic prose."""

from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from src.config import CLAIM_EXTRACTION_MODEL
from src.schemas import Claim


logger = logging.getLogger(__name__)
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "claim_extraction.txt"


class ClaimExtractionError(RuntimeError):
    """Raised when structured claim output is absent or malformed."""


class _ClaimExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[str]

    @field_validator("claims")
    @classmethod
    def validate_claims(cls, claims: list[str]) -> list[str]:
        normalized: list[str] = []
        for claim in claims:
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError("claims must contain only non-empty strings")
            normalized.append(claim.strip())
        return normalized


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def extract_claims(
    text: str,
    *,
    client: OpenAI | None = None,
    model: str = CLAIM_EXTRACTION_MODEL,
) -> list[Claim]:
    """Extract ordered atomic claims using OpenAI structured output.

    Whitespace-only input returns an empty list and makes no API call.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return []
    if not model.strip():
        raise ValueError("model must not be empty")

    api_client = client or OpenAI()
    try:
        response = api_client.responses.parse(
            model=model,
            instructions=_load_prompt(),
            input=text.strip(),
            text_format=_ClaimExtractionOutput,
        )
        if response.output_parsed is None:
            raise ClaimExtractionError("Model returned no parsed claim output")
        parsed = _ClaimExtractionOutput.model_validate(response.output_parsed)
    except ValidationError as exc:
        raise ClaimExtractionError("Model returned malformed claim output") from exc

    claims = [
        Claim(claim_id=f"claim_{index:03d}", text=claim_text)
        for index, claim_text in enumerate(parsed.claims, start=1)
    ]
    logger.info("Extracted %d atomic claims", len(claims))
    return claims
