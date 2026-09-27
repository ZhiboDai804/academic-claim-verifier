"""Structured, evidence-relative verification of one atomic claim."""

from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.config import VERIFICATION_MODEL
from src.schemas import Claim, RetrievedEvidence, VerificationLabel, VerificationResult


logger = logging.getLogger(__name__)
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "claim_verification.txt"


class ClaimVerificationError(RuntimeError):
    """Raised when structured verification output is absent or invalid."""


class _VerificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: VerificationLabel
    reason: str = Field(min_length=1)
    evidence_ids: list[str]

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, reason: str) -> str:
        if not reason.strip():
            raise ValueError("reason must not be empty")
        return reason.strip()

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, evidence_ids: list[str]) -> list[str]:
        normalized: list[str] = []
        for evidence_id in evidence_ids:
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ValueError("evidence IDs must be non-empty strings")
            normalized_id = evidence_id.strip()
            if normalized_id not in normalized:
                normalized.append(normalized_id)
        return normalized


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _format_verification_input(
    claim: Claim,
    evidence: list[RetrievedEvidence],
) -> str:
    sections = ["CLAIM:", claim.text, "", "EVIDENCE:"]
    for index, item in enumerate(evidence, start=1):
        sections.extend(
            [
                "",
                f"[E{index}]",
                f"Source: {item.filename}",
                f"Page: {item.page_number}",
                "Text:",
                item.text,
            ]
        )
    return "\n".join(sections)


def verify_claim(
    claim: Claim,
    evidence: list[RetrievedEvidence],
    *,
    client: OpenAI | None = None,
    model: str = VERIFICATION_MODEL,
) -> VerificationResult:
    """Verify a claim strictly against supplied evidence.

    Unknown evidence IDs are rejected rather than silently removed. With no
    evidence, the evidence-relative verdict is deterministically UNSUPPORTED.
    """
    if not evidence:
        result = VerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            label="UNSUPPORTED",
            reason="No uploaded evidence was retrieved to support this claim.",
            evidence_ids=[],
        )
        logger.info("Claim %s verification label: %s", claim.claim_id, result.label)
        return result
    if not model.strip():
        raise ValueError("model must not be empty")

    api_client = client or OpenAI()
    try:
        response = api_client.responses.parse(
            model=model,
            instructions=_load_prompt(),
            input=_format_verification_input(claim, evidence),
            text_format=_VerificationOutput,
        )
        if response.output_parsed is None:
            raise ClaimVerificationError("Model returned no parsed verification output")
        parsed = _VerificationOutput.model_validate(response.output_parsed)
    except ValidationError as exc:
        raise ClaimVerificationError("Model returned malformed verification output") from exc

    supplied_ids = {f"E{index}" for index in range(1, len(evidence) + 1)}
    unknown_ids = set(parsed.evidence_ids) - supplied_ids
    if unknown_ids:
        unknown = ", ".join(sorted(unknown_ids))
        raise ClaimVerificationError(f"Model cited unknown evidence IDs: {unknown}")

    result = VerificationResult(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        label=parsed.label,
        reason=parsed.reason,
        evidence_ids=parsed.evidence_ids,
    )
    logger.info("Claim %s verification label: %s", claim.claim_id, result.label)
    return result
