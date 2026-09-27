"""Shared, validated data models for the verification pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


VerificationLabel = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]


class StrictModel(BaseModel):
    """Base model that rejects unexpected fields and assignment errors."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Page(StrictModel):
    paper_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)


class Chunk(StrictModel):
    chunk_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class Claim(StrictModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RetrievedEvidence(StrictModel):
    chunk_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    score: float | None = None


class VerificationResult(StrictModel):
    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    label: VerificationLabel
    reason: str = Field(min_length=1)
    evidence_ids: list[str]


class ClaimVerificationReport(StrictModel):
    """One verified claim together with the passages retrieved for it."""

    claim: Claim
    verification: VerificationResult
    retrieved_evidence: list[RetrievedEvidence]


class IngestionStats(StrictModel):
    papers: int = Field(ge=0)
    pages: int = Field(ge=0)
    chunks: int = Field(ge=0)


class EvidenceReference(StrictModel):
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)


class EvaluationExample(StrictModel):
    id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    label: VerificationLabel
    expected_evidence: list[EvidenceReference]
    review_status: Literal[
        "MANUALLY_REVIEWED",
        "DEMO_PLACEHOLDER_REQUIRES_MANUAL_REVIEW",
    ] = "DEMO_PLACEHOLDER_REQUIRES_MANUAL_REVIEW"
    notes: str | None = None


class EvaluationOutcome(StrictModel):
    example: EvaluationExample
    prediction: VerificationResult
    retrieved_evidence: list[RetrievedEvidence]
