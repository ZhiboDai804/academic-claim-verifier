from types import SimpleNamespace

import pytest

from src.schemas import Claim, RetrievedEvidence
from src.verifier import ClaimVerificationError, verify_claim


class FakeResponses:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed: object) -> None:
        self.responses = FakeResponses(parsed)


def _claim(text: str) -> Claim:
    return Claim(claim_id="claim_001", text=text)


def _evidence(text: str, page: int = 1) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=f"paper_p{page}_c1",
        paper_id="paper",
        filename="study.pdf",
        page_number=page,
        text=text,
        score=0.9,
    )


@pytest.mark.parametrize(
    ("claim_text", "evidence_text", "label", "reason"),
    [
        (
            "Model A achieved 82% accuracy.",
            "Model A achieved an accuracy of 82%.",
            "SUPPORTED",
            "The reported accuracy matches exactly.",
        ),
        (
            "Model A significantly outperformed Model B on all datasets.",
            "Model A outperformed Model B on Dataset X.",
            "PARTIALLY_SUPPORTED",
            "Only one dataset is supported, without significance evidence.",
        ),
        (
            "Model A reduces hallucination by 50%.",
            "Model A reduced unsupported outputs.",
            "UNSUPPORTED",
            "The evidence does not support the claimed 50% reduction.",
        ),
    ],
)
def test_parses_all_valid_labels_and_strict_conceptual_cases(
    claim_text: str,
    evidence_text: str,
    label: str,
    reason: str,
) -> None:
    client = FakeClient({"label": label, "reason": reason, "evidence_ids": ["E1"]})

    result = verify_claim(_claim(claim_text), [_evidence(evidence_text)], client=client)

    assert result.label == label
    assert result.reason == reason
    assert result.evidence_ids == ["E1"]
    runtime_input = str(client.responses.calls[0]["input"])
    assert "[E1]" in runtime_input
    assert "Source: study.pdf" in runtime_input
    assert "Page: 1" in runtime_input


def test_unknown_evidence_id_is_rejected() -> None:
    client = FakeClient(
        {"label": "SUPPORTED", "reason": "Supported.", "evidence_ids": ["E2"]}
    )

    with pytest.raises(ClaimVerificationError, match="unknown evidence IDs"):
        verify_claim(_claim("Claim"), [_evidence("Evidence")], client=client)


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        {"label": "CONTRADICTED", "reason": "Wrong label.", "evidence_ids": []},
        {"label": "SUPPORTED", "reason": "", "evidence_ids": []},
        {"label": "SUPPORTED", "reason": "Missing IDs"},
    ],
)
def test_malformed_or_invalid_structured_output_is_rejected(parsed: object) -> None:
    with pytest.raises(ClaimVerificationError):
        verify_claim(_claim("Claim"), [_evidence("Evidence")], client=FakeClient(parsed))


def test_empty_evidence_is_deterministically_unsupported_without_api_call() -> None:
    client = FakeClient(
        {"label": "SUPPORTED", "reason": "Should not run", "evidence_ids": []}
    )

    result = verify_claim(_claim("Claim without retrieved evidence"), [], client=client)

    assert result.label == "UNSUPPORTED"
    assert result.evidence_ids == []
    assert "No uploaded evidence" in result.reason
    assert client.responses.calls == []


def test_prompt_requires_strict_entailment_for_adversarial_qualifiers() -> None:
    client = FakeClient(
        {
            "label": "PARTIALLY_SUPPORTED",
            "reason": "The evidence is narrower than the claim.",
            "evidence_ids": ["E1"],
        }
    )
    verify_claim(
        _claim("The method always significantly improves all datasets causally."),
        [_evidence("The method improved one dataset.")],
        client=client,
    )

    instructions = str(client.responses.calls[0]["instructions"])
    for term in ("percentages", "causal", "comparisons", "statistical", "always"):
        assert term in instructions
