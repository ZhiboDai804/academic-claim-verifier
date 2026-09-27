from types import SimpleNamespace

import pytest

from src.claim_extractor import ClaimExtractionError, extract_claims


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


def test_extracts_one_structured_claim() -> None:
    client = FakeClient({"claims": ["RAG reduces hallucination."]})

    claims = extract_claims("RAG reduces hallucination.", client=client)

    assert [(claim.claim_id, claim.text) for claim in claims] == [
        ("claim_001", "RAG reduces hallucination.")
    ]
    assert client.responses.calls[0]["input"] == "RAG reduces hallucination."


def test_extracts_multiple_claims_from_compound_input() -> None:
    client = FakeClient(
        {
            "claims": [
                "RAG reduces hallucination.",
                "RAG outperforms fine-tuning on biomedical QA benchmarks.",
                "RAG outperforms fine-tuning on legal QA benchmarks.",
            ]
        }
    )

    claims = extract_claims(
        "RAG reduces hallucination and outperforms fine-tuning on biomedical "
        "and legal QA benchmarks.",
        client=client,
    )

    assert len(claims) == 3
    assert claims[2].claim_id == "claim_003"


def test_preserves_numerical_and_comparison_information() -> None:
    client = FakeClient(
        {
            "claims": [
                "Model A achieved 82% accuracy.",
                "Model A significantly outperformed Model B on Dataset X.",
            ]
        }
    )

    claims = extract_claims(
        "Model A achieved 82% accuracy and significantly outperformed Model B "
        "on Dataset X.",
        client=client,
    )

    assert "82%" in claims[0].text
    assert "significantly outperformed Model B" in claims[1].text
    instructions = str(client.responses.calls[0]["instructions"])
    assert "percentages" in instructions
    assert "comparisons" in instructions


def test_empty_input_returns_empty_list_without_model_call() -> None:
    client = FakeClient({"claims": ["should not be used"]})

    assert extract_claims("  \n", client=client) == []
    assert client.responses.calls == []


@pytest.mark.parametrize("parsed", [None, {"claims": [""]}, {"wrong": []}])
def test_malformed_structured_output_raises_clear_error(parsed: object) -> None:
    with pytest.raises(ClaimExtractionError):
        extract_claims("A factual claim.", client=FakeClient(parsed))
