import json
from pathlib import Path

import pytest

from src.evaluation import (
    build_error_analysis,
    classification_accuracy,
    confusion_matrix,
    per_class_counts,
    precision_recall_f1,
    retrieval_hit_at_k,
    retrieval_hit_rate_at_k,
    write_error_analysis,
)
from src.schemas import (
    EvaluationExample,
    EvaluationOutcome,
    EvidenceReference,
    RetrievedEvidence,
    VerificationResult,
)


def _evidence(filename: str, page_number: int) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=f"{filename}_{page_number}",
        paper_id=filename,
        filename=filename,
        page_number=page_number,
        text="Evidence text",
        score=0.9,
    )


def test_classification_accuracy_and_per_class_counts() -> None:
    expected = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "SUPPORTED"]
    predicted = ["SUPPORTED", "UNSUPPORTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED"]

    assert classification_accuracy(expected, predicted) == pytest.approx(0.5)
    assert per_class_counts(expected, predicted) == {
        "SUPPORTED": {"correct": 1, "total": 2},
        "PARTIALLY_SUPPORTED": {"correct": 0, "total": 1},
        "UNSUPPORTED": {"correct": 1, "total": 1},
    }


def test_confusion_matrix_has_all_expected_and_predicted_classes() -> None:
    matrix = confusion_matrix(
        ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"],
        ["SUPPORTED", "UNSUPPORTED", "SUPPORTED"],
    )

    assert matrix["SUPPORTED"]["SUPPORTED"] == 1
    assert matrix["PARTIALLY_SUPPORTED"]["UNSUPPORTED"] == 1
    assert matrix["UNSUPPORTED"]["SUPPORTED"] == 1
    assert matrix["UNSUPPORTED"]["UNSUPPORTED"] == 0


def test_precision_recall_and_f1_are_computed_per_class() -> None:
    metrics = precision_recall_f1(
        ["SUPPORTED", "SUPPORTED", "UNSUPPORTED"],
        ["SUPPORTED", "UNSUPPORTED", "UNSUPPORTED"],
    )

    assert metrics["SUPPORTED"] == {
        "precision": pytest.approx(1.0),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(2 / 3),
    }
    assert metrics["PARTIALLY_SUPPORTED"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_retrieval_hit_at_1_3_and_5_matches_filename_and_page() -> None:
    retrieved = [
        _evidence("other.pdf", 1),
        _evidence("paper.pdf", 2),
        _evidence("other.pdf", 3),
        _evidence("paper.pdf", 8),
    ]
    expected = [EvidenceReference(filename="paper.pdf", page_number=2)]

    assert retrieval_hit_at_k(retrieved, expected, 1) is False
    assert retrieval_hit_at_k(retrieved, expected, 3) is True
    assert retrieval_hit_at_k(retrieved, expected, 5) is True


def test_retrieval_hit_rate_averages_examples() -> None:
    rate = retrieval_hit_rate_at_k(
        [[_evidence("a.pdf", 1)], [_evidence("b.pdf", 2)]],
        [
            [EvidenceReference(filename="a.pdf", page_number=1)],
            [EvidenceReference(filename="missing.pdf", page_number=9)],
        ],
        1,
    )

    assert rate == pytest.approx(0.5)


def test_error_analysis_contains_only_incorrect_examples(tmp_path: Path) -> None:
    incorrect = EvaluationOutcome(
        example=EvaluationExample(
            id="example_1",
            claim="A claim",
            label="SUPPORTED",
            expected_evidence=[EvidenceReference(filename="paper.pdf", page_number=4)],
        ),
        prediction=VerificationResult(
            claim_id="example_1",
            claim_text="A claim",
            label="UNSUPPORTED",
            reason="The retrieved passage did not support the claim.",
            evidence_ids=[],
        ),
        retrieved_evidence=[_evidence("other.pdf", 2)],
    )
    correct = incorrect.model_copy(
        update={
            "prediction": incorrect.prediction.model_copy(
                update={"label": "SUPPORTED"}
            )
        }
    )

    errors = build_error_analysis([incorrect, correct])

    assert len(errors) == 1
    assert errors[0]["ground_truth_label"] == "SUPPORTED"
    assert errors[0]["predicted_label"] == "UNSUPPORTED"
    assert errors[0]["retrieved_evidence"][0]["filename"] == "other.pdf"  # type: ignore[index]

    output_path = tmp_path / "error_analysis.json"
    write_error_analysis([incorrect, correct], output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == errors


@pytest.mark.parametrize(
    ("expected", "predicted"),
    [([], []), (["SUPPORTED"], []), (["UNKNOWN"], ["SUPPORTED"])],
)
def test_accuracy_rejects_empty_mismatched_or_unknown_inputs(
    expected: list[str],
    predicted: list[str],
) -> None:
    with pytest.raises(ValueError):
        classification_accuracy(expected, predicted)
