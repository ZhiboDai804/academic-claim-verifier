"""Deterministic classification, retrieval, and error-analysis utilities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from src.schemas import (
    EvaluationOutcome,
    EvidenceReference,
    RetrievedEvidence,
    VerificationLabel,
)


LABELS: tuple[VerificationLabel, ...] = (
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
)


def _validate_label_pairs(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> None:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted labels must have equal length")
    invalid = (set(expected) | set(predicted)) - set(LABELS)
    if invalid:
        raise ValueError(f"Unknown verification labels: {sorted(invalid)}")


def classification_accuracy(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> float:
    """Return correct predictions divided by the number of examples."""
    _validate_label_pairs(expected, predicted)
    if not expected:
        raise ValueError("at least one classification example is required")
    correct = sum(actual == guess for actual, guess in zip(expected, predicted))
    return correct / len(expected)


def confusion_matrix(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Return rows of expected labels and columns of predicted labels."""
    _validate_label_pairs(expected, predicted)
    matrix = {
        actual: {guess: 0 for guess in LABELS}
        for actual in LABELS
    }
    for actual, guess in zip(expected, predicted):
        matrix[actual][guess] += 1
    return matrix


def per_class_counts(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Return correct and total counts for each expected class."""
    _validate_label_pairs(expected, predicted)
    return {
        label: {
            "correct": sum(
                actual == label and guess == label
                for actual, guess in zip(expected, predicted)
            ),
            "total": sum(actual == label for actual in expected),
        }
        for label in LABELS
    }


def precision_recall_f1(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute one-vs-rest precision, recall, and F1 for each class."""
    _validate_label_pairs(expected, predicted)
    metrics: dict[str, dict[str, float]] = {}
    for label in LABELS:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(expected, predicted)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(expected, predicted)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(expected, predicted)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def retrieval_hit_at_k(
    retrieved: Sequence[RetrievedEvidence],
    expected: Sequence[EvidenceReference],
    k: int,
) -> bool:
    """Return whether any expected filename/page occurs in the top K results."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    expected_pages = {(item.filename, item.page_number) for item in expected}
    return any(
        (item.filename, item.page_number) in expected_pages
        for item in retrieved[:k]
    )


def retrieval_hit_rate_at_k(
    retrieved_runs: Sequence[Sequence[RetrievedEvidence]],
    expected_runs: Sequence[Sequence[EvidenceReference]],
    k: int,
) -> float:
    """Average per-example Hit@K across retrieval runs."""
    if len(retrieved_runs) != len(expected_runs):
        raise ValueError("retrieved and expected runs must have equal length")
    if not retrieved_runs:
        raise ValueError("at least one retrieval example is required")
    hits = [
        retrieval_hit_at_k(retrieved, expected, k)
        for retrieved, expected in zip(retrieved_runs, expected_runs)
    ]
    return sum(hits) / len(hits)


def build_error_analysis(outcomes: Sequence[EvaluationOutcome]) -> list[dict[str, object]]:
    """Serialize only incorrect outcomes with retrieval and verifier context."""
    errors: list[dict[str, object]] = []
    for outcome in outcomes:
        if outcome.example.label == outcome.prediction.label:
            continue
        errors.append(
            {
                "id": outcome.example.id,
                "claim": outcome.example.claim,
                "ground_truth_label": outcome.example.label,
                "predicted_label": outcome.prediction.label,
                "retrieved_evidence": [
                    item.model_dump(mode="json")
                    for item in outcome.retrieved_evidence
                ],
                "expected_evidence": [
                    item.model_dump(mode="json")
                    for item in outcome.example.expected_evidence
                ],
                "verifier_explanation": outcome.prediction.reason,
            }
        )
    return errors


def write_error_analysis(
    outcomes: Sequence[EvaluationOutcome],
    output_path: str | Path,
) -> None:
    """Write inspectable incorrect examples as UTF-8 JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_error_analysis(outcomes), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
