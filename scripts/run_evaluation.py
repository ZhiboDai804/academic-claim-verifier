"""Run reviewed evaluation examples against papers in data/papers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pydantic import TypeAdapter  # noqa: E402

from src.config import TOP_K  # noqa: E402
from src.evaluation import (  # noqa: E402
    classification_accuracy,
    confusion_matrix,
    per_class_counts,
    precision_recall_f1,
    retrieval_hit_rate_at_k,
    write_error_analysis,
)
from src.pipeline import ClaimVerificationPipeline  # noqa: E402
from src.retriever import retrieve_evidence  # noqa: E402
from src.schemas import Claim, EvaluationExample, EvaluationOutcome  # noqa: E402
from src.verifier import verify_claim  # noqa: E402


DATASET_PATH = REPOSITORY_ROOT / "data" / "eval" / "sample_eval.json"
PAPERS_DIRECTORY = REPOSITORY_ROOT / "data" / "papers"
ERROR_ANALYSIS_PATH = REPOSITORY_ROOT / "data" / "eval" / "error_analysis.json"


def _load_examples() -> list[EvaluationExample]:
    raw_examples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return TypeAdapter(list[EvaluationExample]).validate_python(raw_examples)


def main() -> int:
    examples = _load_examples()
    paper_paths = sorted(PAPERS_DIRECTORY.glob("*.pdf"))
    blockers: list[str] = []

    if not os.getenv("OPENAI_API_KEY"):
        blockers.append("OPENAI_API_KEY is unavailable")
    if not paper_paths:
        blockers.append("data/papers contains no PDF files")
    if any(example.review_status != "MANUALLY_REVIEWED" for example in examples):
        blockers.append("the evaluation labels are demo placeholders requiring manual review")
    if any(not example.expected_evidence for example in examples):
        blockers.append("one or more examples have no expected source page")

    if blockers:
        print("Evaluation not run:")
        for blocker in blockers:
            print(f"- {blocker}")
        print("No evaluation metrics were produced.")
        return 2

    pipeline = ClaimVerificationPipeline()
    pipeline.ingest_pdfs([str(path) for path in paper_paths])

    outcomes: list[EvaluationOutcome] = []
    for example in examples:
        claim = Claim(claim_id=example.id, text=example.claim)
        evidence = retrieve_evidence(claim, pipeline.vector_store, top_k=TOP_K)
        prediction = verify_claim(claim, evidence)
        outcomes.append(
            EvaluationOutcome(
                example=example,
                prediction=prediction,
                retrieved_evidence=evidence,
            )
        )

    expected_labels = [outcome.example.label for outcome in outcomes]
    predicted_labels = [outcome.prediction.label for outcome in outcomes]
    retrieved_runs = [outcome.retrieved_evidence for outcome in outcomes]
    expected_runs = [outcome.example.expected_evidence for outcome in outcomes]

    print(f"Evaluation examples: {len(outcomes)}")
    print(f"Classification accuracy: {classification_accuracy(expected_labels, predicted_labels):.3f}")
    for k in (1, 3, 5):
        print(f"Hit@{k}: {retrieval_hit_rate_at_k(retrieved_runs, expected_runs, k):.3f}")

    print("\nPer-class:")
    counts = per_class_counts(expected_labels, predicted_labels)
    detailed = precision_recall_f1(expected_labels, predicted_labels)
    for label in counts:
        print(
            f"{label}: {counts[label]['correct']} / {counts[label]['total']} correct; "
            f"precision={detailed[label]['precision']:.3f}, "
            f"recall={detailed[label]['recall']:.3f}, "
            f"f1={detailed[label]['f1']:.3f}"
        )

    print("\nConfusion matrix (expected rows, predicted columns):")
    print(json.dumps(confusion_matrix(expected_labels, predicted_labels), indent=2))
    write_error_analysis(outcomes, ERROR_ANALYSIS_PATH)
    print(f"Error analysis: {ERROR_ANALYSIS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
