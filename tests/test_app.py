from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.schemas import (
    Claim,
    ClaimVerificationReport,
    RetrievedEvidence,
    VerificationResult,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_app_renders_core_workflow_without_exceptions() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    assert list(app.exception) == []
    assert [button.label for button in app.button] == [
        "Build / Rebuild Index",
        "Verify Claims",
    ]
    assert app.button[0].disabled is True
    assert [text_area.label for text_area in app.text_area] == [
        "Paste an AI-generated literature review, paper summary, or research paragraph here."
    ]


def test_verify_without_prerequisites_shows_a_useful_error() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    app.button[1].click().run(timeout=20)

    assert list(app.exception) == []
    assert len(app.error) == 1


def test_result_ui_shows_summary_verdict_reason_and_source_metadata() -> None:
    report = ClaimVerificationReport(
        claim=Claim(claim_id="claim_001", text="A supported claim."),
        verification=VerificationResult(
            claim_id="claim_001",
            claim_text="A supported claim.",
            label="SUPPORTED",
            reason="The supplied passage supports the claim.",
            evidence_ids=["E1"],
        ),
        retrieved_evidence=[
            RetrievedEvidence(
                chunk_id="paper_p6_c1",
                paper_id="paper",
                filename="study.pdf",
                page_number=6,
                text="A source-backed evidence passage.",
                score=0.87,
            )
        ],
    )
    app = AppTest.from_file(APP_PATH)
    app.session_state["reports"] = [report]

    app.run(timeout=20)

    assert list(app.exception) == []
    assert [(metric.label, metric.value) for metric in app.metric] == [
        ("Total claims", "1"),
        ("Supported", "1"),
        ("Partially supported", "0"),
        ("Unsupported", "0"),
        ("Heuristic groundedness", "100%"),
    ]
    assert [message.value for message in app.success] == ["Verdict: SUPPORTED"]
    rendered_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "The supplied passage supports the claim." in rendered_text
    assert "study.pdf" in rendered_text
    assert "Page 6" in rendered_text
    assert "A source-backed evidence passage." in rendered_text
