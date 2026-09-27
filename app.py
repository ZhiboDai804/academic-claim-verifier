"""Streamlit application for the Academic Claim Verifier MVP."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAIError

from src.pdf_parser import PDFExtractionError
from src.pipeline import ClaimVerificationPipeline
from src.schemas import ClaimVerificationReport, IngestionStats


logger = logging.getLogger(__name__)


def _get_pipeline() -> ClaimVerificationPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = ClaimVerificationPipeline()
    return st.session_state.pipeline


def _upload_signature(uploaded_files: list[object]) -> tuple[tuple[str, int], ...]:
    return tuple((upload.name, upload.size) for upload in uploaded_files)  # type: ignore[attr-defined]


def _write_uploaded_pdfs(uploaded_files: list[object], root: Path) -> list[str]:
    paths: list[str] = []
    for index, upload in enumerate(uploaded_files):
        safe_filename = Path(upload.name).name  # type: ignore[attr-defined]
        if not safe_filename.lower().endswith(".pdf"):
            raise ValueError(f"Uploaded file is not a PDF: {safe_filename}")
        paper_directory = root / f"paper_{index:03d}"
        paper_directory.mkdir(parents=True, exist_ok=True)
        pdf_path = paper_directory / safe_filename
        pdf_path.write_bytes(upload.getvalue())  # type: ignore[attr-defined]
        paths.append(str(pdf_path))
    return paths


def _show_ingestion_stats(stats: IngestionStats) -> None:
    paper_column, page_column, chunk_column = st.columns(3)
    paper_column.metric("Papers", stats.papers)
    page_column.metric("Pages extracted", stats.pages)
    chunk_column.metric("Chunks indexed", stats.chunks)


def _show_verdict(report: ClaimVerificationReport) -> None:
    label = report.verification.label
    message = f"Verdict: {label}"
    if label == "SUPPORTED":
        st.success(message)
    elif label == "PARTIALLY_SUPPORTED":
        st.warning(message)
    else:
        st.error(message)


def _show_results(reports: list[ClaimVerificationReport]) -> None:
    labels = [report.verification.label for report in reports]
    supported = labels.count("SUPPORTED")
    partially_supported = labels.count("PARTIALLY_SUPPORTED")
    unsupported = labels.count("UNSUPPORTED")
    groundedness = (supported + 0.5 * partially_supported) / len(reports)

    st.header("3. Groundedness summary")
    total_column, supported_column, partial_column, unsupported_column = st.columns(4)
    total_column.metric("Total claims", len(reports))
    supported_column.metric("Supported", supported)
    partial_column.metric("Partially supported", partially_supported)
    unsupported_column.metric("Unsupported", unsupported)
    st.metric("Heuristic groundedness", f"{groundedness:.0%}")
    st.caption(
        "Heuristic only: SUPPORTED = 1, PARTIALLY_SUPPORTED = 0.5, and "
        "UNSUPPORTED = 0. This is not a universally validated hallucination metric."
    )

    st.header("4. Claim results")
    for claim_index, report in enumerate(reports, start=1):
        with st.container(border=True):
            st.subheader(f"Claim {claim_index}")
            st.write(report.claim.text)
            _show_verdict(report)
            st.markdown("**Reason**")
            st.write(report.verification.reason)

            st.markdown("**Retrieved evidence**")
            if not report.retrieved_evidence:
                st.caption("No evidence passages were retrieved for this claim.")
                continue

            cited_ids = set(report.verification.evidence_ids)
            for evidence_index, evidence in enumerate(
                report.retrieved_evidence,
                start=1,
            ):
                evidence_id = f"E{evidence_index}"
                cited_text = " — cited by verifier" if evidence_id in cited_ids else ""
                st.markdown(
                    f"**[{evidence_id}] {evidence.filename} — Page "
                    f"{evidence.page_number}{cited_text}**"
                )
                if evidence.score is not None:
                    st.caption(f"Retrieval similarity score: {evidence.score:.4f}")
                st.write(evidence.text)


st.set_page_config(page_title="Academic Claim Verifier", page_icon="🔎")

st.title("Academic Claim Verifier")
st.caption("Check whether academic claims are supported by uploaded papers.")
st.info(
    "This system evaluates whether claims are supported by the uploaded evidence. "
    "It does not determine universal factual truth. UNSUPPORTED does not mean false."
)

if not os.getenv("OPENAI_API_KEY"):
    st.warning(
        "OPENAI_API_KEY is not configured. Add it to a local .env file before "
        "building an index or verifying claims."
    )

st.header("1. Upload papers")
uploaded_files = st.file_uploader(
    "Choose text-based PDF research papers",
    type=["pdf"],
    accept_multiple_files=True,
    help="Scanned PDFs requiring OCR are not supported in this MVP.",
)
uploaded_files = list(uploaded_files or [])
current_signature = _upload_signature(uploaded_files)

if uploaded_files:
    st.caption(f"{len(uploaded_files)} PDF file(s) selected.")

if st.button("Build / Rebuild Index", disabled=not uploaded_files):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Configure OPENAI_API_KEY before building the semantic index.")
    else:
        try:
            with st.spinner("Extracting pages, creating chunks, and building the index..."):
                with tempfile.TemporaryDirectory(prefix="claim-verifier-") as temp_dir:
                    paths = _write_uploaded_pdfs(uploaded_files, Path(temp_dir))
                    pipeline = _get_pipeline()
                    pipeline.ingest_pdfs(paths)
            st.session_state.indexed_signature = current_signature
            st.session_state.index_ready = True
            st.session_state.reports = []
            st.success("Index built successfully.")
        except PDFExtractionError as exc:
            st.session_state.index_ready = False
            st.error(f"A PDF could not be read: {exc}")
        except OpenAIError:
            st.session_state.index_ready = False
            logger.exception("OpenAI request failed while building the index")
            st.error("The embedding request failed. Check the API key and connection.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.session_state.index_ready = False
            logger.exception("Index build failed")
            st.error(f"The index could not be built: {exc}")

if st.session_state.get("index_ready"):
    pipeline = _get_pipeline()
    _show_ingestion_stats(pipeline.last_ingestion_stats)
    if st.session_state.get("indexed_signature") != current_signature:
        st.warning("The selected PDFs changed. Rebuild the index before verification.")

st.header("2. Text to verify")
text_to_verify = st.text_area(
    "Paste an AI-generated literature review, paper summary, or research paragraph here.",
    height=220,
)

if st.button("Verify Claims"):
    index_matches_selection = (
        st.session_state.get("index_ready", False)
        and st.session_state.get("indexed_signature") == current_signature
    )
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Configure OPENAI_API_KEY before verifying claims.")
    elif not uploaded_files or not index_matches_selection:
        st.error("Upload PDFs and build the index before verifying text.")
    elif not text_to_verify.strip():
        st.error("Enter academic text containing at least one verifiable claim.")
    else:
        try:
            with st.spinner("Extracting and verifying atomic claims..."):
                pipeline = _get_pipeline()
                st.session_state.reports = pipeline.verify_text(text_to_verify)
            if not st.session_state.reports:
                st.warning("No independently verifiable factual claims were extracted.")
        except OpenAIError:
            logger.exception("OpenAI request failed during verification")
            st.error("The verification request failed. Check the API key and connection.")
        except (ValueError, RuntimeError) as exc:
            logger.exception("Claim verification failed")
            st.error(f"Claims could not be verified: {exc}")

reports = st.session_state.get("reports", [])
if reports:
    _show_results(reports)
