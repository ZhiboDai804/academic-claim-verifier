from pathlib import Path

import fitz
import pytest

from src.pdf_parser import PDFExtractionError, extract_pdf_pages


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_extracts_text_with_one_indexed_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "study.pdf"
    _write_pdf(pdf_path, ["First   page\nwith text.", "Second page."])

    pages = extract_pdf_pages(str(pdf_path), paper_id="study-2024")

    assert [page.page_number for page in pages] == [1, 2]
    assert all(page.paper_id == "study-2024" for page in pages)
    assert all(page.filename == "study.pdf" for page in pages)
    assert pages[0].text == "First page with text."


def test_skips_empty_pages_without_renumbering(tmp_path: Path) -> None:
    pdf_path = tmp_path / "with-empty-page.pdf"
    _write_pdf(pdf_path, ["Page one.", "", "Page three."])

    pages = extract_pdf_pages(str(pdf_path))

    assert [page.page_number for page in pages] == [1, 3]
    assert pages[0].paper_id == pages[1].paper_id


def test_default_paper_id_is_stable(tmp_path: Path) -> None:
    pdf_path = tmp_path / "stable.pdf"
    _write_pdf(pdf_path, ["Stable content."])

    first = extract_pdf_pages(str(pdf_path))
    second = extract_pdf_pages(str(pdf_path))

    assert first[0].paper_id == second[0].paper_id


def test_unreadable_pdf_has_clear_error(tmp_path: Path) -> None:
    invalid_path = tmp_path / "broken.pdf"
    invalid_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(PDFExtractionError, match="Could not open PDF"):
        extract_pdf_pages(str(invalid_path))


def test_missing_pdf_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(PDFExtractionError, match="does not exist"):
        extract_pdf_pages(str(tmp_path / "missing.pdf"))
