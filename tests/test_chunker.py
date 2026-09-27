import pytest

from src.chunker import chunk_pages
from src.schemas import Page


def _page(page_number: int, text: str, paper_id: str = "paper") -> Page:
    return Page(
        paper_id=paper_id,
        filename="paper.pdf",
        page_number=page_number,
        text=text,
    )


def test_chunk_ids_are_unique_and_metadata_is_preserved() -> None:
    pages = [_page(1, "one two three four five"), _page(2, "six seven eight")]

    chunks = chunk_pages(pages, chunk_size_words=3, overlap_words=1)

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert [chunk.chunk_id for chunk in chunks] == [
        "paper_p1_c1",
        "paper_p1_c2",
        "paper_p2_c1",
    ]
    assert all(chunk.filename == "paper.pdf" for chunk in chunks)
    assert [chunk.page_number for chunk in chunks] == [1, 1, 2]


def test_overlap_repeats_expected_words() -> None:
    chunks = chunk_pages(
        [_page(1, "one two three four five six")],
        chunk_size_words=4,
        overlap_words=2,
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
    ]


def test_chunks_never_cross_page_boundaries() -> None:
    pages = [_page(1, "alpha beta"), _page(2, "gamma delta")]

    chunks = chunk_pages(pages, chunk_size_words=3, overlap_words=1)

    assert [chunk.text for chunk in chunks] == ["alpha beta", "gamma delta"]
    assert [chunk.page_number for chunk in chunks] == [1, 2]


def test_no_empty_chunks_and_empty_input_is_supported() -> None:
    assert chunk_pages([]) == []
    chunks = chunk_pages([_page(1, "  alpha   beta  ")], 2, 0)
    assert [chunk.text for chunk in chunks] == ["alpha   beta"]
    assert all(chunk.text.strip() for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (3, -1), (3, 3), (3, 4)],
)
def test_invalid_chunk_configuration_is_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages([_page(1, "one two three")], chunk_size, overlap)


def test_duplicate_page_identity_is_rejected() -> None:
    duplicate_pages = [_page(1, "one two"), _page(1, "three four")]

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        chunk_pages(duplicate_pages, chunk_size_words=2, overlap_words=0)
