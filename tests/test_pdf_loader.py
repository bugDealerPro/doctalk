from pathlib import Path

import pytest
from pypdf.errors import PdfReadError

from app.pdf_loader import PdfProcessingError, chunk_text, extract_text_from_pdf, validate_pdf_size


def test_chunk_text_splits_long_text(settings) -> None:
    text = "paragraph. " * 200
    chunks = chunk_text(text, settings)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_preserves_short_text(settings) -> None:
    text = "Short document body."
    chunks = chunk_text(text, settings)
    assert chunks == [text]


def test_validate_pdf_size_rejects_large_files(settings) -> None:
    oversized = settings.max_pdf_size_mb * 1024 * 1024 + 1
    with pytest.raises(PdfProcessingError, match="maximum size"):
        validate_pdf_size(oversized, settings)


def test_extract_text_from_pdf_invalid_file(tmp_path: Path) -> None:
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not-a-pdf")

    with pytest.raises(PdfProcessingError, match="valid PDF"):
        extract_text_from_pdf(bad_pdf)


def test_extract_text_from_pdf_read_error(tmp_path: Path, mocker) -> None:
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4")

    mocker.patch("app.pdf_loader.PdfReader", side_effect=PdfReadError("broken"))

    with pytest.raises(PdfProcessingError, match="valid PDF"):
        extract_text_from_pdf(bad_pdf)


def test_extract_text_from_pdf_empty_text(tmp_path: Path, mocker) -> None:
    bad_pdf = tmp_path / "empty.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4")

    mock_page = mocker.Mock()
    mock_page.extract_text.return_value = "   "
    mock_reader = mocker.Mock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [mock_page]
    mocker.patch("app.pdf_loader.PdfReader", return_value=mock_reader)

    with pytest.raises(PdfProcessingError, match="No text could be extracted"):
        extract_text_from_pdf(bad_pdf)
