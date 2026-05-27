"""PDF text extraction and chunking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import Settings

logger = logging.getLogger(__name__)


class PdfProcessingError(Exception):
    """Raised when a PDF cannot be processed."""


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    page_count: int


def extract_text_from_pdf(path: Path) -> ExtractedDocument:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        logger.warning("Failed to read PDF %s: %s", path, exc)
        raise PdfProcessingError("The uploaded file is not a valid PDF.") from exc
    except OSError as exc:
        logger.warning("Failed to open PDF %s: %s", path, exc)
        raise PdfProcessingError("Could not read the uploaded PDF file.") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            logger.warning("Encrypted PDF could not be decrypted: %s", exc)
            raise PdfProcessingError(
                "Password-protected PDFs are not supported."
            ) from exc

    pages: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract text from a page: %s", exc)
            page_text = ""
        pages.append(page_text.strip())

    text = "\n\n".join(part for part in pages if part).strip()
    if not text:
        raise PdfProcessingError(
            "No text could be extracted from this PDF. "
            "Scanned image-only PDFs are not supported."
        )

    return ExtractedDocument(text=text, page_count=len(reader.pages))


def validate_pdf_size(file_size_bytes: int, settings: Settings) -> None:
    max_bytes = settings.max_pdf_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise PdfProcessingError(
            f"PDF exceeds the maximum size of {settings.max_pdf_size_mb} MB."
        )
