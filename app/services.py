"""Business logic for document ingestion and chat."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.graph import run_qa_workflow
from app.pdf_loader import (
    PdfProcessingError,
    chunk_text,
    extract_text_from_pdf,
    validate_pdf_size,
)
from app.vector_store import DocumentRecord, store_document_with_chunks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    document: DocumentRecord
    chunk_count: int
    page_count: int


def ingest_pdf_file(
    *,
    session_id: str,
    filename: str,
    file_bytes: bytes,
    settings: Settings,
) -> IngestResult:
    validate_pdf_size(len(file_bytes), settings)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            temp_path = Path(tmp.name)

        extracted = extract_text_from_pdf(temp_path)
        chunks = chunk_text(extracted.text, settings)
        if not chunks:
            raise PdfProcessingError("The PDF did not produce any usable text chunks.")

        document = store_document_with_chunks(
            session_id=session_id,
            filename=filename,
            chunks=chunks,
            settings=settings,
        )
        return IngestResult(
            document=document,
            chunk_count=len(chunks),
            page_count=extracted.page_count,
        )
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
            logger.debug("Deleted temporary PDF %s", temp_path)


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[str]
    filename: str | None
    error: str | None


def answer_question(
    *,
    session_id: str,
    question: str,
    settings: Settings,
) -> ChatResult:
    result = run_qa_workflow(
        session_id=session_id,
        question=question,
        settings=settings,
    )
    return ChatResult(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        filename=result.get("filename"),
        error=result.get("error"),
    )
