"""Chainlit application entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

import chainlit as cl
from chainlit.element import Element

from app.config import Settings, load_settings
from app.db import wait_for_db
from app.pdf_loader import PdfProcessingError
from app.services import answer_question, ingest_pdf_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def bootstrap(settings: Settings) -> None:
    wait_for_db(settings)


settings = load_settings()
bootstrap(settings)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("document_ready", False)
    await cl.Message(
        content=(
            "Welcome to **DocTalk**.\n\n"
            "Upload one or more PDFs to build a session corpus, then ask questions "
            "across all uploaded documents."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session_id = _session_id()

    pdf_files = [
        element
        for element in message.elements
        if isinstance(element, Element) and element.mime == "application/pdf"
    ]

    if pdf_files:
        for pdf_element in pdf_files:
            await _handle_upload(session_id, pdf_element)
        if message.content.strip():
            await _handle_question(session_id, message.content.strip())
        return

    if not message.content.strip():
        await cl.Message(
            content="Please upload one or more PDFs, or ask a question about them."
        ).send()
        return

    await _handle_question(session_id, message.content.strip())


async def _handle_upload(session_id: str, pdf_element: Element) -> None:
    filename = pdf_element.name or "document.pdf"
    processing_msg = cl.Message(content=f"Processing `{filename}`...")
    await processing_msg.send()

    try:
        file_bytes = _read_upload_bytes(pdf_element)
        result = ingest_pdf_file(
            session_id=session_id,
            filename=filename,
            file_bytes=file_bytes,
            settings=settings,
        )
        cl.user_session.set("document_ready", True)
        processing_msg.content = (
            f"Uploaded `{result.filename}`. "
            f"Current session has {result.session_document_count} document(s) indexed."
        )
    except PdfProcessingError as exc:
        logger.warning("PDF processing failed for session %s: %s", session_id, exc)
        processing_msg.content = str(exc)
    except Exception:
        logger.exception("Unexpected upload failure for session %s", session_id)
        processing_msg.content = (
            "Something went wrong while processing the PDF. Please try again."
        )

    await processing_msg.update()


async def _handle_question(session_id: str, question: str) -> None:
    msg = cl.Message(content="")
    await msg.send()

    try:
        result = answer_question(
            session_id=session_id,
            question=question,
            settings=settings,
        )
        response = result.answer
        if result.sources:
            source_lines = "\n".join(f"- {source}" for source in result.sources)
            response += f"\n\n**Sources**\n{source_lines}"
        msg.content = response
    except Exception:
        logger.exception("Question handling failed for session %s", session_id)
        msg.content = "Something went wrong while generating an answer. Please try again."

    await msg.update()


def _session_id() -> str:
    session = cl.context.session
    if session is None or session.id is None:
        return "anonymous"
    return str(session.id)


def _read_upload_bytes(pdf_element: Element) -> bytes:
    if pdf_element.path:
        return Path(pdf_element.path).read_bytes()

    if pdf_element.content is not None:
        if isinstance(pdf_element.content, bytes):
            return pdf_element.content
        if isinstance(pdf_element.content, str):
            return pdf_element.content.encode("utf-8")

    raise PdfProcessingError("Could not read the uploaded PDF file.")
