"""Vector store operations for documents and chunks."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Sequence

from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.db import get_connection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    session_id: str
    filename: str


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    chunk_index: int
    content: str
    score: float


def _build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    kwargs: dict[str, str] = {
        "model": settings.embedding_model,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)


def embed_texts(texts: Sequence[str], settings: Settings) -> list[list[float]]:
    embeddings = _build_embeddings(settings)
    return embeddings.embed_documents(list(texts))


def delete_document_for_session(session_id: str, settings: Settings) -> None:
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE session_id = %s",
                (session_id,),
            )
    logger.info("Removed previous document for session %s", session_id)


def store_document_with_chunks(
    *,
    session_id: str,
    filename: str,
    chunks: Sequence[str],
    settings: Settings,
) -> DocumentRecord:
    if not chunks:
        raise ValueError("Cannot store a document with no text chunks.")

    delete_document_for_session(session_id, settings)
    vectors = embed_texts(chunks, settings)
    document_id = str(uuid.uuid4())

    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, session_id, filename)
                VALUES (%s, %s, %s)
                RETURNING id, session_id, filename
                """,
                (document_id, session_id, filename),
            )
            row = cur.fetchone()
            assert row is not None

            for index, (content, embedding) in enumerate(zip(chunks, vectors)):
                cur.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (document_id, index, content, embedding),
                )

    logger.info(
        "Stored document %s with %d chunks for session %s",
        document_id,
        len(chunks),
        session_id,
    )
    return DocumentRecord(
        id=str(row["id"]),
        session_id=row["session_id"],
        filename=row["filename"],
    )


def get_document_for_session(
    session_id: str, settings: Settings
) -> DocumentRecord | None:
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, filename
                FROM documents
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return DocumentRecord(
        id=str(row["id"]),
        session_id=row["session_id"],
        filename=row["filename"],
    )


def retrieve_relevant_chunks(
    *,
    document_id: str,
    question: str,
    settings: Settings,
) -> list[RetrievedChunk]:
    query_embedding = embed_texts([question], settings)[0]

    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    chunk_index,
                    content,
                    1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE document_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, document_id, query_embedding, settings.retrieval_top_k),
            )
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            id=str(row["id"]),
            chunk_index=row["chunk_index"],
            content=row["content"],
            score=float(row["score"]),
        )
        for row in rows
    ]


def preview_chunk(content: str, max_length: int = 240) -> str:
    text = " ".join(content.split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."
