"""Vector store operations for documents and chunks."""

from __future__ import annotations

import logging
import re
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


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", query)
    return sorted({term for term in terms if len(term) > 2}, key=len, reverse=True)


def _term_hit_count(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered)


def select_citation_chunks(
    chunks: list[RetrievedChunk], question: str
) -> list[RetrievedChunk]:
    """Keep only chunks that plausibly support the answer for source display."""
    if not chunks:
        return []

    terms = _query_terms(question)
    if not terms:
        return [max(chunks, key=lambda chunk: chunk.score)]

    matching = [
        chunk
        for chunk in chunks
        if _term_hit_count(chunk.content, terms) > 0
    ]
    if matching:
        return sorted(
            matching,
            key=lambda chunk: (_term_hit_count(chunk.content, terms), chunk.score),
            reverse=True,
        )

    return [max(chunks, key=lambda chunk: chunk.score)]


def _best_sentences(text: str, terms: list[str], max_length: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return text[:max_length]

    ranked = sorted(
        sentences,
        key=lambda sentence: _term_hit_count(sentence, terms),
        reverse=True,
    )
    selected: list[str] = []
    length = 0
    for sentence in ranked:
        if terms and _term_hit_count(sentence, terms) == 0:
            continue
        if length + len(sentence) > max_length and selected:
            break
        selected.append(sentence)
        length += len(sentence) + 1
        if length >= max_length:
            break

    if selected:
        return " ".join(selected)[:max_length].rstrip()

    return ranked[0][:max_length].rstrip()


def preview_chunk(content: str, query: str = "", max_length: int = 160) -> str:
    """Show a short excerpt focused on query-related sentences."""
    text = " ".join(content.split())
    terms = _query_terms(query)

    if terms:
        snippet = _best_sentences(text, terms, max_length)
        if _term_hit_count(snippet, terms) > 0:
            return snippet if len(snippet) <= max_length else snippet[: max_length - 3] + "..."

    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."
