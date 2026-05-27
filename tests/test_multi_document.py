"""Multi-document session retrieval tests."""

from __future__ import annotations

import uuid

import pytest

from app.db import get_connection, init_db
from app.pdf_loader import chunk_text
from app.vector_store import (
    count_documents_for_session,
    retrieve_relevant_chunks,
    store_document_with_chunks,
)
from tests.conftest import fake_embedding
from tests.fixtures.planet_texts import NYTHERA_TEXT, ZORVESSA_TEXT


def _db_available(settings) -> bool:
    try:
        init_db(settings)
        return True
    except Exception:
        return False


@pytest.fixture
def db_settings(settings):
    if not _db_available(settings):
        pytest.skip("Postgres is not available for integration tests")
    return settings


@pytest.fixture
def clean_sessions(db_settings, mocker):
    mocker.patch(
        "app.vector_store.embed_texts",
        side_effect=lambda texts, settings: [
            fake_embedding(text, settings.embedding_dimensions) for text in texts
        ],
    )
    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE session_id LIKE 'test-session-%'")
    session_a = f"test-session-{uuid.uuid4()}"
    session_b = f"test-session-{uuid.uuid4()}"
    yield session_a, session_b
    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE session_id LIKE 'test-session-%'")


def _store_planet_corpus(
    *,
    session_id: str,
    settings,
    nythera: bool = True,
    zorvessa: bool = True,
) -> None:
    if nythera:
        store_document_with_chunks(
            session_id=session_id,
            filename="nythera.pdf",
            chunks=chunk_text(NYTHERA_TEXT, settings),
            page_count=1,
            settings=settings,
        )
    if zorvessa:
        store_document_with_chunks(
            session_id=session_id,
            filename="zorvessa.pdf",
            chunks=chunk_text(ZORVESSA_TEXT, settings),
            page_count=1,
            settings=settings,
        )


def test_session_stores_multiple_documents(db_settings, clean_sessions, mock_embeddings) -> None:
    session_id, _ = clean_sessions
    _store_planet_corpus(session_id=session_id, settings=db_settings)
    assert count_documents_for_session(session_id, db_settings) == 2


def test_retrieval_finds_nythera_magnetic_storms(
    db_settings, clean_sessions, mock_embeddings
) -> None:
    session_id, _ = clean_sessions
    _store_planet_corpus(session_id=session_id, settings=db_settings)

    chunks = retrieve_relevant_chunks(
        session_id=session_id,
        question="Which planet has magnetic storms every 17 hours?",
        settings=db_settings,
    )

    assert chunks
    top = chunks[0]
    assert top.filename == "nythera.pdf"
    assert "17 hours" in top.content
    assert "Nythera" in top.content


def test_retrieval_finds_zorvessa_magnetic_storms(
    db_settings, clean_sessions, mock_embeddings
) -> None:
    session_id, _ = clean_sessions
    _store_planet_corpus(session_id=session_id, settings=db_settings)

    chunks = retrieve_relevant_chunks(
        session_id=session_id,
        question="Which planet has magnetic storms every 29 hours?",
        settings=db_settings,
    )

    assert chunks
    top = chunks[0]
    assert top.filename == "zorvessa.pdf"
    assert "29 hours" in top.content
    assert "Zorvessa" in top.content


def test_retrieval_can_surface_both_planets_for_comparison(
    db_settings, clean_sessions, mock_embeddings
) -> None:
    session_id, _ = clean_sessions
    _store_planet_corpus(session_id=session_id, settings=db_settings)

    chunks = retrieve_relevant_chunks(
        session_id=session_id,
        question="Compare Nythera and Zorvessa moons.",
        settings=db_settings,
    )

    filenames = {chunk.filename for chunk in chunks}
    contents = " ".join(chunk.content for chunk in chunks)
    assert "nythera.pdf" in filenames or "zorvessa.pdf" in filenames
    assert "Etris" in contents and "Vallun" in contents
    assert "Pala" in contents and "Rix" in contents and "Ond" in contents


def test_retrieval_isolated_by_session(db_settings, clean_sessions, mock_embeddings) -> None:
    session_a, session_b = clean_sessions
    _store_planet_corpus(session_id=session_a, settings=db_settings, zorvessa=False)
    _store_planet_corpus(
        session_id=session_b,
        settings=db_settings,
        nythera=False,
        zorvessa=True,
    )

    chunks = retrieve_relevant_chunks(
        session_id=session_a,
        question="Which planet has magnetic storms every 29 hours?",
        settings=db_settings,
    )

    assert all(chunk.session_id == session_a for chunk in chunks)
    assert all("29 hours" not in chunk.content for chunk in chunks)
