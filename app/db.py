"""Database connection and schema initialization."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import Settings

logger = logging.getLogger(__name__)


@contextmanager
def get_connection(settings: Settings) -> Generator[psycopg.Connection, None, None]:
    conn = psycopg.connect(settings.database_url, row_factory=dict_row)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(settings: Settings) -> None:
    """Create extensions and tables if they do not exist."""
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(%s),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (document_id, chunk_index)
                )
                """,
                (settings.embedding_dimensions,),
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks (document_id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
    logger.info("Database initialized")


def wait_for_db(settings: Settings, retries: int = 30, delay_seconds: float = 1.0) -> None:
    """Retry database initialization until Postgres is ready."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            init_db(settings)
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Database not ready (attempt %d/%d): %s",
                attempt,
                retries,
                exc,
            )
            time.sleep(delay_seconds)
    raise RuntimeError("Database initialization failed") from last_error
