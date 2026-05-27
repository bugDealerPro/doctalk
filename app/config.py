"""Application configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    database_url: str
    openai_api_key: str
    openai_base_url: str | None
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    max_pdf_size_mb: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int


def load_settings() -> Settings:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url == "":
        base_url = None

    return Settings(
        database_url=os.getenv(
            "DOCTALK_DATABASE_URL",
            "postgresql://doctalk:doctalk@localhost:5432/doctalk",
        ),
        openai_api_key=api_key,
        openai_base_url=base_url,
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dimensions=_get_int("EMBEDDING_DIMENSIONS", 1536),
        max_pdf_size_mb=_get_int("MAX_PDF_SIZE_MB", 20),
        chunk_size=_get_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 200),
        retrieval_top_k=_get_int("RETRIEVAL_TOP_K", 4),
    )
