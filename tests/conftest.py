import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url="postgresql://doctalk:doctalk@localhost:5432/doctalk",
        openai_api_key="test-key",
        openai_base_url=None,
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        max_pdf_size_mb=20,
        chunk_size=100,
        chunk_overlap=20,
        retrieval_top_k=4,
    )
