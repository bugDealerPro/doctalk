import hashlib
import re

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
        chunk_size=400,
        chunk_overlap=50,
        retrieval_top_k=4,
        max_source_previews=3,
        embedding_batch_size=64,
    )


def fake_embedding(text: str, dimensions: int = 1536) -> list[float]:
    vector = [0.0] * dimensions
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.md5(token.encode(), usedforsecurity=False).hexdigest()
        index = int(digest, 16) % dimensions
        vector[index] += 1.0
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


@pytest.fixture
def mock_embeddings(mocker):
    def _embed(texts, settings):
        return [fake_embedding(text, settings.embedding_dimensions) for text in texts]

    return mocker.patch("app.vector_store.embed_texts", side_effect=_embed)
