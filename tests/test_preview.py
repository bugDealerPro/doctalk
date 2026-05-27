from app.vector_store import (
    RetrievedChunk,
    format_source_reference,
    preview_chunk,
    select_citation_chunks,
)


def test_preview_chunk_shows_query_sentence() -> None:
    content = (
        "Overview of Nythera discovery and atmosphere. "
        "Outpost Khepri was abandoned after Incident NT-229 caused navigation failures. "
        "More notes about oceans."
    )
    preview = preview_chunk(content, query="what caused Incident NT-229")
    assert "NT-229" in preview
    assert "Overview" not in preview


def test_select_citation_chunks_filters_irrelevant_chunk() -> None:
    chunks = [
        RetrievedChunk(
            id="1",
            document_id="doc-1",
            session_id="session-1",
            filename="nythera.pdf",
            chunk_index=0,
            content=(
                "Expedition NX-04 reported magnetic storms. "
                "Outpost Khepri was abandoned after Incident NT-229 caused navigation failures."
            ),
            score=0.8,
        ),
        RetrievedChunk(
            id="2",
            document_id="doc-2",
            session_id="session-1",
            filename="zorvessa.pdf",
            chunk_index=0,
            content="Project Dawnbridge proposed a floating research station for 120 scientists.",
            score=0.75,
        ),
    ]

    cited = select_citation_chunks(chunks, "what caused Incident NT-229", max_sources=3)

    assert len(cited) == 1
    assert cited[0].filename == "nythera.pdf"


def test_format_source_reference_includes_filename() -> None:
    chunk = RetrievedChunk(
        id="1",
        document_id="doc-1",
        session_id="session-1",
        filename="nythera.pdf",
        chunk_index=2,
        content="Nythera has 2 moons: Etris and Vallun.",
        score=0.9,
    )
    reference = format_source_reference(chunk, "Nythera moons")
    assert reference.startswith("nythera.pdf (chunk 3):")
