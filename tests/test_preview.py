from app.vector_store import (
    RetrievedChunk,
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
            chunk_index=0,
            content=(
                "Expedition NX-04 reported magnetic storms. "
                "Outpost Khepri was abandoned after Incident NT-229 caused navigation failures."
            ),
            score=0.8,
        ),
        RetrievedChunk(
            id="2",
            chunk_index=1,
            content="Project Dawnbridge proposed a floating research station for 120 scientists.",
            score=0.75,
        ),
    ]

    cited = select_citation_chunks(chunks, "what caused Incident NT-229")

    assert len(cited) == 1
    assert cited[0].chunk_index == 0
