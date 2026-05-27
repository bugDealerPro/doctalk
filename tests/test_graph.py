from app.graph import run_qa_workflow
from app.vector_store import RetrievedChunk


def test_graph_without_uploaded_document(settings, mocker) -> None:
    mocker.patch("app.graph.session_has_documents", return_value=False)

    result = run_qa_workflow(
        session_id="session-1",
        question="What is this document about?",
        settings=settings,
    )

    assert result["error"] == "Please upload one or more PDFs before asking questions."
    assert result["answer"] == "Please upload one or more PDFs before asking questions."
    assert result["sources"] == []


def test_graph_with_document_and_retrieved_chunks(settings, mocker) -> None:
    chunk = RetrievedChunk(
        id="chunk-1",
        document_id="doc-1",
        session_id="session-1",
        filename="sample.pdf",
        chunk_index=0,
        content="The contract starts on January 1.",
        score=0.9,
    )

    mocker.patch("app.graph.session_has_documents", return_value=True)
    mocker.patch("app.graph.retrieve_relevant_chunks", return_value=[chunk])
    mocker.patch(
        "app.graph._build_llm",
        return_value=mocker.Mock(
            invoke=lambda messages: mocker.Mock(content="It starts on January 1.")
        ),
    )

    result = run_qa_workflow(
        session_id="session-1",
        question="When does the contract start?",
        settings=settings,
    )

    assert result["error"] is None
    assert "January 1" in result["answer"]
    assert len(result["sources"]) == 1
    assert "sample.pdf (chunk 1)" in result["sources"][0]
