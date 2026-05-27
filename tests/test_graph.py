from app.graph import run_qa_workflow


def test_graph_without_uploaded_document(settings, mocker) -> None:
    mocker.patch("app.graph.get_document_for_session", return_value=None)

    result = run_qa_workflow(
        session_id="session-1",
        question="What is this document about?",
        settings=settings,
    )

    assert result["error"] == "Please upload a PDF before asking questions."
    assert result["answer"] == "Please upload a PDF before asking questions."
    assert result["sources"] == []


def test_graph_with_document_and_retrieved_chunks(settings, mocker) -> None:
    document = mocker.Mock(id="doc-1", session_id="session-1", filename="sample.pdf")
    chunk = mocker.Mock(
        id="chunk-1",
        chunk_index=0,
        content="The contract starts on January 1.",
        score=0.9,
    )

    mocker.patch("app.graph.get_document_for_session", return_value=document)
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
    assert "Chunk 1" in result["sources"][0]
