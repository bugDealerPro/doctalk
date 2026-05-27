from dataclasses import replace

from app.vector_store import embed_texts


def test_embed_texts_returns_empty_for_no_input(settings, mocker) -> None:
    mocker.patch("app.vector_store._build_embeddings")
    assert embed_texts([], settings) == []


def test_embed_texts_batches_large_inputs(settings, mocker) -> None:
    mock_client = mocker.Mock()
    mock_client.embed_documents.side_effect = lambda batch: [
        [float(index)] * settings.embedding_dimensions
        for index, _ in enumerate(batch)
    ]
    mocker.patch("app.vector_store._build_embeddings", return_value=mock_client)

    settings = replace(settings, embedding_batch_size=2)
    texts = ["chunk-a", "chunk-b", "chunk-c", "chunk-d", "chunk-e"]
    vectors = embed_texts(texts, settings)

    assert len(vectors) == 5
    assert mock_client.embed_documents.call_count == 3
    assert len(mock_client.embed_documents.call_args_list[0].args[0]) == 2
    assert len(mock_client.embed_documents.call_args_list[2].args[0]) == 1
