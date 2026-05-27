import pytest

from app.pdf_loader import PdfProcessingError
from app.services import ingest_pdf_file


def test_ingest_pdf_file_deletes_temp_file(settings, mocker, tmp_path) -> None:
    mocker.patch(
        "app.services.extract_text_from_pdf",
        return_value=mocker.Mock(text="Sample text for chunking.", page_count=1),
    )
    mocker.patch("app.services.chunk_text", return_value=["Sample text for chunking."])
    mocker.patch(
        "app.services.store_document_with_chunks",
        return_value=mocker.Mock(id="doc-1", session_id="s1", filename="file.pdf"),
    )

    created_paths: list[str] = []
    original_named = __import__("tempfile").NamedTemporaryFile

    def track_temp(*args, **kwargs):
        handle = original_named(*args, **kwargs)
        created_paths.append(handle.name)
        return handle

    mocker.patch("app.services.tempfile.NamedTemporaryFile", side_effect=track_temp)

    ingest_pdf_file(
        session_id="s1",
        filename="file.pdf",
        file_bytes=b"%PDF-1.4",
        settings=settings,
    )

    assert created_paths
    assert not __import__("pathlib").Path(created_paths[0]).exists()


def test_ingest_pdf_file_rejects_empty_chunks(settings, mocker) -> None:
    mocker.patch(
        "app.services.extract_text_from_pdf",
        return_value=mocker.Mock(text="ignored", page_count=1),
    )
    mocker.patch("app.services.chunk_text", return_value=[])

    with pytest.raises(PdfProcessingError, match="usable text chunks"):
        ingest_pdf_file(
            session_id="s1",
            filename="file.pdf",
            file_bytes=b"%PDF-1.4",
            settings=settings,
        )
