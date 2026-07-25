"""Tests for PDFService.refresh_ocr — re-OCR a PDF and replace its stored
ChromaDB collection, without touching pdf_id/collection_name."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.api.services.pdf_service import PDFService


def _make_pdf(**overrides):
    defaults = dict(
        pdf_id="pdf_1",
        name="test.pdf",
        collection_name="col_1",
        doc_count=3,
        page_count=3,
        file_path="/tmp/does_not_matter.pdf",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_refresh_ocr_raises_for_missing_pdf():
    service = PDFService()
    db = MagicMock()
    with patch.object(PDFService, "get_pdf", return_value=None):
        with pytest.raises(LookupError):
            service.refresh_ocr("pdf_missing", db)


def test_refresh_ocr_raises_when_not_originally_ocrd():
    # doc_count != page_count is how a native-text (non-OCR) PDF looks —
    # nothing to refresh, since there's no OCR-language issue for it.
    pdf = _make_pdf(doc_count=12, page_count=3)
    service = PDFService()
    db = MagicMock()
    with patch.object(PDFService, "get_pdf", return_value=pdf):
        with pytest.raises(ValueError, match="wasn't OCR'd"):
            service.refresh_ocr("pdf_1", db)


def test_refresh_ocr_raises_when_original_file_missing():
    pdf = _make_pdf()
    service = PDFService()
    db = MagicMock()
    with patch.object(PDFService, "get_pdf", return_value=pdf), \
         patch("os.path.exists", return_value=False):
        with pytest.raises(ValueError, match="not found on disk"):
            service.refresh_ocr("pdf_1", db)


def test_refresh_ocr_raises_when_reocr_yields_no_text():
    pdf = _make_pdf()
    service = PDFService()
    db = MagicMock()
    with patch.object(PDFService, "get_pdf", return_value=pdf), \
         patch("os.path.exists", return_value=True), \
         patch("src.api.services.pdf_service.TextExtractor") as MockExtractor:
        instance = MockExtractor.return_value
        instance.extract_text_from_scanned_pdf.return_value = {"pages": []}
        instance.create_document_chunks.return_value = []
        with pytest.raises(ValueError, match="no text"):
            service.refresh_ocr("pdf_1", db)


def test_refresh_ocr_replaces_collection_and_updates_metadata():
    pdf = _make_pdf(doc_count=3, page_count=3)
    service = PDFService()
    db = MagicMock()

    fresh_docs = [
        Document(page_content=f"page {i} text", metadata={"source_page": i})
        for i in range(1, 4)
    ]

    with patch.object(PDFService, "get_pdf", return_value=pdf), \
         patch("os.path.exists", return_value=True), \
         patch("src.api.services.pdf_service.TextExtractor") as MockExtractor, \
         patch.object(service.vector_store, "delete_collection_by_name") as mock_delete, \
         patch.object(service.vector_store, "create_vector_db") as mock_create:
        instance = MockExtractor.return_value
        instance.extract_text_from_scanned_pdf.return_value = {"pages": ["stub"]}
        instance.create_document_chunks.return_value = fresh_docs

        result = service.refresh_ocr("pdf_1", db)

    # Uses the app's CJK default (no explicit ocr_language passed) since this
    # app's primary use case is CJK scans.
    instance.extract_text_from_scanned_pdf.assert_called_once()
    _, call_kwargs = instance.extract_text_from_scanned_pdf.call_args
    assert call_kwargs["ocr_language"] == "chi_sim+chi_tra+kor"

    # Old collection is torn down and rebuilt under the SAME name — pdf_id
    # and collection_name must never change on refresh.
    mock_delete.assert_called_once_with("col_1")
    mock_create.assert_called_once()
    _, create_kwargs = mock_create.call_args
    assert create_kwargs["collection_name"] == "col_1"
    assert create_kwargs["documents"] == fresh_docs

    # Metadata updated in place and persisted.
    assert pdf.doc_count == 3
    assert pdf.page_count == 3
    db.commit.assert_called_once()
    db.refresh.assert_called_once_with(pdf)
    assert result is pdf


def test_refresh_ocr_accepts_explicit_ocr_language_override():
    pdf = _make_pdf()
    service = PDFService()
    db = MagicMock()
    fresh_docs = [Document(page_content="x", metadata={"source_page": 1})]

    with patch.object(PDFService, "get_pdf", return_value=pdf), \
         patch("os.path.exists", return_value=True), \
         patch("src.api.services.pdf_service.TextExtractor") as MockExtractor, \
         patch.object(service.vector_store, "delete_collection_by_name"), \
         patch.object(service.vector_store, "create_vector_db"):
        instance = MockExtractor.return_value
        instance.extract_text_from_scanned_pdf.return_value = {"pages": ["stub"]}
        instance.create_document_chunks.return_value = fresh_docs

        service.refresh_ocr("pdf_1", db, ocr_language="eng")

    _, call_kwargs = instance.extract_text_from_scanned_pdf.call_args
    assert call_kwargs["ocr_language"] == "eng"
