"""Tests for scanned-PDF detection and watermark-removal OCR fallback."""
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.core.document import _check_single_char_pattern, _is_ocr_artifact_text, detect_if_image_based
from src.core.image_handler import OPENCV_AVAILABLE, ImageHandler
from src.api.services.rag_service import _detect_ocr_language_override

SAMPLE_SCANNED_PDF = Path("data/pdfs/uploads")


def _find_sample_scanned_pdf():
    if not SAMPLE_SCANNED_PDF.exists():
        return None
    pdfs = list(SAMPLE_SCANNED_PDF.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def test_detect_if_image_based_empty_documents():
    assert detect_if_image_based([]) is True


def test_detect_if_image_based_short_text():
    doc = Document(page_content="R1 R2", metadata={})
    assert detect_if_image_based([doc]) is True


def test_detect_if_image_based_clean_text():
    doc = Document(
        page_content="This is a normal, cleanly extracted English paragraph "
        "with plenty of readable words and punctuation, well over fifty characters long.",
        metadata={},
    )
    assert detect_if_image_based([doc]) is False


def test_is_ocr_artifact_text_flags_garbled_output():
    garbled = "AMM ,, .. GONE ~~ BOIINDBA -- **"
    assert _is_ocr_artifact_text(garbled) is True


def test_is_ocr_artifact_text_accepts_chinese():
    chinese = "用魔环起针法起十二针，分到三根棒针上，开始圈织，全下针，右加针"
    assert _is_ocr_artifact_text(chinese) is False


def test_check_single_char_pattern():
    text = "a b c d e f normal words here"
    ratio = _check_single_char_pattern(text)
    assert 0.0 < ratio < 1.0


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
@pytest.mark.skipif(_find_sample_scanned_pdf() is None, reason="No sample scanned PDF available")
def test_remove_watermark_reduces_ocr_garbage():
    """Watermark removal should not make Chinese OCR extraction worse than raw."""
    pytesseract = pytest.importorskip("pytesseract")
    convert_from_path = pytest.importorskip("pdf2image").convert_from_path

    pdf_path = _find_sample_scanned_pdf()
    pages = convert_from_path(str(pdf_path), dpi=150, first_page=1, last_page=1)
    page = pages[0]

    handler = ImageHandler()
    raw_text = pytesseract.image_to_string(page, lang="chi_sim+eng")

    preprocessed = handler.preprocess_for_ocr(page)
    cleaned_text = pytesseract.image_to_string(preprocessed, lang="chi_sim+eng")

    # Garbled OCR output tends to be dominated by non-CJK/non-digit "noise"
    # characters; watermark removal should not increase that noise ratio.
    assert _is_ocr_artifact_text(cleaned_text) is False or _is_ocr_artifact_text(
        raw_text
    ) is True


def test_detect_ocr_language_override_source_and_target_named():
    assert _detect_ocr_language_override("중국어 도안을 한국어로 번역해줘") == "chi_sim+chi_tra+kor"


def test_detect_ocr_language_override_english_and_korean_named():
    assert _detect_ocr_language_override("영어를 한국어로 번역해줘") == "kor+eng"


def test_detect_ocr_language_override_no_translation_keyword():
    assert _detect_ocr_language_override("이 문서를 요약해줘") is None


def test_detect_ocr_language_override_only_one_language_named():
    # Needs an explicit source AND target — a single named language isn't enough.
    assert _detect_ocr_language_override("중국어를 번역해줘") is None
    assert _detect_ocr_language_override("한국어로 요약해줘") is None
