"""Tests for scanned-PDF detection and watermark-removal OCR fallback."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.core.document import _check_single_char_pattern, _is_ocr_artifact_text, detect_if_image_based
from src.core.image_handler import OPENCV_AVAILABLE, ImageHandler
from src.core.text_extractor import _strip_recurring_watermark_lines
from src.api.services.rag_service import (
    NUM_CTX_FLOOR,
    PAGE_TRANSLATION_OUTPUT_BUDGET,
    MAX_CONTINUATION_ATTEMPTS,
    MAX_PAGE_RETRY_ATTEMPTS,
    TRANSLATION_LINE_INSTRUCTIONS,
    VERBATIM_INSTRUCTIONS,
    RAGService,
    _detect_ocr_language_override,
    _estimate_num_ctx,
    _line_instructions_for,
    _looks_correctly_interleaved,
    _looks_duplicated,
    _wants_translation,
    _wants_verbatim_or_translation,
)

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


def _page(text):
    return {"page_number": 1, "text": text}


def test_strip_recurring_watermark_lines_removes_consistent_header():
    pages = [
        _page("请勿二改勿商用，转载标明出处\n更多图解请关注公众号手工坊\nR1 : 全下针"),
        _page("语勿二改勿商用，转载标明出处\n更多图解请关注公众号手工坊\nR2 : 2下针，加针"),
        _page("请勿二改勿商用，转载标明出处\n更多图解请关注公众号手工坊\nR3 : 全下针"),
        _page("请勿二改勿商用，转载标明出处\n更多图解请关注公众号手工坊\nR4 : 全下针"),
    ]
    _strip_recurring_watermark_lines(pages)
    for i, page in enumerate(pages):
        assert "请勿二改" not in page["text"]
        assert "语勿二改" not in page["text"]
        assert "更多图解" not in page["text"]
    assert "R1 : 全下针" in pages[0]["text"]
    assert "R2 : 2下针，加针" in pages[1]["text"]
    assert "R3 : 全下针" in pages[2]["text"]
    assert "R4 : 全下针" in pages[3]["text"]


def test_strip_recurring_watermark_lines_keeps_repeated_short_content():
    # "全下针" legitimately recurs across many pages as a real instruction —
    # must never be stripped just because it's common.
    pages = [
        _page("耳部\nR1 : 全下针\nR2 : 全下针"),
        _page("尾部\nR1 : 全下针\nR3 : 2下针，加针"),
        _page("身体\nR1 : 全下针\nR4 : 全下针"),
    ]
    _strip_recurring_watermark_lines(pages)
    assert "R1 : 全下针" in pages[0]["text"]
    assert "R2 : 全下针" in pages[0]["text"]
    assert "R1 : 全下针" in pages[1]["text"]
    assert "R1 : 全下针" in pages[2]["text"]


def test_strip_recurring_watermark_lines_requires_page_coverage():
    # The header-like line only appears on 1 of 4 pages — not "recurring".
    pages = [
        _page("请勿二改勿商用，转载标明出处\nR1 : 全下针"),
        _page("R2 : 全下针"),
        _page("R3 : 全下针"),
        _page("R4 : 全下针"),
    ]
    _strip_recurring_watermark_lines(pages)
    assert "请勿二改勿商用，转载标明出处" in pages[0]["text"]


def test_strip_recurring_watermark_lines_noop_for_too_few_pages():
    pages = [
        _page("请勿二改勿商用，转载标明出处\nR1 : 全下针"),
        _page("请勿二改勿商用，转载标明出处\nR2 : 全下针"),
    ]
    _strip_recurring_watermark_lines(pages)
    assert "请勿二改勿商用，转载标明出处" in pages[0]["text"]
    assert "请勿二改勿商用，转载标明出处" in pages[1]["text"]


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


def test_looks_correctly_interleaved_true_for_alternating_lines():
    text = "\n".join([
        "R1 : 全下针", "R1: 전체 아래뜨기",
        "R2 : 2下针，加针", "R2: 2 아래뜨기, 코늘림",
        "R3 : 全下针", "R3: 전체 아래뜨기",
    ])
    assert _looks_correctly_interleaved(text) is True


def test_looks_correctly_interleaved_false_for_block_format():
    # All-original lines first, then all-translated lines after — the
    # observed "page 1" failure mode.
    text = "\n".join([
        "R1 : 全下针", "R2 : 2下针，加针", "R3 : 全下针", "R4 : 全下针",
        "R1: 전체 아래뜨기", "R2: 2 아래뜨기, 코늘림", "R3: 전체 아래뜨기", "R4: 전체 아래뜨기",
    ])
    assert _looks_correctly_interleaved(text) is False


def test_looks_correctly_interleaved_false_when_no_translation_present():
    text = "\n".join(["R1 : 全下针", "R2 : 2下针，加针", "R3 : 全下针", "R4 : 全下针"])
    assert _looks_correctly_interleaved(text) is False


def test_looks_correctly_interleaved_true_for_short_text():
    assert _looks_correctly_interleaved("R1 : 全下针") is True


def test_looks_duplicated_true_for_repeated_block():
    lines = ["R1 : 全下针", "R2 : 2下针，加针", "R3 : 全下针", "R4 : 全下针", "R5 : 全下针", "R6 : 全下针"]
    text = "\n".join(lines[:3] + lines[:3])  # whole block repeated
    assert _looks_duplicated(text) is True


def test_looks_duplicated_false_for_distinct_content():
    text = "\n".join([f"R{i} : 全下针" for i in range(1, 7)])
    assert _looks_duplicated(text) is False


def test_looks_duplicated_true_for_localized_prefix_repeat():
    # Observed failure mode: only a short watermark/header snippet echoes
    # twice before the real (non-repeating) content — not a whole-response
    # mirror, so a naive first-half-vs-second-half check would miss this.
    header = ["请勿二改勿商用", "更多图解请关注"]
    body = [f"R{i} : 全下针" for i in range(1, 9)]  # all distinct, no repeats
    text = "\n".join(header + header + body)
    assert _looks_duplicated(text) is True


def test_looks_duplicated_true_for_pairwise_adjacent_duplicate_lines():
    # Real observed pattern: each watermark/junk line doubled up right where
    # it stands (AABBCC), not a repeated multi-line block (ABCABC) — a
    # block-only scan (k>=2) misses this; must be caught at k=1 too.
    text = "\n".join([
        "雪請勿二改勿商用", "雪請勿二改勿商用",
        "( 更多回解請關注 )", "( 更多回解請關注 )",
        "身体:", "身体:",
        "注意 : 左加针，右加针同一渡线位置挑起分别编织",
    ])
    assert _looks_duplicated(text) is True


def test_looks_duplicated_false_for_single_repeated_line():
    # A single recurring instruction line ("全下针") is common and legitimate
    # across different, non-adjacent rows — must not false-positive.
    text = "\n".join(["R1 : 全下针", "R2 : 2下针，加针", "R3 : 全下针", "R4 : 2下针，加针"])
    assert _looks_duplicated(text) is False


def test_line_instructions_for_translation_request():
    instructions = _line_instructions_for("중국어 도안을 한국어로 번역해줘")
    assert instructions == TRANSLATION_LINE_INSTRUCTIONS
    # The whole point of this instruction set: every line gets its own
    # translation immediately below it, never batched into separate blocks.
    assert "never batch all original lines" in instructions


def test_line_instructions_for_plain_verbatim_request():
    instructions = _line_instructions_for("그대로 나열해줘")
    assert instructions == VERBATIM_INSTRUCTIONS
    assert instructions is not TRANSLATION_LINE_INSTRUCTIONS


def test_wants_translation():
    assert _wants_translation("중국어 도안을 한국어로 번역해줘") is True
    assert _wants_translation("이 문서를 요약해줘") is False


def test_wants_verbatim_or_translation_covers_plain_translation_requests():
    # Regression test: a plain "번역해줘" request has no "그대로"-style verbatim
    # keyword, so _wants_verbatim alone would miss it and route it through the
    # "synthesize a summary" template instead of line-by-line reproduction.
    assert _wants_verbatim_or_translation("중국어 도안을 한국어로 번역해줘") is True
    assert _wants_verbatim_or_translation("그대로 나열해줘") is True
    assert _wants_verbatim_or_translation("이 문서를 요약해줘") is False


def test_estimate_num_ctx_floor_applies_regardless_of_output_budget():
    assert _estimate_num_ctx("", output_budget=PAGE_TRANSLATION_OUTPUT_BUDGET) == NUM_CTX_FLOOR


def test_estimate_num_ctx_larger_output_budget_increases_result():
    text = "x" * 20000  # large enough that floor doesn't mask the difference
    default_ctx = _estimate_num_ctx(text)
    page_ctx = _estimate_num_ctx(text, output_budget=PAGE_TRANSLATION_OUTPUT_BUDGET)
    assert page_ctx > default_ctx


def _ai_message(content, done_reason):
    return AIMessage(content=content, response_metadata={"done_reason": done_reason})


def test_invoke_with_continuation_returns_immediately_on_stop():
    llm = MagicMock()
    llm.invoke.return_value = _ai_message("전체 답변", "stop")
    service = RAGService()

    text, truncated = service._invoke_with_continuation(llm, [], reasoning_steps=[])

    assert text == "전체 답변"
    assert truncated is False
    assert llm.invoke.call_count == 1


def test_invoke_with_continuation_retries_then_completes():
    llm = MagicMock()
    llm.invoke.side_effect = [
        _ai_message("첫 부분 ", "length"),
        _ai_message("나머지 부분", "stop"),
    ]
    service = RAGService()
    reasoning_steps = []

    text, truncated = service._invoke_with_continuation(llm, [], reasoning_steps)

    assert text == "첫 부분 나머지 부분"
    assert truncated is False
    assert llm.invoke.call_count == 2
    # Second call's messages should include the first chunk as prior context.
    second_call_messages = llm.invoke.call_args_list[1].args[0]
    assert any(getattr(m, "content", "") == "첫 부분 " for m in second_call_messages)
    assert any("잘림" in s for s in reasoning_steps)


def test_invoke_with_continuation_gives_up_after_max_attempts():
    llm = MagicMock()
    llm.invoke.return_value = _ai_message("계속 잘림", "length")
    service = RAGService()
    reasoning_steps = []

    text, truncated = service._invoke_with_continuation(llm, [], reasoning_steps)

    assert truncated is True
    assert llm.invoke.call_count == MAX_CONTINUATION_ATTEMPTS + 1
    assert any("⚠️" in s for s in reasoning_steps)


def test_translate_pages_calls_once_per_page_in_order():
    docs = [
        Document(page_content=f"page {i} content", metadata={"pdf_name": "test.pdf"})
        for i in range(3)
    ]
    service = RAGService()

    with patch.object(
        RAGService, "_invoke_with_continuation",
        side_effect=[("번역1", False), ("번역2", False), ("번역3", False)],
    ) as mock_invoke:
        result, truncated, failed_pages = service._translate_pages(docs, "번역해줘", "qwen3:14b", [])

    assert mock_invoke.call_count == 3
    assert result == "번역1\n\n번역2\n\n번역3"
    assert truncated is False
    assert failed_pages == []
    # Each call's system message should contain that page's own content, in order.
    for i, call in enumerate(mock_invoke.call_args_list):
        messages = call.args[1]
        assert f"page {i} content" in messages[0].content


def test_translate_pages_propagates_truncation_from_any_page():
    docs = [
        Document(page_content="page 0", metadata={}),
        Document(page_content="page 1", metadata={}),
    ]
    service = RAGService()

    with patch.object(
        RAGService, "_invoke_with_continuation",
        side_effect=[("ok", False), ("cut off", True)],
    ):
        _, truncated, failed_pages = service._translate_pages(docs, "번역해줘", "qwen3:14b", [])

    assert truncated is True
    assert failed_pages == []


def test_translate_pages_retries_a_hard_failure_then_succeeds():
    docs = [Document(page_content="page 0", metadata={})]
    service = RAGService()
    reasoning_steps = []

    with patch.object(
        RAGService, "_invoke_with_continuation",
        side_effect=[RuntimeError("unexpected EOF"), ("복구됨", False)],
    ) as mock_invoke:
        result, truncated, failed_pages = service._translate_pages(
            docs, "번역해줘", "qwen3:14b", reasoning_steps
        )

    assert mock_invoke.call_count == 2
    assert result == "복구됨"
    assert truncated is False
    assert failed_pages == []
    assert any("재시도" in s for s in reasoning_steps)


def test_translate_pages_keeps_earlier_pages_when_a_later_page_fails_permanently():
    docs = [
        Document(page_content="page 0", metadata={}),
        Document(page_content="page 1", metadata={}),
        Document(page_content="page 2", metadata={}),
    ]
    service = RAGService()
    reasoning_steps = []

    # Page 1 (index 1) fails every attempt (1 initial + MAX_PAGE_RETRY_ATTEMPTS
    # retries); pages 0 and 2 succeed on their first try.
    with patch.object(
        RAGService, "_invoke_with_continuation",
        side_effect=[
            ("페이지0 완료", False),
            RuntimeError("crash 1"), RuntimeError("crash 2"), RuntimeError("crash 3"),
            ("페이지2 완료", False),
        ],
    ) as mock_invoke, patch("time.sleep"):
        result, truncated, failed_pages = service._translate_pages(
            docs, "번역해줘", "qwen3:14b", reasoning_steps
        )

    assert mock_invoke.call_count == 5
    assert failed_pages == [2]  # 1-indexed
    # Pages 0 and 2's successful output must survive despite page 1 failing.
    assert "페이지0 완료" in result
    assert "페이지2 완료" in result
    assert any("최종 실패" in s for s in reasoning_steps)


_BLOCK_FORMAT_TEXT = "\n".join([
    "R1 : 全下针", "R2 : 2下针，加针", "R3 : 全下针", "R4 : 全下针",
    "R1: 전체 아래뜨기", "R2: 2 아래뜨기, 코늘림", "R3: 전체 아래뜨기", "R4: 전체 아래뜨기",
])
_INTERLEAVED_TEXT = "\n".join([
    "R1 : 全下针", "R1: 전체 아래뜨기", "R2 : 2下针，加针", "R2: 2 아래뜨기, 코늘림",
])


def test_translate_pages_retries_on_bad_format_then_succeeds():
    docs = [Document(page_content="page 0", metadata={})]
    service = RAGService()
    reasoning_steps = []

    with patch.object(
        RAGService, "_invoke_with_continuation",
        side_effect=[(_BLOCK_FORMAT_TEXT, False), (_INTERLEAVED_TEXT, False)],
    ) as mock_invoke:
        result, truncated, failed_pages = service._translate_pages(
            docs, "중국어를 한국어로 번역해줘", "qwen3:14b", reasoning_steps
        )

    assert mock_invoke.call_count == 2
    assert result == _INTERLEAVED_TEXT
    assert failed_pages == []
    assert any("형식 검증 실패" in s for s in reasoning_steps)
    # The retry attempt's messages should carry a corrective instruction.
    second_call_messages = mock_invoke.call_args_list[1].args[1]
    assert any("rejected" in getattr(m, "content", "") for m in second_call_messages)


def test_translate_pages_keeps_result_when_format_issue_persists_after_retries():
    docs = [Document(page_content="page 0", metadata={})]
    service = RAGService()
    reasoning_steps = []

    with patch.object(
        RAGService, "_invoke_with_continuation",
        return_value=(_BLOCK_FORMAT_TEXT, False),
    ) as mock_invoke:
        result, truncated, failed_pages = service._translate_pages(
            docs, "중국어를 한국어로 번역해줘", "qwen3:14b", reasoning_steps
        )

    assert mock_invoke.call_count == MAX_PAGE_RETRY_ATTEMPTS + 1
    assert result == _BLOCK_FORMAT_TEXT  # kept despite the unresolved format issue
    assert failed_pages == []  # a format issue is not treated as a hard failure
    assert any("형식 문제 남음" in s for s in reasoning_steps)
