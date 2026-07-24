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
    _expects_korean_output,
    _line_instructions_for,
    _detect_page_range,
    _looks_correctly_interleaved,
    _looks_duplicated,
    _looks_untranslated_output,
    _normalize_korean_counts,
    _wants_page_count,
    _wants_raw_page_content,
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


def test_detect_ocr_language_override_falls_back_to_cjk_default_without_two_languages():
    # Translation intent alone (fewer than 2 explicit language names) still
    # narrows away from "eng" — regression test for the exact reported bug:
    # "번역해줘" phrasings that only name the target language (or no language
    # at all) were previously falling through to the un-narrowed, eng-mixed
    # default and using the stale/garbled originally-stored OCR text.
    assert _detect_ocr_language_override("중국어를 번역해줘") == "chi_sim+chi_tra+kor"
    assert _detect_ocr_language_override("1~2페이지 내용을 한국어로 번역해줘") == "chi_sim+chi_tra+kor"
    assert _detect_ocr_language_override("이 문서를 번역해줘") == "chi_sim+chi_tra+kor"
    # No translation intent at all → still None, regardless of language names.
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
    # Translation-mode block (original + Hangul translation lines) repeated
    # wholesale — the real "whole page generated twice" failure mode.
    lines = ["R1 : 全下针", "R1: 전체 아래뜨기", "R2 : 2下针，加针", "R2: 2 아래뜨기, 코늘림"]
    text = "\n".join(lines + lines)
    assert _looks_duplicated(text) is True


def test_looks_duplicated_false_for_distinct_content():
    lines = [f"R{i} : 全下针\nR{i}: 전체 아래뜨기" for i in range(1, 7)]
    text = "\n".join(lines)
    assert _looks_duplicated(text) is False


def test_looks_duplicated_true_for_localized_prefix_repeat():
    # Observed failure mode: only a short watermark/header snippet (with its
    # Hangul translation) echoes twice before the real, non-repeating
    # content — not a whole-response mirror, so a naive first-half-vs-
    # second-half check would miss this.
    header = ["请勿二改勿商用", "재판매 및 상업적 이용 금지"]
    body = [f"R{i} : 全下针\nR{i}: 전체 아래뜨기" for i in range(1, 5)]
    text = "\n".join(header + header + body)
    assert _looks_duplicated(text) is True


def test_looks_duplicated_true_for_pairwise_adjacent_duplicate_lines():
    # Real observed pattern: each translated line doubled up right where it
    # stands (AABBCC), not a repeated multi-line block (ABCABC) — a
    # block-only scan (k>=2) misses this; must be caught at k=1 too.
    text = "\n".join([
        "身体:", "몸통:", "身体:", "몸통:",
        "注意 : 左加针，右加针同一渡线位置挑起分别编织",
        "주의: 왼쪽 코늘림, 오른쪽 코늘림은 같은 실 가닥 위치에서 각각 뜬다",
    ])
    assert _looks_duplicated(text) is True


def test_looks_duplicated_false_for_single_repeated_line():
    # A single recurring instruction line ("全下针") is common and legitimate
    # across different, non-adjacent rows — must not false-positive.
    text = "\n".join([
        "R1 : 全下针", "R1: 전체 아래뜨기",
        "R2 : 2下针，加针", "R2: 2 아래뜨기, 코늘림",
        "R3 : 全下针", "R3: 전체 아래뜨기",
        "R4 : 2下针，加针", "R4: 2 아래뜨기, 코늘림",
    ])
    assert _looks_duplicated(text) is False


def test_looks_duplicated_false_for_repeated_untranslatable_noise():
    # OCR-noise lines with nothing meaningful to translate get echoed
    # verbatim by the model into both the "original" and "translation"
    # slots — this is expected/harmless, not a generation glitch, and must
    # NOT be flagged: flagging it used to burn a full retry budget (minutes
    # per page) on input a retry can never fix.
    text = "\n".join([
        "V 2061 : 人 2", "V 2061 : 人 2",
        "R1 : 全下针", "R1: 전체 아래뜨기",
    ])
    assert _looks_duplicated(text) is False


def test_expects_korean_output_true_for_plain_translation_request():
    assert _expects_korean_output("중국어 도안을 한국어로 번역해줘") is True
    assert _expects_korean_output("1~2페이지 내용을 한국어로 번역해줘") is True


def test_expects_korean_output_false_when_explicit_other_target():
    assert _expects_korean_output("이 문서를 영어로 번역해줘") is False
    assert _expects_korean_output("translate this to Japanese") is False


def test_expects_korean_output_false_when_no_translation_intent():
    assert _expects_korean_output("이 문서를 요약해줘") is False


def test_looks_untranslated_output_true_for_english_only_page():
    text = "\n".join([
        "Long-tail Cast On 69 sts",
        "R1 : 3 sts K, 3 sts P, repeat until end [ 69 sts ]",
        "R2 : Repeat 3 sts P, 3 sts K, repeat until end [ 69 sts ]",
        "R3-7 : Repeat R1-2",
    ])
    assert _looks_untranslated_output(text) is True


def test_looks_untranslated_output_false_for_correctly_interleaved_korean():
    text = "\n".join([
        "R1 : 全下针", "R1: 전체 아래뜨기",
        "R2 : 2下针，加针", "R2: 2 아래뜨기, 코늘림",
        "R3 : 全下针", "R3: 전체 아래뜨기",
        "R4 : 2下针，加针", "R4: 2 아래뜨기, 코늘림",
    ])
    assert _looks_untranslated_output(text) is False


def test_looks_untranslated_output_false_for_short_text():
    assert _looks_untranslated_output("R1 : 全下针") is False


def test_normalize_korean_counts():
    assert _normalize_korean_counts("R32: (1K, k2tog)*6회 [12개의 코]") == "R32: (1K, k2tog)*6회 [12코]"
    assert _normalize_korean_counts("[30개의 코], [18개의 코]") == "[30코], [18코]"
    # Already-correct notation and unrelated text are left untouched.
    assert _normalize_korean_counts("[12코]") == "[12코]"
    assert _normalize_korean_counts("전체 아래뜨기") == "전체 아래뜨기"


def test_normalize_korean_counts_with_attached_particle():
    # Regression test: Korean particles (와/를/...) attach directly to 코
    # with no separating character, so a regex relying on \b for the
    # boundary silently fails here (Python's \b is Unicode-aware and does
    # not treat Hangul-Hangul as a boundary).
    text = "남은 3개의 코와 시작하는 3개의 코를 마커에 넣습니다"
    assert _normalize_korean_counts(text) == "남은 3코와 시작하는 3코를 마커에 넣습니다"


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


def test_wants_page_count():
    assert _wants_page_count("이 pdf는 총 몇 페이지야?") is True
    assert _wants_page_count("How many pages does this document have?") is True
    assert _wants_page_count("이 문서를 요약해줘") is False


def test_query_multi_pdf_answers_page_count_without_llm():
    # The LLM only ever sees chunked page TEXT as context, never page_count
    # metadata, so it can never answer this correctly on its own — must be
    # short-circuited before any model call.
    from types import SimpleNamespace

    service = RAGService()
    fake_pdf = SimpleNamespace(name="test.pdf", pdf_id="pdf_1", page_count=11, doc_count=11)
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [fake_pdf]

    answer, sources, reasoning_steps = service.query_multi_pdf(
        question="이 pdf는 총 몇 페이지야?", model="qwen3:14b", pdf_ids=["pdf_1"], db=db
    )

    assert "11페이지" in answer
    assert sources == [{"pdf_name": "test.pdf", "pdf_id": "pdf_1", "chunk_index": 0}]
    assert any("메타데이터에서 직접 답변" in s for s in reasoning_steps)


def test_detect_page_range_various_formats():
    assert _detect_page_range("1~2페이지 내용을 알려줘") == (1, 2)
    assert _detect_page_range("1-2페이지 내용 보여줘") == (1, 2)
    assert _detect_page_range("1페이지부터 3페이지까지 번역해줘") == (1, 3)
    assert _detect_page_range("pages 2-4 translate please") == (2, 4)
    assert _detect_page_range("page 5 내용 알려줘") == (5, 5)
    assert _detect_page_range("1페이지에 코가 몇 개야?") == (1, 1)
    assert _detect_page_range("이 문서 요약해줘") is None


def test_wants_raw_page_content():
    assert _wants_raw_page_content("1~2페이지 중국어 내용을 알려줘") is True
    assert _wants_raw_page_content("1페이지부터 3페이지까지 번역해줘") is False  # translation intent
    assert _wants_raw_page_content("1페이지에 코가 몇 개야?") is False  # analytical ("몇")


def test_query_multi_pdf_answers_page_range_raw_content_without_llm():
    from types import SimpleNamespace

    service = RAGService()
    fake_pdf = SimpleNamespace(
        name="test.pdf", pdf_id="pdf_1", page_count=3, doc_count=3,
        collection_name="col_1", file_path=None,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [fake_pdf]

    fake_vector_db = MagicMock()
    fake_vector_db.get.return_value = {
        "documents": ["page 1 content", "page 2 content", "page 3 content"],
        "metadatas": [{"source_page": 1}, {"source_page": 2}, {"source_page": 3}],
    }

    with patch("src.api.services.rag_service.Chroma", return_value=fake_vector_db), \
         patch("src.api.services.rag_service.OllamaEmbeddings"):
        answer, sources, reasoning_steps = service.query_multi_pdf(
            question="1~2페이지 내용을 알려줘", model="qwen3:14b", pdf_ids=["pdf_1"], db=db
        )

    assert "page 1 content" in answer
    assert "page 2 content" in answer
    assert "page 3 content" not in answer  # outside the 1-2 range, filtered out
    assert len(sources) == 2
    assert any("페이지 범위 감지" in s for s in reasoning_steps)
    assert any("LLM 호출 없음" in s for s in reasoning_steps)


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
