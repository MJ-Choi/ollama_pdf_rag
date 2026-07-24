"""RAG query service."""
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from langchain_ollama import ChatOllama
import ollama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from ..database import PDFMetadata, ChatSession, ChatMessage
from ..config import settings
from ...core.text_extractor import TextExtractor, CJK_OCR_LANGUAGE

logger = logging.getLogger(__name__)

# Below this many total chunks across the selected PDF(s), skip similarity
# search entirely and feed the model the ENTIRE document instead. Retrieval
# (MultiQueryRetriever, top-k) is a lossy approximation meant for large
# corpora; for a small, deliberately-selected PDF (the common case here —
# users pick specific PDFs to chat with) it can easily miss whole pages for
# vague/exhaustive requests like "extract all the Chinese text", since
# similarity search has no notion of "return everything". Full-document
# mode sidesteps that class of bug for the sizes this app typically sees.
FULL_CONTEXT_CHUNK_LIMIT = 40

# Answer-generation output budget reserved on top of estimated input size
# when sizing Ollama's num_ctx (context window). Without this, Ollama falls
# back to its own default (often just 2048 tokens), which can silently
# truncate context far below what the model actually supports.
NUM_CTX_OUTPUT_BUDGET = 2048
NUM_CTX_FLOOR = 4096
NUM_CTX_CEILING = 32768

# Per-page budget used by _translate_pages (see below) — generous because a
# single page's content is small, so this is cheap even at 2x the default.
PAGE_TRANSLATION_OUTPUT_BUDGET = 4096

# Extra attempts (beyond the first) to let a model continue a response that
# Ollama's done_reason says was cut off before reaching a natural stop.
# Bounds worst-case latency instead of looping indefinitely.
MAX_CONTINUATION_ATTEMPTS = 2

# Extra attempts (beyond the first) for a single page in _translate_pages
# after a HARD failure (e.g. a mid-generation Ollama crash — "unexpected
# EOF" — observed to leave Ollama's own server in a state that recovers a
# few seconds later). Distinct from MAX_CONTINUATION_ATTEMPTS, which handles
# a *successful* call that was merely cut off by num_ctx.
MAX_PAGE_RETRY_ATTEMPTS = 2
PAGE_RETRY_BACKOFF_SECONDS = 2


def _estimate_num_ctx(*texts: str, output_budget: int = NUM_CTX_OUTPUT_BUDGET) -> int:
    """Rough token-budget estimate for Ollama's num_ctx from prompt text length.

    Uses a conservative ~2 chars/token estimate (safe for CJK-heavy text,
    where 1 char is often ~1 token) plus an output budget, clamped to a sane
    floor/ceiling. `output_budget` defaults to NUM_CTX_OUTPUT_BUDGET but can
    be overridden (e.g. PAGE_TRANSLATION_OUTPUT_BUDGET for per-page calls).
    """
    estimated_input_tokens = sum(len(t) for t in texts) // 2
    return max(
        NUM_CTX_FLOOR,
        min(NUM_CTX_CEILING, estimated_input_tokens + output_budget),
    )


# Keywords signaling the user wants raw source text reproduced as-is
# (line-by-line transcription/extraction), not a synthesized/summarized
# answer. The default chain-of-thought prompt actively pushes the model to
# "synthesize a comprehensive answer", which is wrong for these requests —
# e.g. "extract the Chinese text" got turned into a summary glossary table
# instead of the original lines.
VERBATIM_INTENT_KEYWORDS = [
    "그대로", "원문", "나열해", "줄 단위", "요약하지 말고", "요약 없이", "요약하지말고",
    "verbatim", "raw text", "line by line", "line-by-line", "as-is", "word for word",
    "그대로 추출", "그대로 번역",
]


def _wants_verbatim(question: str) -> bool:
    """Detect requests for a verbatim line-by-line reproduction rather than a synthesized answer."""
    lowered = question.lower()
    return any(keyword.lower() in lowered for keyword in VERBATIM_INTENT_KEYWORDS)


# Shared instruction text for line-by-line reproduction, used by both the
# plain-chain prompt template and (via _line_instructions_for) the per-page
# translation loop's system message.
VERBATIM_INSTRUCTIONS = (
    "Reproduce the source text below EXACTLY as it appears, line by line, in its "
    "original order. Do NOT summarize, paraphrase, synthesize into a table, "
    "deduplicate, or reorganize by topic. Keep every line separate, in the same "
    "sequence as the context. Only insert a source label (e.g. [Source: filename]) "
    "when switching between different source documents."
)

# Stricter instructions used instead of VERBATIM_INSTRUCTIONS when the question
# asks for a translation. A weak "if a translation is requested, place it after
# each line" conditional buried inside VERBATIM_INSTRUCTIONS turned out to be
# unreliable across independent per-page calls: some pages skipped translating
# entirely, others dumped a whole translated block after the whole original
# block instead of interleaving line by line. This spells out the exact
# required structure so every page follows the same, unambiguous format.
TRANSLATION_LINE_INSTRUCTIONS = (
    "Reproduce AND translate the source text below, line by line, in its original "
    "order. For EVERY line of source content, output exactly two lines in this "
    "order: (1) the original source line, unmodified, then (2) immediately below "
    "it, its translation. Repeat this original-then-translation pair for every "
    "single line in the source, all the way through — never batch all original "
    "lines together followed by all translations as separate blocks, never merge "
    "multiple original lines into one translated line, and never skip translating "
    "a line. Do NOT summarize, paraphrase, deduplicate, or reorganize by topic — "
    "preserve the exact order and every line, including row/section labels (e.g. "
    "R1, R2, 耳部). Only insert a source label (e.g. [Source: filename]) when "
    "switching between different source documents. If a PRIORITY REFERENCE "
    "CONTEXT section appears above, its term mappings are mandatory for your "
    "translation — use its exact wording for any matching term instead of your "
    "own phrasing, even if your own phrasing seems equally correct. Apply each "
    "mapping literally per occurrence of that exact source text — do NOT "
    "generalize a compound term's mapping to a different, standalone "
    "occurrence of one of its characters. For example, if the context maps a "
    "compound term (e.g. 下针 → K) AND separately maps one of its characters "
    "alone (e.g. 针 → 코), a standalone occurrence of that character — such as "
    "in a stitch count like [12针] — must use ITS OWN mapping (→ [12코]), not "
    "the compound term's mapping (never [12개의 K]). When a count appears with "
    "a unit, join the number directly to the unit with no counter particle in "
    "between — write '12코', never '12개의 코'.\n\n"
    "Example of the required STRUCTURE only (the actual terms below are "
    "placeholders — do NOT reuse this exact wording; always prefer the "
    "priority reference context's terms for real content):\n"
    "R1 : [source line A]\n"
    "R1: [translation of line A]\n"
    "R2 : [source line B]\n"
    "R2: [translation of line B]\n"
    "Follow this exact interleaving pattern for every line below, and do not "
    "repeat any line or block twice."
)


# Explicit language names → tesseract OCR language codes. Used to narrow the
# OCR language pack for a single query when the question names a source/target
# translation pair (e.g. "중국어 도안을 한국어로 번역해줘"). "eng" is deliberately
# NOT in DEFAULT_OCR_LANGUAGE's replacement here unless English is one of the
# named languages — mixing in an unneeded English dictionary measurably
# degrades Tesseract's CJK recognition (e.g. "下针" misread as "FH, Get,").
_LANGUAGE_NAME_TO_OCR_CODE = {
    "중국어": "chi_sim+chi_tra",
    "한국어": "kor",
    "영어": "eng",
}
TRANSLATION_INTENT_KEYWORDS = ["번역", "translate", "translation"]


# Fallback OCR language used for a translation-intent question that doesn't
# explicitly name (or only names one of) the source/target languages — e.g.
# "1~2페이지 내용을 한국어로 번역해줘" names only the target ("한국어"). This
# app's real documents are CJK scans (see chi_knitting.json), and every test
# this session showed "eng" measurably degrading CJK recognition — so
# translation intent alone is treated as enough signal to drop it, rather
# than requiring the user to spell out "중국어" every time. Reuses
# CJK_OCR_LANGUAGE (text_extractor.py) — same default PDFService.refresh_ocr()
# uses to rebuild a stale collection.


def _detect_ocr_language_override(question: str) -> Optional[str]:
    """For a translation-style question, return a narrowed tesseract language
    string to re-OCR with — e.g. "chi_sim+chi_tra+kor" for "중국어가 source,
    한국어가 target", or the same CJK-only default (CJK_OCR_LANGUAGE) when the
    question doesn't name two explicit languages. Returns None (keep
    DEFAULT_OCR_LANGUAGE, which still includes "eng") only when there's no
    translation intent at all.
    """
    if not any(keyword.lower() in question.lower() for keyword in TRANSLATION_INTENT_KEYWORDS):
        return None
    matched_names = [name for name in _LANGUAGE_NAME_TO_OCR_CODE if name in question]
    if len(matched_names) < 2:
        return CJK_OCR_LANGUAGE
    codes: List[str] = []
    for name in matched_names:
        for part in _LANGUAGE_NAME_TO_OCR_CODE[name].split("+"):
            if part not in codes:
                codes.append(part)
    return "+".join(codes)


def _wants_translation(question: str) -> bool:
    """True if the question asks for a translation (reuses TRANSLATION_INTENT_KEYWORDS,
    the same keyword set _detect_ocr_language_override uses). A translation request
    inherently means "reproduce every line, translated" — not "synthesize a summary" —
    even when it doesn't also contain an explicit verbatim keyword like "그대로"."""
    lowered = question.lower()
    return any(keyword.lower() in lowered for keyword in TRANSLATION_INTENT_KEYWORDS)


def _wants_verbatim_or_translation(question: str) -> bool:
    """Combined gate for picking the line-by-line template, disabling thinking
    mode, and (for full-document context) triggering the per-page generation loop."""
    return _wants_verbatim(question) or _wants_translation(question)


def _line_instructions_for(question: str) -> str:
    """Pick strict original+translation interleaving instructions when the
    question asks for a translation, vs plain verbatim reproduction otherwise."""
    return TRANSLATION_LINE_INSTRUCTIONS if _wants_translation(question) else VERBATIM_INSTRUCTIONS


# "How many pages" questions are answerable directly from PDFMetadata.page_count
# — already known with certainty from the upload-time OCR/chunking pass — so
# there's no reason to burn an LLM call on it. Worse, the LLM literally has no
# way to answer this correctly even if asked to: it only ever sees the chunked
# page TEXT as context, never the page_count metadata itself, so without this
# short-circuit it always (correctly, from its perspective) says the context
# doesn't contain that information.
PAGE_COUNT_KEYWORDS = [
    "몇 페이지", "페이지 수", "총 페이지", "몇 장",
    "how many pages", "page count", "number of pages", "total pages",
]


def _wants_page_count(question: str) -> bool:
    lowered = question.lower()
    return any(keyword.lower() in lowered for keyword in PAGE_COUNT_KEYWORDS)


# Detects an explicit page range/single page named in the question (e.g.
# "1~2페이지", "1-2페이지", "1페이지부터 2페이지까지", "pages 1-2", "page 1").
# Every OCR'd page is already stored/re-extracted as its own chunk with a
# `source_page` metadata field — this just lets a question select a subset
# of those chunks instead of always using the whole document.
_PAGE_RANGE_PATTERNS = [
    re.compile(r'(\d+)\s*페이지부터\s*(\d+)\s*페이지까지'),
    re.compile(r'(\d+)\s*[~\-–]\s*(\d+)\s*페이지'),
    re.compile(r'pages?\s*(\d+)\s*(?:-|~|to)\s*(\d+)', re.IGNORECASE),
]
_SINGLE_PAGE_PATTERNS = [
    re.compile(r'(\d+)\s*페이지'),
    re.compile(r'\bpage\s*(\d+)\b', re.IGNORECASE),
]


def _detect_page_range(question: str) -> Optional[Tuple[int, int]]:
    for pattern in _PAGE_RANGE_PATTERNS:
        match = pattern.search(question)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return (min(start, end), max(start, end))
    for pattern in _SINGLE_PAGE_PATTERNS:
        match = pattern.search(question)
        if match:
            page = int(match.group(1))
            return (page, page)
    return None


# When a page range is named without a translation intent, the question is
# usually just "show me what's on those pages" — answerable directly from
# already-stored/extracted page text, no LLM needed. These keywords signal
# the opposite: the user wants the model to actually reason about the
# content (count something, explain it, summarize it), so it still needs
# the engine even though a page range was named.
_ANALYTICAL_INTENT_KEYWORDS = [
    "몇", "얼마", "어떻게", "왜", "설명해", "요약",
    "explain", "how many", "why", "summarize", "summary",
]


def _wants_raw_page_content(question: str) -> bool:
    """True if a page-range question should be answered with the raw stored
    page text directly, instead of being handed to the LLM."""
    if _wants_translation(question):
        return False
    lowered = question.lower()
    return not any(keyword.lower() in lowered for keyword in _ANALYTICAL_INTENT_KEYWORDS)


# Heuristics below catch two failure modes observed from qwen3:14b on the
# per-page translation task even with TRANSLATION_LINE_INSTRUCTIONS' explicit
# instructions + few-shot example: (1) dumping all original lines first
# followed by all translations as a separate block instead of interleaving,
# or skipping translation entirely, and (2) generating the whole page's
# content twice in a single response. Both are checked structurally rather
# than by re-reading the model's own claim about what it did.
_HANGUL_RE = re.compile(r'[가-힣]')


def _looks_correctly_interleaved(text: str) -> bool:
    """False if a translation-mode response looks like a block of original
    lines followed by a separate block of translated (Hangul) lines, or
    contains no Hangul at all despite translation being requested."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return True  # too short to meaningfully judge structure
    has_hangul = [bool(_HANGUL_RE.search(ln)) for ln in lines]
    if not any(has_hangul):
        return False  # no translation anywhere
    midpoint = len(lines) // 2
    first_half_ratio = sum(has_hangul[:midpoint]) / midpoint
    second_half_ratio = sum(has_hangul[midpoint:]) / (len(lines) - midpoint)
    # Properly interleaved output has translated lines spread roughly evenly
    # throughout; almost-none-then-mostly is the "all originals, then all
    # translations" block failure mode.
    return not (first_half_ratio < 0.15 and second_half_ratio > 0.6)


def _looks_duplicated(text: str) -> bool:
    """True if the response contains a contiguous block of lines that is
    immediately repeated back-to-back AND the repeated block includes at
    least one translated (Hangul) line — covers the real failure modes (the
    WHOLE page generated twice, or translated lines doubled up in place)
    while deliberately IGNORING identical adjacent copies of lines with no
    Hangul in them. The latter is the model's expected behavior for
    untranslatable OCR noise (a junk line like "V 2061 : 人 2" gets copied
    verbatim into both the "original" and "translation" slots) — flagging it
    used to burn MAX_PAGE_RETRY_ATTEMPTS retries (minutes per page) on
    something a retry can never fix, since the input itself is untranslatable.
    Only called in translation mode (see _translate_pages), so requiring
    Hangul in the duplicate evidence is safe."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    if n < 2:
        return False
    # Check adjacent-block repeats lines[i:i+k] == lines[i+k:i+2k], from the
    # largest possible block down to a single line.
    for k in range(n // 2, 0, -1):
        for i in range(0, n - 2 * k + 1):
            if lines[i:i + k] == lines[i + k:i + 2 * k] and any(
                _HANGUL_RE.search(ln) for ln in lines[i:i + k]
            ):
                return True
    return False


# --- Target-language validation (fix for "last page drifts to English") ---
# A multi-page translation was observed coming back with one page translated
# into English while every other page was correctly Korean, for the same
# question. Each page is an independent LLM call, so validate each page's
# OUTPUT language and retry through the existing format-issue loop.
_EXPLICIT_NON_KOREAN_TARGET_RE = re.compile(
    r'(?:영어|일본어|중국어|english|japanese|chinese)\s*로\s*번역'
    r'|(?:in)?to\s+(?:english|japanese|chinese)',
    re.IGNORECASE,
)


def _expects_korean_output(question: str) -> bool:
    """True if a translation request's target language is Korean — the app's
    default assumption (Korean-speaking user, Korean UI) unless the question
    explicitly names a different target like "영어로 번역"/"translate to
    English". Note "중국어로 된 문서를 한국어로 번역" does NOT match the
    non-Korean pattern ("중국어로 된" isn't followed by 번역)."""
    if not _wants_translation(question):
        return False
    return not _EXPLICIT_NON_KOREAN_TARGET_RE.search(question)


def _looks_untranslated_output(text: str) -> bool:
    """True if a Korean-target translation response contains (almost) no
    Hangul lines — i.e. the model answered in the wrong language entirely,
    or skipped translating. A correctly interleaved Chinese→Korean page has
    Hangul on roughly half its lines; an English/Chinese-only page has ~0."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False  # too short to judge reliably
    hangul_lines = sum(1 for ln in lines if _HANGUL_RE.search(ln))
    return hangul_lines / len(lines) < 0.15


# Deterministic cleanup for a phrasing quirk qwen3 keeps producing even with
# glossary + instructions: stitch counts rendered as "12개의 코" instead of
# the natural knitting notation "12코". Applied to translation-mode answers
# as a post-processing step, so it holds regardless of model compliance.
# No trailing \b: Korean particles (와/를/가/...) attach directly to 코 with
# no separating character, and \w is Unicode-aware so Python's \b does NOT
# treat a Hangul-Hangul boundary as a word boundary — "12개의 코와" would
# silently fail to match with one.
_KOREAN_COUNT_PARTICLE_RE = re.compile(r'(\d+)\s*개의\s*코')


def _normalize_korean_counts(text: str) -> str:
    return _KOREAN_COUNT_PARTICLE_RE.sub(r'\1코', text)


class RAGService:
    """Service for RAG operations."""

    def __init__(self):
        """Initialize RAG service."""
        self.persist_directory = settings.VECTOR_DB_DIR
        self.priority_context = self._load_priority_context()

    def _load_priority_context(self) -> str:
        """Load priority reference material from data/context/*.json.

        `data/context/` holds any number of JSON files the user maintains
        as authoritative reference material for RAG answers — term
        glossaries (e.g. Chinese knitting-pattern terminology), style/rule
        sheets, domain facts, etc. Every file in the directory is picked up
        automatically and must be consulted BEFORE the model's own built-in
        knowledge whenever it's relevant to the question.

        Two shapes are supported per file:
        - A flat string-to-string dict (e.g. {"下针": "K (겉뜨기)"}) is
          rendered as a "term → value" lookup list.
        - Anything else (nested objects, lists, etc.) is pretty-printed
          as-is under a heading with the filename.

        Returns a formatted prompt section, or "" if no context files exist.
        """
        context_dir = Path(settings.PROJECT_ROOT) / settings.CONTEXT_DIR
        if not context_dir.is_dir():
            return ""

        sections = []
        for json_path in sorted(context_dir.glob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load context file {json_path}: {e}")
                continue

            if not data:
                continue

            if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
                body = "\n".join(f"- {key} → {value}" for key, value in data.items())
            else:
                body = json.dumps(data, ensure_ascii=False, indent=2)

            sections.append(f"[{json_path.name}]\n{body}")

        if not sections:
            return ""

        return (
            "PRIORITY REFERENCE CONTEXT (from data/context/). Treat this as authoritative "
            "and consult it FIRST whenever it's relevant to the question — prefer it over "
            "your own built-in knowledge or assumptions (e.g. use these exact terms/values "
            "instead of your own translation or recollection):\n\n"
            + "\n\n".join(sections)
            + "\n\n"
        )

    def _reocr_pdf_chunks(
        self,
        pdf: PDFMetadata,
        ocr_language: str,
        reasoning_steps: List[str],
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Optional[List[Document]]:
        """Re-run OCR on `pdf`'s original file with a narrower language pack,
        for this query only — the stored vector DB collection is untouched.
        `page_range` (start, end), if given, limits OCR to just those pages
        instead of the whole document — faster, and the returned docs'
        `source_page` metadata already reflects the real page numbers.

        Returns None (caller falls back to the stored/embedded chunks) when
        the PDF wasn't originally OCR'd, the file is missing, or extraction
        yields nothing.
        """
        # doc_count == page_count is how the OCR path stores documents (one
        # chunk per page, see PDFService.upload_and_process); a native-text
        # PDF chunked at 7500 chars/page almost always produces more chunks
        # than pages. Imperfect, but avoids re-OCR'ing large native PDFs.
        if pdf.doc_count != pdf.page_count:
            return None
        if not pdf.file_path or not Path(pdf.file_path).exists():
            reasoning_steps.append(f"⚠️ 재-OCR 건너뜀 — 원본 파일을 찾을 수 없음: {pdf.name}")
            return None

        page_note = f", {page_range[0]}-{page_range[1]}페이지만" if page_range else ""
        reasoning_steps.append(f"🔁 재-OCR 중 ({ocr_language}{page_note}): {pdf.name}")
        extractor = TextExtractor()
        extraction_result = extractor.extract_text_from_scanned_pdf(
            Path(pdf.file_path), ocr_language=ocr_language,
            start_page=page_range[0] if page_range else None,
            end_page=page_range[1] if page_range else None,
        )
        docs = extractor.create_document_chunks(extraction_result)
        if not docs:
            reasoning_steps.append(f"⚠️ 재-OCR 결과 없음 — 기존 저장된 청크로 대체: {pdf.name}")
            return None

        for i, doc in enumerate(docs):
            doc.metadata.update({
                "pdf_id": pdf.pdf_id,
                "pdf_name": pdf.name,
                "chunk_index": i,
            })
        reasoning_steps.append(f"✅ 재-OCR로 {len(docs)}개 청크 재추출: {pdf.name}")
        return docs

    def _invoke_with_continuation(
        self,
        llm: ChatOllama,
        messages: List[BaseMessage],
        reasoning_steps: List[str],
        label: str = "",
    ) -> Tuple[str, bool]:
        """Invoke `llm` and, if Ollama's done_reason signals the response was cut
        off before finishing naturally (anything other than "stop"), retry up to
        MAX_CONTINUATION_ATTEMPTS times asking the model to continue exactly
        where it left off, instead of silently returning a partial answer.

        Returns (full_text, truncated) — truncated is True only if the retry
        cap was hit and the last attempt still didn't end in "stop".
        """
        full_text = ""
        for attempt in range(MAX_CONTINUATION_ATTEMPTS + 1):
            ai_message = llm.invoke(messages)
            chunk = ai_message.content or ""
            full_text += chunk
            done_reason = ai_message.response_metadata.get("done_reason")

            if done_reason == "stop" or not chunk:
                return full_text, False
            if attempt == MAX_CONTINUATION_ATTEMPTS:
                reasoning_steps.append(
                    f"⚠️ {label}응답이 잘렸습니다 (done_reason={done_reason}) — "
                    f"{MAX_CONTINUATION_ATTEMPTS}회 이어쓰기 후에도 미완료"
                )
                return full_text, True

            reasoning_steps.append(
                f"↪️ {label}응답이 중간에 잘림 (done_reason={done_reason}) — "
                f"이어쓰기 {attempt + 1}/{MAX_CONTINUATION_ATTEMPTS}"
            )
            messages = messages + [
                AIMessage(content=chunk),
                HumanMessage(content=(
                    "Continue exactly where you left off. Do not repeat any earlier "
                    "lines and do not add commentary — output only the continuation."
                )),
            ]
        return full_text, True

    def _invoke_ollama_chat_with_continuation(
        self,
        model: str,
        messages: List[Dict[str, str]],
        num_ctx: int,
        reasoning_steps: List[str],
    ) -> Tuple[str, bool]:
        """Raw-ollama-client equivalent of _invoke_with_continuation, for the
        think=True path (kept on the raw client so the existing "💡 Model's
        chain-of-thought" reasoning_steps entries keep working)."""
        full_text = ""
        for attempt in range(MAX_CONTINUATION_ATTEMPTS + 1):
            ollama_response = ollama.chat(
                model=model, messages=messages, think=True, stream=False,
                options={"num_ctx": num_ctx},
            )
            if getattr(ollama_response.message, "thinking", None):
                thinking_text = ollama_response.message.thinking
                reasoning_steps.append(
                    f"💡 Model's chain-of-thought:\n{thinking_text[:500]}"
                    f"{'...' if len(thinking_text) > 500 else ''}"
                )
            chunk = ollama_response.message.content or ""
            full_text += chunk
            done_reason = ollama_response.done_reason

            if done_reason == "stop" or not chunk:
                return full_text, False
            if attempt == MAX_CONTINUATION_ATTEMPTS:
                reasoning_steps.append(f"⚠️ 응답이 잘렸습니다 (done_reason={done_reason}) — "
                                        f"{MAX_CONTINUATION_ATTEMPTS}회 이어쓰기 후에도 미완료")
                return full_text, True

            reasoning_steps.append(
                f"↪️ 응답이 중간에 잘림 (done_reason={done_reason}) — "
                f"이어쓰기 {attempt + 1}/{MAX_CONTINUATION_ATTEMPTS}"
            )
            messages = messages + [
                {"role": "assistant", "content": chunk},
                {"role": "user", "content": (
                    "Continue exactly where you left off. Do not repeat any earlier "
                    "lines and do not add commentary — output only the continuation."
                )},
            ]
        return full_text, True

    def _translate_pages(
        self,
        docs: List[Document],
        question: str,
        model: str,
        reasoning_steps: List[str],
    ) -> Tuple[str, bool, List[int]]:
        """Reproduce/translate a multi-page document one page at a time so each
        call's required OUTPUT size is bounded by a single page's content
        instead of the whole document — the actual fix for full-document
        translations silently cutting off partway through. Thinking mode is
        disabled per-page (reasoning=False): chain-of-thought reasoning adds
        no value to a mechanical line-by-line task and would otherwise eat
        into the same per-call output budget as the visible answer.

        Each page is independent: a HARD failure on one page (e.g. Ollama
        itself crashing mid-generation) is retried up to MAX_PAGE_RETRY_ATTEMPTS
        times, and if it still fails, that page is recorded as failed and the
        loop continues to the next page — every other page's already-completed
        translation is kept. Nothing gets thrown away because one page had a
        bad run.

        Returns (concatenated_answer, any_page_truncated, failed_page_numbers).
        failed_page_numbers is empty when every page eventually succeeded.
        """
        page_outputs: List[str] = []
        any_truncated = False
        failed_pages: List[int] = []

        for i, doc in enumerate(docs):
            source = doc.metadata.get("pdf_name", "Unknown")
            page_label = f"{i + 1}/{len(docs)}"
            reasoning_steps.append(f"📄 페이지 {page_label} 처리 중... ({source})")

            page_num_ctx = _estimate_num_ctx(
                doc.page_content, question, self.priority_context,
                output_budget=PAGE_TRANSLATION_OUTPUT_BUDGET,
            )
            instructions = _line_instructions_for(question)
            system_message = (
                f"{self.priority_context}{instructions} This is page {i + 1} of "
                f"{len(docs)} of the same document — do not add a page-transition "
                "summary or commentary, just the reproduced/translated lines for THIS page.\n\n"
                f"Context (page {i + 1}/{len(docs)}, source: {source}):\n{doc.page_content}"
            )
            messages: List[BaseMessage] = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"Question: {question}\n\nOutput this page's content now:"),
            ]

            translation_requested = _wants_translation(question)
            page_text: Optional[str] = None
            truncated = False
            last_error: Optional[Exception] = None
            format_issue: Optional[str] = None
            attempt_messages = messages

            for retry in range(MAX_PAGE_RETRY_ATTEMPTS + 1):
                last_error = None
                format_issue = None
                try:
                    page_llm = ChatOllama(model=model, num_ctx=page_num_ctx, reasoning=False)
                    page_text, truncated = self._invoke_with_continuation(
                        page_llm, attempt_messages, reasoning_steps, label=f"[페이지 {page_label}] "
                    )
                except Exception as e:
                    last_error = e

                # Format issues only apply when the call itself succeeded and
                # a translation was actually requested for this page.
                if last_error is None and translation_requested and page_text:
                    if _looks_duplicated(page_text):
                        format_issue = "응답이 중복 생성됨"
                    elif _expects_korean_output(question) and _looks_untranslated_output(page_text):
                        format_issue = "요청한 언어(한국어)로 번역되지 않음"
                    elif not _looks_correctly_interleaved(page_text):
                        format_issue = "원문/번역이 줄 단위로 교차되지 않음"

                if last_error is None and format_issue is None:
                    break  # success
                if retry == MAX_PAGE_RETRY_ATTEMPTS:
                    break  # out of retries — handled below

                if last_error is not None:
                    reasoning_steps.append(
                        f"⚠️ 페이지 {page_label} 처리 중 오류 ({last_error}) — "
                        f"재시도 {retry + 1}/{MAX_PAGE_RETRY_ATTEMPTS}"
                    )
                    attempt_messages = messages
                    time.sleep(PAGE_RETRY_BACKOFF_SECONDS)
                else:
                    reasoning_steps.append(
                        f"⚠️ 페이지 {page_label} 형식 검증 실패 ({format_issue}) — "
                        f"재시도 {retry + 1}/{MAX_PAGE_RETRY_ATTEMPTS}"
                    )
                    attempt_messages = messages + [
                        AIMessage(content=page_text),
                        HumanMessage(content=(
                            f"That output was rejected: {format_issue}. Redo this page "
                            "from scratch following the exact original-line-then-"
                            "translation-line interleaved format shown earlier, with "
                            "no duplicated content, translating into the language the "
                            "user asked for (한국어, unless they explicitly requested "
                            "another language)."
                        )),
                    ]

            if last_error is not None:
                failed_pages.append(i + 1)
                page_outputs.append(f"⚠️ [페이지 {page_label} 처리 실패: {last_error}]")
                reasoning_steps.append(f"❌ 페이지 {page_label} 최종 실패 — 다음 페이지로 진행")
                continue

            if format_issue is not None:
                reasoning_steps.append(f"⚠️ 페이지 {page_label} 형식 문제 남음 ({format_issue}) — 결과는 유지")

            any_truncated = any_truncated or truncated
            page_outputs.append(page_text.strip())
            reasoning_steps.append(f"✅ 페이지 {page_label} 완료")

        return "\n\n".join(page_outputs), any_truncated, failed_pages

    def query_multi_pdf(
        self,
        question: str,
        model: str,
        pdf_ids: Optional[List[str]],
        db: Session
    ) -> Tuple[str, List[Dict], List[str]]:
        """Query across multiple PDFs with source attribution.

        Args:
            question: User question
            model: LLM model to use
            pdf_ids: List of PDF IDs to query (None = all PDFs)
            db: Database session

        Returns:
            Tuple of (answer, sources, reasoning_steps)
        """
        reasoning_steps = []

        # Get PDF metadata
        query = db.query(PDFMetadata)
        if pdf_ids:
            query = query.filter(PDFMetadata.pdf_id.in_(pdf_ids))
        pdfs = query.all()

        if not pdfs:
            return "No PDFs found to query.", [], []

        reasoning_steps.append(f"📚 Searching across {len(pdfs)} PDF(s): {', '.join([p.name for p in pdfs])}")

        # Page-count questions are answered directly from stored metadata —
        # see _wants_page_count's docstring for why the LLM can never get
        # this right on its own.
        if _wants_page_count(question):
            reasoning_steps.append("📐 페이지 수 질문 감지 — 저장된 메타데이터에서 직접 답변 (LLM 호출 없음)")
            answer = "\n".join(f"- {pdf.name}: {pdf.page_count}페이지" for pdf in pdfs)
            sources = [
                {"pdf_name": pdf.name, "pdf_id": pdf.pdf_id, "chunk_index": 0}
                for pdf in pdfs
            ]
            reasoning_steps.append("✨ Answer generated successfully!")
            return answer, sources, reasoning_steps

        # An explicit "중국어 도안을 한국어로 번역해줘"-style question narrows OCR to
        # just the named source+target languages for THIS query (re-OCR from the
        # original file, not persisted back to the stored vector DB/collection).
        ocr_language_override = _detect_ocr_language_override(question)
        if ocr_language_override:
            reasoning_steps.append(
                f"🔤 번역 요청 감지 — 이번 질의에 한해 OCR 언어를 '{ocr_language_override}'로 좁혀서 원본 PDF를 다시 읽습니다"
            )

        # An explicit page range/single page named in the question (e.g.
        # "1~2페이지") scopes both re-OCR (if any) and the retrieved chunks
        # to just those pages — every OCR'd page is already its own chunk
        # with a `source_page` field, this just selects a subset of them.
        page_range = _detect_page_range(question)
        if page_range:
            reasoning_steps.append(f"📑 페이지 범위 감지: {page_range[0]}~{page_range[1]}페이지로 컨텍스트를 좁힙니다")

        # Initialize LLM (used for query-generation only; the final answer
        # call gets its own instance sized to the actual context below)
        llm = ChatOllama(model=model)
        reasoning_steps.append(f"🤖 Using model: {model}")

        # Small, deliberately-selected PDF set → skip similarity search and
        # use the whole document(s) as context. See FULL_CONTEXT_CHUNK_LIMIT.
        total_chunks = sum(pdf.doc_count for pdf in pdfs)
        use_full_context = total_chunks <= FULL_CONTEXT_CHUNK_LIMIT

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        all_docs = []

        if use_full_context:
            reasoning_steps.append(
                f"📖 {total_chunks} total chunk(s) across selected PDF(s) — using full-document context "
                f"(no similarity search, so nothing gets missed)"
            )

            for pdf in pdfs:
                docs = None
                if ocr_language_override:
                    docs = self._reocr_pdf_chunks(
                        pdf, ocr_language_override, reasoning_steps, page_range=page_range
                    )

                if docs is not None:
                    all_docs.extend(docs)
                    continue

                vector_db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings,
                    collection_name=pdf.collection_name
                )
                try:
                    reasoning_steps.append(f"📄 Loading full document: {pdf.name}")
                    raw = vector_db.get(include=["documents", "metadatas"])
                    docs = [
                        Document(page_content=text, metadata=meta or {})
                        for text, meta in zip(raw["documents"], raw["metadatas"])
                    ]
                    for doc in docs:
                        doc.metadata.setdefault("pdf_name", pdf.name)
                        doc.metadata.setdefault("pdf_id", pdf.pdf_id)
                    # Preserve original page/chunk order for a coherent read
                    docs.sort(
                        key=lambda d: (
                            d.metadata.get("source_page", d.metadata.get("chunk_index", 0)),
                            d.metadata.get("chunk_index", 0),
                        )
                    )
                    all_docs.extend(docs)
                    reasoning_steps.append(f"✅ Loaded {len(docs)} chunk(s) from {pdf.name}")
                except Exception as e:
                    reasoning_steps.append(f"⚠️ Error loading {pdf.name}: {str(e)}")
                    print(f"Error loading {pdf.name}: {e}")
        else:
            # Query prompt for multi-query retriever
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate 2
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}"""
            )

            reasoning_steps.append("🔍 Generating alternative search queries...")
            if ocr_language_override:
                reasoning_steps.append(
                    "ℹ️ 재-OCR 언어 좁히기는 전체-문서 모드에서만 지원됩니다 (선택한 PDF가 많아 "
                    "유사도 검색 모드로 전환됨) — 기존 저장된 청크를 그대로 사용합니다"
                )

            for pdf in pdfs:
                vector_db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings,
                    collection_name=pdf.collection_name
                )

                retriever = MultiQueryRetriever.from_llm(
                    vector_db.as_retriever(search_kwargs={"k": 3}),
                    llm,
                    prompt=QUERY_PROMPT
                )

                try:
                    reasoning_steps.append(f"📄 Retrieving from: {pdf.name}")
                    # Use invoke instead of deprecated get_relevant_documents
                    docs = retriever.invoke(question)
                    # Ensure metadata is present
                    for doc in docs:
                        if "pdf_name" not in doc.metadata:
                            doc.metadata["pdf_name"] = pdf.name
                        if "pdf_id" not in doc.metadata:
                            doc.metadata["pdf_id"] = pdf.pdf_id
                    all_docs.extend(docs)
                    reasoning_steps.append(f"✅ Found {len(docs)} relevant chunks in {pdf.name}")
                except Exception as e:
                    reasoning_steps.append(f"⚠️ Error retrieving from {pdf.name}: {str(e)}")
                    print(f"Error retrieving from {pdf.name}: {e}")

        reasoning_steps.append(f"📊 Total chunks retrieved: {len(all_docs)}")

        # Scope to the requested page range (full-document mode only — a
        # retrieval/large-corpus query's chunks aren't reliably one-per-page).
        # Docs without a source_page are kept rather than dropped, since we
        # can't confirm whether they're in range.
        if page_range and use_full_context:
            start, end = page_range
            filtered_docs = [
                doc for doc in all_docs
                if doc.metadata.get("source_page") is None or start <= doc.metadata["source_page"] <= end
            ]
            if filtered_docs:
                all_docs = filtered_docs
                reasoning_steps.append(f"✂️ {start}~{end}페이지로 필터링: {len(all_docs)}개 청크")
            else:
                reasoning_steps.append(f"⚠️ {start}~{end}페이지에 해당하는 청크를 찾지 못함 — 전체 컨텍스트 사용")

            # A plain "show me / tell me the content of these pages" request
            # (no translation, no analytical keyword) is answerable directly
            # from the already-extracted page text — no LLM call needed.
            if filtered_docs and _wants_raw_page_content(question):
                reasoning_steps.append("📋 페이지 원문 직접 반환 (LLM 호출 없음)")
                answer_parts = []
                for doc in all_docs:
                    page_label = doc.metadata.get("source_page")
                    header = f"[{page_label}페이지]" if page_label is not None else ""
                    answer_parts.append(f"{header}\n{doc.page_content}".strip())
                answer = "\n\n".join(answer_parts)
                sources = [
                    {
                        "pdf_name": doc.metadata.get("pdf_name", "Unknown"),
                        "pdf_id": doc.metadata.get("pdf_id", ""),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                    }
                    for doc in all_docs
                ]
                reasoning_steps.append("✨ Answer generated successfully!")
                return answer, sources, reasoning_steps

        # In full-document mode, use every chunk; otherwise keep the
        # existing top-10 cap for retrieval-based (large corpus) mode.
        context_limit = len(all_docs) if use_full_context else 10

        # Format context with source labels
        context_parts = []
        for doc in all_docs[:context_limit]:
            source = doc.metadata.get("pdf_name", "Unknown")
            context_parts.append(f"[Source: {source}]\n{doc.page_content}\n")

        formatted_context = "\n---\n".join(context_parts)
        reasoning_steps.append(f"🔗 Using {min(len(all_docs), context_limit)} chunk(s) for context")

        verbatim_mode = _wants_verbatim_or_translation(question)
        reasoning_steps.append(
            f"📝 Answer mode: {'verbatim line-by-line reproduction' if verbatim_mode else 'synthesized answer'}"
        )

        # Check if model supports thinking (e.g., qwen3, deepseek-r1). Thinking
        # is disabled for verbatim/translation requests: chain-of-thought
        # reasoning adds no value to a mechanical line-by-line task and would
        # otherwise eat into the same output budget as the visible answer.
        thinking_models = ['qwen3', 'deepseek-r1', 'qwen', 'deepseek']
        supports_thinking = any(tm in model.lower() for tm in thinking_models)
        use_thinking = supports_thinking and not verbatim_mode

        # A large multi-page verbatim/translation request is translated one
        # page at a time instead of in one giant call, so no single call's
        # required output can exceed a bound tied to one page's content —
        # this is what actually fixes full-document translations silently
        # cutting off partway through (see _translate_pages docstring).
        use_page_loop = use_full_context and verbatim_mode and len(all_docs[:context_limit]) > 1

        if use_page_loop:
            reasoning_steps.append(
                f"🔀 다중 페이지 verbatim/번역 요청 감지 — 페이지별 생성 루프 사용 "
                f"({len(all_docs[:context_limit])}페이지, 응답 잘림 방지 + 사고모드 비활성화)"
            )
            response, truncated, failed_pages = self._translate_pages(
                all_docs[:context_limit], question, model, reasoning_steps
            )
        else:
            failed_pages = []
            if verbatim_mode:
                # Reproduce (and, if requested, translate) source lines as-is —
                # explicitly forbid summarizing, paraphrasing, or reorganizing.
                template = (
                    "{priority_context}" + _line_instructions_for(question) + """

        Context:
        {context}

        Question: {question}

        Output the content line by line, in original order:"""
                )
            else:
                # RAG prompt template with chain-of-thought
                template = """{priority_context}Answer the question based ONLY on the following context from multiple PDF documents.
        Each section is marked with its source document.

        Use chain-of-thought reasoning:
        1. First, identify which parts of the context are relevant to the question
        2. Analyze the information from each source document
        3. Synthesize the information to form a comprehensive answer
        4. Ensure you cite the source document name for each piece of information
        5. If information comes from multiple sources, mention all relevant sources
        6. If sources contradict, note the discrepancy and cite both sources

        Context:
        {context}

        Question: {question}

        Think step-by-step and provide your answer with source citations:"""

            # Size the answer-generation model's context window to what we're
            # actually sending it, instead of relying on Ollama's default
            # (often 2048 tokens) which can silently truncate long contexts.
            num_ctx = _estimate_num_ctx(template, formatted_context, self.priority_context, question)
            reasoning_steps.append(f"🧮 Sizing model context window: num_ctx={num_ctx}")
            reasoning_steps.append("💭 Generating answer with source citations...")

            if use_thinking:
                reasoning_steps.append("🧠 Using thinking-enabled model with chain-of-thought reasoning...")
                cot_system_message = f"""You are an expert AI assistant that uses chain-of-thought reasoning.

{self.priority_context}Answer the question based ONLY on the provided context from PDF documents.

CHAIN-OF-THOUGHT PROCESS:
1. **Read and understand** the question carefully
2. **Scan the context** to identify all relevant information
3. **Break down** the information by source document
4. **Analyze** how each piece relates to the question
5. **Synthesize** a comprehensive answer
6. **Cite sources** explicitly for every claim

Context from PDF documents:
{formatted_context}

Think through each step carefully, showing your reasoning process."""
                try:
                    response, truncated = self._invoke_ollama_chat_with_continuation(
                        model=model,
                        messages=[
                            {"role": "system", "content": cot_system_message},
                            {"role": "user", "content": f"Question: {question}\n\nThink step-by-step and provide a detailed answer with source citations."}
                        ],
                        num_ctx=num_ctx,
                        reasoning_steps=reasoning_steps,
                    )
                except Exception as e:
                    print(f"Error using thinking mode, falling back to standard: {e}")
                    use_thinking = False  # fall through to the plain path below

            if not use_thinking:
                answer_llm = ChatOllama(model=model, num_ctx=num_ctx, reasoning=False)
                prompt = ChatPromptTemplate.from_template(template)
                prompt_messages = prompt.invoke({
                    "context": formatted_context,
                    "question": question,
                    "priority_context": self.priority_context,
                }).to_messages()
                response, truncated = self._invoke_with_continuation(
                    answer_llm, prompt_messages, reasoning_steps
                )

        # Deterministic phrasing cleanup for translation answers ("12개의 코"
        # → "12코") — instructions alone don't fully stop the model from
        # inserting the counter particle, so normalize it here regardless.
        if _wants_translation(question):
            response = _normalize_korean_counts(response)

        # Extract source information
        sources = [
            {
                "pdf_name": doc.metadata.get("pdf_name"),
                "pdf_id": doc.metadata.get("pdf_id"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            }
            for doc in all_docs[:context_limit]
        ]

        # Never silently claim success on a response that Ollama itself
        # reported as cut off before a natural stop, even after retries —
        # and never silently drop pages that hard-failed even after retries.
        if failed_pages:
            page_list = ", ".join(str(p) for p in failed_pages)
            response = response + (
                f"\n\n⚠️ [참고: 페이지 {page_list}는 반복된 오류로 처리하지 못했습니다 — "
                "나머지 페이지는 정상적으로 포함되어 있습니다.]"
            )
            reasoning_steps.append(f"⚠️ 페이지 {page_list} 최종 실패 — 나머지는 정상 반환")

        if truncated:
            response = response + "\n\n⚠️ [참고: 이 답변은 완전히 생성되지 못했을 수 있습니다 — 응답이 중간에 잘렸습니다.]"
            reasoning_steps.append("⚠️ 답변이 불완전하게 생성되었습니다 (이어쓰기 시도 후에도 완료되지 않음)")

        if not failed_pages and not truncated:
            reasoning_steps.append("✨ Answer generated successfully!")

        return response, sources, reasoning_steps

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]],
        db: Session
    ) -> ChatMessage:
        """Save chat message to database.

        Args:
            session_id: Chat session identifier
            role: Message role (user or assistant)
            content: Message content
            sources: Source documents (for assistant messages)
            db: Database session

        Returns:
            Saved chat message
        """
        # Ensure session exists
        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            session = ChatSession(
                session_id=session_id,
                created_at=datetime.now(),
                last_active=datetime.now()
            )
            db.add(session)
        else:
            session.last_active = datetime.now()

        # Save message
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            timestamp=datetime.now()
        )
        db.add(message)
        db.commit()
        db.refresh(message)

        return message

    def get_session_messages(self, session_id: str, db: Session) -> List[ChatMessage]:
        """Get all messages for a session.

        Args:
            session_id: Chat session identifier
            db: Database session

        Returns:
            List of chat messages
        """
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp).all()
