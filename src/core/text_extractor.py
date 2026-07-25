"""Advanced text extraction for scanned/image-based PDFs.

Features:
- OCR extraction from image-based (scanned) PDFs, via a vision-LLM
  (deepseek-ocr:3b through Ollama), with watermark removal
- Language detection
- Quality metrics and preprocessing
"""
import io
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import ollama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from .image_analysis import ImageAnalyzer
from .image_handler import ImageHandler

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not installed. Image-based PDF processing will be limited.")

DEFAULT_OCR_LANGUAGE = "eng+chi_sim+chi_tra+kor"
DEFAULT_DPI = 300

# Vision-LLM OCR via Ollama, replacing pytesseract as of 2026-07-24 — live
# comparison on the sample knitting-pattern PDF (11 pages) showed
# deepseek-ocr:3b recognizing text pytesseract consistently got wrong or
# missed entirely (misread CJK characters, unreadable watermark/caption
# text treated as noise, text embedded in photos). No language parameter is
# needed — the model reads whatever script is present — so `ocr_language`
# is accepted by callers below purely for backward compatibility and is not
# used by this path.
DEEPSEEK_OCR_MODEL = "deepseek-ocr:3b"
DEEPSEEK_OCR_PROMPT = (
    "Extract ALL text from this image exactly as it appears, line by line, "
    "preserving line breaks and original order. Do not translate, "
    "summarize, or explain — output only the raw extracted text."
)

# deepseek-ocr formats its output as loose markdown and, on pages with photo
# grids, sometimes emits the actual photo back as a base64-encoded data URI
# reference (e.g. "[ref1]: data:image/png;base64,iVBORw0K...") — neither is
# real page content; both must be stripped before the text reaches
# chunking/embedding. The base64 payload itself is sometimes long enough
# that the model line-wraps it — [A-Za-z0-9+/=\s]+ (not anchored to one
# line) consumes the whole wrapped block, since CJK/punctuation characters
# aren't in the base64 alphabet and naturally end the match where real text
# resumes. A first version of this regex only matched a single line and, on
# a wrapped payload, left the continuation lines (pure base64, no "data:"
# prefix to match against) sitting in the stored text — inflating that one
# page's chunk past the 7500-char splitter threshold and fragmenting it into
# multiple garbage "pages" that then went through the full translation loop.
_MARKDOWN_HEADER_RE = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MARKDOWN_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
_BASE64_IMAGE_RE = re.compile(
    r'(\[ref\d*\]:\s*)?data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+',
    re.IGNORECASE,
)


def _clean_deepseek_ocr_output(text: str) -> str:
    text = _BASE64_IMAGE_RE.sub('', text)
    text = _MARKDOWN_HEADER_RE.sub('', text)
    text = _MARKDOWN_BOLD_RE.sub(r'\1', text)
    return text

# CJK-only OCR language pack (no "eng"). Mixing "eng" into Tesseract's
# language set measurably degrades CJK recognition (e.g. "下针" misread as
# "FH, Get,"), so anything that already knows it's dealing with a CJK
# document — a query-time translation re-OCR, or refreshing a stale
# collection that was embedded before this was understood — should use
# this instead of DEFAULT_OCR_LANGUAGE. Shared by RAGService (query-time
# re-OCR) and PDFService (refresh_ocr, collection refresh).
CJK_OCR_LANGUAGE = "chi_sim+chi_tra+kor"

# Tesseract treats each CJK character as its own "word" and joins them with
# spaces, e.g. "用 魔 环 起". That hurts readability and embedding/retrieval
# quality, so collapse whitespace directly between two CJK characters.
_CJK_SPACING_RE = re.compile(
    r'(?<=[一-鿿㐀-䶿가-힣])\s+(?=[一-鿿㐀-䶿가-힣])'
)


def _collapse_cjk_spacing(text: str) -> str:
    return _CJK_SPACING_RE.sub('', text)


# A running watermark/caption shows up, OCR noise and all,
# as one of the first few lines
# of nearly every page in a scanned pattern. Only the LEADING lines of each
# page are considered as candidates — never anything mid-page — specifically
# so real pattern content that legitimately repeats short instructions across
# pages (e.g. "全下针") is never at risk of being stripped.
_WATERMARK_HEADER_LINES_TO_CHECK = 3
_WATERMARK_LINE_MIN_LENGTH = 12
_WATERMARK_SIMILARITY_THRESHOLD = 0.65
_WATERMARK_PAGE_COVERAGE = 0.6

# OCR pads watermark lines with inconsistent junk at the edges (stray "|",
# "~", digit fragments from a misread logo, extra spacing) that dilutes a
# straight SequenceMatcher ratio enough to split what's really one recurring
# line into several near-miss clusters. Trimming down to the first/last
# letter-like (Latin/CJK/Hangul) character before comparing removes that
# noise while leaving the actual text - including any internal
# punctuation/digits - untouched.
_LETTER_LIKE_RE = re.compile(r'[A-Za-z一-鿿㐀-䶿가-힣]')


def _normalize_watermark_candidate(line: str) -> str:
    stripped = line.strip()
    matches = list(_LETTER_LIKE_RE.finditer(stripped))
    if not matches:
        return stripped
    return stripped[matches[0].start():matches[-1].end()]


def _strip_recurring_watermark_lines(pages: List[Dict]) -> None:
    """Remove a repeated watermark/caption header that OCR picks up at the
    top of most/all pages. Mutates each page dict's "text" in place.

    Only the first `_WATERMARK_HEADER_LINES_TO_CHECK` non-empty lines of
    each page are ever considered (pooled together, not tied to a fixed line
    index — OCR inconsistently drops one part of a multi-line watermark on
    some pages, which shifts where the surviving part lands), and only lines
    that (fuzzy-)match across at least `_WATERMARK_PAGE_COVERAGE` of all
    pages are removed. Needs at least 3 pages to distinguish "recurring"
    from coincidence.
    """
    if len(pages) < 3:
        return

    # Pool candidates from the first few non-empty lines of every page —
    # position bounds which lines are eligible, but doesn't pin the watermark
    # to one exact index, since it can land on line 1 or line 2 depending on
    # whether OCR also picked up the other watermark line on that page.
    candidates: List[tuple] = [
        (page_idx, line)
        for page_idx, page in enumerate(pages)
        for line in [ln for ln in page["text"].splitlines() if ln.strip()][:_WATERMARK_HEADER_LINES_TO_CHECK]
        if len(line.strip()) >= _WATERMARK_LINE_MIN_LENGTH
    ]

    min_pages = max(2, round(len(pages) * _WATERMARK_PAGE_COVERAGE))
    lines_to_remove_by_page: Dict[int, set] = {}

    # Greedily cluster candidates by fuzzy similarity, tolerant of OCR noise.
    #  A new candidate joins a cluster if it's similar enough to ANY member
    # already in it (not just the cluster's first-seen representative)
    # — a straight chain of drifting OCR variants can have
    # a low first-to-last similarity while each
    # neighboring pair is well within threshold.
    clusters: List[Dict] = []
    for page_idx, line in candidates:
        normalized = _normalize_watermark_candidate(line)
        cluster = next(
            (c for c in clusters
             if any(SequenceMatcher(None, member, normalized).ratio() >= _WATERMARK_SIMILARITY_THRESHOLD
                    for member in c["normalized_members"])),
            None,
        )
        if cluster is None:
            cluster = {"normalized_members": [], "pages": set(), "lines": []}
            clusters.append(cluster)
        cluster["normalized_members"].append(normalized)
        cluster["pages"].add(page_idx)
        cluster["lines"].append((page_idx, line))

    for cluster in clusters:
        if len(cluster["pages"]) >= min_pages:
            for page_idx, line in cluster["lines"]:
                lines_to_remove_by_page.setdefault(page_idx, set()).add(line)

    if not lines_to_remove_by_page:
        return

    for page_idx, page in enumerate(pages):
        to_remove = lines_to_remove_by_page.get(page_idx)
        if not to_remove:
            continue
        kept = [ln for ln in page["text"].splitlines() if ln not in to_remove]
        page["text"] = "\n".join(kept)


class TextExtractor:
    """OCR-based text extraction for scanned PDFs."""

    def __init__(self):
        self.image_handler = ImageHandler()
        self.image_analyzer = ImageAnalyzer()
        self.pdf2image_available = PDF2IMAGE_AVAILABLE

    def _extract_text_with_deepseek_ocr(self, image: Image.Image) -> str:
        """Run the raw (unpreprocessed) page image through deepseek-ocr:3b.

        No grayscale/denoise/Otsu preprocessing: that pipeline was tuned for
        pytesseract's classical recognizer and, in live testing, the vision
        model read watermarked/colored pages fine without it — preprocessing
        would only risk destroying detail (e.g. in photo grids) a vision
        model can actually use. Isolated per-page: a single page's OCR
        failure logs and returns "" rather than aborting the whole document,
        matching this pipeline's existing per-page resilience elsewhere
        (e.g. _translate_pages).
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        try:
            response = ollama.chat(
                model=DEEPSEEK_OCR_MODEL,
                messages=[{
                    "role": "user",
                    "content": DEEPSEEK_OCR_PROMPT,
                    "images": [buffer.getvalue()],
                }],
                stream=False,
            )
        except Exception as e:
            logger.error(f"deepseek-ocr extraction failed: {e}")
            return ""
        return _clean_deepseek_ocr_output(response.message.content or "")

    def extract_text_from_scanned_pdf(
        self,
        file_path: Path,
        ocr_language: str = DEFAULT_OCR_LANGUAGE,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> Dict:
        if not self.pdf2image_available:
            return {
                "pages": [], "total_pages": 0, "extraction_method": "failed",
                "quality_metrics": {}, "languages_detected": [],
                "error": "pdf2image not installed",
            }

        logger.info(f"Converting PDF to images: {file_path}")
        kwargs = {"dpi": DEFAULT_DPI}
        if start_page is not None:
            kwargs["first_page"] = start_page
        if end_page is not None:
            kwargs["last_page"] = end_page
        images = convert_from_path(str(file_path), **kwargs)

        pages = []
        blur_variances = []
        confidences = []
        languages_detected = set()

        for i, image in enumerate(images):
            page_number = (start_page or 1) + i
            logger.info(f"Processing page {page_number}")
            quality = self.image_analyzer.analyze_image_quality(image)
            text = self._extract_text_with_deepseek_ocr(image)
            text = _collapse_cjk_spacing(text)
            text_boxes: List[Dict] = []  # no per-word boxes/confidence from a vision-LLM
            language = self.image_analyzer.detect_language(text)

            confidence = self._calculate_confidence(text_boxes)
            blur_variances.append(quality.get("blur_variance", 0.0))
            confidences.append(confidence)
            if language != "unknown":
                languages_detected.add(language)

            pages.append({
                "page_number": page_number,
                "text": text,
                "text_boxes": text_boxes,
                "quality_metrics": quality,
                "language": language,
                "confidence": confidence,
            })

        _strip_recurring_watermark_lines(pages)

        average_blur_variance = sum(blur_variances) / len(blur_variances) if blur_variances else 0.0
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        total_text_characters = sum(len(p["text"]) for p in pages)

        return {
            "pages": pages,
            "total_pages": len(pages),
            "extraction_method": "ocr",
            "quality_metrics": {
                "average_blur_variance": average_blur_variance,
                "quality_assessment": self._assess_quality(average_blur_variance),
            },
            "total_text_characters": total_text_characters,
            "average_confidence": average_confidence,
            "languages_detected": list(languages_detected),
        }

    def _calculate_confidence(self, text_boxes: List[Dict]) -> float:
        if not text_boxes:
            return 0.0
        confidences = [b["confidence"] for b in text_boxes if b.get("confidence", -1) >= 0]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _assess_quality(self, blur_variance: float) -> str:
        if blur_variance > 500:
            return "excellent"
        if blur_variance > 200:
            return "good"
        if blur_variance > 100:
            return "fair"
        return "poor"

    def format_extracted_text(
        self, extraction_result: Dict, include_page_breaks: bool = True, include_metadata: bool = True
    ) -> str:
        lines = []
        if include_metadata:
            lines.append("# Document Metadata")
            lines.append(f"# Total Pages: {extraction_result.get('total_pages', 0)}")
            lines.append(f"# Extraction Method: {extraction_result.get('extraction_method', 'unknown')}")
            quality = extraction_result.get("quality_metrics", {})
            lines.append(f"# Quality: {quality.get('quality_assessment', 'unknown')}")
            lines.append(f"# Languages Detected: {', '.join(extraction_result.get('languages_detected', []))}")
            lines.append("")

        for page in extraction_result.get("pages", []):
            if include_page_breaks:
                lines.append(f"--- Page {page['page_number']} ---")
            lines.append(page["text"])
            lines.append("")

        return "\n".join(lines)

    def create_document_chunks(
        self,
        extraction_result: Dict,
        chunk_size: int = 7500,
        chunk_overlap: int = 100,
        include_page_info: bool = True,
    ) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []

        for page in extraction_result.get("pages", []):
            text = page["text"]
            if not text or not text.strip():
                continue

            page_chunks = splitter.split_text(text)
            for chunk_index, chunk_text in enumerate(page_chunks):
                metadata = {
                    "language_detected": page.get("language", "unknown"),
                    "ocr_confidence": page.get("confidence", 0.0),
                    "text_boxes_count": len(page.get("text_boxes", [])),
                    "chunk_index": chunk_index,
                }
                if include_page_info:
                    metadata["source_page"] = page["page_number"]
                chunks.append(Document(page_content=chunk_text, metadata=metadata))

        return chunks
