"""Advanced text extraction for scanned/image-based PDFs.

Features:
- OCR extraction from image-based (scanned) PDFs, with watermark removal
- Language detection
- Quality metrics and preprocessing
"""
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .image_analysis import ImageAnalyzer
from .image_handler import ImageHandler

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not installed. Image-based PDF processing will be limited.")

try:
    import pytesseract  # noqa: F401
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR functionality will be limited.")

DEFAULT_OCR_LANGUAGE = "eng+chi_sim+chi_tra+kor"
DEFAULT_DPI = 300

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
        self.ocr_available = TESSERACT_AVAILABLE

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
        if not self.ocr_available:
            return {
                "pages": [], "total_pages": 0, "extraction_method": "failed",
                "quality_metrics": {}, "languages_detected": [],
                "error": "pytesseract not installed",
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
            preprocessed = self.image_handler.preprocess_for_ocr(image)
            quality = self.image_analyzer.analyze_image_quality(preprocessed)
            text = self.image_analyzer.extract_text_with_ocr(preprocessed, lang=ocr_language)
            text = _collapse_cjk_spacing(text)
            text_boxes = self.image_analyzer.extract_text_boxes(preprocessed, lang=ocr_language)
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
