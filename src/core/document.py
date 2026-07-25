"""Document processing functionality."""
import logging
import re
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .text_extractor import TextExtractor

logger = logging.getLogger(__name__)

_KOREAN_RE = re.compile(r'[가-힣]')
_CHINESE_RE = re.compile(r'[一-鿿]')
_ASCII_LETTER_RE = re.compile(r'[A-Za-z]')
_DIGIT_RE = re.compile(r'[0-9]')
_WHITESPACE_RE = re.compile(r'\s')

MIN_TEXT_LENGTH = 50
MEANINGFUL_RATIO_THRESHOLD = 0.3
SPECIAL_RATIO_THRESHOLD = 0.3
SINGLE_CHAR_WORD_RATIO_THRESHOLD = 0.4


def _is_ocr_artifact_text(text: str) -> bool:
    """Heuristic: does this text look like garbled OCR/extraction output?"""
    total = len(text)
    if total == 0:
        return True
    korean = len(_KOREAN_RE.findall(text))
    chinese = len(_CHINESE_RE.findall(text))
    ascii_letters = len(_ASCII_LETTER_RE.findall(text))
    digits = len(_DIGIT_RE.findall(text))
    whitespace = len(_WHITESPACE_RE.findall(text))

    meaningful = korean + chinese + ascii_letters + digits
    meaningful_ratio = meaningful / total
    special = max(total - meaningful - whitespace, 0)
    special_ratio = special / total

    return meaningful_ratio < MEANINGFUL_RATIO_THRESHOLD or special_ratio > SPECIAL_RATIO_THRESHOLD


def _check_single_char_pattern(text: str) -> float:
    """Ratio of space-separated 'words' that are a single character (another OCR-garbling signal)."""
    words = text.split()
    if not words:
        return 0.0
    single_char_words = sum(1 for w in words if len(w) == 1)
    return single_char_words / len(words)


def detect_if_image_based(documents: List) -> bool:
    """Decide whether a loaded PDF is actually a scanned/image-based document
    that needs OCR, based on the quality of text UnstructuredPDFLoader returned."""
    if not documents:
        return True
    combined_text = "\n".join(getattr(doc, "page_content", "") for doc in documents).strip()
    if len(combined_text) < MIN_TEXT_LENGTH:
        return True
    if _is_ocr_artifact_text(combined_text):
        return True
    if _check_single_char_pattern(combined_text) > SINGLE_CHAR_WORD_RATIO_THRESHOLD:
        return True
    return False


class DocumentProcessor:
    """Handles PDF document loading and processing."""

    def __init__(self, chunk_size: int = 7500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.text_extractor = TextExtractor()

    def load_pdf(self, file_path: Path) -> Tuple[List, bool]:
        """Load PDF document.

        Returns (documents, used_ocr). When the PDF turns out to be
        image-based/scanned, `documents` are already page-chunked by the
        OCR path and `used_ocr` is True; callers should skip re-chunking
        via `split_documents` in that case.
        """
        try:
            logger.info(f"Loading PDF from {file_path}")
            # "fast" strategy skips unstructured's own hi-res/OCR handling,
            # since we run our own (with watermark removal) below when needed.
            loader = UnstructuredPDFLoader(str(file_path), strategy="fast")
            documents = loader.load()
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

        if detect_if_image_based(documents):
            logger.info(f"{file_path} appears to be an image-based/scanned PDF; falling back to OCR")
            extraction_result = self.text_extractor.extract_text_from_scanned_pdf(file_path)
            ocr_documents = self.text_extractor.create_document_chunks(
                extraction_result,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            if ocr_documents:
                return ocr_documents, True
            logger.warning(f"OCR fallback produced no text for {file_path}; using original extraction")

        return documents, False

    def split_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        try:
            logger.info("Splitting documents into chunks")
            return self.splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
