# Document Processing API

This page documents the document processing components of Ollama PDF RAG
(`src/core/document.py` and `src/core/text_extractor.py`).

## DocumentProcessor

```python
class DocumentProcessor:
    """Handles PDF document loading and processing."""

    def __init__(self, chunk_size: int = 7500, chunk_overlap: int = 100):
        """Initialize document processor with chunking parameters."""
```

### Methods

#### load_pdf
```python
def load_pdf(self, file_path: Path) -> Tuple[List[Document], bool]:
    """Load a PDF, falling back to OCR if it's scanned/image-based."""
```

Parameters:
- `file_path`: Path to the PDF file

Returns a `(documents, used_ocr)` tuple:
- `documents`: `List[Document]` — native-text extraction result, or (if the
  PDF is scanned/image-based) OCR'd pages already chunked one-per-page
- `used_ocr`: `True` if the OCR fallback ran. When `True`, `documents` is
  already page-chunked — callers should **skip** `split_documents()` (see
  `PDFService.upload_and_process()` for the actual call site)

Internally:
1. Tries `UnstructuredPDFLoader(file_path, strategy="fast")` for native
   text extraction
2. Runs `detect_if_image_based(documents)` on the result — if the text is
   too short, or looks garbled (see below), the PDF is treated as
   scanned/image-based
3. If so, falls back to `TextExtractor.extract_text_from_scanned_pdf()`
   (OCR — see below) instead of the native extraction result

#### split_documents
```python
def split_documents(self, documents: List[Document]) -> List[Document]:
    """Split documents into chunks with overlap, via RecursiveCharacterTextSplitter."""
```

Parameters:
- `documents`: List of Document objects (native-text extraction only —
  OCR'd documents from `load_pdf()` are already page-chunked and should
  not be re-split)

Returns:
- List of chunked Document objects

## detect_if_image_based

```python
def detect_if_image_based(documents: List[Document]) -> bool:
```

Heuristic scan of `UnstructuredPDFLoader`'s output, used by `load_pdf()` to
decide whether the OCR fallback is needed. Returns `True` if any of:

| Signal | Threshold |
|---|---|
| Combined extracted text length | `< MIN_TEXT_LENGTH` (50 chars) |
| "Meaningful" character ratio (Korean + Chinese + ASCII letters + digits) too low, or non-whitespace "special" character ratio too high | `_is_ocr_artifact_text()` — catches garbled extraction, e.g. mostly symbols |
| Ratio of single-character "words" (space-separated) | `> SINGLE_CHAR_WORD_RATIO_THRESHOLD` (0.4) — catches CJK text where each character got split into its own "word" |

## Scanned/Image-Based PDF OCR (`TextExtractor`)

`src/core/text_extractor.py` — used by `DocumentProcessor.load_pdf()` when
`detect_if_image_based()` returns `True`.

```python
class TextExtractor:
    def extract_text_from_scanned_pdf(
        self,
        file_path: Path,
        ocr_language: str = DEFAULT_OCR_LANGUAGE,  # accepted for backward
                                                     # compatibility; unused —
                                                     # see below
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> Dict:
        """Rasterize each page and OCR it via deepseek-ocr:3b (Ollama)."""

    def create_document_chunks(
        self,
        extraction_result: Dict,
        chunk_size: int = 7500,
        chunk_overlap: int = 100,
        include_page_info: bool = True,
    ) -> List[Document]:
        """Turn an extraction result into one Document chunk per page."""
```

OCR pipeline (as of the 2026-07-24 switch from `pytesseract` to a vision-LLM):

1. Each page is rasterized to an image via `pdf2image` (300 DPI) — no
   further preprocessing; the raw page image is sent as-is
2. Each page image is sent to **`deepseek-ocr:3b`** through Ollama, with a
   prompt asking for a verbatim, line-by-line transcription. `ocr_language`
   is accepted by this method purely for signature compatibility with
   callers built for the old pytesseract-based engine — the vision model
   reads whatever script is on the page and does not need it
3. The model's raw output is cleaned up (`_clean_deepseek_ocr_output()`):
   stray markdown headers/bold markers are stripped, and any base64-encoded
   image data the model occasionally echoes back on photo-heavy pages is
   removed (including payloads the model line-wraps across multiple lines)
4. `_collapse_cjk_spacing()` removes unnecessary whitespace between
   consecutive CJK characters
5. `_strip_recurring_watermark_lines()` detects a watermark/caption line
   repeated (via fuzzy matching) across most pages and removes it from all of
   them

Each OCR'd page becomes exactly one chunk (`create_document_chunks()`),
tagged with `source_page` in its metadata — unlike native-text extraction,
which is split by character count via `split_documents()`.

`pytesseract`-based extraction (`ImageAnalyzer.extract_text_with_ocr()` /
`extract_text_boxes()` in `image_analysis.py`, plus `ImageHandler`'s
preprocessing in `image_handler.py`) still exists in the codebase but is no
longer called by this pipeline — it's exercised only by tests, kept pending
a decision on the legacy Streamlit app's direction.

## Usage Example

```python
processor = DocumentProcessor(chunk_size=7500, chunk_overlap=100)

documents, used_ocr = processor.load_pdf(Path("path/to/document.pdf"))
if not used_ocr:
    documents = processor.split_documents(documents)

for doc in documents:
    print(doc.page_content)
    print(doc.metadata)
```

## Error Handling

- `load_pdf()` / `split_documents()` log and re-raise on failure (file not
  found, invalid PDF, etc.) — there's no silent fallback to an empty result
- If the OCR fallback itself produces no text (e.g. a genuinely blank scan),
  `load_pdf()` logs a warning and returns the original (likely empty)
  native-extraction result rather than raising
