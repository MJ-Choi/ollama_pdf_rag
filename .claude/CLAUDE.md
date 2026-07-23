# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 절대규칙
- 클로드는 환경변수 파일(.env*)을 건들지 않는다
- 답변은 한국어로만 한다
- Frontend: @web-ui/claude.md

## Project Overview

A full-stack local RAG (Retrieval Augmented Generation) application for chatting with PDF documents using Ollama and LangChain. The project consists of:
- **Python Backend**: FastAPI REST API with RAG pipeline, PDF processing, and ChromaDB vector storage
- **Next.js Frontend**: Modern React UI with chat persistence, PDF management, and model selection
- **Streamlit App**: Alternative UI for experimentation (legacy)

## Architecture

### Backend (Python)

**Core Components**:
- `src/core/rag.py`: RAG pipeline using LangChain's MultiQueryRetriever and prompt chains
- `src/core/llm.py`: LLM configuration and prompt management via OllamaLLM
- `src/core/embeddings.py`: Vector embeddings via OllamaEmbeddings with ChromaDB storage
- `src/core/document.py`: PDF text/OCR loading and chunking. `DocumentProcessor.load_pdf()` returns `(documents, used_ocr)` — auto-detects scanned/image-based PDFs via `detect_if_image_based()` (text length, CJK-aware garbled-text ratio, single-char-word ratio) and falls back to OCR when native `UnstructuredPDFLoader` extraction is too poor
- `src/core/text_extractor.py`: OCR extraction for scanned/image-based PDFs via `pytesseract` + `pdf2image` (300 DPI, default OCR language `eng+chi_sim+chi_tra+kor`, optional deps that degrade gracefully if missing); collapses spurious spacing Tesseract inserts between CJK characters
- `src/core/image_handler.py`: Image preprocessing for OCR — auto-rotate, denoise, grayscale, and **watermark removal** (`remove_watermark()`, OpenCV Otsu binarization that strips light-gray tiled/ghosted watermarks while preserving ink strokes)
- `src/core/image_analysis.py`: `pytesseract` OCR wrapper (text + text boxes), image quality metrics (blur/brightness/contrast via OpenCV Laplacian), language detection (`langdetect`)

**API Layer** (`src/api/`):
- `main.py`: FastAPI app setup with CORS middleware (allows localhost:3000 for Next.js); registers `pdfs`, `query`, `models`, `health` routers
- `database.py`: SQLite with SQLAlchemy models (PDFMetadata, ChatSession, ChatMessage)
- `routers/`: pdfs (upload/delete), query (RAG), models, health (status)
- `services/`: Business logic — `pdf_service.py`, `rag_service.py`
- `config.py`: Settings for Ollama connection (default: http://localhost:11434)

**Key Flow**:
1. User uploads PDF → text extracted (OCR + watermark removal fallback for scanned PDFs) → chunked → stored in ChromaDB
2. Query arrives → MultiQueryRetriever generates variations → retrieves relevant chunks → priority reference context from `data/context/*.json` injected into the prompt → LLM synthesizes response with sources


### Data Storage

- **Vector DB**: ChromaDB (persisted to `data/vectors/`)
- **PDFs**: Stored in `data/pdfs/` directory
- **API Database**: SQLite at `data/api.db`
- **Frontend Database**: Drizzle ORM (configurable backend)

## Common Development Commands

### Python Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server (on port 8001)
python run_api.py

# Run tests
python -m pytest tests/ -v
python -m pytest tests/ --cov=src

# Run a single test file / test
python -m pytest tests/test_rag.py -v
python -m pytest tests/test_rag.py::test_specific_case -v

# Pre-commit hooks (runs `unittest discover tests` + `pylint` on staged Python files)
pre-commit install
pre-commit run --all-files
```

Note: pre-commit actually runs `python -m unittest discover tests`, not pytest, and there is no `pytest.ini`/`pyproject.toml` at root configuring pytest — `python -m pytest tests/` works via pytest defaults but isn't what pre-commit enforces.

### Frontend (web-ui)

```bash
cd web-ui
pnpm install

# Dev server (Next.js, port 3000)
pnpm dev

# Production build (runs `tsx lib/db/migrate` first, then `next build`)
pnpm build

# Lint / format (Ultracite/Biome, see below)
pnpm lint
pnpm format

# Drizzle schema management
pnpm db:generate   # create a migration file
pnpm db:push       # push schema directly (no migration file; handy for quick local iteration)
pnpm db:studio     # inspect DB in Drizzle Studio

# Playwright e2e tests
pnpm test
```

Frontend TS/TSX/JS/JSX code is governed by an always-applied Cursor rule (`web-ui/.cursor/rules/ultracite.mdc`) wrapping Ultracite/Biome — covers accessibility, complexity, correctness, TypeScript, style, Next.js, and testing conventions. `pnpm lint` checks, `pnpm format` auto-fixes.

### Running Everything

```bash
# Start all services (Terminal 1: Backend, Terminal 2: Frontend)
./start_all.sh

# Or manually:
# Terminal 1
python run_api.py

# Terminal 2
cd web-ui && pnpm dev
```

## Architecture Decisions & Key Patterns

### RAG Pipeline
- Uses **MultiQueryRetriever** to generate multiple query variations before retrieval, improving recall
- Chunks are 7500 characters with 100-character overlap (`DocumentProcessor(chunk_size=7500, chunk_overlap=100)` in `pdf_service.py`)
- ChromaDB provides exact cosine similarity matching
- All processing is local—no data leaves the machine

### Answer generation: truncation safety net & per-page translation loop
- `RAGService.query_multi_pdf()` sizes `num_ctx` per-call via `_estimate_num_ctx()`, but a single flat output budget isn't enough for a full-document line-by-line translation (e.g. "이 중국어 도안을 한국어로 번역해줘" across an 11-page PDF) — the model would silently get cut off mid-answer, with Ollama's `done_reason` never checked.
- Every generation call now goes through `RAGService._invoke_with_continuation()` (LangChain `ChatOllama` path) or `_invoke_ollama_chat_with_continuation()` (raw-`ollama` thinking-model path): if `done_reason != "stop"`, it retries up to `MAX_CONTINUATION_ATTEMPTS` times asking the model to continue from where it left off, and appends a visible `⚠️` warning to the answer text itself if still incomplete after retries — never silently reports `"✨ Answer generated successfully!"` on a truncated response.
- `_wants_verbatim_or_translation()` (verbatim keywords OR `TRANSLATION_INTENT_KEYWORDS`) gates: (1) disabling thinking mode (`reasoning=False` / `think=True` skipped) since chain-of-thought reasoning burns output budget with no value for mechanical transcription, and (2) `RAGService._translate_pages()` — when full-document context has more than one page, translate/reproduce one page per LLM call (bounded per-call output, `PAGE_TRANSLATION_OUTPUT_BUDGET`) instead of one giant call for the whole document, then concatenate in order. Slower (up to one Ollama call per page) but actually completes instead of guessing a big-enough single budget.
- `_translate_pages()` is resilient per page, not all-or-nothing: a HARD failure (e.g. Ollama itself crashing mid-generation — observed "unexpected EOF") is retried up to `MAX_PAGE_RETRY_ATTEMPTS` times with a short backoff; if a page still fails, it's recorded in `failed_pages` and marked inline in the output, but every OTHER page's already-completed translation is kept and returned rather than the whole request raising and losing everything (previously an uncaught exception anywhere in the loop discarded all prior pages' work — see `query.py`'s generic `except Exception` → HTTP 500 with nothing saved).
- Translation-mode pages are also validated structurally after generation via `_looks_correctly_interleaved()` (rejects the model dumping all original lines then all translations as two separate blocks, or skipping translation entirely) and `_looks_duplicated()` (rejects a page's content — or even just a repeated line/header snippet — being generated twice in one response); either failure retries the same page (same `MAX_PAGE_RETRY_ATTEMPTS` budget, with a corrective follow-up message), and if still unresolved after retries the result is kept anyway (flagged, not discarded) rather than treated as a hard failure. `TRANSLATION_LINE_INSTRUCTIONS` includes a concrete few-shot example of the required original-then-translation interleaving to reduce how often this retry path is needed in the first place.

### API Design
- RESTful with `/api/v1/` prefix
- CORS allows only localhost:3000 (configured in `src/api/main.py`)
- Responses include source citations and metadata
- Health endpoint at `/api/v1/health`

### Frontend State
- Chats are persisted to the database via Drizzle ORM
- Each chat can be associated with specific PDFs
- Streaming responses via Vercel's AI SDK
- Real-time chat history updates

### Environment & Configuration
- Backend connects to Ollama at `http://localhost:11434` (configurable in `src/api/config.py`)
- Frontend talks to backend at `http://localhost:8001/api/v1`
- SQLite for local API database (no external dependencies)
- Optional: PostgreSQL support for frontend via Drizzle

### OCR & Watermark Removal (scanned PDFs)
- `DocumentProcessor.load_pdf()` loads via `UnstructuredPDFLoader(strategy="fast")` first, then `detect_if_image_based()` decides whether the extracted text is real or garbage (too short / CJK-aware garbled-character ratio / single-char-word ratio)
- OCR path: `pdf2image` renders pages at 300 DPI → `ImageHandler.preprocess_for_ocr()` (auto-rotate, denoise, grayscale, then **Otsu binarization to strip light-gray watermarks**) → `pytesseract` (`eng+chi_sim+chi_tra+kor`, `--psm 6`) → CJK inter-character spacing cleanup → page-level `Document` chunks
- `PDFService.upload_and_process()` skips re-chunking when `used_ocr=True` since the OCR path already returns page-chunked documents
- Tuning knobs: OCR language list (`text_extractor.DEFAULT_OCR_LANGUAGE`), DPI (`text_extractor.DEFAULT_DPI`), detection thresholds (`document.MIN_TEXT_LENGTH`, `MEANINGFUL_RATIO_THRESHOLD`, `SPECIAL_RATIO_THRESHOLD`, `SINGLE_CHAR_WORD_RATIO_THRESHOLD`)
- Requires the system `tesseract` binary with the needed language packs (`chi_sim`, `chi_tra`, `kor`) installed — the Python packages alone aren't sufficient
- **Query-time OCR language override**: mixing `eng` into Tesseract's language set measurably degrades CJK recognition (e.g. `下针` misread as `FH, Get,`), but dropping `eng` from the default risks English-only scans. `RAGService._detect_ocr_language_override()` (`rag_service.py`) instead narrows the OCR language *per query*: when the question names an explicit source+target translation pair (e.g. "중국어 도안을 한국어로 번역해줘" → `chi_sim+chi_tra+kor`, no `eng`), `RAGService._reocr_pdf_chunks()` re-runs OCR on that PDF's original file with the narrowed language set for that query only — the stored ChromaDB collection is untouched. Only applies in full-document-context mode (`doc_count == page_count`, i.e. the PDF was originally OCR'd); large/retrieval-mode queries and PDFs without a resolvable `file_path` fall back to the stored chunks unchanged.
- **Recurring watermark/caption removal**: `text_extractor._strip_recurring_watermark_lines()` runs after all pages are OCR'd in `extract_text_from_scanned_pdf()`, removing a caption that OCR picks up near the top of most/all pages (e.g. a "do not resell, follow our shop" line stamped on every page of a scanned pattern) before it ever reaches chunking/translation. Only the first `_WATERMARK_HEADER_LINES_TO_CHECK` non-empty lines of each page are eligible candidates (never mid-page content, so a legitimately repeated short instruction like `全下针` is never at risk); candidates are fuzzy-clustered (`difflib.SequenceMatcher`, any-linkage so a chain of drifting OCR variants still merges, edge-trimmed to `[A-Za-z一-鿿㐀-䶿가-힣]` before comparing since OCR pads watermark lines with inconsistent junk characters that otherwise dilute the match) and a cluster is only stripped if it covers `_WATERMARK_PAGE_COVERAGE` of all pages — low-confidence/rarely-OCR'd variants are deliberately left alone rather than risking a false removal.

### Priority Reference Context (`data/context/`)
- `data/context/` is a general-purpose store for **any** `*.json` file the user wants the model to treat as authoritative and consult BEFORE its own built-in knowledge — not limited to translation. Term glossaries (`chi_knitting.json`), rule sheets, domain facts, style guides, etc. can all live here side by side
- Two shapes are supported per file (`RAGService._load_priority_context()` in `src/api/services/rag_service.py`):
  - A flat `{"key": "value"}` string dict is rendered as a `key → value` lookup list (glossary style)
  - Any other JSON shape (nested objects, lists, ...) is pretty-printed as-is under a heading with the filename
- `RAGService` reloads **every** `*.json` file in the directory on each query (no server restart needed after adding/editing files) and injects the combined result into both the standard prompt and the thinking-model (qwen3/deepseek) system message, instructing the LLM to prefer this context over its own knowledge/assumptions
- Files are never merged together — each is kept as its own `[filename.json]`-labeled section, listed in alphabetical filename order

## Important Files & Their Purposes

**Backend**:
- `src/core/rag.py` — RAG pipeline orchestration
- `src/core/document.py`, `text_extractor.py`, `image_handler.py`, `image_analysis.py` — PDF text extraction, OCR fallback, watermark removal
- `src/api/routers/query.py` — Query endpoint that drives RAG
- `src/api/routers/pdfs.py` — PDF upload/deletion/listing
- `src/api/services/rag_service.py` — RAG query orchestration + `data/context/` priority-context injection
- `src/api/database.py` — SQLAlchemy models for metadata
- `data/context/*.json` — priority reference context (glossaries, rules, domain facts, ...) — see Priority Reference Context above

**Config**:
- `requirements.txt` — Python dependencies (LangChain 1.0.0, ChromaDB, FastAPI, Streamlit, `pytesseract`/`pdf2image`/`opencv-python-headless`/`langdetect` for OCR)
- `web-ui/package.json` — Node dependencies (Next.js 16, Vercel AI SDK, Drizzle, Radix UI)
- `run_api.py` — FastAPI entry point with uvicorn config

## Development Workflow

1. **Adding API Endpoints**:
   - Create router in `src/api/routers/`
   - Add to imports in `src/api/main.py`
   - Update CORS if cross-origin access needed
   - Test with curl or Swagger UI at `http://localhost:8001/docs`

2. **Modifying RAG Behavior**:
   - Edit prompt templates in `src/core/llm.py`
   - Adjust chunk size/overlap in `src/core/document.py`
   - Modify retriever settings in `src/core/rag.py`
   - Test with existing PDFs in ChromaDB

3. **Frontend Component Changes**:
   - Use Radix UI for consistency
   - Fetch data via `lib/api/client.ts` methods
   - Update Drizzle schema if schema changes needed
   - Run `pnpm db:generate` to create a migration file, then `pnpm db:migrate` to apply it (works on a clean/empty DB too); `pnpm db:push` is a faster path for quick local iteration when you don't need a migration file

4. **Testing**:
   - Backend: `pytest tests/` for unit tests
   - Frontend: `pnpm test` for Playwright e2e tests
   - Manual: Run `python run_api.py` + `pnpm dev` and test in browser

5. **Adding/Updating Priority Reference Context**:
   - Add or edit a `*.json` file under `data/context/` — a flat `{"key": "value"}` map (e.g. `{"중국어 용어": "한국어 번역"}`) for glossary-style lookups, or any other JSON shape for rules/facts/etc.
   - No restart needed — `RAGService` reloads every file in `data/context/` on each query

## Known Constraints & Considerations

- **Port Assignments**: FastAPI uses 8001, Next.js uses 3000, Streamlit uses 8501
- **ChromaDB Persistence**: Vector DB stored in `data/vectors/`—deleting this resets embeddings
- **CPU/Memory**: Large PDFs or many documents can be memory-intensive; chunk size may need tuning on weaker systems
- **Concurrency**: FastAPI runs with `reload=True` in dev mode; production deployments should remove this
- **OCR Dependencies**: Scanned-PDF OCR requires the system `tesseract` binary with `chi_sim`/`chi_tra`/`kor` language packs installed, not just the Python packages in `requirements.txt`
- **Known issues, full-document translation (`_translate_pages`, unresolved)**: (1) ~7/11 pages on the sample PDF end with a persistent `_looks_duplicated` warning even after `MAX_PAGE_RETRY_ATTEMPTS` retries — root cause traced to a single untranslatable OCR-noise line (garbled text from a graphic/logo) or a page title echoed twice; the page's real content is unaffected and kept (by design), but the warning itself doesn't go away since retrying can't fix genuinely untranslatable input. (2) The last page of a multi-page translation has been observed coming back in English instead of the requested target language (e.g. Korean) while every other page was correctly translated — target-language drift on a single page, cause not yet investigated.

## Debugging Tips

- **API Connection**: Verify backend is at `http://localhost:8001` and CORS allows origin
- **Vector DB Corruption**: Delete `data/vectors/` and re-upload PDFs to rebuild
- **Frontend Database Errors**: `pnpm db:migrate` (`web-ui/lib/db/migrate.ts`) now works on a missing/empty `web-ui/data/chat.db` — it creates the data dir and applies migrations from scratch. `npx tsx web-ui/lib/db/init-db.ts` is an alternative that also works from empty
- **PyCharm/IDE**: Set Python interpreter to `venv/bin/python`; Next.js code completion works with TypeScript language service
- **Garbled/wrong Chinese OCR text**: Check that `ImageHandler.remove_watermark()` is actually stripping the watermark (dump a preprocessed page image and inspect it) before assuming the OCR model itself is at fault

