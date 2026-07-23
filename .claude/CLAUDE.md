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

## Debugging Tips

- **API Connection**: Verify backend is at `http://localhost:8001` and CORS allows origin
- **Vector DB Corruption**: Delete `data/vectors/` and re-upload PDFs to rebuild
- **Frontend Database Errors**: `pnpm db:migrate` (`web-ui/lib/db/migrate.ts`) now works on a missing/empty `web-ui/data/chat.db` — it creates the data dir and applies migrations from scratch. `npx tsx web-ui/lib/db/init-db.ts` is an alternative that also works from empty
- **PyCharm/IDE**: Set Python interpreter to `venv/bin/python`; Next.js code completion works with TypeScript language service
- **Garbled/wrong Chinese OCR text**: Check that `ImageHandler.remove_watermark()` is actually stripping the watermark (dump a preprocessed page image and inspect it) before assuming the OCR model itself is at fault

