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
- `src/core/document.py`: PDF text extraction and chunking (2000-token chunks with 200-token overlap)
- `src/core/text_extractor.py`: OCR extraction for scanned/image-based PDFs via `pytesseract` + `pdf2image` (optional deps, degrades gracefully if missing)
- `src/core/translator.py`: Language detection (`langdetect`) and translation between document languages (optional dep)

**API Layer** (`src/api/`):
- `main.py`: FastAPI app setup with CORS middleware (allows localhost:3000 for Next.js); registers `pdfs`, `query`, `models`, `health` routers
- `database.py`: SQLite with SQLAlchemy models (PDFMetadata, ChatSession, ChatMessage)
- `routers/`: pdfs (upload/delete), query (RAG), models, health (status)
- `services/`: Business logic — `pdf_service.py`, `rag_service.py`
- `config.py`: Settings for Ollama connection (default: http://localhost:11434)

**Key Flow**:
1. User uploads PDF → extracted and chunked → AI → stored in ChromaDB
2. Query arrives → MultiQueryRetriever generates variations → retrieves relevant chunks → LLM synthesizes response with sources


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

# Production build (runs `tsx lib/db/migrate` first — currently broken on a clean DB, see Known Constraints)
pnpm build

# Lint / format (Ultracite/Biome, see below)
pnpm lint
pnpm format

# Drizzle schema management
pnpm db:generate   # create a migration file
pnpm db:push       # push schema directly (preferred for local dev, see Known Constraints)
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
- Chunks are 2000 tokens with 200-token overlap (tuned for balanced context)
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

## Important Files & Their Purposes

**Backend**:
- `src/core/rag.py` — RAG pipeline orchestration
- `src/api/routers/query.py` — Query endpoint that drives RAG
- `src/api/routers/pdfs.py` — PDF upload/deletion/listing
- `src/api/database.py` — SQLAlchemy models for metadata

**Config**:
- `requirements.txt` — Python dependencies (LangChain 1.0.0, ChromaDB, FastAPI, Streamlit)
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
   - Run `pnpm db:generate` to create a migration file, but note `pnpm db:migrate` currently fails from a clean/empty DB (see Known Constraints below) — for local dev, apply schema changes by hand-editing `web-ui/lib/db/init-db.ts` and re-running it, or use `pnpm db:push` instead

4. **Testing**:
   - Backend: `pytest tests/` for unit tests
   - Frontend: `pnpm test` for Playwright e2e tests
   - Manual: Run `python run_api.py` + `pnpm dev` and test in browser

## Known Constraints & Considerations

- **Port Assignments**: FastAPI uses 8001, Next.js uses 3000, Streamlit uses 8501
- **ChromaDB Persistence**: Vector DB stored in `data/vectors/`—deleting this resets embeddings
- **CPU/Memory**: Large PDFs or many documents can be memory-intensive; chunk size may need tuning on weaker systems
- **Concurrency**: FastAPI runs with `reload=True` in dev mode; production deployments should remove this

## Debugging Tips

- **API Connection**: Verify backend is at `http://localhost:8001` and CORS allows origin
- **Vector DB Corruption**: Delete `data/vectors/` and re-upload PDFs to rebuild
- **Frontend Database Errors**: Do NOT run `pnpm db:migrate` on a missing/empty `web-ui/data/chat.db` — it fails (see Known Constraints). Run `npx tsx web-ui/lib/db/init-db.ts` instead, or use the `/reset-webui-db` skill
- **PyCharm/IDE**: Set Python interpreter to `venv/bin/python`; Next.js code completion works with TypeScript language service

