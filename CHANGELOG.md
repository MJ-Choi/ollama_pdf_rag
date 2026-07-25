# Changelog

All notable changes to this project will be documented in this file.

## [v3.1.0] - 2026-07-25

### 🚀 Major Features

#### Scanned/Image-Based PDF Support
- **Automatic OCR Fallback** - Scanned/image-based PDFs are detected automatically and OCR'd via a local vision-LLM (`deepseek-ocr:3b` through Ollama) — no manual pre-processing or external OCR tool needed
- **Watermark & Artifact Removal** - Recurring watermark/caption lines and vision-LLM formatting artifacts (markdown, embedded image data) are stripped before chunking
- **Query-Time Re-OCR & Collection Refresh** - Re-run OCR against a PDF's original file on demand, either scoped to a single query or to permanently refresh a PDF's stored embeddings

#### Full-Document & Page-Range Translation
- **Page-by-Page Generation** - Long documents are translated one page at a time with automatic retry/continuation, so responses don't silently cut off partway through
- **Page-Range Queries** - Ask about or translate just a page range (e.g. "translate pages 1-2") instead of the whole document
- **Structural Validation** - Each translated page is checked for duplicated content, untranslated/wrong-language output, and correct original/translation interleaving, with bounded automatic retries; unresolved issues are marked inline in the answer rather than silently returned as-is
- **Priority Reference Context** - Drop glossaries or domain facts as JSON into `data/context/` and the model consults them before its own knowledge
- **Machine-Readable Truncation Flag** - API responses include a `truncated` boolean so a client can detect an incomplete answer without parsing warning text

#### Server-Side Answer Shortcuts
- Questions the system already knows the answer to (page count, filename, upload date, chunk count, or a page's raw content) are answered directly from stored metadata — no LLM call needed

### 🔧 Infrastructure
- Backend session storage (`ChatSession`/`ChatMessage`) removed — the frontend's own chat database is the sole source of chat history
- `scripts/cleanup_orphans.py` for orphaned ChromaDB collections and unused DB tables
- pre-commit now runs the full pytest suite (previously only ran a small `unittest`-discoverable subset)

## [v3.0.0] - 2025-12-19

### 🚀 Major Features

#### Next.js Frontend
- **Modern Chat UI** - Beautiful, responsive chat interface built with Next.js 16 and React 19
- **Real-time Streaming** - Live message streaming with Vercel AI SDK
- **Chat Persistence** - SQLite database for saving all conversations
- **PDF Selection** - Checkbox-based PDF selection before chatting
- **Question Classification** - Auto-detects if questions need document context
- **Model Selector** - Switch between Ollama models on the fly
- **Theme Support** - Dark/light mode toggle

#### FastAPI Backend
- **REST API** - Production-ready API with OpenAPI docs
- **PDF Management** - Upload, list, delete PDF endpoints
- **RAG Query** - Multi-PDF query with source attribution
- **Session Management** - Chat history per session
- **Health Monitoring** - Ollama connection status

#### Enhanced RAG
- **Multi-Query Retrieval** - Generates alternative queries for better recall
- **Thinking Models** - Support for qwen3, deepseek-r1 chain-of-thought
- **PDF Filtering** - Query specific PDFs, not all
- **Source Citations** - Every answer includes chunk sources

### 📖 Documentation
- **MkDocs Site** - Comprehensive documentation with Material theme
- **API Reference** - Full REST API documentation
- **Architecture Diagrams** - Visual pipeline explanations
- **Screenshots** - UI screenshots in docs

### 🔧 Infrastructure
- **GitHub Actions** - Updated CI for Python 3.10-3.12
- **Security Updates** - Patched Dependabot vulnerabilities
- **Improved Tests** - Mocked tests without Ollama dependency

### 💥 Breaking Changes
- Python 3.9 no longer supported (requires 3.10+)
- New FastAPI backend required on port 8001
- Database schema changes for chat persistence

---

## [v2.1.0] - 2024-01-07

### Added
- Comprehensive test suite with pytest
- GitHub Actions CI pipeline
- Pre-commit hooks for code quality
- Test coverage reporting
- Project restructuring with clean architecture
- New directory structure for better organization
- Sample PDFs in dedicated folder

### Changed
- Moved all source code to src/ directory
- Updated dependencies to latest compatible versions
- Improved README with testing documentation
- Added test status badge
- Reorganized PDF storage structure

### Fixed
- Dependency conflicts with pydantic
- ONNX runtime compatibility issues
- Test coverage configuration

## [v2.0] - 2023-11-05
- Major version release
- Improved RAG implementation
- Enhanced PDF processing

## [v1.0] - 2023-07-08
- Initial release
- Basic RAG functionality
- PDF processing capabilities
- Streamlit interface
- Jupyter notebook for experimentation
