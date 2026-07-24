# 🤖 Chat with PDF locally using Ollama + LangChain

A local RAG (Retrieval Augmented Generation) application that lets you chat with your PDF documents using Ollama and LangChain — including scanned/image-based PDFs via OCR. Includes a Next.js web app (primary), a Streamlit interface, and Jupyter notebooks for experimentation. All processing happens on your machine; no data leaves your device.

[![Python Tests](https://github.com/MJ-Choi/ollama_pdf_rag/actions/workflows/tests.yml/badge.svg)](https://github.com/MJ-Choi/ollama_pdf_rag/actions/workflows/tests.yml)

## About This Project

This is a **fork** of [tonykipkemboi/ollama_pdf_rag](https://github.com/tonykipkemboi/ollama_pdf_rag) — a local RAG chat-with-PDF app by [Tony Kipkemboi](https://tonykipkemboi.com). The original project provides the core RAG pipeline, the Next.js/Streamlit UIs, and the FastAPI backend that this fork builds on.

This fork adds a full OCR and translation pipeline on top of that foundation, built for the primary use case of chatting with **scanned, image-based PDFs in Chinese** (originally: knitting patterns) and translating them into Korean — while remaining general-purpose for any scanned or native-text PDF.

## ✨ Features Added in This Fork

- 🖼️ **Scanned PDF OCR** — Automatic detection of image-based/scanned PDFs, with OCR fallback (`pytesseract` + `pdf2image`) when native text extraction fails or is unreliable
- 🧹 **Watermark Removal** — Both image-level (OpenCV Otsu binarization) and recurring text-caption removal (a watermark/caption line repeated across most pages is detected via fuzzy clustering and stripped)
- 🌐 **Full-Document & Page-Range Translation** — Ask to translate a whole scanned document, or just a page range (e.g. "translate pages 1-2 into Korean"), and get a complete, line-by-line, original-then-translation response:
  - Query-time OCR language narrowing (drops language packs that hurt CJK recognition accuracy, without weakening the default for other documents)
  - Page-by-page generation loop with automatic retry/continuation, so long documents don't silently cut off partway through
  - Structural validation after generation — catches original/translation block-separation, duplicated content, and target-language drift (e.g. a page coming back in the wrong language) — with bounded automatic retries
  - Deterministic post-processing for natural unit/counter notation in the target language (e.g. removing an unnecessary counter word after a number)
- ⚡ **Server-Side Answer Shortcuts** — Questions the system already knows the answer to (page count, filename, upload date, chunk count, or a specific page's raw content) are answered directly from stored metadata — no LLM call, instant and always correct
- 📚 **Priority Reference Context** — Drop glossaries, rules, or domain facts as JSON into `data/context/` and the model consults them before its own knowledge (e.g. a domain-specific term glossary)
- 🔄 **Collection Refresh** — Re-run OCR against a PDF's original file and refresh its stored embeddings without re-uploading — useful after an OCR/language quality fix
- 🔍 **Machine-Readable Truncation Flag** — API responses include a `truncated` boolean in `metadata`, so a client can detect an incomplete answer without parsing warning text
- 🛠️ **Reliability & Cleanup** — Long-running-query timeout fix for full-document translation, markdown line-break rendering fix, a more precise (and Korean-aware) document-context classifier on the frontend, and a maintenance script for orphaned storage

## 🖼️ Screenshots

### Next.js Interface (Recommended)
![Next.js UI](nextjs_ui.png)
*Modern chat interface with PDF management, source citations, and reasoning steps*

### Streamlit Interface
![Streamlit UI](st_app_ui.png)
*Classic Streamlit interface with PDF viewer and chat functionality*

## 🏗️ Project Structure
```
ollama_pdf_rag/
├── src/
│   ├── api/                  # FastAPI REST API
│   │   ├── routers/          # API endpoints
│   │   ├── services/         # Business logic (RAG orchestration, PDF/OCR processing)
│   │   └── main.py           # API entry point
│   ├── app/                  # Streamlit application
│   │   ├── components/       # UI components
│   │   └── main.py           # Streamlit entry point
│   └── core/                 # Core RAG + OCR functionality
│       ├── document.py       # PDF processing + scanned-PDF/OCR fallback detection
│       ├── text_extractor.py # OCR extraction for scanned PDFs (pytesseract + pdf2image)
│       ├── image_handler.py  # Image preprocessing & watermark removal (OpenCV)
│       ├── image_analysis.py # OCR wrapper & image quality analysis
│       └── embeddings.py     # Vector embeddings
├── web-ui/                   # Next.js frontend
│   ├── app/                  # Next.js app router
│   ├── components/           # React components
│   └── lib/                  # Utilities & AI integration
├── data/
│   ├── pdfs/                 # PDF storage
│   ├── vectors/              # ChromaDB storage
│   └── context/              # Priority reference context for the model (*.json: glossaries, rules, facts, ...)
├── scripts/                  # Maintenance scripts (e.g. orphaned-storage cleanup)
├── notebooks/                # Jupyter notebooks
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── run.py                    # Streamlit runner
├── run_api.py                # FastAPI runner
└── start_all.sh              # Start all services
```

## 🚀 Installation

### Prerequisites

1. **Install Ollama**
   - Visit [Ollama's website](https://ollama.ai) to download and install
   - Pull required models:
     ```bash
     ollama pull qwen3:14b  # or your preferred chat model
     ollama pull nomic-embed-text  # for embeddings
     ```

2. **Install OCR system dependencies** (needed for scanned/image-based PDFs)
   - [`tesseract`](https://github.com/tesseract-ocr/tesseract) with the language packs you need (e.g. `chi_sim`, `chi_tra`, `kor`, `eng`)
   - macOS: `brew install tesseract tesseract-lang`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra tesseract-ocr-kor`
   - `poppler` (required by `pdf2image`) — macOS: `brew install poppler`, Ubuntu/Debian: `sudo apt-get install poppler-utils`

3. **Clone this repository**
   ```bash
   git clone https://github.com/MJ-Choi/ollama_pdf_rag.git
   cd ollama_pdf_rag
   git checkout feature/image
   ```

4. **Set Up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Set Up Next.js Frontend** (for the modern UI)
   ```bash
   cd web-ui
   pnpm install
   pnpm db:migrate # If you need to reset the DB from scratch, run ./init-db.sh (⚠️ deletes data/chat.db and lib/db/migrations)
   mv .env.example .env.local # set up env. You can make this file.
   cd ..
   ```

## 💡 Usage

### Running the Application

#### Option 1: Next.js + FastAPI (Recommended)

```bash
# Terminal 1: Start the FastAPI backend
python run_api.py
# Runs on http://localhost:8001

# Terminal 2: Start the Next.js frontend
cd web-ui && pnpm dev
# Runs on http://localhost:3000
```

Or use the convenience script:
```bash
./start_all.sh
```

**Service URLs:**
| Service | URL | Description |
|---------|-----|-------------|
| Next.js Frontend | http://localhost:3000 | Modern chat interface |
| FastAPI Backend | http://localhost:8001 | REST API |
| API Documentation | http://localhost:8001/docs | Swagger UI |

#### Option 2: Streamlit Interface

```bash
python run.py
# Runs on http://localhost:8501
```

#### Option 3: Jupyter Notebook

```bash
jupyter notebook
```
Open `notebooks/experiments/updated_rag_notebook.ipynb` to experiment with the code.

### Using the Next.js Interface

1. **Upload PDFs** — Click the 📎 button or drag & drop files. Scanned/image-based PDFs are automatically detected and run through OCR
2. **View PDFs** — Uploaded PDFs appear in the sidebar with page/chunk counts; hover for a re-OCR (refresh) button
3. **Select Model** — Choose from your locally available Ollama models
4. **Ask Questions** — Type your question and get answers with source citations. Ask to translate a page range or the whole document for a full, line-by-line translation
5. **View Reasoning** — See the AI's thinking process and retrieved chunks

### Using the Streamlit Interface
1. **Upload PDF** — Use the file uploader or toggle "Use sample PDF"
2. **Select Model** — Choose from available Ollama models
3. **Ask Questions** — Chat with your PDF through the interface
4. **Adjust Display** — Use the zoom slider for PDF visibility
5. **Clean Up** — Delete collections when switching documents

### API Reference

The FastAPI backend provides these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/pdfs/upload` | Upload and process a PDF |
| `GET` | `/api/v1/pdfs` | List all uploaded PDFs |
| `DELETE` | `/api/v1/pdfs/{pdf_id}` | Delete a PDF |
| `POST` | `/api/v1/pdfs/{pdf_id}/refresh-ocr` | Re-OCR a PDF and refresh its stored embeddings |
| `POST` | `/api/v1/query` | Query PDFs with RAG |
| `GET` | `/api/v1/models` | List available Ollama models |
| `GET` | `/api/v1/health` | Health check |

See full documentation at http://localhost:8001/docs when running.

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src
```

```bash
# Pre-commit hooks (runs pytest + pylint)
pip install pre-commit
pre-commit install
```

### Maintenance

```bash
# Report (dry-run) orphaned ChromaDB collections and unused DB tables
python scripts/cleanup_orphans.py

# Actually delete/drop what was reported
python scripts/cleanup_orphans.py --apply
```

## ⚠️ Troubleshooting

- **Ollama not responding**: Ensure Ollama is running (`ollama serve`)
- **Model not found**: Pull models with `ollama pull <model-name>`
- **No chunks retrieved**: Re-upload PDFs to rebuild the vector database
- **Port conflicts**: Check if ports 3000, 8001, or 8501 are in use
- **Garbled/wrong OCR text on a scanned PDF**: Confirm the `tesseract` language packs you need are installed, and try the re-OCR (refresh) button on that PDF after fixing the OCR setup

### Common Errors

#### ONNX DLL Error (Windows)
```
DLL load failed while importing onnx_copy2py_export
```
Install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) and restart.

#### CPU-Only Systems
Reduce chunk size if experiencing memory issues:
- Modify `chunk_size` to 500-1000 in `src/core/document.py`

#### Full-document translation: occasional per-page warnings
On a full multi-page translation, the reasoning steps may show a page retrying due to a duplicated-content check and still ending with a "format issue remains" note — this means one page had content the model couldn't usefully translate (e.g. OCR noise from a graphic/logo) and repeated it instead; the page's real content is kept and unaffected. Check the `truncated` field in the API response's `metadata` to detect an incomplete answer programmatically.

## 🤝 Contributing

- Open issues for bugs or suggestions
- Submit pull requests
- ⭐ Star the repository if you find it useful!

## 📝 License & Credits

This project is open source under the [MIT License](LICENSE).

It is a fork of [**ollama_pdf_rag**](https://github.com/tonykipkemboi/ollama_pdf_rag) by [**Tony Kipkemboi**](https://tonykipkemboi.com) — all credit for the original RAG pipeline, UI foundations, and project structure goes to the original author. Please check out the [original repository](https://github.com/tonykipkemboi/ollama_pdf_rag) and consider starring it too.

Follow the original author on [X](https://x.com/tonykipkemboi) | [LinkedIn](https://www.linkedin.com/in/tonykipkemboi/) | [YouTube](https://www.youtube.com/@tonykipkemboi) | [GitHub](https://github.com/tonykipkemboi)
