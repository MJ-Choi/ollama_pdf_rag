# PDF Processing

This guide explains how Ollama PDF RAG processes PDF documents for retrieval.

## Processing Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   PDF File   │───▶│    Load &    │───▶│    Split     │───▶│   Generate   │
│   Upload     │    │    Parse     │    │    Chunks    │    │  Embeddings  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Ready to   │◀───│    Save      │◀───│    Store     │◀───│    Add       │
│    Query     │    │   Metadata   │    │   Vectors    │    │   Metadata   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## Step 1: File Upload

### API Endpoint

```bash
POST http://localhost:8001/api/v1/pdfs/upload
Content-Type: multipart/form-data

file: <PDF file>
```

### Storage Location

```
data/pdfs/uploads/
└── pdf_{hash}_{original_name}.pdf
```

Each uploaded PDF gets a unique ID based on filename + timestamp hash.

## Step 2: Document Loading

We first try LangChain's `UnstructuredPDFLoader` to extract native text:

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(file_path, strategy="fast")
documents = loader.load()
```

### Scanned/Image-Based PDF Fallback (OCR)

If the extracted text is too short, or looks garbled (broken CJK characters,
mostly single-character "words" — a sign the PDF is actually a scanned
image), the app automatically falls back to OCR instead of returning junk
text. This is fully automatic — no external tool or manual pre-processing
needed:

1. Each page is rasterized to an image (`pdf2image`, 300 DPI)
2. Each page image is sent to a local vision-LLM, `deepseek-ocr:3b`, running
   through Ollama — no OCR language configuration needed, the model reads
   whatever script is on the page
3. The model's raw output is cleaned up (stray markdown formatting and any
   embedded image data are stripped) and a repeated watermark/caption line
   (if one appears on most pages) is detected and removed
4. Each OCR'd page becomes its own chunk (see `source_page` in the metadata
   below), instead of being re-split by character count

### What Gets Extracted

| Element | Handled |
|---------|---------|
| Text content | ✅ |
| Headers/titles | ✅ |
| Lists | ✅ |
| Tables (basic) | ✅ |
| Scanned/image-based text | ✅ (automatic OCR fallback, see above) |
| Text embedded in photos on a page | ✅ (vision-LLM reads it directly) |

### Document Object

```python
Document(
    page_content="The extracted text from the PDF...",
    metadata={
        "source": "/path/to/file.pdf",
        "page": 1
    }
)
```

## Step 3: Text Chunking

Large documents are split into smaller chunks for efficient retrieval:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=7500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)
```

### Chunking Strategy

```
Original Document (20,000 chars)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   Chunk 1 (7500)                    │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│                                        ◄──overlap──►│
└─────────────────────────────────────────────────────┘
                                 ┌─────────────────────────────────────────────────────┐
                                 │                   Chunk 2 (7500)                    │
                                 │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
                                 │                                        ◄──overlap──►│
                                 └─────────────────────────────────────────────────────┘
                                                                  ┌─────────────────────┐
                                                                  │   Chunk 3 (5000)    │
                                                                  │ ░░░░░░░░░░░░░░░░░░░ │
                                                                  └─────────────────────┘
```

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunk_size` | 7500 | Maximum characters per chunk |
| `chunk_overlap` | 100 | Characters shared between chunks |
| `separators` | `["\n\n", "\n", " ", ""]` | Split priority |

### Why These Settings?

- **7500 chars**: Large enough for context, small enough for precise retrieval
- **100 overlap**: Preserves context at boundaries
- **Recursive splitting**: Respects document structure (paragraphs > lines > words)

## Step 4: Metadata Enhancement

Each chunk gets enriched metadata:

```python
for i, chunk in enumerate(chunks):
    chunk.metadata.update({
        "pdf_id": "pdf_123456",
        "pdf_name": "Security_Guide.pdf",
        "chunk_index": i,
        "source_file": "Security_Guide.pdf"
    })
```

### Metadata Schema

```python
{
    "source": "/path/to/file.pdf",      # Original file path
    "page": 5,                          # Page number (if available)
    "pdf_id": "pdf_123456",             # Unique PDF identifier
    "pdf_name": "Security_Guide.pdf",   # Display name
    "chunk_index": 3,                   # Chunk sequence number
    "source_file": "Security_Guide.pdf" # Original filename
}
```

## Step 5: Embedding Generation

Chunks are converted to vectors using Ollama's embedding model:

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### Embedding Details

| Property | Value |
|----------|-------|
| Model | nomic-embed-text |
| Dimensions | 768 |
| Type | Float32 |

### Process

```
"The security policy requires..."  ───▶  [0.023, -0.156, 0.892, ...]
                                              768 dimensions
```

## Step 6: Vector Storage

Embeddings are stored in ChromaDB:

```python
from langchain_chroma import Chroma

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=f"pdf_{hash}",
    persist_directory="data/vectors"
)
```

### Collection Structure

```
data/vectors/
├── chroma.sqlite3              # Metadata database
└── collections/
    ├── pdf_123456/             # Collection per PDF
    │   ├── data/
    │   └── metadata/
    └── pdf_789012/
        ├── data/
        └── metadata/
```

Each PDF gets its own collection, enabling:
- Selective querying (specific PDFs)
- Easy deletion
- Isolation between documents

## Step 7: Metadata Persistence

PDF metadata is saved to SQLite:

```python
pdf_metadata = PDFMetadata(
    pdf_id="pdf_123456",
    name="Security_Guide.pdf",
    collection_name="pdf_123456",
    upload_timestamp=datetime.now(),
    doc_count=15,           # Number of chunks
    page_count=12,          # Original pages
    is_sample=False,
    file_path="/data/pdfs/uploads/pdf_123456_Security_Guide.pdf"
)
db.add(pdf_metadata)
db.commit()
```

### Database Schema

```sql
CREATE TABLE pdf_metadata (
    id INTEGER PRIMARY KEY,
    pdf_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    upload_timestamp DATETIME,
    doc_count INTEGER,
    page_count INTEGER,
    is_sample BOOLEAN DEFAULT FALSE,
    file_path TEXT
);
```

## API Endpoints

### Upload PDF

```bash
POST /api/v1/pdfs/upload
Content-Type: multipart/form-data

Response:
{
  "pdf_id": "pdf_123456",
  "name": "Security_Guide.pdf",
  "collection_name": "pdf_123456",
  "doc_count": 15,
  "page_count": 12,
  "upload_timestamp": "2024-12-19T18:00:00Z"
}
```

### List PDFs

```bash
GET /api/v1/pdfs

Response:
[
  {
    "pdf_id": "pdf_123456",
    "name": "Security_Guide.pdf",
    "collection_name": "pdf_123456",
    "upload_timestamp": "2024-12-19T18:00:00Z",
    "doc_count": 15,
    "page_count": 12,
    "is_sample": false
  }
]
```

### Delete PDF

```bash
DELETE /api/v1/pdfs/{pdf_id}

Response:
{
  "message": "PDF deleted successfully"
}
```

Deletion removes:
1. PDF file from disk
2. Vector collection from ChromaDB
3. Metadata from SQLite

## Processing Statistics

For a typical document:

| Document | Pages | Size | Chunks | Processing Time |
|----------|-------|------|--------|-----------------|
| Small PDF | 5 | 200KB | 3-5 | ~5 seconds |
| Medium PDF | 20 | 1MB | 10-20 | ~15 seconds |
| Large PDF | 100 | 10MB | 50-100 | ~60 seconds |

## Troubleshooting

### "Failed to load PDF"

- Check file is valid PDF (not corrupted)
- Verify file permissions
- If the OCR fallback triggered, confirm `ollama pull deepseek-ocr:3b` completed (see below)

### "No chunks created"

- PDF might be empty or image-only
- Check if text extraction worked
- Try a different PDF

### "Embedding failed"

- Verify Ollama is running
- Check `nomic-embed-text` is pulled
- Look for memory issues

### "ChromaDB error"

- Check disk space for vectors
- Verify write permissions on `data/vectors`
- Try deleting and re-uploading

## Best Practices

### Optimal PDF Characteristics

- ✅ Well-structured (headings, paragraphs)
- ✅ Under 100 pages (for speed)
- ✅ Clear language (not heavily formatted)
- ℹ️ Scanned/image-based PDFs work too (automatic OCR fallback) — native-text PDFs just skip the OCR step and process faster

### Pre-processing Tips

1. **Split large PDFs** - Break into chapters/sections
2. **Remove images** - If not needed for context
3. **Clean formatting** - Remove excessive headers/footers
