/**
 * Custom Ollama provider integration for FastAPI backend
 */
import { Agent } from "undici";

// A full-document translation walks the PDF page by page (one Ollama call
// per page, see RAGService._translate_pages on the backend) and can
// legitimately take many minutes. Node's fetch (undici) defaults to a
// 5-minute headersTimeout/bodyTimeout, which was firing as
// UND_ERR_HEADERS_TIMEOUT on long-running-but-otherwise-healthy queries.
// This dispatcher is scoped to just the /api/v1/query call below — other
// endpoints (upload/list/delete) keep the default timeout.
const QUERY_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
const queryDispatcher = new Agent({
  headersTimeout: QUERY_TIMEOUT_MS,
  bodyTimeout: QUERY_TIMEOUT_MS,
});

interface Message {
  role: string;
  content: string;
}

interface Source {
  pdf_name: string;
  pdf_id: string;
  chunk_index: number;
}

interface QueryResponse {
  answer: string;
  sources: Source[];
  metadata: {
    model_used: string;
    chunks_retrieved: number;
    pdfs_queried: number;
    reasoning_steps?: string[];
  };
}

export async function ollamaChat(
  messages: Message[],
  model: string = "mistral:latest",
  pdfIds?: string[]
): Promise<QueryResponse> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  // Extract last user message
  const lastMessage = messages[messages.length - 1];
  const question = lastMessage.content;

  // Query backend with optional PDF filter. Uses queryDispatcher (see
  // above) since a full-document translation can take far longer than
  // fetch's default headers/body timeout.
  const response = await fetch(`${API_URL}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      model,
      pdf_ids: pdfIds, // Filter to specific PDFs if provided
    }),
    // @ts-expect-error -- `dispatcher` is a Node/undici fetch extension not in the DOM lib types
    dispatcher: queryDispatcher,
  });

  if (!response.ok) {
    // Try to get error details from response
    let errorDetail = response.statusText;
    try {
      const errorData = await response.json();
      errorDetail = errorData.detail || errorDetail;
    } catch {
      // If parsing fails, use statusText
    }

    throw new Error(errorDetail);
  }

  const data: QueryResponse = await response.json();

  return data;
}

export async function uploadPDF(file: File): Promise<any> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/api/v1/pdfs/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return response.json();
}

export async function listPDFs(): Promise<any[]> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  const response = await fetch(`${API_URL}/api/v1/pdfs`);

  if (!response.ok) {
    throw new Error(`Failed to list PDFs: ${response.statusText}`);
  }

  return response.json();
}

export async function deletePDF(pdfId: string): Promise<void> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  const response = await fetch(`${API_URL}/api/v1/pdfs/${pdfId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error(`Delete failed: ${response.statusText}`);
  }
}

// Re-runs OCR against a PDF's original file and replaces its stored
// ChromaDB collection with the fresh result — for PDFs uploaded before an
// OCR quality/language fix, whose embedded text is otherwise stale forever
// (upload-time OCR is a one-time snapshot; general, non-translation queries
// read straight from the stored collection).
export async function refreshPdfOcr(pdfId: string): Promise<any> {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

  const response = await fetch(`${API_URL}/api/v1/pdfs/${pdfId}/refresh-ocr`, {
    method: "POST",
  });

  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const errorData = await response.json();
      errorDetail = errorData.detail || errorDetail;
    } catch {
      // If parsing fails, use statusText
    }
    throw new Error(errorDetail);
  }

  return response.json();
}
