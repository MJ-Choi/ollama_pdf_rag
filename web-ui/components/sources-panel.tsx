"use client";

import { FileText } from "lucide-react";
import { usePdfViewer } from "@/hooks/use-pdf-viewer";

interface Source {
  pdf_name: string;
  pdf_id: string;
  chunk_index: number;
  source_page?: number | null;
}

interface SourcesPanelProps {
  sources: Source[];
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const { openPdf } = usePdfViewer();

  if (!sources || sources.length === 0) return null;

  // One citation per (pdf, page) — a page cited by multiple chunks only
  // needs one clickable entry, and chunks with no page info (native-text
  // PDFs) collapse into a single "view PDF" entry per document.
  const seen = new Set<string>();
  const citations = sources.filter((source) => {
    const key = `${source.pdf_id}:${source.source_page ?? "doc"}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-sm">
      <div className="flex items-center gap-1 text-muted-foreground">
        <FileText className="h-3.5 w-3.5" />
        <span>Sources:</span>
      </div>
      {citations.map((source) => (
        <button
          className="rounded-full border px-2 py-0.5 text-xs transition-colors hover:bg-muted"
          key={`${source.pdf_id}-${source.source_page ?? source.chunk_index}`}
          onClick={() =>
            openPdf(source.pdf_id, source.pdf_name, source.source_page ?? 1)
          }
          type="button"
        >
          {source.pdf_name}
          {source.source_page ? ` p.${source.source_page}` : ""}
        </button>
      ))}
    </div>
  );
}
