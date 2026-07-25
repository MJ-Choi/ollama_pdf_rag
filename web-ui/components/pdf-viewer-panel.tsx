"use client";

import { ChevronLeftIcon, ChevronRightIcon, XIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import { Button } from "@/components/ui/button";
import { usePdfViewer } from "@/hooks/use-pdf-viewer";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export function PdfViewerPanel() {
  const { state, closePdf, setPage } = usePdfViewer();
  const [numPages, setNumPages] = useState<number | null>(null);

  // Reset the known page count whenever a different PDF is opened, so a
  // stale count from the previous document doesn't briefly show.
  useEffect(() => {
    setNumPages(null);
  }, [state.pdfId]);

  if (!state.isVisible || !state.pdfId) {
    return null;
  }

  const fileUrl = `${API_URL}/api/v1/pdfs/${state.pdfId}/file`;

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l bg-background shadow-2xl md:max-w-lg">
      <div className="flex items-center justify-between gap-2 border-b p-3">
        <span className="truncate font-medium text-sm" title={state.pdfName ?? ""}>
          {state.pdfName}
        </span>
        <Button onClick={closePdf} size="icon" variant="ghost">
          <XIcon className="size-4" />
        </Button>
      </div>

      <div className="flex flex-1 items-start justify-center overflow-auto bg-muted/30 p-4">
        <Document
          file={fileUrl}
          loading={
            <div className="pt-10 text-muted-foreground text-sm">
              Loading PDF...
            </div>
          }
          onLoadSuccess={({ numPages: loadedPages }) => setNumPages(loadedPages)}
          error={
            <div className="pt-10 text-destructive text-sm">
              Failed to load this PDF.
            </div>
          }
        >
          <Page
            pageNumber={state.page}
            renderAnnotationLayer={false}
            renderTextLayer={false}
            width={440}
          />
        </Document>
      </div>

      <div className="flex items-center justify-center gap-3 border-t p-3">
        <Button
          disabled={state.page <= 1}
          onClick={() => setPage(state.page - 1)}
          size="icon"
          variant="outline"
        >
          <ChevronLeftIcon className="size-4" />
        </Button>
        <span className="text-muted-foreground text-sm">
          Page {state.page}
          {numPages ? ` / ${numPages}` : ""}
        </span>
        <Button
          disabled={!!numPages && state.page >= numPages}
          onClick={() => setPage(state.page + 1)}
          size="icon"
          variant="outline"
        >
          <ChevronRightIcon className="size-4" />
        </Button>
      </div>
    </div>
  );
}
