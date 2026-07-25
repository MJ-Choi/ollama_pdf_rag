"use client";

import { useCallback, useMemo } from "react";
import useSWR from "swr";

export type PdfViewerState = {
  isVisible: boolean;
  pdfId: string | null;
  pdfName: string | null;
  page: number;
};

export const initialPdfViewerState: PdfViewerState = {
  isVisible: false,
  pdfId: null,
  pdfName: null,
  page: 1,
};

/**
 * Global PDF-viewer panel state, following the same SWR-as-a-store pattern
 * as `useArtifact` — lets a source citation deep in the message list open
 * the viewer to a specific page without prop-drilling through Chat/Messages.
 */
export function usePdfViewer() {
  const { data: localState, mutate: setLocalState } = useSWR<PdfViewerState>(
    "pdf-viewer",
    null,
    { fallbackData: initialPdfViewerState }
  );

  const state = useMemo(
    () => localState ?? initialPdfViewerState,
    [localState]
  );

  const openPdf = useCallback(
    (pdfId: string, pdfName: string, page = 1) => {
      setLocalState({ isVisible: true, pdfId, pdfName, page });
    },
    [setLocalState]
  );

  const closePdf = useCallback(() => {
    setLocalState((current) => ({
      ...(current ?? initialPdfViewerState),
      isVisible: false,
    }));
  }, [setLocalState]);

  const setPage = useCallback(
    (page: number) => {
      setLocalState((current) => ({
        ...(current ?? initialPdfViewerState),
        page,
      }));
    },
    [setLocalState]
  );

  return { state, openPdf, closePdf, setPage };
}
