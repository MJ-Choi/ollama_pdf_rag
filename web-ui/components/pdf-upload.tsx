"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload, CheckCircle2 } from "lucide-react";

interface PDFUploadProps {
  onUploadComplete?: () => void;
}

// XMLHttpRequest, not fetch — fetch has no cross-browser way to observe
// upload byte progress (no equivalent of XHR's upload.onprogress), which
// is what the circular indicator below needs.
function uploadWithProgress(
  file: File,
  onProgress: (percent: number) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        reject(new Error(`Upload failed: ${xhr.statusText}`));
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Upload failed")));

    xhr.open("POST", "http://localhost:8001/api/v1/pdfs/upload");
    xhr.send(formData);
  });
}

// Byte-upload progress only covers sending the file — the backend's
// OCR/chunking/embedding work after that has no progress signal at all, so
// reaching 100% here means "upload done, server is now processing," not
// "finished." Callers show a distinct "processing" state once this hits 100.
function CircularProgress({ progress }: { progress: number }) {
  const size = 18;
  const strokeWidth = 2.5;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (Math.min(progress, 100) / 100) * circumference;

  return (
    <svg
      className="-rotate-90 shrink-0"
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      width={size}
    >
      <title>{`${progress}% uploaded`}</title>
      <circle
        cx={size / 2}
        cy={size / 2}
        fill="none"
        opacity={0.25}
        r={radius}
        stroke="currentColor"
        strokeWidth={strokeWidth}
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        fill="none"
        r={radius}
        stroke="currentColor"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        strokeWidth={strokeWidth}
        style={{ transition: "stroke-dashoffset 150ms linear" }}
      />
    </svg>
  );
}

export function PDFUpload({ onUploadComplete }: PDFUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadCount, setUploadCount] = useState(0);
  // Only tracked/shown for a single-file upload — with multiple files the
  // existing "(N completed)" counter already communicates progress, and a
  // single ring can't represent several files' progress at once.
  const [singleFileProgress, setSingleFileProgress] = useState<number | null>(
    null
  );

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileList = Array.from(files);
    const isSingleFile = fileList.length === 1;

    setUploading(true);
    setUploadCount(0);
    setSingleFileProgress(isSingleFile ? 0 : null);
    let successCount = 0;

    for (const file of fileList) {
      try {
        if (isSingleFile) {
          await uploadWithProgress(file, setSingleFileProgress);
        } else {
          const formData = new FormData();
          formData.append("file", file);
          const response = await fetch(
            "http://localhost:8001/api/v1/pdfs/upload",
            { method: "POST", body: formData }
          );
          if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
          }
        }
        successCount++;
        setUploadCount(successCount);
      } catch (error) {
        console.error("Upload error:", error);
      }
    }

    setUploading(false);
    setSingleFileProgress(null);

    // Call the callback after all uploads complete
    if (onUploadComplete) {
      onUploadComplete();
    }

    // Reset the input
    e.target.value = "";
  };

  const buttonLabel = () => {
    if (!uploading) return "Upload PDFs";
    if (singleFileProgress === null) return `Uploading... (${uploadCount} completed)`;
    if (singleFileProgress < 100) return `Uploading... ${singleFileProgress}%`;
    return "Processing...";
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <input
          type="file"
          accept=".pdf"
          multiple
          onChange={handleUpload}
          className="hidden"
          id="pdf-upload"
        />
        <label htmlFor="pdf-upload" className="w-full">
          <Button asChild disabled={uploading} className="w-full">
            <span className="flex items-center justify-center gap-2">
              {uploading && singleFileProgress !== null ? (
                <CircularProgress progress={singleFileProgress} />
              ) : (
                <Upload className="h-4 w-4" />
              )}
              {buttonLabel()}
            </span>
          </Button>
        </label>
      </div>
      {uploadCount > 0 && !uploading && (
        <div className="flex items-center gap-2 text-sm text-green-500">
          <CheckCircle2 className="h-4 w-4" />
          <span>{uploadCount} PDF(s) uploaded successfully</span>
        </div>
      )}
    </div>
  );
}
