"""PDF processing service."""
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session

from ...core.document import DocumentProcessor
from ...core.embeddings import VectorStore
from ...core.text_extractor import TextExtractor, CJK_OCR_LANGUAGE
from ..database import PDFMetadata
from ..config import settings


class PDFService:
    """Service for PDF operations."""

    def __init__(self):
        """Initialize PDF service."""
        self.doc_processor = DocumentProcessor(chunk_size=7500, chunk_overlap=100)
        self.vector_store = VectorStore(
            embedding_model="nomic-embed-text",
            persist_directory=settings.VECTOR_DB_DIR
        )
        self.storage_dir = Path(settings.PDF_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def upload_and_process(
        self,
        file: UploadFile,
        db: Session
    ) -> PDFMetadata:
        """Upload and process a PDF file.

        Args:
            file: Uploaded PDF file
            db: Database session

        Returns:
            PDFMetadata: Metadata for the processed PDF
        """
        # Generate unique ID
        pdf_id = self._generate_pdf_id(file.filename)

        # Save file
        file_path = self.storage_dir / f"{pdf_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process PDF (falls back to OCR internally for scanned/image-based PDFs)
        documents, used_ocr = self.doc_processor.load_pdf(file_path)
        if used_ocr:
            # OCR path already returns page-chunked documents; skip re-chunking.
            chunks = documents
            page_numbers = {
                chunk.metadata.get("source_page")
                for chunk in chunks
                if chunk.metadata.get("source_page") is not None
            }
            page_count = len(page_numbers) if page_numbers else len(chunks)
        else:
            chunks = self.doc_processor.split_documents(documents)
            page_count = len(documents)

        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "pdf_id": pdf_id,
                "pdf_name": file.filename,
                "chunk_index": i,
                "source_file": file.filename
            })

        # Create vector DB collection
        collection_name = f"pdf_{abs(hash(file.filename + pdf_id))}"
        self.vector_store.create_vector_db(
            documents=chunks,
            collection_name=collection_name
        )

        # Store metadata in database
        pdf_metadata = PDFMetadata(
            pdf_id=pdf_id,
            name=file.filename,
            collection_name=collection_name,
            upload_timestamp=datetime.now(),
            doc_count=len(chunks),
            page_count=page_count,
            is_sample=False,
            file_path=str(file_path)
        )
        db.add(pdf_metadata)
        db.commit()
        db.refresh(pdf_metadata)

        return pdf_metadata

    def list_pdfs(self, db: Session) -> List[PDFMetadata]:
        """List all PDFs.

        Args:
            db: Database session

        Returns:
            List of PDF metadata
        """
        return db.query(PDFMetadata).all()

    def get_pdf(self, pdf_id: str, db: Session) -> Optional[PDFMetadata]:
        """Get single PDF metadata.

        Args:
            pdf_id: PDF identifier
            db: Database session

        Returns:
            PDF metadata or None
        """
        return db.query(PDFMetadata).filter(PDFMetadata.pdf_id == pdf_id).first()

    def delete_pdf(self, pdf_id: str, db: Session) -> bool:
        """Delete PDF and its collection.

        Args:
            pdf_id: PDF identifier
            db: Database session

        Returns:
            True if deleted successfully, False otherwise
        """
        pdf = self.get_pdf(pdf_id, db)
        if not pdf:
            return False

        self.vector_store.delete_collection_by_name(pdf.collection_name)

        # Delete file if it exists
        if pdf.file_path and os.path.exists(pdf.file_path):
            os.remove(pdf.file_path)

        # Delete metadata from database
        db.delete(pdf)
        db.commit()

        return True

    def refresh_ocr(
        self,
        pdf_id: str,
        db: Session,
        ocr_language: Optional[str] = None
    ) -> PDFMetadata:
        """Re-OCR a PDF's original file and replace its stored ChromaDB
        collection with the fresh result.

        Upload-time OCR is a one-time snapshot: a PDF uploaded before an OCR
        language/quality fix (e.g. narrowing away from the "eng"-mixed
        default, see CJK_OCR_LANGUAGE) keeps stale/degraded text embedded
        forever, since general (non-translation) queries read straight from
        the stored collection rather than re-OCR'ing. This re-runs OCR
        against the same original file and swaps the collection's contents —
        `pdf_id` and `collection_name` are preserved, only doc_count/
        page_count and the embedded text change.

        Args:
            pdf_id: PDF identifier
            db: Database session
            ocr_language: Tesseract language string to OCR with (defaults to
                CJK_OCR_LANGUAGE — this app's primary use case is CJK scans)

        Returns:
            Updated PDF metadata

        Raises:
            LookupError: no PDF with this pdf_id
            ValueError: PDF wasn't originally OCR'd (nothing to refresh), its
                original file is missing from disk, or re-OCR yields no text
        """
        pdf = self.get_pdf(pdf_id, db)
        if not pdf:
            raise LookupError(f"PDF not found: {pdf_id}")
        # doc_count == page_count is how the OCR path stores documents (one
        # chunk per page, see upload_and_process above) — a native-text PDF
        # chunked at 7500 chars/page almost never matches its page count.
        if pdf.doc_count != pdf.page_count:
            raise ValueError("This PDF wasn't OCR'd at upload — nothing to refresh")
        if not pdf.file_path or not os.path.exists(pdf.file_path):
            raise ValueError(f"Original file not found on disk: {pdf.file_path}")

        extractor = TextExtractor()
        extraction_result = extractor.extract_text_from_scanned_pdf(
            Path(pdf.file_path), ocr_language=ocr_language or CJK_OCR_LANGUAGE
        )
        chunks = extractor.create_document_chunks(extraction_result)
        if not chunks:
            raise ValueError("Re-OCR produced no text")

        page_numbers = {
            chunk.metadata.get("source_page")
            for chunk in chunks
            if chunk.metadata.get("source_page") is not None
        }
        page_count = len(page_numbers) if page_numbers else len(chunks)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "pdf_id": pdf_id,
                "pdf_name": pdf.name,
                "chunk_index": i,
                "source_file": pdf.name,
            })

        self.vector_store.delete_collection_by_name(pdf.collection_name)
        self.vector_store.create_vector_db(
            documents=chunks,
            collection_name=pdf.collection_name
        )

        pdf.doc_count = len(chunks)
        pdf.page_count = page_count
        db.commit()
        db.refresh(pdf)

        return pdf

    def _generate_pdf_id(self, filename: str) -> str:
        """Generate unique PDF ID.

        Args:
            filename: Original filename

        Returns:
            Unique PDF identifier
        """
        timestamp = datetime.now().isoformat()
        return f"pdf_{abs(hash(filename + timestamp))}"
