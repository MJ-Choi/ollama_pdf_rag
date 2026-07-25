"""RAG query endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..dependencies import get_db, get_rag_service
from ..models import QueryRequest, QueryResponse, SourceInfo
from ..services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_pdfs(
    request: QueryRequest,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service)
):
    """Query across PDFs with source attribution."""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"📥 Received query request: question='{request.question[:50]}...', model={request.model}")

    # Query RAG
    logger.info("🚀 Starting RAG query...")
    try:
        answer, sources, reasoning_steps, truncated = rag_service.query_multi_pdf(
            question=request.question,
            model=request.model,
            pdf_ids=request.pdf_ids,
            db=db
        )
        logger.info(f"✅ RAG query complete: answer_length={len(answer)}, sources_count={len(sources)}, reasoning_steps={len(reasoning_steps)}, truncated={truncated}")
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() and "404" in error_msg:
            logger.error(f"❌ Model not found: {request.model}")
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' not found. Please select a different model from the dropdown or install it with: ollama pull {request.model}"
            )
        logger.error(f"❌ Query failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Query failed: {error_msg}")

    response = QueryResponse(
        answer=answer,
        sources=[SourceInfo(**s) for s in sources],
        metadata={
            "model_used": request.model,
            "chunks_retrieved": len(sources),
            "pdfs_queried": len(set(s["pdf_id"] for s in sources)),
            "reasoning_steps": reasoning_steps,
            "truncated": truncated
        }
    )

    logger.info(f"📤 Returning response: answer_length={len(response.answer)}, sources={len(response.sources)}")
    logger.info(f"📊 First 200 chars of answer: {response.answer[:200]}")

    return response
