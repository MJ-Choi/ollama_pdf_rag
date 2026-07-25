"""FastAPI main application."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import time

# Configured before importing routers/services so every module's
# logger.info/error() call — including ones made at import time — picks up
# this format. Plain `logging.basicConfig(level=logging.INFO)` (the
# previous config) has no timestamp or logger name, so a scanned console
# just showed a wall of "INFO:module:message" lines with no way to tell
# when something happened or how long a step took — the actual complaint
# that prompted this. `%(name)s` gives the same "which class/module logged
# this" signal as Java's default logger pattern (e.g. Log4j's `%c`).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from .routers import pdfs, query, models, health
from .database import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(
    title="Ollama PDF RAG API",
    description="REST API for PDF-based RAG with Ollama",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("api.access")

# Every request/response PK a caller might need to debug — id fields the
# server actually keys its data on. Path-based ids (e.g. /api/v1/pdfs/{id})
# are already visible in the logged path itself; this only needs to cover
# ids that travel in a JSON body instead (e.g. POST /api/v1/query's
# {"pdf_ids": [...]}).
_PK_KEYS = ("pdf_id", "pdf_ids", "id", "session_id")


# Single cross-cutting interception point for every route this app serves —
# equivalent to a Spring @Around advice applied via one pointcut, instead of
# a logger.info() call duplicated into each router function. Unlike
# Next.js's middleware, Starlette's `call_next` genuinely wraps the full
# request/response lifecycle, so both sides are logged from this one place.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    started_at = time.time()

    pk = {}
    if request.method in ("POST", "PUT", "PATCH") and "application/json" in request.headers.get("content-type", ""):
        body_bytes = await request.body()
        if body_bytes:
            try:
                body = json.loads(body_bytes)
                if isinstance(body, dict):
                    pk = {k: body[k] for k in _PK_KEYS if k in body}
            except json.JSONDecodeError:
                pass

    access_logger.info(f"--> {request.method} {request.url.path} pk={pk or 'N/A'}")

    response = await call_next(request)

    duration_ms = round((time.time() - started_at) * 1000, 1)
    access_logger.info(f"<-- {request.method} {request.url.path} status={response.status_code} ({duration_ms}ms)")

    return response


# Include routers
app.include_router(pdfs.router)
app.include_router(query.router)
app.include_router(models.router)
app.include_router(health.router)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Ollama PDF RAG API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
