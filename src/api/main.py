"""FastAPI main application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

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
from .config import settings

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

# Include routers
app.include_router(pdfs.router)
app.include_router(query.router)
app.include_router(models.router)
app.include_router(health.router)

logger = logging.getLogger(__name__)


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
