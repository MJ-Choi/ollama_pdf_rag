"""Database models and session management."""
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Database configuration
DATABASE_DIR = Path("data")
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/api.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class PDFMetadata(Base):
    """PDF metadata table."""
    __tablename__ = "pdfs"

    pdf_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    collection_name = Column(String, unique=True, nullable=False)
    upload_timestamp = Column(DateTime, nullable=False)
    doc_count = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=False)
    is_sample = Column(Boolean, default=False)
    file_path = Column(String)


# Note: this backend previously had ChatSession/ChatMessage models logging
# every query to `chat_sessions`/`messages` tables. Removed 2026-07-23: the
# session_id was never read back into an LLM call (fully stateless,
# single-turn RAG regardless), the frontend always sent session_id=null (so
# the backend minted a fresh UUID per query — one "session" per turn, not
# per conversation), and nothing in the repo ever called
# GET /sessions/{id}/messages. It was a write-only log nobody read. The
# Next.js frontend's own `web-ui/data/chat.db` (Drizzle) is the actual,
# load-bearing chat history — sidebar list, resume, delete, titles. The old
# `chat_sessions`/`messages` tables (and their historical rows) are left
# untouched in api.db as harmless residual data — see scripts/cleanup_orphans.py
# for the class of cleanup that could eventually remove them too.


# Create all tables
Base.metadata.create_all(bind=engine)
