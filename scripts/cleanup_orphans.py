"""Maintenance script: clean up storage leftovers that accumulate over time.

1. Orphaned ChromaDB collections — a collection with no corresponding row
   in api.db's `pdfs` table. Happens when a PDF is deleted through some path
   other than PDFService.delete_pdf(), or an upload/refresh is interrupted
   between creating the collection and committing the DB row.
2. The unused `analysis_results` table in api.db — a leftover from an
   unfinished OCR/translation "analyze" feature. No current SQLAlchemy model
   or code path reads or writes it (verified via `grep -rn analysis_results
   src/ web-ui/` finding nothing outside this script).

Defaults to a dry run (report only). Pass --apply to actually delete.

Usage:
    python scripts/cleanup_orphans.py              # dry-run: report only
    python scripts/cleanup_orphans.py --apply       # delete orphans + drop analysis_results
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from sqlalchemy import text

from src.api.config import settings
from src.api.database import PDFMetadata, SessionLocal


def find_orphaned_collections() -> list[dict]:
    """Return ChromaDB collections with no matching PDFMetadata.collection_name."""
    db = SessionLocal()
    try:
        known_collection_names = {pdf.collection_name for pdf in db.query(PDFMetadata).all()}
    finally:
        db.close()

    client = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
    orphans = []
    for collection in client.list_collections():
        if collection.name not in known_collection_names:
            count = client.get_collection(collection.name).count()
            orphans.append({"name": collection.name, "chunk_count": count})
    return orphans


def analysis_results_row_count() -> int | None:
    """Row count of api.db's `analysis_results` table, or None if it doesn't exist."""
    db = SessionLocal()
    try:
        exists = db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results'")
        ).fetchone()
        if not exists:
            return None
        return db.execute(text("SELECT COUNT(*) FROM analysis_results")).scalar()
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Actually delete/drop (default is dry-run)")
    args = parser.parse_args()

    orphans = find_orphaned_collections()
    row_count = analysis_results_row_count()

    print("=== Orphaned ChromaDB collections ===")
    if not orphans:
        print("(none found)")
    for o in orphans:
        print(f"  {o['name']}: {o['chunk_count']} chunk(s)")

    print("\n=== analysis_results table ===")
    if row_count is None:
        print("(table does not exist — nothing to drop)")
    else:
        print(f"exists, {row_count} row(s), unused by any current code path")

    if not args.apply:
        print("\nDry run — no changes made. Re-run with --apply to delete/drop the above.")
        return

    if orphans:
        client = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
        for o in orphans:
            client.delete_collection(o["name"])
            print(f"Deleted collection: {o['name']}")

    if row_count is not None:
        db = SessionLocal()
        try:
            db.execute(text("DROP TABLE analysis_results"))
            db.commit()
            print("Dropped table: analysis_results")
        finally:
            db.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
