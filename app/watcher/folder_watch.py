"""
Folder Watcher — Auto-ingestion of new PDF files using watchdog.

Monitors the documents directory for new .pdf files. When detected,
runs the full ingestion pipeline (loader → chunker → embedder) and
rebuilds the BM25 index.
"""

import time
import threading
from pathlib import Path

from loguru import logger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app.config import DOCS_DIR


class PDFHandler(FileSystemEventHandler):
    """Handles file creation events for PDF files."""

    def on_created(self, event):
        """Triggered when a new file is created in the watched directory."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() != ".pdf":
            return

        logger.info(f"New file detected: {file_path.name}")

        # Small delay to ensure the file is fully written
        time.sleep(1)

        try:
            self._ingest_file(str(file_path))
        except Exception as e:
            logger.error(f"Error ingesting {file_path.name}: {e}")

    def _ingest_file(self, file_path: str):
        """Run the full ingestion pipeline for a single file."""
        from app.ingestion.loader import load_pdf
        from app.ingestion.chunker import chunk_pages
        from app.ingestion.embedder import embed_and_store
        from app.retrieval.bm25_index import rebuild_from_chroma

        # Load
        pages = load_pdf(file_path)
        if not pages:
            logger.warning(f"No pages extracted from {file_path}")
            return

        # Chunk
        chunks = chunk_pages(pages)
        if not chunks:
            logger.warning(f"No chunks created from {file_path}")
            return

        # Embed and store
        num_added = embed_and_store(chunks)

        # Rebuild BM25 index
        rebuild_from_chroma()

        source = Path(file_path).name
        logger.info(f"Indexed: {source} — {num_added} chunks added")


def start_watcher() -> Observer:
    """
    Start the folder watcher in a background thread.

    Returns:
        The watchdog Observer instance (can be stopped with observer.stop()).
    """
    logger.info(f"Starting folder watcher on: {DOCS_DIR}")

    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, DOCS_DIR, recursive=False)
    observer.daemon = True  # Dies when main thread exits
    observer.start()

    logger.info("Folder watcher started")
    return observer


def stop_watcher(observer: Observer) -> None:
    """Stop the folder watcher."""
    if observer and observer.is_alive():
        observer.stop()
        observer.join(timeout=5)
        logger.info("Folder watcher stopped")
