"""
PDF Loader — Extracts text from PDF files page by page using PyMuPDF.

Each page is returned as a dict with text, source filename, and page number.
Pages with fewer than 50 characters (blank/image pages) are skipped.
Also provides SHA-256 hash tracking for delta incremental ingestion.
"""

import hashlib
from pathlib import Path
from typing import List, Dict

import pymupdf as fitz  # PyMuPDF
from loguru import logger


def get_file_hash(file_path: str) -> str:
    """Calculate the SHA-256 hash of a file for delta change detection."""
    file_path = Path(file_path)
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load a PDF file and extract text page by page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of dicts: [{"text": str, "source": str, "page": int, "doc_hash": str}, ...]
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    if not file_path.suffix.lower() == ".pdf":
        logger.warning(f"Not a PDF file: {file_path}")
        return []

    source = file_path.name
    doc_hash = get_file_hash(str(file_path))
    pages = []

    try:
        doc = fitz.open(str(file_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            # Skip pages with fewer than 50 characters (blank/image pages)
            if len(text) < 50:
                logger.debug(
                    f"Skipping page {page_num + 1} of {source} "
                    f"(only {len(text)} chars)"
                )
                continue

            pages.append({
                "text": text,
                "source": source,
                "page": page_num + 1,  # 1-indexed
                "doc_hash": doc_hash,
            })

        doc.close()
        logger.info(
            f"Loaded {source}: {len(pages)} pages extracted "
            f"(out of {page_num + 1} total)"
        )
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

    return pages


def load_all_pdfs(directory: str) -> List[Dict]:
    """
    Load all PDF files from a directory.

    Args:
        directory: Path to the directory containing PDF files.

    Returns:
        List of page dicts from all PDFs.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    all_pages = []
    pdf_files = sorted(dir_path.glob("*.pdf"))

    if not pdf_files:
        logger.info(f"No PDF files found in {directory}")
        return []

    for pdf_file in pdf_files:
        pages = load_pdf(str(pdf_file))
        all_pages.extend(pages)

    logger.info(
        f"Total: {len(all_pages)} pages from {len(pdf_files)} PDF files"
    )
    return all_pages
