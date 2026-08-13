"""
Tests for the ingestion pipeline.

Creates a small dummy PDF with PyMuPDF, runs through loader + chunker,
and asserts chunks are created with proper metadata.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import pymupdf as fitz  # PyMuPDF (new import name)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_pages


@pytest.fixture
def dummy_pdf(tmp_path):
    """Create a small dummy PDF with real text content."""
    pdf_path = tmp_path / "test_document.pdf"

    doc = fitz.open()

    # Page 1 — substantial text
    page1 = doc.new_page()
    text1 = (
        "Machine learning is a subset of artificial intelligence that "
        "provides systems the ability to automatically learn and improve "
        "from experience without being explicitly programmed. "
        "Machine learning focuses on the development of computer programs "
        "that can access data and use it to learn for themselves. "
        "The process of learning begins with observations or data, "
        "such as examples, direct experience, or instruction, in order "
        "to look for patterns in data and make better decisions in the "
        "future based on the examples that we provide. "
        "The primary aim is to allow the computers to learn automatically "
        "without human intervention or assistance and adjust actions accordingly."
    )
    page1.insert_text((72, 72), text1, fontsize=11)

    # Page 2 — more text
    page2 = doc.new_page()
    text2 = (
        "Deep learning is part of a broader family of machine learning "
        "methods based on artificial neural networks with representation "
        "learning. Learning can be supervised, semi-supervised or unsupervised. "
        "Deep learning architectures such as deep neural networks, "
        "recurrent neural networks, and convolutional neural networks "
        "have been applied to fields including computer vision, speech "
        "recognition, natural language processing, and many other tasks "
        "where they have produced results comparable to human experts."
    )
    page2.insert_text((72, 72), text2, fontsize=11)

    # Page 3 — too short (should be skipped)
    page3 = doc.new_page()
    page3.insert_text((72, 72), "Short.", fontsize=11)

    doc.save(str(pdf_path))
    doc.close()

    return str(pdf_path)


def test_load_pdf_extracts_pages(dummy_pdf):
    """Test that the loader extracts pages from a PDF."""
    pages = load_pdf(dummy_pdf)

    # Should have 2 pages (page 3 is too short, <50 chars)
    assert len(pages) >= 2, f"Expected at least 2 pages, got {len(pages)}"

    # Each page should have required fields
    for page in pages:
        assert "text" in page
        assert "source" in page
        assert "page" in page
        assert len(page["text"]) >= 50


def test_load_pdf_source_metadata(dummy_pdf):
    """Test that source metadata is correctly set."""
    pages = load_pdf(dummy_pdf)

    assert len(pages) > 0
    assert pages[0]["source"] == "test_document.pdf"
    assert pages[0]["page"] == 1


def test_load_pdf_skips_short_pages(dummy_pdf):
    """Test that pages with <50 characters are skipped."""
    pages = load_pdf(dummy_pdf)

    # Page 3 has only "Short." — should be skipped
    page_numbers = [p["page"] for p in pages]
    assert 3 not in page_numbers, "Short page should have been skipped"


def test_load_pdf_nonexistent_file():
    """Test that loading a nonexistent file returns empty list."""
    pages = load_pdf("/nonexistent/path/fake.pdf")
    assert pages == []


def test_chunk_pages_creates_chunks(dummy_pdf):
    """Test that the chunker creates chunks from pages."""
    pages = load_pdf(dummy_pdf)
    chunks = chunk_pages(pages)

    assert len(chunks) > 0, "Should create at least one chunk"


def test_chunk_metadata_has_required_fields(dummy_pdf):
    """Test that chunk metadata has source, page, and chunk_id fields."""
    pages = load_pdf(dummy_pdf)
    chunks = chunk_pages(pages)

    for chunk in chunks:
        assert "text" in chunk, "Chunk missing 'text' field"
        assert "source" in chunk, "Chunk missing 'source' field"
        assert "page" in chunk, "Chunk missing 'page' field"
        assert "chunk_id" in chunk, "Chunk missing 'chunk_id' field"
        assert len(chunk["text"]) > 0, "Chunk text should not be empty"


def test_chunk_ids_are_unique(dummy_pdf):
    """Test that chunk IDs are unique."""
    pages = load_pdf(dummy_pdf)
    chunks = chunk_pages(pages)

    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs should be unique"


def test_chunk_id_format(dummy_pdf):
    """Test that chunk IDs follow the expected format."""
    pages = load_pdf(dummy_pdf)
    chunks = chunk_pages(pages)

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        # Format: {source}_p{page}_c{chunk_index}
        assert "_p" in chunk_id, f"Chunk ID missing page marker: {chunk_id}"
        assert "_c" in chunk_id, f"Chunk ID missing chunk marker: {chunk_id}"
