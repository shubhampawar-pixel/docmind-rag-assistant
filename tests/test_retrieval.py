"""
Tests for the retrieval system.

Indexes 3 sample text chunks into a test ChromaDB collection,
runs hybrid search, and tests the reranker.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def test_chroma_dir(tmp_path):
    """Create a temporary ChromaDB directory for testing."""
    chroma_dir = str(tmp_path / "test_chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)

    # Patch config to use test directory
    import app.config as config
    original_dir = config.CHROMA_DIR
    original_collection = config.COLLECTION_NAME
    config.CHROMA_DIR = chroma_dir
    config.COLLECTION_NAME = "test_docmind"

    yield chroma_dir

    # Restore original config
    config.CHROMA_DIR = original_dir
    config.COLLECTION_NAME = original_collection


@pytest.fixture
def sample_chunks():
    """Create 3 sample text chunks for testing."""
    return [
        {
            "text": (
                "BERT is a transformer-based model for natural language "
                "processing. It uses bidirectional training to understand "
                "context from both directions in a sentence."
            ),
            "source": "nlp_guide.pdf",
            "page": 1,
            "chunk_id": "nlp_guide.pdf_p1_c0",
        },
        {
            "text": (
                "Convolutional neural networks are primarily used for "
                "image recognition tasks. They use filters to detect "
                "features like edges, textures, and shapes in images."
            ),
            "source": "deep_learning.pdf",
            "page": 5,
            "chunk_id": "deep_learning.pdf_p5_c0",
        },
        {
            "text": (
                "Gradient descent is an optimization algorithm used to "
                "minimize the loss function in machine learning models. "
                "The learning rate controls the step size during training."
            ),
            "source": "ml_basics.pdf",
            "page": 12,
            "chunk_id": "ml_basics.pdf_p12_c0",
        },
    ]


def test_embed_and_retrieve(test_chroma_dir, sample_chunks):
    """Test that we can embed chunks and retrieve them via dense search."""
    from app.ingestion.embedder import embed_and_store
    from app.retrieval.chroma_store import dense_search

    # Index the sample chunks
    num_added = embed_and_store(sample_chunks)
    assert num_added == 3, f"Expected 3 chunks added, got {num_added}"

    # Search for something related to NLP
    results = dense_search("What is BERT in NLP?", top_k=3)
    assert len(results) > 0, "Dense search should return results"

    # The NLP chunk should be among the results
    result_ids = [r["id"] for r in results]
    assert "nlp_guide.pdf_p1_c0" in result_ids, (
        "NLP chunk should be retrieved for NLP query"
    )


def test_bm25_search(test_chroma_dir, sample_chunks):
    """Test BM25 search returns results."""
    from app.ingestion.embedder import embed_and_store, get_all_documents
    from app.retrieval.bm25_index import build_index, search

    # Index the sample chunks
    embed_and_store(sample_chunks)

    # Build BM25 index from ChromaDB contents
    documents = get_all_documents()
    build_index(documents)

    # Search for "gradient descent"
    results = search("gradient descent optimization", top_k=3)
    assert len(results) > 0, "BM25 search should return results"

    # The ML basics chunk should be top result
    assert results[0]["id"] == "ml_basics.pdf_p12_c0", (
        "ML basics chunk should be top result for gradient descent query"
    )


def test_hybrid_search_returns_results(test_chroma_dir, sample_chunks):
    """Test hybrid search merges BM25 and dense results."""
    from app.ingestion.embedder import embed_and_store, get_all_documents
    from app.retrieval.bm25_index import build_index
    from app.retrieval.hybrid import hybrid_search

    # Index
    embed_and_store(sample_chunks)
    documents = get_all_documents()
    build_index(documents)

    # Hybrid search
    results = hybrid_search("transformer model for language", top_k=3)
    assert len(results) > 0, "Hybrid search should return results"

    # Results should have RRF scores
    for r in results:
        assert "rrf_score" in r, "Results should have RRF scores"


def test_reranker_picks_most_relevant(test_chroma_dir, sample_chunks):
    """Test that the reranker picks the most relevant chunk."""
    from app.agents.reranker import rerank

    # Rerank all 3 sample chunks for a specific query
    query = "What is BERT and how does it work?"
    results = rerank(query, sample_chunks, top_n=1)

    assert len(results) == 1, "Should return exactly 1 result"
    assert results[0]["chunk_id"] == "nlp_guide.pdf_p1_c0", (
        "NLP chunk should be most relevant for BERT query"
    )
    assert "rerank_score" in results[0], "Result should have rerank score"


def test_deduplication(test_chroma_dir, sample_chunks):
    """Test that duplicate chunks are not re-indexed."""
    from app.ingestion.embedder import embed_and_store

    # First insert
    num_added_1 = embed_and_store(sample_chunks)
    assert num_added_1 == 3

    # Second insert — should add 0
    num_added_2 = embed_and_store(sample_chunks)
    assert num_added_2 == 0, "Duplicate chunks should not be re-indexed"
