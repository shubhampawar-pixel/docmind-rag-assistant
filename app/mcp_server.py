"""
DocMind MCP Server — Model Context Protocol (MCP) Integration.

Exposes DocMind's hybrid retrieval & reranking capabilities as an MCP server
compatible with Claude Desktop, Cursor, and enterprise AI agents.

Tools Exposed:
  - search_docmind_kb: Execute hybrid RRF search + Cross-Encoder reranking over PDF corpus.
  - get_corpus_stats: Retrieve vector collection document metrics and indexed sources.

Resources Exposed:
  - docmind://corpus/documents: List of indexed PDF files.
"""

import sys
import json
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from app.retrieval.hybrid import hybrid_search
from app.agents.reranker import rerank
from app.ingestion.embedder import get_collection_stats, get_all_documents


def search_docmind_kb(query: str, top_k: int = 5) -> str:
    """
    Search the DocMind Knowledge Base using Hybrid RRF + Cross-Encoder Reranking.

    Args:
        query: Search query string.
        top_k: Number of reranked results to return (default: 5).

    Returns:
        JSON string containing ranked source passages with metadata.
    """
    logger.info(f"[MCP Server] Tool Invoked: search_docmind_kb(query='{query}', top_k={top_k})")
    
    # Step 1: Hybrid Retrieval (BM25 + Dense RRF)
    candidates = hybrid_search(query, top_k=20)
    
    if not candidates:
        return json.dumps({"status": "empty", "results": [], "message": "No relevant documents found."})
    
    # Step 2: Cross-Encoder Reranking
    top_chunks = rerank(query, candidates, top_n=top_k)
    
    results = [
        {
            "source": c.get("source", "unknown"),
            "page": c.get("page", "?"),
            "chunk_id": c.get("chunk_id", ""),
            "text": c.get("text", ""),
            "score": float(c.get("rerank_score", 0.0)),
        }
        for c in top_chunks
    ]
    
    return json.dumps({
        "status": "success",
        "query": query,
        "results_count": len(results),
        "results": results,
    }, indent=2)


def get_corpus_stats() -> str:
    """
    Get current metadata and statistics for the DocMind vector collection.

    Returns:
        JSON string with total document count, total chunk count, and indexed files.
    """
    stats = get_collection_stats()
    return json.dumps(stats, indent=2)


def list_corpus_resources() -> str:
    """
    List all document resource URIs available in DocMind.

    Returns:
        JSON string with resource URIs.
    """
    docs = get_all_documents()
    sources = list(set(meta.get("source", "unknown") for meta in docs.get("metadatas", [])))
    
    return json.dumps({
        "resource_uri": "docmind://corpus/documents",
        "mime_type": "application/json",
        "sources": sources,
        "count": len(sources),
    }, indent=2)


if __name__ == "__main__":
    print("DocMind MCP Server Initialized.")
    print("Testing MCP Tool Call 'search_docmind_kb':")
    sample_res = search_docmind_kb("What is transformer attention?", top_k=3)
    print(sample_res)
