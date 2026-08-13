"""
BM25 Index — Sparse keyword-based retrieval.

Builds a BM25Okapi index from all documents in ChromaDB.
Caches the index as a pickle file for fast startup.
"""

import os
import pickle
from typing import List, Dict, Optional

from loguru import logger
from rank_bm25 import BM25Okapi

import app.config as config

# Module-level cache
_bm25_index = None
_bm25_doc_ids = None
_bm25_docs_meta = None


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization (as specified in the spec)."""
    return text.lower().split()


def build_index(documents: Dict) -> None:
    """
    Build a BM25 index from ChromaDB documents.

    Args:
        documents: ChromaDB get() result with ids, documents, metadatas.
    """
    global _bm25_index, _bm25_doc_ids, _bm25_docs_meta

    if not documents or not documents.get("ids"):
        logger.warning("No documents to build BM25 index from")
        _bm25_index = None
        _bm25_doc_ids = []
        _bm25_docs_meta = []
        return

    ids = documents["ids"]
    texts = documents["documents"]
    metadatas = documents["metadatas"]

    # Tokenize all documents
    tokenized = [_tokenize(text) for text in texts]

    # Build the BM25 index
    _bm25_index = BM25Okapi(tokenized)
    _bm25_doc_ids = ids
    _bm25_docs_meta = [
        {
            "id": ids[i],
            "text": texts[i],
            "source": metadatas[i].get("source", ""),
            "page": metadatas[i].get("page", 0),
        }
        for i in range(len(ids))
    ]

    # Save to cache
    _save_cache()
    logger.info(f"BM25 index built with {len(ids)} documents")


def _save_cache() -> None:
    """Pickle the BM25 index to disk."""
    global _bm25_index, _bm25_doc_ids, _bm25_docs_meta

    os.makedirs(os.path.dirname(config.BM25_CACHE), exist_ok=True)
    cache_data = {
        "index": _bm25_index,
        "doc_ids": _bm25_doc_ids,
        "docs_meta": _bm25_docs_meta,
    }
    with open(config.BM25_CACHE, "wb") as f:
        pickle.dump(cache_data, f)
    logger.debug(f"BM25 index cached to {config.BM25_CACHE}")


def load_cache() -> bool:
    """
    Load the BM25 index from cache.

    Returns:
        True if cache was loaded successfully, False otherwise.
    """
    global _bm25_index, _bm25_doc_ids, _bm25_docs_meta

    if not os.path.exists(config.BM25_CACHE):
        return False

    try:
        with open(config.BM25_CACHE, "rb") as f:
            cache_data = pickle.load(f)
        _bm25_index = cache_data["index"]
        _bm25_doc_ids = cache_data["doc_ids"]
        _bm25_docs_meta = cache_data["docs_meta"]
        logger.info(
            f"BM25 index loaded from cache ({len(_bm25_doc_ids)} docs)"
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load BM25 cache: {e}")
        return False


def search(query: str, top_k: Optional[int] = None) -> List[Dict]:
    """
    Search the BM25 index.

    Args:
        query: The search query.
        top_k: Number of results to return.

    Returns:
        List of result dicts: [{"id", "text", "source", "page", "score"}, ...]
    """
    global _bm25_index, _bm25_docs_meta

    if top_k is None:
        top_k = config.RETRIEVAL_TOP_K

    if _bm25_index is None:
        logger.warning("BM25 index not built — returning empty results")
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-K by score
    scored_results = [
        (i, scores[i]) for i in range(len(scores))
    ]
    scored_results.sort(key=lambda x: x[1], reverse=True)
    top_results = scored_results[:top_k]

    output = []
    for idx, score in top_results:
        if score > 0:  # Only include docs with non-zero BM25 score
            doc = _bm25_docs_meta[idx]
            output.append({
                "id": doc["id"],
                "text": doc["text"],
                "source": doc["source"],
                "page": doc["page"],
                "score": float(score),
            })

    logger.debug(
        f"BM25 search returned {len(output)} results for: {query[:60]}..."
    )
    return output


def rebuild_from_chroma() -> None:
    """Rebuild the BM25 index from ChromaDB contents."""
    from app.ingestion.embedder import get_all_documents

    documents = get_all_documents()
    build_index(documents)
