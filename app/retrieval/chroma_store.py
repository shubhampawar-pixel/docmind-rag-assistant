"""
ChromaDB Store — Dense vector similarity search.

Provides dense retrieval against the ChromaDB collection using
the same embedding model used for indexing.
"""

from typing import List, Dict, Optional

from loguru import logger
from sentence_transformers import SentenceTransformer

import app.config as config

# Lazy-loaded singleton for the embedding model
_model = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def _get_collection():
    """Get the ChromaDB collection."""
    import chromadb

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def dense_search(
    query: str,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    Perform dense vector similarity search.

    Args:
        query: The search query text.
        top_k: Number of results to return (defaults to RETRIEVAL_TOP_K).

    Returns:
        List of result dicts: [{"id", "text", "source", "page", "score"}, ...]
        Sorted by relevance (most relevant first).
    """
    if top_k is None:
        top_k = config.RETRIEVAL_TOP_K

    collection = _get_collection()
    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty")
        return []

    model = _get_model()
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Convert ChromaDB results to a flat list of dicts
    output = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "page": results["metadatas"][0][i].get("page", 0),
                "score": results["distances"][0][i],
            })

    logger.debug(f"Dense search returned {len(output)} results for: {query[:60]}...")
    return output
