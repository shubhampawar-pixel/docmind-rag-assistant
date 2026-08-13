"""
Embedder — Embeds text chunks and stores them in ChromaDB.

Uses BAAI/bge-small-en-v1.5 via sentence-transformers directly
(not the LangChain wrapper). Deduplicates by chunk_id.
"""

from typing import List, Dict

from loguru import logger
from sentence_transformers import SentenceTransformer

import app.config as config


# Lazy-loaded singleton for the embedding model
_model = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model (singleton, loaded once)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBED_MODEL}")
        _model = SentenceTransformer(config.EMBED_MODEL)
        logger.info("Embedding model loaded successfully")
    return _model


def _get_collection():
    """Get or create the ChromaDB collection."""
    import chromadb

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_store(chunks: List[Dict]) -> int:
    """
    Embed text chunks and store them in ChromaDB.

    Skips chunks whose chunk_id already exists in the collection.

    Args:
        chunks: List of {"text", "source", "page", "chunk_id"} dicts.

    Returns:
        Number of new chunks added.
    """
    if not chunks:
        logger.info("No chunks to embed")
        return 0

    collection = _get_collection()
    model = _get_model()

    # Check which chunks already exist
    existing_ids = set()
    chunk_ids = [c["chunk_id"] for c in chunks]

    try:
        # ChromaDB get() returns existing docs — use it to filter duplicates
        existing = collection.get(ids=chunk_ids)
        if existing and existing["ids"]:
            existing_ids = set(existing["ids"])
    except Exception:
        # If collection is empty or IDs don't exist, that's fine
        pass

    # Filter to only new chunks
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        logger.info("All chunks already indexed — skipping")
        return 0

    # Embed all new chunks in batch
    texts = [c["text"] for c in new_chunks]
    logger.info(f"Embedding {len(texts)} new chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Prepare data for ChromaDB
    ids = [c["chunk_id"] for c in new_chunks]
    documents = texts
    metadatas = [
        {
            "source": c["source"],
            "page": c["page"],
            "chunk_id": c["chunk_id"],
        }
        for c in new_chunks
    ]

    # Insert into ChromaDB (batch insert)
    # ChromaDB has a batch limit, so we chunk the inserts
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info(f"Stored {len(new_chunks)} new chunks in ChromaDB")
    return len(new_chunks)


def get_all_documents() -> Dict:
    """
    Retrieve all documents from ChromaDB.

    Returns:
        ChromaDB get() result dict with ids, documents, metadatas.
    """
    collection = _get_collection()
    return collection.get(include=["documents", "metadatas"])


def get_collection_stats() -> Dict:
    """Get stats about the ChromaDB collection."""
    collection = _get_collection()
    count = collection.count()

    # Get unique sources
    all_docs = collection.get(include=["metadatas"])
    sources = set()
    if all_docs and all_docs["metadatas"]:
        for meta in all_docs["metadatas"]:
            sources.add(meta.get("source", "unknown"))

    return {
        "total_chunks": count,
        "total_documents": len(sources),
        "sources": sorted(sources),
    }


def clear_collection():
    """Delete and recreate the ChromaDB collection."""
    import chromadb

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
        logger.info("ChromaDB collection cleared")
    except Exception:
        pass
    # Recreate
    client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
