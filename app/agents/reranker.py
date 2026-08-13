"""
CrossEncoder Reranker — Re-scores retrieved candidates for better precision.

Uses ms-marco-MiniLM-L-6-v2 from Hugging Face. This is a genuine
cross-encoder call — it takes (query, document) pairs together as input,
not just document embeddings independently.
"""

from typing import List, Dict, Optional

from loguru import logger
from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL, RERANK_TOP_N

# Lazy-loaded singleton for the reranker model
_reranker = None


def _get_reranker() -> CrossEncoder:
    """Load the CrossEncoder model (singleton, loaded once)."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker model loaded successfully")
    return _reranker


def rerank(
    query: str,
    candidates: List[Dict],
    top_n: Optional[int] = None,
) -> List[Dict]:
    """
    Re-score and rerank candidates using the CrossEncoder.

    Args:
        query: The search query.
        candidates: List of candidate dicts with at least a "text" field.
        top_n: Number of top results to return after reranking.

    Returns:
        Top-N candidates sorted by cross-encoder relevance score.
    """
    if top_n is None:
        top_n = RERANK_TOP_N

    if not candidates:
        return []

    reranker = _get_reranker()

    # Create (query, document) pairs for the cross-encoder
    pairs = [(query, doc["text"]) for doc in candidates]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Attach scores to candidates
    scored_candidates = []
    for i, candidate in enumerate(candidates):
        result = candidate.copy()
        result["rerank_score"] = float(scores[i])
        scored_candidates.append(result)

    # Sort by rerank score descending (higher = more relevant)
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Return top-N
    top_results = scored_candidates[:top_n]

    logger.info(
        f"Reranking complete: {len(candidates)} candidates → "
        f"top {len(top_results)} results"
    )
    return top_results
