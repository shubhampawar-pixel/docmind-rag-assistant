"""
Hybrid Retriever — Combines BM25 and dense search using Reciprocal Rank Fusion.

RRF formula: score(d) = Σ 1/(k + rank(d)) where k=60.
Merges ranked lists from both retrieval systems without needing to
normalize scores (which differ in scale between BM25 and cosine similarity).
"""

from typing import List, Dict, Optional

from loguru import logger

from app.config import RETRIEVAL_TOP_K
from app.retrieval import bm25_index
from app.retrieval.chroma_store import dense_search


# RRF constant — empirically chosen to reduce influence of very high-ranked docs
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked result lists. Each result dict must
                      have an "id" field.
        top_k: Number of merged results to return.

    Returns:
        Top-K merged results sorted by RRF score (descending).
    """
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    rrf_scores = {}  # doc_id -> cumulative RRF score
    doc_data = {}     # doc_id -> full document data

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc["id"]
            rrf_score = 1.0 / (RRF_K + rank)

            if doc_id in rrf_scores:
                rrf_scores[doc_id] += rrf_score
            else:
                rrf_scores[doc_id] = rrf_score
                doc_data[doc_id] = doc

    # Sort by RRF score descending
    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda x: rrf_scores[x],
        reverse=True,
    )

    # Build output with RRF scores
    output = []
    for doc_id in sorted_ids[:top_k]:
        result = doc_data[doc_id].copy()
        result["rrf_score"] = rrf_scores[doc_id]
        output.append(result)

    return output


def hybrid_search(
    query: str,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    Perform hybrid retrieval: BM25 + dense search, merged with RRF.

    Args:
        query: The search query.
        top_k: Number of results to return after fusion.

    Returns:
        List of result dicts sorted by RRF score, with full chunk text
        and metadata.
    """
    if top_k is None:
        top_k = RETRIEVAL_TOP_K

    # Run both retrieval systems
    bm25_results = bm25_index.search(query, top_k=top_k)
    dense_results = dense_search(query, top_k=top_k)

    logger.debug(
        f"Hybrid search — BM25: {len(bm25_results)} results, "
        f"Dense: {len(dense_results)} results"
    )

    # Merge with RRF
    merged = reciprocal_rank_fusion(
        [bm25_results, dense_results],
        top_k=top_k,
    )

    logger.info(
        f"Hybrid search returned {len(merged)} merged results "
        f"for: {query[:60]}..."
    )
    return merged
