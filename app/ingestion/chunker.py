"""
Semantic Chunker — Splits page text into semantically coherent chunks.

Uses sentence boundary detection (not fixed character count) to keep
ideas together. Chunks are ~300–500 tokens with 50-token overlap via
the last 2 sentences from the previous chunk.
"""

import re
from typing import List, Dict

from loguru import logger

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex on common sentence boundaries.

    Handles '. ', '? ', '! ', and newline boundaries.
    """
    # Split on sentence-ending punctuation followed by space, or on newlines
    raw_splits = re.split(r'(?<=[.!?])\s+|\n+', text)
    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in raw_splits if s.strip()]
    return sentences


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters (English average)."""
    return len(text) // 4


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Chunk a list of page dicts into semantically coherent chunks.

    Each chunk keeps the source and page metadata from its parent page.
    Chunks get a unique ID: {source}_p{page}_c{chunk_index}.

    Args:
        pages: List of {"text": str, "source": str, "page": int} dicts.

    Returns:
        List of chunk dicts with: text, source, page, chunk_id.
    """
    all_chunks = []
    chunk_counter = {}  # Track chunk index per (source, page)

    for page_data in pages:
        text = page_data["text"]
        source = page_data["source"]
        page = page_data["page"]

        sentences = _split_into_sentences(text)
        if not sentences:
            continue

        key = (source, page)
        if key not in chunk_counter:
            chunk_counter[key] = 0

        current_chunk_sentences = []
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = _estimate_tokens(sentence)

            # If adding this sentence would exceed the chunk size,
            # save the current chunk and start a new one
            if (current_chunk_tokens + sentence_tokens > CHUNK_SIZE
                    and current_chunk_sentences):
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = f"{source}_p{page}_c{chunk_counter[key]}"
                all_chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_id,
                })
                chunk_counter[key] += 1

                # Start new chunk with last 2 sentences as overlap
                overlap_sentences = current_chunk_sentences[-2:]
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = sum(
                    _estimate_tokens(s) for s in overlap_sentences
                )

            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens

        # Don't forget the last chunk for this page
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{source}_p{page}_c{chunk_counter[key]}"
            all_chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
            })
            chunk_counter[key] += 1

    logger.info(
        f"Chunking complete: {len(all_chunks)} chunks from "
        f"{len(pages)} pages"
    )
    return all_chunks
