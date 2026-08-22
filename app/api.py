"""
DocMind Enterprise REST API — FastAPI Backend Service.

Exposes production endpoints for RAG query execution, PDF ingestion,
system health, and Prometheus telemetry.
"""

import sys
import time
import os
from pathlib import Path
from typing import List, Optional

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from app.config import DOCS_DIR, RETRIEVAL_TOP_K, RERANK_TOP_N, GEMINI_API_KEY
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_and_store, get_collection_stats
from app.retrieval.bm25_index import rebuild_from_chroma
from app.retrieval.hybrid import hybrid_search
from app.agents.query_rewriter import rewrite_query
from app.agents.reranker import rerank
from app.agents.citation_validator import validate_citation
from app.llm.ollama_chain import get_llm, ask, build_context
from app.monitoring import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    QUERY_COUNT,
    GROUNDING_STATUS_COUNT,
    get_metrics,
)

# Initialize FastAPI App
app = FastAPI(
    title="DocMind Agentic RAG Service",
    description="Enterprise REST API for PDF Document Intelligence & Agentic RAG",
    version="2.0.0",
)

# Enable CORS for external frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Request/Response Models ───────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the attention mechanism in transformers?")
    top_k: int = Field(default=5, ge=1, le=15, description="Number of reranked chunks")
    provider: Optional[str] = Field(default="gemini", description="LLM provider: gemini | groq | auto | ollama")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    api_key: Optional[str] = Field(default=None, description="Optional Gemini API Key override")


class ChunkMetadata(BaseModel):
    source: str
    page: int
    text: str
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    status: str
    original_query: str
    rewritten_query: str
    answer: str
    grounding_status: str
    grounding_explanation: str
    latency_seconds: float
    sources: List[ChunkMetadata]


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for Docker & Cloud Load Balancers."""
    stats = get_collection_stats()
    return {
        "status": "healthy",
        "service": "DocMind RAG API",
        "indexed_chunks": stats["total_chunks"],
        "indexed_documents": stats["total_documents"],
    }


@app.get("/metrics", tags=["Telemetry"])
def metrics():
    """Exporter endpoint for Prometheus metric scraping."""
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)


@app.get("/api/v1/stats", tags=["Corpus"])
def get_stats():
    """Get corpus indexing statistics and source file list."""
    return get_collection_stats()


def _process_ingestion(save_path: str, filename: str):
    """Background task to load, chunk, embed, and update BM25 index."""
    try:
        pages = load_pdf(save_path)
        if pages:
            chunks = chunk_pages(pages)
            num_added = embed_and_store(chunks)
            rebuild_from_chroma()
            logger.info(f"[Async Ingestion] Successfully indexed {filename}: {num_added} chunks")
        else:
            logger.warning(f"[Async Ingestion] No text extracted from {filename}")
    except Exception as e:
        logger.error(f"[Async Ingestion Error] Failed to ingest {filename}: {e}")


@app.post("/api/v1/upload", tags=["Ingestion"])
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF document. Asynchronously parses, chunks, embeds, and updates BM25 index.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(DOCS_DIR, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Trigger asynchronous background processing
    background_tasks.add_task(_process_ingestion, save_path, file.filename)

    return {
        "status": "queued",
        "filename": file.filename,
        "message": "File uploaded successfully and queued for background ingestion.",
    }


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(req: QueryRequest):
    """
    Execute full Agentic RAG Pipeline:
    Query Rewriting ➔ Hybrid RRF Search ➔ Cross-Encoder Reranking ➔ LLM Generation ➔ Citation Validation
    """
    start_time = time.time()
    endpoint = "/api/v1/query"

    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No documents indexed in corpus. Upload PDFs first.")

    try:
        # Step 1: LLM Instance
        llm = get_llm(temperature=req.temperature, provider=req.provider, api_key=req.api_key)

        # Step 2: Agent 1 — Query Rewriter
        rewritten_query = rewrite_query(req.query, llm=llm)

        # Step 3: Hybrid Retrieval (BM25 + Dense RRF)
        candidates = hybrid_search(rewritten_query, top_k=RETRIEVAL_TOP_K)

        # Step 4: Agent 2 — Cross-Encoder Reranker
        top_chunks = rerank(rewritten_query, candidates, top_n=req.top_k)

        # Step 5: Answer Generation
        answer = ask(
            question=req.query,
            chunks=top_chunks,
            temperature=req.temperature,
            provider=req.provider,
            api_key=req.api_key,
        )

        # Step 6: Agent 3 — Citation Validator
        context_text = build_context(top_chunks)
        validation = validate_citation(answer, context_text, llm=llm)
        grounding_status = validation.get("status", "UNKNOWN")
        grounding_explanation = validation.get("explanation", "")

        # Record Metrics
        latency = time.time() - start_time
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="200").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        QUERY_COUNT.labels(provider=req.provider or "gemini").inc()
        GROUNDING_STATUS_COUNT.labels(status=grounding_status).inc()

        sources_data = [
            ChunkMetadata(
                source=c.get("source", "unknown"),
                page=c.get("page", 1),
                text=c.get("text", ""),
                rerank_score=float(c.get("rerank_score", 0.0)) if "rerank_score" in c else None,
            )
            for c in top_chunks
        ]

        return QueryResponse(
            status="success",
            original_query=req.query,
            rewritten_query=rewritten_query,
            answer=answer,
            grounding_status=grounding_status,
            grounding_explanation=grounding_explanation,
            latency_seconds=round(latency, 3),
            sources=sources_data,
        )

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="500").inc()
        logger.error(f"API Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
