"""
DocMind Configuration — All constants in one place.

No path or model name should be hardcoded outside this file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Data Directories ────────────────────────────────────────────────────────
DOCS_DIR = str(PROJECT_ROOT / "data" / "documents")
CHROMA_DIR = str(PROJECT_ROOT / "data" / "chroma_db")
BM25_CACHE = str(PROJECT_ROOT / "data" / "bm25_cache" / "bm25_index.pkl")

# ─── ChromaDB ────────────────────────────────────────────────────────────────
COLLECTION_NAME = "docmind"

# ─── Embedding Model ─────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ─── Reranker Model ──────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── LLM (Local — Ollama) ────────────────────────────────────────────────────
# qwen2.5:0.5b is the smallest viable model (~400 MB), runs on low-spec laptops
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ─── LLM (Cloud — Google Gemini) ─────────────────────────────────────────────
def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                key = str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass
    return key

GEMINI_API_KEY = get_gemini_api_key()
GEMINI_MODEL = "gemini-2.0-flash"

# ─── LLM Provider Toggle ─────────────────────────────────────────────────────
# Options: "auto" | "gemini" | "ollama"
def get_default_provider() -> str:
    prov = os.getenv("LLM_PROVIDER", "").strip()
    if not prov:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "LLM_PROVIDER" in st.secrets:
                prov = str(st.secrets["LLM_PROVIDER"]).strip()
        except Exception:
            pass
    return prov if prov in ["auto", "gemini", "ollama"] else "gemini"

LLM_PROVIDER = get_default_provider()

# ─── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400        # tokens (approximate)
CHUNK_OVERLAP = 50      # tokens overlap between chunks

# ─── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 20    # candidates before reranking
RERANK_TOP_N = 5        # final results after reranking
MAX_CONTEXT_CHARS = 6000

# ─── Ensure data directories exist ──────────────────────────────────────────
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(BM25_CACHE), exist_ok=True)
