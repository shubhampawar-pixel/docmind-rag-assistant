---
title: DocMind - Agentic RAG Knowledge Assistant
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# DocMind 🧠 — Agentic RAG Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Query your PDFs and study notes with an AI assistant that uses hybrid
> retrieval, cross-encoder reranking, and LLM inference — entirely free to run.

## Architecture

```
PDF Files (local folder or upload via UI)
        │
   Document Watcher (watchdog)
        │
   PDF Loader → Semantic Chunker
        │
   Embedding Model (HuggingFace BGE)
        │
   ChromaDB  +  BM25 Index
        │
─────────────────────────────────
User Question (Streamlit UI)
        │
   [Agent 1] Query Rewriter (LLM)
        │
   Hybrid Retriever (BM25 + Dense)
        │
   Reciprocal Rank Fusion (RRF)
        │
   [Agent 2] CrossEncoder Reranker  → top-5 chunks
        │
   LLM (Ollama local / Gemini API)
        │
   [Agent 3] Citation Validator (LLM)
        │
   Answer + Sources → Streamlit UI
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Retrieval | Hybrid (BM25 + Dense) | Neither alone is sufficient |
| Reranking | CrossEncoder | Better precision than bi-encoder alone |
| Embeddings | BAAI/bge-small-en-v1.5 | MTEB top performer, local, free |
| LLM (local) | qwen2.5:0.5b via Ollama | Smallest viable model, runs on any laptop |
| LLM (cloud) | Gemini 2.0 Flash | Free API, fast inference |
| Vector DB | ChromaDB | Persistent, easy local setup |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/docmind
cd docmind

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# (Optional) Install Ollama + pull smallest model for local inference
# ollama pull qwen2.5:0.5b

# Drop PDFs into data/documents/ (optional — you can also upload via UI)

# Run the app
streamlit run ui/streamlit_app.py
```

## Project Structure

```
docmind/
├── app/
│   ├── config.py          # all config constants
│   ├── ingestion/
│   │   ├── loader.py      # PDF → pages
│   │   ├── chunker.py     # semantic chunking
│   │   └── embedder.py    # embed + store in Chroma
│   ├── retrieval/
│   │   ├── bm25_index.py  # BM25 build + search
│   │   ├── chroma_store.py # dense search
│   │   └── hybrid.py      # RRF fusion
│   ├── agents/
│   │   ├── query_rewriter.py
│   │   ├── reranker.py    # CrossEncoder
│   │   └── citation_validator.py
│   ├── llm/
│   │   └── ollama_chain.py # LangChain + Ollama/Gemini
│   └── watcher/
│       └── folder_watch.py
├── ui/
│   └── streamlit_app.py   # main UI entry point
├── data/
│   └── documents/         # DROP PDFs HERE
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_agents.py
├── requirements.txt
├── .env.example
├── README.md
└── Makefile
```

## LLM Provider Options

DocMind supports three LLM provider modes (configurable in the UI sidebar):

| Mode | How It Works |
|------|-------------|
| **auto** (default) | Tries Ollama first; if not running, falls back to Gemini API |
| **gemini** | Always uses Google Gemini API (requires `GEMINI_API_KEY`) |
| **ollama** | Always uses local Ollama (requires Ollama running) |

### Using Gemini API Only (Recommended for most users)

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey)
2. Add `GEMINI_API_KEY=your_key` to `.env`
3. Select "gemini" in the sidebar settings

### Using Local Ollama

1. Install [Ollama](https://ollama.com)
2. Pull the model: `ollama pull qwen2.5:0.5b`
3. Start Ollama: `ollama serve`
4. Select "ollama" in the sidebar settings

## How It Works

1. **Drop PDFs** into `data/documents/` or upload via the UI — the folder watcher auto-indexes them
2. **Semantic chunking** splits text by sentence boundaries (~400 token chunks)
3. **Hybrid retrieval** combines BM25 keyword search + dense vector search via ChromaDB
4. **Reciprocal Rank Fusion** merges ranked lists from both retrieval systems
5. **CrossEncoder reranking** re-scores the top 20 candidates, keeps the best 5
6. **LLM** (Ollama local or Gemini API) generates a cited answer
7. **Citation validator** checks if the answer is grounded in source chunks

## Running Tests

```bash
pytest tests/ -v
```

## Deployment (Hugging Face Spaces)

The app can be deployed to [Hugging Face Spaces](https://huggingface.co/spaces):

1. Create a new Space (Streamlit SDK, CPU Basic)
2. Set `GEMINI_API_KEY=your_key` and `LLM_PROVIDER=gemini` in Space secrets
3. Push this repo — the `app.py` entry point handles the rest

## License

MIT
