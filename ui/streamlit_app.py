"""
DocMind — Streamlit UI

Single-page chat interface for querying indexed PDF documents.
Features sidebar with document stats, settings, and file upload.
Main area is a conversational Q&A interface with source citations.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from loguru import logger

from app.config import DOCS_DIR, RERANK_TOP_N, RETRIEVAL_TOP_K

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind 🧠",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }

    /* Warning banner for NOT_GROUNDED */
    .warning-banner {
        background-color: #3d2e00;
        border: 1px solid #f0ad4e;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #f0ad4e;
        font-size: 0.9em;
    }

    /* Source expander styling */
    .source-chunk {
        background-color: #1a1f29;
        border-left: 3px solid #58a6ff;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85em;
        color: #c9d1d9;
    }

    .source-meta {
        color: #58a6ff;
        font-weight: 600;
        font-size: 0.8em;
        margin-bottom: 4px;
    }

    /* Stats card */
    .stats-card {
        background: linear-gradient(135deg, #1a1f29, #21262d);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }

    .stats-number {
        font-size: 1.8em;
        font-weight: 700;
        color: #58a6ff;
    }

    .stats-label {
        font-size: 0.85em;
        color: #8b949e;
    }

    /* Log panel */
    .log-entry {
        font-family: 'Courier New', monospace;
        font-size: 0.75em;
        color: #8b949e;
        padding: 2px 0;
        border-bottom: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "watcher_started" not in st.session_state:
    st.session_state.watcher_started = False

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []


def add_log(msg: str):
    """Add a log entry to the pipeline log panel."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.pipeline_logs.append(f"[{ts}] {msg}")
    # Keep last 50 entries
    if len(st.session_state.pipeline_logs) > 50:
        st.session_state.pipeline_logs = st.session_state.pipeline_logs[-50:]


# ─── Initialization (runs once) ─────────────────────────────────────────────
@st.cache_resource
def initialize_pipeline():
    """Initialize the ingestion pipeline and BM25 index on startup."""
    from app.ingestion.loader import load_all_pdfs
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.embedder import embed_and_store, get_collection_stats
    from app.retrieval.bm25_index import rebuild_from_chroma, load_cache

    # Check if there are existing documents in ChromaDB
    stats = get_collection_stats()

    if stats["total_chunks"] == 0:
        # Index any PDFs already in the documents folder
        logger.info("No documents in ChromaDB — checking for PDFs to index...")
        pages = load_all_pdfs(DOCS_DIR)
        if pages:
            chunks = chunk_pages(pages)
            embed_and_store(chunks)

    # Build or load BM25 index
    if not load_cache():
        rebuild_from_chroma()

    return True


@st.cache_resource
def start_folder_watcher():
    """Start the folder watcher (runs once)."""
    from app.watcher.folder_watch import start_watcher
    return start_watcher()


# ─── Run Initialization ─────────────────────────────────────────────────────
with st.spinner("🔄 Initializing DocMind..."):
    initialize_pipeline()

if not st.session_state.watcher_started:
    start_folder_watcher()
    st.session_state.watcher_started = True


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧠 DocMind")
    st.markdown("*Agentic RAG Knowledge Assistant*")
    st.divider()

    # Document stats
    from app.ingestion.embedder import get_collection_stats
    stats = get_collection_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stats-card">'
            f'<div class="stats-number">{stats["total_documents"]}</div>'
            f'<div class="stats-label">Documents</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stats-card">'
            f'<div class="stats-number">{stats["total_chunks"]}</div>'
            f'<div class="stats-label">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    # Indexed documents list
    if stats["sources"]:
        st.markdown("### 📄 Indexed Documents")
        for source in stats["sources"]:
            st.markdown(f"- `{source}`")
    else:
        st.info("No documents indexed yet. Drop PDFs into `data/documents/` or upload below.")

    st.divider()

    # File upload
    st.markdown("### 📤 Upload PDF")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(DOCS_DIR, uploaded_file.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Uploaded: {uploaded_file.name}")

                # Trigger ingestion
                with st.spinner(f"📊 Indexing {uploaded_file.name}..."):
                    from app.ingestion.loader import load_pdf
                    from app.ingestion.chunker import chunk_pages
                    from app.ingestion.embedder import embed_and_store
                    from app.retrieval.bm25_index import rebuild_from_chroma

                    pages = load_pdf(save_path)
                    if pages:
                        chunks = chunk_pages(pages)
                        num_added = embed_and_store(chunks)
                        rebuild_from_chroma()
                        st.success(f"✅ Indexed: {num_added} chunks added")
                        add_log(f"Indexed {uploaded_file.name}: {num_added} chunks")
                    else:
                        st.warning("⚠️ No text could be extracted from this PDF")
                        add_log(f"Failed to extract text from {uploaded_file.name}")

                st.rerun()
            else:
                st.info(f"📄 {uploaded_file.name} is already indexed")

    st.divider()

    # Re-index button
    if st.button("🔄 Re-index All Documents", use_container_width=True):
        with st.spinner("Re-indexing all documents..."):
            from app.ingestion.embedder import clear_collection
            from app.ingestion.loader import load_all_pdfs
            from app.ingestion.chunker import chunk_pages
            from app.ingestion.embedder import embed_and_store
            from app.retrieval.bm25_index import rebuild_from_chroma

            clear_collection()
            pages = load_all_pdfs(DOCS_DIR)
            if pages:
                chunks = chunk_pages(pages)
                num_stored = embed_and_store(chunks)
                add_log(f"Re-indexed all: {num_stored} chunks from {len(set(p['source'] for p in pages))} docs")
            rebuild_from_chroma()

        st.success("✅ Re-indexing complete!")
        st.rerun()

    st.divider()

    # Settings
    with st.expander("⚙️ Settings", expanded=True):
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["auto", "gemini", "ollama"],
            index=0,
            help=(
                "**auto**: Try local Ollama first, fall back to Gemini API.\n\n"
                "**gemini**: Always use Google Gemini API (needs GEMINI_API_KEY).\n\n"
                "**ollama**: Always use local Ollama (needs Ollama running)."
            ),
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Lower = more precise, Higher = more creative",
        )
        top_k = st.slider(
            "Top-K (final chunks for answer)",
            min_value=1,
            max_value=20,
            value=RERANK_TOP_N,
            step=1,
            help="Number of final chunks used for answering after reranking",
        )


# ─── Main Chat Area ─────────────────────────────────────────────────────────
st.markdown("## 💬 Ask DocMind")

# ─── Tabs: Chat | Vector DB | Logs ──────────────────────────────────────────
tab_chat, tab_vectordb, tab_logs = st.tabs(["💬 Chat", "🗄️ Vector DB Explorer", "📋 Pipeline Logs"])

with tab_vectordb:
    st.markdown("### 🗄️ ChromaDB Collection Explorer")
    if stats["total_chunks"] == 0:
        st.info("No documents indexed yet. Upload PDFs to see embeddings here.")
    else:
        st.markdown(
            f"**Collection**: `docmind` — **{stats['total_chunks']}** chunks "
            f"from **{stats['total_documents']}** documents"
        )

        from app.ingestion.embedder import get_all_documents
        all_docs = get_all_documents()

        if all_docs and all_docs.get("ids"):
            # Show a sample of chunks in a table
            import pandas as pd
            rows = []
            for i in range(min(len(all_docs["ids"]), 100)):  # Show max 100
                meta = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
                text = all_docs["documents"][i] if all_docs["documents"] else ""
                rows.append({
                    "Chunk ID": all_docs["ids"][i],
                    "Source": meta.get("source", "?"),
                    "Page": meta.get("page", "?"),
                    "Text Preview": text[:150] + ("..." if len(text) > 150 else ""),
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=400)

            # Per-document breakdown
            st.markdown("#### 📊 Chunks Per Document")
            source_counts = {}
            for meta in all_docs["metadatas"]:
                src = meta.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1

            chart_df = pd.DataFrame(
                list(source_counts.items()),
                columns=["Document", "Chunks"],
            )
            st.bar_chart(chart_df.set_index("Document"))

with tab_logs:
    st.markdown("### 📋 Pipeline Execution Logs")
    st.caption("Logs from the current session's RAG pipeline operations.")
    if st.session_state.pipeline_logs:
        for log_entry in reversed(st.session_state.pipeline_logs):
            st.markdown(f'<div class="log-entry">{log_entry}</div>', unsafe_allow_html=True)
    else:
        st.info("No logs yet. Ask a question or upload a document to see pipeline activity.")

    if st.button("🗑️ Clear Logs"):
        st.session_state.pipeline_logs = []
        st.rerun()

with tab_chat:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                # Show grounding warning if applicable
                if message.get("grounding_status") == "NOT_GROUNDED":
                    st.markdown(
                        '<div class="warning-banner">'
                        '⚠️ <strong>Warning:</strong> This answer may not be '
                        'fully grounded in the source documents. '
                        f'{message.get("grounding_explanation", "")}'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                with st.expander(f"📚 Sources ({len(message['sources'])} chunks)"):
                    for src in message["sources"]:
                        st.markdown(
                            f'<div class="source-chunk">'
                            f'<div class="source-meta">'
                            f'📄 {src["source"]} — Page {src["page"]}'
                            f'</div>'
                            f'{src["text"][:500]}{"..." if len(src["text"]) > 500 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Show pipeline metadata if available
                if "pipeline_meta" in message:
                    with st.expander("🔍 Pipeline Details"):
                        meta = message["pipeline_meta"]
                        st.markdown(f"- **LLM Provider**: `{meta.get('provider', '?')}`")
                        st.markdown(f"- **Original Query**: `{meta.get('original_query', '?')}`")
                        st.markdown(f"- **Rewritten Query**: `{meta.get('rewritten_query', '?')}`")
                        st.markdown(f"- **Hybrid Candidates**: {meta.get('hybrid_count', '?')}")
                        st.markdown(f"- **Reranked Top-K**: {meta.get('reranked_count', '?')}")
                        st.markdown(f"- **Grounding**: `{meta.get('grounding_status', '?')}`")


    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if there are indexed documents
        if stats["total_chunks"] == 0:
            st.warning(
                "⚠️ No documents indexed yet! "
                "Please upload a PDF or drop one into `data/documents/`."
            )
        else:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
            })
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process the question
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        from app.llm.ollama_chain import get_llm, ask, build_context
                        from app.agents.query_rewriter import rewrite_query
                        from app.agents.reranker import rerank
                        from app.agents.citation_validator import validate_citation
                        from app.retrieval.hybrid import hybrid_search

                        # Get settings from sidebar
                        temp = temperature
                        final_top_k = top_k
                        provider = llm_provider

                        add_log(f"Query: {prompt[:80]}...")
                        add_log(f"LLM Provider: {provider}")

                        # Step 1: Query Rewriting
                        add_log("Step 1: Query rewriting...")
                        llm = get_llm(temperature=temp, provider=provider)
                        rewritten_query = rewrite_query(prompt, llm=llm)
                        add_log(f"Rewritten query: {rewritten_query[:80]}...")

                        # Step 2: Hybrid Retrieval
                        add_log("Step 2: Hybrid retrieval (BM25 + Dense)...")
                        candidates = hybrid_search(rewritten_query)
                        add_log(f"Retrieved {len(candidates)} hybrid candidates")

                        # Step 3: Reranking
                        add_log(f"Step 3: Cross-encoder reranking → top {final_top_k}...")
                        top_chunks = rerank(rewritten_query, candidates, top_n=final_top_k)
                        add_log(f"Reranked to {len(top_chunks)} chunks")

                        # Step 4: Generate Answer
                        add_log("Step 4: LLM answer generation...")
                        answer = ask(
                            question=prompt,
                            chunks=top_chunks,
                            temperature=temp,
                            provider=provider,
                        )
                        add_log(f"Answer generated ({len(answer)} chars)")

                        # Step 5: Citation Validation
                        add_log("Step 5: Citation validation...")
                        context_text = build_context(top_chunks)
                        validation = validate_citation(answer, context_text, llm=llm)
                        grounding_status = validation.get("status", "UNKNOWN")
                        grounding_explanation = validation.get("explanation", "")
                        add_log(f"Grounding: {grounding_status}")

                        # Display answer
                        st.markdown(answer)

                        # Show grounding warning if applicable
                        if grounding_status == "NOT_GROUNDED":
                            st.markdown(
                                '<div class="warning-banner">'
                                '⚠️ <strong>Warning:</strong> This answer may not be '
                                'fully grounded in the source documents. '
                                f'{grounding_explanation}'
                                '</div>',
                                unsafe_allow_html=True,
                            )

                        # Show sources
                        sources_data = [
                            {
                                "source": c.get("source", "unknown"),
                                "page": c.get("page", "?"),
                                "text": c.get("text", ""),
                            }
                            for c in top_chunks
                        ]

                        with st.expander(f"📚 Sources ({len(sources_data)} chunks)"):
                            for src in sources_data:
                                st.markdown(
                                    f'<div class="source-chunk">'
                                    f'<div class="source-meta">'
                                    f'📄 {src["source"]} — Page {src["page"]}'
                                    f'</div>'
                                    f'{src["text"][:500]}{"..." if len(src["text"]) > 500 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                        # Pipeline metadata
                        pipeline_meta = {
                            "provider": provider,
                            "original_query": prompt,
                            "rewritten_query": rewritten_query,
                            "hybrid_count": len(candidates),
                            "reranked_count": len(top_chunks),
                            "grounding_status": grounding_status,
                        }

                        with st.expander("🔍 Pipeline Details"):
                            st.markdown(f"- **LLM Provider**: `{provider}`")
                            st.markdown(f"- **Original Query**: `{prompt}`")
                            st.markdown(f"- **Rewritten Query**: `{rewritten_query}`")
                            st.markdown(f"- **Hybrid Candidates**: {len(candidates)}")
                            st.markdown(f"- **Reranked Top-K**: {len(top_chunks)}")
                            st.markdown(f"- **Grounding**: `{grounding_status}`")

                        # Save to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_data,
                            "grounding_status": grounding_status,
                            "grounding_explanation": grounding_explanation,
                            "pipeline_meta": pipeline_meta,
                        })

                        add_log("Pipeline complete ✓")

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Pipeline error: {e}")
                        add_log(f"ERROR: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

            # Trim conversation history to last 10 turns
            if len(st.session_state.messages) > 20:  # 10 turns = 20 messages
                st.session_state.messages = st.session_state.messages[-20:]
