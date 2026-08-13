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

from app.config import DOCS_DIR, RERANK_TOP_N, RETRIEVAL_TOP_K, GEMINI_API_KEY, LLM_PROVIDER

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — Agentic RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── High-Contrast & Premium Design CSS ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main App Background & Typography */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #111827 0%, #0b0f19 90%);
        color: #f1f5f9;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #cbd5e1 !important;
    }

    /* Gradient Brand Title */
    .brand-title {
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: -4px;
        margin-bottom: 1rem;
    }

    /* Stats Card Component */
    .stats-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .stats-number {
        font-size: 1.75rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.2;
    }

    .stats-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Source Reference Chunk Cards */
    .source-chunk {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin: 10px 0;
        color: #e2e8f0;
        font-size: 0.875rem;
        line-height: 1.6;
        border-top: 1px solid #334155;
        border-bottom: 1px solid #334155;
        border-right: 1px solid #334155;
    }

    .source-meta {
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Warning Banner for Hallucination / Grounding Detection */
    .warning-banner {
        background: rgba(234, 88, 12, 0.15);
        border: 1px solid #f97316;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 14px;
        color: #fdba74;
        font-size: 0.88rem;
    }

    /* Pipeline Log Entry */
    .log-entry {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        padding: 6px 8px;
        border-radius: 4px;
        margin-bottom: 4px;
        background: #090d16;
        border: 1px solid #1e293b;
    }

    /* Chat input & containers */
    .stChatFloatingInputContainer {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "watcher_started" not in st.session_state:
    st.session_state.watcher_started = False

if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []


def add_log(msg: str):
    """Add a timestamped log entry to the pipeline log panel."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.pipeline_logs.append(f"[{ts}] {msg}")
    if len(st.session_state.pipeline_logs) > 60:
        st.session_state.pipeline_logs = st.session_state.pipeline_logs[-60:]


# ─── Pipeline Initialization (cached) ─────────────────────────────────────────
@st.cache_resource
def initialize_pipeline():
    """Initialize the ingestion pipeline and BM25 index on startup."""
    from app.ingestion.loader import load_all_pdfs
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.embedder import embed_and_store, get_collection_stats
    from app.retrieval.bm25_index import rebuild_from_chroma, load_cache

    stats = get_collection_stats()

    if stats["total_chunks"] == 0:
        pages = load_all_pdfs(DOCS_DIR)
        if pages:
            chunks = chunk_pages(pages)
            embed_and_store(chunks)

    if not load_cache():
        rebuild_from_chroma()

    return True


@st.cache_resource
def start_folder_watcher():
    """Start the background folder watcher daemon."""
    try:
        from app.watcher.folder_watch import start_watcher
        return start_watcher()
    except Exception as e:
        logger.warning(f"Watchdog could not start: {e}")
        return None


# ─── Run Startup ─────────────────────────────────────────────────────────────
with st.spinner("Initializing DocMind RAG Engine..."):
    initialize_pipeline()

if not st.session_state.watcher_started:
    start_folder_watcher()
    st.session_state.watcher_started = True


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-title">DocMind 🧠</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Agentic RAG Knowledge Assistant</div>', unsafe_allow_html=True)
    st.divider()

    # Document & Collection Metrics
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

    # List of Indexed Files
    st.markdown("### 📄 Indexed Corpus")
    if stats["sources"]:
        for source in stats["sources"]:
            st.markdown(f"• `{source}`")
    else:
        st.caption("No documents in vector index. Upload PDFs below.")

    st.divider()

    # Multi-file PDF Upload
    st.markdown("### 📤 Upload PDFs")
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
                st.success(f"Uploaded: {uploaded_file.name}")

                with st.spinner(f"Embedding {uploaded_file.name}..."):
                    from app.ingestion.loader import load_pdf
                    from app.ingestion.chunker import chunk_pages
                    from app.ingestion.embedder import embed_and_store
                    from app.retrieval.bm25_index import rebuild_from_chroma

                    pages = load_pdf(save_path)
                    if pages:
                        chunks = chunk_pages(pages)
                        num_added = embed_and_store(chunks)
                        rebuild_from_chroma()
                        st.success(f"Indexed {num_added} chunks!")
                        add_log(f"Indexed {uploaded_file.name}: {num_added} chunks")
                    else:
                        st.warning(f"No text extracted from {uploaded_file.name}")
                        add_log(f"Failed extraction: {uploaded_file.name}")

                st.rerun()

    # Re-index Corpus Button
    if st.button("🔄 Re-index All Documents", use_container_width=True):
        with st.spinner("Rebuilding ChromaDB and BM25 index..."):
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
                add_log(f"Re-indexed {num_stored} chunks across {len(set(p['source'] for p in pages))} docs")
            rebuild_from_chroma()

        st.success("Re-indexing complete!")
        st.rerun()

    st.divider()

    # Settings & Model Controls
    with st.expander("⚙️ Inference & RAG Settings", expanded=True):
        user_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=GEMINI_API_KEY,
            placeholder="AIzaSy...",
            help="Get your free key at https://aistudio.google.com/apikey",
        )

        llm_provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "auto", "ollama"],
            index=0 if LLM_PROVIDER == "gemini" else (1 if LLM_PROVIDER == "auto" else 2),
            help=(
                "**gemini**: Google Gemini API (fastest & high accuracy)\n\n"
                "**auto**: Try local Ollama, fallback to Gemini\n\n"
                "**ollama**: Local Ollama model only"
            ),
        )

        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Lower (0.1) = strictly factual and grounded. Higher (0.7) = creative.",
        )

        top_k = st.slider(
            "Top-K Chunks (after Reranker)",
            min_value=1,
            max_value=15,
            value=RERANK_TOP_N,
            step=1,
            help="Number of cross-encoder reranked chunks passed into the LLM context window.",
        )


# ─── Main Interface Tabs ────────────────────────────────────────────────────
st.markdown("## 💬 Knowledge Assistant")

tab_chat, tab_vectordb, tab_logs = st.tabs([
    "💬 Interactive Q&A",
    "🗄️ Vector DB & Embeddings",
    "📋 Pipeline Execution Trace"
])

# ─── Tab 2: Vector DB Explorer ──────────────────────────────────────────────
with tab_vectordb:
    st.markdown("### 🗄️ ChromaDB & Sparse Index Explorer")
    if stats["total_chunks"] == 0:
        st.info("No documents indexed yet. Upload PDFs using the sidebar to view stored embeddings.")
    else:
        st.markdown(
            f"**Active Collection**: `docmind` — **{stats['total_chunks']}** chunks "
            f"across **{stats['total_documents']}** document(s)."
        )

        from app.ingestion.embedder import get_all_documents
        all_docs = get_all_documents()

        if all_docs and all_docs.get("ids"):
            import pandas as pd
            rows = []
            for i in range(min(len(all_docs["ids"]), 150)):
                meta = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
                text = all_docs["documents"][i] if all_docs["documents"] else ""
                rows.append({
                    "Chunk ID": all_docs["ids"][i],
                    "Document": meta.get("source", "?"),
                    "Page": meta.get("page", "?"),
                    "Text Content Preview": text[:160] + ("..." if len(text) > 160 else ""),
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=360)

            # Document Chunk Distribution
            st.markdown("#### 📊 Chunk Distribution by Document")
            source_counts = {}
            for meta in all_docs["metadatas"]:
                src = meta.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1

            chart_df = pd.DataFrame(
                list(source_counts.items()),
                columns=["Document", "Chunk Count"],
            )
            st.bar_chart(chart_df.set_index("Document"))

# ─── Tab 3: Execution Logs ──────────────────────────────────────────────────
with tab_logs:
    st.markdown("### 📋 Real-Time Pipeline Trace")
    st.caption("Detailed execution log for query rewriting, hybrid retrieval, reranking, and verification.")

    if st.session_state.pipeline_logs:
        for log_entry in reversed(st.session_state.pipeline_logs):
            st.markdown(f'<div class="log-entry">{log_entry}</div>', unsafe_allow_html=True)
    else:
        st.info("No query logs yet. Ask a question in the chat tab to view real-time pipeline execution.")

    if st.button("🗑️ Clear Logs"):
        st.session_state.pipeline_logs = []
        st.rerun()

# ─── Tab 1: Chat Interface ──────────────────────────────────────────────────
with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                if message.get("grounding_status") == "NOT_GROUNDED":
                    st.markdown(
                        '<div class="warning-banner">'
                        '⚠️ <strong>Hallucination Alert:</strong> This response may not be '
                        'fully grounded in the context chunks.<br>'
                        f'<small>{message.get("grounding_explanation", "")}</small>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                with st.expander(f"📚 Retrieved Context Sources ({len(message['sources'])} chunks)"):
                    for src in message["sources"]:
                        st.markdown(
                            f'<div class="source-chunk">'
                            f'<div class="source-meta">📄 {src["source"]} — Page {src["page"]}</div>'
                            f'{src["text"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                if "pipeline_meta" in message:
                    with st.expander("🔍 Agentic Pipeline Telemetry"):
                        meta = message["pipeline_meta"]
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**LLM Provider**: `{meta.get('provider')}`")
                            st.write(f"**Original Query**: `{meta.get('original_query')}`")
                            st.write(f"**Rewritten Query**: `{meta.get('rewritten_query')}`")
                        with col_b:
                            st.write(f"**Hybrid Candidates (RRF)**: `{meta.get('hybrid_count')}`")
                            st.write(f"**Final Reranked Chunks**: `{meta.get('reranked_count')}`")
                            st.write(f"**Citation Grounding**: `{meta.get('grounding_status')}`")

    # Chat Input Box
    if prompt := st.chat_input("Ask a question about your uploaded documents..."):
        if stats["total_chunks"] == 0:
            st.warning("⚠️ No documents indexed yet. Please upload a PDF in the sidebar first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Processing through Agentic RAG Pipeline..."):
                    try:
                        from app.llm.ollama_chain import get_llm, ask, build_context
                        from app.agents.query_rewriter import rewrite_query
                        from app.agents.reranker import rerank
                        from app.agents.citation_validator import validate_citation
                        from app.retrieval.hybrid import hybrid_search

                        active_key = user_api_key.strip()
                        active_provider = llm_provider
                        active_temp = temperature
                        final_top_n = top_k

                        add_log(f"User Query: {prompt}")
                        add_log(f"Selected Provider: {active_provider} | Temp: {active_temp}")

                        # ── Step 1: Query Rewriting Agent ──
                        add_log("Agent 1: Evaluating query formulation...")
                        llm = get_llm(temperature=active_temp, provider=active_provider, api_key=active_key)
                        rewritten_query = rewrite_query(prompt, llm=llm)
                        add_log(f"Agent 1 Result: '{rewritten_query}'")

                        # ── Step 2: Hybrid Retrieval (BM25 + Dense) ──
                        add_log("Retrieval: Running BM25 + Dense vector search (RRF)...")
                        candidates = hybrid_search(rewritten_query)
                        add_log(f"Retrieved {len(candidates)} hybrid candidates")

                        # ── Step 3: Cross-Encoder Reranking Agent ──
                        add_log(f"Agent 2: Cross-Encoder reranking ({len(candidates)} -> {final_top_n})...")
                        top_chunks = rerank(rewritten_query, candidates, top_n=final_top_n)
                        add_log(f"Agent 2 Completed: {len(top_chunks)} chunks selected")

                        # ── Step 4: Answer Generation ──
                        add_log("LLM: Generating cited answer from reranked context...")
                        answer = ask(
                            question=prompt,
                            chunks=top_chunks,
                            temperature=active_temp,
                            provider=active_provider,
                            api_key=active_key,
                        )
                        add_log(f"LLM generated answer ({len(answer)} chars)")

                        # ── Step 5: Citation Validation Agent ──
                        add_log("Agent 3: Validating citation grounding & hallucination check...")
                        context_text = build_context(top_chunks)
                        validation = validate_citation(answer, context_text, llm=llm)
                        grounding_status = validation.get("status", "UNKNOWN")
                        grounding_explanation = validation.get("explanation", "")
                        add_log(f"Agent 3 Grounding: {grounding_status}")

                        # Render Answer
                        st.markdown(answer)

                        if grounding_status == "NOT_GROUNDED":
                            st.markdown(
                                '<div class="warning-banner">'
                                '⚠️ <strong>Hallucination Alert:</strong> This response may not be '
                                'fully grounded in the context chunks.<br>'
                                f'<small>{grounding_explanation}</small>'
                                '</div>',
                                unsafe_allow_html=True,
                            )

                        sources_data = [
                            {
                                "source": c.get("source", "unknown"),
                                "page": c.get("page", "?"),
                                "text": c.get("text", ""),
                            }
                            for c in top_chunks
                        ]

                        with st.expander(f"📚 Retrieved Context Sources ({len(sources_data)} chunks)"):
                            for src in sources_data:
                                st.markdown(
                                    f'<div class="source-chunk">'
                                    f'<div class="source-meta">📄 {src["source"]} — Page {src["page"]}</div>'
                                    f'{src["text"]}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                        pipeline_meta = {
                            "provider": active_provider,
                            "original_query": prompt,
                            "rewritten_query": rewritten_query,
                            "hybrid_count": len(candidates),
                            "reranked_count": len(top_chunks),
                            "grounding_status": grounding_status,
                        }

                        with st.expander("🔍 Agentic Pipeline Telemetry"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**LLM Provider**: `{active_provider}`")
                                st.write(f"**Original Query**: `{prompt}`")
                                st.write(f"**Rewritten Query**: `{rewritten_query}`")
                            with col_b:
                                st.write(f"**Hybrid Candidates (RRF)**: `{len(candidates)}`")
                                st.write(f"**Final Reranked Chunks**: `{len(top_chunks)}`")
                                st.write(f"**Citation Grounding**: `{grounding_status}`")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_data,
                            "grounding_status": grounding_status,
                            "grounding_explanation": grounding_explanation,
                            "pipeline_meta": pipeline_meta,
                        })

                        add_log("Pipeline cycle completed successfully ✓")

                    except Exception as e:
                        error_msg = f"❌ Pipeline Error: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Pipeline error: {e}")
                        add_log(f"ERROR: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

            if len(st.session_state.messages) > 20:
                st.session_state.messages = st.session_state.messages[-20:]
